"""Dyadic extraction rendering + deterministic LoCoMo probe engine (role W3).

Extraction rendering (protocol v2 §2.2 item 2, V2-P2; dyadic-gloss §7 item 3):

- ONE extractor call per turn (5,882/family) extracts facts for BOTH speaker
  stores; the frozen dyadic prompts (``extractor_dyadic.txt`` E-M family,
  ``extractor_zep_dyadic.txt`` E-Z family) are used verbatim.
- The current-turn marker NAMES the current speaker (the dyadic analog of
  ``extract._CURRENT_TURN_MARKER``); turn lines are ``{Speaker}: {text}``.
- Context = up to 2 preceding turns (either speaker, crossing session
  boundaries) + session-date headers ``[session of {date}]`` in the
  ``extract.render_extraction_prompt`` style.
- Image turns render their caption as ``[shares photo: <blip_caption>]``
  appended to the turn text — context only; the prompts forbid caption-only
  tuples (captions are demonstrably wrong, locomo_analysis.md §1.2).
- The dyadic tuple schema adds a boolean ``secondhand`` field;
  ``parse_dyadic_tuples`` validates it (gloss §7 item 3).
- Routing (§2.2 item 3): a tuple about person P goes to P's store regardless of
  who uttered it; provenance records ``speaker`` and deterministic
  ``secondhand = (speaker != P)``. Third parties who are not the partner stay
  as ordinary third-party entities in the uttering speaker's store
  (locomo_analysis.md §3.1).

Deterministic probe engine (protocol v2 §4, V2-P4 — no LLM answering layer;
every probe below is a pure store lookup):

- **P-MISATTR** (``p_misattr_probe``): abstention-by-absence on the frozen
  cat-5 sample — correct = the focal speaker's store LACKS the absence key;
  the ``(unknown)`` sentinel counts as absent (§1.2 item 10).
- **P-TIME** (``p_time_probe``): as-of lookup against ZepStore supersession
  history — answer = the value whose ``[valid_from, invalid_from)`` interval
  contains the probe date, for the 13 key-mapped LoCoMo time-indexed items
  (the 6 LME old-value questions are the LME half, out of scope here).
- **P-NATQA** (``p_natqa_probe``): key lookups for the 98 pre-mapped QA rows —
  current-value/count rows read the current entry, as-of rows read the
  validity interval at the row's pinned ``asof_date`` flag, history-tier rows
  return the full supersession history for downstream per-map_type scoring.

Value equivalence stays with the shared machinery: probes compute the
``store.normalize_value`` exact-match fast path and flag ``needs_judge`` for
the blind equivalence judge (never invoked here). Store access is duck-typed
against the ZepStore entry contract ``(value, provenance, valid_from,
invalid_from, created_seq, expired_seq)`` (zep_arm_design.md §4): a store may
expose ``history(key)`` (the ``zepstore`` module, freeze item 10) or be a plain
mapping ``key -> entry | list[entry]`` (pilot snapshots, synthetic test stores).
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from llm import call_llm
from store import normalize_value, values_equivalent

from locomo_data import parse_key_id, parse_session_date

try:  # v2 supersession store (spec §10.1 item 10); optional until it lands
    import zepstore
except ImportError:
    zepstore = None

from paths import PROMPTS_DIR

EXTRACTOR_DYADIC_PROMPT = (PROMPTS_DIR / "extractor_dyadic.txt").read_text()
EXTRACTOR_ZEP_DYADIC_PROMPT = (PROMPTS_DIR / "extractor_zep_dyadic.txt").read_text()

#: extraction-family -> frozen dyadic system prompt (the only lever on the axis)
DYADIC_PROMPTS = {"E-M": EXTRACTOR_DYADIC_PROMPT, "E-Z": EXTRACTOR_ZEP_DYADIC_PROMPT}

#: context window for dyadic extraction: up to 2 preceding turns (spec §2.2 item 2)
CONTEXT_WINDOW = 2

_CURRENT_TURN_MARKER_TEMPLATE = (
    "=== CURRENT MESSAGE from {speaker} (extract fact tuples from this message "
    "only; everything above is context) ==="
)

_PHOTO_NOTE_TEMPLATE = "[shares photo: {caption}]"


# ---------------------------------------------------------------------------
# Dyadic tuple schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DyadicTuple:
    """One extracted dyadic statement tuple — ``extract.Tuple`` plus the
    ``secondhand`` boolean of the dyadic prompt schema."""

    entity: str
    attribute: str
    value: str
    statement_span: str
    secondhand: bool
    confidence: int  # verbalized numeric 0-100 (same contract as extract.Tuple)


def parse_dyadic_tuples(payload: dict) -> list[DyadicTuple]:
    """Parse the dyadic extractor's JSON output.

    Same validation as ``extract.parse_tuples`` (confidence must be an integer
    0-100) plus the dyadic addition: ``secondhand`` must be a JSON boolean
    (gloss §7 item 3 — a string "true"/"false" or 0/1 is a contract violation
    and raises).
    """
    parsed: list[DyadicTuple] = []
    for row in payload["tuples"]:
        confidence = int(round(float(row["confidence"])))
        if not 0 <= confidence <= 100:
            raise ValueError(f"confidence out of range 0-100: {row['confidence']!r}")
        secondhand = row["secondhand"]
        if not isinstance(secondhand, bool):
            raise ValueError(f"secondhand must be a JSON boolean: {secondhand!r}")
        parsed.append(
            DyadicTuple(
                entity=str(row["entity"]).strip(),
                attribute=str(row["attribute"]).strip(),
                value=str(row["value"]).strip(),
                statement_span=str(row["statement_span"]),
                secondhand=secondhand,
                confidence=confidence,
            )
        )
    return parsed


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_turn_text(turn: dict) -> str:
    """A turn's text with its photo note appended when the turn carries an
    image: ``{text} [shares photo: {blip_caption}]``. Captions enter as context
    only — the dyadic prompts forbid tuples supported solely by a caption."""
    text = turn["text"]
    caption = turn.get("blip_caption")
    if caption:
        note = _PHOTO_NOTE_TEMPLATE.format(caption=caption)
        return f"{text} {note}" if text.strip() else note
    return text


def render_turn_line(turn: dict) -> str:
    """``{Speaker}: {text-with-photo-note}`` — every line names its speaker,
    matching the dyadic prompts' 'every turn is labeled' framing."""
    return f"{turn['speaker']}: {render_turn_text(turn)}"


def current_turn_marker(speaker: str) -> str:
    """The speaker-named current-turn marker (dyadic analog of
    ``extract._CURRENT_TURN_MARKER``; gloss §7 item 3)."""
    return _CURRENT_TURN_MARKER_TEMPLATE.format(speaker=speaker)


def render_dyadic_prompt(
    prior_turns: list[dict], current_turn: dict, current_date: str | None
) -> str:
    """Render the dyadic extraction user prompt.

    ``prior_turns`` (up to ``CONTEXT_WINDOW``, either speaker) are dicts
    ``{speaker, text, date, blip_caption?}`` in chronological order;
    ``current_turn`` is a loader turn dict ``{speaker, dia_id, text,
    blip_caption}``. Structure mirrors ``extract.render_extraction_prompt``:
    session-date headers whenever the date changes, then the speaker-named
    current-turn marker.
    """
    lines: list[str] = []
    last_date: str | None = None
    if prior_turns:
        lines.append("=== PRIOR CONVERSATION (context only) ===")
        for turn in prior_turns:
            if turn["date"] != last_date:
                lines.append(f"[session of {turn['date']}]")
                last_date = turn["date"]
            lines.append(render_turn_line(turn))
    lines.append(current_turn_marker(current_turn["speaker"]))
    if current_date is not None and current_date != last_date:
        lines.append(f"[session of {current_date}]")
    lines.append(render_turn_line(current_turn))
    return "\n".join(lines)


def dyadic_context(conv, session_idx: int, turn_idx: int, *, window: int = CONTEXT_WINDOW) -> list[dict]:
    """The up-to-``window`` turns preceding (session_idx, turn_idx) in
    chronological order, either speaker, crossing session boundaries backwards.
    Each turn is tagged with its host session's date (consumed by
    ``render_dyadic_prompt``)."""
    positions: list[tuple[int, int]] = []
    for orig_idx in conv.session_order:
        for t_idx in range(len(conv.sessions[orig_idx])):
            positions.append((orig_idx, t_idx))
    where = positions.index((session_idx, turn_idx))
    context = []
    for s_idx, t_idx in positions[max(0, where - window):where]:
        turn = conv.sessions[s_idx][t_idx]
        context.append(
            {
                "speaker": turn["speaker"],
                "text": turn["text"],
                "blip_caption": turn.get("blip_caption"),
                "date": conv.dates[s_idx],
            }
        )
    return context


def dyadic_turn_job(conv, session_idx: int, turn_idx: int, *, family: str = "E-M") -> dict:
    """Deterministic (LLM-free) job builder for one dyadic extraction call:
    context turns, current turn, current date, and the trace tag. The runner
    feeds this straight into ``extract_dyadic_turn``."""
    if family not in DYADIC_PROMPTS:
        raise ValueError(f"unknown extraction family {family!r}; expected one of {sorted(DYADIC_PROMPTS)}")
    turn = conv.sessions[session_idx][turn_idx]
    return {
        "prior_turns": dyadic_context(conv, session_idx, turn_idx),
        "current_turn": turn,
        "current_date": conv.dates[session_idx],
        "family": family,
        "trace_tag": f"extract-dyadic-{family}-{conv.sample_id}-{turn['dia_id']}",
    }


async def extract_dyadic_turn(
    prior_turns: list[dict],
    current_turn: dict,
    trace_tag: str,
    *,
    family: str = "E-M",
    current_date: str | None = None,
) -> list[DyadicTuple]:
    """ONE dyadic extractor call for one turn (V2-P2): extracts facts for BOTH
    speaker stores in a single pass; ``entity`` carries the person the fact is
    about. ``family`` selects the frozen system prompt: "E-M"
    (extractor_dyadic.txt) or "E-Z" (extractor_zep_dyadic.txt) — the only
    difference between extraction arms."""
    if family not in DYADIC_PROMPTS:
        raise ValueError(f"unknown extraction family {family!r}; expected one of {sorted(DYADIC_PROMPTS)}")
    user_prompt = render_dyadic_prompt(prior_turns, current_turn, current_date)
    output = await call_llm(DYADIC_PROMPTS[family], user_prompt, trace_tag=trace_tag)
    return parse_dyadic_tuples(output)


# ---------------------------------------------------------------------------
# Store routing (spec §2.2 item 3; both speaker-stores update from one pass)
# ---------------------------------------------------------------------------


def route_tuple(tup: DyadicTuple, current_speaker: str, speaker_a: str, speaker_b: str) -> dict:
    """Route one extracted tuple to its speaker-store.

    A tuple about person P is routed to P's store regardless of which speaker
    uttered it; provenance records ``speaker`` and the DETERMINISTIC
    ``secondhand = (speaker != P)`` (spec §2.2 item 3). The model-emitted
    ``secondhand`` field is kept alongside for audit (``model_secondhand``) but
    never decides routing. Third-party entities (neither speaker) stay in the
    uttering speaker's store as ordinary third-party entities.
    """
    entity = tup.entity.strip().lower()
    speakers = {speaker_a.lower(): speaker_a, speaker_b.lower(): speaker_b}
    store_owner = speakers[entity].lower() if entity in speakers else current_speaker.lower()
    return {
        "store": store_owner,
        "entity": entity,
        "speaker": current_speaker,
        "secondhand": entity != current_speaker.lower(),
        "model_secondhand": tup.secondhand,
    }


def route_turn(
    tuples: list[DyadicTuple], current_speaker: str, speaker_a: str, speaker_b: str
) -> list[tuple[DyadicTuple, dict]]:
    """Route a whole turn's tuples; both speaker-stores update from the same
    single extraction pass (no per-speaker duplicate replay)."""
    return [(tup, route_tuple(tup, current_speaker, speaker_a, speaker_b)) for tup in tuples]


# ---------------------------------------------------------------------------
# Deterministic probe engine — shared store access (ZepStore duck-typing)
# ---------------------------------------------------------------------------

# Frozen negation sentinel (spec §1.2 item 10): a key whose current value is
# `(unknown)` counts as ABSENT for presence probes and non-equivalent to any
# concrete value for equivalence scoring. The authoritative sentinel regex is
# a ZepStore freeze-set artifact (`zepstore.is_sentinel`); the canonical form
# below is only the fallback when that module is absent.
_UNKNOWN_SENTINEL = "(unknown)"


def value_is_sentinel(value) -> bool:
    """True when a stored value is the frozen ``(unknown)`` negation sentinel.

    Defers to the freeze-set sentinel regex (``zepstore.is_sentinel``) when the
    module is present: ZepStore-written values are already canonicalized to
    ``(unknown)``, and raw sentinel-form values in synthetic/pilot stores count
    exactly the way the shared write path would have stored them."""
    if value is None:
        return False
    if zepstore is not None:
        return bool(zepstore.is_sentinel(str(value)))
    return normalize_value(str(value)) == _UNKNOWN_SENTINEL


def value_is_absent(value) -> bool:
    """Absence semantics for presence probes: missing key OR sentinel value."""
    return value is None or value_is_sentinel(value)


def _field(entry, name: str, default=None):
    """Read a ZepStore entry field from a dict or an attribute-style entry."""
    if isinstance(entry, dict):
        return entry.get(name, default)
    return getattr(entry, name, default)


def _value_of(entry) -> str | None:
    if isinstance(entry, str):
        return entry
    value = _field(entry, "value")
    return None if value is None else str(value)


def history_entries(store, key: tuple[str, str]) -> list:
    """A key's full supersession history, oldest-first.

    Accepts either a store exposing ``history(key)`` (the ZepStore contract,
    freeze item 10) or a plain mapping ``key -> entry | list[entry]`` (pilot
    snapshot dicts, synthetic test stores). Entries are sorted by
    ``created_seq`` when every entry carries one; otherwise the given order is
    kept (documented, deterministic).
    """
    key = tuple(key)
    history = getattr(store, "history", None)
    if callable(history):
        entries = list(history(key))
    else:
        raw = store.get(key)
        if raw is None:
            entries = []
        elif isinstance(raw, (list, tuple)):
            entries = list(raw)
        else:
            entries = [raw]
    seqs = [_field(e, "created_seq") for e in entries]
    if entries and all(s is not None for s in seqs):
        entries = [entries[i] for _, i in sorted(zip(seqs, range(len(entries))))]
    return entries


def current_entry(store, key: tuple[str, str]):
    """The key's current (non-invalidated) entry, or None. With several
    non-invalidated entries (anomalous), the latest-ingested wins — matching
    last-writer-wins lookup equivalence (zep_arm_design.md §4)."""
    current = [e for e in history_entries(store, key) if _field(e, "invalid_from") is None]
    return current[-1] if current else None


def current_value(store, key: tuple[str, str]) -> str | None:
    """Current value string for ``key`` (sentinel NOT filtered here; presence
    probes apply ``value_is_absent`` on top)."""
    entry = current_entry(store, key)
    return None if entry is None else _value_of(entry)


def as_of_value(entries: list, as_of: datetime, *, parse_date=parse_session_date) -> str | None:
    """P-TIME core rule (spec §4): the value whose ``[valid_from,
    invalid_from)`` interval contains ``as_of``, read from supersession
    history. ``invalid_from=None`` is an open interval; an entry superseded in
    the same instant it was created (empty interval) can never match. With
    anomalous overlaps the latest-ingested matching entry wins."""
    match = None
    for entry in entries:
        valid_from = _field(entry, "valid_from")
        start = parse_date(valid_from) if valid_from is not None else datetime.min
        invalid_from = _field(entry, "invalid_from")
        end = parse_date(invalid_from) if invalid_from is not None else None
        if start <= as_of and (end is None or as_of < end):
            match = entry
    return None if match is None else _value_of(match)


# ---------------------------------------------------------------------------
# P-MISATTR — abstention-by-absence (frozen cat-5 sample; primary stratum
# speaker_swap n=75, focal_fact_distortion n=25 exploratory)
# ---------------------------------------------------------------------------


def p_misattr_probe(focal_store, item: dict) -> dict:
    """Deterministic key-presence check for one cat-5 item against the FOCAL
    speaker's store. Correct = the store LACKS the absence key (the
    ``(unknown)`` sentinel counts as absent). When a value IS present, the
    result carries the normalization fast path against the trap value;
    ``needs_judge`` marks pairs for the blind equivalence judge, which decides
    the cross-attribution vs over-commit sub-type downstream."""
    key = (item["absence_key"]["entity"], item["absence_key"]["attribute"])
    stored = current_value(focal_store, key)
    absent = value_is_absent(stored)
    trap_value = str(item["trap_value"])
    exact_trap_match = (
        not absent and normalize_value(stored) == normalize_value(trap_value)
    )
    return {
        "sample_id": item["sample_id"],
        "qa_index": item["qa_index"],
        "trap_type": item["trap_type"],
        "key": key,
        "stored_value": stored,
        "absent": absent,
        "correct": absent,
        "trap_value": trap_value,
        "exact_trap_match": exact_trap_match,
        "needs_judge": not absent and not exact_trap_match,
    }


async def judge_misattr_subtype(result: dict, *, trace_tag: str) -> str:
    """Sub-type a FAILED misattribution probe: judged equivalence to the trap
    value = ``cross_attribution``; anything else = ``over_commit`` (spec §4
    P-MISATTR row). Uses the shared blind equivalence judge; never called on
    correct (absent) probes."""
    if result["absent"]:
        raise ValueError("subtype is only defined for failed (value-present) probes")
    if result["exact_trap_match"] or await values_equivalent(
        result["key"][1], result["stored_value"], result["trap_value"], trace_tag=trace_tag
    ):
        return "cross_attribution"
    return "over_commit"


# ---------------------------------------------------------------------------
# P-TIME — as-of lookups against ZepStore history (13 key-mapped LoCoMo items)
# ---------------------------------------------------------------------------

# The frozen artifact's as_of_date carries the full session timestamp for
# conversation_end items and a coarser calendar form for explicit dates parsed
# from questions ("August 2023", "July 10, 2022", "27 March, 2022"). Coarse
# forms resolve to the END of the stated period ('as of <period>' = state at
# the period's end), matching parse_asof_flag semantics.
_AS_OF_DAY_FORMATS = ("%d %B, %Y", "%B %d, %Y")


def parse_as_of_date(date: str) -> datetime:
    """Parse a time_indexed_probes ``as_of_date`` in any of its frozen forms."""
    try:
        return parse_session_date(date)  # exact session timestamp
    except ValueError:
        pass
    for fmt in _AS_OF_DAY_FORMATS:
        try:
            return datetime.strptime(date, fmt).replace(hour=23, minute=59)
        except ValueError:
            continue
    try:
        month = datetime.strptime(date, "%B %Y")
    except ValueError:
        raise ValueError(f"unparseable as_of_date: {date!r}") from None
    last_day = calendar.monthrange(month.year, month.month)[1]
    return month.replace(day=last_day, hour=23, minute=59)


def p_time_probe(store, item: dict, *, parse_date=parse_session_date) -> dict:
    """As-of lookup for one key-mapped time-indexed item (spec §4 P-TIME).

    ``store`` is the speaker-store owning the item's key (the entity named in
    ``key_id``); ``parse_date`` parses the STORE entries' session-date fields
    (the item's own as_of_date goes through ``parse_as_of_date``). Raises
    ValueError on the 23 unmapped items — they are episodic counters and cannot
    be scored from supersession history (V2-F4)."""
    if not item.get("key_id"):
        raise ValueError(
            f"time-indexed item ({item['sample_id']}, qa {item['qa_index']}) has no key_id; "
            "only the 13 key-mapped items are P-TIME-scoreable (V2-F4)"
        )
    sample_id, entity, attribute = parse_key_id(item["key_id"])
    key = (entity, attribute)
    as_of = parse_as_of_date(item["as_of_date"])
    value = as_of_value(history_entries(store, key), as_of, parse_date=parse_date)
    found = not value_is_absent(value)
    gold = str(item["gold_answer"])
    exact_match = found and normalize_value(value) == normalize_value(gold)
    return {
        "sample_id": sample_id,
        "qa_index": item["qa_index"],
        "key_id": item["key_id"],
        "key": key,
        "ti_class": item.get("ti_class"),
        "as_of_date": item["as_of_date"],
        "as_of_source": item.get("as_of_source"),
        "value": value,
        "found": found,
        "gold_answer": gold,
        "exact_match": exact_match,
        "needs_judge": found and not exact_match,
    }


# ---------------------------------------------------------------------------
# P-NATQA — key lookups for the 98 pre-mapped QA rows
# ---------------------------------------------------------------------------

_CURRENT_MAP_TYPES = {"current_value", "derived_count"}
_ASOF_MAP_TYPES = {"asof_value"}
_HISTORY_MAP_TYPES = {
    "history_event_time",
    "history_event_value",
    "history_old_value",
    "derived_history",
}

_ASOF_FLAG_RE = re.compile(r"asof_date=(\d{4})-(\d{2})(?:-(\d{2}))?")


def parse_asof_flag(flags: list[str]) -> datetime:
    """Parse a qa_probe_map row's pinned ``asof_date=YYYY-MM[-DD]`` flag into
    the as-of instant. 'As of <period>' means the state at the period's END:
    month precision resolves to the last day of the month, day precision to
    end-of-day (23:59)."""
    for flag in flags:
        m = _ASOF_FLAG_RE.fullmatch(flag)
        if m is not None:
            year, month = int(m.group(1)), int(m.group(2))
            day = int(m.group(3)) if m.group(3) else calendar.monthrange(year, month)[1]
            return datetime(year, month, day, 23, 59)
    raise ValueError(f"no asof_date flag in {flags!r}")


def p_natqa_probe(store, item: dict, *, parse_date=parse_session_date) -> dict:
    """Deterministic key lookup for one pre-mapped QA row (spec §4 P-NATQA).

    ``store`` is the speaker-store owning the row's key. Dispatch by map_type:
    current_value/derived_count read the current entry; asof_value reads the
    validity interval at the row's pinned asof_date flag; history-tier rows
    (transition event times / superseded values) return the full supersession
    history — their per-map_type scoring happens downstream, still without any
    answering LLM. ``exact_match`` is the normalization fast path;
    ``needs_judge`` routes unresolved value pairs to the blind judge.
    """
    sample_id, entity, attribute = parse_key_id(item["key_id"])
    key = (entity, attribute)
    map_type = item["map_type"]
    gold = str(item["gold_answer"])
    result = {
        "sample_id": sample_id,
        "qa_index": item["qa_index"],
        "key_id": item["key_id"],
        "key": key,
        "map_type": map_type,
        "tier": item["tier"],
        "gold_answer": gold,
        "history": None,
        "value": None,
        "found": False,
        "exact_match": False,
        "needs_judge": False,
    }
    if map_type in _CURRENT_MAP_TYPES:
        result["kind"] = "current"
        result["value"] = current_value(store, key)
    elif map_type in _ASOF_MAP_TYPES:
        result["kind"] = "as_of"
        as_of = parse_asof_flag(item.get("flags", []))
        result["as_of"] = as_of.isoformat()
        result["value"] = as_of_value(history_entries(store, key), as_of, parse_date=parse_date)
    elif map_type in _HISTORY_MAP_TYPES:
        result["kind"] = "history"
        result["history"] = [
            {
                "value": _value_of(entry),
                "valid_from": _field(entry, "valid_from"),
                "invalid_from": _field(entry, "invalid_from"),
            }
            for entry in history_entries(store, key)
        ]
        result["found"] = len(result["history"]) > 0
        return result
    else:
        raise ValueError(f"unknown map_type {map_type!r} for {item['key_id']}")
    result["found"] = not value_is_absent(result["value"])
    result["exact_match"] = (
        result["found"] and normalize_value(result["value"]) == normalize_value(gold)
    )
    result["needs_judge"] = result["found"] and not result["exact_match"]
    return result
