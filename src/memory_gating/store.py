"""Flat fact store with write-through replay — protocol_spec.md §3.1 (Test 2).

Replay semantics (v1.1): for each user turn in chronological order, extract
(extract.extract_turn, contract B2), canonicalize each tuple's (entity,
attribute) to schema keys via ONE key_mapper LLM call per tuple
(key_schema.json `canonicalization.extractor_key_mapping`), then dict-write
last-writer-wins into `dict[(entity, attribute)] -> StoreEntry(value,
provenance, timestamp)`. A store snapshot is retained after every user turn.
Unmapped tuples are logged (../results/test2/unmapped.jsonl), never silently
dropped (FM-6). Probing is a deterministic store lookup at the
manifest-designated probe point (B4). Value equivalence tries normalized exact
match first (key_schema `value_normalization_for_exact_match`); only unresolved
pairs go to the blind LLM equivalence judge with seeded-random pair order
(FM-5).

Implementation notes:
- LLM calls are batched with asyncio.gather for speed; because every call is a
  pure function of its prompt (llm.py cache) and extraction inputs never depend
  on the store, the result is identical to strictly sequential per-turn replay.
- Chronological order follows `substrate.session_order` (list of ORIGINAL
  haystack session indices in chronological order, per INTERFACES.md /
  pinned decision P12); `sessions` and `dates` are indexed by original index.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from extract import Tuple, extract_turn
from llm import JUDGE_MODEL, call_llm

SEED = 20260702

from paths import PROMPTS_DIR, RESULTS_DIR
INJECTED_KEYS_PATH = RESULTS_DIR / "injections" / "injected_keys.jsonl"
UNMAPPED_LOG_PATH = RESULTS_DIR / "test2" / "unmapped.jsonl"
KEY_MAPPING_LOG_PATH = RESULTS_DIR / "analysis" / "key_mapping.jsonl"

KEY_MAPPER_PROMPT = (PROMPTS_DIR / "key_mapper.txt").read_text()
EQUIVALENCE_JUDGE_PROMPT = (PROMPTS_DIR / "equivalence_judge.txt").read_text()

PROBE_POINTS = ("full_ingestion", "old_session_boundary", "provenance_timeline")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalKey:
    """One canonical store key: the substrate's gold key from key_schema.json or
    a registered auxiliary key from injected_keys.jsonl (injector_spec §5.3)."""

    entity: str
    attribute: str
    description: str
    probe_question: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.entity, self.attribute)

    @property
    def name(self) -> str:
        return f"{self.entity}/{self.attribute}"


@dataclass(frozen=True)
class StoreEntry:
    """dict[(entity, attribute)] -> (value, provenance, timestamp) per protocol §3.1."""

    value: str
    provenance: tuple  # (session_idx, turn_idx, statement_span, confidence)
    timestamp: str  # host session date (injected turns inherit it, P12)


@dataclass(frozen=True)
class Snapshot:
    """Store state immediately after ingesting one user turn."""

    session_idx: int  # ORIGINAL haystack session index
    turn_idx: int
    store: dict  # {(entity, attribute): StoreEntry}


@dataclass
class StoreTimeline:
    question_id: str
    session_order: list[int]  # original session indices, chronological
    snapshots: list[Snapshot]  # one per user turn, chronological
    writes: list[dict] = field(default_factory=list)
    mapping_log: list[dict] = field(default_factory=list)  # key_mapping.jsonl rows
    unmapped: list[dict] = field(default_factory=list)  # unmapped.jsonl rows


# ---------------------------------------------------------------------------
# Value normalization (key_schema.json canonicalization.value_normalization_for_exact_match)
# ---------------------------------------------------------------------------

_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100",
    "once": "1 time", "twice": "2 times", "thrice": "3 times",
}
_NUMBER_WORD_RE = re.compile(r"\b(" + "|".join(_NUMBER_WORDS) + r")\b")
_PERIODS = r"(?:day|week|month|year|hour|minute)"
_ADVERB_FREQ = {
    "daily": "every 1 day", "weekly": "every 1 week",
    "monthly": "every 1 month", "yearly": "every 1 year",
    "annually": "every 1 year",
}


def normalize_value(value: str) -> str:
    """Normalize a value string for the exact-match fast path. Pairs left unequal
    after normalization go to the LLM equivalence judge (protocol §3.1)."""
    v = " ".join(str(value).lower().split())
    v = v.rstrip(".!?,;:")
    v = _NUMBER_WORD_RE.sub(lambda m: _NUMBER_WORDS[m.group(1)], v)
    # frequencies -> '<n> times per <period>' or 'every <n> <period>'
    v = re.sub(rf"\b(\d+) times? (?:a|per|each|every) ({_PERIODS})s?\b", r"\1 times per \2", v)
    v = re.sub(rf"\bevery ({_PERIODS})\b", r"every 1 \1", v)
    v = re.sub(r"\b(daily|weekly|monthly|yearly|annually)\b", lambda m: _ADVERB_FREQ[m.group(1)], v)
    # currency -> '<integer dollars> usd' (whole-string forms)
    m = re.fullmatch(r"\$\s*([\d,]+)(?:\.0+)?(?:\s*(?:usd|dollars))?", v)
    if m is None:
        m = re.fullmatch(r"([\d,]+)(?:\.0+)?\s*(?:usd|dollars)", v)
    if m is not None:
        return f"{m.group(1).replace(',', '')} usd"
    # clock times -> 'h:mm am/pm' (whole-string forms)
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*([ap])\.?m\.?", v)
    if m is not None:
        return f"{int(m.group(1))}:{m.group(2) or '00'} {m.group(3)}m"
    return v


# ---------------------------------------------------------------------------
# Candidate keys
# ---------------------------------------------------------------------------


def load_aux_keys(question_id: str, path: Path = INJECTED_KEYS_PATH) -> list[CanonicalKey]:
    """Load the substrate's registered auxiliary keys from injected_keys.jsonl."""
    keys = []
    # \n-only iteration (splitlines() splits on U+2028/U+2029 — see label.read_jsonl)
    for line in path.open():
        if not line.strip():
            continue
        row = json.loads(line)
        qid = row.get("substrate_qid", row.get("question_id"))
        if qid != question_id:
            continue
        keys.append(
            CanonicalKey(
                entity=row["entity"],
                attribute=row["attribute"],
                description=row.get("description", ""),
                probe_question=row["probe_question"],
            )
        )
    return keys


def build_candidate_keys(key_schema_entry: dict, aux_keys: list[CanonicalKey]) -> list[CanonicalKey]:
    """Gold key from the key_schema entry + registered aux keys. The candidate
    list never carries gold values, injection provenance, or arm labels
    (key_schema canonicalization: the mapping call is blind to those)."""
    gold = CanonicalKey(
        entity=key_schema_entry["entity"],
        attribute=key_schema_entry["attribute"],
        description=f"canonical single-valued attribute (class: {key_schema_entry['attribute_class']})",
        probe_question=key_schema_entry["probe_question"],
    )
    return [gold, *aux_keys]


def _render_key_mapper_prompt(tup: Tuple, candidates: list[CanonicalKey]) -> str:
    lines = ["CANDIDATE KEYS:"]
    for i, cand in enumerate(candidates, 1):
        lines.append(f"{i}. {cand.name} — {cand.description} Probe question: {cand.probe_question}")
    lines += [
        "",
        "EXTRACTED TUPLE:",
        f"entity: {tup.entity}",
        f"attribute: {tup.attribute}",
        f"value: {tup.value}",
    ]
    return "\n".join(lines)


async def _map_tuple_key(
    tup: Tuple, candidates: list[CanonicalKey], *, trace_tag: str
) -> tuple[CanonicalKey | None, str, str]:
    """ONE key_mapper LLM call for one tuple. Returns (matched candidate or None,
    the model's raw mapped_key string, rationale)."""
    output = await call_llm(
        KEY_MAPPER_PROMPT, _render_key_mapper_prompt(tup, candidates), trace_tag=trace_tag
    )
    raw = str(output.get("mapped_key", "none")).strip()
    rationale = str(output.get("rationale", ""))
    by_name = {c.name: c for c in candidates}
    return by_name.get(raw), raw, rationale


# ---------------------------------------------------------------------------
# v2.4 M1 — plan-stem key-mapper mode (protocol/CHANGELOG.md "Protocol v2.4")
# ---------------------------------------------------------------------------

#: The M1 mapping modes. Exactly ONE fix mode exists (the prereg's "exactly
#: one iteration"); "off" is the frozen v2 mapper, byte-preserved because the
#: wrapper below is simply never installed (gated_store.map_stream_tuples
#: mapper=None default).
MAPPING_MODES = ("off", "plan_stem_wrapper")


def _compile_plan_patterns(patterns) -> tuple:
    """Accept the frozen pattern list as strings or precompiled regexes.
    Strings compile with re.IGNORECASE, mirroring the single frozen home
    (router_sweep._PLAN_FRAME_RE); this module never re-declares the list."""
    return tuple(p if hasattr(p, "search") else re.compile(p, re.IGNORECASE)
                 for p in patterns)


def derive_plan_stem(attribute: str, patterns) -> str | None:
    """Strip the plan-frame tokens of a snake_case attribute; None if the
    attribute is not plan-framed or no usable stem survives.

    Token semantics: the attribute is split on "_" and a token is removed iff
    one of the frozen patterns matches it AT ITS POSITION (each token is
    probed with its real boundary context, so the final-token-only "frame"
    pattern only ever strips a final token, and mid-token hits like the
    "plan" inside `houseplant` never strip anything). Separator handling is
    positional by construction: `trip_plan_details` -> `trip_details`.

    Documented edge (v2.4 M1 pinned behavior): ALL plan-frame tokens are
    stripped, including repeated adjacent ones — `x_plan_plan_y` -> `x_y`
    (the review's single-pass span implementation yielded `x_plan_y`; this
    implementation removes every matching token; no such attribute occurs in
    the frozen corpora, behavior pinned in tests/test_mapper_fix.py).

    Empty-stem guard: when every token is a plan-frame token (`plan`,
    `planning`, `intention`, `plan_intention`, ...) there is no stem to
    re-offer -> None. A stem identical to the input also returns None.
    """
    rxs = _compile_plan_patterns(patterns)
    if not any(rx.search(attribute) for rx in rxs):
        return None
    tokens = attribute.split("_")
    n = len(tokens)
    kept = []
    for i, token in enumerate(tokens):
        probe = ("_" if i > 0 else "") + token + ("_" if i < n - 1 else "")
        if token and any(rx.search(probe) for rx in rxs):
            continue
        kept.append(token)
    stem = "_".join(kept)
    if not stem or stem == attribute:
        return None
    return stem


def _plan_stem_and_pattern(attribute: str, patterns) -> tuple[str | None, str | None]:
    """(derived_stem, pattern_matched) for a plan-framed attribute, or
    (None, None). `pattern_matched` is the FIRST pattern in frozen list order
    whose search matches the original attribute (PIN-1 audit field)."""
    rxs = _compile_plan_patterns(patterns)
    matched = next((rx for rx in rxs if rx.search(attribute)), None)
    if matched is None:
        return None, None
    stem = derive_plan_stem(attribute, rxs)
    if stem is None:
        return None, None
    return stem, matched.pattern


def make_tuple_key_mapper(patterns):
    """The v2.4 M1 "plan_stem_wrapper" mapping mode: a decline-then-stem-
    re-offer wrapper around the frozen `_map_tuple_key` call.

    Contract (per tuple, per call):
      1. FIRST pass = `_map_tuple_key` verbatim — the rendered prompt is
         byte-identical to the frozen v2 mapper call, so the pass is
         cache-shared with the frozen runs (zero fresh spend on every tuple
         the frozen cells already mapped or declined).
      2. If the first pass maps, or the attribute is not plan-framed, or no
         stem is derivable: return the first-pass result with meta == {} —
         the byte path is untouched.
      3. If the first pass DECLINES and `derive_plan_stem` yields a stem:
         ONE extra mapping call re-offers the tuple with the stem attribute
         (same candidates, same blindness, trace tag suffixed `-planstem`).
         - stem pass maps  -> (cand, raw, rationale) of the STEM pass, with
           meta {mapping_mode, original_attribute, derived_stem,
           pattern_matched, stem_mapped: True, plan_framed: True,
           first_pass{...}} — the plan_framed/original_attribute pair is the
           classifier-metadata pin (gated_store packages the classifier item
           with the ORIGINAL plan-framed attribute).
         - stem pass declines -> the FIRST-pass row stays the row of record
           (funnel evidence preserved); meta records the attempt
           (stem_mapped: False + stem_pass{...}, no plan_framed key).

    PIN-1 (orchestrator, v2.4 M1): the stem re-offer may map to ANY
    candidate, not provably the one whose plan-variant matched — every
    stem-mapped row is therefore per-row auditable via
    (original_attribute, derived_stem, mapped_key, pattern_matched) in the
    mapping log, and the M1 runner reports the fraction where mapped_key is
    NOT the candidate whose plan-variant matched (diagnostic stratum, no
    bar).

    Returns an async callable `(tup, candidates, *, trace_tag) ->
    (cand | None, raw_key, rationale, meta)`; `meta` is {} on every
    non-stem-path tuple. The 4th element is consumed ONLY by
    gated_store.map_stream_tuples' mapper seam — no frozen call site's tuple
    shape changes (regression-tested in tests/test_mapper_fix.py against the
    recorded v2 mapper outcomes).
    """
    rxs = _compile_plan_patterns(patterns)

    async def mapper(tup: Tuple, candidates: list[CanonicalKey], *, trace_tag: str):
        cand, raw, rationale = await _map_tuple_key(tup, candidates,
                                                    trace_tag=trace_tag)
        if cand is not None:
            return cand, raw, rationale, {}
        stem, pattern = _plan_stem_and_pattern(tup.attribute, rxs)
        if stem is None:
            return cand, raw, rationale, {}
        stem_tup = replace(tup, attribute=stem)
        cand2, raw2, rationale2 = await _map_tuple_key(
            stem_tup, candidates, trace_tag=f"{trace_tag}-planstem")
        meta = {
            "mapping_mode": "plan_stem_wrapper",
            "original_attribute": tup.attribute,
            "derived_stem": stem,
            "pattern_matched": pattern,
            "first_pass": {"mapped_key": "none", "model_output_key": raw,
                           "rationale": rationale},
        }
        if cand2 is None:
            meta["stem_mapped"] = False
            meta["stem_pass"] = {"mapped_key": "none", "model_output_key": raw2,
                                 "rationale": rationale2}
            return None, raw, rationale, meta  # first pass = row of record
        meta["stem_mapped"] = True
        meta["plan_framed"] = True
        return cand2, raw2, rationale2, meta

    mapper.mode = "plan_stem_wrapper"
    mapper.patterns = tuple(rx.pattern for rx in rxs)
    return mapper


# ---------------------------------------------------------------------------
# Write-through replay
# ---------------------------------------------------------------------------


def _chronological_stream(substrate) -> list[dict]:
    """All turns of the substrate in chronological order, tagged with their
    ORIGINAL session index and host session date."""
    stream = []
    for orig_idx in substrate.session_order:
        date = substrate.dates[orig_idx]
        for turn_idx, turn in enumerate(substrate.sessions[orig_idx]):
            stream.append(
                {
                    "session_idx": orig_idx,
                    "turn_idx": turn_idx,
                    "role": turn["role"],
                    "content": turn["content"],
                    "date": date,
                }
            )
    return stream


async def extract_substrate_tuples(substrate) -> list[dict]:
    """Phase-1 extraction: ONE extractor call per user turn over the substrate's
    chronological stream (contract B2). Returns one row per user turn:

      {"session_idx": int (ORIGINAL), "turn_idx": int, "date": str,
       "turn_text": str, "tuples": list[Tuple]}

    This is THE single extraction pathway (R2-B-2): both consumers —
    `replay_write_through` (Test 2) and the Test 1 extractor-tuples producer
    (`extract.py produce-tuples`) — call it on the same substrates and therefore
    render byte-identical prompts, so the llm.py cache guarantees Test 1's
    `s_conf` comes from literally the same extractor calls that populate the
    Test 2 store (protocol §3.1 B2).
    """
    stream = _chronological_stream(substrate)
    user_positions = [i for i, t in enumerate(stream) if t["role"] == "user"]
    extract_coros = []
    for pos in user_positions:
        turn = stream[pos]
        prior = [
            {"role": t["role"], "content": t["content"], "date": t["date"]}
            for t in stream[:pos]
        ]
        extract_coros.append(
            extract_turn(
                prior,
                turn["content"],
                f"extract-{substrate.question_id}-s{turn['session_idx']}t{turn['turn_idx']}",
                current_date=turn["date"],
            )
        )
    tuples_per_turn = await asyncio.gather(*extract_coros)
    return [
        {
            "session_idx": stream[pos]["session_idx"],
            "turn_idx": stream[pos]["turn_idx"],
            "date": stream[pos]["date"],
            "turn_text": stream[pos]["content"],
            "tuples": list(tuples),
        }
        for pos, tuples in zip(user_positions, tuples_per_turn)
    ]


async def replay_write_through(
    substrate, key_schema_entry: dict, manifest_rows: list[dict], *, aux_keys: list[CanonicalKey] | None = None
) -> StoreTimeline:
    """Direct write-through replay of one substrate (protocol §3.1).

    `manifest_rows` are this substrate's injection-manifest rows; when non-empty
    they are spliced into the turns via inject.build_perturbed_substrate before
    replay. `aux_keys` defaults to the substrate's rows of injected_keys.jsonl.
    """
    question_id = substrate.question_id
    if manifest_rows:
        from inject import build_perturbed_substrate

        substrate = build_perturbed_substrate(substrate, manifest_rows)
    if aux_keys is None:
        aux_keys = load_aux_keys(question_id)
    candidates = build_candidate_keys(key_schema_entry, aux_keys)

    # Phase 1 — one extractor call per user turn (B2), via the shared single
    # extraction pathway (extract_substrate_tuples, R2-B-2).
    turn_rows = await extract_substrate_tuples(substrate)

    # Phase 2 — ONE key_mapper call per extracted tuple.
    map_coros = []
    for row in turn_rows:
        for j, tup in enumerate(row["tuples"]):
            map_coros.append(
                _map_tuple_key(
                    tup,
                    candidates,
                    trace_tag=f"keymap-{question_id}-s{row['session_idx']}t{row['turn_idx']}-{j}",
                )
            )
    map_results = list(await asyncio.gather(*map_coros))

    # Phase 3 — fold writes in chronological order; snapshot after every user turn.
    store_state: dict[tuple[str, str], StoreEntry] = {}
    timeline = StoreTimeline(
        question_id=question_id,
        session_order=list(substrate.session_order),
        snapshots=[],
    )
    cursor = 0
    for row in turn_rows:
        turn = row  # per-user-turn row from extract_substrate_tuples
        for tup in row["tuples"]:
            cand, raw_key, rationale = map_results[cursor]
            cursor += 1
            log_row = {
                "substrate_qid": question_id,
                "session_idx": turn["session_idx"],
                "turn_idx": turn["turn_idx"],
                "extracted_tuple": asdict(tup),
                "candidate_keys": [c.name for c in candidates],
                "mapped_key": cand.name if cand is not None else "none",
                "model_output_key": raw_key,
                "rationale": rationale,
            }
            timeline.mapping_log.append(log_row)
            if cand is None:
                timeline.unmapped.append(log_row)  # logged, never silently dropped (FM-6)
                continue
            store_state[cand.key] = StoreEntry(
                value=tup.value,
                provenance=(turn["session_idx"], turn["turn_idx"], tup.statement_span, tup.confidence),
                timestamp=turn["date"],
            )
            timeline.writes.append(
                {
                    "session_idx": turn["session_idx"],
                    "turn_idx": turn["turn_idx"],
                    "key": cand.name,
                    "value": tup.value,
                    "confidence": tup.confidence,
                    "date": turn["date"],
                }
            )
        timeline.snapshots.append(
            Snapshot(session_idx=turn["session_idx"], turn_idx=turn["turn_idx"], store=dict(store_state))
        )
    return timeline


# ---------------------------------------------------------------------------
# Probing (deterministic store lookup; probe points per B4 / injector §12.A)
# ---------------------------------------------------------------------------


def probe(
    timeline: StoreTimeline,
    key: tuple[str, str],
    probe_point: str,
    *,
    turn: tuple[int, int] | None = None,
) -> str | None:
    """Deterministic lookup of `key` at the manifest-designated probe point.

    - full_ingestion: store after the last turn.
    - old_session_boundary: store after the final turn of the chronologically
      earlier session (chronology = timeline.session_order, never dates: P12
      same-day substrates are safe).
    - provenance_timeline (injector §12.A, `gold_key_spec_not_last` rows): store
      immediately after the injection turn — pass `turn=(orig_session_idx,
      turn_idx)` of the manifest placement; reads whether the injected value was
      committed at that turn even if organic later evidence overwrites it.
    """
    key = tuple(key)
    if probe_point == "full_ingestion":
        snap = timeline.snapshots[-1]
    elif probe_point == "old_session_boundary":
        earlier = timeline.session_order[0]
        session_snaps = [s for s in timeline.snapshots if s.session_idx == earlier]
        snap = session_snaps[-1]
    elif probe_point == "provenance_timeline":
        if turn is None:
            raise ValueError("provenance_timeline probe requires turn=(session_idx, turn_idx)")
        matches = [s for s in timeline.snapshots if (s.session_idx, s.turn_idx) == tuple(turn)]
        if not matches:
            raise ValueError(f"no user-turn snapshot at turn {turn}")
        snap = matches[0]
    else:
        raise ValueError(f"unknown probe_point {probe_point!r}; expected one of {PROBE_POINTS}")
    entry = snap.store.get(key)
    return entry.value if entry is not None else None


# ---------------------------------------------------------------------------
# Equivalence judging (protocol §3.1 / FM-5)
# ---------------------------------------------------------------------------


def _judge_pair_order(attribute: str, value_a: str, value_b: str) -> tuple[str, str]:
    """Seeded-random presentation order for the judge (FM-5: pair order
    randomized). Deterministic per unordered pair so reruns hit the cache."""
    lo, hi = sorted([value_a, value_b])
    rng = random.Random(f"{SEED}|{attribute}|{lo}|{hi}")
    return (lo, hi) if rng.random() < 0.5 else (hi, lo)


async def values_equivalent(attribute: str, value_a: str, value_b: str, *, trace_tag: str) -> bool:
    """Normalized exact match first; only unresolved pairs go to the blind LLM
    equivalence judge (attribute + two value strings, nothing else).

    The judge model is PINNED to llm.JUDGE_MODEL (frozen gpt-5.5): this is
    the single judge entry point every scorer routes through
    (rule5.EquivalenceScorer, locomo_render, driver, task_deltas), and the
    scoring instrument must never follow a MEMGATE_SYSTEM_MODEL override
    (2026-08-11 different-family replication prereg, deviation (d)). The
    pinned string equals the historical default, so every existing judge
    cache entry keeps its key."""
    if normalize_value(value_a) == normalize_value(value_b):
        return True
    first, second = _judge_pair_order(attribute, value_a, value_b)
    user_prompt = f"ATTRIBUTE: {attribute}\nVALUE A: {first}\nVALUE B: {second}"
    output = await call_llm(EQUIVALENCE_JUDGE_PROMPT, user_prompt, trace_tag=trace_tag,
                            model=JUDGE_MODEL)
    return bool(output["equivalent"])


# ---------------------------------------------------------------------------
# Key-collision audit log (key_schema.json canonicalization.collision_audit)
# ---------------------------------------------------------------------------


def collision_audit(mapping_rows: list[dict]) -> dict:
    """Log-level collision audit over key_mapping rows.

    Flags per substrate: (merge risk, key_schema audit clause b) canonical keys
    that received tuples under more than one distinct raw (entity, attribute)
    wording; (split risk, clause a) one raw wording mapped to more than one
    canonical key. These flags are the input queue for the judge-based audit in
    pilot/analysis/ — raw-wording diversity on one key is not itself a
    violation, it is what the audit must inspect.
    """
    key_to_raw: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    raw_to_key: dict[str, dict[tuple, set]] = defaultdict(lambda: defaultdict(set))
    for row in mapping_rows:
        if row["mapped_key"] == "none":
            continue
        qid = row["substrate_qid"]
        raw = (row["extracted_tuple"]["entity"], row["extracted_tuple"]["attribute"])
        key_to_raw[qid][row["mapped_key"]].add(raw)
        raw_to_key[qid][raw].add(row["mapped_key"])
    merge_flags = [
        {"substrate_qid": qid, "canonical_key": key, "raw_pairs": sorted(map(list, raws))}
        for qid in sorted(key_to_raw)
        for key, raws in sorted(key_to_raw[qid].items())
        if len(raws) > 1
    ]
    split_flags = [
        {"substrate_qid": qid, "raw_pair": list(raw), "canonical_keys": sorted(keys)}
        for qid in sorted(raw_to_key)
        for raw, keys in sorted(raw_to_key[qid].items())
        if len(keys) > 1
    ]
    return {
        "n_merge_risk_flags": len(merge_flags),
        "n_split_risk_flags": len(split_flags),
        "merge_risk_flags": merge_flags,
        "split_risk_flags": split_flags,
    }


# ---------------------------------------------------------------------------
# Log writing (overwrite-on-run so reruns stay idempotent)
# ---------------------------------------------------------------------------


def dump_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
