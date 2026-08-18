"""LoCoMo dataset access for protocol v2 (role W3) — Substrate-compatible loader.

Parses ``../data/locomo/locomo10.json`` (10 dyadic conversations, 272 sessions,
5,882 turns) into per-conversation ``LocomoConversation`` objects that follow the
INTERFACES ``data.py`` conventions: a dataclass with ``sessions`` / ``dates`` /
``session_order`` addressable by original index, cached raw parse, fresh copies
per accessor call. Differences from LongMemEval are dyadic-specific and pinned by
protocol v2 §2.2 (V2-P2) and the §2.2.5 data-hygiene rules from
``research/locomo_analysis.md`` §5:

- Turns carry ``speaker`` (there is no user/assistant role) and ``dia_id``
  (``D{session}:{turn}``, both 1-indexed, round-trippable). Image turns keep
  ``blip_caption`` (rendered as ``[shares photo: <caption>]`` by
  ``locomo_render``); other image fields (img_url/query/re-download) are
  dropped at load — captions are auto-generated and never assertions.
- **conv-26 dangling-key anomaly (documented, skipped):** conv-26 carries
  ``session_20_date_time`` … ``session_35_date_time`` keys with NO corresponding
  ``session_20`` … ``session_35`` content (16 dangling date keys). Session
  iteration therefore keys on ``session_N`` presence, never on the date keys;
  the dangling numbers are exposed via ``dangling_session_dates`` for audit.
- QA evidence pointers are regex-extracted defensively (§2.2.5): the corpus has
  6 malformed pointer strings and 2 out-of-range pointers; ``resolve_evidence``
  returns them explicitly instead of raising.

Also exposes the frozen LoCoMo gold schema (``research/key_schema_v2_locomo.json``,
V2-P7) consumed by the deterministic probe engine in ``locomo_render``:
qa_probe_map (98 pre-mapped QA rows, P-NATQA), time_indexed_probes (36 items,
13 key-mapped, P-TIME), cat5_misattribution_sample (100 seeded items, P-MISATTR),
update_keys and censored_staged_hypotheses.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from functools import cache
from pathlib import Path

from paths import DATA_DIR, GOLD_DIR

LOCOMO_PATH = DATA_DIR / "locomo" / "locomo10.json"
LOCOMO_SCHEMA_PATH = GOLD_DIR / "key_schema_v2_locomo.json"

# e.g. "1:56 pm on 8 May, 2023" (schema _meta.dia_id_convention)
SESSION_DATE_FORMAT = "%I:%M %p on %d %B, %Y"

_SESSION_KEY_RE = re.compile(r"session_(\d+)")
_DATE_KEY_RE = re.compile(r"session_(\d+)_date_time")
# Defensive extraction (search anywhere) vs strict single-pointer form (fullmatch).
DIA_ID_RE = re.compile(r"D(\d+):(\d+)")
_DIA_ID_STRICT_RE = re.compile(r"D\d+:\d+")

_TURN_FIELDS = ("speaker", "dia_id", "text", "blip_caption")


@dataclass
class LocomoConversation:
    """One LoCoMo conversation in the Substrate-compatible shape.

    ``sessions[i]`` is the turn list of original session number
    ``session_numbers[i]`` (1-indexed in the corpus; contiguous 1..N in
    locomo10.json — asserted at load) and ``dates[i]`` its
    ``session_N_date_time`` string. ``session_order`` lists indices into
    ``sessions`` sorted chronologically by parsed date — computed, not assumed,
    exactly like ``data.Substrate.session_order`` (LoCoMo dates happen to be
    strictly monotonic, so it is the identity permutation on this corpus).
    Turn dicts are exactly ``{speaker, dia_id, text, blip_caption}`` with
    ``blip_caption=None`` on non-image turns.
    """

    sample_id: str
    speaker_a: str
    speaker_b: str
    sessions: list[list[dict]]
    dates: list[str]
    session_numbers: list[int]
    session_order: list[int]

    @property
    def speakers(self) -> tuple[str, str]:
        return (self.speaker_a, self.speaker_b)


def parse_session_date(date: str) -> datetime:
    """Parse a LoCoMo session date like ``1:56 pm on 8 May, 2023``."""
    return datetime.strptime(date, SESSION_DATE_FORMAT)


def chronological_order(dates: list[str]) -> list[int]:
    """Indices of ``dates`` sorted chronologically (stable on exact ties)."""
    return sorted(range(len(dates)), key=lambda i: parse_session_date(dates[i]))


@cache
def load_locomo() -> list[dict]:
    """Parse ``../data/locomo/locomo10.json`` (10 conversations).

    The parsed list is cached and shared; callers must not mutate it
    (``get_conversation`` hands out fresh copies).
    """
    return json.loads(LOCOMO_PATH.read_text())


@cache
def _locomo_by_id() -> dict[str, dict]:
    return {conv["sample_id"]: conv for conv in load_locomo()}


def conversation_ids() -> list[str]:
    """The 10 sample_ids in corpus order (conv-26 … conv-50)."""
    return [conv["sample_id"] for conv in load_locomo()]


def _session_numbers(conversation: dict) -> list[int]:
    """Session numbers with CONTENT, keyed on ``session_N`` presence (§2.2.5:
    conv-26's dangling ``session_20..35_date_time`` keys must never be iterated)."""
    numbers = sorted(
        int(m.group(1))
        for key in conversation
        if (m := _SESSION_KEY_RE.fullmatch(key)) is not None
    )
    return numbers


def dangling_session_dates(sample_id: str) -> list[int]:
    """Session numbers that have a ``session_N_date_time`` key but no
    ``session_N`` content (the conv-26 anomaly: 16 dangling keys, 20..35;
    empty for every other conversation). Exposed for audit; the loader skips
    them by construction."""
    conversation = _locomo_by_id()[sample_id]["conversation"]
    with_content = set(_session_numbers(conversation))
    with_date = {
        int(m.group(1))
        for key in conversation
        if (m := _DATE_KEY_RE.fullmatch(key)) is not None
    }
    return sorted(with_date - with_content)


def get_conversation(sample_id: str) -> LocomoConversation:
    """Build a ``LocomoConversation`` (fresh copies each call)."""
    raw = _locomo_by_id()[sample_id]
    conversation = raw["conversation"]
    numbers = _session_numbers(conversation)
    if numbers != list(range(1, len(numbers) + 1)):  # contract check, not a guess
        raise ValueError(f"{sample_id}: non-contiguous session numbers {numbers}")
    sessions: list[list[dict]] = []
    dates: list[str] = []
    for n in numbers:
        sessions.append(
            [
                {
                    "speaker": turn["speaker"],
                    "dia_id": turn["dia_id"],
                    "text": turn["text"],
                    "blip_caption": turn.get("blip_caption"),
                }
                for turn in conversation[f"session_{n}"]
            ]
        )
        dates.append(conversation[f"session_{n}_date_time"])
    return LocomoConversation(
        sample_id=sample_id,
        speaker_a=raw["conversation"]["speaker_a"],
        speaker_b=raw["conversation"]["speaker_b"],
        sessions=sessions,
        dates=dates,
        session_numbers=numbers,
        session_order=chronological_order(dates),
    )


def dialog_turns(conv: LocomoConversation) -> list[tuple[int, int, dict]]:
    """(session_idx, turn_idx, turn) for EVERY turn, in chronological order.

    The dyadic analog of ``data.user_turns``: LoCoMo has no privileged role, and
    the single-pass dyadic extraction (V2-P2) makes one extractor call per turn,
    both speakers included. ``session_idx`` indexes ``conv.sessions`` (0-based);
    the corpus session number is ``conv.session_numbers[session_idx]``.
    """
    turns = []
    for orig_idx in conv.session_order:
        for turn_idx, turn in enumerate(conv.sessions[orig_idx]):
            turns.append((orig_idx, turn_idx, turn))
    return turns


# ---------------------------------------------------------------------------
# dia_id round-trip (D{session}:{turn}, both 1-indexed)
# ---------------------------------------------------------------------------


def format_dia_id(session_number: int, turn_number: int) -> str:
    """Compose a dia_id from 1-indexed session and turn numbers."""
    return f"D{session_number}:{turn_number}"


def parse_dia_id(dia_id: str) -> tuple[int, int]:
    """Strict parse of a well-formed dia_id -> (session_number, turn_number),
    both 1-indexed. Raises ValueError on malformed strings (use
    ``resolve_evidence`` for defensive extraction from QA evidence)."""
    m = DIA_ID_RE.fullmatch(dia_id)
    if m is None:
        raise ValueError(f"malformed dia_id: {dia_id!r}")
    return (int(m.group(1)), int(m.group(2)))


def turn_by_dia_id(conv: LocomoConversation, dia_id: str) -> tuple[int, int, dict]:
    """Resolve a dia_id to (session_idx, turn_idx, turn). Raises KeyError when
    the pointer does not resolve (dangling session or out-of-range turn)."""
    session_number, turn_number = parse_dia_id(dia_id)
    try:
        session_idx = conv.session_numbers.index(session_number)
    except ValueError:
        raise KeyError(f"{conv.sample_id}: no session {session_number} for {dia_id!r}") from None
    session = conv.sessions[session_idx]
    if not 1 <= turn_number <= len(session):
        raise KeyError(f"{conv.sample_id}: turn {dia_id!r} out of range (session has {len(session)} turns)")
    return (session_idx, turn_number - 1, session[turn_number - 1])


# ---------------------------------------------------------------------------
# QA access + defensive evidence resolution (§2.2.5)
# ---------------------------------------------------------------------------


def load_qa(sample_id: str) -> list[dict]:
    """The conversation's raw QA list (cached, shared — do not mutate). Schema
    artifacts address items by (sample_id, qa_index) = position in this list."""
    return _locomo_by_id()[sample_id]["qa"]


def qa_item(sample_id: str, qa_index: int) -> dict:
    return load_qa(sample_id)[qa_index]


def question_sha1(question: str) -> str:
    """10-hex-char question fingerprint used by the frozen schema rows."""
    return hashlib.sha1(question.encode()).hexdigest()[:10]


def resolve_evidence(conv: LocomoConversation, evidence: list[str]) -> dict:
    """Defensively resolve a QA item's evidence pointer strings (§2.2.5).

    Pointers are mined with ``DIA_ID_RE.finditer`` from every string, so
    semicolon/space-joined lists still yield their pointers. Returns::

        {"resolved":     [(session_number, turn_number), ...],  # order kept
         "malformed":    [str, ...],   # strings not in strict D{s}:{t} form
         "out_of_range": [(session_number, turn_number), ...]}  # mined, unresolvable

    Corpus-wide there are exactly 6 malformed strings and 2 out-of-range
    pointers (locomo_analysis.md §1.4); nothing is silently dropped.
    """
    resolved: list[tuple[int, int]] = []
    malformed: list[str] = []
    out_of_range: list[tuple[int, int]] = []
    for pointer_str in evidence:
        if _DIA_ID_STRICT_RE.fullmatch(pointer_str) is None:
            malformed.append(pointer_str)
        for m in DIA_ID_RE.finditer(pointer_str):
            session_number, turn_number = int(m.group(1)), int(m.group(2))
            try:
                turn_by_dia_id(conv, format_dia_id(session_number, turn_number))
            except KeyError:
                out_of_range.append((session_number, turn_number))
            else:
                resolved.append((session_number, turn_number))
    return {"resolved": resolved, "malformed": malformed, "out_of_range": out_of_range}


# ---------------------------------------------------------------------------
# Frozen LoCoMo gold schema (research/key_schema_v2_locomo.json, V2-P7)
# ---------------------------------------------------------------------------


@cache
def load_locomo_schema() -> dict:
    """Parse the frozen LoCoMo key schema + probe map (cached, shared — do not
    mutate)."""
    return json.loads(LOCOMO_SCHEMA_PATH.read_text())


def update_keys() -> list[dict]:
    """The 38 gold natural-update keys (47 transitions)."""
    return load_locomo_schema()["update_keys"]


def qa_probe_map_rows() -> list[dict]:
    """The 98 pre-mapped QA rows (P-NATQA; tiers primary/history/secondary)."""
    return load_locomo_schema()["qa_probe_map"]["mapped"]


def time_indexed_probes() -> list[dict]:
    """The 36 distinct time-indexed QA items; exactly 13 carry a key_id and are
    P-TIME-scoreable (V2-F4)."""
    return load_locomo_schema()["time_indexed_probes"]["items"]


def cat5_sample() -> list[dict]:
    """The frozen 100-item seeded cat-5 sample (75 speaker_swap primary +
    25 focal_fact_distortion exploratory; P-MISATTR)."""
    return load_locomo_schema()["cat5_misattribution_sample"]["items"]


def censored_staged_hypotheses() -> list[dict]:
    """The 4 censored natural staged hypotheses (NAT-CENS denominator)."""
    return load_locomo_schema()["censored_staged_hypotheses"]


def parse_key_id(key_id: str) -> tuple[str, str, str]:
    """Split a schema key_id like ``conv-26/melanie/pet_roster`` into
    (sample_id, entity, attribute). The store key is (entity, attribute)
    within the entity's speaker-store of that conversation."""
    parts = key_id.split("/")
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"malformed key_id: {key_id!r}")
    return (parts[0], parts[1], parts[2])
