"""adapter_d1.py — Dataset 1 (standalone conversations) -> runner streams
(external-dataset adapter phase, protocol/CHANGELOG.md 2026-07-30
"EXTERNAL DATASET VALIDATION COMPLETE; adapter phase authorized").

INPUT (model-visible, per the package README use boundary):
  * eval_sets/raw_inputs/dataset1_conversations.jsonl — 200 conversations of
    FLAT turns with per-turn ISO timestamps (sessions are delimited by
    calendar day, per the package schema).
  * eval_sets/raw_inputs/dataset1_test_questions.jsonl — the value-free probe
    questions (one per conversation).

HELD-OUT input (eval_sets/labels_or_answer_keys/dataset1_labels.jsonl), used
for EXACTLY ONE thing here: the FIELD REGISTRY — each conversation registers
ONE canonical key whose attribute comes from the label's ``attribute`` name
(construction metadata, not an answer; the authorized-phase entry pins this).
No other label field is read on the adapter path, and NO label value
(expected_answer / old_value / new_value / final_value) may reach any
model-visible payload — enforced by a hard assert on the full rendered
payload of every adapted conversation (``_assert_no_label_leak``).

OUTPUT (``adapt(conversation_id)``):
  * stream: the m4_eval conversation shape — session-grouped turns
    ``{"session_idx", "date", "turns": [{"role", "text"}]}`` with ONE
    session-level date marker per session (the frozen LME B2 rendering
    contract), date strings in the corpus-wide "%Y/%m/%d (%a) %H:%M" format
    (the LoCoMo/LME-style format every frozen prompt already sees), taken
    from the session's first turn.
  * key_row: the registered field — entity "user", attribute = snake_case of
    the label attribute name, description templated from the attribute name
    only, probe_question = the MODEL-VISIBLE question text
    (dataset1_test_questions.jsonl). FM-4 hygiene asserted: no label value
    appears in attribute/description/probe.

HOLDOUT GUARD: ``adapt`` calls ``holdout_guard.assert_ingest_allowed`` — a
TEST-slice conversation raises unless the one-shot run is authorized.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from functools import cache

import _bootstrap  # noqa: F401  (R1 layout: puts src/ + experiments/ on sys.path)

import holdout_guard
import store

EXTERNAL_PKG = holdout_guard.EXTERNAL_PKG
CONVERSATIONS_PATH = EXTERNAL_PKG / "eval_sets" / "raw_inputs" / "dataset1_conversations.jsonl"
QUESTIONS_PATH = EXTERNAL_PKG / "eval_sets" / "raw_inputs" / "dataset1_test_questions.jsonl"
#: HELD OUT — attribute wiring only on the adapter path; expected answers are
#: read at SCORING time only (runner_external.score_dev), never in a run path.
LABELS_PATH = EXTERNAL_PKG / "eval_sets" / "labels_or_answer_keys" / "dataset1_labels.jsonl"

DATASET = "dataset1"
ENTITY = "user"


def _read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


@cache
def conversations() -> dict[str, dict]:
    return {c["conversation_id"]: c for c in _read_jsonl(CONVERSATIONS_PATH)}


@cache
def questions() -> dict[str, dict]:
    return {q["question_id"]: q for q in _read_jsonl(QUESTIONS_PATH)}


@cache
def _labels() -> dict[str, dict]:
    """HELD-OUT labels keyed by record (conversation) id. Adapter path reads
    ``attribute`` only; ``_assert_no_label_leak`` reads the value fields to
    PROVE they are absent from every model-visible payload (a negative check
    is not ingestion). Scoring-time access goes through
    ``label_for`` (explicitly named scoring seam)."""
    return {l["record_id"]: l for l in _read_jsonl(LABELS_PATH)}


def label_for(conversation_id: str, *, scoring: bool) -> dict:
    """The held-out label row. ``scoring=True`` is the ONLY sanctioned full
    access (deterministic-lookup + judge comparison at scoring time)."""
    assert scoring, (
        "label_for(scoring=False) is not a thing — run-path code must never "
        "read label rows; the adapter uses attribute wiring internally only")
    return _labels()[conversation_id]


# ---------------------------------------------------------------------------
# Session grouping + dates
# ---------------------------------------------------------------------------

def _fmt_date(iso_ts: str) -> str:
    dt = datetime.fromisoformat(iso_ts)
    return f"{dt:%Y/%m/%d} ({dt:%a}) {dt:%H:%M}"


def sessions_for(conv: dict) -> list[dict]:
    """Group the flat turn list into calendar-day sessions (the package's
    session delimiter), preserving order; assert timestamps monotonic."""
    turns = conv["turns"]
    stamps = [datetime.fromisoformat(t["timestamp"]) for t in turns]
    assert all(a <= b for a, b in zip(stamps, stamps[1:])), (
        f"{conv['conversation_id']}: non-monotonic turn timestamps")
    sessions: list[dict] = []
    for turn in turns:
        day = turn["timestamp"][:10]
        if not sessions or sessions[-1]["_day"] != day:
            sessions.append({"_day": day,
                             "session_idx": len(sessions),
                             "date": _fmt_date(turn["timestamp"]),
                             "turns": []})
        assert turn["speaker"] in ("user", "assistant"), (
            f"{conv['conversation_id']}: unexpected speaker {turn['speaker']!r}")
        sessions[-1]["turns"].append({"role": turn["speaker"],
                                      "text": turn["text"]})
    for s in sessions:
        del s["_day"]
    return sessions


# ---------------------------------------------------------------------------
# Field registry (ONE field per conversation)
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(attribute: str) -> str:
    return _SLUG_RE.sub("_", attribute.strip().lower()).strip("_")


def field_registry_row(conversation_id: str) -> dict:
    """The conversation's ONE registered field (m4_keys.jsonl row shape).
    Attribute name from the held-out label (construction metadata, per the
    authorized-phase entry); probe question from the MODEL-VISIBLE questions
    file; description templated from the attribute name only."""
    conv = conversations()[conversation_id]
    label = _labels()[conversation_id]
    qids = conv["test_question_ids"]
    assert len(qids) == 1, f"{conversation_id}: expected 1 question, got {qids}"
    q = questions()[qids[0]]
    assert q["record_id"] == conversation_id, (
        f"{conversation_id}: question {qids[0]} wired to {q['record_id']}")
    attr_name = label["attribute"]
    row = {
        "conversation_id": conversation_id,
        "question_id": qids[0],
        "entity": ENTITY,
        "attribute": _slug(attr_name),
        "description": f"the user's current {attr_name}",
        "probe_question": q["question_text"],
        "key_source": "external_d1_label_attribute_plus_visible_question",
        "fm4_note": ("attribute from the held-out label's attribute NAME "
                     "(construction metadata, never an answer); probe is the "
                     "model-visible test question; no label value may appear "
                     "in any registry field (asserted)"),
    }
    _assert_registry_value_free(row, label)
    return row


def canonical_key(key_row: dict) -> store.CanonicalKey:
    return store.CanonicalKey(entity=key_row["entity"],
                              attribute=key_row["attribute"],
                              description=key_row["description"],
                              probe_question=key_row["probe_question"])


def _label_values(label: dict) -> list[str]:
    return [str(v) for v in (label.get("expected_answer"),
                             label.get("old_value"),
                             label.get("new_value"),
                             label.get("final_value")) if v]


def _assert_registry_value_free(row: dict, label: dict) -> None:
    """FM-4-style hygiene: no label VALUE in any registry field. (The intake
    ladder hard-gated question hygiene, so a trip here is a real leak.)"""
    surface = " | ".join([row["attribute"], row["description"],
                          row["probe_question"]]).lower()
    for v in _label_values(label):
        assert v.lower() not in surface, (
            f"{row['conversation_id']}: label value {v!r} appears in the "
            "field registry — refusing to register a leaking key")


# ---------------------------------------------------------------------------
# The adapted conversation (guarded)
# ---------------------------------------------------------------------------

def model_visible_payload(stream: dict, key_row: dict) -> str:
    """EVERYTHING the adapter makes model-visible for one conversation, as
    one string: every turn text, every session date, and the registered key's
    prompt-facing fields (the mapper prompt renders name/description/probe).
    The leak assert runs over exactly this surface."""
    parts = [key_row["attribute"], key_row["description"],
             key_row["probe_question"]]
    for sess in stream["sessions"]:
        parts.append(sess["date"])
        parts.extend(t["text"] for t in sess["turns"])
    return "\n".join(parts)


def _assert_no_label_leak(stream: dict, key_row: dict, label: dict) -> None:
    """THE prereg assert: the expected_answer string (and every other label
    value not naturally part of the conversation) never enters a
    model-visible payload. expected_answer is a full held-out sentence, so
    its presence anywhere model-visible is an unconditional failure."""
    payload = model_visible_payload(stream, key_row)
    ea = label["expected_answer"]
    assert ea and ea not in payload, (
        f"{stream['conversation_id']}: expected_answer string found in a "
        "model-visible payload — adapter leak, hard stop")
    # registry fields additionally checked value-free in field_registry_row
    # (old/new/final values legitimately occur inside the conversation text
    # itself — that is the dataset; they must never ride on adapter-added
    # surfaces).


def adapt(conversation_id: str) -> tuple[dict, dict]:
    """(stream, key_row) for one conversation. HARD holdout guard + leak
    asserts run on every call."""
    holdout_guard.assert_ingest_allowed(DATASET, conversation_id)
    conv = conversations()[conversation_id]
    stream = {
        "conversation_id": conversation_id,
        "dataset": DATASET,
        "sessions": sessions_for(conv),
    }
    key_row = field_registry_row(conversation_id)
    _assert_no_label_leak(stream, key_row, _labels()[conversation_id])
    return stream, key_row


def stream_turns(stream: dict) -> list[dict]:
    """Flat chronological rows {session_idx, turn_idx, role, content, date}
    (the m4_eval.arc_turns shape the frozen seams consume)."""
    rows = []
    for sess in stream["sessions"]:
        for ti, t in enumerate(sess["turns"]):
            rows.append({"session_idx": sess["session_idx"], "turn_idx": ti,
                         "role": t["role"], "date": sess["date"],
                         "content": t["text"]})
    return rows
