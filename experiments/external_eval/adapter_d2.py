"""adapter_d2.py — Dataset 2 insertion patches -> AUGMENTED REAL HOST streams
(external-dataset adapter phase, protocol/CHANGELOG.md 2026-07-30).

Each patch is applied to its REAL host record:

  * LongMemEval hosts resolve against ``data/longmemeval/longmemeval_oracle
    .json`` — NOT s_cleaned. This is the rung-(a) r2 conformance caveat
    (check H2, results/external_intake at the r2 revision): the patches
    anchor to the ORACLE haystack dates (63/63 exact display_timestamp
    match), while longmemeval_s_cleaned RE-DATES the same sessions (exact
    date match only 4/63 there). Applying to the oracle keeps every
    allowlisted locator byte-consistent with the host; the adapter ASSERTS
    the date equality per patch instead of re-deriving dates.
  * LoCoMo hosts resolve against ``data/locomo/locomo10.json`` via
    ``locomo_data.get_conversation`` (fresh copies — the shared parse is
    never mutated).

INGEST ALLOWLIST (the r2-documented list: package README r2 §"Dataset 2
Host-Application Ingest Allowlist" = qa_reports/dataset2_host_reanchoring_
audit.md §9 at the r2 revision; the r4 README dropped the section (K4
regression, changelog 2026-07-29) — this adapter enforces it regardless):
only ``host_dataset``, ``host_record_id``, ``host_record_ref_policy``,
``target_path``, ``insert_after``, ``timestamp``, ``display_timestamp``,
``speaker_or_role``/``speaker_name`` (LoCoMo ``speaker_a``/``speaker_b`` for
placement only), ``turns``, ``supersedes``, and the LoCoMo locators
``session_key``, ``session_date_time_key``, ``dia_id_sequence``,
``dialog_id_start``, ``dialog_id_end`` are read when applying a patch. ALL
construction/experiment metadata (``insertion_family``, ``wording_distance``,
``mixed_sentence_type``, ``correction_pair_type``, ``calibration_strength``,
``memory_operation``, ``patch_intent``, ``plan_update_type``, ``cluster_id``,
``same_plan_group_id``, ``variant_index``, ...) is DROPPED before anything
model-visible is built, enforced by construction (the allowlisted view is
the only patch object the application code ever sees) and by a lint that no
augmented output carries a banned key. ``patch_id`` is retained as
BOOKKEEPING ONLY (case-record naming + the holdout guard); it is never
rendered into a prompt payload.

OUTPUT (``apply_patch(patch_id)``): an augmented host in the host family's
native replay shape —
  * LME: ``{"sessions": [{session_idx, session_id, date, turns:[{role,
    content, inserted}]}] chronological, "insertion": {...}, "superseded":
    [...]}`` (host ``has_answer`` flags dropped from the stream rows);
  * LoCoMo: a fresh augmented ``LocomoConversation`` (patch turns appended
    to the target session — every Dataset 2 LoCoMo patch is end-of-session,
    asserted) plus the same bookkeeping meta, including
    ``replay_session_idxs`` = the patched session + every superseded
    session (the runner's bounded replay window; graded_eval bounded-world
    precedent, disclosed there).

HOLDOUT GUARD: ``apply_patch`` calls ``holdout_guard.assert_ingest_allowed``
— TEST-slice patches raise unless the one-shot run is authorized.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from functools import cache

import _bootstrap  # noqa: F401  (R1 layout: puts src/ + experiments/ on sys.path)

import holdout_guard
import locomo_data

from paths import DATA_DIR

EXTERNAL_PKG = holdout_guard.EXTERNAL_PKG
PATCHES_PATH = EXTERNAL_PKG / "source_artifacts" / "dataset2_insertion_patches.jsonl"
LME_ORACLE_PATH = DATA_DIR / "longmemeval" / "longmemeval_oracle.json"

DATASET = "dataset2"

#: The r2-documented ingest allowlist (README r2 / audit §9; see docstring).
ALLOWLIST = frozenset({
    "host_dataset", "host_record_id", "host_record_ref_policy",
    "target_path", "insert_after", "timestamp", "display_timestamp",
    "session_id",  # r6: authoritative LME session locator (see _apply_lme)
    "speaker_or_role", "speaker_name", "speaker_a", "speaker_b",
    "turns", "supersedes",
    "session_key", "session_date_time_key", "dia_id_sequence",
    "dialog_id_start", "dialog_id_end",
})

#: Construction/experiment metadata named by the allowlist doc as MUST-DROP.
BANNED_METADATA = frozenset({
    "insertion_family", "wording_distance", "mixed_sentence_type",
    "correction_pair_type", "calibration_strength", "memory_operation",
    "patch_intent", "plan_update_type", "cluster_id", "same_plan_group_id",
    "variant_index", "question_type_target", "fact_handle",
    "event_locator_types", "everyday_domain", "validation_notes",
    "expected_visibility", "leakage_guard", "label_policy",
    "answer_field_policy", "has_answer_policy", "qa_policy",
    "event_summary_policy", "affects_question_date",
})


def _read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


@cache
def patches() -> dict[str, dict]:
    return {p["patch_id"]: p for p in _read_jsonl(PATCHES_PATH)}


@cache
def _lme_oracle() -> dict[str, dict]:
    return {r["question_id"]: r for r in json.loads(LME_ORACLE_PATH.read_text())}


def allowlisted(patch: dict) -> dict:
    """The ONLY patch view the host-application code consumes: allowlisted
    fields + ``patch_id`` (bookkeeping only, never rendered)."""
    view = {k: v for k, v in patch.items() if k in ALLOWLIST and v is not None}
    view["patch_id"] = patch["patch_id"]
    return view


def _assert_no_banned_keys(obj, where: str) -> None:
    """Lint: no construction-metadata key anywhere in an augmented output."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert k not in BANNED_METADATA, (
                f"{where}: construction-metadata key {k!r} leaked into an "
                "augmented output")
            _assert_no_banned_keys(v, where)
    elif isinstance(obj, list):
        for v in obj:
            _assert_no_banned_keys(v, where)


# ---------------------------------------------------------------------------
# LongMemEval application (oracle hosts — see the date-caveat note above)
# ---------------------------------------------------------------------------

_LME_DATE_FMT = "%Y/%m/%d (%a) %H:%M"


def _parse_lme_date(date: str) -> datetime:
    return datetime.strptime(date, _LME_DATE_FMT)


def _apply_lme(view: dict) -> dict:
    host = _lme_oracle().get(view["host_record_id"])
    assert host is not None, (
        f"{view['patch_id']}: host {view['host_record_id']!r} not in the "
        "LongMemEval ORACLE (the declared anchor source)")
    m = re.fullmatch(r"/haystack_sessions/(\d+)", view["target_path"])
    assert m, f"{view['patch_id']}: bad target_path {view['target_path']!r}"
    # r6 locator convention (PROVENANCE Revision 5): target_path indices are
    # computed against longmemeval_s_cleaned, but session_id + display dates
    # still anchor to the ORACLE. session_id is the authoritative locator;
    # the numeric index is not trusted (r2..r5 packages indexed the oracle,
    # r6 indexes s_cleaned — resolving by id is correct for both).
    ia = re.fullmatch(r"(.+):(\d+)", view["insert_after"])
    assert ia, (
        f"{view['patch_id']}: bad insert_after {view['insert_after']!r}")
    sess_id = view["session_id"]
    assert ia.group(1) == sess_id, (
        f"{view['patch_id']}: insert_after {view['insert_after']!r} does not "
        f"name the patch session_id {sess_id!r}")
    id_hits = [i for i, s in enumerate(host["haystack_session_ids"])
               if s == sess_id]
    assert len(id_hits) == 1, (
        f"{view['patch_id']}: session_id {sess_id!r} resolves to "
        f"{len(id_hits)} oracle sessions (need exactly 1)")
    idx = id_hits[0]
    after = int(ia.group(2))
    assert after < len(host["haystack_sessions"][idx]), (
        f"{view['patch_id']}: insert_after turn {after} out of range")
    # THE ORACLE-DATE CAVEAT ASSERT: the patch's display_timestamp must equal
    # the REAL oracle session date (rung-(a) r2 H2 verified 63/63; s_cleaned
    # would fail here — that is why the oracle is the application surface).
    assert view["display_timestamp"] == host["haystack_dates"][idx], (
        f"{view['patch_id']}: display_timestamp "
        f"{view['display_timestamp']!r} != oracle haystack date "
        f"{host['haystack_dates'][idx]!r} — wrong host surface?")

    inserted_span = []
    sessions = []
    for si, (sid, date, turns) in enumerate(zip(host["haystack_session_ids"],
                                                host["haystack_dates"],
                                                host["haystack_sessions"])):
        rows = [{"role": t["role"], "content": t["content"], "inserted": False}
                for t in turns]  # host has_answer flags dropped
        if si == idx:
            patch_rows = [{"role": t["role"], "content": t["content"],
                           "inserted": True} for t in view["turns"]]
            rows = rows[:after + 1] + patch_rows + rows[after + 1:]
            inserted_span = [{"session_idx": si, "turn_idx": after + 1 + j}
                             for j in range(len(patch_rows))]
        sessions.append({"session_idx": si, "session_id": sid, "date": date,
                         "turns": rows})
    # chronological session order (the replay contract; oracle haystack
    # list order is NOT chronological), stable on exact-date ties.
    order = sorted(range(len(sessions)),
                   key=lambda i: (_parse_lme_date(sessions[i]["date"]), i))
    superseded = []
    for loc in view.get("supersedes", []):
        self_ref = re.fullmatch(r"(D2-PATCH-\d+):turn:(\d+)", loc)
        if self_ref:
            # r6 correction pairs are self-contained: the patch inserts the
            # claim AND its correction; supersedes names the patch's OWN
            # inserted turn (1-based over the patch turn list).
            assert self_ref.group(1) == view["patch_id"], (
                f"{view['patch_id']}: cross-patch supersedes {loc!r} "
                "unsupported")
            k = int(self_ref.group(2))
            assert 1 <= k <= len(view["turns"]), (
                f"{view['patch_id']}: supersedes turn {k} outside the patch "
                f"turn list (1..{len(view['turns'])})")
            superseded.append({"session_idx": idx,
                               "turn_idx": after + 1 + (k - 1),
                               "locator": loc})
            continue
        sm = re.fullmatch(r"(.+):(\d+)", loc)
        assert sm and sm.group(1) in host["haystack_session_ids"], (
            f"{view['patch_id']}: unresolvable supersedes locator {loc!r}")
        s_idx = host["haystack_session_ids"].index(sm.group(1))
        t_idx = int(sm.group(2))
        assert t_idx < len(host["haystack_sessions"][s_idx]), (
            f"{view['patch_id']}: supersedes turn out of range {loc!r}")
        if s_idx == idx and t_idx > after:
            # mid-session insertion shifts later host turns in the AUGMENTED
            # coordinates the record consumers read
            t_idx += len(view["turns"])
        superseded.append({"session_idx": s_idx, "turn_idx": t_idx,
                           "locator": loc})
    return {
        "patch_id": view["patch_id"],
        "host_dataset": "longmemeval",
        "host_record_id": view["host_record_id"],
        "sessions": [sessions[i] for i in order],
        "session_order_original_idx": order,
        "insertion": {"session_idx": idx, "session_id": sess_id,
                      "insert_after_turn": after, "turn_span": inserted_span,
                      "date": host["haystack_dates"][idx]},
        "superseded": superseded,
    }


def lme_stream_turns(aug: dict) -> list[dict]:
    """Flat chronological rows {session_idx, turn_idx, role, content, date,
    inserted} over the augmented LME host (the graded_eval 'lme' turn shape)."""
    rows = []
    for sess in aug["sessions"]:
        for ti, t in enumerate(sess["turns"]):
            rows.append({"session_idx": sess["session_idx"], "turn_idx": ti,
                         "role": t["role"], "content": t["content"],
                         "date": sess["date"], "inserted": t["inserted"]})
    return rows


# ---------------------------------------------------------------------------
# LoCoMo application
# ---------------------------------------------------------------------------

def _apply_locomo(view: dict) -> dict:
    conv = locomo_data.get_conversation(view["host_record_id"])  # fresh copy
    m = re.fullmatch(r"session_(\d+)", view["session_key"])
    assert m, f"{view['patch_id']}: bad session_key {view['session_key']!r}"
    s_idx = int(m.group(1)) - 1  # session_numbers are contiguous 1..N (asserted at load)
    assert conv.session_numbers[s_idx] == int(m.group(1))
    assert view["display_timestamp"] == conv.dates[s_idx], (
        f"{view['patch_id']}: display_timestamp != real "
        f"{view['session_key']}_date_time {conv.dates[s_idx]!r}")
    real_pair = (conv.speaker_a, conv.speaker_b)
    assert (view["speaker_a"], view["speaker_b"]) == real_pair, (
        f"{view['patch_id']}: patch speaker pair != real host pair {real_pair}")
    assert view["speaker_name"] in real_pair and view["speaker_or_role"] in real_pair
    session = conv.sessions[s_idx]
    # r2..r5 packages appended at end-of-session; the r6 regeneration inserts
    # MID-SESSION (the owner's topical-fit requirement: the insertion must sit
    # where the surrounding turns motivate it). insert_after names a REAL
    # dia_id in the target session; inserted rows are spliced directly after
    # it. Fresh dia_ids are still collision-checked against the whole session.
    anchor = next((i for i, t in enumerate(session)
                   if t["dia_id"] == view["insert_after"]), None)
    assert anchor is not None, (
        f"{view['patch_id']}: insert_after {view['insert_after']!r} is not a "
        f"dia_id in {view['session_key']}")
    existing_ids = {x["dia_id"] for x in session}
    patch_rows = []
    inserted_dia_ids = []
    for t in view["turns"]:
        assert t["speaker"] in real_pair, (
            f"{view['patch_id']}: inserted turn speaker {t['speaker']!r} not real")
        assert t["dia_id"] not in existing_ids, (
            f"{view['patch_id']}: dia_id collision {t['dia_id']!r}")
        patch_rows.append({"speaker": t["speaker"], "dia_id": t["dia_id"],
                           "text": t["text"], "blip_caption": None})
        inserted_dia_ids.append(t["dia_id"])
    assert inserted_dia_ids == list(view["dia_id_sequence"]), (
        f"{view['patch_id']}: dia_id_sequence mismatch")
    session[anchor + 1:anchor + 1] = patch_rows

    superseded = []
    for loc in view.get("supersedes", []):
        sn, tn = locomo_data.parse_dia_id(loc)
        sup_idx = sn - 1
        turn_idx = next((i for i, t in enumerate(conv.sessions[sup_idx])
                         if t["dia_id"] == loc), None)
        assert turn_idx is not None, (
            f"{view['patch_id']}: unresolvable supersedes dia_id {loc!r}")
        superseded.append({"session_idx": sup_idx, "turn_idx": turn_idx,
                           "locator": loc})
    replay = sorted({s_idx} | {s["session_idx"] for s in superseded})
    return {
        "patch_id": view["patch_id"],
        "host_dataset": "locomo",
        "host_record_id": view["host_record_id"],
        "conversation": conv,
        "insertion": {"session_idx": s_idx, "dia_ids": inserted_dia_ids,
                      "date": conv.dates[s_idx]},
        "superseded": superseded,
        #: the runner's bounded replay window (patched session + superseded
        #: sessions), in chronological session order; extraction CONTEXT is
        #: always the full augmented conversation (dyadic_turn_job), so the
        #: window changes which turns are gate-ingested, never a call's bytes.
        "replay_session_idxs": [i for i in conv.session_order if i in replay],
    }


def locomo_replay_turns(aug: dict) -> list[dict]:
    """Flat rows over the BOUNDED replay window (driver.locomo_stream_rows
    shape: every turn role='user', content = rendered speaker line)."""
    import locomo_render
    conv = aug["conversation"]
    rows = []
    for si in aug["replay_session_idxs"]:
        for ti, turn in enumerate(conv.sessions[si]):
            rows.append({
                "role": "user",
                "content": locomo_render.render_turn_line(turn),
                "session_idx": si, "turn_idx": ti,
                "date": conv.dates[si],
                "speaker": turn["speaker"], "dia_id": turn["dia_id"],
                "inserted": turn["dia_id"] in aug["insertion"]["dia_ids"],
            })
    return rows


# ---------------------------------------------------------------------------
# Entry point (guarded)
# ---------------------------------------------------------------------------

def apply_patch(patch_id: str) -> dict:
    """Augmented host for one patch. HARD holdout guard; allowlisted view
    only; banned-metadata lint on the output bookkeeping."""
    holdout_guard.assert_ingest_allowed(DATASET, patch_id)
    patch = patches()[patch_id]
    view = allowlisted(patch)
    aug = (_apply_lme(view) if view["host_dataset"] == "longmemeval"
           else _apply_locomo(view))
    check = {k: v for k, v in aug.items() if k != "conversation"}
    _assert_no_banned_keys(check, patch_id)
    return aug
