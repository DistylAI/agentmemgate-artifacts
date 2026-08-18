"""scorer_d1.py — held-out D1 endpoint scorer for the one-shot external test
run (protocol/CHANGELOG.md 2026-07-30 "PREREGISTRATION: external held-out
one-shot test run", D1 endpoints 1-4).

Scores the COMPLETED case records under results/external_eval/test/cases_d1/
(147 test-slice conversations x the four frozen systems) against the held-out
labels. It NEVER re-runs extraction: everything is a pure function of the
case record + the held-out label + the frozen equivalence judge
(natural_check_rule5.EquivalenceScorer — imported, never copied; exact-
normalized fast path first, judge call otherwise, m4_eval.score_case
direction: equivalent(attribute, stored_value, label_value)). Judge calls
run under budget stage "external" (runner_external.external_guard — the
SharedCallBudget wiring m4_eval uses for stage "m4").

ENDPOINTS (operationalizations of the prereg entry; every interpretive
decision is recorded in the output's "methods" block):

E1 Contamination (never-true plans stored as current field value; at-risk
   pool = outcome in {abandoned, still_open}):
     while_pending — an entry of the cell's target_value_after_turn timeline
       with user_idx STRICTLY after the plan turn and STRICTLY before the
       outcome turn (still_open: through the LAST user turn — see
       STILL_OPEN_WINDOW_NOTE) whose value matches label new_value.
     at_end — the cell's final_value matches label new_value (the endpoint
       is STORING new_value; the correct final value — label final_value for
       abandoned, old_value for still_open — is not what is counted).
   Reported per strength x outcome cell + pooled at-risk, per system cell.

E2 Staging + promotion (gated cells em_cg / ez_cg):
     staged_at_plan — a staged_target row whose stage coordinates
       (session_idx/turn_idx on the row, mapped to user_idx via the cell's
       target_value_after_turn user-turn index) equal the plan turn and whose
       hypothesized_value matches new_value. Rows without stage coordinates
       fall back to the value-only criterion (usage counted in methods;
       expected 0 — the runner stamps coordinates on every staged row).
     premature_commit (came_true; ALL FOUR cells, reported here under the
       prereg "gated ~ 0 premature commits" expectation) — new_value stored
       as CURRENT in target_value_after_turn strictly after the plan turn and
       strictly before the outcome turn.
     promotion (came_true) — a valid fire with action "promote" for a
       staged_target staged_id at user_idx == outcome turn (t*, primary,
       k=0); window sensitivity k<=1, k<=2 on |user_idx - outcome_u|.
     stale staging (at-risk) — target_pending_at_end non-empty at the end:
       still_open -> held-correctly (correct behavior); abandoned -> reported
       as still-pending vs discarded.

E3 Task probes (all scored conversations incl. controls, per cell, ITT):
   correct = probe.answered AND the frozen judge accepts probe.value against
   the held-out expected_answer — the m4_eval judge-call shape verbatim:
   equivalent(attribute, probe_value, expected); attribute + the two values
   only, the probe QUESTION is not part of the judge prompt (m4_eval's
   score_case passes no question context). Unanswered probe = incorrect.
   Reported per strength x outcome cell, controls separately, pooled.

E4 Controls (already-happened changes; test slice holds 15 of the 20):
     committed_immediately — a target_events entry with a commit-family
       action ("commit" OR "supersede" — m4_eval's "refresh never introduces
       a value" convention; supersede occurs when the control change lands
       over an established current value, observed in the records) at
       user_idx == control turn whose value matches label final_value
       (new_value fallback if final_value were null; all 15 test controls
       carry final_value).
     over_holding — staged_target non-empty in a gated cell.

TURN-ID MAPPING: labels reference turn ids ("D1-CONV-001-T03") into
data/external/ai_memory_plan_handling/dataset1_standalone/conversations.jsonl.
user_idx = the number of 'user'-speaker turns strictly before the referenced
turn, computed by WALKING the conversation's turn list (never arithmetic);
the referenced turn is asserted to be a user turn.

STILL_OPEN_WINDOW_NOTE: the labels DO carry a non-null event_turn_ids.outcome
for still_open items (the unresolved re-mention turn), but the preregistered
while-pending window for still_open runs through the LAST user turn — the
plan never resolves, so storage-as-current any time after the plan turn is
contamination. The label's outcome id is NOT used as a window end.

STATISTICS: x/n with stats.wilson_ci. Cluster = conversation; D1 has exactly
one item per conversation, so Wilson on item counts is the preregistered CI.

OUTPUT: results/external_eval/test/scores_d1.json (nested per endpoint ->
per system cell -> per stratum: x, n, rate, wilson_lo, wilson_hi; plus the
methods block with every operationalization decision, judge-call counts and
a budget snapshot) + scores_d1.md (readable tables).

ONE-SHOT DISCIPLINE: the real (non --dry-run) scoring is run ONCE by the
maintainer; an existing scores_d1.json is refused without --force. --dry-run
stubs the judge with the exact-normalized fast path only (ZERO LLM calls)
and writes to scores_d1_dryrun.{json,md} so it can never clobber the
one-shot output.

CLI (repo root, pilot/.venv/bin/python experiments/external_eval/scorer_d1.py):
  score [--dry-run] [--cases DIR] [--out-dir DIR] [--call-cap N] [--force]
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401  (R1 layout: puts src/ + experiments/ on sys.path)

import adapter_d1
import natural_check_rule5 as rule5
import runner_external as rx
import stats
import store

CASES_DIR = rx.OUT_ROOT / "test" / "cases_d1"
OUT_DIR = rx.OUT_ROOT / "test"
LABELS_PATH = adapter_d1.LABELS_PATH
CONVERSATIONS_PATH = (adapter_d1.EXTERNAL_PKG / "dataset1_standalone"
                      / "conversations.jsonl")

ALL_CELLS = ("em_cw", "em_cg", "ez_cw", "ez_cg")
GATED_CELLS = ("em_cg", "ez_cg")
AT_RISK_OUTCOMES = ("abandoned", "still_open")
STRENGTHS = ("strong_hedge", "weak_hedge", "confident_plan_no_date",
             "confident_plan_with_date")
OUTCOMES = ("came_true", "abandoned", "still_open")
COMMIT_ACTIONS = ("commit", "supersede")  # m4_eval score_case convention
JUDGE_CALL_CAP = 4000  # the m4_eval compute_scores cap


# ---------------------------------------------------------------------------
# Judge (frozen scorer for the real run; exact-only stub for --dry-run)
# ---------------------------------------------------------------------------

class DryEquivalence:
    """--dry-run judge stub: the frozen scorer's deterministic fast path only
    (absent/sentinel never match; exact-normalized match) — pairs the frozen
    scorer would send to the judge are counted and scored NON-equivalent.
    ZERO LLM calls, no budget spend."""

    def __init__(self):
        self.judge_calls = 0
        self.fresh_judge_calls = 0
        self.unresolved_pairs = 0

    async def equivalent(self, attribute: str, probed, hedged) -> tuple[bool, str]:
        if probed is None or hedged is None:
            return False, "absent"
        if rule5.value_is_sentinel(probed) or rule5.value_is_sentinel(hedged):
            return False, "sentinel"
        if store.normalize_value(str(probed)) == store.normalize_value(str(hedged)):
            return True, "exact_normalized"
        self.unresolved_pairs += 1
        return False, "dry_no_judge"


# ---------------------------------------------------------------------------
# Label / conversation loading + turn-id -> user_idx mapping
# ---------------------------------------------------------------------------

def load_labels(path: Path = LABELS_PATH) -> dict[str, dict]:
    """Held-out labels keyed by record_id (== conversation_id)."""
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {r["record_id"]: r for r in rows}


def load_conversations(path: Path = CONVERSATIONS_PATH) -> dict[str, dict]:
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {r["conversation_id"]: r for r in rows}


def load_cases(cases_dir: Path = CASES_DIR) -> list[dict]:
    cases = [json.loads(p.read_text())
             for p in sorted(cases_dir.glob("*.json"))]
    assert cases, f"no case records under {cases_dir}"
    return cases


def user_idx_of_turn(conv: dict, turn_id: str) -> int:
    """Map a label turn id to the 0-based USER-turn index by walking the
    conversation's full alternating turn list (never arithmetic). Asserts the
    referenced turn IS a user turn."""
    count = 0
    for turn in conv["turns"]:
        if turn["turn_id"] == turn_id:
            assert turn["speaker"] == "user", (
                f"label turn id {turn_id} references a "
                f"{turn['speaker']!r} turn, not a user turn")
            return count
        if turn["speaker"] == "user":
            count += 1
    raise KeyError(f"turn id {turn_id!r} not in conversation "
                   f"{conv['conversation_id']!r}")


# ---------------------------------------------------------------------------
# Per-item scoring (pure function of case + label + conversation + judge)
# ---------------------------------------------------------------------------

async def score_item(case: dict, label: dict, conv: dict, scorer,
                     fallback_counter: Counter | None = None) -> list[dict]:
    """Per-conversation score rows, one per system cell. Every flag is either
    a bool (in the endpoint's population) or None (out of it) — the
    aggregator skips None (m4_eval _point pattern)."""
    if fallback_counter is None:
        fallback_counter = Counter()
    attr = case["target_key"].partition("/")[2]
    is_control = bool(label["control_type"])
    strength, outcome = label["plan_strength"], label["outcome"]
    at_risk = (not is_control) and outcome in AT_RISK_OUTCOMES
    n_user = case["geometry"]["n_user_turns"]
    conv_users = sum(1 for t in conv["turns"] if t["speaker"] == "user")
    assert conv_users == n_user, (
        f"{case['conversation_id']}: conversations.jsonl has {conv_users} "
        f"user turns, case geometry says {n_user} — turn mapping unsafe")
    last_user_idx = n_user - 1

    et = label["event_turn_ids"]
    plan_u = outcome_u = control_u = None
    if is_control:
        control_u = user_idx_of_turn(conv, et["control"])
    else:
        plan_u = user_idx_of_turn(conv, et["possible_switch"])
        if outcome in ("came_true", "abandoned"):
            outcome_u = user_idx_of_turn(conv, et["outcome"])
        # still_open: the label's outcome id (unresolved re-mention) is NOT a
        # window end — the preregistered window runs through the last user
        # turn (STILL_OPEN_WINDOW_NOTE).

    new_value = label["new_value"]
    control_target = (label["final_value"] if label["final_value"] is not None
                      else label["new_value"])

    rows = []
    for cell in [c for c in ALL_CELLS if c in case["cells"]]:
        cc = case["cells"][cell]
        gated = cell in GATED_CELLS
        tvat = cc["target_value_after_turn"]
        coord_to_uidx = {(r["session_idx"], r["turn_idx"]): r["user_idx"]
                         for r in tvat}

        async def window_match(lo_u, hi_u) -> bool:
            """new_value stored as CURRENT strictly after lo_u and up to hi_u
            (hi_u itself EXCLUSIVE unless it is the still_open end-of-
            conversation bound, which callers pass inclusive via +1)."""
            for entry in tvat:
                u = entry["user_idx"]
                if not (lo_u < u < hi_u):
                    continue
                eq, _m = await scorer.equivalent(attr, entry["value"], new_value)
                if eq:
                    return True
            return False

        # ---- E1 contamination (at-risk only) --------------------------------
        while_pending = at_end = None
        if at_risk:
            if outcome == "abandoned":
                while_pending = await window_match(plan_u, outcome_u)
            else:  # still_open: strictly after plan, THROUGH the last user turn
                while_pending = await window_match(plan_u, last_user_idx + 1)
            eq, _m = await scorer.equivalent(attr, cc["final_value"], new_value)
            at_end = bool(eq)

        # ---- E2 premature commit (came_true, all four cells) ----------------
        premature_commit = None
        if (not is_control) and outcome == "came_true":
            premature_commit = await window_match(plan_u, outcome_u)

        # ---- E2 staging + promotion (gated cells) ----------------------------
        staged_at_plan = None
        promote_k0 = promote_k1 = promote_k2 = None
        pending_at_end = None
        staged_ids = {s["staged_id"] for s in cc["staged_target"]}
        if gated and not is_control:
            staged_at_plan = False
            for s in cc["staged_target"]:
                if "session_idx" in s and "turn_idx" in s:
                    s_u = coord_to_uidx.get((s["session_idx"], s["turn_idx"]))
                    if s_u != plan_u:
                        continue
                else:  # no stage coordinates: value-only fallback, counted
                    fallback_counter["staged_coord_fallback"] += 1
                eq, _m = await scorer.equivalent(
                    attr, s["hypothesized_value"], new_value)
                if eq:
                    staged_at_plan = True
                    break
            if outcome == "came_true":
                offsets = [abs(f["user_idx"] - outcome_u)
                           for f in cc["valid_fires"]
                           if f["action"] == "promote"
                           and f["staged_id"] in staged_ids]
                promote_k0 = any(o == 0 for o in offsets)
                promote_k1 = any(o <= 1 for o in offsets)
                promote_k2 = any(o <= 2 for o in offsets)
            if at_risk:
                pending_at_end = bool(cc["target_pending_at_end"])

        # ---- E3 task probe (ITT; every scored conversation) ------------------
        probe_answered = bool(cc["probe"]["answered"])
        probe_correct = False
        if probe_answered:  # unanswered = incorrect, no judge call
            eq, _m = await scorer.equivalent(
                attr, cc["probe"]["value"], label["expected_answer"])
            probe_correct = bool(eq)

        # ---- E4 controls -----------------------------------------------------
        committed_immediately = over_holding = None
        if is_control:
            committed_immediately = False
            for ev in cc["target_events"]:
                if ev["action"] not in COMMIT_ACTIONS:
                    continue  # refresh never introduces a value (m4_eval)
                if ev["user_idx"] != control_u:
                    continue
                eq, _m = await scorer.equivalent(attr, ev["value"], control_target)
                if eq:
                    committed_immediately = True
                    break
            if gated:
                over_holding = bool(cc["staged_target"])

        rows.append({
            "conversation_id": case["conversation_id"],
            "cluster": case["conversation_id"],
            "cell": cell,
            "is_control": is_control,
            "strength": strength,
            "outcome": outcome,
            "at_risk": at_risk,
            "plan_turn_user_idx": plan_u,
            "outcome_turn_user_idx": outcome_u,
            "control_turn_user_idx": control_u,
            # E1
            "while_pending_contamination": while_pending,
            "at_end_contamination": at_end,
            # E2
            "premature_commit": premature_commit,
            "staged_at_plan": staged_at_plan,
            "promote_at_tstar_k0": promote_k0,
            "promote_within_k1": promote_k1,
            "promote_within_k2": promote_k2,
            "pending_at_end": pending_at_end,
            "still_open_held_correctly": (
                pending_at_end if (gated and outcome == "still_open") else None),
            "abandoned_still_pending": (
                pending_at_end if (gated and outcome == "abandoned") else None),
            "abandoned_discarded": (
                (not pending_at_end)
                if (gated and outcome == "abandoned") else None),
            # E3
            "probe_answered": probe_answered,
            "probe_correct": probe_correct,
            "probe_value": cc["probe"]["value"],
            # E4
            "committed_immediately": committed_immediately,
            "over_holding": over_holding,
        })
    return rows


# ---------------------------------------------------------------------------
# Aggregation (x/n + Wilson; cluster = conversation, one item each)
# ---------------------------------------------------------------------------

def _pt(rows: list[dict], flag: str) -> dict:
    sel = [r for r in rows if r.get(flag) is not None]
    x = sum(1 for r in sel if r[flag])
    n = len(sel)
    if n == 0:
        return {"x": 0, "n": 0, "rate": None,
                "wilson_lo": None, "wilson_hi": None}
    lo, hi = stats.wilson_ci(x, n)
    return {"x": x, "n": n, "rate": x / n, "wilson_lo": lo, "wilson_hi": hi}


def _cell(rows: list[dict], cell: str) -> list[dict]:
    return [r for r in rows if r["cell"] == cell]


def _strata(rows: list[dict], flag: str,
            outcomes: tuple[str, ...]) -> dict:
    out = {}
    for s in STRENGTHS:
        for o in outcomes:
            sel = [r for r in rows if r["strength"] == s and r["outcome"] == o]
            out[f"{s}|{o}"] = _pt(sel, flag)
    return out


def aggregate(rows: list[dict]) -> dict:
    """Endpoint tree: per endpoint -> per system cell -> per stratum point."""
    at_risk = [r for r in rows if r["at_risk"]]
    non_ctrl = [r for r in rows if not r["is_control"]]
    came_true = [r for r in rows if r["outcome"] == "came_true"]
    ctrl = [r for r in rows if r["is_control"]]

    endpoints: dict = {}
    endpoints["E1_contamination"] = {
        kind: {cell: {
            "pooled_at_risk": _pt(_cell(at_risk, cell), flag),
            **_strata(_cell(at_risk, cell), flag, AT_RISK_OUTCOMES)}
            for cell in ALL_CELLS}
        for kind, flag in (("while_pending", "while_pending_contamination"),
                           ("at_end", "at_end_contamination"))}
    endpoints["E2_staging_promotion"] = {
        "premature_commit_came_true": {cell: {
            "pooled_came_true": _pt(_cell(came_true, cell), "premature_commit"),
            **_strata(_cell(came_true, cell), "premature_commit",
                      ("came_true",))}
            for cell in ALL_CELLS},
        "staged_at_plan": {cell: {
            "pooled_non_control": _pt(_cell(non_ctrl, cell), "staged_at_plan"),
            **_strata(_cell(non_ctrl, cell), "staged_at_plan", OUTCOMES)}
            for cell in GATED_CELLS},
        "promotion_came_true": {cell: {
            "t_star_k0": _pt(_cell(came_true, cell), "promote_at_tstar_k0"),
            "window_k_le_1": _pt(_cell(came_true, cell), "promote_within_k1"),
            "window_k_le_2": _pt(_cell(came_true, cell), "promote_within_k2"),
            **_strata(_cell(came_true, cell), "promote_at_tstar_k0",
                      ("came_true",))}
            for cell in GATED_CELLS},
        "stale_staging": {cell: {
            "still_open_held_correctly": _pt(
                _cell(rows, cell), "still_open_held_correctly"),
            "abandoned_still_pending": _pt(
                _cell(rows, cell), "abandoned_still_pending"),
            "abandoned_discarded": _pt(
                _cell(rows, cell), "abandoned_discarded")}
            for cell in GATED_CELLS},
    }
    endpoints["E3_task_probes"] = {cell: {
        "pooled_all": _pt(_cell(rows, cell), "probe_correct"),
        "pooled_non_control": _pt(_cell(non_ctrl, cell), "probe_correct"),
        "controls": _pt(_cell(ctrl, cell), "probe_correct"),
        "unanswered_ITT": _pt(
            [dict(r, unanswered=not r["probe_answered"])
             for r in _cell(rows, cell)], "unanswered"),
        **_strata(_cell(non_ctrl, cell), "probe_correct", OUTCOMES)}
        for cell in ALL_CELLS}
    endpoints["E4_controls"] = {
        "committed_immediately": {
            cell: _pt(_cell(ctrl, cell), "committed_immediately")
            for cell in ALL_CELLS},
        "over_holding": {
            cell: _pt(_cell(ctrl, cell), "over_holding")
            for cell in GATED_CELLS},
    }
    return endpoints


# ---------------------------------------------------------------------------
# Full scoring pass
# ---------------------------------------------------------------------------

async def score_all(cases: list[dict], labels: dict[str, dict],
                    convs: dict[str, dict], scorer) -> dict:
    """Score exactly the conversations that have case records."""
    fallback_counter: Counter = Counter()
    rows: list[dict] = []
    for case in cases:
        cid = case["conversation_id"]
        rows.extend(await score_item(case, labels[cid], convs[cid], scorer,
                                     fallback_counter))
    n_items = len({r["conversation_id"] for r in rows})
    n_ctrl = len({r["conversation_id"] for r in rows if r["is_control"]})
    n_at_risk = len({r["conversation_id"] for r in rows if r["at_risk"]})
    return {
        "n_conversations_scored": n_items,
        "n_controls": n_ctrl,
        "n_at_risk": n_at_risk,
        "n_came_true": n_items - n_ctrl - n_at_risk,
        "endpoints": aggregate(rows),
        "rows": rows,
        "_fallback_counter": dict(fallback_counter),
    }


def methods_block(scorer, res: dict, mode: str) -> dict:
    return {
        "mode": mode,
        "spec": ("protocol/CHANGELOG.md 2026-07-30 'PREREGISTRATION: external "
                 "held-out one-shot test run' — D1 endpoints 1-4"),
        "scoring_inputs": ("case records + held-out labels only; extraction "
                           "NEVER re-run; the frozen equivalence judge is the "
                           "only model call"),
        "equivalence": ("rule5.EquivalenceScorer (exact-normalized fast path, "
                        "judge otherwise), direction equivalent(attribute, "
                        "stored_value, label_value) per m4_eval.score_case; "
                        "absent/None/sentinel never match"),
        "probe_judge_shape": ("equivalent(attribute, probe.value, "
                              "expected_answer) — attribute + two values only; "
                              "the probe question is NOT in the judge prompt "
                              "(mirrors m4_eval.score_case exactly)"),
        "itt": "unanswered probe = incorrect (no judge call made)",
        "at_risk_pool": list(AT_RISK_OUTCOMES),
        "while_pending_window": ("strictly after the plan turn, strictly "
                                 "before the outcome turn; still_open: "
                                 "through the LAST user turn"),
        "still_open_window_note": (
            "labels carry a non-null event_turn_ids.outcome for still_open "
            "(the unresolved re-mention turn); it is NOT used as a window "
            "end — the preregistered still_open window runs through the "
            "last user turn"),
        "at_end_endpoint": ("final_value matches label new_value (storing "
                            "the never-true plan), NOT final-value "
                            "correctness"),
        "turn_id_mapping": ("walk dataset1_standalone/conversations.jsonl "
                            "turn list; user_idx = count of user turns "
                            "strictly before the referenced turn; referenced "
                            "turn asserted to be a user turn; per-case "
                            "assert conversation user-turn count == case "
                            "geometry n_user_turns"),
        "staged_stage_coordinates": (
            "staged_target rows carry stage-time session_idx/turn_idx; "
            "mapped to user_idx via the cell's target_value_after_turn "
            "(one row per user turn); value-only fallback used "
            f"{res['_fallback_counter'].get('staged_coord_fallback', 0)} "
            "times"),
        "promotion": ("valid fire action=='promote' with staged_id in "
                      "staged_target; primary k=0 at the outcome turn, "
                      "sensitivity |user_idx - outcome_u| <= 1, <= 2"),
        "commit_family_actions": list(COMMIT_ACTIONS),
        "committed_immediately_note": (
            "supersede counted alongside commit (m4_eval 'refresh never "
            "introduces a value' convention): the control change lands over "
            "an established current value in part of the records"),
        "commit_family_layer_note": (
            "the family is evaluated on the target_events (store-event) "
            "layer, where the gate's additive route ('other'-classed "
            "completed facts, commit-if-absent) surfaces as a plain "
            "'commit' event (gate row 'additive_commit'); such landings "
            "count as immediate commits — the E4 endpoint is the change "
            "being recorded as current fact at the control turn, which the "
            "additive route satisfies (RECOUNT verifier finding, one "
            "decisive control cell: ez_cg D1-CONV-182)"),
        "vacuous_windows_note": (
            "every non-control test conversation has exactly 3 user turns "
            "(current value u0, plan u1, outcome/re-mention u2), so the "
            "strictly-between while-pending window is EMPTY for all "
            "came_true and abandoned items: the premature-commit table "
            "(0 in all four cells, ungated included) and the abandoned "
            "while-pending rows are zero BY GEOMETRY, not system behavior, "
            "and the k<=1/k<=2 promotion sensitivity rows coincide with "
            "k=0 for the same reason (SEMANTICS verifier F1)"),
        "while_pending_sensitivity_note": (
            "all counted while-pending contaminations occur at the "
            "still_open re-mention turn (simultaneously the last user "
            "turn); under the alternative window reading that treats the "
            "label's re-mention id as an exclusive end, while-pending is "
            "0/88 in every cell — the ungated-vs-gated while-pending "
            "contrast exists only under the preregistered "
            "through-last-turn window (SEMANTICS verifier F2)"),
        "judge_artifact_note": (
            "two offsetting equivalence-judge artifacts, +-1/88 in "
            "opposite directions, no stratum or pooled direction flips: "
            "D1-CONV-131 em_cg old-value tracking judged equivalent to the "
            "new value (anti-gate direction), D1-CONV-064 em_cw counted "
            "via a degenerate old==new label pair (pro-gate direction) "
            "(SEMANTICS verifier F3)"),
        "control_match_value": ("label final_value, new_value fallback if "
                                "final_value is null"),
        "ci": ("stats.wilson_ci on item counts; cluster = conversation and "
               "D1 has one item per conversation, so Wilson is the "
               "preregistered CI"),
        "population": ("exactly the conversations with case records "
                       f"({res['n_conversations_scored']}; test slice holds "
                       f"{res['n_controls']} of the 20 controls)"),
        "judge_calls": scorer.judge_calls,
        "fresh_judge_calls": scorer.fresh_judge_calls,
        "dry_unresolved_pairs": getattr(scorer, "unresolved_pairs", None),
        "budget_snapshot": rx.advisory_status(),
    }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _fmt(pt: dict) -> str:
    if pt["n"] == 0:
        return "—"
    return (f"{pt['x']}/{pt['n']} = {pt['rate']:.3f} "
            f"[{pt['wilson_lo']:.3f}, {pt['wilson_hi']:.3f}]")


def _table(title: str, tree: dict, cells: tuple[str, ...]) -> list[str]:
    """One markdown table: rows = strata of the first cell's dict, columns =
    system cells."""
    strata = list(tree[cells[0]].keys())
    lines = [f"### {title}", "",
             "| stratum | " + " | ".join(cells) + " |",
             "|---|" + "|".join(["---"] * len(cells)) + "|"]
    for s in strata:
        lines.append("| " + s.replace("|", "\\|") + " | "
                     + " | ".join(_fmt(tree[c][s]) for c in cells) + " |")
    lines.append("")
    return lines


def render_report(res: dict) -> str:
    ep = res["endpoints"]
    m = res["methods"]
    lines = [
        "# External held-out test — Dataset 1 scores",
        "",
        f"Mode: **{m['mode']}** — generated {res['generated_at']}",
        f"Scored: {res['n_conversations_scored']} conversations "
        f"({res['n_at_risk']} at-risk, {res['n_came_true']} came_true, "
        f"{res['n_controls']} controls) x 4 systems.",
        "",
        "Spec: " + m["spec"],
        "",
        "## E1 — Contamination (at-risk: abandoned + still_open)", ""]
    lines += _table("While-pending (new_value stored as current in-window)",
                    ep["E1_contamination"]["while_pending"], ALL_CELLS)
    lines += [
        "> Sensitivity (verifier F2): every counted while-pending event "
        "sits at the still_open re-mention turn (= the last user turn); "
        "the abandoned rows are structurally zero (empty window, see the "
        "premature-commit note below), and under a window reading that "
        "ends at the re-mention turn this whole table is 0/88 in every "
        "cell.", ""]
    lines += _table("At-end (final store holds new_value)",
                    ep["E1_contamination"]["at_end"], ALL_CELLS)
    lines += ["## E2 — Staging + promotion", ""]
    lines += _table("Premature commit (came_true; prereg 'gated ~ 0')",
                    ep["E2_staging_promotion"]["premature_commit_came_true"],
                    ALL_CELLS)
    lines += [
        "> VACUOUS BY GEOMETRY (verifier F1): all non-control "
        "conversations place the outcome on the user turn immediately "
        "after the plan, so the strictly-between window is empty — these "
        "zeros (ungated cells included) are substrate geometry, not "
        "gate behavior, and the k<=1/k<=2 promotion sensitivity rows "
        "below coincide with k=0 for the same reason. Do not cite this "
        "table as a result.", ""]
    lines += _table("Staged at the plan turn (gated cells)",
                    ep["E2_staging_promotion"]["staged_at_plan"], GATED_CELLS)
    lines += _table("Promotion on came_true (gated cells)",
                    ep["E2_staging_promotion"]["promotion_came_true"],
                    GATED_CELLS)
    lines += _table("Stale staging at end (gated cells)",
                    ep["E2_staging_promotion"]["stale_staging"], GATED_CELLS)
    lines += ["## E3 — Task probes (ITT)", ""]
    lines += _table("Probe correct vs held-out expected_answer",
                    ep["E3_task_probes"], ALL_CELLS)
    lines += ["## E4 — Controls (already-happened changes)", ""]
    lines += _table("Committed immediately at the control turn",
                    {c: {"controls": ep["E4_controls"]
                         ["committed_immediately"][c]} for c in ALL_CELLS},
                    ALL_CELLS)
    lines += _table("Over-holding (control change staged; gated cells)",
                    {c: {"controls": ep["E4_controls"]["over_holding"][c]}
                     for c in GATED_CELLS}, GATED_CELLS)
    lines += ["## Methods", ""]
    for k, v in m.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def run_score(cases_dir: Path, out_dir: Path, dry_run: bool,
                    call_cap: int | None, force: bool) -> dict:
    cases = load_cases(cases_dir)
    slices = {c.get("slice") for c in cases}
    assert slices == {"test"}, (
        f"case records span slices {slices}; the scorer targets the "
        "one-shot TEST run only")
    labels = load_labels()
    convs = load_conversations()

    out_json = out_dir / ("scores_d1_dryrun.json" if dry_run
                          else "scores_d1.json")
    out_md = out_dir / ("scores_d1_dryrun.md" if dry_run else "scores_d1.md")
    if not dry_run and out_json.exists() and not force:
        raise SystemExit(
            f"{out_json} already exists — the real scoring is ONE-SHOT; "
            "pass --force only for a logged infrastructure-failure rerun")

    guard = None
    if dry_run:
        scorer = DryEquivalence()
    else:
        scorer = rule5.EquivalenceScorer(cap=JUDGE_CALL_CAP)
        guard = rx.external_guard(local_cap=call_cap)  # budget stage "external"
        guard.install()
    try:
        res = await score_all(cases, labels, convs, scorer)
    finally:
        if guard is not None:
            guard.uninstall()

    res["generated_at"] = datetime.now(timezone.utc).isoformat()
    res["methods"] = methods_block(
        scorer, res, "dry_run (judge stubbed: exact-normalized only, zero "
        "LLM calls)" if dry_run else "full (frozen equivalence judge, budget "
        "stage 'external')")
    if guard is not None:
        res["methods"]["guard_fresh_calls"] = guard.fresh
    res.pop("_fallback_counter")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    out_md.write_text(render_report(res))
    print(f"wrote {out_json}\nwrote {out_md}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("score", help="score the D1 one-shot test run")
    sc.add_argument("--cases", type=Path, default=CASES_DIR)
    sc.add_argument("--out-dir", type=Path, default=OUT_DIR)
    sc.add_argument("--dry-run", action="store_true",
                    help="stub the judge (exact-normalized only, no LLM "
                    "calls); writes scores_d1_dryrun.{json,md}")
    sc.add_argument("--call-cap", type=int, default=None)
    sc.add_argument("--force", action="store_true",
                    help="overwrite an existing scores_d1.json (logged "
                    "infrastructure-failure rerun only)")
    args = ap.parse_args()
    res = asyncio.run(run_score(args.cases, args.out_dir, args.dry_run,
                                args.call_cap, args.force))
    ep = res["endpoints"]
    print(f"\nscored {res['n_conversations_scored']} conversations "
          f"({res['n_at_risk']} at-risk, {res['n_controls']} controls); "
          f"judge_calls={res['methods']['judge_calls']}")
    for cell in ALL_CELLS:
        wp = ep["E1_contamination"]["while_pending"][cell]["pooled_at_risk"]
        ae = ep["E1_contamination"]["at_end"][cell]["pooled_at_risk"]
        pr = ep["E3_task_probes"][cell]["pooled_all"]
        print(f"  {cell}: while_pending {_fmt(wp)} | at_end {_fmt(ae)} | "
              f"probe {_fmt(pr)}")


if __name__ == "__main__":
    main()
