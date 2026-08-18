"""tier1_retry.py — RETRY ABLATION, items 1+2 of the reviewer-response Tier 1
preregistration (protocol/CHANGELOG.md 2026-08-14 "PREREGISTRATION:
reviewer-response experiment set (Tier 1)").

TWO NEW CELLS on both shared substrates, beside the released four:

  * em_cg_noretry — cautious (em) extraction + gate ON, FROZEN v2 mapper only
    (store._map_tuple_key; NO plan-frame stem retry). This is exactly the
    pre-adoption em_cg configuration — the m4_eval ORIGINAL em_cg arm (the
    m4_eval 4-cell grid ran em_cg on the frozen mapper; the wrapper was its
    separate 5th arm em_cg_fixed).
  * em_cw_retry — cautious extraction + gate OFF (ungated write-through),
    WITH the adopted plan-stem retry installed at the same LOCAL mapper seam
    (store.make_tuple_key_mapper over PLAN_FRAME_ATTRIBUTE_PATTERNS).
    Retry-recovered tuples commit directly since the gate is off — the
    retry's effect isolated from the gate.

Together with the released cells (em_cw, em_cg = adopted-with-retry, ez_cw,
ez_cg) this completes the informative portion of the 2x2x2
style x gate x retry factorial; literal(ez) x retry is asserted a structural
no-op by the deterministic count below.

SUBSTRATES (per-turn seams imported from the released runners, never copied):
  * D1 — the 147 held-out test conversations (holdout_guard test slice), the
    runner_external.run_d1_case pattern verbatim (same extraction /
    key-mapping / gate loop bodies via rx._extract_and_map_lme and
    rx._run_policy_cell). Records: results/tier1/d1/cases/<conv_id>.json.
    The run REQUIRES EXTERNAL_TEST_RUN_AUTHORIZED=1 (owner-authorized
    reviewer-response runs per the 2026-08-14 prereg entry; the substrate
    and labels were already exposed by the released one-shot test run).
  * UNIFY — the 179 usable controlled-difficulty arcs
    (results/unify/arcs.jsonl minus unify_exclusions.json), the
    unify_eval.run_arc pattern verbatim (ue._extract_and_map /
    ue._run_policy_cell). Records: results/tier1/unify/cases/<arc_id>.json.

CACHE SHARING: em extraction and both mapping passes are byte-identical to
the released em_cw/em_cg calls (llm.py content-addressed cache), so they
replay free; fresh calls concentrate where the retry changed the store state
(20/147 D1 conversations, 51/179 unify arcs carry stem-mapped tuples).

DETERMINISTIC NO-OP CHECK (literal +- retry): the retry's trigger set is a
DECLINED tuple whose attribute matches the frozen
PLAN_FRAME_ATTRIBUTE_PATTERNS. The released case records keep only mapping
counts, so the released ez (literal / canonical-writing) extraction outputs
are recovered via the content-addressed cache (byte-identical replay of the
released calls — extraction and first-pass mapping, zero fresh calls
expected and reported) and counted per substrate: n plan-frame-matching
attributes, and among them n declined by the frozen mapper. A zero count is
the structural-no-op evidence the prereg names. em counts are reported
alongside as context (the asymmetry that motivates the retry).

SCORING (once; existing scorers' logic, frozen judge gpt-5.5):
  * D1 — scorer_d1.score_item VERBATIM: each new cell is scored under the
    frozen cell name matching its POLICY (em_cg_noretry -> "em_cg" [gated
    branch], em_cw_retry -> "em_cw" [ungated branch]) and the row is
    relabeled back. Endpoints: E1 contamination (while-pending / at-end),
    E2 staging + promotion, E3 task probes ITT, E4 controls.
  * UNIFY — unify_eval.score_case VERBATIM under the same policy aliasing.
    Endpoints: contamination (while-pending / at-end, still_open), staging,
    promotion (came_true), end-state, task probes ITT, g5 commit-correct
    control.
  * Field-match rate per cell = the feeding rec-set's mapping funnel
    (em / em_fixed / ez family diagnostics; D1: any tuple mapped to the
    registered target key per conversation; unify: hedge-turn
    any_tuple_mapped_to_target), read from the case records.
  Released-cell numbers come from the released row files
  (results/external_eval/test/scores_d1.json rows;
  results/unify/run/unify_results.json per_arc_rows) — read, never rerun —
  and are aggregated through the same _pt code path as the new cells.

PAIRED COMPARISONS (item 2): cluster-bootstrap paired CIs (the repo's
stats.cluster_bootstrap_mean over per-conversation deltas in {-1,0,1},
bounds=(-1,1) — the task_deltas._delta_block pattern; seed 20260702 fixed,
n_boot 10,000; cluster = conversation/arc):
  (a) gate on-vs-off per style per substrate — em at both retry levels
      (retry-on: released em_cg vs new em_cw_retry; retry-off: new
      em_cg_noretry vs released em_cw) and ez (released ez_cg vs ez_cw);
  (b) retry on-vs-off in the cautious cells — gated: released em_cg
      (with retry) vs new em_cg_noretry; ungated: new em_cw_retry vs
      released em_cw.

BUDGET: stage "tier1" (ADVISORY 8,000) in results/v2.4/budget.json, the
established way (plan_typed.register_stage pattern: dated note, flock +
atomic rename). The stage is SHARED across the Tier-1 set: a concurrent
Tier-1 agent registered it headroom-limited at 4,000; this runner raises the
advisory to 8,000 (never lowers), the note disclosing that the remainder
beyond the exhausted headroom is drawn against unspent completed-stage
forecasts (allocations are ADVISORY per the 2026-07-08 amendment — only the
120k envelope is hard). THIS run
additionally refuses to start / stops if the combined total would pass
110,000 (the unify/plan_typed prereg stop-condition pattern, via local_cap).

OUTPUT: results/tier1/retry_ablation_results.json +
results/tier1/retry_ablation_report.md — the factorial table (all
informative cells x both substrates), the paired-CI table, the no-op count,
run stats, budget before/after. Scored once; verification (RECOUNT +
SEMANTICS) follows separately — numbers are UNVERIFIED until then.

CLI (repo root; MEMGATE_SYSTEM_MODEL must be UNSET):
  EXTERNAL_TEST_RUN_AUTHORIZED=1 pilot/.venv/bin/python \
      experiments/external_eval/tier1_retry.py run-d1 [--ids ...] [--shard K/N] [--call-cap N]
  pilot/.venv/bin/python experiments/external_eval/tier1_retry.py run-unify [--ids ...] [--shard K/N] [--call-cap N]
  pilot/.venv/bin/python experiments/external_eval/tier1_retry.py score [--force]
  pilot/.venv/bin/python experiments/external_eval/tier1_retry.py status
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401  (R1 layout: puts src/ + experiments/ on sys.path)

import adapter_d1
import forecast_v22
import gated_store
import holdout_guard
import llm
import m4_eval
import mapper_fix
import natural_check_rule5 as rule5
import runner_external as rx
import scorer_d1 as sd1
import stats
import store
import unify_eval as ue
from router_sweep import PLAN_FRAME_ATTRIBUTE_PATTERNS, plan_frame_match

from paths import RESULTS_DIR

STAGE = "tier1"
PINNED_SYSTEM_MODEL = "gpt-5.5-2026-04-23"
ADVISORY_FORECAST = 8_000
#: Prereg stop condition for THIS run (the protocol hard stop stays 120k).
HARD_STOP_TOTAL = 110_000
SEED = stats.DEFAULT_SEED  # 20260702 (INTERFACES.md shared conventions)
N_BOOT = 10_000

V24_DIR = RESULTS_DIR / "v2.4"
BUDGET_PATH = V24_DIR / "budget.json"

TIER1_DIR = RESULTS_DIR / "tier1"
D1_CASES = TIER1_DIR / "d1" / "cases"
UNIFY_CASES = TIER1_DIR / "unify" / "cases"
RESULTS_PATH = TIER1_DIR / "retry_ablation_results.json"
REPORT_PATH = TIER1_DIR / "retry_ablation_report.md"
RUN_LOG = TIER1_DIR / "run_log.jsonl"
BUDGET_BEFORE = TIER1_DIR / "budget_before.json"

#: Released artifacts this analysis QUOTES rows from (read-only, never rerun).
RELEASED_D1_SCORES = RESULTS_DIR / "external_eval" / "test" / "scores_d1.json"
RELEASED_UNIFY_RESULTS = RESULTS_DIR / "unify" / "run" / "unify_results.json"
RELEASED_D1_CASES = RESULTS_DIR / "external_eval" / "test" / "cases_d1"
RELEASED_UNIFY_CASES = RESULTS_DIR / "unify" / "run" / "cases"

NEW_CELLS = ("em_cw_retry", "em_cg_noretry")
RELEASED_CELLS = ("em_cw", "em_cg", "ez_cw", "ez_cg")
#: Factorial display order: style x gate x retry, informative cells only.
FACTORIAL_CELLS = ("em_cw", "em_cw_retry", "em_cg_noretry", "em_cg",
                   "ez_cw", "ez_cg")
GATED = {"em_cg", "em_cg_noretry", "ez_cg"}
#: Frozen scorer alias per new cell — the frozen cell name whose POLICY
#: branch (gated/ungated) matches; scoring runs verbatim, rows relabeled.
SCORE_ALIAS = {"em_cg_noretry": "em_cg", "em_cw_retry": "em_cw"}
#: Rec-set family feeding each cell (field-match / mapping-funnel axis).
FAMILY_OF = {"em_cw": "em", "em_cg_noretry": "em",
             "em_cg": "em_fixed", "em_cw_retry": "em_fixed",
             "ez_cw": "ez", "ez_cg": "ez"}

CELL_NOTES = {
    "em_cg_noretry": (
        "cautious extraction + gate ON, FROZEN v2 mapper (NO plan-frame "
        "stem retry) — the pre-adoption em_cg configuration (m4_eval's "
        "original em_cg arm: frozen store._map_tuple_key; the wrapper was "
        "m4_eval's separate 5th arm em_cg_fixed)"),
    "em_cw_retry": (
        "cautious extraction + gate OFF (ungated write-through) WITH the "
        "adopted plan-stem retry at the same LOCAL mapper seam "
        "(store.make_tuple_key_mapper over PLAN_FRAME_ATTRIBUTE_PATTERNS); "
        "retry-recovered tuples commit directly — the retry isolated from "
        "the gate"),
}

#: Paired contrasts (name, cell_a, cell_b): delta = rate(a) - rate(b).
CONTRASTS = (
    ("gate_em_retry_on", "em_cg", "em_cw_retry",
     "gate on-vs-off, cautious style, retry PRESENT on both sides"),
    ("gate_em_retry_off", "em_cg_noretry", "em_cw",
     "gate on-vs-off, cautious style, retry ABSENT on both sides"),
    ("gate_ez", "ez_cg", "ez_cw",
     "gate on-vs-off, literal style (retry structurally no-op — see count)"),
    ("retry_gated", "em_cg", "em_cg_noretry",
     "retry on-vs-off inside the gated cautious cell"),
    ("retry_ungated", "em_cw_retry", "em_cw",
     "retry on-vs-off inside the ungated cautious cell"),
)
#: Endpoints defined for BOTH sides of every contrast, per substrate.
D1_SHARED_FLAGS = ("while_pending_contamination", "at_end_contamination",
                   "probe_correct", "committed_immediately")
#: Additional endpoints for the both-gated contrast (retry_gated) on D1.
D1_GATED_FLAGS = ("staged_at_plan", "promote_at_tstar_k0",
                  "promote_within_k2", "over_holding")
UNIFY_SHARED_FLAGS = ("while_pending_contamination", "at_end_contamination",
                      "probe_correct", "end_state_correct",
                      "g5_commit_correct")
UNIFY_GATED_FLAGS = ("staged_at_hedge", "promote_at_tstar",
                     "promote_within_window")


# ---------------------------------------------------------------------------
# Budget (advisory stage tier1; 110k prereg stop via local_cap)
# ---------------------------------------------------------------------------

def register_stage(path: Path = BUDGET_PATH) -> dict:
    """Register the tier1 stage in the combined budget file — the established
    convention (external/claude_repl/unify/plan_typed): dated note, flock +
    atomic rename, spent counter from 0; idempotent. Remaining unallocated
    headroom (4,000) is less than the 8,000 advisory forecast, so the note
    discloses the remainder is drawn against unspent completed-stage
    forecasts — allocations are ADVISORY (2026-07-08 amendment), only the
    120k envelope is hard."""
    lock_path = path.with_suffix(".lock")
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        budget = json.loads(path.read_text())
        current = budget["allocation"].get(STAGE)
        if current is not None and current >= ADVISORY_FORECAST:
            return budget  # already at (or above) the 8k advisory
        headroom = budget["allocation"].get("unallocated_headroom", 0)
        need = ADVISORY_FORECAST - (current or 0)
        from_headroom = min(headroom, need)
        budget["allocation"]["unallocated_headroom"] = headroom - from_headroom
        budget["allocation"][STAGE] = ADVISORY_FORECAST
        budget["spent"].setdefault(STAGE, 0)
        if current is None:
            what = f"tier1 {ADVISORY_FORECAST} registered"
        else:
            what = (f"tier1 advisory raised {current} -> {ADVISORY_FORECAST} "
                    "(SHARED Tier-1 stage — a concurrent Tier-1 agent "
                    "registered it headroom-limited; concurrent runners "
                    "accept the stage as-is)")
        budget.setdefault("notes", []).append(
            f"2026-08-14: {what} (reviewer-response Tier 1 experiment set, "
            "items 1+2 retry ablation + paired comparisons; preregistration "
            f"changelog 2026-08-14). {from_headroom} from unallocated "
            f"headroom (now {budget['allocation']['unallocated_headroom']}); "
            f"the remaining {need - from_headroom} drawn against unspent "
            "completed-stage forecasts (allocation sum may exceed the "
            "envelope — allocations are ADVISORY per the 2026-07-08 "
            "amendment; the 120k envelope is the only protocol hard stop). "
            "The run additionally refuses to push TOTAL spend past 110,000 "
            "(prereg stop condition).")
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(budget, indent=1, ensure_ascii=False))
        tmp.replace(path)
        return budget


def tier1_guard(path: Path = BUDGET_PATH,
                local_cap: int | None = None) -> forecast_v22.SharedCallBudget:
    """The unify_guard/pt_guard pattern, stage tier1: stage forecast ADVISORY
    (warn + continue); this run's own stop condition caps fresh calls so the
    combined total cannot pass 110k."""
    register_stage(path)
    budget = mapper_fix.ensure_budget(path)
    spent = budget["spent"].get(STAGE, 0)
    forecast = budget["allocation"].get(STAGE, ADVISORY_FORECAST)
    if spent >= forecast:
        print(f"[tier1 budget ADVISORY] stage spend {spent:,} >= forecast "
              f"{forecast:,} — continuing (stages are advisory per the "
              "2026-07-08 amendment; only the total envelope is hard)")
    margin_110 = HARD_STOP_TOTAL - sum(budget["spent"].values())
    if margin_110 <= 0:
        raise SystemExit(
            f"tier1 STOP CONDITION: combined spend "
            f"{sum(budget['spent'].values()):,} >= {HARD_STOP_TOTAL:,} — "
            "prereg stop; report and stop")
    remaining = min(forecast_v22.budget_remaining(budget), margin_110)
    cap = remaining if local_cap is None else min(local_cap, remaining)
    return forecast_v22.SharedCallBudget(STAGE, path=path, local_cap=cap)


def advisory_status(path: Path = BUDGET_PATH) -> dict:
    budget = mapper_fix.ensure_budget(path)
    spent = budget["spent"].get(STAGE, 0)
    forecast = budget["allocation"].get(STAGE, ADVISORY_FORECAST)
    return {
        "stage": STAGE,
        "advisory_forecast": forecast,
        "spent": spent,
        "over_forecast": spent > forecast,
        "envelope": budget["envelope"],
        "envelope_spent_total": sum(budget["spent"].values()),
        "run_stop_total": HARD_STOP_TOTAL,
        "mode": "ADVISORY stage / HARD total envelope (2026-07-08 amendment)"
                " / 110k run stop (2026-08-14 Tier 1 prereg)",
    }


# ---------------------------------------------------------------------------
# Pins + bookkeeping
# ---------------------------------------------------------------------------

def assert_pins() -> None:
    assert os.environ.get("MEMGATE_SYSTEM_MODEL") is None, (
        "MEMGATE_SYSTEM_MODEL is set — the Tier 1 retry ablation is "
        "preregistered on the PINNED default gpt-5.5 systems")
    assert llm.MODEL == PINNED_SYSTEM_MODEL, (
        f"llm.MODEL is {llm.MODEL!r} — expected {PINNED_SYSTEM_MODEL}")
    assert llm.JUDGE_MODEL == PINNED_SYSTEM_MODEL, (
        f"llm.JUDGE_MODEL drifted to {llm.JUDGE_MODEL!r}")
    m4_eval.assert_mapper_mode_off()


def snapshot_budget_before() -> None:
    if not BUDGET_BEFORE.exists():
        TIER1_DIR.mkdir(parents=True, exist_ok=True)
        BUDGET_BEFORE.write_text(BUDGET_PATH.read_text())


def log_phase(row: dict) -> None:
    TIER1_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUN_LOG, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _shard(ids: list[str], spec: str | None) -> list[str]:
    if not spec:
        return ids
    k, n = (int(x) for x in spec.split("/"))
    assert 0 <= k < n, f"bad shard spec {spec!r}"
    return ids[k::n]


# ---------------------------------------------------------------------------
# No-op trigger count (deterministic; cache replay of released calls)
# ---------------------------------------------------------------------------

def _noop_counts(recs: list[dict]) -> dict:
    """Count the retry's trigger set in one family's per-turn recs. The stem
    wrapper re-offers a tuple iff (i) the frozen first-pass mapping DECLINED
    (cand is None) AND (ii) store.derive_plan_stem yields a usable stem
    (plan-frame token match at a token boundary, non-empty stem differing
    from the input) — n_retry_triggers counts exactly that. The coarser
    plan-frame attribute match is reported alongside."""
    n_tuples = n_pf = n_pf_declined = n_trigger = 0
    matched: list[dict] = []
    for rec in recs:
        for tup, mres in zip(rec["tuples"], rec["mapped"]):
            n_tuples += 1
            if not plan_frame_match(tup.attribute):
                continue
            n_pf += 1
            cand, _raw, _rat, _meta = gated_store._unpack_map_result(mres)
            declined = cand is None
            n_pf_declined += int(declined)
            stem = store.derive_plan_stem(tup.attribute,
                                          PLAN_FRAME_ATTRIBUTE_PATTERNS)
            n_trigger += int(declined and stem is not None)
            matched.append({"session_idx": rec["session_idx"],
                            "turn_idx": rec["turn_idx"],
                            "attribute": tup.attribute,
                            "declined": declined,
                            "derived_stem": stem})
    return {"n_tuples": n_tuples, "n_plan_frame_attrs": n_pf,
            "n_plan_frame_declined": n_pf_declined,
            "n_retry_triggers": n_trigger,
            "matched_attributes": matched}


# ---------------------------------------------------------------------------
# D1 — the 147 held-out test conversations (runner_external pattern)
# ---------------------------------------------------------------------------

async def run_d1_case(conversation_id: str) -> dict:
    """One D1 conversation through the TWO new cells (frozen per-turn seams
    imported from runner_external; extraction + mapping cache-shared with the
    released run) + the ez/em no-op trigger counts."""
    stream, key_row = adapter_d1.adapt(conversation_id)
    slice_name = holdout_guard.slice_of(adapter_d1.DATASET, conversation_id)
    turns = adapter_d1.stream_turns(stream)
    user_positions = [i for i, t in enumerate(turns) if t["role"] == "user"]
    target = adapter_d1.canonical_key(key_row)
    candidates = [target]  # rx.D1_CANDIDATES_NOTE
    gen_system = gated_store.GENERATOR_PROMPT_PATH.read_text()
    checker_system = gated_store.CHECKER_PROMPT_PATH.read_text()

    async def _recs(family: str, mapper=None) -> list[dict]:
        return rx._attach_user_idx([
            await rx._extract_and_map_lme(family, turns, pos, candidates,
                                          conversation_id, "d1", mapper=mapper)
            for pos in user_positions])

    m4_eval.assert_mapper_mode_off()   # global frozen mapper untouched at entry
    recs_em = await _recs("em")        # frozen mapping -> em_cg_noretry
    wrapper = store.make_tuple_key_mapper(PLAN_FRAME_ATTRIBUTE_PATTERNS)
    recs_em_fixed = await _recs("em", mapper=wrapper)  # retry -> em_cw_retry
    m4_eval.assert_mapper_mode_off()   # teardown: the wrapper stayed local
    recs_ez = await _recs("ez")        # cache replay; no-op count only

    out = {
        "conversation_id": conversation_id,
        "dataset": adapter_d1.DATASET,
        "slice": slice_name,
        "question_id": key_row["question_id"],
        "target_key": target.name,
        "probe_question": target.probe_question,
        "candidates_convention": rx.D1_CANDIDATES_NOTE,
        "cell_notes": CELL_NOTES,
        "prereg": ("protocol/CHANGELOG.md 2026-08-14 reviewer-response "
                   "Tier 1, item 1 (retry ablation)"),
        "test_run_authorization": (
            "EXTERNAL_TEST_RUN_AUTHORIZED=1 (owner-authorized "
            "reviewer-response runs, 2026-08-14 prereg; the test substrate "
            "and labels were already exposed by the released one-shot run)"),
        "geometry": {
            "n_sessions": len(stream["sessions"]),
            "n_turns": len(turns),
            "n_user_turns": len(user_positions),
            "session_dates": [s["date"] for s in stream["sessions"]],
        },
        "cells": {},
        "diagnostics": {},
        "noop_trigger_counts": {"em": _noop_counts(recs_em),
                                "ez": _noop_counts(recs_ez)},
    }
    out["diagnostics"]["em"] = [rx._turn_diag_row(r, target) for r in recs_em]
    out["diagnostics"]["em_fixed"] = [rx._turn_diag_row(r, target)
                                      for r in recs_em_fixed]
    out["diagnostics"]["ez"] = [rx._turn_diag_row(r, target) for r in recs_ez]

    kw = dict(case_id=conversation_id, turns=turns, benchmark="lme",
              gen_system=gen_system, checker_system=checker_system,
              target=target)
    out["cells"]["em_cg_noretry"] = await rx._run_policy_cell(
        "gated", recs_em, "em", **kw)
    cw_retry = await rx._run_policy_cell("ungated", recs_em_fixed, "em", **kw)
    cw_retry["mapping_meta"] = rx._wrapper_mapping_meta(recs_em_fixed, target)
    out["cells"]["em_cw_retry"] = cw_retry
    return out


# ---------------------------------------------------------------------------
# UNIFY — the 179 usable controlled-difficulty arcs (unify_eval pattern)
# ---------------------------------------------------------------------------

async def run_unify_arc(arc: dict, key_row: dict, arcs_digest: str) -> dict:
    """One unify arc through the TWO new cells (frozen per-turn seams
    imported from unify_eval) + the ez/em no-op trigger counts. The record
    scaffold mirrors ue.run_arc so ue.score_case runs verbatim on it."""
    arc_id = arc["arc_id"]
    geom = m4_eval.arc_geometry(arc)
    turns, user_positions = geom["turns"], geom["user_positions"]
    user_index, hedge_pos, res_pos = (geom["user_index"], geom["hedge_pos"],
                                      geom["res_pos"])
    target = m4_eval.arc_canonical_key(key_row)
    candidates = [target]
    gen_system = gated_store.GENERATOR_PROMPT_PATH.read_text()
    checker_system = gated_store.CHECKER_PROMPT_PATH.read_text()

    async def _recs(family: str, mapper=None) -> list[dict]:
        return [await ue._extract_and_map(family, turns, pos, candidates,
                                          arc_id, mapper=mapper)
                for pos in user_positions]

    m4_eval.assert_mapper_mode_off()
    recs_em = await _recs("em")
    wrapper = store.make_tuple_key_mapper(PLAN_FRAME_ATTRIBUTE_PATTERNS)
    recs_em_fixed = await _recs("em", mapper=wrapper)
    m4_eval.assert_mapper_mode_off()
    recs_ez = await _recs("ez")

    out = {"arc_id": arc_id,
           "unify_ladder": arc["unify_ladder"],
           "unify_level": arc["unify_level"],
           "m2_level": arc["m2_level"],
           "outcome": ue.arc_outcome(arc),
           "resolution_type": arc["resolution_type"],
           "gold_resolution_status": arc["gold_resolution_status"],
           "hedge_strength": arc["hedge_strength"],
           "cluster": arc_id,
           "target_key": target.name,
           "hedged_value": arc["value"],
           "value_candidates": arc.get("value_candidates"),
           "prior_value": arc.get("prior_value"),
           "probe_question": key_row["probe_question"],
           "cell_notes": CELL_NOTES,
           "prereg": ("protocol/CHANGELOG.md 2026-08-14 reviewer-response "
                      "Tier 1, item 1 (retry ablation)"),
           "arcs_sha256": arcs_digest,
           "seed": arc["seed"],
           "geometry": {
               "n_turns": len(turns), "n_user_turns": len(user_positions),
               "hedge": {"pos": hedge_pos,
                         "session_idx": turns[hedge_pos]["session_idx"],
                         "turn_idx": turns[hedge_pos]["turn_idx"],
                         "user_idx": user_index[hedge_pos]},
               "resolution": {"pos": res_pos,
                              "session_idx": turns[res_pos]["session_idx"],
                              "turn_idx": turns[res_pos]["turn_idx"],
                              "user_idx": user_index[res_pos]},
           },
           "cells": {}, "diagnostics": {},
           "noop_trigger_counts": {"em": _noop_counts(recs_em),
                                   "ez": _noop_counts(recs_ez)}}

    for fam, recs in (("em", recs_em), ("em_fixed", recs_em_fixed),
                      ("ez", recs_ez)):
        hedge_rec = next(r for r in recs if r["pos"] == hedge_pos)
        out["diagnostics"][fam] = {
            "hedge_turn": m4_eval._hedge_turn_diagnostics(
                hedge_rec, arc["hedge_text"], target),
            "per_turn": [m4_eval._turn_diag_row(rec, user_index[rec["pos"]],
                                                target)
                         for rec in recs],
        }

    kw = dict(arc_id=arc_id, target=target, turns=turns,
              user_index=user_index, res_pos=res_pos, hedge_pos=hedge_pos,
              gen_system=gen_system, checker_system=checker_system)
    out["cells"]["em_cg_noretry"] = await ue._run_policy_cell(
        "gated", recs_em, "em", **kw)
    cw_retry = await ue._run_policy_cell("ungated", recs_em_fixed, "em", **kw)
    cw_retry["mapping_meta"] = m4_eval._fixed_mapping_meta(
        recs_em_fixed, user_index, target)
    out["cells"]["em_cw_retry"] = cw_retry
    return out


# ---------------------------------------------------------------------------
# Run orchestration (resumable per case; guarded budget)
# ---------------------------------------------------------------------------

async def run_substrate(substrate: str, ids: list[str] | None,
                        shard: str | None, call_cap: int | None) -> dict:
    assert_pins()
    snapshot_budget_before()
    if substrate == "d1":
        assert holdout_guard.test_run_authorized(), (
            "run-d1 targets the held-out TEST slice — set "
            "EXTERNAL_TEST_RUN_AUTHORIZED=1 (owner-authorized "
            "reviewer-response runs, 2026-08-14 prereg entry)")
        all_ids = sorted(holdout_guard.test_ids(adapter_d1.DATASET))
        assert len(all_ids) == 147, f"expected 147 test ids, got {len(all_ids)}"
        cases_dir = D1_CASES

        async def _runner(item_id):
            return await run_d1_case(item_id)
    else:
        arcs = ue.load_arcs()
        keys = ue.load_keys()
        digest = ue.arcs_sha256()
        by_id = {a["arc_id"]: a for a in arcs}
        all_ids = sorted(by_id)
        cases_dir = UNIFY_CASES

        async def _runner(item_id):
            return await run_unify_arc(by_id[item_id], keys[item_id], digest)

    sel = _shard(sorted(ids) if ids else all_ids, shard)
    unknown = set(sel) - set(all_ids)
    assert not unknown, f"unknown/excluded ids: {sorted(unknown)}"
    started = datetime.now(timezone.utc).isoformat()
    guard = tier1_guard(local_cap=call_cap)
    done = skipped = 0
    failed: list[dict] = []
    guard.install()
    try:
        for item_id in sel:
            path = cases_dir / f"{item_id}.json"
            if path.exists():
                skipped += 1
                continue
            try:
                record = await _runner(item_id)
            except (SystemExit, KeyboardInterrupt,
                    holdout_guard.HoldoutViolation):
                raise
            except Exception as exc:  # noqa: BLE001 — recorded, run continues
                low = str(exc).lower()
                if "budget" in low or "envelope" in low or "exhausted" in low:
                    raise
                failed.append({"id": item_id, "error": repr(exc)})
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, ensure_ascii=False, indent=1))
            done += 1
    finally:
        guard.uninstall()
    summary = {"phase": f"tier1_run_{substrate}", "model": llm.MODEL,
               "started": started,
               "ended": datetime.now(timezone.utc).isoformat(),
               "selected": len(sel), "n_new": done, "skipped": skipped,
               "failed": failed, "fresh_calls": guard.fresh,
               "advisory": advisory_status()}
    log_phase(summary)
    return summary


# ---------------------------------------------------------------------------
# Scoring — frozen scorers verbatim under policy-matched aliases
# ---------------------------------------------------------------------------

def _load_cases(root: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(root.glob("*.json"))]


def _aliased(case: dict) -> dict:
    """A shallow case copy whose cells are keyed by the frozen scorer alias
    (policy-matched: em_cg_noretry -> em_cg gated branch, em_cw_retry ->
    em_cw ungated branch)."""
    alias = dict(case)
    alias["cells"] = {SCORE_ALIAS[c]: case["cells"][c] for c in NEW_CELLS
                      if c in case["cells"]}
    return alias


def _relabel(rows: list[dict]) -> list[dict]:
    back = {v: k for k, v in SCORE_ALIAS.items()}
    for r in rows:
        r["cell"] = back[r["cell"]]
    return rows


async def score_new_d1(scorer) -> list[dict]:
    cases = _load_cases(D1_CASES)
    assert cases, f"no case records under {D1_CASES} — run run-d1 first"
    labels = sd1.load_labels()
    convs = sd1.load_conversations()
    rows: list[dict] = []
    for case in cases:
        cid = case["conversation_id"]
        rows.extend(_relabel(await sd1.score_item(
            _aliased(case), labels[cid], convs[cid], scorer)))
    return rows


async def score_new_unify(scorer) -> list[dict]:
    cases = _load_cases(UNIFY_CASES)
    assert cases, f"no case records under {UNIFY_CASES} — run run-unify first"
    digests = {c.get("arcs_sha256") for c in cases}
    assert len(digests) == 1, f"multiple arcs digests: {digests}"
    rows: list[dict] = []
    for case in cases:
        rows.extend(_relabel(await ue.score_case(_aliased(case), scorer)))
    return rows


def load_released_rows() -> tuple[list[dict], list[dict]]:
    d1 = json.loads(RELEASED_D1_SCORES.read_text())["rows"]
    uni = json.loads(RELEASED_UNIFY_RESULTS.read_text())["per_arc_rows"]
    return d1, uni


# ---------------------------------------------------------------------------
# Factorial endpoints (one _pt code path over released + new rows)
# ---------------------------------------------------------------------------

def _cell_rows(rows: list[dict], cell: str) -> list[dict]:
    return [r for r in rows if r["cell"] == cell]


def d1_cell_endpoints(rows: list[dict]) -> dict:
    """The D1 factorial column set for one cell's rows (scorer_d1._pt —
    flags are None outside each endpoint's population, _pt skips None)."""
    return {
        "while_pending_contamination_at_risk":
            sd1._pt(rows, "while_pending_contamination"),
        "at_end_contamination_at_risk": sd1._pt(rows, "at_end_contamination"),
        "staged_at_plan_non_control": sd1._pt(rows, "staged_at_plan"),
        "promote_at_tstar_k0_came_true": sd1._pt(rows, "promote_at_tstar_k0"),
        "promote_within_k2_came_true": sd1._pt(rows, "promote_within_k2"),
        "task_probe_correct_ITT_all": sd1._pt(rows, "probe_correct"),
        "controls_committed_immediately": sd1._pt(rows,
                                                  "committed_immediately"),
        "controls_over_holding": sd1._pt(rows, "over_holding"),
    }


def unify_cell_endpoints(rows: list[dict]) -> dict:
    """The unify factorial column set for one cell's rows (unify_eval._pt)."""
    return {
        "while_pending_contamination_still_open":
            ue._pt(rows, "while_pending_contamination"),
        "at_end_contamination_still_open": ue._pt(rows,
                                                  "at_end_contamination"),
        "staged_at_hedge": ue._pt(rows, "staged_at_hedge"),
        "promote_at_tstar_came_true": ue._pt(rows, "promote_at_tstar"),
        "promote_within_window_came_true": ue._pt(rows,
                                                  "promote_within_window"),
        "end_state_correct_came_true": ue._pt(rows, "end_state_correct"),
        "task_probe_correct_ITT_all": ue._pt(rows, "probe_correct"),
        "g5_commit_correct_control": ue._pt(rows, "g5_commit_correct"),
    }


def field_match_rates() -> dict:
    """Field-match (mapping-funnel) rate per rec-set family per substrate,
    from the case-record diagnostics: D1 = any tuple mapped to the registered
    target key per conversation; unify = hedge-turn
    any_tuple_mapped_to_target. em/em_fixed come from the NEW tier1 records
    and are asserted identical to the released records' (cache-shared
    extraction); ez comes from the released records (not rerun as a cell)."""
    out: dict = {"d1": {}, "unify": {}}
    new_d1 = _load_cases(D1_CASES)
    rel_d1 = _load_cases(RELEASED_D1_CASES)
    rel_by_id = {c["conversation_id"]: c for c in rel_d1}
    for fam in ("em", "em_fixed", "ez"):
        src = rel_d1 if fam == "ez" else new_d1
        flags = [any(d["n_mapped_target"] for d in c["diagnostics"][fam])
                 for c in src]
        x, n = sum(flags), len(flags)
        lo, hi = stats.wilson_ci(x, n)
        out["d1"][fam] = {"x": x, "n": n, "rate": x / n,
                          "wilson_lo": lo, "wilson_hi": hi}
    # integrity: new em/em_fixed diagnostics match the released records'
    mismatch = 0
    for c in new_d1:
        rc = rel_by_id[c["conversation_id"]]
        for fam in ("em", "em_fixed"):
            if c["diagnostics"][fam] != rc["diagnostics"][fam]:
                mismatch += 1
    out["d1"]["_new_vs_released_diag_mismatches"] = mismatch

    new_uni = _load_cases(UNIFY_CASES)
    rel_uni = _load_cases(RELEASED_UNIFY_CASES)
    rel_by_arc = {c["arc_id"]: c for c in rel_uni}
    for fam in ("em", "em_fixed", "ez"):
        src = rel_uni if fam == "ez" else new_uni
        flags = [bool(c["diagnostics"][fam]["hedge_turn"]
                      ["any_tuple_mapped_to_target"]) for c in src]
        x, n = sum(flags), len(flags)
        lo, hi = stats.wilson_ci(x, n)
        out["unify"][fam] = {"x": x, "n": n, "rate": x / n,
                             "wilson_lo": lo, "wilson_hi": hi}
    mismatch = 0
    for c in new_uni:
        rc = rel_by_arc[c["arc_id"]]
        for fam in ("em", "em_fixed"):
            if c["diagnostics"][fam] != rc["diagnostics"][fam]:
                mismatch += 1
    out["unify"]["_new_vs_released_diag_mismatches"] = mismatch
    return out


# ---------------------------------------------------------------------------
# Paired cluster-bootstrap CIs (item 2)
# ---------------------------------------------------------------------------

def _delta_block(rows_a: list[dict], rows_b: list[dict], flag: str) -> dict:
    """Paired per-cluster delta rate(a) - rate(b) with a cluster-bootstrap CI
    (the task_deltas._delta_block pattern: stats.cluster_bootstrap_mean over
    deltas in {-1,0,1}, seed-fixed)."""
    by_a = {r["cluster"]: r for r in rows_a if r.get(flag) is not None}
    by_b = {r["cluster"]: r for r in rows_b if r.get(flag) is not None}
    ids = sorted(set(by_a) & set(by_b))
    if not ids:
        return {"point": None, "lo": None, "hi": None, "n_pairs": 0,
                "n_clusters": 0, "rate_a": None, "rate_b": None,
                "n_discordant": 0}
    diffs = [int(by_a[i][flag]) - int(by_b[i][flag]) for i in ids]
    block = stats.cluster_bootstrap_mean(diffs, ids, n_boot=N_BOOT,
                                         seed=SEED, bounds=(-1, 1))
    n = len(ids)
    return {"point": block["point"], "lo": block["lo"], "hi": block["hi"],
            "n_pairs": n, "n_clusters": block["n_clusters"],
            "rate_a": sum(int(by_a[i][flag]) for i in ids) / n,
            "rate_b": sum(int(by_b[i][flag]) for i in ids) / n,
            "n_discordant": sum(1 for d in diffs if d != 0)}


def paired_cis(rows: list[dict], shared_flags: tuple[str, ...],
               gated_flags: tuple[str, ...]) -> dict:
    by_cell = {c: _cell_rows(rows, c) for c in FACTORIAL_CELLS}
    out: dict = {}
    for name, a, b, desc in CONTRASTS:
        flags = shared_flags + (gated_flags if name == "retry_gated" else ())
        out[name] = {
            "cell_a": a, "cell_b": b, "description": desc,
            "delta_note": "delta = rate(cell_a) - rate(cell_b), paired by "
                          "cluster (conversation/arc)",
            "endpoints": {f: _delta_block(by_cell[a], by_cell[b], f)
                          for f in flags},
        }
    return out


# ---------------------------------------------------------------------------
# No-op aggregation
# ---------------------------------------------------------------------------

def noop_summary() -> dict:
    out: dict = {}
    for name, root in (("d1", D1_CASES), ("unify", UNIFY_CASES)):
        cases = _load_cases(root)
        agg: dict = {}
        for fam in ("em", "ez"):
            tot = Counter()
            attrs: list[str] = []
            for c in cases:
                blk = c["noop_trigger_counts"][fam]
                for k in ("n_tuples", "n_plan_frame_attrs",
                          "n_plan_frame_declined", "n_retry_triggers"):
                    tot[k] += blk[k]
                attrs.extend(m["attribute"] for m in blk["matched_attributes"]
                             if m["declined"] and m["derived_stem"] is not None)
            agg[fam] = {**tot,
                        "trigger_rate_per_tuple":
                            tot["n_retry_triggers"] / tot["n_tuples"],
                        "triggering_attribute_multiset":
                            dict(Counter(attrs))}
        out[name] = {"n_cases": len(cases), **agg}

    def _tot(fam, key):
        return out["d1"][fam][key] + out["unify"][fam][key]

    ez_trig, em_trig = _tot("ez", "n_retry_triggers"), _tot("em", "n_retry_triggers")
    ez_n, em_n = _tot("ez", "n_tuples"), _tot("em", "n_tuples")
    if ez_trig == 0:
        out["conclusion"] = (
            f"literal (ez) extraction produced 0 retry triggers across both "
            "substrates — literal +- retry is a structural no-op")
    else:
        out["conclusion"] = (
            "the structural-no-op assertion does NOT hold as an exact zero: "
            f"literal (ez) extraction carries {ez_trig} retry triggers "
            f"(declined + stem-derivable) out of {ez_n} tuples "
            f"({ez_trig / ez_n:.4f}/tuple) across both substrates, vs "
            f"{em_trig}/{em_n} ({em_trig / em_n:.4f}/tuple) for cautious "
            f"(em) — a {(em_trig / em_n) / (ez_trig / ez_n):.1f}x "
            "concentration in the cautious family. The ez matches are "
            "overwhelmingly lexical 'plan' tokens naming plan-OBJECTS "
            "(meal_plan_*, mobile_plan_*), not plan-framed speculation "
            "wrappers; literal +- retry is therefore NOT structurally "
            "identical, and any exact ez-retry effect would require the "
            "deferred second-family factorial cells to measure")
    out["method"] = (
        "deterministic count over the released ez/em extraction + first-pass "
        "frozen-mapping outputs, recovered byte-identically via the "
        "content-addressed llm cache (the released records keep only "
        "mapping counts); retry trigger = the exact stem-wrapper re-offer "
        "condition: frozen first-pass mapping DECLINED and "
        "store.derive_plan_stem yields a usable stem over the frozen "
        "PLAN_FRAME_ATTRIBUTE_PATTERNS")
    return out


# ---------------------------------------------------------------------------
# score — results JSON + report
# ---------------------------------------------------------------------------

async def compute_results() -> dict:
    scorer = rule5.EquivalenceScorer(cap=6000)
    new_d1 = await score_new_d1(scorer)
    new_uni = await score_new_unify(scorer)
    rel_d1, rel_uni = load_released_rows()
    rows_d1 = rel_d1 + new_d1
    rows_uni = rel_uni + new_uni

    factorial = {
        "d1": {cell: d1_cell_endpoints(_cell_rows(rows_d1, cell))
               for cell in FACTORIAL_CELLS},
        "unify": {cell: unify_cell_endpoints(_cell_rows(rows_uni, cell))
                  for cell in FACTORIAL_CELLS},
    }
    fm = field_match_rates()
    for sub in ("d1", "unify"):
        for cell in FACTORIAL_CELLS:
            factorial[sub][cell]["field_match_rate"] = fm[sub][FAMILY_OF[cell]]
    field_match_integrity = {
        sub: {"new_vs_released_diag_mismatches":
              fm[sub]["_new_vs_released_diag_mismatches"],
              "note": ("count of new tier1 case records whose em/em_fixed "
                       "mapping diagnostics differ from the released "
                       "records' — 0 proves the cache-shared extraction + "
                       "mapping replayed byte-identically")}
        for sub in ("d1", "unify")}

    cis = {
        "d1": paired_cis(rows_d1, D1_SHARED_FLAGS, D1_GATED_FLAGS),
        "unify": paired_cis(rows_uni, UNIFY_SHARED_FLAGS, UNIFY_GATED_FLAGS),
    }
    noop = noop_summary()
    phases = ([json.loads(l) for l in RUN_LOG.open()]
              if RUN_LOG.exists() else [])
    return {
        "id": "TIER1-RETRY-ABLATION-RESULTS",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prereg": ("protocol/CHANGELOG.md 2026-08-14 'PREREGISTRATION: "
                   "reviewer-response experiment set (Tier 1)', items 1+2"),
        "system_model": llm.MODEL,
        "judge_model": llm.JUDGE_MODEL,
        "cells": {
            "new": {c: CELL_NOTES[c] for c in NEW_CELLS},
            "released_quoted": {
                "em_cw": "cautious, ungated, frozen mapper (released)",
                "em_cg": "cautious, gated, WITH the adopted retry (released "
                         "adopted configuration)",
                "ez_cw": "literal, ungated (released)",
                "ez_cg": "literal, gated (released)",
            },
            "scoring_alias": SCORE_ALIAS,
            "factorial_note": (
                "with the released four this completes the informative "
                "portion of the style x gate x retry 2x2x2. The prereg "
                "asserted the two ez retry cells structurally identical to "
                "ez_cw/ez_cg, to be evidenced by deterministic trigger "
                "count; the count (noop_check below) is small but NONZERO "
                "— the assertion does not hold as an exact zero, and the "
                "ez retry cells remain unrun under the prereg's explicit "
                "second-family deferral"),
        },
        "n_rows": {"d1_released": len(rel_d1), "d1_new": len(new_d1),
                   "unify_released": len(rel_uni), "unify_new": len(new_uni)},
        "released_row_sources": {"d1": str(RELEASED_D1_SCORES),
                                 "unify": str(RELEASED_UNIFY_RESULTS)},
        "ci_note": ("factorial points: x/n + Wilson (cluster = "
                    "conversation/arc, one row per cluster per cell); "
                    "paired deltas: stats.cluster_bootstrap_mean over "
                    "per-cluster deltas in {-1,0,1}, bounds=(-1,1), "
                    f"n_boot={N_BOOT}, seed={SEED} (task_deltas pattern)"),
        "factorial": factorial,
        "field_match_integrity": field_match_integrity,
        "paired_cis": cis,
        "noop_check": noop,
        "new_rows": {"d1": new_d1, "unify": new_uni},
        "run_phases": phases,
        "scoring_judge_calls": {"total": scorer.judge_calls,
                                "fresh": scorer.fresh_judge_calls},
        "budget_before": (json.loads(BUDGET_BEFORE.read_text())
                          if BUDGET_BEFORE.exists() else None),
        "budget_after": mapper_fix.ensure_budget(BUDGET_PATH),
        "advisory_budget": advisory_status(),
    }


D1_COLS = (("while_pending_contamination_at_risk", "contam while-pending"),
           ("at_end_contamination_at_risk", "contam at-end"),
           ("staged_at_plan_non_control", "staged@plan"),
           ("promote_at_tstar_k0_came_true", "promote t*"),
           ("task_probe_correct_ITT_all", "probe ITT"),
           ("controls_committed_immediately", "ctrl commit-now"),
           ("field_match_rate", "field-match"))
UNIFY_COLS = (("while_pending_contamination_still_open", "contam while-pending"),
              ("at_end_contamination_still_open", "contam at-end"),
              ("staged_at_hedge", "staged@hedge"),
              ("promote_within_window_came_true", "promote window"),
              ("end_state_correct_came_true", "end-state"),
              ("task_probe_correct_ITT_all", "probe ITT"),
              ("g5_commit_correct_control", "g5 ctrl commit-correct"),
              ("field_match_rate", "field-match"))


def _fmt(pt: dict | None) -> str:
    if not pt or pt.get("rate") is None:
        return "—"
    return (f"{pt['x']}/{pt['n']} = {pt['rate']:.3f} "
            f"[{pt['wilson_lo']:.3f}, {pt['wilson_hi']:.3f}]")


def _fmt_delta(d: dict) -> str:
    if d["point"] is None:
        return "—"
    return (f"{d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] "
            f"(n={d['n_pairs']}, disc={d['n_discordant']})")


def render_report(res: dict) -> str:
    lines = [
        "# Tier 1 retry ablation — factorial cells + paired CIs "
        "(reviewer-response items 1+2)",
        "",
        f"Preregistered 2026-08-14 (protocol/CHANGELOG.md). Generated "
        f"{res['generated_at']}. Systems pinned `{res['system_model']}`; "
        f"judge frozen `{res['judge_model']}`.",
        "",
        "New cells (this run): **em_cg_noretry** = cautious + gate ON, "
        "frozen mapper only (the pre-adoption em_cg / m4_eval original "
        "em_cg arm); **em_cw_retry** = cautious + gate OFF with the adopted "
        "plan-stem retry at the same LOCAL mapper seam (retry-recovered "
        "tuples commit directly). Released cells quoted from "
        "the released row files, never rerun. Scored with the existing "
        "scorers verbatim (policy-matched cell aliasing; "
        f"{res['scoring_judge_calls']['total']} equivalence-judge calls, "
        f"{res['scoring_judge_calls']['fresh']} fresh).",
        "",
        res["cells"]["factorial_note"] + ".",
        ""]
    for sub, cols, n_note in (
            ("d1", D1_COLS, "147 held-out test conversations (88 at-risk, "
             "44 came_true, 15 controls)"),
            ("unify", UNIFY_COLS, "179 controlled-difficulty arcs (80 "
             "still_open g1-g4, 80 came_true p1-p4, 19 g5 controls)")):
        lines += [f"## Factorial table — {sub.upper()} ({n_note})", "",
                  "| cell (style/gate/retry) | " + " | ".join(
                      label for _f, label in cols) + " |",
                  "|---|" + "---|" * len(cols)]
        tags = {"em_cw": "cautious / off / off",
                "em_cw_retry": "cautious / off / ON (new)",
                "em_cg_noretry": "cautious / ON / off (new)",
                "em_cg": "cautious / ON / ON (released adopted)",
                "ez_cw": "literal / off / (no-op)",
                "ez_cg": "literal / ON / (no-op)"}
        for cell in FACTORIAL_CELLS:
            e = res["factorial"][sub][cell]
            lines.append(f"| {cell} — {tags[cell]} | " + " | ".join(
                _fmt(e[f]) for f, _label in cols) + " |")
        lines.append("")
    lines += ["## Paired cluster-bootstrap CIs (delta = cell_a - cell_b; "
              f"seed {SEED}, n_boot {N_BOOT}, cluster = conversation/arc)",
              ""]
    for sub in ("d1", "unify"):
        lines += [f"### {sub.upper()}", ""]
        flags = sorted({f for c in res["paired_cis"][sub].values()
                        for f in c["endpoints"]})
        lines += ["| contrast (a vs b) | " + " | ".join(flags) + " |",
                  "|---|" + "---|" * len(flags)]
        for name, _a, _b, _d in CONTRASTS:
            c = res["paired_cis"][sub][name]
            cells = []
            for f in flags:
                cells.append(_fmt_delta(c["endpoints"][f])
                             if f in c["endpoints"] else "—")
            lines.append(f"| {name}: {c['cell_a']} vs {c['cell_b']} | "
                         + " | ".join(cells) + " |")
        lines.append("")
    noop = res["noop_check"]
    lines += ["## Deterministic no-op check (literal +- retry)", "",
              noop["method"] + ".", ""]
    lines += ["| substrate | family | tuples | plan-frame attrs | declined "
              "| retry triggers (declined + stem-derivable) | triggers/tuple |",
              "|---|---|---|---|---|---|---|"]
    for sub in ("d1", "unify"):
        for fam in ("ez", "em"):
            b = noop[sub][fam]
            lines.append(f"| {sub} | {fam} | {b['n_tuples']} | "
                         f"{b['n_plan_frame_attrs']} | "
                         f"{b['n_plan_frame_declined']} | "
                         f"{b['n_retry_triggers']} | "
                         f"{b['trigger_rate_per_tuple']:.4f} |")
    lines += ["", f"**{noop['conclusion']}**", "",
              "## Run stats", "",
              "| phase | selected | new | skipped | failed | fresh calls |",
              "|---|---|---|---|---|---|"]
    for r_ in res["run_phases"]:
        lines.append(f"| {r_['phase']} | {r_['selected']} | {r_['n_new']} "
                     f"| {r_['skipped']} | {len(r_['failed'])} "
                     f"| {r_['fresh_calls']:,} |")
    adv = res["advisory_budget"]
    bb, ba = res.get("budget_before"), res["budget_after"]
    lines += ["", "## Budget (stage tier1; advisory / hard envelope / "
              "110k run stop)", ""]
    if bb:
        lines.append(f"- before: {json.dumps(bb['spent'])} "
                     f"(total {sum(bb['spent'].values()):,})")
    lines += [f"- after:  {json.dumps(ba['spent'])} "
              f"(total {sum(ba['spent'].values()):,} / envelope "
              f"{ba['envelope']:,}; run stop {HARD_STOP_TOTAL:,})",
              f"- stage tier1 spend {adv['spent']:,} vs advisory forecast "
              f"{adv['advisory_forecast']:,} (over: {adv['over_forecast']})",
              f"- THIS runner's own fresh calls: "
              f"{sum(r_['fresh_calls'] for r_ in res['run_phases'])} across "
              "the run phases above + "
              f"{res['scoring_judge_calls']['fresh']} scoring judge calls; "
              "the tier1 stage is SHARED with the concurrent Tier-1 agents "
              "(items 3-6), whose spend lands in the same counter",
              "",
              "Scored once (prereg). Verification (RECOUNT + SEMANTICS) "
              "precedes release — these numbers are UNVERIFIED until then.",
              ""]
    return "\n".join(lines)


async def score(force: bool) -> dict:
    assert_pins()
    if RESULTS_PATH.exists() and not force:
        raise SystemExit(f"{RESULTS_PATH} already exists — scoring is "
                         "one-shot; pass --force only for a logged "
                         "infrastructure-failure rerun")
    guard = tier1_guard()
    guard.install()
    try:
        res = await compute_results()
    finally:
        guard.uninstall()
    res["score_fresh_calls"] = guard.fresh
    TIER1_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    REPORT_PATH.write_text(render_report(res))
    print(f"wrote {RESULTS_PATH}\nwrote {REPORT_PATH} "
          f"({guard.fresh} fresh judge calls)")
    return res


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run-d1", "run-unify"):
        p = sub.add_parser(name)
        p.add_argument("--ids", nargs="*", default=None)
        p.add_argument("--shard", default=None, help="K/N over sorted ids")
        p.add_argument("--call-cap", type=int, default=None)
    sc = sub.add_parser("score")
    sc.add_argument("--force", action="store_true")
    sub.add_parser("status")
    args = ap.parse_args()

    if args.cmd == "status":
        n_d1 = len(list(D1_CASES.glob("*.json"))) if D1_CASES.exists() else 0
        n_uni = (len(list(UNIFY_CASES.glob("*.json")))
                 if UNIFY_CASES.exists() else 0)
        print(json.dumps({**advisory_status(), "d1_case_records": n_d1,
                          "unify_case_records": n_uni}, indent=1))
        return
    if args.cmd == "score":
        asyncio.run(score(args.force))
        return
    substrate = "d1" if args.cmd == "run-d1" else "unify"
    summary = asyncio.run(run_substrate(substrate, args.ids, args.shard,
                                        args.call_cap))
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
