"""plan_typed.py — the preregistered PLAN-TYPED-COMMIT baseline
(protocol/CHANGELOG.md 2026-08-12 "PREREGISTRATION: plan-typed-commit
baseline (reviewer-motivated ablation)" — the binding spec).

MECHANISM (two arms, em_pt / ez_pt): the identical frozen pipeline through
classification — extraction (gated_store.extract_turn_family under
syn_res.LME_PROMPTS), field matching (store._map_tuple_key / the adopted
LOCAL stem wrapper, see IC-2), classifier (classify.classify_item) — with
the gate's ROUTING replaced, at the same seam the gate policy occupies
(the graded_eval/m4_eval `_gated_tuples` vs `_ungated_tuples` policy-body
seam):

  speculative            -> COMMIT (store_id main) with type=plan on the
                            stored value record (provenance metadata) and
                            on the write event; validity history preserved
                            — the prior value's interval closes normally.
  completed_event /
  explicit_correction /
  other                  -> EXACTLY the frozen ungated write behavior
                            (graded_eval._ungated_tuples: write with
                            correction=False; supersede-on-different,
                            refresh-on-equal; no additive_noop, no
                            retro-invalidation).

READ RULE (deterministic, part of the arm definition — the arm's ONLY
notion of current state, used for the probe path AND the pipeline's
stored-value lookup): the most recent NON-plan-typed committed value for
the field; plan-typed values are NEVER returned as current state; if no
non-plan value exists, abstention-by-absence. Confirmations resolve via
route 2 only (the confirming turn's own commit) — this baseline has no
route 1: a non-plan commit of a (normalize-)equal value refreshes the
plan-typed current entry (the frozen Z7 behavior, history byte-identical)
and the arm-local wrapper marks that entry CONFIRMED so the read rule
returns it.

NO prompt, store, gate or mapper default is modified: the plan-type marker
lives ONLY in the arm-local provenance/event metadata this runner passes to
the frozen ZepStore.write; assert_mapper_mode_off holds at entry and
teardown of every arc.

SUBSTRATES (one run each, resumable per arc):
  - internal M4: the 60 r4 arcs (results/v2.4/m4/arcs.jsonl), the m4_eval
    harness pattern; case records -> results/plan_typed/m4/cases/.
  - unified ladders: the 179 usable arcs (results/unify/arcs.jsonl minus
    unify_exclusions.json), the unify_eval harness pattern; case records ->
    results/plan_typed/unify/cases/.

MODELS: systems = the PINNED default gpt-5.5 (MEMGATE_SYSTEM_MODEL unset —
asserted); scoring judge = the frozen llm.JUDGE_MODEL via
rule5.EquivalenceScorer.

ENDPOINTS (score once -> results/plan_typed/plan_typed_results.json +
plan_typed_report.md; per the prereg, each substrate's released endpoint
set; comparison rows QUOTED from the released artifacts — never rerun):
  - contamination while-pending / at-end: a plan-typed value RETURNED AS
    CURRENT (i.e. the hedged value returned by the arm's read rule) counts
    as contamination; a plan-typed value present but MASKED by the read
    rule does not — BOTH counts are reported.
  - task probes ITT (deterministic read-rule lookup; unanswered =
    incorrect on unify; M4 abstention branch per m4_eval).
  - came_true / resolve_true end-state correctness overall and by
    confirmation-wording distance (unified cw ladder p1-p4 — the
    discriminating comparison vs the gate's route-1 promotion).
  - controls: unify g5 commit-correct.
  - directional-expectation scorecard (3 preregistered rows): contamination
    ~= gated; oblique/implied came-true correctness BELOW gated; no
    over-holding by construction — evaluated under the disclosed
    operationalizations below.

CI conventions mirror each substrate's released artifact: M4 = cluster
bootstrap (cluster = arc, SEED_M4); unify = Wilson (one row per
conversation per arm per endpoint).

INTERPRETIVE CHOICES (spec ambiguities resolved here, disclosed):
  IC-1  The classifier's stored_value = the arm's READ-RULE current state
        (the spec places the read rule on "the probe/lookup path"; the read
        rule is the arm's only notion of current state, so the store-
        conditioned classifier sees it — which also keeps classifier calls
        cache-shared with the released gated cells wherever the fact-
        visible state coincides).
  IC-2  em field matching mirrors each substrate's released gated
        comparison cell: M4 em_pt uses the frozen store._map_tuple_key
        (byte-identical to the released em_cw/em_cg mapping calls); unify
        em_pt uses the ADOPTED M1 plan-stem wrapper as a LOCAL mapper
        (byte-identical to the released unify em_cg recs). ez_pt uses the
        frozen mapper on both. Never installed globally.
  IC-3  explicit_correction writes with correction=False — the prereg's
        "exactly the frozen ungated write behavior" (zep_arm_design §4: the
        ungated column has no correction/update distinction; lookup-
        identical either way).
  IC-4  Route-2 refresh-confirm: when a NON-plan commit refreshes a
        plan-typed current entry (equal value, frozen Z7 refresh — no
        history change), the arm-local wrapper flips that entry's
        provenance type to fact (confirmed_seq/class recorded). Without
        this, the baseline's only resolution route could never resolve a
        confirmation onto an equal stored plan value.
  IC-5  M4 while-pending is computed from the recorded per-turn value
        timeline (D1 window semantics; window = strictly after the hedge
        turn, through the pre-resolution turn — through the last user turn
        for open arcs) as a SUPPLEMENTARY row; M4's released comparison
        rows are at-end contamination + the pre-resolution-boundary
        supplementary, quoted as such.
  IC-6  Scorecard operationalizations (the prereg gives directions only):
        row 1 holds iff |pt - released gated| <= 0.05 on pooled at-end
        contamination, per family, on BOTH substrates (unify pooled g1-g4;
        M4 pooled abandoned+open); row 2 holds iff pt came-true end-state
        < the released gated cell on pooled p3+p4, per family; row 3 holds
        iff (a) zero arcs where the read rule masked a NON-plan current
        value (structural, counted from the records) and (b) unify g5
        commit-correct >= the released gated cell - 0.05, per family.

BUDGET: stage "plan_typed" ADVISORY 12,000, registered in
results/v2.4/budget.json from unallocated headroom (the claude_repl/unify
registration convention); spend recorded per fresh call via the
flock-guarded forecast_v22.SharedCallBudget. The 120k envelope stays the
protocol hard stop; THIS run additionally refuses to start / stops if the
combined total would pass 110,000 (prereg stop condition), enforced via
local_cap, plus an explicit early projection check that aborts when the
projected total exceeds 110k. Heavy cache-sharing expected (extraction /
mapping / classification replay from the released runs' cache).

CLI (repo root; MEMGATE_SYSTEM_MODEL must be UNSET):
  pilot/.venv/bin/python experiments/v2_full/plan_typed.py \
      run-m4    [--ids ...] [--call-cap N]
      run-unify [--ids ...] [--shard K/N] [--call-cap N]
      score
      status
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401  (R1 layout: puts src/ + experiments/ on sys.path)

import classify
import forecast_v22
import gated_store
import llm
import m4_eval
import mapper_fix
import natural_check_rule5 as rule5
import stats
import store
import unify_eval
import zepstore
from graded_eval import _CellSim, _read_jsonl, _rel
from router_sweep import PLAN_FRAME_ATTRIBUTE_PATTERNS

from paths import RESULTS_DIR

STAGE = "plan_typed"
PINNED_SYSTEM_MODEL = "gpt-5.5-2026-04-23"
ADVISORY_FORECAST = 12_000
#: Prereg stop condition for THIS run (the protocol hard stop stays 120k).
HARD_STOP_TOTAL = 110_000

V24_DIR = RESULTS_DIR / "v2.4"
BUDGET_PATH = V24_DIR / "budget.json"

PT_DIR = RESULTS_DIR / "plan_typed"
M4_PT_DIR = PT_DIR / "m4"
M4_CASES = M4_PT_DIR / "cases"
UNIFY_PT_DIR = PT_DIR / "unify"
UNIFY_CASES = UNIFY_PT_DIR / "cases"
RESULTS_PATH = PT_DIR / "plan_typed_results.json"
REPORT_PATH = PT_DIR / "plan_typed_report.md"
RUN_LOG = PT_DIR / "run_log.jsonl"
BUDGET_BEFORE = PT_DIR / "budget_before.json"

#: Released artifacts the comparison rows QUOTE (read-only, never rerun).
RELEASED_M4 = V24_DIR / "m4" / "eval" / "m4_eval_results.json"
RELEASED_UNIFY = RESULTS_DIR / "unify" / "run" / "unify_results.json"

ARMS = ("em_pt", "ez_pt")
FAMILIES = ("em", "ez")
#: Released comparison cell per pt arm (gated; ungated quoted alongside).
GATED_OF = {"em_pt": "em_cg", "ez_pt": "ez_cg"}
UNGATED_OF = {"em_pt": "em_cw", "ez_pt": "ez_cw"}

M4_CLASSES = ("resolve_true", "abandoned", "open")
G_LEVELS = unify_eval.G_LEVELS
P_LEVELS = unify_eval.P_LEVELS
STILL_OPEN_LEVELS = unify_eval.STILL_OPEN_LEVELS

#: Scorecard tolerance (the repo's APPROX_ZERO convention, disclosed IC-6).
APPROX = 0.05

MECHANISM_NOTE = (
    "plan-typed-commit baseline: speculative -> commit with type=plan on the "
    "stored value record/event metadata (validity history preserved; prior "
    "interval closes normally); completed_event/explicit_correction/other -> "
    "exactly the frozen ungated write behavior. Read rule: most recent "
    "NON-plan-typed committed value, else abstention; plan values are never "
    "returned as current. No holding area, no resolution predicates, no "
    "checker; confirmations resolve via route 2 only (the confirming turn's "
    "own commit — refresh-confirm on equal values, IC-4).")


# ---------------------------------------------------------------------------
# Budget (advisory stage plan_typed; 110k prereg stop via local_cap)
# ---------------------------------------------------------------------------

def register_stage(path: Path = BUDGET_PATH) -> dict:
    """Register the plan_typed stage in the combined budget file — the
    established convention (external/claude_repl/unify): allocation from
    unallocated headroom + a dated note; spent counter starts at 0.
    Idempotent; flock + atomic rename (the SharedCallBudget discipline)."""
    lock_path = path.with_suffix(".lock")
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        budget = json.loads(path.read_text())
        if STAGE in budget["allocation"] and STAGE in budget["spent"]:
            assert budget["allocation"][STAGE] == ADVISORY_FORECAST, (
                f"stage {STAGE} already registered with allocation "
                f"{budget['allocation'][STAGE]} != {ADVISORY_FORECAST}")
            return budget
        headroom = budget["allocation"].get("unallocated_headroom", 0)
        assert headroom >= ADVISORY_FORECAST, (
            f"unallocated_headroom {headroom} < {ADVISORY_FORECAST} — cannot "
            "register plan_typed from headroom")
        budget["allocation"]["unallocated_headroom"] = (
            headroom - ADVISORY_FORECAST)
        budget["allocation"][STAGE] = ADVISORY_FORECAST
        budget["spent"].setdefault(STAGE, 0)
        budget.setdefault("notes", []).append(
            "2026-08-12: plan_typed 12000 from headroom (plan-typed-commit "
            "baseline, reviewer-motivated ablation; preregistration changelog "
            "2026-08-12). Advisory stage per the 2026-07-08 amendment; 120k "
            "envelope hard; the run refuses to push TOTAL spend past 110,000 "
            "(prereg stop condition).")
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(budget, indent=1, ensure_ascii=False))
        tmp.replace(path)
        return budget


def pt_guard(path: Path = BUDGET_PATH,
             local_cap: int | None = None) -> forecast_v22.SharedCallBudget:
    """The unify_guard pattern, stage plan_typed: the stage forecast is
    ADVISORY (warn + continue); this run's own stop condition caps fresh
    calls so the combined total cannot pass 110k."""
    register_stage(path)
    budget = mapper_fix.ensure_budget(path)
    spent = budget["spent"].get(STAGE, 0)
    forecast = budget["allocation"].get(STAGE, ADVISORY_FORECAST)
    if spent >= forecast:
        print(f"[plan_typed budget ADVISORY] stage spend {spent:,} >= "
              f"forecast {forecast:,} — continuing (stages are advisory per "
              "the 2026-07-08 amendment; only the total envelope is hard)")
    margin_110 = HARD_STOP_TOTAL - sum(budget["spent"].values())
    if margin_110 <= 0:
        raise SystemExit(
            f"plan_typed STOP CONDITION: combined spend "
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
                " / 110k run stop (2026-08-12 plan-typed prereg)",
    }


def total_spent(path: Path = BUDGET_PATH) -> int:
    return sum(mapper_fix.ensure_budget(path)["spent"].values())


# ---------------------------------------------------------------------------
# Pins + bookkeeping
# ---------------------------------------------------------------------------

def assert_pins() -> None:
    assert os.environ.get("MEMGATE_SYSTEM_MODEL") is None, (
        "MEMGATE_SYSTEM_MODEL is set — the plan-typed baseline is "
        "preregistered on the PINNED default gpt-5.5 systems")
    assert llm.MODEL == PINNED_SYSTEM_MODEL, (
        f"llm.MODEL is {llm.MODEL!r} — expected the pinned "
        f"{PINNED_SYSTEM_MODEL}")
    assert llm.JUDGE_MODEL == PINNED_SYSTEM_MODEL, (
        f"llm.JUDGE_MODEL drifted to {llm.JUDGE_MODEL!r}")
    m4_eval.assert_mapper_mode_off()


def snapshot_budget_before() -> None:
    if not BUDGET_BEFORE.exists():
        PT_DIR.mkdir(parents=True, exist_ok=True)
        BUDGET_BEFORE.write_text(BUDGET_PATH.read_text())


def log_phase(row: dict) -> None:
    PT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUN_LOG, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _shard(ids: list[str], spec: str | None) -> list[str]:
    if not spec:
        return ids
    k, n = (int(x) for x in spec.split("/"))
    assert 0 <= k < n, f"bad shard spec {spec!r}"
    return ids[k::n]


# ---------------------------------------------------------------------------
# The arm-local read rule + write wrapper (the plan-type marker lives here)
# ---------------------------------------------------------------------------

PLAN = "plan"
FACT = "fact"


def _entry_type(entry) -> str:
    """The arm-local value type of one ZepEntry: read from the provenance
    metadata this runner wrote. Entries written by frozen cells carry no
    marker and read as fact — but pt stores only ever contain pt writes."""
    prov = entry.provenance if isinstance(entry.provenance, dict) else {}
    return prov.get("value_type", FACT)


def read_current(zstore, key: tuple) -> str | None:
    """THE READ RULE (deterministic, part of the arm definition): the most
    recent NON-plan-typed committed value for the field — plan-typed values
    are never returned as current state; if no non-plan value exists,
    abstention-by-absence (None). Superseded fact entries stay recoverable
    (their validity interval was closed normally by the plan-typed commit,
    which is exactly why the prior value is returnable)."""
    best = None
    for entry in zstore.history(key):
        if _entry_type(entry) == PLAN:
            continue
        if best is None or entry.created_seq > best.created_seq:
            best = entry
    return best.value if best is not None else None


def plan_masked_value(zstore, key: tuple) -> str | None:
    """The raw CURRENT value when it is a plan-typed entry the read rule is
    masking, else None (the 'plan-typed present but masked' count)."""
    entry = zstore.current_entry(key)
    if entry is not None and _entry_type(entry) == PLAN:
        return entry.value
    return None


async def _plan_typed_tuples(sim: _CellSim, rec: dict, turns: list[dict],
                             family: str, arc_id: str,
                             classify_fn=None) -> None:
    """The plan-typed policy body for one user turn — the arm-level wrapper
    at the same seam the gate policy occupies (the m4_eval/unify_eval
    `_gated_tuples` / `_ungated_tuples` policy-body convention).

    Identical frozen pipeline through classification (the item packaging is
    the `_gated_tuples` body verbatim; stored_value = the arm's read-rule
    current state, IC-1); the routing is the prereg's: speculative ->
    plan-typed commit; every other class -> exactly the frozen ungated
    write (`graded_eval._ungated_tuples` semantics: correction=False,
    supersede/refresh, nothing else)."""
    classify_call = classify.classify_item if classify_fn is None else classify_fn
    stream = [{"session_idx": t["session_idx"], "role": t["role"],
               "content": t["content"], "date": t["date"]} for t in turns]
    for j, (tup, mres) in enumerate(zip(rec["tuples"], rec["mapped"])):
        cand, _raw_key, _rationale, _meta = gated_store._unpack_map_result(mres)
        sim.diag["n_tuples"] += 1
        if cand is None:
            sim.diag["n_unmapped"] += 1
            continue
        sim.diag["n_mapped"] += 1
        store_id = sim.route(cand.key)
        zstore = sim.zstore(store_id)
        stored_now = read_current(zstore, cand.key)  # IC-1: read-rule state
        item = {
            "item_id": f"pt-{family}-{arc_id}-s{rec['session_idx']}"
                       f"t{rec['turn_idx']}-{j}",
            "entity": cand.entity, "attribute": cand.attribute,
            "asserted_value": tup.value, "stored_value": stored_now,
            "turn_text": rec["turn_text"],
            "preceding_turns": gated_store._preceding_turns(stream, rec["pos"]),
        }
        cls_out = await classify_call(item)
        cls = cls_out["argmax"]
        sim.diag[f"class_{cls}"] += 1
        gate_row = {"item_id": item["item_id"],
                    "session_idx": rec["session_idx"],
                    "turn_idx": rec["turn_idx"], "store_id": store_id,
                    "key": cand.name, "value": tup.value, "class": cls,
                    "s_ev": cls_out["s_ev"], "probs": cls_out["probs"]}
        if cls == "speculative":
            # plan-typed commit: the frozen write (correction=False), with
            # the type marker on the value record + the event metadata.
            event = zstore.write(
                cand.key, tup.value, valid_from=rec["date"],
                provenance={"class": cls, "value_type": PLAN},
                correction=False)
            event["value_type"] = PLAN
            gate_row["action"] = f"plan_{event['action']}"
            gate_row["value_type"] = PLAN
            sim.diag["n_plan_writes"] += int(event["action"] != "refresh")
        else:
            # exactly the frozen ungated write behavior (prereg; IC-3).
            event = zstore.write(
                cand.key, tup.value, valid_from=rec["date"],
                provenance={"class": cls, "value_type": FACT},
                correction=False)
            event["value_type"] = FACT
            gate_row["action"] = event["action"]
            gate_row["value_type"] = FACT
            sim.diag["n_fact_writes"] += int(event["action"] != "refresh")
            if event["action"] == "refresh":
                cur = zstore.current_entry(cand.key)
                if cur is not None and _entry_type(cur) == PLAN:
                    # IC-4 route-2 refresh-confirm: history untouched (the
                    # frozen Z7 refresh); the arm-local type flips to fact.
                    cur.provenance["value_type"] = FACT
                    cur.provenance["confirmed_seq"] = event["seq"]
                    cur.provenance["confirmed_class"] = cls
                    event["plan_confirmed"] = True
                    gate_row["action"] = "refresh_confirms_plan"
                    sim.diag["n_plan_confirms"] += 1
        sim.gate_rows.append(gate_row)


async def _run_plan_typed_cell(recs: list[dict], family: str, arc_id: str,
                               target: store.CanonicalKey, turns: list[dict],
                               user_index: dict, res_pos: int, hedge_pos: int,
                               classify_fn=None) -> dict:
    """One pt arm over a family's per-turn recs — the `_run_policy_cell`
    record shape (final/pre-resolution/timeline/events) with the read-rule
    value as the arm's answer and the raw + masked values reported beside.
    A fresh _CellSim per call keeps every arm's store state independent of
    every frozen cell (arm locality)."""
    sim = _CellSim("plan_typed", "lme")
    store_id = sim.route(target.key)
    pre_resolution_value = pre_resolution_raw = None
    pre_resolution_masked = False
    target_events: list[dict] = []
    value_timeline: list[dict] = []
    for rec in recs:
        offset = user_index[rec["pos"]] - user_index[res_pos]
        if rec["pos"] == res_pos:
            z = sim.zstore(store_id)
            pre_resolution_value = read_current(z, target.key)
            pre_resolution_raw = z.lookup(target.key)
            pre_resolution_masked = plan_masked_value(z, target.key) is not None
        n_events = len(sim.zstore(store_id).events)
        await _plan_typed_tuples(sim, rec, turns, family, arc_id,
                                 classify_fn=classify_fn)
        for event in sim.zstore(store_id).events[n_events:]:
            if tuple(event["key"]) != target.key:
                continue
            target_events.append({
                **m4_eval._jsonable_event(event),
                "session_idx": rec["session_idx"],
                "turn_idx": rec["turn_idx"],
                "user_idx": user_index[rec["pos"]],
                "user_offset_from_resolution": offset,
                "at_hedge_turn": rec["pos"] == hedge_pos,
            })
        z = sim.zstore(store_id)
        masked_now = plan_masked_value(z, target.key)
        value_timeline.append({
            "session_idx": rec["session_idx"], "turn_idx": rec["turn_idx"],
            "user_idx": user_index[rec["pos"]],
            "user_offset_from_resolution": offset,
            "value": read_current(z, target.key),
            "raw_value": z.lookup(target.key),
            "masked_plan": masked_now is not None})
    z = sim.zstore(store_id)
    final = read_current(z, target.key)
    final_raw = z.lookup(target.key)
    final_masked = plan_masked_value(z, target.key) is not None
    return {
        "final_value": final,
        "final_value_raw": final_raw,
        "final_masked_plan": final_masked,
        "final_present": (final is not None
                          and final != zepstore.UNKNOWN_SENTINEL),
        "pre_resolution_value": pre_resolution_value,
        "pre_resolution_value_raw": pre_resolution_raw,
        "pre_resolution_masked_plan": pre_resolution_masked,
        "target_value_after_turn": value_timeline,
        "target_events": target_events,
        "plan_writes_target": sum(1 for e in target_events
                                  if e.get("value_type") == PLAN
                                  and e["action"] != "refresh"),
        "fact_writes_target": sum(1 for e in target_events
                                  if e.get("value_type") == FACT
                                  and e["action"] != "refresh"),
        "plan_confirms_target": sum(1 for e in target_events
                                    if e.get("plan_confirmed")),
        "gate_rows_target": [g for g in sim.gate_rows
                             if g["key"] == target.name],
        "diag": dict(sim.diag),
    }


# ---------------------------------------------------------------------------
# run — M4 substrate (m4_eval harness pattern)
# ---------------------------------------------------------------------------

async def run_arc_m4(arc: dict, key_row: dict, arcs_digest: str) -> dict:
    arc_id = arc["arc_id"]
    geom = m4_eval.arc_geometry(arc)
    turns, user_positions = geom["turns"], geom["user_positions"]
    user_index, hedge_pos, res_pos = (geom["user_index"], geom["hedge_pos"],
                                      geom["res_pos"])
    target = m4_eval.arc_canonical_key(key_row)
    candidates = [target]  # m4_eval interpretive choice #3

    out = {"arc_id": arc_id, "substrate": "m4",
           "resolution_type": arc["resolution_type"],
           "hedge_strength": arc["hedge_strength"],
           "cluster": arc_id,
           "target_key": target.name,
           "hedged_value": arc["value"],
           "probe_question": key_row["probe_question"],
           "established_prior_value": m4_eval.expected_prior_memory(arc),
           "prior_value": arc.get("prior_value"),
           "abandoned_mode": arc.get("abandoned_mode"),
           "alternative_value": arc.get("alternative_value"),
           "arc_status": arc.get("status"), "arc_usable": arc.get("usable"),
           "arcs_sha256": arcs_digest,
           "mechanism": MECHANISM_NOTE,
           "mapper_note": ("em_pt: frozen store._map_tuple_key (the released "
                           "M4 em_cw/em_cg mapping, cache-shared; IC-2); "
                           "ez_pt: frozen mapper"),
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
           "cells": {}, "diagnostics": {}}

    m4_eval.assert_mapper_mode_off()  # frozen mapper untouched at arc entry
    for family in FAMILIES:
        recs = [await m4_eval._extract_and_map(family, turns, pos, candidates,
                                               arc_id)
                for pos in user_positions]
        hedge_rec = next(r for r in recs if r["pos"] == hedge_pos)
        out["diagnostics"][family] = {
            "hedge_turn": m4_eval._hedge_turn_diagnostics(
                hedge_rec, arc["hedge_text"], target),
            "per_turn": [m4_eval._turn_diag_row(rec, user_index[rec["pos"]],
                                                target) for rec in recs],
        }
        out["cells"][f"{family}_pt"] = await _run_plan_typed_cell(
            recs, family, arc_id, target, turns, user_index, res_pos,
            hedge_pos)
    m4_eval.assert_mapper_mode_off()  # teardown
    return out


# ---------------------------------------------------------------------------
# run — unified substrate (unify_eval harness pattern)
# ---------------------------------------------------------------------------

async def run_arc_unify(arc: dict, key_row: dict, arcs_digest: str) -> dict:
    arc_id = arc["arc_id"]
    geom = m4_eval.arc_geometry(arc)
    turns, user_positions = geom["turns"], geom["user_positions"]
    user_index, hedge_pos, res_pos = (geom["user_index"], geom["hedge_pos"],
                                      geom["res_pos"])
    target = m4_eval.arc_canonical_key(key_row)
    candidates = [target]

    out = {"arc_id": arc_id, "substrate": "unify",
           "unify_ladder": arc["unify_ladder"],
           "unify_level": arc["unify_level"],
           "outcome": unify_eval.arc_outcome(arc),
           "resolution_type": arc["resolution_type"],
           "hedge_strength": arc["hedge_strength"],
           "cluster": arc_id,
           "target_key": target.name,
           "hedged_value": arc["value"],
           "prior_value": arc.get("prior_value"),
           "probe_question": key_row["probe_question"],
           "arcs_sha256": arcs_digest,
           "mechanism": MECHANISM_NOTE,
           "mapper_note": ("em_pt: the ADOPTED M1 plan-stem wrapper as a "
                           "LOCAL mapper (the released unify em_cg mapping, "
                           "cache-shared; IC-2); ez_pt: frozen mapper"),
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
           "cells": {}, "diagnostics": {}}

    # em: the ADOPTED stem wrapper (LOCAL — cache-shared with the released
    # unify em_cg recs); ez: frozen mapper. Never installed globally.
    m4_eval.assert_mapper_mode_off()
    wrapper = store.make_tuple_key_mapper(PLAN_FRAME_ATTRIBUTE_PATTERNS)
    recs_em = [await unify_eval._extract_and_map("em", turns, pos, candidates,
                                                 arc_id, mapper=wrapper)
               for pos in user_positions]
    m4_eval.assert_mapper_mode_off()  # the wrapper stayed local
    recs_ez = [await unify_eval._extract_and_map("ez", turns, pos, candidates,
                                                 arc_id)
               for pos in user_positions]

    for fam, recs in (("em", recs_em), ("ez", recs_ez)):
        hedge_rec = next(r for r in recs if r["pos"] == hedge_pos)
        out["diagnostics"][fam] = {
            "hedge_turn": m4_eval._hedge_turn_diagnostics(
                hedge_rec, arc["hedge_text"], target),
            "per_turn": [m4_eval._turn_diag_row(rec, user_index[rec["pos"]],
                                                target) for rec in recs],
        }
    out["cells"]["em_pt"] = await _run_plan_typed_cell(
        recs_em, "em", arc_id, target, turns, user_index, res_pos, hedge_pos)
    out["cells"]["ez_pt"] = await _run_plan_typed_cell(
        recs_ez, "ez", arc_id, target, turns, user_index, res_pos, hedge_pos)
    m4_eval.assert_mapper_mode_off()
    return out


# ---------------------------------------------------------------------------
# run drivers (resumable; early projection checks against the 110k stop)
# ---------------------------------------------------------------------------

def _project_and_maybe_stop(done: int, remaining: int, fresh_so_far: int,
                            base_total: int) -> None:
    """Early projection (prereg stop condition): abort loudly if the
    projected combined total would pass 110k."""
    if done == 0:
        return
    per_arc = fresh_so_far / done
    projected = base_total + fresh_so_far + per_arc * remaining
    print(f"[plan_typed projection] {done} arcs done, {fresh_so_far} fresh "
          f"({per_arc:.1f}/arc); projected combined total "
          f"{projected:,.0f} / {HARD_STOP_TOTAL:,} run stop")
    if projected > HARD_STOP_TOTAL:
        raise SystemExit(
            f"plan_typed STOP: projected combined total {projected:,.0f} > "
            f"{HARD_STOP_TOTAL:,} — prereg stop condition; report and stop")


async def _run_substrate(name: str, arcs: list[dict], keys: dict[str, dict],
                         digest: str, cases_dir: Path, run_arc, sel_ids,
                         call_cap: int | None) -> dict:
    assert_pins()
    snapshot_budget_before()
    base_total = total_spent()
    by_id = {a["arc_id"]: a for a in arcs}
    started = datetime.now(timezone.utc).isoformat()
    guard = pt_guard(local_cap=call_cap)
    done = skipped = 0
    failed: list[dict] = []
    n_todo = sum(1 for i in sel_ids if not (cases_dir / f"{i}.json").exists())
    guard.install()
    try:
        for arc_id in sel_ids:
            arc = by_id.get(arc_id)
            assert arc is not None, f"unknown/excluded arc id {arc_id!r}"
            path = cases_dir / f"{arc_id}.json"
            if path.exists():
                skipped += 1
                continue
            try:
                record = await run_arc(arc, keys[arc_id], digest)
            except (SystemExit, KeyboardInterrupt):
                raise
            except Exception as exc:  # noqa: BLE001 — recorded, run continues
                low = str(exc).lower()
                if "budget" in low or "envelope" in low or "exhausted" in low:
                    raise
                failed.append({"id": arc_id, "error": repr(exc)})
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, ensure_ascii=False, indent=1))
            done += 1
            if done in (5, 15) or done % 30 == 0:
                _project_and_maybe_stop(done, n_todo - done, guard.fresh,
                                        base_total)
    finally:
        guard.uninstall()
    summary = {"phase": f"plan_typed_run_{name}", "model": llm.MODEL,
               "started": started,
               "ended": datetime.now(timezone.utc).isoformat(),
               "selected": len(sel_ids), "n_new": done, "skipped": skipped,
               "failed": failed, "fresh_calls": guard.fresh,
               "advisory": advisory_status()}
    log_phase(summary)
    return summary


async def run_m4(ids: list[str] | None, call_cap: int | None) -> dict:
    arcs = m4_eval.load_arcs()
    keys = m4_eval.load_keys()
    digest = m4_eval.arcs_sha256()
    sel = sorted(ids if ids is not None else [a["arc_id"] for a in arcs])
    return await _run_substrate("m4", arcs, keys, digest, M4_CASES,
                                run_arc_m4, sel, call_cap)


async def run_unify(ids: list[str] | None, shard: str | None,
                    call_cap: int | None) -> dict:
    arcs = unify_eval.load_arcs()  # 179 usable — exclusions honored inside
    keys = unify_eval.load_keys()
    digest = unify_eval.arcs_sha256()
    sel = _shard(sorted(ids if ids is not None
                        else [a["arc_id"] for a in arcs]), shard)
    return await _run_substrate("unify", arcs, keys, digest, UNIFY_CASES,
                                run_arc_unify, sel, call_cap)


# ---------------------------------------------------------------------------
# score — per-arc rows (pure functions of the case records; never rerun)
# ---------------------------------------------------------------------------

async def score_case_m4(case: dict, scorer) -> list[dict]:
    """Per-arc rows, one per pt arm — the m4_eval.score_case class logic on
    the READ-RULE value, with the masked-plan counts reported beside
    (returned-as-current = contamination; masked = not, counted)."""
    attr = case["target_key"].partition("/")[2]
    hv = case["hedged_value"]
    cls = case["resolution_type"]
    prior = case.get("established_prior_value")
    hedge_u = case["geometry"]["hedge"]["user_idx"]
    res_u = case["geometry"]["resolution"]["user_idx"]
    last_u = case["geometry"]["n_user_turns"] - 1
    rows = []
    for arm in [a for a in ARMS if a in case["cells"]]:
        c = case["cells"][arm]
        final_eq, _m = await scorer.equivalent(attr, c["final_value"], hv)
        raw_final_eq, _m = await scorer.equivalent(
            attr, c["final_value_raw"], hv)
        masked_final = (bool(c["final_masked_plan"]) and bool(raw_final_eq)
                        and not final_eq)
        pre_eq, _m = await scorer.equivalent(
            attr, c["pre_resolution_value"], hv)
        raw_pre_eq, _m = await scorer.equivalent(
            attr, c["pre_resolution_value_raw"], hv)
        masked_pre = (bool(c["pre_resolution_masked_plan"])
                      and bool(raw_pre_eq) and not pre_eq)
        boundary_eq = final_eq if cls == "open" else pre_eq
        boundary_masked = masked_final if cls == "open" else masked_pre

        # IC-5 while-pending window over the recorded timeline (D1
        # semantics; open arcs run through the last user turn).
        hi = (last_u + 1) if cls == "open" else res_u
        wp_returned = wp_masked = False
        for entry in c.get("target_value_after_turn", ()):
            u = entry["user_idx"]
            if not (hedge_u < u < hi):
                continue
            eq, _mm = await scorer.equivalent(attr, entry["value"], hv)
            if eq:
                wp_returned = True
            if entry.get("masked_plan"):
                req, _mm = await scorer.equivalent(attr, entry["raw_value"], hv)
                if req:
                    wp_masked = True

        probe_value = (c["final_value"]
                       if m4_eval._answered(c["final_value"]) else None)
        if cls == "resolve_true":
            probe_correct = bool(final_eq)
            probe_expected = hv
        elif prior is not None:
            eq, _mm = await scorer.equivalent(attr, probe_value, prior)
            probe_correct = bool(eq)
            probe_expected = prior
        else:
            probe_correct = probe_value is None  # abstention-by-absence
            probe_expected = "(abstention-by-absence)"

        rows.append({
            "arc_id": case["arc_id"], "substrate": "m4",
            "resolution_type": cls,
            "hedge_strength": case["hedge_strength"],
            "cell": arm, "cluster": case["cluster"],
            "hedged_value": hv,
            "final_value": c["final_value"],
            "final_value_raw": c["final_value_raw"],
            "final_masked_plan": bool(c["final_masked_plan"]),
            "pre_resolution_value": c["pre_resolution_value"],
            "final_eq_hedged": bool(final_eq),
            # returned-as-current = contamination; masked = counted, NOT
            # contamination (prereg endpoint semantics)
            "contamination": (bool(final_eq)
                              if cls in ("abandoned", "open") else None),
            "contamination_masked_plan": (masked_final
                                          if cls in ("abandoned", "open")
                                          else None),
            "premature_commit": bool(boundary_eq),
            "premature_masked_plan": bool(boundary_masked),
            "pre_resolution_returned": bool(pre_eq),
            "pre_resolution_masked_plan_eq": bool(masked_pre),
            "while_pending_returned": wp_returned,
            "while_pending_masked_plan": wp_masked,
            "end_state_eq_new_value": (bool(final_eq)
                                       if cls == "resolve_true" else None),
            "end_state_masked_plan": (masked_final
                                      if cls == "resolve_true" else None),
            "probe_value": probe_value,
            "probe_expected": probe_expected,
            "probe_answered": probe_value is not None,
            "probe_correct": bool(probe_correct),
        })
    return rows


async def score_case_unify(case: dict, scorer) -> list[dict]:
    """Per-arc rows, one per pt arm — the unify_eval.score_case endpoint
    logic on the READ-RULE value + masked-plan counts."""
    attr = case["target_key"].partition("/")[2]
    hv = case["hedged_value"]
    outcome = case["outcome"]
    prior = case.get("prior_value")
    hedge_u = case["geometry"]["hedge"]["user_idx"]
    res_u = case["geometry"]["resolution"]["user_idx"]
    last_u = case["geometry"]["n_user_turns"] - 1
    rows = []
    for arm in [a for a in ARMS if a in case["cells"]]:
        c = case["cells"][arm]
        final_eq, _m = await scorer.equivalent(attr, c["final_value"], hv)
        raw_final_eq, _m = await scorer.equivalent(
            attr, c["final_value_raw"], hv)
        masked_final = (bool(c["final_masked_plan"]) and bool(raw_final_eq)
                        and not final_eq)

        async def window(lo_u: int, hi_u: int, field: str,
                         masked_only: bool) -> bool:
            for entry in c["target_value_after_turn"]:
                u = entry["user_idx"]
                if not (lo_u < u < hi_u):
                    continue
                if masked_only and not entry.get("masked_plan"):
                    continue
                eq, _mm = await scorer.equivalent(attr, entry[field], hv)
                if eq:
                    return True
            return False

        while_pending = at_end = None
        while_pending_masked = at_end_masked = None
        commit_correct = None
        g5_masked = None
        if outcome == "still_open":
            while_pending = await window(hedge_u, last_u + 1, "value", False)
            while_pending_masked = await window(hedge_u, last_u + 1,
                                                "raw_value", True)
            at_end = bool(final_eq)
            at_end_masked = masked_final
        elif outcome == "already_completed":
            commit_correct = bool(final_eq)  # a commit is CORRECT (g5)
            g5_masked = masked_final         # completed value plan-masked

        promote_analog_note = "no staging/promotion in this baseline"
        end_state = premature = None
        end_state_masked = premature_masked = None
        if outcome == "came_true":
            end_state = bool(final_eq)
            end_state_masked = masked_final  # the no-route-1 signature
            premature = await window(hedge_u, res_u, "value", False)
            premature_masked = await window(hedge_u, res_u, "raw_value", True)

        probe_value = (c["final_value"]
                       if m4_eval._answered(c["final_value"]) else None)
        probe_expected = prior if outcome == "still_open" else hv
        eq, _mm = await scorer.equivalent(attr, probe_value, probe_expected)
        probe_correct = bool(eq)

        rows.append({
            "arc_id": case["arc_id"], "substrate": "unify",
            "cluster": case["cluster"], "cell": arm,
            "unify_ladder": case["unify_ladder"],
            "unify_level": case["unify_level"],
            "outcome": outcome,
            "hedged_value": hv,
            "final_value": c["final_value"],
            "final_value_raw": c["final_value_raw"],
            "final_masked_plan": bool(c["final_masked_plan"]),
            "final_eq_hedged": bool(final_eq),
            "while_pending_contamination": while_pending,
            "while_pending_masked_plan": while_pending_masked,
            "at_end_contamination": at_end,
            "at_end_masked_plan": at_end_masked,
            "g5_commit_correct": commit_correct,
            "g5_completed_value_masked_plan": g5_masked,
            "end_state_correct": end_state,
            "end_state_masked_plan": end_state_masked,
            "premature_commit_before_resolution": premature,
            "premature_masked_plan": premature_masked,
            "promotion_note": promote_analog_note,
            "plan_confirms_target": c.get("plan_confirms_target", 0),
            "probe_value": probe_value,
            "probe_expected": probe_expected,
            "probe_answered": probe_value is not None,
            "probe_correct": probe_correct,
        })
    return rows


def _load_cases(root: Path) -> list[dict]:
    if not root.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(root.glob("*.json"))]


# --- released-artifact quoting (READ-ONLY) ----------------------------------

def _q(pt_dict: dict | None) -> dict | None:
    """Quote one released point verbatim (x/n/rate|point + CI fields)."""
    return pt_dict


def _pool_released(points: list[dict]) -> dict:
    """Arithmetic pooling of quoted released x/n rows (sum x, sum n) — the
    m2 adoption-arithmetic convention; quoting, never rerunning."""
    x = sum(p["x"] for p in points if p and p.get("n"))
    n = sum(p["n"] for p in points if p and p.get("n"))
    return {"x": x, "n": n, "rate": (x / n) if n else None}


def _pt_pt(rows: list[dict], flag: str) -> dict:
    """x/n + Wilson over per-conversation rows (the unify convention)."""
    sel = [r for r in rows if r.get(flag) is not None]
    x = sum(1 for r in sel if r[flag])
    n = len(sel)
    if n == 0:
        return {"x": 0, "n": 0, "rate": None,
                "wilson_lo": None, "wilson_hi": None}
    lo, hi = stats.wilson_ci(x, n)
    return {"x": x, "n": n, "rate": x / n, "wilson_lo": lo, "wilson_hi": hi}


def _usel(rows, cell, level=None, outcome=None):
    return [r for r in rows if r["cell"] == cell
            and (level is None or r["unify_level"] == level)
            and (outcome is None or r["outcome"] == outcome)]


def _msel(rows, cell, cls=None):
    return [r for r in rows if r["cell"] == cell
            and (cls is None or r["resolution_type"] == cls)]


def _rate(p: dict | None):
    if not p:
        return None
    return p.get("rate", p.get("point"))


# ---------------------------------------------------------------------------
# score — endpoints + comparison + the 3-row scorecard
# ---------------------------------------------------------------------------

async def compute_scores() -> dict:
    m4_cases = _load_cases(M4_CASES)
    unify_cases = _load_cases(UNIFY_CASES)
    assert m4_cases, f"no M4 case records under {M4_CASES}"
    assert unify_cases, f"no unify case records under {UNIFY_CASES}"

    # substrate integrity: same frozen arcs as the released comparison runs
    released_m4 = json.loads(RELEASED_M4.read_text())
    released_unify = json.loads(RELEASED_UNIFY.read_text())
    m4_digests = {c["arcs_sha256"] for c in m4_cases}
    unify_digests = {c["arcs_sha256"] for c in unify_cases}
    assert m4_digests == {released_m4["arcs_sha256"]}, (
        f"M4 case digests {m4_digests} != released "
        f"{released_m4['arcs_sha256']}")
    assert unify_digests == {released_unify["arcs_sha256"]}, (
        f"unify case digests {unify_digests} != released "
        f"{released_unify['arcs_sha256']}")
    excluded = unify_eval.load_exclusions()
    assert not ({c["arc_id"] for c in unify_cases} & excluded), (
        "an excluded arc has a case record")

    scorer = rule5.EquivalenceScorer(cap=10_000)
    m4_rows: list[dict] = []
    for case in m4_cases:
        m4_rows.extend(await score_case_m4(case, scorer))
    unify_rows: list[dict] = []
    for case in unify_cases:
        unify_rows.extend(await score_case_unify(case, scorer))

    # ---- M4 endpoints (cluster bootstrap — the released M4 convention) ----
    m4_ep: dict = {}
    m4_ep["contamination_final_store"] = {
        cls: {arm: {"returned_as_current": m4_eval._point(
                        _msel(m4_rows, arm, cls), "contamination"),
                    "masked_plan": m4_eval._point(
                        _msel(m4_rows, arm, cls), "contamination_masked_plan")}
              for arm in ARMS}
        for cls in ("abandoned", "open")}
    m4_ep["premature_commit_resolve_true"] = {
        arm: {"returned_as_current": m4_eval._point(
                  _msel(m4_rows, arm, "resolve_true"), "premature_commit"),
              "masked_plan": m4_eval._point(
                  _msel(m4_rows, arm, "resolve_true"), "premature_masked_plan")}
        for arm in ARMS}
    m4_ep["pre_resolution_boundary_contamination_supplementary"] = {
        cls: {arm: {"returned_as_current": m4_eval._point(
                        _msel(m4_rows, arm, cls), "pre_resolution_returned"),
                    "masked_plan": m4_eval._point(
                        _msel(m4_rows, arm, cls),
                        "pre_resolution_masked_plan_eq")}
              for arm in ARMS}
        for cls in ("abandoned", "open")}
    m4_ep["while_pending_supplementary_IC5"] = {
        cls: {arm: {"returned_as_current": m4_eval._point(
                        _msel(m4_rows, arm, cls), "while_pending_returned"),
                    "masked_plan": m4_eval._point(
                        _msel(m4_rows, arm, cls), "while_pending_masked_plan")}
              for arm in ARMS}
        for cls in M4_CLASSES}
    m4_ep["end_state_resolve_true"] = {
        arm: m4_eval._point(_msel(m4_rows, arm, "resolve_true"),
                            "end_state_eq_new_value") for arm in ARMS}
    m4_ep["end_state_masked_plan_resolve_true"] = {
        arm: m4_eval._point(_msel(m4_rows, arm, "resolve_true"),
                            "end_state_masked_plan") for arm in ARMS}
    m4_probes = {cls: {arm: m4_eval._point(_msel(m4_rows, arm, cls),
                                           "probe_correct") for arm in ARMS}
                 for cls in M4_CLASSES}
    m4_probes["pooled"] = {arm: m4_eval._point(_msel(m4_rows, arm),
                                               "probe_correct")
                           for arm in ARMS}
    m4_ep["task_probes"] = m4_probes

    # ---- unify endpoints (Wilson — the released unify convention) ---------
    u_ep: dict = {}
    u_ep["contamination_by_level"] = {
        lv: {arm: {
            "while_pending": _pt_pt(_usel(unify_rows, arm, lv),
                                    "while_pending_contamination"),
            "while_pending_masked_plan": _pt_pt(
                _usel(unify_rows, arm, lv), "while_pending_masked_plan"),
            "at_end": _pt_pt(_usel(unify_rows, arm, lv),
                             "at_end_contamination"),
            "at_end_masked_plan": _pt_pt(_usel(unify_rows, arm, lv),
                                         "at_end_masked_plan")}
            for arm in ARMS}
        for lv in STILL_OPEN_LEVELS}
    u_ep["g5_commit_correct_control"] = {
        arm: {"commit_correct_at_end": _pt_pt(
                  _usel(unify_rows, arm, "g5_completed_control"),
                  "g5_commit_correct"),
              "completed_value_masked_plan": _pt_pt(
                  _usel(unify_rows, arm, "g5_completed_control"),
                  "g5_completed_value_masked_plan")}
        for arm in ARMS}
    u_ep["end_state_by_distance"] = {
        lv: {arm: {
            "end_state_correct_ITT": _pt_pt(_usel(unify_rows, arm, lv),
                                            "end_state_correct"),
            "end_state_masked_plan": _pt_pt(_usel(unify_rows, arm, lv),
                                            "end_state_masked_plan"),
            "premature_commit_before_resolution_supplementary": _pt_pt(
                _usel(unify_rows, arm, lv),
                "premature_commit_before_resolution"),
            "premature_masked_plan": _pt_pt(_usel(unify_rows, arm, lv),
                                            "premature_masked_plan")}
            for arm in ARMS}
        for lv in P_LEVELS}
    u_ep["end_state_came_true_pooled"] = {
        arm: _pt_pt(_usel(unify_rows, arm, outcome="came_true"),
                    "end_state_correct") for arm in ARMS}
    u_probes = {"by_outcome": {}, "by_level": {}, "pooled": {}}
    for oc in unify_eval.OUTCOME_CLASSES:
        u_probes["by_outcome"][oc] = {
            arm: _pt_pt(_usel(unify_rows, arm, outcome=oc), "probe_correct")
            for arm in ARMS}
    for lv in G_LEVELS + P_LEVELS:
        u_probes["by_level"][lv] = {
            arm: _pt_pt(_usel(unify_rows, arm, lv), "probe_correct")
            for arm in ARMS}
    u_probes["pooled"] = {arm: _pt_pt(_usel(unify_rows, arm), "probe_correct")
                          for arm in ARMS}
    u_ep["task_probes_ITT"] = u_probes

    # ---- comparison rows (QUOTED from the released artifacts) -------------
    rm4_ep = released_m4["endpoints"]
    ru_ep = released_unify["endpoints"]
    released_cells = ("em_cw", "em_cg", "ez_cw", "ez_cg", "em_cg_fixed")
    comparison = {
        "released_sources": {"m4": str(RELEASED_M4),
                             "unify": str(RELEASED_UNIFY)},
        "note": ("comparison rows quoted verbatim from the released "
                 "artifacts — nothing released was rerun. Unify em_cg IS "
                 "the adopted repaired configuration (its M4 planted analog "
                 "is em_cg_fixed)."),
        "m4": {
            "contamination_final_store": {
                cls: {cell: _q(rm4_ep["contamination_final_store"][cls]
                               .get(cell)) for cell in released_cells}
                for cls in ("abandoned", "open")},
            "premature_commit_resolve_true": {
                cell: _q(rm4_ep["premature_commit_resolve_true"].get(cell))
                for cell in released_cells},
            "pre_resolution_boundary_contamination_supplementary": {
                cls: {cell: _q(
                    rm4_ep["pre_resolution_boundary_contamination_supplementary"]
                    [cls].get(cell)) for cell in released_cells}
                for cls in ("abandoned", "open")},
            "end_state_resolve_true": {
                cell: _q(rm4_ep["promotion_resolve_true"][cell]
                         ["end_state_eq_new_value_ITT"])
                for cell in released_cells},
            "task_probes": {
                cls: {cell: _q(rm4_ep["task_probes"][cls].get(cell))
                      for cell in released_cells}
                for cls in M4_CLASSES + ("pooled",)},
        },
        "unify": {
            "contamination_by_level": {
                lv: {cell: {
                    "while_pending": _q(ru_ep["contamination_by_level"][lv]
                                        [cell]["while_pending"]),
                    "at_end": _q(ru_ep["contamination_by_level"][lv][cell]
                                 ["at_end"])}
                    for cell in unify_eval.CELLS}
                for lv in STILL_OPEN_LEVELS},
            "g5_commit_correct_control": {
                cell: _q(ru_ep["g5_commit_correct_control"][cell]
                         ["commit_correct_at_end"])
                for cell in unify_eval.CELLS},
            "end_state_by_distance": {
                lv: {cell: _q(ru_ep["promotion_by_distance"][lv][cell]
                              ["end_state_correct_ITT"])
                     for cell in unify_eval.CELLS}
                for lv in P_LEVELS},
            "promotion_window_by_distance": {
                lv: {cell: _q(ru_ep["promotion_by_distance"][lv][cell]
                              ["promote_fire_within_window_ITT"])
                     for cell in unify_eval.GATED_CELLS}
                for lv in P_LEVELS},
            "task_probes_by_outcome": {
                oc: {cell: _q(ru_ep["task_probes_ITT"]["by_outcome"][oc]
                              [cell]) for cell in unify_eval.CELLS}
                for oc in unify_eval.OUTCOME_CLASSES},
            "task_probes_pooled": {
                cell: _q(ru_ep["task_probes_ITT"]["pooled"][cell])
                for cell in unify_eval.CELLS},
        },
    }

    # the KEY TABLE: came_true end-state by wording distance, pt vs released
    key_table = {}
    for lv in P_LEVELS:
        key_table[lv] = {
            "em_pt": u_ep["end_state_by_distance"][lv]["em_pt"]
                     ["end_state_correct_ITT"],
            "ez_pt": u_ep["end_state_by_distance"][lv]["ez_pt"]
                     ["end_state_correct_ITT"],
            "released": {cell: comparison["unify"]["end_state_by_distance"]
                         [lv][cell] for cell in unify_eval.CELLS},
            "released_promotion_window": comparison["unify"]
                ["promotion_window_by_distance"][lv],
        }

    # ---- the 3-row directional-expectation scorecard (IC-6) ---------------
    def upool(arm, levels, flag):
        sel = [r for lv in levels for r in _usel(unify_rows, arm, lv)]
        return _pt_pt(sel, flag)

    def released_upool(cell, levels, endpoint):
        return _pool_released([ru_ep["contamination_by_level"][lv][cell]
                               [endpoint] for lv in levels])

    def released_m4_pool(cell):
        return _pool_released([rm4_ep["contamination_final_store"][cls][cell]
                               for cls in ("abandoned", "open")])

    def m4pool(arm, flag):
        sel = [r for cls in ("abandoned", "open")
               for r in _msel(m4_rows, arm, cls)]
        return m4_eval._point(sel, flag)

    scorecard = []
    # Row 1 — contamination ~= gated levels (both classifier-limited)
    r1_obs, r1_checks = {}, []
    for arm in ARMS:
        cg = GATED_OF[arm]
        u_pt = upool(arm, STILL_OPEN_LEVELS, "at_end_contamination")
        u_cg = released_upool(cg, STILL_OPEN_LEVELS, "at_end")
        m_pt = m4pool(arm, "contamination")
        m_cg = released_m4_pool(cg)
        r1_obs[arm] = {
            "unify_at_end_pooled_g1_g4": {"pt": u_pt, "released_gated": u_cg},
            "unify_while_pending_pooled_g1_g4_descriptive": {
                "pt": upool(arm, STILL_OPEN_LEVELS,
                            "while_pending_contamination"),
                "released_gated": released_upool(
                    cg, STILL_OPEN_LEVELS, "while_pending")},
            "m4_final_store_pooled_abandoned_open": {
                "pt": m_pt, "released_gated": m_cg},
        }
        for a, b in ((u_pt, u_cg), (m_pt, m_cg)):
            ra, rb = _rate(a), _rate(b)
            r1_checks.append(ra is not None and rb is not None
                             and abs(ra - rb) <= APPROX)
    scorecard.append({
        "row": 1,
        "expectation": ("contamination ~= gated levels (both are "
                        "classifier-limited)"),
        "operationalization": (
            f"holds iff |pt - released gated| <= {APPROX} on pooled at-end "
            "contamination, per family, on BOTH substrates (unify pooled "
            "g1-g4 at-end; M4 pooled abandoned+open final store); "
            "while-pending reported descriptively"),
        "observed": r1_obs,
        "holds": all(r1_checks),
    })
    # Row 2 — came-true correctness BELOW gated at oblique/implied wordings
    r2_obs, r2_checks = {}, []
    oblique = ("p3_oblique", "p4_situational")
    for arm in ARMS:
        cg = GATED_OF[arm]
        pt_p34 = upool(arm, oblique, "end_state_correct")
        cg_p34 = _pool_released([ru_ep["promotion_by_distance"][lv][cg]
                                 ["end_state_correct_ITT"] for lv in oblique])
        r2_obs[arm] = {"pt_p3_p4_pooled_end_state": pt_p34,
                       "released_gated_p3_p4_pooled_end_state": cg_p34,
                       "per_level": {lv: {
                           "pt": u_ep["end_state_by_distance"][lv][arm]
                                 ["end_state_correct_ITT"],
                           "released_gated": comparison["unify"]
                               ["end_state_by_distance"][lv][cg]}
                           for lv in oblique}}
        ra, rb = _rate(pt_p34), _rate(cg_p34)
        r2_checks.append(ra is not None and rb is not None and ra < rb)
    scorecard.append({
        "row": 2,
        "expectation": ("came-true correctness BELOW gated at oblique/"
                        "implied wordings (no route 1)"),
        "operationalization": (
            "holds iff pt came_true end-state < the released gated cell on "
            "pooled p3+p4, per family (points)"),
        "observed": r2_obs,
        "holds": all(r2_checks),
    })
    # Row 3 — no over-holding by construction
    violations = []
    for case in m4_cases + unify_cases:
        for arm in [a for a in ARMS if a in case["cells"]]:
            c = case["cells"][arm]
            if (c["final_value"] != c["final_value_raw"]
                    and not c["final_masked_plan"]):
                violations.append({"arc_id": case["arc_id"], "cell": arm})
    r3_obs = {"masked_nonplan_violations": violations,
              "n_violations": len(violations), "g5": {}}
    r3_checks = [not violations]
    for arm in ARMS:
        cg = GATED_OF[arm]
        pt_g5 = u_ep["g5_commit_correct_control"][arm]["commit_correct_at_end"]
        cg_g5 = ru_ep["g5_commit_correct_control"][cg]["commit_correct_at_end"]
        r3_obs["g5"][arm] = {
            "pt_commit_correct": pt_g5, "released_gated": _q(cg_g5),
            "pt_completed_value_masked_plan":
                u_ep["g5_commit_correct_control"][arm]
                ["completed_value_masked_plan"]}
        ra, rb = _rate(pt_g5), _rate(cg_g5)
        r3_checks.append(ra is not None and rb is not None
                         and ra >= rb - APPROX)
    scorecard.append({
        "row": 3,
        "expectation": "no over-holding by construction",
        "operationalization": (
            "holds iff (a) ZERO arcs across both substrates where the read "
            "rule masked a NON-plan current value (structural — counted "
            "from the case records) AND (b) unify g5 commit-correct >= the "
            f"released gated cell - {APPROX}, per family (a completed "
            "control mistyped as plan WOULD be over-holding — evaluated, "
            "not assumed)"),
        "observed": r3_obs,
        "holds": all(r3_checks),
    })

    phases = ([json.loads(l) for l in RUN_LOG.open()]
              if RUN_LOG.exists() else [])
    m4_class_counts = {cls: sum(c["resolution_type"] == cls
                                for c in m4_cases) for cls in M4_CLASSES}
    level_counts = {lv: sum(c["unify_level"] == lv for c in unify_cases)
                    for lv in G_LEVELS + P_LEVELS}
    return {
        "id": "PLAN-TYPED-BASELINE-RESULTS",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prereg": ("protocol/CHANGELOG.md 2026-08-12 'PREREGISTRATION: "
                   "plan-typed-commit baseline (reviewer-motivated "
                   "ablation)'"),
        "mechanism": MECHANISM_NOTE,
        "system_model": llm.MODEL,
        "judge_model": llm.JUDGE_MODEL,
        "arms": list(ARMS),
        "m4": {"n_arcs_scored": len(m4_cases),
               "partial": len(m4_cases) < 60,
               "class_counts": m4_class_counts,
               "arcs_sha256": next(iter(m4_digests)),
               "ci_note": ("cluster-bootstrap CIs (cluster = arc, "
                           "SEED_M4) — the released M4 convention"),
               "endpoints": m4_ep,
               "per_arc_rows": m4_rows},
        "unify": {"n_arcs_scored": len(unify_cases),
                  "partial": len(unify_cases) < 179,
                  "level_counts": level_counts,
                  "arcs_sha256": next(iter(unify_digests)),
                  "excluded": sorted(excluded),
                  "ci_note": "Wilson CIs — the released unify convention",
                  "endpoints": u_ep,
                  "per_arc_rows": unify_rows},
        "wording_distance_key_table": key_table,
        "released_comparison": comparison,
        "directional_scorecard": scorecard,
        "run_phases": phases,
        "scoring_judge_calls": {"total": scorer.judge_calls,
                                "fresh": scorer.fresh_judge_calls},
        "budget_before": (json.loads(BUDGET_BEFORE.read_text())
                          if BUDGET_BEFORE.exists() else None),
        "budget_after": mapper_fix.ensure_budget(BUDGET_PATH),
        "advisory_budget": advisory_status(),
    }


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _fmt_w(p: dict | None) -> str:
    if not p or p.get("rate") is None:
        return "—"
    return (f"{p['x']}/{p['n']} = {p['rate']:.3f} "
            f"[{p['wilson_lo']:.3f}, {p['wilson_hi']:.3f}]")


def _fmt_b(p: dict | None) -> str:
    if not p or p.get("point") is None:
        return "—"
    return (f"{p['x']}/{p['n']} = {p['point']:.3f} "
            f"[{p['ci'][0]:.3f}, {p['ci'][1]:.3f}]")


def _fmt_q(p: dict | None) -> str:
    if not p:
        return "—"
    r = p.get("rate", p.get("point"))
    if r is None:
        return "—"
    return f"{p['x']}/{p['n']} = {r:.3f}"


def render_report(res: dict) -> str:
    m4r, unir = res["m4"], res["unify"]
    m4_ep, u_ep = m4r["endpoints"], unir["endpoints"]
    cmp_ = res["released_comparison"]
    lines = [
        "# Plan-typed-commit baseline (em_pt / ez_pt) — preregistered "
        "reviewer-motivated ablation",
        "",
        f"Preregistered 2026-08-12 (protocol/CHANGELOG.md). Generated "
        f"{res['generated_at']}. Systems under pinned "
        f"`{res['system_model']}`; judge = frozen `{res['judge_model']}`.",
        "",
        f"Mechanism: {res['mechanism']}",
        "",
        f"Substrates: internal M4 {m4r['n_arcs_scored']}/60 arcs"
        + (" (PARTIAL)" if m4r["partial"] else "")
        + f" (classes {m4r['class_counts']}); unified ladders "
        f"{unir['n_arcs_scored']}/179 arcs"
        + (" (PARTIAL)" if unir["partial"] else "")
        + f" (excluded: {', '.join(unir['excluded'])}). Comparison rows are "
        "QUOTED from the released artifacts (never rerun): "
        f"`{cmp_['released_sources']['m4']}`, "
        f"`{cmp_['released_sources']['unify']}`. Unify em_cg is the adopted "
        "repaired configuration; em_pt mirrors each substrate's released "
        "gated mapping (IC-2). No adoption decision — a reported baseline.",
        "",
        "Contamination semantics (prereg): a plan-typed value RETURNED AS "
        "CURRENT by the arm's read rule counts as contamination; a "
        "plan-typed value present but MASKED by the read rule does not — "
        "BOTH counts are reported in every table.",
        "",
        "## THE KEY TABLE — came_true end-state correctness by "
        "confirmation-wording distance (unified cw ladder; Wilson CIs)",
        "",
        "The discriminating comparison vs the gate's route-1 promotion: "
        "this baseline resolves via route 2 only (the confirming turn's "
        "own commit).",
        "",
        "| distance | em_pt | ez_pt | released em_cg | released ez_cg | "
        "released em_cw | released ez_cw |",
        "|---|---|---|---|---|---|---|"]
    for lv in P_LEVELS:
        k = res["wording_distance_key_table"][lv]
        rel = k["released"]
        lines.append(
            f"| {lv} | {_fmt_w(k['em_pt'])} | {_fmt_w(k['ez_pt'])} "
            f"| {_fmt_q(rel['em_cg'])} | {_fmt_q(rel['ez_cg'])} "
            f"| {_fmt_q(rel['em_cw'])} | {_fmt_q(rel['ez_cw'])} |")
    lines += [
        "",
        "Pooled came_true end-state: "
        + "; ".join(f"{arm} {_fmt_w(u_ep['end_state_came_true_pooled'][arm])}"
                    for arm in ARMS)
        + ". Released gated window-promotion (context, quoted): "
        + "; ".join(
            f"{lv}: em_cg {_fmt_q(cmp_['unify']['promotion_window_by_distance'][lv]['em_cg'])}, "
            f"ez_cg {_fmt_q(cmp_['unify']['promotion_window_by_distance'][lv]['ez_cg'])}"
            for lv in P_LEVELS) + ".",
        "",
        "Masked-at-end on came_true (the no-route-1 failure signature: the "
        "confirmed value sits plan-typed and masked):",
        "",
        "| distance | em_pt masked | ez_pt masked |", "|---|---|---|"]
    for lv in P_LEVELS:
        e = u_ep["end_state_by_distance"][lv]
        lines.append(f"| {lv} | {_fmt_w(e['em_pt']['end_state_masked_plan'])} "
                     f"| {_fmt_w(e['ez_pt']['end_state_masked_plan'])} |")
    lines += [
        "", "## Unified ladders — contamination by hedge-strength level "
        "(still_open g1-g4; Wilson CIs)", "",
        "returned = read-rule current (contamination); masked = plan-typed "
        "present but masked (NOT contamination; reported).", "",
        "### At-end", "",
        "| level | em_pt returned | em_pt masked | ez_pt returned | "
        "ez_pt masked | released em_cg | released ez_cg | released em_cw | "
        "released ez_cw |",
        "|---|---|---|---|---|---|---|---|---|"]
    for lv in STILL_OPEN_LEVELS:
        e = u_ep["contamination_by_level"][lv]
        q = cmp_["unify"]["contamination_by_level"][lv]
        lines.append(
            f"| {lv} | {_fmt_w(e['em_pt']['at_end'])} "
            f"| {_fmt_w(e['em_pt']['at_end_masked_plan'])} "
            f"| {_fmt_w(e['ez_pt']['at_end'])} "
            f"| {_fmt_w(e['ez_pt']['at_end_masked_plan'])} "
            f"| {_fmt_q(q['em_cg']['at_end'])} "
            f"| {_fmt_q(q['ez_cg']['at_end'])} "
            f"| {_fmt_q(q['em_cw']['at_end'])} "
            f"| {_fmt_q(q['ez_cw']['at_end'])} |")
    lines += ["", "### While-pending", "",
              "| level | em_pt returned | em_pt masked | ez_pt returned | "
              "ez_pt masked | released em_cg | released ez_cg |",
              "|---|---|---|---|---|---|---|"]
    for lv in STILL_OPEN_LEVELS:
        e = u_ep["contamination_by_level"][lv]
        q = cmp_["unify"]["contamination_by_level"][lv]
        lines.append(
            f"| {lv} | {_fmt_w(e['em_pt']['while_pending'])} "
            f"| {_fmt_w(e['em_pt']['while_pending_masked_plan'])} "
            f"| {_fmt_w(e['ez_pt']['while_pending'])} "
            f"| {_fmt_w(e['ez_pt']['while_pending_masked_plan'])} "
            f"| {_fmt_q(q['em_cg']['while_pending'])} "
            f"| {_fmt_q(q['ez_cg']['while_pending'])} |")
    g5 = u_ep["g5_commit_correct_control"]
    g5q = cmp_["unify"]["g5_commit_correct_control"]
    lines += [
        "", "## Controls — unify g5 commit-correct (a commit is CORRECT)",
        "",
        "| arm | commit-correct | completed value plan-masked | "
        "released gated | released ungated |", "|---|---|---|---|---|"]
    for arm in ARMS:
        lines.append(
            f"| {arm} | {_fmt_w(g5[arm]['commit_correct_at_end'])} "
            f"| {_fmt_w(g5[arm]['completed_value_masked_plan'])} "
            f"| {GATED_OF[arm]} {_fmt_q(g5q[GATED_OF[arm]])} "
            f"| {UNGATED_OF[arm]} {_fmt_q(g5q[UNGATED_OF[arm]])} |")
    lines += [
        "", "## Internal M4 — contamination + boundaries (cluster-bootstrap "
        "CIs, cluster = arc)", "",
        "| endpoint | em_pt returned | em_pt masked | ez_pt returned | "
        "ez_pt masked | released em_cg | released ez_cg | released em_cw | "
        "released ez_cw |",
        "|---|---|---|---|---|---|---|---|---|"]
    qm = cmp_["m4"]
    for cls in ("abandoned", "open"):
        e = m4_ep["contamination_final_store"][cls]
        lines.append(
            f"| contamination at end ({cls}) "
            f"| {_fmt_b(e['em_pt']['returned_as_current'])} "
            f"| {_fmt_b(e['em_pt']['masked_plan'])} "
            f"| {_fmt_b(e['ez_pt']['returned_as_current'])} "
            f"| {_fmt_b(e['ez_pt']['masked_plan'])} "
            f"| {_fmt_q(qm['contamination_final_store'][cls]['em_cg'])} "
            f"| {_fmt_q(qm['contamination_final_store'][cls]['ez_cg'])} "
            f"| {_fmt_q(qm['contamination_final_store'][cls]['em_cw'])} "
            f"| {_fmt_q(qm['contamination_final_store'][cls]['ez_cw'])} |")
    e = m4_ep["premature_commit_resolve_true"]
    lines.append(
        f"| premature commit (resolve_true, boundary) "
        f"| {_fmt_b(e['em_pt']['returned_as_current'])} "
        f"| {_fmt_b(e['em_pt']['masked_plan'])} "
        f"| {_fmt_b(e['ez_pt']['returned_as_current'])} "
        f"| {_fmt_b(e['ez_pt']['masked_plan'])} "
        f"| {_fmt_q(qm['premature_commit_resolve_true']['em_cg'])} "
        f"| {_fmt_q(qm['premature_commit_resolve_true']['ez_cg'])} "
        f"| {_fmt_q(qm['premature_commit_resolve_true']['em_cw'])} "
        f"| {_fmt_q(qm['premature_commit_resolve_true']['ez_cw'])} |")
    for cls in ("abandoned", "open"):
        e = m4_ep["pre_resolution_boundary_contamination_supplementary"][cls]
        q = qm["pre_resolution_boundary_contamination_supplementary"][cls]
        lines.append(
            f"| pre-resolution boundary ({cls}, supplementary) "
            f"| {_fmt_b(e['em_pt']['returned_as_current'])} "
            f"| {_fmt_b(e['em_pt']['masked_plan'])} "
            f"| {_fmt_b(e['ez_pt']['returned_as_current'])} "
            f"| {_fmt_b(e['ez_pt']['masked_plan'])} "
            f"| {_fmt_q(q['em_cg'])} | {_fmt_q(q['ez_cg'])} "
            f"| {_fmt_q(q['em_cw'])} | {_fmt_q(q['ez_cw'])} |")
    for cls in M4_CLASSES:
        e = m4_ep["while_pending_supplementary_IC5"][cls]
        lines.append(
            f"| while-pending ({cls}, IC-5 supplementary — no released "
            f"analog) | {_fmt_b(e['em_pt']['returned_as_current'])} "
            f"| {_fmt_b(e['em_pt']['masked_plan'])} "
            f"| {_fmt_b(e['ez_pt']['returned_as_current'])} "
            f"| {_fmt_b(e['ez_pt']['masked_plan'])} | — | — | — | — |")
    lines += [
        "", "## M4 resolve_true end-state (route-2 tracking)", "",
        "| arm | end-state correct | masked (no-route-1 signature) | "
        "released gated | released ungated | released em_cg_fixed |",
        "|---|---|---|---|---|---|"]
    for arm in ARMS:
        lines.append(
            f"| {arm} | {_fmt_b(m4_ep['end_state_resolve_true'][arm])} "
            f"| {_fmt_b(m4_ep['end_state_masked_plan_resolve_true'][arm])} "
            f"| {GATED_OF[arm]} "
            f"{_fmt_q(qm['end_state_resolve_true'][GATED_OF[arm]])} "
            f"| {UNGATED_OF[arm]} "
            f"{_fmt_q(qm['end_state_resolve_true'][UNGATED_OF[arm]])} "
            f"| {_fmt_q(qm['end_state_resolve_true']['em_cg_fixed'])} |")
    lines += [
        "", "## Task probes (ITT; deterministic read-rule lookup)", "",
        "### M4 (expected: resolve_true -> new value; abandoned/open -> "
        "established prior)", "",
        "| class | em_pt | ez_pt | released em_cg | released ez_cg | "
        "released em_cw | released ez_cw |", "|---|---|---|---|---|---|---|"]
    for cls in M4_CLASSES + ("pooled",):
        p = m4_ep["task_probes"][cls]
        q = qm["task_probes"][cls]
        lines.append(f"| {cls} | {_fmt_b(p['em_pt'])} | {_fmt_b(p['ez_pt'])} "
                     f"| {_fmt_q(q['em_cg'])} | {_fmt_q(q['ez_cg'])} "
                     f"| {_fmt_q(q['em_cw'])} | {_fmt_q(q['ez_cw'])} |")
    lines += ["", "### Unified (expected: still_open -> prior; came_true / "
              "g5 -> VALUE)", "",
              "| stratum | em_pt | ez_pt | released em_cg | released ez_cg | "
              "released em_cw | released ez_cw |",
              "|---|---|---|---|---|---|---|"]
    for oc in unify_eval.OUTCOME_CLASSES:
        p = u_ep["task_probes_ITT"]["by_outcome"][oc]
        q = cmp_["unify"]["task_probes_by_outcome"][oc]
        lines.append(f"| {oc} | {_fmt_w(p['em_pt'])} | {_fmt_w(p['ez_pt'])} "
                     f"| {_fmt_q(q['em_cg'])} | {_fmt_q(q['ez_cg'])} "
                     f"| {_fmt_q(q['em_cw'])} | {_fmt_q(q['ez_cw'])} |")
    p = u_ep["task_probes_ITT"]["pooled"]
    q = cmp_["unify"]["task_probes_pooled"]
    lines.append(f"| pooled | {_fmt_w(p['em_pt'])} | {_fmt_w(p['ez_pt'])} "
                 f"| {_fmt_q(q['em_cg'])} | {_fmt_q(q['ez_cg'])} "
                 f"| {_fmt_q(q['em_cw'])} | {_fmt_q(q['ez_cw'])} |")

    lines += ["", "## Preregistered directional-expectation scorecard "
              "(3 rows; operationalizations IC-6, disclosed)", "",
              "| row | expectation | holds |", "|---|---|---|"]
    for row in res["directional_scorecard"]:
        lines.append(f"| {row['row']} | {row['expectation']} "
                     f"| **{'HOLDS' if row['holds'] else 'DOES NOT HOLD'}** |")
    lines += [""]
    for row in res["directional_scorecard"]:
        lines.append(f"- Row {row['row']}: {row['operationalization']}")
        if row["row"] == 1:
            for arm in ARMS:
                o = row["observed"][arm]
                lines.append(
                    f"  - {arm}: unify at-end pooled pt "
                    f"{_fmt_w(o['unify_at_end_pooled_g1_g4']['pt'])} vs "
                    f"released {GATED_OF[arm]} "
                    f"{_fmt_q(o['unify_at_end_pooled_g1_g4']['released_gated'])}; "
                    f"M4 pooled pt "
                    f"{_fmt_b(o['m4_final_store_pooled_abandoned_open']['pt'])} "
                    f"vs released "
                    f"{_fmt_q(o['m4_final_store_pooled_abandoned_open']['released_gated'])}; "
                    f"while-pending (descriptive) pt "
                    f"{_fmt_w(o['unify_while_pending_pooled_g1_g4_descriptive']['pt'])} "
                    f"vs released "
                    f"{_fmt_q(o['unify_while_pending_pooled_g1_g4_descriptive']['released_gated'])}")
        elif row["row"] == 2:
            for arm in ARMS:
                o = row["observed"][arm]
                lines.append(
                    f"  - {arm}: p3+p4 pooled end-state pt "
                    f"{_fmt_w(o['pt_p3_p4_pooled_end_state'])} vs released "
                    f"{GATED_OF[arm]} "
                    f"{_fmt_q(o['released_gated_p3_p4_pooled_end_state'])}")
        else:
            o = row["observed"]
            lines.append(f"  - masked-non-plan violations: "
                         f"{o['n_violations']} (structural check)")
            for arm in ARMS:
                g = o["g5"][arm]
                lines.append(
                    f"  - {arm}: g5 commit-correct pt "
                    f"{_fmt_w(g['pt_commit_correct'])} vs released "
                    f"{GATED_OF[arm]} {_fmt_q(g['released_gated'])}; "
                    f"completed value plan-masked "
                    f"{_fmt_w(g['pt_completed_value_masked_plan'])}")

    lines += ["", "## Run stats", "",
              "| phase | selected | new | skipped | failed | fresh calls |",
              "|---|---|---|---|---|---|"]
    for r_ in res["run_phases"]:
        lines.append(f"| {r_['phase']} | {r_['selected']} | {r_['n_new']} "
                     f"| {r_['skipped']} | {len(r_['failed'])} "
                     f"| {r_['fresh_calls']:,} |")
    adv = res["advisory_budget"]
    jc = res["scoring_judge_calls"]
    bb = res.get("budget_before")
    ba = res["budget_after"]
    lines += ["", "## Budget (stage plan_typed; advisory / hard envelope / "
              "110k run stop)", ""]
    if bb:
        lines.append(f"- before: {json.dumps(bb['spent'])} "
                     f"(total {sum(bb['spent'].values()):,})")
    lines += [f"- after:  {json.dumps(ba['spent'])} "
              f"(total {sum(ba['spent'].values()):,} / envelope "
              f"{ba['envelope']:,}; run stop {HARD_STOP_TOTAL:,})",
              f"- stage plan_typed spend {adv['spent']:,} vs advisory "
              f"forecast {adv['advisory_forecast']:,} "
              f"(over: {adv['over_forecast']})",
              f"- scoring equivalence-judge calls: {jc['total']} "
              f"({jc['fresh']} fresh, stage plan_typed)",
              "",
              "Scored once (prereg). Verification (RECOUNT + SEMANTICS) "
              "precedes release — these numbers are UNVERIFIED until then.",
              ""]
    return "\n".join(lines)


async def score() -> dict:
    assert llm.JUDGE_MODEL == PINNED_SYSTEM_MODEL
    guard = pt_guard()
    guard.install()
    try:
        res = await compute_scores()
    finally:
        guard.uninstall()
    res["score_fresh_calls"] = guard.fresh
    PT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    REPORT_PATH.write_text(render_report(res))
    print(f"wrote {_rel(RESULTS_PATH)}\nwrote {_rel(REPORT_PATH)} "
          f"(m4 {res['m4']['n_arcs_scored']}/60, "
          f"unify {res['unify']['n_arcs_scored']}/179; "
          f"{guard.fresh} fresh judge calls)")
    return res


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    rm = sub.add_parser("run-m4")
    rm.add_argument("--ids", nargs="*", default=None)
    rm.add_argument("--call-cap", type=int, default=None)
    ru = sub.add_parser("run-unify")
    ru.add_argument("--ids", nargs="*", default=None)
    ru.add_argument("--shard", default=None, help="K/N over sorted ids")
    ru.add_argument("--call-cap", type=int, default=None)
    sub.add_parser("score")
    sub.add_parser("status")
    args = ap.parse_args()

    if args.cmd == "status":
        n_m4 = len(list(M4_CASES.glob("*.json"))) if M4_CASES.exists() else 0
        n_u = (len(list(UNIFY_CASES.glob("*.json")))
               if UNIFY_CASES.exists() else 0)
        print(json.dumps({**advisory_status(), "m4_case_records": n_m4,
                          "unify_case_records": n_u}, indent=1))
        return
    if args.cmd == "score":
        asyncio.run(score())
        return
    if args.cmd == "run-m4":
        summary = asyncio.run(run_m4(args.ids, args.call_cap))
    else:
        summary = asyncio.run(run_unify(args.ids, args.shard, args.call_cap))
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
