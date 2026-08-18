"""unify_eval.py — the preregistered UNIFIED-LADDER run (protocol/CHANGELOG.md
2026-08-11 "Unified-substrate ladders ACCEPTED by owner; unified-ladder run
PREREGISTERED" — the binding spec; substrate contract in the same-day
"SUBSTRATE-UNIFICATION generation contract FROZEN" entry).

SUBSTRATE: results/unify/arcs.jsonl — 180 owner-reviewed conversation arcs
(r4-shaped, unify_v5 render), MINUS the ids in unify_exclusions.json
(UNIFY-hs-g5-009, defect inside protected factor sentences) = 179 usable.
Two factor ladders: hedge-strength g1..g5 (g1-g4 outcome STILL_OPEN, g5
ALREADY_COMPLETED commit-correct control) and confirmation-wording p1..p4
(all CAME_TRUE). Keys registered from results/unify/unify_keys.jsonl (FM-4,
frozen before generation).

SYSTEMS: the four frozen cells em/ez x ungated(cw)/gated(cg) via the frozen
seams exactly as M2/M4/D1 (imported, never copied): gated_store.
extract_turn_family under syn_res.LME_PROMPTS, store._map_tuple_key (mode
OFF, asserted), classify.classify_item + gated_store.ARGMAX_ROUTE,
gated_store._generate_predicates, the frozen checker (test3 fire semantics,
consume-on-fire), ZepStore writes. em_cg carries the ADOPTED M1 plan-stem
repair as the default cautious-gated configuration (changelog 2026-07-09
"REPAIR ADOPTED"; reconfirmed v2.5 2026-08 repair-arm entries):
store.make_tuple_key_mapper(PLAN_FRAME_ATTRIBUTE_PATTERNS) passed as a
LOCAL mapper into the em_cg extract+map recs only — never installed
globally (assert_mapper_mode_off at entry AND teardown); em extraction and
the wrapper's first mapping pass are byte-identical to em_cw's calls
(llm.py cache-shared), so only `-planstem` re-offers on DECLINED
plan-framed tuples + their downstream gate calls are fresh. This is the
runner_external.run_d1_case cell convention verbatim (the most recent
m4_eval-pattern reuse, after claude_repl).

MODELS: systems = the PINNED default gpt-5.5 (llm.MODEL with
MEMGATE_SYSTEM_MODEL unset — asserted); scoring judge = the frozen
llm.JUDGE_MODEL (gpt-5.5-2026-04-23) via rule5.EquivalenceScorer
(exact-normalized fast path + judge fallback).

REPLAY (m4_eval semantics): the FULL conversation stream per arc in order;
extraction/mapping/classification on user turns only (assistant turns are
context); a single user store per arc; session-level dates (interpretive
choice #2 of m4_eval — the frozen LME B2 rendering marks one session date
per session); mapper candidates = [the arc key] alone (m4_eval interpretive
choice #3 — the unify conversations are self-contained, the arc key is the
substrate's only registered key, the mapper's "none" option keeps doing the
real work).

PER-ARC CASE RECORD (results/unify/run/cases/<arc_id>.json, resumable per
arc): per cell — final store value for the arc key; pre-final-beat boundary
snapshot; a value-after-each-user-turn timeline for the arc key (the
runner_external addition — the scorer evaluates the while-pending window
without rerun); staging rows with predicates + status + coordinates;
checker rows + valid fires with user-turn offsets relative to the final
beat; the arc-key commit/supersession event log; em_cg wrapper mapping meta
(PIN-1 audit fields); diagnostics (hedge-turn extraction/mapping funnel per
family em / em_fixed / ez + per-turn rows). Scoring never re-runs.

BEAT NAMING: "hedge" = the change-statement beat (for g5 it is the
completed-change assertion itself — the M2 rule-(d) semantics); "resolution"
= the final beat at the final session's user turn 0 (came_true: the
resolution-as-opener; still_open: the neutral still-pending nod; g5: the
state continuation). Promotion offsets anchor on it; promotion endpoints
are read only for came_true arcs.

ENDPOINTS (score once, after the run -> unify_results.json +
unify_report.md; Wilson CIs, cluster = conversation — one row per
conversation per cell per endpoint, so the Wilson interval on x/n is the
preregistered instrument, the scorer_d1 aggregation pattern):
  (1) per-gradient-level contamination, g1-g4 (still_open): hedged VALUE as
      current — while-pending (stored as current at any user turn strictly
      after the change-statement turn through the last user turn; the D1
      E1 window semantics, computed on the recorded value timeline) and
      at-end (final store equivalent). g5 = commit-correct CONTROL: a
      commit of the completed value is CORRECT, never contamination —
      reported as commit-correct-at-end per cell.
  (2) staging per level (gated cells): arc-key item staged at the
      change-statement beat.
  (3) per-wording-distance promotion (came_true, gated cells): valid
      promote fire at t* (the resolution turn, primary) / within
      [t*, t*+2] user turns (window sensitivity, test3 semantics), ITT +
      conditional-on-staged; end-state correctness (final store equivalent
      to VALUE) for ALL cells. Premature commit before the resolution turn
      reported as a supplementary (timeline window, D1 semantics).
  (4) task probes, ITT (all arcs; deterministic final-store lookup,
      unanswered = incorrect): expected answer BY OUTCOME CLASS —
      still_open -> the established prior_value; came_true -> VALUE; g5
      already_completed -> the completed value (= VALUE). This is the
      m4_eval score_case class logic carried to the unify outcome classes
      per the substrate-unification contract entry.

REPORTING (prereg): the two headline curve tables — contamination by
g-level per cell; promotion by p-distance for gated cells — plus a
comparison block quoting the RELEASED planted-probe numbers
(results/v2.4/m2/m2_results.json, read only, never rerun; unify em_cg's
planted analog is em_cg_fixed = the adopted configuration) and the
directional expectation evaluated per row: onset at g3, recovery at g4,
promotion declining with wording distance — holds/doesn't.

BUDGET: stage "unify" (ADVISORY allocation 20,000; spent 3,774 at prereg by
generation), spend recorded per fresh call to results/v2.4/budget.json via
the flock-guarded forecast_v22.SharedCallBudget. The 120k envelope remains
the protocol hard stop; THIS run additionally refuses to start / stops if
the combined total would pass 110,000 (prereg stop condition), enforced via
local_cap (the claude_repl guard pattern).

CLI (repo root; MEMGATE_SYSTEM_MODEL must be UNSET):
  pilot/.venv/bin/python experiments/v2_full/unify_eval.py \
      run [--ids ...] [--shard K/N] [--call-cap N]
  pilot/.venv/bin/python experiments/v2_full/unify_eval.py score
  pilot/.venv/bin/python experiments/v2_full/unify_eval.py status
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
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
import run_predicates as test3
import stats
import store
import syn_res
import zepstore
from gated_store import ARGMAX_ROUTE
from graded_eval import _CellSim, _read_jsonl, _rel, _ungated_tuples
from router_sweep import PLAN_FRAME_ATTRIBUTE_PATTERNS

from paths import PROMPTS_DIR, RESULTS_DIR

STAGE = "unify"
PINNED_SYSTEM_MODEL = "gpt-5.5-2026-04-23"
ADVISORY_FORECAST = 20_000
#: Prereg stop condition for THIS run (the protocol hard stop stays 120k).
HARD_STOP_TOTAL = 110_000

V24_DIR = RESULTS_DIR / "v2.4"
BUDGET_PATH = V24_DIR / "budget.json"

UNIFY_DIR = RESULTS_DIR / "unify"
ARCS_PATH = UNIFY_DIR / "arcs.jsonl"
KEYS_PATH = UNIFY_DIR / "unify_keys.jsonl"
EXCLUSIONS_PATH = UNIFY_DIR / "unify_exclusions.json"
RUN_DIR = UNIFY_DIR / "run"
CASES_DIR = RUN_DIR / "cases"
RESULTS_PATH = RUN_DIR / "unify_results.json"
REPORT_PATH = RUN_DIR / "unify_report.md"
RUN_LOG = RUN_DIR / "run_log.jsonl"
BUDGET_BEFORE = RUN_DIR / "budget_before.json"

#: Released planted-probe numbers the comparison block quotes (READ-ONLY —
#: the prereg forbids rerunning them).
RELEASED_M2 = V24_DIR / "m2" / "m2_results.json"

CELLS = ("em_cw", "em_cg", "ez_cw", "ez_cg")
GATED_CELLS = ("em_cg", "ez_cg")

G_LEVELS = ("g1_strong_hedge", "g2_weak_hedge", "g3_confident_plan",
            "g4_scheduled_event", "g5_completed_control")
P_LEVELS = ("p1_verbatim", "p2_paraphrase", "p3_oblique", "p4_situational")
STILL_OPEN_LEVELS = G_LEVELS[:4]

#: unify outcome class per arc resolution_type (contract entry semantics).
OUTCOME_BY_RESOLUTION_TYPE = {"open": "still_open",
                              "completed_control": "already_completed",
                              "resolve_true": "came_true"}
OUTCOME_CLASSES = ("still_open", "already_completed", "came_true")

#: unify level -> released planted (M2) level for the comparison block.
M2_LEVEL_OF = {**{lv: lv for lv in G_LEVELS[:4] + P_LEVELS},
               "g5_completed_control": "g5_present_tense_change"}
#: unify cell -> released planted cell (em_cg here IS the adopted repaired
#: configuration, whose planted row is em_cg_fixed).
M2_CELL_OF = {"em_cw": "em_cw", "em_cg": "em_cg_fixed",
              "ez_cw": "ez_cw", "ez_cg": "ez_cg"}

SENSITIVITY_WINDOW = test3.SENSITIVITY_WINDOW  # = 2 (v1 §3.4 semantics)

EM_CG_REPAIR_NOTE = (
    "em_cg = the ADOPTED cautious-gated configuration (changelog 2026-07-09 "
    "'REPAIR ADOPTED'): the M1 plan-stem wrapper "
    "(store.make_tuple_key_mapper over "
    "router_sweep.PLAN_FRAME_ATTRIBUTE_PATTERNS, mode 'plan_stem_wrapper') "
    "passed as a LOCAL mapper for the em_cg recs only — never installed "
    "globally; extraction + first-pass mapping cache-shared with em_cw "
    "(the runner_external.run_d1_case convention)")


# ---------------------------------------------------------------------------
# Budget (advisory stage unify; 110k prereg stop via local_cap)
# ---------------------------------------------------------------------------

def unify_guard(path: Path = BUDGET_PATH,
                local_cap: int | None = None) -> forecast_v22.SharedCallBudget:
    """The claude_repl.claude_guard pattern, stage unify: the stage forecast
    is ADVISORY (warn + continue, 2026-07-08 amendment); this run's own stop
    condition caps fresh calls so the combined total cannot pass 110k."""
    budget = mapper_fix.ensure_budget(path)
    spent = budget["spent"].get(STAGE, 0)
    forecast = budget["allocation"].get(STAGE, ADVISORY_FORECAST)
    if spent >= forecast:
        print(f"[unify budget ADVISORY] stage spend {spent:,} >= forecast "
              f"{forecast:,} — continuing (stages are advisory per the "
              "2026-07-08 amendment; only the total envelope is hard)")
    margin_110 = HARD_STOP_TOTAL - sum(budget["spent"].values())
    if margin_110 <= 0:
        raise SystemExit(
            f"unify STOP CONDITION: combined spend "
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
                " / 110k run stop (2026-08-11 unified-ladder prereg)",
    }


# ---------------------------------------------------------------------------
# Pins + bookkeeping
# ---------------------------------------------------------------------------

def assert_pins() -> None:
    assert os.environ.get("MEMGATE_SYSTEM_MODEL") is None, (
        "MEMGATE_SYSTEM_MODEL is set — the unified-ladder run is "
        "preregistered on the PINNED default gpt-5.5 systems")
    assert llm.MODEL == PINNED_SYSTEM_MODEL, (
        f"llm.MODEL is {llm.MODEL!r} — expected the pinned "
        f"{PINNED_SYSTEM_MODEL}")
    assert llm.JUDGE_MODEL == PINNED_SYSTEM_MODEL, (
        f"llm.JUDGE_MODEL drifted to {llm.JUDGE_MODEL!r}")
    m4_eval.assert_mapper_mode_off()


def snapshot_budget_before() -> None:
    if not BUDGET_BEFORE.exists():
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        BUDGET_BEFORE.write_text(BUDGET_PATH.read_text())


def log_phase(row: dict) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUN_LOG, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _shard(ids: list[str], spec: str | None) -> list[str]:
    if not spec:
        return ids
    k, n = (int(x) for x in spec.split("/"))
    assert 0 <= k < n, f"bad shard spec {spec!r}"
    return ids[k::n]


# ---------------------------------------------------------------------------
# Substrate loading (owner-accepted 179 = 180 minus the registered exclusion)
# ---------------------------------------------------------------------------

def load_exclusions(path: Path = EXCLUSIONS_PATH) -> set[str]:
    excl = json.loads(path.read_text())
    ids = set(excl["excluded"])
    assert ids == {"UNIFY-hs-g5-009"}, (
        f"unexpected exclusion set {ids} — the prereg registers exactly "
        "UNIFY-hs-g5-009")
    return ids


def load_arcs(path: Path = ARCS_PATH) -> list[dict]:
    arcs = _read_jsonl(path)
    assert len(arcs) == 180, f"expected 180 unify arcs, got {len(arcs)}"
    excluded = load_exclusions()
    arcs = [a for a in arcs if a["arc_id"] not in excluded]
    assert len(arcs) == 179, f"expected 179 usable arcs, got {len(arcs)}"
    assert all(a.get("render_round") == "unify_v5" for a in arcs), (
        "non-unify_v5 arc present")
    for a in arcs:
        assert a["unify_level"] in G_LEVELS + P_LEVELS, a["arc_id"]
        assert a["resolution_type"] in OUTCOME_BY_RESOLUTION_TYPE, a["arc_id"]
    return sorted(arcs, key=lambda a: a["arc_id"])


def load_keys(path: Path = KEYS_PATH) -> dict[str, dict]:
    return {k["arc_id"]: k for k in _read_jsonl(path)}


def arcs_sha256(path: Path = ARCS_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arc_outcome(arc: dict) -> str:
    return OUTCOME_BY_RESOLUTION_TYPE[arc["resolution_type"]]


def case_path(arc_id: str, out_root: Path = CASES_DIR) -> Path:
    return out_root / f"{arc_id}.json"


# ---------------------------------------------------------------------------
# run — per-turn seams (frozen, imported; unify trace tags — the established
# per-experiment tag pattern of graded_eval/m4_eval/runner_external)
# ---------------------------------------------------------------------------

async def _extract_and_map(family: str, turns: list[dict], pos: int,
                           candidates: list[store.CanonicalKey],
                           arc_id: str, mapper=None) -> dict:
    """(tuples, mapped) for one user turn — the m4_eval._extract_and_map body
    with unify trace tags. ``mapper`` None = the frozen store._map_tuple_key;
    the em_cg recs pass the adopted stem wrapper (LOCAL — its first pass is
    byte-identical to the frozen mapping, cache-shared)."""
    si, ti = turns[pos]["session_idx"], turns[pos]["turn_idx"]
    tag = f"unify-extract-{family}-{arc_id}-s{si}t{ti}"
    prior = [{"role": t["role"], "content": t["content"], "date": t["date"]}
             for t in turns[:pos]]
    tuples = await gated_store.extract_turn_family(
        prior, turns[pos]["content"], tag,
        system_prompt=(PROMPTS_DIR / syn_res.LME_PROMPTS[family]).read_text(),
        current_date=turns[pos]["date"])
    map_call = store._map_tuple_key if mapper is None else mapper
    mapped = list(await asyncio.gather(*[
        map_call(tup, candidates,
                 trace_tag=f"unify-keymap-{family}-{arc_id}-s{si}t{ti}-{j}")
        for j, tup in enumerate(tuples)]))
    return {"pos": pos, "session_idx": si, "turn_idx": ti,
            "date": turns[pos]["date"], "turn_text": turns[pos]["content"],
            "tuples": tuples, "mapped": mapped}


async def _gated_checker(sim: _CellSim, rec: dict, family: str, arc_id: str,
                         checker_system: str, turn_offset: int) -> None:
    """The m4_eval._gated_checker body (frozen fire semantics,
    consume-on-fire, user-turn offsets) with unify trace tags."""
    for store_id in sorted(sim.pending):
        items = sim.pending[store_id]
        if not items:
            continue
        pending_ids = {p["staged_id"] for p in items}
        output = await llm.call_llm(
            checker_system,
            test3.build_checker_user_prompt(
                rec["turn_text"], rec["date"],
                [gated_store._pending_item(p) for p in items]),
            trace_tag=f"unify-check-{family}-{arc_id}-{store_id}-"
                      f"s{rec['session_idx']}t{rec['turn_idx']}")
        fires, malformed, _extra = test3.parse_checker_fires(
            output, pending_ids, rec["turn_text"])
        consumed = {f["staged_id"] for f in fires if f["staged_id"] in pending_ids}
        for item in items:
            sid = item["staged_id"]
            if sid not in consumed:
                continue
            item_fires = [f for f in fires if f["staged_id"] == sid]
            valid = [f for f in item_fires if not f["invalid"]]
            if valid and valid[0]["action"] == "promote":
                outcome = "promoted"
                sim.zstore(store_id).write(
                    (item["entity"], item["attribute"]),
                    item["hypothesized_value"], valid_from=rec["date"],
                    provenance={"kind": "gated_promotion", "staged_id": sid},
                    correction=False)
            elif valid and valid[0]["action"] == "invalidate":
                outcome = "invalidated"
            else:
                outcome = "discarded_invalid"
            sim.staged_rows[sid]["status"] = outcome
        sim.pending[store_id] = [p for p in items
                                 if p["staged_id"] not in consumed]
        sim.checker_rows.append({
            "store_id": store_id, "session_idx": rec["session_idx"],
            "turn_idx": rec["turn_idx"], "turn_offset": turn_offset,
            "pending_ids": sorted(pending_ids), "fires": fires,
            "malformed_output": malformed})


async def _gated_tuples(sim: _CellSim, rec: dict, turns: list[dict],
                        family: str, arc_id: str, gen_system: str) -> None:
    """The m4_eval._gated_tuples body (frozen argmax routing verbatim; staged
    rows carry their staging-turn coordinates) with unify ids/tags."""
    stream = [{"session_idx": t["session_idx"], "role": t["role"],
               "content": t["content"], "date": t["date"]} for t in turns]
    for j, (tup, mres) in enumerate(zip(rec["tuples"], rec["mapped"])):
        cand, raw_key, _rationale, _meta = gated_store._unpack_map_result(mres)
        sim.diag["n_tuples"] += 1
        if cand is None:
            sim.diag["n_unmapped"] += 1
            continue
        sim.diag["n_mapped"] += 1
        store_id = sim.route(cand.key)
        zstore = sim.zstore(store_id)
        stored_now = zstore.lookup(cand.key)
        item = {
            "item_id": f"unify-{family}-{arc_id}-s{rec['session_idx']}"
                       f"t{rec['turn_idx']}-{j}",
            "entity": cand.entity, "attribute": cand.attribute,
            "asserted_value": tup.value, "stored_value": stored_now,
            "turn_text": rec["turn_text"],
            "preceding_turns": gated_store._preceding_turns(stream, rec["pos"]),
        }
        cls_out = await classify.classify_item(item)
        cls = cls_out["argmax"]
        route_to = ARGMAX_ROUTE[cls]
        gate_row = {"item_id": item["item_id"], "session_idx": rec["session_idx"],
                    "turn_idx": rec["turn_idx"], "store_id": store_id,
                    "key": cand.name, "value": tup.value, "class": cls,
                    "s_ev": cls_out["s_ev"], "probs": cls_out["probs"]}
        if route_to == "commit":
            event = zstore.write(cand.key, tup.value, valid_from=rec["date"],
                                 provenance={"class": cls},
                                 correction=(cls == "explicit_correction"))
            gate_row["action"] = event["action"]
        elif route_to == "stage":
            staged_id = f"stg-unify-{family}-{arc_id}-s{rec['session_idx']}" \
                        f"t{rec['turn_idx']}-{j}"
            row_shim = {"pos": rec["pos"], "turn_text": rec["turn_text"]}
            staged = await gated_store._generate_predicates(
                stream, row_shim, cand, tup, staged_id=staged_id,
                stream_id=arc_id, family=family, gen_system=gen_system,
                diagnostics=defaultdict(int))
            staged["status"] = "pending"
            staged["store_id"] = store_id
            staged["session_idx"] = rec["session_idx"]
            staged["turn_idx"] = rec["turn_idx"]
            sim.staged_rows[staged_id] = staged
            sim.pending.setdefault(store_id, []).append(staged)
            gate_row["action"] = "staged"
            gate_row["staged_id"] = staged_id
        else:  # additive: commit-if-absent-or-equal, never a supersession
            new_stored = zepstore.apply_sentinel(tup.value)
            if stored_now is None or (store.normalize_value(new_stored)
                                      == store.normalize_value(stored_now)):
                event = zstore.write(cand.key, tup.value, valid_from=rec["date"],
                                     provenance={"class": cls}, correction=False)
                gate_row["action"] = f"additive_{event['action']}"
            else:
                gate_row["action"] = "additive_noop"
        sim.gate_rows.append(gate_row)


async def _run_policy_cell(policy: str, recs: list[dict], family: str,
                           arc_id: str, target: store.CanonicalKey,
                           turns: list[dict], user_index: dict, res_pos: int,
                           hedge_pos: int, gen_system: str,
                           checker_system: str) -> dict:
    """The m4_eval._run_policy_cell body + the runner_external value timeline
    (target_value_after_turn) so the scorer can evaluate the while-pending
    window without rerun. Mapper-agnostic: the recs already carry the mapping
    results (frozen or adopted wrapper); a fresh _CellSim per call keeps
    every cell's store state independent."""
    sim = _CellSim(policy, "lme")
    store_id = sim.route(target.key)
    pre_resolution_value = None
    target_events: list[dict] = []
    value_timeline: list[dict] = []
    for rec in recs:
        offset = user_index[rec["pos"]] - user_index[res_pos]
        if rec["pos"] == res_pos:
            pre_resolution_value = sim.lookup(target.key)
        n_events = len(sim.zstore(store_id).events)
        if policy == "gated":
            await _gated_checker(sim, rec, family, arc_id, checker_system, offset)
            await _gated_tuples(sim, rec, turns, family, arc_id, gen_system)
        else:
            await _ungated_tuples(sim, rec)
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
        value_timeline.append({
            "session_idx": rec["session_idx"], "turn_idx": rec["turn_idx"],
            "user_idx": user_index[rec["pos"]],
            "user_offset_from_resolution": offset,
            "value": sim.lookup(target.key)})
    final = sim.lookup(target.key)
    staged_target = []
    for sid, srow in sim.staged_rows.items():
        if (srow["entity"], srow["attribute"]) != target.key:
            continue
        s_pos = m4_eval.turn_pos(turns, {"session_idx": srow["session_idx"],
                                         "turn_idx": srow["turn_idx"]})
        staged_target.append({
            "staged_id": sid, "status": srow["status"],
            "hypothesized_value": srow["hypothesized_value"],
            "promote_when": srow["promote_when"],
            "invalidate_when": srow["invalidate_when"],
            "session_idx": srow["session_idx"],
            "turn_idx": srow["turn_idx"],
            "user_idx": user_index[s_pos],
            "at_hedge_turn": s_pos == hedge_pos,
        })
    valid_fires = [
        {"turn_offset": c["turn_offset"], "staged_id": f["staged_id"],
         "action": f["action"], "evidence_span": f["evidence_span"],
         "span_valid": f["span_valid"]}
        for c in sim.checker_rows for f in c["fires"] if not f["invalid"]]
    promote_offsets = [f["turn_offset"] for f in valid_fires
                       if f["action"] == "promote"]
    nonneg = [o for o in promote_offsets if o >= 0]
    return {
        "final_value": final,
        "final_present": (final is not None
                          and final != zepstore.UNKNOWN_SENTINEL),
        "pre_resolution_value": pre_resolution_value,
        "target_staged_at_hedge": any(s["at_hedge_turn"]
                                      for s in staged_target),
        "staged_target": staged_target,
        "checker_rows": sim.checker_rows,
        "valid_fires": valid_fires,
        "promote_fire_at_tstar": 0 in promote_offsets,
        "promote_fire_within_window": any(
            0 <= o <= SENSITIVITY_WINDOW for o in promote_offsets),
        "first_valid_promote_offset": (min(nonneg) if nonneg
                                       else (min(promote_offsets)
                                             if promote_offsets else None)),
        "invalidate_fire_offsets": [f["turn_offset"] for f in valid_fires
                                    if f["action"] == "invalidate"],
        "target_events": target_events,
        "target_value_after_turn": value_timeline,
        "gate_rows_target": [g for g in sim.gate_rows
                             if g["key"] == target.name],
        "diag": dict(sim.diag),
    }


async def run_arc(arc: dict, key_row: dict, arcs_digest: str) -> dict:
    """One arc through the four frozen cells over the FULL conversation
    stream (em_cg = adopted repaired configuration, LOCAL wrapper)."""
    arc_id = arc["arc_id"]
    geom = m4_eval.arc_geometry(arc)
    turns, user_positions = geom["turns"], geom["user_positions"]
    user_index, hedge_pos, res_pos = (geom["user_index"], geom["hedge_pos"],
                                      geom["res_pos"])
    target = m4_eval.arc_canonical_key(key_row)
    candidates = [target]  # m4_eval interpretive choice #3 (docstring)
    gen_system = gated_store.GENERATOR_PROMPT_PATH.read_text()
    checker_system = gated_store.CHECKER_PROMPT_PATH.read_text()

    out = {"arc_id": arc_id,
           "unify_ladder": arc["unify_ladder"],
           "unify_level": arc["unify_level"],
           "m2_level": arc["m2_level"],
           "outcome": arc_outcome(arc),
           "resolution_type": arc["resolution_type"],
           "gold_resolution_status": arc["gold_resolution_status"],
           "hedge_strength": arc["hedge_strength"],
           "cluster": arc_id,  # cluster = conversation (prereg)
           "target_key": target.name,
           "hedged_value": arc["value"],
           "value_candidates": arc.get("value_candidates"),
           "prior_value": arc.get("prior_value"),
           "probe_question": key_row["probe_question"],
           "pending_window_user_turns": arc.get("pending_window_user_turns"),
           "realized_pending_window_user_turns":
               arc.get("realized_pending_window_user_turns"),
           "arc_status": arc.get("status"), "arc_usable": arc.get("usable"),
           "stratum_note": ("owner ACCEPTED the 179-arc stratum "
                            "(changelog 2026-08-11) — the frozen arcs.jsonl "
                            "rows retain their pre-review status fields"),
           "em_cg_repair": EM_CG_REPAIR_NOTE,
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
           "beat_note": ("hedge = the change-statement beat (g5: the "
                         "completed-change assertion); resolution = the "
                         "final beat (came_true: resolution opener; "
                         "still_open: still-pending nod; g5: state "
                         "continuation). Promotion endpoints read offsets "
                         "only for came_true arcs."),
           "cells": {}, "diagnostics": {}}

    async def _recs(family: str, mapper=None) -> list[dict]:
        return [await _extract_and_map(family, turns, pos, candidates,
                                       arc_id, mapper=mapper)
                for pos in user_positions]

    # em: frozen mapping feeds em_cw; the ADOPTED stem wrapper feeds em_cg
    # (default cautious-gated config). Extraction + first-pass mapping are
    # byte-identical across the two rec sets (llm.py cache-shared).
    m4_eval.assert_mapper_mode_off()  # global frozen mapper untouched at entry
    recs_em = await _recs("em")
    wrapper = store.make_tuple_key_mapper(PLAN_FRAME_ATTRIBUTE_PATTERNS)
    recs_em_fixed = await _recs("em", mapper=wrapper)
    m4_eval.assert_mapper_mode_off()  # teardown: the wrapper stayed local
    recs_ez = await _recs("ez")

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
    out["cells"]["em_cw"] = await _run_policy_cell("ungated", recs_em, "em", **kw)
    em_cg = await _run_policy_cell("gated", recs_em_fixed, "em", **kw)
    em_cg["mapping_meta"] = m4_eval._fixed_mapping_meta(
        recs_em_fixed, user_index, target)
    out["cells"]["em_cg"] = em_cg
    out["cells"]["ez_cw"] = await _run_policy_cell("ungated", recs_ez, "ez", **kw)
    out["cells"]["ez_cg"] = await _run_policy_cell("gated", recs_ez, "ez", **kw)
    return out


async def run(ids: list[str] | None, shard: str | None,
              call_cap: int | None) -> dict:
    assert_pins()
    snapshot_budget_before()
    arcs = load_arcs()
    keys = load_keys()
    digest = arcs_sha256()
    sel_ids = _shard(sorted(ids if ids is not None
                            else [a["arc_id"] for a in arcs]), shard)
    by_id = {a["arc_id"]: a for a in arcs}
    started = datetime.now(timezone.utc).isoformat()
    guard = unify_guard(local_cap=call_cap)
    done = skipped = 0
    failed: list[dict] = []
    guard.install()
    try:
        for arc_id in sel_ids:
            arc = by_id.get(arc_id)
            assert arc is not None, f"unknown/excluded arc id {arc_id!r}"
            path = case_path(arc_id)
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
    finally:
        guard.uninstall()
    summary = {"phase": "unify_run", "model": llm.MODEL, "started": started,
               "ended": datetime.now(timezone.utc).isoformat(),
               "selected": len(sel_ids), "n_new": done, "skipped": skipped,
               "failed": failed, "fresh_calls": guard.fresh,
               "advisory": advisory_status()}
    log_phase(summary)
    return summary


# ---------------------------------------------------------------------------
# score — preregistered endpoints (Wilson CIs; cluster = conversation)
# ---------------------------------------------------------------------------

def _pt(rows: list[dict], flag: str) -> dict:
    """x/n + Wilson over per-conversation rows (one row per conversation per
    cell, so cluster = conversation holds trivially — the scorer_d1
    aggregation pattern; the prereg names Wilson CIs)."""
    sel = [r for r in rows if r.get(flag) is not None]
    x = sum(1 for r in sel if r[flag])
    n = len(sel)
    if n == 0:
        return {"x": 0, "n": 0, "rate": None,
                "wilson_lo": None, "wilson_hi": None}
    lo, hi = stats.wilson_ci(x, n)
    return {"x": x, "n": n, "rate": x / n, "wilson_lo": lo, "wilson_hi": hi}


def _sel(rows: list[dict], cell: str, level: str | None = None,
         outcome: str | None = None) -> list[dict]:
    return [r for r in rows if r["cell"] == cell
            and (level is None or r["unify_level"] == level)
            and (outcome is None or r["outcome"] == outcome)]


async def score_case(case: dict, scorer) -> list[dict]:
    """Per-arc score rows, one per cell — pure function of the case record +
    the frozen equivalence scorer (never re-runs anything). The m4_eval
    score_case class logic carried to the unify outcome classes."""
    attr = case["target_key"].partition("/")[2]
    hv = case["hedged_value"]
    outcome = case["outcome"]
    prior = case.get("prior_value")
    hedge_u = case["geometry"]["hedge"]["user_idx"]
    res_u = case["geometry"]["resolution"]["user_idx"]
    last_u = case["geometry"]["n_user_turns"] - 1
    rows = []
    for cell in [c for c in CELLS if c in case["cells"]]:
        c = case["cells"][cell]
        final_eq, _m = await scorer.equivalent(attr, c["final_value"], hv)

        async def window_match(lo_u: int, hi_u: int) -> bool:
            """hedged VALUE stored as CURRENT strictly after lo_u and before
            hi_u (D1 scorer semantics; still_open passes last_u + 1 so the
            window runs THROUGH the last user turn)."""
            for entry in c["target_value_after_turn"]:
                u = entry["user_idx"]
                if not (lo_u < u < hi_u):
                    continue
                eq, _mm = await scorer.equivalent(attr, entry["value"], hv)
                if eq:
                    return True
            return False

        committed_ever = False
        committed_at_hedge = False
        for ev in c["target_events"]:
            if ev["action"] not in ("commit", "supersede"):
                continue  # refresh never introduces a value
            eq, _mm = await scorer.equivalent(attr, ev["value"], hv)
            if eq:
                committed_ever = True
                if ev["at_hedge_turn"]:
                    committed_at_hedge = True

        # (1) contamination (still_open) / g5 commit-correct control
        while_pending = at_end = None
        commit_correct = None
        if outcome == "still_open":
            while_pending = await window_match(hedge_u, last_u + 1)
            at_end = bool(final_eq)
        elif outcome == "already_completed":
            commit_correct = bool(final_eq)  # a commit is CORRECT (rule (d))

        # (3) promotion (came_true, gated) + end-state + premature supplement
        promote_t = promote_w = None
        end_state = None
        premature = None
        if outcome == "came_true":
            end_state = bool(final_eq)
            premature = await window_match(hedge_u, res_u)
            if cell in GATED_CELLS:
                promote_t = bool(c.get("promote_fire_at_tstar"))
                promote_w = bool(c.get("promote_fire_within_window"))

        # (4) task probe, ITT (deterministic lookup; unanswered = incorrect)
        probe_value = (c["final_value"]
                       if m4_eval._answered(c["final_value"]) else None)
        if outcome == "still_open":
            probe_expected = prior
        else:  # came_true and already_completed: the (completed) VALUE
            probe_expected = hv
        eq, _mm = await scorer.equivalent(attr, probe_value, probe_expected)
        probe_correct = bool(eq)

        rows.append({
            "arc_id": case["arc_id"],
            "cluster": case["cluster"],
            "cell": cell,
            "unify_ladder": case["unify_ladder"],
            "unify_level": case["unify_level"],
            "outcome": outcome,
            "hedged_value": hv,
            "final_value": c["final_value"],
            "pre_resolution_value": c["pre_resolution_value"],
            "final_eq_hedged": bool(final_eq),
            # (1)
            "while_pending_contamination": while_pending,
            "at_end_contamination": at_end,
            "g5_commit_correct": commit_correct,
            "g5_committed_ever": (committed_ever
                                  if outcome == "already_completed" else None),
            "committed_at_hedge": committed_at_hedge,
            # (2)
            "staged_at_hedge": (bool(c["target_staged_at_hedge"])
                                if cell in GATED_CELLS else None),
            # (3)
            "promote_at_tstar": promote_t,
            "promote_within_window": promote_w,
            "end_state_correct": end_state,
            "premature_commit_before_resolution": premature,
            "promotion_delay_user_turns": (
                c.get("first_valid_promote_offset")
                if (outcome == "came_true" and cell in GATED_CELLS
                    and c.get("first_valid_promote_offset") is not None
                    and c["first_valid_promote_offset"] >= 0) else None),
            # (4)
            "probe_value": probe_value,
            "probe_expected": probe_expected,
            "probe_answered": probe_value is not None,
            "probe_correct": probe_correct,
        })
    return rows


def _load_cases(out_root: Path = CASES_DIR) -> list[dict]:
    if not out_root.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(out_root.glob("*.json"))]


# --- released planted-probe comparison (READ-ONLY) --------------------------

def _released_pooled(m2: dict, level: str, cell: str, endpoint: str) -> dict:
    """Pool one released planted endpoint across benchmarks (n=60/block —
    the m2 adoption-arithmetic convention; quoted, never rerun)."""
    trees = (m2["gradient_curves"], m2["nearmiss_promotion_curves"],
             m2["nearmiss_staging"])
    x = n = 0
    for bench in ("lme", "locomo"):
        for tree in trees:
            k = f"{bench}/{level}"
            if k in tree and cell in tree[k]["cells"]:
                p = tree[k]["cells"][cell].get(endpoint)
                if p and p.get("n"):
                    x += p["x"]
                    n += p["n"]
                break
    return {"x": x, "n": n, "rate": (x / n) if n else None}


def planted_comparison(rows: list[dict]) -> dict:
    """The prereg comparison block: unify curve points next to the released
    planted-probe numbers (results/v2.4/m2/m2_results.json, quoted), and the
    directional expectation evaluated per row — onset at g3, recovery at g4,
    promotion declining with wording distance. Unify contamination analog =
    at-end (final store, the planted endpoint); unify em_cg's planted analog
    is em_cg_fixed (the adopted configuration)."""
    m2 = json.loads(RELEASED_M2.read_text())

    def unify_pt(cell, level, flag):
        return _pt(_sel(rows, cell, level), flag)

    def _fmt_dir(name, expectation, uni_obs, uni_holds, rel_obs, rel_holds):
        return {"row": name, "expectation": expectation,
                "unify_observed": uni_obs, "unify_holds": uni_holds,
                "released_planted_observed": rel_obs,
                "released_planted_holds": rel_holds,
                "matches_released": (uni_holds == rel_holds
                                     if None not in (uni_holds, rel_holds)
                                     else None)}

    def r(p):
        return p.get("rate")

    quotes: dict = {"contamination_at_end_by_level": {}, "promotion_by_distance": {}}
    for lv in G_LEVELS:
        quotes["contamination_at_end_by_level"][lv] = {}
        for cell in CELLS:
            flag = ("g5_commit_correct" if lv == "g5_completed_control"
                    else "at_end_contamination")
            quotes["contamination_at_end_by_level"][lv][cell] = {
                "unify": unify_pt(cell, lv, flag),
                "released_planted": _released_pooled(
                    m2, M2_LEVEL_OF[lv], M2_CELL_OF[cell], "contamination"),
                "released_cell": M2_CELL_OF[cell],
            }
    for lv in P_LEVELS:
        quotes["promotion_by_distance"][lv] = {}
        for cell in GATED_CELLS:
            quotes["promotion_by_distance"][lv][cell] = {
                "unify_window_ITT": unify_pt(cell, lv, "promote_within_window"),
                "unify_tstar_ITT": unify_pt(cell, lv, "promote_at_tstar"),
                "released_planted_window_ITT": _released_pooled(
                    m2, M2_LEVEL_OF[lv], M2_CELL_OF[cell],
                    "promote_fire_within_window_ITT"),
                "released_cell": M2_CELL_OF[cell],
            }

    scorecard = []
    for cell in CELLS:
        flag = "at_end_contamination"
        u = {lv: unify_pt(cell, lv, flag) for lv in STILL_OPEN_LEVELS}
        p = {lv: _released_pooled(m2, M2_LEVEL_OF[lv], M2_CELL_OF[cell],
                                  "contamination") for lv in STILL_OPEN_LEVELS}

        def _obs(d):
            return "; ".join(
                f"{lv.split('_')[0]} {d[lv]['x']}/{d[lv]['n']}="
                f"{d[lv]['rate']:.3f}" if d[lv]["rate"] is not None
                else f"{lv.split('_')[0]} —" for lv in STILL_OPEN_LEVELS)

        def _onset(d):
            if any(r(d[lv]) is None for lv in STILL_OPEN_LEVELS[:3]):
                return None
            return (r(d["g3_confident_plan"]) > r(d["g1_strong_hedge"])
                    and r(d["g3_confident_plan"]) > r(d["g2_weak_hedge"]))

        def _recovery(d):
            if (r(d["g4_scheduled_event"]) is None
                    or r(d["g3_confident_plan"]) is None):
                return None
            return r(d["g4_scheduled_event"]) < r(d["g3_confident_plan"])

        scorecard.append(_fmt_dir(
            f"onset-at-g3/{cell}",
            "failure onset at confident-plan phrasing: at-end contamination "
            "g3 > g1 AND g3 > g2",
            _obs(u), _onset(u), _obs(p), _onset(p)))
        scorecard.append(_fmt_dir(
            f"recovery-at-g4/{cell}",
            "recovery with explicit dates: at-end contamination g4 < g3",
            _obs(u), _recovery(u), _obs(p), _recovery(p)))
    for cell in GATED_CELLS:
        u1 = unify_pt(cell, "p1_verbatim", "promote_within_window")
        u4 = unify_pt(cell, "p4_situational", "promote_within_window")
        p1 = _released_pooled(m2, "p1_verbatim", M2_CELL_OF[cell],
                              "promote_fire_within_window_ITT")
        p4 = _released_pooled(m2, "p4_situational", M2_CELL_OF[cell],
                              "promote_fire_within_window_ITT")

        def _pobs(a, b):
            def f(p):
                return (f"{p['x']}/{p['n']}={p['rate']:.3f}"
                        if p["rate"] is not None else "—")
            return f"p1 {f(a)}; p4 {f(b)}"

        scorecard.append(_fmt_dir(
            f"promotion-declines/{cell}",
            "promotion declines with wording distance: window-ITT p1 > p4",
            _pobs(u1, u4),
            (None if None in (r(u1), r(u4)) else r(u1) > r(u4)),
            _pobs(p1, p4),
            (None if None in (r(p1), r(p4)) else r(p1) > r(p4))))
    return {
        "released_source": str(RELEASED_M2),
        "cell_mapping_note": ("unify em_cg = the ADOPTED repaired "
                              "configuration; its released planted analog is "
                              "em_cg_fixed. Released numbers are pooled "
                              "lme+locomo (n=60/block), quoted — never rerun."),
        "contamination_analog_note": ("unify contamination analog for the "
                                      "comparison = at-end (final store), the "
                                      "planted suite's contamination endpoint"),
        "quotes": quotes,
        "directional_scorecard": scorecard,
    }


async def compute_scores(out_root: Path = CASES_DIR) -> dict:
    cases = _load_cases(out_root)
    assert cases, f"no case records under {out_root} — run `run` first"
    digests = {c.get("arcs_sha256") for c in cases}
    assert len(digests) == 1, (
        f"case records span multiple arcs.jsonl digests: {digests}")
    excluded = load_exclusions()
    assert not ({c["arc_id"] for c in cases} & excluded), (
        "an excluded arc has a case record")
    scorer = rule5.EquivalenceScorer(cap=6000)

    rows: list[dict] = []
    for case in cases:
        rows.extend(await score_case(case, scorer))

    endpoints: dict = {}
    # (1) contamination by gradient level (g1-g4) + g5 commit-correct control
    endpoints["contamination_by_level"] = {
        lv: {cell: {
            "while_pending": _pt(_sel(rows, cell, lv),
                                 "while_pending_contamination"),
            "at_end": _pt(_sel(rows, cell, lv), "at_end_contamination")}
            for cell in CELLS}
        for lv in STILL_OPEN_LEVELS}
    endpoints["g5_commit_correct_control"] = {
        cell: {"commit_correct_at_end": _pt(
                   _sel(rows, cell, "g5_completed_control"),
                   "g5_commit_correct"),
               "committed_ever": _pt(
                   _sel(rows, cell, "g5_completed_control"),
                   "g5_committed_ever")}
        for cell in CELLS}
    # (2) staging per level (gated cells; both ladders)
    endpoints["staging_by_level"] = {
        lv: {cell: _pt(_sel(rows, cell, lv), "staged_at_hedge")
             for cell in GATED_CELLS}
        for lv in G_LEVELS + P_LEVELS}
    # (3) promotion by wording distance (came_true) + end-state + supplements
    endpoints["promotion_by_distance"] = {}
    for lv in P_LEVELS:
        entry: dict = {}
        for cell in CELLS:
            sel = _sel(rows, cell, lv)
            e: dict = {"end_state_correct_ITT": _pt(sel, "end_state_correct"),
                       "premature_commit_before_resolution_supplementary":
                           _pt(sel, "premature_commit_before_resolution")}
            if cell in GATED_CELLS:
                e["promote_fire_at_tstar_ITT"] = _pt(sel, "promote_at_tstar")
                e["promote_fire_within_window_ITT"] = _pt(
                    sel, "promote_within_window")
                staged_sel = [x for x in sel if x["staged_at_hedge"]]
                e["conditional_on_staged"] = {
                    "n": len(staged_sel),
                    "promote_fire_within_window": _pt(
                        staged_sel, "promote_within_window"),
                } if staged_sel else {"n": 0}
                delays = [x["promotion_delay_user_turns"] for x in sel
                          if x["promotion_delay_user_turns"] is not None]
                e["promotion_delay_user_turns"] = {
                    "n_fired": len(delays), "delays": sorted(delays),
                    "mean": (sum(delays) / len(delays)) if delays else None,
                    "max": max(delays) if delays else None}
            entry[cell] = e
        endpoints["promotion_by_distance"][lv] = entry
    # (4) task probes ITT — by outcome class, by level, pooled
    probes: dict = {"by_outcome": {}, "by_level": {}, "pooled": {}}
    for oc in OUTCOME_CLASSES:
        probes["by_outcome"][oc] = {
            cell: _pt(_sel(rows, cell, outcome=oc), "probe_correct")
            for cell in CELLS}
    for lv in G_LEVELS + P_LEVELS:
        probes["by_level"][lv] = {
            cell: _pt(_sel(rows, cell, lv), "probe_correct")
            for cell in CELLS}
    probes["pooled"] = {cell: _pt(_sel(rows, cell), "probe_correct")
                        for cell in CELLS}
    endpoints["task_probes_ITT"] = probes

    # hedge-turn funnel diagnostics (per family per ladder level; never a bar)
    hedge_diag: dict = {}
    for fam in ("em", "em_fixed", "ez"):
        hedge_diag[fam] = {}
        for lv in ("pooled",) + G_LEVELS + P_LEVELS:
            sel = [c for c in cases
                   if lv == "pooled" or c["unify_level"] == lv]
            drows = [{"cluster": c["cluster"],
                      "extracted": int(c["diagnostics"][fam]["hedge_turn"]
                                       ["target_extracted"]),
                      "mapped": int(c["diagnostics"][fam]["hedge_turn"]
                                    ["any_tuple_mapped_to_target"]),
                      "tuple_mapped": int(
                          c["diagnostics"][fam]["hedge_turn"]
                          ["target_mapped_via_matched_tuple"])}
                     for c in sel if fam in c.get("diagnostics", {})]
            hedge_diag[fam][lv] = {
                "target_extracted": _pt(
                    [{"cluster": d["cluster"], "f": d["extracted"]}
                     for d in drows], "f"),
                "any_tuple_mapped": _pt(
                    [{"cluster": d["cluster"], "f": d["mapped"]}
                     for d in drows], "f"),
                "hedge_tuple_mapped": _pt(
                    [{"cluster": d["cluster"], "f": d["tuple_mapped"]}
                     for d in drows], "f")}

    comparison = planted_comparison(rows)

    level_counts = {lv: sum(c["unify_level"] == lv for c in cases)
                    for lv in G_LEVELS + P_LEVELS}
    phases = ([json.loads(l) for l in RUN_LOG.open()]
              if RUN_LOG.exists() else [])
    return {
        "id": "UNIFY-LADDER-RESULTS",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prereg": ("protocol/CHANGELOG.md 2026-08-11 'Unified-substrate "
                   "ladders ACCEPTED by owner; unified-ladder run "
                   "PREREGISTERED'"),
        "system_model": llm.MODEL,
        "judge_model": llm.JUDGE_MODEL,
        "n_arcs_scored": len(cases),
        "partial": len(cases) < 179,
        "level_counts": level_counts,
        "arcs_sha256": next(iter(digests)),
        "excluded": sorted(excluded),
        "arms": list(CELLS),
        "em_cg_repair": EM_CG_REPAIR_NOTE,
        "ci_note": ("Wilson CIs; cluster = conversation (one row per "
                    "conversation per cell per endpoint — the scorer_d1 "
                    "aggregation pattern, per the prereg)"),
        "endpoint_note": ("directional curves only — no go/no-go bars; the "
                          "dose-response and wording-distance curves REPLACE "
                          "the probe-based Result 3 curves as primary (prereg "
                          "REPORTING clause)"),
        "endpoints": endpoints,
        "hedge_turn_diagnostics": hedge_diag,
        "planted_comparison": comparison,
        "per_arc_rows": rows,
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

def _fmt(pt: dict | None) -> str:
    if not pt or pt.get("rate") is None:
        return "—"
    return (f"{pt['x']}/{pt['n']} = {pt['rate']:.3f} "
            f"[{pt['wilson_lo']:.3f}, {pt['wilson_hi']:.3f}]")


def _fmt_q(pt: dict | None) -> str:
    if not pt or pt.get("rate") is None:
        return "—"
    return f"{pt['x']}/{pt['n']} = {pt['rate']:.3f}"


def render_report(res: dict) -> str:
    ep = res["endpoints"]
    cmp_ = res["planted_comparison"]
    lines = [
        "# Unified-ladder run — both graded-difficulty factor ladders on the "
        "conversation substrate (four frozen systems, pinned gpt-5.5)",
        "",
        f"Preregistered 2026-08-11 (protocol/CHANGELOG.md). Generated "
        f"{res['generated_at']}. Scored arcs: {res['n_arcs_scored']}/179"
        + (" (PARTIAL)" if res["partial"] else "")
        + f" (180 generated, {len(res['excluded'])} excluded: "
        f"{', '.join(res['excluded'])}); arcs.jsonl sha256 "
        f"{res['arcs_sha256'][:16]}…. Systems: the four frozen cells em/ez x "
        f"ungated(cw)/gated(cg) under pinned `{res['system_model']}`; em_cg "
        "IS the adopted repaired configuration (M1 plan-stem wrapper as "
        "LOCAL mapper — its released planted analog is em_cg_fixed). Judge "
        f"= frozen `{res['judge_model']}`. Wilson CIs, cluster = "
        "conversation. n = 20 per level cell (19 for g5) — small-n CIs "
        "expected.",
        "",
        "## Headline table 1 — contamination by hedge-strength level "
        "(g1-g4 still_open; final-session opener is a neutral still-pending "
        "nod)",
        "",
        "At-end = hedged VALUE current in the FINAL store (the planted "
        "suite's contamination endpoint). While-pending = VALUE current at "
        "any user turn strictly after the change-statement turn through the "
        "last user turn (D1 E1 window semantics). g5 = commit-correct "
        "CONTROL: the change is already completed, a commit is CORRECT — "
        "never contamination.",
        "",
        "### At-end", "",
        "| level | " + " | ".join(CELLS) + " |",
        "|---|" + "---|" * len(CELLS)]
    for lv in ("g1_strong_hedge", "g2_weak_hedge", "g3_confident_plan",
               "g4_scheduled_event"):
        cells = ep["contamination_by_level"][lv]
        lines.append(f"| {lv} | " + " | ".join(
            _fmt(cells[c]["at_end"]) for c in CELLS) + " |")
    g5 = ep["g5_commit_correct_control"]
    lines.append("| g5_completed_control (commit-correct rate — higher is "
                 "BETTER) | " + " | ".join(
                     _fmt(g5[c]["commit_correct_at_end"]) for c in CELLS) + " |")
    lines += ["", "### While-pending", "",
              "| level | " + " | ".join(CELLS) + " |",
              "|---|" + "---|" * len(CELLS)]
    for lv in ("g1_strong_hedge", "g2_weak_hedge", "g3_confident_plan",
               "g4_scheduled_event"):
        cells = ep["contamination_by_level"][lv]
        lines.append(f"| {lv} | " + " | ".join(
            _fmt(cells[c]["while_pending"]) for c in CELLS) + " |")
    lines += [
        "",
        "## Headline table 2 — promotion by confirmation-wording distance "
        "(came_true; gated cells; valid promote fire at the resolution turn)",
        "",
        "| distance | em_cg t* | em_cg window [t*,t*+2] | ez_cg t* | "
        "ez_cg window [t*,t*+2] | em_cg end-state | ez_cg end-state |",
        "|---|---|---|---|---|---|---|"]
    for lv in P_LEVELS:
        e = ep["promotion_by_distance"][lv]
        lines.append(
            f"| {lv} "
            f"| {_fmt(e['em_cg']['promote_fire_at_tstar_ITT'])} "
            f"| {_fmt(e['em_cg']['promote_fire_within_window_ITT'])} "
            f"| {_fmt(e['ez_cg']['promote_fire_at_tstar_ITT'])} "
            f"| {_fmt(e['ez_cg']['promote_fire_within_window_ITT'])} "
            f"| {_fmt(e['em_cg']['end_state_correct_ITT'])} "
            f"| {_fmt(e['ez_cg']['end_state_correct_ITT'])} |")
    lines += ["", "End-state correctness (final store = VALUE) for the "
              "ungated cells, by distance:", "",
              "| distance | em_cw | ez_cw |", "|---|---|---|"]
    for lv in P_LEVELS:
        e = ep["promotion_by_distance"][lv]
        lines.append(f"| {lv} | {_fmt(e['em_cw']['end_state_correct_ITT'])} "
                     f"| {_fmt(e['ez_cw']['end_state_correct_ITT'])} |")
    lines += ["", "## Staging at the change-statement beat (gated cells)", "",
              "| level | em_cg | ez_cg |", "|---|---|---|"]
    for lv in G_LEVELS + P_LEVELS:
        s = ep["staging_by_level"][lv]
        lines.append(f"| {lv} | {_fmt(s['em_cg'])} | {_fmt(s['ez_cg'])} |")
    lines += ["", "## Task probes (ITT; deterministic final-store lookup, "
              "unanswered = incorrect)", "",
              "Expected by outcome class: still_open -> the established "
              "prior value; came_true -> VALUE; already_completed (g5) -> "
              "the completed value.", "",
              "| stratum | " + " | ".join(CELLS) + " |",
              "|---|" + "---|" * len(CELLS)]
    for oc in OUTCOME_CLASSES:
        p = ep["task_probes_ITT"]["by_outcome"][oc]
        lines.append(f"| {oc} | " + " | ".join(_fmt(p[c]) for c in CELLS) + " |")
    p = ep["task_probes_ITT"]["pooled"]
    lines.append("| pooled | " + " | ".join(_fmt(p[c]) for c in CELLS) + " |")
    lines += ["", "Per-level probe rows live in unify_results.json "
              "(task_probes_ITT.by_level).",
              "",
              "## Comparison block — released planted-probe numbers "
              "(quoted, never rerun) + directional scorecard",
              "",
              f"Source: `{cmp_['released_source']}`. "
              f"{cmp_['cell_mapping_note']} "
              f"{cmp_['contamination_analog_note']}.",
              "",
              "### Contamination at end, unify vs planted (per level per "
              "cell; planted pooled lme+locomo)", "",
              "| level | " + " | ".join(
                  f"{c} (unify / planted {M2_CELL_OF[c]})" for c in CELLS)
              + " |",
              "|---|" + "---|" * len(CELLS)]
    for lv in G_LEVELS:
        q = cmp_["quotes"]["contamination_at_end_by_level"][lv]
        lines.append(
            f"| {lv}{' (commit-correct)' if lv == 'g5_completed_control' else ''} | "
            + " | ".join(f"{_fmt_q(q[c]['unify'])} / "
                         f"{_fmt_q(q[c]['released_planted'])}"
                         for c in CELLS) + " |")
    lines += ["", "(planted g5 row quotes the released g5 commit rate — "
              "commit-is-CORRECT semantics on both substrates)", "",
              "### Promotion (window ITT), unify vs planted", "",
              "| distance | em_cg (unify / planted em_cg_fixed) | "
              "ez_cg (unify / planted ez_cg) |", "|---|---|---|"]
    for lv in P_LEVELS:
        q = cmp_["quotes"]["promotion_by_distance"][lv]
        lines.append(
            f"| {lv} | {_fmt_q(q['em_cg']['unify_window_ITT'])} / "
            f"{_fmt_q(q['em_cg']['released_planted_window_ITT'])} "
            f"| {_fmt_q(q['ez_cg']['unify_window_ITT'])} / "
            f"{_fmt_q(q['ez_cg']['released_planted_window_ITT'])} |")
    lines += ["", "### Directional scorecard (prereg expectation: onset at "
              "g3, recovery at g4, promotion declining with wording "
              "distance)", "",
              "| row | expectation | unify observed | unify | planted "
              "observed | planted | matches |",
              "|---|---|---|---|---|---|---|"]

    def v(h):
        return "—" if h is None else ("HOLDS" if h else "does not hold")

    for row in cmp_["directional_scorecard"]:
        m = row["matches_released"]
        lines.append(
            f"| {row['row']} | {row['expectation']} "
            f"| {row['unify_observed']} | **{v(row['unify_holds'])}** "
            f"| {row['released_planted_observed']} "
            f"| {v(row['released_planted_holds'])} "
            f"| {'—' if m is None else ('YES' if m else 'NO')} |")
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
    lines += ["", "## Budget (stage unify; advisory / hard envelope / 110k "
              "run stop)", ""]
    if bb:
        lines.append(f"- before: {json.dumps(bb['spent'])} "
                     f"(total {sum(bb['spent'].values()):,})")
    lines += [f"- after:  {json.dumps(ba['spent'])} "
              f"(total {sum(ba['spent'].values()):,} / envelope "
              f"{ba['envelope']:,}; run stop {HARD_STOP_TOTAL:,})",
              f"- stage unify spend {adv['spent']:,} vs advisory forecast "
              f"{adv['advisory_forecast']:,} (over: {adv['over_forecast']})",
              f"- scoring equivalence-judge calls: {jc['total']} "
              f"({jc['fresh']} fresh, stage unify)",
              "",
              "Scored once (prereg). Verification (RECOUNT + SEMANTICS) "
              "precedes release — these numbers are UNVERIFIED until then.",
              ""]
    return "\n".join(lines)


async def score() -> dict:
    assert llm.JUDGE_MODEL == PINNED_SYSTEM_MODEL
    guard = unify_guard()
    guard.install()
    try:
        res = await compute_scores(CASES_DIR)
    finally:
        guard.uninstall()
    res["score_fresh_calls"] = guard.fresh
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    REPORT_PATH.write_text(render_report(res))
    print(f"wrote {_rel(RESULTS_PATH)}\nwrote {_rel(REPORT_PATH)} "
          f"(n_arcs={res['n_arcs_scored']}"
          f"{' PARTIAL' if res['partial'] else ''}; "
          f"{guard.fresh} fresh judge calls)")
    return res


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    runp = sub.add_parser("run")
    runp.add_argument("--ids", nargs="*", default=None)
    runp.add_argument("--shard", default=None, help="K/N over sorted ids")
    runp.add_argument("--call-cap", type=int, default=None)
    sub.add_parser("score")
    sub.add_parser("status")
    args = ap.parse_args()

    if args.cmd == "status":
        n_cases = len(list(CASES_DIR.glob("*.json"))) if CASES_DIR.exists() else 0
        print(json.dumps({**advisory_status(), "case_records": n_cases},
                         indent=1))
        return
    if args.cmd == "score":
        asyncio.run(score())
        return
    summary = asyncio.run(run(args.ids, args.shard, args.call_cap))
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
