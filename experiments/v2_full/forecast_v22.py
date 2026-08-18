"""forecast_v22.py — v2.2 COMBINED C1+C2 fresh-call forecast + shared budget.

Closes the two BLOCKING findings of research/impl_review_v22.md under the
orchestrator-pinned resolution (logged as a v2.2 amendment note in
research/protocol_changelog.md, 2026-07-06, appended pre-run):

B-1  The preregistered <= 40,000 fresh-call envelope is C1+C2 COMBINED, but
     the C2 forecaster (router_sweep.py forecast) granted itself the full
     envelope and the C1 cell runs (expand_natural.py jobs --run) were
     unforecast and unguarded. This module produces the ONE combined forecast
     that drives the preregistered de-scope, and provides the shared runtime
     guard (SharedCallBudget over results/v2.2/budget.json) that BOTH run
     paths — expand_natural.py jobs --run AND router_sweep.py run — decrement;
     each fails loudly when the envelope is exhausted.

B-2  The amendment's "Target n >= 80" is arithmetically unreachable inside
     its own envelope under the realized freeze-list geometry (the same error
     class as pilot critique S1: a target and a budget preregistered jointly
     without checking joint feasibility). Pinned rule, encoded here: apply
     the preregistered de-scope order to C2 (drop the s_ev {0.3, 0.7} points;
     then drop the LoCoMo gated-column re-runs of non-argmax non-fix points —
     LME kept everywhere; the fix, oracle and confidence points are never
     dropped from the plan; argmax keeps LoCoMo as the zero-fresh-cost
     regression reference and the fix keeps LoCoMo as the both-benchmark
     readout); after de-scope, ALL remaining envelope headroom goes to C1
     source-question additions selected cheapest-first from the freeze
     list's expected-computable (`if_source_question_runs`) rows — the frozen
     greedy rule: ascending (cost, question_id), stop at the first question
     that does not fit. The achieved n is reported with the shortfall vs 80
     stated explicitly, never silently reinterpreted.

Forecast model (FM-20 style: realized artifacts only, zero LLM calls)
---------------------------------------------------------------------

C2 per point x benchmark: router_sweep.compute_point_forecast — the C2
implementer's own offline projection over the frozen v2 gate logs
(generator + checker + judging), unchanged.

C1 per question q (all four cells via arms.run_cells; T = user turns of the
s_cleaned haystack):

  extraction   2 x T fresh calls (1 per user turn per family; diagnostics of
               every frozen timeline show n_extraction_calls == n_user_turns);
               0 when q is one of the 22 questions in the frozen v2 s_cleaned
               extraction files (byte-identical replay = llm cache hit —
               0/17 pair substrates, 13/121 addition candidates).
  key mapping  one store._map_tuple_key call per DISTINCT (entity, attribute,
               value) triple: the mapper prompt is a pure function of
               (tuple, candidate list), so duplicate triples are llm-cache
               hits ACROSS policies and families (the candidate list is
               per-substrate and family-independent). Covered questions use
               their exact realized distinct-triple count; uncovered use the
               measured corpus rate over the 22 frozen streams.
  classifier   gated cells only: mapped-tuple rate per turn measured on the
               NATURAL (no construction-label join) gate rows of the frozen
               v2 LME gated cells, per family. Conservative: the v2.2
               candidate lists are 1-3 keys vs the frozen v2 lists' larger
               gold+aux pools.
  predicates   one generation call per staged item per gated cell: 2 x hosted
               freeze rows (ITT-conservative: every hosted hedge/pair row
               stages) + naturally staged tuples at the measured rate.
  checker      one call per user turn per store while the pending set is
               non-empty (LME replays are single-store): per family
               min(T, W(q) + ceil(stage_rate x T^2 / 2)) where W(q) is the
               union length of the hosted rows' pending windows (pair rows:
               anchor -> anchor + realized checker_horizon_user_turns from
               the frozen schema; hedge rows have no resolution turn ->
               pending to stream end) and the quadratic term is the expected
               coverage of naturally staged tuples (staging uniform over the
               stream, pending to stream end — v2 realized n_pending_final
               ~= n_staged).
  rule-5 judge JUDGE_PER_ROW = 4 cells x run_v2.JUDGE_UNRESOLVED_FRACTION
               (0.4) = 1.6 fresh equivalence-judge calls per computable
               estimand row (realized old-block rate: 28/20 = 1.4).

Outputs
-------
  results/v2.2/forecast_v22.json   full combined forecast + de-scope + greedy
  results/v2.2/budget.json         the SHARED envelope file: per-stage
                                   allocation, planned n (+ shortfall vs 80),
                                   de-scope steps, and the mutable `spent`
                                   counters that BOTH run paths decrement
                                   through SharedCallBudget.

CLI (from pilot/, .venv/bin/python):
  python forecast_v22.py forecast [--envelope 40000]
  python forecast_v22.py status
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import sys
from functools import cache
from pathlib import Path

import _bootstrap  # noqa: F401  (R1 layout: puts src/ + experiments/ on sys.path)

import expand_natural as en
import router_sweep
import driver as run_v2
import eval_classifier as test1_eval

from paths import ROOT, RESULTS_DIR

V22_DIR = RESULTS_DIR / "v2.2"
FORECAST_PATH = V22_DIR / "forecast_v22.json"
BUDGET_PATH = V22_DIR / "budget.json"

ENVELOPE = 40_000          # preregistered v2.2 C1+C2 COMBINED envelope
TARGET_N = 80              # the amendment's (infeasible, B-2) target
OLD_STRATUM_N = 20         # frozen v2 rule-5 stratum size
JUDGE_PER_ROW = 4 * run_v2.JUDGE_UNRESOLVED_FRACTION   # 1.6 (§7 row 12 anchor)

#: C2 points whose LoCoMo halves SURVIVE de-scope step 2 (pinned B-2 rule):
#: argmax is the zero-fresh-cost regression reference; plan_frame_fix is THE
#: fix readout, kept on both benchmarks.
KEEP_LOCOMO = ("argmax", "plan_frame_fix")
#: points never dropped from the plan (preregistered "never the fix or oracle
#: points"; the pinned amendment adds confidence).
NEVER_DROPPED_POINTS = ("plan_frame_fix", "oracle", "confidence")

#: the run-path stage keys of budget.json `spent` (shared-guard accounting).
STAGES = ("c1_cell_runs", "c1_rule5_judging", "c2_router_sweep")


# ---------------------------------------------------------------------------
# Measured realized rates (zero LLM; loaders cached)
# ---------------------------------------------------------------------------

@cache
def s_cleaned_realized() -> dict:
    """Realized geometry of the 22 frozen v2 s_cleaned extraction streams:
    per-question user-turn counts, DISTINCT (entity, attribute, value) triple
    counts (mapper llm-cache units, deduplicated across BOTH families — the
    mapper prompt is family-independent), and the corpus mapping rate."""
    turns: dict[str, int] = {}
    uniq: dict[str, set] = {}
    for family in ("mem0", "zep"):
        path = run_v2.extraction_path("lme", family, "s_cleaned")
        if not path.exists():
            raise SystemExit(f"missing frozen v2 s_cleaned extraction: {path}")
        with path.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                uniq.setdefault(r["question_id"], set()).update(
                    (t["entity"], t["attribute"], t["value"])
                    for t in r["tuples"])
                if family == "mem0":
                    turns[r["question_id"]] = turns.get(r["question_id"], 0) + 1
    total_turns = sum(turns.values())
    total_uniq = sum(len(v) for v in uniq.values())
    return {
        "turns": turns,
        "distinct_triples": {q: len(v) for q, v in uniq.items()},
        "n_questions": len(turns),
        "total_turns": total_turns,
        "total_distinct_triples": total_uniq,
        "map_rate": total_uniq / total_turns,
    }


@cache
def lme_natural_rates() -> dict:
    """Per-family NATURAL-row rates measured on the frozen v2 LME gated
    cells: gate rows with no construction-label join (= natural tuples that
    mapped to a candidate key and reached the classifier) per user turn, and
    the naturally-STAGED subset per user turn."""
    out: dict = {}
    for family in router_sweep.FAMILIES:
        turns = nat = staged = 0
        for stream_id in router_sweep.frozen_stream_ids(family):
            if router_sweep.benchmark_of_stream(stream_id) != "lme":
                continue
            stream = router_sweep.enriched_stream_rows(family, stream_id)
            turns += stream["n_user_turns"]
            for row in stream["rows"]:
                if row["construction_label"] is None:
                    nat += 1
                    staged += (row["frozen_route"] == "stage")
        out[family] = {
            "n_user_turns": turns,
            "n_natural_rows": nat,
            "n_natural_staged": staged,
            "classify_rate": nat / turns,
            "stage_rate": staged / turns,
        }
    return out


@cache
def pair_horizons() -> dict[str, int]:
    """question_id -> realized checker_horizon_user_turns (stage->resolution
    + 2-turn window) from the frozen natural_pair_keys schema entries."""
    schema = json.loads(en.LME_SCHEMA_PATH.read_text())
    return {e["question_id"]: e["checker_horizon_user_turns"]
            for e in schema["natural_pair_keys"]["entries"]}


# ---------------------------------------------------------------------------
# Pure forecast arithmetic (unit-tested)
# ---------------------------------------------------------------------------

def interval_union_length(intervals: list[tuple[int, int]]) -> int:
    """Total length of the union of half-open [start, end) intervals."""
    total = 0
    cur_s = cur_e = None
    for s, e in sorted(intervals):
        if e <= s:
            continue
        if cur_s is None:
            cur_s, cur_e = s, e
        elif s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    if cur_s is not None:
        total += cur_e - cur_s
    return total


def question_cost(*, n_user_turns: int, distinct_triples: int | None,
                  extraction_cached: bool, windows: list[tuple[int, int]],
                  n_rows: int, map_rate: float, rates: dict,
                  include_judge: bool) -> dict:
    """Projected fresh calls for ONE C1 substrate through all four cells.
    Pure arithmetic over the module-docstring model; every input is a
    realized quantity. ``rates`` is lme_natural_rates()-shaped."""
    T = n_user_turns
    extraction = 0 if extraction_cached else 2 * T
    mapping = (distinct_triples if distinct_triples is not None
               else math.ceil(T * map_rate))
    classifier = sum(math.ceil(T * rates[f]["classify_rate"]) for f in rates)
    W = interval_union_length(windows)
    checker = sum(
        min(T, W + math.ceil(rates[f]["stage_rate"] * T * T / 2))
        for f in rates)
    predicates = 2 * n_rows + math.ceil(
        T * sum(rates[f]["stage_rate"] for f in rates))
    judge = math.ceil(JUDGE_PER_ROW * n_rows) if include_judge else 0
    return {
        "n_user_turns": T,
        "extraction": extraction,
        "mapping": mapping,
        "classifier": classifier,
        "checker": checker,
        "predicates": predicates,
        "judge": judge,
        "cells_total": extraction + mapping + classifier + checker + predicates,
        "total": (extraction + mapping + classifier + checker + predicates
                  + judge),
    }


def apply_combined_descope(c1_scheduled_total: int, c2_point_costs: dict,
                           envelope: int) -> dict:
    """The pinned COMBINED de-scope (B-1/B-2): the trigger total is C1
    scheduled + C2, and the preregistered order acts on C2 only.

      step 1  drop the s_ev threshold points {0.3, 0.7} entirely;
      step 2  drop the LoCoMo halves of every remaining point EXCEPT
              KEEP_LOCOMO (argmax: zero-fresh-cost regression reference;
              plan_frame_fix: the both-benchmark fix readout) — LME kept
              everywhere; NEVER_DROPPED_POINTS survive as points.

    Pure arithmetic; returns the scoped plan + step log + verdict."""
    assert not set(router_sweep.DESCOPE_DROP_FIRST) & set(NEVER_DROPPED_POINTS)
    scope = {p: [b for b in router_sweep.BENCHMARKS if b in c2_point_costs[p]]
             for p in c2_point_costs}
    steps: list[dict] = []

    def c2_total() -> int:
        return sum(c2_point_costs[p][b] for p, bs in scope.items() for b in bs)

    def total() -> int:
        return c1_scheduled_total + c2_total()

    if total() > envelope:
        dropped = [p for p in router_sweep.DESCOPE_DROP_FIRST if p in scope]
        for p in dropped:
            del scope[p]
        steps.append({"step": "drop_sev_threshold_points",
                      "dropped_points": dropped,
                      "combined_total_after": total()})
    if total() > envelope:
        restricted = [p for p, bs in scope.items()
                      if "locomo" in bs and p not in KEEP_LOCOMO]
        for p in restricted:
            scope[p] = [b for b in scope[p] if b != "locomo"]
        steps.append({"step": "drop_locomo_non_argmax_non_fix",
                      "restricted_points": restricted,
                      "locomo_kept_for": [p for p in scope
                                          if "locomo" in scope[p]],
                      "combined_total_after": total()})
    for p in NEVER_DROPPED_POINTS:
        assert p in scope and "lme" in scope[p], \
            f"de-scope must never drop the {p} point (v2.2 C2 budget rule)"
    return {
        "envelope": envelope,
        "c1_scheduled_total": c1_scheduled_total,
        "c2_points": {p: list(bs) for p, bs in scope.items()},
        "c2_total": c2_total(),
        "steps": steps,
        "combined_total": total(),
        "over_envelope": total() > envelope,
        "rule": ("pinned v2.2 combined de-scope (B-1/B-2 amendment note): "
                 "trigger on C1 scheduled + C2 vs the combined envelope; "
                 "drop s_ev {0.3, 0.7} first, then the LoCoMo halves of "
                 "non-argmax non-fix points (LME kept everywhere; fix, "
                 "oracle and confidence never dropped as points)"),
    }


def greedy_additions(candidates: list[dict], headroom: int) -> tuple[list[dict], int]:
    """The frozen greedy rule (pinned B-2): candidates sorted ascending by
    (total cost, question_id); take while the next question fits the
    remaining headroom; STOP at the first that does not fit (ascending cost —
    nothing later fits either). Returns (selected, headroom_left)."""
    remaining = headroom
    selected: list[dict] = []
    for cand in sorted(candidates, key=lambda c: (c["cost"]["total"],
                                                  c["question_id"])):
        if cand["cost"]["total"] > remaining:
            break
        selected.append(cand)
        remaining -= cand["cost"]["total"]
    return selected, remaining


# ---------------------------------------------------------------------------
# C1 geometry from the frozen freeze list
# ---------------------------------------------------------------------------

def _user_pos(substrate, session_idx: int, turn_idx: int) -> int:
    """0-based user-turn position of an anchor turn within a substrate."""
    pos = 0
    for si, session in enumerate(substrate.sessions):
        for ti, turn in enumerate(session):
            if turn["role"] != "user":
                continue
            if (si, ti) == (session_idx, turn_idx):
                return pos
            pos += 1
    raise AssertionError(f"anchor ({session_idx}, {turn_idx}) is not a user "
                         "turn of the substrate")


def substrate_user_turns(qid: str) -> int:
    substrate = test1_eval.s_cleaned_substrate(qid)
    return sum(1 for s in substrate.sessions for t in s if t["role"] == "user")


def c1_question_geometry(qid: str, freeze_rows: list[dict],
                         host_kinds: tuple[str, ...]) -> tuple[int, list, list]:
    """(n_user_turns, checker windows, hosted row_ids) of one v2.2 run
    substrate. Windows are user-pos intervals during which a hosted row keeps
    the store's pending set non-empty: pair rows resolve after their realized
    checker horizon; hedge rows have no resolution turn -> stream end."""
    substrate = test1_eval.s_cleaned_substrate(qid)
    T = sum(1 for s in substrate.sessions for t in s if t["role"] == "user")
    covered = s_cleaned_realized()["turns"].get(qid)
    if covered is not None:
        assert covered == T, (
            f"{qid}: frozen extraction has {covered} user turns but the "
            f"substrate walk found {T} — realized-artifact mismatch")
    windows: list[tuple[int, int]] = []
    row_ids: list[str] = []
    for fr in freeze_rows:
        for host in fr["hosts"]:
            if host["kind"] not in host_kinds or host["question_id"] != qid:
                continue
            pos = _user_pos(substrate, host["session_idx"], host["turn_idx"])
            if fr["row_kind"] == "lme_pair_stage":
                end = min(T, pos + 1 + pair_horizons()[qid])
            else:
                end = T
            windows.append((pos + 1, end))
            row_ids.append(fr["row_id"])
    return T, windows, sorted(set(row_ids))


def _cost_for(qid: str, freeze_rows: list[dict], host_kinds: tuple[str, ...],
              include_judge: bool) -> tuple[dict, list[str]]:
    T, windows, row_ids = c1_question_geometry(qid, freeze_rows, host_kinds)
    realized = s_cleaned_realized()
    cost = question_cost(
        n_user_turns=T,
        distinct_triples=realized["distinct_triples"].get(qid),
        extraction_cached=qid in realized["turns"],
        windows=windows,
        n_rows=len(row_ids),
        map_rate=realized["map_rate"],
        rates=lme_natural_rates(),
        include_judge=include_judge,
    )
    return cost, row_ids


def c1_scheduled_forecast(freeze_rows: list[dict]) -> dict:
    """The C1(a) scheduled plan: the 17 LME pair substrates through all four
    cells, plus rule-5 judging for every scheduled-computable row (the
    cross-hosted hedges ride the same runs; the one existing-v2-timelines row
    is judge-only)."""
    per_question: dict[str, dict] = {}
    cells_total = 0
    for qid in sorted(en.lme_pair_qids()):
        cost, row_ids = _cost_for(qid, freeze_rows, ("v22_pair_run",),
                                  include_judge=False)
        per_question[qid] = {**cost, "hosted_rows": row_ids}
        cells_total += cost["cells_total"]
    scheduled_rows = [fr["row_id"] for fr in freeze_rows
                      if fr["expected_computable"] in
                      ("scheduled_v22_pair_run", "existing_v2_timelines")]
    judge = math.ceil(JUDGE_PER_ROW * len(scheduled_rows))
    return {
        "question_ids": sorted(en.lme_pair_qids()),
        "per_question": per_question,
        "cells_total": cells_total,
        "computable_rows": sorted(scheduled_rows),
        "n_computable_rows": len(scheduled_rows),
        "judge_total": judge,
        "total": cells_total + judge,
    }


def c1_addition_candidates(freeze_rows: list[dict]) -> list[dict]:
    """Every `if_source_question_runs` source question with its marginal cost
    (all four cells + rule-5 judging of its hosted rows), the greedy's input."""
    by_q: dict[str, list[dict]] = {}
    for fr in freeze_rows:
        if fr["expected_computable"] == "if_source_question_runs":
            by_q.setdefault(fr["source_question_id"], []).append(fr)
    out = []
    for qid in sorted(by_q):
        cost, row_ids = _cost_for(qid, by_q[qid], ("v22_expansion_run",),
                                  include_judge=True)
        assert len(row_ids) == len(by_q[qid])
        out.append({"question_id": qid, "rows": row_ids, "cost": cost})
    return out


# ---------------------------------------------------------------------------
# The combined forecast
# ---------------------------------------------------------------------------

def build_forecast(envelope: int = ENVELOPE) -> dict:
    freeze_rows = en.load_freeze_rows()

    c2_per_point = {p: router_sweep.compute_point_forecast(p)
                    for p in router_sweep.POINTS}
    c2_costs = {p: {b: c2_per_point[p][b]["total"]
                    for b in router_sweep.BENCHMARKS}
                for p in router_sweep.POINTS}
    c2_full_total = sum(c2_costs[p][b] for p in c2_costs for b in c2_costs[p])

    c1 = c1_scheduled_forecast(freeze_rows)
    plan = apply_combined_descope(c1["total"], c2_costs, envelope)

    headroom = envelope - plan["combined_total"]
    candidates = c1_addition_candidates(freeze_rows)
    if plan["over_envelope"]:
        additions, headroom_left = [], headroom
    else:
        additions, headroom_left = greedy_additions(candidates, headroom)
    additions_total = sum(a["cost"]["total"] for a in additions)
    addition_rows = sum(len(a["rows"]) for a in additions)

    unaffordable = [c for c in sorted(candidates,
                                      key=lambda c: (c["cost"]["total"],
                                                     c["question_id"]))
                    if c not in additions]
    cheapest_unaffordable = (
        {"question_id": unaffordable[0]["question_id"],
         "cost": unaffordable[0]["cost"]["total"],
         "rows": len(unaffordable[0]["rows"])}
        if unaffordable else None)

    planned_n = OLD_STRATUM_N + c1["n_computable_rows"] + addition_rows
    realized = s_cleaned_realized()
    rates = lme_natural_rates()
    return {
        "id": "V2.2-COMBINED-FORECAST",
        "endpoint": ("v2.2 COMBINED C1+C2 fresh-call forecast (FM-20 style; "
                     "review B-1/B-2 fix, orchestrator-pinned resolution)"),
        "envelope": envelope,
        "notes": [
            "ONE combined forecast drives the preregistered de-scope (B-1); "
            "projection from realized frozen artifacts only, zero LLM calls",
            "C2 costs are router_sweep.compute_point_forecast unchanged; "
            "C1 costs follow the module-docstring model (extraction 1 call/"
            "user turn/family; mapping 1 call/distinct triple, cache-shared "
            "across policies AND families; classifier/staging at the "
            "measured natural rates of the frozen v2 LME gated cells; "
            "checker = hosted-row pending windows + expected natural-staging "
            "coverage; rule-5 judging at 1.6/computable row)",
            "runtime enforcement is SharedCallBudget over "
            "results/v2.2/budget.json — the forecast is the plan, the guard "
            "is the envelope",
        ],
        "measured_rates": {
            "map_rate_distinct_triples_per_turn": realized["map_rate"],
            "s_cleaned_covered_questions": realized["n_questions"],
            "s_cleaned_total_turns": realized["total_turns"],
            "s_cleaned_total_distinct_triples":
                realized["total_distinct_triples"],
            "lme_natural": rates,
            "judge_per_row": JUDGE_PER_ROW,
        },
        "c1": {
            "scheduled": c1,
            "additions": {
                "candidates_considered": len(candidates),
                "selected": [{"question_id": a["question_id"],
                              "rows": a["rows"],
                              "cost": a["cost"]} for a in additions],
                "selected_total": additions_total,
                "addition_rows": addition_rows,
                "cheapest_unaffordable": cheapest_unaffordable,
                "greedy_rule": ("frozen: ascending (cost, question_id) over "
                                "the freeze list's if_source_question_runs "
                                "source questions; stop at the first that "
                                "does not fit the remaining headroom"),
            },
        },
        "c2": {"per_point": c2_per_point, "point_totals": c2_costs,
               "full_total": c2_full_total},
        "descope": plan,
        "totals": {
            "combined_before_descope": c1["total"] + c2_full_total,
            "combined_after_descope": plan["combined_total"],
            "headroom_after_descope": headroom,
            "combined_planned": plan["combined_total"] + additions_total,
            "headroom_final": headroom_left,
        },
        "planned_n": {
            "old_stratum": OLD_STRATUM_N,
            "scheduled_new_rows": c1["n_computable_rows"],
            "addition_rows": addition_rows,
            "total": planned_n,
            "target": TARGET_N,
            "target_met": planned_n >= TARGET_N,
            "shortfall": max(0, TARGET_N - planned_n),
        },
        "over_envelope": plan["over_envelope"],
    }


# ---------------------------------------------------------------------------
# budget.json — the shared envelope file
# ---------------------------------------------------------------------------

def budget_from_forecast(fc: dict, prior_spent: dict | None = None) -> dict:
    spent = {stage: 0 for stage in STAGES}
    if prior_spent:
        for stage, n in prior_spent.items():
            if stage in spent:
                spent[stage] = n
    c1 = fc["c1"]
    additions = c1["additions"]["selected"]
    alloc_cells = (c1["scheduled"]["cells_total"]
                   + sum(a["cost"]["cells_total"] for a in additions))
    alloc_judge = (c1["scheduled"]["judge_total"]
                   + sum(a["cost"]["judge"] for a in additions))
    alloc_c2 = fc["descope"]["c2_total"]
    return {
        "id": "V2.2-COMBINED-BUDGET",
        "envelope": fc["envelope"],
        "forecast": str(FORECAST_PATH.relative_to(ROOT)),
        "over_envelope": fc["over_envelope"],
        "descope_steps": fc["descope"]["steps"],
        "planned_n": fc["planned_n"],
        "allocation": {
            "c1_cell_runs": alloc_cells,
            "c1_rule5_judging": alloc_judge,
            "c2_router_sweep": alloc_c2,
            "unallocated_headroom": fc["totals"]["headroom_final"],
        },
        "c1": {
            "scheduled_question_ids": c1["scheduled"]["question_ids"],
            "additions": [{"question_id": a["question_id"],
                           "rows": a["rows"],
                           "cost": a["cost"]["total"]} for a in additions],
        },
        "c2": {"points": fc["descope"]["c2_points"]},
        "spent": spent,
        "spent_note": ("FRESH llm calls (cache hits free, run_v2.CallBudget "
                       "semantics) recorded per call by "
                       "forecast_v22.SharedCallBudget; the combined envelope "
                       "is enforced over sum(spent.values())"),
    }


def load_budget(path: Path | None = None) -> dict:
    path = BUDGET_PATH if path is None else path
    if not path.exists():
        raise SystemExit(
            f"missing v2.2 combined budget file {path} — run "
            "`forecast_v22.py forecast` first (B-1 gate: no C1 cell run and "
            "no C2 sweep without the COMBINED in-envelope forecast)")
    return json.loads(path.read_text())


def budget_remaining(budget: dict) -> int:
    return budget["envelope"] - sum(budget["spent"].values())


class SharedCallBudget:
    """The B-1 runtime guard: ONE budget file (results/v2.2/budget.json) that
    BOTH the C1 jobs-run path and the C2 sweep path decrement.

    Wraps run_v2.CallBudget (fresh = llm-cache misses only) with an on_fresh
    hook that persists EVERY fresh call to the budget file immediately
    (crash-safe accounting) and aborts — loudly — the moment the COMBINED
    envelope is exhausted. Refuses to install at all when the plan is
    over-envelope or the envelope is already spent."""

    def __init__(self, stage: str, *, path: Path | None = None,
                 local_cap: int | None = None):
        self.stage = stage
        self.path = BUDGET_PATH if path is None else path
        budget = load_budget(self.path)
        if stage not in budget["spent"]:
            raise SystemExit(
                f"unknown budget stage {stage!r}; expected one of "
                f"{sorted(budget['spent'])}")
        if budget.get("over_envelope"):
            raise SystemExit(
                "the combined forecast is over the envelope even after "
                "de-scope — do not run (v2.2 hard stop)")
        remaining = budget_remaining(budget)
        if remaining <= 0:
            raise SystemExit(
                f"v2.2 combined envelope EXHAUSTED: "
                f"{sum(budget['spent'].values()):,} fresh calls spent of "
                f"{budget['envelope']:,} — hard stop (stage {stage!r})")
        cap = remaining if local_cap is None else min(local_cap, remaining)
        self._inner = run_v2.CallBudget(cap, on_fresh=self._record_fresh)

    @property
    def fresh(self) -> int:
        return self._inner.fresh

    def install(self) -> None:
        self._inner.install()

    def uninstall(self) -> None:
        self._inner.uninstall()

    def _record_fresh(self, _guard) -> None:
        # Exclusive lock around the read-modify-write: concurrent shard
        # processes each increment this file; without the lock two shards can
        # read the same snapshot and one increment is silently lost
        # (undercounted spend defeats the envelope's forecast-error alarm).
        lock_path = self.path.with_suffix(".lock")
        with open(lock_path, "w") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            budget = json.loads(self.path.read_text())
            budget["spent"][self.stage] = budget["spent"].get(self.stage, 0) + 1
            # Atomic write (temp + rename): a plain write_text lets a reader
            # catch it empty mid-truncate (observed 2026-07-08: parallel
            # M2/M3 runs -> JSONDecodeError at char 0). pid-suffixed temp so
            # writers never share a temp path.
            tmp = self.path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(budget, indent=1, ensure_ascii=False))
            tmp.replace(self.path)
        total = sum(budget["spent"].values())
        if total > budget["envelope"]:
            raise RuntimeError(
                f"v2.2 combined envelope EXHAUSTED at {total:,} fresh calls "
                f"> {budget['envelope']:,} (stage {self.stage!r}) — hard "
                "stop; the call was recorded and aborted")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_forecast(args) -> None:
    fc = build_forecast(args.envelope)
    prior_spent = None
    if BUDGET_PATH.exists():
        prior = json.loads(BUDGET_PATH.read_text())
        prior_spent = prior.get("spent")
        if prior_spent and any(prior_spent.values()):
            print(f"[forecast] NOTE: preserving prior spent counters "
                  f"{prior_spent} from {BUDGET_PATH}")
    V22_DIR.mkdir(parents=True, exist_ok=True)
    FORECAST_PATH.write_text(json.dumps(fc, indent=1, ensure_ascii=False))
    budget = budget_from_forecast(fc, prior_spent)
    BUDGET_PATH.write_text(json.dumps(budget, indent=1, ensure_ascii=False))

    t = fc["totals"]
    n = fc["planned_n"]
    print(f"[forecast_v22] combined before de-scope "
          f"{t['combined_before_descope']:,} vs envelope "
          f"{fc['envelope']:,}; {len(fc['descope']['steps'])} de-scope "
          f"step(s) -> {t['combined_after_descope']:,}; headroom "
          f"{t['headroom_after_descope']:,}")
    for step in fc["descope"]["steps"]:
        print(f"  de-scope {step['step']}: -> combined "
              f"{step['combined_total_after']:,}")
    adds = fc["c1"]["additions"]
    print(f"  C1 scheduled {fc['c1']['scheduled']['total']:,} "
          f"(cells {fc['c1']['scheduled']['cells_total']:,} + judge "
          f"{fc['c1']['scheduled']['judge_total']}); C2 scoped "
          f"{fc['descope']['c2_total']:,}")
    print(f"  C1 additions: {len(adds['selected'])} question(s), "
          f"{adds['addition_rows']} row(s), {adds['selected_total']:,} calls"
          + (f"; cheapest unaffordable "
             f"{adds['cheapest_unaffordable']['question_id']} at "
             f"{adds['cheapest_unaffordable']['cost']:,}"
             if adds["cheapest_unaffordable"] else ""))
    print(f"  planned n = {n['total']} ({n['old_stratum']} old + "
          f"{n['scheduled_new_rows']} scheduled + {n['addition_rows']} "
          f"added) vs target {n['target']} -> "
          f"{'met' if n['target_met'] else f'SHORTFALL {n_shortfall(n)}'}")
    print(f"  planned combined {t['combined_planned']:,}; final headroom "
          f"{t['headroom_final']:,} -> {FORECAST_PATH}\n"
          f"  budget file (shared guard) -> {BUDGET_PATH}")
    for p in router_sweep.POINTS:
        scoped = fc["descope"]["c2_points"].get(p, [])
        costs = fc["c2"]["point_totals"][p]
        print(f"    C2 {p:15s} lme={costs['lme']:>6,} "
              f"locomo={costs['locomo']:>6,}  scoped={scoped}")
    if fc["over_envelope"]:
        print("[forecast_v22] OVER ENVELOPE after full de-scope — do not run")
        sys.exit(1)


def n_shortfall(n: dict) -> int:
    return n["shortfall"]


def cmd_status(args) -> None:
    budget = load_budget()
    print(json.dumps({
        "envelope": budget["envelope"],
        "spent": budget["spent"],
        "remaining": budget_remaining(budget),
        "allocation": budget["allocation"],
        "planned_n": budget["planned_n"],
        "descope_steps": [s["step"] for s in budget["descope_steps"]],
    }, indent=1, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="v2.2 combined C1+C2 forecast + shared budget (B-1/B-2)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("forecast", help="combined forecast + de-scope + "
                                        "greedy C1 additions -> budget.json")
    p.add_argument("--envelope", type=int, default=ENVELOPE)
    sub.add_parser("status", help="print spent/remaining from budget.json")
    args = parser.parse_args()
    {"forecast": cmd_forecast, "status": cmd_status}[args.cmd](args)


if __name__ == "__main__":
    main()
