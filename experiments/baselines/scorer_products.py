"""scorer_products.py — scorer for the Mem0 + Graphiti BLACK-BOX product
baselines (protocol/CHANGELOG.md 2026-08-10 "PREREGISTRATION: Mem0 +
Zep/Graphiti black-box product baselines", endpoints 1-3).

Scores the COMPLETED raw-answer records under results/baselines/
{mem0,graphiti}/{internal,external_d1}/<id>.json (runner_products.py output;
414 = 207 conversations x 2 products). Labels are read HERE and only here —
the runner is label-free by construction. Nothing model-visible is produced.

GROUND TRUTH
  internal (60 M4 arcs, results/v2.4/m4/arcs.jsonl): the expected
    current-value answer exactly as m4_eval.score_case derives it —
    resolve_true -> the hedged value (arc["value"]); abandoned/open -> the
    established prior (m4_eval.expected_prior_memory: alternative_value for
    abandoned 'alternative' mode, prior_value otherwise); the
    abstention-by-absence branch is kept for fidelity but no r4 arc uses it.
  external_d1 (147 test conversations): the held-out labels
    (scorer_d1.load_labels — expected_answer, plan_strength, outcome,
    new_value), joined on record_id.

JUDGE: rule5.EquivalenceScorer — the SAME frozen instrument scorer_d1 and
m4_eval use (imported, never copied): exact-normalized fast path, judge call
otherwise; absent/None/sentinel never match. Direction identical to
m4_eval.score_case / scorer_d1: equivalent(attribute, answer_value,
label_value); the probe question is never part of the judge prompt.

ENDPOINTS (per prereg, per product x substrate):
  E1 Task accuracy (ITT): the record's final_answer vs the expected answer.
     The runner's frozen template replies exactly "unknown" on retrieval
     miss; "unknown" matches the frozen sentinel regex, so the judge scores
     it non-equivalent — retrieval-miss/"unknown"/empty = incorrect (no
     judge call). Wilson CIs. Break-outs: internal by outcome class
     (resolve_true/abandoned/open); external by plan_strength x outcome,
     controls separately. Comparison rows are placed beside the four frozen
     systems' already-released numbers (read-only from
     results/external_eval/test/scores_d1.json and
     results/v2.4/m4/eval/m4_eval_results.json — never modified).
  E2 Contamination analog (session-probe timeline): does the never-true
     plan value (external: label new_value on abandoned/still_open;
     internal: the hedged value on abandoned/open arcs) appear as the
     CURRENT answer in a pre-outcome session probe (while_pending) or as
     the final answer (at_end)? Window rules are the scorer_d1 rules mapped
     to the session-probe granularity: the probe after session s sits after
     every turn of sessions <= s and before every turn of sessions > s, so
       abandoned:  probes with plan_session <= s < outcome_session
                   (strictly after the plan turn, strictly before the
                   outcome turn);
       still_open/open: probes with s >= plan_session through the FINAL
                   probe inclusive (scorer_d1's through-the-last-user-turn
                   window — the label re-mention id is NOT a window end).
     at_end = final_answer matches the never-true value (STORING the
     never-true plan, not final-value correctness) — scorer_d1's at_end.
  E3 Overhead: API calls / tokens / wall-clock per conversation from the
     records, cross-checked against results/baselines/usage.json.

STATISTICS: x/n with stats.wilson_ci; cluster = conversation (one item per
conversation on both substrates), so Wilson on item counts is the
preregistered CI. Frozen comparison numbers are reported with their own
released CI kinds; internal pooled rows summed from the released per-class
x/n carry a Wilson interval recomputed for display (marked as such).

BUDGET: judge calls run under runner_external.external_guard (the scorer_d1
wiring — ADVISORY stage "external" under the hard 120k envelope). The
products' own calls were accounted separately in results/baselines/usage.json
per the prereg entry; this scorer makes no product calls.

ONE-SHOT DISCIPLINE: real scoring runs ONCE; an existing
scores_baselines.json is refused without --force. --dry-run stubs the judge
(scorer_d1.DryEquivalence — exact-normalized only, ZERO LLM calls) and
writes scores_baselines_dryrun.{json,md}.

CLI (repo root):
  pilot/.venv/bin/python experiments/baselines/scorer_products.py score
      [--dry-run] [--records-root DIR] [--out-dir DIR] [--call-cap N]
      [--force]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_EXTERNAL_EVAL = _ROOT / "experiments" / "external_eval"
if str(_EXTERNAL_EVAL) not in sys.path:
    sys.path.insert(0, str(_EXTERNAL_EVAL))
import _bootstrap  # noqa: F401,E402  (R1 layout: src/ + experiments/ on sys.path)

import adapter_d1  # noqa: E402
import m4_eval  # noqa: E402
import natural_check_rule5 as rule5  # noqa: E402
import runner_external as rx  # noqa: E402
import scorer_d1  # noqa: E402
import stats  # noqa: E402

import runner_products as rp  # noqa: E402  (same package directory)

# ---------------------------------------------------------------------------
# Frozen constants
# ---------------------------------------------------------------------------

RECORDS_ROOT = rp.DEFAULT_RESULTS_ROOT           # results/baselines
OUT_DIR = rp.DEFAULT_RESULTS_ROOT
PRODUCTS = rp.PRODUCTS                            # ("mem0", "graphiti")
SUBSTRATES = rp.SUBSTRATES                        # ("internal", "external_d1")

INTERNAL_OUTCOMES = ("resolve_true", "abandoned", "open")
INTERNAL_AT_RISK = ("abandoned", "open")
EXTERNAL_OUTCOMES = scorer_d1.OUTCOMES            # came_true/abandoned/still_open
EXTERNAL_AT_RISK = scorer_d1.AT_RISK_OUTCOMES     # abandoned/still_open
STRENGTHS = scorer_d1.STRENGTHS
JUDGE_CALL_CAP = scorer_d1.JUDGE_CALL_CAP         # 4000 (the m4_eval cap)

#: Released frozen-system score files (READ-ONLY comparison sources).
FROZEN_D1_SCORES = rx.OUT_ROOT / "test" / "scores_d1.json"
FROZEN_M4_SCORES = _ROOT / "results" / "v2.4" / "m4" / "eval" / "m4_eval_results.json"
FROZEN_CELLS = ("em_cw", "em_cg", "ez_cw", "ez_cg")

#: Product -> frozen family (headline framing): mem0 is an extract-and-merge
#: fact store (the em family's design lineage); graphiti is Zep's temporal
#: knowledge-graph engine (the ez family's design lineage). PRODUCT-LEVEL
#: mapping only — see the confound disclosure in the methods block.
PRODUCT_FAMILY = {"mem0": "em", "graphiti": "ez"}


# ---------------------------------------------------------------------------
# Ground truth (labels are read HERE and only here)
# ---------------------------------------------------------------------------

def load_internal_truth() -> dict[str, dict]:
    """Per-arc ground truth from results/v2.4/m4/arcs.jsonl, expected answer
    exactly as m4_eval.score_case derives it (its logic imported)."""
    truth = {}
    for arc in m4_eval.load_arcs():
        rt = arc["resolution_type"]
        expected = (arc["value"] if rt == "resolve_true"
                    else m4_eval.expected_prior_memory(arc))
        truth[arc["arc_id"]] = {
            "attribute": arc["key"]["attribute"],
            "outcome": rt,
            "strength": arc["hedge_strength"],
            "is_control": False,
            "expected_answer": expected,
            "at_risk": rt in INTERNAL_AT_RISK,
            "never_true_value": arc["value"] if rt in INTERNAL_AT_RISK else None,
            "plan_session": arc["hedge"]["session_idx"],
            "outcome_session": arc["resolution"]["session_idx"],
            "n_sessions": len(arc["conversation"]),
        }
    return truth


def session_of_turn(raw_conv: dict, turn_id: str) -> int:
    """Calendar-day session index of a labelled turn id — the SAME grouping
    adapter_d1.sessions_for applies (day boundary = timestamp[:10] change),
    walked over the raw turn list. Asserts the referenced turn is a user
    turn (scorer_d1.user_idx_of_turn's check)."""
    day = None
    sidx = -1
    for turn in raw_conv["turns"]:
        d = turn["timestamp"][:10]
        if d != day:
            day, sidx = d, sidx + 1
        if turn["turn_id"] == turn_id:
            assert turn["speaker"] == "user", (
                f"label turn id {turn_id} references a "
                f"{turn['speaker']!r} turn, not a user turn")
            return sidx
    raise KeyError(f"turn id {turn_id!r} not in conversation "
                   f"{raw_conv['conversation_id']!r}")


def load_external_truth() -> dict[str, dict]:
    """Per-conversation ground truth for the 147 frozen test ids: held-out
    labels (scorer_d1.load_labels) + label turn ids mapped to session
    indices via the adapter's raw conversations."""
    import holdout_guard
    labels = scorer_d1.load_labels()
    convs = adapter_d1.conversations()
    truth = {}
    for cid in holdout_guard.test_ids("dataset1"):
        assert holdout_guard.slice_of("dataset1", cid) == "test"
        label = labels[cid]
        raw = convs[cid]
        n_sessions = len(adapter_d1.sessions_for(raw))
        is_control = bool(label["control_type"])
        plan_s = outcome_s = None
        if not is_control:
            et = label["event_turn_ids"]
            plan_s = session_of_turn(raw, et["possible_switch"])
            if label["outcome"] in ("came_true", "abandoned"):
                outcome_s = session_of_turn(raw, et["outcome"])
            # still_open: the label's re-mention id is NOT a window end
            # (scorer_d1 STILL_OPEN_WINDOW_NOTE) — window runs to the end.
        truth[cid] = {
            "attribute": label["attribute"],
            "outcome": label["outcome"],
            "strength": label["plan_strength"],
            "is_control": is_control,
            "expected_answer": label["expected_answer"],
            "at_risk": (not is_control) and label["outcome"] in EXTERNAL_AT_RISK,
            "never_true_value": (label["new_value"]
                                 if (not is_control)
                                 and label["outcome"] in EXTERNAL_AT_RISK
                                 else None),
            "plan_session": plan_s,
            "outcome_session": outcome_s,
            "n_sessions": n_sessions,
        }
    return truth


def load_records(records_root: Path, product: str, substrate: str) -> list[dict]:
    cell_dir = records_root / product / substrate
    recs = [json.loads(p.read_text()) for p in sorted(cell_dir.glob("*.json"))]
    expected_n = 60 if substrate == "internal" else 147
    assert len(recs) == expected_n, (
        f"{product}/{substrate}: {len(recs)} records, expected {expected_n}")
    for rec in recs:
        assert rec["record_version"] == 1
        assert rec["product"] == product and rec["substrate"] == substrate
    return recs


# ---------------------------------------------------------------------------
# Per-record scoring (pure function of record + truth + judge)
# ---------------------------------------------------------------------------

def _answered(answer) -> bool:
    """ITT answered rule: a present, non-empty, non-sentinel final answer.
    The runner's frozen template replies exactly 'unknown' on retrieval
    miss; 'unknown' matches the frozen sentinel regex."""
    if answer is None:
        return False
    text = str(answer).strip()
    return bool(text) and not rule5.value_is_sentinel(text)


async def score_record(rec: dict, truth: dict, scorer) -> dict:
    attr = truth["attribute"]
    assert rec["n_sessions"] == truth["n_sessions"], (
        f"{rec['conversation_id']}: record has {rec['n_sessions']} sessions, "
        f"substrate says {truth['n_sessions']} — window mapping unsafe")

    # ---- E1 task accuracy (ITT) -----------------------------------------
    answered = _answered(rec["final_answer"])
    expected = truth["expected_answer"]
    if expected is None:              # abstention-by-absence (m4_eval branch;
        correct = not answered        # no r4 arc / D1 label reaches it)
    elif not answered:
        correct = False               # ITT: unanswered = incorrect, no judge
    else:
        eq, _m = await scorer.equivalent(attr, rec["final_answer"], expected)
        correct = bool(eq)

    # ---- E2 contamination analog (at-risk only) --------------------------
    while_pending = at_end = None
    window_probe_count = None
    if truth["at_risk"]:
        nv = truth["never_true_value"]
        plan_s = truth["plan_session"]
        if truth["outcome"] == "abandoned":
            window = [p for p in rec["session_probes"]
                      if plan_s <= p["session_idx"] < truth["outcome_session"]]
        else:  # open / still_open: through the final probe inclusive
            window = [p for p in rec["session_probes"]
                      if p["session_idx"] >= plan_s]
        window_probe_count = len(window)
        while_pending = False
        for probe in window:
            eq, _m = await scorer.equivalent(attr, probe["answer"], nv)
            if eq:
                while_pending = True
                break
        eq, _m = await scorer.equivalent(attr, rec["final_answer"], nv)
        at_end = bool(eq)

    return {
        "conversation_id": rec["conversation_id"],
        "cluster": rec["conversation_id"],
        "product": rec["product"],
        "substrate": rec["substrate"],
        "outcome": truth["outcome"],
        "strength": truth["strength"],
        "is_control": truth["is_control"],
        "at_risk": truth["at_risk"],
        "answered": answered,
        "unanswered": not answered,
        "task_correct": correct,
        "final_answer": rec["final_answer"],
        "expected_answer": expected,
        "while_pending_contamination": while_pending,
        "at_end_contamination": at_end,
        "window_probe_count": window_probe_count,
        # E3 raw material
        "total_calls": rec["usage"]["total_calls"],
        "tokens": rec["usage"]["tokens"],
        "wall_clock_s": rec["wall_clock_s"],
        "kuzu_fallback": rec.get("kuzu_fallback", False),
    }


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


def _by_outcome(rows: list[dict], flag: str, outcomes) -> dict:
    return {o: _pt([r for r in rows if r["outcome"] == o], flag)
            for o in outcomes}


def _strength_by_outcome(rows: list[dict], flag: str) -> dict:
    out = {}
    for s in STRENGTHS:
        for o in EXTERNAL_OUTCOMES:
            sel = [r for r in rows if r["strength"] == s and r["outcome"] == o]
            out[f"{s}|{o}"] = _pt(sel, flag)
    return out


def aggregate_cell(rows: list[dict], substrate: str) -> dict:
    """E1 + E2 tree for one product x substrate cell."""
    non_ctrl = [r for r in rows if not r["is_control"]]
    ctrl = [r for r in rows if r["is_control"]]
    at_risk = [r for r in rows if r["at_risk"]]
    task: dict = {
        "pooled_all": _pt(rows, "task_correct"),
        "unanswered_ITT": _pt(rows, "unanswered"),
    }
    if substrate == "internal":
        task["by_outcome"] = _by_outcome(rows, "task_correct",
                                         INTERNAL_OUTCOMES)
    else:
        task["pooled_non_control"] = _pt(non_ctrl, "task_correct")
        task["controls"] = _pt(ctrl, "task_correct")
        task["by_strength_outcome"] = _strength_by_outcome(
            non_ctrl, "task_correct")
    at_risk_outcomes = (INTERNAL_AT_RISK if substrate == "internal"
                        else EXTERNAL_AT_RISK)
    contamination = {
        kind: {
            "pooled_at_risk": _pt(at_risk, flag),
            "by_outcome": _by_outcome(at_risk, flag, at_risk_outcomes),
        }
        for kind, flag in (("while_pending", "while_pending_contamination"),
                           ("at_end", "at_end_contamination"))
    }
    return {"task_accuracy": task, "contamination_analog": contamination}


def overhead_cell(rows: list[dict], usage_cell: dict | None) -> dict:
    calls = [r["total_calls"] for r in rows]
    wall = [r["wall_clock_s"] for r in rows]
    tok_in = [sum(v for k, v in r["tokens"].items() if k.endswith("_input"))
              for r in rows]
    tok_out = [sum(v for k, v in r["tokens"].items() if k.endswith("_output"))
               for r in rows]
    out = {
        "n_conversations": len(rows),
        "api_calls_per_conversation": {
            "mean": statistics.mean(calls), "median": statistics.median(calls),
            "min": min(calls), "max": max(calls), "total": sum(calls)},
        "wall_clock_s_per_conversation": {
            "mean": statistics.mean(wall), "median": statistics.median(wall),
            "min": min(wall), "max": max(wall), "total": round(sum(wall), 1)},
        "tokens_per_conversation": {
            "input_mean": statistics.mean(tok_in),
            "output_mean": statistics.mean(tok_out),
            "input_total": sum(tok_in), "output_total": sum(tok_out)},
        "kuzu_fallback_count": sum(1 for r in rows if r["kuzu_fallback"]),
    }
    if usage_cell is not None:
        assert usage_cell["total_calls"] == sum(calls), (
            "usage.json total_calls disagrees with per-record sum")
        out["usage_json_cell"] = usage_cell
    return out


# ---------------------------------------------------------------------------
# Frozen-system comparison rows (READ-ONLY; released files never modified)
# ---------------------------------------------------------------------------

def _wilson_pt(x: int, n: int) -> dict:
    lo, hi = stats.wilson_ci(x, n)
    return {"x": x, "n": n, "rate": x / n, "wilson_lo": lo, "wilson_hi": hi,
            "ci_note": "wilson_recomputed_for_display"}


def frozen_comparisons() -> dict:
    """The four frozen systems' already-released task-accuracy and
    contamination numbers, read from the released score files."""
    d1 = json.loads(FROZEN_D1_SCORES.read_text())
    m4 = json.loads(FROZEN_M4_SCORES.read_text())
    ext = {}
    for cell in FROZEN_CELLS:
        e3 = d1["endpoints"]["E3_task_probes"][cell]
        e1 = d1["endpoints"]["E1_contamination"]
        ext[cell] = {
            "task_pooled_all": e3["pooled_all"],
            "task_pooled_non_control": e3["pooled_non_control"],
            "task_controls": e3["controls"],
            "task_by_strength_outcome": {
                k: v for k, v in e3.items()
                if "|" in k},
            "contamination_while_pending_pooled":
                e1["while_pending"][cell]["pooled_at_risk"],
            "contamination_at_end_pooled":
                e1["at_end"][cell]["pooled_at_risk"],
        }
    internal = {}
    tp = m4["endpoints"]["task_probes"]
    cont = m4["endpoints"]["contamination_final_store"]
    for cell in FROZEN_CELLS:
        by_outcome = {o: tp[o][cell] for o in INTERNAL_OUTCOMES}
        x = sum(tp[o][cell]["x"] for o in INTERNAL_OUTCOMES)
        n = sum(tp[o][cell]["n"] for o in INTERNAL_OUTCOMES)
        cx = sum(cont[o][cell]["x"] for o in INTERNAL_AT_RISK)
        cn = sum(cont[o][cell]["n"] for o in INTERNAL_AT_RISK)
        internal[cell] = {
            "task_by_outcome": by_outcome,       # released cluster-bootstrap CIs
            "task_pooled_all": _wilson_pt(x, n),  # summed from released x/n
            "contamination_final_store_by_outcome": {
                o: cont[o][cell] for o in INTERNAL_AT_RISK},
            "contamination_final_store_pooled": _wilson_pt(cx, cn),
        }
    return {
        "sources": {"external_d1": str(FROZEN_D1_SCORES.relative_to(_ROOT)),
                    "internal": str(FROZEN_M4_SCORES.relative_to(_ROOT))},
        "external_d1": ext,
        "internal": internal,
    }


def build_headline(endpoints: dict, frozen: dict) -> dict:
    """Product task accuracy vs the ungated and gated frozen systems of the
    matching family — PRODUCT-LEVEL framing (whole-product comparison, not a
    component ablation; the confound is disclosed in methods)."""
    headline = {"framing": (
        "Whole-product baselines: each product differs from our systems in "
        "extraction, storage, retrieval AND answering, so every delta below "
        "is product-level, not attributable to the gate or any single "
        "component."), "cells": {}}
    for product in PRODUCTS:
        fam = PRODUCT_FAMILY[product]
        for substrate in SUBSTRATES:
            prod_pt = (endpoints["E1_task_accuracy"][product][substrate]
                       ["pooled_all"])
            fz = frozen[substrate]
            ungated = fz[f"{fam}_cw"]["task_pooled_all"]
            gated = fz[f"{fam}_cg"]["task_pooled_all"]
            headline["cells"][f"{product}/{substrate}"] = {
                "family": fam,
                "product": prod_pt,
                "ungated_frozen": {"cell": f"{fam}_cw", **ungated},
                "gated_frozen": {"cell": f"{fam}_cg", **gated},
                "delta_vs_ungated_pp": round(
                    100 * (prod_pt["rate"] - ungated["rate"]), 1),
                "delta_vs_gated_pp": round(
                    100 * (prod_pt["rate"] - gated["rate"]), 1),
            }
    return headline


# ---------------------------------------------------------------------------
# Full scoring pass
# ---------------------------------------------------------------------------

async def score_all(records_root: Path, scorer) -> dict:
    truths = {"internal": load_internal_truth(),
              "external_d1": load_external_truth()}
    usage_path = records_root / "usage.json"
    usage = (json.loads(usage_path.read_text())
             if usage_path.exists() else {"per_cell": {}})

    rows: list[dict] = []
    endpoints: dict = {"E1_task_accuracy": {}, "E2_contamination_analog": {},
                       "E3_overhead": {}}
    n_records = {}
    for product in PRODUCTS:
        endpoints["E1_task_accuracy"][product] = {}
        endpoints["E2_contamination_analog"][product] = {}
        endpoints["E3_overhead"][product] = {}
        for substrate in SUBSTRATES:
            recs = load_records(records_root, product, substrate)
            truth_map = truths[substrate]
            cell_rows = []
            for rec in recs:
                cid = rec["conversation_id"]
                assert cid in truth_map, f"no ground truth for {cid}"
                cell_rows.append(await score_record(rec, truth_map[cid],
                                                    scorer))
            rows.extend(cell_rows)
            agg = aggregate_cell(cell_rows, substrate)
            endpoints["E1_task_accuracy"][product][substrate] = (
                agg["task_accuracy"])
            endpoints["E2_contamination_analog"][product][substrate] = (
                agg["contamination_analog"])
            endpoints["E3_overhead"][product][substrate] = overhead_cell(
                cell_rows, usage["per_cell"].get(f"{product}/{substrate}"))
            n_records[f"{product}/{substrate}"] = len(cell_rows)

    frozen = frozen_comparisons()
    return {
        "n_records": n_records,
        "n_records_total": sum(n_records.values()),
        "endpoints": endpoints,
        "frozen_comparisons": frozen,
        "headline": build_headline(endpoints, frozen),
        "rows": rows,
    }


def methods_block(scorer, res: dict, mode: str) -> dict:
    empty_abandoned_windows = sum(
        1 for r in res["rows"]
        if r["at_risk"] and r["outcome"] == "abandoned"
        and r["window_probe_count"] == 0)
    return {
        "mode": mode,
        "spec": ("protocol/CHANGELOG.md 2026-08-10 'PREREGISTRATION: Mem0 + "
                 "Zep/Graphiti black-box product baselines' — endpoints 1-3"),
        "scoring_inputs": ("raw answer records + ground truth only; products "
                           "NEVER re-run; the frozen equivalence judge is the "
                           "only model call"),
        "equivalence": ("rule5.EquivalenceScorer (exact-normalized fast path, "
                        "judge otherwise), direction equivalent(attribute, "
                        "answer_value, label_value) per m4_eval.score_case / "
                        "scorer_d1; absent/None/sentinel never match; the "
                        "probe question is never in the judge prompt"),
        "itt": ("final_answer empty/'unknown'/sentinel = unanswered = "
                "incorrect (no judge call); the runner's frozen template "
                "replies exactly 'unknown' on retrieval miss, which the "
                "frozen sentinel regex catches"),
        "ground_truth_internal": (
            "results/v2.4/m4/arcs.jsonl via m4_eval: resolve_true -> the "
            "hedged value (arc['value']); abandoned 'alternative' -> "
            "alternative_value; abandoned 'kept_prior'/open -> prior_value "
            "(m4_eval.expected_prior_memory, imported); every r4 arc has an "
            "established prior, so the abstention-by-absence branch is "
            "never reached"),
        "ground_truth_external": (
            "held-out dataset1_labels.jsonl via scorer_d1.load_labels, "
            "joined on record_id; expected_answer for task accuracy, "
            "new_value for the contamination analog"),
        "contamination_window": (
            "scorer_d1 window rules mapped to the session-probe timeline "
            "(the probe after session s sits after all turns of sessions "
            "<= s and before all turns of sessions > s): abandoned = probes "
            "with plan_session <= s < outcome_session; still_open/open = "
            "probes with s >= plan_session through the FINAL probe "
            "inclusive (the through-the-last-user-turn window; the label's "
            "re-mention id is NOT a window end). at_end = final_answer "
            "matches the never-true value — STORING the never-true plan, "
            "not final-value correctness"),
        "contamination_granularity_note": (
            "the frozen systems' released while-pending numbers are scored "
            "on per-user-turn store timelines; the product analog is scored "
            "on after-each-session answer probes (the only black-box "
            "observable), so while-pending granularities differ across the "
            "comparison — at_end is directly comparable. On this substrate "
            "geometry every abandoned item has a non-empty session-probe "
            "window (external: plan session 1, outcome session 2, so exactly "
            "one in-window probe), unlike the frozen D1 turn-level window "
            "which is empty by geometry; "
            f"empty abandoned windows observed: {empty_abandoned_windows}"),
        "external_still_open_note": (
            "still_open windows include the final probe, so still_open "
            "while_pending >= at_end by construction on that stratum"),
        "oss_timestamp_fidelity_caveat": (
            "Mem0 OSS 2.0.17 Memory.add has no ingestion-timestamp "
            "parameter; the session date was carried in metadata AND as a "
            "date line prepended to every message (preregistered, disclosed "
            "fidelity caveat — the hosted product's timestamping may "
            "behave differently)"),
        "whole_product_confound": (
            "these are WHOLE-PRODUCT comparisons: the products differ from "
            "the four frozen systems in extraction, storage, retrieval AND "
            "answering (and mem0's answers ride on its own extraction "
            "summaries); deltas are product-level baselines, NOT component "
            "ablations, and say nothing attributable about gating per se"),
        "frozen_comparison_sources": res["frozen_comparisons"]["sources"],
        "frozen_comparison_note": (
            "frozen-system numbers are reproduced read-only from the "
            "released score files (never modified); internal pooled rows "
            "are summed from the released per-class x/n with a Wilson "
            "interval recomputed for display (released internal CIs are "
            "cluster bootstrap); external D1 rows carry their released "
            "Wilson CIs"),
        "population": ("all 414 records: 207 conversations (60 internal + "
                       "147 external D1 test) x 2 products; the external "
                       "slice was de-novo-consumed by the frozen systems' "
                       "one-shot run and is reported as the same comparison "
                       "set, NOT a fresh held-out claim (prereg)"),
        "ci": ("stats.wilson_ci on item counts; cluster = conversation, one "
               "item per conversation on both substrates"),
        "budget": ("judge calls under runner_external.external_guard "
                   "(ADVISORY stage 'external', hard 120k envelope); the "
                   "products' own calls were accounted separately in "
                   "results/baselines/usage.json per the prereg entry"),
        "judge_calls": scorer.judge_calls,
        "fresh_judge_calls": scorer.fresh_judge_calls,
        "dry_unresolved_pairs": getattr(scorer, "unresolved_pairs", None),
        "budget_snapshot": rx.advisory_status(),
    }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _fmt(pt: dict | None) -> str:
    if pt is None or pt.get("n") in (0, None):
        return "—"
    if "rate" in pt and pt.get("wilson_lo") is not None:
        return (f"{pt['x']}/{pt['n']} = {pt['rate']:.3f} "
                f"[{pt['wilson_lo']:.3f}, {pt['wilson_hi']:.3f}]")
    if "point" in pt:  # released m4_eval shape (cluster-bootstrap CI)
        return (f"{pt['x']}/{pt['n']} = {pt['point']:.3f} "
                f"[{pt['ci'][0]:.3f}, {pt['ci'][1]:.3f}]*")
    return f"{pt['x']}/{pt['n']}"


def _md_table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    lines.append("")
    return lines


def render_report(res: dict) -> str:
    ep = res["endpoints"]
    fz = res["frozen_comparisons"]
    m = res["methods"]
    hl = res["headline"]
    systems_int = [("mem0", None), ("graphiti", None)] + [
        (c, "frozen") for c in FROZEN_CELLS]

    lines = [
        "# Black-box product baselines — scores (Mem0 + Graphiti)",
        "",
        f"Mode: **{m['mode']}** — generated {res['generated_at']}",
        f"Records: {res['n_records_total']} "
        "(60 internal + 147 external D1 test, x 2 products).",
        "",
        "Spec: " + m["spec"],
        "",
        "**Read this first:** whole-product comparisons — the products "
        "differ from our four frozen systems in extraction, storage, "
        "retrieval AND answering. Product-level baselines, not component "
        "ablations.",
        "",
        "## Headline — task accuracy vs the matching frozen family",
        "",
    ]
    hrows = []
    for key, cell in hl["cells"].items():
        hrows.append([
            key, cell["family"],
            _fmt(cell["product"]),
            f"{cell['ungated_frozen']['cell']}: "
            + _fmt(cell["ungated_frozen"]),
            f"{cell['gated_frozen']['cell']}: " + _fmt(cell["gated_frozen"]),
            f"{cell['delta_vs_ungated_pp']:+.1f}pp",
            f"{cell['delta_vs_gated_pp']:+.1f}pp",
        ])
    lines += _md_table(
        ["product/substrate", "family", "product accuracy",
         "ungated frozen", "gated frozen", "Δ vs ungated", "Δ vs gated"],
        hrows)
    lines += [f"> {hl['framing']}", ""]

    # ---- E1 internal ----
    lines += ["## E1 — Task accuracy (ITT), internal (60 M4 arcs)", ""]
    rows_md = []
    for o in list(INTERNAL_OUTCOMES) + ["pooled"]:
        row = [o]
        for sysname, kind in systems_int:
            if kind is None:
                t = ep["E1_task_accuracy"][sysname]["internal"]
                pt = (t["pooled_all"] if o == "pooled"
                      else t["by_outcome"][o])
            else:
                t = fz["internal"][sysname]
                pt = (t["task_pooled_all"] if o == "pooled"
                      else t["task_by_outcome"][o])
            row.append(_fmt(pt))
        rows_md.append(row)
    lines += _md_table(["outcome class", "mem0", "graphiti"]
                       + list(FROZEN_CELLS), rows_md)
    lines += ["\\* released cluster-bootstrap CI (m4_eval_results.json); "
              "frozen pooled rows: Wilson recomputed for display from "
              "released x/n.", ""]

    # ---- E1 external ----
    lines += ["## E1 — Task accuracy (ITT), external D1 test (147)", ""]
    rows_md = []
    strata = ([f"{s}|{o}" for s in STRENGTHS for o in EXTERNAL_OUTCOMES]
              + ["controls", "pooled_non_control", "pooled_all"])
    for stratum in strata:
        row = [stratum.replace("|", " / ")]
        for sysname, kind in systems_int:
            if kind is None:
                t = ep["E1_task_accuracy"][sysname]["external_d1"]
                pt = (t.get(stratum) if stratum.startswith("pooled")
                      or stratum == "controls"
                      else t["by_strength_outcome"].get(stratum))
            else:
                t = fz["external_d1"][sysname]
                pt = {"pooled_all": t["task_pooled_all"],
                      "pooled_non_control": t["task_pooled_non_control"],
                      "controls": t["task_controls"]}.get(
                    stratum, t["task_by_strength_outcome"].get(stratum))
            row.append(_fmt(pt))
        rows_md.append(row)
    lines += _md_table(["stratum", "mem0", "graphiti"] + list(FROZEN_CELLS),
                       rows_md)

    # ---- E2 ----
    lines += ["## E2 — Contamination analog (session-probe timeline, "
              "at-risk items)", "",
              "While-pending: the never-true plan value asserted as the "
              "current answer in a pre-outcome session probe. At-end: the "
              "final answer equals the never-true value. Frozen-system "
              "while-pending numbers are turn-level store timelines "
              "(different granularity — see methods); at_end is directly "
              "comparable.", ""]
    for substrate, at_risk_outcomes, fz_key in (
            ("internal", INTERNAL_AT_RISK,
             "contamination_final_store"),
            ("external_d1", EXTERNAL_AT_RISK, None)):
        lines += [f"### {substrate}", ""]
        rows_md = []
        for kind in ("while_pending", "at_end"):
            for o in list(at_risk_outcomes) + ["pooled_at_risk"]:
                row = [f"{kind} / {o}"]
                for product in PRODUCTS:
                    t = ep["E2_contamination_analog"][product][substrate][kind]
                    pt = (t["pooled_at_risk"] if o == "pooled_at_risk"
                          else t["by_outcome"][o])
                    row.append(_fmt(pt))
                for cell in FROZEN_CELLS:
                    if substrate == "internal":
                        t = fz["internal"][cell]
                        if kind == "at_end":
                            pt = (t["contamination_final_store_pooled"]
                                  if o == "pooled_at_risk" else
                                  t["contamination_final_store_by_outcome"][o])
                        else:
                            pt = None  # no released turn-level analog table
                    else:
                        t = fz["external_d1"][cell]
                        pt = (t[f"contamination_{kind}_pooled"]
                              if o == "pooled_at_risk" else None)
                    row.append(_fmt(pt))
                rows_md.append(row)
        lines += _md_table(["endpoint / stratum", "mem0", "graphiti"]
                           + list(FROZEN_CELLS), rows_md)
    lines += ["Frozen internal while-pending cells are omitted (the "
              "released M4 endpoints report boundary/premature-commit "
              "measures, not a session-probe timeline); frozen external "
              "per-outcome strata are in the released scores_d1.json.", ""]

    # ---- E3 ----
    lines += ["## E3 — Overhead (per conversation)", ""]
    rows_md = []
    for product in PRODUCTS:
        for substrate in SUBSTRATES:
            o = ep["E3_overhead"][product][substrate]
            rows_md.append([
                f"{product}/{substrate}",
                str(o["n_conversations"]),
                f"{o['api_calls_per_conversation']['mean']:.1f} "
                f"(med {o['api_calls_per_conversation']['median']:.0f})",
                f"{o['wall_clock_s_per_conversation']['mean']:.1f} "
                f"(med {o['wall_clock_s_per_conversation']['median']:.1f})",
                f"{o['tokens_per_conversation']['input_mean']:,.0f}",
                f"{o['tokens_per_conversation']['output_mean']:,.0f}",
                str(o["kuzu_fallback_count"]),
            ])
    lines += _md_table(
        ["cell", "n", "API calls/conv", "wall s/conv",
         "input tok/conv", "output tok/conv", "kuzu fallbacks"], rows_md)

    lines += ["## Methods", ""]
    for k, v in m.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def run_score(records_root: Path, out_dir: Path, dry_run: bool,
                    call_cap: int | None, force: bool) -> dict:
    out_json = out_dir / ("scores_baselines_dryrun.json" if dry_run
                          else "scores_baselines.json")
    out_md = out_dir / ("scores_baselines_dryrun.md" if dry_run
                        else "scores_baselines.md")
    if not dry_run and out_json.exists() and not force:
        raise SystemExit(
            f"{out_json} already exists — the real scoring is ONE-SHOT; "
            "pass --force only for a logged infrastructure-failure rerun")

    guard = None
    if dry_run:
        scorer = scorer_d1.DryEquivalence()
    else:
        scorer = rule5.EquivalenceScorer(cap=JUDGE_CALL_CAP)
        guard = rx.external_guard(local_cap=call_cap)
        guard.install()
    try:
        res = await score_all(records_root, scorer)
    finally:
        if guard is not None:
            guard.uninstall()

    res["generated_at"] = datetime.now(timezone.utc).isoformat()
    res["methods"] = methods_block(
        scorer, res,
        "dry_run (judge stubbed: exact-normalized only, zero LLM calls)"
        if dry_run else
        "full (frozen equivalence judge, budget stage 'external')")
    if guard is not None:
        res["methods"]["guard_fresh_calls"] = guard.fresh

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    out_md.write_text(render_report(res))
    print(f"wrote {out_json}\nwrote {out_md}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("score", help="score the product baseline records")
    sc.add_argument("--records-root", type=Path, default=RECORDS_ROOT)
    sc.add_argument("--out-dir", type=Path, default=OUT_DIR)
    sc.add_argument("--dry-run", action="store_true",
                    help="stub the judge (exact-normalized only, no LLM "
                    "calls); writes scores_baselines_dryrun.{json,md}")
    sc.add_argument("--call-cap", type=int, default=None)
    sc.add_argument("--force", action="store_true",
                    help="overwrite an existing scores_baselines.json "
                    "(logged infrastructure-failure rerun only)")
    args = ap.parse_args()
    res = asyncio.run(run_score(args.records_root, args.out_dir,
                                args.dry_run, args.call_cap, args.force))
    ep = res["endpoints"]
    print(f"\nscored {res['n_records_total']} records; "
          f"judge_calls={res['methods']['judge_calls']} "
          f"(fresh {res['methods']['fresh_judge_calls']})")
    for product in PRODUCTS:
        for substrate in SUBSTRATES:
            task = ep["E1_task_accuracy"][product][substrate]["pooled_all"]
            cont = ep["E2_contamination_analog"][product][substrate]
            wp = cont["while_pending"]["pooled_at_risk"]
            ae = cont["at_end"]["pooled_at_risk"]
            print(f"  {product}/{substrate}: task {_fmt(task)} | "
                  f"while_pending {_fmt(wp)} | at_end {_fmt(ae)}")


if __name__ == "__main__":
    main()
