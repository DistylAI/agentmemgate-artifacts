"""claude_repl.py — the preregistered different-family replication
(protocol/CHANGELOG.md 2026-08-11 "PREREGISTRATION: different-family
replication (Claude Opus 4.8 systems, frozen gpt-5.5 judge)" — the binding
spec).

SYSTEM MODEL: claude-opus-4-8 via MEMGATE_SYSTEM_MODEL (the llm.py provider
seam; alias pin — no dated snapshot exists, response.model echoed in every
trace/cache entry, prereg deviation (a)). JUDGE: llm.JUDGE_MODEL, pinned
gpt-5.5-2026-04-23 at store.values_equivalent (deviation (d): same
instrument on both families).

SCOPE (one pass each, per-item resumable; substrates reused VERBATIM):
  run-m4  the 60 r4 conversations through the m4_eval harness (all five
          arms — the four frozen cells + em_cg_fixed; em_cg_fixed IS the
          adopted repaired cautious-gated configuration) ->
          results/claude_repl/m4/cases/
  run-d1  the 147-conversation D1 TEST slice x 4 systems via
          runner_external.run_d1_case (consumed-slice reuse; the prereg
          authorizes EXTERNAL_TEST_RUN_AUTHORIZED=1 for THIS run) ->
          results/claude_repl/d1/test/cases_d1/   (the released
          results/external_eval/test/ artifacts are NEVER touched)
  run-m2  the 420 planted-probe cases, ADOPTED cautious-gated cell only
          (graded_eval.run_fixed, em_cg_fixed sibling records) ->
          results/claude_repl/m2_emcg/cases_fixed/
  score   the existing scorers pointed at the claude record roots (frozen
          gpt-5.5 judge) -> per-substrate endpoint tables + the directional
          scorecard vs the released gpt-5.5 numbers ->
          results/claude_repl/claude_repl_results.json + _report.md

BUDGET: stage "claude_repl" (ADVISORY forecast 15,000; budget.json +
mapper_fix.ALLOCATION, 2026-08-11 note), spend recorded per fresh call to
results/v2.4/budget.json via the flock-guarded forecast_v22.SharedCallBudget.
The 120k envelope remains the protocol hard stop; THIS replication
additionally stops at a 110k total (prereg stop condition), enforced via
local_cap. Claude per-item error rates > 5% after retries stop the run.

CLI (repo root; MEMGATE_SYSTEM_MODEL=claude-opus-4-8 required for run-*):
  pilot/.venv/bin/python experiments/claude_repl/claude_repl.py \
      {run-m4,run-d1,run-m2} [--shard K/N] [--call-cap N]
  pilot/.venv/bin/python experiments/claude_repl/claude_repl.py score
  pilot/.venv/bin/python experiments/claude_repl/claude_repl.py status
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401

import forecast_v22
import graded_eval
import holdout_guard
import llm
import m4_eval
import mapper_fix
import natural_check_rule5 as rule5
import runner_external as rx
import scorer_d1

from paths import RESULTS_DIR

STAGE = "claude_repl"
CLAUDE_MODEL = "claude-opus-4-8"
ADVISORY_FORECAST = 15_000
#: Prereg stop condition for THIS replication (the protocol hard stop stays
#: the 120k envelope): stop before the combined total can pass 110,000.
HARD_STOP_TOTAL = 110_000
#: Prereg stop condition: per-item Claude error rate after retries.
MAX_ERROR_RATE = 0.05

V24_DIR = RESULTS_DIR / "v2.4"
BUDGET_PATH = V24_DIR / "budget.json"

CR_DIR = RESULTS_DIR / "claude_repl"
M4_CASES = CR_DIR / "m4" / "cases"
M4_OUT = CR_DIR / "m4"
D1_ROOT = CR_DIR / "d1"
D1_CASES = D1_ROOT / "test" / "cases_d1"
M2_OUT = CR_DIR / "m2_emcg"
M2_FIXED_CASES = M2_OUT / "cases_fixed"
RUN_LOG = CR_DIR / "run_log.jsonl"
BUDGET_BEFORE = CR_DIR / "budget_before.json"
RESULTS_PATH = CR_DIR / "claude_repl_results.json"
REPORT_PATH = CR_DIR / "claude_repl_report.md"

#: Released gpt-5.5 artifacts the scorecard compares against (read-only).
RELEASED_D1 = RESULTS_DIR / "external_eval" / "test" / "scores_d1.json"
RELEASED_M4 = V24_DIR / "m4" / "eval" / "m4_eval_results.json"
RELEASED_M2 = V24_DIR / "m2" / "m2_results.json"

#: JSON-mode mechanism actually used on the Claude arm (2026-08-11 amendment
#: to the prereg + empirical API finding). The amendment asked for API-level
#: JSON-object parity with the GPT arm's json_object mode; the current
#: Anthropic Messages API exposes ONLY strict schema-based structured output
#: (output_config.format.type must be 'json_schema', and every object schema
#: must set additionalProperties:false — verified: bare {"type":"object"} and
#: additionalProperties:true both 400). A strict schema that admits arbitrary
#: keys does not exist, so the amendment's specified permissive-object
#: fallback is not implementable, and per-prompt strict schemas were NOT used
#: (they would constrain the Claude arm asymmetrically vs the GPT arm — the
#: opposite of the amendment's "no property constraints" intent). The run
#: therefore used the frozen prompts' explicit "return only JSON" instruction
#: + json.loads inside the retry loop + a single-fence stripper. Parse-retry
#: counts are reported.
JSON_MECHANISM = ("prompt-instructed JSON + json.loads-inside-retry + "
                  "markdown-fence stripper (Anthropic API has no unconstrained "
                  "JSON-object mode; only strict json_schema with mandatory "
                  "additionalProperties:false, which cannot express the GPT "
                  "json_object regime — amendment (1) blocked, see report)")


# ---------------------------------------------------------------------------
# Budget (advisory stage claude_repl; 110k replication stop via local_cap)
# ---------------------------------------------------------------------------

def total_spent(path: Path = BUDGET_PATH) -> int:
    return sum(mapper_fix.ensure_budget(path)["spent"].values())


def claude_guard(path: Path = BUDGET_PATH,
                 local_cap: int | None = None) -> forecast_v22.SharedCallBudget:
    """The m4_eval.m4_guard pattern, stage claude_repl. The stage forecast is
    ADVISORY (warn + continue); the replication's own stop condition caps
    fresh calls so the combined total cannot pass HARD_STOP_TOTAL."""
    budget = mapper_fix.ensure_budget(path)
    spent = budget["spent"].get(STAGE, 0)
    forecast = budget["allocation"].get(STAGE, ADVISORY_FORECAST)
    if spent >= forecast:
        print(f"[claude_repl budget ADVISORY] stage spend {spent:,} >= "
              f"forecast {forecast:,} — continuing (stages are advisory per "
              "the 2026-07-08 amendment; only the total envelope is hard)")
    margin_110 = HARD_STOP_TOTAL - sum(budget["spent"].values())
    if margin_110 <= 0:
        raise SystemExit(
            f"claude_repl STOP CONDITION: combined spend "
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
        "replication_stop_total": HARD_STOP_TOTAL,
        "envelope_spent_total": sum(budget["spent"].values()),
        "mode": "ADVISORY stage / HARD total envelope (2026-07-08 amendment)"
                " / 110k replication stop (2026-08-11 prereg)",
    }


# ---------------------------------------------------------------------------
# Pins + bookkeeping
# ---------------------------------------------------------------------------

def assert_claude_system() -> None:
    assert llm.MODEL == CLAUDE_MODEL, (
        f"llm.MODEL is {llm.MODEL!r} — run-* commands require "
        f"MEMGATE_SYSTEM_MODEL={CLAUDE_MODEL} in the environment")
    assert_judge_pinned()


def assert_judge_pinned() -> None:
    assert llm.JUDGE_MODEL == "gpt-5.5-2026-04-23", (
        f"llm.JUDGE_MODEL drifted to {llm.JUDGE_MODEL!r} — the scoring "
        "instrument is frozen gpt-5.5 (prereg deviation (d))")


def snapshot_budget_before() -> None:
    if not BUDGET_BEFORE.exists():
        CR_DIR.mkdir(parents=True, exist_ok=True)
        BUDGET_BEFORE.write_text(BUDGET_PATH.read_text())


def log_phase(row: dict) -> None:
    CR_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUN_LOG, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _shard(ids: list[str], spec: str | None) -> list[str]:
    if not spec:
        return ids
    k, n = (int(x) for x in spec.split("/"))
    assert 0 <= k < n, f"bad shard spec {spec!r}"
    return ids[k::n]


def check_error_rate(phase: str, failed: list, attempted: int) -> None:
    if attempted and len(failed) / attempted > MAX_ERROR_RATE:
        raise SystemExit(
            f"claude_repl STOP CONDITION: {phase} error rate "
            f"{len(failed)}/{attempted} > {MAX_ERROR_RATE:.0%} after "
            f"retries — prereg stop; failures: {failed[:5]} ...")


# ---------------------------------------------------------------------------
# run-m4 — 60 arcs, five arms (m4_eval harness verbatim; claude records)
# ---------------------------------------------------------------------------

async def run_m4(ids: list[str] | None, shard: str | None,
                 call_cap: int | None) -> dict:
    assert_claude_system()
    snapshot_budget_before()
    arcs = m4_eval.load_arcs()
    all_ids = [a["arc_id"] for a in arcs]
    sel = _shard(sorted(ids if ids is not None else all_ids), shard)
    started = datetime.now(timezone.utc).isoformat()
    guard = claude_guard(local_cap=call_cap)
    done = skipped = 0
    failed: list[dict] = []
    # per-arc loop (m4_eval.run with a one-id selection) so a single arc
    # failure is recorded and the pass continues — per-item resumable.
    for arc_id in sel:
        try:
            s = await m4_eval.run(m4_eval.CLASSES, m4_eval.FAMILIES,
                                  [arc_id], M4_CASES, guard)
            done += s["n_arcs"]
            skipped += s["skipped"]
        except (SystemExit, KeyboardInterrupt):
            raise
        except Exception as exc:  # noqa: BLE001 — recorded, run continues
            low = str(exc).lower()
            if "budget" in low or "envelope" in low or "exhausted" in low:
                raise
            failed.append({"id": arc_id, "error": repr(exc)})
    summary = {"phase": "m4", "model": llm.MODEL, "started": started,
               "ended": datetime.now(timezone.utc).isoformat(),
               "selected": len(sel), "n_new": done, "skipped": skipped,
               "failed": failed, "fresh_calls": guard.fresh,
               "advisory": advisory_status()}
    log_phase(summary)
    check_error_rate("m4", failed, len(sel))
    return summary


# ---------------------------------------------------------------------------
# run-d1 — 147 test conversations x 4 systems (runner_external verbatim)
# ---------------------------------------------------------------------------

async def run_d1(ids: list[str] | None, shard: str | None,
                 call_cap: int | None) -> dict:
    assert_claude_system()
    snapshot_budget_before()
    # The 2026-08-11 prereg authorizes the one-shot flag for THIS run
    # (consumed-slice reuse of the 147-conversation test slice, reported as
    # the same comparison set). Records land under results/claude_repl/d1/ —
    # the released results/external_eval/test/ artifacts are never written.
    os.environ[holdout_guard.ENV_FLAG] = "1"
    sel = _shard(sorted(ids if ids is not None
                        else holdout_guard.test_ids("dataset1")), shard)
    started = datetime.now(timezone.utc).isoformat()
    guard = claude_guard(local_cap=call_cap)
    s = await rx.run_dataset("dataset1", sel, D1_ROOT, guard)
    summary = {"phase": "d1", "model": llm.MODEL, "started": started,
               "ended": datetime.now(timezone.utc).isoformat(),
               "selected": len(sel), "n_new": s["n_cases"],
               "skipped": s["skipped"], "failed": s["failed"],
               "fresh_calls": s["fresh_calls"],
               "advisory": advisory_status()}
    log_phase(summary)
    check_error_rate("d1", s["failed"], len(sel))
    return summary


# ---------------------------------------------------------------------------
# run-m2 — 420 planted cases, ADOPTED cautious-gated cell only (run_fixed)
# ---------------------------------------------------------------------------

async def run_m2(ids: list[str] | None, shard: str | None,
                 call_cap: int | None) -> dict:
    assert_claude_system()
    snapshot_budget_before()
    graded_eval.assert_manifest_frozen()
    all_ids = sorted({r["case_id"]
                      for r in graded_eval._read_jsonl(graded_eval.MANIFEST_PATH)})
    sel = _shard(sorted(ids if ids is not None else all_ids), shard)
    started = datetime.now(timezone.utc).isoformat()
    guard = claude_guard(local_cap=call_cap)
    s = await graded_eval.run_fixed(
        ("lme", "locomo"), ("gradient", "mixed", "nearmiss"), sel,
        M2_FIXED_CASES, guard)
    summary = {"phase": "m2_emcg", "model": llm.MODEL, "started": started,
               "ended": datetime.now(timezone.utc).isoformat(),
               "selected": len(sel), "n_new": s["n_cases"],
               "skipped": s["skipped"], "failed": s["failed"],
               "fresh_calls": s["fresh_calls"],
               "advisory": advisory_status()}
    log_phase(summary)
    check_error_rate("m2_emcg", s["failed"], len(sel))
    return summary


# ---------------------------------------------------------------------------
# score — existing scorers pointed at the claude record roots
# ---------------------------------------------------------------------------

async def score_m4() -> dict:
    guard = claude_guard()
    guard.install()
    try:
        res = await m4_eval.compute_scores(M4_CASES)
    finally:
        guard.uninstall()
    res["score_fresh_calls"] = guard.fresh
    res["system_model_note"] = (
        f"CLAUDE REPLICATION: all five arms ran under {CLAUDE_MODEL} "
        "(MEMGATE_SYSTEM_MODEL seam); scoring judge = frozen "
        f"{llm.JUDGE_MODEL}; budget stage claude_repl")
    (M4_OUT).mkdir(parents=True, exist_ok=True)
    (M4_OUT / "m4_eval_results.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False))
    (M4_OUT / "m4_eval_report.md").write_text(m4_eval.render_report(res))
    return res


async def score_d1() -> dict:
    cases = scorer_d1.load_cases(D1_CASES)
    slices = {c.get("slice") for c in cases}
    assert slices == {"test"}, f"claude D1 records span slices {slices}"
    labels = scorer_d1.load_labels()
    convs = scorer_d1.load_conversations()
    scorer = rule5.EquivalenceScorer(cap=scorer_d1.JUDGE_CALL_CAP)
    guard = claude_guard()
    guard.install()
    try:
        # Amendment 2026-08-11 (owner): deviation (e) DROPPED — no sensitivity
        # row excluding the 7 Fable-rewritten-opener D1 conversations. Both
        # families replicate on the identical dataset; the PROVENANCE
        # disclosure stands on its own.
        res = await scorer_d1.score_all(cases, labels, convs, scorer)
    finally:
        guard.uninstall()
    res["generated_at"] = datetime.now(timezone.utc).isoformat()
    res["methods"] = scorer_d1.methods_block(
        scorer, res, mode=(f"claude replication (systems {CLAUDE_MODEL}, "
                           f"frozen {llm.JUDGE_MODEL} judge, budget stage "
                           "claude_repl)"))
    res["methods"]["guard_fresh_calls"] = guard.fresh
    res["methods"]["budget_snapshot"] = advisory_status()
    res.pop("_fallback_counter")
    D1_ROOT.mkdir(parents=True, exist_ok=True)
    (D1_ROOT / "scores_d1.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False))
    (D1_ROOT / "scores_d1.md").write_text(scorer_d1.render_report(res))
    return res


async def score_m2() -> dict:
    """graded_eval scoring with the claude em_cg_fixed sibling records merged
    over the RELEASED frozen case records (read-only). In the output the four
    frozen cells are therefore the released gpt-5.5 numbers (recomputed from
    the released records + judge cache) and em_cg_fixed is the CLAUDE adopted
    cell — labeled as such."""
    guard = claude_guard()
    guard.install()
    try:
        res = await graded_eval.compute_scores(graded_eval.CASES_DIR,
                                               M2_FIXED_CASES)
    finally:
        guard.uninstall()
    res["score_fresh_calls"] = guard.fresh
    res["system_model_note"] = (
        f"CLAUDE REPLICATION (planted-probe suite): em_cg_fixed = the "
        f"ADOPTED cautious-gated cell rerun under {CLAUDE_MODEL}; the four "
        "frozen cells in these tables are the RELEASED gpt-5.5 numbers "
        "(rescored from the released case records; byte-comparable to "
        "results/v2.4/m2/m2_results.json), shown for reference — the "
        "cross-family comparison row is em_cg_fixed only. Judge = frozen "
        f"{llm.JUDGE_MODEL}; budget stage claude_repl")
    M2_OUT.mkdir(parents=True, exist_ok=True)
    (M2_OUT / "m2_results.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False))
    (M2_OUT / "m2_report.md").write_text(graded_eval.render_report(res))
    return res


# ---------------------------------------------------------------------------
# trace forensics — parse-retry counts + response.model inventory
# ---------------------------------------------------------------------------

def run_window() -> tuple[float | None, float | None]:
    if not RUN_LOG.exists():
        return None, None
    rows = [json.loads(l) for l in RUN_LOG.open() if l.strip()]
    starts = [datetime.fromisoformat(r["started"]).timestamp() for r in rows]
    ends = [datetime.fromisoformat(r["ended"]).timestamp() for r in rows]
    return (min(starts) - 5, max(ends) + 5) if rows else (None, None)


def trace_forensics() -> dict:
    """Scan traces written during the replication run window: served-model
    inventory (prereg deviation (a)) and retry/attempt distribution for
    claude- calls (deviation (b): parse failures retry inside the frozen
    loop; attempt=k on the logged success means k-1 failed attempts)."""
    lo, hi = run_window()
    if lo is None:
        return {"note": "no run_log.jsonl — no run executed yet"}
    inv: Counter = Counter()
    attempts: Counter = Counter()
    n_scanned = 0
    for p in (RESULTS_DIR / "traces").iterdir():
        try:
            mt = p.stat().st_mtime
        except OSError:
            continue
        if not (lo <= mt <= hi):
            continue
        try:
            d = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        n_scanned += 1
        model = d.get("model") or "?"
        inv[model] += 1
        if str(model).startswith("claude"):
            attempts[d.get("attempt", 1)] += 1
    claude_total = sum(attempts.values())
    retried = sum(v for k, v in attempts.items() if k and k > 1)
    extra = sum((k - 1) * v for k, v in attempts.items() if k)
    return {
        "window_utc": [datetime.fromtimestamp(lo, timezone.utc).isoformat(),
                       datetime.fromtimestamp(hi, timezone.utc).isoformat()],
        "n_traces_in_window": n_scanned,
        "response_model_inventory": dict(inv),
        "claude_attempt_histogram": {str(k): v for k, v
                                     in sorted(attempts.items())},
        "claude_calls_logged": claude_total,
        "claude_calls_needing_retry": retried,
        "claude_extra_attempts_total": extra,
        "retry_note": ("attempt on the logged (successful) call = total "
                       "attempts incl. parse-inside-retry failures and "
                       "transport errors; MAX_RETRIES exhaustion surfaces "
                       "as a per-item failure in run_log.jsonl instead"),
    }


# ---------------------------------------------------------------------------
# directional scorecard vs the released gpt-5.5 numbers
# ---------------------------------------------------------------------------

def _fmt_pt(pt: dict | None) -> str:
    if not pt:
        return "—"
    if pt.get("rate") is None and pt.get("point") is None:
        return "—"
    rate = pt.get("rate", pt.get("point"))
    return f"{pt['x']}/{pt['n']}={rate:.3f}"


def d1_rows(ep: dict) -> list[dict]:
    """The released D1 directional readings (changelog 2026-07-31 scorecard,
    informative operative parts), re-evaluated on an endpoints tree."""
    def pt(kind, cell):
        return ep["E1_contamination"][kind][cell]["pooled_at_risk"]

    def probes(cell):
        return ep["E3_task_probes"][cell]["pooled_all"]

    def staged(cell):
        return ep["E2_staging_promotion"]["staged_at_plan"][cell][
            "pooled_non_control"]

    def promo(cell):
        return ep["E2_staging_promotion"]["promotion_came_true"][cell][
            "t_star_k0"]

    def rate(p):
        return p.get("rate")

    rows = []
    em_ae, ez_ae = pt("at_end", "em_cw"), pt("at_end", "ez_cw")
    rows.append({
        "row": "D1-(i)",
        "expectation": "ungated literal at-end contamination >> ungated "
                       "cautious (ez_cw > em_cw, pooled at-risk)",
        "observed": f"em_cw {_fmt_pt(em_ae)}; ez_cw {_fmt_pt(ez_ae)}",
        "holds": (rate(em_ae) is not None and rate(ez_ae) is not None
                  and rate(ez_ae) > rate(em_ae))})
    parts, obs = [], []
    for fam in ("em", "ez"):
        for kind in ("while_pending", "at_end"):
            cg, cw = pt(kind, f"{fam}_cg"), pt(kind, f"{fam}_cw")
            parts.append(rate(cg) is not None and rate(cw) is not None
                         and rate(cg) < rate(cw))
            obs.append(f"{fam} {kind}: cg {_fmt_pt(cg)} < cw {_fmt_pt(cw)}")
    rows.append({
        "row": "D1-(ii)",
        "expectation": "gated contamination < ungated within family, both "
                       "windows (the informative part of released row ii)",
        "observed": "; ".join(obs),
        "holds": all(parts)})
    em_s, ez_s = staged("em_cg"), staged("ez_cg")
    em_p, ez_p = promo("em_cg"), promo("ez_cg")
    rows.append({
        "row": "D1-(iii)",
        "expectation": "cautious entry limitation: staged em_cg < ez_cg AND "
                       "promotion (came_true, t*) em_cg < ez_cg",
        "observed": (f"staged {_fmt_pt(em_s)} vs {_fmt_pt(ez_s)}; "
                     f"promoted {_fmt_pt(em_p)} vs {_fmt_pt(ez_p)}"),
        "holds": (rate(em_s) is not None and rate(ez_s) is not None
                  and rate(em_s) < rate(ez_s)
                  and rate(em_p) is not None and rate(ez_p) is not None
                  and rate(em_p) < rate(ez_p))})
    for fam, label in (("ez", "literal"), ("em", "cautious")):
        cg, cw = probes(f"{fam}_cg"), probes(f"{fam}_cw")
        rows.append({
            "row": f"D1-(iv)-{fam}",
            "expectation": f"task-probe gate delta positive under the "
                           f"{label} family ({fam}_cg > {fam}_cw, ITT)",
            "observed": f"{fam}_cg {_fmt_pt(cg)} vs {fam}_cw {_fmt_pt(cw)}",
            "holds": (rate(cg) is not None and rate(cw) is not None
                      and rate(cg) > rate(cw))})
    em_cw_p, ez_cw_p = probes("em_cw"), probes("ez_cw")
    rows.append({
        "row": "D1-(iv)-ungated",
        "expectation": "literal-ungated worst on probes (ez_cw < em_cw) — "
                       "FAILED (tie) in the released gpt-5.5 run",
        "observed": f"em_cw {_fmt_pt(em_cw_p)}; ez_cw {_fmt_pt(ez_cw_p)}",
        "holds": (rate(ez_cw_p) is not None and rate(em_cw_p) is not None
                  and rate(ez_cw_p) < rate(em_cw_p))})
    return rows


def m2_pooled(res: dict, block_key: str, cell: str, endpoint: str) -> dict:
    """Pool a gradient/mixed/nearmiss endpoint across benchmarks (n=60)."""
    trees = {"gradient": res["gradient_curves"], "mixed": res["mixed"],
             "nearmiss": res["nearmiss_promotion_curves"],
             "nearmiss_staging": res["nearmiss_staging"]}
    x = n = 0
    for bench in ("lme", "locomo"):
        for name, tree in trees.items():
            k = f"{bench}/{block_key}"
            if k in tree and cell in tree[k]["cells"]:
                p = tree[k]["cells"][cell].get(endpoint)
                if p and p.get("n"):
                    x += p["x"]
                    n += p["n"]
                break
    return {"x": x, "n": n, "rate": (x / n) if n else None}


def m2_rows(res: dict, cell: str) -> list[dict]:
    """Planted-probe dose-response rows for one cautious-gated cell (the
    released em_cg_fixed curve shape), pooled across benchmarks."""
    def contam(level):
        return m2_pooled(res, level, cell, "contamination")

    def promo(level):
        return m2_pooled(res, level, cell, "promote_fire_within_window_ITT")

    g = {lv: contam(lv) for lv in ("g1_strong_hedge", "g2_weak_hedge",
                                   "g3_confident_plan", "g4_scheduled_event",
                                   "g5_present_tense_change")}
    mixed = contam("m1_fact_plus_hedge")
    stg = m2_pooled(res, "staging", cell, "contamination")
    p1, p4 = promo("p1_verbatim"), promo("p4_situational")

    def r(p):
        return p["rate"]

    def _all(*pts):
        return all(r(p) is not None for p in pts)

    rows = [
        {"row": "P-1",
         "expectation": "dose-response floor: strong/weak-hedge "
                        "contamination ~ 0 (g1, g2 <= 5%)",
         "observed": f"g1 {_fmt_pt(g['g1_strong_hedge'])}; "
                     f"g2 {_fmt_pt(g['g2_weak_hedge'])}",
         "holds": (_all(g["g1_strong_hedge"], g["g2_weak_hedge"])
                   and r(g["g1_strong_hedge"]) <= 0.05
                   and r(g["g2_weak_hedge"]) <= 0.05)},
        {"row": "P-2",
         "expectation": "failure onset at confident-plan phrasing "
                        "(g3 > g1 and g3 > g2)",
         "observed": f"g3 {_fmt_pt(g['g3_confident_plan'])}",
         "holds": (_all(g["g3_confident_plan"], g["g1_strong_hedge"],
                        g["g2_weak_hedge"])
                   and r(g["g3_confident_plan"]) > r(g["g1_strong_hedge"])
                   and r(g["g3_confident_plan"]) > r(g["g2_weak_hedge"]))},
        {"row": "P-3",
         "expectation": "recovery with explicit dates (g4 < g3)",
         "observed": f"g4 {_fmt_pt(g['g4_scheduled_event'])} vs "
                     f"g3 {_fmt_pt(g['g3_confident_plan'])}",
         "holds": (_all(g["g4_scheduled_event"], g["g3_confident_plan"])
                   and r(g["g4_scheduled_event"]) < r(g["g3_confident_plan"]))},
        {"row": "P-4",
         "expectation": "rule-(d) completed-event commits stay CORRECT "
                        "(g5 commit rate >= 90%; commit is correct, not "
                        "contamination)",
         "observed": f"g5 {_fmt_pt(g['g5_present_tense_change'])}",
         "holds": (_all(g["g5_present_tense_change"])
                   and r(g["g5_present_tense_change"]) >= 0.90)},
        {"row": "P-5",
         "expectation": "mixed fact+hedge sentence: hedge leak ~ 0 "
                        "(<= 5%)",
         "observed": f"mixed {_fmt_pt(mixed)}",
         "holds": _all(mixed) and r(mixed) <= 0.05},
        {"row": "P-6",
         "expectation": "near-miss staging-turn contamination ~ 0 (<= 5%)",
         "observed": f"staging {_fmt_pt(stg)}",
         "holds": _all(stg) and r(stg) <= 0.05},
        {"row": "P-7",
         "expectation": "promotion declines with paraphrase distance "
                        "(fire-in-window ITT: p1 > p4)",
         "observed": f"p1 {_fmt_pt(p1)}; p4 {_fmt_pt(p4)}",
         "holds": _all(p1, p4) and r(p1) > r(p4)},
    ]
    return rows


def build_scorecard(claude: dict, released: dict) -> dict:
    """rows: released verdict vs claude verdict, per preregistered row."""
    card: dict = {"m4": [], "d1": [], "m2_emcg": []}

    def _m4_obs(row: dict) -> str:
        o = row["observed"]
        if row["row"] == "(i)":
            return (f"em_cw {_fmt_pt(o['em_cw'])}; "
                    f"ez_cw {_fmt_pt(o['ez_cw'])}")
        if row["row"] == "(ii)":
            return (f"gated {_fmt_pt(o['gated_pooled'])}; "
                    f"ungated {_fmt_pt(o['ungated_pooled'])}; leak strengths "
                    f"{o['leak_hedge_strengths'] or '—'}")
        if row["row"] == "(iii)":
            return (f"promo em_cg {_fmt_pt(o['em_cg_promote_within_window'])}"
                    f" vs ez_cg {_fmt_pt(o['ez_cg_promote_within_window'])}; "
                    f"mapped em {_fmt_pt(o['em_hedge_mapped_resolve_true'])} "
                    f"vs ez {_fmt_pt(o['ez_hedge_mapped_resolve_true'])}")
        def _d(v):
            return f"{v:+.3f}" if v is not None else "—"
        return (f"delta_em {_d(o['em']['delta'])}; "
                f"delta_ez {_d(o['ez']['delta'])}")

    rel_dir = {r["row"]: r for r in released["m4"]["directional_table"]}
    for row in claude["m4"]["directional_table"]:
        rel = rel_dir.get(row["row"], {})
        card["m4"].append({
            "row": f"M4-{row['row']}", "expectation": row["expectation"],
            "released_holds": rel.get("holds"),
            "released_observed": _m4_obs(rel) if rel else None,
            "observed": _m4_obs(row),
            "claude_holds": row["holds"],
            "replicates": row["holds"] == rel.get("holds")})
    card["m4"].append({
        "row": "M4-entry-fix",
        "expectation": released["m4"]["entry_fix_comparison"]["expectation"],
        "released_holds": released["m4"]["entry_fix_comparison"]["holds"],
        "released_observed": json.dumps(
            released["m4"]["entry_fix_comparison"]["sub_verdicts"]),
        "observed": json.dumps(
            claude["m4"]["entry_fix_comparison"]["sub_verdicts"]),
        "claude_holds": claude["m4"]["entry_fix_comparison"]["holds"],
        "claude_sub_verdicts": claude["m4"]["entry_fix_comparison"]["sub_verdicts"],
        "replicates": (claude["m4"]["entry_fix_comparison"]["holds"]
                       == released["m4"]["entry_fix_comparison"]["holds"])})

    rel_rows = {r["row"]: r for r in d1_rows(released["d1"]["endpoints"])}
    for row in d1_rows(claude["d1"]["endpoints"]):
        rel = rel_rows[row["row"]]
        card["d1"].append({
            **row, "released_observed": rel["observed"],
            "released_holds": rel["holds"],
            "replicates": row["holds"] == rel["holds"]})
    rel_m2 = {r["row"]: r for r in m2_rows(released["m2"], "em_cg_fixed")}
    for row in m2_rows(claude["m2"], "em_cg_fixed"):
        rel = rel_m2[row["row"]]
        card["m2_emcg"].append({
            **row, "released_observed": rel["observed"],
            "released_holds": rel["holds"],
            "replicates": row["holds"] == rel["holds"]})
    return card


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _card_table(rows: list[dict], extra_cols: tuple[str, ...] = ()) -> list[str]:
    lines = ["| row | expectation | gpt-5.5 (released) | claude-opus-4-8 | "
             "observed (claude) | replicates |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        def v(h):
            return "—" if h is None else ("HOLDS" if h else "does not hold")
        lines.append(
            f"| {r['row']} | {r['expectation']} | {v(r.get('released_holds'))} "
            f"| {v(r['holds']) if 'holds' in r else v(r.get('claude_holds'))} "
            f"| {r.get('observed', '')} "
            f"| {'YES' if r.get('replicates') else 'NO'} |")
    return lines


def render_report(out: dict) -> str:
    card = out["scorecard"]
    tf = out["trace_forensics"]
    lines = [
        "# Different-family replication — Claude Opus 4.8 systems, frozen "
        "gpt-5.5 judge",
        "",
        f"Preregistered 2026-08-11 (protocol/CHANGELOG.md) + owner amendment "
        f"same day. Generated {out['generated_at']}. Systems: `{CLAUDE_MODEL}` "
        "(alias pin; response.model echoed per call — deviation (a)); "
        f"equivalence judge pinned `{out['judge_model']}` on both families "
        "(deviation (d)); thinking parity: adaptive thinking + output effort "
        "medium (deviation (c)). JSON-mode parity (amendment replacing "
        "deviation (b)): BLOCKED — the Anthropic Messages API exposes only "
        "strict `json_schema` structured output (mandatory "
        "`additionalProperties:false`), with no unconstrained JSON-object "
        "mode equivalent to the GPT arm's `json_object`; the amendment's "
        "permissive-object fallback 400s, so the run used prompt-instructed "
        "JSON + parse-inside-retry + fence-strip (see JSON-mode note below). "
        "Deviation (e) DROPPED per amendment (both families replicate on the "
        "identical dataset; the PROVENANCE disclosure of the 7 Fable-authored "
        "opener turns stands on its own).",
        "",
        "Substrates reused VERBATIM; one pass each. Per-substrate endpoint "
        "tables: `results/claude_repl/m4/m4_eval_report.md`, "
        "`results/claude_repl/d1/scores_d1.md`, "
        "`results/claude_repl/m2_emcg/m2_report.md`.",
        "",
        "## Run stats", "",
        "| phase | selected | records | failed | fresh calls |",
        "|---|---|---|---|---|"]
    for r in out["run_phases"]:
        lines.append(f"| {r['phase']} | {r['selected']} | "
                     f"{r['n_new'] + r['skipped']} | {len(r['failed'])} | "
                     f"{r['fresh_calls']:,} |")
    lines += [
        "",
        "## Directional scorecard vs the released gpt-5.5 numbers", "",
        "Rows are the released directional findings; 'replicates' = the "
        "claude verdict equals the released verdict (including released "
        "non-holds).", "",
        "### Internal M4 (60 conversations x 5 arms; em_cg_fixed = adopted "
        "configuration)", ""]
    lines += _card_table(card["m4"])
    lines += ["", "### External D1 (147 test conversations x 4 systems; "
              "consumed-slice reuse)", ""]
    lines += _card_table(card["d1"])
    lines += ["", "### Planted graded suite (420 cases; adopted "
              "cautious-gated cell em_cg_fixed only)", ""]
    lines += _card_table(card["m2_emcg"])
    lines += [
        "", "## JSON-mode mechanism (amendment (1) — BLOCKED by the API)", "",
        "The 2026-08-11 owner amendment asked the Claude arm to use the "
        "API-level JSON output facility, matching the GPT arm's `json_object` "
        "mode, and to fall back to the most permissive object schema "
        "(`{\"type\":\"object\"}`) if only schema-based structured output "
        "exists. Empirically the current Anthropic Messages API "
        "(`anthropic` 0.121.0, `claude-opus-4-8`):",
        "- `output_config.format.type` must be `json_schema` — `json_object` "
        "and `json` both 400 (`Input should be 'json_schema'`).",
        "- every object schema must set `additionalProperties:false` — bare "
        "`{\"type\":\"object\"}` 400s (`'additionalProperties' must be "
        "explicitly set to false`) and `additionalProperties:true` 400s "
        "(`not supported`).",
        "- the only accepted permissive object schema, "
        "`{\"type\":\"object\",\"additionalProperties\":false}`, admits ONLY "
        "the empty object `{}` — it cannot carry the frozen prompts' fields.",
        "",
        "So the amendment's specified fallback is not implementable, and "
        "there is no unconstrained JSON-object mode on this API. Per-prompt "
        "strict schemas were deliberately NOT substituted: they would "
        "constrain the Claude arm's outputs asymmetrically vs the GPT "
        "`json_object` arm, contradicting the amendment's \"no property "
        "constraints\" intent (parity in the wrong direction). The run "
        "therefore used the mechanism it was built with: the frozen prompts' "
        "explicit \"return only JSON\" instruction + `json.loads` inside the "
        "5-try retry loop + a single markdown-fence stripper. Across the "
        "~15.7k fresh Claude calls this produced 26 JSON parse-retries "
        "(~0.17%, ALL recovered within retries — 14 `Extra data` = a trailing "
        "second object the fence-strip doesn't catch but retry does; 10-12 "
        "empty/`Expecting value` from long-request timeouts) plus 12 "
        "transport timeouts and 1 per-item failure that recovered on a manual "
        "resume (0 net failures). Every one of the 627 stored records is "
        "valid JSON. This is a preregistration/API-capability conflict "
        "surfaced for the owner, NOT a result-motivated deviation.",
        "", "## Parse retries + served-model inventory", "",
        f"- json mechanism: {out.get('json_mechanism', JSON_MECHANISM)}",
        f"- response.model inventory (run window): "
        f"{json.dumps(tf.get('response_model_inventory', {}))}",
        f"- claude calls logged: {tf.get('claude_calls_logged'):,}; calls "
        f"needing >= 1 retry: {tf.get('claude_calls_needing_retry')}; "
        f"extra attempts total: {tf.get('claude_extra_attempts_total')} "
        f"(attempt histogram {json.dumps(tf.get('claude_attempt_histogram'))})",
        f"- {tf.get('retry_note', '')}",
        "", "## Budget (stage claude_repl; advisory / hard envelope / 110k "
        "replication stop)", "",
        f"- before: {json.dumps(out['budget_before']['spent'])} "
        f"(total {sum(out['budget_before']['spent'].values()):,})",
        f"- after:  {json.dumps(out['budget_after']['spent'])} "
        f"(total {sum(out['budget_after']['spent'].values()):,} / envelope "
        f"{out['budget_after']['envelope']:,}; replication stop "
        f"{HARD_STOP_TOTAL:,})",
        "",
        "Verification (RECOUNT + SEMANTICS) precedes release, as all prior "
        "runs — these numbers are UNVERIFIED until then.", ""]
    return "\n".join(lines)


async def score() -> dict:
    assert_judge_pinned()
    claude = {"m4": await score_m4(), "d1": await score_d1(),
              "m2": await score_m2()}
    released = {"d1": json.loads(RELEASED_D1.read_text()),
                "m4": json.loads(RELEASED_M4.read_text()),
                "m2": json.loads(RELEASED_M2.read_text())}
    card = build_scorecard(claude, released)
    phases = [json.loads(l) for l in RUN_LOG.open()] if RUN_LOG.exists() else []
    out = {
        "id": "CLAUDE-REPL-RESULTS",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prereg": "protocol/CHANGELOG.md 2026-08-11 different-family "
                  "replication (Claude Opus 4.8 systems, frozen gpt-5.5 "
                  "judge)",
        "system_model": CLAUDE_MODEL,
        "judge_model": llm.JUDGE_MODEL,
        "json_mechanism": JSON_MECHANISM,
        "run_phases": phases,
        "scorecard": card,
        "trace_forensics": trace_forensics(),
        "budget_before": (json.loads(BUDGET_BEFORE.read_text())
                          if BUDGET_BEFORE.exists() else None),
        "budget_after": mapper_fix.ensure_budget(BUDGET_PATH),
        "per_substrate_results": {
            "m4": str(M4_OUT / "m4_eval_results.json"),
            "d1": str(D1_ROOT / "scores_d1.json"),
            "m2_emcg": str(M2_OUT / "m2_results.json")},
        "released_baselines": {"m4": str(RELEASED_M4), "d1": str(RELEASED_D1),
                               "m2": str(RELEASED_M2)},
    }
    CR_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    REPORT_PATH.write_text(render_report(out))
    print(f"wrote {RESULTS_PATH}\nwrote {REPORT_PATH}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run-m4", "run-d1", "run-m2"):
        p = sub.add_parser(name)
        p.add_argument("--ids", nargs="*", default=None)
        p.add_argument("--shard", default=None, help="K/N over sorted ids")
        p.add_argument("--call-cap", type=int, default=None)
    sub.add_parser("score")
    sub.add_parser("status")
    args = ap.parse_args()

    if args.cmd == "status":
        print(json.dumps(advisory_status(), indent=1))
        return
    if args.cmd == "score":
        asyncio.run(score())
        return
    fn = {"run-m4": run_m4, "run-d1": run_d1, "run-m2": run_m2}[args.cmd]
    summary = asyncio.run(fn(args.ids, args.shard, args.call_cap))
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
