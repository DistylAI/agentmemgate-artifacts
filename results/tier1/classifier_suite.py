"""Tier-1 reviewer-response items 3 + 4 (preregistration: protocol/CHANGELOG.md
2026-08-14 "PREREGISTRATION: reviewer-response experiment set (Tier 1)").

ITEM 3 — CLASSIFIER STABILITY SUITE on the 163 natural statements
(results/test1/eval_items.jsonl, provenance == "natural"; gold = the finalized
labels re-derived and asserted per results/analysis/classifier_standalone_eval):

  (a) Re-analysis of the FROZEN v1 Test-1 classifier outputs (0 LLM calls):
      bootstrap 95% CI (10,000 statement-level resamples, seed 20260702,
      percentile) for the released AUROC 0.928388 of s_ev vs the binary
      should-update label; macro-F1 (point + bootstrap CI, same resamples);
      four-class argmax accuracy (must reproduce the released 69.9%);
      per-class precision/recall (must reproduce the released values,
      asserted against results/analysis/classifier_standalone_eval.json);
      and the two routing-critical error rates with Wilson CIs:
        speculation-committed  = gold speculative, argmax in COMMIT_CLASSES
        valid-update-held      = gold completed/correction, argmax speculative

  (b) Stochasticity: 5 repeated classifications per statement (815 fresh
      calls) with the frozen classifier prompt (prompts/classifier_short.txt),
      pinned gpt-5.5-2026-04-23, the pipeline's normal sampling settings
      (llm.call_llm: reasoning effort medium, JSON-object output; the
      Responses call exposes no temperature/seed parameter).

      STOCHASTICITY DESIGN (prereg item 3 "fresh calls" branch): the pipeline
      is DETERMINISTIC-BY-CACHE — llm.call_llm content-addresses every call by
      sha256(model, effort, system_prompt, user_prompt), so a repeat with the
      byte-identical frozen prompt replays the frozen output for free. The
      established bypass in THIS pipeline is call_llm(use_cache=False), the
      exact mechanism of the v1 FM-12 self-consistency rerun
      (experiments/v1_pilot/eval_classifier.py cmd_self_consistency ->
      classify.classify_item(it, use_cache=False)). The unify `retry_cycle`
      salt is a payload-schema field of the unify GENERATION pipeline
      (unify_arcs.py render/plan payloads), not a mechanism of the frozen
      classifier packaging — salting the classifier user prompt would change
      the frozen prompt under measurement. We therefore measure true API
      sampling stochasticity with direct UNCACHED calls through the
      byte-identical frozen prompt (verified byte-identical: every
      reconstructed prompt cache-key hits the frozen v1 cache entry and its
      stored probs equal the frozen classifier block exactly). Fresh repeat
      outputs are persisted to classifier_suite_repeats.jsonl (resumable),
      never written into the shared frozen cache.

ITEM 4 — PAIRED PROMPT-OBJECTIVE BASELINES on exactly the same 163 statements,
same model (supersedes the unpaired 0.93 vs 0.69/0.62 comparison):
  - "is it true now": the frozen extraction-embedded confidence
    (prompts/extractor.txt: probability the tuple "is true of the entity's
    current, real-world state") — per-item scores are the frozen s_conf field
    of eval_items.jsonl (historical per-item outputs REUSED; 0 fresh calls).
  - "should it commit": the dedicated commit-confidence probe (frozen SYSTEM
    prompt of src/memory_gating/analysis/exploratory_commit_confidence.py) —
    per-item scores recovered from the content-addressed cache written by the
    v1 exploratory run (historical per-item outputs REUSED where present;
    fresh calls only for cache misses, expected 0/163).
  Paired AUROC differences (evidence-type s_ev minus each) with
  cluster-bootstrap percentile CIs over statements
  (stats.paired_bootstrap_delta_auc, 10k resamples, seed 20260702; natural
  statements are singleton clusters per the frozen eval-set convention, so the
  cluster bootstrap is exactly statement-level resampling).

BUDGET: stage "tier1" in results/v2.4/budget.json (SHARED with the Tier-1
retry-ablation runner; flock-guarded registration + per-fresh-call accounting
via forecast_v22.SharedCallBudget). Advisory stage per the 2026-07-08
amendment; the 120k envelope is the hard stop; this runner additionally caps
its own fresh calls at LOCAL_CAP.

CLI (repo root; MEMGATE_SYSTEM_MODEL must be UNSET):
  pilot/.venv/bin/python results/tier1/classifier_suite.py run      # full
  pilot/.venv/bin/python results/tier1/classifier_suite.py analyze  # 0 calls
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src" / "memory_gating",
           ROOT / "experiments" / "v1_pilot",
           ROOT / "experiments" / "v2_full"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import classify  # noqa: E402
import forecast_v22  # noqa: E402
import label  # noqa: E402
import llm  # noqa: E402
import stats  # noqa: E402
from label import CLASSES, COMMIT_CLASSES  # noqa: E402
from paths import RESULTS_DIR  # noqa: E402

STAGE = "tier1"
PINNED_SYSTEM_MODEL = "gpt-5.5-2026-04-23"
N_NATURAL = 163
N_REPEATS = 5
N_BOOT = 10_000
SEED = stats.DEFAULT_SEED  # 20260702
LOCAL_CAP = 2_500          # this runner's own fresh-call stop (task estimate)
#: tier1 advisory registration is headroom-limited (4,000 remaining at prereg
#: time); the prereg's +5-8k estimate covers the WHOLE Tier-1 set — stages are
#: advisory, only the 120k envelope is hard.
ADVISORY_FORECAST = 4_000

V24_DIR = RESULTS_DIR / "v2.4"
BUDGET_PATH = V24_DIR / "budget.json"
TIER1_DIR = RESULTS_DIR / "tier1"
EVAL_ITEMS_PATH = RESULTS_DIR / "test1" / "eval_items.jsonl"
RESULTS_V1_PATH = RESULTS_DIR / "test1" / "results.json"
STANDALONE_EVAL_PATH = RESULTS_DIR / "analysis" / "classifier_standalone_eval.json"
COMMITCONF_RELEASED_PATH = RESULTS_DIR / "test1" / "exploratory_commit_confidence.json"
NATURAL_GOLD_PATH = RESULTS_DIR / "labels" / "natural_gold.jsonl"
ADJUDICATIONS_PATH = RESULTS_DIR / "labels" / "adjudications.jsonl"

REPEATS_PATH = TIER1_DIR / "classifier_suite_repeats.jsonl"
OUT_JSON = TIER1_DIR / "classifier_suite_results.json"
OUT_MD = TIER1_DIR / "classifier_suite_report.md"
BUDGET_BEFORE = TIER1_DIR / "classifier_suite_budget_before.json"

#: The frozen dedicated commit-confidence probe, byte-identical to the v1
#: exploratory run (src/memory_gating/analysis/exploratory_commit_confidence.py
#: SYSTEM / render_item) — verified: all 163 reconstructed cache keys hit the
#: historical cache and the recovered AUC equals the released 0.6211835334476844.
COMMITCONF_SYSTEM = """You score memory updates for a personal assistant. You are given one user statement from a conversation (with brief context), one extracted (entity, attribute, value) tuple from it, and the currently stored value for that attribute if one exists.

Question: what is the probability that this statement should, right now, update the entity's canonical CURRENT state for this attribute to the asserted value? Consider alternative readings before answering — including that the change has not happened yet, may never happen, is about the past, or does not concern current state at all — and weigh how likely each is. A statement can be faithfully extracted and still not warrant updating current state.

Return only JSON: {"commit_probability": <integer 0-100>}"""


def commitconf_user_prompt(it: dict) -> str:
    ctx = "\n".join(f"[{t['role']}] {t['content']}" for t in it["preceding_turns"])
    stored = it["stored_value"] if it["stored_value"] not in (None, "None") else "(none)"
    return (f"Preceding turns:\n{ctx}\n\nSource user turn:\n{it['turn_text']}\n\n"
            f"Target entity: {it['entity']}\nTarget attribute: {it['attribute']}\n"
            f"Asserted value: {it['asserted_value']}\n"
            f"Stored value for this (entity, attribute): {stored}")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def wilson(x: int, n: int) -> dict:
    lo, hi = stats.wilson_ci(x, n)
    return {"x": x, "n": n, "point": x / n, "wilson_lo": lo, "wilson_hi": hi}


def load_natural() -> list[dict]:
    nat = [r for r in _jsonl(EVAL_ITEMS_PATH) if r["provenance"] == "natural"]
    assert len(nat) == N_NATURAL, f"expected {N_NATURAL} natural rows, got {len(nat)}"
    # label provenance: re-derive finalized gold and assert identity (the
    # classifier_standalone_eval provenance check, repeated here verbatim)
    adjudications = label.read_adjudications(ADJUDICATIONS_PATH)
    rederived = {}
    for row in _jsonl(NATURAL_GOLD_PATH):
        if row.get("extraction_miss"):
            continue
        rederived[row["item_id"]] = label.finalize_label(row["labeling"], adjudications)
    for r in nat:
        fin = rederived.get(r["item_id"])
        assert fin is not None and fin["class"] == r["gold"], (
            f"gold provenance mismatch for {r['item_id']}")
    return nat


def frozen_prompt_reproduction(nat: list[dict]) -> dict:
    """Verify the reconstructed frozen classifier prompt is byte-identical to
    the v1 run: every cache key (model, effort, system, user) must hit the
    frozen cache and the stored probs must equal the frozen classifier block."""
    sysp = label.load_prompt(classify.CLASSIFIER_PROMPT_NAME)
    hits = probs_match = 0
    for it in nat:
        key = llm._cache_key(llm.MODEL, llm.REASONING_EFFORT, sysp,
                             label.format_item_prompt(it))
        p = llm.CACHE_DIR / f"{key}.json"
        if not p.exists():
            continue
        hits += 1
        probs, _ = classify.validate_probs(json.loads(p.read_text())["output"]["probs"])
        if all(abs(probs[c] - it["classifier"]["probs"][c]) < 1e-9 for c in CLASSES):
            probs_match += 1
    ok = hits == probs_match == len(nat)
    assert ok, f"frozen prompt reconstruction failed: hits={hits} match={probs_match}"
    return {"cache_key_hits": hits, "probs_match_frozen": probs_match,
            "n": len(nat), "byte_identical": ok}


# ---------------------------------------------------------------------------
# Item 3(a): re-analysis of the frozen outputs
# ---------------------------------------------------------------------------

def macro_f1_from(golds: list[str], preds: list[str]) -> float:
    """Macro-F1 over classes with any gold or predicted mass (F1_c =
    2*TP_c / (n_gold_c + n_pred_c)); classes absent from a (re)sample are
    skipped in the macro mean."""
    f1s = []
    for cls in CLASSES:
        tp = sum(1 for g, p in zip(golds, preds) if g == cls and p == cls)
        denom = sum(1 for g in golds if g == cls) + sum(1 for p in preds if p == cls)
        if denom:
            f1s.append(2 * tp / denom)
    return float(np.mean(f1s))


def analyze_frozen(nat: list[dict]) -> dict:
    golds = [r["gold"] for r in nat]
    preds = [r["classifier"]["argmax"] for r in nat]
    y = np.array([1 if g in COMMIT_CLASSES else 0 for g in golds])
    s_ev = np.array([r["classifier"]["s_ev"] for r in nat], dtype=float)

    # point estimates
    auc_ev = stats.mann_whitney_auc(y.tolist(), s_ev.tolist())
    n_correct = sum(g == p for g, p in zip(golds, preds))
    accuracy = wilson(n_correct, len(nat))
    macro_f1 = macro_f1_from(golds, preds)

    per_class = {}
    for cls in CLASSES:
        tp = sum(1 for g, p in zip(golds, preds) if g == cls and p == cls)
        n_gold = golds.count(cls)
        n_pred = preds.count(cls)
        per_class[cls] = {
            "support": n_gold, "predicted": n_pred,
            "precision": wilson(tp, n_pred) if n_pred else
                {"x": tp, "n": 0, "point": None, "wilson_lo": None, "wilson_hi": None},
            "recall": wilson(tp, n_gold),
            "f1": 2 * tp / (n_gold + n_pred) if n_gold + n_pred else None,
        }

    # one bootstrap loop, shared resamples: AUROC + macro-F1 (10k, seed 20260702,
    # statement-level i.i.d. resampling, percentile CI; single-class resamples
    # skipped for AUROC only — stats.paired_bootstrap_delta_auc convention)
    golds_arr = np.array(golds)
    preds_arr = np.array(preds)
    rng = np.random.default_rng(SEED)
    n = len(nat)
    auc_bs, f1_bs = [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() != yb.max():
            auc_bs.append(stats._auc(yb, s_ev[idx]))
        f1_bs.append(macro_f1_from(golds_arr[idx].tolist(), preds_arr[idx].tolist()))
    auc_lo, auc_hi = np.quantile(auc_bs, [0.025, 0.975])
    f1_lo, f1_hi = np.quantile(f1_bs, [0.025, 0.975])

    # routing-critical error rates (argmax operating point)
    spec_committed = [r["item_id"] for r in nat if r["gold"] == "speculative"
                      and r["classifier"]["argmax"] in COMMIT_CLASSES]
    n_spec = golds.count("speculative")
    held = [r["item_id"] for r in nat if r["gold"] in COMMIT_CLASSES
            and r["classifier"]["argmax"] == "speculative"]
    n_valid = sum(golds.count(c) for c in COMMIT_CLASSES)

    # reproduction asserts vs the released standalone-eval numbers
    released = json.loads(STANDALONE_EVAL_PATH.read_text())
    rel_v1 = json.loads(RESULTS_V1_PATH.read_text())
    rel_auc = rel_v1["metrics"]["natural_vs_injected_FM1"]["natural"]["auc_ev"]
    assert abs(auc_ev - rel_auc) < 1e-12, (auc_ev, rel_auc)
    assert accuracy["x"] == released["overall_accuracy"]["x"] == 114
    assert abs(accuracy["point"] - released["overall_accuracy"]["point"]) < 1e-12
    for cls in CLASSES:
        rel_pc = released["per_class"][cls]
        assert per_class[cls]["recall"]["x"] == rel_pc["recall"]["x"]
        assert per_class[cls]["recall"]["n"] == rel_pc["recall"]["n"]
        assert per_class[cls]["precision"]["x"] == rel_pc["precision"]["x"]
        assert per_class[cls]["precision"]["n"] == rel_pc["precision"]["n"]

    return {
        "n": len(nat),
        "auroc_s_ev": {
            "point": auc_ev, "released": rel_auc,
            "bootstrap_ci95": [float(auc_lo), float(auc_hi)],
            "n_boot": N_BOOT, "n_boot_used_auc": len(auc_bs), "seed": SEED,
            "scheme": "statement-level i.i.d. resampling, percentile CI; "
                      "single-class resamples skipped",
        },
        "macro_f1": {
            "point": macro_f1,
            "bootstrap_ci95": [float(f1_lo), float(f1_hi)],
            "n_boot": N_BOOT, "seed": SEED,
            "definition": "mean over classes of 2*TP_c/(n_gold_c + n_pred_c); "
                          "classes absent from a resample (no gold, no pred) "
                          "skipped in that resample's mean",
        },
        "four_class_accuracy": accuracy,
        "per_class": per_class,
        "routing_critical": {
            "speculation_committed": {
                **wilson(len(spec_committed), n_spec),
                "definition": "gold speculative, argmax in "
                              "{completed_event, explicit_correction}",
                "item_ids": sorted(spec_committed),
            },
            "valid_update_held": {
                **wilson(len(held), n_valid),
                "definition": "gold completed_event/explicit_correction, "
                              "argmax speculative",
                "item_ids": sorted(held),
            },
        },
        "reproduction_check": {
            "auroc_equals_released": True,
            "accuracy_equals_released_114_of_163": True,
            "per_class_pr_equals_released": True,
        },
    }


# ---------------------------------------------------------------------------
# Item 3(b): 5 uncached repeats per statement
# ---------------------------------------------------------------------------

def load_repeats() -> dict[tuple[str, int], dict]:
    return {(r["item_id"], r["repeat_idx"]): r for r in _jsonl(REPEATS_PATH)}


async def run_repeats(nat: list[dict]) -> None:
    done = load_repeats()
    todo = [(it, k) for it in nat for k in range(1, N_REPEATS + 1)
            if (it["item_id"], k) not in done]
    if not todo:
        print("[repeats] all present; 0 calls")
        return
    print(f"[repeats] {len(todo)} calls to make ({len(done)} already present)")
    TIER1_DIR.mkdir(parents=True, exist_ok=True)
    write_lock = asyncio.Lock()

    async def one(it: dict, k: int) -> None:
        out = await classify.classify_item(it, use_cache=False)
        row = {"item_id": it["item_id"], "repeat_idx": k, **out}
        async with write_lock:
            with REPEATS_PATH.open("a") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    for sweep in range(1, 4):  # up to 3 sweeps over failures (resumable)
        results = await asyncio.gather(*(one(it, k) for it, k in todo),
                                       return_exceptions=True)
        failures = [(t, r) for t, r in zip(todo, results) if isinstance(r, Exception)]
        if not failures:
            return
        print(f"[repeats sweep {sweep}] {len(failures)} failures; "
              f"first: {failures[0][1]!r}")
        todo = [t for t, _ in failures]
    raise RuntimeError(f"{len(todo)} repeat calls still failing after 3 sweeps")


def analyze_repeats(nat: list[dict]) -> dict:
    done = load_repeats()
    missing = [(it["item_id"], k) for it in nat for k in range(1, N_REPEATS + 1)
               if (it["item_id"], k) not in done]
    assert not missing, f"{len(missing)} repeat outputs missing; run `run` first"

    y = [1 if r["gold"] in COMMIT_CLASSES else 0 for r in nat]
    unanimous = 0
    unanimous_commit = 0
    pairwise = []
    all5_match_frozen = 0
    majority_match_frozen = 0
    call_match_frozen = 0
    sd_list, mean_list = [], []
    renorm = 0
    per_repeat_auc = []
    mean_sev_by_item = {}

    for it in nat:
        reps = [done[(it["item_id"], k)] for k in range(1, N_REPEATS + 1)]
        labels = [r["argmax"] for r in reps]
        sevs = [r["s_ev"] for r in reps]
        renorm += sum(1 for r in reps if r.get("renormalized"))
        unanimous += len(set(labels)) == 1
        unanimous_commit += len({l in COMMIT_CLASSES for l in labels}) == 1
        agree = [a == b for a, b in combinations(labels, 2)]
        pairwise.append(sum(agree) / len(agree))
        frozen = it["classifier"]["argmax"]
        all5_match_frozen += all(l == frozen for l in labels)
        call_match_frozen += sum(l == frozen for l in labels)
        majority_match_frozen += Counter(labels).most_common(1)[0][0] == frozen
        mean_list.append(float(np.mean(sevs)))
        sd_list.append(float(np.std(sevs, ddof=1)))
        mean_sev_by_item[it["item_id"]] = float(np.mean(sevs))

    for k in range(1, N_REPEATS + 1):
        s = [done[(it["item_id"], k)]["s_ev"] for it in nat]
        per_repeat_auc.append(stats.mann_whitney_auc(y, s))
    auc_of_mean = stats.mann_whitney_auc(
        y, [mean_sev_by_item[it["item_id"]] for it in nat])

    frozen_sev = [it["classifier"]["s_ev"] for it in nat]
    return {
        "n_statements": len(nat), "n_repeats": N_REPEATS,
        "fresh_calls_design": "classify.classify_item(use_cache=False) — the "
                              "established FM-12 self-consistency bypass; "
                              "byte-identical frozen prompt, true API sampling "
                              "stochasticity (no salt added to the prompt)",
        "label_agreement": {
            "fraction_unanimous_argmax": wilson(unanimous, len(nat)),
            "fraction_unanimous_commit_decision": wilson(unanimous_commit, len(nat)),
            "mean_pairwise_argmax_agreement": float(np.mean(pairwise)),
        },
        "agreement_with_frozen_run": {
            "all_5_repeats_match_frozen_argmax": wilson(all5_match_frozen, len(nat)),
            "majority_matches_frozen_argmax": wilson(majority_match_frozen, len(nat)),
            "per_call_match_frozen_argmax": wilson(call_match_frozen,
                                                   len(nat) * N_REPEATS),
        },
        "s_ev_commit_probability_mass": {
            "mean_of_per_statement_means": float(np.mean(mean_list)),
            "frozen_run_mean": float(np.mean(frozen_sev)),
            "per_statement_sd": {
                "mean": float(np.mean(sd_list)),
                "median": float(np.median(sd_list)),
                "p90": float(np.quantile(sd_list, 0.9)),
                "max": float(np.max(sd_list)),
                "fraction_zero": float(np.mean([s == 0 for s in sd_list])),
            },
        },
        "auroc_across_repeats": {
            "per_repeat": per_repeat_auc,
            "mean": float(np.mean(per_repeat_auc)),
            "sd": float(np.std(per_repeat_auc, ddof=1)),
            "min": float(np.min(per_repeat_auc)),
            "max": float(np.max(per_repeat_auc)),
            "auroc_of_mean_s_ev": auc_of_mean,
            "frozen_run_auroc": stats.mann_whitney_auc(y, frozen_sev),
        },
        "renormalized_outputs": renorm,
    }


# ---------------------------------------------------------------------------
# Item 4: paired prompt-objective baselines
# ---------------------------------------------------------------------------

async def recover_commitconf(nat: list[dict]) -> tuple[dict[str, float], dict]:
    """Per-item dedicated commit-confidence scores: recovered from the
    historical content-addressed cache (v1 exploratory run) where present;
    fresh calls (cached thereafter) only for misses."""
    vals: dict[str, float] = {}
    misses: list[dict] = []
    for it in nat:
        key = llm._cache_key(llm.MODEL, llm.REASONING_EFFORT, COMMITCONF_SYSTEM,
                             commitconf_user_prompt(it))
        p = llm.CACHE_DIR / f"{key}.json"
        if p.exists():
            out = json.loads(p.read_text())["output"]
            vals[it["item_id"]] = max(0, min(100, int(out["commit_probability"]))) / 100.0
        else:
            misses.append(it)
    fresh = 0
    if misses:
        async def one(it: dict) -> None:
            out = await llm.call_llm(COMMITCONF_SYSTEM, commitconf_user_prompt(it),
                                     trace_tag=f"tier1-commitconf-{it['item_id']}")
            vals[it["item_id"]] = max(0, min(100, int(out["commit_probability"]))) / 100.0
        await asyncio.gather(*(one(it) for it in misses))
        fresh = len(misses)
    prov = {"recovered_from_cache": len(nat) - fresh, "fresh_calls": fresh,
            "note": "historical per-item outputs of the v1 exploratory run "
                    "reused via the content-addressed cache (byte-identical "
                    "frozen probe prompt)"}
    return vals, prov


def analyze_baselines(nat: list[dict], commitconf: dict[str, float],
                      provenance: dict) -> dict:
    y = [1 if r["gold"] in COMMIT_CLASSES else 0 for r in nat]
    clusters = [r["cluster"] for r in nat]  # singleton clusters (frozen convention)
    s_ev = [float(r["classifier"]["s_ev"]) for r in nat]
    s_truenow = [float(r["s_conf"]) for r in nat]
    s_commit = [commitconf[r["item_id"]] for r in nat]

    d_truenow = stats.paired_bootstrap_delta_auc(y, s_ev, s_truenow,
                                                 n_boot=N_BOOT, seed=SEED,
                                                 clusters=clusters)
    d_commit = stats.paired_bootstrap_delta_auc(y, s_ev, s_commit,
                                                n_boot=N_BOOT, seed=SEED,
                                                clusters=clusters)
    # reproduction vs released exploratory/natural numbers
    rel = json.loads(COMMITCONF_RELEASED_PATH.read_text())["natural"]
    assert abs(d_truenow["auc_b"] - rel["auc_conf_extraction_embedded"]) < 1e-12
    assert abs(d_commit["auc_b"] - rel["auc_commitconf_dedicated"]) < 1e-12
    assert abs(d_commit["auc_a"] - rel["auc_ev"]) < 1e-12

    return {
        "n": len(nat),
        "objectives": {
            "evidence_type": "s_ev = P(completed_event)+P(explicit_correction), "
                             "frozen classifier (prompts/classifier_short.txt)",
            "true_now": "extraction-embedded confidence (prompts/extractor.txt: "
                        "probability the tuple is true of current, real-world "
                        "state) — frozen s_conf per item",
            "should_commit": "dedicated commit-confidence probe (frozen "
                             "exploratory_commit_confidence.py SYSTEM prompt)",
        },
        "per_item_output_provenance": provenance,
        "auroc": {"evidence_type": d_commit["auc_a"],
                  "true_now": d_truenow["auc_b"],
                  "should_commit": d_commit["auc_b"]},
        "paired_delta_evidence_minus_true_now": {
            "delta": d_truenow["delta"], "ci95": [d_truenow["lo"], d_truenow["hi"]],
            "n_boot": N_BOOT, "n_boot_used": d_truenow["n_boot_used"], "seed": SEED,
        },
        "paired_delta_evidence_minus_should_commit": {
            "delta": d_commit["delta"], "ci95": [d_commit["lo"], d_commit["hi"]],
            "n_boot": N_BOOT, "n_boot_used": d_commit["n_boot_used"], "seed": SEED,
        },
        "bootstrap_scheme": "paired cluster bootstrap over statements "
                            "(stats.paired_bootstrap_delta_auc; natural items "
                            "are singleton clusters, so cluster resampling == "
                            "statement-level resampling), percentile CI",
        "reproduction_check": {
            "true_now_auc_equals_released_0_69": True,
            "should_commit_auc_equals_released_0_62": True,
            "evidence_auc_equals_released_0_93": True,
        },
    }


# ---------------------------------------------------------------------------
# Budget (stage tier1, shared with the retry-ablation runner)
# ---------------------------------------------------------------------------

def register_stage(path: Path = BUDGET_PATH) -> dict:
    """Idempotent flock-guarded registration of the SHARED tier1 stage (the
    plan_typed/claude_repl/unify convention). If another Tier-1 runner already
    registered the stage, accept it as-is (allocation may differ)."""
    lock_path = path.with_suffix(".lock")
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        budget = json.loads(path.read_text())
        if STAGE in budget["spent"]:
            return budget
        headroom = budget["allocation"].get("unallocated_headroom", 0)
        alloc = min(ADVISORY_FORECAST, headroom)
        assert alloc > 0, f"no unallocated headroom left to register {STAGE}"
        budget["allocation"]["unallocated_headroom"] = headroom - alloc
        budget["allocation"][STAGE] = alloc
        budget["spent"][STAGE] = 0
        budget.setdefault("notes", []).append(
            f"2026-08-14: tier1 {alloc} from headroom (reviewer-response "
            "Tier-1 experiment set, preregistration changelog 2026-08-14; "
            "stage SHARED by the retry-ablation and classifier-suite "
            "runners). Advisory stage per the 2026-07-08 amendment; the "
            "prereg's whole-set estimate is +5-8k with the 120k envelope "
            "hard; headroom covered only the registered amount.")
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(budget, indent=1, ensure_ascii=False))
        tmp.replace(path)
        return budget


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def pct(v: float | None) -> str:
    return "—" if v is None else f"{100 * v:.1f}%"


def ci_str(w: dict) -> str:
    if w.get("point") is None:
        return "—"
    return f"{pct(w['point'])} [{pct(w['wilson_lo'])}, {pct(w['wilson_hi'])}]"


def write_md(out: dict) -> None:
    a = out["item3a_frozen_reanalysis"]
    b = out["item3b_stochasticity"]
    c = out["item4_paired_baselines"]
    L = []
    L.append("# Tier-1 classifier suite: stability + paired prompt-objective baselines")
    L.append("")
    L.append("Preregistration: protocol/CHANGELOG.md 2026-08-14 (reviewer-response "
             "Tier 1), items 3 and 4. Substrate: the 163 natural statements of the "
             "frozen v1 Test-1 eval set (results/test1/eval_items.jsonl, provenance "
             "== 'natural') with finalized gold labels (provenance re-derived and "
             "asserted per results/analysis/classifier_standalone_eval.md).")
    L.append("")
    L.append("## Item 3(a) — frozen-output re-analysis (0 LLM calls)")
    L.append("")
    au = a["auroc_s_ev"]
    L.append(f"- AUROC of s_ev vs should-update: **{au['point']:.6f}** "
             f"(= released 0.928388), bootstrap 95% CI "
             f"**[{au['bootstrap_ci95'][0]:.3f}, {au['bootstrap_ci95'][1]:.3f}]** "
             f"({au['n_boot']:,} statement-level resamples, seed {au['seed']}; "
             f"{au['n_boot_used_auc']:,} usable).")
    mf = a["macro_f1"]
    L.append(f"- Macro-F1 (four-class argmax): **{mf['point']:.3f}**, bootstrap "
             f"95% CI [{mf['bootstrap_ci95'][0]:.3f}, {mf['bootstrap_ci95'][1]:.3f}] "
             "(same resamples).")
    acc = a["four_class_accuracy"]
    L.append(f"- Four-class accuracy: **{ci_str(acc)}** ({acc['x']}/{acc['n']}) — "
             "reproduces the released 69.9%.")
    L.append("")
    L.append("Per-class precision / recall (reproduce the released values exactly; "
             "95% Wilson CIs):")
    L.append("")
    L.append("| class | support | precision | recall | F1 |")
    L.append("|---|---|---|---|---|")
    for cls in CLASSES:
        p = a["per_class"][cls]
        L.append(f"| {cls} | {p['support']} | {ci_str(p['precision'])} | "
                 f"{ci_str(p['recall'])} | {p['f1']:.3f} |")
    L.append("")
    rc = a["routing_critical"]
    sc, vh = rc["speculation_committed"], rc["valid_update_held"]
    L.append("Routing-critical error rates (argmax operating point, Wilson CIs):")
    L.append("")
    L.append(f"- Speculation committed (gold speculative → argmax commit class): "
             f"**{ci_str(sc)}** ({sc['x']}/{sc['n']}; ids: "
             f"{', '.join(sc['item_ids']) or 'none'}).")
    L.append(f"- Valid update held (gold completed/correction → argmax speculative): "
             f"**{ci_str(vh)}** ({vh['x']}/{vh['n']}; ids: "
             f"{', '.join(vh['item_ids']) or 'none'}).")
    L.append("")
    L.append("## Item 3(b) — stochasticity (5 uncached repeats per statement)")
    L.append("")
    L.append("**Design.** The pipeline is deterministic-by-cache: llm.call_llm "
             "content-addresses calls by sha256(model, effort, system, user), so "
             "same-prompt repeats replay the frozen output. Repeats therefore "
             "bypass the cache via `classify_item(use_cache=False)` — the "
             "pipeline's established self-consistency mechanism (v1 FM-12), NOT a "
             "prompt salt: the unify `retry_cycle` salt belongs to the generation "
             "pipeline's payload schema, and salting the frozen classifier "
             "packaging would change the prompt under measurement. Prompts were "
             "verified byte-identical to the frozen v1 run (all 163 reconstructed "
             "cache keys hit the frozen cache with exactly the frozen probs). "
             "Settings: pinned gpt-5.5-2026-04-23, reasoning effort medium, "
             "JSON-object output — the Responses call exposes no temperature/seed "
             "parameter, so repeats measure true API sampling stochasticity.")
    L.append("")
    la, af = b["label_agreement"], b["agreement_with_frozen_run"]
    L.append(f"- Unanimous argmax across 5 repeats: **{ci_str(la['fraction_unanimous_argmax'])}** "
             f"({la['fraction_unanimous_argmax']['x']}/{la['fraction_unanimous_argmax']['n']}); "
             f"unanimous commit-vs-hold decision: "
             f"{ci_str(la['fraction_unanimous_commit_decision'])}.")
    L.append(f"- Mean pairwise argmax agreement: "
             f"**{la['mean_pairwise_argmax_agreement']:.3f}**.")
    L.append(f"- Agreement with the frozen run: per-call "
             f"{ci_str(af['per_call_match_frozen_argmax'])}; repeat-majority "
             f"{ci_str(af['majority_matches_frozen_argmax'])}; all-5 "
             f"{ci_str(af['all_5_repeats_match_frozen_argmax'])}.")
    sev = b["s_ev_commit_probability_mass"]
    sd = sev["per_statement_sd"]
    L.append(f"- Commit-probability mass s_ev: mean of per-statement means "
             f"{sev['mean_of_per_statement_means']:.3f} (frozen run "
             f"{sev['frozen_run_mean']:.3f}); per-statement SD mean "
             f"**{sd['mean']:.3f}**, median {sd['median']:.3f}, p90 {sd['p90']:.3f}, "
             f"max {sd['max']:.3f} (SD exactly 0 on {pct(sd['fraction_zero'])} of "
             "statements).")
    ar = b["auroc_across_repeats"]
    L.append(f"- AUROC across the 5 repeats: "
             f"{', '.join(f'{v:.4f}' for v in ar['per_repeat'])} — mean "
             f"**{ar['mean']:.4f}** (SD {ar['sd']:.4f}, range [{ar['min']:.4f}, "
             f"{ar['max']:.4f}]); frozen-run AUROC {ar['frozen_run_auroc']:.4f}; "
             f"AUROC of the 5-repeat mean s_ev {ar['auroc_of_mean_s_ev']:.4f}.")
    L.append(f"- Renormalized outputs among the {b['n_statements'] * b['n_repeats']} "
             f"repeat calls: {b['renormalized_outputs']}.")
    L.append("")
    L.append("## Item 4 — paired prompt-objective baselines (same 163 statements, "
             "same model)")
    L.append("")
    pr = c["per_item_output_provenance"]
    L.append(f"Historical per-item outputs REUSED: 'true now' = the frozen "
             f"extraction-embedded confidence (s_conf, eval_items.jsonl); 'should "
             f"commit' = the dedicated commit-confidence probe recovered from the "
             f"content-addressed cache of the v1 exploratory run "
             f"({pr['recovered_from_cache']}/163 recovered, {pr['fresh_calls']} "
             "fresh calls). Both reproduce the released unpaired AUROCs exactly.")
    L.append("")
    L.append("| objective | AUROC | paired Δ (evidence − alternative) | 95% CI |")
    L.append("|---|---|---|---|")
    L.append(f"| evidence-type (s_ev) | {c['auroc']['evidence_type']:.4f} | — | — |")
    dt = c["paired_delta_evidence_minus_true_now"]
    L.append(f"| 'is it true now' (extraction-embedded confidence) | "
             f"{c['auroc']['true_now']:.4f} | **{dt['delta']:+.4f}** | "
             f"[{dt['ci95'][0]:.4f}, {dt['ci95'][1]:.4f}] |")
    dc = c["paired_delta_evidence_minus_should_commit"]
    L.append(f"| 'should it commit' (dedicated probe) | "
             f"{c['auroc']['should_commit']:.4f} | **{dc['delta']:+.4f}** | "
             f"[{dc['ci95'][0]:.4f}, {dc['ci95'][1]:.4f}] |")
    L.append("")
    L.append(f"CIs: {c['bootstrap_scheme']}; {N_BOOT:,} resamples, seed {SEED}.")
    L.append("")
    L.append("## Methods and budget")
    L.append("")
    m = out["methods"]
    L.append(f"- **Reproduction checks:** {m['reproduction']}")
    L.append(f"- **Gold provenance:** {m['gold_provenance']}")
    L.append(f"- **Fresh LLM calls:** {m['fresh_llm_calls']} "
             f"(item 3b repeats {m['fresh_calls_repeats']}, item 4 "
             f"{m['fresh_calls_commitconf']}); budget stage '{STAGE}' "
             f"(results/v2.4/budget.json, flock-guarded SharedCallBudget, "
             f"advisory; 120k envelope hard).")
    bd = out["budget"]
    L.append(f"- **Budget delta:** this runner {bd['this_runner_fresh_calls']} "
             f"fresh calls; stage {STAGE} spend {bd['stage_spent_before']} → "
             f"{bd['stage_spent_after']} (stage SHARED with the concurrent "
             f"Tier-1 retry-ablation and plan-typed-D1 runners, whose in-window "
             f"spend the snapshots include); combined total "
             f"{bd['total_before']:,} → {bd['total_after']:,} of "
             f"{bd['envelope']:,} (hard envelope).")
    L.append(f"- **Repeat outputs:** {m['repeats_file']} (never written into the "
             "shared frozen cache).")
    L.append("")
    OUT_MD.write_text("\n".join(L))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=["run", "analyze"])
    args = parser.parse_args()

    assert os.environ.get("MEMGATE_SYSTEM_MODEL") is None, (
        "MEMGATE_SYSTEM_MODEL must be UNSET (pinned frozen-family run)")
    assert llm.MODEL == PINNED_SYSTEM_MODEL, llm.MODEL

    nat = load_natural()
    repro = frozen_prompt_reproduction(nat)
    item3a = analyze_frozen(nat)

    budget_before = json.loads(BUDGET_PATH.read_text()) if BUDGET_PATH.exists() else {}
    fresh_repeats = fresh_commitconf = 0
    if args.cmd == "run":
        register_stage()
        TIER1_DIR.mkdir(parents=True, exist_ok=True)
        if not BUDGET_BEFORE.exists():
            BUDGET_BEFORE.write_text(json.dumps(budget_before, indent=1))
        guard = forecast_v22.SharedCallBudget(STAGE, path=BUDGET_PATH,
                                              local_cap=LOCAL_CAP)
        guard.install()
        try:
            before = guard.fresh
            asyncio.run(run_repeats(nat))
            fresh_repeats = guard.fresh - before
            before = guard.fresh
            commitconf, prov = asyncio.run(recover_commitconf(nat))
            fresh_commitconf = guard.fresh - before
        finally:
            guard.uninstall()
    else:
        commitconf, prov = asyncio.run(recover_commitconf(nat))
        assert prov["fresh_calls"] == 0, "analyze mode must not need fresh calls"
        # re-analysis must not clobber the run's fresh-call/budget accounting
        if OUT_JSON.exists():
            prior = json.loads(OUT_JSON.read_text())
            fresh_repeats = prior["methods"]["fresh_calls_repeats"]
            fresh_commitconf = prior["methods"]["fresh_calls_commitconf"]
            budget_before = {"spent": {STAGE: prior["budget"]["stage_spent_before"],
                                       "_others": prior["budget"]["total_before"]
                                       - prior["budget"]["stage_spent_before"]}}

    item3b = analyze_repeats(nat)
    item4 = analyze_baselines(nat, commitconf, prov)

    budget_after = json.loads(BUDGET_PATH.read_text())
    out = {
        "endpoint": "Tier-1 reviewer-response items 3+4: classifier stability "
                    "suite + paired prompt-objective baselines",
        "preregistration": "protocol/CHANGELOG.md 2026-08-14 items 3 and 4",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": llm.MODEL,
        "frozen_prompt_reproduction": repro,
        "item3a_frozen_reanalysis": item3a,
        "item3b_stochasticity": item3b,
        "item4_paired_baselines": item4,
        "methods": {
            "reproduction": "frozen classifier prompt reconstructed and verified "
                            "byte-identical via cache-key hits with exact probs "
                            "equality (163/163); AUROC, accuracy and per-class "
                            "P/R asserted equal to the released values "
                            "(results/analysis/classifier_standalone_eval.json, "
                            "results/test1/results.json)",
            "gold_provenance": "finalized labels re-derived from "
                               "results/labels/{natural_gold,adjudications}.jsonl "
                               "via label.finalize_label and asserted identical "
                               "to eval_items.jsonl gold for all 163 items",
            "fresh_llm_calls": fresh_repeats + fresh_commitconf,
            "fresh_calls_repeats": fresh_repeats,
            "fresh_calls_commitconf": fresh_commitconf,
            "repeats_file": str(REPEATS_PATH.relative_to(ROOT)),
            "stochasticity_design": item3b["fresh_calls_design"],
        },
        "budget": {
            "stage": STAGE,
            "stage_spent_before": (budget_before.get("spent", {}) or {}).get(STAGE, 0),
            "stage_spent_after": budget_after["spent"].get(STAGE, 0),
            "total_before": sum((budget_before.get("spent") or {}).values()),
            "total_after": sum(budget_after["spent"].values()),
            "envelope": budget_after["envelope"],
            "this_runner_fresh_calls": fresh_repeats + fresh_commitconf,
            "shared_stage_note": "stage tier1 is SHARED by the concurrent "
                                 "Tier-1 runners (retry-ablation, plan-typed "
                                 "D1 held-out, this suite); before/after "
                                 "snapshots include their in-window spend — "
                                 "this runner's own fresh calls are counted "
                                 "exactly by its SharedCallBudget guard",
        },
    }
    TIER1_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    write_md(out)
    print(f"AUROC {item3a['auroc_s_ev']['point']:.6f} "
          f"CI {item3a['auroc_s_ev']['bootstrap_ci95']}; "
          f"unanimous {item3b['label_agreement']['fraction_unanimous_argmax']['x']}"
          f"/{N_NATURAL}; fresh calls {fresh_repeats + fresh_commitconf}")
    print(f"-> {OUT_JSON}\n-> {OUT_MD}")


if __name__ == "__main__":
    main()
