"""Test 1 runner: eval-set construction, classifier + confidence scoring, and
the preregistered metrics of protocol_spec.md v1.1 sections 3.3 / 4.2.

Subcommands (run from inside pilot/ with .venv/bin/python):
  python test1_eval.py label-natural        # gold-label the natural pool (section 2.2)
  python test1_eval.py run [--allow-pending]
  python test1_eval.py self-consistency     # 20% subsample rerun, use_cache=False

Inputs (produced by other owners; read as files, never imported):
  ../results/injections/manifest.jsonl      injection manifest (I2, INTERFACES schema)
  ../results/test1/extractor_tuples.jsonl   per-turn extractor output (I3's
                                            `extract.py produce-tuples`, R2-B-2; row
                                            schema in label.load_extractor_tuples)
Outputs:
  ../results/labels/natural_gold.jsonl, ../results/labels/escalation_queue.jsonl
  ../results/test1/eval_items.jsonl, ../results/test1/results.json,
  ../results/test1/self_consistency.json

Other owners' modules (data.py, stats.py, inject.py) are imported lazily inside
runner functions so the pure-logic functions here stay unit-testable while the
modules are implemented in parallel.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import _bootstrap  # noqa: F401  (R1 layout: puts src/ + experiments/ on sys.path)

import classify
import label
import llm
from label import (
    CLASSES,
    COMMIT_CLASSES,
    SEED,
    normalize_key_part,
    read_jsonl,
    session_content_hash,
)

logger = logging.getLogger(__name__)

from paths import RESEARCH_DIR, RESULTS_DIR
from paths import DATA_DIR as _REPO_DATA_DIR

DATA_DIR = _REPO_DATA_DIR / "longmemeval"

MANIFEST_PATH = RESULTS_DIR / "injections" / "manifest.jsonl"
EXTRACTOR_TUPLES_PATH = RESULTS_DIR / "test1" / "extractor_tuples.jsonl"
NATURAL_CANDIDATES_PATH = RESEARCH_DIR / "natural_candidates.jsonl"
NATURAL_CANDIDATES_OTHER_PATH = RESEARCH_DIR / "natural_candidates_other.jsonl"
NATURAL_GOLD_PATH = RESULTS_DIR / "labels" / "natural_gold.jsonl"
EVAL_ITEMS_PATH = RESULTS_DIR / "test1" / "eval_items.jsonl"
RESULTS_PATH = RESULTS_DIR / "test1" / "results.json"
SELF_CONSISTENCY_PATH = RESULTS_DIR / "test1" / "self_consistency.json"

DATE_FORMAT = "%Y/%m/%d (%a) %H:%M"

# Injection types that enter the Test 1 stream, with their gold class
# (injector_spec section 11; establishes and Test 3 rows are excluded).
TEST1_INJ_TYPES = {
    "spec_gold": "speculative",
    "spec_aux": "speculative",
    "caux_correct": "explicit_correction",
    "cgold_correct": "explicit_correction",
    "completed_topup": "completed_event",
    "hedged_recall": "other",
}

# Per-class eval-set minimums (protocol section 4.2, pinned decision P4).
CLASS_FLOORS = {
    "explicit_correction": 60,
    "speculative": 80,
    "completed_event": 70,
    "other": 50,
}
HEDGED_RECALL_MIN = 15
EDGE_CASE_QUOTA = 0.30

SELF_CONSISTENCY_FRACTION = 0.20


# ---------------------------------------------------------------------------
# Substrate resolution (oracle via data.py at runtime; s_cleaned loaded here
# because the INTERFACES data.py contract only covers the oracle file)
# ---------------------------------------------------------------------------

@dataclass
class SimpleSubstrate:
    """Duck-type of data.py's Substrate for s_cleaned-sourced questions:
    sessions/dates keep the ORIGINAL haystack order (sessions[i] is original
    session_idx i); session_order lists original indices chronologically."""
    question_id: str
    sessions: list[list[dict]]
    dates: list[str]
    session_order: list[int]


_S_CLEANED_INDEX: Optional[dict[str, dict]] = None


def _s_cleaned_index() -> dict[str, dict]:
    global _S_CLEANED_INDEX
    if _S_CLEANED_INDEX is None:
        rows = json.loads((DATA_DIR / "longmemeval_s_cleaned.json").read_text())
        _S_CLEANED_INDEX = {q["question_id"]: q for q in rows}
    return _S_CLEANED_INDEX


def s_cleaned_substrate(question_id: str) -> SimpleSubstrate:
    q = _s_cleaned_index()[question_id]
    sessions = [
        [{"role": t["role"], "content": t["content"]} for t in session]
        for session in q["haystack_sessions"]
    ]
    order = sorted(
        range(len(sessions)),
        key=lambda i: datetime.strptime(q["haystack_dates"][i], DATE_FORMAT),
    )
    return SimpleSubstrate(
        question_id=question_id,
        sessions=sessions,
        dates=list(q["haystack_dates"]),
        session_order=order,
    )


def substrate_for(source_file: str, question_id: str) -> Any:
    if source_file == "s_cleaned":
        return s_cleaned_substrate(question_id)
    import data  # I1's module (INTERFACES contract), resolved at runtime

    return data.get_substrate(question_id)


def session_by_orig_idx(substrate: Any, orig_session_idx: int) -> list[dict]:
    return substrate.sessions[orig_session_idx]


# ---------------------------------------------------------------------------
# Natural candidate pool
# ---------------------------------------------------------------------------

def parse_source(note: str) -> str:
    m = re.search(r"\[source=(oracle|s_cleaned)\]", note)
    if not m:
        raise ValueError(f"natural candidate note lacks a [source=...] tag: {note!r}")
    return m.group(1)


def load_natural_candidates() -> list[dict]:
    """BOTH harvest pools (R2-B-1): natural_candidates.jsonl (101 rows) and the
    `other`-class pool natural_candidates_other.jsonl (73 rows — hedged-recall,
    schedule-facts, assistant-world-knowledge corrections, plain additive
    facts, per §4.2/P7). P8's re-adjudication mandate extends to the new pool
    (changelog): every row passes the §2.2 pipeline; `kind` is a pool tag,
    never gold."""
    cands = read_jsonl(NATURAL_CANDIDATES_PATH) + read_jsonl(NATURAL_CANDIDATES_OTHER_PATH)
    for c in cands:
        c["source_file"] = parse_source(c["note"])
    return cands


# ---------------------------------------------------------------------------
# Manifest access
# ---------------------------------------------------------------------------

def load_manifest() -> list[dict]:
    rows = read_jsonl(MANIFEST_PATH)
    if not rows:
        sys.exit(f"missing or empty injection manifest: {MANIFEST_PATH} "
                 "(run inject.py plan+render first)")
    return rows


def non_t3_rows(manifest: list[dict], question_id: str) -> list[dict]:
    """Manifest rows spliced together for a Test-0/Test-1 substrate: every
    non-Test-3 injection of that question (Test 3 lives in clean copies)."""
    return [r for r in manifest
            if r["question_id"] == question_id
            and r["inj_type"] not in ("t3_staging", "t3_variant")]


def perturbed_substrate(manifest_rows: list[dict], question_id: str) -> Any:
    import data
    import inject  # I2's module: build_perturbed_substrate is INTERFACES-named

    return inject.build_perturbed_substrate(data.get_substrate(question_id), manifest_rows)


def natural_item_substrate(cand: dict, manifest: list[dict], cache: dict) -> Any:
    """Substrate a natural candidate is packaged from (R2-M-d): a candidate on
    a schema substrate (source oracle, question hosting non-T3 manifest rows)
    uses the PERTURBED substrate — the same splice the extractor saw — so the
    packaged turn text and preceding context match the extractor's information
    set. All other candidates use their unperturbed source substrate (the
    extractor saw those unperturbed too, R2-B-2)."""
    key = (cand["source_file"], cand["question_id"])
    if key not in cache:
        qid = cand["question_id"]
        rows = non_t3_rows(manifest, qid) if cand["source_file"] == "oracle" else []
        cache[key] = (perturbed_substrate(rows, qid) if rows
                      else substrate_for(cand["source_file"], qid))
    return cache[key]


# ---------------------------------------------------------------------------
# Eval-set construction: content-hash dedupe + ID-anchored rubric holdout
# ---------------------------------------------------------------------------

def dedupe_and_holdout(
    items: list[dict],
    anchor_turns: set[tuple[str, int]],
) -> tuple[list[dict], dict[str, list[str]]]:
    """Protocol section 2.3: apply the content-hash session dedupe first, then
    drop rubric-anchored turns keyed by (session content hash, turn_idx) —
    never by surface text. Each item must carry a "session_hash" field
    (hash of its UNPERTURBED source session).

    Dedupe key: (session_hash, turn_idx, entity, attribute) — the same
    underlying statement tuple surfacing under multiple question_ids enters
    once (first item in sorted item_id order wins). Injected items are unique
    by injection_id but are deduped and held out under the same rule.
    """
    dropped: dict[str, list[str]] = {"anchored": [], "duplicates": []}
    kept: list[dict] = []
    seen: set[tuple] = set()
    for item in sorted(items, key=lambda x: x["item_id"]):
        if (item["session_hash"], item["turn_idx"]) in anchor_turns:
            dropped["anchored"].append(item["item_id"])
            continue
        key = (
            item["session_hash"],
            item["turn_idx"],
            normalize_key_part(item["entity"]),
            normalize_key_part(item["attribute"]),
        )
        if key in seen:
            dropped["duplicates"].append(item["item_id"])
            continue
        seen.add(key)
        kept.append(item)
    return kept, dropped


def class_floor_shortfalls(counts: dict[str, int]) -> dict[str, dict]:
    out = {}
    for cls, floor in CLASS_FLOORS.items():
        n = counts.get(cls, 0)
        out[cls] = {"n": n, "floor": floor, "shortfall": max(0, floor - n)}
    return out


# ---------------------------------------------------------------------------
# Pure metric helpers (unit-tested with synthetic scores; stats.py math is
# I1's and is injected as a module argument)
# ---------------------------------------------------------------------------

def rankdata(values: list[float]) -> list[float]:
    """Average ranks (1-based), ties get the mean rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        mean_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = mean_rank
        i = j + 1
    return ranks


def spearman_rho(a: list[float], b: list[float]) -> float:
    """Spearman rank correlation (Pearson on average ranks)."""
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("spearman_rho needs two equal-length lists, n >= 2")
    ra, rb = rankdata(a), rankdata(b)
    ma = sum(ra) / len(ra)
    mb = sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra)
    vb = sum((y - mb) ** 2 for y in rb)
    if va == 0 or vb == 0:
        return 0.0
    return cov / math.sqrt(va * vb)


def youden_threshold(scores: list[float], y: list[int]) -> float:
    """Threshold tau maximizing Youden's J (TPR - FPR) for `commit iff
    score >= tau` on the tuning fold. Deterministic: smallest tau among ties.
    Candidates = every observed score plus one above the max (never-commit)."""
    n_pos = sum(y)
    n_neg = len(y) - n_pos
    candidates = sorted(set(scores)) + [max(scores) + 1.0]
    best_tau, best_j = candidates[0], -2.0
    for tau in candidates:
        tp = sum(1 for s, t in zip(scores, y) if s >= tau and t == 1)
        fp = sum(1 for s, t in zip(scores, y) if s >= tau and t == 0)
        tpr = tp / n_pos if n_pos else 0.0
        fpr = fp / n_neg if n_neg else 0.0
        j = tpr - fpr
        if j > best_j:
            best_tau, best_j = tau, j
    return best_tau


def cross_tuned_decisions(scores: list[float], y: list[int], seed: int = SEED) -> tuple[list[int], list[float]]:
    """2-fold cross-tuning of the confidence threshold in the baseline's favor
    (protocol section 3.3): tune tau maximizing Youden's J on fold A, decide
    fold B; swap; pool decisions. Returns (decisions aligned to input, taus)."""
    idx = list(range(len(scores)))
    rng = random.Random(f"{seed}-conf-folds")
    rng.shuffle(idx)
    fold_a, fold_b = idx[: len(idx) // 2], idx[len(idx) // 2:]
    decisions = [0] * len(scores)
    taus = []
    for tune, apply_ in ((fold_a, fold_b), (fold_b, fold_a)):
        tau = youden_threshold([scores[i] for i in tune], [y[i] for i in tune])
        taus.append(tau)
        for i in apply_:
            decisions[i] = 1 if scores[i] >= tau else 0
    return decisions, taus


def mcnemar_counts(y: list[int], dec_a: list[int], dec_b: list[int]) -> tuple[int, int]:
    """Discordant-pair counts: n01 = A correct & B wrong, n10 = B correct & A wrong."""
    n01 = sum(1 for t, a, b in zip(y, dec_a, dec_b) if (a == t) and (b != t))
    n10 = sum(1 for t, a, b in zip(y, dec_a, dec_b) if (b == t) and (a != t))
    return n01, n10


def sensitivity_score(argmax: str, max_prob: float) -> float:
    """M6 sensitivity ranking: argmax mapped to {0,1} with the max class
    probability from the same frozen {probs} output breaking ranks. Commit
    predictions rank above non-commit; within each side, more confident
    predictions are more extreme."""
    if argmax in COMMIT_CLASSES:
        return 0.5 + max_prob / 2
    return 0.5 - max_prob / 2


def redundancy_verdict(delta_lo: float, delta_hi: float, rho: float,
                       mcnemar_p: float, n_discordant: int) -> dict:
    """Preregistered redundancy verdict (protocol section 3.3): signals are
    redundant iff the delta-AUC 95% CI covers 0 AND rho > 0.7 AND McNemar is
    non-significant. n_d < 15 is itself redundancy evidence (section 4.2)."""
    ci_covers_zero = delta_lo <= 0.0 <= delta_hi
    mcnemar_ns = mcnemar_p >= 0.05
    return {
        "redundant": bool(ci_covers_zero and rho > 0.7 and mcnemar_ns),
        "delta_auc_ci_covers_zero": ci_covers_zero,
        "spearman_rho_gt_0.7": rho > 0.7,
        "mcnemar_nonsignificant": mcnemar_ns,
        "n_discordant": n_discordant,
        "n_discordant_lt_15_redundancy_evidence": n_discordant < 15,
    }


def _rate(x: int, n: int, wilson: Callable) -> dict:
    if n == 0:
        return {"x": x, "n": 0, "point": None, "wilson_lo": None, "wilson_hi": None}
    lo, hi = wilson(x, n)
    return {"x": x, "n": n, "point": x / n, "wilson_lo": lo, "wilson_hi": hi}


def per_class_pr(golds: list[str], preds: list[str], wilson: Callable) -> dict:
    out = {}
    for cls in CLASSES:
        tp = sum(1 for g, p in zip(golds, preds) if g == cls and p == cls)
        fp = sum(1 for g, p in zip(golds, preds) if g != cls and p == cls)
        fn = sum(1 for g, p in zip(golds, preds) if g == cls and p != cls)
        out[cls] = {
            "precision": _rate(tp, tp + fp, wilson),
            "recall": _rate(tp, tp + fn, wilson),
        }
    return out


def letter_tags(tags: list[str]) -> set[str]:
    return {t for t in tags if t in label.RULE_FLAGS}


def compute_metrics(records: list[dict], stats_mod: Any, seed: int = SEED,
                    n_boot: int = 10000) -> dict:
    """All preregistered Test 1 metrics over classified eval records.

    Each record: {item_id, gold, provenance, cluster, s_ev, s_conf, argmax,
    max_prob, confidence_raw, tags}. stats_mod provides the INTERFACES stats.py
    functions (wilson_ci, mann_whitney_auc, paired_bootstrap_delta_auc,
    exact_mcnemar).
    """
    wilson = stats_mod.wilson_ci
    y = [1 if r["gold"] in COMMIT_CLASSES else 0 for r in records]
    s_ev = [r["s_ev"] for r in records]
    s_conf = [r["s_conf"] for r in records]
    golds = [r["gold"] for r in records]
    preds = [r["argmax"] for r in records]

    # --- AUCs + paired bootstrap delta (statement-level primary; M9 cluster line)
    auc_ev = stats_mod.mann_whitney_auc(y, s_ev)
    auc_conf = stats_mod.mann_whitney_auc(y, s_conf)
    delta = stats_mod.paired_bootstrap_delta_auc(y, s_ev, s_conf, n_boot=n_boot, seed=seed)
    delta_cluster = stats_mod.paired_bootstrap_delta_auc(
        y, s_ev, s_conf, n_boot=n_boot, seed=seed,
        clusters=[r["cluster"] for r in records])

    # --- M6 sensitivity: argmax + max-prob ranking
    s_argmax = [sensitivity_score(r["argmax"], r["max_prob"]) for r in records]
    auc_argmax = stats_mod.mann_whitney_auc(y, s_argmax)

    # --- redundancy components
    rho = spearman_rho(s_ev, s_conf)
    conf_decisions, taus = cross_tuned_decisions(s_conf, y, seed=seed)
    ev_decisions = [1 if r["argmax"] in COMMIT_CLASSES else 0 for r in records]
    n01, n10 = mcnemar_counts(y, ev_decisions, conf_decisions)
    mcnemar_p = stats_mod.exact_mcnemar(n01, n10)

    # --- per-class P/R + the two primary per-class endpoints
    pr = per_class_pr(golds, preds, wilson)
    corr = pr["explicit_correction"]["recall"]
    correction_recall = {
        **corr,
        "n_ge_60": corr["n"] >= 60,
        "pass": (corr["point"] is not None and corr["point"] >= 0.9
                 and corr["wilson_lo"] >= 0.8),
    }
    spec_idx = [i for i, g in enumerate(golds) if g == "speculative"]
    licensed = sum(1 for i in spec_idx if preds[i] in COMMIT_CLASSES)
    lic = _rate(licensed, len(spec_idx), wilson)
    speculation_licensing = {
        **lic,
        "pass_10pct": lic["wilson_hi"] is not None and lic["wilson_hi"] <= 0.10,
        "pass_5pct_aspirational": lic["wilson_hi"] is not None and lic["wilson_hi"] <= 0.05,
    }

    # --- confidence histogram (FM-9 clustering check) over raw 0-100 ints
    conf_hist: dict[str, int] = {}
    for r in records:
        key = str(r["confidence_raw"])
        conf_hist[key] = conf_hist.get(key, 0) + 1

    # --- s_conf stratified by gold class (the redundancy diagnostic figure)
    conf_by_class = {}
    for cls in CLASSES:
        vals = sorted(r["s_conf"] for r in records if r["gold"] == cls)
        conf_by_class[cls] = {
            "n": len(vals),
            "median": vals[len(vals) // 2] if vals else None,
            "mean": sum(vals) / len(vals) if vals else None,
        }

    # --- FM-1 split: natural vs injected
    fm1 = {}
    for prov in ("natural", "injected"):
        sub = [i for i, r in enumerate(records) if r["provenance"] == prov]
        sub_y = [y[i] for i in sub]
        entry: dict[str, Any] = {"n": len(sub)}
        if len(set(sub_y)) == 2:
            entry["auc_ev"] = stats_mod.mann_whitney_auc(sub_y, [s_ev[i] for i in sub])
            entry["auc_conf"] = stats_mod.mann_whitney_auc(sub_y, [s_conf[i] for i in sub])
        else:
            entry["auc_ev"] = entry["auc_conf"] = None
        entry["per_class_recall"] = {
            cls: _rate(
                sum(1 for i in sub if golds[i] == cls and preds[i] == cls),
                sum(1 for i in sub if golds[i] == cls),
                wilson,
            )
            for cls in CLASSES
        }
        fm1[prov] = entry

    # --- FM-10 edge-case quota + per-tag accuracy (tags with n >= 10)
    n_edge = sum(1 for r in records if letter_tags(r["tags"]))
    tag_counts: dict[str, list[int]] = {}
    for r, g, p in zip(records, golds, preds):
        for t in r["tags"]:
            tag_counts.setdefault(t, []).append(1 if g == p else 0)
    per_tag = {
        t: {"n": len(v), "accuracy": sum(v) / len(v)}
        for t, v in sorted(tag_counts.items()) if len(v) >= 10
    }
    hedged_n = sum(
        1 for r in records
        if {"hedged_recall", "hedged_recall_wrongval"} & set(r["tags"]))

    return {
        "n": len(records),
        "n_natural": sum(1 for r in records if r["provenance"] == "natural"),
        "n_injected": sum(1 for r in records if r["provenance"] == "injected"),
        "class_counts": {cls: golds.count(cls) for cls in CLASSES},
        "class_floor_shortfalls": class_floor_shortfalls(
            {cls: golds.count(cls) for cls in CLASSES}),
        "hedged_recall_stratum": {"n": hedged_n, "min_required": HEDGED_RECALL_MIN,
                                  "met": hedged_n >= HEDGED_RECALL_MIN},
        "per_class": pr,
        "correction_recall": correction_recall,
        "speculation_licensing": speculation_licensing,
        "auc": {
            "s_ev": auc_ev,
            "s_conf": auc_conf,
            "s_argmax_sensitivity_M6": auc_argmax,
            "delta_paired_bootstrap": delta,
            "delta_cluster_bootstrap_M9": delta_cluster,
            "confidence_baseline_broken_FM9": auc_conf < 0.55,
        },
        "spearman_rho": rho,
        "mcnemar": {
            "n01_ev_correct_conf_wrong": n01,
            "n10_conf_correct_ev_wrong": n10,
            "n_discordant": n01 + n10,
            "p_exact": mcnemar_p,
            "cross_tuned_taus": taus,
        },
        "redundancy_verdict": redundancy_verdict(
            delta["lo"], delta["hi"], rho, mcnemar_p, n01 + n10),
        "confidence_histogram": conf_hist,
        "s_conf_by_gold_class": conf_by_class,
        "natural_vs_injected_FM1": fm1,
        "edge_case_FM10": {
            "quota_fraction": n_edge / len(records) if records else None,
            "quota_met": bool(records) and n_edge / len(records) >= EDGE_CASE_QUOTA,
            "n_edge_items": n_edge,
            "per_tag_accuracy_n_ge_10": per_tag,
        },
    }


# ---------------------------------------------------------------------------
# Stage 1: gold-label the natural pool (protocol section 2.2, pinned P8)
# ---------------------------------------------------------------------------

def build_all_natural_items(
    tuples_index: dict,
    manifest: list[dict],
    turn_texts: Optional[dict] = None,
) -> tuple[list[dict], list[dict]]:
    """Build items for ALL rows of BOTH harvest pools (P8, extended to the
    `other` pool by R2-B-1: the whole pool passes through the section 2.2
    pipeline before any use as gold). Candidates on schema substrates are
    packaged from the PERTURBED substrate (R2-M-d) and, when `turn_texts`
    (extraction-time turn texts) is provided, a build-time assert enforces
    packaging text == extractor stream text."""
    items, misses = [], []
    substrate_cache: dict = {}
    turn_texts = turn_texts or {}
    for cand in load_natural_candidates():
        substrate = natural_item_substrate(cand, manifest, substrate_cache)
        expected = turn_texts.get((cand["source_file"], cand["question_id"],
                                   cand["session_idx"], cand["turn_idx"]))
        built = label.build_natural_item(cand, substrate, tuples_index,
                                         cand["source_file"],
                                         expected_turn_text=expected)
        if built.get("extraction_miss"):
            misses.append(built)
        else:
            items.append(built)
    return items, misses


def cmd_label_natural() -> None:
    if not EXTRACTOR_TUPLES_PATH.exists():
        sys.exit(f"missing extractor output: {EXTRACTOR_TUPLES_PATH} "
                 "(run `python extract.py produce-tuples` first)")
    manifest = load_manifest()
    tuples_index = label.load_extractor_tuples(EXTRACTOR_TUPLES_PATH)
    turn_texts = label.load_extractor_turn_texts(EXTRACTOR_TUPLES_PATH)
    items, misses = build_all_natural_items(tuples_index, manifest, turn_texts)
    rng = random.Random(SEED)
    rng.shuffle(items)  # randomized item order (protocol section 2.1)

    async def run_all():
        return await asyncio.gather(*[label.label_item(it) for it in items])

    results = asyncio.run(run_all())
    n_queued = label.append_escalations(items, results)
    adjudications = label.read_adjudications()

    NATURAL_GOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NATURAL_GOLD_PATH.open("w") as fh:
        for miss in misses:
            fh.write(json.dumps(miss, ensure_ascii=False) + "\n")
        for item, res in zip(items, results):
            row = {**item, "labeling": res, "final": label.finalize_label(res, adjudications)}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_pending = sum(1 for res in results
                    if label.finalize_label(res, adjudications)["class"] is None)
    print(f"natural pool: {len(items)} labeled, {len(misses)} extraction misses, "
          f"{sum(r['escalate'] for r in results)} escalated "
          f"({n_queued} newly queued), {n_pending} pending adjudication")
    print(f"wrote {NATURAL_GOLD_PATH}")
    if n_pending:
        print("ACTION REQUIRED: adjudicate the escalation queue "
              f"({label.ESCALATION_QUEUE_PATH}) via Claude subagents, write "
              f"{label.ADJUDICATIONS_PATH}, then rerun label-natural (cached calls "
              "are free) before `run`.")


# ---------------------------------------------------------------------------
# Stage 2: eval set + classification + metrics
# ---------------------------------------------------------------------------

def anchor_turn_set() -> set[tuple[str, int]]:
    """(session content hash, turn_idx) of every rubric-anchored [real] example
    turn — the P9 ID-anchored holdout, generalized through content-hash dedupe
    so anchored sessions recurring under other question_ids are caught too."""
    anchors = set()
    for qid, sidx, tidx in label.rubric_anchors():
        substrate = substrate_for("oracle", qid)
        anchors.add((session_content_hash(session_by_orig_idx(substrate, sidx)), tidx))
    return anchors


def build_injected_eval_items(manifest: list[dict], tuples_index: dict) -> tuple[list[dict], list[dict]]:
    items, misses = [], []
    test1_rows = [r for r in manifest if r["inj_type"] in TEST1_INJ_TYPES]
    perturbed_cache: dict[str, Any] = {}
    for row in test1_rows:
        qid = row["question_id"]
        if qid not in perturbed_cache:
            perturbed_cache[qid] = perturbed_substrate(non_t3_rows(manifest, qid), qid)
        built = label.build_injected_item(row, perturbed_cache[qid], tuples_index)
        if built.get("extraction_miss"):
            misses.append(built)
        else:
            built["gold"] = built["construction_label"]
            built["tags"] = built["edge_case_tags"]
            items.append(built)
    return items, misses


def load_natural_eval_items(allow_pending: bool) -> tuple[list[dict], dict]:
    rows = read_jsonl(NATURAL_GOLD_PATH)
    if not rows:
        sys.exit(f"missing {NATURAL_GOLD_PATH}: run `python test1_eval.py label-natural` first")
    items, info = [], {"misses": 0, "pending": 0}
    for row in rows:
        if row.get("extraction_miss"):
            info["misses"] += 1
            continue
        if row["final"]["class"] is None:
            info["pending"] += 1
            continue
        row["gold"] = row["final"]["class"]
        # M8 labeler/adjudicator flags + R2-M-c build-time harvest tags (e.g.
        # kind=other_hedged_recall -> "hedged_recall"), so the >=15
        # hedged-recall stratum and the FM-10 table see natural items.
        flags = list(row["final"]["edge_case_flags"])
        row["tags"] = flags + [t for t in row.get("harvest_tags", []) if t not in flags]
        items.append(row)
    if info["pending"] and not allow_pending:
        sys.exit(f"{info['pending']} natural items are pending adjudication; "
                 "adjudicate the escalation queue and rerun label-natural, or pass "
                 "--allow-pending to exclude them (excluded items are reported).")
    return items, info


def same_hash_turn_survivors(kept: list[dict]) -> list[list[str]]:
    """R2-M-g: groups of KEPT items sharing (session_hash, turn_idx) — duplicate
    sessions surfacing under different question_ids whose extractor wording
    differed enough to evade the tuple-level dedupe key (extraction prompts
    differ per question, so tuple wording can differ). Reported alongside
    dropped_duplicates; the anchor holdout is unaffected (keys on hash+turn)."""
    groups: dict[tuple, list[str]] = {}
    for item in kept:
        groups.setdefault((item["session_hash"], item["turn_idx"]), []).append(item["item_id"])
    return sorted(sorted(ids) for ids in groups.values() if len(ids) > 1)


def attach_session_hashes(items: list[dict]) -> None:
    """Hash each item's UNPERTURBED source session for dedupe + holdout."""
    cache: dict[tuple[str, str, int], str] = {}
    for item in items:
        key = (item["source_file"], item["question_id"], item["session_idx"])
        if key not in cache:
            substrate = substrate_for(item["source_file"], item["question_id"])
            cache[key] = session_content_hash(
                session_by_orig_idx(substrate, item["session_idx"]))
        item["session_hash"] = cache[key]


def cmd_run(allow_pending: bool) -> None:
    import stats  # I1's module (INTERFACES contract)

    if not EXTRACTOR_TUPLES_PATH.exists():
        sys.exit(f"missing extractor output: {EXTRACTOR_TUPLES_PATH} "
                 "(run `python extract.py produce-tuples` first)")
    manifest = load_manifest()
    tuples_index = label.load_extractor_tuples(EXTRACTOR_TUPLES_PATH)

    injected, injected_misses = build_injected_eval_items(manifest, tuples_index)
    natural, natural_info = load_natural_eval_items(allow_pending)
    items = injected + natural
    attach_session_hashes(items)
    kept, dropped = dedupe_and_holdout(items, anchor_turn_set())
    # R2-M-b: items where several extractor tuples matched the target key and
    # the highest-confidence one was taken (M5 tie-break interpolation) — the
    # choice biases s_conf upward on these items, so their count is reported.
    multi_match_ids = sorted(
        it["item_id"] for it in kept if it.get("tuple_match_count", 1) > 1)

    # extraction hit rates per class, intent-to-treat style reporting (M5)
    hit_rates = {}
    for cls in CLASSES:
        hits = sum(1 for it in injected if it["gold"] == cls)
        misses_cls = sum(1 for m in injected_misses if m["construction_label"] == cls)
        hit_rates[cls] = {"hits": hits, "misses": misses_cls,
                          "rate": hits / (hits + misses_cls) if hits + misses_cls else None}

    async def classify_all():
        return await asyncio.gather(*[classify.classify_item(it) for it in kept])

    outputs = asyncio.run(classify_all())

    records = []
    EVAL_ITEMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVAL_ITEMS_PATH.open("w") as fh:
        for item, out in zip(kept, outputs):
            record = {
                "item_id": item["item_id"],
                "gold": item["gold"],
                "provenance": item["provenance"],
                "cluster": (item["question_id"] if item["provenance"] == "injected"
                            else item["item_id"]),  # M9: natural = singleton clusters
                "s_ev": out["s_ev"],
                "s_conf": item["confidence"] / 100.0,
                "confidence_raw": item["confidence"],
                "argmax": out["argmax"],
                "max_prob": out["max_prob"],
                "tags": item["tags"],
            }
            records.append(record)
            fh.write(json.dumps({**item, "classifier": out, **{
                k: record[k] for k in ("s_ev", "s_conf", "cluster")}},
                ensure_ascii=False) + "\n")

    metrics = compute_metrics(records, stats)
    shortfalls = metrics["class_floor_shortfalls"]
    for cls, row in shortfalls.items():
        if row["shortfall"]:
            print(f"CLASS FLOOR SHORTFALL: {cls} has {row['n']} < {row['floor']} "
                  f"(short {row['shortfall']})", file=sys.stderr)

    results = {
        "seed": SEED,
        "model": llm.MODEL,
        "generated_by": "test1_eval.py run",
        "eval_set": {
            "n_kept": len(kept),
            "n_injected_built": len(injected),
            "n_natural_built": len(natural),
            "natural_pending_excluded": natural_info["pending"],
            "natural_extraction_misses": natural_info["misses"],
            "injected_extraction_misses": len(injected_misses),
            "injected_extraction_miss_ids": [m["item_id"] for m in injected_misses],
            "extraction_hit_rates_injected": hit_rates,
            "dropped_anchored": dropped["anchored"],
            "dropped_duplicates": dropped["duplicates"],
            # R2-M-g: same-(session_hash, turn_idx) items that SURVIVED the
            # tuple-level dedupe (extractor wording differed across question
            # contexts), reported next to dropped_duplicates.
            "same_hash_turn_survivors": {
                "n_groups": len(same_hash_turn_survivors(kept)),
                "groups": same_hash_turn_survivors(kept),
            },
            # R2-M-b: M5 multi-match tie-break (highest confidence wins).
            "m5_multi_match_tiebreak": {
                "n_items": len(multi_match_ids),
                "item_ids": multi_match_ids,
                "note": ("several extractor tuples matched the target key; the "
                         "highest-confidence tuple entered the eval set (pinned "
                         "M5 reading, changelog) — biases s_conf upward on these items"),
            },
        },
        "metrics": metrics,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"wrote {RESULTS_PATH} (n={len(kept)})")


# ---------------------------------------------------------------------------
# Stage 3: 20% self-consistency rerun (FM-12; protocol section 4.3)
# ---------------------------------------------------------------------------

def cmd_self_consistency() -> None:
    rows = read_jsonl(EVAL_ITEMS_PATH)
    if not rows:
        sys.exit(f"missing {EVAL_ITEMS_PATH}: run `python test1_eval.py run` first")
    rng = random.Random(f"{SEED}-selfcheck")
    n_sample = math.ceil(SELF_CONSISTENCY_FRACTION * len(rows))
    sample = rng.sample(rows, n_sample)

    async def rerun():
        return await asyncio.gather(*[
            classify.classify_item(it, use_cache=False) for it in sample])

    outputs = asyncio.run(rerun())
    argmax_agree = sum(1 for it, out in zip(sample, outputs)
                       if it["classifier"]["argmax"] == out["argmax"])
    commit_agree = sum(1 for it, out in zip(sample, outputs)
                       if (it["classifier"]["argmax"] in COMMIT_CLASSES)
                       == (out["argmax"] in COMMIT_CLASSES))
    report = {
        "seed": f"{SEED}-selfcheck",
        "fraction": SELF_CONSISTENCY_FRACTION,
        "n": n_sample,
        "argmax_agreement": {"x": argmax_agree, "n": n_sample,
                             "point": argmax_agree / n_sample},
        "commit_decision_agreement": {"x": commit_agree, "n": n_sample,
                                      "point": commit_agree / n_sample},
        "item_ids": [it["item_id"] for it in sample],
    }
    SELF_CONSISTENCY_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote {SELF_CONSISTENCY_PATH} "
          f"(argmax agreement {argmax_agree}/{n_sample})")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("label-natural")
    run_p = sub.add_parser("run")
    run_p.add_argument("--allow-pending", action="store_true")
    sub.add_parser("self-consistency")
    args = parser.parse_args()
    if args.cmd == "label-natural":
        cmd_label_natural()
    elif args.cmd == "run":
        cmd_run(allow_pending=args.allow_pending)
    elif args.cmd == "self-consistency":
        cmd_self_consistency()


if __name__ == "__main__":
    main()
