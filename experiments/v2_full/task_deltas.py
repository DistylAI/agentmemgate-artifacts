"""task_deltas.py — M3: downstream task deltas (protocol v2.4, headroom endpoint).

Preregistration (protocol/CHANGELOG.md, "Protocol v2.4" M3 + 2026-07-08 budget
amendment): answer the benchmarks' OWN QA sets from each cell's store —
deterministic lookup + the frozen equivalence judge, no answering LLM — and
report per-cell task accuracy with cluster CIs; the gate's value expressed as
a task-level delta on questions whose answers route through hedge/update
handling, and its cost on all other questions. Baselines have headroom by
construction (v2 probe accuracies 0.26–0.48).

Question universe (pinned by the M3 build brief; frozen at authoring time)
--------------------------------------------------------------------------
  LoCoMo   the 98 pre-mapped QA rows (qa_probe_map.mapped) + the 13 key-mapped
           time-indexed items, both from protocol/gold/key_schema_v2_locomo.json.
           The 23 unmapped time-indexed items are episodic counters with no
           schema key (V2-F4) and are out of scope, as everywhere in v2.
  LME      the 67 KU probe questions of protocol/gold/key_schema_v2_lme.json
           (probe_question per substrate; gold = new_value, the post-update
           current value). The 60 MS/TR aux substrates carry no KU question.

Conditions (frozen stores; NO cell is re-run here)
--------------------------------------------------
  mem0_gated / mem0_ungated / zep_gated / zep_ungated   results/v2/<cell>
  mem0_gated_after                                      results/v2.4/m1_after/v2/mem0_gated

The AFTER tree is M1's controlled re-run of the mem0 gated column with the
key-mapper fix ON — the mem0-family triple (before-fix gated, after-fix gated,
ungated) is the fix's task-level effect. The m1_after/v2.2 tree (17 natural-
expansion hosts) maps to NO benchmark QA item and is not consumed here.

Scoring rules (deterministic; the frozen blind equivalence judge is the only
LLM in the loop, store.values_equivalent semantics unchanged)
-----------------------------------------------------------------------------
  current / derived_count / asof / time-indexed / LME KU:
      value = current or interval lookup (locomo_render probe engine, spec §4);
      correct = found AND (normalized-exact OR judge-equivalent vs gold).
      Negation golds (normalized in {"no","none","never","0"}; exactly one row,
      conv-44 qa 8): correct = value ABSENT at the as-of instant, or zero-like.
  history_event_time:
      correct = ANY history entry's valid_from denotes the gold event time
      (normalized-exact, else judge on attribute "<attribute> (event date)").
  history_event_value / history_old_value:
      correct = ANY history entry's value matches the gold (normalized-exact,
      else gold-token containment, else judge). Token containment (all gold
      tokens present in the entry's normalized tokens) is deterministic and
      covers answers that are a COMPONENT of a stored state (e.g. gold
      "Shadow" inside a roster/count value "two (Coco, Shadow)") — the
      question asks for a recoverable part, not the whole state.
  derived_history (10 rows, tier=secondary, arithmetic across two events):
      EXCLUDED from accuracy — not recoverable by lookup + equivalence (needs
      date arithmetic); reported with excluded_reason, never silently dropped.

Question-family strata (the headline split)
-------------------------------------------
  update_sensitive = answers routing through hedge/update handling:
    - every history / asof map_type row (the answer IS the supersession
      history or an interval lookup through it);
    - every time-indexed item (as-of through validity intervals);
    - every LME KU question (in-dialog old->new supersession by construction);
    - current/derived_count rows whose QA evidence spans >= 2 sessions — the
      integration/update-sensitive marker of research_notes/locomo_analysis.md
      (the answer integrates cross-session state).
  other = current/derived_count rows with single-session evidence (n=7; the
  pre-mapped universe was constructed from update keys — FM-21 scoping — so
  the "cost on all other questions" readout is thin and flagged as such).

Headline: gated-vs-ungated accuracy delta per stratum (paired per item,
cluster-bootstrap CI on the mean of Y_gated - Y_ungated in {-1,0,1}); mem0
family additionally before-fix vs after-fix vs ungated.

Clusters: LoCoMo = conversation (n=10); LME = substrate qid with the frozen
shared-session merge (driver.SHARED_SESSION_CLUSTER); pooled rows prefix the
benchmark to keep cluster ids disjoint.

Budget (2026-07-08 amendment): every fresh call is recorded under stage "m3"
of results/v2.4/budget.json via forecast_v22.SharedCallBudget. Stage
allocations are ADVISORY — an m3 overrun past 8,000 raises a forecast-error
ALARM in the log and the results JSON, never an abort. The ONLY hard stop is
the 120,000 combined envelope. Judge calls are llm-cached, so an aborted or
re-run pass is idempotent and near-free.

Outputs (results/v2.4/m3/)
--------------------------
  per_item.jsonl    one row per condition x question (scored, excluded and
                    missing rows all present, labeled)
  m3_results.json   universe counts, per cell x benchmark accuracy blocks
                    (overall / update_sensitive / other / per qa_family) with
                    cluster CIs, headline deltas, budget block
  m3_report.md      compact human-readable summary
  smoke/smoke.json  the 5-item smoke readout (`smoke` command)

CLI
---
  python experiments/v2_full/task_deltas.py run [--force] [--fresh-cap N]
  python experiments/v2_full/task_deltas.py smoke
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  (R1 layout: puts src/ + experiments/ on sys.path)

import driver as run_v2
import forecast_v22
import locomo_data
import locomo_render
import mapper_fix
import stats
import store
from paths import RESULTS_DIR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 20260708
N_BOOT = 10000

STAGE = "m3"
V24_DIR = RESULTS_DIR / "v2.4"
BUDGET_PATH = V24_DIR / "budget.json"
M3_DIR = V24_DIR / "m3"
SMOKE_DIR = M3_DIR / "smoke"
PER_ITEM_PATH = M3_DIR / "per_item.jsonl"
RESULTS_PATH = M3_DIR / "m3_results.json"
REPORT_PATH = M3_DIR / "m3_report.md"

V2_TREE = RESULTS_DIR / "v2"
AFTER_TREE = V24_DIR / "m1_after" / "v2"

#: condition -> (frozen timeline tree, cell directory name). Order is the
#: scoring order (deterministic budget spend / cache warm-up).
CONDITIONS: dict[str, tuple[Path, str]] = {
    "mem0_gated": (V2_TREE, "mem0_gated"),
    "mem0_gated_after": (AFTER_TREE, "mem0_gated"),
    "mem0_ungated": (V2_TREE, "mem0_ungated"),
    "zep_gated": (V2_TREE, "zep_gated"),
    "zep_ungated": (V2_TREE, "zep_ungated"),
}

#: (family, delta name, condition_a, condition_b): reported as acc_a - acc_b.
DELTAS: tuple[tuple[str, str, str, str], ...] = (
    ("mem0", "gated_vs_ungated_before_fix", "mem0_gated", "mem0_ungated"),
    ("mem0", "gated_vs_ungated_after_fix", "mem0_gated_after", "mem0_ungated"),
    ("mem0", "fix_effect_after_vs_before", "mem0_gated_after", "mem0_gated"),
    ("zep", "gated_vs_ungated", "zep_gated", "zep_ungated"),
)

STRATA = ("update_sensitive", "other")

#: Scoreable history map_types (locomo_render owns the full frozen sets).
HISTORY_TIME_MAP_TYPES = {"history_event_time"}
HISTORY_VALUE_MAP_TYPES = {"history_event_value", "history_old_value"}
EXCLUDED_MAP_TYPES = {"derived_history"}

#: Negation golds (normalized): the answer is the ABSENCE of a value.
NEGATION_GOLD_NORMS = {"no", "none", "never", "0"}
#: Stored values that count as a negation answer when the gold is one.
ZERO_LIKE_NORMS = {"0", "none", "no"}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


# ---------------------------------------------------------------------------
# Budget — advisory stage m3, hard envelope only (2026-07-08 amendment)
# ---------------------------------------------------------------------------

def m3_guard(path: Path = BUDGET_PATH,
             local_cap: int | None = None) -> forecast_v22.SharedCallBudget:
    """The M3 shared-envelope guard. Per the 2026-07-08 owner-approved
    amendment, the m3 allocation (8,000) is an ADVISORY forecast: spend is
    recorded per fresh call under stage 'm3', but the guard NEVER refuses to
    install or aborts on a stage overrun — the only hard stop is the 120,000
    combined envelope (SharedCallBudget's own envelope check). ``local_cap``
    is a harness convenience for bounded smokes, not a stage cap."""
    mapper_fix.ensure_budget(path)  # shape/prereg validation; never clobbers
    return forecast_v22.SharedCallBudget(STAGE, path=path, local_cap=local_cap)


def budget_block(path: Path, fresh_this_run: int) -> dict:
    budget = json.loads(path.read_text())
    alloc = budget["allocation"][STAGE]
    spent = budget["spent"][STAGE]
    return {
        "stage": STAGE,
        "fresh_calls_this_run": fresh_this_run,
        "stage_spent_total": spent,
        "stage_allocation_advisory": alloc,
        "advisory_overrun_alarm": spent > alloc,
        "envelope": budget["envelope"],
        "envelope_spent_total": sum(budget["spent"].values()),
        "allocation_mode": budget.get("allocation_mode"),
    }


# ---------------------------------------------------------------------------
# Question universe
# ---------------------------------------------------------------------------

def _evidence_sessions(evidence: list[str] | None) -> set[int]:
    sessions: set[int] = set()
    for pointer in evidence or ():
        for m in locomo_data.DIA_ID_RE.finditer(str(pointer)):
            sessions.add(int(m.group(1)))
    return sessions


def natqa_update_sensitive(row: dict) -> bool:
    """The frozen stratum rule for pre-mapped QA rows (module docstring):
    history/asof map_types always; current/count rows iff their QA evidence
    spans >= 2 sessions (locomo_analysis.md integration/update-sensitive)."""
    if row["map_type"] not in locomo_render._CURRENT_MAP_TYPES:
        return True
    return len(_evidence_sessions(row.get("evidence"))) >= 2


def _natqa_family(map_type: str) -> str:
    if map_type in EXCLUDED_MAP_TYPES:
        return "natqa_derived_history"
    if map_type in HISTORY_TIME_MAP_TYPES | HISTORY_VALUE_MAP_TYPES:
        return "natqa_history"
    if map_type in locomo_render._ASOF_MAP_TYPES:
        return "natqa_asof"
    return "natqa_current"


def build_items() -> list[dict]:
    """The frozen M3 question universe, in deterministic (file) order."""
    items: list[dict] = []
    for row in locomo_data.qa_probe_map_rows():
        excluded = row["map_type"] in EXCLUDED_MAP_TYPES
        items.append({
            "benchmark": "locomo",
            "item_id": f"natqa/{row['sample_id']}/qa{row['qa_index']}",
            "qa_family": _natqa_family(row["map_type"]),
            "stream_id": row["sample_id"],
            "cluster": row["sample_id"],
            "update_sensitive": natqa_update_sensitive(row),
            "tier": row["tier"],
            "excluded": excluded,
            "excluded_reason": (
                "derived_history: arithmetic across two events is not "
                "recoverable by deterministic lookup + equivalence judging"
                if excluded else None),
            "payload": row,
        })
    for row in locomo_data.time_indexed_probes():
        if not row.get("key_id"):
            continue  # episodic counters, no schema key (V2-F4)
        items.append({
            "benchmark": "locomo",
            "item_id": f"ti/{row['sample_id']}/qa{row['qa_index']}",
            "qa_family": "time_indexed",
            "stream_id": row["sample_id"],
            "cluster": row["sample_id"],
            "update_sensitive": True,
            "tier": "time_indexed",
            "excluded": False,
            "excluded_reason": None,
            "payload": row,
        })
    for qid, entry in run_v2.lme_ku_entries().items():
        items.append({
            "benchmark": "lme",
            "item_id": f"ku/{qid}",
            "qa_family": "lme_ku",
            "stream_id": qid,
            "cluster": run_v2.SHARED_SESSION_CLUSTER.get(qid, qid),
            "update_sensitive": True,
            "tier": entry.get("tier"),
            "excluded": False,
            "excluded_reason": None,
            "payload": entry,
        })
    return items


# ---------------------------------------------------------------------------
# Scoring — deterministic lookup + the frozen equivalence judge
# ---------------------------------------------------------------------------

async def _judge(attribute: str, value, gold, *, trace_tag: str) -> bool:
    """Single judge chokepoint (tests monkeypatch this): the FROZEN blind
    equivalence judge, store.values_equivalent semantics unchanged."""
    return await store.values_equivalent(attribute, str(value), str(gold),
                                         trace_tag=trace_tag)


def _norm(value) -> str:
    return store.normalize_value(str(value))


def _tokens(value) -> list[str]:
    return _TOKEN_RE.findall(_norm(value))


def gold_token_contained(value, gold) -> bool:
    """All gold tokens present among the value's normalized tokens (the
    deterministic component-recoverability fast path for history VALUE rows)."""
    gold_tokens = _tokens(gold)
    return bool(gold_tokens) and set(gold_tokens) <= set(_tokens(value))


def gold_is_negation(gold) -> bool:
    return _norm(gold) in NEGATION_GOLD_NORMS


async def _score_value(value, gold, attribute: str, *, tag: str,
                       counter: dict) -> bool:
    """Shared value-vs-gold rule for current/asof/count/TI/KU lookups."""
    if gold_is_negation(gold):
        return locomo_render.value_is_absent(value) or _norm(value) in ZERO_LIKE_NORMS
    if locomo_render.value_is_absent(value):
        return False
    if _norm(value) == _norm(gold):
        return True
    counter["n_judge_calls"] += 1
    return await _judge(attribute, value, gold, trace_tag=tag)


async def _score_candidates(candidates: list, gold, attribute: str, *,
                            tag: str, counter: dict,
                            containment: bool) -> tuple[bool, str | None]:
    """ANY-match rule for history rows. Two passes, both oldest-first with
    short-circuit: (1) the deterministic checks (normalized exact; token
    containment for value rows) across ALL candidates before any LLM, then
    (2) the frozen judge over the remaining candidates — deterministic
    judge-call order, minimal fresh spend."""
    present = [(idx, cand) for idx, cand in enumerate(candidates)
               if cand is not None and not locomo_render.value_is_absent(cand)]
    for _, cand in present:
        if _norm(cand) == _norm(gold):
            return True, str(cand)
        if containment and gold_token_contained(cand, gold):
            return True, str(cand)
    for idx, cand in present:
        counter["n_judge_calls"] += 1
        if await _judge(attribute, cand, gold, trace_tag=f"{tag}-h{idx}"):
            return True, str(cand)
    return False, None


async def score_item(item: dict, mapping, *, tag: str) -> dict:
    """Score ONE universe item against one condition's merged store view.

    Returns the per-item outcome fields (correct/found/value/judge use);
    the caller stamps condition/benchmark/cluster fields on top.
    """
    counter = {"n_judge_calls": 0}
    out: dict = {"correct": None, "found": None, "value": None,
                 "n_history": None, "matched_candidate": None,
                 "n_judge_calls": 0, "excluded": item["excluded"],
                 "excluded_reason": item["excluded_reason"], "missing": False}
    if item["excluded"]:
        return out
    payload = item["payload"]
    family = item["qa_family"]

    if family in ("natqa_current", "natqa_asof", "natqa_history"):
        probe = locomo_render.p_natqa_probe(mapping, payload)
        gold = probe["gold_answer"]
        attribute = probe["key"][1]
        if family == "natqa_history":
            history = probe["history"] or []
            out["n_history"] = len(history)
            out["found"] = probe["found"]
            if payload["map_type"] in HISTORY_TIME_MAP_TYPES:
                candidates = [e["valid_from"] for e in history]
                ok, match = await _score_candidates(
                    candidates, gold, f"{attribute} (event date)",
                    tag=tag, counter=counter, containment=False)
            else:
                candidates = [e["value"] for e in history]
                ok, match = await _score_candidates(
                    candidates, gold, attribute,
                    tag=tag, counter=counter, containment=True)
            out["correct"] = ok
            out["matched_candidate"] = match
        else:
            out["value"] = probe["value"]
            out["found"] = probe["found"]
            out["correct"] = await _score_value(
                probe["value"], gold, attribute, tag=tag, counter=counter)

    elif family == "time_indexed":
        probe = locomo_render.p_time_probe(mapping, payload)
        out["value"] = probe["value"]
        out["found"] = probe["found"]
        out["correct"] = await _score_value(
            probe["value"], probe["gold_answer"], probe["key"][1],
            tag=tag, counter=counter)

    elif family == "lme_ku":
        key = (payload["entity"], payload["attribute"])
        value = locomo_render.current_value(mapping, key)
        gold = str(payload["new_value"])
        out["value"] = value
        out["found"] = not locomo_render.value_is_absent(value)
        out["correct"] = await _score_value(
            value, gold, payload["attribute"], tag=tag, counter=counter)

    else:  # pragma: no cover - build_items never emits other families
        raise ValueError(f"unknown qa_family {family!r}")

    out["n_judge_calls"] = counter["n_judge_calls"]
    return out


def _load_mapping(tree: Path, cell: str, stream_id: str):
    timeline = run_v2.load_cell_timeline(tree, cell, stream_id)
    if timeline is None:
        return None
    return run_v2._MappingStore(run_v2.timeline_store_mapping(timeline))


async def score_all(items: list[dict],
                    conditions: dict[str, tuple[Path, str]]) -> list[dict]:
    """One row per condition x item (sequential: deterministic spend order)."""
    rows: list[dict] = []
    for condition, (tree, cell) in conditions.items():
        mappings: dict[str, object] = {}
        for item in items:
            sid = item["stream_id"]
            if sid not in mappings:
                mappings[sid] = _load_mapping(tree, cell, sid)
            mapping = mappings[sid]
            base = {k: item[k] for k in
                    ("benchmark", "item_id", "qa_family", "stream_id",
                     "cluster", "update_sensitive", "tier")}
            base["condition"] = condition
            if mapping is None:
                base.update({"correct": None, "found": None, "value": None,
                             "n_history": None, "matched_candidate": None,
                             "n_judge_calls": 0, "excluded": item["excluded"],
                             "excluded_reason": item["excluded_reason"],
                             "missing": True})
            else:
                outcome = await score_item(
                    item, mapping, tag=f"m3-{condition}-{item['item_id']}")
                base.update(outcome)
            rows.append(base)
    return rows


# ---------------------------------------------------------------------------
# Aggregation — cluster CIs + the headline paired deltas
# ---------------------------------------------------------------------------

def _scored(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["correct"] is not None]


def _accuracy_block(rows: list[dict], n_boot: int) -> dict:
    rows = _scored(rows)
    if not rows:
        return {"point": None, "lo": None, "hi": None, "n": 0,
                "n_clusters": 0, "x": 0}
    block = stats.cluster_bootstrap_rate(
        [int(r["correct"]) for r in rows], [r["cluster"] for r in rows],
        n_boot=n_boot, seed=SEED)
    block["x"] = sum(int(r["correct"]) for r in rows)
    return block


def _delta_block(rows_a: list[dict], rows_b: list[dict], n_boot: int) -> dict:
    """Paired per-item delta acc_a - acc_b with a cluster-bootstrap CI."""
    by_a = {r["item_id"]: r for r in _scored(rows_a)}
    by_b = {r["item_id"]: r for r in _scored(rows_b)}
    ids = sorted(set(by_a) & set(by_b))
    if not ids:
        return {"point": None, "lo": None, "hi": None, "n_pairs": 0,
                "n_clusters": 0, "acc_a": None, "acc_b": None,
                "n_discordant": 0}
    diffs = [int(by_a[i]["correct"]) - int(by_b[i]["correct"]) for i in ids]
    clusters = [by_a[i]["cluster"] for i in ids]
    block = stats.cluster_bootstrap_mean(diffs, clusters, n_boot=n_boot,
                                         seed=SEED, bounds=(-1, 1))
    n = len(ids)
    return {"point": block["point"], "lo": block["lo"], "hi": block["hi"],
            "n_pairs": n, "n_clusters": block["n_clusters"],
            "acc_a": sum(int(by_a[i]["correct"]) for i in ids) / n,
            "acc_b": sum(int(by_b[i]["correct"]) for i in ids) / n,
            "n_discordant": sum(1 for d in diffs if d != 0)}


def _stratum_slices(rows: list[dict]) -> dict[str, list[dict]]:
    return {
        "overall": rows,
        "update_sensitive": [r for r in rows if r["update_sensitive"]],
        "other": [r for r in rows if not r["update_sensitive"]],
    }


def _bench_slices(rows: list[dict]) -> dict[str, list[dict]]:
    """Per-benchmark row slices + the pooled view (disjoint cluster ids)."""
    slices = {
        bench: [r for r in rows if r["benchmark"] == bench]
        for bench in ("locomo", "lme")
    }
    slices["pooled"] = [
        {**r, "cluster": f"{r['benchmark']}:{r['cluster']}"} for r in rows]
    return slices


def universe_block(items: list[dict]) -> dict:
    def count(pred) -> int:
        return sum(1 for it in items if pred(it))
    block: dict = {"n_items": len(items)}
    for bench in ("locomo", "lme"):
        b_items = [it for it in items if it["benchmark"] == bench]
        block[bench] = {
            "n": len(b_items),
            "n_excluded": sum(it["excluded"] for it in b_items),
            "n_scoreable": sum(not it["excluded"] for it in b_items),
            "update_sensitive": sum(
                it["update_sensitive"] and not it["excluded"]
                for it in b_items),
            "other": sum(
                not it["update_sensitive"] and not it["excluded"]
                for it in b_items),
            "by_family": {
                fam: count(lambda it, f=fam: it["benchmark"] == bench
                           and it["qa_family"] == f)
                for fam in sorted({it["qa_family"] for it in b_items})},
        }
    return block


def aggregate(rows: list[dict], items: list[dict],
              n_boot: int = N_BOOT) -> dict:
    """The m3_results.json body (budget block added by the caller)."""
    results: dict = {
        "id": "V2.4-M3-TASK-DELTAS",
        "protocol": ("Protocol v2.4 M3 (protocol/CHANGELOG.md, preregistered "
                     "2026-07-07; 2026-07-08 budget amendment)"),
        "seed": SEED,
        "n_boot": n_boot,
        "definitions": {
            "update_sensitive": (
                "history/asof map_types; time-indexed items; LME KU "
                "questions; current/derived_count rows with QA evidence "
                "spanning >= 2 sessions (locomo_analysis.md integration/"
                "update-sensitive marker)"),
            "other": ("current/derived_count rows with single-session "
                      "evidence (thin by construction: the pre-mapped "
                      "universe derives from update keys, FM-21)"),
            "scoring": ("deterministic store lookup (locomo_render probe "
                        "engine) + frozen blind equivalence judge only; "
                        "derived_history rows excluded (arithmetic); "
                        "negation golds score by value absence; history "
                        "rows score by any-entry match (see module "
                        "docstring for the full rules)"),
            "ci": "cluster bootstrap (LoCoMo: conversation; LME: substrate "
                  "with the frozen shared-session merge), percentile 95%",
        },
        "universe": universe_block(items),
    }

    conditions_block: dict = {}
    for condition in CONDITIONS:
        cond_rows = [r for r in rows if r["condition"] == condition]
        cond_block: dict = {}
        for bench, b_rows in _bench_slices(cond_rows).items():
            bench_block: dict = {
                stratum: _accuracy_block(s_rows, n_boot)
                for stratum, s_rows in _stratum_slices(b_rows).items()}
            bench_block["by_family"] = {
                fam: _accuracy_block(
                    [r for r in b_rows if r["qa_family"] == fam], n_boot)
                for fam in sorted({r["qa_family"] for r in b_rows})}
            bench_block["n_missing_timelines"] = sum(
                r["missing"] for r in b_rows)
            cond_block[bench] = bench_block
        conditions_block[condition] = cond_block
    results["conditions"] = conditions_block

    deltas_block: dict = {}
    for family, name, cond_a, cond_b in DELTAS:
        rows_a = [r for r in rows if r["condition"] == cond_a]
        rows_b = [r for r in rows if r["condition"] == cond_b]
        entry: dict = {"family": family, "condition_a": cond_a,
                       "condition_b": cond_b}
        slices_a, slices_b = _bench_slices(rows_a), _bench_slices(rows_b)
        for bench in slices_a:
            entry[bench] = {
                stratum: _delta_block(sa, sb, n_boot)
                for (stratum, sa), (_, sb) in zip(
                    _stratum_slices(slices_a[bench]).items(),
                    _stratum_slices(slices_b[bench]).items())}
        deltas_block[f"{family}/{name}"] = entry
    results["headline_deltas"] = deltas_block

    results["judge_use"] = {
        "n_rows_judged": sum(1 for r in rows if r["n_judge_calls"]),
        "n_judge_calls_total": sum(r["n_judge_calls"] for r in rows),
        "note": "judge calls are llm-cached; fresh spend is the budget block",
    }
    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt(block: dict, point_key: str = "point") -> str:
    if block.get(point_key) is None:
        return "—"
    lo, hi = block.get("lo"), block.get("hi")
    n = block.get("n", block.get("n_pairs"))
    return f"{block[point_key]:+.3f} [{lo:+.3f}, {hi:+.3f}] (n={n})"


def _fmt_acc(block: dict) -> str:
    if block.get("point") is None:
        return "—"
    return (f"{block['point']:.3f} [{block['lo']:.3f}, {block['hi']:.3f}] "
            f"(x={block.get('x')}/{block['n']})")


def write_report(results: dict, path: Path = REPORT_PATH) -> None:
    u = results["universe"]
    lines = [
        "# M3 — downstream task deltas (protocol v2.4, preregistered)",
        "",
        f"Seed {results['seed']}; cluster-bootstrap CIs (n_boot="
        f"{results['n_boot']}). Frozen stores only; the equivalence judge is "
        "the only LLM in the loop.",
        "",
        f"Universe: {u['n_items']} questions — LoCoMo "
        f"{u['locomo']['n_scoreable']} scoreable "
        f"({u['locomo']['n_excluded']} derived_history excluded), LME "
        f"{u['lme']['n_scoreable']} KU. Strata: update-sensitive "
        f"{u['locomo']['update_sensitive'] + u['lme']['update_sensitive']}, "
        f"other {u['locomo']['other'] + u['lme']['other']} (thin by "
        "construction, FM-21).",
        "",
        "## Task accuracy per cell x benchmark (overall)",
        "",
        "| condition | locomo | lme | pooled |",
        "|---|---|---|---|",
    ]
    for condition, block in results["conditions"].items():
        lines.append(
            f"| {condition} | {_fmt_acc(block['locomo']['overall'])} "
            f"| {_fmt_acc(block['lme']['overall'])} "
            f"| {_fmt_acc(block['pooled']['overall'])} |")
    lines += [
        "",
        "## Headline: paired deltas per stratum (acc_a - acc_b)",
        "",
        "| delta | benchmark | update_sensitive | other |",
        "|---|---|---|---|",
    ]
    for name, entry in results["headline_deltas"].items():
        for bench in ("locomo", "lme", "pooled"):
            lines.append(
                f"| {name} | {bench} "
                f"| {_fmt(entry[bench]['update_sensitive'])} "
                f"| {_fmt(entry[bench]['other'])} |")
    b = results.get("budget", {})
    lines += [
        "",
        "## Budget (stage m3, advisory per the 2026-07-08 amendment)",
        "",
        f"Fresh calls this run: {b.get('fresh_calls_this_run')}; stage total "
        f"{b.get('stage_spent_total')} vs advisory {b.get('stage_allocation_advisory')}"
        f"{' — ADVISORY OVERRUN ALARM' if b.get('advisory_overrun_alarm') else ''}; "
        f"envelope {b.get('envelope_spent_total')}/{b.get('envelope')}.",
        "",
    ]
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def cmd_run(args) -> None:
    if RESULTS_PATH.exists() and not args.force:
        raise SystemExit(f"{RESULTS_PATH} exists — pass --force to re-score "
                         "(judge calls are cached; re-runs are near-free)")
    M3_DIR.mkdir(parents=True, exist_ok=True)
    items = build_items()
    guard = m3_guard(local_cap=args.fresh_cap)
    guard.install()
    try:
        rows = asyncio.run(score_all(items, CONDITIONS))
    finally:
        guard.uninstall()
    results = aggregate(rows, items, n_boot=args.n_boot)
    results["budget"] = budget_block(BUDGET_PATH, guard.fresh)
    _write_jsonl(PER_ITEM_PATH, rows)
    RESULTS_PATH.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    write_report(results)
    if results["budget"]["advisory_overrun_alarm"]:
        print(f"[m3][ALARM] stage m3 spent "
              f"{results['budget']['stage_spent_total']:,} fresh calls > "
              f"advisory allocation "
              f"{results['budget']['stage_allocation_advisory']:,} — "
              "forecast-error alarm per the 2026-07-08 amendment (no stop; "
              "the 120k envelope is the only hard limit)")
    for name, entry in results["headline_deltas"].items():
        us = entry["pooled"]["update_sensitive"]
        print(f"[m3] {name}: pooled update-sensitive delta "
              f"{_fmt(us)}")
    print(f"[m3] fresh calls this run: {guard.fresh}; outputs -> {M3_DIR}")


def cmd_smoke(args) -> None:
    """5 QA items through ONE cell (mem0_gated, frozen v2 tree), budget-
    guarded under stage m3 with a hard local cap of 10 fresh calls. Items are
    the first 5 single-lookup rows (current/asof) of the pre-mapped file
    order, so each row costs at most one judge call."""
    items = [it for it in build_items()
             if it["qa_family"] in ("natqa_current", "natqa_asof")][:5]
    guard = m3_guard(local_cap=10)
    guard.install()
    try:
        rows = asyncio.run(score_all(
            items, {"mem0_gated": CONDITIONS["mem0_gated"]}))
    finally:
        guard.uninstall()
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    out = {"items": rows, "fresh_calls": guard.fresh,
           "n_correct": sum(int(bool(r["correct"])) for r in rows)}
    (SMOKE_DIR / "smoke.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False))
    for r in rows:
        print(f"[m3-smoke] {r['item_id']} ({r['qa_family']}, "
              f"US={r['update_sensitive']}): correct={r['correct']} "
              f"value={r['value']!r} judges={r['n_judge_calls']}")
    print(f"[m3-smoke] {out['n_correct']}/5 correct; fresh calls "
          f"{guard.fresh} (cap 10) -> {SMOKE_DIR / 'smoke.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="M3 downstream task deltas (protocol v2.4)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="score all conditions + aggregate")
    p.add_argument("--force", action="store_true")
    p.add_argument("--fresh-cap", type=int, default=None,
                   help="optional extra cap on fresh calls this run "
                        "(the envelope remains the only preregistered stop)")
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    sub.add_parser("smoke", help="5 QA items through one cell (<=10 fresh)")
    args = parser.parse_args()
    {"run": cmd_run, "smoke": cmd_smoke}[args.cmd](args)


if __name__ == "__main__":
    main()
