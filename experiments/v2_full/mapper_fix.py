"""mapper_fix.py — Protocol v2.4 M1: key-mapper fix for plan-shaped facts
(controlled BEFORE/AFTER), preregistered in protocol/CHANGELOG.md
("Protocol v2.4", M1 entry).

THE ONE TARGETED CHANGE (exactly one iteration, no tunables beyond the frozen
pattern list): mapping mode "plan_stem_wrapper" (store.make_tuple_key_mapper
over router_sweep.PLAN_FRAME_ATTRIBUTE_PATTERNS — the list's single frozen
home; this module passes it, never re-declares it). Implementation reading,
blessed per orchestrator PIN-1: the frozen mapper call runs first
(byte-cache-shared with the v2 runs); only on a DECLINE of a plan-framed
attribute is the tuple re-offered ONCE with its derived stem attribute (same
candidates, same blindness). PIN-1 audit obligation: the stem re-offer may map
to ANY candidate, not provably the one whose plan-variant matched — every
stem-mapped row logs (original_attribute, derived_stem, mapped_key,
pattern_matched) and `score` reports the fraction where mapped_key is NOT the
candidate whose plan-variant matched (diagnostic stratum, NO bar).

FRESH-SPEND ACCOUNTING (orchestrator PIN-2): extraction and first-pass mapping
are fully cache-shared with the frozen runs; the fresh spend is (a) the stem
re-offer mapping calls, (b) classifier calls on newly stem-mapped tuples,
(c) predicate-generator calls on newly staged tuples, (d) CASCADING checker
calls (each newly staged item changes every subsequent checker prompt on its
store), and (e) equivalence-judge calls in scoring. ALL of these count against
the M1 8,000 allocation of the v2.4 combined envelope
(results/v2.4/budget.json, envelope 45,000; stages m1 8k / m2 25k / m3 8k /
m4 3k / headroom 1k), enforced per call by forecast_v22.SharedCallBudget
(stage "m1", local_cap = the unspent m1 allocation).

BEFORE = the frozen artifacts, READ AND HARD-ASSERTED, never recomputed:
  * SYN-RES promotion ITT per block: results/v2/test3_{em,ez}/results.json
    (the prereg's "baseline 46-76%": em/lme 23/50, ez/lme 35/50,
    ez/locomo 36/50, em/locomo 38/50);
  * the v2.3 staging funnel: results/v2.3/staging_funnel.jsonl (loss
    concentration anchor: em planted_spec 270/360 extracted-but-unmapped with
    the target key in the candidate list);
  * natural strata: results/v2.3/resolution_aware_rule5.json (at-risk n=16 =
    the contamination stratum; resolved-true n=29 = correct tracking) over the
    frozen judged rows of results/v2.2/rule5_expanded_rows.jsonl;
  * contamination / correction-recall BEFORE: recomputed from the frozen
    GATED-cell timelines through the same JSON prober used for AFTER
    (equivalence prompts are byte-identical to the frozen metrics pass, so
    the judge calls replay from cache). Per the review's drift note (§1.1,
    impl_review_v24_m1.md) every BEFORE determination anchors on the
    gated/cache side — the *_ungated timelines carry stale rationale text.

AFTER = the gated columns re-run with the fix ON, into results/v2.4/m1_after/
mirroring the frozen trees (m1_after/v2/<cell>/timelines/... for the 127 LME +
10 LoCoMo main-cell streams; m1_after/v2.2/<cell>/timelines/... for the 17
v2.2 natural-expansion hosts — the two trees share stream ids, so they must
not be merged). Extraction is never re-invoked fresh (cache-shared by
construction); reruns resume (existing AFTER timelines are skipped), so a
budget abort is recoverable.

ENDPOINTS (`score`; results/v2.4/m1_results.json + m1_report.md):
  * PRIMARY — end-to-end promotion rate of resolving hedges, BEFORE/AFTER
    delta with cluster-bootstrap CIs. Operationalization: the resolving-hedge
    populations are (a) the natural resolved-true stratum (n=29 hedges whose
    resolution verifiably arrives in-stream; per family, through the gated
    columns' key mapper — the mapper-sensitive measurement) and (b) the
    SYN-RES t3 blocks (the prereg's 46-76% anchor). NOTE, disclosed here and
    in the report: the frozen SYN-RES pathway binds its staging key by
    construction and contains NO key-mapper call, so its M1 delta is
    STRUCTURALLY ZERO — it is carried as the baseline anchor, and the
    measured delta lives on the natural resolving hedges + the funnel's
    staging stages.
  * GUARDS — contamination: gated-cell hedged-value commit rate over the
    planted spec_* probes (BEFORE 0), tolerated Wilson upper <= 5%;
    correction recall: must not fall below the preregistered v2 rule
    (corr_recall_rule: point >= 0.9 AND Wilson lower >= 0.8); unmapped-tuple
    rate per cell (reported, no bar).
  * natural at-risk (n=16) re-score: hedged-value-held-as-current in the
    AFTER gated stores (frozen hedge-value sets; must not rise) and
    resolved-true (n=29) re-score (correct tracking must not degrade).
  * PIN-1 diagnostic stratum (above) + funnel re-run on AFTER via the
    imported staging_funnel record builders (same record schema; the v2.3
    reconciliation asserts apply to BEFORE only — AFTER staging rates change
    by design).

CLI (repo root, pilot/.venv/bin/python experiments/v2_full/mapper_fix.py ...):
  budget                       write/refresh results/v2.4/budget.json (never
                               clobbers spent counters)
  before                       BEFORE block only (0 fresh calls enforced)
  run   [--groups lme,locomo,expansion] [--families mem0,zep] [--qids ...]
                               AFTER re-run, fix ON, budget-guarded, resumable
  score                        endpoints from BEFORE + AFTER -> m1_results.json
                               + m1_report.md (judge calls budget-guarded)
  smoke --qid QID [--fresh-cap N]
                               ONE substrate before/after (default cap 10
                               fresh calls; artifacts under results/v2.4/smoke/,
                               the real AFTER tree untouched; spend recorded
                               under stage m1 — it is real M1 spend)
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401  (R1 layout: puts src/ + experiments/ on sys.path)

import arms
import data  # noqa: F401  (driver deps)
import driver
import expand_natural
import forecast_v22
import gated_store
import locomo_data
import metrics_v2
import natural_check_rule5 as rule5
import staging_funnel as sf
import stats
import store
from router_sweep import PLAN_FRAME_ATTRIBUTE_PATTERNS, plan_frame_match
from zepstore import ZepStore

from paths import RESULTS_DIR, ROOT

SEED = 20260702

V2_DIR = RESULTS_DIR / "v2"
V22_DIR = RESULTS_DIR / "v2.2"
V23_DIR = RESULTS_DIR / "v2.3"
V24_DIR = RESULTS_DIR / "v2.4"
BUDGET_PATH = V24_DIR / "budget.json"
AFTER_DIR = V24_DIR / "m1_after"
SMOKE_DIR = V24_DIR / "smoke"
RESULTS_PATH = V24_DIR / "m1_results.json"
REPORT_PATH = V24_DIR / "m1_report.md"

STAGE = "m1"
ENVELOPE = 120000  # 2026-07-08 amendment: total-only hard limit
# m1 raised 8000->9000 from preregistered headroom (changelog 2026-07-07:
# cascade cost of the fix succeeding; zep AFTER dropped as mechanism-absent).
# m4 raised 3000->12000 from headroom (changelog 2026-07-09: M4-EVAL
# preregistration; allocations ADVISORY per the 2026-07-08 amendment).
# external: 15000 from headroom (changelog 2026-07-30 "adapter phase
# authorized": external-dataset adapter dev smoke + the preregistered
# one-shot test run; advisory like every stage — only the envelope is hard).
# claude_repl: 15000 from headroom, registered IN results/v2.4/budget.json
# (changelog 2026-08-11 "different-family replication" preregistration:
# Claude Opus 4.8 systems, frozen gpt-5.5 judge). Stages added after this
# module's freeze live in the budget file; ensure_budget accepts them as a
# superset and still pins every stage listed here.
ALLOCATION = {"m1": 11000, "m2": 23000, "m3": 8000, "m4": 12000,
              "external": 15000}
HEADROOM = 51000  # 2026-07-08 amendment envelope minus stage forecasts
assert sum(ALLOCATION.values()) + HEADROOM == ENVELOPE

FAMILIES = ("mem0", "zep")
FAM_KEY = {"mem0": "em", "zep": "ez"}          # metrics-family aliases
FAM_NAME = {"em": "mem0", "ez": "zep"}
GATED_CELL = {"em": "mem0_gated", "ez": "zep_gated"}

#: Scoring annotation for a family whose AFTER tree was never run. The zep
#: AFTER run was DROPPED as mechanism-absent (changelog 2026-07-07, M1
#: execution deviations: the fix's precondition barely exists for the
#: canonical-writing extractor — frozen funnel 233/240 hedges already staged;
#: unmapped plan-frame rows concentrate in mem0, 270/360). Such a family is
#: scored N/A: frozen BEFORE values still reported, AFTER/delta null, guards
#: and PIN-1 diagnostics computed for present families only.
NA_STATUS = "N/A — mechanism absent, AFTER not run (changelog 2026-07-07)"

#: AFTER tree names mirror the frozen trees they re-run (stream ids collide
#: across v2 and v2.2 — e.g. 0977f2af exists in both — so they never merge).
TREE_V2 = "v2"
TREE_V22 = "v2.2"

#: Frozen BEFORE anchors (impl_review_v24_m1.md §4; drift in any of these is
#: an artifact-integrity failure -> hard stop, never silently recomputed).
SYNRES_PROMOTION_BEFORE = {
    ("em", "lme"): (23, 50), ("em", "locomo"): (38, 50),
    ("ez", "lme"): (35, 50), ("ez", "locomo"): (36, 50),
}
FUNNEL_EM_SPEC_UNMAPPED_TARGET_BEFORE = (270, 360)
NATURAL_AT_RISK_N = 16       # v2.3 contamination stratum (unresolved+abandoned)
NATURAL_RESOLVED_TRUE_N = 29  # v2.3 correct-tracking stratum
CONTAM_GATED_COMMITS_BEFORE = 0  # per family x benchmark, planted spec probes


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open() if line.strip()]


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def _wilson(x: int, n: int) -> dict:
    lo, hi = stats.wilson_ci(x, n) if n else (None, None)
    return {"x": x, "n": n, "point": (x / n if n else None),
            "wilson_lo": lo, "wilson_hi": hi}


# ---------------------------------------------------------------------------
# Budget (PIN-2) — results/v2.4/budget.json + the shared M1 guard
# ---------------------------------------------------------------------------

def ensure_budget(path: Path = BUDGET_PATH) -> dict:
    """Create the v2.4 combined budget file if missing; validate and return it
    if present. NEVER clobbers spent counters (idempotent)."""
    if path.exists():
        budget = json.loads(path.read_text())
        assert budget["envelope"] == ENVELOPE, (
            f"{path}: envelope {budget['envelope']} != preregistered {ENVELOPE}")
        # every stage frozen here must be present and pinned; stages added
        # AFTER this module's freeze (changelog-preregistered, e.g.
        # claude_repl 2026-08-11) are registered in the budget file itself
        # and accepted as a superset — a MISSING stage or a drifted
        # allocation still fails hard.
        assert set(ALLOCATION) <= set(budget["spent"]), (
            f"{path}: stages {sorted(budget['spent'])} missing "
            f"{sorted(set(ALLOCATION) - set(budget['spent']))}")
        for s, cap in ALLOCATION.items():
            assert budget["allocation"][s] == cap, (
                f"{path}: allocation[{s}] = {budget['allocation'][s]} != {cap}")
        return budget
    budget = {
        "id": "V2.4-COMBINED-BUDGET",
        "protocol": "Protocol v2.4 (protocol/CHANGELOG.md, preregistered): "
                    "<= 45,000 fresh calls total across M1-M4",
        "envelope": ENVELOPE,
        "allocation": {**ALLOCATION, "unallocated_headroom": HEADROOM},
        "spent": {s: 0 for s in ALLOCATION},
        "spent_note": (
            "FRESH llm calls (cache hits free, run_v2.CallBudget semantics) "
            "recorded per call by forecast_v22.SharedCallBudget(stage, "
            "path=results/v2.4/budget.json); combined envelope enforced over "
            "sum(spent.values()); per-stage caps enforced via local_cap by "
            "each stage runner. M1 accounting (orchestrator PIN-2): stem "
            "re-offer mapping calls + classifier + predicate-generator calls "
            "on newly stem-mapped tuples + cascading checker calls + scoring "
            "equivalence-judge calls ALL count under m1 (8,000)."),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(budget, indent=1, ensure_ascii=False))
    return budget


def m1_guard(path: Path = BUDGET_PATH,
             local_cap: int | None = None) -> forecast_v22.SharedCallBudget:
    """The M1 shared-envelope guard: SharedCallBudget stage 'm1' against the
    v2.4 budget file, capped at the UNSPENT m1 allocation (PIN-2). Refuses to
    install when the m1 allocation is exhausted."""
    budget = ensure_budget(path)
    stage_left = ALLOCATION[STAGE] - budget["spent"][STAGE]
    if stage_left <= 0:
        raise SystemExit(
            f"M1 stage allocation EXHAUSTED: {budget['spent'][STAGE]:,} fresh "
            f"calls spent of {ALLOCATION[STAGE]:,} — hard stop (de-scope order "
            "in the prereg never touches M1's guard endpoints; do not raid "
            "other stages)")
    cap = stage_left if local_cap is None else min(local_cap, stage_left)
    return forecast_v22.SharedCallBudget(STAGE, path=path, local_cap=cap)


# ---------------------------------------------------------------------------
# The fix-ON mapper + AFTER replays
# ---------------------------------------------------------------------------

def make_mapper():
    """The one M1 mapping mode, over the FROZEN pattern list (single home:
    router_sweep.PLAN_FRAME_ATTRIBUTE_PATTERNS — passed, never re-declared)."""
    return store.make_tuple_key_mapper(PLAN_FRAME_ATTRIBUTE_PATTERNS)


async def replay_lme_after(job: arms.CellJob, family: str, mapper) -> gated_store.GatedTimeline:
    return await gated_store.replay_gated(
        job.substrate, job.key_schema_entry, job.manifest_rows,
        family=family, aux_keys=job.aux_keys, mapper=mapper)


async def replay_locomo_after(job: arms.CellJob, family: str, mapper) -> gated_store.GatedTimeline:
    """driver.make_locomo_replay's gated branch with the mapper threaded
    through (the dyadic single-pass seam; extraction cache-shared)."""
    conv = driver.perturbed_conversation(job.question_id)
    stream = driver.locomo_stream_rows(conv)
    extracted = await driver.dyadic_extract_stream(
        conv, family, stream_id=f"{job.question_id}-perturbed",
        turn_range=(0, len(stream)))
    stores = {s.lower(): ZepStore() for s in conv.speakers}
    return await gated_store.run_gated_replay(
        stream, job.aux_keys, stream_id=job.question_id, family=family,
        stores=stores, route_store=lambda key: key[0],
        session_order=list(conv.session_order), extracted=extracted,
        mapper=mapper)


def expansion_qids() -> list[str]:
    """The v2.2 natural-expansion host streams that actually ran (frozen
    gated timelines on disk) — the AFTER re-run mirrors exactly that set."""
    return sorted(p.stem for p in (V22_DIR / "mem0_gated" / "timelines").glob("*.json"))


def build_after_jobs(groups: tuple[str, ...], qids: list[str] | None = None) -> list[dict]:
    """[{tree, group, job}] for the requested AFTER groups. `qids` filters
    within each group (partial/resume runs); an empty intersection runs
    nothing for that group."""
    out = []
    if "lme" in groups:
        lme_qids = [q for q in driver.lme_v2_qids()
                    if qids is None or q in qids]
        if lme_qids:
            for job in driver.lme_cell_jobs(lme_qids):
                out.append({"tree": TREE_V2, "group": "lme", "job": job})
    if "locomo" in groups:
        sids = [s for s in locomo_data.conversation_ids()
                if qids is None or s in qids]
        if sids:
            for job in driver.locomo_cell_jobs(sids):
                out.append({"tree": TREE_V2, "group": "locomo", "job": job})
    if "expansion" in groups:
        exp = [q for q in expansion_qids() if qids is None or q in qids]
        if exp:
            for job in expand_natural.build_jobs(exp):
                out.append({"tree": TREE_V22, "group": "expansion", "job": job})
    return out


async def run_after(groups: tuple[str, ...], families: tuple[str, ...],
                    qids: list[str] | None, out_root: Path,
                    guard: forecast_v22.SharedCallBudget) -> dict:
    """Re-run the GATED columns with the fix ON. One timeline JSON per stream,
    the arms.run_cells serializer verbatim (byte-comparable with the frozen
    artifacts); existing AFTER timelines are SKIPPED (resume after a budget
    abort). Only gated cells run — extraction/mapping first passes replay
    from the frozen cache, all fresh spend is guarded (PIN-2)."""
    mapper = make_mapper()
    entries = build_after_jobs(groups, qids)
    summary: dict = {"mapping_mode": mapper.mode, "cells": {}, "skipped": 0,
                     "n_jobs": len(entries)}
    guard.install()
    try:
        for family in families:
            cell = arms.cell_name(family, "gated")
            for entry in entries:
                job, tree, group = entry["job"], entry["tree"], entry["group"]
                cell_dir = out_root / tree / cell
                tl_path = cell_dir / "timelines" / f"{job.question_id}.json"
                if tl_path.exists():
                    summary["skipped"] += 1
                    continue
                if group == "locomo":
                    timeline = await replay_locomo_after(job, family, mapper)
                else:
                    timeline = await replay_lme_after(job, family, mapper)
                tl_path.parent.mkdir(parents=True, exist_ok=True)
                tl_path.write_text(json.dumps(
                    gated_store.timeline_to_json(timeline),
                    ensure_ascii=False, indent=1))
                bucket = summary["cells"].setdefault(f"{tree}/{cell}", {
                    "n_streams": 0, "n_tuples": 0, "n_mapped": 0,
                    "n_unmapped": 0, "n_staged": 0, "n_promoted": 0})
                d = timeline.diagnostics
                bucket["n_streams"] += 1
                for k in ("n_tuples", "n_mapped", "n_unmapped", "n_staged",
                          "n_promoted"):
                    bucket[k] += d[k]
    finally:
        guard.uninstall()
    summary["fresh_calls"] = guard.fresh
    return summary


# ---------------------------------------------------------------------------
# JSON-timeline probing (frozen + AFTER trees, same code path both sides)
# ---------------------------------------------------------------------------

def probe_timeline_json(tl: dict, key_name: str, probe_point: str,
                        *, turn: tuple | None = None,
                        store_id: str | None = None) -> str | None:
    """gated_store.probe_timeline over a SERIALIZED timeline (store routing
    per arms.answer_probes R1-B1: the entity-named store when it exists,
    else the default store)."""
    if store_id is None:
        entity = key_name.partition("/")[0]
        store_id = entity if entity in tl["stores"] else gated_store.DEFAULT_STORE
    if probe_point == "full_ingestion":
        snap = tl["snapshots"][-1]
    elif probe_point == "old_session_boundary":
        earlier = tl["session_order"][0]
        snaps = [s for s in tl["snapshots"] if s["session_idx"] == earlier]
        snap = snaps[-1]
    elif probe_point == "provenance_timeline":
        matches = [s for s in tl["snapshots"]
                   if (s["session_idx"], s["turn_idx"]) == tuple(turn)]
        snap = matches[0]
    else:
        raise ValueError(f"unknown probe_point {probe_point!r}")
    hist = tl["stores"].get(store_id, {}).get("history", {}).get(key_name, [])
    return rule5.as_of_seq_json(hist, snap["seq"].get(store_id, 0))


def _load_tl(base: Path, cell: str, stream_id: str) -> dict:
    return rule5.load_timeline(cell, stream_id, base)


async def probe_manifest_rows(base: Path, fam_key: str, inj_types: set[str],
                              scorer: rule5.EquivalenceScorer) -> list[dict]:
    """{injection_id, benchmark, cluster, present, y} per probed manifest row
    against ONE tree's gated cell (`y` = probed value equivalent to the
    planted target — the contamination/recall outcome)."""
    cell = GATED_CELL[fam_key]
    rows = []
    for man in driver.load_manifest_v2():
        if man["inj_type"] not in inj_types:
            continue
        tl = _load_tl(base, cell, man["question_id"])
        turn = None
        if man["probe_point"] == "provenance_timeline":
            turn = (man["placement"]["session_idx"], man["placement"]["turn_idx"])
        key_name = f"{man['entity']}/{man['attribute']}"
        value = probe_timeline_json(tl, key_name, man["probe_point"], turn=turn)
        eq, method = await scorer.equivalent(man["attribute"], value, man["value"])
        rows.append({
            "injection_id": man["injection_id"], "benchmark": man["benchmark"],
            "question_id": man["question_id"], "inj_type": man["inj_type"],
            "cluster": driver._cluster_of({"question_id": man["question_id"]}),
            "key": key_name, "value": value,
            "present": value is not None, "y": int(eq), "method": method,
        })
    return rows


def _agg(rows: list[dict]) -> dict[tuple, dict]:
    out: dict[tuple, dict] = defaultdict(lambda: {"x": 0, "n": 0})
    for r in rows:
        b = out[(r["benchmark"],)]
        b["n"] += 1
        b["x"] += r["y"]
    return out


# ---------------------------------------------------------------------------
# BEFORE (frozen artifacts, hard-asserted)
# ---------------------------------------------------------------------------

def before_synres() -> dict:
    """SYN-RES promotion ITT per block from the frozen results.json files —
    asserted equal to the preregistered 46-76% anchors."""
    out = {}
    for fam in ("em", "ez"):
        res = json.loads((V2_DIR / f"test3_{fam}" / "results.json").read_text())
        for bench in ("lme", "locomo"):
            p = res["benchmarks"][bench]["promotion_recall_ITT"]["primary"]
            x, n = p["x"], p["n"]
            assert (x, n) == SYNRES_PROMOTION_BEFORE[(fam, bench)], (
                f"frozen SYN-RES drift: {fam}/{bench} = {x}/{n} != "
                f"{SYNRES_PROMOTION_BEFORE[(fam, bench)]}")
            out[f"{fam}/{bench}"] = {"x": x, "n": n, "point": p["point"],
                                     "ci": p["ci"], "ci_kind": p["ci_kind"]}
    return out


def funnel_stage_table(records: list[dict]) -> dict:
    """Per (population, family, benchmark): stage-survival counts + the
    unmapped-with-target-in-candidates loss cell."""
    table: dict[str, dict] = {}
    for r in records:
        k = f"{r['population']}/{r['family']}/{r['benchmark']}"
        b = table.setdefault(k, {"n": 0, "extracted": 0, "mapped": 0,
                                 "routed_speculative": 0,
                                 "staged_with_predicates": 0,
                                 "unmapped_target_in_candidates": 0,
                                 "promoted": 0})
        b["n"] += 1
        reached = r["stage_reached"]
        if reached is not None:
            idx = sf.STAGES.index(reached)
            for s in sf.STAGES[:idx + 1]:
                b[s] += 1
        if (r["dropout_stage"] == "mapped"
                and (r.get("mechanism_detail") or {}).get("target_key_in_candidates")):
            b["unmapped_target_in_candidates"] += 1
        srow = (r.get("evidence") or {}).get("staged_row") or {}
        if reached == "staged_with_predicates" and srow.get("status") == "promoted":
            b["promoted"] += 1
    return table


def before_funnel() -> tuple[list[dict], dict]:
    records = _read_jsonl(V23_DIR / "staging_funnel.jsonl")
    assert records, "missing frozen results/v2.3/staging_funnel.jsonl"
    em_spec = [r for r in records
               if r["population"] == "planted_spec" and r["family"] == "em"]
    x = sum(1 for r in em_spec
            if r["dropout_stage"] == "mapped"
            and (r.get("mechanism_detail") or {}).get("target_key_in_candidates"))
    assert (x, len(em_spec)) == FUNNEL_EM_SPEC_UNMAPPED_TARGET_BEFORE, (
        f"frozen funnel drift: em spec unmapped-with-target = {x}/{len(em_spec)}"
        f" != {FUNNEL_EM_SPEC_UNMAPPED_TARGET_BEFORE}")
    return records, funnel_stage_table(records)


def natural_strata() -> dict[str, list[dict]]:
    """{'at_risk': [...16 frozen rescore rows], 'resolved_true': [...29]}."""
    doc = json.loads((V23_DIR / "resolution_aware_rule5.json").read_text())
    rows = doc["resolution_aware"]["rows"]
    strata = {"at_risk": [r for r in rows if r["stratum"] == "contamination"],
              "resolved_true": [r for r in rows
                                if r["stratum"] == "correct_tracking"]}
    assert len(strata["at_risk"]) == NATURAL_AT_RISK_N, len(strata["at_risk"])
    assert len(strata["resolved_true"]) == NATURAL_RESOLVED_TRUE_N, (
        len(strata["resolved_true"]))
    return strata


def rule5_rows_by_id() -> dict[str, dict]:
    return {r["row_id"]: r for r in _read_jsonl(V22_DIR / "rule5_expanded_rows.jsonl")}


def _r5_tree(row: dict) -> str:
    """Which frozen tree hosts a rule-5 row's cells (mirrors sf._r5_base)."""
    return Path(row["_base"]).name if row.get("_base") else TREE_V2


def natural_promotion_rows(strata: dict, funnel_records: list[dict]) -> list[dict]:
    """Per (resolved-true row, family): the end-to-end promotion outcome of a
    RESOLVING natural hedge, read from funnel records (staged AND the staged
    item's status == promoted). Same determination BEFORE (frozen funnel
    jsonl) and AFTER (rebuilt records) — pure log reads."""
    resolved_ids = {r["row_id"] for r in strata["resolved_true"]}
    by_key = {}
    for rec in funnel_records:
        if rec["population"] != "natural":
            continue
        rid = rec["hedge_source_id"]
        if rid not in resolved_ids:
            continue
        srow = (rec.get("evidence") or {}).get("staged_row") or {}
        promoted = int(rec["stage_reached"] == "staged_with_predicates"
                       and srow.get("status") == "promoted")
        staged = int(rec["stage_reached"] == "staged_with_predicates")
        by_key[(rid, rec["family"])] = {
            "row_id": rid, "family": rec["family"],
            "cluster": rec["stream_id"], "staged": staged, "promoted": promoted}
    return [by_key[k] for k in sorted(by_key)]


# ---------------------------------------------------------------------------
# AFTER funnel rebuild (staging_funnel imported, pointed at the AFTER trees)
# ---------------------------------------------------------------------------

def build_after_funnel_records(after_root: Path,
                               families: tuple[str, ...] = sf.FAMILIES) -> list[dict]:
    """The v2.3 funnel's spec + natural populations rebuilt against the AFTER
    trees (same record schema, same extraction dumps — extraction is
    cache-shared and unchanged by the fix). The t3 population is NOT rebuilt:
    its pathway has no key-mapper call (structural zero; carried from the
    frozen records by the scorer). `families` restricts the rebuild to the
    families whose AFTER trees exist (zep dropped, changelog 2026-07-07)."""
    records = []
    man = [r for r in sf._read_jsonl(sf.MANIFEST_PATH)
           if r["inj_type"] in sf.SPEC_TYPES]
    for row in man:
        si, ti = row["placement"]["session_idx"], row["placement"]["turn_idx"]
        for family in families:
            cell = sf.FAM_GATED[family]
            dump = sf.dump_tuples(row["benchmark"], sf.FAM_NAME[family])
            d = dump[(row["question_id"], si, ti)]
            tl = sf.load_timeline(after_root / TREE_V2, cell, row["question_id"])
            gates, unmapped = sf.turn_rows(tl, si, ti)
            records.append(sf.spec_funnel_record(
                row, family, d["tuples"], gates, unmapped, sf.staged_index(tl),
                _rel(after_root / TREE_V2 / cell / "timelines"
                     / f"{row['question_id']}.json")))
    texts = sf.natural_hedge_texts()
    for row in sf._read_jsonl(sf.RULE5_ROWS_PATH):
        tree = _r5_tree(row)
        si, ti = row["anchor"]
        for family in families:
            cell = sf.FAM_GATED[family]
            tl = sf.load_timeline(after_root / tree, cell, row["sample_id"])
            if tree == TREE_V22:
                tuples = sf.v22_anchor_tuples(cell, row["sample_id"], si, ti, tl)
            else:
                d = sf.dump_tuples("locomo", sf.FAM_NAME[family])[
                    (row["sample_id"], si, ti)]
                tuples = d["tuples"]
            gates, unmapped = sf.turn_rows(tl, si, ti)
            records.append(sf.natural_funnel_record(
                row, family, texts[row["row_id"]], tuples, gates, unmapped,
                sf.staged_index(tl),
                _rel(after_root / tree / cell / "timelines"
                     / f"{row['sample_id']}.json")))
    return records


# ---------------------------------------------------------------------------
# Natural strata re-scores (frozen hedge-value sets; AFTER store state)
# ---------------------------------------------------------------------------

async def rescore_natural_gated(base_for_tree, scorer: rule5.EquivalenceScorer,
                                families: tuple[str, ...] = ("em", "ez")) -> list[dict]:
    """Per frozen rule-5 row: Y per GATED cell recomputed from the tree that
    `base_for_tree(tree_name) -> Path` resolves (frozen trees for BEFORE, the
    AFTER trees for AFTER). The hedged-value sets are the FROZEN judged sets
    (rule5_expanded_rows.jsonl hedged_values_all — extraction-level, unchanged
    by the fix); only the store state and the equivalence of NEW final values
    are recomputed. Estimand arithmetic identical to the frozen §8 rule 5.
    `families` restricts the re-score to families whose trees exist under the
    resolved bases; y_frozen always carries BOTH families (frozen row data,
    so a mechanism-absent family's BEFORE stays reportable)."""
    out = []
    for row in _read_jsonl(V22_DIR / "rule5_expanded_rows.jsonl"):
        tree = _r5_tree(row)
        base = base_for_tree(tree)
        key_name = row["key"]
        store_id = row.get("store_id", row["entity"])
        y, finals = {}, {}
        for fam in families:
            tl = _load_tl(base, GATED_CELL[fam], row["sample_id"])
            if store_id not in tl["stores"]:
                sid = gated_store.DEFAULT_STORE
            else:
                sid = store_id
            final = rule5.full_ingestion_value(tl, sid, key_name)
            finals[f"{fam}_cg"] = final
            yv = 0
            for hedged in row["hedged_values_all"][fam]:
                eq, _method = await scorer.equivalent(row["attribute"], final, hedged)
                if eq:
                    yv = 1
            y[f"{fam}_cg"] = yv
        out.append({"row_id": row["row_id"], "cluster": row["cluster"],
                    "tree": tree, "y_gated": y, "finals": finals,
                    "y_frozen": {"em_cg": row["y_primary"]["em_cg"],
                                 "ez_cg": row["y_primary"]["ez_cg"]}})
    return out


def _stratum_before(rows: list[dict], row_ids: set[str], fam: str) -> dict:
    """Frozen BEFORE counts of a stratum. y_frozen carries BOTH families
    (read from the frozen rule-5 rows), so this stays reportable for a
    family whose AFTER tree was never run (N/A path)."""
    sel = [r for r in rows if r["row_id"] in row_ids]
    return {"x": sum(r["y_frozen"][f"{fam}_cg"] for r in sel), "n": len(sel)}


def _stratum_delta(rows: list[dict], row_ids: set[str], fam: str,
                   before_key: str = "y_frozen", after_key: str = "y_gated") -> dict:
    sel = [r for r in rows if r["row_id"] in row_ids]
    d = [r[after_key][f"{fam}_cg"] - r[before_key][f"{fam}_cg"] for r in sel]
    boot = stats.cluster_bootstrap_mean(
        d, [r["cluster"] for r in sel], seed=SEED, bounds=(-1, 1)) if sel else None
    return {
        "n": len(sel),
        "before_x": sum(r[before_key][f"{fam}_cg"] for r in sel),
        "after_x": sum(r[after_key][f"{fam}_cg"] for r in sel),
        "delta_mean": (boot["point"] if boot else None),
        "delta_ci": ([boot["lo"], boot["hi"]] if boot else None),
        "ci_kind": "cluster_bootstrap",
    }


# ---------------------------------------------------------------------------
# PIN-1 diagnostic stratum
# ---------------------------------------------------------------------------

def pin1_diagnostic(after_root: Path, families: tuple[str, ...] = FAMILIES) -> dict:
    """Over every AFTER mapping-log row the wrapper stem-MAPPED: the fraction
    where mapped_key is NOT the candidate whose plan-variant matched (i.e.
    the mapped key's attribute != the derived stem). Diagnostic stratum, NO
    bar (orchestrator PIN-1). Reads the serialized `unmapped`+gate provenance
    is not enough (mapping_log is not serialized) — the per-row audit fields
    ride on the GATE rows of stem-mapped tuples (mapping_mode/plan_framed/
    original_attribute/derived_stem), which are one-to-one with stem-mapped
    mapping rows."""
    total = mismatch = 0
    rows = []
    for tree in (TREE_V2, TREE_V22):
        for family in families:
            cell = arms.cell_name(family, "gated")
            tdir = after_root / tree / cell / "timelines"
            if not tdir.exists():
                continue
            for p in sorted(tdir.glob("*.json")):
                tl = json.loads(p.read_text())
                for g in tl["gate_log"]:
                    if not g.get("plan_framed"):
                        continue
                    total += 1
                    mapped_attr = g["key"].partition("/")[2]
                    off_variant = mapped_attr != g["derived_stem"]
                    mismatch += int(off_variant)
                    rows.append({
                        "tree": tree, "cell": cell, "stream_id": tl["stream_id"],
                        "item_id": g["item_id"],
                        "original_attribute": g["original_attribute"],
                        "derived_stem": g["derived_stem"],
                        "mapped_key": g["key"],
                        "off_variant_candidate": off_variant})
    return {"n_stem_mapped": total, "n_off_variant": mismatch,
            "fraction_off_variant": (mismatch / total if total else None),
            "note": "diagnostic stratum, no bar (PIN-1); a row is "
                    "off-variant when the mapped key's attribute != the "
                    "derived stem of the original plan-framed attribute",
            "rows": rows}


# ---------------------------------------------------------------------------
# score — endpoints from BEFORE + AFTER
# ---------------------------------------------------------------------------

def present_families(after_root: Path) -> tuple[str, ...]:
    """Metrics-family keys ('em'/'ez') whose AFTER gated tree exists under
    `after_root` in ANY of the two mirrored trees. A family with no AFTER
    tree at all was dropped from the AFTER run (zep: mechanism absent,
    changelog 2026-07-07) and is scored N/A with its frozen BEFORE values
    carried. A PARTIAL tree is deliberately NOT masked as absent: the
    loaders fail loudly and the resumable `run` command completes it."""
    return tuple(
        fam for fam in ("em", "ez")
        if any((after_root / tree / GATED_CELL[fam] / "timelines").exists()
               for tree in (TREE_V2, TREE_V22)))


def unmapped_rates(tree_bases: dict[str, Path], families=FAMILIES) -> dict:
    out = {}
    for tree, base in tree_bases.items():
        for family in families:
            cell = arms.cell_name(family, "gated")
            tdir = base / cell / "timelines"
            if not tdir.exists():
                continue
            n_t = n_u = 0
            for p in sorted(tdir.glob("*.json")):
                d = json.loads(p.read_text())["diagnostics"]
                n_t += d["n_tuples"]
                n_u += d["n_unmapped"]
            out[f"{tree}/{cell}"] = {"n_tuples": n_t, "n_unmapped": n_u,
                                     "rate": (n_u / n_t if n_t else None)}
    return out


def _promotion_block(rows: list[dict], fam: str) -> dict:
    sel = [r for r in rows if r["family"] == fam]
    x = sum(r["promoted"] for r in sel)
    boot = stats.cluster_bootstrap_mean(
        [r["promoted"] for r in sel], [r["cluster"] for r in sel],
        seed=SEED, bounds=(0, 1)) if sel else None
    return {"x": x, "n": len(sel),
            "point": (x / len(sel) if sel else None),
            "ci": ([boot["lo"], boot["hi"]] if boot else None),
            "ci_kind": "cluster_bootstrap"}


async def compute_scores(after_root: Path) -> dict:
    strata = natural_strata()
    before_records, before_table = before_funnel()
    scorer = rule5.EquivalenceScorer(cap=4000)

    # Families with an AFTER tree; an absent family (zep, dropped as
    # mechanism-absent, changelog 2026-07-07) is scored N/A — frozen BEFORE
    # still reported, AFTER/delta null; guards + PIN-1 present-only.
    present = present_families(after_root)
    absent = tuple(f for f in ("em", "ez") if f not in present)

    # --- AFTER funnel + promotion ------------------------------------------
    after_records = build_after_funnel_records(after_root, present)
    after_table = funnel_stage_table(after_records)

    prom_before = natural_promotion_rows(strata, before_records)
    prom_after = natural_promotion_rows(strata, after_records)
    prom_delta = {}
    for fam in ("em", "ez"):
        if fam not in present:
            prom_delta[fam] = {"before": _promotion_block(prom_before, fam),
                               "after": None, "delta": None,
                               "status": NA_STATUS}
            continue
        b = {(r["row_id"]): r for r in prom_before if r["family"] == fam}
        a = {(r["row_id"]): r for r in prom_after if r["family"] == fam}
        ids = sorted(set(b) & set(a))
        d = [a[i]["promoted"] - b[i]["promoted"] for i in ids]
        boot = stats.cluster_bootstrap_mean(
            d, [b[i]["cluster"] for i in ids], seed=SEED, bounds=(-1, 1)) if ids else None
        prom_delta[fam] = {
            "before": _promotion_block(prom_before, fam),
            "after": _promotion_block(prom_after, fam),
            "delta_mean": (boot["point"] if boot else None),
            "delta_ci": ([boot["lo"], boot["hi"]] if boot else None),
            "ci_kind": "cluster_bootstrap", "n_paired": len(ids)}

    # --- guards (present families only — N/A families never ran AFTER) ------
    contamination, corr_recall = {}, {}
    for fam in present:
        spec_after = await probe_manifest_rows(
            after_root / TREE_V2, fam, set(arms.SPEC_TYPES), scorer)
        spec_before = await probe_manifest_rows(
            V2_DIR, fam, set(arms.SPEC_TYPES), scorer)
        corr_after = await probe_manifest_rows(
            after_root / TREE_V2, fam, set(arms.CORR_TYPES), scorer)
        corr_before = await probe_manifest_rows(
            V2_DIR, fam, set(arms.CORR_TYPES), scorer)
        for bench in ("lme", "locomo"):
            sa = [r for r in spec_after if r["benchmark"] == bench]
            sb = [r for r in spec_before if r["benchmark"] == bench]
            if not sa:
                continue
            xb = sum(r["y"] for r in sb)
            assert xb == CONTAM_GATED_COMMITS_BEFORE, (
                f"frozen contamination drift: {fam}/{bench} BEFORE hedged-value "
                f"commits = {xb} != {CONTAM_GATED_COMMITS_BEFORE}")
            w = _wilson(sum(r["y"] for r in sa), len(sa))
            contamination[f"{fam}/{bench}"] = {
                "before": _wilson(xb, len(sb)), "after": w,
                "bar": "Wilson upper <= 0.05 (M1 prereg)",
                "pass": (w["wilson_hi"] is not None and w["wilson_hi"] <= 0.05)}
            ca = [r for r in corr_after if r["benchmark"] == bench]
            cb = [r for r in corr_before if r["benchmark"] == bench]
            if ca:
                after_rule = metrics_v2.corr_recall_rule(
                    sum(r["y"] for r in ca), len(ca))
                before_rule = metrics_v2.corr_recall_rule(
                    sum(r["y"] for r in cb), len(cb))
                corr_recall[f"{fam}/{bench}"] = {
                    "before": before_rule, "after": after_rule,
                    "bar": "preregistered v2 CORR-RECALL rule must still pass "
                           "and the point must not fall below BEFORE",
                    "pass": (bool(after_rule["pass"])
                             and after_rule["point"] >= before_rule["point"])}

    # --- natural strata re-scores -------------------------------------------
    frozen_bases = {TREE_V2: V2_DIR, TREE_V22: V22_DIR}
    nat_rows = await rescore_natural_gated(
        lambda tree: after_root / tree, scorer, present)
    at_risk_ids = {r["row_id"] for r in strata["at_risk"]}
    resolved_ids = {r["row_id"] for r in strata["resolved_true"]}
    natural = {}
    for fam in ("em", "ez"):
        if fam not in present:
            natural[fam] = {
                "at_risk_contamination": {
                    "before": _stratum_before(nat_rows, at_risk_ids, fam),
                    "after": None, "delta": None, "status": NA_STATUS,
                    "bar": "must not rise (hedged value held while unresolved)"},
                "resolved_true_correct_tracking": {
                    "before": _stratum_before(nat_rows, resolved_ids, fam),
                    "after": None, "delta": None, "status": NA_STATUS,
                    "bar": "must not degrade (correct tracking)"}}
            continue
        natural[fam] = {
            "at_risk_contamination": {
                **_stratum_delta(nat_rows, at_risk_ids, fam),
                "bar": "must not rise (hedged value held while unresolved)"},
            "resolved_true_correct_tracking": {
                **_stratum_delta(nat_rows, resolved_ids, fam),
                "bar": "must not degrade (correct tracking)"}}

    # --- assemble -----------------------------------------------------------
    return {
        "id": "V2.4-M1-RESULTS",
        "mapping_mode": "plan_stem_wrapper",
        "pattern_list_home": "router_sweep.PLAN_FRAME_ATTRIBUTE_PATTERNS",
        "families": {"present": list(present),
                     "absent": {fam: NA_STATUS for fam in absent}},
        "before": {
            "synres_promotion_ITT": before_synres(),
            "funnel": before_table,
            "contamination_gated_commits": CONTAM_GATED_COMMITS_BEFORE,
            "natural_strata_n": {"at_risk": NATURAL_AT_RISK_N,
                                 "resolved_true": NATURAL_RESOLVED_TRUE_N},
        },
        "after": {
            "funnel": after_table,
            "unmapped_tuple_rate": {
                "before": unmapped_rates(frozen_bases),
                "after": unmapped_rates({t: after_root / t
                                         for t in (TREE_V2, TREE_V22)})},
        },
        "primary_promotion_resolving_hedges": {
            "natural_resolved_true": prom_delta,
            "synres_carried": {
                "blocks": before_synres(),
                "delta": 0.0,
                "note": "STRUCTURAL ZERO, disclosed: the frozen SYN-RES "
                        "pathway binds its staging key by construction and "
                        "contains no key-mapper call — the M1 fix cannot "
                        "move it; carried as the prereg's 46-76% baseline "
                        "anchor only."}},
        "guards": {"contamination": contamination,
                   "correction_recall": corr_recall},
        "natural_rescore": {"rows": nat_rows, "per_family": natural},
        "pin1_diagnostic": pin1_diagnostic(
            after_root, tuple(FAM_NAME[f] for f in present)),
        "scoring_judge_calls": {"total": scorer.judge_calls,
                                "fresh": scorer.fresh_judge_calls},
        "after_funnel_records_n": len(after_records),
    }


def render_report(res: dict) -> str:
    lines = [
        "# v2.4 M1 — key-mapper fix for plan-shaped facts (BEFORE/AFTER)",
        "",
        "Mapping mode: `plan_stem_wrapper` (decline-then-stem-re-offer; frozen "
        "pattern list `router_sweep.PLAN_FRAME_ATTRIBUTE_PATTERNS`). BEFORE = "
        "frozen v2 gated artifacts (read, hard-asserted, never recomputed; "
        "anchored on the gated/cache side per impl_review_v24_m1.md §1.1). "
        "AFTER = gated columns re-run with the fix ON "
        "(results/v2.4/m1_after/).",
        "",
    ]
    fams = res.get("families") or {}
    if fams.get("absent"):
        lines += [
            "AFTER family coverage: "
            + ", ".join(FAM_NAME.get(f, f) for f in fams["present"])
            + " only — "
            + "; ".join(f"{FAM_NAME.get(f, f)}: {status}"
                        for f, status in fams["absent"].items())
            + ". Absent families carry frozen BEFORE values; guards and "
            "PIN-1 diagnostics cover present families only.",
            "",
        ]
    lines += [
        "## Primary — end-to-end promotion of resolving hedges",
        "",
        "SYN-RES anchor (frozen; structural zero under M1 — no key-mapper "
        "call exists in that pathway):",
        "",
        "| block | promotion ITT |",
        "|---|---|",
    ]
    for k, v in res["before"]["synres_promotion_ITT"].items():
        lines.append(f"| {k} | {v['x']}/{v['n']} = {v['point']:.2f} |")
    lines += ["", "Natural resolved-true stratum (n=29 resolving hedges; "
              "mapper-sensitive measurement, cluster-bootstrap CIs):", "",
              "| family | before | after | delta [CI] |", "|---|---|---|---|"]
    for fam, d in res["primary_promotion_resolving_hedges"][
            "natural_resolved_true"].items():
        if d.get("status"):
            b = d["before"]
            lines.append(f"| {fam} | {b['x']}/{b['n']} | — | {d['status']} |")
            continue
        b, a = d["before"], d["after"]
        ci = d["delta_ci"]
        lines.append(
            f"| {fam} | {b['x']}/{b['n']} | {a['x']}/{a['n']} | "
            f"{d['delta_mean']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] |"
            if d["delta_mean"] is not None else f"| {fam} | — | — | — |")
    lines += ["", "## Guards", "", "### Contamination (gated hedged-value "
              "commit rate; bar: Wilson upper <= 5%)", "",
              "| block | before | after | Wilson upper | pass |",
              "|---|---|---|---|---|"]
    for k, v in res["guards"]["contamination"].items():
        a = v["after"]
        hi = a["wilson_hi"]
        lines.append(f"| {k} | {v['before']['x']}/{v['before']['n']} | "
                     f"{a['x']}/{a['n']} | "
                     f"{hi:.4f} | {'PASS' if v['pass'] else 'FAIL'} |")
    lines += ["", "### Correction recall (v2 CORR-RECALL rule)", "",
              "| block | before | after | pass |", "|---|---|---|---|"]
    for k, v in res["guards"]["correction_recall"].items():
        lines.append(f"| {k} | {v['before']['x']}/{v['before']['n']} | "
                     f"{v['after']['x']}/{v['after']['n']} | "
                     f"{'PASS' if v['pass'] else 'FAIL'} |")
    lines += ["", "### Unmapped-tuple rate (reported, no bar)", "",
              "| cell | before | after |", "|---|---|---|"]
    ur = res["after"]["unmapped_tuple_rate"]
    for cell in sorted(ur["before"]):
        b = ur["before"][cell]
        a = ur["after"].get(cell)
        lines.append(
            f"| {cell} | {b['n_unmapped']}/{b['n_tuples']} = {b['rate']:.3f} | "
            + (f"{a['n_unmapped']}/{a['n_tuples']} = {a['rate']:.3f} |"
               if a else "— |"))
    lines += ["", "## Natural strata re-scores (frozen hedge-value sets, "
              "AFTER gated stores)", "",
              "| family | stratum | before | after | delta [CI] | bar |",
              "|---|---|---|---|---|---|"]
    for fam, blocks in res["natural_rescore"]["per_family"].items():
        for name, s in blocks.items():
            if s.get("status"):
                b = s["before"]
                lines.append(f"| {fam} | {name} | {b['x']}/{b['n']} | — | "
                             f"{s['status']} | {s['bar']} |")
                continue
            ci = s["delta_ci"]
            lines.append(
                f"| {fam} | {name} | {s['before_x']}/{s['n']} | "
                f"{s['after_x']}/{s['n']} | {s['delta_mean']:+.3f} "
                f"[{ci[0]:+.3f}, {ci[1]:+.3f}] | {s['bar']} |"
                if s["delta_mean"] is not None
                else f"| {fam} | {name} | — | — | — | {s['bar']} |")
    p = res["pin1_diagnostic"]
    frac = (f"{p['fraction_off_variant']:.3f}"
            if p["fraction_off_variant"] is not None else "n/a")
    lines += ["", "## PIN-1 diagnostic stratum (no bar)", "",
              f"Stem-mapped rows: {p['n_stem_mapped']}; mapped to a key other "
              f"than the derived-stem candidate: {p['n_off_variant']} "
              f"(fraction {frac}).",
              "", "## Funnel (AFTER; planted_spec + natural rebuilt via "
              "staging_funnel; t3 carried frozen)", "",
              "| population/family/benchmark | n | extracted | mapped | "
              "routed_spec | staged | unmapped-target-in-cands |",
              "|---|---|---|---|---|---|---|"]
    for k in sorted(res["after"]["funnel"]):
        b = res["after"]["funnel"][k]
        lines.append(f"| {k} | {b['n']} | {b['extracted']} | {b['mapped']} | "
                     f"{b['routed_speculative']} | "
                     f"{b['staged_with_predicates']} | "
                     f"{b['unmapped_target_in_candidates']} |")
    jc = res["scoring_judge_calls"]
    lines += ["", f"Scoring equivalence-judge calls: {jc['total']} "
              f"({jc['fresh']} fresh, counted under stage m1).", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# smoke — ONE substrate before/after, <= fresh-cap calls (default 10)
# ---------------------------------------------------------------------------

async def smoke(qid: str, fresh_cap: int = 10) -> dict:
    """BEFORE: mode-off replay of `qid` (mem0_gated) under CallBudget(0) —
    any cache miss aborts loudly — byte-compared against the frozen timeline.
    AFTER: the same stream with the fix ON, budget-guarded (stage m1, capped
    at `fresh_cap`), artifacts under results/v2.4/smoke/ only."""
    job = driver.lme_cell_jobs([qid])[0]
    frozen_path = V2_DIR / "mem0_gated" / "timelines" / f"{qid}.json"
    frozen = frozen_path.read_text()

    off_guard = driver.CallBudget(0)
    off_guard.install()
    try:
        tl_off = await gated_store.replay_gated(
            job.substrate, job.key_schema_entry, job.manifest_rows,
            family="mem0", aux_keys=job.aux_keys, mapper=None)
    finally:
        off_guard.uninstall()
    off_bytes = json.dumps(gated_store.timeline_to_json(tl_off),
                           ensure_ascii=False, indent=1)
    byte_identical = off_bytes == frozen

    guard = m1_guard(local_cap=fresh_cap)
    guard.install()
    try:
        tl_on = await gated_store.replay_gated(
            job.substrate, job.key_schema_entry, job.manifest_rows,
            family="mem0", aux_keys=job.aux_keys, mapper=make_mapper())
    finally:
        guard.uninstall()
    on_json = gated_store.timeline_to_json(tl_on)
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    (SMOKE_DIR / f"{qid}_mem0_gated_fix_on.json").write_text(
        json.dumps(on_json, ensure_ascii=False, indent=1))

    stem_gates = [g for g in on_json["gate_log"] if g.get("plan_framed")]
    stem_attempts = [u for u in on_json["unmapped"]
                     if (u.get("plan_stem") or {}).get("stem_mapped") is False]
    d_off, d_on = tl_off.diagnostics, tl_on.diagnostics
    out = {
        "qid": qid,
        "before": {"byte_identical_to_frozen": byte_identical,
                   "fresh_calls": off_guard.fresh,
                   "n_mapped": d_off["n_mapped"],
                   "n_unmapped": d_off["n_unmapped"],
                   "n_staged": d_off["n_staged"]},
        "after": {"fresh_calls": guard.fresh,
                  "n_mapped": d_on["n_mapped"],
                  "n_unmapped": d_on["n_unmapped"],
                  "n_staged": d_on["n_staged"],
                  "n_stem_mapped": len(stem_gates),
                  "n_stem_attempted_declined": len(stem_attempts),
                  "stem_rows": [{
                      "original_attribute": g["original_attribute"],
                      "derived_stem": g["derived_stem"],
                      "mapped_key": g["key"], "class": g["class"],
                      "action": g["action"]} for g in stem_gates]},
    }
    (SMOKE_DIR / f"{qid}_smoke_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="v2.4 M1 mapper-fix runner")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("budget")
    sub.add_parser("before")
    runp = sub.add_parser("run")
    runp.add_argument("--groups", default="lme,locomo,expansion")
    runp.add_argument("--families", default="mem0,zep")
    runp.add_argument("--qids", default=None)
    runp.add_argument("--out", default=str(AFTER_DIR))
    runp.add_argument("--call-cap", type=int, default=None)
    scorep = sub.add_parser("score")
    scorep.add_argument("--after", default=str(AFTER_DIR))
    smokep = sub.add_parser("smoke")
    smokep.add_argument("--qid", required=True)
    smokep.add_argument("--fresh-cap", type=int, default=10)
    args = ap.parse_args()

    if args.cmd == "budget":
        budget = ensure_budget()
        print(f"[mapper_fix] budget -> {BUDGET_PATH}: envelope "
              f"{budget['envelope']:,}, spent {budget['spent']}")
        return
    if args.cmd == "before":
        guard = driver.CallBudget(0)
        guard.install()
        try:
            synres = before_synres()
            _, table = before_funnel()
            strata = natural_strata()
        finally:
            guard.uninstall()
        print(json.dumps({"synres_promotion_ITT": synres,
                          "natural_strata_n": {k: len(v) for k, v in strata.items()},
                          "funnel_blocks": len(table),
                          "fresh_calls": guard.fresh}, indent=1))
        return
    if args.cmd == "run":
        groups = tuple(g.strip() for g in args.groups.split(",") if g.strip())
        families = tuple(f.strip() for f in args.families.split(",") if f.strip())
        qids = ([q.strip() for q in args.qids.split(",") if q.strip()]
                if args.qids else None)
        guard = m1_guard(local_cap=args.call_cap)
        summary = asyncio.run(run_after(groups, families, qids,
                                        Path(args.out), guard))
        print(json.dumps(summary, indent=1))
        return
    if args.cmd == "score":
        guard = m1_guard()
        guard.install()
        try:
            res = asyncio.run(compute_scores(Path(args.after)))
        finally:
            guard.uninstall()
        res["score_fresh_calls"] = guard.fresh
        V24_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(res, indent=1, ensure_ascii=False))
        REPORT_PATH.write_text(render_report(res))
        print(f"[mapper_fix] score -> {RESULTS_PATH} + {REPORT_PATH} "
              f"({guard.fresh} fresh calls)")
        return
    if args.cmd == "smoke":
        out = asyncio.run(smoke(args.qid, args.fresh_cap))
        print(json.dumps(out, indent=1, ensure_ascii=False))
        return


if __name__ == "__main__":
    main()
