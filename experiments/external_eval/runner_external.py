"""runner_external.py — the four frozen systems over the ADAPTED external
datasets (protocol/CHANGELOG.md 2026-07-30 "EXTERNAL DATASET VALIDATION
COMPLETE; adapter phase authorized").

SYSTEMS (the frozen four-cell grid, WITH the adopted repair):
  * em_cw / ez_cw — ungated write-through, frozen v2 mapper.
  * ez_cg        — zep family, gated, frozen v2 mapper.
  * em_cg        — cautious family, gated, WITH the adopted M1 plan-stem
    repair as the DEFAULT configuration (changelog 2026-07-09 "REPAIR
    ADOPTED": the stem re-offer is part of the default cautious-gated system
    definition). Implementation is the m4_eval 5th-arm LOCAL-mapper pattern
    verbatim: store.make_tuple_key_mapper(mapper_fix.PLAN_FRAME_ATTRIBUTE_
    PATTERNS) passed into the extract+map seam for the em_cg recs only —
    never installed globally (assert_mapper_mode_off at entry AND teardown);
    em extraction and the wrapper's first mapping pass are byte-identical to
    em_cw's calls (llm.py cache-shared), so only `-planstem` re-offers on
    DECLINED plan-framed tuples + their downstream gate calls are fresh.

All per-turn seams are the frozen ones, imported never copied where
possible: gated_store.extract_turn_family under syn_res.LME_PROMPTS (D1 +
D2/LongMemEval), locomo_render dyadic single-pass (D2/LoCoMo),
store._map_tuple_key, classify.classify_item + gated_store.ARGMAX_ROUTE,
gated_store._generate_predicates, the frozen checker (test3 fire semantics,
consume-on-fire), ZepStore writes. The _gated_checker/_gated_tuples loop
bodies are the graded_eval/m4_eval bodies with external trace tags (the
established per-experiment pattern).

HOLDOUT GUARD: every case goes through adapter_d1.adapt / adapter_d2.
apply_patch, which call holdout_guard.assert_ingest_allowed — TEST-slice ids
raise HoldoutViolation unless EXTERNAL_TEST_RUN_AUTHORIZED=1 (set ONLY for
the preregistered one-shot test run). Case records land under
results/external_eval/<slice>/ so dev artifacts can never be mistaken for
the one-shot test run's.

DATASET 1 (per-conversation resumable case records, m4_eval case shape):
the full conversation stream (calendar-day sessions from the adapter), user
turns ingested, assistant turns context. ONE registered field per
conversation (the adapter's registry: attribute NAME from the held-out
label — construction metadata; probe question from the model-visible
questions file). Interpretive choice (the m4_eval choice-#3 analog,
disclosed): mapper candidates = [the registered key] alone — the
conversations are standalone, so the external key is the only registered
key and the mapper's "none" option keeps doing the real work. The record
carries, per cell: final + pending field values, staging rows with
predicates + status, checker rows and valid fires (absolute turn
coordinates — D1 has NO run-path notion of the resolution turn; that is
label metadata, read only by the scorer), the target-key event log, a
value-after-each-user-turn timeline (so the scorer can evaluate any
label-anchored boundary without rerun), and the probe answer by
DETERMINISTIC LOOKUP of the registered field (present + non-sentinel = the
answer, else abstention-by-absence). NO label value is read on this path;
equivalence judging against held-out expected answers happens ONLY in the
scorer (preregistered endpoints entry, before the one-shot test run).

DATASET 2 (per-patch endpoint records): the adapter's augmented REAL host
(LongMemEval ORACLE / LoCoMo via locomo_data). Replay window = the patched
session + every superseded session, chronological (the graded_eval
bounded-world precedent, disclosed in the record): LME ingests the window's
user turns with the window prefix as context; LoCoMo ingests every window
turn dyadically with the full augmented conversation as extraction context
(the adapter's contract — the window changes which turns are gate-ingested,
never a call's bytes). Mapper candidates (interpretive choices, disclosed
in-record and unresolved-by-design until the preregistered endpoints
entry):
  * LoCoMo — driver.locomo_candidates(host) = the substrate's frozen v2
    registered keys (the M2 "substrate's registered candidates" convention;
    blindness preserved). No external target key exists: D2 labels carry NO
    attribute names, so a D1-style external field registry is impossible.
  * LongMemEval — NO frozen registry exists for the oracle hosts (the v2
    registries cover the 67-question s_cleaned KU substrate; overlap with
    the 45 D2 oracle hosts is zero) -> candidates = [] and every tuple is
    DETERMINISTICALLY unmapped (the frozen mapper cannot return a candidate
    from an empty list; short-circuited without an LLM call, zero spend).
    LME D2 cells therefore measure extraction mechanics only, and the
    endpoints entry MUST settle the D2 key convention before the test run.
The record carries per cell: every gate row (with an inserted-turn subset +
action counts), staged rows + statuses, checker rows/fires, and the FULL
final store dump; plus the raw per-family extraction output of the inserted
turns, so any later key convention can be evaluated against what the
extractors actually produced.

BUDGET: stage "external" (ADVISORY forecast 15,000 — mapper_fix.ALLOCATION,
budget.json note 2026-07-30), spend recorded per fresh call to
results/v2.4/budget.json via forecast_v22.SharedCallBudget (flock-guarded —
safe for parallel background shards); the ONLY hard stop is the 120,000
total envelope.

CLI (repo root, pilot/.venv/bin/python experiments/external_eval/runner_external.py):
  run-d1  [--ids ...] [--call-cap N] [--shard K/N] [--out-root DIR]
          resumable per conversation; default ids = the frozen DEV slice
  run-d2  [--ids ...] [--call-cap N] [--shard K/N] [--out-root DIR]
          resumable per patch; default ids = the frozen DEV slice
  summary [--out-root DIR]   dev-slice MECHANICS summary (per-cell counts,
          marked DEV — no directional conclusions) -> dev_smoke_summary.{json,md}
  status  budget stage status
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401  (R1 layout: puts src/ + experiments/ on sys.path)

import adapter_d1
import adapter_d2
import classify
import driver
import forecast_v22
import gated_store
import holdout_guard
import llm
import locomo_render
import mapper_fix
import run_predicates as test3
import store
import syn_res
import zepstore
from gated_store import ARGMAX_ROUTE
from graded_eval import _CellSim, _ungated_tuples
from m4_eval import assert_mapper_mode_off

from paths import PROMPTS_DIR, RESULTS_DIR

V24_DIR = RESULTS_DIR / "v2.4"
BUDGET_PATH = V24_DIR / "budget.json"
OUT_ROOT = RESULTS_DIR / "external_eval"

STAGE = "external"
#: ADVISORY stage forecast (2026-07-30 adapter-phase authorization; 15,000
#: from headroom) — tracked and reported vs actuals, never a hard stop.
EXTERNAL_ADVISORY_FORECAST = 15_000

FAMILIES = ("em", "ez")
CELLS = ("em_cw", "em_cg", "ez_cw", "ez_cg")
GATED_CELLS = ("em_cg", "ez_cg")

D1_CANDIDATES_NOTE = (
    "candidates = [the conversation's one registered external field] (m4_eval "
    "interpretive-choice-#3 analog: standalone conversations, no benchmark "
    "host, the mapper's 'none' option does the real work)")
D2_LME_CANDIDATES_NOTE = (
    "candidates = [] — no frozen registry exists for the LongMemEval ORACLE "
    "hosts (v2 registries cover the s_cleaned KU substrate; 0/45 overlap) "
    "and D2 labels carry no attribute names for a D1-style external "
    "registry; every tuple is deterministically unmapped (no mapper call). "
    "The preregistered endpoints entry must settle the D2 key convention "
    "BEFORE the one-shot test run.")
D2_LOCOMO_CANDIDATES_NOTE = (
    "candidates = the host conversation's frozen v2 registered keys "
    "(driver.locomo_candidates — the M2 'substrate's registered candidates' "
    "convention, blindness preserved). No external target key is registered: "
    "D2 labels carry no attribute names.")

EM_CG_REPAIR_NOTE = (
    "em_cg runs WITH the adopted M1 plan-stem repair as the default "
    "cautious-gated configuration (changelog 2026-07-09 REPAIR ADOPTED): "
    "store.make_tuple_key_mapper(mapper_fix.PLAN_FRAME_ATTRIBUTE_PATTERNS), "
    "LOCAL mapper for this cell's recs only — the global frozen mapper is "
    "assert-checked untouched at entry and teardown.")


# ---------------------------------------------------------------------------
# Budget (advisory stage 'external'; hard limit = the 120k envelope only)
# ---------------------------------------------------------------------------

def external_guard(path: Path = BUDGET_PATH,
                   local_cap: int | None = None) -> forecast_v22.SharedCallBudget:
    """The external-eval shared-envelope guard (the m4_eval.m4_guard pattern,
    stage 'external'): the stage allocation is ADVISORY — exceeding the
    forecast prints a warning and CONTINUES; the only hard stop is the
    combined 120k envelope. SharedCallBudget's read-modify-write is
    flock-guarded, so parallel background shards share it safely."""
    budget = mapper_fix.ensure_budget(path)
    spent = budget["spent"].get(STAGE, 0)
    forecast = budget["allocation"].get(STAGE, EXTERNAL_ADVISORY_FORECAST)
    if spent >= forecast:
        print(f"[external budget ADVISORY] stage spend {spent:,} >= forecast "
              f"{forecast:,} — continuing (stages are advisory per the "
              "2026-07-08 amendment; only the total envelope is hard)")
    remaining = forecast_v22.budget_remaining(budget)
    cap = remaining if local_cap is None else min(local_cap, remaining)
    return forecast_v22.SharedCallBudget(STAGE, path=path, local_cap=cap)


def advisory_status(path: Path = BUDGET_PATH) -> dict:
    budget = mapper_fix.ensure_budget(path)
    spent = budget["spent"].get(STAGE, 0)
    forecast = budget["allocation"].get(STAGE, EXTERNAL_ADVISORY_FORECAST)
    return {
        "stage": STAGE,
        "advisory_forecast": forecast,
        "spent": spent,
        "over_forecast": spent > forecast,
        "envelope": budget["envelope"],
        "envelope_spent_total": sum(budget["spent"].values()),
        "mode": "ADVISORY stage / HARD total envelope (2026-07-08 amendment)",
    }


# ---------------------------------------------------------------------------
# Paths (per-slice roots: dev artifacts can never shadow the one-shot test's)
# ---------------------------------------------------------------------------

def case_path(dataset: str, slice_name: str, item_id: str,
              out_root: Path = OUT_ROOT) -> Path:
    sub = "cases_d1" if dataset == "dataset1" else "cases_d2"
    return out_root / slice_name / sub / f"{item_id}.json"


# ---------------------------------------------------------------------------
# Shared per-turn machinery (frozen seams; external trace tags)
# ---------------------------------------------------------------------------

async def _map_all(tuples, candidates, tag_base: str, mapper=None) -> list:
    """Map every tuple of one turn. Empty candidate list short-circuits
    deterministically (the frozen mapper's by_name lookup over [] can only
    return None — no call, zero spend), identically for the frozen mapper and
    the stem wrapper (a re-offer against [] is equally structurally None)."""
    if not candidates:
        return [(None, "none", "no registered candidate keys (deterministic)")
                for _ in tuples]
    map_call = store._map_tuple_key if mapper is None else mapper
    return list(await asyncio.gather(*[
        map_call(tup, candidates, trace_tag=f"{tag_base}-{j}")
        for j, tup in enumerate(tuples)]))


async def _extract_and_map_lme(family: str, turns: list[dict], pos: int,
                               candidates, case_id: str, dataset_tag: str,
                               mapper=None) -> dict:
    """(tuples, mapped) for one user turn under the frozen LME B2 pathway
    (full-prefix context, session-level dates) — the m4_eval._extract_and_map
    body with external trace tags. ``mapper`` is the m4 5th-arm seam: None =
    the frozen store._map_tuple_key; em_cg passes the adopted stem wrapper
    (first pass byte-identical -> cache-shared with em_cw)."""
    si, ti = turns[pos]["session_idx"], turns[pos]["turn_idx"]
    tag = f"ext-{dataset_tag}-extract-{family}-{case_id}-s{si}t{ti}"
    prior = [{"role": t["role"], "content": t["content"], "date": t["date"]}
             for t in turns[:pos]]
    tuples = await gated_store.extract_turn_family(
        prior, turns[pos]["content"], tag,
        system_prompt=(PROMPTS_DIR / syn_res.LME_PROMPTS[family]).read_text(),
        current_date=turns[pos]["date"])
    mapped = await _map_all(
        tuples, candidates,
        f"ext-{dataset_tag}-keymap-{family}-{case_id}-s{si}t{ti}", mapper)
    return {"pos": pos, "session_idx": si, "turn_idx": ti,
            "date": turns[pos]["date"], "turn_text": turns[pos]["content"],
            "inserted": bool(turns[pos].get("inserted")),
            "tuples": tuples, "mapped": mapped}


async def _extract_and_map_locomo(family: str, conv, turns: list[dict],
                                  pos: int, candidates, case_id: str,
                                  mapper=None) -> dict:
    """(tuples, mapped) for one LoCoMo turn under the frozen dyadic
    single-pass pathway (graded_eval's locomo branch with external tags).
    Extraction context is the FULL augmented conversation (dyadic_turn_job);
    the bounded replay window only selects which turns are ingested."""
    row = turns[pos]
    si, ti = row["session_idx"], row["turn_idx"]
    tag = f"ext-d2-extract-{family}-{case_id}-s{si}t{ti}"
    job = locomo_render.dyadic_turn_job(conv, si, ti,
                                        family=syn_res.DYADIC_FAMILY[family])
    tuples = await locomo_render.extract_dyadic_turn(
        job["prior_turns"], job["current_turn"], tag,
        family=syn_res.DYADIC_FAMILY[family], current_date=job["current_date"])
    mapped = await _map_all(
        tuples, candidates, f"ext-d2-keymap-{family}-{case_id}-s{si}t{ti}",
        mapper)
    return {"pos": pos, "session_idx": si, "turn_idx": ti,
            "date": row["date"], "turn_text": row["content"],
            "inserted": bool(row.get("inserted")),
            "tuples": tuples, "mapped": mapped}


async def _gated_checker(sim: _CellSim, rec: dict, family: str, case_id: str,
                         checker_system: str) -> None:
    """Checker over pending items on one turn (the frozen fire semantics —
    graded_eval._gated_checker with external trace tags; absolute turn
    coordinates + user_idx instead of resolution-relative offsets, which do
    not exist on the external run path)."""
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
            trace_tag=f"ext-check-{family}-{case_id}-{store_id}-"
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
            "turn_idx": rec["turn_idx"], "user_idx": rec.get("user_idx"),
            "pending_ids": sorted(pending_ids), "fires": fires,
            "malformed_output": malformed})


async def _gated_tuples(sim: _CellSim, rec: dict, turns: list[dict],
                        family: str, case_id: str, gen_system: str) -> None:
    """Phase-3 gate loop body for one turn (graded_eval._gated_tuples with
    external ids/tags; frozen argmax routing verbatim). Staged rows carry
    their staging-turn coordinates (the m4_eval convention)."""
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
            "item_id": f"ext-{family}-{case_id}-s{rec['session_idx']}"
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
            staged_id = f"stg-ext-{family}-{case_id}-s{rec['session_idx']}" \
                        f"t{rec['turn_idx']}-{j}"
            row_shim = {"pos": rec["pos"], "turn_text": rec["turn_text"]}
            staged = await gated_store._generate_predicates(
                stream, row_shim, cand, tup, staged_id=staged_id,
                stream_id=case_id, family=family, gen_system=gen_system,
                diagnostics=sim.diag)
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


def _answered(value) -> bool:
    """Deterministic-lookup semantics for the task probe (m4_eval._answered):
    a present, non-sentinel current value answers; anything else is
    abstention-by-absence."""
    return value is not None and not zepstore.is_sentinel(str(value))


def _jsonable_event(event: dict) -> dict:
    out = dict(event)
    out["key"] = list(out["key"])
    return out


def _staged_row_public(sid: str, srow: dict) -> dict:
    """The serialized staged row (the m4_eval staged_target field subset +
    identity/coordinates)."""
    return {
        "staged_id": sid, "status": srow["status"],
        "store_id": srow.get("store_id"),
        "entity": srow["entity"], "attribute": srow["attribute"],
        "hypothesized_value": srow["hypothesized_value"],
        "promote_when": srow["promote_when"],
        "invalidate_when": srow["invalidate_when"],
        "session_idx": srow.get("session_idx"),
        "turn_idx": srow.get("turn_idx"),
    }


def _valid_fires(checker_rows: list[dict]) -> list[dict]:
    return [
        {"session_idx": c["session_idx"], "turn_idx": c["turn_idx"],
         "user_idx": c.get("user_idx"), "store_id": c["store_id"],
         "staged_id": f["staged_id"], "action": f["action"],
         "evidence_span": f["evidence_span"], "span_valid": f["span_valid"]}
        for c in checker_rows for f in c["fires"] if not f["invalid"]]


def _turn_diag_row(rec: dict, target: store.CanonicalKey | None) -> dict:
    """Per-turn tuples-extracted/-mapped diagnostics (m4_eval._turn_diag_row,
    target optional for D2)."""
    target_values = []
    n_mapped_any = 0
    for tup, mres in zip(rec["tuples"], rec["mapped"]):
        cand, _raw, _rat, _meta = gated_store._unpack_map_result(mres)
        if cand is None:
            continue
        n_mapped_any += 1
        if target is not None and cand.key == target.key:
            target_values.append(tup.value)
    row = {"session_idx": rec["session_idx"], "turn_idx": rec["turn_idx"],
           "user_idx": rec.get("user_idx"), "inserted": rec.get("inserted"),
           "n_tuples": len(rec["tuples"]), "n_mapped_any": n_mapped_any}
    if target is not None:
        row["n_mapped_target"] = len(target_values)
        row["target_values"] = target_values
    return row


def _wrapper_mapping_meta(recs: list[dict],
                          target: store.CanonicalKey | None) -> list[dict]:
    """Per-turn stem-wrapper mapping meta for em_cg (m4_eval._fixed_mapping_
    meta — PIN-1 audit fields: original_attribute, derived_stem,
    pattern_matched, mapped_key)."""
    out = []
    for rec in recs:
        tuples_meta = []
        for tup, mres in zip(rec["tuples"], rec["mapped"]):
            cand, _raw, _rat, meta = gated_store._unpack_map_result(mres)
            tuples_meta.append({
                "attribute": tup.attribute, "value": tup.value,
                "mapped_key": cand.name if cand is not None else None,
                "mapped_to_target": (target is not None and cand is not None
                                     and cand.key == target.key),
                "stem_mapped": meta.get("stem_mapped"),
                "derived_stem": meta.get("derived_stem"),
                "pattern_matched": meta.get("pattern_matched"),
                "original_attribute": meta.get("original_attribute"),
            })
        out.append({"session_idx": rec["session_idx"],
                    "turn_idx": rec["turn_idx"],
                    "user_idx": rec.get("user_idx"), "tuples": tuples_meta})
    return out


async def _run_policy_cell(policy: str, recs: list[dict], family: str,
                           case_id: str, turns: list[dict], benchmark: str,
                           gen_system: str, checker_system: str,
                           target: store.CanonicalKey | None = None,
                           inserted_coords: set | None = None) -> dict:
    """Run ONE policy over one family's per-turn recs; returns the per-cell
    record. The body is mapper-agnostic (the recs already carry the mapping
    results — frozen or adopted-wrapper), so all four cells share this exact
    code path; a fresh _CellSim per call keeps every cell's store state
    independent."""
    sim = _CellSim(policy, benchmark)
    target_events: list[dict] = []
    value_timeline: list[dict] = []
    store_id = sim.route(target.key) if target is not None else None
    for rec in recs:
        if target is not None:
            n_events = len(sim.zstore(store_id).events)
        if policy == "gated":
            await _gated_checker(sim, rec, family, case_id, checker_system)
            await _gated_tuples(sim, rec, turns, family, case_id, gen_system)
        else:
            await _ungated_tuples(sim, rec)
        if target is not None:
            for event in sim.zstore(store_id).events[n_events:]:
                if tuple(event["key"]) != target.key:
                    continue
                target_events.append({
                    **_jsonable_event(event),
                    "session_idx": rec["session_idx"],
                    "turn_idx": rec["turn_idx"],
                    "user_idx": rec.get("user_idx"),
                })
            value_timeline.append({
                "session_idx": rec["session_idx"], "turn_idx": rec["turn_idx"],
                "user_idx": rec.get("user_idx"),
                "value": sim.lookup(target.key)})
    staged_rows = [_staged_row_public(sid, srow)
                   for sid, srow in sim.staged_rows.items()]
    checker_rows = sim.checker_rows
    out = {
        "final_store": {sid: {f"{k[0]}/{k[1]}": z.lookup(k) for k in z.keys()}
                        for sid, z in sim.stores.items()},
        "staged_rows": staged_rows,
        "checker_rows": checker_rows,
        "valid_fires": _valid_fires(checker_rows),
        "gate_rows": sim.gate_rows,
        "diag": dict(sim.diag),
    }
    if target is not None:
        final = sim.lookup(target.key)
        staged_target = [s for s in staged_rows
                         if (s["entity"], s["attribute"]) == target.key]
        probe_value = final if _answered(final) else None
        out.update({
            "final_value": final,
            "final_present": _answered(final),
            "staged_target": staged_target,
            "target_pending_at_end": [s["staged_id"] for s in staged_target
                                      if s["status"] == "pending"],
            "pending_values_at_end": [s["hypothesized_value"]
                                      for s in staged_target
                                      if s["status"] == "pending"],
            "target_events": target_events,
            "target_value_after_turn": value_timeline,
            "gate_rows_target": [g for g in sim.gate_rows
                                 if g.get("key") == target.name],
            "probe": {
                "value": probe_value,
                "answered": probe_value is not None,
                "semantics": ("deterministic lookup of the registered field; "
                              "equivalence judging vs the held-out expected "
                              "answer happens ONLY in the scorer"),
            },
        })
    if inserted_coords is not None:
        ins = [g for g in sim.gate_rows
               if (g["session_idx"], g["turn_idx"]) in inserted_coords]
        out["inserted_gate_rows"] = ins
        out["inserted_action_counts"] = dict(Counter(
            g["action"] for g in ins))
        out["inserted_staged_ids"] = [g["staged_id"] for g in ins
                                      if g.get("staged_id")]
    return out


def _attach_user_idx(recs: list[dict]) -> list[dict]:
    for k, rec in enumerate(recs):
        rec["user_idx"] = k
    return recs


# ---------------------------------------------------------------------------
# Dataset 1 — per-conversation case records
# ---------------------------------------------------------------------------

async def run_d1_case(conversation_id: str) -> dict:
    """One D1 conversation through all four systems over the full adapted
    stream (HARD holdout guard + leak asserts inside adapter_d1.adapt)."""
    stream, key_row = adapter_d1.adapt(conversation_id)
    slice_name = holdout_guard.slice_of(adapter_d1.DATASET, conversation_id)
    turns = adapter_d1.stream_turns(stream)
    user_positions = [i for i, t in enumerate(turns) if t["role"] == "user"]
    target = adapter_d1.canonical_key(key_row)
    candidates = [target]  # D1_CANDIDATES_NOTE (module docstring)
    gen_system = gated_store.GENERATOR_PROMPT_PATH.read_text()
    checker_system = gated_store.CHECKER_PROMPT_PATH.read_text()

    out = {
        "conversation_id": conversation_id,
        "dataset": adapter_d1.DATASET,
        "slice": slice_name,
        "question_id": key_row["question_id"],
        "target_key": target.name,
        "probe_question": target.probe_question,
        "candidates_convention": D1_CANDIDATES_NOTE,
        "em_cg_repair": EM_CG_REPAIR_NOTE,
        "geometry": {
            "n_sessions": len(stream["sessions"]),
            "n_turns": len(turns),
            "n_user_turns": len(user_positions),
            "session_dates": [s["date"] for s in stream["sessions"]],
        },
        "cells": {},
        "diagnostics": {},
    }

    async def _recs(family: str, mapper=None) -> list[dict]:
        return _attach_user_idx([
            await _extract_and_map_lme(family, turns, pos, candidates,
                                       conversation_id, "d1", mapper=mapper)
            for pos in user_positions])

    # em: frozen mapping feeds em_cw; the ADOPTED stem wrapper feeds em_cg
    # (default cautious-gated config). Extraction + first-pass mapping are
    # byte-identical across the two rec sets (llm.py cache-shared).
    assert_mapper_mode_off()          # global frozen mapper untouched at entry
    recs_em = await _recs("em")
    wrapper = store.make_tuple_key_mapper(mapper_fix.PLAN_FRAME_ATTRIBUTE_PATTERNS)
    recs_em_fixed = await _recs("em", mapper=wrapper)
    assert_mapper_mode_off()          # teardown: the wrapper stayed local
    recs_ez = await _recs("ez")

    out["diagnostics"]["em"] = [_turn_diag_row(r, target) for r in recs_em]
    out["diagnostics"]["em_fixed"] = [_turn_diag_row(r, target)
                                      for r in recs_em_fixed]
    out["diagnostics"]["ez"] = [_turn_diag_row(r, target) for r in recs_ez]

    kw = dict(case_id=conversation_id, turns=turns, benchmark="lme",
              gen_system=gen_system, checker_system=checker_system,
              target=target)
    out["cells"]["em_cw"] = await _run_policy_cell("ungated", recs_em, "em", **kw)
    em_cg = await _run_policy_cell("gated", recs_em_fixed, "em", **kw)
    em_cg["mapping_meta"] = _wrapper_mapping_meta(recs_em_fixed, target)
    out["cells"]["em_cg"] = em_cg
    out["cells"]["ez_cw"] = await _run_policy_cell("ungated", recs_ez, "ez", **kw)
    out["cells"]["ez_cg"] = await _run_policy_cell("gated", recs_ez, "ez", **kw)
    return out


# ---------------------------------------------------------------------------
# Dataset 2 — per-patch endpoint records
# ---------------------------------------------------------------------------

def _tuple_public(tup) -> dict:
    d = tup.__dict__ if hasattr(tup, "__dict__") else dict(tup)
    return {k: v for k, v in d.items()}


def _mapped_key_names(rec: dict) -> list:
    out = []
    for mres in rec["mapped"]:
        cand, _raw, _rat, _meta = gated_store._unpack_map_result(mres)
        out.append(cand.name if cand is not None else None)
    return out


async def run_d2_case(patch_id: str) -> dict:
    """One D2 patch: apply to the REAL host (adapter_d2, guarded), replay the
    bounded window (patched + superseded sessions) through all four systems,
    record per-patch endpoint mechanics."""
    aug = adapter_d2.apply_patch(patch_id)
    slice_name = holdout_guard.slice_of(adapter_d2.DATASET, patch_id)
    gen_system = gated_store.GENERATOR_PROMPT_PATH.read_text()
    checker_system = gated_store.CHECKER_PROMPT_PATH.read_text()

    if aug["host_dataset"] == "longmemeval":
        benchmark = "lme"
        window = sorted({aug["insertion"]["session_idx"]}
                        | {s["session_idx"] for s in aug["superseded"]})
        turns = [r for r in adapter_d2.lme_stream_turns(aug)
                 if r["session_idx"] in window]  # already chronological
        ingest_positions = [i for i, t in enumerate(turns)
                            if t["role"] == "user"]
        candidates: list = []
        candidates_note = D2_LME_CANDIDATES_NOTE
        conv = None
    else:
        benchmark = "locomo"
        conv = aug["conversation"]
        window = aug["replay_session_idxs"]
        turns = adapter_d2.locomo_replay_turns(aug)
        ingest_positions = list(range(len(turns)))
        candidates = driver.locomo_candidates(aug["host_record_id"])
        candidates_note = D2_LOCOMO_CANDIDATES_NOTE

    inserted_coords = {(t["session_idx"], t["turn_idx"])
                       for t in turns if t.get("inserted")}
    assert inserted_coords, f"{patch_id}: no inserted turns in the window"

    out = {
        "patch_id": patch_id,
        "dataset": adapter_d2.DATASET,
        "slice": slice_name,
        "host_dataset": aug["host_dataset"],
        "host_record_id": aug["host_record_id"],
        "insertion": aug["insertion"],
        "superseded": aug["superseded"],
        "replay_window": {
            "kind": ("bounded: patched session + superseded sessions, "
                     "chronological (graded_eval bounded-world precedent)"),
            "session_idxs": list(window),
            "n_turns": len(turns),
            "n_ingest_positions": len(ingest_positions),
            "n_inserted_turns": len(inserted_coords),
        },
        "candidates": [c.name for c in candidates],
        "candidates_convention": candidates_note,
        "em_cg_repair": EM_CG_REPAIR_NOTE,
        "cells": {},
        "diagnostics": {},
        "inserted_turn_extraction": {},
    }

    async def _recs(family: str, mapper=None) -> list[dict]:
        if benchmark == "lme":
            recs = [await _extract_and_map_lme(family, turns, pos, candidates,
                                               patch_id, "d2", mapper=mapper)
                    for pos in ingest_positions]
        else:
            recs = [await _extract_and_map_locomo(family, conv, turns, pos,
                                                  candidates, patch_id,
                                                  mapper=mapper)
                    for pos in ingest_positions]
        return _attach_user_idx(recs)

    assert_mapper_mode_off()
    recs_em = await _recs("em")
    wrapper = store.make_tuple_key_mapper(mapper_fix.PLAN_FRAME_ATTRIBUTE_PATTERNS)
    recs_em_fixed = await _recs("em", mapper=wrapper)
    assert_mapper_mode_off()
    recs_ez = await _recs("ez")

    for family, recs in (("em", recs_em), ("em_fixed", recs_em_fixed),
                         ("ez", recs_ez)):
        out["diagnostics"][family] = [_turn_diag_row(r, None) for r in recs]
        out["inserted_turn_extraction"][family] = [
            {"session_idx": r["session_idx"], "turn_idx": r["turn_idx"],
             "turn_text": r["turn_text"],
             "tuples": [_tuple_public(t) for t in r["tuples"]],
             "mapped_keys": _mapped_key_names(r)}
            for r in recs if r["inserted"]]

    kw = dict(case_id=patch_id, turns=turns, benchmark=benchmark,
              gen_system=gen_system, checker_system=checker_system,
              inserted_coords=inserted_coords)
    out["cells"]["em_cw"] = await _run_policy_cell("ungated", recs_em, "em", **kw)
    em_cg = await _run_policy_cell("gated", recs_em_fixed, "em", **kw)
    em_cg["mapping_meta"] = _wrapper_mapping_meta(recs_em_fixed, None)
    out["cells"]["em_cg"] = em_cg
    out["cells"]["ez_cw"] = await _run_policy_cell("ungated", recs_ez, "ez", **kw)
    out["cells"]["ez_cg"] = await _run_policy_cell("gated", recs_ez, "ez", **kw)
    return out


# ---------------------------------------------------------------------------
# Run orchestration (resumable; shard-friendly; guarded budget)
# ---------------------------------------------------------------------------

async def run_dataset(dataset: str, ids: list[str], out_root: Path,
                      guard) -> dict:
    runner = run_d1_case if dataset == "dataset1" else run_d2_case
    summary = {"dataset": dataset, "n_cases": 0, "skipped": 0, "failed": []}
    guard.install()
    try:
        for item_id in ids:
            slice_name = holdout_guard.slice_of(dataset, item_id)  # fail closed
            path = case_path(dataset, slice_name, item_id, out_root)
            if path.exists():
                summary["skipped"] += 1
                continue
            try:
                record = await runner(item_id)
            except (SystemExit, holdout_guard.HoldoutViolation):
                raise
            except Exception as exc:  # noqa: BLE001 — shard keeps going
                if "EXHAUSTED" in str(exc) or "budget" in str(exc).lower():
                    raise  # budget stops are never swallowed
                summary["failed"].append({"id": item_id, "error": repr(exc)})
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, ensure_ascii=False, indent=1))
            summary["n_cases"] += 1
    finally:
        guard.uninstall()
    summary["fresh_calls"] = guard.fresh
    summary["advisory"] = advisory_status()
    return summary


def _shard(ids: list[str], spec: str | None) -> list[str]:
    if not spec:
        return ids
    k, n = spec.split("/")
    k, n = int(k), int(n)
    assert 0 <= k < n, f"bad shard spec {spec!r}"
    return ids[k::n]


# ---------------------------------------------------------------------------
# Dev-slice mechanics summary (per-cell counts; DEV — no directional claims)
# ---------------------------------------------------------------------------

def _load_cases(out_root: Path, slice_name: str, sub: str) -> list[dict]:
    root = out_root / slice_name / sub
    if not root.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(root.glob("*.json"))]


def _d1_cell_counts(cases: list[dict]) -> dict:
    per_cell = {}
    for cell in CELLS:
        n = len(cases)
        cc = {
            "n_cases": n,
            "final_present": sum(1 for c in cases
                                 if c["cells"][cell]["final_present"]),
            "probe_answered": sum(1 for c in cases
                                  if c["cells"][cell]["probe"]["answered"]),
            "any_target_event": sum(
                1 for c in cases if c["cells"][cell]["target_events"]),
            "any_target_commit_or_supersede": sum(
                1 for c in cases
                if any(e["action"] in ("commit", "supersede")
                       for e in c["cells"][cell]["target_events"])),
            "tuples_extracted": sum(c["cells"][cell]["diag"].get("n_tuples", 0)
                                    for c in cases),
            "tuples_mapped": sum(c["cells"][cell]["diag"].get("n_mapped", 0)
                                 for c in cases),
        }
        if cell in GATED_CELLS:
            statuses = Counter()
            for c in cases:
                statuses.update(s["status"]
                                for s in c["cells"][cell]["staged_target"])
            cc.update({
                "staged_target_any": sum(
                    1 for c in cases if c["cells"][cell]["staged_target"]),
                "target_pending_at_end": sum(
                    1 for c in cases
                    if c["cells"][cell]["target_pending_at_end"]),
                "staged_target_status_totals": dict(statuses),
                "checker_calls_total": sum(
                    len(c["cells"][cell]["checker_rows"]) for c in cases),
                "valid_fires_total": sum(
                    len(c["cells"][cell]["valid_fires"]) for c in cases),
            })
        per_cell[cell] = cc
    # family mapping diagnostics (the entry-repair visibility axis)
    fam = {}
    for family in ("em", "em_fixed", "ez"):
        rows = [d for c in cases for d in c["diagnostics"][family]]
        fam[family] = {
            "turns": len(rows),
            "tuples": sum(r["n_tuples"] for r in rows),
            "mapped_any": sum(r["n_mapped_any"] for r in rows),
            "mapped_target": sum(r["n_mapped_target"] for r in rows),
            "cases_with_any_target_map": sum(
                1 for c in cases
                if any(d["n_mapped_target"] for d in c["diagnostics"][family])),
        }
    stem = {"re_offers": 0, "stem_mapped": 0}
    for c in cases:
        for row in c["cells"]["em_cg"].get("mapping_meta", []):
            for t in row["tuples"]:
                if t["derived_stem"] is not None:
                    stem["re_offers"] += 1
                    if t["stem_mapped"]:
                        stem["stem_mapped"] += 1
    return {"per_cell": per_cell, "per_family_mapping": fam,
            "em_cg_stem_wrapper": stem}


def _d2_cell_counts(cases: list[dict]) -> dict:
    out = {}
    for host in ("longmemeval", "locomo"):
        sel = [c for c in cases if c["host_dataset"] == host]
        per_cell = {}
        for cell in CELLS:
            actions = Counter()
            ins_actions = Counter()
            staged_status = Counter()
            for c in sel:
                cc = c["cells"][cell]
                actions.update(g["action"] for g in cc["gate_rows"])
                ins_actions.update(cc.get("inserted_action_counts", {}))
                staged_status.update(s["status"] for s in cc["staged_rows"])
            per_cell[cell] = {
                "n_cases": len(sel),
                "tuples_extracted": sum(
                    c["cells"][cell]["diag"].get("n_tuples", 0) for c in sel),
                "tuples_mapped": sum(
                    c["cells"][cell]["diag"].get("n_mapped", 0) for c in sel),
                "gate_action_totals": dict(actions),
                "inserted_turn_action_totals": dict(ins_actions),
                "staged_status_totals": dict(staged_status),
                "checker_calls_total": sum(
                    len(c["cells"][cell]["checker_rows"]) for c in sel),
                "valid_fires_total": sum(
                    len(c["cells"][cell]["valid_fires"]) for c in sel),
                "final_store_rows_total": sum(
                    sum(len(v) for v in c["cells"][cell]["final_store"].values())
                    for c in sel),
            }
        ins_extract = {}
        for family in ("em", "em_fixed", "ez"):
            ins_extract[family] = {
                "inserted_turns": sum(len(c["inserted_turn_extraction"][family])
                                      for c in sel),
                "inserted_tuples": sum(
                    len(r["tuples"])
                    for c in sel for r in c["inserted_turn_extraction"][family]),
                "inserted_tuples_mapped": sum(
                    sum(1 for k in r["mapped_keys"] if k)
                    for c in sel for r in c["inserted_turn_extraction"][family]),
            }
        out[host] = {"n_cases": len(sel), "per_cell": per_cell,
                     "inserted_turn_extraction": ins_extract}
    return out


def build_summary(out_root: Path = OUT_ROOT,
                  slice_name: str = "dev") -> dict:
    d1 = _load_cases(out_root, slice_name, "cases_d1")
    d2 = _load_cases(out_root, slice_name, "cases_d2")
    expected = {"dataset1": len(holdout_guard.dev_ids("dataset1")),
                "dataset2": len(holdout_guard.dev_ids("dataset2"))}
    return {
        "slice": slice_name.upper(),
        "banner": ("DEV-SLICE MECHANICS SUMMARY — adapter/runner mechanics "
                   "only; NO directional conclusions (the preregistered "
                   "endpoints entry + one-shot test run come later)"),
        "coverage": {"dataset1": {"cases": len(d1), "expected_dev": expected["dataset1"]},
                     "dataset2": {"cases": len(d2), "expected_dev": expected["dataset2"]}},
        "dataset1": _d1_cell_counts(d1) if d1 else None,
        "dataset2": _d2_cell_counts(d2) if d2 else None,
        "budget": advisory_status(),
    }


def render_summary_md(s: dict) -> str:
    lines = [
        f"# External-eval {s['slice']} smoke — mechanics summary",
        "",
        f"**{s['banner']}**",
        "",
        f"Coverage: D1 {s['coverage']['dataset1']['cases']}/"
        f"{s['coverage']['dataset1']['expected_dev']} conversations; "
        f"D2 {s['coverage']['dataset2']['cases']}/"
        f"{s['coverage']['dataset2']['expected_dev']} patches.",
        "",
        f"Budget: stage `{s['budget']['stage']}` spent "
        f"{s['budget']['spent']:,} / advisory {s['budget']['advisory_forecast']:,}; "
        f"envelope total {s['budget']['envelope_spent_total']:,} / "
        f"{s['budget']['envelope']:,}.",
        "",
    ]
    if s["dataset1"]:
        lines += ["## Dataset 1 (per-cell counts, DEV)", "",
                  "| cell | n | final_present | probe_answered | staged_target | pending_at_end | checker calls | valid fires |",
                  "|---|---|---|---|---|---|---|---|"]
        for cell in CELLS:
            c = s["dataset1"]["per_cell"][cell]
            lines.append(
                f"| {cell} | {c['n_cases']} | {c['final_present']} | "
                f"{c['probe_answered']} | {c.get('staged_target_any', '—')} | "
                f"{c.get('target_pending_at_end', '—')} | "
                f"{c.get('checker_calls_total', '—')} | "
                f"{c.get('valid_fires_total', '—')} |")
        fam = s["dataset1"]["per_family_mapping"]
        lines += ["", "| family | tuples | mapped_any | mapped_target | cases w/ target map |",
                  "|---|---|---|---|---|"]
        for f in ("em", "em_fixed", "ez"):
            r = fam[f]
            lines.append(f"| {f} | {r['tuples']} | {r['mapped_any']} | "
                         f"{r['mapped_target']} | {r['cases_with_any_target_map']} |")
        stem = s["dataset1"]["em_cg_stem_wrapper"]
        lines += ["", f"em_cg stem wrapper: {stem['re_offers']} re-offers, "
                      f"{stem['stem_mapped']} stem-mapped.", ""]
    if s["dataset2"]:
        lines += ["## Dataset 2 (per-cell counts, DEV)", ""]
        for host, block in s["dataset2"].items():
            lines += [f"### {host} (n={block['n_cases']})", "",
                      "| cell | tuples | mapped | inserted-turn actions | staged statuses | checker calls |",
                      "|---|---|---|---|---|---|"]
            for cell in CELLS:
                c = block["per_cell"][cell]
                lines.append(
                    f"| {cell} | {c['tuples_extracted']} | {c['tuples_mapped']} | "
                    f"{json.dumps(c['inserted_turn_action_totals'])} | "
                    f"{json.dumps(c['staged_status_totals'])} | "
                    f"{c['checker_calls_total']} |")
            lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, dataset in (("run-d1", "dataset1"), ("run-d2", "dataset2")):
        p = sub.add_parser(name)
        p.set_defaults(dataset=dataset)
        p.add_argument("--ids", nargs="*", default=None)
        p.add_argument("--call-cap", type=int, default=None)
        p.add_argument("--shard", default=None, help="K/N over sorted ids")
        p.add_argument("--out-root", type=Path, default=OUT_ROOT)
    p = sub.add_parser("summary")
    p.add_argument("--out-root", type=Path, default=OUT_ROOT)
    p.add_argument("--slice", default="dev")
    sub.add_parser("status")
    args = ap.parse_args()

    if args.cmd == "status":
        print(json.dumps(advisory_status(), indent=1))
        return
    if args.cmd == "summary":
        s = build_summary(args.out_root, args.slice)
        out_dir = args.out_root / args.slice
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "dev_smoke_summary.json").write_text(
            json.dumps(s, ensure_ascii=False, indent=1))
        (out_dir / "dev_smoke_summary.md").write_text(render_summary_md(s))
        print(render_summary_md(s))
        return

    ids = args.ids if args.ids else holdout_guard.dev_ids(args.dataset)
    ids = _shard(sorted(ids), args.shard)
    guard = external_guard(local_cap=args.call_cap)
    summary = asyncio.run(run_dataset(args.dataset, ids, args.out_root, guard))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
