"""m4_eval.py — Protocol v2.4 M4-EVAL-r4: the full FIVE-arm experiment set on
the r4 realistic substrate (preregistered in protocol/CHANGELOG.md,
"2026-07-09 M4-EVAL-r4 preregistration: rerun on the r4 substrate + a 5th
entry-fix arm" — the binding spec; frozen BEFORE any run call; supersedes the
earlier r3 four-cell M4-EVAL spec).

SUBSTRATE: the 60 r4 themed synthetic conversations
(results/v2.4/m4/arcs.jsonl, coherent resolution openers + established prior
value; contract shas 3809ff7185→arcs 162032ef24; blind-judge-cleared),
20/20/20 resolve_true/abandoned/open. PROVISIONAL pending owner realism
review — every score row carries arc_id so the owner's accept/reject sheet
filters the reported stratum WITHOUT any rerun; the final reported stratum
is the owner-accepted subset.

ARMS: the four FROZEN cells em/ez × ungated(cw)/gated(cg) via the frozen
seams exactly as M2 (graded_eval): gated_store.extract_turn_family under the
family's frozen LME prompt (syn_res.LME_PROMPTS), store._map_tuple_key
(frozen v2 key mapper, mode OFF — the M1 stem wrapper is NOT installed;
asserted at run time), classify.classify_item + gated_store.ARGMAX_ROUTE
routing, gated_store._generate_predicates, the frozen checker (test3 fire
semantics, consume-on-fire), ZepStore writes; PLUS the 5th preregistered
entry-fix arm **em_cg_fixed** = the cautious (em) family + gated policy with
the M1 stem-wrapper mapper (store.make_tuple_key_mapper over
router_sweep.PLAN_FRAME_ATTRIBUTE_PATTERNS, mode 'plan_stem_wrapper')
installed as a LOCAL mapper in place of the frozen store._map_tuple_key for
THAT arm only. The wrapper is never installed globally (assert_mapper_mode_off
still guards the four frozen cells and is re-asserted after the arm); the em
extraction/first-pass mapping is cache-shared with em_cw/em_cg, so the only
fresh spend is the `-planstem` re-offer on DECLINED plan-framed em tuples plus
the downstream gate/checker on newly-mapped tuples. The four frozen cells are
byte-identical whether or not the 5th arm runs.

REPLAY (per the spec entry): the FULL conversation stream per arc, in order;
extraction/mapping/classification on user turns only (assistant turns are
context); a single user store per arc (entity=user → gated_store
DEFAULT_STORE); the arc key registered from results/v2.4/m4/m4_keys.jsonl.
Stream dates are the SESSION-level dates (conversation[si]["date"]),
matching the frozen LME B2 rendering contract of one session-date marker per
session; the arcs' per-turn timestamps stay in the raw arc record
(interpretive choice #2 below).

PER-ARC CASE RECORD (results/v2.4/m4/eval/cases/<arc_id>.json, resumable per
arc — the M2 case_path pattern): per cell — final store value for the arc
key; pre-resolution boundary store value (snapshot just before the
resolution turn's checker+gate; for open arcs the spec's boundary = end of
conversation, so scoring uses the final value there — the pre-resolution
snapshot is recorded for every arc anyway); the staging record at the hedge
turn (staged ids + frozen predicates); checker rows over the pending window
+ resolution turn (and every later turn while items stay pending — the
frozen gated replay behavior) with user-turn offsets relative to the
resolution turn; promote/invalidate fires with validity; the
commit/supersession event log for the arc key annotated with turn
coordinates; diagnostics (tuples extracted / mapped to the arc key per user
turn + the hedge-turn natres match). Everything needed to score all six
endpoint families is in the record — scoring never re-runs.

ENDPOINTS (score → m4_eval_results.json + m4_eval_report.md; NO go/no-go
bars; cluster = conversation (arc); ITT denominators = all arcs in the
outcome class; equivalence = rule5.EquivalenceScorer, exact-normalized fast
path + frozen judge fallback):
  (1) contamination — hedged value asserted as current in the FINAL store
      for abandoned/open arcs; premature-commit variant for resolve_true =
      value asserted at the PRE-RESOLUTION boundary. For resolve_true arcs a
      final-store match is CORRECT tracking, never contamination.
  (2) staging rate at the hedge turn (gated cells).
  (3) promotion on resolve_true — valid promote fire at t* (primary) /
      within [t*, t*+2] user turns (sensitivity, test3 semantics);
      end-state equivalence (all arms — the write-through cells' own
      resolution tracking); delay in user turns (first valid promote fire at
      offset >= 0, conditional on firing). ITT + conditional-on-staged.
  (4) abandoned handling — gated correct: staged item (from the hedge turn)
      invalidated or still pending at end AND the hedged value never
      committed; ungated failure mode: committed at the hedge.
      `no_false_memory` (never committed + final store lacks the value) is
      reported beside as the unstaged-inclusive variant.
  (5) open handling — no commit of the hedged value anywhere in the replay;
      gated correct hold = staged item still pending at end.
  (6) task probes — each arc's value-free probe question answered by
      DETERMINISTIC lookup of the arc key in the cell's final store
      (present + non-sentinel = the answer; else abstention-by-absence).
      Expected per class: resolve_true → the new value; abandoned/open →
      the established prior value if one exists in-arc, else
      abstention-by-absence (see interpretive choice #1).

PREREGISTERED DIRECTIONAL TABLE (rows (i)-(iv), report holds/doesn't — no
bars; the spec entry gives directions, the operationalizations below are
this module's, stated in full in the report):
  (i)   ez_cw contamination ≫ em_cw: pooled abandoned+open final-store
        contamination; holds iff point(ez_cw) > point(em_cw).
  (ii)  gated premature commits ≈ 0, leakage concentrated in
        confident/scheduled phrasings: premature commit per arc per gated
        cell = hedged value asserted as current at the pre-resolution
        boundary (open arcs: end of conversation, per the spec's boundary);
        holds iff the pooled gated rate <= 0.05 AND < the pooled ungated
        rate; concentration in hedge_strength `scheduled_but_future` (the
        M4 pool's only textually-unhedged strength — the M2 g4 analog; the
        pool has no g3 confident_plan analog) is reported descriptively.
  (iii) promotion high in ez_cg, em_cg entry (mapping) the bottleneck:
        holds iff ez_cg promote-fire-within-window (resolve_true, ITT) >
        em_cg AND em's hedge-turn arc-key mapping rate < ez's.
  (iv)  task delta gated>ungated under ez, ≈0 under em: delta_fam =
        probe-correct(fam_cg) − probe-correct(fam_cw) pooled over all
        scored arcs; holds iff delta_ez > 0 AND |delta_em| <= 0.05.

INTERPRETIVE CHOICES (spec ambiguities resolved here, disclosed):
  1. "prior value if established in-arc": every r4 arc ESTABLISHES a
     designed prior current value for the arc key (prior_establish at
     s0 t0; r4 contract), so the expected probe answer is prior_value for
     abandoned kept_prior and open arcs, alternative_value for abandoned
     alternative arcs, and the hedge target value for resolve_true
     (score_case reads case["established_prior_value"] =
     expected_prior_memory(arc)). The abstention-by-absence branch remains
     implemented for any substrate without an established prior (the r3
     arcs were scored under it) but no r4 arc uses it. Raw probe values
     ride on every per-arc row for any post-hoc re-read.
  2. Stream dates are session-level (see REPLAY above).
  3. Mapper candidate list = [the arc key] alone: the r4 conversations are
     self-contained (no benchmark hosts — carried from the r3 freeze), so
     the arc key is the substrate's ONLY registered key; the M2 convention
     (substrate's registered candidates + target key) degenerates to this,
     and the mapper's "none" option keeps doing the real work.
  4. Checker turn offsets are USER-turn offsets relative to the resolution
     turn (negative = pending window before it, 0 = t*), the spec's "delay
     in user turns" axis; for open arcs the still-pending mention turn
     anchors the offsets (promotion endpoints never read them).

BUDGET: stage "m4" ADVISORY forecast 12,000 (raised 3,000 → 12,000 by the
M4-EVAL prereg entry; run estimate 6-10k). Spend recorded per fresh call to
results/v2.4/budget.json via forecast_v22.SharedCallBudget; stage overruns
WARN and continue; the ONLY hard stop is the 120,000 total envelope.

CLI (repo root, pilot/.venv/bin/python experiments/v2_full/m4_eval.py):
  run    [--classes resolve_true,abandoned,open] [--families em,ez]
         [--ids M4-...] [--call-cap N] [--out DIR]
         five-arm full-stream replays (4 frozen cells + em_cg_fixed),
         resumable per arc
  score  [--cases DIR] [--out-dir DIR]
         endpoints + directional table -> m4_eval_results.json +
         m4_eval_report.md (judge calls budget-guarded, stage m4)
  smoke  [--call-cap N]   run the 2 preregistered smoke arcs (first
         resolve_true + first abandoned in arc_id order) through all five
         arms (default cap 250 fresh calls), then score them into
         results/v2.4/m4/eval/smoke/ — case records are REAL (the full run
         resumes past them)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401  (R1 layout: puts src/ + experiments/ on sys.path)

import classify
import forecast_v22
import gated_store
import graded_eval
import llm
import mapper_fix
import natres
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

SEED_M4 = 20260708  # the frozen M4 plan seed (arcs carry it per row)

V24_DIR = RESULTS_DIR / "v2.4"
BUDGET_PATH = V24_DIR / "budget.json"
M4_DIR = V24_DIR / "m4"
ARCS_PATH = M4_DIR / "arcs.jsonl"
KEYS_PATH = M4_DIR / "m4_keys.jsonl"
EVAL_DIR = M4_DIR / "eval"
CASES_DIR = EVAL_DIR / "cases"
RESULTS_PATH = EVAL_DIR / "m4_eval_results.json"
REPORT_PATH = EVAL_DIR / "m4_eval_report.md"
SMOKE_DIR = EVAL_DIR / "smoke"

STAGE = "m4"
#: ADVISORY stage forecast — raised 3,000 -> 12,000 by the M4-EVAL prereg
#: entry (2026-07-09); tracked and reported vs actuals, never a hard stop.
M4_EVAL_ADVISORY_FORECAST = 12_000

FAMILIES = ("em", "ez")
POLICIES = ("ungated", "gated")
#: The four FROZEN cells (mapper mode OFF; the directional table (i)-(iv) is
#: computed over these and stays byte-identical when the 5th arm is present).
CELLS = ("em_cw", "em_cg", "ez_cw", "ez_cg")
GATED_CELLS = ("em_cg", "ez_cg")
UNGATED_CELLS = ("em_cw", "ez_cw")
#: The preregistered 5th "entry-fix" arm (M4-EVAL-r4, 2026-07-09): cautious
#: (em) family + gated policy, but the tuple->key mapping uses the M1
#: stem-wrapper (store.make_tuple_key_mapper, mode 'plan_stem_wrapper') in
#: place of the frozen store._map_tuple_key for THIS arm only. The wrapper is a
#: LOCAL mapper (never installed globally — assert_mapper_mode_off still guards
#: the four frozen cells); it is torn down (re-asserted) after the arm.
FIXED_CELL = "em_cg_fixed"
#: Every arm that gets a per-cell score row + endpoint column.
SCORED_CELLS = CELLS + (FIXED_CELL,)
#: Gated arms (staging / promotion / gated-correct endpoints) including the fix.
GATED_CELLS_EXT = GATED_CELLS + (FIXED_CELL,)
#: The em/ez frozen families + the wrapper-mapped em ("em_fixed") whose
#: hedge-turn funnel feeds the entry-fix mapping comparison.
DIAG_FAMILIES = FAMILIES + ("em_fixed",)
CLASSES = ("resolve_true", "abandoned", "open")

SENSITIVITY_WINDOW = test3.SENSITIVITY_WINDOW  # = 2 (v1 §3.4 semantics)

#: Row (ii) concentration stratum: the pool's textually-unhedged strengths
#: (the M2 g3/g4 "confident/scheduled phrasings" analog; the r3 pool carries
#: only the scheduled one).
LEAKY_STRENGTHS = ("scheduled_but_future",)

#: Row (ii)/(iv) "≈ 0" operationalization (this module's, disclosed in the
#: docstring + report — the spec entry gives directions only).
APPROX_ZERO = 0.05


# ---------------------------------------------------------------------------
# Budget (advisory stage m4; hard limit = the 120k combined envelope only)
# ---------------------------------------------------------------------------

def m4_guard(path: Path = BUDGET_PATH,
             local_cap: int | None = None) -> forecast_v22.SharedCallBudget:
    """The M4-EVAL shared-envelope guard (the graded_eval.m2_guard pattern,
    stage m4): the stage allocation is ADVISORY — exceeding the m4 forecast
    prints a warning and CONTINUES; the only hard stop is the combined 120k
    envelope."""
    budget = mapper_fix.ensure_budget(path)
    spent = budget["spent"].get(STAGE, 0)
    forecast = budget["allocation"].get(STAGE, M4_EVAL_ADVISORY_FORECAST)
    if spent >= forecast:
        print(f"[m4 budget ADVISORY] stage spend {spent:,} >= forecast "
              f"{forecast:,} — continuing (stages are advisory per the "
              "2026-07-08 amendment; only the total envelope is hard)")
    remaining = forecast_v22.budget_remaining(budget)
    cap = remaining if local_cap is None else min(local_cap, remaining)
    return forecast_v22.SharedCallBudget(STAGE, path=path, local_cap=cap)


def advisory_status(path: Path = BUDGET_PATH) -> dict:
    budget = mapper_fix.ensure_budget(path)
    spent = budget["spent"].get(STAGE, 0)
    forecast = budget["allocation"].get(STAGE, M4_EVAL_ADVISORY_FORECAST)
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
# Frozen configuration asserts
# ---------------------------------------------------------------------------

def assert_mapper_mode_off() -> None:
    """The spec's "key-mapper frozen mode OFF" pin: this runner calls
    store._map_tuple_key directly (never a mapper override), and the M1 stem
    wrapper must not have been monkey-installed over it —
    store.make_tuple_key_mapper closures carry ``mode == 'plan_stem_wrapper'``;
    the frozen function carries no mode attribute."""
    mode = getattr(store._map_tuple_key, "mode", "off")
    assert mode == "off", (
        f"store._map_tuple_key carries mapping mode {mode!r} — the M1 stem "
        "wrapper is installed; M4-EVAL is preregistered with the frozen "
        "mapper, mode OFF (matching M2)")


def arcs_sha256(path: Path = ARCS_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Substrate loading (the frozen r3 arcs + FM-4 key registry)
# ---------------------------------------------------------------------------

def load_arcs(path: Path = ARCS_PATH) -> list[dict]:
    arcs = _read_jsonl(path)
    assert len(arcs) == 60, f"expected 60 M4 arcs, got {len(arcs)}"
    # r4 is the current owner-authorized substrate (coherence + established
    # prior value); r3 remains readable for regression on the retired file.
    assert all(a.get("render_round") == 4 for a in arcs), "non-r4 arc present"
    return sorted(arcs, key=lambda a: a["arc_id"])


def expected_prior_memory(arc: dict) -> str | None:
    """The correct-current-memory value for the abstention/prior probe branch
    (r4 schema, protocol/CHANGELOG.md 2026-07-09). resolve_true is scored
    against the hedged VALUE by score_case's own branch, so it returns None
    here; abandoned/open return the value a correct memory should hold."""
    rt = arc["resolution_type"]
    if rt == "resolve_true":
        return None
    if rt == "abandoned" and arc.get("abandoned_mode") == "alternative":
        return arc.get("alternative_value")
    return arc.get("prior_value")            # abandoned kept_prior / open


def load_keys(path: Path = KEYS_PATH) -> dict[str, dict]:
    return {k["arc_id"]: k for k in _read_jsonl(path)}


def arc_canonical_key(key_row: dict) -> store.CanonicalKey:
    return store.CanonicalKey(entity=key_row["entity"],
                              attribute=key_row["attribute"],
                              description=key_row["description"],
                              probe_question=key_row["probe_question"])


def arc_turns(arc: dict) -> list[dict]:
    """The full chronological stream: {session_idx, turn_idx, role, date,
    content} with SESSION-level dates (interpretive choice #2 — the frozen
    LME B2 rendering marks one session date per session)."""
    rows = []
    for sess in arc["conversation"]:
        for ti, t in enumerate(sess["turns"]):
            rows.append({"session_idx": sess["session_idx"], "turn_idx": ti,
                         "role": t["role"], "date": sess["date"],
                         "content": t["text"]})
    return rows


def turn_pos(turns: list[dict], placement: dict) -> int:
    for i, t in enumerate(turns):
        if (t["session_idx"] == placement["session_idx"]
                and t["turn_idx"] == placement["turn_idx"]):
            return i
    raise ValueError(f"turn {placement} not found")


def arc_geometry(arc: dict) -> dict:
    """Positions + user-turn indices for the hedge and resolution beats.
    Asserts the beats sit on user turns and carry their beat sentences."""
    turns = arc_turns(arc)
    user_positions = [i for i, t in enumerate(turns) if t["role"] == "user"]
    user_index = {pos: k for k, pos in enumerate(user_positions)}
    hedge_pos = turn_pos(turns, arc["hedge"])
    res_pos = turn_pos(turns, arc["resolution"])
    assert turns[hedge_pos]["role"] == "user", f"{arc['arc_id']}: hedge not a user turn"
    assert turns[res_pos]["role"] == "user", f"{arc['arc_id']}: resolution not a user turn"
    assert arc["hedge_text"] in turns[hedge_pos]["content"], (
        f"{arc['arc_id']}: hedge_text not in hedge turn")
    assert arc["resolution_text"] in turns[res_pos]["content"], (
        f"{arc['arc_id']}: resolution_text not in resolution turn")
    return {"turns": turns, "user_positions": user_positions,
            "user_index": user_index, "hedge_pos": hedge_pos,
            "res_pos": res_pos}


# ---------------------------------------------------------------------------
# run — the four-cell full-stream replays (M2 machinery, M4 stimuli)
# ---------------------------------------------------------------------------

async def _extract_and_map(family: str, turns: list[dict], pos: int,
                           candidates: list[store.CanonicalKey],
                           arc_id: str, mapper=None) -> dict:
    """(tuples, mapped) for one user turn — shared by BOTH policies of a
    family (in-memory; byte-identical llm calls replay from the cache). The
    frozen LME pathway of graded_eval._extract_and_map, M4 trace tags.

    ``mapper`` (M4-EVAL-r4 5th-arm seam, default None = the frozen v2 mapper
    ``store._map_tuple_key``): an optional async override with the same
    ``(tup, candidates, *, trace_tag)`` convention. When None the mapping call
    is byte-identical to em_cw/em_cg (cache-shared). The 5th arm passes the M1
    stem-wrapper (store.make_tuple_key_mapper) here: its FIRST pass renders the
    byte-identical prompt (cache hit) and only the `-planstem` re-offer on a
    DECLINED, plan-framed tuple is fresh. The extraction call is identical
    either way (the em extraction is shared with em_cw/em_cg), so no fresh
    extraction spend. Results pass through verbatim — the wrapper's 4-tuple
    (with meta) and the frozen 3-tuple are both accepted downstream by
    gated_store._unpack_map_result."""
    si, ti = turns[pos]["session_idx"], turns[pos]["turn_idx"]
    tag = f"m4-extract-{family}-{arc_id}-s{si}t{ti}"
    prior = [{"role": t["role"], "content": t["content"], "date": t["date"]}
             for t in turns[:pos]]
    tuples = await gated_store.extract_turn_family(
        prior, turns[pos]["content"], tag,
        system_prompt=(PROMPTS_DIR / syn_res.LME_PROMPTS[family]).read_text(),
        current_date=turns[pos]["date"])
    map_call = store._map_tuple_key if mapper is None else mapper
    mapped = list(await asyncio.gather(*[
        map_call(tup, candidates,
                 trace_tag=f"m4-keymap-{family}-{arc_id}-s{si}t{ti}-{j}")
        for j, tup in enumerate(tuples)]))
    return {"pos": pos, "session_idx": si, "turn_idx": ti,
            "date": turns[pos]["date"], "turn_text": turns[pos]["content"],
            "tuples": tuples, "mapped": mapped}


async def _gated_checker(sim: _CellSim, rec: dict, family: str, arc_id: str,
                         checker_system: str, turn_offset: int) -> None:
    """Checker over pending items on one turn (graded_eval._gated_checker
    with M4 trace tags + user-turn offsets relative to the resolution turn)."""
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
            trace_tag=f"m4-check-{family}-{arc_id}-{store_id}-"
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
    """Phase-3 gate loop body for one turn (graded_eval._gated_tuples with M4
    ids/tags; frozen argmax routing verbatim). Staged rows additionally carry
    their staging-turn coordinates (the M4 per-arc record needs them)."""
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
            "item_id": f"m4-{family}-{arc_id}-s{rec['session_idx']}"
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
            staged_id = f"stg-m4-{family}-{arc_id}-s{rec['session_idx']}" \
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


def _hedge_turn_diagnostics(rec: dict, hedge_text: str,
                            target: store.CanonicalKey) -> dict:
    """Extraction-hit + mapping diagnostics at the hedge turn (the M1
    mapping-loss axis — graded_eval._target_diagnostics semantics)."""
    tuples = [t.__dict__ if hasattr(t, "__dict__") else dict(t)
              for t in rec["tuples"]]
    matched, method = natres.match_staged_tuple(
        tuples, hedge_text,
        {"entity": target.entity, "attribute": target.attribute})
    target_mapped = False
    matched_cand = None
    if matched is not None:
        idx = tuples.index(matched)
        cand, _raw, _rat, _meta = gated_store._unpack_map_result(rec["mapped"][idx])
        matched_cand = cand.name if cand is not None else None
        target_mapped = cand is not None and cand.key == target.key
    any_target_map = any(
        (gated_store._unpack_map_result(m)[0] is not None
         and gated_store._unpack_map_result(m)[0].key == target.key)
        for m in rec["mapped"])
    return {"n_tuples": len(tuples), "target_extracted": matched is not None,
            "match_method": method, "matched_tuple_mapped_to": matched_cand,
            "target_mapped_via_matched_tuple": target_mapped,
            "any_tuple_mapped_to_target": any_target_map}


def _turn_diag_row(rec: dict, user_idx: int,
                   target: store.CanonicalKey) -> dict:
    """Per-user-turn tuples-extracted/-mapped diagnostics for the arc key."""
    target_values = []
    n_mapped_any = 0
    for tup, mres in zip(rec["tuples"], rec["mapped"]):
        cand, _raw, _rat, _meta = gated_store._unpack_map_result(mres)
        if cand is None:
            continue
        n_mapped_any += 1
        if cand.key == target.key:
            target_values.append(tup.value)
    return {"session_idx": rec["session_idx"], "turn_idx": rec["turn_idx"],
            "user_idx": user_idx, "n_tuples": len(rec["tuples"]),
            "n_mapped_any": n_mapped_any,
            "n_mapped_target": len(target_values),
            "target_values": target_values}


def _jsonable_event(event: dict) -> dict:
    out = dict(event)
    out["key"] = list(out["key"])
    return out


def case_path(arc_id: str, out_root: Path = CASES_DIR) -> Path:
    return out_root / f"{arc_id}.json"


async def _run_policy_cell(policy: str, recs: list[dict], family: str,
                           arc_id: str, target: store.CanonicalKey,
                           turns: list[dict], user_index: dict, res_pos: int,
                           hedge_pos: int, gen_system: str,
                           checker_system: str) -> dict:
    """Run ONE policy over a family's per-turn recs; returns the per-cell
    record (identical fields for every cell — em_cw/em_cg/ez_cw/ez_cg AND the
    5th arm em_cg_fixed). The body is mapper-agnostic: the recs already carry
    the mapping results (frozen or wrapper), so the four frozen cells and the
    fixed arm share this exact code path. A fresh _CellSim per call means the
    fixed arm's store/staging state is entirely independent of the frozen
    cells (regression: em_cg is byte-identical with or without the 5th arm)."""
    sim = _CellSim(policy, "lme")
    store_id = sim.route(target.key)
    pre_resolution_value = None
    target_events: list[dict] = []
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
                **_jsonable_event(event),
                "session_idx": rec["session_idx"],
                "turn_idx": rec["turn_idx"],
                "user_idx": user_index[rec["pos"]],
                "user_offset_from_resolution": offset,
                "at_hedge_turn": rec["pos"] == hedge_pos,
            })
    final = sim.lookup(target.key)
    staged_target = []
    for sid, srow in sim.staged_rows.items():
        if (srow["entity"], srow["attribute"]) != target.key:
            continue
        s_pos = turn_pos(turns, {"session_idx": srow["session_idx"],
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
        "gate_rows_target": [g for g in sim.gate_rows
                             if g["key"] == target.name],
        "diag": dict(sim.diag),
    }


def _fixed_mapping_meta(recs: list[dict], user_index: dict,
                        target: store.CanonicalKey) -> list[dict]:
    """Per-user-turn wrapper mapping meta for em_cg_fixed — the entry funnel
    can be recomputed / the stem re-offer audited from the case record alone
    (PIN-1: original_attribute, derived_stem, pattern_matched, mapped_key)."""
    out = []
    for rec in recs:
        tuples_meta = []
        for tup, mres in zip(rec["tuples"], rec["mapped"]):
            cand, _raw, _rat, meta = gated_store._unpack_map_result(mres)
            tuples_meta.append({
                "attribute": tup.attribute, "value": tup.value,
                "mapped_key": cand.name if cand is not None else None,
                "mapped_to_target": (cand is not None
                                     and cand.key == target.key),
                "stem_mapped": meta.get("stem_mapped"),
                "derived_stem": meta.get("derived_stem"),
                "pattern_matched": meta.get("pattern_matched"),
                "original_attribute": meta.get("original_attribute"),
            })
        out.append({"session_idx": rec["session_idx"],
                    "turn_idx": rec["turn_idx"],
                    "user_idx": user_index[rec["pos"]], "tuples": tuples_meta})
    return out


async def run_arc(arc: dict, key_row: dict, families: tuple[str, ...],
                  arcs_digest: str, include_fixed: bool = True) -> dict:
    """One arc through all five arms (4 frozen cells + em_cg_fixed) over the
    FULL conversation stream."""
    arc_id = arc["arc_id"]
    geom = arc_geometry(arc)
    turns, user_positions = geom["turns"], geom["user_positions"]
    user_index, hedge_pos, res_pos = (geom["user_index"], geom["hedge_pos"],
                                      geom["res_pos"])
    target = arc_canonical_key(key_row)
    candidates = [target]  # interpretive choice #3 (module docstring)
    gen_system = gated_store.GENERATOR_PROMPT_PATH.read_text()
    checker_system = gated_store.CHECKER_PROMPT_PATH.read_text()

    out = {"arc_id": arc_id, "question_id": arc["question_id"],
           "resolution_type": arc["resolution_type"],
           "gold_resolution_status": arc["gold_resolution_status"],
           "hedge_strength": arc["hedge_strength"],
           "cluster": arc_id,  # cluster = conversation (spec)
           "target_key": target.name,
           "hedged_value": arc["value"],
           "value_candidates": arc.get("value_candidates"),
           "probe_question": key_row["probe_question"],
           # r4 arcs establish a designed prior value in-conversation (session 0
           # user turn 0); the abstention/prior probe branch reads it. abandoned
           # correct memory = alternative_value (mode 'alternative') or
           # prior_value (mode 'kept_prior'); open = prior_value; resolve_true
           # is scored against the hedged value by score_case's own branch.
           "established_prior_value": expected_prior_memory(arc),
           "prior_value": arc.get("prior_value"),
           "abandoned_mode": arc.get("abandoned_mode"),
           "alternative_value": arc.get("alternative_value"),
           "arc_status": arc.get("status"), "arc_usable": arc.get("usable"),
           "pending_window_user_turns": arc.get("pending_window_user_turns"),
           "realized_pending_window_user_turns":
               arc.get("realized_pending_window_user_turns"),
           "arcs_sha256": arcs_digest,
           "seed": SEED_M4,
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
           "boundary_note": ("pre_resolution_value = store state just before "
                             "the resolution turn's checker+gate; the spec's "
                             "pre-resolution boundary for open arcs = end of "
                             "conversation (scoring uses final_value there)"),
           "cells": {}, "diagnostics": {}}

    for family in families:
        recs = [await _extract_and_map(family, turns, pos, candidates, arc_id)
                for pos in user_positions]
        hedge_rec = next(r for r in recs if r["pos"] == hedge_pos)
        out["diagnostics"][family] = {
            "hedge_turn": _hedge_turn_diagnostics(hedge_rec,
                                                  arc["hedge_text"], target),
            "per_turn": [_turn_diag_row(rec, user_index[rec["pos"]], target)
                         for rec in recs],
        }
        for policy in POLICIES:
            cell = f"{family}_{'cg' if policy == 'gated' else 'cw'}"
            out["cells"][cell] = await _run_policy_cell(
                policy, recs, family, arc_id, target, turns, user_index,
                res_pos, hedge_pos, gen_system, checker_system)

    # 5th preregistered arm (M4-EVAL-r4, 2026-07-09): em_cg_fixed — the
    # cautious (em) family + gated policy, but the tuple->key mapping uses the
    # M1 stem-wrapper (mode 'plan_stem_wrapper') in place of the frozen
    # store._map_tuple_key for THIS arm only. The wrapper is a LOCAL mapper
    # passed into _extract_and_map (never installed globally — the frozen four
    # cells above already ran against store._map_tuple_key, and
    # assert_mapper_mode_off holds throughout); it is re-asserted (torn down)
    # after the arm. Extraction is cache-shared with em_cw/em_cg; only the
    # `-planstem` re-offer on DECLINED plan-framed em tuples + the downstream
    # gate/checker on newly-mapped tuples is fresh.
    if include_fixed and "em" in families:
        assert_mapper_mode_off()  # global frozen mapper untouched entering arm
        fixed_mapper = store.make_tuple_key_mapper(PLAN_FRAME_ATTRIBUTE_PATTERNS)
        recs_fixed = [
            await _extract_and_map("em", turns, pos, candidates, arc_id,
                                   mapper=fixed_mapper)
            for pos in user_positions]
        hedge_rec_fixed = next(r for r in recs_fixed if r["pos"] == hedge_pos)
        out["diagnostics"]["em_fixed"] = {
            "hedge_turn": _hedge_turn_diagnostics(hedge_rec_fixed,
                                                  arc["hedge_text"], target),
            "per_turn": [_turn_diag_row(rec, user_index[rec["pos"]], target)
                         for rec in recs_fixed],
        }
        fixed_cell = await _run_policy_cell(
            "gated", recs_fixed, "em", arc_id, target, turns, user_index,
            res_pos, hedge_pos, gen_system, checker_system)
        fixed_cell["mapping_meta"] = _fixed_mapping_meta(
            recs_fixed, user_index, target)
        out["cells"][FIXED_CELL] = fixed_cell
        assert_mapper_mode_off()  # teardown: local mapper, global still frozen
    return out


async def run(classes: tuple[str, ...], families: tuple[str, ...],
              ids: list[str] | None, out_root: Path, guard) -> dict:
    assert_mapper_mode_off()
    arcs = load_arcs()
    keys = load_keys()
    digest = arcs_sha256()
    summary = {"n_arcs": 0, "skipped": 0, "failed": []}
    guard.install()
    try:
        for arc in arcs:
            if arc["resolution_type"] not in classes:
                continue
            if ids is not None and arc["arc_id"] not in ids:
                continue
            path = case_path(arc["arc_id"], out_root)
            if path.exists():
                summary["skipped"] += 1
                continue
            record = await run_arc(arc, keys[arc["arc_id"]], families, digest)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, ensure_ascii=False, indent=1))
            summary["n_arcs"] += 1
    finally:
        guard.uninstall()
    summary["fresh_calls"] = guard.fresh
    summary["advisory"] = advisory_status()
    return summary


# ---------------------------------------------------------------------------
# score — the six endpoint families + the directional table (no bars)
# ---------------------------------------------------------------------------

def _answered(value) -> bool:
    """Deterministic lookup semantics for the task probe: a present,
    non-sentinel current value answers; anything else is
    abstention-by-absence."""
    return value is not None and not zepstore.is_sentinel(str(value))


async def score_case(case: dict, scorer) -> list[dict]:
    """Per-arc score rows, one per cell (arc_id rides on every row so the
    owner's accept/reject sheet filters without rerun). Pure function of the
    case record + the equivalence scorer — never re-runs anything."""
    attr = case["target_key"].partition("/")[2]
    hv = case["hedged_value"]
    cls = case["resolution_type"]
    prior = case.get("established_prior_value")
    rows = []
    # score every arm present (the four frozen cells + the 5th em_cg_fixed arm
    # when the run produced it; tolerant of legacy 4-cell records + synthetic
    # test cases so their row count is unchanged).
    for cell in [c for c in SCORED_CELLS if c in case["cells"]]:
        c = case["cells"][cell]
        final_eq, _m = await scorer.equivalent(attr, c["final_value"], hv)
        pre_eq, _m = await scorer.equivalent(
            attr, c["pre_resolution_value"], hv)
        # the spec's pre-resolution boundary: open arcs = end of conversation
        boundary_value = (c["final_value"] if cls == "open"
                          else c["pre_resolution_value"])
        boundary_eq = final_eq if cls == "open" else pre_eq
        committed_ever = False
        committed_at_hedge = False
        for ev in c["target_events"]:
            if ev["action"] not in ("commit", "supersede"):
                continue  # refresh never introduces a value
            eq, _m = await scorer.equivalent(attr, ev["value"], hv)
            if eq:
                committed_ever = True
                if ev["at_hedge_turn"]:
                    committed_at_hedge = True
        hedge_staged = [s for s in c["staged_target"] if s["at_hedge_turn"]]
        statuses = sorted({s["status"] for s in hedge_staged})
        probe_value = c["final_value"] if _answered(c["final_value"]) else None
        if cls == "resolve_true":
            probe_correct = bool(final_eq)
            probe_expected = hv
        elif prior is not None:
            eq, _m = await scorer.equivalent(attr, probe_value, prior)
            probe_correct = bool(eq)
            probe_expected = prior
        else:
            probe_correct = probe_value is None  # abstention-by-absence
            probe_expected = "(abstention-by-absence)"
        row = {
            "arc_id": case["arc_id"],
            "resolution_type": cls,
            "hedge_strength": case["hedge_strength"],
            "cell": cell,
            "cluster": case["cluster"],
            "arc_status": case.get("arc_status"),
            "arc_usable": case.get("arc_usable"),
            "hedged_value": hv,
            "final_value": c["final_value"],
            "pre_resolution_value": c["pre_resolution_value"],
            "boundary_value": boundary_value,
            "final_eq_hedged": bool(final_eq),
            "pre_resolution_eq_hedged": bool(pre_eq),
            "boundary_eq_hedged": bool(boundary_eq),
            # endpoint (1): contamination for abandoned/open (final store);
            # premature-commit variant for resolve_true (boundary); a
            # resolve_true final match is CORRECT tracking, never counted
            "contamination": bool(final_eq) if cls in ("abandoned", "open") else None,
            "premature_commit": bool(boundary_eq),
            "staged_at_hedge": bool(c["target_staged_at_hedge"]),
            "hedge_staged_statuses": statuses,
            "committed_ever": committed_ever,
            "committed_at_hedge": committed_at_hedge,
            # endpoint (3) fields (resolve_true; gated fire semantics)
            "promote_at_tstar": bool(c.get("promote_fire_at_tstar")),
            "promote_within_window": bool(c.get("promote_fire_within_window")),
            "promotion_delay_user_turns": (
                c.get("first_valid_promote_offset")
                if (c.get("first_valid_promote_offset") is not None
                    and c["first_valid_promote_offset"] >= 0) else None),
            "end_state_eq_new_value": bool(final_eq) if cls == "resolve_true" else None,
            # endpoint (4): the spec's gated correct-handling conjunction
            "abandoned_correct": (
                bool(hedge_staged)
                and all(s in ("invalidated", "pending") for s in statuses)
                and not committed_ever) if cls == "abandoned" else None,
            "no_false_memory": ((not committed_ever and not final_eq)
                                if cls in ("abandoned", "open") else None),
            # endpoint (5)
            "no_commit_anywhere": (not committed_ever) if cls == "open" else None,
            "open_correct_hold": (
                bool(hedge_staged) and statuses == ["pending"]
                and not committed_ever) if cls == "open" else None,
            # endpoint (6)
            "probe_value": probe_value,
            "probe_expected": probe_expected,
            "probe_answered": probe_value is not None,
            "probe_correct": bool(probe_correct),
        }
        rows.append(row)
    return rows


def _point(rows: list[dict], flag: str) -> dict:
    """Cluster-bootstrap rate over per-arc rows (cluster = arc)."""
    sel = [r for r in rows if r.get(flag) is not None]
    values = [int(bool(r[flag])) for r in sel]
    clusters = [r["cluster"] for r in sel]
    if not values:
        return {"x": 0, "n": 0, "point": None, "ci": None,
                "ci_kind": "cluster_bootstrap"}
    boot = stats.cluster_bootstrap_rate(values, clusters, seed=SEED_M4)
    return {"x": sum(values), "n": len(values), "point": boot["point"],
            "ci": [boot["lo"], boot["hi"]], "ci_kind": "cluster_bootstrap",
            "n_clusters": boot["n_clusters"]}


def _cell_rows(rows: list[dict], cell: str,
               cls: str | None = None) -> list[dict]:
    return [r for r in rows if r["cell"] == cell
            and (cls is None or r["resolution_type"] == cls)]


def directional_table(rows: list[dict], diagnostics: dict) -> list[dict]:
    """The preregistered rows (i)-(iv); operationalizations per the module
    docstring (disclosed — the spec entry gives directions, not numbers)."""
    def pooled(cells, cls_set, flag):
        sel = [r for r in rows if r["cell"] in cells
               and r["resolution_type"] in cls_set and r.get(flag) is not None]
        return _point(sel, flag)

    table = []
    # (i) ez_cw contamination >> em_cw (abandoned+open pooled, final store)
    em = pooled(("em_cw",), ("abandoned", "open"), "contamination")
    ez = pooled(("ez_cw",), ("abandoned", "open"), "contamination")
    table.append({
        "row": "(i)", "expectation": "ez_cw contamination >> em_cw "
        "(v2/M2 interaction direction; abandoned+open pooled, final store)",
        "observed": {"em_cw": em, "ez_cw": ez},
        "holds": (em["point"] is not None and ez["point"] is not None
                  and ez["point"] > em["point"]),
        "operationalization": "holds iff point(ez_cw) > point(em_cw)",
    })
    # (ii) gated premature commits ~ 0; leakage concentrated in
    # confident/scheduled phrasings
    gated = pooled(GATED_CELLS, CLASSES, "premature_commit")
    ungated = pooled(UNGATED_CELLS, CLASSES, "premature_commit")
    leaks = [r for r in rows if r["cell"] in GATED_CELLS
             and r["premature_commit"]]
    leak_strengths = sorted({r["hedge_strength"] for r in leaks})
    concentrated = bool(leaks) and all(
        r["hedge_strength"] in LEAKY_STRENGTHS for r in leaks)
    table.append({
        "row": "(ii)", "expectation": "gated premature commits ~ 0; possible "
        "leakage concentrated in confident/scheduled phrasings (M2 g3/g4)",
        "observed": {"gated_pooled": gated, "ungated_pooled": ungated,
                     "leak_arcs": [{"arc_id": r["arc_id"], "cell": r["cell"],
                                    "hedge_strength": r["hedge_strength"]}
                                   for r in leaks],
                     "leak_hedge_strengths": leak_strengths,
                     "leakage_concentrated_in_scheduled": concentrated},
        "holds": (gated["point"] is not None
                  and gated["point"] <= APPROX_ZERO
                  and (ungated["point"] is None
                       or gated["point"] < ungated["point"])),
        "operationalization": (
            f"holds iff pooled gated premature-commit rate <= {APPROX_ZERO} "
            "AND < the pooled ungated rate; concentration in "
            f"{LEAKY_STRENGTHS} reported descriptively (the r4 pool has no "
            "confident_plan strength)"),
    })
    # (iii) promotion high in ez_cg; em entry (mapping) the bottleneck
    em_p = pooled(("em_cg",), ("resolve_true",), "promote_within_window")
    ez_p = pooled(("ez_cg",), ("resolve_true",), "promote_within_window")
    em_map = diagnostics.get("hedge_mapped_rate", {}).get("em", {}).get("resolve_true")
    ez_map = diagnostics.get("hedge_mapped_rate", {}).get("ez", {}).get("resolve_true")
    holds_iii = (em_p["point"] is not None and ez_p["point"] is not None
                 and ez_p["point"] > em_p["point"]
                 and em_map is not None and ez_map is not None
                 and em_map["point"] is not None and ez_map["point"] is not None
                 and em_map["point"] < ez_map["point"])
    table.append({
        "row": "(iii)", "expectation": "promotion high in ez_cg on "
        "plain-completion resolutions; em_cg entry (mapping) remains the "
        "bottleneck direction (M1/M2 funnel)",
        "observed": {"em_cg_promote_within_window": em_p,
                     "ez_cg_promote_within_window": ez_p,
                     "em_hedge_mapped_resolve_true": em_map,
                     "ez_hedge_mapped_resolve_true": ez_map},
        "holds": holds_iii,
        "operationalization": ("holds iff ez_cg window-promotion point > "
                               "em_cg AND em hedge-turn arc-key mapping rate "
                               "< ez (resolve_true arcs)"),
    })
    # (iv) task delta gated>ungated under ez, ~0 under em
    deltas = {}
    for fam in FAMILIES:
        cg = pooled((f"{fam}_cg",), CLASSES, "probe_correct")
        cw = pooled((f"{fam}_cw",), CLASSES, "probe_correct")
        deltas[fam] = {"gated": cg, "ungated": cw,
                       "delta": (None if cg["point"] is None
                                 or cw["point"] is None
                                 else cg["point"] - cw["point"])}
    holds_iv = (deltas["ez"]["delta"] is not None
                and deltas["em"]["delta"] is not None
                and deltas["ez"]["delta"] > 0
                and abs(deltas["em"]["delta"]) <= APPROX_ZERO)
    table.append({
        "row": "(iv)", "expectation": "task delta gated>ungated under ez, "
        "~0 under em (M3 direction)",
        "observed": deltas,
        "holds": holds_iv,
        "operationalization": (f"holds iff delta_ez > 0 AND |delta_em| <= "
                               f"{APPROX_ZERO} (pooled over all scored arcs)"),
    })
    return table


def _paired_delta(rows: list[dict], cell_a: str, cell_b: str,
                  cls_set: tuple[str, ...], flag: str) -> dict:
    """Cluster-bootstrap CI on the per-arc paired delta flag(cell_a) −
    flag(cell_b), paired by arc (cluster = arc). Used for the entry-fix arm's
    task-probe delta (em_cg_fixed − em_cg)."""
    a = {r["arc_id"]: r for r in rows if r["cell"] == cell_a
         and r["resolution_type"] in cls_set and r.get(flag) is not None}
    b = {r["arc_id"]: r for r in rows if r["cell"] == cell_b
         and r["resolution_type"] in cls_set and r.get(flag) is not None}
    ids = sorted(set(a) & set(b))
    d = [int(bool(a[i][flag])) - int(bool(b[i][flag])) for i in ids]
    if not d:
        return {"n": 0, "delta": None, "ci": None,
                "ci_kind": "cluster_bootstrap"}
    boot = stats.cluster_bootstrap_mean(d, ids, seed=SEED_M4, bounds=(-1, 1))
    return {"n": len(d), "a_x": sum(int(bool(a[i][flag])) for i in ids),
            "b_x": sum(int(bool(b[i][flag])) for i in ids),
            "delta": boot["point"], "ci": [boot["lo"], boot["hi"]],
            "ci_kind": "cluster_bootstrap"}


def entry_fix_comparison(rows: list[dict], diagnostics: dict) -> dict:
    """The preregistered 5th-arm comparison: em_cg_fixed vs em_cg (M4-EVAL-r4,
    2026-07-09). Cluster-bootstrap CIs (cluster = arc) on: hedge-turn arc-key
    mapping rate, staging rate, promotion (resolve_true within-window),
    task-probe delta (pooled + per class), and contamination cost
    (abandoned+open final store + the pre-resolution supplementary).

    Directional expectation (spec): em_cg_fixed mapping >> em_cg mapping (the
    entry funnel recovered); staging & promotion up; contamination stays
    gated-low (the gate still holds what now enters)."""
    def pooled(cell, cls_set, flag):
        sel = [r for r in rows if r["cell"] == cell
               and r["resolution_type"] in cls_set and r.get(flag) is not None]
        return _point(sel, flag)

    def per_class(cell, flag):
        return {cls: _point(_cell_rows(rows, cell, cls), flag)
                for cls in CLASSES}

    # primary funnel metric = the matched hedge tuple's own mapping (the loss
    # the stem-wrapper mechanically recovers); any_tuple reported alongside.
    htm = diagnostics.get("hedge_tuple_mapped_rate", {})
    hm = diagnostics.get("hedge_mapped_rate", {})
    mapping_rate = {cls: {"em_cg": htm.get("em", {}).get(cls),
                          "em_cg_fixed": htm.get("em_fixed", {}).get(cls)}
                    for cls in CLASSES + ("pooled",)}
    mapping_rate_any_tuple = {cls: {"em_cg": hm.get("em", {}).get(cls),
                                    "em_cg_fixed": hm.get("em_fixed", {}).get(cls)}
                              for cls in CLASSES + ("pooled",)}

    staging_rate = {"em_cg": per_class("em_cg", "staged_at_hedge"),
                    "em_cg_fixed": per_class("em_cg_fixed", "staged_at_hedge"),
                    "pooled": {"em_cg": pooled("em_cg", CLASSES, "staged_at_hedge"),
                               "em_cg_fixed": pooled("em_cg_fixed", CLASSES,
                                                     "staged_at_hedge")}}

    promotion = {"em_cg": _point(_cell_rows(rows, "em_cg", "resolve_true"),
                                 "promote_within_window"),
                 "em_cg_fixed": _point(
                     _cell_rows(rows, "em_cg_fixed", "resolve_true"),
                     "promote_within_window")}

    task_probe_delta = {
        cls: {"em_cg": _point(_cell_rows(rows, "em_cg", cls), "probe_correct"),
              "em_cg_fixed": _point(_cell_rows(rows, "em_cg_fixed", cls),
                                    "probe_correct"),
              "delta": _paired_delta(rows, "em_cg_fixed", "em_cg", (cls,),
                                     "probe_correct")}
        for cls in CLASSES}
    task_probe_delta["pooled"] = {
        "em_cg": pooled("em_cg", CLASSES, "probe_correct"),
        "em_cg_fixed": pooled("em_cg_fixed", CLASSES, "probe_correct"),
        "delta": _paired_delta(rows, "em_cg_fixed", "em_cg", CLASSES,
                               "probe_correct")}

    contamination_cost = {
        "final_store_abandoned_open": {
            "em_cg": pooled("em_cg", ("abandoned", "open"), "contamination"),
            "em_cg_fixed": pooled("em_cg_fixed", ("abandoned", "open"),
                                  "contamination")},
        "pre_resolution_supplementary_abandoned_open": {
            "em_cg": pooled("em_cg", ("abandoned", "open"),
                            "pre_resolution_eq_hedged"),
            "em_cg_fixed": pooled("em_cg_fixed", ("abandoned", "open"),
                                  "pre_resolution_eq_hedged")}}

    def pt(d):
        return d.get("point") if d else None
    map_base = pt(mapping_rate["pooled"]["em_cg"])
    map_fix = pt(mapping_rate["pooled"]["em_cg_fixed"])
    stg_base = pt(staging_rate["pooled"]["em_cg"])
    stg_fix = pt(staging_rate["pooled"]["em_cg_fixed"])
    pro_base = pt(promotion["em_cg"])
    pro_fix = pt(promotion["em_cg_fixed"])
    cont_base = pt(contamination_cost["final_store_abandoned_open"]["em_cg"])
    cont_fix = pt(contamination_cost["final_store_abandoned_open"]["em_cg_fixed"])
    sub = {
        "mapping_fixed_gt_em_cg": (map_base is not None and map_fix is not None
                                   and map_fix > map_base),
        "staging_up": (stg_base is not None and stg_fix is not None
                       and stg_fix >= stg_base),
        "promotion_up": (pro_base is not None and pro_fix is not None
                         and pro_fix >= pro_base),
        "contamination_gated_low": (
            cont_fix is not None and cont_fix <= APPROX_ZERO
            and (cont_base is None or cont_fix <= cont_base + APPROX_ZERO)),
    }
    return {
        "expectation": ("em_cg_fixed mapping >> em_cg mapping (entry funnel "
                        "recovered); staging & promotion up; contamination "
                        "stays gated-low (the gate still holds what now "
                        "enters)"),
        "mapping_rate": mapping_rate,
        "mapping_rate_any_tuple": mapping_rate_any_tuple,
        "staging_rate": staging_rate,
        "promotion_within_window_resolve_true": promotion,
        "task_probe_delta": task_probe_delta,
        "contamination_cost": contamination_cost,
        "sub_verdicts": sub,
        "holds": all(sub.values()),
        "operationalization": (
            f"holds iff pooled hedge-TUPLE mapping(em_cg_fixed) > "
            f"mapping(em_cg) [the matched hedge tuple's own mapping — the "
            f"funnel the stem-wrapper targets; any-tuple rate reported "
            f"alongside] AND pooled staging(fixed) >= staging(em_cg) AND "
            f"pooled resolve_true window-promotion(fixed) >= (em_cg) AND "
            f"pooled abandoned+open final-store contamination(fixed) <= "
            f"{APPROX_ZERO} and not worse than em_cg by > {APPROX_ZERO}"),
    }


def _load_cases(out_root: Path = CASES_DIR) -> list[dict]:
    if not out_root.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(out_root.glob("*.json"))]


async def compute_scores(out_root: Path = CASES_DIR) -> dict:
    cases = _load_cases(out_root)
    assert cases, f"no case records under {out_root} — run `run` first"
    digests = {c.get("arcs_sha256") for c in cases}
    assert len(digests) == 1, (
        f"case records span multiple arcs.jsonl digests: {digests}")
    scorer = rule5.EquivalenceScorer(cap=4000)

    rows: list[dict] = []
    for case in cases:
        rows.extend(await score_case(case, scorer))

    endpoints: dict = {}
    # (1) contamination / premature commit (all arms incl. the 5th em_cg_fixed)
    endpoints["contamination_final_store"] = {
        cls: {cell: _point(_cell_rows(rows, cell, cls), "contamination")
              for cell in SCORED_CELLS}
        for cls in ("abandoned", "open")}
    endpoints["premature_commit_resolve_true"] = {
        cell: _point(_cell_rows(rows, cell, "resolve_true"), "premature_commit")
        for cell in SCORED_CELLS}
    endpoints["premature_commit_all_classes"] = {
        cell: _point(_cell_rows(rows, cell), "premature_commit")
        for cell in SCORED_CELLS}
    # SUPPLEMENTARY (2026-07-09 changelog addendum, logged before any
    # full-run scoring; no rule changed): abandoned/open arcs additionally
    # report PRE-RESOLUTION-BOUNDARY contamination — an in-arc corrective
    # resolution can mask mid-arc exposure (the smoke abandoned arc: ez_cw
    # pre-resolution = hedged value, end state = superseded). This is the
    # recorded pre_resolution snapshot itself, NOT the spec's open-arc
    # end-of-conversation boundary substitution.
    endpoints["pre_resolution_boundary_contamination_supplementary"] = {
        cls: {cell: _point(_cell_rows(rows, cell, cls),
                           "pre_resolution_eq_hedged") for cell in SCORED_CELLS}
        for cls in ("abandoned", "open")}
    # (2) staging at the hedge turn (gated arms incl. the fix)
    endpoints["staging_at_hedge"] = {
        cls: {cell: _point(_cell_rows(rows, cell, cls), "staged_at_hedge")
              for cell in GATED_CELLS_EXT} for cls in CLASSES}
    endpoints["staging_at_hedge"]["pooled"] = {
        cell: _point(_cell_rows(rows, cell), "staged_at_hedge")
        for cell in GATED_CELLS_EXT}
    # (3) promotion on resolve_true
    promo: dict = {}
    for cell in SCORED_CELLS:
        sel = _cell_rows(rows, cell, "resolve_true")
        entry: dict = {"end_state_eq_new_value_ITT": _point(
            sel, "end_state_eq_new_value")}
        if cell in GATED_CELLS_EXT:
            entry["promote_fire_at_tstar_ITT"] = _point(sel, "promote_at_tstar")
            entry["promote_fire_within_window_ITT"] = _point(
                sel, "promote_within_window")
            staged_sel = [r for r in sel if r["staged_at_hedge"]]
            entry["conditional_on_staged"] = {
                "n": len(staged_sel),
                "promote_fire_within_window": _point(
                    staged_sel, "promote_within_window"),
            } if staged_sel else {"n": 0}
            delays = [r["promotion_delay_user_turns"] for r in sel
                      if r["promotion_delay_user_turns"] is not None]
            entry["promotion_delay_user_turns"] = {
                "n_fired": len(delays), "delays": sorted(delays),
                "mean": (sum(delays) / len(delays)) if delays else None,
                "max": max(delays) if delays else None}
        promo[cell] = entry
    endpoints["promotion_resolve_true"] = promo
    # (4) abandoned handling
    endpoints["abandoned_handling"] = {
        "gated_correct_ITT": {cell: _point(
            _cell_rows(rows, cell, "abandoned"), "abandoned_correct")
            for cell in GATED_CELLS_EXT},
        "committed_at_hedge": {cell: _point(
            _cell_rows(rows, cell, "abandoned"), "committed_at_hedge")
            for cell in SCORED_CELLS},
        "no_false_memory": {cell: _point(
            _cell_rows(rows, cell, "abandoned"), "no_false_memory")
            for cell in SCORED_CELLS},
    }
    # (5) open handling
    endpoints["open_handling"] = {
        "no_commit_anywhere": {cell: _point(
            _cell_rows(rows, cell, "open"), "no_commit_anywhere")
            for cell in SCORED_CELLS},
        "gated_correct_hold_ITT": {cell: _point(
            _cell_rows(rows, cell, "open"), "open_correct_hold")
            for cell in GATED_CELLS_EXT},
        "no_false_memory": {cell: _point(
            _cell_rows(rows, cell, "open"), "no_false_memory")
            for cell in SCORED_CELLS},
    }
    # (6) task probes
    probes: dict = {cls: {cell: _point(_cell_rows(rows, cell, cls),
                                       "probe_correct") for cell in SCORED_CELLS}
                    for cls in CLASSES}
    probes["pooled"] = {cell: _point(_cell_rows(rows, cell), "probe_correct")
                        for cell in SCORED_CELLS}
    endpoints["task_probes"] = probes

    # hedge-turn funnel diagnostics (per family per class; never a bar).
    # DIAG_FAMILIES adds em_fixed (the wrapper-mapped em) so the entry-fix
    # comparison can read the recovered mapping rate.
    hedge_mapped: dict = {fam: {} for fam in DIAG_FAMILIES}
    hedge_extracted: dict = {fam: {} for fam in DIAG_FAMILIES}
    # hedge-TUPLE-specific mapping (did the matched hedge tuple itself map to
    # the arc key) — the precise entry-funnel metric the M1 stem-wrapper
    # targets. `hedge_mapped_rate` (any tuple at the hedge turn) can be
    # saturated by non-hedge tuples; row (iii) keeps using it for continuity,
    # while the entry-fix comparison drives its verdict off this one.
    hedge_tuple_mapped: dict = {fam: {} for fam in DIAG_FAMILIES}
    for fam in DIAG_FAMILIES:
        for cls in CLASSES + ("pooled",):
            sel = [c for c in cases
                   if cls == "pooled" or c["resolution_type"] == cls]
            drows = [{"cluster": c["cluster"],
                      "mapped": int(c["diagnostics"][fam]["hedge_turn"]
                                    ["any_tuple_mapped_to_target"]),
                      "tuple_mapped": int(c["diagnostics"][fam]["hedge_turn"]
                                          ["target_mapped_via_matched_tuple"]),
                      "extracted": int(c["diagnostics"][fam]["hedge_turn"]
                                       ["target_extracted"])}
                     for c in sel if fam in c.get("diagnostics", {})]
            hedge_mapped[fam][cls] = _point(
                [{"cluster": d["cluster"], "f": d["mapped"]} for d in drows], "f")
            hedge_tuple_mapped[fam][cls] = _point(
                [{"cluster": d["cluster"], "f": d["tuple_mapped"]} for d in drows], "f")
            hedge_extracted[fam][cls] = _point(
                [{"cluster": d["cluster"], "f": d["extracted"]} for d in drows], "f")
    diagnostics = {"hedge_mapped_rate": hedge_mapped,
                   "hedge_tuple_mapped_rate": hedge_tuple_mapped,
                   "hedge_extracted_rate": hedge_extracted}

    # row (ii) hedge-strength breakdown (descriptive)
    strength_breakdown: dict = {}
    for strength in sorted({r["hedge_strength"] for r in rows}):
        sel = [r for r in rows if r["hedge_strength"] == strength]
        strength_breakdown[strength] = {
            "gated_premature": _point(
                [r for r in sel if r["cell"] in GATED_CELLS], "premature_commit"),
            "ungated_premature": _point(
                [r for r in sel if r["cell"] in UNGATED_CELLS], "premature_commit"),
        }

    return {
        "id": "V2.4-M4-EVAL-RESULTS",
        "seed": SEED_M4,
        "n_arcs_scored": len(cases),
        "partial": len(cases) < 60,
        "arcs_sha256": next(iter(digests)),
        "stratum_note": ("FINAL — owner realism review round 4 ACCEPTED all 60 "
                         "arcs (owner_review.json, 2026-07-09); per-arc rows "
                         "carry arc_id"),
        "endpoint_note": ("directional endpoints only — no go/no-go bars "
                          "(M4-EVAL preregistration, 2026-07-09)"),
        "mapper": ("four frozen cells: frozen v2 key mapper, mode OFF (M1 stem "
                   "wrapper not installed); 5th arm em_cg_fixed: M1 stem "
                   "wrapper (store.make_tuple_key_mapper, mode "
                   "'plan_stem_wrapper') installed as a LOCAL mapper for that "
                   "arm only"),
        "arms": list(SCORED_CELLS),
        "class_counts": {cls: sum(c["resolution_type"] == cls for c in cases)
                         for cls in CLASSES},
        "endpoints": endpoints,
        "directional_table": directional_table(rows, diagnostics),
        "entry_fix_comparison": entry_fix_comparison(rows, diagnostics),
        "hedge_strength_breakdown": strength_breakdown,
        "hedge_turn_diagnostics": diagnostics,
        "per_arc_rows": rows,
        "scoring_judge_calls": {"total": scorer.judge_calls,
                                "fresh": scorer.fresh_judge_calls},
        "advisory_budget": advisory_status(),
    }


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _fmt(pt: dict | None) -> str:
    if not pt or pt.get("point") is None:
        return "—"
    return (f"{pt['x']}/{pt['n']} = {pt['point']:.3f} "
            f"[{pt['ci'][0]:.3f}, {pt['ci'][1]:.3f}]")


def _fmt_delta(d: dict | None) -> str:
    if not d or d.get("delta") is None:
        return "—"
    return (f"{d['delta']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}] "
            f"(fixed {d.get('a_x')}/{d['n']} vs em_cg {d.get('b_x')}/{d['n']})")


def _render_entry_fix(efc: dict) -> list[str]:
    """The dedicated 'Entry-fix arm (em_cg_fixed vs em_cg)' section."""
    mr = efc["mapping_rate"]
    sr = efc["staging_rate"]
    pr = efc["promotion_within_window_resolve_true"]
    tpd = efc["task_probe_delta"]
    cc = efc["contamination_cost"]
    sub = efc["sub_verdicts"]
    lines = ["", "## Entry-fix arm — em_cg_fixed vs em_cg (5th preregistered "
             "arm; cluster-bootstrap CIs, cluster = arc)", "",
             f"Directional expectation: {efc['expectation']}", "",
             "**Overall verdict: "
             f"{'HOLDS' if efc['holds'] else 'DOES NOT HOLD'}** "
             f"(sub-verdicts: {sub}).", "",
             "### Hedge-turn arc-key mapping rate (the entry funnel)", "",
             "Primary = the matched HEDGE TUPLE's own mapping to the arc key "
             "(the loss the stem-wrapper mechanically recovers); the any-tuple "
             "rate (any tuple at the hedge turn mapping to the arc key — row "
             "(iii)'s metric, saturable by non-hedge tuples) is reported "
             "alongside.", "",
             "| class | em_cg (hedge tuple) | em_cg_fixed (hedge tuple) "
             "| em_cg (any tuple) | em_cg_fixed (any tuple) |",
             "|---|---|---|---|---|"]
    mra = efc.get("mapping_rate_any_tuple", {})
    for cls in CLASSES + ("pooled",):
        a = mra.get(cls, {})
        lines.append(f"| {cls} | {_fmt(mr[cls]['em_cg'])} "
                     f"| {_fmt(mr[cls]['em_cg_fixed'])} "
                     f"| {_fmt(a.get('em_cg'))} | {_fmt(a.get('em_cg_fixed'))} |")
    lines += ["", "### Staging rate at the hedge turn", "",
              "| class | em_cg | em_cg_fixed |", "|---|---|---|"]
    for cls in CLASSES:
        lines.append(f"| {cls} | {_fmt(sr['em_cg'][cls])} "
                     f"| {_fmt(sr['em_cg_fixed'][cls])} |")
    lines.append(f"| pooled | {_fmt(sr['pooled']['em_cg'])} "
                 f"| {_fmt(sr['pooled']['em_cg_fixed'])} |")
    lines += ["", "### Promotion (resolve_true, within window)", "",
              "| em_cg | em_cg_fixed |", "|---|---|",
              f"| {_fmt(pr['em_cg'])} | {_fmt(pr['em_cg_fixed'])} |"]
    lines += ["", "### Task-probe correctness + paired delta "
              "(em_cg_fixed − em_cg)", "",
              "| class | em_cg | em_cg_fixed | paired delta [CI] |",
              "|---|---|---|---|"]
    for cls in CLASSES + ("pooled",):
        lines.append(f"| {cls} | {_fmt(tpd[cls]['em_cg'])} "
                     f"| {_fmt(tpd[cls]['em_cg_fixed'])} "
                     f"| {_fmt_delta(tpd[cls]['delta'])} |")
    lines += ["", "### Contamination cost (does routing more plan-frame "
              "tuples to the gate raise leakage?)", "",
              "| metric | em_cg | em_cg_fixed |", "|---|---|---|",
              "| final-store contamination (abandoned+open) "
              f"| {_fmt(cc['final_store_abandoned_open']['em_cg'])} "
              f"| {_fmt(cc['final_store_abandoned_open']['em_cg_fixed'])} |",
              "| pre-resolution supplementary (abandoned+open) "
              f"| {_fmt(cc['pre_resolution_supplementary_abandoned_open']['em_cg'])} "
              f"| {_fmt(cc['pre_resolution_supplementary_abandoned_open']['em_cg_fixed'])} |",
              "", f"Operationalization: {efc['operationalization']}"]
    return lines


def render_report(res: dict) -> str:
    ep = res["endpoints"]
    lines = [
        "# v2.4 M4-EVAL-r4 — five-arm experiment set on the r4 realistic "
        "substrate",
        "",
        "Preregistered 2026-07-09 (protocol/CHANGELOG.md, M4-EVAL-r4 entry). "
        f"Scored arcs: {res['n_arcs_scored']}/60"
        + (" (PARTIAL — smoke or in-progress run)" if res["partial"] else "")
        + f"; class counts {res['class_counts']}. Substrate arcs.jsonl sha256 "
        f"{res['arcs_sha256'][:16]}…; stratum FINAL — owner review round 4 "
        "accepted all 60 arcs (2026-07-09). Arms: the four FROZEN "
        "cells em/ez × ungated (cw) / gated (cg), frozen seams as M2, key "
        "mapper mode OFF; PLUS the 5th preregistered entry-fix arm "
        "**em_cg_fixed** (cautious em + gated, with the M1 stem-wrapper mapper "
        "installed as a LOCAL mapper for that arm only — the four frozen cells "
        "are byte-identical with or without it). Cluster = conversation (arc); "
        "ITT denominators = all arcs in class. n = 20 per class — small-n CIs "
        "expected.",
        "",
        "## (1) Contamination (abandoned/open: final store) + premature "
        "commit (resolve_true: pre-resolution boundary)",
        "",
        "A resolve_true final-store match is CORRECT tracking, never "
        "contamination (spec). Open-arc boundary = end of conversation.",
        "",
        "| endpoint | cell | rate |",
        "|---|---|---|",
    ]
    for cls in ("abandoned", "open"):
        for cell in SCORED_CELLS:
            lines.append(f"| contamination ({cls}) | {cell} "
                         f"| {_fmt(ep['contamination_final_store'][cls][cell])} |")
    for cell in SCORED_CELLS:
        lines.append(f"| premature commit (resolve_true) | {cell} "
                     f"| {_fmt(ep['premature_commit_resolve_true'][cell])} |")
    sup = ep["pre_resolution_boundary_contamination_supplementary"]
    for cls in ("abandoned", "open"):
        for cell in SCORED_CELLS:
            lines.append(f"| pre-resolution boundary contamination ({cls}, "
                         "supplementary — 2026-07-09 addendum) "
                         f"| {cell} | {_fmt(sup[cls][cell])} |")
    lines += ["", "Supplementary line per the 2026-07-09 changelog addendum "
              "(logged before any full-run scoring): an in-arc corrective "
              "resolution can mask mid-arc exposure, so abandoned/open arcs "
              "also report the recorded pre-resolution snapshot.",
              "", "## (2) Staging at the hedge turn (gated arms)", "",
              "| class | cell | staging rate |", "|---|---|---|"]
    for cls in CLASSES + ("pooled",):
        for cell in GATED_CELLS_EXT:
            lines.append(f"| {cls} | {cell} "
                         f"| {_fmt(ep['staging_at_hedge'][cls][cell])} |")
    lines += ["", "## (3) Promotion on resolve_true", "",
              "Fire semantics: valid promote at t* (primary) / within "
              "[t*, t*+2] user turns (sensitivity, test3). End-state = final "
              "store equivalent to the new value (all arms).", "",
              "| cell | fire at t* | fire in window | end-state | "
              "cond-on-staged (window) | delay (user turns) |",
              "|---|---|---|---|---|---|"]
    for cell in SCORED_CELLS:
        e = ep["promotion_resolve_true"][cell]
        cond = e.get("conditional_on_staged") or {}
        delay = e.get("promotion_delay_user_turns") or {}
        delay_s = ("—" if not delay.get("n_fired")
                   else f"mean {delay['mean']:.2f}, max {delay['max']} "
                        f"(n={delay['n_fired']})")
        lines.append(
            f"| {cell} "
            f"| {_fmt(e.get('promote_fire_at_tstar_ITT')) if cell in GATED_CELLS_EXT else 'n/a'} "
            f"| {_fmt(e.get('promote_fire_within_window_ITT')) if cell in GATED_CELLS_EXT else 'n/a'} "
            f"| {_fmt(e['end_state_eq_new_value_ITT'])} "
            f"| {_fmt(cond.get('promote_fire_within_window')) if cond.get('n') else 'n/a'} "
            f"| {delay_s if cell in GATED_CELLS_EXT else 'n/a'} |")
    lines += ["", "## (4) Abandoned handling", "",
              "Gated correct (spec): hedge-turn staged item invalidated or "
              "still pending at end AND the hedged value never committed. "
              "`no_false_memory` (never committed + final store lacks the "
              "value) reported beside as the unstaged-inclusive variant.", "",
              "| metric | cell | rate |", "|---|---|---|"]
    for cell in GATED_CELLS_EXT:
        lines.append(f"| gated correct (ITT) | {cell} "
                     f"| {_fmt(ep['abandoned_handling']['gated_correct_ITT'][cell])} |")
    for cell in SCORED_CELLS:
        lines.append(f"| committed at the hedge | {cell} "
                     f"| {_fmt(ep['abandoned_handling']['committed_at_hedge'][cell])} |")
    for cell in SCORED_CELLS:
        lines.append(f"| no false memory | {cell} "
                     f"| {_fmt(ep['abandoned_handling']['no_false_memory'][cell])} |")
    lines += ["", "## (5) Open handling", "",
              "| metric | cell | rate |", "|---|---|---|"]
    for cell in SCORED_CELLS:
        lines.append(f"| no commit anywhere | {cell} "
                     f"| {_fmt(ep['open_handling']['no_commit_anywhere'][cell])} |")
    for cell in GATED_CELLS_EXT:
        lines.append(f"| gated correct hold (still pending, ITT) | {cell} "
                     f"| {_fmt(ep['open_handling']['gated_correct_hold_ITT'][cell])} |")
    lines += ["", "## (6) Task probes (deterministic final-store lookup + "
              "frozen equivalence judge)", "",
              "Expected: resolve_true → the hedge target value; abandoned "
              "kept_prior and open → the established prior value; abandoned "
              "alternative → the adopted alternative value. Every r4 arc "
              "establishes a designed prior current value before the hedge "
              "(prior_establish at s0 t0), so a correct abandoned/open probe "
              "means the store HOLDS the right current value — the "
              "abstention-by-absence branch is unused on this substrate "
              "(interpretive choice #1, module docstring — raw probe values "
              "ride on every per-arc row).", "",
              "| class | cell | probe correct |", "|---|---|---|"]
    for cls in CLASSES + ("pooled",):
        for cell in SCORED_CELLS:
            lines.append(f"| {cls} | {cell} "
                         f"| {_fmt(ep['task_probes'][cls][cell])} |")
    lines += ["", "## Preregistered directional comparisons — four frozen "
              "cells (holds / doesn't — no bars)", "",
              "| row | expectation | observed | holds |", "|---|---|---|---|"]
    for row in res["directional_table"]:
        if row["row"] == "(i)":
            obs = (f"em_cw {_fmt(row['observed']['em_cw'])}; "
                   f"ez_cw {_fmt(row['observed']['ez_cw'])}")
        elif row["row"] == "(ii)":
            o = row["observed"]
            obs = (f"gated {_fmt(o['gated_pooled'])}; "
                   f"ungated {_fmt(o['ungated_pooled'])}; "
                   f"leak strengths {o['leak_hedge_strengths'] or '—'}")
        elif row["row"] == "(iii)":
            o = row["observed"]
            obs = (f"promotion em_cg {_fmt(o['em_cg_promote_within_window'])} "
                   f"vs ez_cg {_fmt(o['ez_cg_promote_within_window'])}; "
                   f"hedge-mapped em {_fmt(o['em_hedge_mapped_resolve_true'])} "
                   f"vs ez {_fmt(o['ez_hedge_mapped_resolve_true'])}")
        else:
            o = row["observed"]
            def _d(v):
                return f"{v:+.3f}" if v is not None else "—"
            obs = (f"delta_em {_d(o['em']['delta'])}; "
                   f"delta_ez {_d(o['ez']['delta'])}")
        lines.append(f"| {row['row']} | {row['expectation']} | {obs} "
                     f"| **{'HOLDS' if row['holds'] else 'DOES NOT HOLD'}** |")
    lines += ["", "Operationalizations (this module's, the spec entry gives "
              "directions only):"]
    for row in res["directional_table"]:
        lines.append(f"- {row['row']}: {row['operationalization']}")

    # Entry-fix arm comparison (em_cg_fixed vs em_cg)
    efc = res.get("entry_fix_comparison")
    if efc is not None:
        lines += _render_entry_fix(efc)

    lines += ["", "## Hedge-turn funnel diagnostics (no bar)", "",
              "em_fixed = the em family re-mapped by the M1 stem-wrapper (the "
              "5th arm's entry funnel).", "",
              "| family | class | target extracted | mapped to arc key |",
              "|---|---|---|---|"]
    hd = res["hedge_turn_diagnostics"]
    for fam in DIAG_FAMILIES:
        for cls in CLASSES + ("pooled",):
            lines.append(
                f"| {fam} | {cls} "
                f"| {_fmt(hd['hedge_extracted_rate'].get(fam, {}).get(cls))} "
                f"| {_fmt(hd['hedge_mapped_rate'].get(fam, {}).get(cls))} |")
    adv = res["advisory_budget"]
    jc = res["scoring_judge_calls"]
    lines += ["", "## Budget (advisory stage / hard envelope)", "",
              f"Stage m4 spend {adv['spent']:,} vs advisory forecast "
              f"{adv['advisory_forecast']:,} (over: {adv['over_forecast']}); "
              f"envelope {adv['envelope_spent_total']:,}/{adv['envelope']:,} "
              "(hard limit is the total envelope only, 2026-07-08 amendment).",
              "",
              f"Scoring equivalence-judge calls: {jc['total']} "
              f"({jc['fresh']} fresh, stage m4).", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# smoke — 2 preregistered arcs end-to-end (run + score), then STOP
# ---------------------------------------------------------------------------

def smoke_arc_ids(arcs: list[dict] | None = None) -> tuple[str, str]:
    """Deterministic smoke picks: the first resolve_true and the first
    abandoned arc in arc_id sort order."""
    arcs = load_arcs() if arcs is None else arcs
    rt = next(a["arc_id"] for a in arcs if a["resolution_type"] == "resolve_true")
    ab = next(a["arc_id"] for a in arcs if a["resolution_type"] == "abandoned")
    return rt, ab


async def smoke(call_cap: int = 250) -> dict:
    """Run the two smoke arcs through all FIVE arms (case records land in
    the REAL cases dir — the full run resumes past them), then score the
    available case records into eval/smoke/."""
    ids = list(smoke_arc_ids())
    guard = m4_guard(local_cap=call_cap)
    run_summary = await run(CLASSES, FAMILIES, ids, CASES_DIR, guard)

    score_guard = m4_guard()
    score_guard.install()
    try:
        res = await compute_scores(CASES_DIR)
    finally:
        score_guard.uninstall()
    res["score_fresh_calls"] = score_guard.fresh
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    (SMOKE_DIR / "m4_eval_results.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False))
    (SMOKE_DIR / "m4_eval_report.md").write_text(render_report(res))
    return {"smoke_arc_ids": ids, "run": run_summary,
            "n_arcs_scored": res["n_arcs_scored"],
            "score_fresh_calls": score_guard.fresh,
            "results": str(SMOKE_DIR / "m4_eval_results.json"),
            "advisory": advisory_status()}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="v2.4 M4-EVAL runner")
    sub = ap.add_subparsers(dest="cmd", required=True)
    runp = sub.add_parser("run")
    runp.add_argument("--classes", default=",".join(CLASSES))
    runp.add_argument("--families", default=",".join(FAMILIES))
    runp.add_argument("--ids", default=None)
    runp.add_argument("--out", default=str(CASES_DIR))
    runp.add_argument("--call-cap", type=int, default=None)
    sc = sub.add_parser("score")
    sc.add_argument("--cases", default=str(CASES_DIR))
    sc.add_argument("--out-dir", default=str(EVAL_DIR))
    sm = sub.add_parser("smoke")
    sm.add_argument("--call-cap", type=int, default=250)
    args = ap.parse_args()

    if args.cmd == "run":
        guard = m4_guard(local_cap=args.call_cap)
        summary = asyncio.run(run(
            tuple(args.classes.split(",")),
            tuple(args.families.split(",")),
            ([s for s in args.ids.split(",") if s] if args.ids else None),
            Path(args.out), guard))
        print(json.dumps(summary, indent=1))
    elif args.cmd == "score":
        guard = m4_guard()
        guard.install()
        try:
            res = asyncio.run(compute_scores(Path(args.cases)))
        finally:
            guard.uninstall()
        res["score_fresh_calls"] = guard.fresh
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        results_path = out_dir / RESULTS_PATH.name
        report_path = out_dir / REPORT_PATH.name
        results_path.write_text(json.dumps(res, indent=1, ensure_ascii=False))
        report_path.write_text(render_report(res))
        print(f"[score] -> {_rel(results_path)} + {_rel(report_path)} "
              f"({guard.fresh} fresh calls; n_arcs={res['n_arcs_scored']}"
              f"{' PARTIAL' if res['partial'] else ''})")
    elif args.cmd == "smoke":
        out = asyncio.run(smoke(args.call_cap))
        print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
