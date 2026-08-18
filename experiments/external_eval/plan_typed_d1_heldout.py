"""plan_typed_d1_heldout.py — Tier-1 item 5: the plan-typed-commit baseline
(em_pt / ez_pt) extended to the HELD-OUT D1 test slice (147 conversations).

Preregistration: protocol/CHANGELOG.md 2026-08-14 "PREREGISTRATION:
reviewer-response experiment set (Tier 1)", item 5 — "extend the plan-typed
arms (both styles, both readings, same read rule) to the 147 held-out
conversations; endpoints as the D1 scorer". Binding mechanism spec =
the 2026-08-12 plan-typed prereg + its release entry (IC-1..IC-6 as
implemented in experiments/v2_full/plan_typed.py, REUSED here, not copied).

MECHANISM (unchanged; plan_typed._plan_typed_tuples imported verbatim):
identical frozen pipeline through classification; speculative -> COMMIT with
type=plan (validity history preserved, prior interval closes normally);
completed_event / explicit_correction / other -> exactly the frozen ungated
write. READ RULE: most recent NON-plan-typed committed value, else
abstention-by-absence. Confirmations resolve via route 2 only (the confirming
turn's own commit); on a refresh-equal non-plan commit the IC-4
refresh-confirm flip marks the plan entry confirmed (run-time semantics =
the released with-flip arm; IC-1 stored_value = the with-flip read state).

DUAL READING (the release framing is bound to it — recorded NATIVELY here):
  * with-flip — the released arm definition: a flipped (confirmed) entry is
    returnable as current.
  * strict    — no resolution mechanism at all: a flipped entry is STILL a
    plan; only never-plan commits are returnable.
Every case record carries the per-user-turn value timeline, final values and
masked flags under BOTH readings (the strict reading is a deterministic
re-read of the same store history — same records, no second run).

SUBSTRATE (the runner_external D1 pattern, reused): adapter_d1.adapt streams
(holdout guard enforced — the test slice requires EXTERNAL_TEST_RUN_
AUTHORIZED=1, set ONLY for this preregistered Tier-1 run), the frozen LME B2
extraction + mapping seams via runner_external._extract_and_map_lme.
IC-2 analog, disclosed: em_pt mirrors the released D1 gated comparison cell
em_cg — the ADOPTED M1 plan-stem wrapper as a LOCAL mapper (byte-identical
mapping calls -> cache-shared with the released test em_cg recs);
ez_pt uses the frozen mapper (cache-shared with the released ez recs).
Extraction calls are byte-identical to the released D1 test run for both
families (heavy cache-sharing expected). assert_mapper_mode_off at entry and
teardown of every conversation.

ENDPOINTS (scorer_d1's endpoint logic pointed at the new records; frozen
judge rule5.EquivalenceScorer; comparison rows QUOTED verbatim from the
released results/external_eval/test/scores_d1.json — never rerun):
  * E1 at-end contamination (at-risk = abandoned + still_open): the read
    rule's final value matches label new_value = RETURNED-AS-CURRENT
    (contamination); the raw current entry is plan-typed under the reading
    and matches new_value while the read value does not = MASKED-PRESENT
    (NOT contamination; reported separately). Both readings. While-pending
    reported as the released scorer defines it (abandoned windows are empty
    by geometry — the released F1 note is carried).
  * E3 task probes ITT: probe value = the reading's read-rule final value
    (present + non-sentinel, else abstention = incorrect); judged vs the
    held-out expected_answer exactly as scorer_d1 (attribute + two values,
    no question text).
  * came_true end-state correctness: the E3 came_true strata (directly
    comparable to the released cells) + the supplementary
    end-state-eq-new-value and the masked-at-end no-route-1 signature.
  * E4 controls: committed_immediately = a FACT-typed commit-family event
    (commit/supersede; with-flip additionally counts a refresh-confirm flip)
    at the control turn matching label final_value; plus the pt analog of
    over-holding: the control change plan-typed (masked) at end.
  * masked-residue rate: per-arm share of plan-typed commits still
    unresolved (never flipped) in the final store, store-wide and on the
    target field; plus per-conversation masked-at-end rates per reading.

STATISTICS: x/n + stats.wilson_ci per the released D1 convention (cluster =
conversation, one item each).

BUDGET: stage "tier1" (registered from unallocated headroom, ADVISORY —
flock-shared with the concurrent Tier-1 agents via forecast_v22.
SharedCallBudget); the 120k envelope stays the protocol hard stop; THIS run
additionally refuses to start / stops if the combined total is projected
past 110,000 (the Tier-1 prereg stop condition).

OUTPUT: case records -> results/tier1/d1_plan_typed/cases_d1/;
score once  -> results/tier1/plan_typed_heldout_results.json + _report.md.
Scored once; RECOUNT + SEMANTICS verification precedes any release.

CLI (repo root; MEMGATE_SYSTEM_MODEL must be UNSET; run needs
EXTERNAL_TEST_RUN_AUTHORIZED=1):
  pilot/.venv/bin/python experiments/external_eval/plan_typed_d1_heldout.py \
      run   [--ids ...] [--shard K/N] [--call-cap N]
      score [--force] [--call-cap N]
      status
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401  (R1 layout: src/ + experiments/* on sys.path)

import adapter_d1
import forecast_v22
import holdout_guard
import llm
import mapper_fix
import natural_check_rule5 as rule5
import plan_typed as pt
import runner_external as rx
import scorer_d1 as sd1
import store
from graded_eval import _CellSim
from m4_eval import assert_mapper_mode_off

from paths import RESULTS_DIR

STAGE = "tier1"
#: ADVISORY stage allocation, registered from the remaining unallocated
#: headroom (the Tier-1 prereg estimates +5-8k for the WHOLE set; stages are
#: advisory per the 2026-07-08 amendment — only the envelope is hard).
TIER1_ADVISORY = 4_000
PINNED_SYSTEM_MODEL = pt.PINNED_SYSTEM_MODEL
#: Tier-1 prereg stop condition (the protocol hard stop stays 120k).
HARD_STOP_TOTAL = 110_000

V24_DIR = RESULTS_DIR / "v2.4"
BUDGET_PATH = V24_DIR / "budget.json"

TIER1_DIR = RESULTS_DIR / "tier1"
PT_D1_DIR = TIER1_DIR / "d1_plan_typed"
CASES_DIR = PT_D1_DIR / "cases_d1"
RUN_LOG = PT_D1_DIR / "run_log.jsonl"
BUDGET_BEFORE = PT_D1_DIR / "budget_before.json"
RESULTS_PATH = TIER1_DIR / "plan_typed_heldout_results.json"
REPORT_PATH = TIER1_DIR / "plan_typed_heldout_report.md"

#: Released one-shot D1 test scores — comparison rows are QUOTED, never rerun.
RELEASED_SCORES = rx.OUT_ROOT / "test" / "scores_d1.json"

ARMS = ("em_pt", "ez_pt")
READINGS = ("with_flip", "strict")
RELEASED_CELLS = ("em_cw", "em_cg", "ez_cw", "ez_cg")
GATED_OF = {"em_pt": "em_cg", "ez_pt": "ez_cg"}
UNGATED_OF = {"em_pt": "em_cw", "ez_pt": "ez_cw"}
JUDGE_CALL_CAP = 10_000
APPROX = pt.APPROX  # the disclosed IC-6 tolerance convention (0.05)

MECHANISM_NOTE = (
    pt.MECHANISM_NOTE
    + " DUAL READING (Tier-1 item 5): every value is recorded under BOTH the "
      "with-flip reading (released arm definition; a confirmed/flipped entry "
      "is returnable) and the strict reading (no resolution mechanism; a "
      "flipped entry is still a plan). Run-time pipeline semantics (IC-1 "
      "stored_value) = with-flip, matching the released arm.")

EM_PT_MAPPER_NOTE = (
    "em_pt: the ADOPTED M1 plan-stem wrapper as a LOCAL mapper — the "
    "released D1 gated comparison cell em_cg's mapping, byte-identical and "
    "cache-shared (IC-2 analog); ez_pt: frozen mapper (the released ez "
    "recs). Never installed globally.")


# ---------------------------------------------------------------------------
# Budget (stage tier1; advisory; 110k projected-total stop)
# ---------------------------------------------------------------------------

def register_stage(path: Path = BUDGET_PATH) -> dict:
    """Register the tier1 stage from unallocated headroom (idempotent;
    flock + atomic rename). Tolerates an existing registration by a
    concurrent Tier-1 agent regardless of its allocation (stages are
    advisory; the envelope + the 110k stop are the binding limits)."""
    lock_path = path.with_suffix(".lock")
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        budget = json.loads(path.read_text())
        if STAGE in budget["spent"]:
            return budget
        headroom = budget["allocation"].get("unallocated_headroom", 0)
        alloc = min(TIER1_ADVISORY, headroom)
        budget["allocation"]["unallocated_headroom"] = headroom - alloc
        budget["allocation"][STAGE] = alloc
        budget["spent"][STAGE] = 0
        budget.setdefault("notes", []).append(
            f"2026-08-14: tier1 {alloc} from headroom (reviewer-response "
            "experiment set, Tier-1 preregistration 2026-08-14; the prereg "
            "estimates +5-8k for the whole set — the stage is ADVISORY per "
            "the 2026-07-08 amendment, shared by the concurrent Tier-1 "
            "agents; 120k envelope hard; Tier-1 runs stop if the combined "
            "total is projected past 110,000).")
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(budget, indent=1, ensure_ascii=False))
        tmp.replace(path)
        return budget


def tier1_guard(path: Path = BUDGET_PATH,
                local_cap: int | None = None) -> forecast_v22.SharedCallBudget:
    register_stage(path)
    budget = mapper_fix.ensure_budget(path)
    spent = budget["spent"].get(STAGE, 0)
    forecast = budget["allocation"].get(STAGE, TIER1_ADVISORY)
    if spent >= forecast:
        print(f"[tier1 budget ADVISORY] stage spend {spent:,} >= forecast "
              f"{forecast:,} — continuing (stages advisory; only the total "
              "envelope is hard)")
    margin_110 = HARD_STOP_TOTAL - sum(budget["spent"].values())
    if margin_110 <= 0:
        raise SystemExit(
            f"tier1 STOP CONDITION: combined spend "
            f"{sum(budget['spent'].values()):,} >= {HARD_STOP_TOTAL:,} — "
            "prereg stop; report and stop")
    remaining = min(forecast_v22.budget_remaining(budget), margin_110)
    cap = remaining if local_cap is None else min(local_cap, remaining)
    return forecast_v22.SharedCallBudget(STAGE, path=path, local_cap=cap)


def advisory_status(path: Path = BUDGET_PATH) -> dict:
    budget = mapper_fix.ensure_budget(path)
    return {
        "stage": STAGE,
        "advisory_forecast": budget["allocation"].get(STAGE),
        "spent": budget["spent"].get(STAGE, 0),
        "envelope": budget["envelope"],
        "envelope_spent_total": sum(budget["spent"].values()),
        "run_stop_total": HARD_STOP_TOTAL,
        "mode": ("ADVISORY stage / HARD total envelope (2026-07-08 "
                 "amendment) / 110k Tier-1 run stop (2026-08-14 prereg)"),
    }


def total_spent(path: Path = BUDGET_PATH) -> int:
    return sum(mapper_fix.ensure_budget(path)["spent"].values())


def assert_pins() -> None:
    assert os.environ.get("MEMGATE_SYSTEM_MODEL") is None, (
        "MEMGATE_SYSTEM_MODEL is set — the Tier-1 plan-typed extension is "
        "preregistered on the PINNED default gpt-5.5 systems")
    assert llm.MODEL == PINNED_SYSTEM_MODEL, (
        f"llm.MODEL is {llm.MODEL!r} — expected {PINNED_SYSTEM_MODEL}")
    assert llm.JUDGE_MODEL == PINNED_SYSTEM_MODEL, (
        f"llm.JUDGE_MODEL drifted to {llm.JUDGE_MODEL!r}")
    assert_mapper_mode_off()


def log_phase(row: dict) -> None:
    PT_D1_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUN_LOG, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# The strict reading (deterministic re-read of the same store history)
# ---------------------------------------------------------------------------

def _was_plan(entry) -> bool:
    """True iff the entry was ORIGINALLY plan-typed: still typed plan, or
    flipped by the IC-4 refresh-confirm (confirmed_seq stamped)."""
    prov = entry.provenance if isinstance(entry.provenance, dict) else {}
    return prov.get("value_type", pt.FACT) == pt.PLAN or "confirmed_seq" in prov


def read_current_strict(zstore, key: tuple) -> str | None:
    """The STRICT read rule: only never-plan commits are returnable; a
    flipped (confirmed) plan entry is still a plan. Else abstention."""
    best = None
    for entry in zstore.history(key):
        if _was_plan(entry):
            continue
        if best is None or entry.created_seq > best.created_seq:
            best = entry
    return best.value if best is not None else None


def masked_value(zstore, key: tuple, reading: str) -> str | None:
    """The raw CURRENT value when the reading's read rule masks it (the
    current entry is plan-typed under that reading), else None."""
    if reading == "with_flip":
        return pt.plan_masked_value(zstore, key)
    entry = zstore.current_entry(key)
    if entry is not None and _was_plan(entry):
        return entry.value
    return None


def _plan_store_summary(sim: _CellSim, target: store.CanonicalKey) -> dict:
    """Final-store plan-residue accounting, store-wide and on the target
    field: every ORIGINALLY-plan entry is exactly one of flipped (confirmed,
    resolved by the IC-4 route), current_masked (still plan AND the field's
    current entry — the masked residue the read rule hides), or superseded
    (still plan, interval closed by a later write)."""

    def bucket(entry, cur) -> str:
        prov = entry.provenance if isinstance(entry.provenance, dict) else {}
        if "confirmed_seq" in prov:
            return "flipped"
        is_cur = cur is not None and entry.created_seq == cur.created_seq
        return "current_masked" if is_cur else "superseded"

    out = {"store_wide": Counter(), "target": Counter()}
    for _sid, z in sim.stores.items():
        for key in z.keys():
            cur = z.current_entry(key)
            for entry in z.history(key):
                if not _was_plan(entry):
                    continue
                out["store_wide"]["plan_commits"] += 1
                out["store_wide"][bucket(entry, cur)] += 1
                if key == target.key:
                    out["target"]["plan_commits"] += 1
                    out["target"][bucket(entry, cur)] += 1
    return {k: dict(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Run — one conversation, both pt arms (runner_external D1 pattern)
# ---------------------------------------------------------------------------

async def _run_pt_cell(recs: list[dict], family: str, cid: str,
                       target: store.CanonicalKey,
                       turns: list[dict]) -> dict:
    """One pt arm over one family's per-turn recs. The policy body is
    plan_typed._plan_typed_tuples VERBATIM (imported — run-time semantics =
    the released with-flip arm, IC-1 included); this wrapper only records the
    dual-reading timeline + final state."""
    sim = _CellSim("plan_typed", "lme")
    store_id = sim.route(target.key)
    target_events: list[dict] = []
    timeline: list[dict] = []
    for rec in recs:
        n_events = len(sim.zstore(store_id).events)
        await pt._plan_typed_tuples(sim, rec, turns, family, cid)
        z = sim.zstore(store_id)
        for event in z.events[n_events:]:
            if tuple(event["key"]) != target.key:
                continue
            target_events.append({
                **rx._jsonable_event(event),
                "session_idx": rec["session_idx"],
                "turn_idx": rec["turn_idx"],
                "user_idx": rec["user_idx"],
            })
        timeline.append({
            "session_idx": rec["session_idx"], "turn_idx": rec["turn_idx"],
            "user_idx": rec["user_idx"],
            "value": pt.read_current(z, target.key),          # with-flip
            "value_strict": read_current_strict(z, target.key),
            "raw_value": z.lookup(target.key),
            "masked_plan": masked_value(z, target.key, "with_flip") is not None,
            "masked_plan_strict": masked_value(z, target.key, "strict") is not None,
        })
    z = sim.zstore(store_id)
    finals = {"with_flip": pt.read_current(z, target.key),
              "strict": read_current_strict(z, target.key)}
    raw_final = z.lookup(target.key)
    probe = {}
    for reading, val in finals.items():
        pv = val if rx._answered(val) else None
        probe[reading] = {"value": pv, "answered": pv is not None}
    return {
        "final_value": finals["with_flip"],
        "final_value_strict": finals["strict"],
        "final_value_raw": raw_final,
        "final_masked_plan": masked_value(z, target.key, "with_flip") is not None,
        "final_masked_plan_strict": masked_value(z, target.key, "strict") is not None,
        "final_present": rx._answered(finals["with_flip"]),
        "final_present_strict": rx._answered(finals["strict"]),
        "target_events": target_events,
        "target_value_after_turn": timeline,
        "gate_rows": sim.gate_rows,
        "gate_rows_target": [g for g in sim.gate_rows
                             if g["key"] == target.name],
        "plan_writes_target": sum(1 for e in target_events
                                  if e.get("value_type") == pt.PLAN
                                  and e["action"] != "refresh"),
        "fact_writes_target": sum(1 for e in target_events
                                  if e.get("value_type") == pt.FACT
                                  and e["action"] != "refresh"),
        "plan_confirms_target": sum(1 for e in target_events
                                    if e.get("plan_confirmed")),
        "plan_store_summary": _plan_store_summary(sim, target),
        "probe": {**probe,
                  "semantics": ("deterministic read-rule lookup per reading; "
                                "equivalence vs the held-out expected_answer "
                                "happens ONLY in the scorer")},
        "diag": dict(sim.diag),
    }


async def run_case(conversation_id: str) -> dict:
    stream, key_row = adapter_d1.adapt(conversation_id)  # holdout guard inside
    slice_name = holdout_guard.slice_of(adapter_d1.DATASET, conversation_id)
    assert slice_name == "test", (
        f"{conversation_id} is {slice_name!r} — this Tier-1 extension runs "
        "the held-out TEST slice only")
    turns = adapter_d1.stream_turns(stream)
    user_positions = [i for i, t in enumerate(turns) if t["role"] == "user"]
    target = adapter_d1.canonical_key(key_row)
    candidates = [target]  # the released D1 candidates convention

    out = {
        "conversation_id": conversation_id,
        "dataset": adapter_d1.DATASET,
        "slice": slice_name,
        "question_id": key_row["question_id"],
        "target_key": target.name,
        "probe_question": target.probe_question,
        "candidates_convention": rx.D1_CANDIDATES_NOTE,
        "mechanism": MECHANISM_NOTE,
        "mapper_note": EM_PT_MAPPER_NOTE,
        "prereg": ("protocol/CHANGELOG.md 2026-08-14 Tier-1 item 5; "
                   "mechanism frozen 2026-08-12 (plan-typed prereg + "
                   "release entry, IC-1..IC-4)"),
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
        return rx._attach_user_idx([
            await rx._extract_and_map_lme(family, turns, pos, candidates,
                                          conversation_id, "d1", mapper=mapper)
            for pos in user_positions])

    # em_pt: the ADOPTED stem wrapper, LOCAL (byte-identical to the released
    # test em_cg mapping calls — cache-shared); ez_pt: frozen mapper.
    assert_mapper_mode_off()
    wrapper = store.make_tuple_key_mapper(mapper_fix.PLAN_FRAME_ATTRIBUTE_PATTERNS)
    recs_em = await _recs("em", mapper=wrapper)
    assert_mapper_mode_off()  # the wrapper stayed local
    recs_ez = await _recs("ez")

    out["diagnostics"]["em_fixed"] = [rx._turn_diag_row(r, target)
                                      for r in recs_em]
    out["diagnostics"]["ez"] = [rx._turn_diag_row(r, target) for r in recs_ez]
    out["em_pt_mapping_meta"] = rx._wrapper_mapping_meta(recs_em, target)

    out["cells"]["em_pt"] = await _run_pt_cell(recs_em, "em",
                                               conversation_id, target, turns)
    out["cells"]["ez_pt"] = await _run_pt_cell(recs_ez, "ez",
                                               conversation_id, target, turns)
    assert_mapper_mode_off()
    return out


def _project_and_maybe_stop(done: int, remaining: int, fresh_so_far: int,
                            base_total: int) -> None:
    if done == 0:
        return
    per_case = fresh_so_far / done
    projected = base_total + fresh_so_far + per_case * remaining
    print(f"[tier1 pt-d1 projection] {done} cases done, {fresh_so_far} fresh "
          f"({per_case:.1f}/case); projected combined total "
          f"{projected:,.0f} / {HARD_STOP_TOTAL:,} run stop")
    if projected > HARD_STOP_TOTAL:
        raise SystemExit(
            f"tier1 STOP: projected combined total {projected:,.0f} > "
            f"{HARD_STOP_TOTAL:,} — prereg stop condition; report and stop")


def _shard(ids: list[str], spec: str | None) -> list[str]:
    if not spec:
        return ids
    k, n = (int(x) for x in spec.split("/"))
    assert 0 <= k < n, f"bad shard spec {spec!r}"
    return ids[k::n]


async def run(ids: list[str] | None, shard: str | None,
              call_cap: int | None) -> dict:
    assert_pins()
    assert holdout_guard.test_run_authorized(), (
        f"{holdout_guard.ENV_FLAG}=1 is required — this preregistered Tier-1 "
        "extension ingests the held-out D1 TEST slice (changelog 2026-08-14 "
        "item 5 authorizes the run)")
    sel = _shard(sorted(ids if ids is not None
                        else holdout_guard.test_ids("dataset1")), shard)
    if not BUDGET_BEFORE.exists():
        PT_D1_DIR.mkdir(parents=True, exist_ok=True)
        BUDGET_BEFORE.write_text(BUDGET_PATH.read_text())
    base_total = total_spent()
    guard = tier1_guard(local_cap=call_cap)
    started = datetime.now(timezone.utc).isoformat()
    done = skipped = 0
    failed: list[dict] = []
    n_todo = sum(1 for i in sel if not (CASES_DIR / f"{i}.json").exists())
    guard.install()
    try:
        for cid in sel:
            path = CASES_DIR / f"{cid}.json"
            if path.exists():
                skipped += 1
                continue
            try:
                record = await run_case(cid)
            except (SystemExit, KeyboardInterrupt,
                    holdout_guard.HoldoutViolation):
                raise
            except Exception as exc:  # noqa: BLE001 — recorded, run continues
                low = str(exc).lower()
                if "budget" in low or "envelope" in low or "exhausted" in low:
                    raise
                failed.append({"id": cid, "error": repr(exc)})
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, ensure_ascii=False, indent=1))
            done += 1
            if done in (5, 15) or done % 30 == 0:
                _project_and_maybe_stop(done, n_todo - done, guard.fresh,
                                        base_total)
    finally:
        guard.uninstall()
    summary = {"phase": "tier1_pt_d1_run", "model": llm.MODEL,
               "started": started,
               "ended": datetime.now(timezone.utc).isoformat(),
               "selected": len(sel), "n_new": done, "skipped": skipped,
               "failed": failed, "fresh_calls": guard.fresh,
               "advisory": advisory_status()}
    log_phase(summary)
    return summary


# ---------------------------------------------------------------------------
# Score — scorer_d1 endpoint logic on the pt records, both readings
# ---------------------------------------------------------------------------

def _load_cases() -> list[dict]:
    cases = [json.loads(p.read_text())
             for p in sorted(CASES_DIR.glob("*.json"))]
    assert cases, f"no case records under {CASES_DIR}"
    assert {c["slice"] for c in cases} == {"test"}
    return cases


async def score_item(case: dict, label: dict, conv: dict,
                     scorer) -> list[dict]:
    """Per-conversation rows, one per (pt arm x reading) — the scorer_d1
    score_item endpoint logic on the read-rule values."""
    attr = case["target_key"].partition("/")[2]
    is_control = bool(label["control_type"])
    strength, outcome = label["plan_strength"], label["outcome"]
    at_risk = (not is_control) and outcome in sd1.AT_RISK_OUTCOMES
    n_user = case["geometry"]["n_user_turns"]
    conv_users = sum(1 for t in conv["turns"] if t["speaker"] == "user")
    assert conv_users == n_user, (
        f"{case['conversation_id']}: turn-mapping unsafe "
        f"({conv_users} vs {n_user})")
    last_user_idx = n_user - 1

    et = label["event_turn_ids"]
    plan_u = outcome_u = control_u = None
    if is_control:
        control_u = sd1.user_idx_of_turn(conv, et["control"])
    else:
        plan_u = sd1.user_idx_of_turn(conv, et["possible_switch"])
        if outcome in ("came_true", "abandoned"):
            outcome_u = sd1.user_idx_of_turn(conv, et["outcome"])
        # still_open: window through the LAST user turn (released
        # STILL_OPEN_WINDOW_NOTE, carried verbatim)

    new_value = label["new_value"]
    control_target = (label["final_value"] if label["final_value"] is not None
                      else label["new_value"])

    rows = []
    for arm in [a for a in ARMS if a in case["cells"]]:
        cc = case["cells"][arm]
        tvat = cc["target_value_after_turn"]
        for reading in READINGS:
            vfield = "value" if reading == "with_flip" else "value_strict"
            mfield = ("masked_plan" if reading == "with_flip"
                      else "masked_plan_strict")
            final = (cc["final_value"] if reading == "with_flip"
                     else cc["final_value_strict"])
            final_masked_flag = (cc["final_masked_plan"]
                                 if reading == "with_flip"
                                 else cc["final_masked_plan_strict"])

            async def window(lo_u, hi_u, masked_only: bool) -> bool:
                for entry in tvat:
                    u = entry["user_idx"]
                    if not (lo_u < u < hi_u):
                        continue
                    if masked_only:
                        if not entry.get(mfield):
                            continue
                        eq, _m = await scorer.equivalent(
                            attr, entry["raw_value"], new_value)
                    else:
                        eq, _m = await scorer.equivalent(
                            attr, entry[vfield], new_value)
                    if eq:
                        return True
                return False

            # E1 — at-risk contamination: returned vs masked-present
            while_pending = at_end = None
            while_pending_masked = at_end_masked = None
            if at_risk:
                hi = (outcome_u if outcome == "abandoned"
                      else last_user_idx + 1)
                while_pending = await window(plan_u, hi, False)
                while_pending_masked = await window(plan_u, hi, True)
                eq, _m = await scorer.equivalent(attr, final, new_value)
                at_end = bool(eq)
                raw_eq, _m = await scorer.equivalent(
                    attr, cc["final_value_raw"], new_value)
                at_end_masked = (bool(final_masked_flag) and bool(raw_eq)
                                 and not at_end)

            # came_true — end-state (eq new_value) + masked signature +
            # premature (vacuous by geometry; recorded for completeness)
            end_state_eq_new = end_state_masked = premature = None
            if (not is_control) and outcome == "came_true":
                eq, _m = await scorer.equivalent(attr, final, new_value)
                end_state_eq_new = bool(eq)
                raw_eq, _m = await scorer.equivalent(
                    attr, cc["final_value_raw"], new_value)
                end_state_masked = (bool(final_masked_flag) and bool(raw_eq)
                                    and not end_state_eq_new)
                premature = await window(plan_u, outcome_u, False)

            # E3 — task probe ITT (per reading)
            pv = cc["probe"][reading]["value"]
            probe_answered = bool(cc["probe"][reading]["answered"])
            probe_correct = False
            if probe_answered:
                eq, _m = await scorer.equivalent(
                    attr, pv, label["expected_answer"])
                probe_correct = bool(eq)

            # E4 — controls
            committed_immediately = None
            control_plan_typed_write = None
            control_masked_at_end = None
            if is_control:
                committed_immediately = False
                for ev in cc["target_events"]:
                    if ev["user_idx"] != control_u:
                        continue
                    fact_commit = (ev["action"] in sd1.COMMIT_ACTIONS
                                   and ev.get("value_type") == pt.FACT)
                    flip_here = (reading == "with_flip"
                                 and ev.get("plan_confirmed"))
                    if not (fact_commit or flip_here):
                        continue
                    eq, _m = await scorer.equivalent(
                        attr, ev["value"], control_target)
                    if eq:
                        committed_immediately = True
                        break
                control_plan_typed_write = any(
                    ev.get("value_type") == pt.PLAN
                    and ev["action"] != "refresh"
                    for ev in cc["target_events"])
                control_masked_at_end = bool(final_masked_flag)

            rows.append({
                "conversation_id": case["conversation_id"],
                "cluster": case["conversation_id"],
                "cell": arm, "reading": reading,
                "is_control": is_control,
                "strength": strength, "outcome": outcome, "at_risk": at_risk,
                "while_pending_contamination": while_pending,
                "while_pending_masked_plan": while_pending_masked,
                "at_end_contamination": at_end,
                "at_end_masked_plan": at_end_masked,
                "premature_commit": premature,
                "end_state_eq_new_value": end_state_eq_new,
                "end_state_masked_plan": end_state_masked,
                "probe_answered": probe_answered,
                "probe_correct": probe_correct,
                "probe_unanswered": not probe_answered,
                "probe_value": pv,
                "came_true_probe_correct": (probe_correct
                                            if (not is_control)
                                            and outcome == "came_true"
                                            else None),
                "committed_immediately": committed_immediately,
                "control_plan_typed_write": control_plan_typed_write,
                "control_masked_at_end": control_masked_at_end,
                "target_masked_at_end": (bool(final_masked_flag)
                                         if not is_control else None),
            })
    return rows


def _sel(rows, arm, reading, **kw):
    out = [r for r in rows if r["cell"] == arm and r["reading"] == reading]
    for k, v in kw.items():
        out = [r for r in out if r[k] == v]
    return out


def _strata(rows, flag, outcomes) -> dict:
    out = {}
    for s in sd1.STRENGTHS:
        for o in outcomes:
            out[f"{s}|{o}"] = sd1._pt(
                [r for r in rows if r["strength"] == s
                 and r["outcome"] == o], flag)
    return out


def aggregate(rows: list[dict], cases: list[dict]) -> dict:
    ep: dict = {}
    for arm in ARMS:
        ep[arm] = {}
        for reading in READINGS:
            sel = _sel(rows, arm, reading)
            at_risk = [r for r in sel if r["at_risk"]]
            non_ctrl = [r for r in sel if not r["is_control"]]
            came_true = [r for r in non_ctrl if r["outcome"] == "came_true"]
            ctrl = [r for r in sel if r["is_control"]]
            ep[arm][reading] = {
                "E1_contamination": {
                    "at_end": {
                        "returned_as_current": {
                            "pooled_at_risk": sd1._pt(at_risk,
                                                      "at_end_contamination"),
                            **_strata(at_risk, "at_end_contamination",
                                      sd1.AT_RISK_OUTCOMES)},
                        "masked_present": {
                            "pooled_at_risk": sd1._pt(at_risk,
                                                      "at_end_masked_plan"),
                            **_strata(at_risk, "at_end_masked_plan",
                                      sd1.AT_RISK_OUTCOMES)},
                    },
                    "while_pending": {
                        "returned_as_current": {
                            "pooled_at_risk": sd1._pt(
                                at_risk, "while_pending_contamination"),
                            **_strata(at_risk, "while_pending_contamination",
                                      sd1.AT_RISK_OUTCOMES)},
                        "masked_present": {
                            "pooled_at_risk": sd1._pt(
                                at_risk, "while_pending_masked_plan")},
                    },
                },
                "E3_task_probes": {
                    "pooled_all": sd1._pt(sel, "probe_correct"),
                    "pooled_non_control": sd1._pt(non_ctrl, "probe_correct"),
                    "controls": sd1._pt(ctrl, "probe_correct"),
                    "unanswered_ITT": sd1._pt(sel, "probe_unanswered"),
                    **_strata(non_ctrl, "probe_correct", sd1.OUTCOMES),
                },
                "came_true_end_state": {
                    "probe_correct_pooled": sd1._pt(came_true,
                                                    "probe_correct"),
                    "probe_correct_by_strength": _strata(
                        came_true, "probe_correct", ("came_true",)),
                    "eq_new_value_pooled": sd1._pt(came_true,
                                                   "end_state_eq_new_value"),
                    "eq_new_value_by_strength": _strata(
                        came_true, "end_state_eq_new_value", ("came_true",)),
                    "masked_at_end_pooled": sd1._pt(came_true,
                                                    "end_state_masked_plan"),
                    "masked_at_end_by_strength": _strata(
                        came_true, "end_state_masked_plan", ("came_true",)),
                    "premature_commit_pooled": sd1._pt(came_true,
                                                       "premature_commit"),
                },
                "E4_controls": {
                    "committed_immediately": sd1._pt(ctrl,
                                                     "committed_immediately"),
                    "control_plan_typed_write": sd1._pt(
                        ctrl, "control_plan_typed_write"),
                    "control_masked_at_end": sd1._pt(ctrl,
                                                     "control_masked_at_end"),
                },
                "masked_at_end_non_control": sd1._pt(non_ctrl,
                                                     "target_masked_at_end"),
            }
    # masked residue from the case store summaries (reading-independent
    # components; residue readings differ only in whether flips resolve)
    residue = {}
    for arm in ARMS:
        agg = {"store_wide": Counter(), "target": Counter()}
        for c in cases:
            s = c["cells"][arm]["plan_store_summary"]
            for scope in agg:
                agg[scope].update(s.get(scope, {}))
        residue[arm] = {}
        for scope, cnt in agg.items():
            total = cnt.get("plan_commits", 0)
            flipped = cnt.get("flipped", 0)
            cur = cnt.get("current_masked", 0)
            sup = cnt.get("superseded", 0)
            residue[arm][scope] = {
                "plan_commits": total, "flipped": flipped,
                "current_masked": cur, "superseded": sup,
                "unresolved_with_flip_rate": ((cur + sup) / total
                                              if total else None),
                "unresolved_strict_rate": ((cur + sup + flipped) / total
                                           if total else None),
                "current_masked_rate": (cur / total if total else None),
            }
    return {"per_arm": ep, "masked_residue": residue}


def _quote_released() -> dict:
    rel = json.loads(RELEASED_SCORES.read_text())
    ep = rel["endpoints"]
    out = {
        "source": str(RELEASED_SCORES),
        "note": ("comparison rows quoted VERBATIM from the released one-shot "
                 "D1 test scores — nothing released was rerun"),
        "n_conversations": rel["n_conversations_scored"],
        "E1_at_end": {c: ep["E1_contamination"]["at_end"][c]
                      for c in RELEASED_CELLS},
        "E1_while_pending": {c: ep["E1_contamination"]["while_pending"][c]
                             for c in RELEASED_CELLS},
        "E3_task_probes": {c: ep["E3_task_probes"][c]
                           for c in RELEASED_CELLS},
        "E4_committed_immediately":
            ep["E4_controls"]["committed_immediately"],
        "E4_over_holding": ep["E4_controls"]["over_holding"],
    }
    return out


def _rate(p):
    if not p:
        return None
    return p.get("rate")


def build_verdict(agg: dict, released: dict) -> dict:
    """The dual-reading evaluation on the held-out substrate: does the
    released with-flip parity hold, and what does the strict reading show?
    Operationalized with the disclosed plan-typed APPROX convention (IC-6
    style, 0.05 on points); Wilson CIs reported beside every cell."""
    checks = {"with_flip_parity": {}, "strict_reading": {}}
    parity_ok, strict_ok = [], []
    for arm in ARMS:
        cg = GATED_OF[arm]
        pt_ct = agg["per_arm"][arm]["with_flip"]["came_true_end_state"][
            "probe_correct_pooled"]
        # released came_true pooled across strengths:
        cg_ct_pool = {"x": 0, "n": 0}
        for s in sd1.STRENGTHS:
            p = released["E3_task_probes"][cg][f"{s}|came_true"]
            cg_ct_pool["x"] += p["x"]
            cg_ct_pool["n"] += p["n"]
        cg_ct_pool["rate"] = (cg_ct_pool["x"] / cg_ct_pool["n"]
                              if cg_ct_pool["n"] else None)
        pt_ae = agg["per_arm"][arm]["with_flip"]["E1_contamination"][
            "at_end"]["returned_as_current"]["pooled_at_risk"]
        cg_ae = released["E1_at_end"][cg]["pooled_at_risk"]
        ct_close = (abs(_rate(pt_ct) - cg_ct_pool["rate"]) <= APPROX)
        ae_close = (abs(_rate(pt_ae) - _rate(cg_ae)) <= APPROX)
        checks["with_flip_parity"][arm] = {
            "came_true_probe_correct": {"pt_with_flip": pt_ct,
                                        "released_gated_pooled": cg_ct_pool,
                                        "within_0.05": ct_close},
            "at_end_contamination": {"pt_with_flip": pt_ae,
                                     "released_gated": cg_ae,
                                     "within_0.05": ae_close},
        }
        parity_ok.append(ct_close and ae_close)
        st_ct = agg["per_arm"][arm]["strict"]["came_true_end_state"][
            "probe_correct_pooled"]
        below = (_rate(st_ct) is not None
                 and _rate(st_ct) < cg_ct_pool["rate"])
        checks["strict_reading"][arm] = {
            "came_true_probe_correct_strict": st_ct,
            "released_gated_pooled": cg_ct_pool,
            "below_gated": below,
            "collapse_vs_with_flip": {
                "with_flip": agg["per_arm"][arm]["with_flip"][
                    "came_true_end_state"]["probe_correct_pooled"],
                "strict": st_ct},
        }
        strict_ok.append(below)
    checks["with_flip_parity"]["holds"] = all(parity_ok)
    checks["strict_reading"]["strict_below_gated_holds"] = all(strict_ok)
    checks["operationalization"] = (
        f"with-flip parity holds iff |pt - released gated| <= {APPROX} on "
        "BOTH pooled came_true probe-correct (ITT) and pooled at-risk "
        "at-end contamination, per family (the plan-typed IC-6 APPROX "
        "convention); the strict row holds iff strict came_true "
        "probe-correct < the released gated pooled point, per family (the "
        "2026-08-12 release entry's prediction-row-2 analog)")
    return checks


async def compute_scores(scorer) -> dict:
    cases = _load_cases()
    labels = sd1.load_labels()
    convs = sd1.load_conversations()
    rows: list[dict] = []
    for case in cases:
        cid = case["conversation_id"]
        rows.extend(await score_item(case, labels[cid], convs[cid], scorer))
    agg = aggregate(rows, cases)
    released = _quote_released()
    verdict = build_verdict(agg, released)
    n_items = len({r["conversation_id"] for r in rows})
    n_ctrl = len({r["conversation_id"] for r in rows if r["is_control"]})
    n_at_risk = len({r["conversation_id"] for r in rows if r["at_risk"]})
    phases = ([json.loads(l) for l in RUN_LOG.open()]
              if RUN_LOG.exists() else [])
    return {
        "id": "TIER1-PLAN-TYPED-HELDOUT-D1-RESULTS",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prereg": ("protocol/CHANGELOG.md 2026-08-14 'PREREGISTRATION: "
                   "reviewer-response experiment set (Tier 1)' item 5; "
                   "mechanism per the 2026-08-12 plan-typed prereg + "
                   "release entry (IC-4 dual reading)"),
        "mechanism": MECHANISM_NOTE,
        "system_model": llm.MODEL,
        "judge_model": llm.JUDGE_MODEL,
        "arms": list(ARMS),
        "readings": list(READINGS),
        "n_conversations_scored": n_items,
        "n_controls": n_ctrl,
        "n_at_risk": n_at_risk,
        "n_came_true": n_items - n_ctrl - n_at_risk,
        "partial": n_items < 147,
        "ci_note": ("x/n + Wilson (cluster = conversation, one item each — "
                    "the released D1 convention)"),
        "endpoints": agg["per_arm"],
        "masked_residue": agg["masked_residue"],
        "released_comparison": released,
        "dual_reading_verdict": verdict,
        "geometry_notes": {
            "vacuous_windows": (
                "carried from the released scorer (verifier F1): every "
                "non-control test conversation has exactly 3 user turns, so "
                "the strictly-between while-pending window is EMPTY for "
                "abandoned and came_true items — those zeros are geometry, "
                "not behavior; still_open runs through the last user turn"),
            "while_pending_reading_note": (
                "carried from the released scorer (verifier F2): counted "
                "while-pending events sit at the still_open re-mention "
                "turn (= the last user turn)"),
        },
        "methods": {
            "endpoint_logic": ("scorer_d1 operationalizations pointed at "
                               "the pt records: turn-id mapping by walking "
                               "conversations.jsonl; at-risk pool "
                               "abandoned+still_open; still_open window "
                               "through the last user turn; probe judge "
                               "shape equivalent(attribute, value, "
                               "expected_answer); unanswered = incorrect"),
            "contamination_semantics": (
                "plan-typed prereg: RETURNED-AS-CURRENT by the reading's "
                "read rule = contamination; plan-typed present but MASKED "
                "= not contamination, reported separately"),
            "committed_immediately": (
                "FACT-typed commit/supersede event at the control turn "
                "matching label final_value; the with-flip reading "
                "additionally counts an IC-4 refresh-confirm flip landing "
                "at the control turn"),
            "masked_residue": (
                "per plan-typed commit in the final store: flipped "
                "(IC-4-resolved) / current_masked (read rule hides it) / "
                "superseded (interval closed by a later write); "
                "unresolved_with_flip = current_masked+superseded over all "
                "plan commits; unresolved_strict additionally counts "
                "flipped entries (no resolution mechanism)"),
        },
        "run_phases": phases,
        "scoring_judge_calls": {"total": scorer.judge_calls,
                                "fresh": scorer.fresh_judge_calls},
        "per_item_rows": rows,
        "budget_before": (json.loads(BUDGET_BEFORE.read_text())
                          if BUDGET_BEFORE.exists() else None),
        "budget_after": mapper_fix.ensure_budget(BUDGET_PATH),
        "advisory_budget": advisory_status(),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt(p: dict | None) -> str:
    if not p or p.get("rate") is None:
        return "—"
    return (f"{p['x']}/{p['n']} = {p['rate']:.3f} "
            f"[{p['wilson_lo']:.3f}, {p['wilson_hi']:.3f}]")


def _fmt_q(p: dict | None) -> str:
    if not p or p.get("rate") is None:
        return "—"
    return f"{p['x']}/{p['n']} = {p['rate']:.3f}"


def render_report(res: dict) -> str:
    ep = res["endpoints"]
    rel = res["released_comparison"]
    v = res["dual_reading_verdict"]
    lines = [
        "# Plan-typed baseline on the HELD-OUT D1 test slice (Tier-1 item 5)",
        "",
        f"Preregistered 2026-08-14 (Tier-1 item 5); mechanism frozen "
        f"2026-08-12. Generated {res['generated_at']}. Systems under pinned "
        f"`{res['system_model']}`; judge = frozen `{res['judge_model']}`.",
        "",
        f"Mechanism: {res['mechanism']}",
        "",
        f"Scored {res['n_conversations_scored']}/147 held-out conversations "
        + ("(PARTIAL) " if res["partial"] else "")
        + f"({res['n_at_risk']} at-risk, {res['n_came_true']} came_true, "
        f"{res['n_controls']} controls) x 2 pt arms x 2 readings. "
        f"Comparison rows QUOTED from `{rel['source']}`.",
        "",
        "Contamination semantics (plan-typed prereg): RETURNED-AS-CURRENT "
        "by the reading's read rule = contamination; plan-typed present but "
        "MASKED = not contamination, reported separately.",
        "",
        "## E1 — At-end contamination (at-risk = abandoned + still_open; "
        "Wilson CIs)", ""]
    for reading in READINGS:
        lines += [f"### {reading} reading", "",
                  "| stratum | em_pt returned | em_pt masked | ez_pt "
                  "returned | ez_pt masked | released em_cg | released "
                  "ez_cg | released em_cw | released ez_cw |",
                  "|---|---|---|---|---|---|---|---|---|"]
        em = ep["em_pt"][reading]["E1_contamination"]["at_end"]
        ez = ep["ez_pt"][reading]["E1_contamination"]["at_end"]
        strata = ["pooled_at_risk"] + [
            k for k in em["returned_as_current"] if k != "pooled_at_risk"]
        for s in strata:
            rel_key = "pooled_at_risk" if s == "pooled_at_risk" else s
            lines.append(
                f"| {s.replace('|', chr(92) + '|')} "
                f"| {_fmt(em['returned_as_current'][s])} "
                f"| {_fmt(em['masked_present'].get(s))} "
                f"| {_fmt(ez['returned_as_current'][s])} "
                f"| {_fmt(ez['masked_present'].get(s))} "
                f"| {_fmt_q(rel['E1_at_end']['em_cg'].get(rel_key))} "
                f"| {_fmt_q(rel['E1_at_end']['ez_cg'].get(rel_key))} "
                f"| {_fmt_q(rel['E1_at_end']['em_cw'].get(rel_key))} "
                f"| {_fmt_q(rel['E1_at_end']['ez_cw'].get(rel_key))} |")
        lines.append("")
    lines += [
        "### While-pending (pooled at-risk; geometry note carried from the "
        "released scorer)", "",
        f"> {res['geometry_notes']['vacuous_windows']}", "",
        "| reading | em_pt returned | em_pt masked | ez_pt returned | "
        "ez_pt masked | released em_cg | released ez_cg | released em_cw | "
        "released ez_cw |",
        "|---|---|---|---|---|---|---|---|---|"]
    for reading in READINGS:
        em = ep["em_pt"][reading]["E1_contamination"]["while_pending"]
        ez = ep["ez_pt"][reading]["E1_contamination"]["while_pending"]
        lines.append(
            f"| {reading} "
            f"| {_fmt(em['returned_as_current']['pooled_at_risk'])} "
            f"| {_fmt(em['masked_present']['pooled_at_risk'])} "
            f"| {_fmt(ez['returned_as_current']['pooled_at_risk'])} "
            f"| {_fmt(ez['masked_present']['pooled_at_risk'])} "
            f"| {_fmt_q(rel['E1_while_pending']['em_cg']['pooled_at_risk'])} "
            f"| {_fmt_q(rel['E1_while_pending']['ez_cg']['pooled_at_risk'])} "
            f"| {_fmt_q(rel['E1_while_pending']['em_cw']['pooled_at_risk'])} "
            f"| {_fmt_q(rel['E1_while_pending']['ez_cw']['pooled_at_risk'])} |")
    lines += ["", "## Came-true end-state correctness (the dual-reading "
              "discriminator; E3 came_true probe-correct, ITT)", "",
              "| stratum | em_pt with-flip | em_pt strict | ez_pt with-flip "
              "| ez_pt strict | released em_cg | released ez_cg | released "
              "em_cw | released ez_cw |",
              "|---|---|---|---|---|---|---|---|---|"]
    ct_rows = (["probe_correct_pooled"]
               + [f"{s}|came_true" for s in sd1.STRENGTHS])
    for s in ct_rows:
        def _pt_cell(arm, reading):
            c = ep[arm][reading]["came_true_end_state"]
            if s == "probe_correct_pooled":
                return c["probe_correct_pooled"]
            return c["probe_correct_by_strength"].get(s)
        def _rel_cell(cell):
            if s == "probe_correct_pooled":
                x = sum(rel["E3_task_probes"][cell][f"{st}|came_true"]["x"]
                        for st in sd1.STRENGTHS)
                n = sum(rel["E3_task_probes"][cell][f"{st}|came_true"]["n"]
                        for st in sd1.STRENGTHS)
                return {"x": x, "n": n, "rate": x / n if n else None}
            return rel["E3_task_probes"][cell].get(s)
        lines.append(
            f"| {s.replace('|', chr(92) + '|')} "
            f"| {_fmt(_pt_cell('em_pt', 'with_flip'))} "
            f"| {_fmt(_pt_cell('em_pt', 'strict'))} "
            f"| {_fmt(_pt_cell('ez_pt', 'with_flip'))} "
            f"| {_fmt(_pt_cell('ez_pt', 'strict'))} "
            f"| {_fmt_q(_rel_cell('em_cg'))} | {_fmt_q(_rel_cell('ez_cg'))} "
            f"| {_fmt_q(_rel_cell('em_cw'))} | {_fmt_q(_rel_cell('ez_cw'))} |")
    lines += [
        "",
        "Supplementary (pt-only): end-state eq new_value pooled — "
        + "; ".join(
            f"{arm} {reading} "
            f"{_fmt(ep[arm][reading]['came_true_end_state']['eq_new_value_pooled'])}"
            for arm in ARMS for reading in READINGS) + ".",
        "",
        "Masked-at-end on came_true (the no-route-1 failure signature):",
        "",
        "| arm | with-flip masked | strict masked |", "|---|---|---|"]
    for arm in ARMS:
        lines.append(
            f"| {arm} "
            f"| {_fmt(ep[arm]['with_flip']['came_true_end_state']['masked_at_end_pooled'])} "
            f"| {_fmt(ep[arm]['strict']['came_true_end_state']['masked_at_end_pooled'])} |")
    lines += ["", "## E3 — Task probes (ITT; deterministic read-rule "
              "lookup per reading)", "",
              "| stratum | em_pt with-flip | em_pt strict | ez_pt with-flip "
              "| ez_pt strict | released em_cg | released ez_cg | released "
              "em_cw | released ez_cw |",
              "|---|---|---|---|---|---|---|---|---|"]
    probe_rows = (["pooled_all", "pooled_non_control", "controls",
                   "unanswered_ITT"]
                  + [f"{s}|{o}" for s in sd1.STRENGTHS for o in sd1.OUTCOMES])
    for s in probe_rows:
        lines.append(
            f"| {s.replace('|', chr(92) + '|')} "
            f"| {_fmt(ep['em_pt']['with_flip']['E3_task_probes'].get(s))} "
            f"| {_fmt(ep['em_pt']['strict']['E3_task_probes'].get(s))} "
            f"| {_fmt(ep['ez_pt']['with_flip']['E3_task_probes'].get(s))} "
            f"| {_fmt(ep['ez_pt']['strict']['E3_task_probes'].get(s))} "
            f"| {_fmt_q(rel['E3_task_probes']['em_cg'].get(s))} "
            f"| {_fmt_q(rel['E3_task_probes']['ez_cg'].get(s))} "
            f"| {_fmt_q(rel['E3_task_probes']['em_cw'].get(s))} "
            f"| {_fmt_q(rel['E3_task_probes']['ez_cw'].get(s))} |")
    lines += ["", "## E4 — Controls (already-happened changes; n=15)", "",
              "| endpoint | em_pt with-flip | em_pt strict | ez_pt "
              "with-flip | ez_pt strict | released em_cg | released ez_cg | "
              "released em_cw | released ez_cw |",
              "|---|---|---|---|---|---|---|---|---|"]
    for name, rel_block in (("committed_immediately",
                             rel["E4_committed_immediately"]),
                            ("control_plan_typed_write (over-holding "
                             "analog; released = over_holding, gated only)",
                             rel["E4_over_holding"]),
                            ("control_masked_at_end", None)):
        key = name.split(" ")[0]
        cells = []
        for arm in ARMS:
            for reading in READINGS:
                cells.append(_fmt(ep[arm][reading]["E4_controls"][key]))
        rel_cells = []
        for c in ("em_cg", "ez_cg", "em_cw", "ez_cw"):
            rel_cells.append(_fmt_q(rel_block.get(c)) if rel_block else "—")
        lines.append(f"| {name} | " + " | ".join(cells + rel_cells) + " |")
    lines += ["", "## Masked-residue rate (plan entries in the final store)",
              "",
              "Every plan-typed commit ends as exactly one of: flipped "
              "(IC-4-confirmed — the baseline's only resolution route), "
              "current & masked (the read rule is hiding it at conversation "
              "end — the masked residue), or superseded (a later write "
              "closed its interval). never-confirmed = 1 - flipped/total "
              "(under the STRICT reading this is 1.000 by construction — "
              "no resolution mechanism exists).", "",
              "| arm | scope | plan commits | flipped (IC-4) | current & "
              "masked | superseded by later write | never-confirmed "
              "(with-flip) | masked-current residue |",
              "|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        for scope in ("target", "store_wide"):
            r = res["masked_residue"][arm][scope]
            uf = (f"{r['unresolved_with_flip_rate']:.3f}"
                  if r["unresolved_with_flip_rate"] is not None else "—")
            cm = (f"{r['current_masked_rate']:.3f}"
                  if r["current_masked_rate"] is not None else "—")
            lines.append(
                f"| {arm} | {scope} | {r['plan_commits']} | {r['flipped']} "
                f"| {r['current_masked']} | {r['superseded']} | {uf} | {cm} |")
    lines += [
        "",
        "Per-conversation masked-at-end (non-control): "
        + "; ".join(
            f"{arm} {reading} "
            f"{_fmt(ep[arm][reading]['masked_at_end_non_control'])}"
            for arm in ARMS for reading in READINGS) + ".",
        "",
        "## Dual-reading verdict (held-out substrate)", "",
        f"- Operationalization: {v['operationalization']}", "",
        f"- **With-flip parity: "
        f"{'HOLDS' if v['with_flip_parity']['holds'] else 'DOES NOT HOLD'}**"]
    for arm in ARMS:
        w = v["with_flip_parity"][arm]
        lines.append(
            f"  - {arm}: came_true probe-correct "
            f"{_fmt(w['came_true_probe_correct']['pt_with_flip'])} vs "
            f"released {GATED_OF[arm]} "
            f"{_fmt_q(w['came_true_probe_correct']['released_gated_pooled'])} "
            f"(within 0.05: {w['came_true_probe_correct']['within_0.05']}); "
            f"at-end contamination "
            f"{_fmt(w['at_end_contamination']['pt_with_flip'])} vs "
            f"{_fmt_q(w['at_end_contamination']['released_gated'])} "
            f"(within 0.05: {w['at_end_contamination']['within_0.05']})")
    lines.append(
        f"- **Strict reading below gated: "
        f"{'HOLDS' if v['strict_reading']['strict_below_gated_holds'] else 'DOES NOT HOLD'}**")
    for arm in ARMS:
        s = v["strict_reading"][arm]
        lines.append(
            f"  - {arm}: strict came_true probe-correct "
            f"{_fmt(s['came_true_probe_correct_strict'])} vs released "
            f"{GATED_OF[arm]} {_fmt_q(s['released_gated_pooled'])} "
            f"(below: {s['below_gated']}); with-flip -> strict collapse "
            f"{_fmt(s['collapse_vs_with_flip']['with_flip'])} -> "
            f"{_fmt(s['collapse_vs_with_flip']['strict'])}")
    lines += ["", "## Run stats", "",
              "| phase | selected | new | skipped | failed | fresh calls |",
              "|---|---|---|---|---|---|"]
    for r_ in res["run_phases"]:
        lines.append(f"| {r_['phase']} | {r_['selected']} | {r_['n_new']} "
                     f"| {r_['skipped']} | {len(r_['failed'])} "
                     f"| {r_['fresh_calls']:,} |")
    adv = res["advisory_budget"]
    jc = res["scoring_judge_calls"]
    bb, ba = res.get("budget_before"), res["budget_after"]
    lines += ["", "## Budget (stage tier1; advisory / hard envelope / "
              "110k run stop)", ""]
    if bb:
        lines.append(f"- before: {json.dumps(bb['spent'])} "
                     f"(total {sum(bb['spent'].values()):,})")
    lines += [
        f"- after:  {json.dumps(ba['spent'])} "
        f"(total {sum(ba['spent'].values()):,} / envelope "
        f"{ba['envelope']:,}; run stop {HARD_STOP_TOTAL:,})",
        f"- stage tier1 spend {adv['spent']:,} vs advisory forecast "
        f"{adv['advisory_forecast']:,} (shared with the concurrent Tier-1 "
        "agents)",
        f"- scoring equivalence-judge calls: {jc['total']} "
        f"({jc['fresh']} fresh)",
        "",
        "Scored once (prereg). Verification (RECOUNT + SEMANTICS) precedes "
        "release — these numbers are UNVERIFIED until then.",
        ""]
    return "\n".join(lines)


async def score(call_cap: int | None, force: bool) -> dict:
    assert llm.JUDGE_MODEL == PINNED_SYSTEM_MODEL
    if RESULTS_PATH.exists() and not force:
        raise SystemExit(
            f"{RESULTS_PATH} already exists — scored ONCE per prereg; pass "
            "--force only for a logged infrastructure-failure rerun")
    scorer = rule5.EquivalenceScorer(cap=JUDGE_CALL_CAP)
    guard = tier1_guard(local_cap=call_cap)
    guard.install()
    try:
        res = await compute_scores(scorer)
    finally:
        guard.uninstall()
    res["score_fresh_calls"] = guard.fresh
    TIER1_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    REPORT_PATH.write_text(render_report(res))
    print(f"wrote {RESULTS_PATH}\nwrote {REPORT_PATH} "
          f"({res['n_conversations_scored']}/147 conversations; "
          f"{guard.fresh} fresh judge calls)")
    return res


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    rn = sub.add_parser("run")
    rn.add_argument("--ids", nargs="*", default=None)
    rn.add_argument("--shard", default=None, help="K/N over sorted ids")
    rn.add_argument("--call-cap", type=int, default=None)
    sc = sub.add_parser("score")
    sc.add_argument("--call-cap", type=int, default=None)
    sc.add_argument("--force", action="store_true")
    sub.add_parser("status")
    args = ap.parse_args()

    if args.cmd == "status":
        n = len(list(CASES_DIR.glob("*.json"))) if CASES_DIR.exists() else 0
        print(json.dumps({**advisory_status(), "case_records": n,
                          "expected": 147}, indent=1))
        return
    if args.cmd == "score":
        asyncio.run(score(args.call_cap, args.force))
        return
    summary = asyncio.run(run(args.ids, args.shard, args.call_cap))
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
