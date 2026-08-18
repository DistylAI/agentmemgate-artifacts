"""run_v2.py — the staged v2 execution driver (protocol_spec_v2.md §10.2 steps 3-10).

Orchestrates the preregistered v2 experiment over the frozen artifacts. Every
stage is idempotent (all LLM calls go through llm.call_llm and are cached on
byte content) and checkpointed: a completed stage writes
``results/v2/checkpoints/<stage>.done`` with its counts and reruns skip
completed work (``--force`` re-runs; cached calls make reruns cheap).

Stages (CLI subcommands, in execution order):

  forecast       FM-20 gate (§10.2 step 7): recompute the projected call
                 budget from REALIZED artifacts (manifest rows, substrate turn
                 counts, the natural selection, checker horizons) and compare
                 row-by-row to the §7 table. Applies the preregistered de-scope
                 rules when a capped row overruns (row 7 checker cap 18,000 ->
                 seeded 7-of-10 LoCoMo conversation subsample, seed 20260702;
                 row 10 replay cap 10,000 -> injected top-up, §2.3). Writes
                 results/v2/forecast.json. NON-ZERO EXIT if the forecast
                 exceeds the 130,000-call hard cap.
  extract        Both extraction families over all v2 substrates — the single
                 shared extraction pathway (§1.2 item 1-2): LME perturbed
                 streams (67 KU + 60 aux, non-T3 manifest rows spliced) +
                 the natural-selection bought questions' s_cleaned streams
                 (§2.3 stored-value replay); LoCoMo natural + perturbed
                 streams via the dyadic single-pass driver (ONE call per turn,
                 both speaker stores; prompts extractor_dyadic.txt /
                 extractor_zep_dyadic.txt). Emits per-family extractor-tuples
                 files (label.load_extractor_tuples row schema) under
                 results/v2/extraction/ — cells replay from this llm cache.
  cells          The 4 cells (E-M/E-Z x C-W/C-G) over both benchmarks via
                 arms.run_cells. LME jobs use the v2 gold schemas (aux
                 substrates carry no gold key); LoCoMo jobs use per-speaker
                 stores + route_store by entity and the dyadic replay seam
                 (gated_store `extracted=`). Manifest probe rows pass
                 expected_subject through (FM-16). Gated LoCoMo cells also
                 emit the NAT-RES in-stream trace (natres.LOCOMO_TRACE_CONTRACT).
  label-natural  The selected natural items (results/v2/natural_selection.json)
                 through the v1 §2.2 labeling pipeline: LME items with
                 stored values from the s_cleaned extraction cache (§2.3 one
                 cached pathway); LoCoMo items dyadically (rubric + dyadic
                 addendum) with stored values from the natural-pass cache;
                 the 70 NAT-RES pair rows STORELESS (LME half via
                 natres.label_pairs). Escalations queue blind (R2-B-3).
  probes         All probe families against every cell's timelines (§4, no
                 LLM answering layer): P-KEY (manifest probe points — the
                 cells' probe_answers), P-NATQA (98 pre-mapped QA rows),
                 P-MISATTR (frozen 100-item cat-5 sample,
                 abstention-by-absence), P-TIME (13 key-mapped LoCoMo items +
                 the 6 pinned LME old-value KU questions, as-of lookups), plus
                 the expected_subject store check on third-party probe rows.
  natres         pilot/natres.py over the 35 pairs: the STORELESS LME pass
                 (both families) + score (LoCoMo half read from the in-stream
                 trace the cells stage wrote).
  metrics        metrics_v2 over everything: H-INT (+ REP-LOCO) and the §5.1
                 rules computable from the driver's artifacts, per benchmark,
                 natural/injected split (V2-P9). Equivalence scoring uses the
                 normalize_value fast path with the blind judge on unresolved
                 pairs (§7 row 12).

Smoke/partial runs: ``--qids``, ``--turns a:b`` (LoCoMo chronological turn
slice; extraction context always comes from the full conversation) and
``--call-budget N`` (hard abort after N fresh — non-cached — LLM calls).
Partial runs NEVER write a stage checkpoint.

LLM calls go through llm.call_llm only. T3/SYN-RES variant replays are the
pilot test3 flow and are out of this driver's scope (arms excludes T3 rows);
the forecast still budgets their §7 rows.

Run from pilot/:  .venv/bin/python run_v2.py <stage> [options]
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import random
import sys
from datetime import timedelta
from functools import cache
from pathlib import Path

import _bootstrap  # noqa: F401  (R1 layout: puts src/ + experiments/ on sys.path)

import arms
import data
import gated_store
import label
import llm
import locomo_data
import locomo_render
import metrics_v2
import natres
import store
import run_predicates as test3
from zepstore import ZepStore

from paths import GOLD_DIR, RESULTS_DIR
V2_DIR = RESULTS_DIR / "v2"
CHECKPOINT_DIR = V2_DIR / "checkpoints"
EXTRACTION_DIR = V2_DIR / "extraction"
PROBES_DIR = V2_DIR / "probes"
LABELS_DIR = V2_DIR / "labels"
FORECAST_PATH = V2_DIR / "forecast.json"
METRICS_PATH = V2_DIR / "metrics.json"
NATURAL_LABELS_PATH = LABELS_DIR / "natural_labels.jsonl"
LME_ESCALATION_QUEUE_PATH = LABELS_DIR / "escalation_queue_v2.jsonl"

MANIFEST_V2_PATH = V2_DIR / "injections" / "manifest.jsonl"
INJECTED_KEYS_V2_PATH = V2_DIR / "injections" / "injected_keys.jsonl"
SELECTION_PATH = V2_DIR / "natural_selection.json"
GATE_ITEMS_PATHS = {b: V2_DIR / "injections" / f"gate_items_{b}.jsonl"
                    for b in ("lme", "locomo")}
LME_SCHEMA_PATH = GOLD_DIR / "key_schema_v2_lme.json"

SEED = 20260702           # v1/v2 analysis seed (de-scope draws, §7 cap rules)
HARD_CAP = 130_000        # §7 total (V2-P8)
CHECKER_CAP = 18_000      # §7 row 7 hard cap (FM-20)
REPLAY_CAP = 10_000       # §7 row 10 hard cap (FM-20)
RESERVE_FRACTION = 0.088  # §6.3 / §7: 8.8% reserve over the subtotal

FAMILIES = arms.FAMILIES                    # ("mem0", "zep")
FAMILY_DYADIC = {"mem0": "E-M", "zep": "E-Z"}  # locomo_render.DYADIC_PROMPTS keys

#: shared-session pairs count as ONE cluster (§2.1); shadow -> designated.
SHARED_SESSION_CLUSTER = {"89941a94": "89941a93", "07741c44": "07741c45"}

#: the 6 pinned asks-old-value KU questions (lme_schema_v2_notes.md
#: "asks-old-value: 6 pinned") — the LME half of P-TIME (§4, V2-F4).
LME_OLD_VALUE_QIDS = ("50635ada", "e66b632c", "9bbe84a2", "0977f2af",
                      "dfde3500", "07741c44")

STAGES = ("forecast", "extract", "cells", "label-natural", "probes",
          "natres", "metrics")


# ---------------------------------------------------------------------------
# §7 preregistered budget table (the comparison reference; calls column)
# ---------------------------------------------------------------------------

S7_TABLE = {
    1: ("extraction E-M (natural + perturbed + variant replays)", 20_233),
    2: ("extraction E-Z (same streams, second family)", 20_233),
    3: ("key mapping E-M", 11_051),
    4: ("key mapping E-Z (x1.55 tuple volume)", 16_051),
    5: ("gate classifier (2 gated cells)", 6_000),
    6: ("predicate generation", 1_500),
    7: ("resolution checker (2 gated cells) [HARD CAP]", 18_000),
    8: ("injection plan + render", 6_000),
    9: ("naturalness-gate judging", 1_500),
    10: ("natural-item stored-value replay [HARD CAP]", 10_000),
    11: ("gold labeling", 1_200),
    12: ("equivalence judging", 3_000),
    13: ("FM-12 reliability reruns", 1_800),
    14: ("dyadic Test-0 calibration", 500),
    15: ("NAT-RES storeless pass, LME half", 1_500),
}
S7_SUBTOTAL = 118_568
S7_RESERVE = 11_432

# §7 pilot-calibrated forecast anchors (formula constants, cited per row).
REPLAY_FACTOR = 2.6              # rows 1-2: replays over natural-pass turns
KEYMAP_EM = (1.05, 2.0)          # row 3: (per natural-pass call, per manifest row)
KEYMAP_EZ = (1.6, 2.5)           # row 4
MAPPED_FRACTION = {"mem0": 0.20, "zep": 0.35}   # row 5
EVAL_ITEMS_OUTSIDE_STREAMS = 600                # row 5: Test-1 eval items
PREDICATE_FRACTION = 0.25                       # row 6: speculative share
T3_TAIL_TURNS = 5                # rows 5/7: variant tail = resolution + 2-turn
                                 # window (MINOR-4a; LoCoMo tails ~3-5 -> 5)
CHECKER_OCCUPANCY = {"locomo": 1.0, "lme": 0.4}  # row 7 (§7 anchors)
RENDER_CALLS_PER_ROW = 4                        # row 8: plan+render
GATE_ROUNDS = 2                                 # row 9: naturalness-gate rounds
GATE_CAL_ITEMS = 150                            # row 9: calibration items
LABEL_VOTES = 3                                 # row 11
LOCOMO_CAL_LABELED = 75                         # row 11: Test-0 items x 3 votes
ESCALATION_ADJUDICATIONS = 90                   # row 11
PROBE_POINTS_PER_INJECTION = 1.1                # row 12
NATURAL_KEY_BOUNDARIES = 6                      # row 12
MISC_PROBE_LOOKUPS = 1_250                      # row 12: P-NATQA/P-TIME/P-MISATTR
JUDGE_UNRESOLVED_FRACTION = 0.4                 # row 12
RERUN_FRACTION = 0.2                            # row 13 (of rows 5 + 12)
T0_CALIBRATION_CALLS = 500                      # row 14 (executed pre-driver)
NATRES_STAGE_CALLS = 3                          # row 15: extract+classify+gen /pair


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def checkpoint_path(stage: str) -> Path:
    return CHECKPOINT_DIR / f"{stage}.done"


def stage_complete(stage: str) -> bool:
    return checkpoint_path(stage).exists()


def write_checkpoint(stage: str, counts: dict) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path(stage).write_text(json.dumps(
        {"stage": stage, "counts": counts}, indent=1, ensure_ascii=False))


def skip_if_complete(stage: str, force: bool) -> bool:
    """True -> the stage already completed and this run should skip."""
    if force or not stage_complete(stage):
        return False
    counts = json.loads(checkpoint_path(stage).read_text()).get("counts", {})
    print(f"[{stage}] checkpoint exists ({checkpoint_path(stage)}); "
          f"skipping completed work (counts: {json.dumps(counts)}) — use --force to re-run")
    return True


# ---------------------------------------------------------------------------
# Fresh-call budget guard (smoke discipline: cache hits are free)
# ---------------------------------------------------------------------------

class CallBudget:
    """Wraps llm.call_llm; counts FRESH calls (cache misses) and aborts past
    the budget. Cache hits never count — idempotent reruns are free.

    ``on_fresh`` (optional): callback invoked with this guard after every
    fresh-call increment, BEFORE the budget check — the v2.2 shared-envelope
    guard (forecast_v22.SharedCallBudget) uses it to persist per-stage spend
    to results/v2.2/budget.json; an aborted over-budget call is therefore
    recorded (conservative accounting). The callback may itself raise to
    abort (e.g. combined-envelope exhaustion)."""

    def __init__(self, budget: int | None, on_fresh=None):
        self.budget = budget
        self.fresh = 0
        self._orig = llm.call_llm
        self._on_fresh = on_fresh

    def install(self) -> None:
        orig = self._orig

        async def guarded(system_prompt, user_prompt, *, trace_tag, **kwargs):
            model = kwargs.get("model", llm.MODEL)
            effort = kwargs.get("reasoning_effort", llm.REASONING_EFFORT)
            cached = (kwargs.get("use_cache", True)
                      and (llm.CACHE_DIR / (llm._cache_key(
                          model, effort, system_prompt, user_prompt) + ".json")).exists())
            if not cached:
                self.fresh += 1
                if self._on_fresh is not None:
                    self._on_fresh(self)
                if self.budget is not None and self.fresh > self.budget:
                    raise RuntimeError(
                        f"call budget exceeded: {self.fresh} fresh LLM calls > "
                        f"budget {self.budget} (trace_tag={trace_tag})")
            return await orig(system_prompt, user_prompt, trace_tag=trace_tag,
                              **kwargs)

        import extract as _extract

        # llm.call_llm covers every `import llm` caller (classify, label,
        # natres, gated_store, inject_v2); modules that bound the NAME at
        # import time need their own attribute patched.
        llm.call_llm = guarded
        store.call_llm = guarded
        locomo_render.call_llm = guarded
        _extract.call_llm = guarded

    def uninstall(self) -> None:
        import extract as _extract

        llm.call_llm = self._orig
        store.call_llm = self._orig
        locomo_render.call_llm = self._orig
        _extract.call_llm = self._orig


# ---------------------------------------------------------------------------
# Frozen-artifact loaders (cached, read-only)
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open() if line.strip()]


@cache
def load_manifest_v2() -> list[dict]:
    rows = _read_jsonl(MANIFEST_V2_PATH)
    if not rows:
        raise SystemExit(f"missing v2 manifest: {MANIFEST_V2_PATH}")
    return rows


@cache
def load_lme_schema_v2() -> dict:
    return json.loads(LME_SCHEMA_PATH.read_text())


@cache
def lme_ku_entries() -> dict[str, dict]:
    """question_id -> KU gold entry (67 rows: 44 pilot-inherited + 23 v2)."""
    return {e["question_id"]: e for e in load_lme_schema_v2()["substrates"]}


@cache
def lme_aux_qids() -> list[str]:
    return [r["question_id"] for r in sorted(
        load_lme_schema_v2()["aux_injection_substrates"],
        key=lambda r: r["selection_rank"])]


def lme_v2_qids() -> list[str]:
    """All 127 v2 LME substrate qids (67 KU + 60 aux), KU first."""
    return list(lme_ku_entries()) + lme_aux_qids()


@cache
def _injected_keys_v2() -> list[dict]:
    return _read_jsonl(INJECTED_KEYS_V2_PATH)


def v2_aux_keys(benchmark: str, question_id: str) -> list[store.CanonicalKey]:
    """Registered v2 aux keys for one substrate (injected_keys.jsonl, FM-4)."""
    return [
        store.CanonicalKey(entity=r["entity"], attribute=r["attribute"],
                           description=r.get("description", ""),
                           probe_question=r["probe_question"])
        for r in _injected_keys_v2()
        if r["benchmark"] == benchmark and r["question_id"] == question_id
    ]


def locomo_gold_keys(sample_id: str) -> list[store.CanonicalKey]:
    """The conversation's frozen gold update keys as mapper candidates
    (§1.2 item 3: gold + registered aux + LoCoMo schema keys)."""
    keys = []
    for row in locomo_data.update_keys():
        if row["sample_id"] != sample_id:
            continue
        keys.append(store.CanonicalKey(
            entity=row["entity"], attribute=row["attribute"],
            description=("canonical single-valued attribute "
                         f"(class: {row['attribute_class']})"),
            probe_question=row["probe_question"]))
    return keys


def locomo_candidates(sample_id: str) -> list[store.CanonicalKey]:
    return locomo_gold_keys(sample_id) + v2_aux_keys("locomo", sample_id)


def load_selection() -> dict:
    if not SELECTION_PATH.exists():
        raise SystemExit(f"missing natural selection: {SELECTION_PATH} "
                         "(run select_natural.py)")
    return json.loads(SELECTION_PATH.read_text())


def manifest_rows_for(benchmark: str, question_id: str,
                      *, include_t3: bool = False) -> list[dict]:
    return [r for r in load_manifest_v2()
            if r["benchmark"] == benchmark and r["question_id"] == question_id
            and (include_t3 or r["inj_type"] not in arms.T3_TYPES)]


# ---------------------------------------------------------------------------
# forecast — FM-20 (§7 recomputation + de-scope rules)
# ---------------------------------------------------------------------------

def count_user_turns(substrate) -> int:
    return sum(1 for s in substrate.sessions for t in s if t["role"] == "user")


def forecast_inputs() -> dict:
    """All realized quantities the §7 recomputation consumes (no LLM)."""
    manifest = load_manifest_v2()
    by_bench: dict[str, list[dict]] = {"lme": [], "locomo": []}
    for r in manifest:
        by_bench[r["benchmark"]].append(r)

    lme_turns = sum(count_user_turns(data.get_substrate(qid))
                    for qid in lme_v2_qids())
    convs = locomo_data.conversation_ids()
    loco_turns_by_conv = {
        sid: len(locomo_data.dialog_turns(locomo_data.get_conversation(sid)))
        for sid in convs}

    selection = load_selection()
    schema = json.loads(natres.LME_SCHEMA_PATH.read_text())
    horizon_sum = sum(e["checker_horizon_user_turns"]
                      for e in schema["natural_pair_keys"]["entries"])

    def n_type(bench, types):
        return sum(1 for r in by_bench[bench] if r["inj_type"] in types)

    gate_items = {b: len(_read_jsonl(p)) for b, p in GATE_ITEMS_PATHS.items()}
    n_update_keys = len(locomo_data.update_keys())

    return {
        "lme_natural_turns": lme_turns,                      # §7: 1,900
        "locomo_turns_by_conversation": loco_turns_by_conv,  # §7: 5,882 total
        "locomo_natural_turns": sum(loco_turns_by_conv.values()),
        "n_manifest_rows": len(manifest),
        "n_manifest_rows_by_benchmark": {b: len(rows) for b, rows in by_bench.items()},
        "n_spec_rows": n_type("lme", arms.SPEC_TYPES) + n_type("locomo", arms.SPEC_TYPES),
        "n_corr_rows": n_type("lme", arms.CORR_TYPES) + n_type("locomo", arms.CORR_TYPES),
        "n_t3_variant_rows": {b: n_type(b, {"t3_variant"}) for b in by_bench},
        "natural_replay_charge": selection["replay_budget"]["total_charge"],  # 9,714
        "natural_replay_within_cap": selection["replay_budget"]["within_cap"],
        "n_selected_items": selection["n_selected_items"],
        "shortfalls_injected_topup": selection["shortfalls_injected_topup"],
        "natres_horizon_sum": horizon_sum,                   # 632
        "n_lme_pairs": 17,
        "n_gate_items": gate_items,
        "n_natural_keys": len(lme_ku_entries()) + n_update_keys,  # row 12 anchor
    }


def descope_locomo_subsample(sample_ids: list[str], seed: int = SEED) -> list[str]:
    """FM-20 row-7 de-scope: the seeded 7-of-10 conversation subsample.
    Deterministic: sorted ids, random.Random(seed).sample, re-sorted."""
    rng = random.Random(seed)
    return sorted(rng.sample(sorted(sample_ids), 7))


def compute_forecast(inputs: dict) -> dict:
    """Pure §7 recomputation (unit-tested arithmetic). Applies the FM-20
    de-scope rules on the capped rows before totalling."""
    lme_t = inputs["lme_natural_turns"]
    loco_by_conv = dict(inputs["locomo_turns_by_conversation"])
    loco_t = sum(loco_by_conv.values())
    n_rows = inputs["n_manifest_rows"]
    t3v = inputs["n_t3_variant_rows"]
    natural_total = lme_t + loco_t

    descope = {"checker_subsample": None, "natural_replay_topup": None}

    # --- row 7 first (its cap decides the LoCoMo replay scope) -------------
    def checker_calls(loco_turns: int) -> int:
        per_family = (
            CHECKER_OCCUPANCY["locomo"] * (loco_turns + t3v["locomo"] * T3_TAIL_TURNS)
            + CHECKER_OCCUPANCY["lme"] * (lme_t + t3v["lme"] * T3_TAIL_TURNS))
        return round(2 * per_family)

    row7 = checker_calls(loco_t)
    if row7 > CHECKER_CAP:
        subsample = descope_locomo_subsample(list(loco_by_conv))
        loco_scoped = sum(loco_by_conv[s] for s in subsample)
        descope["checker_subsample"] = {
            "rule": "checker forecast > 18,000 -> gated LoCoMo replay drops to "
                    "the seeded 7-of-10 conversation subsample (§7 cap "
                    "discipline / FM-20; seed 20260702); log in the changelog "
                    "BEFORE execution",
            "seed": SEED,
            "conversations": subsample,
            "forecast_before": row7,
            "forecast_after": checker_calls(loco_scoped),
        }
        row7 = checker_calls(loco_scoped)

    # --- row 10 (cap enforced at selection; topup is the §2.3 rule) --------
    row10 = inputs["natural_replay_charge"]
    if row10 > REPLAY_CAP:
        descope["natural_replay_topup"] = {
            "rule": "greedy fills the cap before floors -> injected top-up "
                    "per v1 §4.2 (§2.3), logged per class",
            "shortfalls": inputs["shortfalls_injected_topup"],
        }
        row10 = REPLAY_CAP  # realized replay calls never exceed the cap

    row5_events = natural_total + (t3v["lme"] + t3v["locomo"]) * T3_TAIL_TURNS
    row5 = round(MAPPED_FRACTION["mem0"] * row5_events
                 + MAPPED_FRACTION["zep"] * row5_events
                 + EVAL_ITEMS_OUTSIDE_STREAMS)
    row12 = round(JUDGE_UNRESOLVED_FRACTION * (
        (inputs["n_spec_rows"] + inputs["n_corr_rows"]) * 4 * PROBE_POINTS_PER_INJECTION
        + inputs["n_natural_keys"] * 4 * NATURAL_KEY_BOUNDARIES
        + MISC_PROBE_LOOKUPS))

    rows = {
        1: round(natural_total * REPLAY_FACTOR),
        2: round(natural_total * REPLAY_FACTOR),
        3: round(KEYMAP_EM[0] * natural_total + KEYMAP_EM[1] * n_rows),
        4: round(KEYMAP_EZ[0] * natural_total + KEYMAP_EZ[1] * n_rows),
        5: row5,
        6: round(PREDICATE_FRACTION * row5),
        7: row7,
        8: RENDER_CALLS_PER_ROW * n_rows,
        9: (GATE_ROUNDS * sum(inputs["n_gate_items"].values()) + GATE_CAL_ITEMS),
        10: row10,
        11: (LABEL_VOTES * inputs["n_selected_items"]
             + LABEL_VOTES * LOCOMO_CAL_LABELED + ESCALATION_ADJUDICATIONS),
        12: row12,
        13: round(RERUN_FRACTION * (row5 + row12)),
        14: T0_CALIBRATION_CALLS,
        15: (2 * inputs["n_lme_pairs"] * NATRES_STAGE_CALLS
             + 2 * inputs["natres_horizon_sum"]),
    }

    table = []
    for i in sorted(rows):
        name, budget = S7_TABLE[i]
        table.append({"row": i, "component": name, "budget_s7": budget,
                      "forecast": rows[i], "delta": rows[i] - budget})
    subtotal = sum(rows.values())
    reserve = round(subtotal * RESERVE_FRACTION)
    total = subtotal + reserve
    return {
        "inputs": {k: v for k, v in inputs.items()
                   if k != "locomo_turns_by_conversation"},
        "rows": table,
        "subtotal_forecast": subtotal,
        "subtotal_s7": S7_SUBTOTAL,
        "reserve": {"fraction": RESERVE_FRACTION, "forecast": reserve,
                    "s7": S7_RESERVE},
        "total_forecast": total,
        "hard_cap": HARD_CAP,
        "headroom": HARD_CAP - total,
        "exceeds_hard_cap": total > HARD_CAP,
        "caps": {
            "checker_row7": {"forecast": rows[7], "cap": CHECKER_CAP,
                             "descope_applied": descope["checker_subsample"] is not None},
            "natural_replay_row10": {"charge": rows[10], "cap": REPLAY_CAP,
                                     "descope_applied": descope["natural_replay_topup"] is not None},
        },
        "descope": descope,
        "verdict": ("OVER HARD CAP — do not run (FM-20)" if total > HARD_CAP
                    else "within hard cap"),
    }


def cmd_forecast(args) -> None:
    stage = "forecast"
    if skip_if_complete(stage, args.force):
        return
    fc = compute_forecast(forecast_inputs())
    V2_DIR.mkdir(parents=True, exist_ok=True)
    FORECAST_PATH.write_text(json.dumps(fc, indent=1, ensure_ascii=False))
    over = [r for r in fc["rows"] if r["delta"] > 0]
    print(f"[forecast] total {fc['total_forecast']:,} vs hard cap "
          f"{fc['hard_cap']:,} (headroom {fc['headroom']:,}); checker row 7 "
          f"{fc['caps']['checker_row7']['forecast']:,}/{CHECKER_CAP:,}; replay "
          f"row 10 {fc['caps']['natural_replay_row10']['charge']:,}/{REPLAY_CAP:,}; "
          f"{len(over)} row(s) over their §7 line "
          f"({', '.join(str(r['row']) for r in over) or 'none'}); "
          f"verdict: {fc['verdict']} -> {FORECAST_PATH}")
    if fc["exceeds_hard_cap"]:
        sys.exit(1)
    write_checkpoint(stage, {
        "total_forecast": fc["total_forecast"], "hard_cap": fc["hard_cap"],
        "rows_over_budget": [r["row"] for r in over],
        "descope_applied": {k: v is not None for k, v in fc["descope"].items()},
    })


# ---------------------------------------------------------------------------
# LoCoMo dyadic single-pass extraction (shared by extract + cells)
# ---------------------------------------------------------------------------

def perturbed_conversation(sample_id: str):
    """The conversation's main perturbed world (all non-T3 rows spliced) —
    the exact world the calibration attribution smoke extracted from."""
    from inject_v2 import build_perturbed_conversation

    conv = locomo_data.get_conversation(sample_id)
    rows = manifest_rows_for("locomo", sample_id)
    return build_perturbed_conversation(conv, rows) if rows else conv


def locomo_stream_rows(conv) -> list[dict]:
    """The gate-loop stream for a LoCoMo conversation: one row per turn (all
    turns are human — V2-P3 preserved), content = '{Speaker}: {text}' with
    photo notes (locomo_render.render_turn_line), tagged with the ORIGINAL
    session index and session date."""
    rows = []
    for si, ti, turn in locomo_data.dialog_turns(conv):
        rows.append({
            "role": "user",
            "content": locomo_render.render_turn_line(turn),
            "session_idx": si,
            "turn_idx": ti,
            "date": conv.dates[si],
            "speaker": turn["speaker"],
            "dia_id": turn["dia_id"],
        })
    return rows


def parse_turn_range(spec: str | None, n: int) -> tuple[int, int]:
    """'a:b' -> (a, b) clamped to [0, n] (chronological turn positions)."""
    if not spec:
        return (0, n)
    a, _, b = spec.partition(":")
    lo = max(0, int(a or 0))
    hi = min(n, int(b)) if b else n
    if lo >= hi:
        raise SystemExit(f"empty turn range {spec!r} over {n} turns")
    return (lo, hi)


async def dyadic_extract_stream(conv, family: str, *, stream_id: str,
                                turn_range: tuple[int, int] | None = None
                                ) -> tuple[list[dict], list[str]]:
    """ONE dyadic extractor call per turn (V2-P2), both speaker stores from
    the single pass. Returns (turn_rows, call_keys) in gated_store's phase-1
    shape — the `extracted=` seam input. Context is ALWAYS computed from the
    full conversation (dyadic_turn_job), so a turn slice changes only which
    turns are replayed, never any call's bytes."""
    dyfam = FAMILY_DYADIC[family]
    system_prompt = locomo_render.DYADIC_PROMPTS[dyfam]
    positions = locomo_data.dialog_turns(conv)
    lo, hi = turn_range if turn_range is not None else (0, len(positions))
    call_keys, coros, meta = [], [], []
    for pos in range(lo, hi):
        si, ti, turn = positions[pos]
        job = locomo_render.dyadic_turn_job(conv, si, ti, family=dyfam)
        user_prompt = locomo_render.render_dyadic_prompt(
            job["prior_turns"], job["current_turn"], job["current_date"])
        call_keys.append(test3._cache_key(system_prompt, user_prompt))
        coros.append(locomo_render.extract_dyadic_turn(
            job["prior_turns"], job["current_turn"],
            f"extract-dyadic-{dyfam}-{stream_id}-{turn['dia_id']}",
            family=dyfam, current_date=job["current_date"]))
        meta.append((pos, si, ti, turn))
    tuples_per_turn = await asyncio.gather(*coros)
    turn_rows = []
    for (pos, si, ti, turn), tuples in zip(meta, tuples_per_turn):
        turn_rows.append({
            "pos": pos,
            "user_pos": pos,   # every LoCoMo turn is a human turn
            "session_idx": si,
            "turn_idx": ti,
            "date": conv.dates[si],
            "turn_text": locomo_render.render_turn_line(turn),
            "dia_id": turn["dia_id"],
            "speaker": turn["speaker"],
            "tuples": list(tuples),
        })
    return turn_rows, call_keys


# ---------------------------------------------------------------------------
# extract — the shared per-family extraction pathway (§10.2 steps 4 + 7)
# ---------------------------------------------------------------------------

def _tuples_out(tuples) -> list[dict]:
    return [dataclasses.asdict(t) if dataclasses.is_dataclass(t) else dict(t)
            for t in tuples]


def extraction_path(benchmark: str, family: str, stream: str) -> Path:
    return EXTRACTION_DIR / f"{benchmark}_{family}_{stream}.jsonl"


async def extract_lme_stream(qid: str, family: str, stream: str) -> list[dict]:
    """One LME substrate stream: 'perturbed' (v2 substrate, non-T3 rows
    spliced — what the cells replay) or 's_cleaned' (natural-selection bought
    question, §2.3 stored-value pathway). Rows follow the
    label.load_extractor_tuples schema."""
    import inject
    import eval_classifier as test1_eval

    if stream == "perturbed":
        substrate = data.get_substrate(qid)
        rows = manifest_rows_for("lme", qid)
        if rows:
            substrate = inject.build_perturbed_substrate(substrate, rows)
        source_file = "oracle"
    elif stream == "s_cleaned":
        substrate = test1_eval.s_cleaned_substrate(qid)
        source_file = "s_cleaned"
    else:
        raise ValueError(f"unknown LME stream {stream!r}")
    turn_rows, call_keys = await gated_store.extract_stream_tuples(
        gated_store.substrate_stream(substrate), stream_id=qid, family=family)
    return [
        {"benchmark": "lme", "family": family, "stream": stream,
         "source_file": source_file, "question_id": qid,
         "session_idx": r["session_idx"], "turn_idx": r["turn_idx"],
         "date": r["date"], "turn_text": r["turn_text"],
         "tuples": _tuples_out(r["tuples"]), "cache_key": k}
        for r, k in zip(turn_rows, call_keys)
    ]


async def extract_locomo_stream(sample_id: str, family: str, stream: str,
                                turn_range=None) -> list[dict]:
    """One LoCoMo conversation stream: 'natural' (unperturbed — the store-state
    stream for natural items, §2.3) or 'perturbed' (main world — what the
    cells replay). ONE dyadic call per turn (V2-P2)."""
    conv = (perturbed_conversation(sample_id) if stream == "perturbed"
            else locomo_data.get_conversation(sample_id))
    turn_rows, call_keys = await dyadic_extract_stream(
        conv, family, stream_id=f"{sample_id}-{stream}", turn_range=turn_range)
    return [
        {"benchmark": "locomo", "family": family, "stream": stream,
         "source_file": "locomo", "question_id": sample_id,
         "session_idx": r["session_idx"], "turn_idx": r["turn_idx"],
         "dia_id": r["dia_id"], "speaker": r["speaker"], "date": r["date"],
         "turn_text": r["turn_text"], "tuples": _tuples_out(r["tuples"]),
         "cache_key": k}
        for r, k in zip(turn_rows, call_keys)
    ]


def selection_bought_qids() -> list[str]:
    return [b["question_id"] for b in load_selection()["bought_questions"]]


def _append_rows(path: Path, rows: list[dict]) -> None:
    """Idempotent-by-key merge: later rows for the same turn supersede earlier
    ones on read (label.load_extractor_tuples), so append is safe; a fresh
    full run rewrites the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _stream_filter(args, valid: tuple[str, ...]) -> list[str]:
    if not getattr(args, "streams", None):
        return list(valid)
    return [s for s in args.streams.split(",") if s in valid]


def cmd_extract(args) -> None:
    stage = "extract"
    partial = bool(args.qids or args.turns or args.streams)
    if args.turns and args.benchmarks != "locomo":
        raise SystemExit("--turns slices LoCoMo turn streams only; "
                         "pass --benchmarks locomo")
    if not partial and skip_if_complete(stage, args.force):
        return
    families = _families(args)
    benchmarks = args.benchmarks.split(",")
    qids_filter = set(args.qids.split(",")) if args.qids else None
    counts: dict[str, int] = {}

    async def run() -> None:
        if "lme" in benchmarks:
            perturbed = [q for q in lme_v2_qids()
                         if qids_filter is None or q in qids_filter]
            s_cleaned = [q for q in selection_bought_qids()
                         if qids_filter is None or q in qids_filter]
            for family in families:
                for stream in _stream_filter(args, ("perturbed", "s_cleaned")):
                    qids = perturbed if stream == "perturbed" else s_cleaned
                    out_rows = []
                    for qid in qids:
                        out_rows.extend(await extract_lme_stream(qid, family, stream))
                    if out_rows:
                        _append_rows(extraction_path("lme", family, stream), out_rows)
                    counts[f"lme_{family}_{stream}"] = len(out_rows)
        if "locomo" in benchmarks:
            convs = [c for c in locomo_data.conversation_ids()
                     if qids_filter is None or c in qids_filter]
            for family in families:
                for stream in _stream_filter(args, ("natural", "perturbed")):
                    out_rows = []
                    for sid in convs:
                        conv = locomo_data.get_conversation(sid)
                        rng = parse_turn_range(
                            args.turns, len(locomo_data.dialog_turns(conv)))
                        out_rows.extend(await extract_locomo_stream(
                            sid, family, stream, turn_range=rng))
                    if out_rows:
                        _append_rows(extraction_path("locomo", family, stream),
                                     out_rows)
                    counts[f"locomo_{family}_{stream}"] = len(out_rows)

    _run_with_budget(args, run())
    total = sum(counts.values())
    print(f"[extract] {total} extractor calls recorded across "
          f"{len(counts)} (benchmark, family, stream) files -> {EXTRACTION_DIR} "
          f"({json.dumps(counts)})" + (" [partial run: no checkpoint]" if partial else ""))
    if not partial:
        write_checkpoint(stage, counts)


# ---------------------------------------------------------------------------
# cells — job builders (v2 schemas) + the 2x2 runner (§10.2 step 8)
# ---------------------------------------------------------------------------

def lme_cell_jobs(qids: list[str] | None = None) -> list[arms.CellJob]:
    """CellJob rows for the 127 v2 LME substrates: KU rows carry their gold
    schema entry; aux substrates carry none (candidates = registered aux keys
    only); manifest rows are the substrate's non-T3 rows; probe rows default
    to the PROBED_TYPES rows (arms)."""
    ku = lme_ku_entries()
    jobs = []
    for qid in (qids if qids is not None else lme_v2_qids()):
        entry = ku.get(qid)
        if entry is None and qid not in lme_aux_qids():
            raise ValueError(f"{qid}: not a v2 LME substrate")
        jobs.append(arms.CellJob(
            question_id=qid,
            substrate=data.get_substrate(qid),
            key_schema_entry=entry,
            manifest_rows=manifest_rows_for("lme", qid),
            aux_keys=v2_aux_keys("lme", qid),
        ))
    return jobs


def locomo_probe_rows(sample_id: str) -> list[dict]:
    """Manifest probe rows for one conversation. `entity` stays the focal
    store-routing key; expected_subject rides along (FM-16) and answer_probes
    carries it into the probe answers."""
    return [dict(r) for r in manifest_rows_for("locomo", sample_id)
            if r["inj_type"] in arms.PROBED_TYPES]


def make_locomo_replay(turns_spec: str | None = None):
    """The dyadic single-pass replay callable for CellJob.replay: perturbed
    main world, dyadic extraction (shared llm cache across policies), gate
    loop via gated_store with per-speaker stores + route_store by entity
    (spec §2.2 item 3: a tuple about P routes to P's store; third parties
    get their own entity-named stores)."""

    async def replay(job: arms.CellJob, family: str, policy: str):
        conv = perturbed_conversation(job.question_id)
        stream = locomo_stream_rows(conv)
        rng = parse_turn_range(turns_spec, len(stream))
        extracted = await dyadic_extract_stream(
            conv, family, stream_id=f"{job.question_id}-perturbed",
            turn_range=rng)
        stores = {s.lower(): ZepStore() for s in conv.speakers}
        run = (gated_store.run_gated_replay if policy == "gated"
               else gated_store.run_ungated_replay)
        return await run(
            stream, job.aux_keys, stream_id=job.question_id, family=family,
            stores=stores, route_store=lambda key: key[0],
            session_order=list(conv.session_order), extracted=extracted)

    return replay


def locomo_cell_jobs(sample_ids: list[str] | None = None,
                     turns_spec: str | None = None) -> list[arms.CellJob]:
    """CellJob rows for the 10 LoCoMo conversations. `aux_keys` doubles as the
    full candidate list (gold update keys + registered aux); the dyadic
    replay seam does the actual work."""
    replay = make_locomo_replay(turns_spec)
    jobs = []
    for sid in (sample_ids if sample_ids is not None
                else locomo_data.conversation_ids()):
        jobs.append(arms.CellJob(
            question_id=sid,
            substrate=locomo_data.get_conversation(sid),
            key_schema_entry=None,
            manifest_rows=manifest_rows_for("locomo", sid),
            probe_rows=locomo_probe_rows(sid),
            aux_keys=locomo_candidates(sid),
            replay=replay,
        ))
    return jobs


# --- NAT-RES in-stream trace (natres.LOCOMO_TRACE_CONTRACT) ----------------

def _snapshot_user_pos(timeline_json: dict) -> dict[tuple[int, int], int]:
    return {(s["session_idx"], s["turn_idx"]): s["user_pos"]
            for s in timeline_json["snapshots"]}


def locomo_instream_rows(timeline_json: dict, pairs: list[dict],
                         family_alias: str) -> list[dict]:
    """NAT-RES rows for one gated LoCoMo timeline (§4: the 18 LoCoMo pairs are
    staged and checked IN-STREAM; scoring reads the resolution turn + the
    2-turn window from the run trace)."""
    sample_id = timeline_json["stream_id"]
    conv = locomo_data.get_conversation(sample_id)
    pos_of = _snapshot_user_pos(timeline_json)
    wanted = {"promote": "promoted", "invalidate": "invalidated"}
    out = []
    for pair in pairs:
        if pair["corpus"] != "locomo" or pair["source_question_id"] != sample_id:
            continue
        s_si, s_ti, _ = locomo_data.turn_by_dia_id(conv, pair["stage"]["dia_id"])
        r_si, r_ti, _ = locomo_data.turn_by_dia_id(conv, pair["resolution"]["dia_id"])
        res_pos = pos_of.get((r_si, r_ti))
        speaker_store = pair["speaker"].strip().lower()
        staged_rows = [
            s for s in timeline_json["staged_log"]
            if s["staged_at"]["session_idx"] == s_si
            and s["staged_at"]["turn_idx"] == s_ti
            and s["store_id"] == speaker_store]
        target = wanted[pair["gold_action"]]

        def _hit(window: int) -> bool:
            if res_pos is None:
                return False
            return any(
                s["status"] == target and s["resolved_at"] is not None
                and res_pos <= s["resolved_at"]["user_pos"] <= res_pos + window
                for s in staged_rows)

        n_false = sum(
            1 for s in staged_rows
            if s["status"] not in ("pending",) and s["resolved_at"] is not None
            and not (s["status"] == target and res_pos is not None
                     and res_pos <= s["resolved_at"]["user_pos"] <= res_pos + natres.SENSITIVITY_WINDOW))
        out.append({
            "pair_id": pair["pair_id"],
            "family": family_alias,
            "staged": bool(staged_rows),
            "hit_primary": _hit(0),
            "hit_sensitivity": _hit(natres.SENSITIVITY_WINDOW),
            "n_false_fires": n_false,
            "notes": f"in-stream trace from {sample_id} gated timeline "
                     f"({len(staged_rows)} staged item(s) at the stage anchor)",
        })
    return out


def write_locomo_instream_trace(out_dir: Path, families: tuple[str, ...]) -> int:
    """Scan the gated LoCoMo timelines under out_dir and (re)write the NAT-RES
    in-stream trace. Only writes when every conversation with a pair has a
    timeline (partial runs never produce a partial trace)."""
    pairs = [p for p in natres.load_pairs() if p["corpus"] == "locomo"]
    needed = sorted({p["source_question_id"] for p in pairs})
    rows: list[dict] = []
    family_alias = {"mem0": "em", "zep": "ez"}
    for family in families:
        cell_dir = out_dir / arms.cell_name(family, "gated") / "timelines"
        paths = {sid: cell_dir / f"{sid}.json" for sid in needed}
        if not all(p.exists() for p in paths.values()):
            return 0
        for sid, path in sorted(paths.items()):
            rows.extend(locomo_instream_rows(
                json.loads(path.read_text()), pairs, family_alias[family]))
    natres.LOCOMO_TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with natres.LOCOMO_TRACE_PATH.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def cmd_cells(args) -> None:
    stage = "cells"
    partial = bool(args.qids or args.turns)
    if args.turns and args.benchmarks != "locomo":
        raise SystemExit("--turns slices LoCoMo turn streams only; "
                         "pass --benchmarks locomo")
    if not partial and skip_if_complete(stage, args.force):
        return
    families = _families(args)
    policies = tuple(p for p in args.policies.split(",") if p)
    benchmarks = args.benchmarks.split(",")
    qids_filter = set(args.qids.split(",")) if args.qids else None
    out_dir = Path(args.out) if args.out else arms.V2_DIR

    jobs: list[arms.CellJob] = []
    if "lme" in benchmarks:
        qids = [q for q in lme_v2_qids()
                if qids_filter is None or q in qids_filter]
        jobs.extend(lme_cell_jobs(qids))
    if "locomo" in benchmarks:
        sids = [c for c in locomo_data.conversation_ids()
                if qids_filter is None or c in qids_filter]
        loco_jobs = locomo_cell_jobs(sids, turns_spec=args.turns)
        if partial:  # sliced replays cannot honor full-stream probe points
            for j in loco_jobs:
                j.probe_rows = []
        jobs.extend(loco_jobs)
    if not jobs:
        raise SystemExit("no jobs selected")

    summary = _run_with_budget(args, arms.run_cells(
        jobs, families=families, policies=policies, out_dir=out_dir))

    n_trace = 0
    if "locomo" in benchmarks and "gated" in policies and not partial:
        n_trace = write_locomo_instream_trace(out_dir, families)
    counts = {cell: info["n_substrates"] for cell, info in summary["cells"].items()}
    identity = all(v["identical"] for v in summary["extraction_identity"])
    print(f"[cells] {len(jobs)} substrate job(s) x {len(summary['cells'])} cell(s) "
          f"-> {out_dir}; extraction byte-identity across policies: "
          f"{'OK' if identity else 'VIOLATED'}; NAT-RES in-stream trace rows: "
          f"{n_trace}" + (" [partial run: no checkpoint]" if partial else ""))
    if not partial:
        write_checkpoint(stage, {"jobs": len(jobs), "cells": counts,
                                 "extraction_identity_ok": identity,
                                 "natres_instream_rows": n_trace})


# ---------------------------------------------------------------------------
# label-natural — the v1 §2.2 pipeline over the selected items (§10.2 step 5)
# ---------------------------------------------------------------------------

def _pool_by_anchor() -> dict[tuple, dict]:
    import select_natural
    return {(r["source"], r["question_id"], r["session_idx"], r["turn_idx"]): r
            for r in select_natural.load_pool()}


def selection_items_with_pool() -> list[dict]:
    """The selected items joined back to their full pool rows (text/about/...)."""
    pool = _pool_by_anchor()
    items = []
    for it in load_selection()["selected_items"]:
        key = (it["source"], it["question_id"], it["session_idx"], it["turn_idx"])
        row = pool.get(key)
        if row is None:
            raise SystemExit(f"selected item not found in pool: {key}")
        items.append({**row, **it})
    return items


async def _label_locomo_item(cand: dict, tuples_at: dict, stored_streams: bool) -> dict:
    """One selected LoCoMo row through the dyadic pipeline. Pair rows are
    STORELESS (stored_value=None, rule 1.5d no-stored-value branch, §2.3);
    non-pair rows read the stored value from the natural-pass E-M dyadic
    tuples (the store-state stream, §2.3)."""
    import calibrate_locomo

    item = calibrate_locomo.build_natural_item(cand)
    conv = locomo_data.get_conversation(cand["question_id"])
    item_turn = (item["session_idx"], item["turn_idx"])
    turn_tuples = tuples_at.get(item_turn, [])
    matched, score = label.match_tuple_to_span(turn_tuples, cand["text"])
    if matched is not None:
        # M5 natural branch: the span-matched tuple's own key is the target.
        item["entity"] = matched["entity"]
        item["attribute"] = matched["attribute"]
        item["asserted_value"] = matched["value"]
        if not cand.get("pair_id") and stored_streams:
            turn_order = [(si, ti) for si, ti, _ in locomo_data.dialog_turns(conv)]
            item["stored_value"] = label.stored_value_replay(
                turn_order, item_turn, tuples_at,
                matched["entity"], matched["attribute"])
    result = await calibrate_locomo.label_item_dyadic(item)
    return {"item": item, "labeling": result, "span_overlap": score,
            "extraction_miss": matched is None}


def cmd_label_natural(args) -> None:
    stage = "label-natural"
    if skip_if_complete(stage, args.force):
        return
    import calibrate_locomo

    items = selection_items_with_pool()
    lme_pair = [i for i in items if i.get("pair_id") and i["source"] != "locomo"]
    loco_rows = [i for i in items if i["source"] == "locomo"]
    lme_rows = [i for i in items
                if i["source"] != "locomo" and not i.get("pair_id")]

    # LoCoMo natural-pass tuples (E-M — labels are arm-independent, anchored
    # on the frozen extractor, natres docstring) from the extract-stage file.
    loco_tuples_path = extraction_path("locomo", "mem0", "natural")
    loco_index = label.load_extractor_tuples(loco_tuples_path)
    have_loco_stream = bool(loco_index)
    lme_tuples_path = extraction_path("lme", "mem0", "s_cleaned")
    lme_index = label.load_extractor_tuples(lme_tuples_path)

    rows_out: list[dict] = []
    lme_items, lme_results = [], []
    loco_items, loco_results = [], []

    async def run() -> None:
        import eval_classifier as test1_eval

        # --- LME non-pair rows (bought s_cleaned questions, §2.3) ----------
        for cand in lme_rows:
            substrate = test1_eval.s_cleaned_substrate(cand["question_id"])
            item = label.build_natural_item(
                cand, substrate, lme_index, "s_cleaned")
            if item.get("extraction_miss"):
                rows_out.append({"item_id": item["item_id"], "corpus": "lme",
                                 "extraction_miss": True})
                continue
            res = await label.label_item(item)
            rows_out.append({"item_id": item["item_id"], "corpus": "lme",
                             "extraction_miss": False, "labeling": res})
            lme_items.append(item)
            lme_results.append(res)

        # --- LoCoMo rows (dyadic rubric + addendum; pair rows storeless) ---
        for cand in loco_rows:
            tuples_at = label.tuples_for_question(
                loco_index, "locomo", cand["question_id"])
            out = await _label_locomo_item(cand, tuples_at, have_loco_stream)
            rows_out.append({
                "item_id": out["item"]["item_id"], "corpus": "locomo",
                "pair_id": cand.get("pair_id"),
                "storeless_pair_row": bool(cand.get("pair_id")),
                "extraction_miss": out["extraction_miss"],
                "labeling": out["labeling"]})
            loco_items.append(out["item"])
            loco_results.append(out["labeling"])

        # --- the 17 LME pairs (34 rows): STORELESS via natres (§4/V2-F3) ---
        if lme_pair and not args.skip_pairs:
            await natres.label_pairs()

    _run_with_budget(args, run())

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    with NATURAL_LABELS_PATH.open("w") as fh:
        for r in rows_out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_esc_lme = label.append_escalations(
        lme_items, lme_results, queue_path=LME_ESCALATION_QUEUE_PATH)
    n_esc_loco = calibrate_locomo.append_escalations_locomo(
        loco_items, loco_results)
    counts = {"labeled": len(rows_out), "lme": len(lme_rows),
              "locomo": len(loco_rows), "lme_pair_rows": len(lme_pair),
              "escalations_queued_lme": n_esc_lme,
              "escalations_queued_locomo": n_esc_loco}
    print(f"[label-natural] {len(rows_out)} items labeled "
          f"(+{len(lme_pair)} LME pair rows via natres.label_pairs); "
          f"escalations queued: lme {n_esc_lme} -> {LME_ESCALATION_QUEUE_PATH.name}, "
          f"locomo {n_esc_loco} (blind convention R2-B-3)")
    write_checkpoint(stage, counts)


# ---------------------------------------------------------------------------
# probes — deterministic probe families over the cell timelines (§4, V2-P4)
# ---------------------------------------------------------------------------

def timeline_store_mapping(timeline_json: dict) -> dict[tuple, list[dict]]:
    """Merge a timeline's per-entity stores into one {(entity, attribute):
    [entry, ...]} mapping (keys carry their entity, so merging is lossless) —
    the plain-mapping duck-type of the locomo_render probe engine."""
    merged: dict[tuple, list[dict]] = {}
    for zstore in timeline_json["stores"].values():
        for key_name, entries in zstore["history"].items():
            entity, _, attribute = key_name.partition("/")
            merged[(entity, attribute)] = list(entries)
    return merged


class _MappingStore:
    """Minimal store facade over the merged mapping (get() protocol of
    locomo_render.history_entries)."""

    def __init__(self, mapping: dict):
        self._m = mapping

    def get(self, key):
        return self._m.get(tuple(key))


_TIMELINE_CACHE: dict[str, dict] = {}


def load_cell_timeline(out_dir: Path, cell: str, stream_id: str) -> dict | None:
    """Read (and memoize) one cell timeline JSON. The metrics stage scans
    timelines once per manifest row, so the cache matters; probes/metrics run
    after the cells stage finished writing them."""
    path = out_dir / cell / "timelines" / f"{stream_id}.json"
    key = str(path)
    if key not in _TIMELINE_CACHE:
        if not path.exists():
            return None
        _TIMELINE_CACHE[key] = json.loads(path.read_text())
    return _TIMELINE_CACHE[key]


def lme_ptime_probe(timeline_json: dict, entry: dict) -> dict:
    """The LME half of P-TIME (§4): as-of lookup of the KU key one minute
    before the new-evidence session — the old value must hold there (the
    history diagnostic: write-through-with-overwrite destroys it)."""
    store_map = _MappingStore(timeline_store_mapping(timeline_json))
    key = (entry["entity"], entry["attribute"])
    new_si = entry["new_evidence"]["session_idx"]
    substrate = data.get_substrate(entry["question_id"])
    as_of = data.parse_haystack_date(substrate.dates[new_si]) - timedelta(minutes=1)
    value = locomo_render.as_of_value(
        locomo_render.history_entries(store_map, key), as_of,
        parse_date=data.parse_haystack_date)
    found = not locomo_render.value_is_absent(value)
    gold = str(entry["old_value"])
    exact = found and store.normalize_value(value) == store.normalize_value(gold)
    return {"question_id": entry["question_id"], "key": list(key),
            "as_of": as_of.isoformat(), "value": value, "found": found,
            "gold_answer": gold, "exact_match": exact,
            "needs_judge": found and not exact}


def expected_subject_checks(out_dir: Path, cell: str) -> list[dict]:
    """FM-16 probe-row check: for probe answers carrying expected_subject, read
    the SUBJECT's store too — correct routing = focal key absent, subject key
    holding the value."""
    path = out_dir / cell / "probe_answers.jsonl"
    checks = []
    for row in _read_jsonl(path):
        subject = row.get("expected_subject")
        if not subject:
            continue
        timeline = load_cell_timeline(out_dir, cell, row["question_id"])
        if timeline is None:
            continue
        mapping = timeline_store_mapping(timeline)
        entity, _, attribute = row["key"].partition("/")
        subj_entries = mapping.get((subject, attribute), [])
        subj_value = locomo_render.current_value(_MappingStore(mapping),
                                                 (subject, attribute))
        checks.append({
            "injection_id": row["injection_id"],
            "question_id": row["question_id"],
            "expected_subject": subject,
            "focal_key": row["key"],
            "focal_absent": not row["present"],
            "subject_value": subj_value,
            "subject_present": not locomo_render.value_is_absent(subj_value),
            "n_subject_entries": len(subj_entries),
        })
    return checks


def cmd_probes(args) -> None:
    stage = "probes"
    if skip_if_complete(stage, args.force):
        return
    out_dir = Path(args.out) if args.out else arms.V2_DIR
    families = _families(args)
    policies = tuple(p for p in args.policies.split(",") if p)
    schema = locomo_data.load_locomo_schema()
    qa_rows = locomo_data.qa_probe_map_rows()
    cat5 = locomo_data.cat5_sample()
    ptime_items = [i for i in locomo_data.time_indexed_probes() if i.get("key_id")]
    ku = lme_ku_entries()
    counts: dict[str, dict] = {}
    PROBES_DIR.mkdir(parents=True, exist_ok=True)

    for family in families:
        for policy in policies:
            cell = arms.cell_name(family, policy)
            cell_counts = {"p_key": 0, "p_natqa": 0, "p_misattr": 0,
                           "p_time": 0, "expected_subject_checks": 0}
            out = {"cell": cell, "p_key": [], "p_natqa": [], "p_misattr": [],
                   "p_time": [], "expected_subject_checks": []}

            # P-KEY: the cells' deterministic manifest-point lookups (B4).
            out["p_key"] = _read_jsonl(out_dir / cell / "probe_answers.jsonl")

            # LoCoMo store views per conversation.
            mappings = {}
            for sid in locomo_data.conversation_ids():
                tl = load_cell_timeline(out_dir, cell, sid)
                if tl is not None:
                    mappings[sid] = _MappingStore(timeline_store_mapping(tl))
            for item in qa_rows:
                sid = item["sample_id"]
                if sid in mappings:
                    out["p_natqa"].append(
                        locomo_render.p_natqa_probe(mappings[sid], item))
            for item in cat5:
                sid = item["sample_id"]
                if sid in mappings:
                    out["p_misattr"].append(
                        locomo_render.p_misattr_probe(mappings[sid], item))
            for item in ptime_items:
                sid = item["sample_id"]
                if sid in mappings:
                    out["p_time"].append(
                        locomo_render.p_time_probe(mappings[sid], item))
            for qid in LME_OLD_VALUE_QIDS:
                tl = load_cell_timeline(out_dir, cell, qid)
                if tl is not None and qid in ku:
                    out["p_time"].append(lme_ptime_probe(tl, ku[qid]))

            out["expected_subject_checks"] = expected_subject_checks(out_dir, cell)
            for k in cell_counts:
                cell_counts[k] = len(out[k])
            counts[cell] = cell_counts
            (PROBES_DIR / f"{cell}.json").write_text(
                json.dumps(out, indent=1, ensure_ascii=False))

    total = {k: sum(c[k] for c in counts.values())
             for k in next(iter(counts.values()))} if counts else {}
    print(f"[probes] {len(counts)} cell(s) -> {PROBES_DIR} "
          f"(totals: {json.dumps(total)})")
    write_checkpoint(stage, counts)


# ---------------------------------------------------------------------------
# natres — the 35-pair STORELESS endpoint (§4/V2-F3, §7 row 15)
# ---------------------------------------------------------------------------

def cmd_natres(args) -> None:
    stage = "natres"
    if skip_if_complete(stage, args.force):
        return
    _run_with_budget(args, natres.run_lme(("em", "ez")))
    results = natres.score()
    counts = {fam: {"primary_hits": block["rule_primary_resolution_turn"]["x"],
                    "n_run": block["n_run"]}
              for fam, block in results["per_family"].items()}
    print(f"[natres] 35-pair NAT-RES scored (LME storeless x2 families; "
          f"LoCoMo half from the in-stream trace) -> {natres.RESULTS_PATH}")
    write_checkpoint(stage, counts)


# ---------------------------------------------------------------------------
# metrics — metrics_v2 over everything (§5.1; §8 natural-first)
# ---------------------------------------------------------------------------

async def _equivalent(attribute: str, value_a, value_b, *, trace_tag: str) -> bool:
    """normalize_value fast path; blind equivalence judge on unresolved pairs
    (§7 row 12). The `(unknown)` sentinel is non-equivalent to any concrete
    value (§1.2 item 10)."""
    if value_a is None or value_b is None:
        return False
    if locomo_render.value_is_sentinel(value_a) or locomo_render.value_is_sentinel(value_b):
        return False
    if store.normalize_value(str(value_a)) == store.normalize_value(str(value_b)):
        return True
    return await store.values_equivalent(attribute, str(value_a), str(value_b),
                                         trace_tag=trace_tag)


def _cluster_of(row: dict) -> str:
    qid = row["question_id"]
    return SHARED_SESSION_CLUSTER.get(qid, qid)


def load_probe_answers(out_dir: Path) -> dict[str, dict[str, dict]]:
    """{metrics_cell: {injection_id: probe answer row}} for the four cells."""
    table: dict[str, dict[str, dict]] = {}
    for (family, policy), mcell in arms.METRICS_CELL.items():
        rows = _read_jsonl(out_dir / arms.cell_name(family, policy)
                           / "probe_answers.jsonl")
        table[mcell] = {r["injection_id"]: r for r in rows}
    return table


async def build_y_table(answers: dict[str, dict[str, dict]],
                        inj_types: set[str], benchmark: str) -> list[dict]:
    """One row per injection present in ALL FOUR cells: the paired §1.1
    outcomes Y_i^cell = [probed value equivalent to the injected value]."""
    ids = None
    for mcell in metrics_v2.CELLS:
        cell_ids = {i for i, r in answers[mcell].items()
                    if r["inj_type"] in inj_types
                    and _bench_of(r) == benchmark}
        ids = cell_ids if ids is None else ids & cell_ids
    rows = []
    for inj_id in sorted(ids or ()):
        y = {}
        base = answers["em_cw"][inj_id]
        for mcell in metrics_v2.CELLS:
            r = answers[mcell][inj_id]
            attr = r["key"].partition("/")[2]
            y[mcell] = int(r["present"] and await _equivalent(
                attr, r["value"], r["v_target"],
                trace_tag=f"metrics-eq-{inj_id}-{mcell}"))
        row = {"injection_id": inj_id, "question_id": base["question_id"],
               "cluster": _cluster_of(base), "provenance": "injected", "y": y}
        if benchmark == "locomo":
            row["conversation"] = base["question_id"]
            row["speaker"] = base["key"].partition("/")[0]
        rows.append(row)
    return rows


@cache
def _manifest_by_id() -> dict[str, dict]:
    return {r["injection_id"]: r for r in load_manifest_v2()}


def _bench_of(answer_row: dict) -> str:
    if answer_row.get("benchmark"):
        return answer_row["benchmark"]
    return _manifest_by_id()[answer_row["injection_id"]]["benchmark"]


def _old_value_of(row: dict) -> str | None:
    m = _manifest_by_id()[row["injection_id"]]
    if m.get("old_value") is not None:
        return m["old_value"]
    paired = m.get("paired_with")
    if paired and paired in _manifest_by_id():
        return _manifest_by_id()[paired].get("value")
    return None


def canonicalization_check(out_dir: Path, family: str, row: dict) -> bool | None:
    """C1-i: the injected tuple reached the target key at the placement turn
    (read from the family's UNGATED timeline gate_log — extraction+mapping
    are policy-shared, so the ungated cell is the policy-free witness)."""
    tl = load_cell_timeline(out_dir, arms.cell_name(family, "ungated"),
                            row["question_id"])
    if tl is None:
        return None
    p = row["placement"]
    key_name = f"{row['entity']}/{row['attribute']}"
    return any(g["key"] == key_name
               and g["session_idx"] == p["session_idx"]
               and g["turn_idx"] == p["turn_idx"]
               for g in tl["gate_log"])


def spec_licensing_check(out_dir: Path, family: str, row: dict) -> bool | None:
    """SPEC-LIC: in the gated cell, was the injected speculative tuple
    classified into a COMMIT class at its own turn (licensing)?"""
    tl = load_cell_timeline(out_dir, arms.cell_name(family, "gated"),
                            row["question_id"])
    if tl is None:
        return None
    p = row["placement"]
    key_name = f"{row['entity']}/{row['attribute']}"
    hits = [g for g in tl["gate_log"]
            if g["key"] == key_name and g["session_idx"] == p["session_idx"]
            and g["turn_idx"] == p["turn_idx"]]
    if not hits:
        return None  # not canonicalized -> not in the licensing denominator
    return any(g["class"] in gated_store.CLASSES_COMMIT for g in hits)


def natcens_false_promotes(out_dir: Path, family: str) -> dict:
    """NAT-CENS: false-promote count over the 4 frozen censored LoCoMo arcs
    (a staged item at the censored dia_id that PROMOTED is a false promote)."""
    x, n, scanned = 0, 0, 0
    for row in locomo_data.censored_staged_hypotheses():
        n += 1
        tl = load_cell_timeline(out_dir, arms.cell_name(family, "gated"),
                                row["sample_id"])
        if tl is None:
            continue
        scanned += 1
        conv = locomo_data.get_conversation(row["sample_id"])
        si, ti, _ = locomo_data.turn_by_dia_id(conv, row["dia_id"])
        x += any(s["staged_at"]["session_idx"] == si
                 and s["staged_at"]["turn_idx"] == ti
                 and s["status"] == "promoted"
                 for s in tl["staged_log"])
    return {"x": x, "n": n, "n_scanned": scanned}


def cmd_metrics(args) -> None:
    stage = "metrics"
    if skip_if_complete(stage, args.force):
        return
    out_dir = Path(args.out) if args.out else arms.V2_DIR
    answers = load_probe_answers(out_dir)
    manifest = load_manifest_v2()
    results: dict = {"seed": metrics_v2.SEED, "benchmarks": {}, "pending": []}

    async def run() -> None:
        for bench in ("lme", "locomo"):
            block: dict = {}
            spec_rows = await build_y_table(answers, arms.SPEC_TYPES, bench)
            corr_rows = await build_y_table(answers, arms.CORR_TYPES, bench)

            # MANIP + CANON (C1-i) from the manifest x ungated timelines.
            manip_by_family = {}
            for family in FAMILIES:
                per_class: dict[str, dict] = {}
                for row in manifest:
                    if (row["benchmark"] != bench
                            or row["inj_type"] in arms.T3_TYPES):
                        continue
                    can = canonicalization_check(out_dir, family, row)
                    if can is None:
                        continue
                    cls = row["construction_label"]
                    b = per_class.setdefault(cls, {"x": 0, "n": 0})
                    b["n"] += 1
                    b["x"] += int(can)
                spec_c = per_class.get("speculative", {"x": 0, "n": 0})
                manip_by_family[family] = metrics_v2.manip_rule(
                    spec_c["x"], spec_c["n"])
                block.setdefault("CANON", {})[family] = {
                    cls: metrics_v2.canon_report(
                        c["x"], c["n"], arm=family, item_class=cls,
                        benchmark=bench)
                    for cls, c in sorted(per_class.items())}
            block["MANIP"] = {"ez": manip_by_family["zep"],
                              "em_reference": manip_by_family["mem0"]}

            # H-INT / REP-LOCO + gate contrasts + the E-Z/C-W cell rule.
            if spec_rows:
                if bench == "lme":
                    block["H-INT"] = metrics_v2.h_int(spec_rows)
                else:
                    block["REP-LOCO"] = metrics_v2.rep_loco(spec_rows)
                for fam_key, fam in (("ez", "zep"), ("em", "mem0")):
                    grows = [{"cluster": r["cluster"],
                              "y_cw": r["y"][f"{fam_key}_cw"],
                              "y_cg": r["y"][f"{fam_key}_cg"],
                              "provenance": r["provenance"]} for r in spec_rows]
                    block[f"GATE-{fam_key.upper()}"] = metrics_v2.gate_contrast(
                        grows, family=fam_key)
                if bench == "lme":
                    block["CELL-EZCW"] = metrics_v2.cell_ezcw_rule(
                        [r["y"]["ez_cw"] for r in spec_rows],
                        [r["cluster"] for r in spec_rows])

            # Corrections: recall (gated cells) + paired residue.
            if corr_rows:
                for fam_key in ("em", "ez"):
                    x = sum(r["y"][f"{fam_key}_cg"] for r in corr_rows)
                    block.setdefault("CORR-RECALL", {})[fam_key] = \
                        metrics_v2.corr_recall_rule(x, len(corr_rows))
                for fam_key in ("em", "ez"):
                    rrows = []
                    for r in corr_rows:
                        old = _old_value_of(
                            answers["em_cw"][r["injection_id"]])
                        stale = {}
                        for pol in ("cw", "cg"):
                            a = answers[f"{fam_key}_{pol}"][r["injection_id"]]
                            attr = a["key"].partition("/")[2]
                            stale[pol] = int(a["present"] and await _equivalent(
                                attr, a["value"], old,
                                trace_tag=f"metrics-stale-{r['injection_id']}-{fam_key}_{pol}"))
                        rrows.append({"cluster": r["cluster"],
                                      "stale_gated": stale["cg"],
                                      "stale_ungated": stale["cw"]})
                    block.setdefault("CORR-RESIDUE", {})[fam_key] = \
                        metrics_v2.corr_residue_rule(rrows)

            # SPEC-LIC per extraction arm (gated cells).
            for family in FAMILIES:
                x = n = 0
                for row in manifest:
                    if (row["benchmark"] != bench
                            or row["inj_type"] not in arms.SPEC_TYPES):
                        continue
                    lic = spec_licensing_check(out_dir, family, row)
                    if lic is None:
                        continue
                    n += 1
                    x += int(lic)
                if n:
                    block.setdefault("SPEC-LIC", {})[family] = \
                        metrics_v2.spec_lic_rule(x, n)

            if bench == "locomo":
                for family in FAMILIES:
                    nc = natcens_false_promotes(out_dir, family)
                    if nc["n_scanned"]:
                        block.setdefault("NAT-CENS", {})[family] = \
                            metrics_v2.natcens_report(nc["x"], nc["n"])
            results["benchmarks"][bench] = block

        # Probe-family endpoints (per cell, from the probes stage).
        for (family, policy), mcell in arms.METRICS_CELL.items():
            path = PROBES_DIR / f"{arms.cell_name(family, policy)}.json"
            if not path.exists():
                continue
            probes = json.loads(path.read_text())
            cellblock = results.setdefault("probe_endpoints", {}).setdefault(mcell, {})
            swap = [r for r in probes["p_misattr"]
                    if r["trap_type"] == "speaker_swap"]
            ffd = [r for r in probes["p_misattr"]
                   if r["trap_type"] == "focal_fact_distortion"]
            if swap:
                cellblock["MISATTR"] = metrics_v2.misattr_report(
                    sum(r["correct"] for r in swap), len(swap),
                    stratum="speaker_swap", cell=mcell)
            if ffd:
                cellblock["MISATTR_exploratory"] = metrics_v2.misattr_report(
                    sum(r["correct"] for r in ffd), len(ffd),
                    stratum="focal_fact_distortion", cell=mcell)
            ptime = probes["p_time"]
            if ptime:
                x = 0
                for r in ptime:
                    ok = r["exact_match"]
                    if not ok and r.get("needs_judge"):
                        attr = (r["key"][1] if isinstance(r["key"], (list, tuple))
                                else r["key"])
                        ok = await _equivalent(
                            attr, r["value"], r["gold_answer"],
                            trace_tag=f"metrics-ptime-{mcell}-{r.get('qa_index', r.get('question_id'))}")
                    x += int(ok)
                cellblock["TIME-IDX"] = metrics_v2.time_idx_report(
                    x, len(ptime), cell=mcell)
            natqa = [r for r in probes["p_natqa"] if r["map_type"] not in
                     ("history_event_time", "history_event_value",
                      "history_old_value", "derived_history")]
            if natqa:
                x = 0
                for r in natqa:
                    ok = r["exact_match"]
                    if not ok and r.get("needs_judge"):
                        ok = await _equivalent(
                            r["key"][1] if isinstance(r["key"], (list, tuple)) else "",
                            r["value"], r["gold_answer"],
                            trace_tag=f"metrics-natqa-{mcell}-{r['qa_index']}")
                    x += int(ok)
                cellblock["QA-NAT"] = metrics_v2.qa_nat_report(
                    x, len(natqa), cell=mcell)

        # NAT-RES passthrough (natres stage owns the computation).
        if natres.RESULTS_PATH.exists():
            nat = json.loads(natres.RESULTS_PATH.read_text())
            results["NAT-RES"] = {fam: block["rule_primary_resolution_turn"]
                                  for fam, block in nat["per_family"].items()}
        else:
            results["pending"].append(
                {"endpoint": "NAT-RES", "reason": "natres stage not run"})
        for endpoint, reason in (
                ("SYN-RES", "T3 variant replays are the pilot test3 flow, "
                            "outside this driver (§7 rows budgeted)"),
                ("TTC", "staleness aggregation over genuine natural updates "
                        "lands with the analysis pass"),
                ("dAUC", "evidence-vs-confidence machinery is the v1 test1 "
                         "flow, per extraction arm (C2)")):
            results["pending"].append({"endpoint": endpoint, "reason": reason})

    _run_with_budget(args, run())
    METRICS_PATH.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    summary_bits = []
    for bench, block in results["benchmarks"].items():
        head = block.get("H-INT") or block.get("REP-LOCO")
        if head:
            summary_bits.append(
                f"{bench}: {head['id']} point={head['point']:.3f} "
                f"CI=[{head['ci'][0]:.3f},{head['ci'][1]:.3f}] pass={head['pass']}")
        else:
            summary_bits.append(f"{bench}: no paired spec outcomes yet")
    print(f"[metrics] {'; '.join(summary_bits)}; "
          f"{len(results['pending'])} endpoint(s) pending -> {METRICS_PATH}")
    write_checkpoint(stage, {
        "benchmarks": {b: sorted(k for k in blk)
                       for b, blk in results["benchmarks"].items()},
        "pending": [p["endpoint"] for p in results["pending"]]})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _families(args) -> tuple[str, ...]:
    fams = tuple(f for f in args.families.split(",") if f)
    for f in fams:
        assert f in FAMILIES, f"unknown family {f!r}"
    return fams


def _run_with_budget(args, coro):
    """Run a stage coroutine under the optional fresh-call budget guard."""
    budget = getattr(args, "call_budget", None)
    guard = CallBudget(budget)
    guard.install()
    try:
        result = asyncio.run(coro)
    finally:
        guard.uninstall()
        if budget is not None:
            print(f"[budget] fresh LLM calls this run: {guard.fresh} "
                  f"(budget {budget}; cache hits are free)")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="v2 staged execution driver (protocol_spec_v2 §10.2)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add(name, **kw):
        p = sub.add_parser(name, **kw)
        p.add_argument("--force", action="store_true",
                       help="re-run even if the stage checkpoint exists")
        return p

    add("forecast", help="FM-20 §7 budget recomputation gate")

    def add_runstage(name, **kw):
        p = add(name, **kw)
        p.add_argument("--families", default=",".join(FAMILIES))
        p.add_argument("--call-budget", type=int, default=None,
                       help="abort after N fresh (non-cached) LLM calls")
        return p

    p = add_runstage("extract", help="shared per-family extraction pathway")
    p.add_argument("--benchmarks", default="lme,locomo")
    p.add_argument("--qids", default=None,
                   help="comma-separated substrate/conversation ids (partial run)")
    p.add_argument("--turns", default=None,
                   help="LoCoMo chronological turn slice a:b (partial run)")
    p.add_argument("--streams", default=None,
                   help="stream subset (lme: perturbed,s_cleaned; "
                        "locomo: natural,perturbed) — partial run")

    p = add_runstage("cells", help="the 4 cells over both benchmarks")
    p.add_argument("--benchmarks", default="lme,locomo")
    p.add_argument("--policies", default=",".join(arms.POLICIES))
    p.add_argument("--qids", default=None)
    p.add_argument("--turns", default=None)
    p.add_argument("--out", default=None)

    p = add_runstage("label-natural",
                     help="v1 §2.2 labeling over the selected natural items")
    p.add_argument("--skip-pairs", action="store_true",
                   help="skip the natres.label_pairs LME pair-row pass")

    p = add("probes", help="deterministic probe families over cell timelines")
    p.add_argument("--families", default=",".join(FAMILIES))
    p.add_argument("--policies", default=",".join(arms.POLICIES))
    p.add_argument("--out", default=None)

    p = add_runstage("natres", help="NAT-RES storeless pass + score")

    p = add_runstage("metrics", help="metrics_v2 over everything (§5.1)")
    p.add_argument("--out", default=None)

    args = parser.parse_args()
    {
        "forecast": cmd_forecast,
        "extract": cmd_extract,
        "cells": cmd_cells,
        "label-natural": cmd_label_natural,
        "probes": cmd_probes,
        "natres": cmd_natres,
        "metrics": cmd_metrics,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
