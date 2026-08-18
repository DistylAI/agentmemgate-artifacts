"""graded_eval.py — Protocol v2.4 M2: difficulty-graded contamination
evaluation (preregistered in protocol/CHANGELOG.md, "Protocol v2.4" M2 entry
+ the 2026-07-08 owner-approved budget amendment: advisory stage budgets,
120k total envelope, revised de-scope order protecting the near-miss
stratum).

Anti-saturation directive: every endpoint is reported as a DIFFICULTY CURVE
(per stratum level, per cell, with cluster CIs per substrate) — no go/no-go
bars anywhere in this module.

STIMULI (seeded, SEED_M2 = 20260708; ~30 per stratum level per benchmark):

  (a) hedging-strength gradient, 5 levels (GRADIENT_LEVELS): strong hedge /
      weak hedge / confident plan / scheduled event / present-tense-implying-
      change. The last three are the frozen taxonomy's known-hard rules
      (a), (b), (d) (protocol_spec.md §1.5). Levels g1-g4 are designed-gold
      `speculative` (a commit of the planted prospective value IS
      contamination); g5 is designed-gold `completed_event` per rule (d)
      (a commit is CORRECT behavior; staging it is over-gating) — the
      designed_gold_* fields ride on every row so the curves stay
      interpretable across the class boundary. Every gradient stimulus is
      paired with an ESTABLISH aside (current value V0) placed strictly
      earlier, so g5's rule-(d) hard case (conflict with stored state) exists
      by construction and no level is confounded with prior-value presence.
  (b) mixed statements (MIXED_LEVEL): hedge + committed fact in ONE sentence
      (switch-frame): the sentence asserts the CURRENT value as unhedged fact
      and floats a hedged prospective switch to a second novel value.
  (c) near-miss resolutions (NEARMISS_LEVELS): a staged hedge (SPECULATION,
      `intending`) + a designed RESOLVING turn at 4 paraphrase-distance
      levels (verbatim-ish / paraphrase / oblique reference / situational
      implication) — the promote-predicate difficulty axis (the M1 cascade
      finding named promote-predicates the weakest stage; the 2026-07-08
      de-scope order protects this stratum).

REUSE (import, never copy — the frozen v2 machinery):
  * geometry: every placement is a FROZEN v2 manifest placement
    (results/v2/injections/manifest.jsonl): gradient/mixed reuse the audited
    (establish, dependent) pair slots of spec_aux/caux rows; near-miss reuses
    the frozen t3 case geometries (staging + variant slots, whose R1-B1
    sensitivity window >= 2 user turns after t* is guaranteed by the frozen
    plan). No new host-slot auditing is introduced.
  * render: the frozen two-call pipeline (injector_plan.txt +
    injector_render.txt, byte-untouched) with payload-level adaptation only
    (the inject_v2 §3.3 precedent): stratum notes, LoCoMo register/author
    notes, round-2 naturalness machinery (ROUND2_INTEGRATION_NOTE, tell
    blacklist, style exemplars, inject._render_ok) + M2-specific
    deterministic construction checks (_m2_render_ok) that enforce the level
    definitions (hedge-marker sets, one-sentence mixed, paraphrase-distance
    checks). Construction labels are BY DESIGN (stratum table), never
    post-hoc.
  * keys: fresh aux keys from inject_v2.AUX_POOL_V2 with FM-4 value-free
    probe questions registered at `stimuli` time — before any value exists;
    `freeze` lints that no probe question contains any sampled value.
  * run path: all four cells (em/ez x ungated/gated = em_cw/em_cg/ez_cw/
    ez_cg) via the frozen per-turn seams: gated_store.extract_turn_family /
    locomo_render dyadic single-pass (extraction CACHE-AWARE: byte-identical
    calls across policies and across the staging/variant worlds replay from
    the llm.py cache), store._map_tuple_key (frozen mapper, mode OFF — the
    M1 stem wrapper is deliberately NOT installed; mapping loss per level is
    part of the difficulty signal and is reported as a diagnostic),
    classify.classify_item + gated_store.ARGMAX_ROUTE routing,
    gated_store._generate_predicates, and the frozen checker over the pinned
    resolution tail [t*, t*+2] (test3 fire semantics).

BOUNDED-WORLD DEVIATION (disclosed, syn_res/NAT-RES precedent): each
stimulus runs in its own clean-copy world and only the injected turns (+ the
near-miss resolution tail) are ingested — intermediate natural turns are not
replayed (they cannot touch the fresh target keys by value-novelty
construction; checker false-fires on natural turns are NOT an M2 endpoint).

ENDPOINTS per stratum level per cell (score -> m2_results.json +
m2_report.md, curves only):
  * contamination: final stored value for the target key EQUIVALENT to the
    planted prospective/changed value (EquivalenceScorer: exact-normalized
    fast path, frozen judge fallback). Mixed additionally reports
    fact-retention (final == the in-sentence committed fact).
  * staging (gated cells): the target tuple staged at the stimulus turn.
  * promotion (near-miss): gated = valid promote fire at t* (primary) /
    within [t*, t*+2] (sensitivity, test3 semantics) + end-state store
    equivalence; ungated = end-state store equivalence after the tail
    (the write-through cell's own resolution tracking). ITT denominators
    (never-staged cases count as failures), conditional-on-staged reported
    beside.
  * diagnostics: target-tuple extraction hit + target-key mapping rate per
    level per family (the M1 mapping-loss axis), never a bar.
Cluster CIs: stats.cluster_bootstrap_rate clustered per substrate
(driver._cluster_of: LME shared-session pairs collapse; LoCoMo sample_id).

REPAIR-COVERAGE ARM (preregistered 2026-07-09, protocol/CHANGELOG.md
"REPAIR-COVERAGE preregistration" — the binding spec; frozen BEFORE any run
call): **em_cg_fixed** = the cautious (em) family + gated policy with the M1
stem-wrapper mapper (store.make_tuple_key_mapper over
router_sweep.PLAN_FRAME_ATTRIBUTE_PATTERNS via mapper_fix.make_mapper, mode
'plan_stem_wrapper') installed as a LOCAL mapper in place of the frozen
store._map_tuple_key for THAT arm only, replayed over the FULL M2 stimulus
suite (420 cases: gradient/mixed/nearmiss, both benchmarks) — the m4_eval
5th-arm pattern verbatim. The wrapper is never installed globally
(assert_mapper_mode_off guards entry AND teardown); the em extraction +
first-pass mapping replay from the llm.py cache (byte-identical prompts —
the cache key excludes trace tags), so the fresh spend is the `-planstem`
re-offer on DECLINED plan-framed em tuples + the downstream gate/checker on
newly-mapped tuples. Results land as SIBLING records per case
(results/v2.4/m2/cases_fixed/<benchmark>/<case_id>.json) — the frozen case
files under cases/ are NEVER modified (byte-checked per case at write time).
Scoring merges the sibling arm in memory: every M2 endpoint table gains an
em_cg_fixed row (contamination/staging per gradient level, mixed
contamination/fact-retention/staging, near-miss staging + promotion curves
incl. cond-on-staged, funnel diagnostics incl. stem-recovered mapping
rates), the four frozen cells' numbers byte-identical with or without the
arm present (regression-tested). ADOPTION CRITERION (owner-approved, decided
on these results): the repair is adopted as the default cautious-gated
configuration iff its planted-probe contamination is not worse than em_cg's
by more than ADOPTION_THRESHOLD_PP = 2.5 percentage points on any gradient
level or mixed block (one arc = 1/60 ~ 1.7pp; threshold allows one-case
noise; blocks pooled across benchmarks, n = 60, matching the prereg's 1/60
arithmetic — per-benchmark deltas reported descriptively). Interpretive
choice (disclosed): g5_present_tense_change's per-level endpoint measures a
commit that is designed-gold CORRECT (taxonomy rule (d) — the frozen M2
design and the released report both pin "a commit is CORRECT; staging is
over-gating"), so g5 is NOT a contamination block; it is shown in the
adoption table descriptively, and whether binding it would change the
verdict is reported explicitly. Near-miss staging-turn contamination is
likewise shown descriptively (the prereg names gradient levels + the mixed
block only).

BUDGET (2026-07-08 amendment): stage "m2" is an ADVISORY forecast
(M2_ADVISORY_FORECAST = the 23,000 reallocation) — spend is recorded per
fresh call to results/v2.4/budget.json via forecast_v22.SharedCallBudget,
overruns of the stage forecast WARN and continue; the ONLY hard stop is the
120,000 total envelope (SharedCallBudget's built-in combined-envelope
abort).

CLI (repo root, pilot/.venv/bin/python experiments/v2_full/graded_eval.py):
  stimuli   [--force]          deterministic manifest + FM-4 key registry
                               (no LLM calls) -> results/v2.4/m2/
  render    [--only ids] [--benchmark b]   two-call render, budget-guarded,
                               resumable (values persist per group)
  freeze                       lint + sha256 sidecar over the rendered
                               manifest (the freeze artifact)
  run       [--benchmarks] [--families em,ez] [--strata] [--ids] [--call-cap]
                               four-cell micro-replays, resumable per case
  run-fixed [--benchmarks] [--strata] [--ids] [--call-cap] [--out]
                               em_cg_fixed arm ONLY (REPAIR-COVERAGE prereg):
                               sibling records -> cases_fixed/, resumable per
                               case; frozen case files never modified
  score     [--cases] [--fixed-cases] [--out-dir]
                               per-stratum curves -> m2_results.json +
                               m2_report.md (judge calls budget-guarded);
                               merges the em_cg_fixed sibling arm when its
                               records exist + evaluates the preregistered
                               adoption criterion
  gate-items                   naturalness FM-1 item builder (injected vs
                               length-decile-matched natural; freeze artifact
                               for the frozen judging pathway)
  smoke     [--fresh-cap N]    render 3 stimuli (one near-miss case: staging
                               + p1 + p2) + run the staging stimulus through
                               ONE gated cell (em_cg); default cap 10 fresh
                               calls; artifacts under results/v2.4/m2/smoke/;
                               spend recorded under stage m2 (real M2 spend)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
from collections import defaultdict
from functools import cache
from pathlib import Path

import _bootstrap  # noqa: F401  (R1 layout: puts src/ + experiments/ on sys.path)

import classify
import data
import driver
import forecast_v22
import gated_store
import inject
import inject_v2
import llm
import locomo_data
import locomo_render
import mapper_fix
import natres
import natural_check_rule5 as rule5
import run_predicates as test3
import stats
import store
import syn_res
from gated_store import ARGMAX_ROUTE, DEFAULT_STORE
from inject import (BLACKLIST, MARKER_WORDS, ROUND2_INTEGRATION_NOTE,
                    ROUND2_TELL_BLACKLIST, Draws, _align_plan_value, _norm,
                    _render_ok, novelty_ok, sibling_value_exclusions)
from inject_v2 import (AUX_POOL_V2, LOCOMO_LENGTH_TARGET, LOCOMO_REGISTER_NOTE,
                       _locomo_author_note, _render_ok_v2)
from zepstore import UNKNOWN_SENTINEL, ZepStore, apply_sentinel

from paths import PROMPTS_DIR, RESULTS_DIR, ROOT

SEED_M2 = 20260708

V24_DIR = RESULTS_DIR / "v2.4"
BUDGET_PATH = V24_DIR / "budget.json"
M2_DIR = V24_DIR / "m2"
MANIFEST_PATH = M2_DIR / "stimuli_manifest.jsonl"
KEYS_PATH = M2_DIR / "m2_keys.jsonl"
FREEZE_SIDECAR_PATH = M2_DIR / "stimuli_manifest.sha256"
CASES_DIR = M2_DIR / "cases"
GATE_ITEMS_PATH = {b: M2_DIR / f"gate_items_{b}.jsonl" for b in ("lme", "locomo")}
GATE_DISPLAY_PATH = {b: M2_DIR / f"gate_display_{b}.jsonl" for b in ("lme", "locomo")}
RESULTS_PATH = M2_DIR / "m2_results.json"
REPORT_PATH = M2_DIR / "m2_report.md"
SMOKE_DIR = M2_DIR / "smoke"

STAGE = "m2"
#: ADVISORY stage forecast (2026-07-07 reallocation, made advisory by the
#: 2026-07-08 amendment): tracked and reported vs actuals, never a hard stop.
M2_ADVISORY_FORECAST = 23_000

BENCHMARKS = ("lme", "locomo")
FAMILIES = ("em", "ez")                      # metrics-family keys
POLICIES = ("ungated", "gated")
CELLS = ("em_cw", "em_cg", "ez_cw", "ez_cg")  # arms.METRICS_CELL aliases
GATED_CELLS = ("em_cg", "ez_cg")
#: The preregistered repair arm (REPAIR-COVERAGE, 2026-07-09): cautious (em)
#: family + gated policy with the M1 stem-wrapper mapper installed as a LOCAL
#: mapper for THIS arm only (mode 'plan_stem_wrapper'; never global —
#: assert_mapper_mode_off guards entry and teardown).
FIXED_CELL = "em_cg_fixed"
#: Diagnostics-family key for the wrapper-mapped em funnel (m4_eval's
#: em_fixed convention — the stem-recovered mapping rates read it).
FIXED_DIAG_FAMILY = "em_fixed"
#: Every arm that can get a score row: the four frozen cells + the repair arm
#: (scored only where its sibling record exists; frozen numbers unchanged).
SCORED_CELLS = CELLS + (FIXED_CELL,)
GATED_CELLS_EXT = GATED_CELLS + (FIXED_CELL,)
#: Sibling-record root for the repair arm — NEVER the frozen CASES_DIR.
CASES_FIXED_DIR = M2_DIR / "cases_fixed"

#: ADOPTION CRITERION (owner-approved, REPAIR-COVERAGE prereg): adopt iff
#: em_cg_fixed contamination is not worse than em_cg by more than 2.5pp on
#: any gradient level or mixed block (pooled n=60; 1/60 ~ 1.7pp).
ADOPTION_THRESHOLD_PP = 2.5
#: The binding blocks: contamination-semantics gradient levels + mixed. g5 is
#: designed-gold completed_event (a commit is CORRECT, taxonomy rule (d)) so
#: its per-level endpoint is not contamination — shown descriptively, with a
#: would-the-verdict-change line if it were bound (module docstring).
ADOPTION_BINDING_LEVELS = ("g1_strong_hedge", "g2_weak_hedge",
                           "g3_confident_plan", "g4_scheduled_event",
                           "m1_fact_plus_hedge")
ADOPTION_DESCRIPTIVE_LEVELS = ("g5_present_tense_change", "staging")
LME_LENGTH_TARGET = inject_v2.LME_LENGTH_TARGET

N_PER_STRATUM = 30            # per level per benchmark (de-scope: 30 -> 24)
NEARMISS_TAIL = 3             # checker/ingest tail = [t*, t*+1, t*+2]
SENSITIVITY_WINDOW = test3.SENSITIVITY_WINDOW  # = 2 (v1 §3.4 semantics)


# ---------------------------------------------------------------------------
# Frozen stratum tables (level definitions ARE the construction labels)
# ---------------------------------------------------------------------------

#: Deterministic hedge-marker vocabularies enforcing the gradient's level
#: separation at render time (_m2_render_ok). g1 requires a STRONG marker,
#: g2 a WEAK one, g3/g4/g5 ban BOTH sets — so the textual hedging strength
#: is monotone non-increasing in rank by construction.
HEDGE_MARKERS_STRONG = (
    "might", "maybe", "torn", "mulling", "debating", "not sure", "undecided",
    "on the fence", "toying with", "weighing", "still deciding",
    "haven't decided", "can't decide", "thinking about", "considering",
    "back and forth", "haven't made up my mind",
)
HEDGE_MARKERS_WEAK = (
    "probably", "planning to", "plan to", "planning on", "leaning toward",
    "leaning towards", "likely", "hoping to", "aiming to", "thinking about",
    "considering", "tempted to", "keen to", "may ",
)

GRADIENT_LEVELS: tuple[dict, ...] = (
    {
        "level": "g1_strong_hedge", "rank": 1, "contract": "SPECULATION",
        "hedge_strength": "deliberating", "construction_label": "speculative",
        "designed_gold_class": "speculative", "designed_gold_route": "stage",
        "taxonomy_rule": None,
        "note": ("strong hedge: explicit deliberation required — the speaker "
                 "is still weighing the change and says so ('torn between', "
                 "'still mulling', 'might'). The change must read as clearly "
                 "unresolved."),
    },
    {
        "level": "g2_weak_hedge", "rank": 2, "contract": "SPECULATION",
        "hedge_strength": "intending", "construction_label": "speculative",
        "designed_gold_class": "speculative", "designed_gold_route": "stage",
        "taxonomy_rule": None,
        "note": ("weak hedge: exactly one soft prospective hedge ('probably', "
                 "'planning to', 'leaning toward') — no strong uncertainty "
                 "language, but the change is explicitly still prospective."),
    },
    {
        "level": "g3_confident_plan", "rank": 3, "contract": "SPECULATION",
        "hedge_strength": "intending", "construction_label": "speculative",
        "designed_gold_class": "speculative", "designed_gold_route": "stage",
        "taxonomy_rule": "a",
        "note": ("confident plan (taxonomy rule (a)): state the plan as "
                 "settled and definite — 'I'm moving next month' shape. NO "
                 "hedging words at all (no probably/maybe/thinking about/"
                 "considering/planning to). The change itself must remain "
                 "UNREALIZED: no completion language, the event is still "
                 "ahead."),
    },
    {
        "level": "g4_scheduled_event", "rank": 4, "contract": "SPECULATION",
        "hedge_strength": "scheduled_but_future", "construction_label": "speculative",
        "designed_gold_class": "speculative", "designed_gold_route": "stage",
        "taxonomy_rule": "b",
        "note": ("scheduled event stated as fact (taxonomy rule (b)): the "
                 "switch is locked in on the calendar and stated as bare "
                 "fact — 'I start Monday' / 'the switch goes through next "
                 "week' shape, no hedging words at all — but the event has "
                 "NOT happened yet."),
    },
    {
        "level": "g5_present_tense_change", "rank": 5, "contract": "COMPLETED_D",
        "hedge_strength": None, "construction_label": "completed_event",
        "designed_gold_class": "completed_event", "designed_gold_route": "commit",
        "taxonomy_rule": "d",
        "note": ("present-tense state implying a completed change (taxonomy "
                 "rule (d)): assert the NEW value as current state in the "
                 "present tense ('these days it's X', 'it's X now'), "
                 "conflicting with the established value; do NOT narrate the "
                 "change event and use no future/plan language."),
    },
)

MIXED_LEVEL: dict = {
    "level": "m1_fact_plus_hedge", "rank": 1, "contract": "SPECULATION",
    "hedge_strength": "intending", "construction_label": "speculative",
    "designed_gold_class": "speculative", "designed_gold_route": "stage",
    "taxonomy_rule": None,
    "note": ("mixed fact+hedge, ONE sentence (switch-frame): the single "
             "sentence must (1) assert the CURRENT value "
             "(established_value) as a committed, unhedged fact about the "
             "speaker and (2) float a possible future switch to the sampled "
             "value with a clear hedge — e.g. \"I've been with <current> "
             "for years, though I'm thinking about moving to <new> next "
             "quarter.\" Both values MUST appear in that one sentence; no "
             "second sentence about the change."),
}

NEARMISS_STAGING: dict = {
    "contract": "SPECULATION", "hedge_strength": "intending",
    "construction_label": "speculative", "designed_gold_class": "speculative",
    "designed_gold_route": "stage",
    "note": ("staged hedge for a near-miss resolution case: a plainly hedged "
             "prospective switch ('planning to', 'probably') that the gated "
             "cells should stage."),
}

NEARMISS_LEVELS: tuple[dict, ...] = (
    {
        "level": "p1_verbatim", "rank": 1, "value_required": True,
        "no_six_gram_overlap": False, "attribute_noun_banned": False,
        "note": ("verbatim-ish resolution: assert that the previously mooted "
                 "change WENT THROUGH, reusing the staged aside's own key "
                 "wording where natural, and name the value explicitly."),
    },
    {
        "level": "p2_paraphrase", "rank": 2, "value_required": True,
        "no_six_gram_overlap": True, "attribute_noun_banned": False,
        "note": ("paraphrase resolution: assert the completion in completely "
                 "fresh wording — share no 6-word phrase with the staged "
                 "aside — and still name the value explicitly."),
    },
    {
        "level": "p3_oblique", "rank": 3, "value_required": True,
        "no_six_gram_overlap": True, "attribute_noun_banned": True,
        "note": ("oblique resolution: the change is asserted as done, but "
                 "referred back to only obliquely ('went ahead with it', "
                 "'pulled the trigger on that switch') — the value must "
                 "appear, but do NOT name the attribute with the nouns of "
                 "the staged aside."),
    },
    {
        "level": "p4_situational", "rank": 4, "value_required": False,
        "no_six_gram_overlap": True, "attribute_noun_banned": False,
        "note": ("situational-implication resolution: never state that the "
                 "change happened and use NO completion verbs about the "
                 "plan; describe a concrete first experience/situation that "
                 "only makes sense if the change already happened (e.g. the "
                 "first use of the new arrangement). The value may appear "
                 "incidentally but must not be announced as a switch."),
    },
)

#: Manifest inj_type vocabulary.
TYPE_ESTABLISH = "m2_establish"
TYPE_STIMULUS = "m2_stimulus"       # gradient + mixed + nearmiss staging
TYPE_RESOLUTION = "m2_resolution"   # nearmiss resolving turns

STRATA = ("gradient", "mixed", "nearmiss")

CONTRACTS_USED = ("SPECULATION", "ESTABLISH", "COMPLETED_D", "T3_PROMOTE")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


class _TranscriptShim:
    """Minimal object satisfying inject.novelty_ok's ``sub.transcript_text()``."""

    def __init__(self, text: str):
        self._text = text

    def transcript_text(self) -> str:
        return self._text


@cache
def _lme_transcript(qid: str) -> str:
    sub = data.get_substrate(qid)
    return "\n".join(t["content"] for s in sub.sessions for t in s).lower()


@cache
def _locomo_transcript(sample_id: str) -> str:
    conv = locomo_data.get_conversation(sample_id)
    return "\n".join(t["text"] for s in conv.sessions for t in s).lower()


def _transcript(benchmark: str, qid: str) -> str:
    return _lme_transcript(qid) if benchmark == "lme" else _locomo_transcript(qid)


def _cluster_of(benchmark: str, qid: str) -> str:
    return driver._cluster_of({"question_id": qid})


def assert_mapper_mode_off() -> None:
    """The frozen-cell pin (m4_eval.assert_mapper_mode_off verbatim): this
    runner calls store._map_tuple_key directly for the four frozen cells, and
    the M1 stem wrapper must not have been monkey-installed over it —
    store.make_tuple_key_mapper closures carry ``mode == 'plan_stem_wrapper'``;
    the frozen function carries no mode attribute. The repair arm passes its
    wrapper as a LOCAL mapper only, and re-asserts this at teardown."""
    mode = getattr(store._map_tuple_key, "mode", "off")
    assert mode == "off", (
        f"store._map_tuple_key carries mapping mode {mode!r} — the M1 stem "
        "wrapper is installed globally; M2 is preregistered with the frozen "
        "mapper, mode OFF (the em_cg_fixed arm uses a LOCAL wrapper only)")


# ---------------------------------------------------------------------------
# Budget (advisory stage m2; hard limit = the 120k combined envelope only)
# ---------------------------------------------------------------------------

def m2_guard(path: Path = BUDGET_PATH,
             local_cap: int | None = None) -> forecast_v22.SharedCallBudget:
    """The M2 shared-envelope guard. Per the 2026-07-08 amendment the stage
    allocation is ADVISORY: exceeding the m2 forecast prints a warning and
    CONTINUES; the only hard stop is the combined 120k envelope
    (SharedCallBudget refuses/aborts on envelope exhaustion and records
    every fresh call to the budget file)."""
    budget = mapper_fix.ensure_budget(path)
    spent = budget["spent"].get(STAGE, 0)
    forecast = budget["allocation"].get(STAGE, M2_ADVISORY_FORECAST)
    if spent >= forecast:
        print(f"[m2 budget ADVISORY] stage spend {spent:,} >= forecast "
              f"{forecast:,} — continuing (stages are advisory per the "
              "2026-07-08 amendment; only the total envelope is hard)")
    remaining = forecast_v22.budget_remaining(budget)
    cap = remaining if local_cap is None else min(local_cap, remaining)
    return forecast_v22.SharedCallBudget(STAGE, path=path, local_cap=cap)


def advisory_status(path: Path = BUDGET_PATH) -> dict:
    budget = mapper_fix.ensure_budget(path)
    spent = budget["spent"].get(STAGE, 0)
    forecast = budget["allocation"].get(STAGE, M2_ADVISORY_FORECAST)
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
# Frozen v2 geometries (placements are reused, never re-audited here)
# ---------------------------------------------------------------------------

def pair_geometries(benchmark: str) -> list[dict]:
    """Audited (establish, dependent) slot pairs from the frozen v2 manifest
    (spec_aux + caux_correct rows with their paired establishes), one geometry
    per dependent row, sorted by the frozen dependent injection_id."""
    rows = driver.load_manifest_v2()
    by_id = {r["injection_id"]: r for r in rows}
    out = []
    for r in rows:
        if r["benchmark"] != benchmark:
            continue
        if r["inj_type"] not in ("spec_aux", "caux_correct"):
            continue
        est = by_id.get(r["paired_with"]) if r["paired_with"] else None
        if est is None:
            continue
        if benchmark == "locomo" and est.get("speaker") != r.get("speaker"):
            continue  # same-speaker pairs only (aside is about its author)
        out.append({
            "question_id": r["question_id"],
            "speaker": r.get("speaker"),
            "est_placement": dict(est["placement"]),
            "dep_placement": dict(r["placement"]),
            "source_dep_id": r["injection_id"],
        })
    return sorted(out, key=lambda g: g["source_dep_id"])


def t3_geometries(benchmark: str) -> list[dict]:
    """The 50 frozen t3 case geometries (staging + variant slots; the R1-B1
    sensitivity window is guaranteed by the frozen plan)."""
    rows = driver.load_manifest_v2()
    staging = {r["t3_case_id"]: r for r in rows
               if r["benchmark"] == benchmark and r["inj_type"] == "t3_staging"}
    variant_slot: dict[str, dict] = {}
    for r in rows:
        if r["benchmark"] == benchmark and r["inj_type"] == "t3_variant":
            variant_slot.setdefault(r["t3_case_id"], dict(r["placement"]))
    out = []
    for case_id in sorted(staging):
        s = staging[case_id]
        out.append({
            "question_id": s["question_id"],
            "speaker": s.get("speaker"),
            "staging_placement": dict(s["placement"]),
            "variant_placement": variant_slot[case_id],
            "source_case_id": case_id,
        })
    return out


# ---------------------------------------------------------------------------
# stimuli — deterministic manifest + FM-4 key registry (no LLM calls)
# ---------------------------------------------------------------------------

def _registered_attrs(benchmark: str, qid: str) -> set[str]:
    return {k.attribute for k in driver.v2_aux_keys(benchmark, qid)}


def _who(benchmark: str, speaker: str | None) -> str:
    return "the user" if benchmark == "lme" else str(speaker)


def _pick_attribute(draws: Draws, benchmark: str, qid: str,
                    transcript: str) -> str:
    """Seeded aux-attribute pick: not already registered for this substrate
    in the frozen v2 registry, and keyword-collision-free (AUX_POOL_V2)."""
    taken = _registered_attrs(benchmark, qid)
    eligible = [a for a, (_d, _p, kws) in sorted(AUX_POOL_V2.items())
                if a not in taken and not any(k in transcript for k in kws)]
    if not eligible:
        raise RuntimeError(f"aux pool exhausted for {benchmark}/{qid}")
    return draws.choice(eligible)


def _base_row(*, injection_id, benchmark, question_id, speaker, inj_type,
              case_id, stratum, level, rank, contract, entity, attribute,
              placement, construction_label, designed_gold_class,
              designed_gold_route, taxonomy_rule, note, hedge_strength,
              paired_with=None, edge_case_tags=(), extra=None) -> dict:
    row = {
        "injection_id": injection_id,
        "benchmark": benchmark,
        "question_id": question_id,
        "speaker": speaker,
        "inj_type": inj_type,
        "case_id": case_id,
        "stratum": stratum,
        "level": level,
        "rank": rank,
        "contract": contract,
        "entity": entity,
        "attribute": attribute,
        "value": None,
        "old_value": None,
        "fact_value": None,
        "placement": dict(placement),
        "probe_point": "full_ingestion",
        "construction_label": construction_label,
        "designed_gold_class": designed_gold_class,
        "designed_gold_route": designed_gold_route,
        "taxonomy_rule": taxonomy_rule,
        "stratum_note": note,
        "hedge_strength": hedge_strength,
        "edge_case_tags": list(edge_case_tags),
        "paired_with": paired_with,
        "t3_case_id": None,
        "t3_variant_kind": None,
        "cluster": _cluster_of(benchmark, question_id),
        "seed": SEED_M2,
        "text": None,
    }
    if extra:
        row.update(extra)
    return row


def build_stimuli(n_per_stratum: int = N_PER_STRATUM) -> tuple[list[dict], list[dict]]:
    """Pure, deterministic stimulus plan (seed SEED_M2). Returns
    (manifest_rows, key_registry). Keys + FM-4 probe questions are registered
    HERE — before any value exists (values are sampled at render Call 1)."""
    draws = Draws(SEED_M2)
    rows: list[dict] = []
    keys: list[dict] = []

    def register_key(benchmark, qid, speaker, case_id, stratum, level, attr):
        who = _who(benchmark, speaker)
        desc_tpl, probe_tpl, _kws = AUX_POOL_V2[attr]
        entity = "user" if benchmark == "lme" else str(speaker).lower()
        keys.append({
            "benchmark": benchmark, "question_id": qid, "case_id": case_id,
            "stratum": stratum, "level": level,
            "entity": entity, "attribute": attr,
            "description": desc_tpl.format(who=who),
            "probe_question": probe_tpl.format(who=who),
            "created_by": case_id, "seed": SEED_M2,
        })
        return entity

    for benchmark in BENCHMARKS:
        pair_geoms = pair_geometries(benchmark)
        n_pairs_needed = n_per_stratum * (len(GRADIENT_LEVELS) + 1)  # + mixed
        assert len(pair_geoms) >= n_pairs_needed, (
            f"{benchmark}: {len(pair_geoms)} frozen pair geometries < "
            f"{n_pairs_needed} needed")
        chosen_pairs = draws.sample(pair_geoms, n_pairs_needed)

        # (a) hedging-strength gradient — establish + stimulus per level
        for li, spec in enumerate(GRADIENT_LEVELS):
            for i in range(n_per_stratum):
                geom = chosen_pairs[li * n_per_stratum + i]
                qid, speaker = geom["question_id"], geom["speaker"]
                case_id = f"m2-{benchmark}-{spec['level']}-{i:03d}"
                attr = _pick_attribute(draws, benchmark, qid,
                                       _transcript(benchmark, qid))
                entity = register_key(benchmark, qid, speaker, case_id,
                                      "gradient", spec["level"], attr)
                est_id = f"m2est-{case_id}"
                rows.append(_base_row(
                    injection_id=est_id, benchmark=benchmark, question_id=qid,
                    speaker=speaker, inj_type=TYPE_ESTABLISH, case_id=case_id,
                    stratum="gradient", level=spec["level"], rank=spec["rank"],
                    contract="ESTABLISH", entity=entity, attribute=attr,
                    placement=geom["est_placement"],
                    construction_label="other", designed_gold_class="other",
                    designed_gold_route="additive", taxonomy_rule=None,
                    note=None, hedge_strength=None,
                    edge_case_tags=["m2", "establish"],
                    extra={"source_geometry": geom["source_dep_id"]}))
                tags = ["m2", f"level:{spec['level']}"]
                if spec["taxonomy_rule"]:
                    tags.append(spec["taxonomy_rule"])
                if spec["hedge_strength"]:
                    tags.append(f"hedge:{spec['hedge_strength']}")
                rows.append(_base_row(
                    injection_id=f"m2stim-{case_id}", benchmark=benchmark,
                    question_id=qid, speaker=speaker, inj_type=TYPE_STIMULUS,
                    case_id=case_id, stratum="gradient", level=spec["level"],
                    rank=spec["rank"], contract=spec["contract"],
                    entity=entity, attribute=attr,
                    placement=geom["dep_placement"],
                    construction_label=spec["construction_label"],
                    designed_gold_class=spec["designed_gold_class"],
                    designed_gold_route=spec["designed_gold_route"],
                    taxonomy_rule=spec["taxonomy_rule"], note=spec["note"],
                    hedge_strength=spec["hedge_strength"],
                    paired_with=est_id, edge_case_tags=tags,
                    extra={"source_geometry": geom["source_dep_id"]}))

        # (b) mixed statements — ONE sentence, fact + hedge, no establish turn
        for i in range(n_per_stratum):
            geom = chosen_pairs[len(GRADIENT_LEVELS) * n_per_stratum + i]
            qid, speaker = geom["question_id"], geom["speaker"]
            case_id = f"m2-{benchmark}-{MIXED_LEVEL['level']}-{i:03d}"
            attr = _pick_attribute(draws, benchmark, qid,
                                   _transcript(benchmark, qid))
            entity = register_key(benchmark, qid, speaker, case_id,
                                  "mixed", MIXED_LEVEL["level"], attr)
            rows.append(_base_row(
                injection_id=f"m2stim-{case_id}", benchmark=benchmark,
                question_id=qid, speaker=speaker, inj_type=TYPE_STIMULUS,
                case_id=case_id, stratum="mixed", level=MIXED_LEVEL["level"],
                rank=MIXED_LEVEL["rank"], contract=MIXED_LEVEL["contract"],
                entity=entity, attribute=attr,
                placement=geom["dep_placement"],
                construction_label=MIXED_LEVEL["construction_label"],
                designed_gold_class=MIXED_LEVEL["designed_gold_class"],
                designed_gold_route=MIXED_LEVEL["designed_gold_route"],
                taxonomy_rule=None, note=MIXED_LEVEL["note"],
                hedge_strength=MIXED_LEVEL["hedge_strength"],
                edge_case_tags=["m2", "mixed_fact_hedge",
                                f"hedge:{MIXED_LEVEL['hedge_strength']}"],
                extra={"source_geometry": geom["source_dep_id"]}))

        # (c) near-miss resolutions — staged hedge + 4 paraphrase levels
        t3_geoms = t3_geometries(benchmark)
        assert len(t3_geoms) >= n_per_stratum, (
            f"{benchmark}: {len(t3_geoms)} frozen t3 geometries < {n_per_stratum}")
        chosen_t3 = draws.sample(t3_geoms, n_per_stratum)
        for i in range(n_per_stratum):
            geom = chosen_t3[i]
            qid, speaker = geom["question_id"], geom["speaker"]
            case_id = f"m2-{benchmark}-nearmiss-{i:03d}"
            attr = _pick_attribute(draws, benchmark, qid,
                                   _transcript(benchmark, qid))
            entity = register_key(benchmark, qid, speaker, case_id,
                                  "nearmiss", "staging", attr)
            stg_id = f"m2stg-{case_id}"
            rows.append(_base_row(
                injection_id=stg_id, benchmark=benchmark, question_id=qid,
                speaker=speaker, inj_type=TYPE_STIMULUS, case_id=case_id,
                stratum="nearmiss", level="staging", rank=0,
                contract=NEARMISS_STAGING["contract"], entity=entity,
                attribute=attr, placement=geom["staging_placement"],
                construction_label=NEARMISS_STAGING["construction_label"],
                designed_gold_class=NEARMISS_STAGING["designed_gold_class"],
                designed_gold_route=NEARMISS_STAGING["designed_gold_route"],
                taxonomy_rule=None, note=NEARMISS_STAGING["note"],
                hedge_strength=NEARMISS_STAGING["hedge_strength"],
                edge_case_tags=["m2", "nearmiss_staging",
                                f"hedge:{NEARMISS_STAGING['hedge_strength']}"],
                extra={"source_geometry": geom["source_case_id"]}))
            for spec in NEARMISS_LEVELS:
                rows.append(_base_row(
                    injection_id=f"m2res-{case_id}-{spec['level']}",
                    benchmark=benchmark, question_id=qid, speaker=speaker,
                    inj_type=TYPE_RESOLUTION, case_id=case_id,
                    stratum="nearmiss", level=spec["level"],
                    rank=spec["rank"], contract="T3_PROMOTE",
                    entity=entity, attribute=attr,
                    placement=geom["variant_placement"],
                    construction_label="completed_event",
                    designed_gold_class="completed_event",
                    designed_gold_route="commit", taxonomy_rule=None,
                    note=spec["note"], hedge_strength=None,
                    paired_with=stg_id,
                    edge_case_tags=["m2", f"paraphrase_distance:{spec['rank']}"],
                    extra={"source_geometry": geom["source_case_id"],
                           "value_required": spec["value_required"],
                           "no_six_gram_overlap": spec["no_six_gram_overlap"],
                           "attribute_noun_banned": spec["attribute_noun_banned"]}))
    return rows, keys


def cmd_stimuli(force: bool = False) -> None:
    if MANIFEST_PATH.exists() and not force:
        raise SystemExit(
            f"{MANIFEST_PATH} exists (may carry rendered values) — refusing "
            "to clobber; pass --force to regenerate the UNRENDERED plan")
    rows, keys = build_stimuli()
    _write_jsonl(MANIFEST_PATH, rows)
    _write_jsonl(KEYS_PATH, keys)
    n = defaultdict(int)
    for r in rows:
        n[(r["benchmark"], r["stratum"], r["inj_type"])] += 1
    print(f"stimuli: {len(rows)} manifest rows, {len(keys)} FM-4 keys "
          f"-> {_rel(MANIFEST_PATH)}")
    for k in sorted(n):
        print(f"  {k}: {n[k]}")


# ---------------------------------------------------------------------------
# render — the frozen two-call pipeline, payload-level adaptation only
# ---------------------------------------------------------------------------

def _host_turn(benchmark: str, qid: str, placement: dict) -> tuple[str, str]:
    si, ti = placement["session_idx"], placement["turn_idx"]
    if benchmark == "lme":
        sub = data.get_substrate(qid)
        return sub.sessions[si][ti]["content"], sub.dates[si]
    conv = locomo_data.get_conversation(qid)
    return conv.sessions[si][ti]["text"], conv.dates[si]


def _style_exemplars_m2(benchmark: str, qid: str, speaker: str | None,
                        rng: random.Random) -> list[str]:
    if benchmark == "lme":
        return inject._style_exemplars(data.get_substrate(qid), rng)
    conv = locomo_data.get_conversation(qid)
    turns = [t["text"] for s in conv.sessions for t in s
             if t["speaker"] == speaker]
    turns = [" ".join(t.split()[:40]) for t in turns if len(t.split()) >= 6]
    return rng.sample(turns, min(3, len(turns)))


def _attr_ban_tokens(attribute: str) -> list[str]:
    return [tok for tok in attribute.split("_") if len(tok) > 3]


#: Placeholder for masking a pinned value string out of a checked text. Free
#: of sentence punctuation and never a natural token, so masked spans neither
#: fabricate sentence boundaries nor collide with real n-gram tokens.
_VALUE_MASK = " m2valuemask "


def _mask_value(text: str, value: str | None) -> str:
    """Replace verbatim occurrences of ``value`` (case-insensitive) with the
    neutral mask token. Used by per-stratum assert waivers that must ignore
    the REQUIRED value string itself while staying strict everywhere else."""
    if not value:
        return text
    return re.sub(re.escape(str(value)), _VALUE_MASK, text, flags=re.IGNORECASE)


def _m2_render_ok(aside: str, row: dict, staging_text: str | None) -> bool:
    """M2 construction checks on top of the frozen inject._render_ok +
    inject_v2._render_ok_v2 (which are applied by the caller). Deterministic
    — these enforce the level DEFINITIONS at render time."""
    low = aside.lower()
    level = row["level"]
    if row["inj_type"] == TYPE_STIMULUS and (
            row["stratum"] == "gradient"
            or (row["stratum"] == "nearmiss" and level == "staging")):
        banned = tuple(set(HEDGE_MARKERS_STRONG) | set(HEDGE_MARKERS_WEAK))
        if level == "g1_strong_hedge":
            if not any(m in low for m in HEDGE_MARKERS_STRONG):
                return False
        elif level in ("g2_weak_hedge", "staging"):
            if not any(m in low for m in HEDGE_MARKERS_WEAK):
                return False
        else:  # g3/g4/g5: textually unhedged by construction
            checked = low
            if level == "g4_scheduled_event":
                # g4 waiver (stratum design, taxonomy rule (b)): a scheduled
                # event is "locked in on the calendar and stated as bare
                # fact", so calendar-date phrasing is REQUIRED by the level
                # definition — the month name in "May 1" / "May 15" is not
                # the modal hedge the weak-marker "may " entry targets (the
                # modal is never followed by a numeral). Mask month-day date
                # expressions before marker matching; every hedge marker,
                # including modal "may", still fails everywhere else.
                checked = re.sub(r"\bmay\s+\d{1,2}(?:st|nd|rd|th)?\b", " ",
                                 checked)
            if any(m in checked for m in banned):
                return False
    if row["stratum"] == "mixed":
        if not row.get("fact_value"):
            return False
        if _norm(row["fact_value"]) not in _norm(aside):
            return False
        # one sentence: no sentence boundary strictly inside the aside.
        # m1 waiver (stratum design: BOTH values must appear verbatim in the
        # ONE sentence): value-internal terminal punctuation ("Supergoop!")
        # is brand orthography, not a sentence boundary — with the value
        # required verbatim mid-sentence the unmasked check is unsatisfiable
        # by construction. Mask the two required value strings before
        # boundary detection; a real second sentence anywhere else still
        # fails.
        checked = aside.strip()
        for v in (row.get("value"), row.get("fact_value")):
            checked = _mask_value(checked, v)
        if [b for b in inject._sentence_boundaries(checked) if b < len(checked)]:
            return False
    if row["inj_type"] == TYPE_RESOLUTION and staging_text is not None:
        if row.get("no_six_gram_overlap"):
            # near-miss waiver (stratum design + T3 fixed-value semantics):
            # the resolution asserts the STAGED value verbatim (p2/p3:
            # value_required; p4: the value may appear incidentally), and the
            # staged aside names the same value — for multi-word values a
            # 6-gram spanning the pinned value string is shared by
            # construction, not by paraphrase failure. Mask the value in
            # BOTH texts (the mask token still occupies its n-gram slot, so
            # copied wording AROUND the value keeps failing, as does any
            # shared 6-gram elsewhere).
            if test3.find_six_gram_overlaps(
                    _mask_value(aside, row.get("value")),
                    [_mask_value(staging_text, row.get("value"))]):
                return False
        if row.get("attribute_noun_banned"):
            vn = _norm(row["value"] or "")
            for tok in _attr_ban_tokens(row["attribute"]):
                if tok in vn:
                    continue  # value itself carries the token — not banned
                if re.search(rf"(?<![\w]){re.escape(tok)}(?![\w])", low):
                    return False
        if row["level"] == "p4_situational":
            completion = ("went ahead", "pulled the trigger", "made the switch",
                          "switched to", "went through with", "officially",
                          "finalized", "signed up", "it's done", "did it")
            if any(m in low for m in completion):
                return False
    return True


async def render_m2_row(row: dict, by_id: dict[str, dict],
                        plan_sys: str, render_sys: str,
                        keydesc: dict, key_group: list[dict] = ()) -> None:
    """Two-call render of one M2 row — the frozen prompts byte-untouched;
    all M2-specific guidance rides in the JSON payload (stratum_note,
    register/author notes), the inject_v2 §3.3 payload-level-adaptation
    precedent."""
    from llm import call_llm

    benchmark = row["benchmark"]
    p = row["placement"]
    host_turn, host_date = _host_turn(benchmark, row["question_id"], p)
    shim = _TranscriptShim(_transcript(benchmark, row["question_id"]))

    paired = by_id.get(row["paired_with"]) if row["paired_with"] else None
    if paired and paired["value"] is None:
        raise RuntimeError(f"{row['injection_id']}: paired row not rendered yet")

    # near-miss resolutions assert the staged value (fixed, T3 semantics)
    fixed = paired["value"] if row["inj_type"] == TYPE_RESOLUTION else None
    established = paired["value"] if (paired and row["inj_type"] != TYPE_RESOLUTION) else None
    staging_text = paired["text"] if (paired and row["inj_type"] == TYPE_RESOLUTION) else None

    exclusions = [v for v in [established] if v]
    for v in sibling_value_exclusions(row, list(key_group)):
        if v not in exclusions:
            exclusions.append(v)

    # mixed: sample the in-sentence CURRENT value first (ESTABLISH plan call;
    # content plan discarded — value-sampling only; flagged in the module
    # docstring as an interpretive choice)
    if row["stratum"] == "mixed" and row.get("fact_value") is None:
        fact_payload = {
            "contract": "ESTABLISH",
            "injection_type": TYPE_ESTABLISH,
            "target_key": {"entity": row["entity"], "attribute": row["attribute"]},
            "established_value": None, "gold_old_value": None,
            "fixed_value": None, "value_exclusions": list(exclusions),
            "host_turn_text": host_turn, "insertion_position": p["position"],
            "host_session_date": host_date, "other_session_date": host_date,
            "hedge_strength": None, "near_miss_subtype": None,
            "attribute_description": keydesc.get(row["case_id"], ""),
        }
        fout = await call_llm(plan_sys, json.dumps(fact_payload, ensure_ascii=False),
                              trace_tag=f"m2-plan-fact-{row['injection_id']}")
        fact = next((str(c) for c in fout.get("value_candidates", [])
                     if novelty_ok(str(c), shim, exclusions)), None)
        if fact is None:
            raise RuntimeError(f"{row['injection_id']}: no fact value passed novelty")
        row["fact_value"] = fact
    if row.get("fact_value"):
        established = row["fact_value"]
        if established not in exclusions:
            exclusions.append(established)

    payload = {
        "contract": row["contract"],
        "injection_type": row["inj_type"],
        "target_key": {"entity": row["entity"], "attribute": row["attribute"]},
        "established_value": established,
        "gold_old_value": None,
        "fixed_value": fixed,
        "value_exclusions": exclusions,
        "host_turn_text": host_turn,
        "insertion_position": p["position"],
        "host_session_date": host_date,
        "other_session_date": host_date,
        "hedge_strength": row["hedge_strength"],
        "near_miss_subtype": None,
        "attribute_description": keydesc.get(row["case_id"], ""),
    }
    if row.get("stratum_note"):
        payload["stratum_note"] = row["stratum_note"]
    if benchmark == "locomo":
        payload["register_note"] = LOCOMO_REGISTER_NOTE
        payload["speaker_name"] = row["speaker"]
        payload["author_note"] = _locomo_author_note(row["speaker"])
    if staging_text is not None:
        payload["staged_aside_text"] = staging_text

    out1 = await call_llm(plan_sys, json.dumps(payload, ensure_ascii=False),
                          trace_tag=f"m2-plan-{row['injection_id']}")
    if fixed is not None:
        value = fixed
    else:
        candidates = [str(c) for c in out1.get("value_candidates", [])]
        value = next((c for c in candidates if novelty_ok(c, shim, exclusions)), None)
        for retry_round in range(1, 4):
            if value is not None:
                break
            retry_payload = dict(payload)
            retry_payload["value_exclusions"] = exclusions + candidates
            retry_payload["retry_note"] = (
                f"Retry round {retry_round}: every previous candidate was "
                "rejected because it appears verbatim in the source "
                "transcript or collides with an excluded value. Propose "
                "distinctly different candidates.")
            out1 = await call_llm(plan_sys,
                                  json.dumps(retry_payload, ensure_ascii=False),
                                  trace_tag=f"m2-plan-retry{retry_round}-{row['injection_id']}")
            retry_candidates = [str(c) for c in out1.get("value_candidates", [])]
            value = next((c for c in retry_candidates
                          if novelty_ok(c, shim, exclusions)), None)
            candidates = candidates + retry_candidates
        if value is None:
            raise RuntimeError(f"{row['injection_id']}: no candidate passed "
                               f"novelty after retries: {candidates}")
        row["value_candidates"] = candidates
    row["value"] = value

    rng = random.Random(f"{SEED_M2}:{row['injection_id']}")
    exemplars = _style_exemplars_m2(benchmark, row["question_id"],
                                    row["speaker"], rng)
    length_target = (LOCOMO_LENGTH_TARGET if benchmark == "locomo"
                     else LME_LENGTH_TARGET)
    # p4 waives the frozen value-containment assert (situational implication
    # may omit the value) — disclosed deviation; every other check applies.
    check_value = "" if (row["inj_type"] == TYPE_RESOLUTION
                         and not row.get("value_required", True)) else value
    for attempt in range(3):
        rpayload = {
            "contract": row["contract"],
            "content_plan": _align_plan_value(out1.get("content_plan", ""),
                                              row.get("value_candidates") or [],
                                              value),
            "value": value,
            "established_value": established,
            "old_value": None,
            "connective": None,
            "integration_note": ROUND2_INTEGRATION_NOTE,
            "insertion_position": p["position"],
            "host_turn_text": host_turn,
            "style_exemplars": exemplars,
            "blacklist": BLACKLIST + ROUND2_TELL_BLACKLIST,
            "marker_words_allowed": False,
            "marker_words": MARKER_WORDS,
            "length_target_words": length_target,
            "attempt": attempt,
        }
        if row.get("stratum_note"):
            rpayload["stratum_note"] = row["stratum_note"]
        if benchmark == "locomo":
            rpayload["register_note"] = LOCOMO_REGISTER_NOTE
            rpayload["author_note"] = _locomo_author_note(row["speaker"])
        if staging_text is not None:
            rpayload["staged_aside_text"] = staging_text
            if row.get("no_six_gram_overlap") and row["level"] == "p3_oblique":
                # Payload-level constraint enforcement (inject_v2 §3.3
                # precedent, frozen prompts untouched): p3 renders were
                # observed re-using the staged aside's motivation clause
                # verbatim, tripping the level's deterministic no-6-gram
                # check. Restates the frozen level definition only.
                rpayload["fresh_wording_note"] = (
                    "Deterministic construction check on this level: the "
                    "aside must not share ANY six-word phrase with "
                    "staged_aside_text. Express the motivation and framing "
                    "in completely fresh wording (do not reuse the staged "
                    "aside's rationale clause); only the value string itself "
                    "may recur.")
        out2 = await call_llm(render_sys, json.dumps(rpayload, ensure_ascii=False),
                              trace_tag=f"m2-render-{row['injection_id']}")
        aside = str(out2.get("aside", "")).strip()
        if (_render_ok(aside, check_value, False, row)
                and _render_ok_v2(aside, row)
                and _m2_render_ok(aside, row, staging_text)):
            row["text"] = aside
            return
        exemplars = _style_exemplars_m2(benchmark, row["question_id"],
                                        row["speaker"], rng)
    raise RuntimeError(f"{row['injection_id']}: render failed asserts after 3 attempts")


def _keydesc() -> dict[str, str]:
    return {k["case_id"]: k["description"] for k in _read_jsonl(KEYS_PATH)}


async def render(only: set[str] | None = None,
                 benchmark: str | None = None) -> None:
    rows = _read_jsonl(MANIFEST_PATH)
    assert rows, f"missing manifest {MANIFEST_PATH} — run `stimuli` first"
    by_id = {r["injection_id"]: r for r in rows}
    keydesc = _keydesc()
    plan_sys = (PROMPTS_DIR / "injector_plan.txt").read_text()
    render_sys = (PROMPTS_DIR / "injector_render.txt").read_text()

    def selected(r):
        return ((only is None or r["injection_id"] in only)
                and (benchmark is None or r["benchmark"] == benchmark)
                and r["text"] is None)

    groups: dict[str, list[dict]] = {}
    for r in rows:  # manifest order: establish/staging precede dependents
        groups.setdefault(r["case_id"], []).append(r)

    failures: list[str] = []
    budget_aborted = False

    async def render_group(group: list[dict]) -> None:
        nonlocal budget_aborted
        for r in group:
            if budget_aborted:
                return
            if selected(r):
                try:
                    await render_m2_row(r, by_id, plan_sys, render_sys,
                                        keydesc, key_group=group)
                except Exception as exc:  # noqa: BLE001 — surfaced + resumable
                    failures.append(f"{r['injection_id']}: {exc}")
                    if "budget" in str(exc).lower() or "envelope" in str(exc).lower():
                        budget_aborted = True  # stop scheduling further rows
                    return
            elif r["text"] is None and any(selected(x) for x in group):
                return  # unrendered ancestor of a selected dependent

    await asyncio.gather(*(render_group(g) for g in groups.values()))
    _write_jsonl(MANIFEST_PATH, rows)
    done = sum(1 for r in rows if r["text"])
    print(f"rendered: {done}/{len(rows)} rows have text")
    if failures:
        for f in failures:
            print(f"RENDER FAILURE: {f}")
        raise SystemExit(f"{len(failures)} row(s) failed to render "
                         "(progress persisted; re-run resumes)")


# ---------------------------------------------------------------------------
# freeze — lint + sha256 sidecar (the stimuli freeze artifact)
# ---------------------------------------------------------------------------

def lint_manifest(rows: list[dict], keys: list[dict]) -> list[str]:
    errors: list[str] = []
    counts = defaultdict(int)
    for r in rows:
        counts[(r["benchmark"], r["stratum"], r["level"], r["inj_type"])] += 1
    for benchmark in BENCHMARKS:
        for spec in GRADIENT_LEVELS:
            for t, n in ((TYPE_ESTABLISH, N_PER_STRATUM),
                         (TYPE_STIMULUS, N_PER_STRATUM)):
                got = counts[(benchmark, "gradient", spec["level"], t)]
                if got != n:
                    errors.append(f"{benchmark}/gradient/{spec['level']}/{t}: "
                                  f"{got} != {n}")
        if counts[(benchmark, "mixed", MIXED_LEVEL["level"], TYPE_STIMULUS)] != N_PER_STRATUM:
            errors.append(f"{benchmark}/mixed budget != {N_PER_STRATUM}")
        if counts[(benchmark, "nearmiss", "staging", TYPE_STIMULUS)] != N_PER_STRATUM:
            errors.append(f"{benchmark}/nearmiss staging budget != {N_PER_STRATUM}")
        for spec in NEARMISS_LEVELS:
            got = counts[(benchmark, "nearmiss", spec["level"], TYPE_RESOLUTION)]
            if got != N_PER_STRATUM:
                errors.append(f"{benchmark}/nearmiss/{spec['level']}: "
                              f"{got} != {N_PER_STRATUM}")
    unrendered = [r["injection_id"] for r in rows if not r["text"]]
    if unrendered:
        errors.append(f"{len(unrendered)} unrendered rows (first: {unrendered[:3]})")
    # FM-4: no probe question contains a sampled value (len >= 4) FOR ITS OWN
    # KEY — scoped per (question_id, entity, attribute), the frozen v2 lint
    # precedent (inject_v2.lint_v2: "a commute-mode value 'bike' appearing
    # inside the bike_lock_brand probe text leaks nothing about either key's
    # answer — the global cross-product check false-positived on exactly
    # that"). fact_value counts too: it is the mixed rows' in-sentence answer.
    values_by_key: dict[tuple, list[str]] = {}
    for r in rows:
        kk = (r["question_id"], r["entity"], r["attribute"])
        for field in ("value", "fact_value"):
            v = r.get(field)
            if v and len(str(v)) >= 4:
                values_by_key.setdefault(kk, []).append(str(v))
    for k in keys:
        pq = k["probe_question"].lower()
        if "{" in pq:
            errors.append(f"unsubstituted probe template for {k['case_id']}")
        kk = (k["question_id"], k["entity"], k["attribute"])
        for v in values_by_key.get(kk, []):
            if v.lower() in pq:
                errors.append(f"FM-4 leak: probe for {k['case_id']} contains {v!r}")
    for r in rows:
        if r["text"]:
            low = r["text"].lower()
            if any(b.lower() in low for b in BLACKLIST + ROUND2_TELL_BLACKLIST):
                errors.append(f"{r['injection_id']}: blacklist phrase in text")
    return errors


def cmd_freeze() -> None:
    rows = _read_jsonl(MANIFEST_PATH)
    keys = _read_jsonl(KEYS_PATH)
    errors = lint_manifest(rows, keys)
    if errors:
        for e in errors:
            print("ERROR", e)
        raise SystemExit(f"{len(errors)} lint error(s) — not freezing")
    digest = _sha256(MANIFEST_PATH.read_text())
    FREEZE_SIDECAR_PATH.write_text(digest)
    print(f"froze {len(rows)} rows: sha256 {digest} -> {_rel(FREEZE_SIDECAR_PATH)}")


def assert_manifest_frozen() -> str:
    assert MANIFEST_PATH.exists(), f"missing {MANIFEST_PATH}"
    assert FREEZE_SIDECAR_PATH.exists(), (
        f"missing freeze sidecar {FREEZE_SIDECAR_PATH} — run `freeze` first")
    digest = _sha256(MANIFEST_PATH.read_text())
    frozen = FREEZE_SIDECAR_PATH.read_text().strip()
    assert digest == frozen, "stimuli manifest modified after freeze (sha256 mismatch)"
    return digest


# ---------------------------------------------------------------------------
# run — the four-cell micro-replays (extraction cache-aware)
# ---------------------------------------------------------------------------

def _candidates(benchmark: str, qid: str,
                m2_key: store.CanonicalKey) -> list[store.CanonicalKey]:
    """The mapper's candidate list: the substrate's frozen v2 candidates
    (blindness preserved) + the M2 key appended last."""
    if benchmark == "lme":
        entry = driver.lme_ku_entries().get(qid)
        aux = driver.v2_aux_keys("lme", qid)
        base = (store.build_candidate_keys(entry, aux)
                if entry is not None else list(aux))
    else:
        base = driver.locomo_candidates(qid)
    return base + [m2_key]


def _m2_key(key_row: dict) -> store.CanonicalKey:
    return store.CanonicalKey(entity=key_row["entity"],
                              attribute=key_row["attribute"],
                              description=key_row["description"],
                              probe_question=key_row["probe_question"])


async def _extract_turn(benchmark: str, family: str, world, turns: list[dict],
                        pos: int, tag: str) -> list:
    """One extraction call at ``turns[pos]`` under the family's frozen prompt
    — byte-identical to the cells' pathway (LME B2 full-prefix rendering;
    LoCoMo dyadic single-pass), so replays across policies/worlds are llm.py
    cache hits."""
    if benchmark == "lme":
        prior = [{"role": t["role"], "content": t["content"], "date": t["date"]}
                 for t in turns[:pos]]
        return await gated_store.extract_turn_family(
            prior, turns[pos]["content"], tag,
            system_prompt=(PROMPTS_DIR / syn_res.LME_PROMPTS[family]).read_text(),
            current_date=turns[pos]["date"])
    si, ti = turns[pos]["session_idx"], turns[pos]["turn_idx"]
    job = locomo_render.dyadic_turn_job(world, si, ti,
                                        family=syn_res.DYADIC_FAMILY[family])
    return await locomo_render.extract_dyadic_turn(
        job["prior_turns"], job["current_turn"], tag,
        family=syn_res.DYADIC_FAMILY[family], current_date=job["current_date"])


async def _extract_and_map(benchmark: str, family: str, world, turns, pos,
                           candidates, case_id: str, mapper=None) -> dict:
    """(tuples, mapped) for one turn — shared by BOTH policies of a family
    (in-memory here; byte-identical llm calls across worlds hit the cache).

    ``mapper`` (the m4_eval 5th-arm seam, default None = the frozen v2 mapper
    ``store._map_tuple_key``): an optional async override with the same
    ``(tup, candidates, *, trace_tag)`` convention. When None the mapping call
    is byte-identical to the frozen cells (cache-shared). The em_cg_fixed arm
    passes the M1 stem-wrapper here: its FIRST pass renders the byte-identical
    prompt (cache hit against the frozen em run) and only the `-planstem`
    re-offer on a DECLINED, plan-framed tuple is fresh. The extraction call is
    identical either way. Downstream, gated_store._unpack_map_result accepts
    both the frozen 3-tuple and the wrapper's 4-tuple (with meta)."""
    si, ti = turns[pos]["session_idx"], turns[pos]["turn_idx"]
    tag = f"m2-extract-{family}-{case_id}-s{si}t{ti}"
    tuples = await _extract_turn(benchmark, family, world, turns, pos, tag)
    map_call = store._map_tuple_key if mapper is None else mapper
    mapped = list(await asyncio.gather(*[
        map_call(tup, candidates,
                 trace_tag=f"m2-keymap-{family}-{case_id}-s{si}t{ti}-{j}")
        for j, tup in enumerate(tuples)]))
    return {"pos": pos, "session_idx": si, "turn_idx": ti,
            "date": turns[pos]["date"], "turn_text": turns[pos]["content"],
            "tuples": tuples, "mapped": mapped}


class _CellSim:
    """One cell's store state over a bounded world (micro-replay)."""

    def __init__(self, policy: str, benchmark: str):
        self.policy = policy
        self.benchmark = benchmark
        self.stores: dict[str, ZepStore] = {}
        self.pending: dict[str, list[dict]] = {}
        self.staged_rows: dict[str, dict] = {}
        self.gate_rows: list[dict] = []
        self.checker_rows: list[dict] = []
        self.diag = defaultdict(int)

    def route(self, key: tuple) -> str:
        return key[0] if self.benchmark == "locomo" else DEFAULT_STORE

    def zstore(self, store_id: str) -> ZepStore:
        if store_id not in self.stores:
            self.stores[store_id] = ZepStore()
        return self.stores[store_id]

    def lookup(self, key: tuple):
        return self.zstore(self.route(key)).lookup(key)


async def _gated_checker(sim: _CellSim, rec: dict, family: str,
                         case_id: str, checker_system: str,
                         turn_offset: int | None) -> None:
    """Checker over pending items on one turn (frozen fire semantics:
    consume-on-fire; valid promote commits the hypothesized value)."""
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
            trace_tag=f"m2-check-{family}-{case_id}-{store_id}-"
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
                        family: str, case_id: str, gen_system: str) -> None:
    """Phase-3 gate loop body for one turn (frozen argmax routing verbatim)."""
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
            "item_id": f"m2-{family}-{case_id}-s{rec['session_idx']}"
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
            staged_id = f"stg-m2-{family}-{case_id}-s{rec['session_idx']}" \
                        f"t{rec['turn_idx']}-{j}"
            row_shim = {"pos": rec["pos"], "turn_text": rec["turn_text"]}
            staged = await gated_store._generate_predicates(
                stream, row_shim, cand, tup, staged_id=staged_id,
                stream_id=case_id, family=family, gen_system=gen_system,
                diagnostics=defaultdict(int))
            staged["status"] = "pending"
            staged["store_id"] = store_id
            sim.staged_rows[staged_id] = staged
            sim.pending.setdefault(store_id, []).append(staged)
            gate_row["action"] = "staged"
            gate_row["staged_id"] = staged_id
        else:  # additive: commit-if-absent-or-equal, never a supersession
            new_stored = apply_sentinel(tup.value)
            if stored_now is None or (store.normalize_value(new_stored)
                                      == store.normalize_value(stored_now)):
                event = zstore.write(cand.key, tup.value, valid_from=rec["date"],
                                     provenance={"class": cls}, correction=False)
                gate_row["action"] = f"additive_{event['action']}"
            else:
                gate_row["action"] = "additive_noop"
        sim.gate_rows.append(gate_row)


async def _ungated_tuples(sim: _CellSim, rec: dict) -> None:
    for tup, mres in zip(rec["tuples"], rec["mapped"]):
        cand, _raw, _rat, _meta = gated_store._unpack_map_result(mres)
        sim.diag["n_tuples"] += 1
        if cand is None:
            sim.diag["n_unmapped"] += 1
            continue
        sim.diag["n_mapped"] += 1
        store_id = sim.route(cand.key)
        event = sim.zstore(store_id).write(cand.key, tup.value,
                                           valid_from=rec["date"],
                                           provenance={"class": None},
                                           correction=False)
        sim.gate_rows.append({"session_idx": rec["session_idx"],
                              "turn_idx": rec["turn_idx"], "store_id": store_id,
                              "key": cand.name, "value": tup.value,
                              "class": None, "action": event["action"]})


def _target_diagnostics(rec: dict, stim_row: dict,
                        target: store.CanonicalKey) -> dict:
    """Extraction-hit + mapping diagnostics for the stimulus turn (the M1
    mapping-loss axis, reported per level — never a bar)."""
    tuples = [t.__dict__ if hasattr(t, "__dict__") else dict(t)
              for t in rec["tuples"]]
    matched, method = natres.match_staged_tuple(
        tuples, stim_row["text"],
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


def _cell_summary(sim: _CellSim, target: store.CanonicalKey) -> dict:
    final = sim.lookup(target.key)
    staged = [sid for sid, srow in sim.staged_rows.items()
              if (srow["entity"], srow["attribute"]) == target.key]
    target_rows = [g for g in sim.gate_rows if g["key"] == target.name]
    return {
        "final_value": final,
        "final_present": final is not None and final != UNKNOWN_SENTINEL,
        "target_staged": bool(staged),
        "staged_ids": staged,
        "staged_status": {sid: sim.staged_rows[sid]["status"] for sid in staged},
        "target_gate_rows": target_rows,
        "diag": dict(sim.diag),
    }


def _rows_by_case(rows: list[dict]) -> dict[str, list[dict]]:
    by_case: dict[str, list[dict]] = {}
    for r in rows:
        by_case.setdefault(r["case_id"], []).append(r)
    return by_case


def case_path(benchmark: str, case_id: str, out_root: Path = CASES_DIR) -> Path:
    return out_root / benchmark / f"{case_id}.json"


async def run_case_simple(case_rows: list[dict], key_row: dict,
                          families: tuple[str, ...]) -> dict:
    """Gradient / mixed case: world = [establish?, stimulus]; all four cells."""
    stim = next(r for r in case_rows if r["inj_type"] == TYPE_STIMULUS)
    benchmark, qid, case_id = stim["benchmark"], stim["question_id"], stim["case_id"]
    world_rows = [r for r in case_rows if r["inj_type"] in (TYPE_ESTABLISH,
                                                            TYPE_STIMULUS)]
    for r in world_rows:
        assert r["text"], f"{r['injection_id']}: not rendered"
    world = syn_res.build_world(benchmark, qid, world_rows)
    turns = syn_res.world_turns(benchmark, world)
    positions = sorted(syn_res.staging_pos(turns, r["placement"])
                       for r in world_rows)
    stim_pos = syn_res.staging_pos(turns, stim["placement"])
    target = _m2_key(key_row)
    gen_system = gated_store.GENERATOR_PROMPT_PATH.read_text()
    checker_system = gated_store.CHECKER_PROMPT_PATH.read_text()

    out = {"case_id": case_id, "benchmark": benchmark, "stratum": stim["stratum"],
           "level": stim["level"], "rank": stim["rank"], "question_id": qid,
           "cluster": stim["cluster"], "target_key": target.name,
           "planted_value": stim["value"], "fact_value": stim.get("fact_value"),
           "established_value": next((r["value"] for r in world_rows
                                      if r["inj_type"] == TYPE_ESTABLISH), None),
           "designed_gold_class": stim["designed_gold_class"],
           "designed_gold_route": stim["designed_gold_route"],
           "taxonomy_rule": stim["taxonomy_rule"],
           "cells": {}, "diagnostics": {}}

    for family in families:
        candidates = _candidates(benchmark, qid, target)
        recs = [await _extract_and_map(benchmark, family, world, turns, pos,
                                       candidates, case_id)
                for pos in positions]
        for pos, rec in zip(positions, recs):
            rec["pos"] = pos
        stim_rec = next(r for r in recs if r["pos"] == stim_pos)
        out["diagnostics"][family] = _target_diagnostics(stim_rec, stim, target)
        for policy in POLICIES:
            sim = _CellSim(policy, benchmark)
            for rec in recs:
                if policy == "gated":
                    await _gated_checker(sim, rec, family, case_id,
                                         checker_system, None)
                    await _gated_tuples(sim, rec, turns, family, case_id,
                                        gen_system)
                else:
                    await _ungated_tuples(sim, rec)
            out["cells"][f"{family}_{'cg' if policy == 'gated' else 'cw'}"] = \
                _cell_summary(sim, target)
    return out


async def run_case_nearmiss(case_rows: list[dict], key_row: dict,
                            families: tuple[str, ...]) -> dict:
    """Near-miss case: staging world + one world per paraphrase level; the
    checker replays the pinned tail [t*, t*+2]; both policies also ingest the
    tail turns (the write-through cell's own resolution tracking)."""
    staging = next(r for r in case_rows if r["inj_type"] == TYPE_STIMULUS)
    resolutions = sorted((r for r in case_rows
                          if r["inj_type"] == TYPE_RESOLUTION),
                         key=lambda r: r["rank"])
    benchmark, qid, case_id = (staging["benchmark"], staging["question_id"],
                               staging["case_id"])
    for r in [staging] + resolutions:
        assert r["text"], f"{r['injection_id']}: not rendered"
    target = _m2_key(key_row)
    gen_system = gated_store.GENERATOR_PROMPT_PATH.read_text()
    checker_system = gated_store.CHECKER_PROMPT_PATH.read_text()

    out = {"case_id": case_id, "benchmark": benchmark, "stratum": "nearmiss",
           "question_id": qid, "cluster": staging["cluster"],
           "target_key": target.name, "planted_value": staging["value"],
           "staging": {"cells": {}, "diagnostics": {}}, "levels": {}}

    for family in families:
        candidates = _candidates(benchmark, qid, target)
        # --- staging world (contamination + staging endpoints) --------------
        world_s = syn_res.build_world(benchmark, qid, [staging])
        turns_s = syn_res.world_turns(benchmark, world_s)
        s_pos = syn_res.staging_pos(turns_s, staging["placement"])
        staging_sha = _sha256(turns_s[s_pos]["content"])
        rec_s = await _extract_and_map(benchmark, family, world_s, turns_s,
                                       s_pos, candidates, case_id)
        rec_s["pos"] = s_pos
        out["staging"]["diagnostics"][family] = _target_diagnostics(
            rec_s, staging, target)
        for policy in POLICIES:
            sim = _CellSim(policy, benchmark)
            if policy == "gated":
                await _gated_tuples(sim, rec_s, turns_s, family, case_id,
                                    gen_system)
            else:
                await _ungated_tuples(sim, rec_s)
            out["staging"]["cells"][f"{family}_{'cg' if policy == 'gated' else 'cw'}"] = \
                _cell_summary(sim, target)

        # --- per paraphrase level: staging + resolution world ---------------
        for res in resolutions:
            world_v = syn_res.build_world(benchmark, qid, [staging, res])
            turns_v = syn_res.world_turns(benchmark, world_v)
            s_pos_v = syn_res.staging_pos(turns_v, staging["placement"])
            # byte-identity of the staging turn across worlds -> cache hits
            assert _sha256(turns_v[s_pos_v]["content"]) == staging_sha, (
                f"{case_id}/{res['level']}: staging turn drift across worlds")
            cont = syn_res.continuation_user_turns(turns_v, s_pos_v)
            t_star = syn_res.staging_pos(cont, res["placement"])
            assert res["text"] in cont[t_star]["content"], (
                f"{case_id}/{res['level']}: resolution text not in t*")
            tail = cont[t_star:t_star + NEARMISS_TAIL]
            tail_positions = [syn_res.staging_pos(turns_v, {
                "session_idx": t["session_idx"], "turn_idx": t["turn_idx"]})
                for t in tail]
            recs = [await _extract_and_map(benchmark, family, world_v, turns_v,
                                           s_pos_v, candidates, case_id)]
            recs[0]["pos"] = s_pos_v
            for pos in tail_positions:
                rec = await _extract_and_map(benchmark, family, world_v,
                                             turns_v, pos, candidates, case_id)
                rec["pos"] = pos
                recs.append(rec)
            level_out = out["levels"].setdefault(res["level"], {
                "rank": res["rank"], "cells": {}})
            for policy in POLICIES:
                sim = _CellSim(policy, benchmark)
                for i, rec in enumerate(recs):
                    offset = None if i == 0 else i - 1  # 0 = t*
                    if policy == "gated":
                        if i > 0:
                            await _gated_checker(sim, rec, family, case_id,
                                                 checker_system, offset)
                        await _gated_tuples(sim, rec, turns_v, family,
                                            case_id, gen_system)
                    else:
                        await _ungated_tuples(sim, rec)
                summary = _cell_summary(sim, target)
                fires = [(c["turn_offset"], f) for c in sim.checker_rows
                         for f in c["fires"] if not f["invalid"]]
                promote_fires = [(off, f) for off, f in fires
                                 if f["action"] == "promote"]
                summary["checker_rows"] = sim.checker_rows
                summary["promote_fire_at_tstar"] = any(off == 0 for off, _ in promote_fires)
                summary["promote_fire_within_window"] = any(
                    off is not None and off <= SENSITIVITY_WINDOW
                    for off, _ in promote_fires)
                level_out["cells"][f"{family}_{'cg' if policy == 'gated' else 'cw'}"] = summary
    return out


async def run(benchmarks: tuple[str, ...], families: tuple[str, ...],
              strata: tuple[str, ...], ids: list[str] | None,
              out_root: Path, guard) -> dict:
    rows = _read_jsonl(MANIFEST_PATH)
    keys = {k["case_id"]: k for k in _read_jsonl(KEYS_PATH)}
    by_case = _rows_by_case(rows)
    summary = {"n_cases": 0, "skipped": 0, "failed": []}
    guard.install()
    try:
        for case_id in sorted(by_case):
            case_rows = by_case[case_id]
            head = case_rows[0]
            if head["benchmark"] not in benchmarks:
                continue
            if head["stratum"] not in strata:
                continue
            if ids is not None and case_id not in ids:
                continue
            path = case_path(head["benchmark"], case_id, out_root)
            if path.exists():
                summary["skipped"] += 1
                continue
            runner = (run_case_nearmiss if head["stratum"] == "nearmiss"
                      else run_case_simple)
            record = await runner(case_rows, keys[case_id], families)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, ensure_ascii=False, indent=1))
            summary["n_cases"] += 1
    finally:
        guard.uninstall()
    summary["fresh_calls"] = guard.fresh
    summary["advisory"] = advisory_status()
    return summary


# ---------------------------------------------------------------------------
# run-fixed — the em_cg_fixed repair arm ONLY (REPAIR-COVERAGE prereg,
# 2026-07-09): sibling records, frozen case files never modified
# ---------------------------------------------------------------------------

def _fixed_mapper():
    """The M1 stem-wrapper as a LOCAL mapper (mapper_fix.make_mapper — the
    single frozen constructor over router_sweep.PLAN_FRAME_ATTRIBUTE_PATTERNS;
    import, never copy)."""
    mapper = mapper_fix.make_mapper()
    assert getattr(mapper, "mode", None) == "plan_stem_wrapper", (
        "wrapper constructor did not produce the plan_stem_wrapper mode")
    return mapper


def _fixed_mapping_meta(recs: list[dict],
                        target: store.CanonicalKey) -> list[dict]:
    """Per-turn wrapper mapping meta for em_cg_fixed — the entry funnel can
    be recomputed / the stem re-offer audited from the sibling record alone
    (the m4_eval._fixed_mapping_meta convention: original_attribute,
    derived_stem, pattern_matched, mapped_key per tuple)."""
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
                    "turn_idx": rec["turn_idx"], "tuples": tuples_meta})
    return out


async def run_case_simple_fixed(case_rows: list[dict], key_row: dict) -> dict:
    """Gradient / mixed case, em_cg_fixed arm ONLY: world = [establish?,
    stimulus]; the em family under the LOCAL stem-wrapper mapper, gated
    policy. Extraction + first-pass mapping replay from cache (byte-identical
    to the frozen em cells); the record matches the frozen em_cg cell
    structure + wrapper mapping meta."""
    stim = next(r for r in case_rows if r["inj_type"] == TYPE_STIMULUS)
    benchmark, qid, case_id = stim["benchmark"], stim["question_id"], stim["case_id"]
    world_rows = [r for r in case_rows if r["inj_type"] in (TYPE_ESTABLISH,
                                                            TYPE_STIMULUS)]
    for r in world_rows:
        assert r["text"], f"{r['injection_id']}: not rendered"
    world = syn_res.build_world(benchmark, qid, world_rows)
    turns = syn_res.world_turns(benchmark, world)
    positions = sorted(syn_res.staging_pos(turns, r["placement"])
                       for r in world_rows)
    stim_pos = syn_res.staging_pos(turns, stim["placement"])
    target = _m2_key(key_row)
    gen_system = gated_store.GENERATOR_PROMPT_PATH.read_text()
    checker_system = gated_store.CHECKER_PROMPT_PATH.read_text()

    assert_mapper_mode_off()  # global frozen mapper untouched entering arm
    mapper = _fixed_mapper()
    candidates = _candidates(benchmark, qid, target)
    recs = [await _extract_and_map(benchmark, "em", world, turns, pos,
                                   candidates, case_id, mapper=mapper)
            for pos in positions]
    for pos, rec in zip(positions, recs):
        rec["pos"] = pos
    stim_rec = next(r for r in recs if r["pos"] == stim_pos)

    sim = _CellSim("gated", benchmark)
    for rec in recs:
        await _gated_checker(sim, rec, "em", case_id, checker_system, None)
        await _gated_tuples(sim, rec, turns, "em", case_id, gen_system)
    summary = _cell_summary(sim, target)
    summary["mapping_meta"] = _fixed_mapping_meta(recs, target)
    assert_mapper_mode_off()  # teardown: local mapper, global still frozen

    return {"case_id": case_id, "benchmark": benchmark,
            "stratum": stim["stratum"], "level": stim["level"],
            "rank": stim["rank"], "question_id": qid,
            "cluster": stim["cluster"], "target_key": target.name,
            "planted_value": stim["value"], "fact_value": stim.get("fact_value"),
            "established_value": next((r["value"] for r in world_rows
                                       if r["inj_type"] == TYPE_ESTABLISH), None),
            "designed_gold_class": stim["designed_gold_class"],
            "designed_gold_route": stim["designed_gold_route"],
            "taxonomy_rule": stim["taxonomy_rule"],
            "arm": FIXED_CELL,
            "mapper": ("M1 stem wrapper (store.make_tuple_key_mapper, mode "
                       "'plan_stem_wrapper') as a LOCAL mapper for this arm "
                       "only — REPAIR-COVERAGE prereg 2026-07-09"),
            "cells": {FIXED_CELL: summary},
            "diagnostics": {FIXED_DIAG_FAMILY: _target_diagnostics(
                stim_rec, stim, target)}}


async def run_case_nearmiss_fixed(case_rows: list[dict], key_row: dict) -> dict:
    """Near-miss case, em_cg_fixed arm ONLY: staging world + one world per
    paraphrase level, em family under the LOCAL stem-wrapper mapper, gated
    policy; the frozen run_case_nearmiss replay geometry verbatim (pinned
    tail [t*, t*+2], test3 fire semantics)."""
    staging = next(r for r in case_rows if r["inj_type"] == TYPE_STIMULUS)
    resolutions = sorted((r for r in case_rows
                          if r["inj_type"] == TYPE_RESOLUTION),
                         key=lambda r: r["rank"])
    benchmark, qid, case_id = (staging["benchmark"], staging["question_id"],
                               staging["case_id"])
    for r in [staging] + resolutions:
        assert r["text"], f"{r['injection_id']}: not rendered"
    target = _m2_key(key_row)
    gen_system = gated_store.GENERATOR_PROMPT_PATH.read_text()
    checker_system = gated_store.CHECKER_PROMPT_PATH.read_text()

    assert_mapper_mode_off()
    mapper = _fixed_mapper()
    candidates = _candidates(benchmark, qid, target)

    out = {"case_id": case_id, "benchmark": benchmark, "stratum": "nearmiss",
           "question_id": qid, "cluster": staging["cluster"],
           "target_key": target.name, "planted_value": staging["value"],
           "arm": FIXED_CELL,
           "mapper": ("M1 stem wrapper (store.make_tuple_key_mapper, mode "
                      "'plan_stem_wrapper') as a LOCAL mapper for this arm "
                      "only — REPAIR-COVERAGE prereg 2026-07-09"),
           "staging": {"cells": {}, "diagnostics": {}}, "levels": {}}

    # --- staging world (contamination + staging endpoints) ------------------
    world_s = syn_res.build_world(benchmark, qid, [staging])
    turns_s = syn_res.world_turns(benchmark, world_s)
    s_pos = syn_res.staging_pos(turns_s, staging["placement"])
    staging_sha = _sha256(turns_s[s_pos]["content"])
    rec_s = await _extract_and_map(benchmark, "em", world_s, turns_s, s_pos,
                                   candidates, case_id, mapper=mapper)
    rec_s["pos"] = s_pos
    out["staging"]["diagnostics"][FIXED_DIAG_FAMILY] = _target_diagnostics(
        rec_s, staging, target)
    sim = _CellSim("gated", benchmark)
    await _gated_tuples(sim, rec_s, turns_s, "em", case_id, gen_system)
    s_summary = _cell_summary(sim, target)
    s_summary["mapping_meta"] = _fixed_mapping_meta([rec_s], target)
    out["staging"]["cells"][FIXED_CELL] = s_summary

    # --- per paraphrase level: staging + resolution world --------------------
    for res in resolutions:
        world_v = syn_res.build_world(benchmark, qid, [staging, res])
        turns_v = syn_res.world_turns(benchmark, world_v)
        s_pos_v = syn_res.staging_pos(turns_v, staging["placement"])
        # byte-identity of the staging turn across worlds -> cache hits
        assert _sha256(turns_v[s_pos_v]["content"]) == staging_sha, (
            f"{case_id}/{res['level']}: staging turn drift across worlds")
        cont = syn_res.continuation_user_turns(turns_v, s_pos_v)
        t_star = syn_res.staging_pos(cont, res["placement"])
        assert res["text"] in cont[t_star]["content"], (
            f"{case_id}/{res['level']}: resolution text not in t*")
        tail = cont[t_star:t_star + NEARMISS_TAIL]
        tail_positions = [syn_res.staging_pos(turns_v, {
            "session_idx": t["session_idx"], "turn_idx": t["turn_idx"]})
            for t in tail]
        recs = [await _extract_and_map(benchmark, "em", world_v, turns_v,
                                       s_pos_v, candidates, case_id,
                                       mapper=mapper)]
        recs[0]["pos"] = s_pos_v
        for pos in tail_positions:
            rec = await _extract_and_map(benchmark, "em", world_v, turns_v,
                                         pos, candidates, case_id,
                                         mapper=mapper)
            rec["pos"] = pos
            recs.append(rec)
        sim = _CellSim("gated", benchmark)
        for i, rec in enumerate(recs):
            offset = None if i == 0 else i - 1  # 0 = t*
            if i > 0:
                await _gated_checker(sim, rec, "em", case_id,
                                     checker_system, offset)
            await _gated_tuples(sim, rec, turns_v, "em", case_id, gen_system)
        summary = _cell_summary(sim, target)
        fires = [(c["turn_offset"], f) for c in sim.checker_rows
                 for f in c["fires"] if not f["invalid"]]
        promote_fires = [(off, f) for off, f in fires
                         if f["action"] == "promote"]
        summary["checker_rows"] = sim.checker_rows
        summary["promote_fire_at_tstar"] = any(off == 0 for off, _ in promote_fires)
        summary["promote_fire_within_window"] = any(
            off is not None and off <= SENSITIVITY_WINDOW
            for off, _ in promote_fires)
        summary["mapping_meta"] = _fixed_mapping_meta(recs, target)
        out["levels"][res["level"]] = {"rank": res["rank"],
                                       "cells": {FIXED_CELL: summary}}
    assert_mapper_mode_off()  # teardown: local mapper, global still frozen
    return out


async def run_fixed(benchmarks: tuple[str, ...], strata: tuple[str, ...],
                    ids: list[str] | None, out_root: Path, guard,
                    frozen_root: Path = CASES_DIR) -> dict:
    """Replay ONLY the em_cg_fixed arm for each selected case, writing a
    SIBLING record under ``out_root`` (default cases_fixed/). The frozen case
    record must already exist (the arm replays the frozen em extraction from
    cache) and is byte-checked untouched after each case; per-case failures
    are recorded and the run continues (resumable), except on budget/envelope
    aborts."""
    assert out_root.resolve() != frozen_root.resolve(), (
        "run-fixed must write SIBLING records — out_root == frozen cases dir")
    assert_mapper_mode_off()
    rows = _read_jsonl(MANIFEST_PATH)
    keys = {k["case_id"]: k for k in _read_jsonl(KEYS_PATH)}
    by_case = _rows_by_case(rows)
    summary = {"arm": FIXED_CELL, "n_cases": 0, "skipped": 0, "failed": []}
    guard.install()
    try:
        for case_id in sorted(by_case):
            case_rows = by_case[case_id]
            head = case_rows[0]
            if head["benchmark"] not in benchmarks:
                continue
            if head["stratum"] not in strata:
                continue
            if ids is not None and case_id not in ids:
                continue
            path = case_path(head["benchmark"], case_id, out_root)
            if path.exists():
                summary["skipped"] += 1
                continue
            frozen_path = case_path(head["benchmark"], case_id, frozen_root)
            assert frozen_path.exists(), (
                f"{case_id}: frozen case record missing at {frozen_path} — "
                "the repair arm replays the frozen em run from cache; run "
                "the four-cell `run` first")
            frozen_bytes = frozen_path.read_bytes()
            frozen_rec = json.loads(frozen_bytes)
            try:
                runner = (run_case_nearmiss_fixed
                          if head["stratum"] == "nearmiss"
                          else run_case_simple_fixed)
                record = await runner(case_rows, keys[case_id])
                # sibling-consistency: same case identity as the frozen record
                assert record["planted_value"] == frozen_rec["planted_value"], (
                    f"{case_id}: planted_value drift vs frozen record")
                assert record["target_key"] == frozen_rec["target_key"], (
                    f"{case_id}: target_key drift vs frozen record")
            except Exception as exc:  # noqa: BLE001 — surfaced + resumable
                summary["failed"].append(f"{case_id}: {exc}")
                low = str(exc).lower()
                if "budget" in low or "envelope" in low:
                    break  # hard stop: no point scheduling further cases
                continue
            # the frozen case file is NEVER modified (byte-checked)
            assert frozen_path.read_bytes() == frozen_bytes, (
                f"{case_id}: frozen case record changed during run-fixed")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, ensure_ascii=False, indent=1))
            summary["n_cases"] += 1
    finally:
        guard.uninstall()
        assert_mapper_mode_off()
    summary["fresh_calls"] = guard.fresh
    summary["advisory"] = advisory_status()
    return summary


# ---------------------------------------------------------------------------
# score — per-stratum difficulty curves (no bars)
# ---------------------------------------------------------------------------

def _load_cases(out_root: Path = CASES_DIR) -> list[dict]:
    out = []
    for benchmark in BENCHMARKS:
        d = out_root / benchmark
        if not d.exists():
            continue
        for p in sorted(d.glob("*.json")):
            out.append(json.loads(p.read_text()))
    return out


def merge_fixed_arm(cases: list[dict], fixed_root: Path | None) -> int:
    """Merge em_cg_fixed SIBLING records (run-fixed output) into the loaded
    case dicts IN MEMORY — the frozen case files stay untouched on disk, and
    the frozen cells' entries in each dict are never written to (regression:
    frozen scored numbers identical with/without the arm present). Returns
    the number of cases that gained the arm."""
    if fixed_root is None:
        return 0
    fixed = {c["case_id"]: c for c in _load_cases(fixed_root)}
    merged = 0
    for case in cases:
        f = fixed.get(case["case_id"])
        if f is None:
            continue
        assert f.get("arm") == FIXED_CELL, (
            f"{case['case_id']}: sibling record carries arm {f.get('arm')!r}")
        assert f["planted_value"] == case["planted_value"], (
            f"{case['case_id']}: sibling planted_value drift")
        assert f["target_key"] == case["target_key"], (
            f"{case['case_id']}: sibling target_key drift")
        if case["stratum"] == "nearmiss":
            case["staging"]["cells"][FIXED_CELL] = \
                f["staging"]["cells"][FIXED_CELL]
            case["staging"]["diagnostics"][FIXED_DIAG_FAMILY] = \
                f["staging"]["diagnostics"][FIXED_DIAG_FAMILY]
            for level, lv in f["levels"].items():
                case["levels"][level]["cells"][FIXED_CELL] = \
                    lv["cells"][FIXED_CELL]
        else:
            case["cells"][FIXED_CELL] = f["cells"][FIXED_CELL]
            case["diagnostics"][FIXED_DIAG_FAMILY] = \
                f["diagnostics"][FIXED_DIAG_FAMILY]
        merged += 1
    return merged


def _stem_wrapper_stats(fixed_root: Path | None) -> dict:
    """Aggregate stem re-offer audit rows from the sibling records' wrapper
    mapping meta (per benchmark x stratum; descriptive, never a bar)."""
    stats_out: dict = {}
    if fixed_root is None:
        return stats_out

    def _consume(meta_rows, bucket):
        for turn in meta_rows:
            for t in turn["tuples"]:
                if t["stem_mapped"] is None:
                    continue  # no re-offer attempted for this tuple
                bucket["n_stem_reoffers"] += 1
                if t["stem_mapped"]:
                    bucket["n_stem_mapped"] += 1
                    if t["mapped_to_target"]:
                        bucket["n_stem_mapped_to_target"] += 1

    for c in _load_cases(fixed_root):
        key = f"{c['benchmark']}/{c['stratum']}"
        bucket = stats_out.setdefault(key, {
            "n_cases": 0, "n_stem_reoffers": 0, "n_stem_mapped": 0,
            "n_stem_mapped_to_target": 0})
        bucket["n_cases"] += 1
        if c["stratum"] == "nearmiss":
            _consume(c["staging"]["cells"][FIXED_CELL]["mapping_meta"], bucket)
            for lv in c["levels"].values():
                _consume(lv["cells"][FIXED_CELL]["mapping_meta"], bucket)
        else:
            _consume(c["cells"][FIXED_CELL]["mapping_meta"], bucket)
    return stats_out


def _curve_point(values: list[int], clusters: list[str]) -> dict:
    x = sum(values)
    n = len(values)
    if not n:
        return {"x": 0, "n": 0, "point": None, "ci": None,
                "ci_kind": "cluster_bootstrap"}
    boot = stats.cluster_bootstrap_rate(values, clusters, seed=SEED_M2)
    return {"x": x, "n": n, "point": boot["point"],
            "ci": [boot["lo"], boot["hi"]], "ci_kind": "cluster_bootstrap",
            "n_clusters": boot["n_clusters"]}


async def compute_scores(out_root: Path = CASES_DIR,
                         fixed_root: Path | None = CASES_FIXED_DIR) -> dict:
    cases = _load_cases(out_root)
    assert cases, f"no case records under {out_root} — run `run` first"
    n_fixed = merge_fixed_arm(cases, fixed_root)
    scorer = rule5.EquivalenceScorer(cap=4000)

    async def contaminated(case: dict, cell_summary: dict, value) -> int:
        eq, _m = await scorer.equivalent(
            case["target_key"].partition("/")[2],
            cell_summary["final_value"], value)
        return int(eq)

    gradient_curves: dict = {}
    mixed_block: dict = {}
    nearmiss_curves: dict = {}
    staging_curves: dict = {}
    diagnostics: dict = {}
    #: per-case contamination flags for the adoption criterion, keyed
    #: (level, cell) -> list of (benchmark, y) — pooled ACROSS benchmarks
    #: (n=60/block, the prereg's 1/60 arithmetic).
    contam_flags: dict[tuple, list[tuple]] = defaultdict(list)

    def _diag_rows(key: str, diags: list[dict]) -> None:
        """Funnel diagnostics per family (em/ez + em_fixed where present),
        incl. the stem-RECOVERED mapping count for em_fixed (mapped by the
        wrapper where the frozen em mapping declined)."""
        for fam in FAMILIES + (FIXED_DIAG_FAMILY,):
            dsel = [d[fam] for d in diags if fam in d]
            if not dsel:
                continue
            row = {"target_extracted": sum(d["target_extracted"] for d in dsel),
                   "target_mapped": sum(d["any_tuple_mapped_to_target"]
                                        for d in dsel),
                   "n": len(dsel)}
            if fam == FIXED_DIAG_FAMILY:
                pairs = [(d["em"], d[FIXED_DIAG_FAMILY]) for d in diags
                         if "em" in d and FIXED_DIAG_FAMILY in d]
                row["stem_recovered_vs_em"] = sum(
                    int(dfx["any_tuple_mapped_to_target"]
                        and not dem["any_tuple_mapped_to_target"])
                    for dem, dfx in pairs)
            diagnostics[f"{key}/{fam}"] = row

    for benchmark in BENCHMARKS:
        bcases = [c for c in cases if c["benchmark"] == benchmark]

        # --- gradient + mixed: contamination / staging per level per cell ---
        for stratum, level_specs in (("gradient", GRADIENT_LEVELS),
                                     ("mixed", (MIXED_LEVEL,))):
            for spec in level_specs:
                sel = [c for c in bcases if c["stratum"] == stratum
                       and c["level"] == spec["level"]]
                if not sel:
                    continue
                key = f"{benchmark}/{spec['level']}"
                bucket = {"rank": spec["rank"],
                          "designed_gold_class": spec["designed_gold_class"],
                          "designed_gold_route": spec["designed_gold_route"],
                          "taxonomy_rule": spec.get("taxonomy_rule"),
                          "cells": {}}
                for cell in SCORED_CELLS:
                    csel = [c for c in sel if cell in c["cells"]]
                    if not csel:
                        continue  # the repair arm scores where its record exists
                    clusters = [c["cluster"] for c in csel]
                    ys = [await contaminated(c, c["cells"][cell],
                                             c["planted_value"]) for c in csel]
                    contam_flags[(spec["level"], cell)] += [
                        (benchmark, y) for y in ys]
                    bucket["cells"][cell] = {
                        "contamination": _curve_point(ys, clusters)}
                    if cell in GATED_CELLS_EXT:
                        st = [int(c["cells"][cell]["target_staged"]) for c in csel]
                        bucket["cells"][cell]["staging"] = _curve_point(st, clusters)
                    if stratum == "mixed":
                        fh = [await contaminated(c, c["cells"][cell],
                                                 c.get("fact_value")) for c in csel]
                        bucket["cells"][cell]["fact_retention"] = _curve_point(
                            fh, clusters)
                (gradient_curves if stratum == "gradient" else mixed_block)[key] = bucket
                _diag_rows(key, [c.get("diagnostics", {}) for c in sel])

        # --- near-miss: staging-turn endpoints + promotion curves ------------
        nm = [c for c in bcases if c["stratum"] == "nearmiss"]
        if nm:
            skey = f"{benchmark}/staging"
            staging_curves[skey] = {"cells": {}}
            for cell in SCORED_CELLS:
                csel = [c for c in nm if cell in c["staging"]["cells"]]
                if not csel:
                    continue
                clusters = [c["cluster"] for c in csel]
                ys = [await contaminated(c, c["staging"]["cells"][cell],
                                         c["planted_value"]) for c in csel]
                contam_flags[("staging", cell)] += [
                    (benchmark, y) for y in ys]
                staging_curves[skey]["cells"][cell] = {
                    "contamination": _curve_point(ys, clusters)}
                if cell in GATED_CELLS_EXT:
                    st = [int(c["staging"]["cells"][cell]["target_staged"])
                          for c in csel]
                    staging_curves[skey]["cells"][cell]["staging"] = \
                        _curve_point(st, clusters)
            _diag_rows(skey, [c["staging"].get("diagnostics", {})
                              for c in nm])
            for spec in NEARMISS_LEVELS:
                sel = [c for c in nm if spec["level"] in c.get("levels", {})]
                if not sel:
                    continue
                key = f"{benchmark}/{spec['level']}"
                bucket = {"rank": spec["rank"], "cells": {}}
                for cell in SCORED_CELLS:
                    csel = [c for c in sel
                            if cell in c["levels"][spec["level"]]["cells"]]
                    if not csel:
                        continue
                    clusters = [c["cluster"] for c in csel]
                    cells = [c["levels"][spec["level"]]["cells"][cell]
                             for c in csel]
                    end_state = [await contaminated(c, cc, c["planted_value"])
                                 for c, cc in zip(csel, cells)]
                    entry = {"promoted_end_state_ITT": _curve_point(end_state,
                                                                    clusters)}
                    if cell in GATED_CELLS_EXT:
                        prim = [int(cc.get("promote_fire_at_tstar", False))
                                for cc in cells]
                        wind = [int(cc.get("promote_fire_within_window", False))
                                for cc in cells]
                        entry["promote_fire_at_tstar_ITT"] = _curve_point(
                            prim, clusters)
                        entry["promote_fire_within_window_ITT"] = _curve_point(
                            wind, clusters)
                        staged_sel = [
                            (c, cc, w) for c, cc, w in zip(csel, cells, wind)
                            if c["staging"]["cells"][cell]["target_staged"]]
                        entry["conditional_on_staged"] = {
                            "n": len(staged_sel),
                            "promote_fire_within_window": _curve_point(
                                [w for _c, _cc, w in staged_sel],
                                [c["cluster"] for c, _cc, _w in staged_sel]),
                        } if staged_sel else {"n": 0}
                    bucket["cells"][cell] = entry
                nearmiss_curves[key] = bucket

    out = {
        "id": "V2.4-M2-RESULTS",
        "seed": SEED_M2,
        "n_cases_scored": len(cases),
        "endpoint_note": ("difficulty curves only — no go/no-go bars "
                          "(anti-saturation directive, Protocol v2.4 M2)"),
        "mapper": "frozen v2 key mapper, mode OFF (M1 stem wrapper not installed)",
        "gradient_curves": gradient_curves,
        "mixed": mixed_block,
        "nearmiss_staging": staging_curves,
        "nearmiss_promotion_curves": nearmiss_curves,
        "target_pipeline_diagnostics": diagnostics,
        "scoring_judge_calls": {"total": scorer.judge_calls,
                                "fresh": scorer.fresh_judge_calls},
        "advisory_budget": advisory_status(),
    }
    if n_fixed:
        out["arms"] = list(SCORED_CELLS)
        out["n_cases_with_fixed_arm"] = n_fixed
        out["mapper_fixed_arm"] = (
            "em_cg_fixed: M1 stem wrapper (store.make_tuple_key_mapper over "
            "router_sweep.PLAN_FRAME_ATTRIBUTE_PATTERNS, mode "
            "'plan_stem_wrapper') installed as a LOCAL mapper for that arm "
            "only — REPAIR-COVERAGE prereg 2026-07-09; the four frozen "
            "cells' numbers are byte-identical with or without the arm")
        out["adoption_criterion"] = adoption_criterion(contam_flags)
        out["stem_wrapper_reoffers"] = _stem_wrapper_stats(fixed_root)
    return out


def adoption_criterion(contam_flags: dict[tuple, list[tuple]]) -> dict:
    """The preregistered ADOPTION CRITERION (REPAIR-COVERAGE, 2026-07-09,
    owner-approved): adopt the repair as the default cautious-gated
    configuration iff em_cg_fixed's planted-probe contamination is not worse
    than em_cg's by more than ADOPTION_THRESHOLD_PP on any gradient level or
    mixed block. Blocks are POOLED across benchmarks (n=60 — the prereg's
    "one arc = 1/60 ~ 1.7pp" arithmetic); per-benchmark deltas are shown
    descriptively. g5 (designed-gold completed_event: a commit is CORRECT per
    taxonomy rule (d)) and the near-miss staging turn are DESCRIPTIVE blocks
    — g5's would-it-change-the-verdict line is reported explicitly."""
    all_levels = ([s["level"] for s in GRADIENT_LEVELS]
                  + [MIXED_LEVEL["level"]] + ["staging"])
    blocks: dict = {}
    binding_trips: list[str] = []
    g5_trips = False
    evaluable = True
    for level in all_levels:
        base = contam_flags.get((level, "em_cg"), [])
        fix = contam_flags.get((level, FIXED_CELL), [])
        binding = level in ADOPTION_BINDING_LEVELS

        def _rates(rows):
            per_b = {}
            for b in BENCHMARKS:
                ys = [y for bb, y in rows if bb == b]
                per_b[b] = {"x": sum(ys), "n": len(ys),
                            "rate": (sum(ys) / len(ys)) if ys else None}
            ys = [y for _bb, y in rows]
            pooled = {"x": sum(ys), "n": len(ys),
                      "rate": (sum(ys) / len(ys)) if ys else None}
            return pooled, per_b

        b_pooled, b_per = _rates(base)
        f_pooled, f_per = _rates(fix)
        delta_pp = (None if b_pooled["rate"] is None or f_pooled["rate"] is None
                    else (f_pooled["rate"] - b_pooled["rate"]) * 100.0)
        exceeds = delta_pp is not None and delta_pp > ADOPTION_THRESHOLD_PP
        blocks[level] = {
            "binding": binding,
            "pooled": {"em_cg": b_pooled, "em_cg_fixed": f_pooled,
                       "delta_pp": delta_pp,
                       "exceeds_threshold": exceeds},
            "per_benchmark": {
                b: {"em_cg": b_per[b], "em_cg_fixed": f_per[b],
                    "delta_pp": (None if b_per[b]["rate"] is None
                                 or f_per[b]["rate"] is None
                                 else (f_per[b]["rate"] - b_per[b]["rate"]) * 100.0)}
                for b in BENCHMARKS},
        }
        if level == "g5_present_tense_change":
            blocks[level]["note"] = (
                "descriptive — g5's per-level endpoint is a commit of the "
                "planted value, designed-gold CORRECT (taxonomy rule (d)); "
                "not a contamination block")
            g5_trips = exceeds
        elif level == "staging":
            blocks[level]["note"] = (
                "descriptive — near-miss staging-turn contamination; the "
                "prereg names gradient levels + the mixed block only")
        if binding:
            if (delta_pp is None or f_pooled["n"] == 0
                    or f_pooled["n"] != b_pooled["n"]):
                # partial em_cg_fixed coverage on a binding block (e.g. a
                # smoke subset) must never produce a silent verdict — the
                # prereg's 1/60 arithmetic assumes complete blocks
                evaluable = False
            elif exceeds:
                binding_trips.append(level)
    if not evaluable:
        verdict = "NOT EVALUABLE (missing em_cg_fixed data on a binding block)"
    else:
        verdict = "DO NOT ADOPT" if binding_trips else "ADOPT"
    return {
        "rule": ("REPAIR-COVERAGE prereg (2026-07-09, owner-approved): adopt "
                 "the repair as the default cautious-gated configuration iff "
                 "its planted-probe contamination is not worse than em_cg's "
                 f"by more than {ADOPTION_THRESHOLD_PP}pp on any gradient "
                 "level or mixed block (one arc = 1/60 ~ 1.7pp; threshold "
                 "allows one-case noise); near-miss promotion gains reported "
                 "either way"),
        "threshold_pp": ADOPTION_THRESHOLD_PP,
        "binding_blocks": list(ADOPTION_BINDING_LEVELS),
        "pooling": "blocks pooled across benchmarks (n=60/block); "
                   "per-benchmark deltas descriptive",
        "g5_interpretation": (
            "g5_present_tense_change excluded from the binding set: its "
            "endpoint measures the designed-gold-CORRECT commit (rule (d)), "
            "not contamination (frozen M2 design + released report)"),
        "blocks": blocks,
        "binding_trips": binding_trips,
        "verdict": verdict,
        "verdict_would_change_if_g5_bound": (
            evaluable and not binding_trips and g5_trips),
    }


def _fmt(pt: dict) -> str:
    if not pt or pt.get("point") is None:
        return "—"
    return (f"{pt['x']}/{pt['n']} = {pt['point']:.3f} "
            f"[{pt['ci'][0]:.3f}, {pt['ci'][1]:.3f}]")


def _cells_in(bucket: dict) -> list[str]:
    """Row order for a bucket's cells: the four frozen cells first (always
    all present in frozen records), then the repair arm where its sibling
    record was merged."""
    return [cell for cell in SCORED_CELLS if cell in bucket.get("cells", {})]


def _render_adoption(ad: dict) -> list[str]:
    lines = ["", "## ADOPTION CRITERION — em_cg_fixed vs em_cg "
             "(REPAIR-COVERAGE prereg, 2026-07-09; owner-approved)", "",
             ad["rule"], "",
             f"**VERDICT: {ad['verdict']}**"
             + (f" — binding blocks tripped: {ad['binding_trips']}"
                if ad["binding_trips"] else
                (" — no binding block worse by more than "
                 f"{ad['threshold_pp']}pp" if ad["verdict"] == "ADOPT" else "")),
             "",
             "Blocks pooled across benchmarks (n = 60/block, the prereg's "
             "1/60 arithmetic); per-benchmark deltas descriptive. "
             + ad["g5_interpretation"] + ".", "",
             "| block | binding | em_cg (pooled) | em_cg_fixed (pooled) "
             "| delta (pp) | > 2.5pp? |", "|---|---|---|---|---|---|"]

    def _r(d):
        if d["rate"] is None:
            return "—"
        return f"{d['x']}/{d['n']} = {d['rate']:.3f}"

    for level, blk in ad["blocks"].items():
        p = blk["pooled"]
        d = ("—" if p["delta_pp"] is None else f"{p['delta_pp']:+.2f}")
        lines.append(
            f"| {level} | {'BINDING' if blk['binding'] else 'descriptive'} "
            f"| {_r(p['em_cg'])} | {_r(p['em_cg_fixed'])} | {d} "
            f"| {'YES' if p['exceeds_threshold'] else 'no'} |")
    lines += ["", "Per-benchmark deltas (descriptive):", "",
              "| block | benchmark | em_cg | em_cg_fixed | delta (pp) |",
              "|---|---|---|---|---|"]
    for level, blk in ad["blocks"].items():
        for b, pb in blk["per_benchmark"].items():
            d = ("—" if pb["delta_pp"] is None else f"{pb['delta_pp']:+.2f}")
            lines.append(f"| {level} | {b} | {_r(pb['em_cg'])} "
                         f"| {_r(pb['em_cg_fixed'])} | {d} |")
    if ad.get("verdict_would_change_if_g5_bound"):
        lines += ["", "NOTE: binding g5 (against the frozen design's "
                  "commit-is-correct semantics) WOULD change the verdict — "
                  "disclosed per the module docstring's interpretive choice."]
    else:
        lines += ["", "Binding g5 would NOT change the verdict (g5 shown "
                  "descriptively; its endpoint is the designed-gold-correct "
                  "commit, taxonomy rule (d))."]
    return lines


def render_report(res: dict) -> str:
    has_fixed = bool(res.get("n_cases_with_fixed_arm"))
    lines = [
        "# v2.4 M2 — difficulty-graded contamination evaluation (curves)",
        "",
        "Preregistered under Protocol v2.4 (M2) + the 2026-07-08 amendments "
        "(advisory stage budgets, 120k hard total envelope, near-miss "
        "stratum protected in the de-scope order). Endpoints are difficulty "
        "curves per stratum level per cell with cluster-bootstrap CIs per "
        "substrate — NO go/no-go bars. Cells: em/ez extraction families x "
        "ungated (cw) / gated (cg). Frozen v2 machinery throughout; frozen "
        "key mapper mode OFF."
        + ((" PLUS the preregistered repair arm **em_cg_fixed** "
            "(REPAIR-COVERAGE, 2026-07-09): cautious em + gated with the M1 "
            "stem-wrapper mapper installed as a LOCAL mapper for that arm "
            "only over sibling case records "
            f"({res.get('n_cases_with_fixed_arm')} cases) — the four frozen "
            "cells' numbers are byte-identical with or without it.")
           if has_fixed else ""),
        "",
        "## (a) Hedging-strength gradient",
        "",
        "Contamination = final stored value for the target key equals the "
        "planted prospective/changed value. Designed gold: g1–g4 speculative "
        "(a commit IS contamination), g5 completed_event per taxonomy rule "
        "(d) (a commit is CORRECT; staging is over-gating).",
        "",
        "| level (rank) | gold | cell | contamination | staging (gated) |",
        "|---|---|---|---|---|",
    ]
    for key in sorted(res["gradient_curves"],
                      key=lambda k: (k.split("/")[0],
                                     res["gradient_curves"][k]["rank"])):
        b = res["gradient_curves"][key]
        for cell in _cells_in(b):
            c = b["cells"].get(cell, {})
            lines.append(
                f"| {key} ({b['rank']}) | {b['designed_gold_class']} | {cell} "
                f"| {_fmt(c.get('contamination'))} "
                f"| {_fmt(c.get('staging')) if cell in GATED_CELLS_EXT else 'n/a'} |")
    lines += ["", "## (b) Mixed statements (hedge + fact, one sentence)", "",
              "| block | cell | contamination (hedged value) | fact retention "
              "| staging (gated) |", "|---|---|---|---|---|"]
    for key in sorted(res["mixed"]):
        b = res["mixed"][key]
        for cell in _cells_in(b):
            c = b["cells"].get(cell, {})
            lines.append(
                f"| {key} | {cell} | {_fmt(c.get('contamination'))} "
                f"| {_fmt(c.get('fact_retention'))} "
                f"| {_fmt(c.get('staging')) if cell in GATED_CELLS_EXT else 'n/a'} |")
    lines += ["", "## (c) Near-miss resolutions", "",
              "Staging-turn endpoints (the staged hedge):", "",
              "| block | cell | contamination | staging (gated) |",
              "|---|---|---|---|"]
    for key in sorted(res["nearmiss_staging"]):
        b = res["nearmiss_staging"][key]
        for cell in _cells_in(b):
            c = b["cells"].get(cell, {})
            lines.append(
                f"| {key} | {cell} | {_fmt(c.get('contamination'))} "
                f"| {_fmt(c.get('staging')) if cell in GATED_CELLS_EXT else 'n/a'} |")
    lines += ["", "Promotion difficulty curves (paraphrase distance 1-4; "
              "ITT over all cases; gated fire semantics = test3 primary at "
              "t*, sensitivity within [t*, t*+2]; end-state = final store "
              "equals the planned value):", "",
              "| level (distance) | cell | fire at t* | fire in window | "
              "end-state | cond-on-staged (window) |",
              "|---|---|---|---|---|---|"]
    for key in sorted(res["nearmiss_promotion_curves"],
                      key=lambda k: (k.split("/")[0],
                                     res["nearmiss_promotion_curves"][k]["rank"])):
        b = res["nearmiss_promotion_curves"][key]
        for cell in _cells_in(b):
            c = b["cells"].get(cell, {})
            cond = c.get("conditional_on_staged") or {}
            lines.append(
                f"| {key} ({b['rank']}) | {cell} "
                f"| {_fmt(c.get('promote_fire_at_tstar_ITT')) if cell in GATED_CELLS_EXT else 'n/a'} "
                f"| {_fmt(c.get('promote_fire_within_window_ITT')) if cell in GATED_CELLS_EXT else 'n/a'} "
                f"| {_fmt(c.get('promoted_end_state_ITT'))} "
                f"| {_fmt(cond.get('promote_fire_within_window')) if cond.get('n') else 'n/a'} |")
    if has_fixed:
        lines += _render_adoption(res["adoption_criterion"])
    if has_fixed:
        lines += ["", "## Target-pipeline diagnostics (no bar)", "",
                  "em_fixed = the em family re-mapped by the M1 stem-wrapper "
                  "(the repair arm's entry funnel); stem-recovered = mapped "
                  "by the wrapper where the frozen em mapping declined.", "",
                  "| block/family | target extracted | mapped to target | n "
                  "| stem-recovered vs em |",
                  "|---|---|---|---|---|"]
        for key in sorted(res["target_pipeline_diagnostics"]):
            d = res["target_pipeline_diagnostics"][key]
            rec = (f"{d['stem_recovered_vs_em']}/{d['n']}"
                   if "stem_recovered_vs_em" in d else "—")
            lines.append(f"| {key} | {d['target_extracted']}/{d['n']} "
                         f"| {d['target_mapped']}/{d['n']} | {d['n']} | {rec} |")
    else:
        lines += ["", "## Target-pipeline diagnostics (no bar)", "",
                  "| block/family | target extracted | mapped to target | n |",
                  "|---|---|---|---|"]
        for key in sorted(res["target_pipeline_diagnostics"]):
            d = res["target_pipeline_diagnostics"][key]
            lines.append(f"| {key} | {d['target_extracted']}/{d['n']} "
                         f"| {d['target_mapped']}/{d['n']} | {d['n']} |")
    if has_fixed and res.get("stem_wrapper_reoffers"):
        lines += ["", "## Stem-wrapper re-offer audit (em_cg_fixed; "
                  "descriptive)", "",
                  "Re-offers = frozen-mapper-DECLINED, plan-framed em tuples "
                  "re-offered with the derived stem attribute (`-planstem`; "
                  "any turn of the arm's worlds).", "",
                  "| benchmark/stratum | cases | re-offers | stem-mapped "
                  "| stem-mapped to target key |", "|---|---|---|---|---|"]
        for key in sorted(res["stem_wrapper_reoffers"]):
            s = res["stem_wrapper_reoffers"][key]
            lines.append(f"| {key} | {s['n_cases']} | {s['n_stem_reoffers']} "
                         f"| {s['n_stem_mapped']} "
                         f"| {s['n_stem_mapped_to_target']} |")
    adv = res["advisory_budget"]
    jc = res["scoring_judge_calls"]
    lines += ["", "## Budget (advisory stage / hard envelope)", "",
              f"Stage m2 spend {adv['spent']:,} vs advisory forecast "
              f"{adv['advisory_forecast']:,} (over: {adv['over_forecast']}); "
              f"envelope {adv['envelope_spent_total']:,}/{adv['envelope']:,} "
              "(hard limit is the total envelope only, 2026-07-08 amendment).",
              "",
              f"Scoring equivalence-judge calls: {jc['total']} "
              f"({jc['fresh']} fresh, stage m2).", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# gate-items — FM-1 naturalness item builder (freeze artifact)
# ---------------------------------------------------------------------------

GATE_PER_LEVEL = 12


def build_gate_items(benchmark: str) -> list[dict]:
    """Injected vs length-decile-matched natural comparison items for the
    frozen FM-1 judging pathway (inject_v2 §3.3 semantics: natural turns come
    from the same substrates; injected text = the SPLICED host turn). Item
    building is deterministic; judging itself is the frozen external
    pathway."""
    rows = [r for r in _read_jsonl(MANIFEST_PATH)
            if r["benchmark"] == benchmark and r["text"]
            and r["inj_type"] != TYPE_ESTABLISH]
    rng = random.Random(f"{SEED_M2}:gate:{benchmark}")
    by_level: dict[str, list[dict]] = {}
    for r in rows:
        by_level.setdefault(f"{r['stratum']}/{r['level']}", []).append(r)
    injected: list[dict] = []
    for level in sorted(by_level):
        pool = sorted(by_level[level], key=lambda r: r["injection_id"])
        injected += rng.sample(pool, min(GATE_PER_LEVEL, len(pool)))

    def spliced_text(r: dict) -> str:
        world = syn_res.build_world(benchmark, r["question_id"], [r])
        turns = syn_res.world_turns(benchmark, world)
        pos = syn_res.staging_pos(turns, r["placement"])
        return turns[pos]["content"]

    def natural_pool(qid: str, speaker) -> list[tuple[int, int, str]]:
        if benchmark == "lme":
            sub = data.get_substrate(qid)
            return [(si, ti, t["content"])
                    for si, s in enumerate(sub.sessions)
                    for ti, t in enumerate(s) if t["role"] == "user"]
        conv = locomo_data.get_conversation(qid)
        return [(si, ti, t["text"])
                for si, s in enumerate(conv.sessions)
                for ti, t in enumerate(s) if t["speaker"] == speaker]

    def decile(wc: int) -> int:
        return min(wc // 10, 9)

    items, used = [], set()
    for r in injected:
        text = spliced_text(r)
        items.append({"benchmark": benchmark, "provenance": "injected",
                      "stratum": f"{r['stratum']}/{r['level']}",
                      "question_id": r["question_id"],
                      "session_idx": r["placement"]["session_idx"],
                      "turn_idx": r["placement"]["turn_idx"], "text": text})
        want = decile(len(text.split()))
        pool = [(si, ti, t) for si, ti, t in
                natural_pool(r["question_id"], r["speaker"])
                if (r["question_id"], si, ti) not in used
                and (si, ti) != (r["placement"]["session_idx"],
                                 r["placement"]["turn_idx"])]
        match = ([p for p in pool if decile(len(p[2].split())) == want]
                 or pool)
        si, ti, t = match[rng.randrange(len(match))]
        used.add((r["question_id"], si, ti))
        items.append({"benchmark": benchmark, "provenance": "natural",
                      "stratum": f"{r['stratum']}/{r['level']}",
                      "question_id": r["question_id"], "session_idx": si,
                      "turn_idx": ti, "text": t})
    return items


#: The ONLY fields a judge-facing display row may carry (v2 judge_display /
#: R2-B-3 display-safety precedent): everything else in gate_items —
#: provenance, stratum, question_id, placement — is bookkeeping the judge
#: runner must never see.
GATE_DISPLAY_ALLOWLIST = ("item_id", "display")


def blind_display_rows(items: list[dict]) -> list[dict]:
    """item_id + display ONLY. item_ids are assigned by the caller over a
    seeded shuffle, so neither the id sequence nor the file order encodes
    provenance (as built, injected/natural alternate pairwise)."""
    rows = [{"item_id": it["item_id"], "display": it["text"]} for it in items]
    for r in rows:
        assert set(r) == set(GATE_DISPLAY_ALLOWLIST), (
            f"display row leaks non-allowlisted fields: {sorted(set(r))}")
    return rows


def cmd_gate_items() -> None:
    for benchmark in BENCHMARKS:
        items = build_gate_items(benchmark)
        # blind ordering + ids: as built the list alternates injected/natural
        # (each injected item is followed by its decile-matched natural), so
        # both file order and any order-derived id would leak provenance.
        rng = random.Random(f"{SEED_M2}:gate-ids:{benchmark}")
        rng.shuffle(items)
        for i, item in enumerate(items):
            item["item_id"] = f"m2gate-{benchmark}-{i:03d}"
        _write_jsonl(GATE_ITEMS_PATH[benchmark], items)
        _write_jsonl(GATE_DISPLAY_PATH[benchmark], blind_display_rows(items))
        n_inj = sum(i["provenance"] == "injected" for i in items)
        print(f"[gate-items {benchmark}] {n_inj} injected + "
              f"{len(items) - n_inj} natural -> {_rel(GATE_ITEMS_PATH[benchmark])}"
              f" + blind display -> {_rel(GATE_DISPLAY_PATH[benchmark])}")


# ---------------------------------------------------------------------------
# smoke — render 3 stimuli + run 1 through ONE gated cell (<= fresh-cap)
# ---------------------------------------------------------------------------

async def smoke(fresh_cap: int = 10) -> dict:
    """Render one near-miss case's staging + p1 + p2 rows (persisted into the
    manifest — real M2 work) and run the STAGING stimulus through the em
    gated cell (target tuple only; predicates skipped if < 2 calls remain).
    Spend recorded under stage m2 (advisory); artifacts -> m2/smoke/."""
    rows = _read_jsonl(MANIFEST_PATH)
    assert rows, "run `stimuli` first"
    by_id = {r["injection_id"]: r for r in rows}
    case_id = next(r["case_id"] for r in rows
                   if r["benchmark"] == "lme" and r["stratum"] == "nearmiss")
    case_rows = [r for r in rows if r["case_id"] == case_id]
    staging = next(r for r in case_rows if r["inj_type"] == TYPE_STIMULUS)
    p1 = next(r for r in case_rows if r["level"] == "p1_verbatim")
    p2 = next(r for r in case_rows if r["level"] == "p2_paraphrase")
    keydesc = _keydesc()
    key_row = next(k for k in _read_jsonl(KEYS_PATH) if k["case_id"] == case_id)
    plan_sys = (PROMPTS_DIR / "injector_plan.txt").read_text()
    render_sys = (PROMPTS_DIR / "injector_render.txt").read_text()

    guard = m2_guard(local_cap=fresh_cap)
    guard.install()
    rendered, run_out = [], None
    try:
        for r in (staging, p1, p2):
            if r["text"] is None:
                await render_m2_row(r, by_id, plan_sys, render_sys, keydesc,
                                    key_group=case_rows)
            rendered.append({"injection_id": r["injection_id"],
                             "level": r["level"], "value": r["value"],
                             "text": r["text"]})
        _write_jsonl(MANIFEST_PATH, rows)

        # run the staging stimulus through em_cg (target tuple only — smoke
        # deviation to hard-bound the call count; the full `run` maps every
        # extracted tuple)
        family = "em"
        target = _m2_key(key_row)
        world = syn_res.build_world("lme", staging["question_id"], [staging])
        turns = syn_res.world_turns("lme", world)
        s_pos = syn_res.staging_pos(turns, staging["placement"])
        tuples = await _extract_turn("lme", family, world, turns, s_pos,
                                     f"m2-extract-{family}-{case_id}-smoke")
        tdicts = [t.__dict__ for t in tuples]
        matched, method = natres.match_staged_tuple(
            tdicts, staging["text"],
            {"entity": target.entity, "attribute": target.attribute})
        run_out = {"cell": "em_cg", "n_tuples": len(tuples),
                   "match_method": method, "staged": False, "class": None,
                   "mapped_key": None, "predicates": None}
        if matched is not None:
            idx = tdicts.index(matched)
            tup = tuples[idx]
            candidates = _candidates("lme", staging["question_id"], target)
            cand, _raw, _rat = await store._map_tuple_key(
                tup, candidates, trace_tag=f"m2-keymap-{family}-{case_id}-smoke")
            run_out["mapped_key"] = cand.name if cand else None
            if cand is not None:
                stream = [{"session_idx": t["session_idx"], "role": t["role"],
                           "content": t["content"], "date": t["date"]}
                          for t in turns]
                item = {"item_id": f"m2-{family}-{case_id}-s"
                                   f"{staging['placement']['session_idx']}"
                                   f"t{staging['placement']['turn_idx']}-smoke",
                        "entity": cand.entity, "attribute": cand.attribute,
                        "asserted_value": tup.value, "stored_value": None,
                        "turn_text": turns[s_pos]["content"],
                        "preceding_turns": gated_store._preceding_turns(
                            stream, s_pos)}
                cls_out = await classify.classify_item(item)
                run_out["class"] = cls_out["argmax"]
                run_out["probs"] = cls_out["probs"]
                if cls_out["argmax"] == "speculative":
                    run_out["staged"] = True
                    if guard.fresh <= fresh_cap - 2:
                        staged = await gated_store._generate_predicates(
                            stream, {"pos": s_pos,
                                     "turn_text": turns[s_pos]["content"]},
                            cand, tup, staged_id=f"stg-m2-{family}-{case_id}-smoke",
                            stream_id=case_id, family=family,
                            gen_system=gated_store.GENERATOR_PROMPT_PATH.read_text(),
                            diagnostics=defaultdict(int))
                        run_out["predicates"] = {
                            "promote_when": staged["promote_when"],
                            "invalidate_when": staged["invalidate_when"]}
                    else:
                        run_out["predicates"] = "skipped (fresh-cap headroom < 2)"
    finally:
        guard.uninstall()
    out = {"case_id": case_id, "rendered": rendered, "gated_run": run_out,
           "fresh_calls": guard.fresh, "fresh_cap": fresh_cap,
           "advisory": advisory_status()}
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    (SMOKE_DIR / f"{case_id}_smoke.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="v2.4 M2 graded-eval runner")
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("stimuli")
    st.add_argument("--force", action="store_true")
    rp = sub.add_parser("render")
    rp.add_argument("--only", default=None)
    rp.add_argument("--benchmark", default=None)
    rp.add_argument("--call-cap", type=int, default=None)
    sub.add_parser("freeze")
    runp = sub.add_parser("run")
    runp.add_argument("--benchmarks", default="lme,locomo")
    runp.add_argument("--families", default="em,ez")
    runp.add_argument("--strata", default="gradient,mixed,nearmiss")
    runp.add_argument("--ids", default=None)
    runp.add_argument("--out", default=str(CASES_DIR))
    runp.add_argument("--call-cap", type=int, default=None)
    runf = sub.add_parser("run-fixed")
    runf.add_argument("--benchmarks", default="lme,locomo")
    runf.add_argument("--strata", default="gradient,mixed,nearmiss")
    runf.add_argument("--ids", default=None)
    runf.add_argument("--ids-file", default=None,
                      help="newline-separated case ids (shard lists)")
    runf.add_argument("--out", default=str(CASES_FIXED_DIR))
    runf.add_argument("--call-cap", type=int, default=None)
    sc = sub.add_parser("score")
    sc.add_argument("--cases", default=str(CASES_DIR))
    sc.add_argument("--fixed-cases", default=str(CASES_FIXED_DIR),
                    help="em_cg_fixed sibling records root ('' disables)")
    sc.add_argument("--out-dir", default=str(M2_DIR))
    sub.add_parser("gate-items")
    sm = sub.add_parser("smoke")
    sm.add_argument("--fresh-cap", type=int, default=10)
    args = ap.parse_args()

    if args.cmd == "stimuli":
        cmd_stimuli(force=args.force)
    elif args.cmd == "render":
        only = set(args.only.split(",")) if args.only else None
        guard = m2_guard(local_cap=args.call_cap)
        guard.install()
        try:
            asyncio.run(render(only, args.benchmark))
        finally:
            guard.uninstall()
        print(f"[render] fresh calls: {guard.fresh}; "
              f"advisory: {json.dumps(advisory_status())}")
    elif args.cmd == "freeze":
        cmd_freeze()
    elif args.cmd == "run":
        assert_manifest_frozen()
        guard = m2_guard(local_cap=args.call_cap)
        summary = asyncio.run(run(
            tuple(args.benchmarks.split(",")),
            tuple(args.families.split(",")),
            tuple(args.strata.split(",")),
            ([s for s in args.ids.split(",") if s] if args.ids else None),
            Path(args.out), guard))
        print(json.dumps(summary, indent=1))
    elif args.cmd == "run-fixed":
        assert_manifest_frozen()
        ids = [s for s in args.ids.split(",") if s] if args.ids else None
        if args.ids_file:
            ids = (ids or []) + [s.strip() for s in
                                 Path(args.ids_file).read_text().splitlines()
                                 if s.strip()]
        guard = m2_guard(local_cap=args.call_cap)
        summary = asyncio.run(run_fixed(
            tuple(args.benchmarks.split(",")),
            tuple(args.strata.split(",")),
            ids, Path(args.out), guard))
        print(json.dumps(summary, indent=1))
    elif args.cmd == "score":
        guard = m2_guard()
        guard.install()
        try:
            res = asyncio.run(compute_scores(
                Path(args.cases),
                Path(args.fixed_cases) if args.fixed_cases else None))
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
              f"({guard.fresh} fresh calls; fixed-arm cases: "
              f"{res.get('n_cases_with_fixed_arm', 0)})")
    elif args.cmd == "gate-items":
        cmd_gate_items()
    elif args.cmd == "smoke":
        out = asyncio.run(smoke(args.fresh_cap))
        print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
