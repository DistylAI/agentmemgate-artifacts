"""router_sweep.py — protocol v2.2 C2: staging-router sensitivity sweep + THE FIX.

Implements the preregistered amendment (research/protocol_changelog.md, entry
"Protocol v2.2", item C2): re-run the GATED columns only (both extraction
families, both benchmarks) at frozen router operating points, over the FROZEN
v2 extraction + classifier outputs. The classify->route step of the gate loop
is pluggable (gated_store.run_gated_replay ``router=`` / ``classify_fn=`` /
``construction_label_of=`` seams); everything else — extraction rendering, key
mapping, staging predicates, checker, probe machinery — is the frozen v2
pathway byte-for-byte.

Router points (the amendment's five, seven operating points total):

  argmax           the frozen v2 behavior (gated_store.ARGMAX_ROUTE) — the
                   regression reference; reproduces the recorded gate_logs
                   EXACTLY (unit-tested against 2 substrates; the run verifies
                   all substrates).
  sev_0.3/0.5/0.7  commit iff s_ev >= t; below threshold the tuple stages
                   unless the frozen argmax is `other` (additive — `other`
                   never licenses or stages a canonical revision, v1 §1
                   taxonomy; staging every below-threshold `other` would
                   change the taxonomy, not the router).
  confidence       confidence-as-router: commit iff confidence/100 >= tau,
                   same below-threshold disposition as the s_ev thresholds.
                   tau = the cross-tuned threshold from the FROZEN v2 dAUC
                   artifacts (results/v2/dauc/results.json
                   mcnemar.cross_tuned_taus, per family x benchmark; the two
                   fold taus are averaged — they differ by <= 0.005
                   everywhere; the Youden rule there is `commit iff score >=
                   tau`, matched here).
  oracle           construction labels route (ceiling): speculative -> stage,
                   explicit_correction / completed_event -> commit, other ->
                   additive. INJECTED ROWS ONLY — natural rows carry no
                   construction label and FALL BACK TO ARGMAX (documented
                   limitation: the oracle ceiling is exact on the injected
                   stratum that GATE-*/SYN-RES score, argmax elsewhere).
  plan_frame_fix   THE one preregistered fix: tuples whose attribute matches
                   the frozen plan-frame regex list (PLAN_FRAME_ATTRIBUTE_
                   PATTERNS below) route to stage REGARDLESS of argmax;
                   non-matching tuples keep argmax routing. Matching is
                   applied to BOTH the raw extractor attribute and the mapped
                   canonical attribute.

Cache discipline (the amendment's "fresh spend = routing + checker + judging
only"): extraction and key-mapper calls replay byte-identically and hit the
llm.py cache (hard-asserted per substrate: the replay's extraction cache keys
must equal the frozen run's results/v2/<cell>/extraction_calls.jsonl rows);
classifier outputs are consumed from the FROZEN gated-cell gate_logs via
``RecordedClassifier`` (zero classifier calls — the recorded output was
conditioned on the argmax-policy store state; that conditioning noise is
accepted by the amendment's cache-sharing rule and documented here). Fresh
spend is therefore: predicate generation for NEWLY staged tuples, checker
calls on divergent pending sets, and equivalence judging.

Readouts per point (summary.json): GATE-EZ + GATE-EM (metrics_v2.gate_contrast
against the FROZEN v2 ungated cells, per benchmark, never pooled — FM-22),
SYN-RES staging rate (offline over the frozen per-family
test3_<family>/predicates_manifest.jsonl classifier records; the t3 staging
rows are all construction-label speculative), and the realized fresh-call
count.

Budget (FM-20 style; review B-1 fix): the ``forecast`` subcommand is the
C2-ONLY projection (fresh calls per point from REALIZED artifacts — frozen
gate_logs / extraction files / probe answers — with zero LLM calls). It is
kept as a byte-reproducible diagnostic, but it does NOT gate runs: the
preregistered 40,000-call envelope is C1+C2 COMBINED, so the run gate is the
combined forecast (forecast_v22.py), whose de-scope follows the preregistered
order — drop s_ev threshold points {0.3, 0.7} first, then the LoCoMo halves
of non-argmax non-fix points (keep LME everywhere; never drop the fix,
oracle or confidence points) — as pinned in the protocol-changelog v2.2
amendment note. ``run`` (full, non-partial) refuses to start without the
combined budget file (results/v2.2/budget.json), takes its scoped C2 plan
from it, and decrements the SHARED envelope at runtime through
forecast_v22.SharedCallBudget — the same file the C1 cell runs decrement;
it fails loudly when the combined envelope is exhausted.

Outputs -> ../results/v2.2/router_sweep/<point>/ (arms.run_cells artifact
shapes: timelines/, probe_answers.jsonl, extraction_calls.jsonl, summary.json)
+ <point>/metrics.json + the sweep-level summary.json and forecast.json.

CLI (from pilot/, .venv/bin/python):
  python router_sweep.py forecast [--envelope 40000]
  python router_sweep.py run [--points p1,p2] [--families mem0,zep]
                             [--qids ...] [--out DIR] [--call-budget N]
  python router_sweep.py smoke     one substrate through plan_frame_fix
                                   (<= 10 fresh calls, hard-capped)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from functools import cache
from pathlib import Path

import _bootstrap  # noqa: F401  (R1 layout: puts src/ + experiments/ on sys.path)

import arms
import gated_store
import label
import metrics_v2
import driver as run_v2
import store
from zepstore import ZepStore

from paths import RESULTS_DIR

V2_DIR = RESULTS_DIR / "v2"
SWEEP_DIR = RESULTS_DIR / "v2.2" / "router_sweep"
FORECAST_PATH = SWEEP_DIR / "forecast.json"
SUMMARY_PATH = SWEEP_DIR / "summary.json"
DAUC_RESULTS_PATH = V2_DIR / "dauc" / "results.json"

SEED = 20260702
ENVELOPE = 40_000          # v2.2 preregistered C1+C2 fresh-call envelope
JUDGE_UNRESOLVED_FRACTION = run_v2.JUDGE_UNRESOLVED_FRACTION  # §7 row 12 anchor

POINTS = ("argmax", "sev_0.3", "sev_0.5", "sev_0.7",
          "confidence", "oracle", "plan_frame_fix")
#: preregistered de-scope order: these two points drop FIRST; the fix and
#: oracle points are never dropped (v2.2 C2 budget rule).
DESCOPE_DROP_FIRST = ("sev_0.3", "sev_0.7")
NEVER_DROPPED = ("plan_frame_fix", "oracle")

FAMILIES = arms.FAMILIES                       # ("mem0", "zep")
FAMILY_ALIAS = {"mem0": "em", "zep": "ez"}     # metrics/dauc/test3 aliases
BENCHMARKS = ("lme", "locomo")

GATE_ITEM_RE = re.compile(r"^gate-(mem0|zep)-(.+)-s(\d+)t(\d+)-(\d+)$")

# ---------------------------------------------------------------------------
# THE FIX — frozen plan-frame attribute patterns
# ---------------------------------------------------------------------------

#: FROZEN regex list for the v2.2 plan_frame_fix router point (protocol
#: changelog "Protocol v2.2" C2 item 5: "tuples whose attribute matches
#: plan-frame patterns (frozen regex list, committed with the harness) route
#: to staging regardless of classifier argmax"). Exactly ONE fix iteration;
#: this list is committed with the harness and never re-tuned.
#:
#: The four pattern families are the amendment's own
#: (plan / intention / consideration / deliberation / "-frame"); the concrete
#: regex forms were derived from the plan-shaped attributes actually observed
#: in the frozen v2 artifacts BEFORE any sweep run:
#:
#:   * results/v2/extraction/*_perturbed.jsonl raw extractor attributes
#:     (realized token counts over 38,830 tuples: plan-family 2,110;
#:     intention-family 714; consideration-family 329; deliberation 0;
#:     "-frame"-suffix 0), e.g. `weekend_plan`, `planned_trip_destination`,
#:     `planned_purchase`, `hiking_plan`, `support_intention`,
#:     `detergent_brand_intention`, `toothpaste_switch_consideration`,
#:     `considering_purchase`;
#:   * results/v2/natres/failure_audit.md (§2.1 A1 / §3 G5 plan-shaped
#:     reframings): `guitar_service_intention`,
#:     `moonlit_ocean_print_display_plan`, `business_plan`,
#:     `planned_trip_region`, `camera_under_consideration`,
#:     `endorsement_brand_considerations`;
#:   * results/v2/test3_*/predicates_manifest.jsonl staging attributes:
#:     `barbershop_switch_plan`, `budgeting_app_plan`,
#:     `car_insurance_switch_consideration`, `detergent_purchase_intent`.
#:
#: Matching is TOKEN-BOUNDED on snake_case names (`(?:^|_)…(?:_|$)`), so
#: `houseplant_species` (plan inside "houseplant"), `meal_prep_timeframe`
#: ("timeframe"), `framed_photo` and `custom_frame_service_search` (physical
#: picture frames) do NOT match, while every attribute above does. The
#: "-frame" family is kept as a final-token pattern per the amendment even
#: though zero realized attributes match it (the observed `frame` tokens are
#: physical objects, deliberately excluded). `goal`/`interest`-shaped
#: attributes (the other reframing family the audit observed) are DELIBERATELY
#: not included: the amendment pins these four families, preferences are
#: explicitly `other` under the frozen rubric, and widening the list would be
#: a second fix iteration.
PLAN_FRAME_ATTRIBUTE_PATTERNS = (
    r"(?:^|_)plan(?:s|ned|ning)?(?:_|$)",
    r"(?:^|_)inten(?:t|ts|tion|tions|ded|ding)(?:_|$)",
    r"(?:^|_)consider(?:s|ed|ing|ation|ations)?(?:_|$)",
    r"(?:^|_)deliberat(?:e|es|ed|ing|ion|ions)(?:_|$)",
    r"(?:^|_)frame(?:s)?$",
)
_PLAN_FRAME_RE = tuple(re.compile(p, re.IGNORECASE)
                       for p in PLAN_FRAME_ATTRIBUTE_PATTERNS)


def plan_frame_match(attribute: str) -> bool:
    """True iff `attribute` matches the FROZEN plan-frame regex list."""
    return any(rx.search(attribute) for rx in _PLAN_FRAME_RE)


# ---------------------------------------------------------------------------
# Routers — route(tuple_record, classifier_output, construction_label_or_None)
# ---------------------------------------------------------------------------

class Router:
    """One router operating point. ``route`` returns one of gated_store.ROUTES
    ("commit" | "stage" | "additive"); the gate loop's branch bodies are the
    frozen v2 implementations regardless of router."""

    name = "base"

    def route(self, tuple_record: dict, classifier_output: dict,
              construction_label: str | None) -> str:
        raise NotImplementedError


class ArgmaxRouter(Router):
    """The frozen v2 classify->route mapping — reproduces current behavior
    EXACTLY (regression-asserted against the recorded gated-cell gate_logs)."""

    name = "argmax"

    def route(self, tuple_record, classifier_output, construction_label):
        return gated_store.ARGMAX_ROUTE[classifier_output["argmax"]]


class _ScoreThresholdRouter(Router):
    """Shared threshold shape: commit iff score >= threshold; below threshold
    the tuple STAGES unless the argmax is `other` (additive) — `other` never
    licenses or stages a canonical revision (v1 §1 taxonomy), so the
    thresholds move the commit/stage boundary without changing the additive
    class's semantics."""

    def __init__(self, threshold: float):
        self.threshold = float(threshold)

    def score(self, tuple_record: dict, classifier_output: dict) -> float:
        raise NotImplementedError

    def route(self, tuple_record, classifier_output, construction_label):
        if self.score(tuple_record, classifier_output) >= self.threshold:
            return "commit"
        return "additive" if classifier_output["argmax"] == "other" else "stage"


class SevThresholdRouter(_ScoreThresholdRouter):
    """s_ev threshold point (v2.2 C2 point 2): s_ev = P(completed) +
    P(correction) from the frozen classifier output."""

    def __init__(self, threshold: float):
        super().__init__(threshold)
        self.name = f"sev_{threshold:g}"

    def score(self, tuple_record, classifier_output):
        return classifier_output["s_ev"]


class ConfidenceRouter(_ScoreThresholdRouter):
    """confidence-as-router (v2.2 C2 point 3) at the cross-tuned threshold
    from the frozen v2 dAUC artifacts (see confidence_tau)."""

    name = "confidence"

    def score(self, tuple_record, classifier_output):
        return tuple_record["confidence"] / 100.0


class OracleRouter(Router):
    """Construction-label oracle (v2.2 C2 point 4, ceiling). INJECTED rows
    route by their construction label; NATURAL rows carry no label and fall
    back to argmax (documented: exact on the injected stratum only)."""

    name = "oracle"

    LABEL_ROUTE = {
        "speculative": "stage",
        "explicit_correction": "commit",
        "completed_event": "commit",
        "other": "additive",
    }

    def route(self, tuple_record, classifier_output, construction_label):
        if construction_label is None:  # natural row -> argmax fallback
            return gated_store.ARGMAX_ROUTE[classifier_output["argmax"]]
        return self.LABEL_ROUTE[construction_label]


class PlanFrameRouter(Router):
    """THE FIX (v2.2 C2 point 5): attributes matching the frozen plan-frame
    regex list stage regardless of argmax; everything else keeps argmax
    routing. Matches the raw extractor attribute OR the mapped canonical
    attribute (the audit observed the plan framing on the raw side; the
    canonical schemas are mostly status-shaped)."""

    name = "plan_frame_fix"

    def route(self, tuple_record, classifier_output, construction_label):
        if (plan_frame_match(tuple_record.get("raw_attribute") or "")
                or plan_frame_match(tuple_record.get("attribute") or "")):
            return "stage"
        return gated_store.ARGMAX_ROUTE[classifier_output["argmax"]]


@cache
def confidence_tau(family: str, benchmark: str) -> float:
    """The confidence router's frozen threshold: the mean of the two
    cross-tuned fold taus recorded in the v2 dAUC artifacts
    (results/v2/dauc/results.json, test1_eval.cross_tuned_decisions — Youden's
    J, `commit iff s_conf >= tau`). Per (family, benchmark); realized values:
    em/lme 0.97, em/locomo 0.96, ez/lme 0.975, ez/locomo 0.96."""
    results = json.loads(DAUC_RESULTS_PATH.read_text())
    alias = FAMILY_ALIAS[family]
    taus = results["families"][alias]["benchmarks"][benchmark]["mcnemar"][
        "cross_tuned_taus"]
    return sum(taus) / len(taus)


def make_router(point: str, family: str, benchmark: str) -> Router:
    if point == "argmax":
        return ArgmaxRouter()
    if point.startswith("sev_"):
        t = float(point[len("sev_"):])
        assert t in (0.3, 0.5, 0.7), f"non-preregistered s_ev threshold {t}"
        return SevThresholdRouter(t)
    if point == "confidence":
        return ConfidenceRouter(confidence_tau(family, benchmark))
    if point == "oracle":
        return OracleRouter()
    if point == "plan_frame_fix":
        return PlanFrameRouter()
    raise ValueError(f"unknown router point {point!r}; expected one of {POINTS}")


# ---------------------------------------------------------------------------
# Frozen v2 artifact loaders
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open() if line.strip()]


@cache
def _locomo_qids() -> frozenset:
    return frozenset(r["question_id"] for r in run_v2.load_manifest_v2()
                     if r["benchmark"] == "locomo")


def benchmark_of_stream(stream_id: str) -> str:
    return "locomo" if stream_id in _locomo_qids() else "lme"


@cache
def _raw_attribute_index(family: str) -> dict:
    """(question_id, session_idx, turn_idx) -> [raw attribute per tuple j] from
    the frozen v2 perturbed extraction files (later rows supersede earlier —
    the load_extractor_tuples convention)."""
    index: dict = {}
    for bench in BENCHMARKS:
        for r in _read_jsonl(run_v2.extraction_path(bench, family, "perturbed")):
            index[(r["question_id"], r["session_idx"], r["turn_idx"])] = [
                t["attribute"] for t in r["tuples"]]
    return index


@cache
def _frozen_extraction_keys(family: str) -> dict:
    """question_id -> frozen extraction llm-cache key list (gated cell)."""
    rows = _read_jsonl(V2_DIR / f"{family}_gated" / "extraction_calls.jsonl")
    if not rows:
        raise SystemExit(
            f"missing frozen v2 extraction_calls.jsonl for {family}_gated — "
            "the sweep replays against the frozen v2 run only")
    return {r["question_id"]: r["cache_keys"] for r in rows}


def assert_extraction_matches_frozen(family: str, question_id: str,
                                     keys: list[str]) -> None:
    """v2.2 C2 hard assert: the sweep's extraction cache keys must be
    IDENTICAL to the frozen v2 run's (cache-shared extraction; any divergence
    means fresh extraction spend and a broken comparison)."""
    frozen = _frozen_extraction_keys(family).get(question_id)
    assert frozen is not None, (
        f"{question_id}: no frozen v2 extraction record for family {family}")
    assert list(keys) == list(frozen), (
        f"EXTRACTION CACHE DIVERGENCE ({family}/{question_id}): the sweep "
        f"replay issued extraction calls that differ from the frozen v2 run "
        f"(v2.2 C2 requires cache-shared extraction)")


def load_frozen_timeline(family: str, stream_id: str) -> dict:
    path = V2_DIR / f"{family}_gated" / "timelines" / f"{stream_id}.json"
    if not path.exists():
        raise SystemExit(f"missing frozen v2 gated timeline: {path}")
    return json.loads(path.read_text())


def frozen_route_of(action: str) -> str:
    """Map a frozen gate_log action back to its route."""
    if action == "staged":
        return "stage"
    if action.startswith("additive"):
        return "additive"
    return "commit"  # commit / supersede / refresh / retro variants


@cache
def _manifest_label_index(benchmark: str, question_id: str) -> dict:
    """(session_idx, turn_idx, norm_entity, norm_attribute) ->
    construction_label for the substrate's non-T3 injected rows (the dauc
    gate-row join, normalize_key_part on both sides)."""
    index = {}
    for r in run_v2.manifest_rows_for(benchmark, question_id):
        key = (r["placement"]["session_idx"], r["placement"]["turn_idx"],
               label.normalize_key_part(r["entity"]),
               label.normalize_key_part(r["attribute"]))
        index[key] = r["construction_label"]
    return index


def manifest_label_lookup(benchmark: str, question_id: str):
    """construction_label_of callable for gated_store (oracle point): injected
    rows resolve to their construction label at the placement turn; natural
    rows resolve to None."""
    index = _manifest_label_index(benchmark, question_id)

    def lookup(turn_row: dict, cand: store.CanonicalKey) -> str | None:
        return index.get((turn_row["session_idx"], turn_row["turn_idx"],
                          label.normalize_key_part(cand.entity),
                          label.normalize_key_part(cand.attribute)))

    return lookup


# ---------------------------------------------------------------------------
# Recorded classifier provider (cache-shared classifier outputs)
# ---------------------------------------------------------------------------

class RecordedClassifier:
    """classify_fn provider returning the FROZEN v2 gated-cell classifier
    outputs by item_id (the gate item ids are deterministic given the shared
    extraction + mapping, both asserted/cached). Zero LLM calls.

    The recorded output was conditioned on the argmax-policy store state
    (item packaging shows the CURRENT stored value); under a different router
    the store state can differ, so this is the amendment's cache-sharing
    approximation, accepted and documented (module docstring)."""

    def __init__(self, v2_dir: Path = V2_DIR):
        self._v2_dir = v2_dir
        self._by_stream: dict[tuple, dict] = {}

    def _table(self, family: str, stream_id: str) -> dict:
        key = (family, stream_id)
        if key not in self._by_stream:
            tl = json.loads((self._v2_dir / f"{family}_gated" / "timelines"
                             / f"{stream_id}.json").read_text())
            self._by_stream[key] = {
                g["item_id"]: {"probs": g["probs"], "s_ev": g["s_ev"],
                               "argmax": g["class"],
                               "max_prob": max(g["probs"].values()),
                               "renormalized": False}
                for g in tl["gate_log"]
            }
        return self._by_stream[key]

    async def __call__(self, item: dict) -> dict:
        m = GATE_ITEM_RE.match(item["item_id"])
        assert m, f"unparseable gate item id {item['item_id']!r}"
        family, stream_id = m.group(1), m.group(2)
        table = self._table(family, stream_id)
        if item["item_id"] not in table:
            raise KeyError(
                f"{item['item_id']}: no recorded v2 classifier output — the "
                "replay's mapped-tuple set diverged from the frozen run "
                "(extraction/mapping must be cache-identical)")
        return dict(table[item["item_id"]])


# ---------------------------------------------------------------------------
# Offline route simulation over the frozen gate logs (forecast + regression)
# ---------------------------------------------------------------------------

def enriched_stream_rows(family: str, stream_id: str) -> dict:
    """One frozen gated timeline joined offline with (a) the raw extractor
    attribute per gate row (extraction files, tuple index from the item id)
    and (b) the injected construction label (manifest placement join).
    Returns {stream_id, benchmark, n_user_turns, rows: [...]}, each row
    carrying everything a Router needs plus the frozen action."""
    tl = load_frozen_timeline(family, stream_id)
    bench = benchmark_of_stream(stream_id)
    raw_index = _raw_attribute_index(family)
    label_index = _manifest_label_index(bench, stream_id)
    rows = []
    for g in tl["gate_log"]:
        m = GATE_ITEM_RE.match(g["item_id"])
        assert m, f"unparseable gate item id {g['item_id']!r}"
        si, ti, j = int(m.group(3)), int(m.group(4)), int(m.group(5))
        raw_attrs = raw_index.get((m.group(2), si, ti))
        assert raw_attrs is not None and j < len(raw_attrs), (
            f"{g['item_id']}: no raw extraction tuple recorded "
            f"(extraction files and gate log out of sync)")
        entity, _, attribute = g["key"].partition("/")
        rows.append({
            "item_id": g["item_id"],
            "session_idx": g["session_idx"],
            "turn_idx": g["turn_idx"],
            "user_pos": g["user_pos"],
            "store_id": g["store_id"],
            "key": g["key"],
            "entity": entity,
            "attribute": attribute,
            "raw_attribute": raw_attrs[j],
            "value": g["value"],
            "confidence": g["confidence"],
            "stored_value": g["stored_value"],
            "argmax": g["class"],
            "probs": g["probs"],
            "s_ev": g["s_ev"],
            "action": g["action"],
            "frozen_route": frozen_route_of(g["action"]),
            "construction_label": label_index.get(
                (g["session_idx"], g["turn_idx"],
                 label.normalize_key_part(entity),
                 label.normalize_key_part(attribute))),
        })
    return {"stream_id": stream_id, "benchmark": bench,
            "n_user_turns": tl["diagnostics"]["n_user_turns"], "rows": rows}


def frozen_stream_ids(family: str) -> list[str]:
    return sorted(p.stem for p in
                  (V2_DIR / f"{family}_gated" / "timelines").glob("*.json"))


def route_offline(router: Router, row: dict) -> str:
    """Apply a router to one enriched frozen gate row (no replay)."""
    cls_out = {"argmax": row["argmax"], "probs": row["probs"],
               "s_ev": row["s_ev"], "max_prob": max(row["probs"].values())}
    record = {k: row[k] for k in
              ("key", "entity", "attribute", "raw_attribute", "value",
               "confidence", "stored_value", "store_id", "session_idx",
               "turn_idx", "user_pos")}
    record["raw_entity"] = None  # not recorded in gate logs; no router uses it
    return router.route(record, cls_out, row["construction_label"])


# ---------------------------------------------------------------------------
# Forecast — FM-20-style projection from realized artifacts (pure arithmetic)
# ---------------------------------------------------------------------------

def fresh_checker_estimate(n_user_turns: int,
                           store_rows: list[tuple[int, bool, bool]]) -> int:
    """Projected FRESH checker calls for one (stream, store) under a point.

    ``store_rows``: (user_pos, frozen_is_stage, point_is_stage) per gate row
    routed to this store. Model (documented estimate, unit-tested):
      * an item staged at user_pos u keeps the store's pending set non-empty
        for every later turn (frozen resolution rates are low — v2 realized
        n_pending_final ~= n_staged — so pending-until-end is the projection);
      * a checker call at turn v is FRESH iff the pending set at v differs
        from the frozen run's, i.e. iff some row with user_pos < v changed
        stage-membership; identical pending sets are byte-identical prompts =
        llm-cache hits (free).
    => fresh = #{v : v > min staged user_pos under the point AND v > first
    stage-membership divergence} = n_user_turns - 1 - max(min_staged, div),
    floored at 0; zero when the point stages nothing here or nothing diverges.
    """
    staged = [u for u, _f, p in store_rows if p]
    if not staged:
        return 0
    diverged = [u for u, f, p in store_rows if f != p]
    if not diverged:
        return 0
    anchor = max(min(staged), min(diverged))
    return max(0, n_user_turns - 1 - anchor)


def point_stream_costs(point: str, family: str, stream: dict) -> dict:
    """Offline per-stream projection: route every frozen gate row under the
    point, count newly-staged rows (fresh predicate-generation calls) and
    project fresh checker calls per store (fresh_checker_estimate)."""
    router = make_router(point, family, stream["benchmark"])
    by_store: dict[str, list[tuple[int, bool, bool]]] = {}
    generator_fresh = 0
    n_stage = n_diff = 0
    for row in stream["rows"]:
        r = route_offline(router, row)
        f_stage = row["frozen_route"] == "stage"
        p_stage = r == "stage"
        n_stage += p_stage
        n_diff += (r != row["frozen_route"])
        generator_fresh += (p_stage and not f_stage)
        by_store.setdefault(row["store_id"], []).append(
            (row["user_pos"], f_stage, p_stage))
    checker_fresh = sum(
        fresh_checker_estimate(stream["n_user_turns"], rows)
        for rows in by_store.values())
    return {"benchmark": stream["benchmark"], "n_rows": len(stream["rows"]),
            "n_stage": n_stage, "n_route_diffs": n_diff,
            "generator_fresh": generator_fresh, "checker_fresh": checker_fresh}


@cache
def _n_spec_probe_rows(family: str) -> dict:
    """Realized spec-type probe-answer counts per benchmark (frozen gated
    cell) — the judging surface per (family, benchmark, point)."""
    rows = _read_jsonl(V2_DIR / f"{family}_gated" / "probe_answers.jsonl")
    out = {b: 0 for b in BENCHMARKS}
    for r in rows:
        if r["inj_type"] in arms.SPEC_TYPES:
            out[r.get("benchmark") or "lme"] += 1
    return out


def compute_point_forecast(point: str) -> dict:
    """Projected fresh calls for one point, per benchmark (both families)."""
    per_bench = {b: {"generator": 0, "checker": 0, "judging": 0,
                     "n_route_diffs": 0, "n_stage": 0, "n_rows": 0}
                 for b in BENCHMARKS}
    for family in FAMILIES:
        diffs_by_bench = {b: 0 for b in BENCHMARKS}
        for stream_id in frozen_stream_ids(family):
            c = point_stream_costs(point, family,
                                   enriched_stream_rows(family, stream_id))
            b = per_bench[c["benchmark"]]
            b["generator"] += c["generator_fresh"]
            b["checker"] += c["checker_fresh"]
            b["n_route_diffs"] += c["n_route_diffs"]
            b["n_stage"] += c["n_stage"]
            b["n_rows"] += c["n_rows"]
            diffs_by_bench[c["benchmark"]] += c["n_route_diffs"]
        # judging: fresh equivalence-judge calls only when the gated column
        # can differ from the frozen one (else every pair is llm-cached).
        for bench in BENCHMARKS:
            if diffs_by_bench[bench]:
                per_bench[bench]["judging"] += round(
                    JUDGE_UNRESOLVED_FRACTION * _n_spec_probe_rows(family)[bench])
    for bench in BENCHMARKS:
        b = per_bench[bench]
        b["total"] = b["generator"] + b["checker"] + b["judging"]
    return per_bench


def apply_descope(point_costs: dict, envelope: int) -> dict:
    """The preregistered v2.2 C2 de-scope order over projected costs.

    ``point_costs``: {point: {benchmark: cost_int}}. Steps, applied only while
    the projected total exceeds ``envelope``:
      1. drop the s_ev threshold points {0.3, 0.7} entirely;
      2. drop the LoCoMo halves of every remaining point (keep LME).
    The fix and oracle points are never dropped. Returns the scoped plan with
    the step log and the over-envelope verdict (pure arithmetic; unit-tested).
    """
    scope = {p: [b for b in BENCHMARKS if b in point_costs[p]]
             for p in point_costs}
    steps: list[dict] = []

    def total() -> int:
        return sum(point_costs[p][b] for p, bs in scope.items() for b in bs)

    if total() > envelope:
        dropped = [p for p in DESCOPE_DROP_FIRST if p in scope]
        for p in dropped:
            assert p not in NEVER_DROPPED
            del scope[p]
        steps.append({"step": "drop_sev_threshold_points",
                      "dropped_points": dropped, "total_after": total()})
    if total() > envelope:
        restricted = [p for p, bs in scope.items() if "locomo" in bs]
        for p in restricted:
            scope[p] = [b for b in scope[p] if b != "locomo"]
        steps.append({"step": "drop_locomo_keep_lme",
                      "restricted_points": restricted, "total_after": total()})
    return {
        "envelope": envelope,
        "points": {p: bs for p, bs in scope.items()},
        "steps": steps,
        "total_projected": total(),
        "over_envelope": total() > envelope,
        "rule": ("v2.2 C2 de-scope order: drop s_ev threshold points "
                 "{0.3, 0.7} first, then LoCoMo router points (keep LME), "
                 "never the fix or oracle points"),
    }


def cmd_forecast(args) -> None:
    per_point = {p: compute_point_forecast(p) for p in POINTS}
    point_costs = {p: {b: per_point[p][b]["total"] for b in BENCHMARKS}
                   for p in POINTS}
    plan = apply_descope(point_costs, args.envelope)
    fc = {
        "endpoint": "v2.2 C2 router-sweep fresh-call forecast (FM-20 style)",
        "seed": SEED,
        "envelope": args.envelope,
        "judge_fraction_anchor": JUDGE_UNRESOLVED_FRACTION,
        "notes": [
            "projection from REALIZED frozen v2 artifacts only (gate logs, "
            "extraction files, probe answers); zero LLM calls",
            "extraction/key-mapping/classifier are cache-shared by "
            "construction (asserted at run time) and carry no fresh cost",
            "checker projection assumes staged items stay pending to stream "
            "end (v2 realized n_pending_final ~= n_staged) and counts only "
            "byte-divergent pending sets as fresh (identical sets are llm "
            "cache hits)",
        ],
        "points": per_point,
        "descope": plan,
    }
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    FORECAST_PATH.write_text(json.dumps(fc, indent=1, ensure_ascii=False))
    full_total = sum(point_costs[p][b] for p in POINTS for b in BENCHMARKS)
    print(f"[forecast] full sweep projection {full_total:,} fresh calls vs "
          f"envelope {args.envelope:,}; scoped plan total "
          f"{plan['total_projected']:,} after {len(plan['steps'])} de-scope "
          f"step(s); verdict: "
          f"{'OVER ENVELOPE - do not run' if plan['over_envelope'] else 'within envelope'} "
          f"-> {FORECAST_PATH}")
    for p in POINTS:
        scoped = plan["points"].get(p, [])
        print(f"  {p:15s} lme={point_costs[p]['lme']:>6,} "
              f"locomo={point_costs[p]['locomo']:>6,}  scoped={scoped}")
    print("[forecast] NOTE (review B-1): this is the C2-ONLY projection; the "
          "40,000 envelope is C1+C2 COMBINED and the run gate is "
          "`forecast_v22.py forecast` -> results/v2.2/budget.json")
    if plan["over_envelope"]:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Sweep replay (GATED columns only, frozen inputs)
# ---------------------------------------------------------------------------

def make_sweep_replay(point: str, benchmark: str,
                      provider: RecordedClassifier):
    """CellJob.replay callable running the gated column under `point` with the
    frozen classifier outputs; asserts extraction cache-key identity with the
    frozen v2 run per substrate."""

    async def replay(job: arms.CellJob, family: str, policy: str):
        assert policy == "gated", "the v2.2 sweep re-runs GATED columns only"
        router = make_router(point, family, benchmark)
        label_of = manifest_label_lookup(benchmark, job.question_id)
        if benchmark == "lme":
            substrate, candidates = gated_store._prepare_substrate(
                job.substrate, job.key_schema_entry, job.manifest_rows,
                job.aux_keys)
            timeline = await gated_store.run_gated_replay(
                gated_store.substrate_stream(substrate), candidates,
                stream_id=job.question_id, family=family,
                session_order=list(substrate.session_order),
                router=router, classify_fn=provider,
                construction_label_of=label_of)
        else:
            conv = run_v2.perturbed_conversation(job.question_id)
            stream = run_v2.locomo_stream_rows(conv)
            extracted = await run_v2.dyadic_extract_stream(
                conv, family, stream_id=f"{job.question_id}-perturbed")
            stores = {s.lower(): ZepStore() for s in conv.speakers}
            timeline = await gated_store.run_gated_replay(
                stream, job.aux_keys, stream_id=job.question_id,
                family=family, stores=stores, route_store=lambda key: key[0],
                session_order=list(conv.session_order), extracted=extracted,
                router=router, classify_fn=provider,
                construction_label_of=label_of)
        assert_extraction_matches_frozen(family, job.question_id,
                                         timeline.extraction_call_keys)
        return timeline

    return replay


def build_jobs(point: str, benchmarks: list[str],
               qids: set[str] | None) -> list[arms.CellJob]:
    provider = RecordedClassifier()
    jobs: list[arms.CellJob] = []
    if "lme" in benchmarks:
        lme_qids = [q for q in run_v2.lme_v2_qids()
                    if qids is None or q in qids]
        lme_jobs = run_v2.lme_cell_jobs(lme_qids)
        replay = make_sweep_replay(point, "lme", provider)
        for j in lme_jobs:
            j.replay = replay
        jobs.extend(lme_jobs)
    if "locomo" in benchmarks:
        import locomo_data
        sids = [s for s in locomo_data.conversation_ids()
                if qids is None or s in qids]
        loco_jobs = run_v2.locomo_cell_jobs(sids)
        replay = make_sweep_replay(point, "locomo", provider)
        for j in loco_jobs:
            j.replay = replay
        jobs.extend(loco_jobs)
    if not jobs:
        raise SystemExit("no jobs selected")
    return jobs


def verify_argmax_regression(point_dir: Path, families: tuple[str, ...],
                             stream_ids: list[str] | None = None) -> dict:
    """argmax-point regression: the re-run gate logs must reproduce the frozen
    v2 gated-cell gate logs row for row ((item_id, class, action) sequences —
    checker outputs are llm-cached, so resolutions replay identically)."""
    verdict = {"identical": True, "n_streams": 0, "mismatches": []}
    for family in families:
        tl_dir = point_dir / arms.cell_name(family, "gated") / "timelines"
        for path in sorted(tl_dir.glob("*.json")):
            if stream_ids is not None and path.stem not in stream_ids:
                continue
            new = json.loads(path.read_text())
            frozen = load_frozen_timeline(family, path.stem)
            got = [(g["item_id"], g["class"], g["action"])
                   for g in new["gate_log"]]
            want = [(g["item_id"], g["class"], g["action"])
                    for g in frozen["gate_log"]]
            verdict["n_streams"] += 1
            if got != want:
                verdict["identical"] = False
                verdict["mismatches"].append(
                    {"family": family, "stream_id": path.stem})
    assert verdict["identical"], (
        f"ARGMAX REGRESSION FAILURE: re-run gate logs diverge from the frozen "
        f"v2 run: {verdict['mismatches']}")
    return verdict


# ---------------------------------------------------------------------------
# Per-point readouts: GATE-EZ / GATE-EM + SYN-RES staging rate
# ---------------------------------------------------------------------------

@cache
def _frozen_ungated_answers(family: str) -> dict:
    rows = _read_jsonl(V2_DIR / f"{family}_ungated" / "probe_answers.jsonl")
    if not rows:
        raise SystemExit(f"missing frozen v2 ungated probe answers for {family}")
    return {r["injection_id"]: r for r in rows}


async def _y_outcome(row: dict, trace_tag: str) -> int:
    attr = row["key"].partition("/")[2]
    return int(row["present"] and await run_v2._equivalent(
        attr, row["value"], row["v_target"], trace_tag=trace_tag))


async def point_gate_metrics(point: str, point_dir: Path,
                             benchmarks: list[str]) -> dict:
    """GATE-EZ / GATE-EM per benchmark: paired ITT contrast CW-CG where CW is
    the FROZEN v2 ungated column (router-invariant: routers act on the gated
    commit policy only) and CG is this point's re-run gated column. Estimator,
    clusters and provenance handling are metrics_v2.gate_contrast exactly as
    in the frozen v2 metrics stage."""
    out: dict = {}
    for family in FAMILIES:
        alias = FAMILY_ALIAS[family]
        gated = _read_jsonl(point_dir / arms.cell_name(family, "gated")
                            / "probe_answers.jsonl")
        ungated = _frozen_ungated_answers(family)
        for bench in benchmarks:
            rows = []
            for r in gated:
                if r["inj_type"] not in arms.SPEC_TYPES:
                    continue
                if (r.get("benchmark") or "lme") != bench:
                    continue
                cw = ungated.get(r["injection_id"])
                if cw is None:
                    continue
                y_cg = await _y_outcome(
                    r, f"v22-eq-{point}-{r['injection_id']}-{alias}_cg")
                y_cw = await _y_outcome(
                    cw, f"v22-eq-{point}-{r['injection_id']}-{alias}_cw")
                rows.append({"cluster": run_v2._cluster_of(r),
                             "y_cw": y_cw, "y_cg": y_cg,
                             "provenance": "injected"})
            if rows:
                out.setdefault(bench, {})[f"GATE-{alias.upper()}"] = \
                    metrics_v2.gate_contrast(rows, family=alias)
    return out


def synres_staging_rate(point: str, family: str, benchmark: str) -> dict:
    """SYN-RES staging rate under a router point, computed OFFLINE from the
    frozen per-family t3 predicate manifests (test3_v2 generate records the
    staging-turn tuple + the frozen classifier output per case; every t3
    staging row's construction label is `speculative`). ITT denominator = the
    50 frozen cases; extraction misses can never stage under any router."""
    alias = FAMILY_ALIAS[family]
    rows = [r for r in _read_jsonl(V2_DIR / f"test3_{alias}"
                                   / "predicates_manifest.jsonl")
            if r["benchmark"] == benchmark]
    if not rows:
        return {"x": 0, "n": 0, "rate": None,
                "note": f"no frozen t3 manifest for {alias}/{benchmark}"}
    router = make_router(point, family, benchmark)
    x = frozen_x = n_miss = 0
    for r in rows:
        frozen_x += bool(r.get("staged"))
        tup = r.get("staged_tuple")
        cls = r.get("classifier")
        if tup is None or cls is None:  # extraction miss: ITT not staged
            n_miss += 1
            continue
        record = {
            "key": f"{r['entity']}/{r['attribute']}",
            "entity": r["entity"], "attribute": r["attribute"],
            "raw_entity": tup["entity"], "raw_attribute": tup["attribute"],
            "value": tup["value"], "confidence": tup["confidence"],
            "stored_value": None, "store_id": None,
            "session_idx": r["staged_at_turn"]["session_idx"],
            "turn_idx": r["staged_at_turn"]["turn_idx"], "user_pos": None,
        }
        cls_out = {"argmax": cls["argmax"], "probs": cls["probs"],
                   "s_ev": cls["s_ev"], "max_prob": cls["max_prob"]}
        x += router.route(record, cls_out, "speculative") == "stage"
    n = len(rows)
    return {"x": x, "n": n, "rate": x / n, "n_extraction_miss": n_miss,
            "frozen_argmax": {"x": frozen_x, "n": n, "rate": frozen_x / n}}


def point_synres_block(point: str, benchmarks: list[str]) -> dict:
    return {FAMILY_ALIAS[family]: {bench: synres_staging_rate(point, family, bench)
                                   for bench in benchmarks}
            for family in FAMILIES}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _merge_summary(point: str, block: dict, envelope: int | None) -> None:
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    summary = (json.loads(SUMMARY_PATH.read_text())
               if SUMMARY_PATH.exists() else {"points": {}})
    summary["points"][point] = block
    summary["envelope"] = envelope
    summary["total_fresh_calls"] = sum(
        p.get("fresh_calls", 0) for p in summary["points"].values())
    SUMMARY_PATH.write_text(json.dumps(summary, indent=1, ensure_ascii=False))


async def _run_point(point: str, benchmarks: list[str],
                     families: tuple[str, ...], qids: set[str] | None,
                     out_root: Path, compute_metrics: bool) -> dict:
    point_dir = out_root / point
    jobs = build_jobs(point, benchmarks, qids)
    cells = await arms.run_cells(jobs, families=families,
                                 policies=("gated",), out_dir=point_dir)
    block: dict = {
        "point": point,
        "benchmarks": benchmarks,
        "families": list(families),
        "n_jobs": len(jobs),
        "cells": {c: {k: v for k, v in info.items() if k != "diagnostics"}
                  for c, info in cells["cells"].items()},
    }
    if point == "argmax":
        block["argmax_regression"] = verify_argmax_regression(
            point_dir, families,
            stream_ids=sorted(qids) if qids is not None else None)
    if compute_metrics:
        block["GATE"] = await point_gate_metrics(point, point_dir, benchmarks)
        block["SYNRES_staging_rate"] = point_synres_block(point, benchmarks)
    return block


def _run_guarded(coro, guard) -> tuple[object, int]:
    guard.install()
    try:
        result = asyncio.run(coro)
    finally:
        guard.uninstall()
    return result, guard.fresh


def cmd_run(args) -> None:
    families = tuple(f for f in args.families.split(",") if f)
    for f in families:
        assert f in FAMILIES, f"unknown family {f!r}"
    qids = set(args.qids.split(",")) if args.qids else None
    partial = qids is not None
    out_root = Path(args.out) if args.out else SWEEP_DIR
    if partial and not args.out:
        raise SystemExit("partial runs (--qids) require --out: never write "
                         "partial artifacts into the canonical sweep tree")

    if partial:
        points = (args.points.split(",") if args.points
                  else ["plan_frame_fix"])
        scoped = {p: [b for b in (args.benchmarks.split(",")
                                  if args.benchmarks else BENCHMARKS)]
                  for p in points}
        envelope = None
    else:
        # B-1 fix: full sweep runs are gated on the COMBINED C1+C2 forecast
        # (forecast_v22.py), take their scoped plan from it, and decrement
        # the SHARED envelope file (results/v2.2/budget.json) that the C1
        # cell runs decrement too. Import is function-local: forecast_v22
        # imports this module for the C2 point projections.
        import forecast_v22
        shared = forecast_v22.load_budget()
        if shared.get("over_envelope"):
            raise SystemExit("the combined forecast is over the envelope "
                             "even after de-scope — do not run (v2.2 hard "
                             "stop)")
        envelope = shared["envelope"]
        scoped = shared["c2"]["points"]
        points = args.points.split(",") if args.points else list(scoped)
        for p in points:
            if p not in scoped:
                raise SystemExit(f"point {p!r} is outside the combined "
                                 f"forecast's scoped C2 plan {sorted(scoped)} "
                                 "(de-scoped or unknown); re-run "
                                 "forecast_v22.py to change scope")

    for point in points:
        benchmarks = scoped[point]
        if partial:
            guard = run_v2.CallBudget(args.call_budget)
        else:
            import forecast_v22
            guard = forecast_v22.SharedCallBudget(
                "c2_router_sweep", local_cap=args.call_budget)
        block, fresh = _run_guarded(
            _run_point(point, benchmarks, families, qids, out_root,
                       compute_metrics=not partial),
            guard)
        block["fresh_calls"] = fresh
        if not partial:
            _merge_summary(point, block, envelope)
        gate = block.get("GATE", {})
        bits = []
        for bench, per in gate.items():
            for rid, res in per.items():
                bits.append(f"{bench}/{rid} point={res['point']:.3f} "
                            f"CI=[{res['ci'][0]:.3f},{res['ci'][1]:.3f}]")
        print(f"[{point}] {block['n_jobs']} substrate job(s) over "
              f"{benchmarks}; fresh calls {fresh:,}"
              + (f"; {'; '.join(bits)}" if bits else "")
              + ("" if not partial else " [partial run: no summary merge]"))
    if not partial:
        print(f"[run] sweep summary -> {SUMMARY_PATH}")


# ---------------------------------------------------------------------------
# Smoke — one substrate through one non-default point (<= 10 fresh calls)
# ---------------------------------------------------------------------------

SMOKE_POINT = "plan_frame_fix"
SMOKE_QID = "9bbe84a2"     # LME KU substrate: exactly one plan-frame route
                           # flip vs argmax, <= 3 trailing user turns
SMOKE_BUDGET = 10


def cmd_smoke(args) -> None:
    out_root = SWEEP_DIR / "smoke"
    block, fresh = _run_guarded(
        _run_point(SMOKE_POINT, ["lme"], ("mem0",), {SMOKE_QID}, out_root,
                   compute_metrics=False),
        run_v2.CallBudget(SMOKE_BUDGET))
    tl = json.loads((out_root / SMOKE_POINT / "mem0_gated" / "timelines"
                     / f"{SMOKE_QID}.json").read_text())
    frozen = load_frozen_timeline("mem0", SMOKE_QID)
    frozen_actions = {g["item_id"]: g["action"] for g in frozen["gate_log"]}
    diffs = [
        {"item_id": g["item_id"], "class": g["class"], "route": g.get("route"),
         "raw_action": g["action"], "frozen_action": frozen_actions.get(g["item_id"])}
        for g in tl["gate_log"]
        if g["action"] != frozen_actions.get(g["item_id"])]
    print(f"[smoke] {SMOKE_POINT} on {SMOKE_QID} (mem0/lme): "
          f"{len(tl['gate_log'])} gate rows, "
          f"{tl['diagnostics']['n_staged']} staged "
          f"(frozen {frozen['diagnostics']['n_staged']}); "
          f"{len(diffs)} action diff(s) vs frozen; fresh calls {fresh} "
          f"(cap {SMOKE_BUDGET})")
    for d in diffs:
        print(f"  DIFF {d['item_id']}: class={d['class']} "
              f"frozen={d['frozen_action']} -> now={d['raw_action']} "
              f"(route={d['route']})")
    for s in tl["staged_log"]:
        print(f"  staged {s['staged_id']}: {s['entity']}/{s['attribute']} = "
              f"{s['hypothesized_value']!r} status={s['status']}")
    assert fresh <= SMOKE_BUDGET


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="v2.2 C2 staging-router sensitivity sweep")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("forecast", help="FM-20-style fresh-call projection "
                                        "+ preregistered de-scope")
    p.add_argument("--envelope", type=int, default=ENVELOPE)

    p = sub.add_parser("run", help="re-run the gated columns per router point")
    p.add_argument("--points", default=None,
                   help="comma-separated subset of the scoped plan "
                        "(default: every scoped point)")
    p.add_argument("--families", default=",".join(FAMILIES))
    p.add_argument("--benchmarks", default=None,
                   help="partial runs only (full runs take scope from the "
                        "forecast)")
    p.add_argument("--qids", default=None,
                   help="substrate/conversation subset (partial run; "
                        "requires --out)")
    p.add_argument("--out", default=None)
    p.add_argument("--call-budget", type=int, default=None,
                   help="abort after N fresh (non-cached) LLM calls")

    sub.add_parser("smoke", help="one substrate through plan_frame_fix "
                                 "(<= 10 fresh calls)")

    args = parser.parse_args()
    {"forecast": cmd_forecast, "run": cmd_run, "smoke": cmd_smoke}[args.cmd](args)


if __name__ == "__main__":
    main()
