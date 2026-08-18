"""scorer_d2.py — held-out D2 endpoint scorer for the one-shot external test
run (protocol/CHANGELOG.md 2026-07-30 "PREREGISTRATION: external held-out
one-shot test run", D2 endpoint convention paragraph; 2026-08-11 "D2
VALIDATION COMPLETE" + "LoCoMo structural exclusion" entries: the run
executes on the USABLE test slice — 77 items = the 100 frozen test ids minus
results/external_intake/d2_exclusions.json).

Scores the COMPLETED per-patch case records under
results/external_eval/test/cases_d2/ (77 patches x the four frozen systems)
against the held-out D2 labels. It NEVER re-runs extraction or host
application: everything is a pure function of the case record + the held-out
label + the model-visible question row + the patch construction metadata +
the frozen equivalence judge (natural_check_rule5.EquivalenceScorer —
imported, never copied; exact-normalized fast path first, judge call
otherwise). Judge calls run under budget stage "external"
(runner_external.external_guard), exactly the scorer_d1 wiring.

D2 ENDPOINT CONVENTION (binding, from the preregistration entry): endpoints
are BEHAVIORAL, keyed to the tuples extracted from the inserted turns under
each family — expected_memory_state realized (committed as current fact /
staged / promoted at confirmation / superseded at correction / discarded) —
plus the patch probe answered from the store state at the keys those tuples
landed on, judged against the held-out expected_answer by the frozen
equivalence judge. LME hosts carry NO key registry (deterministic no-map,
recorded in every case record); LoCoMo uses the frozen host registries.

OPERATIONALIZATION (every decision restated in the output "methods" block):

"The keys those tuples landed on": for the cell's extraction family
(em_cw->em, em_cg->em_fixed [adopted stem-wrapper repair], ez_cw/ez_cg->ez)
the record's inserted_turn_extraction rows give, per inserted turn, the
tuples and their mapped_keys. A tuple LANDED iff its mapped key is non-null.
Landed tuples are paired 1:1, in order, with the cell's inserted_gate_rows
at the same (session_idx, turn_idx) — the runner appends exactly one gate
row per mapped tuple in tuple order — and the pairing asserts key and value
agreement. LME cells have zero landings BY CONSTRUCTION (candidates == []);
their behavioral strata are therefore structurally zero and are reported
with the deterministic-no-map disclosure, never as gate behavior
(anti-saturation: the landing-coverage table is itself a reported quantity).

Realized tuple state (mechanics distribution + endpoint inputs):
  no_map | committed_current | committed_not_current (a later write
  displaced it) | staged_pending | promoted_current | promoted_not_current |
  invalidated | discarded_invalid | additive_noop | refresh.
"Stored as current" for a landed tuple = the cell's final_store value at the
landed (store_id, key) is judge-equivalent to the tuple's value (promoted
tuples also match on the staged row's hypothesized_value); this is
route-agnostic — path attribution (direct commit vs staged->promoted) is
reported separately for the confirmation family.

Per-family endpoints (per system cell; Wilson CIs on patch counts; ITT over
the 77 — a patch with zero landings scores False on every stored/applied
flag; landed-only strata are ALSO reported with their disclosed smaller n):

E-CONF confirmation_wording: stored_current (the inserted plan's value
  current at end), by wording_distance (verbatim/paraphrased/oblique/
  implied) and by host; path split via_direct_commit (all cells) vs
  via_staged_promotion (gated cells only; staging undefined ungated ->
  D1-style disclosed n/a).
E-MIX mixed_sentence: fact_retention = some landed key's final value
  judge-equivalent to the held-out expected_answer (the settled component);
  contamination = a PLAN-FLAGGED landed tuple stored as current. Plan
  flagging is deterministic and disclosed: the tuple's attribute contains
  "plan" OR its attribute+value+statement_span text matches the fixed hedge/
  intent cue regex (PLAN_CUE_RE); tuples judge-equivalent to expected_answer
  are the settled component and never count as contamination (overlap
  counted in methods). plan_staged (gated) reported as the correct-gating
  complement.
E-CORR correction_pair: applied = a landed key of the CORRECTING turn(s)
  holds a final value judge-equivalent to expected_answer. Correcting turns
  come from the label's evidence_refs: patch_turn_id "…:turn:N" (1-based
  into the patch's turns; LME coordinates via the record's insertion
  turn_span) or dialog_id (index into the record's insertion dia_ids);
  unresolvable refs fall back to the LAST inserted turn (counted).
  stale_handled = at each correcting-landed key where a commit-family gate
  row wrote a PRIOR value (record order, before the correcting turn's write
  at that key) that is NOT equivalent to expected_answer, the final value is
  no longer equivalent to that prior value; patches with no such prior value
  are disclosed unevaluable (n/a), never forced into the denominator.
E-CAL calibration_anchor: stored_current (the anchor content committed as
  current fact — for these unsettled anchors a commit IS the contamination
  analog) and staged (gated), by calibration_strength (strong_hedge /
  confident_plan_with_date); reported beside the released M2 planted-probe
  numbers (results/v2.4/m2/m2_results.json, READ ONLY, never rerun): the
  label metadata names the bridge strong_hedge -> lme|locomo g1_strong_hedge
  exactly; confident_plan_with_date -> g3_confident_plan by construct name,
  with g4_scheduled_event carried as the adjacent dated level (disclosed:
  M2's g3 stimuli are undated). External em_cg maps to M2's em_cg_fixed (the
  adopted repair IS the default cautious-gated configuration).
E-PROBE (all 77, ITT): answered = some landed key holds a present,
  non-sentinel final value; correct = an answered value judge-equivalent to
  expected_answer; unanswered = incorrect UNLESS the model-visible question
  row says allowed_abstention (9 calibration anchors) AND the probe is
  unanswered — abstention-by-absence is then the labeled correct behavior.

STATISTICS: x/n with stats.wilson_ci on patch counts. ITT unit = patch;
confirmation clusters (same fact, 4 wording distances) repeat across
distance strata but are unique WITHIN each stratum; pooled rows therefore
carry a disclosed cluster note rather than a cluster CI (mirrors the D1
Wilson-on-items convention).

POPULATION: exactly the 77 usable test ids — holdout_guard.test_ids
("dataset2") minus results/external_intake/d2_exclusions.json — asserted
one-to-one against the case records; any excluded id among the cases is a
hard failure.

ONE-SHOT DISCIPLINE: the real (non --dry-run) scoring is run ONCE by the
maintainer; an existing scores_d2.json is refused without --force. --dry-run
stubs the judge with scorer_d1.DryEquivalence (exact-normalized fast path
only, ZERO LLM calls, unresolved pairs counted) and writes to
scores_d2_dryrun.{json,md} so it can never clobber the one-shot output.

CLI (repo root, pilot/.venv/bin/python experiments/external_eval/scorer_d2.py):
  score [--dry-run] [--cases DIR] [--out-dir DIR] [--call-cap N] [--force]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401  (R1 layout: puts src/ + experiments/ on sys.path)

import adapter_d2
import holdout_guard
import natural_check_rule5 as rule5
import runner_external as rx
import scorer_d1 as sd1
import stats
import zepstore
from scorer_d1 import DryEquivalence  # the frozen --dry-run judge stub

from paths import RESULTS_DIR

CASES_DIR = rx.OUT_ROOT / "test" / "cases_d2"
OUT_DIR = rx.OUT_ROOT / "test"
LABELS_PATH = (adapter_d2.EXTERNAL_PKG / "eval_sets"
               / "labels_or_answer_keys" / "dataset2_labels.jsonl")
QUESTIONS_PATH = (adapter_d2.EXTERNAL_PKG / "eval_sets" / "raw_inputs"
                  / "dataset2_test_questions.jsonl")
EXCLUSIONS_PATH = RESULTS_DIR / "external_intake" / "d2_exclusions.json"
M2_RESULTS_PATH = RESULTS_DIR / "v2.4" / "m2" / "m2_results.json"

ALL_CELLS = ("em_cw", "em_cg", "ez_cw", "ez_cg")
GATED_CELLS = ("em_cg", "ez_cg")
#: cell -> the extraction/mapping family whose recs fed it (runner_external).
CELL_FAMILY = {"em_cw": "em", "em_cg": "em_fixed", "ez_cw": "ez", "ez_cg": "ez"}
FAMILIES = ("confirmation_wording", "mixed_sentence", "correction_pair",
            "calibration_anchor")
WORDING_DISTANCES = ("verbatim", "paraphrased", "oblique", "implied")
CALIBRATION_STRENGTHS = ("strong_hedge", "confident_plan_with_date")
HOSTS = ("longmemeval", "locomo")
#: store-event layer commit family (scorer_d1 COMMIT_ACTIONS + the gate's
#: additive route, which surfaces as 'additive_commit' on the gate-row layer;
#: same-key current-fact writes all count — the D1 commit_family_layer_note).
COMMIT_ACTIONS = ("commit", "supersede", "additive_commit")
USABLE_TEST_N = 77  # 100 frozen test ids - 23 excluded (changelog 2026-08-11)
JUDGE_CALL_CAP = sd1.JUDGE_CALL_CAP  # the m4_eval compute_scores cap

#: Deterministic plan/hedge cue rule for the mixed_sentence UNSETTLED
#: component (attribute containing 'plan' also flags — the em extractor's
#: naming convention). Fixed BEFORE scoring; hit counts land in methods.
PLAN_CUE_RE = re.compile(
    r"\b(?:may|might|maybe|possibly|perhaps|hop(?:e|es|ing) to|"
    r"thinking (?:of|about)|considering|leaning toward(?:s)?|"
    r"tentative(?:ly)?|undecided|not sure|"
    r"ha(?:s|ve)(?: not|n't) decided|no decision|we'?ll see|"
    r"toying with|tempted to|planning to|plans? to|want(?:s)? to|"
    r"would like to|looking to|open to|still deciding)\b")


# ---------------------------------------------------------------------------
# Loading + population assertions
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_labels(path: Path = LABELS_PATH) -> dict[str, dict]:
    """Held-out D2 labels keyed by patch_id (== record_id)."""
    return {r["patch_id"]: r for r in _read_jsonl(path)}


def load_questions(path: Path = QUESTIONS_PATH) -> dict[str, dict]:
    """Model-visible test questions keyed by record_id (== patch_id)."""
    return {q["record_id"]: q for q in _read_jsonl(path)}


def load_cases(cases_dir: Path = CASES_DIR) -> list[dict]:
    cases = [json.loads(p.read_text())
             for p in sorted(cases_dir.glob("*.json"))]
    assert cases, f"no case records under {cases_dir}"
    return cases


def excluded_ids(path: Path = EXCLUSIONS_PATH) -> list[str]:
    return list(json.loads(path.read_text())["excluded"])


def usable_test_ids() -> list[str]:
    """The preregistered run population: frozen test slice minus the owner/
    audit exclusions (changelog 2026-08-11 RUN AUTHORIZATION + LoCoMo
    structural exclusion). Hard-asserted to the entry's 77."""
    usable = sorted(set(holdout_guard.test_ids("dataset2"))
                    - set(excluded_ids()))
    assert len(usable) == USABLE_TEST_N, (
        f"usable D2 test slice is {len(usable)}, expected {USABLE_TEST_N} "
        "(changelog 2026-08-11) — split or exclusion file drifted")
    return usable


def assert_population(case_ids: list[str], usable: list[str],
                      excluded: list[str]) -> None:
    """The scored population is EXACTLY the usable test slice; excluded ids
    are never scored (they were never ingested; a case record for one is a
    hard failure, not a skip)."""
    cids, uset = set(case_ids), set(usable)
    assert len(case_ids) == len(cids), "duplicate case ids"
    hit = cids & set(excluded)
    assert not hit, f"EXCLUDED ids have case records / would be scored: " \
                    f"{sorted(hit)}"
    missing = uset - cids
    assert not missing, f"usable test ids without case records: " \
                        f"{sorted(missing)}"
    extra = cids - uset
    assert not extra, f"case records outside the usable test slice: " \
                      f"{sorted(extra)}"


# ---------------------------------------------------------------------------
# Inserted-tuple <-> gate-row pairing (the "keys those tuples landed on")
# ---------------------------------------------------------------------------

def inserted_tuple_rows(case: dict, cell: str) -> list[dict]:
    """One row per tuple extracted from an inserted turn, for the cell's
    extraction family, paired IN ORDER with the cell's inserted_gate_rows at
    the same coordinates (the runner appends exactly one gate row per MAPPED
    tuple, in tuple order; unmapped tuples produce none). Asserts key and
    value agreement on every pairing and that no inserted gate row is left
    unclaimed."""
    cc = case["cells"][cell]
    by_coord: dict[tuple, list[dict]] = {}
    for g in cc["inserted_gate_rows"]:
        by_coord.setdefault((g["session_idx"], g["turn_idx"]), []).append(g)
    staged_status = {s["staged_id"]: s for s in cc["staged_rows"]}
    out = []
    ext = case["inserted_turn_extraction"][CELL_FAMILY[cell]]
    for turn_ord, row in enumerate(ext):
        coord = (row["session_idx"], row["turn_idx"])
        queue = list(by_coord.get(coord, []))
        for tup, key in zip(row["tuples"], row["mapped_keys"]):
            item = {"turn_ordinal": turn_ord, "coord": coord,
                    "attribute_raw": tup["attribute"], "value": tup["value"],
                    "text": " ".join(str(tup.get(f) or "") for f in
                                     ("attribute", "value", "statement_span")),
                    "key": key, "gate": None, "staged": None}
            if key is not None:
                assert queue, (f"{case['patch_id']}/{cell}: tuple mapped to "
                               f"{key!r} at {coord} has no gate row left")
                g = queue.pop(0)
                assert g["key"] == key and g["value"] == tup["value"], (
                    f"{case['patch_id']}/{cell}: gate-row pairing mismatch at "
                    f"{coord}: gate ({g['key']!r}, {g['value']!r}) vs tuple "
                    f"({key!r}, {tup['value']!r})")
                item["gate"] = g
                if g["action"] == "staged":
                    item["staged"] = staged_status[g["staged_id"]]
            out.append(item)
        assert not queue, (f"{case['patch_id']}/{cell}: "
                           f"{len(queue)} unclaimed inserted gate rows at "
                           f"{coord}")
    return out


def _attr_of(key: str) -> str:
    return key.partition("/")[2]


async def _current_at_end(cc: dict, item: dict, scorer) -> bool:
    """Landed tuple stored as current: the final_store value at the landed
    (store_id, key) is equivalent to the tuple's value (promoted tuples also
    match on the staged hypothesized_value)."""
    g = item["gate"]
    final = cc["final_store"].get(g["store_id"], {}).get(g["key"])
    eq, _m = await scorer.equivalent(_attr_of(g["key"]), final, item["value"])
    if eq:
        return True
    if item["staged"] is not None and item["staged"]["status"] == "promoted":
        eq, _m = await scorer.equivalent(
            _attr_of(g["key"]), final, item["staged"]["hypothesized_value"])
        return bool(eq)
    return False


async def realized_state(cc: dict, item: dict, scorer) -> str:
    """The preregistered behavioral tuple states (docstring list)."""
    if item["key"] is None:
        return "no_map"
    action = item["gate"]["action"]
    if action == "staged":
        status = item["staged"]["status"]
        if status == "pending":
            return "staged_pending"
        if status == "promoted":
            return ("promoted_current"
                    if await _current_at_end(cc, item, scorer)
                    else "promoted_not_current")
        return status  # invalidated | discarded_invalid
    if action in COMMIT_ACTIONS:
        return ("committed_current"
                if await _current_at_end(cc, item, scorer)
                else "committed_not_current")
    return action  # additive_noop | refresh (never introduces a value)


# ---------------------------------------------------------------------------
# Correcting-turn resolution (correction_pair family)
# ---------------------------------------------------------------------------

def correcting_turn_ordinals(case: dict, label: dict, patch: dict,
                             counters: Counter) -> set[int]:
    """0-based ordinals INTO the cell-family extraction rows (== inserted
    turns in insertion order) named by the label's evidence_refs. LME:
    patch_turn_id '…:turn:N' -> the record's insertion turn_span[N-1]
    coordinates -> the extraction row at those coordinates (assistant-turn
    refs cannot land: LME ingests user turns only — asserted user here).
    LoCoMo: dialog_id -> index into the record's insertion dia_ids.
    Unresolvable -> fall back to the LAST inserted turn (counted)."""
    ins = case["insertion"]
    ords: set[int] = set()
    ext = case["inserted_turn_extraction"][CELL_FAMILY["em_cw"]]
    coord_to_ord = {(r["session_idx"], r["turn_idx"]): i
                    for i, r in enumerate(ext)}
    for ref in label.get("evidence_refs", []):
        rid = ref.get("ref_id") or ""
        if ref.get("ref_type") == "patch_turn_id" and ":turn:" in rid:
            n = int(rid.rsplit(":turn:", 1)[1])  # 1-based patch turn
            span = ins.get("turn_span")
            if span and 1 <= n <= len(span):
                turn = patch["turns"][n - 1]
                assert (turn.get("role") or "user") == "user", (
                    f"{case['patch_id']}: correcting ref {rid} is not a "
                    "user turn — it was never ingested on LME")
                coord = (span[n - 1]["session_idx"], span[n - 1]["turn_idx"])
                if coord in coord_to_ord:
                    ords.add(coord_to_ord[coord])
                    continue
        elif ref.get("ref_type") == "dialog_id":
            dia = ins.get("dia_ids") or []
            if rid in dia:
                ords.add(dia.index(rid))
                continue
        counters["correcting_ref_fallback"] += 1
    if not ords:
        counters["correcting_all_fallback_last_turn"] += 1
        ords = {len(ext) - 1}
    return ords


# ---------------------------------------------------------------------------
# Per-patch scoring (pure function of case + label + question + patch + judge)
# ---------------------------------------------------------------------------

async def score_patch(case: dict, label: dict, question: dict, patch: dict,
                      scorer, counters: Counter | None = None) -> list[dict]:
    """Per-patch score rows, one per system cell. Every flag is a bool (in
    the endpoint's population) or None (out of it) — the aggregator skips
    None (the scorer_d1 _pt pattern)."""
    if counters is None:
        counters = Counter()
    pid = case["patch_id"]
    family = label["insertion_family"]
    assert family == patch["insertion_family"], (
        f"{pid}: label/patch insertion_family disagree")
    expected = label["expected_answer"]
    abstention_ok = bool(question["allowed_abstention"])
    corr_ords = (correcting_turn_ordinals(case, label, patch, counters)
                 if family == "correction_pair" else None)

    rows = []
    for cell in [c for c in ALL_CELLS if c in case["cells"]]:
        cc = case["cells"][cell]
        gated = cell in GATED_CELLS
        items = inserted_tuple_rows(case, cell)
        landed = [i for i in items if i["key"] is not None]
        states = [await realized_state(cc, i, scorer) for i in items]
        current = {}  # id(item) -> stored-as-current bool (landed only)
        for i in landed:
            current[id(i)] = await _current_at_end(cc, i, scorer)
        any_current = any(current[id(i)] for i in landed)

        # distinct landed keys + their final values (the probe's store view)
        landed_keys = {(i["gate"]["store_id"], i["gate"]["key"])
                       for i in landed}
        finals = {}
        for sid, key in sorted(landed_keys):
            v = cc["final_store"].get(sid, {}).get(key)
            if v is not None and not zepstore.is_sentinel(str(v)):
                finals[(sid, key)] = v

        async def _final_matches(value) -> bool:
            for (sid, key), v in finals.items():
                eq, _m = await scorer.equivalent(_attr_of(key), v, value)
                if eq:
                    return True
            return False

        # ---- E-CONF confirmation_wording -----------------------------------
        conf_stored = conf_direct = conf_promoted = None
        if family == "confirmation_wording":
            conf_stored = any_current
            conf_direct = any(
                current[id(i)] and i["gate"]["action"] in COMMIT_ACTIONS
                for i in landed)
            if gated:
                conf_promoted = any(
                    s == "promoted_current"
                    for s in states)  # staged->promoted path, still current
            # ungated: staging undefined -> disclosed n/a (None)

        # ---- E-MIX mixed_sentence -------------------------------------------
        mix_fact = mix_cont = mix_staged = None
        if family == "mixed_sentence":
            mix_fact = await _final_matches(expected)
            mix_cont = False
            plan_flagged = 0
            for i in landed:
                is_settled, _m = await scorer.equivalent(
                    _attr_of(i["gate"]["key"]), i["value"], expected)
                plan_flag = ("plan" in i["attribute_raw"].lower()
                             or bool(PLAN_CUE_RE.search(i["text"].lower())))
                if plan_flag:
                    plan_flagged += 1
                    if is_settled:
                        counters["mixed_settled_and_plan_flagged"] += 1
                if plan_flag and not is_settled and current[id(i)]:
                    mix_cont = True
            counters["mixed_plan_flagged_tuples"] += plan_flagged
            if gated:
                mix_staged = any(
                    i["staged"] is not None
                    and ("plan" in i["attribute_raw"].lower()
                         or PLAN_CUE_RE.search(i["text"].lower()))
                    for i in landed)

        # ---- E-CORR correction_pair -----------------------------------------
        corr_applied = corr_stale = None
        corr_stale_evaluable = None
        if family == "correction_pair":
            corr_landed = [i for i in landed
                           if i["turn_ordinal"] in corr_ords]
            corr_applied = False
            for i in corr_landed:
                sid, key = i["gate"]["store_id"], i["gate"]["key"]
                if (sid, key) in finals:
                    eq, _m = await scorer.equivalent(
                        _attr_of(key), finals[(sid, key)], expected)
                    if eq:
                        corr_applied = True
                        break
            # stale handling: prior commit-family value at a correcting-landed
            # key, written (record order) before the correcting write there.
            corr_stale_evaluable = False
            handled = True
            gate_rows = cc["gate_rows"]
            for i in corr_landed:
                sid, key = i["gate"]["store_id"], i["gate"]["key"]
                coord = i["coord"]
                cut = next((j for j, g in enumerate(gate_rows)
                            if (g["session_idx"], g["turn_idx"]) == coord
                            and g["store_id"] == sid and g["key"] == key),
                           len(gate_rows))
                prior = None
                for g in gate_rows[:cut]:
                    if (g["store_id"], g["key"]) == (sid, key) \
                            and g["action"] in COMMIT_ACTIONS:
                        prior = g["value"]
                if prior is None:
                    continue
                eq_new, _m = await scorer.equivalent(_attr_of(key), prior,
                                                     expected)
                if eq_new:
                    continue  # host already held the corrected value
                corr_stale_evaluable = True
                final = cc["final_store"].get(sid, {}).get(key)
                still_old, _m = await scorer.equivalent(_attr_of(key), final,
                                                        prior)
                if still_old:
                    handled = False
            corr_stale = handled if corr_stale_evaluable else None
            if not corr_stale_evaluable:
                counters[f"corr_stale_unevaluable|{cell}"] += 1

        # ---- E-CAL calibration_anchor ----------------------------------------
        cal_stored = cal_staged = None
        if family == "calibration_anchor":
            cal_stored = any_current
            if gated:
                cal_staged = any(i["staged"] is not None for i in landed)

        # ---- E-PROBE (all families, ITT) --------------------------------------
        probe_answered = bool(finals)
        if probe_answered:
            probe_correct = await _final_matches(expected)
        else:
            probe_correct = abstention_ok  # abstention-by-absence, disclosed
            if abstention_ok:
                counters["probe_abstention_credited"] += 1

        rows.append({
            "patch_id": pid,
            "cell": cell,
            "family": family,
            "host": case["host_dataset"],
            "wording_distance": label["wording_distance"],
            "mixed_sentence_type": label["mixed_sentence_type"],
            "correction_pair_type": label["correction_pair_type"],
            "calibration_strength": label["calibration_strength"],
            "n_inserted_tuples": len(items),
            "n_landed_tuples": len(landed),
            "any_landed": bool(landed),
            "realized_states": dict(Counter(states)),
            # E-CONF
            "conf_stored_current": conf_stored,
            "conf_via_direct_commit": conf_direct,
            "conf_via_staged_promotion": conf_promoted,
            # E-MIX
            "mixed_fact_retained": mix_fact,
            "mixed_contamination": mix_cont,
            "mixed_plan_staged": mix_staged,
            # E-CORR
            "corr_applied": corr_applied,
            "corr_stale_handled": corr_stale,
            "corr_stale_evaluable": corr_stale_evaluable,
            # E-CAL
            "calib_stored_current": cal_stored,
            "calib_staged": cal_staged,
            # E-PROBE
            "probe_answered": probe_answered,
            "probe_correct": probe_correct,
            "probe_abstention_allowed": abstention_ok,
        })
    return rows


# ---------------------------------------------------------------------------
# Aggregation (x/n + Wilson on patch counts; scorer_d1._pt reused)
# ---------------------------------------------------------------------------

def _cell(rows: list[dict], cell: str) -> list[dict]:
    return [r for r in rows if r["cell"] == cell]


def _fam(rows: list[dict], family: str) -> list[dict]:
    return [r for r in rows if r["family"] == family]


def _tree(rows: list[dict], flag: str, strata: dict[str, callable],
          cells=ALL_CELLS) -> dict:
    """Per-cell {stratum: point}; each stratum is (name -> row filter)."""
    return {cell: {name: sd1._pt([r for r in _cell(rows, cell) if pred(r)],
                                 flag)
                   for name, pred in strata.items()}
            for cell in cells}


def _itt_and_landed(extra: dict[str, callable] | None = None) -> dict:
    strata = {"pooled_itt": lambda r: True,
              "pooled_landed_only": lambda r: r["any_landed"]}
    for host in HOSTS:
        strata[f"{host}|itt"] = (lambda r, h=host: r["host"] == h)
        strata[f"{host}|landed_only"] = (
            lambda r, h=host: r["host"] == h and r["any_landed"])
    strata.update(extra or {})
    return strata


def aggregate(rows: list[dict]) -> dict:
    conf = _fam(rows, "confirmation_wording")
    mix = _fam(rows, "mixed_sentence")
    corr = _fam(rows, "correction_pair")
    cal = _fam(rows, "calibration_anchor")

    conf_strata = _itt_and_landed(
        {f"{d}|itt": (lambda r, d=d: r["wording_distance"] == d)
         for d in WORDING_DISTANCES}
        | {f"{d}|landed_only": (
            lambda r, d=d: r["wording_distance"] == d and r["any_landed"])
           for d in WORDING_DISTANCES})
    cal_strata = _itt_and_landed(
        {f"{s}|itt": (lambda r, s=s: r["calibration_strength"] == s)
         for s in CALIBRATION_STRENGTHS}
        | {f"{s}|landed_only": (
            lambda r, s=s: r["calibration_strength"] == s and r["any_landed"])
           for s in CALIBRATION_STRENGTHS})

    endpoints = {
        "landing_coverage": {
            # the evaluability axis of the whole convention — a reported
            # quantity, not a footnote (LME structurally 0 by no-registry).
            "any_landed": _tree(rows, "any_landed", {
                "pooled": lambda r: True,
                **{f"{h}": (lambda r, h=h: r["host"] == h) for h in HOSTS},
                **{f"{f}|locomo": (
                    lambda r, f=f: r["family"] == f and r["host"] == "locomo")
                   for f in FAMILIES},
            }),
        },
        "E_confirmation_wording": {
            "stored_current": _tree(conf, "conf_stored_current", conf_strata),
            "via_direct_commit": _tree(conf, "conf_via_direct_commit",
                                       _itt_and_landed()),
            "via_staged_promotion": _tree(conf, "conf_via_staged_promotion",
                                          _itt_and_landed(),
                                          cells=GATED_CELLS),
        },
        "E_mixed_sentence": {
            "contamination": _tree(mix, "mixed_contamination",
                                   _itt_and_landed()),
            "fact_retention": _tree(mix, "mixed_fact_retained",
                                    _itt_and_landed()),
            "plan_staged": _tree(mix, "mixed_plan_staged", _itt_and_landed(),
                                 cells=GATED_CELLS),
        },
        "E_correction_pair": {
            "applied": _tree(corr, "corr_applied", _itt_and_landed()),
            "stale_handled": _tree(corr, "corr_stale_handled", {
                "evaluable": lambda r: True,  # _pt skips the None rows
                **{f"{h}|evaluable": (lambda r, h=h: r["host"] == h)
                   for h in HOSTS},
            }),
        },
        "E_calibration_anchor": {
            "stored_current": _tree(cal, "calib_stored_current", cal_strata),
            "staged": _tree(cal, "calib_staged", cal_strata,
                            cells=GATED_CELLS),
        },
        "E_probe": _tree(rows, "probe_correct", {
            "pooled_itt": lambda r: True,
            "answered": lambda r: r["probe_answered"],
            **{f"{f}|itt": (lambda r, f=f: r["family"] == f)
               for f in FAMILIES},
            **{f"{h}|itt": (lambda r, h=h: r["host"] == h) for h in HOSTS},
            "abstention_allowed|itt": lambda r: r["probe_abstention_allowed"],
        }),
    }
    # probe answered-rate (mechanics for the ITT reading)
    endpoints["E_probe_answered"] = _tree(rows, "probe_answered", {
        "pooled": lambda r: True,
        **{f"{h}": (lambda r, h=h: r["host"] == h) for h in HOSTS}})
    # realized-state mechanics (tuple counts, no CI — a distribution)
    mech: dict = {}
    for cell in ALL_CELLS:
        mech[cell] = {}
        for host in HOSTS:
            c = Counter()
            for r in _cell(rows, cell):
                if r["host"] == host:
                    c.update(r["realized_states"])
            mech[cell][host] = dict(c)
    endpoints["realized_state_mechanics"] = mech
    return endpoints


# ---------------------------------------------------------------------------
# M2 planted-probe bridge (released numbers, READ ONLY — never rerun)
# ---------------------------------------------------------------------------

#: external cell -> the M2 arm it reproduces (em_cg ships the adopted
#: stem-wrapper repair, i.e. M2's em_cg_fixed arm).
CELL_TO_M2 = {"em_cw": "em_cw", "em_cg": "em_cg_fixed",
              "ez_cw": "ez_cw", "ez_cg": "ez_cg"}
#: label calibration_strength -> M2 gradient level(s); strong_hedge is an
#: exact name match, confident_plan_with_date matches g3 by construct name
#: with g4 carried as the adjacent dated level (M2's g3 stimuli are undated).
STRENGTH_TO_M2 = {
    "strong_hedge": ("g1_strong_hedge",),
    "confident_plan_with_date": ("g3_confident_plan", "g4_scheduled_event"),
}


def m2_bridge(path: Path = M2_RESULTS_PATH) -> dict:
    """The corresponding ORIGINAL planted-probe numbers for the calibration
    anchors, copied verbatim from the released results/v2.4/m2 summary."""
    if not path.exists():
        return {"available": False, "path": str(path)}
    curves = json.loads(path.read_text())["gradient_curves"]
    out = {"available": True, "source": str(path),
           "cell_mapping": dict(CELL_TO_M2),
           "note": ("released M2 numbers, read from the frozen summary — "
                    "nothing rerun; external em_cg == M2 em_cg_fixed (the "
                    "adopted repair is the default cautious-gated config); "
                    "confident_plan_with_date bridges to g3_confident_plan "
                    "by construct name, g4_scheduled_event shown as the "
                    "adjacent dated level")}
    for strength, levels in STRENGTH_TO_M2.items():
        out[strength] = {}
        for level in levels:
            out[strength][level] = {}
            for sub in ("lme", "locomo"):
                block = curves.get(f"{sub}/{level}", {}).get("cells", {})
                out[strength][level][sub] = {
                    cell: block.get(m2cell)
                    for cell, m2cell in CELL_TO_M2.items()}
    return out


# ---------------------------------------------------------------------------
# Full pass + methods
# ---------------------------------------------------------------------------

async def score_all(cases: list[dict], labels: dict[str, dict],
                    questions: dict[str, dict], patches: dict[str, dict],
                    scorer) -> dict:
    """Score exactly the patches that have case records."""
    counters: Counter = Counter()
    rows: list[dict] = []
    for case in cases:
        pid = case["patch_id"]
        rows.extend(await score_patch(case, labels[pid], questions[pid],
                                      patches[pid], scorer, counters))
    pids = {r["patch_id"] for r in rows}
    by_family = Counter(labels[p]["insertion_family"] for p in pids)
    by_host = Counter(labels[p]["host_dataset"] for p in pids)
    return {
        "n_patches_scored": len(pids),
        "by_family": dict(by_family),
        "by_host": dict(by_host),
        "endpoints": aggregate(rows),
        "m2_bridge": m2_bridge(),
        "rows": rows,
        "_counters": dict(counters),
    }


def methods_block(scorer, res: dict, mode: str) -> dict:
    c = res["_counters"]
    return {
        "mode": mode,
        "spec": ("protocol/CHANGELOG.md 2026-07-30 preregistration, 'D2 "
                 "endpoint convention' paragraph (retained by the 2026-07-31 "
                 "amendment) + 2026-08-11 RUN AUTHORIZATION on the usable "
                 "77-item test slice"),
        "population": (f"{res['n_patches_scored']} patches == the frozen "
                       "test slice minus results/external_intake/"
                       "d2_exclusions.json; excluded ids asserted absent"),
        "scoring_inputs": ("case records + held-out labels + model-visible "
                           "question rows (allowed_abstention only) + patch "
                           "construction metadata (family/turn structure); "
                           "extraction and host application NEVER re-run; "
                           "the frozen equivalence judge is the only model "
                           "call"),
        "equivalence": ("rule5.EquivalenceScorer (exact-normalized fast "
                        "path, judge otherwise), the m4_eval/scorer_d1 pair "
                        "shape: equivalent(attribute, stored_value, "
                        "reference); absent/None/sentinel never match; the "
                        "probe question is never in the judge prompt"),
        "landed_keys": ("a tuple landed iff its mapped key is non-null in "
                        "the record's inserted_turn_extraction for the "
                        "cell's family (em_cw->em, em_cg->em_fixed [adopted "
                        "stem-wrapper repair], ez_cw/ez_cg->ez); landed "
                        "tuples paired in order with inserted_gate_rows at "
                        "the same coordinates, key+value asserted"),
        "lme_no_registry": ("LME hosts carry NO key registry (preregistered "
                            "deterministic no-map): every LME tuple is "
                            "unmapped, so LME behavioral strata are "
                            "STRUCTURALLY zero-landing — reported under the "
                            "landing_coverage table and per-host splits, "
                            "never citable as gate behavior"),
        "judge_artifact_note": (
            "verifier F1: one lenient frozen-judge verdict (D2-PATCH-113: an "
            "incidental employment side fact judged equivalent to the hedged "
            "Thursday-plan expected answer) carries that patch's probe in "
            "ALL FOUR cells identically — counterfactually removing it "
            "shifts each cell's probe count by -1/77 and no cell contrast "
            "or the entry-bottleneck finding moves; disclosed per the D1 "
            "precedent, not rescored"),
        "calibration_reading_note": (
            "verifier F2: the only nonzero calibration stored-as-current "
            "signal (1/17, same patch in all cells) is that settled side "
            "fact, not the hedged anchor value; under a plan-flagged "
            "reading (the E-MIX rule) external calibration contamination "
            "is 0/17 in every cell — the released 1/17 is the honest "
            "number under the operationalization as written"),
        "itt": ("ITT unit = patch over the usable 77; a patch with zero "
                "landings scores False on every stored/applied flag and its "
                "probe is unanswered (incorrect unless abstention-credited); "
                "landed_only strata re-report each endpoint over patches "
                "with >=1 landed tuple (disclosed n)"),
        "stored_as_current": ("final_store value at the landed (store_id, "
                              "key) equivalent to the tuple's value; "
                              "promoted tuples also match the staged row's "
                              "hypothesized_value; route-agnostic"),
        "commit_family_actions": list(COMMIT_ACTIONS) + [
            "(additive_commit counted per the D1 commit_family_layer_note; "
            "refresh never introduces a value)"],
        "confirmation_path_split": ("via_direct_commit = a commit-family "
                                    "inserted gate row still current; "
                                    "via_staged_promotion = an inserted "
                                    "staged row promoted and still current; "
                                    "staging undefined for ungated cells -> "
                                    "disclosed n/a (None)"),
        "mixed_component_rule": ("settled component = tuple judge-equivalent "
                                 "to the held-out expected_answer; unsettled "
                                 "(plan) component = attribute contains "
                                 "'plan' OR attribute+value+statement_span "
                                 "matches PLAN_CUE_RE (fixed hedge/intent "
                                 "cues); settled tuples never count as "
                                 "contamination; plan-flagged tuple count "
                                 f"{c.get('mixed_plan_flagged_tuples', 0)}, "
                                 "settled-and-flagged overlap "
                                 f"{c.get('mixed_settled_and_plan_flagged', 0)}"),
        "correcting_turn_resolution": (
            "label evidence_refs: patch_turn_id via the record's insertion "
            "turn_span (user turns only on LME); dialog_id via insertion "
            "dia_ids; fallback to the last inserted turn counted "
            f"{c.get('correcting_ref_fallback', 0)} ref-level / "
            f"{c.get('correcting_all_fallback_last_turn', 0)} patch-level"),
        "stale_handling": ("evaluable iff a commit-family gate row wrote a "
                           "prior value (record order, before the correcting "
                           "write) at a correcting-landed key and that prior "
                           "value is not already equivalent to "
                           "expected_answer; handled = final value no longer "
                           "equivalent to the prior value; unevaluable "
                           "patches are disclosed n/a per cell: "
                           + json.dumps({k: v for k, v in c.items()
                                         if k.startswith("corr_stale_unevaluable")})),
        "calibration_endpoint": ("stored_current is the contamination "
                                 "analog (anchors are unsettled: a commit IS "
                                 "the failure the M2 g1-g4 convention "
                                 "counts); staged reported for gated cells; "
                                 "bridge numbers in m2_bridge, read from the "
                                 "released summary only"),
        "probe": ("answered = some landed key holds a present, non-sentinel "
                  "final value; correct = an answered value equivalent to "
                  "expected_answer; unanswered = incorrect, EXCEPT "
                  "allowed_abstention questions (model-visible flag) where "
                  "abstention-by-absence is the labeled correct behavior — "
                  f"credited {c.get('probe_abstention_credited', 0)} "
                  "patch-cell times"),
        "ci": ("stats.wilson_ci on patch counts; confirmation clusters (one "
               "fact x 4 wording distances) are unique within each distance "
               "stratum; pooled confirmation rows carry this cluster note "
               "in lieu of a cluster CI"),
        "judge_calls": scorer.judge_calls,
        "fresh_judge_calls": scorer.fresh_judge_calls,
        "dry_unresolved_pairs": getattr(scorer, "unresolved_pairs", None),
        "budget_snapshot": rx.advisory_status(),
    }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _table(title: str, tree: dict, cells: tuple[str, ...]) -> list[str]:
    strata = list(tree[cells[0]].keys())
    lines = [f"### {title}", "",
             "| stratum | " + " | ".join(cells) + " |",
             "|---|" + "|".join(["---"] * len(cells)) + "|"]
    for s in strata:
        lines.append("| " + s.replace("|", "\\|") + " | "
                     + " | ".join(sd1._fmt(tree[c][s]) for c in cells) + " |")
    lines.append("")
    return lines


def render_report(res: dict) -> str:
    ep = res["endpoints"]
    m = res["methods"]
    lines = [
        "# External held-out test — Dataset 2 scores",
        "",
        f"Mode: **{m['mode']}** — generated {res['generated_at']}",
        f"Scored: {res['n_patches_scored']} patches "
        f"({json.dumps(res['by_family'])}; {json.dumps(res['by_host'])}) "
        "x 4 systems.",
        "",
        "Spec: " + m["spec"],
        "",
        "## Landing coverage (evaluability of the behavioral convention)",
        "",
        "> LME hosts carry NO key registry (preregistered deterministic "
        "no-map): LME landings are structurally zero — those strata measure "
        "the convention, not gate behavior. LoCoMo landing is partial (the "
        "frozen host registries predate the insertions).", ""]
    lines += _table("Patches with >=1 landed inserted tuple",
                    ep["landing_coverage"]["any_landed"], ALL_CELLS)
    lines += ["## E-CONF — confirmation_wording (promotion/commitment of "
              "the inserted value as current fact)", ""]
    lines += _table("Stored as current (ITT + landed-only + wording ladder)",
                    ep["E_confirmation_wording"]["stored_current"], ALL_CELLS)
    lines += _table("Path: via direct commit",
                    ep["E_confirmation_wording"]["via_direct_commit"],
                    ALL_CELLS)
    lines += _table("Path: via staged->promotion (gated cells; ungated n/a "
                    "— staging undefined)",
                    ep["E_confirmation_wording"]["via_staged_promotion"],
                    GATED_CELLS)
    lines += ["## E-MIX — mixed_sentence", ""]
    lines += _table("Contamination (unsettled component stored as current)",
                    ep["E_mixed_sentence"]["contamination"], ALL_CELLS)
    lines += _table("Fact retention (settled component stored)",
                    ep["E_mixed_sentence"]["fact_retention"], ALL_CELLS)
    lines += _table("Plan component staged (gated cells)",
                    ep["E_mixed_sentence"]["plan_staged"], GATED_CELLS)
    lines += ["## E-CORR — correction_pair", ""]
    lines += _table("Correction applied (new value current at a "
                    "correcting-turn landed key)",
                    ep["E_correction_pair"]["applied"], ALL_CELLS)
    lines += _table("Stale handling (old value no longer current; "
                    "evaluable subpopulation)",
                    ep["E_correction_pair"]["stale_handled"], ALL_CELLS)
    lines += ["## E-CAL — calibration_anchor", ""]
    lines += _table("Stored as current (contamination analog) by strength",
                    ep["E_calibration_anchor"]["stored_current"], ALL_CELLS)
    lines += _table("Staged (gated cells) by strength",
                    ep["E_calibration_anchor"]["staged"], GATED_CELLS)
    br = res["m2_bridge"]
    lines += ["### Original planted-probe bridge (released M2 numbers, "
              "not rerun)", ""]
    if br.get("available"):
        lines += [br["note"], ""]
        for strength, levels in STRENGTH_TO_M2.items():
            for level in levels:
                lines += [f"**{strength} -> {level}**", "",
                          "| substrate | cell | contamination | staging |",
                          "|---|---|---|---|"]
                for sub in ("lme", "locomo"):
                    for cell in ALL_CELLS:
                        pt = (br[strength][level][sub] or {}).get(cell)
                        if not pt:
                            continue
                        cont = pt.get("contamination")
                        stg = pt.get("staging")

                        def f(p):
                            if not p:
                                return "n/a"
                            return (f"{p['x']}/{p['n']} = {p['point']:.3f} "
                                    f"[{p['ci'][0]:.3f}, {p['ci'][1]:.3f}]")
                        lines.append(f"| {sub} | {cell} | {f(cont)} | "
                                     f"{f(stg)} |")
                lines.append("")
    else:
        lines += [f"m2 summary unavailable at {br.get('path')}", ""]
    lines += ["## E-PROBE — patch probes (ITT over the 77)", ""]
    lines += _table("Probe correct vs held-out expected_answer",
                    ep["E_probe"], ALL_CELLS)
    lines += _table("Probe answered (mechanics for the ITT reading)",
                    ep["E_probe_answered"], ALL_CELLS)
    lines += ["## Realized tuple states (mechanics; tuple counts)", ""]
    for cell in ALL_CELLS:
        lines.append(f"- **{cell}**: "
                     + json.dumps(ep["realized_state_mechanics"][cell]))
    lines += ["", "## Methods", ""]
    for k, v in m.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def run_score(cases_dir: Path, out_dir: Path, dry_run: bool,
                    call_cap: int | None, force: bool) -> dict:
    cases = load_cases(cases_dir)
    slices = {c.get("slice") for c in cases}
    assert slices == {"test"}, (
        f"case records span slices {slices}; the scorer targets the "
        "one-shot TEST run only")
    usable = usable_test_ids()
    assert_population([c["patch_id"] for c in cases], usable, excluded_ids())
    labels = load_labels()
    questions = load_questions()
    patches = adapter_d2.patches()

    out_json = out_dir / ("scores_d2_dryrun.json" if dry_run
                          else "scores_d2.json")
    out_md = out_dir / ("scores_d2_dryrun.md" if dry_run else "scores_d2.md")
    if not dry_run and out_json.exists() and not force:
        raise SystemExit(
            f"{out_json} already exists — the real scoring is ONE-SHOT; "
            "pass --force only for a logged infrastructure-failure rerun")

    guard = None
    if dry_run:
        scorer = DryEquivalence()
    else:
        scorer = rule5.EquivalenceScorer(cap=JUDGE_CALL_CAP)
        guard = rx.external_guard(local_cap=call_cap)  # budget stage "external"
        guard.install()
    try:
        res = await score_all(cases, labels, questions, patches, scorer)
    finally:
        if guard is not None:
            guard.uninstall()

    res["generated_at"] = datetime.now(timezone.utc).isoformat()
    res["methods"] = methods_block(
        scorer, res, "dry_run (judge stubbed: exact-normalized only, zero "
        "LLM calls)" if dry_run else "full (frozen equivalence judge, budget "
        "stage 'external')")
    if guard is not None:
        res["methods"]["guard_fresh_calls"] = guard.fresh
    res.pop("_counters")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    out_md.write_text(render_report(res))
    print(f"wrote {out_json}\nwrote {out_md}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("score", help="score the D2 one-shot test run")
    sc.add_argument("--cases", type=Path, default=CASES_DIR)
    sc.add_argument("--out-dir", type=Path, default=OUT_DIR)
    sc.add_argument("--dry-run", action="store_true",
                    help="stub the judge (exact-normalized only, no LLM "
                    "calls); writes scores_d2_dryrun.{json,md}")
    sc.add_argument("--call-cap", type=int, default=None)
    sc.add_argument("--force", action="store_true",
                    help="overwrite an existing scores_d2.json (logged "
                    "infrastructure-failure rerun only)")
    args = ap.parse_args()
    res = asyncio.run(run_score(args.cases, args.out_dir, args.dry_run,
                                args.call_cap, args.force))
    ep = res["endpoints"]
    print(f"\nscored {res['n_patches_scored']} patches "
          f"{json.dumps(res['by_family'])}; "
          f"judge_calls={res['methods']['judge_calls']}, "
          f"dry_unresolved={res['methods']['dry_unresolved_pairs']}")
    for cell in ALL_CELLS:
        land = ep["landing_coverage"]["any_landed"][cell]["pooled"]
        probe = ep["E_probe"][cell]["pooled_itt"]
        print(f"  {cell}: any_landed {sd1._fmt(land)} | "
              f"probe {sd1._fmt(probe)}")


if __name__ == "__main__":
    main()
