"""staging_funnel.py — protocol v2.3 N2: held-statement gap characterization.

Preregistration (protocol/CHANGELOG.md, entry "Protocol v2.3", N2): for every
hedge entering a gated cell — planted spec_* rows and the frozen natural
stratum, both extraction families, both benchmarks — emit ONE per-hedge
funnel record

    hedge -> tuple extracted -> key mapped -> routed speculative
          -> staged with predicates

with the drop-out stage and a mechanism tag at each loss point, built from
EXISTING FROZEN LOGS (target: 0 fresh LLM calls; this module never invokes
llm.call_llm — the only "LLM" reads are frozen results/cache/ files written
by past runs).

Populations (all four funnel stages probed per (hedge, family)):

  planted_spec  the 360 manifest spec_gold/spec_aux rows (LME 240, LoCoMo
                120), evaluated inside the frozen v2 MAIN gated cells
                (results/v2/{mem0,zep}_gated/timelines) — 720 records.
  planted_t3    the 100 t3_staging hedges (50/benchmark) evaluated through
                the frozen per-family SYN-RES gate entries
                (results/v2/test3_{em,ez}/predicates_manifest.jsonl) — 200
                records. This population is included because the v2.2
                sweep's staging-completeness numbers (the preregistered
                reconciliation target — the "74-86%" band of
                docs/RESULTS_V22.md C2) are DEFINED on it; without it the
                reconciliation is impossible. Its "key mapped" stage is
                vacuous (t3 staging keys are fresh aux keys bound by
                construction; no key-mapper call exists in that pathway).
  natural       the 45 frozen §8-rule-5 expansion-stratum rows
                (results/v2.2/rule5_expanded_rows.jsonl), evaluated in their
                host gated cells (results/v2 LoCoMo cells / results/v2.2
                pair-run cells) — 90 records.

RECONCILIATION (hard-asserted at build time, `reconcile()`):
  * per t3 row: the funnel's bottom-up staged determination (classifier
    argmax == speculative AND predicates present AND final lint clean) must
    equal the frozen manifest's `staged` flag;
  * aggregated per (family, benchmark): x/n must equal the v2.2 router
    sweep's frozen-argmax SYN-RES staging rates
    (results/v2.2/router_sweep/summary.json — em/lme 37/50 = 0.74,
    ez/lme 42/50 = 0.84, em/locomo 46/50 = 0.92, ez/locomo 44/50 = 0.88;
    the amendment's "74-86%" quotes the LME band of RESULTS_V22.md C2);
  * per natural row and family: the funnel's key-anchored staged
    determination must equal non-emptiness of the frozen row's
    hedged_values_speculative[family], and its "hedge reached the anchor
    key" determination must equal non-emptiness of hedged_values_all[family]
    (the rule-5 ungated witness — extraction/mapping are cache-shared
    across policies, so the gated log must agree).

Mechanism taxonomy — derived bottom-up from the frozen logs (see
results/v2.3/funnel_report.md for the system-agnostic definitions and the
realized counts):

  no-tuple-extracted                        dropout at `extracted`
  tuple-extracted-but-unmapped-key          dropout at `mapped` (FM-6 row is
                                            the evidence; detail records
                                            whether the target key was in
                                            the mapper's candidate list)
  mapped-but-classified-completed_event     dropout at `routed_speculative`
  mapped-but-classified-explicit_correction (anticipated; realized 0)
  mapped-but-classified-other               dropout at `routed_speculative`
                                            (additive branch: detail records
                                            additive_commit / _refresh /
                                            _noop — commit/refresh REIFY the
                                            hedged value as current)
  plan-shaped-attribute-committed-additively  the classified-other subset
                                            whose RAW extracted attribute
                                            matches the frozen v2.2
                                            plan-frame regex list
                                            (router_sweep.plan_frame_match)
  staged-but-predicate-lint-failed          dropout at
                                            `staged_with_predicates`
                                            (anticipated as "staged-but-
                                            predicate-lint-dropped"; in this
                                            implementation a residual lint
                                            failure after the single retry
                                            RETAINS the item, so the family
                                            is structurally empty here —
                                            realized 0 — but the stage is
                                            probed for portability)

Outputs:
  results/v2.3/staging_funnel.jsonl   one record per (hedge, family)
  results/v2.3/funnel_report.md       loss-by-stage tables, ranked
                                      mechanisms, reconciliation, method

CLI (repo root): pilot/.venv/bin/python experiments/v2_full/staging_funnel.py
"""

from __future__ import annotations

import collections
import dataclasses
import json
from pathlib import Path

import _bootstrap  # noqa: F401  (R1 layout: puts src/ + experiments/ on sys.path)

import extract
import locomo_data
import store
from natres import match_staged_tuple
from router_sweep import PLAN_FRAME_ATTRIBUTE_PATTERNS, plan_frame_match

from paths import GOLD_DIR, RESULTS_DIR, ROOT

V2_DIR = RESULTS_DIR / "v2"
V22_DIR = RESULTS_DIR / "v2.2"
OUT_DIR = RESULTS_DIR / "v2.3"
FUNNEL_PATH = OUT_DIR / "staging_funnel.jsonl"
REPORT_PATH = OUT_DIR / "funnel_report.md"

MANIFEST_PATH = V2_DIR / "injections" / "manifest.jsonl"
RULE5_ROWS_PATH = V22_DIR / "rule5_expanded_rows.jsonl"
EXPANSION_PATH = V22_DIR / "natural_stratum_expansion.jsonl"
PAIRS_PATH = GOLD_DIR / "natural_pairs_v2.jsonl"
SWEEP_SUMMARY_PATH = V22_DIR / "router_sweep" / "summary.json"
CACHE_DIR = RESULTS_DIR / "cache"

FAMILIES = ("em", "ez")
FAM_NAME = {"em": "mem0", "ez": "zep"}          # extraction-family dir names
FAM_GATED = {"em": "mem0_gated", "ez": "zep_gated"}
BENCHMARKS = ("lme", "locomo")

#: Funnel stages in order. `stage_reached` = LAST stage passed (None if the
#: hedge never even yielded a matching tuple); `dropout_stage` = FIRST stage
#: failed (None iff stage_reached == "staged_with_predicates").
STAGES = ("extracted", "mapped", "routed_speculative", "staged_with_predicates")

SPEC_TYPES = ("spec_gold", "spec_aux")


# ---------------------------------------------------------------------------
# small helpers (pure)
# ---------------------------------------------------------------------------

def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def value_carries(container: str, value: str) -> bool:
    """True when one normalized value string contains the other (the planted
    value survives paraphrase: 'Sleep Cycle' inside 'planning to move ... to
    Sleep Cycle'). Symmetric containment via store.normalize_value."""
    a, b = store.normalize_value(str(container)), store.normalize_value(str(value))
    return bool(a) and bool(b) and (a == b or b in a or a in b)


def match_hedge_tuple_indices(tuples: list[dict], planted_value: str,
                              hedge_text: str, entity: str,
                              attribute: str) -> list[int]:
    """Indices of extracted tuples that carry the hedge. Primary: normalized
    value containment against the planted value (tuple value or its
    statement_span). Fallback: the frozen M5 span match against the hedge
    text with the target key preferred (natres.match_staged_tuple — the
    machinery SYN-RES/NAT-RES staging used)."""
    hits = [j for j, t in enumerate(tuples)
            if value_carries(t["value"], planted_value)
            or value_carries(t.get("statement_span") or "", planted_value)]
    if hits:
        return hits
    matched, _method = match_staged_tuple(
        list(tuples), hedge_text, {"entity": entity, "attribute": attribute})
    if matched is not None:
        return [tuples.index(matched)]
    return []


# ---------------------------------------------------------------------------
# disposition of one extracted tuple inside a gated replay (pure)
# ---------------------------------------------------------------------------

def tuple_disposition(j: int, tup: dict, turn_gates: list[dict],
                      turn_unmapped: list[dict]) -> dict:
    """Where tuple #j of a turn landed in the frozen gate loop.

    Every extracted tuple is EITHER a gate_log row (mapped; its item_id ends
    with the tuple index) OR an FM-6 unmapped row (mapped_key 'none', full
    extracted_tuple retained) — asserted, never guessed.
    Returns {"kind": "staged"|"classified"|"unmapped", "gate_row"|
    "unmapped_row": <frozen log row>}.
    """
    gate = [g for g in turn_gates if g["item_id"].endswith(f"-{j}")]
    if gate:
        g = gate[0]
        return {"kind": "staged" if g["action"] == "staged" else "classified",
                "gate_row": g}
    unm = [u for u in turn_unmapped if u["extracted_tuple"] == tup]
    assert unm, (
        f"tuple #{j} ({tup.get('attribute')!r}) is in neither the gate_log "
        f"nor the unmapped log of its turn — frozen-log inconsistency")
    return {"kind": "unmapped", "unmapped_row": unm[0]}


_RANK = {"unmapped": 1, "classified": 2, "staged": 3}


def best_disposition(dispositions: list[tuple[int, dict, dict]],
                     target_key: str | None) -> tuple[int, dict, dict]:
    """ITT selection among the hedge's candidate tuples: the hedge is staged
    if ANY carrying tuple staged; otherwise prefer a mapped tuple over an
    unmapped one, a target-key-mapped tuple over another key, then the
    earliest tuple (deterministic)."""
    def rank(item):
        j, tup, disp = item
        on_target = int(disp.get("gate_row", {}).get("key") == target_key)
        return (_RANK[disp["kind"]], on_target, -j)
    return max(dispositions, key=rank)


# ---------------------------------------------------------------------------
# mechanism tagging (pure)
# ---------------------------------------------------------------------------

ADDITIVE_ACTIONS = ("additive_commit", "additive_refresh", "additive_noop")


def classify_mechanism(gate_row: dict, raw_attribute: str | None) -> tuple[str, dict]:
    """Mechanism tag for a mapped-but-not-staged tuple (dropout at
    routed_speculative). `raw_attribute` = the EXTRACTOR's attribute name
    (the plan-frame surface the v2.2 fix targeted); falls back to the mapped
    key's attribute when the raw tuple is unavailable."""
    cls = gate_row["class"]
    action = gate_row.get("action")
    attr = raw_attribute or gate_row["key"].split("/", 1)[1]
    plan = plan_frame_match(attr)
    detail = {"class": cls, "action": action, "raw_attribute": attr,
              "plan_frame_attribute": plan,
              "stored_value_at_gate": gate_row.get("stored_value")}
    if cls == "other" and plan:
        return "plan-shaped-attribute-committed-additively", detail
    return f"mapped-but-classified-{cls}", detail


def staged_record_fields(gate_row: dict | None, staged_row: dict | None) -> dict:
    """Terminal fields for a hedge that reached staging. A residual lint
    failure after the single retry fails the `staged_with_predicates` stage
    (mechanism `staged-but-predicate-lint-failed`) — anticipated family;
    realized 0 in the frozen v2 artifacts (the implementation retains such
    items, so nothing is silently dropped either way)."""
    lint = (staged_row or {}).get("lint") or {}
    residual = lint.get("final_failures") or []
    if residual:
        return {"stage_reached": "routed_speculative",
                "dropout_stage": "staged_with_predicates",
                "mechanism_tag": "staged-but-predicate-lint-failed",
                "mechanism_detail": {"final_failures": residual,
                                     "retained_in_store": True}}
    return {"stage_reached": "staged_with_predicates",
            "dropout_stage": None, "mechanism_tag": None,
            "mechanism_detail": None}


# ---------------------------------------------------------------------------
# per-population record builders (pure given the frozen rows)
# ---------------------------------------------------------------------------

def _base_record(*, hedge_id, source_id, population, benchmark, family,
                 stream_id, cell_path, anchor, key, hedged_value, hedge_text):
    return {
        "hedge_id": hedge_id, "hedge_source_id": source_id,
        "population": population, "benchmark": benchmark, "family": family,
        "stream_id": stream_id, "cell": cell_path,
        "anchor": {"session_idx": anchor[0], "turn_idx": anchor[1]},
        "key": key, "hedged_value": hedged_value, "hedge_text": hedge_text,
    }


def spec_funnel_record(man_row: dict, family: str, tuples: list[dict],
                       turn_gates: list[dict], turn_unmapped: list[dict],
                       staged_by_id: dict[str, dict], cell_path: str) -> dict:
    """Funnel record for one planted spec_* hedge inside one family's frozen
    main gated cell."""
    target_key = f"{man_row['entity']}/{man_row['attribute']}"
    rec = _base_record(
        hedge_id=f"{man_row['injection_id']}::{family}",
        source_id=man_row["injection_id"], population="planted_spec",
        benchmark=man_row["benchmark"], family=family,
        stream_id=man_row["question_id"], cell_path=cell_path,
        anchor=(man_row["placement"]["session_idx"],
                man_row["placement"]["turn_idx"]),
        key=target_key, hedged_value=man_row["value"],
        hedge_text=man_row["text"])
    hits = match_hedge_tuple_indices(
        tuples, man_row["value"], man_row["text"],
        man_row["entity"], man_row["attribute"])
    if not hits:
        rec.update(stage_reached=None, dropout_stage="extracted",
                   mechanism_tag="no-tuple-extracted", mechanism_detail=None,
                   evidence={"source": cell_path,
                             "note": "no extracted tuple at the placement "
                                     "turn carries the planted value (norm-"
                                     "containment) or span-matches the aside",
                             "turn_tuples": [(t["entity"], t["attribute"],
                                              t["value"]) for t in tuples]})
        return rec
    dispositions = [(j, tuples[j],
                     tuple_disposition(j, tuples[j], turn_gates, turn_unmapped))
                    for j in hits]
    j, tup, disp = best_disposition(dispositions, target_key)
    rec["matched_tuple"] = {"index": j, "entity": tup["entity"],
                            "attribute": tup["attribute"], "value": tup["value"],
                            "n_candidate_tuples": len(hits)}
    _finish_from_disposition(rec, tup, disp, staged_by_id, cell_path, target_key)
    return rec


def _finish_from_disposition(rec: dict, tup: dict, disp: dict,
                             staged_by_id: dict, cell_path: str,
                             target_key: str | None) -> None:
    """Shared terminal logic for a hedge attributed to one extracted tuple
    inside a gated replay (spec + natural span-fallback paths)."""
    if disp["kind"] == "unmapped":
        u = disp["unmapped_row"]
        rec.update(
            stage_reached="extracted", dropout_stage="mapped",
            mechanism_tag="tuple-extracted-but-unmapped-key",
            mechanism_detail={
                "raw_attribute": tup["attribute"],
                "plan_frame_attribute": plan_frame_match(tup["attribute"]),
                "target_key_in_candidates": (target_key in u["candidate_keys"]
                                             if target_key else None),
                "mapper_rationale": u.get("rationale")},
            evidence={"source": f"{cell_path}#unmapped", "row": u})
        return
    g = disp["gate_row"]
    on_target = target_key is None or g["key"] == target_key
    if disp["kind"] == "staged":
        srow = staged_by_id.get(g.get("staged_id"))
        rec.update(staged_record_fields(g, srow))
        rec["evidence"] = {"source": f"{cell_path}#gate_log+staged_log",
                           "gate_row": g,
                           "staged_row": srow}
        if not on_target:
            rec.setdefault("mechanism_detail", None)
            rec["cross_key"] = {"mapped_key": g["key"], "target_key": target_key}
        return
    tag, detail = classify_mechanism(g, tup.get("attribute"))
    if not on_target:
        detail["cross_key"] = {"mapped_key": g["key"], "target_key": target_key}
    rec.update(stage_reached="mapped", dropout_stage="routed_speculative",
               mechanism_tag=tag, mechanism_detail=detail,
               evidence={"source": f"{cell_path}#gate_log", "row": g})


def t3_funnel_record(t3_row: dict, family: str, hedge_text: str | None) -> dict:
    """Funnel record for one t3_staging hedge through the frozen per-family
    SYN-RES gate entry. The `mapped` stage is VACUOUS here: t3 staging keys
    are fresh aux keys bound by construction (no key-mapper call exists in
    the pathway) — a tuple that is extracted passes `mapped` by definition."""
    src = f"results/v2/test3_{family}/predicates_manifest.jsonl"
    rec = _base_record(
        hedge_id=f"{t3_row['case_id']}::{family}",
        source_id=t3_row["case_id"], population="planted_t3",
        benchmark=t3_row["benchmark"], family=family,
        stream_id=t3_row["question_id"], cell_path=src,
        anchor=(t3_row["staged_at_turn"]["session_idx"],
                t3_row["staged_at_turn"]["turn_idx"]),
        key=f"{t3_row['entity']}/{t3_row['attribute']}",
        hedged_value=t3_row["designed_value"], hedge_text=hedge_text)
    rec["mapped_stage_vacuous"] = True
    tup = t3_row.get("staged_tuple")
    if tup is None:
        assert t3_row["outcome_reason"] == "extraction_miss"
        rec.update(stage_reached=None, dropout_stage="extracted",
                   mechanism_tag="no-tuple-extracted", mechanism_detail=None,
                   evidence={"source": src,
                             "row": {k: t3_row[k] for k in
                                     ("case_id", "outcome_reason", "extraction")}})
        return rec
    rec["matched_tuple"] = {"entity": tup["entity"], "attribute": tup["attribute"],
                            "value": tup["value"],
                            "match_method": t3_row["extraction"]["match_method"]}
    cls = t3_row["classifier"]["argmax"]
    if cls != "speculative":
        # storeless gate entry: the additive/commit branch bodies never ran,
        # so `action` is None — the class alone is the routing witness.
        gate_like = {"class": cls, "action": None,
                     "key": rec["key"], "stored_value": None}
        tag, detail = classify_mechanism(gate_like, tup["attribute"])
        detail["storeless_gate_entry"] = True
        rec.update(stage_reached="mapped", dropout_stage="routed_speculative",
                   mechanism_tag=tag, mechanism_detail=detail,
                   evidence={"source": src,
                             "row": {"case_id": t3_row["case_id"],
                                     "outcome_reason": t3_row["outcome_reason"],
                                     "classifier": t3_row["classifier"],
                                     "staged_tuple": tup}})
        return rec
    assert t3_row["staged"], (
        f"{t3_row['case_id']}/{family}: classifier argmax speculative but "
        f"manifest staged={t3_row['staged']} — frozen-route inconsistency")
    staged_like = {"lint": t3_row.get("lint")}
    rec.update(staged_record_fields(None, staged_like))
    rec["evidence"] = {"source": src,
                       "row": {"case_id": t3_row["case_id"], "staged": True,
                               "promote_when": t3_row["promote_when"],
                               "invalidate_when": t3_row["invalidate_when"],
                               "lint": t3_row["lint"]}}
    return rec


def natural_funnel_record(r5_row: dict, family: str, hedge_text: str,
                          tuples: list[dict], turn_gates: list[dict],
                          turn_unmapped: list[dict],
                          staged_by_id: dict[str, dict],
                          cell_path: str) -> dict:
    """Funnel record for one frozen natural-stratum hedge in one family's
    host gated cell. Attribution precedence (documented in the report):

      1. KEY-ANCHORED (the frozen rule-5 convention): gate rows at the
         anchor turn on the row's schema key + store. Staged iff any such
         row staged; otherwise the first such row's class is the dropout.
      2. else the frozen M5 span/key match picks the hedge's tuple and its
         disposition (mapped to another key / unmapped) applies.
      3. else no-tuple-extracted.
    """
    key_name = r5_row["key"]
    store_id = r5_row.get("store_id", r5_row["entity"])
    rec = _base_record(
        hedge_id=f"{r5_row['row_id']}::{family}",
        source_id=r5_row["row_id"], population="natural",
        benchmark=("locomo" if r5_row["sample_id"].startswith("conv-") else "lme"),
        family=family, stream_id=r5_row["sample_id"], cell_path=cell_path,
        anchor=tuple(r5_row["anchor"]), key=key_name,
        hedged_value=(r5_row["hedged_values_all"][family] or None),
        hedge_text=hedge_text)
    rec["row_kind"] = r5_row["row_kind"]
    rec["frozen_crosscheck"] = {
        "hedged_values_all": r5_row["hedged_values_all"][family],
        "hedged_values_speculative": r5_row["hedged_values_speculative"][family]}

    anchored = [g for g in turn_gates
                if g["key"] == key_name and g["store_id"] == store_id]
    if anchored:
        staged_rows = [g for g in anchored if g["action"] == "staged"]
        if staged_rows:
            g = staged_rows[0]
            srow = staged_by_id.get(g.get("staged_id"))
            rec.update(staged_record_fields(g, srow))
            rec["evidence"] = {"source": f"{cell_path}#gate_log+staged_log",
                               "gate_row": g, "staged_row": srow}
        else:
            g = anchored[0]
            tag, detail = classify_mechanism(g, None)
            detail["n_key_anchored_rows"] = len(anchored)
            rec.update(stage_reached="mapped",
                       dropout_stage="routed_speculative",
                       mechanism_tag=tag, mechanism_detail=detail,
                       evidence={"source": f"{cell_path}#gate_log", "row": g})
        return rec

    matched, method = match_staged_tuple(
        list(tuples), hedge_text,
        {"entity": r5_row["entity"], "attribute": r5_row["attribute"]})
    if matched is None:
        rec.update(stage_reached=None, dropout_stage="extracted",
                   mechanism_tag="no-tuple-extracted", mechanism_detail=None,
                   evidence={"source": cell_path,
                             "note": "no anchor-turn tuple span/key-matches "
                                     "the hedge quote (frozen M5 matcher)",
                             "turn_tuples": [(t["entity"], t["attribute"],
                                              t["value"]) for t in tuples]})
        return rec
    j = tuples.index(matched)
    rec["matched_tuple"] = {"index": j, "entity": matched["entity"],
                            "attribute": matched["attribute"],
                            "value": matched["value"], "match_method": method}
    disp = tuple_disposition(j, matched, turn_gates, turn_unmapped)
    _finish_from_disposition(rec, matched, disp, staged_by_id, cell_path, key_name)
    return rec


# ---------------------------------------------------------------------------
# frozen-artifact loaders (I/O layer)
# ---------------------------------------------------------------------------

_TL_CACHE: dict[tuple, dict] = {}


def load_timeline(base: Path, cell: str, stream_id: str) -> dict:
    k = (str(base), cell, stream_id)
    if k not in _TL_CACHE:
        _TL_CACHE[k] = json.loads(
            (base / cell / "timelines" / f"{stream_id}.json").read_text())
    return _TL_CACHE[k]


def turn_rows(timeline: dict, si: int, ti: int) -> tuple[list[dict], list[dict]]:
    gates = [g for g in timeline["gate_log"]
             if (g["session_idx"], g["turn_idx"]) == (si, ti)]
    unmapped = [u for u in timeline["unmapped"]
                if (u["session_idx"], u["turn_idx"]) == (si, ti)]
    return gates, unmapped


def staged_index(timeline: dict) -> dict[str, dict]:
    return {s["staged_id"]: s for s in timeline["staged_log"]}


_DUMP_CACHE: dict[tuple, dict] = {}


def dump_tuples(benchmark: str, fam_name: str) -> dict[tuple, dict]:
    """(question_id, session_idx, turn_idx) -> extraction dump row of the
    frozen PERTURBED stream (results/v2/extraction/) — the exact per-turn
    tuple lists (with statement spans) the v2 cells replayed."""
    k = (benchmark, fam_name)
    if k not in _DUMP_CACHE:
        idx = {}
        for r in _read_jsonl(V2_DIR / "extraction"
                             / f"{benchmark}_{fam_name}_perturbed.jsonl"):
            idx[(r["question_id"], r["session_idx"], r["turn_idx"])] = r
        _DUMP_CACHE[k] = idx
    return _DUMP_CACHE[k]


_EC_CACHE: dict[tuple, dict] = {}


def v22_anchor_tuples(cell: str, stream_id: str, si: int, ti: int,
                      timeline: dict) -> list[dict]:
    """Anchor-turn tuple list for a v2.2 pair-run host (no extraction dump
    exists for that run): snapshots give the anchor's user_pos, the cell's
    extraction_calls.jsonl gives the llm cache key at that position, and the
    FROZEN results/cache/ entry replays the parse (extract.parse_tuples, the
    B2 contract) — zero LLM calls."""
    k = ("v2.2", cell)
    if k not in _EC_CACHE:
        _EC_CACHE[k] = {r["question_id"]: r["cache_keys"] for r in
                        _read_jsonl(V22_DIR / cell / "extraction_calls.jsonl")}
    ups = [s["user_pos"] for s in timeline["snapshots"]
           if (s["session_idx"], s["turn_idx"]) == (si, ti)]
    assert len(ups) == 1, (stream_id, si, ti, ups)
    cache_key = _EC_CACHE[k][stream_id][ups[0]]
    out = json.loads((CACHE_DIR / f"{cache_key}.json").read_text())["output"]
    return [dataclasses.asdict(t) for t in extract.parse_tuples(out)]


def natural_hedge_texts() -> dict[str, str]:
    """row_id -> verbatim hedge quote, joined from the frozen sources:
    natural_pairs_v2.jsonl (P-LOCO-*), the expansion freeze list (XPAIR-*/
    XNAT-*), and the LoCoMo schema's censored hypotheses (CENS-*)."""
    texts: dict[str, str] = {}
    for p in _read_jsonl(PAIRS_PATH):
        texts[p["pair_id"]] = p["stage"]["text"]
    for r in _read_jsonl(EXPANSION_PATH):
        texts[r["row_id"]] = r["text"]
    for c in locomo_data.censored_staged_hypotheses():
        texts[f"CENS-{c['sample_id']}-{c['dia_id']}"] = c["quote"]
    return texts


def _r5_base(r5_row: dict) -> Path:
    """Host results tree of a frozen rule-5 row. The artifact records an
    absolute path in `_base` (or omits it: the frozen v2 tree) — resolved by
    tree NAME under the current RESULTS_DIR so the artifact stays portable."""
    raw = r5_row.get("_base")
    return RESULTS_DIR / Path(raw).name if raw else V2_DIR


# ---------------------------------------------------------------------------
# population builders
# ---------------------------------------------------------------------------

def build_spec_records() -> list[dict]:
    man = [r for r in _read_jsonl(MANIFEST_PATH)
           if r["inj_type"] in SPEC_TYPES]
    records = []
    for row in man:
        si, ti = row["placement"]["session_idx"], row["placement"]["turn_idx"]
        for family in FAMILIES:
            fam_name = FAM_NAME[family]
            cell = FAM_GATED[family]
            dump = dump_tuples(row["benchmark"], fam_name)
            d = dump[(row["question_id"], si, ti)]
            assert row["text"] in d["turn_text"], (
                f"{row['injection_id']}: aside text not found in the frozen "
                f"perturbed stream turn — placement mismatch")
            tl = load_timeline(V2_DIR, cell, row["question_id"])
            gates, unmapped = turn_rows(tl, si, ti)
            records.append(spec_funnel_record(
                row, family, d["tuples"], gates, unmapped, staged_index(tl),
                _rel(V2_DIR / cell / "timelines" / f"{row['question_id']}.json")))
    return records


def build_t3_records() -> list[dict]:
    texts = {r["t3_case_id"]: r["text"] for r in _read_jsonl(MANIFEST_PATH)
             if r["inj_type"] == "t3_staging"}
    records = []
    for family in FAMILIES:
        for row in _read_jsonl(V2_DIR / f"test3_{family}"
                               / "predicates_manifest.jsonl"):
            records.append(t3_funnel_record(row, family,
                                            texts.get(row["case_id"])))
    return records


def build_natural_records() -> list[dict]:
    texts = natural_hedge_texts()
    records = []
    for row in _read_jsonl(RULE5_ROWS_PATH):
        base = _r5_base(row)
        si, ti = row["anchor"]
        for family in FAMILIES:
            cell = FAM_GATED[family]
            tl = load_timeline(base, cell, row["sample_id"])
            if base == V22_DIR:
                tuples = v22_anchor_tuples(cell, row["sample_id"], si, ti, tl)
            else:
                d = dump_tuples("locomo", FAM_NAME[family])[
                    (row["sample_id"], si, ti)]
                tuples = d["tuples"]
            gates, unmapped = turn_rows(tl, si, ti)
            rec = natural_funnel_record(
                row, family, texts[row["row_id"]], tuples, gates, unmapped,
                staged_index(tl),
                _rel(base / cell / "timelines" / f"{row['sample_id']}.json"))
            records.append(rec)
    return records


# ---------------------------------------------------------------------------
# reconciliation (preregistered hard checks)
# ---------------------------------------------------------------------------

def reconcile(records: list[dict]) -> dict:
    """Assert the funnel against the frozen v2.2 numbers.

    (1) t3 staged-with-predicates x/n per (family, benchmark) == the router
        sweep's frozen-argmax SYN-RES staging rates (the C2 staging-
        completeness numbers: LME 0.74/0.84 — the amendment's 74-86% band —
        and LoCoMo 0.92/0.88); plus PER ROW, in both directions, the funnel
        staged determination == the frozen manifest's `staged` flag.
    (2) per natural row+family: staged determination == non-emptiness of the
        frozen hedged_values_speculative; reaching the anchor key ==
        non-emptiness of hedged_values_all (the rule-5 witness).
    """
    summary = json.loads(SWEEP_SUMMARY_PATH.read_text())
    target = summary["points"]["argmax"]["SYNRES_staging_rate"]
    out = {"t3_vs_sweep": {}, "natural_agreement": {"n": 0, "mismatches": []}}
    t3 = [r for r in records if r["population"] == "planted_t3"]
    for family in FAMILIES:
        for bench in BENCHMARKS:
            sub = [r for r in t3
                   if r["family"] == family and r["benchmark"] == bench]
            x = sum(r["stage_reached"] == "staged_with_predicates" for r in sub)
            t = target[family][bench]["frozen_argmax"]
            entry = {"funnel_x": x, "funnel_n": len(sub),
                     "sweep_x": t["x"], "sweep_n": t["n"],
                     "rate": (x / len(sub)) if sub else None,
                     "match": x == t["x"] and len(sub) == t["n"]}
            out["t3_vs_sweep"][f"{family}/{bench}"] = entry
            assert entry["match"], (
                f"RECONCILIATION FAILURE {family}/{bench}: funnel "
                f"{x}/{len(sub)} vs sweep frozen-argmax {t['x']}/{t['n']}")
    # Per-row, BOTH directions (M-4): funnel staged determination == the
    # frozen manifest's `staged` flag (the aggregate x/n above alone could
    # hide a symmetric pair of opposite-direction per-row errors in a cell).
    for family in FAMILIES:
        flags = {r["case_id"]: bool(r["staged"]) for r in _read_jsonl(
            V2_DIR / f"test3_{family}" / "predicates_manifest.jsonl")}
        for r in (x for x in t3 if x["family"] == family):
            assert ((r["stage_reached"] == "staged_with_predicates")
                    == flags[r["hedge_source_id"]]), (
                f"t3 per-row staged mismatch vs frozen manifest: {r['hedge_id']}")
    nat = [r for r in records if r["population"] == "natural"]
    for r in nat:
        frozen_spec = bool(r["frozen_crosscheck"]["hedged_values_speculative"])
        frozen_all = bool(r["frozen_crosscheck"]["hedged_values_all"])
        staged = r["stage_reached"] == "staged_with_predicates"
        # "reached the anchor key": a key-anchored gate row existed (staged
        # or classified evidence on the row's key), i.e. the evidence gate
        # row's key equals the row key.
        ev = r.get("evidence", {})
        grow = ev.get("gate_row") or (ev.get("row")
                                      if "gate_row" not in ev else None)
        on_key = bool(grow and grow.get("key") == r["key"]
                      and "class" in grow)
        out["natural_agreement"]["n"] += 1
        if staged != frozen_spec or on_key != frozen_all:
            out["natural_agreement"]["mismatches"].append(r["hedge_id"])
    assert not out["natural_agreement"]["mismatches"], (
        "natural-row cross-check failed for: "
        f"{out['natural_agreement']['mismatches']}")
    return out


# ---------------------------------------------------------------------------
# aggregation + report
# ---------------------------------------------------------------------------

def loss_table(records: list[dict]) -> dict:
    """Per (population, benchmark, family): n, survivors per stage
    (cumulative), and dropouts per stage."""
    out: dict = {}
    for r in records:
        k = (r["population"], r["benchmark"], r["family"])
        e = out.setdefault(k, {"n": 0,
                               "reached": {s: 0 for s in STAGES},
                               "lost_at": {s: 0 for s in STAGES}})
        e["n"] += 1
        reached_idx = (-1 if r["stage_reached"] is None
                       else STAGES.index(r["stage_reached"]))
        for i, s in enumerate(STAGES):
            if i <= reached_idx:
                e["reached"][s] += 1
        if r["dropout_stage"] is not None:
            e["lost_at"][r["dropout_stage"]] += 1
    return out


def mechanism_ranking(records: list[dict]) -> list[tuple[str, int]]:
    c = collections.Counter(r["mechanism_tag"] for r in records
                            if r["mechanism_tag"])
    return c.most_common()


def run(out_dir: Path = OUT_DIR) -> dict:
    records = build_spec_records() + build_t3_records() + build_natural_records()
    n_expected = 360 * 2 + 100 * 2 + 45 * 2
    assert len(records) == n_expected, (len(records), n_expected)
    recon = reconcile(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "staging_funnel.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = render_report(records, recon)
    (out_dir / "funnel_report.md").write_text(report)
    return {"n_records": len(records), "reconciliation": recon,
            "funnel_path": str(out_dir / "staging_funnel.jsonl"),
            "report_path": str(out_dir / "funnel_report.md")}


# --- report rendering -------------------------------------------------------

def _pct(x: int, n: int) -> str:
    return f"{x}/{n} ({x / n:.0%})" if n else "–"


def _stage_table(table: dict, population: str) -> list[str]:
    lines = ["| benchmark | family | n | extracted | mapped | routed spec. "
             "| staged w/ predicates | lost@extract | lost@map | lost@route "
             "| lost@stage |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (pop, bench, fam), e in sorted(table.items()):
        if pop != population:
            continue
        n = e["n"]
        lines.append(
            f"| {bench} | {fam} | {n} | "
            + " | ".join(_pct(e["reached"][s], n) for s in STAGES) + " | "
            + " | ".join(str(e["lost_at"][s]) for s in STAGES) + " |")
    return lines


def render_report(records: list[dict], recon: dict) -> str:
    table = loss_table(records)
    parts: list[str] = []
    a = parts.append
    a("# v2.3 N2 — staging-funnel report (held-statement gap "
      "characterization)\n")
    a("Preregistration: protocol/CHANGELOG.md, entry \"Protocol v2.3\", N2. "
      "Built exclusively from frozen artifacts; **fresh LLM calls: 0** "
      "(this module contains no LLM invocation; the only cache reads replay "
      "frozen extraction outputs).\n")

    a("## 1. Method — the funnel, system-agnostically\n")
    a("A *hedge* is a hedged/speculative statement about a trackable "
      "attribute (planted or naturally occurring). For a staged-memory "
      "system — any system that (i) extracts candidate facts from "
      "conversation turns, (ii) resolves them to canonical attribute keys, "
      "(iii) routes each resolved fact to a revision action, and (iv) holds "
      "speculative facts in a pending buffer with explicit resolution "
      "conditions instead of committing them — every hedge either survives "
      "four probe stages or drops out at exactly one of them:\n")
    a("1. **extracted** — some extracted fact-tuple *carries* the hedge, "
      "per the frozen stage-1 attribution rule (fully specified below).\n"
      "2. **mapped** — that tuple received a canonical key from the key-"
      "resolution step (an unmapped tuple is logged, never silently "
      "dropped, so the evidence row always exists).\n"
      "3. **routed_speculative** — the routing step sent the tuple to the "
      "staging branch (here: frozen classifier argmax = `speculative`).\n"
      "4. **staged_with_predicates** — the staged item exists in the "
      "pending buffer with machine-checkable promote/invalidate predicates "
      "that pass lint.\n")
    a("**Stage-1 attribution rule (frozen; two tiers, in order).**\n"
      "(a) *Value containment*: a tuple carries the hedge when its `value` "
      "(or its `statement_span`), after value normalization, equals or "
      "contains-either-way the normalized planted/hedged value. "
      "Normalization (reference implementation `src/memory_gating/store.py`"
      ", `normalize_value`): lowercase + whitespace collapse, trailing-"
      "punctuation strip, number-words→digits, frequency/currency/clock-"
      "time canonicalization.\n"
      "(b) *The frozen M5 span match* (`src/memory_gating/natres.py`, "
      "`match_staged_tuple`, built on `label.match_tuple_to_span`), used "
      "when (a) matches nothing: score every tuple's `statement_span` "
      "against the hedge quote by token-containment overlap "
      "|A∩B| / min(|A|, |B|) over the two spans' lowercased alphanumeric "
      "token sets; the best-scoring tuple is the match iff its score "
      "≥ 0.5, with ties at the best score broken in favor of a tuple whose "
      "normalized (entity, attribute) equals the target key (the v2.2 C1 "
      "tie-break); if no span reaches 0.5, a tuple whose normalized "
      "(entity, attribute) equals the target key matches (highest "
      "extractor confidence on ties); otherwise no tuple carries the hedge "
      "(`no-tuple-extracted`). Natural rows apply the frozen rule-5 "
      "key-anchored convention BEFORE this rule (§2).\n")
    a("`stage_reached` records the LAST stage passed (null when even stage "
      "1 fails); `dropout_stage` the FIRST stage failed (null iff the hedge "
      "is fully staged). Each dropout carries a `mechanism_tag` (taxonomy "
      "below, derived bottom-up from the logs) and `evidence` — the verbatim "
      "frozen log row that shows the drop. To instantiate the funnel on "
      "another system you need only: per-turn extraction outputs, a key-"
      "resolution log that retains unmapped tuples, the per-tuple routing "
      "decision, and the pending-buffer rows with their lint status.\n")

    a("### Record schema (one JSONL row per hedge × extraction family)\n")
    a("```\nhedge_id            \"<source_id>::<family>\"\n"
      "hedge_source_id     manifest injection_id / t3 case_id / rule-5 row_id\n"
      "population          planted_spec | planted_t3 | natural\n"
      "benchmark           lme | locomo\n"
      "family              em | ez (the two frozen extraction families)\n"
      "stream_id           substrate / conversation id\n"
      "cell                repo-relative path of the frozen host artifact\n"
      "anchor              {session_idx, turn_idx} of the hedge turn\n"
      "key                 target canonical key (entity/attribute)\n"
      "hedged_value        planted/designed value (or frozen hedged set)\n"
      "hedge_text          the verbatim hedge sentence/quote\n"
      "matched_tuple       the attributed extractor tuple (when stage 1 passes)\n"
      "stage_reached       null|extracted|mapped|routed_speculative|staged_with_predicates\n"
      "dropout_stage       null | first failed stage\n"
      "mechanism_tag       null | taxonomy tag (dropouts only)\n"
      "mechanism_detail    tag-specific fields (action, raw attribute,\n"
      "                    plan-frame flag, candidate-list membership, ...)\n"
      "evidence            {source: artifact path, row(s): frozen log row(s)}\n```\n")

    a("## 2. Populations and hosts\n")
    a("| population | hedges | records (×2 families) | host logs |\n"
      "|---|---|---|---|\n"
      "| planted_spec (spec_gold + spec_aux) | 360 (lme 240, locomo 120) | "
      "720 | frozen v2 main gated cells (results/v2/{mem0,zep}_gated) + "
      "results/v2/extraction dumps |\n"
      "| planted_t3 (t3_staging) | 100 (50/benchmark) | 200 | frozen "
      "SYN-RES gate entries (results/v2/test3_{em,ez}/"
      "predicates_manifest.jsonl) |\n"
      "| natural (frozen §8-rule-5 stratum) | 45 (lme 24, locomo 21) | 90 | "
      "host gated cells: results/v2 LoCoMo cells, results/v2.2 pair-run "
      "cells (+ frozen llm cache for anchor-turn tuples) |\n")
    a("planted_t3 is included because the preregistered reconciliation "
      "target — the v2.2 sweep's staging-completeness numbers — is defined "
      "on it. Its `mapped` stage is vacuous (fresh aux keys bound by "
      "construction; `mapped_stage_vacuous: true` on every record). The "
      "natural rows use the frozen rule-5 key-anchored convention first and "
      "the M5 span match only when no anchor-key gate row exists.\n")

    a("## 3. Loss by stage\n")
    a("Survivor columns are cumulative (share of n that passed the stage); "
      "`lost@` columns are the dropouts at that stage.\n")
    for pop, label_ in (("planted_spec", "Planted spec_* hedges in the main "
                                         "gated cells"),
                        ("planted_t3", "Planted t3/SYN-RES hedges (the "
                                       "reconciliation population)"),
                        ("natural", "Natural hedges (frozen rule-5 "
                                    "stratum)")):
        a(f"### {label_}\n")
        a("\n".join(_stage_table(table, pop)) + "\n")

    a("## 4. Ranked mechanisms\n")
    a("### 4.1 The preregistered 14–26% gap (planted_t3, the population the "
      "changelog's \"14–26%\" refers to)\n")
    t3_recs = [r for r in records if r["population"] == "planted_t3"]
    a(_mech_table(t3_recs))
    a("### 4.2 Planted spec_* hedges in the main cells\n")
    spec_recs = [r for r in records if r["population"] == "planted_spec"]
    a(_mech_table(spec_recs))
    a("### 4.3 Natural hedges\n")
    nat_recs = [r for r in records if r["population"] == "natural"]
    a(_mech_table(nat_recs))
    a("### 4.4 Pooled ranking (all populations)\n")
    a(_mech_table(records))

    a("## 5. Mechanism taxonomy (bottom-up; every family documented)\n")
    a(_taxonomy_section(records))

    a("## 6. Reconciliation against the v2.2 sweep\n")
    a("The v2.2 router sweep's staging-completeness numbers (docs/"
      "RESULTS_V22.md C2; results/v2.2/router_sweep/summary.json "
      "`SYNRES_staging_rate`, frozen-argmax) are the preregistered target. "
      "The amendment's \"74–86%\" band quotes the LME table (0.74 em / 0.84 "
      "ez at the frozen argmax point, up to 0.76/0.86 at the confidence "
      "point); the LoCoMo frozen-argmax rates from the same summary are "
      "0.92/0.88. The funnel's staged-with-predicates rates on the "
      "planted_t3 population, derived bottom-up (classifier argmax + "
      "predicate/lint evidence, never the manifest's `staged` flag), must "
      "reproduce all four frozen-argmax numbers exactly — **asserted at "
      "build time**:\n")
    a("| family/benchmark | funnel x/n | sweep frozen-argmax x/n | rate | "
      "match |\n|---|---|---|---|---|")
    for k, e in sorted(recon["t3_vs_sweep"].items()):
        a(f"| {k} | {e['funnel_x']}/{e['funnel_n']} | "
          f"{e['sweep_x']}/{e['sweep_n']} | {e['rate']:.2f} | "
          f"{'PASS' if e['match'] else 'FAIL'} |")
    na = recon["natural_agreement"]
    a(f"\nNatural cross-check: {na['n']}/{na['n']} (family × row) records "
      "agree with the frozen rule-5 artifact — staged determination vs "
      "`hedged_values_speculative` non-emptiness AND anchor-key arrival vs "
      "`hedged_values_all` non-emptiness (the rule-5 ungated witness; "
      "extraction/key-mapping are cache-shared across commit policies, so "
      "the gated logs must and do agree). Mismatches: "
      f"{na['mismatches'] or 'none'}.\n")
    a("The planted_spec staging rates (§3) are NOT expected to equal the "
      "t3 band and do not: the t3 staging asides were designed to extract "
      "onto fresh aux keys with no key-mapping step, while spec_* hedges "
      "must traverse the mapper onto existing current-state keys — the "
      "funnel shows that difference is almost entirely the `mapped` stage "
      "(see §7).\n")

    a("## 7. Findings\n")
    a(_findings_section(records, table))

    a("## 8. Coverage, determinism, budget\n")
    a(f"- Records: {len(records)} = 720 planted_spec + 200 planted_t3 + 90 "
      "natural; every (hedge, family) pair attributed — no row skipped.\n"
      "- Fresh LLM calls: 0 (target met). All evidence is frozen: gate/"
      "staged/unmapped logs, extraction dumps, the t3 predicate manifests, "
      "the rule-5 row artifact, the router-sweep summary, and frozen llm-"
      "cache entries (read-only replay of past extraction outputs).\n"
      "- Deterministic: pure functions over the frozen rows; re-running "
      "reproduces the JSONL byte-for-byte.\n"
      "- Anticipated-but-empty families (`mapped-but-classified-"
      "explicit_correction`, `staged-but-predicate-lint-failed`) are "
      "retained in the taxonomy with realized count 0 so the funnel stays "
      "portable to systems where they are non-empty.\n")
    return "\n".join(parts)


def _mech_table(records: list[dict]) -> str:
    ranked = mechanism_ranking(records)
    n_drop = sum(1 for r in records if r["dropout_stage"])
    if not ranked:
        return "(no dropouts)\n"
    lines = [f"Dropouts: {n_drop}/{len(records)} records.\n",
             "| rank | mechanism | count | share of dropouts |",
             "|---|---|---|---|"]
    for i, (tag, c) in enumerate(ranked, 1):
        lines.append(f"| {i} | `{tag}` | {c} | {c / n_drop:.0%} |")
    return "\n".join(lines) + "\n"


def _taxonomy_section(records: list[dict]) -> str:
    c = collections.Counter(r["mechanism_tag"] for r in records
                            if r["mechanism_tag"])
    plan_patterns = "\n".join(f"      {p}"
                              for p in PLAN_FRAME_ATTRIBUTE_PATTERNS)
    unm = [r for r in records
           if r["mechanism_tag"] == "tuple-extracted-but-unmapped-key"]
    unm_in_cand = sum(1 for r in unm
                      if r["mechanism_detail"]["target_key_in_candidates"])
    unm_plan = sum(1 for r in unm
                   if r["mechanism_detail"]["plan_frame_attribute"])
    other = [r for r in records
             if r["mechanism_tag"] == "mapped-but-classified-other"]
    other_actions = collections.Counter(
        r["mechanism_detail"]["action"] for r in other)
    return (
        f"- **no-tuple-extracted** ({c.get('no-tuple-extracted', 0)}): the "
        "extractor emitted no tuple at the hedge turn that carries the "
        "hedged value or span-matches the hedge quote. Evidence: the turn's "
        "full extracted tuple list.\n"
        f"- **tuple-extracted-but-unmapped-key** ({c.get('tuple-extracted-but-unmapped-key', 0)}): "
        "the carrying tuple reached the key-resolution step and was mapped "
        "to `none` (retained in the unmapped log — FM-6). Of these, "
        f"{unm_in_cand}/{len(unm)} had the hedge's target key PRESENT in "
        f"the mapper's candidate list (the mapper declined — its recorded "
        f"rationales distinguish future/plan reframings from the current-"
        f"state key it was offered), and "
        f"{unm_plan}/{len(unm)} carry an explicitly plan-shaped raw "
        "attribute (the frozen v2.2 plan-frame regex list, verbatim under "
        "`plan-shaped-attribute-committed-additively` below). Evidence: the "
        "unmapped row incl. the mapper's verbatim rationale.\n"
        f"- **mapped-but-classified-completed_event** ({c.get('mapped-but-classified-completed_event', 0)}): "
        "the hedge tuple was mapped but the revision classifier read it as "
        "a completed event — it COMMITS as canonical current state (the "
        "contaminating failure mode). Evidence: the gate row (class, "
        "action, stored value).\n"
        f"- **mapped-but-classified-explicit_correction** ({c.get('mapped-but-classified-explicit_correction', 0)}): "
        "anticipated (commit with retro-invalidation); realized 0.\n"
        f"- **mapped-but-classified-other** ({c.get('mapped-but-classified-other', 0)}): "
        "classified `other` → the additive branch: additive_commit / "
        "additive_refresh reify the hedged value as current when the key "
        "was empty or already equal (a quieter contamination door); "
        "additive_noop silently drops the hedge when a different value is "
        f"stored. Realized actions: {dict(other_actions)} (storeless t3 "
        "entries have action `None`).\n"
        f"- **plan-shaped-attribute-committed-additively** ({c.get('plan-shaped-attribute-committed-additively', 0)}): "
        "the classified-`other` subset whose RAW extracted attribute "
        "matches the frozen v2.2 plan-frame regex list "
        "(`router_sweep.PLAN_FRAME_ATTRIBUTE_PATTERNS`, applied "
        "case-insensitively via `re.search`, token-bounded on snake_case "
        "attribute names), the five patterns verbatim:\n\n"
        f"{plan_patterns}\n\n"
        "  — the exact pathway the v2.2 "
        "plan_frame_fix router point targeted (+1 staging event on LoCoMo "
        "only, consistent with the counts here).\n"
        f"- **staged-but-predicate-lint-failed** ({c.get('staged-but-predicate-lint-failed', 0)}): "
        "staged but the resolution predicates still fail lint after the "
        "single feedback retry. In this implementation such items are "
        "RETAINED (nothing is dropped at this stage by design — the "
        "anticipated \"staged-but-predicate-lint-dropped\" family is "
        "therefore structurally empty here; realized residual lint "
        "failures across all frozen gated cells and t3 manifests: 0). The "
        "stage is probed anyway so the funnel ports to systems that do "
        "drop on lint failure.\n")


def _findings_section(records: list[dict], table: dict) -> str:
    def rate(pop, bench, fam):
        e = table[(pop, bench, fam)]
        return e["reached"]["staged_with_predicates"], e["n"]

    s_em_l, n_em_l = rate("planted_spec", "lme", "em")
    s_ez_l, n_ez_l = rate("planted_spec", "lme", "ez")
    s_em_c, n_em_c = rate("planted_spec", "locomo", "em")
    s_ez_c, n_ez_c = rate("planted_spec", "locomo", "ez")
    spec_em = [r for r in records if r["population"] == "planted_spec"
               and r["family"] == "em"]
    em_unm = sum(1 for r in spec_em
                 if r["mechanism_tag"] == "tuple-extracted-but-unmapped-key")
    return (
        "1. **The t3 14–26% gap is a routing gap, not an extraction gap**: "
        "on the reconciliation population every loss is at "
        "`routed_speculative` (misclassification as `other` or "
        "`completed_event`); extraction and staging machinery lose "
        "nothing.\n"
        "2. **For planted spec_* hedges the two families fail in different "
        f"places.** E-Z stages {s_ez_l}/{n_ez_l} (lme) and {s_ez_c}/"
        f"{n_ez_c} (locomo) — its extractor preserves the target attribute "
        "name, the mapper accepts it, the classifier reads the hedge as "
        f"speculative. E-M stages only {s_em_l}/{n_em_l} (lme) and "
        f"{s_em_c}/{n_em_c} (locomo): its extractor reifies hedges into "
        "NEW plan-shaped attribute names, and the key mapper then declines "
        f"to map them onto the current-state target keys ({em_unm}/"
        f"{len(spec_em)} E-M records drop as "
        "`tuple-extracted-but-unmapped-key`, most with the target key "
        "sitting in the candidate list). The hedge never reaches the "
        "store — which is also why the ungated E-M column showed little "
        "contamination (GATE-EM ≈ 0.04 vs GATE-EZ ≈ 0.97): the extraction "
        "family, not the gate, was absorbing planted hedges in E-M.\n"
        "3. **Natural hedges replicate the E-M pattern for both families**: "
        "the dominant natural loss is `tuple-extracted-but-unmapped-key` "
        "(hedges reframed to plan-shaped or novel attributes that the "
        "frozen candidate lists cannot host), followed by "
        "misclassification at the routing stage; outright extraction "
        "misses are rare.\n"
        "4. **The additive door exists but is secondary**: classified-"
        "`other` hedges additively commit/refresh the hedged value as "
        "current when the key is empty — a contamination pathway the "
        "primary metrics under-count because it needs no supersession.\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = run()
    print(json.dumps({"n_records": result["n_records"],
                      "t3_vs_sweep": result["reconciliation"]["t3_vs_sweep"],
                      "natural_agreement":
                          result["reconciliation"]["natural_agreement"],
                      "outputs": [result["funnel_path"],
                                  result["report_path"]]},
                     indent=2))
