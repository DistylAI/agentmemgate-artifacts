"""§8-rule-5 natural interaction consistency check — EXECUTED LATE (deviation D2).

The frozen protocol (research/protocol_spec_v2.md §8 rule 5) requires Δ̂_int
to be additionally computed on the NATURAL speculation stratum (pair-stage
rows + censored hedges anchored to schema keys, all four cells) and printed
next to H-INT; if its sign opposes H-INT and its cluster CI excludes the
injected point estimate, the headline claim downgrades to injected-scope.
This check was not executed with the 2026-07-04 run (RESULTS_V2.md §11, D2).
This module executes it late, on the COMPUTABLE subset, with no re-runs and
no fresh LLM calls except the equivalence judge on pairs the exact
normalize_value path cannot resolve (hard cap: 50 judge calls, frozen
prompts/equivalence_judge.txt via store.values_equivalent).

Computable subset (coverage stated, not hidden):
  (a) the 18 LoCoMo pair-stage rows (research/natural_pairs_v2.jsonl) — the
      LoCoMo pairs were staged and checked IN-STREAM inside the cell replays
      (protocol §4 / §7 row 7), so all four cells have full store timelines
      for their conversations;
  (b) the frozen censored staged hypotheses of key_schema_v2_locomo.json that
      are anchored to schema update keys: conv-41 D28:5 →
      conv-41/john/employment_status and conv-41 D17:11 →
      conv-41/john/dog_ownership (the schema's own notes cross-reference
      both). The other two frozen censored hedges (conv-26 D4:11 career,
      conv-47 D2:18 pet) have NO schema key and are excluded by the rule's
      own "anchored to schema keys" predicate.
NOT computable: the 17 LME pair-stage rows — the LME half of NAT-RES ran
STORELESS (natres.py, per §4/V2-F3), so no cell store timeline contains the
pair keys. Verified, not assumed: 8/17 pair source questions double as v2
injection substrates and DO have cell timelines, but their frozen candidate
lists never included the natural pair keys — 0/8 timelines contain the pair
key and 0/8 have any stage-anchor gate_log row for it.

Per computable stage row i and cell c ∈ {em_cw, em_cg, ez_cw, ez_cg}:
  Y_i^c = 1  iff the hedged value is canonical at the row's schema key at the
             designated read point — full_ingestion (the probe point of
             356/360 v2 speculation injections), read through the recorded
             last snapshot's per-store sequence numbers exactly as
             gated_store.probe_timeline does (ZepStore.as_of_seq semantics
             reconstructed from the serialized timeline), with the store
             resolved per arms.answer_probes' R1-B1 convention (the
             entity-routed speaker store).
  Hedged value set (primary, fully mechanical, policy-free): every value the
  family's shared extraction contributed to the schema key AT the stage
  anchor turn (gate_log of the family's ungated cell — the policy-free
  witness, exactly canonicalization_check's convention). Y = 1 iff the read
  value is present (non-sentinel) and equivalent to ANY of them.
  Sensitivity variant (recorded per row, no extra LLM): the subset of those
  values the family's gated replay classified `speculative` at the anchor
  (hedge-routed only; a strict subset, so it reuses the same judge results).

  Z_i = (Y_i^ez_cw − Y_i^ez_cg) − (Y_i^em_cw − Y_i^em_cg)   ∈ {−2..2}
  Δ̂_nat = mean(Z); CI by conversation-level cluster bootstrap
  (stats.cluster_bootstrap_mean, seed 20260702, n_boot 10000 — the §1.1
  machinery H-INT used).

Outputs:
  results/v2/natural_consistency_rule5_rows.jsonl   per-row audit trail
  results/v2/metrics.json                           key natural_consistency_check_rule5
                                                    (+ D4 pending-array fix)

CLI (from pilot/): .venv/bin/python natural_check_rule5.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import _bootstrap  # noqa: F401  (R1 layout: puts src/ + experiments/ on sys.path)

import llm
import locomo_data
import metrics_v2
import stats
import store
from locomo_render import value_is_sentinel

from paths import RESEARCH_DIR, RESULTS_DIR

V2_DIR = RESULTS_DIR / "v2"
PAIRS_PATH = RESEARCH_DIR / "natural_pairs_v2.jsonl"
LOCOMO_SCHEMA_PATH = RESEARCH_DIR / "key_schema_v2_locomo.json"
METRICS_PATH = V2_DIR / "metrics.json"
ROWS_PATH = V2_DIR / "natural_consistency_rule5_rows.jsonl"

SEED = 20260702
N_BOOT = 10000
JUDGE_CALL_CAP = 50

CELL_DIRS = {"em_cw": "mem0_ungated", "em_cg": "mem0_gated",
             "ez_cw": "zep_ungated", "ez_cg": "zep_gated"}
FAM_UNGATED = {"em": "mem0_ungated", "ez": "zep_ungated"}
FAM_GATED = {"em": "mem0_gated", "ez": "zep_gated"}

#: The two frozen censored staged hypotheses that anchor to schema update
#: keys (key_schema_v2_locomo.json: the update-key notes cross-reference the
#: censored dia_ids). The other two frozen hedges have no schema key.
CENSORED_ANCHORED = {
    ("conv-41", "D28:5"): ("john", "employment_status"),
    ("conv-41", "D17:11"): ("john", "dog_ownership"),
}
CENSORED_UNANCHORED = {
    ("conv-26", "D4:11"): "career -> counseling/mental health: no caroline "
                          "career key in update_keys",
    ("conv-47", "D2:18"): "pet_ownership none -> dog/pet: no conv-47 pet key "
                          "in update_keys",
}

N_LME_PAIR_STAGE = 17   # storeless (natres.py §4/V2-F3) — not computable
N_LOCOMO_PAIR_STAGE = 18
N_CENSORED_FROZEN = 4


# ---------------------------------------------------------------------------
# Row construction (frozen artifacts only)
# ---------------------------------------------------------------------------

def build_rows() -> list[dict]:
    pairs = [json.loads(line) for line in PAIRS_PATH.read_text().splitlines()
             if line.strip()]
    loco_pairs = [p for p in pairs if p["corpus"] == "locomo"]
    assert len(loco_pairs) == N_LOCOMO_PAIR_STAGE
    assert sum(p["corpus"] == "lme" for p in pairs) == N_LME_PAIR_STAGE
    rows = []
    for p in loco_pairs:
        conv = locomo_data.get_conversation(p["source_question_id"])
        # positional anchor via turn_by_dia_id (the pairs artifact records
        # D-numbers; timelines use positional session indices) — the exact
        # convention of run_v2.locomo_instream_rows.
        si, ti, _ = locomo_data.turn_by_dia_id(conv, p["stage"]["dia_id"])
        rows.append({
            "row_id": p["pair_id"], "row_kind": "pair_stage",
            "sample_id": p["source_question_id"],
            "entity": p["key"]["entity"], "attribute": p["key"]["attribute"],
            "anchor": [si, ti], "stage_dia_id": p["stage"]["dia_id"],
        })
    censored = locomo_data.censored_staged_hypotheses()
    assert len(censored) == N_CENSORED_FROZEN
    for c in censored:
        tag = (c["sample_id"], c["dia_id"])
        if tag not in CENSORED_ANCHORED:
            assert tag in CENSORED_UNANCHORED, f"unclassified censored hedge {tag}"
            continue
        conv = locomo_data.get_conversation(c["sample_id"])
        si, ti, _ = locomo_data.turn_by_dia_id(conv, c["dia_id"])
        entity, attribute = CENSORED_ANCHORED[tag]
        rows.append({
            "row_id": f"CENS-{c['sample_id']}-{c['dia_id']}",
            "row_kind": "censored_hedge", "sample_id": c["sample_id"],
            "entity": entity, "attribute": attribute,
            "anchor": [si, ti], "stage_dia_id": c["dia_id"],
        })
    return rows


# ---------------------------------------------------------------------------
# Timeline reads (serialized-JSON reconstruction of probe_timeline semantics)
# ---------------------------------------------------------------------------

_TL_CACHE: dict[tuple, dict] = {}


def load_timeline(cell_dir: str, sample_id: str, base_dir: Path = V2_DIR) -> dict:
    """Load one serialized cell timeline. ``base_dir`` defaults to the frozen
    v2 results tree; the v2.2 C1 expansion (expand_natural.py) passes its own
    results/v2.2 tree — the read semantics are identical."""
    key = (str(base_dir), cell_dir, sample_id)
    if key not in _TL_CACHE:
        _TL_CACHE[key] = json.loads(
            (base_dir / cell_dir / "timelines" / f"{sample_id}.json").read_text())
    return _TL_CACHE[key]


def as_of_seq_json(history_entries: list[dict], seq: int) -> str | None:
    """ZepStore.as_of_seq over the serialized history (same semantics)."""
    best = None
    for e in history_entries:
        if e["created_seq"] > seq:
            continue
        if e["expired_seq"] is not None and e["expired_seq"] <= seq:
            continue
        if best is None or e["created_seq"] > best["created_seq"]:
            best = e
    return best["value"] if best else None


def full_ingestion_value(timeline: dict, store_id: str, key_name: str) -> str | None:
    """probe_timeline(..., 'full_ingestion') on the serialized timeline: the
    LAST snapshot's recorded per-store seq, read via as_of_seq. Store routing
    per arms.answer_probes R1-B1: the entity-named speaker store."""
    snap = timeline["snapshots"][-1]
    seq = snap["seq"].get(store_id, 0)
    hist = timeline["stores"].get(store_id, {}).get("history", {}).get(key_name, [])
    return as_of_seq_json(hist, seq)


def anchor_values(timeline: dict, row: dict, *, speculative_only: bool) -> list[str]:
    """Values the extraction contributed to the row's schema key AT the stage
    anchor turn (gate_log; store-routed to the row's entity — or to the row's
    explicit ``store_id`` when set: LME single-store replays route every key
    to gated_store.DEFAULT_STORE 'main', the v2.2 C1 expansion case; rows
    without ``store_id`` behave exactly as before)."""
    key_name = f"{row['entity']}/{row['attribute']}"
    store_id = row.get("store_id", row["entity"])
    si, ti = row["anchor"]
    out = []
    for g in timeline["gate_log"]:
        if (g["session_idx"], g["turn_idx"]) != (si, ti):
            continue
        if g["key"] != key_name or g["store_id"] != store_id:
            continue
        if speculative_only and g.get("class") != "speculative":
            continue
        if g["value"] not in out:
            out.append(g["value"])
    return out


# ---------------------------------------------------------------------------
# Equivalence (exact path first; judge only on unresolved pairs, capped)
# ---------------------------------------------------------------------------

class EquivalenceScorer:
    def __init__(self, cap: int = JUDGE_CALL_CAP):
        self.cap = cap
        self.judge_calls = 0
        self.fresh_judge_calls = 0
        self._cache: dict[tuple, bool] = {}

    async def equivalent(self, attribute: str, probed, hedged) -> tuple[bool, str]:
        """Returns (equivalent, method). Sentinel/None are non-equivalent to
        any concrete value (protocol §1.2 item 10 — run_v2._equivalent)."""
        if probed is None or hedged is None:
            return False, "absent"
        if value_is_sentinel(probed) or value_is_sentinel(hedged):
            return False, "sentinel"
        a, b = str(probed), str(hedged)
        if store.normalize_value(a) == store.normalize_value(b):
            return True, "exact_normalized"
        lo, hi = sorted([a, b])
        ck = (attribute, lo, hi)
        if ck not in self._cache:
            if self.judge_calls >= self.cap:
                raise RuntimeError(
                    f"judge-call cap {self.cap} exceeded — refusing to continue")
            first, second = store._judge_pair_order(attribute, a, b)
            user_prompt = f"ATTRIBUTE: {attribute}\nVALUE A: {first}\nVALUE B: {second}"
            # judge accounting keys on the PINNED judge model (JUDGE_MODEL ==
            # the historical default, so pre-seam cache entries still count as
            # cached); store.values_equivalent pins the actual call the same way
            cache_key = llm._cache_key(llm.JUDGE_MODEL, llm.REASONING_EFFORT,
                                       store.EQUIVALENCE_JUDGE_PROMPT, user_prompt)
            was_cached = (llm.CACHE_DIR / f"{cache_key}.json").exists()
            self.judge_calls += 1
            self.fresh_judge_calls += int(not was_cached)
            self._cache[ck] = await store.values_equivalent(
                attribute, a, b, trace_tag=f"rule5-eq-{attribute}")
        return self._cache[ck], "judge"


# ---------------------------------------------------------------------------
# Per-row scoring (the estimand body; reused by the v2.2 C1 expansion via
# expand_natural.py — import, don't copy)
# ---------------------------------------------------------------------------

async def score_row(row: dict, scorer: EquivalenceScorer,
                    *, base_dir: Path = V2_DIR) -> dict:
    """Score ONE natural stage row across the four cells (module docstring):
    hedged value set from the family's UNGATED gate_log at the anchor,
    Y per cell at full_ingestion, Z the paired interaction contrast. The
    default ``base_dir`` reads the frozen v2 cell timelines; the v2.2 C1
    expansion passes its own tree per row. Semantics are byte-identical to
    the original inline loop of run()."""
    key_name = f"{row['entity']}/{row['attribute']}"
    store_id = row.get("store_id", row["entity"])
    hedged_all = {fam: anchor_values(
        load_timeline(FAM_UNGATED[fam], row["sample_id"], base_dir),
        row, speculative_only=False)
        for fam in ("em", "ez")}
    hedged_spec = {fam: anchor_values(
        load_timeline(FAM_GATED[fam], row["sample_id"], base_dir),
        row, speculative_only=True)
        for fam in ("em", "ez")}
    y_primary, y_spec, finals, methods = {}, {}, {}, {}
    for mcell, cell_dir in CELL_DIRS.items():
        fam = mcell[:2]
        timeline = load_timeline(cell_dir, row["sample_id"], base_dir)
        final = full_ingestion_value(timeline, store_id, key_name)
        finals[mcell] = final
        yp = ys = 0
        meth = []
        for h in hedged_all[fam]:
            eq, method = await scorer.equivalent(row["attribute"], final, h)
            meth.append({"hedged": h, "equivalent": eq, "method": method,
                         "speculative_classified": h in hedged_spec[fam]})
            if eq:
                yp = 1
                if h in hedged_spec[fam]:
                    ys = 1
        y_primary[mcell] = yp
        y_spec[mcell] = ys
        methods[mcell] = meth
    z = ((y_primary["ez_cw"] - y_primary["ez_cg"])
         - (y_primary["em_cw"] - y_primary["em_cg"]))
    z_s = ((y_spec["ez_cw"] - y_spec["ez_cg"])
           - (y_spec["em_cw"] - y_spec["em_cg"]))
    return {
        **row, "key": key_name, "cluster": row["sample_id"],
        "hedged_values_all": hedged_all, "hedged_values_speculative": hedged_spec,
        "final_values_full_ingestion": finals,
        "y_primary": y_primary, "y_speculative_only": y_spec,
        "z_primary": z, "z_speculative_only": z_s,
        "equivalence_detail": methods,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run() -> dict:
    rows = build_rows()
    scorer = EquivalenceScorer()
    out_rows = []
    for row in rows:
        out_rows.append(await score_row(row, scorer))

    clusters = [r["cluster"] for r in out_rows]
    boot = stats.cluster_bootstrap_mean([r["z_primary"] for r in out_rows], clusters,
                                        n_boot=N_BOOT, seed=SEED, bounds=(-2, 2))
    boot_spec = stats.cluster_bootstrap_mean([r["z_speculative_only"] for r in out_rows],
                                             clusters, n_boot=N_BOOT, seed=SEED,
                                             bounds=(-2, 2))

    metrics = json.loads(METRICS_PATH.read_text())
    injected = metrics["benchmarks"]["lme"]["H-INT"]
    natural = {"point": boot["point"], "ci": [boot["lo"], boot["hi"]]}
    check = metrics_v2.natural_consistency_check(
        {"point": injected["point"], "ci": injected["ci"]}, natural)

    block = {
        "id": "NAT-CONSISTENCY-RULE5",
        "rule": ("§8 rule 5: Δ̂_int on the natural speculation stratum printed "
                 "next to H-INT; sign opposing H-INT AND cluster CI excluding "
                 "the injected point ⇒ headline downgraded to injected-scope "
                 "(FM-1 discipline)"),
        "delta_nat": boot["point"],
        "ci": [boot["lo"], boot["hi"]],
        "n": boot["n"],
        "n_clusters": boot["n_clusters"],
        "ci_kind": "cluster_bootstrap_by_conversation",
        "seed": SEED,
        "n_boot": N_BOOT,
        "injected_reference": {"id": "H-INT", "point": injected["point"],
                               "ci": injected["ci"]},
        "sign_opposes_hint": check.get("sign_opposes", False),
        "natural_ci_excludes_injected_point":
            check.get("natural_ci_excludes_injected_point"),
        "downgrade_triggered": check.get("downgrade_to_injected_scope", False),
        "z_distribution": {str(v): sum(r["z_primary"] == v for r in out_rows)
                           for v in sorted({r["z_primary"] for r in out_rows})},
        "sensitivity_speculative_classified_only": {
            "delta_nat": boot_spec["point"],
            "ci": [boot_spec["lo"], boot_spec["hi"]],
            "note": "hedged-value set restricted to stage-anchor tuples the "
                    "family's gated replay classified `speculative` (subset; "
                    "same judged pairs)"},
        "coverage": {
            "computable": len(out_rows),
            "total_natural_stage_rows": (N_LME_PAIR_STAGE + N_LOCOMO_PAIR_STAGE
                                         + N_CENSORED_FROZEN),
            "excluded_lme_storeless": N_LME_PAIR_STAGE,
            "excluded_censored_unanchored": len(CENSORED_UNANCHORED),
            "detail": ("computable = 18 LoCoMo pair-stage rows (in-stream in "
                       "all four cell replays) + 2 frozen censored hedges "
                       "anchored to schema update keys (conv-41 D28:5 → "
                       "john/employment_status; conv-41 D17:11 → "
                       "john/dog_ownership). Excluded: the 17 LME pair-stage "
                       "rows (LME NAT-RES ran STORELESS per §4/V2-F3 — "
                       "verified: 8/17 pair source questions have cell "
                       "timelines as injection substrates, but 0/8 candidate "
                       "lists included the natural pair keys, 0/8 timelines "
                       "contain them) and the 2 frozen censored hedges with "
                       "no schema key (conv-26 D4:11, conv-47 D2:18 — "
                       "excluded by the rule's own 'anchored to schema keys' "
                       "predicate)."),
        },
        "judge_calls": {"cap": JUDGE_CALL_CAP, "used": scorer.judge_calls,
                        "fresh": scorer.fresh_judge_calls,
                        "prompt": "prompts/equivalence_judge.txt (frozen)"},
        "note": ("EXECUTED LATE (2026-07-04, deviation D2 → executed-late "
                 "status): computed post-hoc from the frozen run artifacts "
                 "(cell timelines + natural_pairs_v2.jsonl + "
                 "key_schema_v2_locomo.json), no replays. Read point = "
                 "full_ingestion (the probe point of 356/360 v2 speculation "
                 "injections), read via the recorded last-snapshot store "
                 "sequence numbers (gated_store.probe_timeline semantics), "
                 "store routed per arms.answer_probes R1-B1. Hedged value "
                 "set = the family's stage-anchor extraction contribution to "
                 "the schema key (policy-free ungated gate_log witness); "
                 "Y = probed value non-sentinel AND equivalent "
                 "(store.normalize_value exact path first, frozen blind "
                 "equivalence judge on unresolved pairs). Per-row audit "
                 "trail: results/v2/natural_consistency_rule5_rows.jsonl."),
        "rows_artifact": "results/v2/natural_consistency_rule5_rows.jsonl",
    }

    metrics["natural_consistency_check_rule5"] = block
    # D4 fix: SYN-RES / TTC / dAUC were listed pending although their results
    # shipped in the same commit under their own artifact files.
    still_pending = [p for p in metrics.get("pending", [])
                     if p["endpoint"] not in ("SYN-RES", "TTC", "dAUC")]
    metrics["pending"] = still_pending
    metrics["pending_resolved_D4_fix"] = {
        "date": "2026-07-04",
        "note": ("stale-bookkeeping fix (RESULTS_V2.md §11 D4): these "
                 "endpoints were never actually pending at commit time — "
                 "their results landed in the same commit under their own "
                 "artifact files, listed here"),
        "endpoints": {
            "SYN-RES": ["results/v2/test3_em/results.json",
                        "results/v2/test3_ez/results.json"],
            "TTC": ["results/v2/ttc/ttc.json"],
            "dAUC": ["results/v2/dauc/results.json"],
        },
    }

    ROWS_PATH.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                 for r in out_rows))
    METRICS_PATH.write_text(json.dumps(metrics, indent=1, ensure_ascii=False))
    return block


def main() -> None:
    block = asyncio.run(run())
    print(json.dumps({k: block[k] for k in
                      ("delta_nat", "ci", "n", "n_clusters", "sign_opposes_hint",
                       "natural_ci_excludes_injected_point", "downgrade_triggered",
                       "z_distribution", "judge_calls")},
                     ensure_ascii=False, indent=1))
    print(f"-> {METRICS_PATH}")
    print(f"-> {ROWS_PATH}")


if __name__ == "__main__":
    main()
