"""Test 2 runner — write-through contamination (protocol_spec.md §3.1, v1.1).

Consumes the frozen injection manifest (../results/injections/manifest.jsonl),
the key schema, and the aux-key registry; replays direct write-through per core
substrate (store.replay_write_through) and scores:

- Test 2a speculation-contamination: numerator = injections whose probed store
  value is semantically equivalent to v_spec; denominator = all speculation
  injections, intent-to-treat (extraction misses count as non-contamination but
  stay in the denominator). Secondary: conditional-on-extraction.
- Test 2b correction-residue: numerator = injections whose probed value is NOT
  equivalent to v_new, split-reported stale-old (equals v_old) vs missing/other;
  denominator = all correction injections. Never pooled with 2a.

Probe-point accounting per B4: every injection is scored at its
manifest-designated probe point; deviant strata (C-GOLD boundary probes,
`gold_key_spec_not_last` provenance-timeline probes) are ADDITIONALLY reported
separately and their counts footnote the headline. FM-6 extraction hit-rates
per injection type; FM-7 realized-density strata. Every rate carries x and n,
with Wilson CI and cluster bootstrap by substrate (stats.py).

After scoring, the §3.2 turns-to-canonical analysis (analysis_ttc.py, R2-M-f)
runs over the same timelines and writes ../results/test2/ttc.json.

Run from pilot/: .venv/bin/python test2_writethrough.py
Idempotent: reruns hit the llm.py cache and reproduce identical outputs.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter, defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401  (R1 layout: puts src/ + experiments/ on sys.path)

import analysis_ttc
import data
import stats
import store

from paths import RESULTS_DIR
MANIFEST_PATH = RESULTS_DIR / "injections" / "manifest.jsonl"
OUT_DIR = RESULTS_DIR / "test2"

SPEC_TYPES = {"spec_gold", "spec_aux"}
CORR_TYPES = {"caux_correct", "cgold_correct"}
T3_TYPES = {"t3_staging", "t3_variant"}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def rate_block(successes: list[int], clusters: list[str]) -> dict:
    """One rate with its denominator, Wilson CI, and by-substrate cluster bootstrap."""
    x, n = sum(successes), len(successes)
    block: dict = {"x": x, "n": n, "rate": (x / n) if n else None}
    if n:
        lo, hi = stats.wilson_ci(x, n)
        block["wilson_ci"] = [lo, hi]
        block["cluster_bootstrap_by_substrate"] = stats.cluster_bootstrap_rate(successes, clusters)
    return block


def agg(outcomes: list[dict], flag: str) -> dict:
    return rate_block([int(o[flag]) for o in outcomes], [o["question_id"] for o in outcomes])


def extraction_hit(row: dict, timeline: store.StoreTimeline) -> bool:
    """FM-6/M5: did a tuple extracted from the injection turn map to the target key?"""
    target = f"{row['entity']}/{row['attribute']}"
    s, t = row["placement"]["session_idx"], row["placement"]["turn_idx"]
    return any(
        m["session_idx"] == s and m["turn_idx"] == t and m["mapped_key"] == target
        for m in timeline.mapping_log
    )


def density_stratum(row: dict, session_counts: Counter) -> str:
    """FM-7 / injector §7.4: realized injected-aside count of the session hosting
    the injection (all non-Test-3 manifest rows, establishes included)."""
    count = session_counts[(row["question_id"], row["placement"]["session_idx"])]
    return "<=1" if count <= 1 else ("2" if count == 2 else ">=3")


async def score_speculation(row: dict, timeline: store.StoreTimeline) -> dict:
    key = (row["entity"], row["attribute"])
    probe_point = row["probe_point"]
    turn = None
    if probe_point == "provenance_timeline":
        turn = (row["placement"]["session_idx"], row["placement"]["turn_idx"])
    looked_up = store.probe(timeline, key, probe_point, turn=turn)
    contaminated = looked_up is not None and await store.values_equivalent(
        row["attribute"], looked_up, row["value"], trace_tag=f"equiv2a-{row['injection_id']}"
    )
    return {
        "injection_id": row["injection_id"],
        "question_id": row["question_id"],
        "inj_type": row["inj_type"],
        "probe_point": probe_point,
        "key": "/".join(key),
        "looked_up_value": looked_up,
        "v_spec": row["value"],
        "contaminated": contaminated,
        "extraction_hit": extraction_hit(row, timeline),
    }


async def score_correction(row: dict, timeline: store.StoreTimeline) -> dict:
    key = (row["entity"], row["attribute"])
    probe_point = row["probe_point"]
    looked_up = store.probe(timeline, key, probe_point)
    v_new, v_old = row["value"], row["old_value"]
    residue = looked_up is None or not await store.values_equivalent(
        row["attribute"], looked_up, v_new, trace_tag=f"equiv2b-new-{row['injection_id']}"
    )
    stale_old = (
        residue
        and looked_up is not None
        and await store.values_equivalent(
            row["attribute"], looked_up, v_old, trace_tag=f"equiv2b-old-{row['injection_id']}"
        )
    )
    return {
        "injection_id": row["injection_id"],
        "question_id": row["question_id"],
        "inj_type": row["inj_type"],
        "probe_point": probe_point,
        "key": "/".join(key),
        "looked_up_value": looked_up,
        "v_new": v_new,
        "v_old": v_old,
        "residue": residue,
        "stale_old": stale_old,
        "missing_other": residue and not stale_old,
        "extraction_hit": extraction_hit(row, timeline),
    }


def by_stratum(outcomes: list[dict], flag: str, key_fn) -> dict:
    grouped = defaultdict(list)
    for o in outcomes:
        grouped[key_fn(o)].append(o)
    return {name: agg(rows, flag) for name, rows in sorted(grouped.items())}


def scored_question_ids(manifest: list[dict]) -> list[str]:
    """Substrates hosting scored (speculation/correction) injections."""
    return sorted({r["question_id"] for r in manifest
                   if r["inj_type"] in SPEC_TYPES | CORR_TYPES})


async def replay_all_timelines(
    manifest: list[dict], key_schema: dict, qids: list[str]
) -> dict[str, store.StoreTimeline]:
    """Write-through replay of the given substrates (perturbed with all their
    non-T3 manifest rows). Shared by run() and analysis_ttc.main()."""
    aux_rows = load_jsonl(store.INJECTED_KEYS_PATH)
    aux_keys_by_qid: dict[str, list[store.CanonicalKey]] = defaultdict(list)
    for row in aux_rows:
        qid = row.get("substrate_qid", row.get("question_id"))
        aux_keys_by_qid[qid].append(
            store.CanonicalKey(
                entity=row["entity"],
                attribute=row["attribute"],
                description=row.get("description", ""),
                probe_question=row["probe_question"],
            )
        )

    splice_rows_by_qid: dict[str, list[dict]] = defaultdict(list)
    for row in manifest:
        if row["inj_type"] not in T3_TYPES:  # Test 3 lives in its own clean copies (injector §9.1)
            splice_rows_by_qid[row["question_id"]].append(row)

    async def replay_one(qid: str) -> tuple[str, store.StoreTimeline]:
        substrate = data.get_substrate(qid)
        timeline = await store.replay_write_through(
            substrate, key_schema[qid], splice_rows_by_qid[qid], aux_keys=aux_keys_by_qid.get(qid, [])
        )
        return qid, timeline

    return dict(await asyncio.gather(*(replay_one(qid) for qid in qids)))


async def run() -> dict:
    key_schema = data.load_key_schema()
    manifest = load_jsonl(MANIFEST_PATH)

    spec_rows = [r for r in manifest if r["inj_type"] in SPEC_TYPES]
    corr_rows = [r for r in manifest if r["inj_type"] in CORR_TYPES]
    scored_qids = scored_question_ids(manifest)

    timelines = await replay_all_timelines(manifest, key_schema, scored_qids)

    spec_outcomes = list(
        await asyncio.gather(*(score_speculation(r, timelines[r["question_id"]]) for r in spec_rows))
    )
    corr_outcomes = list(
        await asyncio.gather(*(score_correction(r, timelines[r["question_id"]]) for r in corr_rows))
    )

    # FM-7 density strata over the sessions hosting each scored injection.
    session_counts = Counter(
        (r["question_id"], r["placement"]["session_idx"])
        for r in manifest
        if r["inj_type"] not in T3_TYPES
    )
    for o, r in zip(spec_outcomes + corr_outcomes, spec_rows + corr_rows):
        o["density_stratum"] = density_stratum(r, session_counts)

    spec_hits = [o for o in spec_outcomes if o["extraction_hit"]]
    corr_hits = [o for o in corr_outcomes if o["extraction_hit"]]
    spec_deviant = [o for o in spec_outcomes if o["probe_point"] == "provenance_timeline"]
    corr_deviant = [o for o in corr_outcomes if o["inj_type"] == "cgold_correct"]

    results = {
        "meta": {
            "seed": store.SEED,
            "manifest": str(MANIFEST_PATH),
            "n_substrates_replayed": len(timelines),
            "n_speculation_injections": len(spec_rows),
            "n_correction_injections": len(corr_rows),
            "probe_point_counts_2a": dict(Counter(o["probe_point"] for o in spec_outcomes)),
            "probe_point_counts_2b": dict(Counter(o["probe_point"] for o in corr_outcomes)),
            "headline_footnote": (
                "Headline rates score every injection at its manifest-designated probe point (B4). "
                f"Deviant strata, additionally reported separately: gold_key_spec_not_last "
                f"provenance-timeline speculation rows n={len(spec_deviant)}; C-GOLD "
                f"old-session-boundary correction rows n={len(corr_deviant)}."
            ),
        },
        "test2a_speculation_contamination": {
            "intent_to_treat": agg(spec_outcomes, "contaminated"),
            "conditional_on_extraction": agg(spec_hits, "contaminated"),
            "by_inj_type": by_stratum(spec_outcomes, "contaminated", lambda o: o["inj_type"]),
            "deviant_stratum_gold_key_spec_not_last": agg(spec_deviant, "contaminated"),
            "by_density_stratum": by_stratum(spec_outcomes, "contaminated", lambda o: o["density_stratum"]),
            "extraction_hit_rate": agg(spec_outcomes, "extraction_hit"),
        },
        "test2b_correction_residue": {
            "intent_to_treat": agg(corr_outcomes, "residue"),
            "stale_old": agg(corr_outcomes, "stale_old"),
            "missing_other": agg(corr_outcomes, "missing_other"),
            "conditional_on_extraction": agg(corr_hits, "residue"),
            "by_inj_type": by_stratum(corr_outcomes, "residue", lambda o: o["inj_type"]),
            "deviant_stratum_cgold": agg(corr_deviant, "residue"),
            "by_density_stratum": by_stratum(corr_outcomes, "residue", lambda o: o["density_stratum"]),
            "extraction_hit_rate": agg(corr_outcomes, "extraction_hit"),
        },
        "collision_audit": store.collision_audit(
            [m for t in timelines.values() for m in t.mapping_log]
        ),
        "unmapped_tuple_count": sum(len(t.unmapped) for t in timelines.values()),
        "per_injection": spec_outcomes + corr_outcomes,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    store.dump_jsonl(
        [u for t in timelines.values() for u in t.unmapped], store.UNMAPPED_LOG_PATH
    )
    store.dump_jsonl(
        [m for t in timelines.values() for m in t.mapping_log], store.KEY_MAPPING_LOG_PATH
    )

    # §3.2 turns-to-canonical over the same timelines (R2-M-f) -> test2/ttc.json
    ttc = await analysis_ttc.compute_ttc(timelines, key_schema, manifest)
    analysis_ttc.write_ttc(ttc)
    return results


def main() -> None:
    results = asyncio.run(run())
    print(json.dumps({k: v for k, v in results.items() if k != "per_injection"}, indent=2))
    print(f"\nwrote {OUT_DIR / 'results.json'} and {analysis_ttc.TTC_PATH}")


if __name__ == "__main__":
    main()
