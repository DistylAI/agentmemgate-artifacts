"""arms.py — the v2 2x2 cell runner (owner: W2).

A cell is (extraction_family, commit_policy) with family in {mem0, zep} and
policy in {ungated, gated} (protocol_spec_v2 §1.1; metrics_v2 cell aliases in
`METRICS_CELL`: mem0 = E-M, zep = E-Z; ungated = C-W, gated = C-G).

Per cell, per substrate job this module runs the matching gated_store replay
and emits to ../results/v2/<cell>/:

  timelines/<question_id>.json   full store timeline (both ZepStore timelines,
                                 gate/staged/checker logs, per-turn snapshots,
                                 diagnostics — gated_store.timeline_to_json)
  probe_answers.jsonl            deterministic lookups at the manifest-designated
                                 probe points (B4): raw probed values only —
                                 equivalence judging/scoring is downstream (W5)
  extraction_calls.jsonl         per-substrate llm.py cache keys of the
                                 extraction calls (the byte-identity witness)
  summary.json                   aggregated diagnostics + the identity-check
                                 verdicts

Fairness hard assert (protocol_spec_v2 §1.2 item 1-2): the extraction pass is
shared per family via the llm.py cache — the gated and ungated cells of a
family must issue BYTE-IDENTICAL extraction calls. Both policies route through
`gated_store.extract_stream_tuples` (one pathway), and after running both
cells of a family `assert_extraction_identity` hard-compares the recorded
llm.py cache-key sequences per substrate (a cache key hashes (model, effort,
system_prompt, user_prompt), so equal keys == byte-identical calls).

The commit axis therefore adds no extraction cost (§7): the second policy's
extraction calls are cache hits by construction.

CLI (from pilot/, .venv/bin/python):
  python arms.py run [--families mem0,zep] [--policies ungated,gated]
                     [--qids q1,q2,...] [--manifest PATH] [--out DIR]
      Substrate jobs are built from the v1 key schema (data.load_key_schema)
      and the injection manifest's non-T3 rows — the pilot-compatible wiring;
      the v2 gold artifacts plug in by constructing CellJob rows directly
      (run_cells is loader-agnostic).
  python arms.py smoke
      Tiny end-to-end pass over every gate stage on ONE synthetic tuple
      (<= 5 LLM calls, hard-capped by construction: extract, key-map,
      classify, predicate-generation, checker).
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

import gated_store
import store
from zepstore import UNKNOWN_SENTINEL, ZepStore

from paths import RESULTS_DIR
V2_DIR = RESULTS_DIR / "v2"
MANIFEST_PATH = RESULTS_DIR / "injections" / "manifest.jsonl"

FAMILIES = ("mem0", "zep")
POLICIES = ("ungated", "gated")

#: metrics_v2.CELLS aliases (mem0 = E-M plan-aware, zep = E-Z canonical-writing;
#: ungated = C-W immediate supersession, gated = C-G evidence gate).
METRICS_CELL = {
    ("mem0", "ungated"): "em_cw",
    ("mem0", "gated"): "em_cg",
    ("zep", "ungated"): "ez_cw",
    ("zep", "gated"): "ez_cg",
}

# Manifest rows that carry probe points to answer (test2 conventions).
SPEC_TYPES = {"spec_gold", "spec_aux"}
CORR_TYPES = {"caux_correct", "cgold_correct"}
PROBED_TYPES = SPEC_TYPES | CORR_TYPES
T3_TYPES = {"t3_staging", "t3_variant"}


def cell_name(family: str, policy: str) -> str:
    assert family in FAMILIES, f"unknown family {family!r}"
    assert policy in POLICIES, f"unknown policy {policy!r}"
    return f"{family}_{policy}"


@dataclass
class CellJob:
    """One substrate's inputs, loader-agnostic (v1 schema wiring or v2 gold)."""

    question_id: str
    substrate: object  # data.Substrate-shaped (sessions/dates/session_order)
    key_schema_entry: dict | None  # None: no gold KU key (v2 aux substrates)
    manifest_rows: list = field(default_factory=list)   # spliced into the stream
    probe_rows: list | None = None  # rows to probe; default: PROBED_TYPES of manifest_rows
    aux_keys: list | None = None    # default: store.load_aux_keys(question_id)
    #: optional replay override: async (job, family, policy) -> GatedTimeline.
    #: The v2 LoCoMo dyadic single-pass driver (run_v2) plugs in here — the
    #: cell runner (artifacts, byte-identity assert, summaries) stays shared.
    replay: object | None = None


# ---------------------------------------------------------------------------
# Byte-identity of extraction across commit policies (per family)
# ---------------------------------------------------------------------------


def assert_extraction_identity(family: str,
                               keys_by_policy: dict[str, dict[str, list[str]]]) -> dict:
    """Hard-assert that every policy of `family` issued the identical extraction
    call sequence per substrate. Input: {policy: {question_id: [cache_key, ...]}}.
    Returns the audit verdict dict (also written into summary.json)."""
    policies = sorted(keys_by_policy)
    verdict = {"family": family, "policies": policies, "identical": True,
               "n_substrates": 0, "mismatches": []}
    if len(policies) < 2:
        verdict["note"] = "single policy run — identity check vacuous"
        return verdict
    ref_policy = policies[0]
    ref = keys_by_policy[ref_policy]
    for policy in policies[1:]:
        other = keys_by_policy[policy]
        assert set(ref) == set(other), (
            f"extraction identity ({family}): policies {ref_policy}/{policy} ran "
            f"different substrate sets {sorted(set(ref) ^ set(other))}")
        for qid in sorted(ref):
            verdict["n_substrates"] += 1
            if ref[qid] != other[qid]:
                verdict["identical"] = False
                verdict["mismatches"].append(
                    {"question_id": qid, "policy_a": ref_policy, "policy_b": policy})
    assert verdict["identical"], (
        f"FAIRNESS VIOLATION (protocol_spec_v2 §1.2): gated and ungated cells of "
        f"family {family!r} issued non-identical extraction calls: "
        f"{verdict['mismatches']}")
    return verdict


# ---------------------------------------------------------------------------
# Probe answering (deterministic lookups; scoring/judging is downstream)
# ---------------------------------------------------------------------------


def answer_probes(timeline: gated_store.GatedTimeline, probe_rows: list[dict]) -> list[dict]:
    """Raw probed value per manifest-designated probe row (B4). The `(unknown)`
    sentinel counts as ABSENT for `present` (protocol_spec_v2 §1.2 item 10);
    equivalence judging happens downstream, never here."""
    answers = []
    for row in probe_rows:
        key = (row["entity"], row["attribute"])
        turn = None
        if row["probe_point"] == "provenance_timeline":
            turn = (row["placement"]["session_idx"], row["placement"]["turn_idx"])
        # R1-B1 fix: resolve the store the way the replay routed writes —
        # per-entity stores (LoCoMo speaker stores) when one exists, else the
        # default store (LME). Probing only DEFAULT_STORE silently returned
        # absent for every speaker-store key.
        store_id = row["entity"] if row["entity"] in timeline.stores else gated_store.DEFAULT_STORE
        value = gated_store.probe_timeline(timeline, key, row["probe_point"], turn=turn,
                                           store_id=store_id)
        zstore = timeline.stores.get(store_id)
        answers.append(
            {
                "injection_id": row["injection_id"],
                "question_id": row["question_id"],
                "benchmark": row.get("benchmark"),
                "inj_type": row["inj_type"],
                "construction_label": row.get("construction_label"),
                "key": "/".join(key),
                "probe_point": row["probe_point"],
                "value": value,
                "present": value is not None and value != UNKNOWN_SENTINEL,
                "v_target": row.get("value"),
                "old_value": row.get("old_value"),
                # FM-16: the subject a third-party aside asserts about (the
                # manifest `entity` stays the focal store-routing key); carried
                # through so downstream scoring can check the subject's store.
                "expected_subject": row.get("expected_subject"),
                "history_depth": zstore.history_depth(key) if zstore is not None else 0,
                "family": timeline.family,
                "policy": timeline.policy,
            }
        )
    return answers


# ---------------------------------------------------------------------------
# Cell runner
# ---------------------------------------------------------------------------


async def run_cell_job(job: CellJob, family: str, policy: str) -> gated_store.GatedTimeline:
    if job.replay is not None:  # v2 LoCoMo dyadic driver (run_v2) or test seam
        return await job.replay(job, family, policy)
    replay = gated_store.replay_gated if policy == "gated" else gated_store.replay_ungated
    return await replay(job.substrate, job.key_schema_entry, job.manifest_rows,
                        family=family, aux_keys=job.aux_keys)


def _dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


async def run_cells(
    jobs: list[CellJob],
    *,
    families: tuple = FAMILIES,
    policies: tuple = POLICIES,
    out_dir: Path = V2_DIR,
) -> dict:
    """Run the requested cells over all jobs and emit per-cell artifacts.

    Per family: BOTH policies replay the same streams; extraction is shared via
    the llm.py cache (the gated pass's extraction calls are cache hits of the
    ungated pass's, or vice versa) and the byte-identity is hard-asserted from
    the recorded cache keys after both policies complete."""
    summary: dict = {"cells": {}, "extraction_identity": []}
    for family in families:
        keys_by_policy: dict[str, dict[str, list[str]]] = {}
        for policy in policies:
            cell = cell_name(family, policy)
            cell_dir = out_dir / cell
            (cell_dir / "timelines").mkdir(parents=True, exist_ok=True)
            probe_answers: list[dict] = []
            extraction_rows: list[dict] = []
            diag_total: dict = {}
            keys_by_policy[policy] = {}
            for job in jobs:
                timeline = await run_cell_job(job, family, policy)
                keys_by_policy[policy][job.question_id] = list(
                    timeline.extraction_call_keys)
                (cell_dir / "timelines" / f"{job.question_id}.json").write_text(
                    json.dumps(gated_store.timeline_to_json(timeline),
                               ensure_ascii=False, indent=1))
                probe_rows = (job.probe_rows if job.probe_rows is not None
                              else [r for r in job.manifest_rows
                                    if r["inj_type"] in PROBED_TYPES])
                probe_answers.extend(answer_probes(timeline, probe_rows))
                extraction_rows.append(
                    {"question_id": job.question_id, "family": family, "policy": policy,
                     "n_calls": len(timeline.extraction_call_keys),
                     "cache_keys": timeline.extraction_call_keys})
                for k, v in timeline.diagnostics.items():
                    if isinstance(v, dict):
                        bucket = diag_total.setdefault(k, {})
                        for kk, vv in v.items():
                            bucket[kk] = bucket.get(kk, 0) + vv
                    else:
                        diag_total[k] = diag_total.get(k, 0) + v
            _dump_jsonl(cell_dir / "probe_answers.jsonl", probe_answers)
            _dump_jsonl(cell_dir / "extraction_calls.jsonl", extraction_rows)
            summary["cells"][cell] = {
                "family": family,
                "policy": policy,
                "metrics_cell": METRICS_CELL[(family, policy)],
                "n_substrates": len(jobs),
                "n_probe_answers": len(probe_answers),
                "diagnostics": diag_total,
            }
        verdict = assert_extraction_identity(family, keys_by_policy)
        summary["extraction_identity"].append(verdict)

    out_dir.mkdir(parents=True, exist_ok=True)
    for cell_info in summary["cells"]:
        (out_dir / cell_info / "summary.json").write_text(
            json.dumps({**summary["cells"][cell_info],
                        "extraction_identity": summary["extraction_identity"]},
                       ensure_ascii=False, indent=1))
    return summary


# ---------------------------------------------------------------------------
# v1-schema job loader (pilot-compatible wiring; v2 gold constructs CellJob
# rows directly)
# ---------------------------------------------------------------------------


def load_jobs(qids: list[str] | None, manifest_path: Path = MANIFEST_PATH) -> list[CellJob]:
    import data

    schema = data.load_key_schema()
    manifest = []
    if manifest_path.exists():
        manifest = [json.loads(line) for line in manifest_path.read_text().splitlines()
                    if line.strip()]
    if qids is None:
        qids = sorted(schema)
    jobs = []
    for qid in qids:
        rows = [r for r in manifest
                if r["question_id"] == qid and r["inj_type"] not in T3_TYPES]
        jobs.append(CellJob(question_id=qid, substrate=data.get_substrate(qid),
                            key_schema_entry=schema[qid], manifest_rows=rows))
    return jobs


# ---------------------------------------------------------------------------
# Smoke (<= 5 LLM calls, hard-capped by construction)
# ---------------------------------------------------------------------------


async def smoke() -> None:
    """Exercise every LLM stage of the gate ONCE on one synthetic tuple:
    extract -> key-map -> classify -> generate predicates -> checker; then
    apply the promote fire to a ZepStore and print the lookup. Exactly one
    call per stage (the generator lint result is printed, the retry is skipped
    in smoke mode), so the call count is <= 5 by construction."""
    import classify
    import extract
    import llm
    import run_predicates as test3

    family = "mem0"
    system_prompt = gated_store.family_system_prompt(family)
    turn_text = ("By the way, I'm seriously considering switching my cell phone "
                 "carrier to Meteor Mobile once my current contract wraps up — "
                 "still comparing coverage maps, nothing decided yet.")
    date = "2023/05/14 (Sun) 09:02"

    # 1) extraction (family-selected prompt, B2 rendering)
    tuples = await gated_store.extract_turn_family(
        [], turn_text, "arms_smoke_extract", system_prompt=system_prompt,
        current_date=date)
    print(f"=== extraction ({family}) === {len(tuples)} tuple(s)")
    assert tuples, "smoke: extractor returned no tuples"
    tup = tuples[0]  # smoke processes ONE tuple to hard-cap the call count
    print(json.dumps({"entity": tup.entity, "attribute": tup.attribute,
                      "value": tup.value, "confidence": tup.confidence},
                     ensure_ascii=False))

    # 2) key canonicalization (existing key_mapper path)
    candidate = store.CanonicalKey(
        entity="user", attribute="cell_phone_carrier",
        description="canonical single-valued attribute (class: categorical)",
        probe_question="Which cell phone carrier does the user currently use?")
    cand, raw_key, _rationale = await store._map_tuple_key(
        tup, [candidate], trace_tag="arms_smoke_keymap")
    print(f"=== key mapper === mapped_key={raw_key}")

    # 3) classifier (frozen classifier_short via classify.classify_item)
    key = cand.key if cand is not None else candidate.key
    entity, attribute = key
    item = {"item_id": "arms-smoke-0", "entity": entity, "attribute": attribute,
            "asserted_value": tup.value, "stored_value": None,
            "turn_text": turn_text, "preceding_turns": []}
    cls_out = await classify.classify_item(item)
    print(f"=== classifier === argmax={cls_out['argmax']} s_ev={cls_out['s_ev']:.3f}")

    # 4) predicate generation (storeless B2-style prefix input)
    prefix = [{"date": date, "role": "user", "content": turn_text}]
    gen_out = await llm.call_llm(
        gated_store.GENERATOR_PROMPT_PATH.read_text(),
        test3.build_generator_user_prompt(prefix, entity, attribute, tup.value),
        trace_tag="arms_smoke_gen")
    failures = test3.lint_predicates(str(gen_out["promote_when"]),
                                     str(gen_out["invalidate_when"]), tup.value,
                                     test3.attribute_aliases(attribute))
    print(f"=== generator === lint failures: {failures or 'none'} "
          "(smoke skips the retry)")

    # 5) checker on a synthetic resolution turn + promote commit to a ZepStore
    staged = {"staged_id": "stg-arms-smoke-0", "entity": entity,
              "attribute": attribute, "hypothesized_value": tup.value,
              "promote_when": str(gen_out["promote_when"]),
              "invalidate_when": str(gen_out["invalidate_when"])}
    resolution_text = ("Quick update on the phone thing — I pulled the trigger and "
                       "moved my number over to Meteor Mobile on Tuesday; coverage "
                       "at the house has been great so far.")
    resolution_date = "2023/08/15 (Tue) 18:40"
    checker_out = await llm.call_llm(
        gated_store.CHECKER_PROMPT_PATH.read_text(),
        test3.build_checker_user_prompt(resolution_text, resolution_date, [staged]),
        trace_tag="arms_smoke_check")
    fires, malformed, _extra = test3.parse_checker_fires(
        checker_out, {staged["staged_id"]}, resolution_text)
    print(f"=== checker === fires={json.dumps(fires, ensure_ascii=False)} "
          f"malformed={malformed}")
    zstore = ZepStore()
    for fire in fires:
        if not fire["invalid"] and fire["action"] == "promote":
            zstore.write(key, staged["hypothesized_value"],
                         valid_from=resolution_date, correction=False)
    print(f"=== store after resolution === lookup({'/'.join(key)}) = "
          f"{zstore.lookup(key)!r}")
    print("smoke complete: 5 LLM calls (one per gate stage)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="v2 2x2 cell runner (W2)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    runp = sub.add_parser("run", help="run cells over substrates")
    runp.add_argument("--families", default=",".join(FAMILIES))
    runp.add_argument("--policies", default=",".join(POLICIES))
    runp.add_argument("--qids", default=None,
                      help="comma-separated question ids (default: all v1 schema)")
    runp.add_argument("--manifest", default=str(MANIFEST_PATH))
    runp.add_argument("--out", default=str(V2_DIR))
    sub.add_parser("smoke", help="tiny end-to-end smoke (<= 5 LLM calls)")
    args = parser.parse_args()

    if args.cmd == "smoke":
        asyncio.run(smoke())
        return

    families = tuple(f.strip() for f in args.families.split(",") if f.strip())
    policies = tuple(p.strip() for p in args.policies.split(",") if p.strip())
    qids = ([q.strip() for q in args.qids.split(",") if q.strip()]
            if args.qids else None)
    jobs = load_jobs(qids, Path(args.manifest))
    summary = asyncio.run(run_cells(jobs, families=families, policies=policies,
                                    out_dir=Path(args.out)))
    print(json.dumps({"cells": {c: {k: v for k, v in info.items() if k != "diagnostics"}
                                for c, info in summary["cells"].items()},
                      "extraction_identity": summary["extraction_identity"]},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
