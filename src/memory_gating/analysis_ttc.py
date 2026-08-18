"""Turns-to-canonical (TTC) staleness analysis — protocol_spec.md §3.2 (R2-M-f).

Consumes the Test 2 write-through StoreTimelines (per-user-turn snapshots),
the frozen key schema, and the injection manifest, and computes, for every
genuine update (each replayed substrate's gold key old -> new change):

- `t_L` per M12: the EARLIEST licensing turn — the key_schema `new_evidence`
  turn plus any `extra_evidence` rows in the new-value session (e.g.
  `e493bb7c` s1 t0 unmarked before s1 t6 marked; `8fb83627` s1 t2 before
  s1 t6, per the key_schema notes);
- `TTC(u) = min{t >= t_L : lookup(k_u) ≡ v_new} − t_L` in USER turns, with the
  session-boundary version reported too; never reaching canonical ⇒
  right-censored at transcript end (reported as a fraction, never imputed);
- median / IQR (Tukey hinges) and censoring fractions, split per P5:
  * immediately-licensed updates — every Test 2 write-through genuine update
    (completed-event evidence); expected censoring ≈ 0 (a censored item here
    indicates extractor or gate failure and is reported as such);
  * staged-then-resolved items — fed by Test 3's predicate firing lag
    (`ttc_feed_p5` in ../results/test3/results.json, P5: the gated arm's TTC
    for staged items equals the firing lag), read when available.

Value equivalence follows §3.1: normalized exact match first, unresolved pairs
to the blind LLM equivalence judge (store.values_equivalent; llm.py-cached, so
pairs already judged by test2_writethrough are free).

Writes ../results/test2/ttc.json. Wired into test2_writethrough.run(); can be
rerun standalone (replays hit the llm.py cache):
    .venv/bin/python analysis_ttc.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import store

from paths import RESULTS_DIR
TTC_PATH = RESULTS_DIR / "test2" / "ttc.json"
TEST3_RESULTS_PATH = RESULTS_DIR / "test3" / "results.json"

T3_TYPES = {"t3_staging", "t3_variant"}


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without LLM calls)
# ---------------------------------------------------------------------------


def licensing_candidates(entry: dict) -> list[dict]:
    """All turns evidencing the NEW value: the key_schema `new_evidence` turn
    plus every `extra_evidence` row in the new-value session (the substrates
    are 2-session; new-session extra evidence restates v_new — see the
    key_schema notes for e493bb7c / 8fb83627)."""
    new = entry["new_evidence"]
    cands = [{"session_idx": new["session_idx"], "turn_idx": new["turn_idx"]}]
    for extra in entry.get("extra_evidence", []):
        if extra["session_idx"] == new["session_idx"]:
            cands.append({"session_idx": extra["session_idx"],
                          "turn_idx": extra["turn_idx"]})
    return sorted(cands, key=lambda c: c["turn_idx"])


def licensing_turn(entry: dict) -> tuple[int, int]:
    """M12: `t_L` = the EARLIEST licensing turn (changelog entry 31)."""
    first = licensing_candidates(entry)[0]
    return (first["session_idx"], first["turn_idx"])


def median_iqr(values: list) -> dict:
    """Median + Tukey-hinge IQR (median of lower/upper half; deterministic,
    exact on the small preregistered ns)."""
    if not values:
        return {"n": 0, "median": None, "iqr": None}
    ordered = sorted(values)

    def med(xs: list) -> float:
        mid = len(xs) // 2
        return xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2

    half = len(ordered) // 2
    if half == 0:
        iqr = [ordered[0], ordered[-1]]
    else:
        iqr = [med(ordered[:half]), med(ordered[-half:])]
    return {"n": len(ordered), "median": med(ordered), "iqr": iqr}


def gold_key_injections(entry: dict, question_id: str, manifest: list[dict]) -> list[str]:
    """Injection ids of non-T3 manifest rows targeting the substrate's gold key
    — context for interpreting a delayed/censored TTC (a spec_gold aside can
    overwrite the gold key after t_L in the write-through arm)."""
    key = (entry["entity"], entry["attribute"])
    return [
        row["injection_id"]
        for row in manifest
        if row["question_id"] == question_id
        and row["inj_type"] not in T3_TYPES
        and (row["entity"], row["attribute"]) == key
    ]


# ---------------------------------------------------------------------------
# Per-update TTC over StoreTimeline snapshots
# ---------------------------------------------------------------------------


async def ttc_for_update(timeline: store.StoreTimeline, entry: dict,
                         manifest: list[dict]) -> dict:
    """TTC of one genuine update against one substrate's snapshot timeline."""
    key = (entry["entity"], entry["attribute"])
    t_l = licensing_turn(entry)
    coords = [(s.session_idx, s.turn_idx) for s in timeline.snapshots]
    if t_l not in coords:
        raise ValueError(
            f"licensing turn {t_l} of {timeline.question_id} has no user-turn "
            "snapshot — key_schema evidence turns must be user turns")
    idx_l = coords.index(t_l)
    v_new = entry["new_value"]

    verdicts: dict[str, bool] = {}
    reach_idx: Optional[int] = None
    for idx in range(idx_l, len(coords)):
        stored = timeline.snapshots[idx].store.get(key)
        if stored is None:
            continue
        value = stored.value
        if value not in verdicts:
            verdicts[value] = await store.values_equivalent(
                entry["attribute"], value, v_new,
                trace_tag=f"ttc-{timeline.question_id}-{idx}")
        if verdicts[value]:
            reach_idx = idx
            break

    censored = reach_idx is None
    session_pos = {s: i for i, s in enumerate(timeline.session_order)}
    entry_at_tl = timeline.snapshots[idx_l].store.get(key)
    return {
        "question_id": timeline.question_id,
        "key": "/".join(key),
        "t_L": {"session_idx": t_l[0], "turn_idx": t_l[1]},
        "licensing_candidates": licensing_candidates(entry),
        "censored": censored,
        "ttc_user_turns": None if censored else reach_idx - idx_l,
        "ttc_session_boundaries": None if censored else (
            session_pos[coords[reach_idx][0]] - session_pos[t_l[0]]),
        "value_at_t_L": entry_at_tl.value if entry_at_tl is not None else None,
        "v_new": v_new,
        "gold_key_injections": gold_key_injections(entry, timeline.question_id, manifest),
    }


# ---------------------------------------------------------------------------
# Aggregation (P5 split) + Test 3 staged-then-resolved feed
# ---------------------------------------------------------------------------


def staged_then_resolved_feed(path: Path = TEST3_RESULTS_PATH) -> dict:
    """P5: the gated arm's TTC for staged-then-resolved items equals the
    predicate firing lag; test3.py publishes it as `ttc_feed_p5`."""
    block = {"source": str(path)}
    if not path.exists():
        return {**block, "available": False,
                "note": "Test 3 not yet run — staged-then-resolved TTC pending"}
    feed = json.loads(path.read_text())["ttc_feed_p5"]
    return {
        **block,
        "available": True,
        "ttc_user_turns": median_iqr(feed["firing_lags_user_turns"]),
        "right_censored": feed["right_censored"],
        "censoring_bound": feed.get("censoring_bound"),
        "censoring_within_bound": feed.get("censoring_within_bound"),
        "note": feed.get("note"),
    }


async def compute_ttc(timelines: dict, key_schema: dict,
                      manifest: list[dict]) -> dict:
    """§3.2 TTC over all replayed substrates' genuine updates."""
    per_update = []
    for qid in sorted(timelines):
        entry = key_schema.get(qid)
        if entry is None:
            continue
        per_update.append(await ttc_for_update(timelines[qid], entry, manifest))

    n = len(per_update)
    n_censored = sum(u["censored"] for u in per_update)
    uncensored_turns = [u["ttc_user_turns"] for u in per_update if not u["censored"]]
    uncensored_sessions = [u["ttc_session_boundaries"] for u in per_update
                           if not u["censored"]]
    return {
        "meta": {
            "seed": store.SEED,
            "generated_by": "analysis_ttc.py (protocol §3.2, M12/P5; R2-M-f)",
            "definition": ("TTC(u) = min{t >= t_L : lookup(k_u) equivalent v_new} - t_L "
                           "in user turns; t_L = earliest licensing turn (M12); "
                           "right-censored at transcript end, censored fraction "
                           "reported, never imputed"),
        },
        "immediately_licensed": {
            "n": n,
            "n_censored": n_censored,
            "censoring_fraction": (n_censored / n) if n else None,
            "ttc_user_turns": median_iqr(uncensored_turns),
            "ttc_session_boundaries": median_iqr(uncensored_sessions),
            "note": ("write-through arm: every Test 2 genuine update is immediately "
                     "licensed (completed-event evidence). Expected censoring ~ 0; a "
                     "censored item indicates extractor or gate failure (P5) — "
                     "gold-key injections are annotated per update for diagnosis"),
            "per_update": per_update,
        },
        "staged_then_resolved": staged_then_resolved_feed(),
    }


def write_ttc(results: dict, path: Path = TTC_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Standalone rerun (replays via the llm.py cache)
# ---------------------------------------------------------------------------


def main() -> None:
    import data
    import run_writethrough as t2  # lazy: t2 imports this module at top level

    key_schema = data.load_key_schema()
    manifest = t2.load_jsonl(t2.MANIFEST_PATH)
    qids = t2.scored_question_ids(manifest)

    async def _run() -> dict:
        timelines = await t2.replay_all_timelines(manifest, key_schema, qids)
        return await compute_ttc(timelines, key_schema, manifest)

    results = asyncio.run(_run())
    write_ttc(results)
    block = results["immediately_licensed"]
    print(f"wrote {TTC_PATH} (n={block['n']}, censored={block['n_censored']}, "
          f"median TTC={block['ttc_user_turns']['median']})")


if __name__ == "__main__":
    main()
