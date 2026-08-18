"""SYN-RES v2 — the pilot Test-3 flow over the v2 manifest's t3 rows, PER
EXTRACTION FAMILY (protocol_spec_v2 §5.1 SYN-RES; §7 rows 5-7 T3 tails;
MINOR-4a pinned variant-replay horizon).

Per family (em = frozen `extractor.txt`/`extractor_dyadic.txt`, ez =
`extractor_zep.txt`/`extractor_zep_dyadic.txt`) and per benchmark (lme,
locomo; never pooled — FM-22), for each of the 50 frozen t3 cases:

  stage    the staging aside is spliced alone into a CLEAN copy of the
           substrate/conversation (pilot injector §9.1 flow); ONE extraction
           call at the staging turn under the family's frozen prompt (B2
           rendering: ≤2 preceding turns; LoCoMo via the dyadic single-pass
           rendering); staged-tuple selection = M5 span match against the
           aside text with key-match fallback (natres.match_staged_tuple);
           ONE classifier call (frozen classifier_short.txt, storeless —
           t3 staging keys are fresh aux keys, old_value null on all 100
           rows); the tuple stages iff argmax = speculative (the gated
           cells' route, gated_store phase 3). This is the per-family seam:
           the v1 flow staged the designed tuple directly; v2 composes the
           gate end-to-end per extraction arm exactly as the gated cells and
           the NAT-RES storeless pass (natres.py) do — non-staged cases are
           scored intent-to-treat (misses / trivially-clean distractors) with
           the conditional-on-staged stratum reported beside.
  generate predicates from the staged tuple + the transcript prefix through
           the staging turn (both roles, dates — v1 B6 firewall; identical to
           gated_store._generate_predicates and the pilot flow), frozen
           predicate_generator.txt; §3.3 lint with exactly one feedback retry.
           B6 ordering hook: test3.assert_naturalness_gate_passed against the
           v2 flag results/v2/naturalness_gate_resolved.flag (same assert
           semantics, adapted path — the v2 gates resolved 2026-07-04 under
           the preregistered FM-1 fallback BEFORE any call here).
  freeze   per-family predicate manifest at
           results/v2/test3_<family>/predicates_manifest.jsonl with
           staging_turn_sha256 + generator llm cache keys. FREEZE INTEGRITY:
           the orchestrator's "git commit" precondition of the pilot flow is
           not available in this run (no-commit execution); the same
           frozen-and-unmodified semantics is enforced by a sha256 sidecar
           (frozen_manifest.sha256) written at generate time and ASSERTED by
           `check` before any checker call (deviation recorded in results).
  check    attach variants (already rendered in the v2 manifest — 4/case):
           replay = clean copy + [staging, variant]; the staging-turn hash is
           re-asserted against the frozen manifest per replay (B6); 6-gram
           leak scan of every frozen condition vs the ORIGINAL transcript
           turns after the staging turn and all variant texts (§3.2 item 4);
           checker (frozen checker.txt) one call per replayed user turn with
           the case's single pending item, consume-on-fire. VARIANT-REPLAY
           HORIZON (MINOR-4a, §7 anchor T3_TAIL_TURNS = 5): each variant tail
           replays ONLY the user turns [max(0, t*−2) .. t*+2] — the designated
           resolution/distractor turn plus the v1 §3.4 2-turn sensitivity
           window, with a 2-turn pre-window false-fire surface (the §7 rows
           5/7 budget five checker turns per variant; the pilot's
           full-continuation replay is deliberately NOT carried: rows 1-2/7
           "carry no other unbounded term").
  score    v1 §3.4 / predicate_spec §4 verbatim (test3.score_variant /
           aggregate_counts), per benchmark per family: promotion recall
           (point ≥ 0.8 AND cluster-by-case lower ≥ 0.65), false-fire
           (point ≤ 0.05 AND cluster upper ≤ 0.075, n = 100 distractor
           probes), invalidation reported; + the TTC staged-then-resolved
           feed (P5). Primary denominators are the frozen 50 cases (ITT);
           the conditional-on-staged stratum is printed beside (FM-1-style
           two-row reporting).

Outputs -> ../results/v2/test3_em/ and ../results/v2/test3_ez/
CLI (from pilot/): .venv/bin/python test3_v2.py {generate,check,score}
                   [--families em,ez] [--benchmarks lme,locomo]
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import _bootstrap  # noqa: F401  (R1 layout: puts src/ + experiments/ on sys.path)

import classify
import data
import gated_store
import inject
import llm
import locomo_data
import locomo_render
import metrics_v2
import natres
import run_predicates as test3

from paths import PROMPTS_DIR, RESULTS_DIR
V2_DIR = RESULTS_DIR / "v2"
MANIFEST_V2_PATH = V2_DIR / "injections" / "manifest.jsonl"
GATE_FLAG_V2_PATH = V2_DIR / "naturalness_gate_resolved.flag"

SEED = 20260702
TAIL_PRE_WINDOW = 2       # pinned tail = [max(0, t*-2) .. t*+2]  (≤ 5 turns)
SENSITIVITY_WINDOW = test3.SENSITIVITY_WINDOW  # = 2 (v1 §3.4)

FAMILIES = ("em", "ez")
BENCHMARKS = ("lme", "locomo")
LME_PROMPTS = {"em": "extractor.txt", "ez": "extractor_zep.txt"}
DYADIC_FAMILY = {"em": "E-M", "ez": "E-Z"}


def out_dir(family: str) -> Path:
    return V2_DIR / f"test3_{family}"


def manifest_path(family: str) -> Path:
    return out_dir(family) / "predicates_manifest.jsonl"


def sidecar_path(family: str) -> Path:
    return out_dir(family) / "frozen_manifest.sha256"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


# ---------------------------------------------------------------------------
# Fresh-call accounting (budget discipline; cache hits are free)
# ---------------------------------------------------------------------------

class FreshCounter:
    def __init__(self):
        self.fresh = 0
        self._orig = llm.call_llm

    def install(self):
        orig = self._orig

        async def counted(system_prompt, user_prompt, *, trace_tag, **kw):
            model = kw.get("model", llm.MODEL)
            effort = kw.get("reasoning_effort", llm.REASONING_EFFORT)
            cached = (kw.get("use_cache", True)
                      and (llm.CACHE_DIR / (llm._cache_key(
                          model, effort, system_prompt, user_prompt) + ".json")).exists())
            if not cached:
                self.fresh += 1
            return await orig(system_prompt, user_prompt, trace_tag=trace_tag, **kw)

        llm.call_llm = counted
        gated_store.llm.call_llm = counted
        locomo_render.call_llm = counted

    def uninstall(self):
        llm.call_llm = self._orig
        locomo_render.call_llm = self._orig


# ---------------------------------------------------------------------------
# Frozen t3 case loading (v2 manifest) + world/stream construction
# ---------------------------------------------------------------------------

def load_cases(benchmark: str, *, require_variants: bool) -> dict[str, dict]:
    rows = [r for r in _load_jsonl(MANIFEST_V2_PATH) if r["benchmark"] == benchmark]
    assert rows, f"no v2 manifest rows for benchmark {benchmark}"
    cases = test3.load_t3_cases(rows, require_variants=require_variants)
    assert len(cases) == 50, f"{benchmark}: expected 50 t3 cases, got {len(cases)}"
    return cases


def build_world(benchmark: str, question_id: str, rows: list[dict]):
    """Clean copy of the substrate/conversation with ONLY `rows` spliced."""
    if benchmark == "lme":
        return inject.build_perturbed_substrate(data.get_substrate(question_id), rows)
    from inject_v2 import build_perturbed_conversation
    return build_perturbed_conversation(locomo_data.get_conversation(question_id), rows)


def world_turns(benchmark: str, world) -> list[dict]:
    """Chronological turn rows {session_idx, turn_idx, role, date, content}.
    LoCoMo turns render as '{Speaker}: {text}' lines (all human, V2-P3)."""
    if benchmark == "lme":
        rows = []
        for row in gated_store.substrate_stream(world):
            rows.append({"session_idx": row["session_idx"], "turn_idx": row["turn_idx"],
                         "role": row["role"], "date": row["date"],
                         "content": row["content"]})
        return rows
    rows = []
    for si, ti, turn in locomo_data.dialog_turns(world):
        rows.append({"session_idx": si, "turn_idx": ti, "role": "user",
                     "date": world.dates[si],
                     "content": locomo_render.render_turn_line(turn)})
    return rows


def staging_pos(turns: list[dict], placement: dict) -> int:
    for i, t in enumerate(turns):
        if (t["session_idx"] == placement["session_idx"]
                and t["turn_idx"] == placement["turn_idx"]):
            return i
    raise ValueError(f"staging turn {placement} not found")


def continuation_user_turns(turns: list[dict], s_pos: int) -> list[dict]:
    out = []
    for t in turns[s_pos + 1:]:
        if t["role"] == "user":
            out.append({**t, "pos": len(out)})
    return out


# ---------------------------------------------------------------------------
# generate — per-family staging (extract -> classify -> predicates) + freeze
# ---------------------------------------------------------------------------

async def _extract_staging(benchmark: str, family: str, world, staging: dict,
                           turns: list[dict], s_pos: int) -> list[dict]:
    """ONE extraction call at the staging turn under the family prompt (B2)."""
    case_id = staging["t3_case_id"]
    if benchmark == "lme":
        # Full-prefix prior context — byte-identical rendering contract to the
        # cells' extraction pathway (gated_store.extract_stream_tuples).
        prior = [{"role": t["role"], "content": t["content"], "date": t["date"]}
                 for t in turns[:s_pos]]
        tuples = await gated_store.extract_turn_family(
            prior, turns[s_pos]["content"],
            f"t3v2-extract-{family}-{case_id}",
            system_prompt=(PROMPTS_DIR / LME_PROMPTS[family]).read_text(),
            current_date=turns[s_pos]["date"])
        return [dataclasses.asdict(t) for t in tuples]
    si, ti = staging["placement"]["session_idx"], staging["placement"]["turn_idx"]
    job = locomo_render.dyadic_turn_job(world, si, ti, family=DYADIC_FAMILY[family])
    tuples = await locomo_render.extract_dyadic_turn(
        job["prior_turns"], job["current_turn"],
        f"t3v2-extract-dyadic-{family}-{case_id}",
        family=DYADIC_FAMILY[family], current_date=job["current_date"])
    return [dataclasses.asdict(t) for t in tuples]


async def _generate_case(benchmark: str, family: str, staging: dict) -> dict:
    case_id = staging["t3_case_id"]
    world = build_world(benchmark, staging["question_id"], [staging])
    turns = world_turns(benchmark, world)
    s_pos = staging_pos(turns, staging["placement"])
    staging_text = turns[s_pos]["content"]
    assert turns[s_pos]["role"] == "user", f"{case_id}: staging turn is not a user turn"
    assert staging["text"] in staging_text, \
        f"{case_id}: rendered staging aside not found in the replayed staging turn"

    record: dict[str, Any] = {
        "benchmark": benchmark,
        "case_id": case_id,
        "staged_id": f"stg_{case_id}_01",
        "question_id": staging["question_id"],
        "entity": staging["entity"],
        "attribute": staging["attribute"],
        "designed_value": staging["value"],
        "staged_at_turn": {"question_id": staging["question_id"],
                           "session_idx": staging["placement"]["session_idx"],
                           "turn_idx": staging["placement"]["turn_idx"],
                           "date": turns[s_pos]["date"]},
        "staging_turn_sha256": _sha256(staging_text),
        "seed": SEED,
    }

    # 1 — extraction (family seam; one call, B2/dyadic rendering)
    tuples = await _extract_staging(benchmark, family, world, staging, turns, s_pos)
    tup, match_method = natres.match_staged_tuple(
        tuples, staging["text"],
        {"entity": staging["entity"], "attribute": staging["attribute"]})
    record["extraction"] = {"n_tuples": len(tuples), "match_method": match_method}
    if tup is None:
        record.update(staged=False, outcome_reason="extraction_miss")
        return record
    record["staged_tuple"] = tup
    record["hypothesized_value"] = tup["value"]

    # 2 — classifier route (storeless: fresh aux key, old_value null by design)
    prior_ctx = gated_store._preceding_turns(
        [{"session_idx": t["session_idx"], "role": t["role"],
          "content": t["content"]} for t in turns], s_pos)
    item = {
        "item_id": f"t3v2-{family}-{case_id}",
        "entity": staging["entity"],
        "attribute": staging["attribute"],
        "asserted_value": tup["value"],
        "stored_value": None,
        "turn_text": staging_text,
        "preceding_turns": prior_ctx,
    }
    cls = await classify.classify_item(item)
    record["classifier"] = {"argmax": cls["argmax"], "s_ev": cls["s_ev"],
                            "max_prob": cls["max_prob"], "probs": cls["probs"]}
    staged = cls["argmax"] == "speculative"
    record["staged"] = staged
    if not staged:
        record["outcome_reason"] = f"not_staged_argmax_{cls['argmax']}"
        return record

    # 3 — predicate generation (B6 firewall: full prefix through staging turn,
    # both roles, dates — identical to the pilot flow / gated_store)
    gen_system = test3.GENERATOR_PROMPT_PATH.read_text()
    prefix = [{"date": t["date"], "role": t["role"], "content": t["content"]}
              for t in turns[:s_pos + 1]]
    user_prompt = test3.build_generator_user_prompt(
        prefix, staging["entity"], staging["attribute"], tup["value"])
    cache_keys = [test3._cache_key(gen_system, user_prompt)]
    out = await llm.call_llm(gen_system, user_prompt,
                             trace_tag=f"t3v2_gen_{family}_{case_id}")
    aliases = test3.attribute_aliases(staging["attribute"])
    promote_when, invalidate_when = str(out["promote_when"]), str(out["invalidate_when"])
    first_failures = test3.lint_predicates(promote_when, invalidate_when,
                                           tup["value"], aliases)
    final_failures, retried = first_failures, False
    if first_failures:  # §3.3: exactly one regeneration with lint feedback
        retried = True
        retry_prompt = user_prompt + test3.lint_feedback_block(first_failures)
        cache_keys.append(test3._cache_key(gen_system, retry_prompt))
        out = await llm.call_llm(gen_system, retry_prompt,
                                 trace_tag=f"t3v2_gen_retry_{family}_{case_id}")
        promote_when, invalidate_when = str(out["promote_when"]), str(out["invalidate_when"])
        final_failures = test3.lint_predicates(promote_when, invalidate_when,
                                               tup["value"], aliases)
    stated = (out.get("certainty_meta") or {}).get("stated_certainty")
    record.update({
        "promote_when": promote_when,
        "invalidate_when": invalidate_when,
        "certainty_meta": {"stated_certainty": stated,
                           "stated_certainty_in_vocab": stated in test3.CERTAINTY_VOCAB,
                           "horizon_hint": (out.get("certainty_meta") or {}).get("horizon_hint"),
                           "secondhand": (out.get("certainty_meta") or {}).get("secondhand")},
        "generator_cache_keys": cache_keys,
        "generator_prompt_sha256": _sha256(gen_system),
        "lint": {"first_failures": first_failures, "retried": retried,
                 "final_failures": final_failures},
        "outcome_reason": "staged",
    })
    return record


async def generate(families: tuple[str, ...], benchmarks: tuple[str, ...]) -> None:
    # B6 ordering hook — v2 flag path; assert semantics unchanged (test3.py).
    test3.assert_naturalness_gate_passed(flag_path=GATE_FLAG_V2_PATH)
    for family in families:
        rows: list[dict] = []
        for benchmark in benchmarks:
            cases = load_cases(benchmark, require_variants=False)
            results = await asyncio.gather(*[
                _generate_case(benchmark, family, cases[cid]["staging"])
                for cid in sorted(cases)])
            rows.extend(results)
            n_staged = sum(r["staged"] for r in results)
            n_miss = sum(r["outcome_reason"] == "extraction_miss" for r in results)
            retried = sum(1 for r in results if r.get("lint", {}).get("retried"))
            residual = [r["case_id"] for r in results
                        if r.get("lint", {}).get("final_failures")]
            print(f"[generate {family}/{benchmark}] staged {n_staged}/50 "
                  f"(extraction misses {n_miss}, "
                  f"non-spec routes {50 - n_staged - n_miss}); lint retried "
                  f"{retried}, residual failures {residual}")
        _dump_jsonl(manifest_path(family), rows)
        sidecar_path(family).write_text(_sha256(manifest_path(family).read_text()))
        print(f"froze {len(rows)} rows -> {manifest_path(family)} "
              f"(sha256 sidecar written)")


# ---------------------------------------------------------------------------
# check — freeze assert -> staging-hash asserts -> leak scan -> checker
# ---------------------------------------------------------------------------

def assert_manifest_frozen(family: str) -> str:
    """No-commit analog of test3.assert_predicates_manifest_committed: the
    manifest bytes must match the sha256 sidecar written at generate time
    (frozen-and-unmodified semantics; the git-commit step is unavailable in
    this execution and recorded as a deviation)."""
    path, side = manifest_path(family), sidecar_path(family)
    assert path.exists(), f"predicates manifest missing: {path}"
    assert side.exists(), f"freeze sidecar missing: {side}"
    digest = _sha256(path.read_text())
    assert digest == side.read_text().strip(), \
        f"{family}: predicates manifest modified after freeze (sha256 mismatch)"
    return digest


def run_leak_scan(family: str, benchmark: str, pred_rows: dict[str, dict],
                  cases: dict[str, dict]) -> dict:
    """§3.2 item 4: 6-gram overlap of every frozen condition vs (a) ORIGINAL
    (unperturbed) transcript turns after the staging turn and (b) all variant
    texts of the benchmark. Flags reported, never auto-failed."""
    all_variant_texts = [v["text"] for case in cases.values()
                        for v in case["variants"].values()]
    flags = []
    n_scanned = 0
    for case_id in sorted(pred_rows):
        pred = pred_rows[case_id]
        if not pred.get("staged"):
            continue
        staging = cases[case_id]["staging"]
        original = (data.get_substrate(pred["question_id"]) if benchmark == "lme"
                    else locomo_data.get_conversation(pred["question_id"]))
        turns = world_turns(benchmark, original)
        s_pos = staging_pos(turns, staging["placement"])
        future_texts = [t["content"] for t in turns[s_pos + 1:]]
        for field in ("promote_when", "invalidate_when"):
            n_scanned += 1
            for source, texts in (("original_future_turns", future_texts),
                                  ("variant_texts", all_variant_texts)):
                for ngram in test3.find_six_gram_overlaps(pred[field], texts):
                    flags.append({"case_id": case_id, "staged_id": pred["staged_id"],
                                  "field": field, "source": source,
                                  "ngram": " ".join(ngram)})
    return {"benchmark": benchmark, "n_conditions_scanned": n_scanned,
            "n_flags": len(flags), "flags": flags,
            "note": "flag list reported for manual review, not auto-failed (§3.2 item 4)"}


async def _run_variant(family: str, benchmark: str, case_id: str, kind: str,
                       pred: dict, tail: list[dict],
                       checker_system: str) -> list[dict]:
    """Checker over the pinned tail; consume-on-fire (pilot semantics)."""
    pending = ([test3.checker_pending_item(pred)] if pred.get("staged") else [])
    log_rows = []
    for turn in tail:
        row = {"benchmark": benchmark, "case_id": case_id, "variant_kind": kind,
               "turn_pos": turn["pos"], "session_idx": turn["session_idx"],
               "turn_idx": turn["turn_idx"], "turn_text": turn["content"],
               "turn_date": turn["date"]}
        if not pending:
            log_rows.append({**row, "skipped": True, "output": None, "fires": [],
                             "malformed_output": False, "extra_keys": []})
            continue
        assert len(pending) == 1  # M2: one staged item per injected case
        pending_ids = {p["staged_id"] for p in pending}
        output = await llm.call_llm(
            checker_system,
            test3.build_checker_user_prompt(turn["content"], turn["date"], pending),
            trace_tag=f"t3v2_check_{family}_{case_id}_{kind}_{turn['pos']}")
        fires, malformed, extra_keys = test3.parse_checker_fires(
            output, pending_ids, turn["content"])
        consumed = {f["staged_id"] for f in fires if f["staged_id"] in pending_ids}
        pending = [p for p in pending if p["staged_id"] not in consumed]
        log_rows.append({**row, "skipped": False, "output": output, "fires": fires,
                         "malformed_output": malformed, "extra_keys": extra_keys})
    return log_rows


async def check(families: tuple[str, ...], benchmarks: tuple[str, ...]) -> None:
    checker_system = test3.CHECKER_PROMPT_PATH.read_text()
    for family in families:
        digest = assert_manifest_frozen(family)
        all_rows = _load_jsonl(manifest_path(family))
        replay_meta, coros, leaks = [], [], []
        for benchmark in benchmarks:
            pred_rows = {r["case_id"]: r for r in all_rows
                         if r["benchmark"] == benchmark}
            cases = load_cases(benchmark, require_variants=True)
            assert set(pred_rows) == set(cases), \
                f"{family}/{benchmark}: manifest and t3 case set disagree"
            leaks.append(run_leak_scan(family, benchmark, pred_rows, cases))

            for case_id in sorted(cases):
                case, pred = cases[case_id], pred_rows[case_id]
                staging = case["staging"]
                for kind in test3.VARIANT_KINDS:
                    vrow = case["variants"][kind]
                    world = build_world(benchmark, staging["question_id"],
                                        [staging, vrow])
                    turns = world_turns(benchmark, world)
                    s_pos = staging_pos(turns, staging["placement"])
                    # B6: staging hash re-asserted against the frozen manifest.
                    assert _sha256(turns[s_pos]["content"]) == pred["staging_turn_sha256"], \
                        f"{case_id} {kind}: staging turn hash mismatch vs frozen manifest (B6)"
                    cont = continuation_user_turns(turns, s_pos)
                    t_star_pos = staging_pos(cont, vrow["placement"])
                    assert vrow["text"] in cont[t_star_pos]["content"], \
                        f"{case_id} {kind}: variant text not found in designed turn t*"
                    # §4.1: the sensitivity window must exist.
                    assert len(cont) > t_star_pos + SENSITIVITY_WINDOW, \
                        f"{case_id} {kind}: continuation too short for the sensitivity window"
                    tail = cont[max(0, t_star_pos - TAIL_PRE_WINDOW):
                                t_star_pos + SENSITIVITY_WINDOW + 1]
                    replay_meta.append({
                        "benchmark": benchmark, "case_id": case_id,
                        "variant_kind": kind,
                        "question_id": staging["question_id"],
                        "staged_id": pred["staged_id"],
                        "staged": bool(pred.get("staged")),
                        "not_staged_reason": (None if pred.get("staged")
                                              else pred["outcome_reason"]),
                        "gold_action": test3.GOLD_ACTION_BY_KIND[kind],
                        "invalidate_style": test3.invalidate_style_of(vrow),
                        "t_star_pos": t_star_pos,
                        "t_star_session_idx": vrow["placement"]["session_idx"],
                        "t_star_turn_idx": vrow["placement"]["turn_idx"],
                        "n_continuation_user_turns": len(cont),
                        "tail_positions": [t["pos"] for t in tail],
                        "staging_date": turns[s_pos]["date"],
                        "t_star_date": cont[t_star_pos]["date"],
                        "staged_duration_user_turns": t_star_pos + 1,
                        "staged_duration_days": test3.days_between(
                            turns[s_pos]["date"], cont[t_star_pos]["date"]),
                        "manifest_sha256": digest,
                    })
                    coros.append(_run_variant(family, benchmark, case_id, kind,
                                              pred, tail, checker_system))
        all_logs = await asyncio.gather(*coros)
        checker_log = [row for rows in all_logs for row in rows]
        _dump_jsonl(out_dir(family) / "variants_replayed.jsonl", replay_meta)
        _dump_jsonl(out_dir(family) / "checker_log.jsonl", checker_log)
        (out_dir(family) / "leak_scan.json").write_text(
            json.dumps({"scans": leaks}, ensure_ascii=False, indent=1))
        n_calls = sum(not r["skipped"] for r in checker_log)
        print(f"[check {family}] {len(replay_meta)} variant replays, "
              f"{n_calls} checker calls "
              f"({len(checker_log) - n_calls} tail turns skipped); leak flags: "
              f"{[l['n_flags'] for l in leaks]}")


# ---------------------------------------------------------------------------
# score — v1 §3.4 verbatim per (family, benchmark), ITT + conditional strata
# ---------------------------------------------------------------------------

def _scores_for(replays: list[dict], checker_log: list[dict]
                ) -> list[test3.VariantScore]:
    log_by_variant: dict[tuple, list[dict]] = {}
    for row in checker_log:
        log_by_variant.setdefault((row["case_id"], row["variant_kind"]), []).append(row)
    for rows in log_by_variant.values():
        rows.sort(key=lambda r: r["turn_pos"])
    scores = []
    for meta in replays:
        gold = test3.VariantGold(
            case_id=meta["case_id"], variant_kind=meta["variant_kind"],
            gold_action=meta["gold_action"], gold_staged_id=meta["staged_id"],
            t_star_pos=meta["t_star_pos"],
            invalidate_style=meta["invalidate_style"])
        rows = log_by_variant.get((meta["case_id"], meta["variant_kind"]), [])
        fires = [test3.FireEvent(turn_pos=row["turn_pos"], staged_id=f["staged_id"],
                                 action=f["action"], invalid=f["invalid"])
                 for row in rows for f in row["fires"]]
        scores.append(test3.score_variant(gold, fires))
    return scores


def _block(scores: list[test3.VariantScore], replays: list[dict],
           preds: dict[str, dict], benchmark: str, family: str) -> dict:
    counts = test3.aggregate_counts(scores)
    promote = counts["promote"]
    ff = counts["false_fire"]

    promotion = metrics_v2.synres_promotion_rule(
        promote["successes_primary"], promote["clusters"])
    promotion_sens = metrics_v2.synres_promotion_rule(
        promote["successes_sensitivity"], promote["clusters"])
    false_fire = metrics_v2.synres_false_fire_rule(
        ff["probe"]["successes"], ff["probe"]["clusters"])
    invalid = counts["invalidate"]

    staged_ids = {r["case_id"] for r in replays if r["staged"]}
    cond_scores = [s for s in scores if s.gold.case_id in staged_ids]
    cond_counts = test3.aggregate_counts(cond_scores) if cond_scores else None

    def _cond_rule(fn, block_key, succ_key):
        if not cond_counts:
            return None
        b = (cond_counts[block_key] if block_key != "false_fire"
             else cond_counts["false_fire"]["probe"])
        return fn(b[succ_key], b["clusters"])

    lags = promote["lags"]
    n_promote = promote["n"]
    censored = promote["censored"]
    reasons: dict[str, int] = {}
    for r in replays:
        if not r["staged"] and r["variant_kind"] == "promote":
            reasons[r["not_staged_reason"]] = reasons.get(r["not_staged_reason"], 0) + 1
    return {
        "benchmark": benchmark,
        "family": family,
        "n_cases": len({s.gold.case_id for s in scores}),
        "n_staged_cases": len(staged_ids),
        "staging": {
            "n_staged": len(staged_ids), "n_cases": 50,
            "not_staged_reasons_promote_variants": reasons,
            "note": ("per-family gate entry (extract -> classify -> stage): "
                     "non-staged cases are ITT misses / trivially-clean "
                     "distractors; conditional-on-staged stratum below"),
        },
        "promotion_recall_ITT": {"primary": promotion, "sensitivity": promotion_sens},
        "promotion_recall_conditional_on_staged": {
            "primary": _cond_rule(metrics_v2.synres_promotion_rule,
                                  "promote", "successes_primary"),
            "sensitivity": _cond_rule(metrics_v2.synres_promotion_rule,
                                      "promote", "successes_sensitivity"),
        },
        "invalidation_recall_reported": {
            "primary": {"x": invalid["x_primary"], "n": invalid["n"]},
            "sensitivity": {"x": invalid["x_sensitivity"], "n": invalid["n"]},
            "by_style": {style: {"x": invalid["by_style"][style]["x_sensitivity"],
                                 "n": invalid["by_style"][style]["n"]}
                         for style in ("soft", "hard")},
        },
        "false_fire_ITT": false_fire,
        "false_fire_conditional_on_staged": _cond_rule(
            metrics_v2.synres_false_fire_rule, "false_fire", "successes"),
        "total_false_fire_events": counts["total_false_fire_events"],
        "ttc_feed_p5": {
            "firing_lags_user_turns": lags,
            "lag_distribution": {str(k): lags.count(k) for k in (0, 1, 2)},
            "right_censored": {"x": censored, "n": n_promote,
                               "fraction": censored / n_promote if n_promote else None},
            "censoring_bound": 0.2,
            "censoring_within_bound": (n_promote > 0 and censored / n_promote <= 0.2),
            "note": ("gated-arm TTC for staged-then-resolved items = firing lag "
                     "(P5); ITT censoring includes never-staged cases"),
        },
        "lint_residual_failures": sorted(
            c for c, r in preds.items()
            if r["benchmark"] == benchmark and r.get("lint", {}).get("final_failures")),
    }


def score(families: tuple[str, ...], benchmarks: tuple[str, ...]) -> None:
    for family in families:
        digest = assert_manifest_frozen(family)
        preds = {r["case_id"]: r for r in _load_jsonl(manifest_path(family))}
        replays = _load_jsonl(out_dir(family) / "variants_replayed.jsonl")
        checker_log = _load_jsonl(out_dir(family) / "checker_log.jsonl")
        leak = json.loads((out_dir(family) / "leak_scan.json").read_text())
        results: dict[str, Any] = {
            "endpoint": "SYN-RES",
            "family": family,
            "seed": SEED,
            "horizon_pin": ("variant tail = user turns [max(0, t*-2) .. t*+2] "
                            "(MINOR-4a; §7 T3 tail anchor = 5 turns/variant)"),
            "freeze": {"manifest_sha256": digest,
                       "deviation": ("git-commit freeze step unavailable in this "
                                     "execution; sha256 sidecar asserted instead "
                                     "(same frozen-and-unmodified semantics)")},
            "provenance": {
                "generator_prompt_sha256": _sha256(test3.GENERATOR_PROMPT_PATH.read_text()),
                "checker_prompt_sha256": _sha256(test3.CHECKER_PROMPT_PATH.read_text()),
                "model": llm.MODEL,
                "gate_flag": str(GATE_FLAG_V2_PATH),
            },
            "leak_scan_flags": {s["benchmark"]: s["n_flags"] for s in leak["scans"]},
            "benchmarks": {},
        }
        for benchmark in benchmarks:
            b_replays = [r for r in replays if r["benchmark"] == benchmark]
            b_log = [r for r in checker_log if r["benchmark"] == benchmark]
            scores = _scores_for(b_replays, b_log)
            results["benchmarks"][benchmark] = _block(
                scores, b_replays, preds, benchmark, family)
            blk = results["benchmarks"][benchmark]
            p = blk["promotion_recall_ITT"]["primary"]
            f = blk["false_fire_ITT"]
            print(f"[score {family}/{benchmark}] staged {blk['n_staged_cases']}/50; "
                  f"promotion ITT {p['x']}/{p['n']} point={p['point']:.3f} "
                  f"lower={p['ci'][0]:.3f} pass={p['pass']}; false-fire "
                  f"{f['x']}/{f['n']} point={f['point']:.3f} upper={f['ci'][1]:.3f} "
                  f"pass={f['pass']}")
        (out_dir(family) / "results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=1))
        print(f"wrote {out_dir(family) / 'results.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SYN-RES v2 per-family runner")
    parser.add_argument("stage", choices=["generate", "check", "score"])
    parser.add_argument("--families", default="em,ez")
    parser.add_argument("--benchmarks", default="lme,locomo")
    args = parser.parse_args()
    families = tuple(f for f in args.families.split(",") if f in FAMILIES)
    benchmarks = tuple(b for b in args.benchmarks.split(",") if b in BENCHMARKS)
    counter = FreshCounter()
    counter.install()
    try:
        if args.stage == "generate":
            asyncio.run(generate(families, benchmarks))
        elif args.stage == "check":
            asyncio.run(check(families, benchmarks))
        else:
            score(families, benchmarks)
    finally:
        counter.uninstall()
        print(f"[budget] fresh LLM calls this run: {counter.fresh}")


if __name__ == "__main__":
    main()
