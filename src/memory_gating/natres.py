"""NAT-RES — the preregistered NATURAL staging→resolution endpoint (owner: W5).

Implements the protocol_spec_v2.md §4 STORELESS contract (V2-F3, fixes
BLOCKING-3) and feeds the §5.1 NAT-RES rule (metrics_v2.natres_rule). The
35-pair denominator is frozen (research/natural_pairs_v2.jsonl); the two
supplementary arcs (42ec0761, c6853660) are NEVER in the denominator.

LME half (17 pairs) — storeless, per pair × extractor family (em = frozen
`extractor.txt`, ez = `extractor_zep.txt`):
  1. ONE extraction call at the stage turn: up to 2 preceding turns as text
     context (B2 rendering via extract.render_extraction_prompt) — NO store
     replay. Staged tuple = M5 span match against the pair's stage quote
     (key-match fallback against the authored pair key, method recorded).
  2. ONE classifier call (frozen classifier_short.txt via classify.classify_item),
     stored_value = None (rule 1.5d no-stored-value branch). Route: staged iff
     argmax = speculative (commit-class argmax ⇒ the tuple bypasses staging;
     scored as not-promoted with the route recorded). Dual-branch sensitivity
     (V2-F3): P-LME-08 / P-LME-14 (`prior_value_before_stage`) get a second
     classifier call with the gold-sourced quoted prior assertion as
     stored_value (NO replay); the primary route stays the no-stored-value
     branch; divergence reported.
  3. Predicate generation from the staged tuple + the stage-turn text prefix
     (the same ≤2-preceding-turns + stage-turn prefix — v1 B6 firewall: nothing
     after the stage turn enters the call), frozen predicate_generator.txt,
     §3.3 lint with one retry (test3.lint_predicates).
  4. Checker (frozen checker.txt) over EXACTLY the user turns strictly after
     the stage anchor through the resolution anchor plus the v1 §3.4 2-turn
     window — the frozen per-pair horizons of key_schema_v2_lme.json
     `natural_pair_keys` (asserted per pair; sum = 632/family, §7 row 15).
     One call per (user turn, pending set); default no-fire; a fire consumes
     the pending item and remaining turns are skipped (logged).
  5. Gold-action scoring: hit iff a valid fire matches the pair's authored
     gold action (promote/invalidate) at the resolution turn (primary) or
     within the +2-turn sensitivity window; everything else is a false fire.

LoCoMo half (18 pairs): staged and checked IN-STREAM inside the gated LoCoMo
replay (§4/§7 row 7) — this module never replays LoCoMo. `score` reads the
in-stream trace at results/v2/natres/locomo_instream.jsonl when the gated
replay produces it (row contract in LOCOMO_TRACE_CONTRACT) and reports the
half as pending otherwise.

Labels (`label-pairs`): the 34 LME pair rows (stage + resolution) go through
the v1 §2.2 pipeline STORELESS — items packaged from the frozen-extractor
(E-M) tuple at their own turn, stored_value = None (rule 1.5d no-stored-value
branch); P-LME-08 / P-LME-14 rows additionally labeled under the stored-value
branch (gold-sourced prior quote), divergence reported. Escalations queue to
results/v2/natres/escalation_queue.jsonl (blinding per label.py R2-B-3).

Turn order: the frozen horizons are computed over the ORIGINAL haystack
session order (verified 17/17 against the artifact; the artifact is
authoritative per §4/V2-F3) — the runner walks sessions in original order.

Outputs -> ../results/v2/natres/
CLI (from pilot/): .venv/bin/python natres.py {run,label-pairs,score,smoke}
  run --family {em,ez,both}   # the storeless LME pass
  smoke                       # no-LLM dry run over all 17 pairs + one
                              # SYNTHETIC pipeline pass (<= 5 LLM calls; no
                              # preregistered item is touched pre-gate)
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Optional

import classify
import extract
import label
import llm
import metrics_v2
import run_predicates as test3

from paths import PROMPTS_DIR, RESEARCH_DIR, RESULTS_DIR
NATRES_DIR = RESULTS_DIR / "v2" / "natres"

PAIRS_PATH = RESEARCH_DIR / "natural_pairs_v2.jsonl"
LME_SCHEMA_PATH = RESEARCH_DIR / "key_schema_v2_lme.json"
LME_RUNS_PATH = NATRES_DIR / "lme_runs.jsonl"
PAIR_LABELS_PATH = NATRES_DIR / "pair_labels.jsonl"
ESCALATION_QUEUE_PATH = NATRES_DIR / "escalation_queue.jsonl"
LOCOMO_TRACE_PATH = NATRES_DIR / "locomo_instream.jsonl"
RESULTS_PATH = NATRES_DIR / "results.json"

SEED = 20260702
SENSITIVITY_WINDOW = 2   # v1 §3.4 window, carried by §5.1 NAT-RES
TOTAL_LME_HORIZON = 632  # §7 row 15 (per extractor family)
N_PAIRS_FROZEN = 35      # frozen denominator (V2-F3 / FM-19)
DUAL_LABEL_PAIRS = ("P-LME-08", "P-LME-14")  # prior_value_before_stage (§4)

FAMILY_PROMPTS = {"em": "extractor.txt", "ez": "extractor_zep.txt"}

# Contract for the gated-LoCoMo-replay owner (§4: in-stream, no separate
# replay). One row per (pair, family):
LOCOMO_TRACE_CONTRACT = {
    "pair_id": "P-LOCO-NN",
    "family": "em|ez",
    "staged": "bool — the stage-turn tuple was routed to staging in-stream",
    "hit_primary": "bool — valid gold-action fire ON the resolution turn",
    "hit_sensitivity": "bool — valid gold-action fire within resolution+2 user turns",
    "n_false_fires": "int — non-matching fires over the pair's horizon",
    "notes": "optional",
}


# ---------------------------------------------------------------------------
# Frozen-artifact loading and verification
# ---------------------------------------------------------------------------

def load_pairs() -> list[dict]:
    rows = [json.loads(line) for line in PAIRS_PATH.read_text().splitlines()
            if line.strip()]
    assert len(rows) == N_PAIRS_FROZEN, \
        f"frozen pair artifact must have {N_PAIRS_FROZEN} rows, got {len(rows)}"
    return rows


def load_lme_pair_keys() -> dict[str, dict]:
    schema = json.loads(LME_SCHEMA_PATH.read_text())
    entries = {e["pair_id"]: e for e in schema["natural_pair_keys"]["entries"]}
    assert len(entries) == 17, f"expected 17 LME pair keys, got {len(entries)}"
    total = sum(e["checker_horizon_user_turns"] for e in entries.values())
    assert total == TOTAL_LME_HORIZON, \
        f"authored horizons sum {total} != {TOTAL_LME_HORIZON} (§7 row 15)"
    return entries


def load_lme_pairs() -> list[dict]:
    """The 17 LME pairs joined with their authored keys (V2-F3)."""
    keys = load_lme_pair_keys()
    out = []
    for pair in load_pairs():
        if pair["corpus"] != "lme":
            continue
        key = keys[pair["pair_id"]]
        assert key["question_id"] == pair["source_question_id"]
        assert key["gold_action"] == pair["gold_action"], \
            f"{pair['pair_id']}: gold_action disagrees between artifacts"
        out.append({"pair": pair, "key": key})
    assert len(out) == 17
    return out


_PRIOR_QUOTE_RE = re.compile(r"s(\d+) t(\d+) '(.+?)' asserts")


def parse_prior_assertion(key_entry: dict) -> tuple[int, int, str]:
    """The gold-sourced quoted prior assertion for a `prior_value_before_stage`
    pair, parsed from the frozen prior_mention_scan (NO replay — V2-F3)."""
    m = _PRIOR_QUOTE_RE.search(key_entry["prior_mention_scan"])
    if not m:
        raise ValueError(
            f"{key_entry['pair_id']}: no quoted prior assertion in prior_mention_scan")
    return int(m.group(1)), int(m.group(2)), m.group(3)


# ---------------------------------------------------------------------------
# Stream construction (ORIGINAL haystack session order — matches the frozen
# horizons, verified 17/17; asserted per pair below)
# ---------------------------------------------------------------------------

def user_turns_original_order(substrate: Any) -> list[dict]:
    """User turns in ORIGINAL haystack session order with session dates.
    The frozen `natural_pair_keys` horizons were computed over this order;
    the artifact is authoritative (§4/V2-F3)."""
    out = []
    for sidx, session in enumerate(substrate.sessions):
        for tidx, turn in enumerate(session):
            if turn["role"] == "user":
                out.append({"session_idx": sidx, "turn_idx": tidx,
                            "text": turn["content"],
                            "date": substrate.dates[sidx]})
    return out


def build_pair_stream(pair: dict, key: dict, substrate: Any) -> dict:
    """Anchors, extraction context, and the frozen checker horizon for one
    LME pair. Asserts quote-verified anchors and the authored horizon."""
    stage = pair["stage"]
    resolution = pair["resolution"]
    turns = user_turns_original_order(substrate)
    index = {(t["session_idx"], t["turn_idx"]): i for i, t in enumerate(turns)}
    stage_pos = index[(stage["session_idx"], stage["turn_idx"])]
    res_pos = index[(resolution["session_idx"], resolution["turn_idx"])]
    assert res_pos > stage_pos, f"{pair['pair_id']}: resolution precedes stage"

    stage_turn = turns[stage_pos]
    res_turn = turns[res_pos]
    assert stage["text"] in stage_turn["text"], \
        f"{pair['pair_id']}: stage quote not verbatim in the stage turn"
    assert key["stage_evidence"]["quote"] in stage_turn["text"], \
        f"{pair['pair_id']}: authored stage evidence not verbatim"
    assert resolution["text"] in res_turn["text"], \
        f"{pair['pair_id']}: resolution quote not verbatim in the resolution turn"

    # user turns strictly after the stage anchor through resolution + 2 (§4)
    horizon = [dict(t, horizon_pos=i)
               for i, t in enumerate(turns[stage_pos + 1: res_pos + 1 + SENSITIVITY_WINDOW])]
    authored = key["checker_horizon_user_turns"]
    assert len(horizon) == authored, \
        (f"{pair['pair_id']}: computed horizon {len(horizon)} != authored {authored} "
         "(frozen natural_pair_keys, original haystack order)")
    resolution_horizon_pos = res_pos - stage_pos - 1

    # up to 2 preceding turns (either role) within the stage session, dated —
    # the B2 text-prefix context (§4: no store replay)
    prior = [dict(t, date=substrate.dates[stage["session_idx"]])
             for t in label.preceding_turns(
                 substrate, stage["session_idx"], stage["turn_idx"])]
    return {
        "pair_id": pair["pair_id"],
        "question_id": pair["source_question_id"],
        "stage_turn": stage_turn,
        "prior_turns": prior,
        "horizon_turns": horizon,
        "resolution_horizon_pos": resolution_horizon_pos,
        "gold_action": pair["gold_action"],
    }


# ---------------------------------------------------------------------------
# Staged-tuple matching (M5) and scoring — pure logic
# ---------------------------------------------------------------------------

def match_staged_tuple(tuples: list[dict], stage_quote: str,
                       key_entry: dict) -> tuple[Optional[dict], str]:
    """M5 natural branch: span match against the pair's stage quote; fallback
    to a key match against the authored pair key (highest confidence).

    v2.2 C1 harness correction (audited defect A1, protocol_changelog.md
    'Protocol v2.2'): the span match now prefers the authored pair key's
    tuple ON TIES at the best overlap score (label.match_tuple_to_span
    ``preferred_key``). The frozen 2026-07-04 lme_runs.jsonl artifacts are
    untouched; the correction applies to any post-v2.2 re-run only.
    """
    matched, _score = label.match_tuple_to_span(
        tuples, stage_quote,
        preferred_key=(key_entry["entity"], key_entry["attribute"]))
    if matched is not None:
        return matched, "span"
    key_hits = [t for t in tuples
                if label.key_matches(t, key_entry["entity"], key_entry["attribute"])]
    if key_hits:
        return max(key_hits, key=lambda t: t["confidence"]), "key"
    return None, "miss"


def score_checker_log(log_rows: list[dict], resolution_pos: int,
                      gold_action: str, staged_id: str,
                      window: int = SENSITIVITY_WINDOW) -> dict:
    """§5.1 NAT-RES scoring over one pair's checker log. Hit = valid fire with
    the authored gold action on the resolution turn (primary) / within the
    +2-turn sensitivity window; every other fire is a false fire."""
    fires = [(row["horizon_pos"], f)
             for row in log_rows for f in row.get("fires", [])]

    def matched(pos: int, fire: dict, w: int) -> bool:
        return (not fire["invalid"] and fire["action"] == gold_action
                and fire["staged_id"] == staged_id
                and resolution_pos <= pos <= resolution_pos + w)

    hit_primary = any(matched(p, f, 0) for p, f in fires)
    hit_sensitivity = any(matched(p, f, window) for p, f in fires)
    false_fires = [{"horizon_pos": p, "staged_id": f["staged_id"],
                    "action": f["action"], "invalid": f["invalid"]}
                   for p, f in fires if not matched(p, f, window)]
    lag = None
    if hit_sensitivity:
        lag = min(p for p, f in fires if matched(p, f, window)) - resolution_pos
    return {"hit_primary": hit_primary, "hit_sensitivity": hit_sensitivity,
            "firing_lag": lag, "n_fires": len(fires),
            "false_fires": false_fires}


# ---------------------------------------------------------------------------
# The storeless pass (LLM stages 1-4)
# ---------------------------------------------------------------------------

def _tuples_as_dicts(tuples: list) -> list[dict]:
    return [dataclasses.asdict(t) for t in tuples]


def _packaged_item(pair_id: str, family: str, stream: dict, tup: dict,
                   stored_value: Optional[str], suffix: str = "") -> dict:
    """Item-packaging fields consumed by label.format_item_prompt (v1 §2.1):
    stored_value=None is the rule-1.5d no-stored-value branch (STORELESS)."""
    return {
        "item_id": f"natres-{pair_id}-{family}{suffix}",
        "entity": tup["entity"],
        "attribute": tup["attribute"],
        "asserted_value": tup["value"],
        "stored_value": stored_value,
        "turn_text": stream["stage_turn"]["text"],
        "preceding_turns": [{"role": t["role"], "content": t["content"]}
                            for t in stream["prior_turns"]],
    }


async def run_pair_family(item: dict, family: str, extractor_system: str,
                          gen_system: str, checker_system: str,
                          *, max_checker_turns: Optional[int] = None) -> dict:
    """Stages 1-5 of the module docstring for one (pair, family)."""
    pair, key = item["pair"], item["key"]
    pair_id = pair["pair_id"]
    stream = item["stream"]
    record: dict[str, Any] = {
        "pair_id": pair_id, "family": family,
        "question_id": pair["source_question_id"],
        "gold_action": pair["gold_action"],
        "resolution_horizon_pos": stream["resolution_horizon_pos"],
        "horizon_user_turns": len(stream["horizon_turns"]),
        "quality": pair["quality"], "flags": pair["flags"],
        "resolution_strength": pair["resolution_strength"],
    }

    # 1 — extraction at the stage turn (text prefix input; no store replay)
    user_prompt = extract.render_extraction_prompt(
        stream["prior_turns"], stream["stage_turn"]["text"],
        stream["stage_turn"]["date"])
    raw = await llm.call_llm(extractor_system, user_prompt,
                             trace_tag=f"natres-extract-{pair_id}-{family}")
    tuples = _tuples_as_dicts(extract.parse_tuples(raw))
    tup, match_method = match_staged_tuple(tuples, pair["stage"]["text"], key)
    record["extraction"] = {"n_tuples": len(tuples), "match_method": match_method}
    if tup is None:
        record.update(extraction_miss=True, staged=False,
                      score={"hit_primary": False, "hit_sensitivity": False,
                             "firing_lag": None, "n_fires": 0, "false_fires": []},
                      outcome_reason="extraction_miss")
        return record
    record["staged_tuple"] = tup

    # 2 — classifier route (stored_value=None: rule 1.5d no-stored-value branch)
    cls = await classify.classify_item(
        _packaged_item(pair_id, family, stream, tup, None))
    staged = cls["argmax"] == "speculative"
    record["classifier"] = {"argmax": cls["argmax"], "s_ev": cls["s_ev"],
                            "max_prob": cls["max_prob"]}
    record["staged"] = staged
    if pair_id in DUAL_LABEL_PAIRS:
        _, _, prior_quote = parse_prior_assertion(key)
        alt = await classify.classify_item(
            _packaged_item(pair_id, family, stream, tup, prior_quote,
                           suffix="-storedbranch"))
        record["classifier_stored_branch_sensitivity"] = {
            "stored_value_gold_sourced": prior_quote,
            "argmax": alt["argmax"], "s_ev": alt["s_ev"],
            "diverges_from_primary": alt["argmax"] != cls["argmax"],
            "note": "V2-F3 dual-branch; primary route stays the "
                    "no-stored-value branch",
        }
    if not staged:
        record.update(score={"hit_primary": False, "hit_sensitivity": False,
                             "firing_lag": None, "n_fires": 0, "false_fires": []},
                      outcome_reason=f"not_staged_argmax_{cls['argmax']}")
        return record

    # 3 — predicate generation from the staged tuple + the stage-turn text
    # prefix (B6: nothing after the stage turn enters the call)
    prefix = [{"date": t["date"], "role": t["role"], "content": t["content"]}
              for t in stream["prior_turns"]] + [
        {"date": stream["stage_turn"]["date"], "role": "user",
         "content": stream["stage_turn"]["text"]}]
    gen_prompt = test3.build_generator_user_prompt(
        prefix, tup["entity"], tup["attribute"], tup["value"])
    out = await llm.call_llm(gen_system, gen_prompt,
                             trace_tag=f"natres-gen-{pair_id}-{family}")
    promote_when, invalidate_when = out["promote_when"], out["invalidate_when"]
    aliases = test3.attribute_aliases(tup["attribute"])
    failures = test3.lint_predicates(promote_when, invalidate_when,
                                     tup["value"], aliases)
    retried = False
    if failures:  # §3.3: exactly one regeneration with lint feedback
        retried = True
        out = await llm.call_llm(
            gen_system, gen_prompt + test3.lint_feedback_block(failures),
            trace_tag=f"natres-gen-retry-{pair_id}-{family}")
        promote_when, invalidate_when = out["promote_when"], out["invalidate_when"]
        failures = test3.lint_predicates(promote_when, invalidate_when,
                                         tup["value"], aliases)
    staged_id = f"natres_{pair_id}"
    pending_item = {"staged_id": staged_id, "entity": tup["entity"],
                    "attribute": tup["attribute"],
                    "hypothesized_value": tup["value"],
                    "promote_when": promote_when,
                    "invalidate_when": invalidate_when}
    record["predicates"] = {**pending_item,
                            "lint": {"retried": retried,
                                     "final_failures": failures}}

    # 4 — checker over the frozen horizon; fire consumes; remaining skipped
    horizon = stream["horizon_turns"]
    if max_checker_turns is not None:
        horizon = horizon[:max_checker_turns]
    pending = [pending_item]
    log_rows = []
    for turn in horizon:
        row = {"horizon_pos": turn["horizon_pos"],
               "session_idx": turn["session_idx"], "turn_idx": turn["turn_idx"],
               "turn_date": turn["date"]}
        if not pending:
            log_rows.append({**row, "skipped": True, "fires": []})
            continue
        output = await llm.call_llm(
            checker_system,
            test3.build_checker_user_prompt(turn["text"], turn["date"], pending),
            trace_tag=f"natres-check-{pair_id}-{family}-{turn['horizon_pos']}")
        fires, malformed, extra = test3.parse_checker_fires(
            output, {staged_id}, turn["text"])
        if any(f["staged_id"] == staged_id for f in fires):
            pending = []
        log_rows.append({**row, "skipped": False, "fires": fires,
                         "malformed_output": malformed, "extra_keys": extra})
    record["checker_log"] = log_rows
    record["checker_calls"] = sum(not r["skipped"] for r in log_rows)

    # 5 — gold-action scoring
    record["score"] = score_checker_log(
        log_rows, stream["resolution_horizon_pos"], pair["gold_action"], staged_id)
    record["outcome_reason"] = "scored"
    return record


def prepare_lme_items() -> list[dict]:
    """Load + verify the 17 pairs and build their streams (no LLM)."""
    import eval_classifier as test1_eval  # s_cleaned pathway (reuse; v1 §2.1-B3 information set)

    items = []
    for entry in load_lme_pairs():
        substrate = test1_eval.s_cleaned_substrate(
            entry["pair"]["source_question_id"])
        entry["stream"] = build_pair_stream(entry["pair"], entry["key"], substrate)
        items.append(entry)
    total = sum(len(e["stream"]["horizon_turns"]) for e in items)
    assert total == TOTAL_LME_HORIZON, \
        f"summed horizons {total} != {TOTAL_LME_HORIZON}"
    return items


async def run_lme(families: tuple[str, ...] = ("em", "ez")) -> list[dict]:
    items = prepare_lme_items()
    gen_system = test3.GENERATOR_PROMPT_PATH.read_text()
    checker_system = test3.CHECKER_PROMPT_PATH.read_text()
    records = []
    for family in families:
        extractor_system = (PROMPTS_DIR / FAMILY_PROMPTS[family]).read_text()
        results = await asyncio.gather(*[
            run_pair_family(item, family, extractor_system, gen_system,
                            checker_system)
            for item in items])
        records.extend(results)
    NATRES_DIR.mkdir(parents=True, exist_ok=True)
    with LME_RUNS_PATH.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"ran {len(records)} (pair × family) storeless passes -> {LME_RUNS_PATH}")
    return records


# ---------------------------------------------------------------------------
# Pair-row labeling (v1 §2.2 pipeline, STORELESS; dual-branch for the two
# prior_value_before_stage pairs)
# ---------------------------------------------------------------------------

async def _label_row(pair_id: str, role: str, stream_like: dict, tup: dict,
                     stored_value: Optional[str], suffix: str = "") -> dict:
    item = {
        "item_id": f"natres-pair-{pair_id}-{role}{suffix}",
        "entity": tup["entity"], "attribute": tup["attribute"],
        "asserted_value": tup["value"], "stored_value": stored_value,
        "turn_text": stream_like["turn_text"],
        "preceding_turns": stream_like["preceding_turns"],
        # bookkeeping (never shown to labelers; format_item_prompt ignores it)
        "question_id": stream_like["question_id"],
        "session_idx": stream_like["session_idx"],
        "turn_idx": stream_like["turn_idx"],
        "source_file": "s_cleaned",
    }
    res = await label.label_item(item)
    return {"item": item, "labeling": res}


async def label_pairs() -> None:
    """Label the 34 LME pair rows under rule 1.5d's no-stored-value branch;
    dual-branch P-LME-08/-14 (V2-F3). E-M (frozen extractor.txt) tuples anchor
    the packaged items — labels are arm-independent; the stage-turn extraction
    reuses the run's cache, resolution turns get their own storeless call."""
    import eval_classifier as test1_eval

    extractor_system = (PROMPTS_DIR / FAMILY_PROMPTS["em"]).read_text()
    rows_out, items_for_queue, results_for_queue = [], [], []
    for entry in load_lme_pairs():
        pair, key = entry["pair"], entry["key"]
        pair_id = pair["pair_id"]
        substrate = test1_eval.s_cleaned_substrate(pair["source_question_id"])
        dual = pair_id in DUAL_LABEL_PAIRS
        prior_quote = parse_prior_assertion(key)[2] if dual else None
        for role in ("stage", "resolution"):
            anchor = pair[role]
            sidx, tidx = anchor["session_idx"], anchor["turn_idx"]
            prior = [dict(t, date=substrate.dates[sidx])
                     for t in label.preceding_turns(substrate, sidx, tidx)]
            turn_text = substrate.sessions[sidx][tidx]["content"]
            raw = await llm.call_llm(
                extractor_system,
                extract.render_extraction_prompt(prior, turn_text,
                                                 substrate.dates[sidx]),
                trace_tag=f"natres-extract-{pair_id}-em"
                if role == "stage" else f"natres-label-extract-{pair_id}-{role}")
            tuples = _tuples_as_dicts(extract.parse_tuples(raw))
            tup, method = match_staged_tuple(tuples, anchor["text"], key)
            base = {"pair_id": pair_id, "pair_role": role,
                    "question_id": pair["source_question_id"],
                    "match_method": method}
            if tup is None:
                rows_out.append({**base, "extraction_miss": True})
                continue
            stream_like = {
                "turn_text": turn_text,
                "preceding_turns": [{"role": t["role"], "content": t["content"]}
                                    for t in prior],
                "question_id": pair["source_question_id"],
                "session_idx": sidx, "turn_idx": tidx,
            }
            primary = await _label_row(pair_id, role, stream_like, tup, None)
            row = {**base, "extraction_miss": False,
                   "primary_no_stored_value_branch": primary["labeling"],
                   "item_id": primary["item"]["item_id"]}
            items_for_queue.append(primary["item"])
            results_for_queue.append(primary["labeling"])
            if dual:
                sens = await _label_row(pair_id, role, stream_like, tup,
                                        prior_quote, suffix="-storedbranch")
                row["stored_value_branch_sensitivity"] = {
                    "stored_value_gold_sourced": prior_quote,
                    "labeling": sens["labeling"],
                    "diverges_from_primary": (sens["labeling"]["vote"]
                                              != primary["labeling"]["vote"]),
                    "note": "V2-F3: primary label remains the no-stored-value "
                            "branch; divergence reported",
                }
                items_for_queue.append(sens["item"])
                results_for_queue.append(sens["labeling"])
            rows_out.append(row)
    NATRES_DIR.mkdir(parents=True, exist_ok=True)
    with PAIR_LABELS_PATH.open("w") as fh:
        for r in rows_out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_queued = label.append_escalations(items_for_queue, results_for_queue,
                                        queue_path=ESCALATION_QUEUE_PATH)
    print(f"labeled {len(rows_out)} pair rows -> {PAIR_LABELS_PATH} "
          f"({n_queued} escalations queued -> {ESCALATION_QUEUE_PATH})")


# ---------------------------------------------------------------------------
# Scoring / aggregation (§5.1 NAT-RES via metrics_v2)
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def aggregate(records: list[dict], locomo_rows: list[dict],
              *, survived: Optional[int] = None) -> dict:
    """NAT-RES per family over the frozen 35-pair denominator (pooled across
    benchmarks BY NECESSITY, flagged — the only FM-22 exception); strata
    reported. LoCoMo rows follow LOCOMO_TRACE_CONTRACT."""
    pairs_by_corpus = {"lme": 17, "locomo": 18}
    out: dict[str, Any] = {
        "endpoint": "NAT-RES",
        "denominator_frozen": N_PAIRS_FROZEN,
        "pooled_across_benchmarks": "by necessity, flagged (§5.1); strata reported",
        "supplementary_arcs_never_in_denominator": ["42ec0761", "c6853660"],
        "per_family": {},
    }
    for family in ("em", "ez"):
        fam_lme = [r for r in records if r["family"] == family]
        fam_loco = [r for r in locomo_rows if r.get("family") == family]
        hits_p = (sum(r["score"]["hit_primary"] for r in fam_lme)
                  + sum(bool(r.get("hit_primary")) for r in fam_loco))
        hits_s = (sum(r["score"]["hit_sensitivity"] for r in fam_lme)
                  + sum(bool(r.get("hit_sensitivity")) for r in fam_loco))
        n_run = len(fam_lme) + len(fam_loco)
        surv = survived if survived is not None else n_run
        strata = {
            "lme": {"n_run": len(fam_lme), "n_frozen": pairs_by_corpus["lme"],
                    "hit_primary": sum(r["score"]["hit_primary"] for r in fam_lme),
                    "hit_sensitivity": sum(r["score"]["hit_sensitivity"]
                                           for r in fam_lme)},
            "locomo": {"n_run": len(fam_loco), "n_frozen": pairs_by_corpus["locomo"],
                       "hit_primary": sum(bool(r.get("hit_primary")) for r in fam_loco),
                       "hit_sensitivity": sum(bool(r.get("hit_sensitivity"))
                                              for r in fam_loco),
                       "status": "in-stream (§7 row 7)" if fam_loco
                                 else "PENDING gated LoCoMo replay"},
        }
        by_action: dict[str, dict] = {}
        for r in fam_lme:
            b = by_action.setdefault(r["gold_action"], {"n": 0, "hit_sensitivity": 0})
            b["n"] += 1
            b["hit_sensitivity"] += int(r["score"]["hit_sensitivity"])
        out["per_family"][family] = {
            "rule_primary_resolution_turn": metrics_v2.natres_rule(
                hits_p, N_PAIRS_FROZEN, survived=surv),
            "rule_sensitivity_plus2": metrics_v2.natres_rule(
                hits_s, N_PAIRS_FROZEN, survived=surv),
            "n_run": n_run,
            "survived": surv,
            "strata": strata,
            "lme_by_gold_action": by_action,
            "lme_not_staged": sorted(r["pair_id"] for r in fam_lme
                                     if not r.get("staged")),
            "lme_extraction_misses": sorted(r["pair_id"] for r in fam_lme
                                            if r.get("extraction_miss")),
            "lme_false_fires_total": sum(len(r["score"]["false_fires"])
                                         for r in fam_lme),
            "lme_dual_branch_divergence": {
                r["pair_id"]: r["classifier_stored_branch_sensitivity"][
                    "diverges_from_primary"]
                for r in fam_lme
                if "classifier_stored_branch_sensitivity" in r},
        }
    return out


def score() -> dict:
    records = _load_jsonl(LME_RUNS_PATH)
    locomo_rows = _load_jsonl(LOCOMO_TRACE_PATH)
    if not records:
        raise SystemExit(f"missing {LME_RUNS_PATH}: run `python natres.py run` first")
    labels = _load_jsonl(PAIR_LABELS_PATH)
    survived = None
    if labels:
        # a pair survives labeling when neither of its rows dropped (FM-19);
        # LoCoMo pair survival is folded in once its labels ship
        dropped = {r["pair_id"] for r in labels if r.get("extraction_miss")}
        survived = 18 + (17 - len(dropped))  # LoCoMo half counted until its labels land
    results = {
        **aggregate(records, locomo_rows, survived=survived),
        "provenance": {
            "seed": SEED, "model": llm.MODEL,
            "checker_horizon_total_lme": TOTAL_LME_HORIZON,
            "prompts_sha256": {
                name: hashlib.sha256((PROMPTS_DIR / f).read_bytes()).hexdigest()
                for name, f in (("extractor_em", "extractor.txt"),
                                ("extractor_ez", "extractor_zep.txt"),
                                ("classifier", "classifier_short.txt"),
                                ("predicate_generator", "predicate_generator.txt"),
                                ("checker", "checker.txt"))},
            "locomo_trace_contract": LOCOMO_TRACE_CONTRACT,
        },
    }
    NATRES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    for family, block in results["per_family"].items():
        rule = block["rule_primary_resolution_turn"]
        print(f"{family}: primary {rule['x']}/{rule['n']} "
              f"pass={rule['pass']} (status {rule['status']}); "
              f"locomo {block['strata']['locomo']['status']}")
    print(f"wrote {RESULTS_PATH}")
    return results


# ---------------------------------------------------------------------------
# Smoke: no-LLM dry run over the frozen pairs + one SYNTHETIC pipeline pass
# ---------------------------------------------------------------------------

SMOKE_STAGE_TURN = (
    "Also — I've been thinking about signing up for the community pottery "
    "class at the arts centre this autumn; still checking whether the "
    "Tuesday sessions fit my work schedule, so nothing booked yet.")
SMOKE_RESOLUTION_TURN = (
    "Good news on the pottery front: I went ahead and enrolled in the "
    "community pottery class at the arts centre yesterday — Tuesday evenings, "
    "starting next month.")


async def smoke() -> None:
    """(a) No-LLM dry run: build all 17 frozen pair streams (asserts anchors +
    horizons; prints the 632 total). (b) SYNTHETIC storeless pipeline pass —
    extraction -> match -> classify -> predicate gen -> 1 checker turn —
    on invented content so no preregistered item is touched pre-gate
    (<= 5 LLM calls, cf. test3.smoke)."""
    items = prepare_lme_items()
    total = sum(len(e["stream"]["horizon_turns"]) for e in items)
    print(f"dry run: {len(items)} LME pairs verified; horizon total {total} "
          f"user turns/family (authored {TOTAL_LME_HORIZON})")
    for e in items[:3]:
        s = e["stream"]
        print(f"  {s['pair_id']}: horizon {len(s['horizon_turns'])}, "
              f"resolution at pos {s['resolution_horizon_pos']}, "
              f"gold {s['gold_action']}, prior context {len(s['prior_turns'])} turns")

    synth_pair = {
        "pair_id": "P-SMOKE-00", "source_question_id": "smoke",
        "gold_action": "promote", "quality": "clear", "flags": [],
        "resolution_strength": "full",
        "stage": {"text": SMOKE_STAGE_TURN},
    }
    synth_key = {"pair_id": "P-SMOKE-00", "entity": "user",
                 "attribute": "pottery_class_enrollment"}
    stream = {
        "pair_id": "P-SMOKE-00",
        "stage_turn": {"text": SMOKE_STAGE_TURN,
                       "date": "2023/09/03 (Sun) 10:12"},
        "prior_turns": [],
        "horizon_turns": [{"horizon_pos": 0, "session_idx": 0, "turn_idx": 0,
                           "text": SMOKE_RESOLUTION_TURN,
                           "date": "2023/09/20 (Wed) 19:05"}],
        "resolution_horizon_pos": 0,
        "gold_action": "promote",
    }
    record = await run_pair_family(
        {"pair": synth_pair, "key": synth_key, "stream": stream},
        "em",
        (PROMPTS_DIR / FAMILY_PROMPTS["em"]).read_text(),
        test3.GENERATOR_PROMPT_PATH.read_text(),
        test3.CHECKER_PROMPT_PATH.read_text(),
        max_checker_turns=1,
    )
    printable = {k: v for k, v in record.items() if k != "checker_log"}
    print("=== synthetic storeless pass ===")
    print(json.dumps(printable, ensure_ascii=False, indent=1))
    print(f"score: {json.dumps(record['score'], ensure_ascii=False)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NAT-RES storeless runner (§4, V2-F3)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("--family", choices=["em", "ez", "both"], default="both")
    sub.add_parser("label-pairs")
    sub.add_parser("score")
    sub.add_parser("smoke")
    args = parser.parse_args()
    if args.cmd == "run":
        families = ("em", "ez") if args.family == "both" else (args.family,)
        asyncio.run(run_lme(families))
    elif args.cmd == "label-pairs":
        asyncio.run(label_pairs())
    elif args.cmd == "score":
        score()
    elif args.cmd == "smoke":
        asyncio.run(smoke())


if __name__ == "__main__":
    main()
