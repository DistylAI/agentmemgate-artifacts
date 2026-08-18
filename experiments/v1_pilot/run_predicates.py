"""Test 3 — resolution predicate generation and firing (owner: I5).

Implements predicate_spec.md end to end, under the pinned pipeline order (§3.2, B6):

    stage (splice t3_staging asides; injector-owned rendering)
    -> naturalness-gate hook (assert_naturalness_gate_passed: flag file must exist)
    -> generate_predicates (input = replayed prefix through the staging turn ONLY)
    -> groundedness lint (§3.3; one retry with feedback, logged)
    -> freeze manifest ../results/test3/predicates_manifest.jsonl
       (incl. staging_turn_sha256 + call_llm cache keys)
    -> [orchestrator commits the manifest; 15-item hand audit happens outside python]
    -> attach_variants (asserts manifest git-committed AND staging hashes match the
       replayed prefix) + 6-gram leak scan (§3.2 item 4)
    -> run_checker (one call per user turn of each variant replay; |pending| = 1 per M2,
       asserted; no cross-turn memory)
    -> score (protocol §3.4 / predicate_spec §4; cluster bootstrap by case per P3/B5;
       TTC lag feed per P5) -> ../results/test3/results.json
    -> failure-audit queue writer (§6 inputs; bucket assignment happens externally)

CLI (run from pilot/ with .venv/bin/python):
    python test3.py generate   # gate hook -> predicates -> lint -> freeze manifest
    python test3.py check      # attach_variants asserts -> leak scan -> run checker
    python test3.py score      # score + results.json + failure_audit_queue.jsonl
    python test3.py smoke      # tiny end-to-end smoke test (<= 3 real LLM calls)

Cross-module imports (data, inject, stats, llm) are function-local so this module's
pure-logic functions stay importable/testable before those owners' modules land.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import _bootstrap  # noqa: F401  (R1 layout: puts src/ + experiments/ on sys.path)

from paths import PROMPTS_DIR, RESULTS_DIR
GENERATOR_PROMPT_PATH = PROMPTS_DIR / "predicate_generator.txt"
CHECKER_PROMPT_PATH = PROMPTS_DIR / "checker.txt"

TEST3_DIR = RESULTS_DIR / "test3"
INJECTION_MANIFEST_PATH = RESULTS_DIR / "injections" / "manifest.jsonl"
PREDICATES_MANIFEST_PATH = TEST3_DIR / "predicates_manifest.jsonl"
CHECKER_LOG_PATH = TEST3_DIR / "checker_log.jsonl"
VARIANTS_REPLAYED_PATH = TEST3_DIR / "variants_replayed.jsonl"
LEAK_SCAN_PATH = TEST3_DIR / "leak_scan.json"
RESULTS_PATH = TEST3_DIR / "results.json"
AUDIT_QUEUE_PATH = TEST3_DIR / "failure_audit_queue.jsonl"
# B6 hook: the Test 0 owner writes this flag once the injector §10 naturalness gate has
# passed for the Test 3 staging stratum (including any regeneration rounds). Staging text
# is immutable from that point on; generate_predicates refuses to run without the flag.
GATE_FLAG_PATH = RESULTS_DIR / "test0" / "naturalness_gate_passed.flag"

SEED = 20260702
SENSITIVITY_WINDOW = 2  # matched fires at t*, t*+1, t*+2 (user turns) — §4.3
MIN_CONDITION_WORDS = 15  # §3.3 crude vagueness floor
AUDIT_QUEUE_CAP = 60  # §6.2: audit a stratified random 60 if failures exceed 60
# §3.3 G4 keyword family — any one suffices.
SUSPENSION_KEYWORDS = ("on hold", "suspend", "shelv", "indefinite", "no longer plan", "foreseeable")
VARIANT_KINDS = ("promote", "invalidate", "distractor_a", "distractor_b")
GOLD_ACTION_BY_KIND = {
    "promote": "promote",
    "invalidate": "invalidate",
    "distractor_a": "none",
    "distractor_b": "none",
}
CERTAINTY_VOCAB = ("deliberating", "intended", "planned", "scheduled")
# §3.4: the two real staging turns used in predicate_spec §5 worked examples are excluded
# from the scored case set, keyed by (question_id, session_idx, turn_idx) — ID-anchored (P9).
WORKED_EXAMPLE_HOLDOUT = {("gpt4_76048e76", 1, 10), ("gpt4_e05b82a6", 7, 0)}

# §6.1 bucket definitions + attribution question, shipped verbatim with every audit-queue
# row so the external (Claude-agent) audit calls are self-contained per §6.2.
BUCKET_DEFINITIONS = {
    "F1": "miss-because-vague-predicate: gold fire missed, and the frozen predicate "
          "condition — read literally but charitably by a competent reader with no other "
          "context — does NOT cover the gold turn's event (the predicate under-scoped or "
          "under-specified the resolution).",
    "F2": "miss-because-checker: gold fire missed (no fire, wrong action, or wrong "
          "staged_id), and the predicate condition DOES cover the gold turn's event — the "
          "checker failed to apply a sufficient predicate.",
    "F3": "false-fire-because-predicate-overbroad: a fire on a non-gold (turn, action), "
          "and the predicate condition, read literally, DOES license the fired action on "
          "that turn (the predicate was written too broadly).",
    "F4": "false-fire-because-checker: a fire on a non-gold (turn, action), and the "
          "predicate condition does NOT license it — the checker fired beyond the predicate.",
}
ATTRIBUTION_QUESTION = (
    "Does the predicate text, interpreted literally but charitably by a competent reader "
    "given only the predicate and this turn, license the gold action (for misses) / the "
    "fired action (for false fires) on this turn?"
)


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def _cache_key(system_prompt: str, user_prompt: str) -> str:
    """Recompute the llm.py P10 cache key for a call at the pinned model/default effort.

    Mirrors llm._cache_key (INTERFACES.md exposes only call_llm, so the formula is
    restated here); recomputing it from the archived prefix + tuple is the §3.2(2)
    prompt-content proof that no future content entered the generation call.
    """
    import llm

    payload = json.dumps([llm.MODEL, llm.REASONING_EFFORT, system_prompt, user_prompt],
                         ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


_DATE_RE = re.compile(r"(\d{4})/(\d{1,2})/(\d{1,2})")


def days_between(date_a: str, date_b: str) -> int | None:
    """Calendar days between two session datelines like '2023/06/02 (Fri) 10:35'."""
    import datetime

    ma, mb = _DATE_RE.search(date_a or ""), _DATE_RE.search(date_b or "")
    if not ma or not mb:
        return None
    da = datetime.date(int(ma.group(1)), int(ma.group(2)), int(ma.group(3)))
    db = datetime.date(int(mb.group(1)), int(mb.group(2)), int(mb.group(3)))
    return (db - da).days


# ---------------------------------------------------------------------------
# Prefix slicing and replay walking (§3.2 item 3 — code-enforced temporal firewall)
# ---------------------------------------------------------------------------

def _walk_turns(substrate) -> list[dict]:
    """All turns of a Substrate in chronological order, tagged with ORIGINAL session_idx.

    Substrate contract (data.py / INTERFACES.md): ``.sessions`` and ``.dates`` keep
    the ORIGINAL haystack order (never assume it is chronological — P12);
    ``.session_order`` lists the original indices in chronological order and is
    the iteration order. Content and dates are therefore indexed by ``orig_idx``.
    """
    rows = []
    for orig_idx in substrate.session_order:
        date = substrate.dates[orig_idx]
        for turn_idx, turn in enumerate(substrate.sessions[orig_idx]):
            rows.append({
                "session_idx": orig_idx,
                "turn_idx": turn_idx,
                "date": date,
                "role": turn["role"],
                "content": turn["content"],
            })
    return rows


def slice_prefix(substrate, staging_session_idx: int, staging_turn_idx: int) -> list[dict]:
    """Transcript prefix up to AND INCLUDING the staging turn (both roles, with dates).

    The last returned row is asserted to be the staging turn; nothing after it is
    included (predicate_spec §3.2 item 3 — the live leakage channel at generation time).
    """
    prefix = []
    for row in _walk_turns(substrate):
        prefix.append(row)
        if row["session_idx"] == staging_session_idx and row["turn_idx"] == staging_turn_idx:
            assert prefix[-1]["session_idx"] == staging_session_idx
            assert prefix[-1]["turn_idx"] == staging_turn_idx
            return prefix
    raise ValueError(
        f"staging turn (session {staging_session_idx}, turn {staging_turn_idx}) not found")


def turns_after(substrate, staging_session_idx: int, staging_turn_idx: int) -> list[dict]:
    """All turns (both roles) strictly after the staging turn, chronological."""
    rows = _walk_turns(substrate)
    for i, row in enumerate(rows):
        if row["session_idx"] == staging_session_idx and row["turn_idx"] == staging_turn_idx:
            return rows[i + 1:]
    raise ValueError(
        f"staging turn (session {staging_session_idx}, turn {staging_turn_idx}) not found")


def continuation_user_turns(substrate, staging_session_idx: int, staging_turn_idx: int) -> list[dict]:
    """User turns strictly after the staging turn — the checked turns of a replay (§2.1)."""
    out = []
    for row in turns_after(substrate, staging_session_idx, staging_turn_idx):
        if row["role"] == "user":
            out.append({**row, "pos": len(out)})
    return out


def find_turn_pos(cont: list[dict], session_idx: int, turn_idx: int) -> int:
    """Position of the designed turn t* within the continuation user turns."""
    for row in cont:
        if row["session_idx"] == session_idx and row["turn_idx"] == turn_idx:
            return row["pos"]
    raise ValueError(f"designed turn (session {session_idx}, turn {turn_idx}) not in continuation")


# ---------------------------------------------------------------------------
# Groundedness lint (§3.3) — pure logic
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.lower()))


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text))


def attribute_aliases(attribute: str) -> list[str]:
    """Surface referents of a snake_case attribute: the full phrase + content words >= 4 chars."""
    phrase = attribute.replace("_", " ").strip().lower()
    aliases = [phrase]
    aliases.extend(tok for tok in phrase.split() if len(tok) >= 4)
    return aliases


def mentions_referent(condition: str, hypothesized_value: str, aliases: list[str]) -> bool:
    """Normalized word-boundary substring match against value + per-attribute aliases."""
    hay = f" {_normalize(condition)} "
    candidates = [_normalize(hypothesized_value)]
    candidates.extend(t for t in _normalize(hypothesized_value).split() if len(t) >= 4)
    candidates.extend(_normalize(a) for a in aliases)
    return any(c and f" {c} " in hay for c in candidates)


def lint_predicates(promote_when: str, invalidate_when: str,
                    hypothesized_value: str, aliases: list[str]) -> list[str]:
    """§3.3 checks. Returns failure messages (empty list = pass)."""
    failures = []
    for name, cond in (("promote_when", promote_when), ("invalidate_when", invalidate_when)):
        if not mentions_referent(cond, hypothesized_value, aliases):
            failures.append(f"{name}: does not mention the hypothesized value or the "
                            f"attribute's surface referent (G1)")
        if _word_count(cond) < MIN_CONDITION_WORDS:
            failures.append(f"{name}: fewer than {MIN_CONDITION_WORDS} words (vagueness floor)")
    low = invalidate_when.lower()
    if not any(k in low for k in SUSPENSION_KEYWORDS):
        failures.append("invalidate_when: does not cover indefinite suspension (G4); "
                        f"keyword family: {list(SUSPENSION_KEYWORDS)}")
    return failures


def lint_feedback_block(failures: list[str]) -> str:
    lines = "\n".join(f"- {f}" for f in failures)
    return ("\n\nLINT FEEDBACK — your previous output failed these groundedness checks:\n"
            f"{lines}\nRegenerate the full JSON object, fixing every listed issue while "
            "keeping all four groundedness requirements satisfied.")


# ---------------------------------------------------------------------------
# 6-gram leak scan (§3.2 item 4) — pure python
# ---------------------------------------------------------------------------

def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _ngrams(tokens: list[str], n: int = 6) -> set[tuple[str, ...]]:
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def find_six_gram_overlaps(condition: str, texts: list[str]) -> list[tuple[str, ...]]:
    """6-grams (lowercased word tokens) of `condition` that occur in any of `texts`."""
    cond = _ngrams(_tokens(condition))
    if not cond:
        return []
    hits: set[tuple[str, ...]] = set()
    for text in texts:
        hits |= cond & _ngrams(_tokens(text))
    return sorted(hits)


# ---------------------------------------------------------------------------
# Checker output parsing (§2.3) — pure logic
# ---------------------------------------------------------------------------

def parse_checker_fires(output, pending_ids: set[str], turn_text: str):
    """Validate one checker call's output.

    Returns (fires, malformed, extra_keys). Each fire dict carries span_valid and
    invalid/invalid_reason flags. Per §2.3: a non-verbatim span still counts as a fire
    (span_valid=False, flagged); double action for one staged_id or a fire on an unknown
    staged_id → invalid fires that can never be hits (scored as false fires).
    """
    if not isinstance(output, dict) or not isinstance(output.get("fires"), list):
        return [], True, []
    extra_keys = sorted(set(output) - {"fires"})
    fires = []
    for entry in output["fires"]:
        if not isinstance(entry, dict):
            fires.append({"staged_id": None, "action": None, "evidence_span": None,
                          "span_valid": False, "invalid": True,
                          "invalid_reason": "malformed_entry"})
            continue
        staged_id = entry.get("staged_id")
        action = entry.get("action")
        span = entry.get("evidence_span")
        invalid_reason = None
        if staged_id not in pending_ids:
            invalid_reason = "unknown_id"
        elif action not in ("promote", "invalidate"):
            invalid_reason = "bad_action"
        fires.append({
            "staged_id": staged_id,
            "action": action,
            "evidence_span": span,
            "span_valid": isinstance(span, str) and span in turn_text,
            "invalid": invalid_reason is not None,
            "invalid_reason": invalid_reason,
        })
    # Both actions emitted for the same staged_id in one call → all its fires invalid (§2.3).
    by_id: dict[str, set] = {}
    for f in fires:
        if f["staged_id"] in pending_ids:
            by_id.setdefault(f["staged_id"], set()).add(f["action"])
    for f in fires:
        if f["staged_id"] in by_id and len(by_id[f["staged_id"]]) > 1 and not f["invalid"]:
            f["invalid"] = True
            f["invalid_reason"] = "double_action"
    return fires, False, extra_keys


# ---------------------------------------------------------------------------
# Scoring semantics (protocol §3.4 / predicate_spec §4.2–§4.4) — pure logic
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FireEvent:
    turn_pos: int  # index into the variant's continuation user turns
    staged_id: str | None
    action: str | None
    invalid: bool = False


@dataclass(frozen=True)
class VariantGold:
    case_id: str
    variant_kind: str  # promote | invalidate | distractor_a | distractor_b
    gold_action: str  # promote | invalidate | none
    gold_staged_id: str
    t_star_pos: int
    invalidate_style: str | None = None  # "soft" | "hard" for invalidate variants (§5.4)


@dataclass(frozen=True)
class VariantScore:
    gold: VariantGold
    hit_primary: bool
    hit_sensitivity: bool
    firing_lag: int | None  # in {0,1,2} for sensitivity-window hits (P5 TTC feed)
    fires: tuple[FireEvent, ...]
    false_fires: tuple[FireEvent, ...]

    @property
    def fired(self) -> bool:
        return bool(self.fires)


def fire_matched(gold: VariantGold, fire: FireEvent, window: int) -> bool:
    """§4.2 'Match': action == gold action AND case-matched staged item, at t*..t*+window."""
    return (not fire.invalid
            and gold.gold_action in ("promote", "invalidate")
            and fire.action == gold.gold_action
            and fire.staged_id == gold.gold_staged_id
            and gold.t_star_pos <= fire.turn_pos <= gold.t_star_pos + window)


def score_variant(gold: VariantGold, fires: list[FireEvent]) -> VariantScore:
    """Classify one variant replay's fires per the §4.2 table.

    Every fire that is not a sensitivity-window match is a false fire: any fire on a
    distractor variant, wrong action, wrong staged_id, filler-turn fires before t*,
    fires beyond t*+2, and invalid outputs (never hits). Promote-fire on the
    continued-speculation distractor is explicitly a false fire (gold_action = none).
    """
    hit_primary = any(fire_matched(gold, f, 0) for f in fires)
    hit_sensitivity = any(fire_matched(gold, f, SENSITIVITY_WINDOW) for f in fires)
    firing_lag = None
    if hit_sensitivity:
        firing_lag = min(f.turn_pos for f in fires
                         if fire_matched(gold, f, SENSITIVITY_WINDOW)) - gold.t_star_pos
    false_fires = tuple(f for f in fires if not fire_matched(gold, f, SENSITIVITY_WINDOW))
    return VariantScore(gold=gold, hit_primary=hit_primary, hit_sensitivity=hit_sensitivity,
                        firing_lag=firing_lag, fires=tuple(fires), false_fires=false_fires)


def _recall_block(scores: list[VariantScore]) -> dict:
    return {
        "n": len(scores),
        "x_primary": sum(s.hit_primary for s in scores),
        "x_sensitivity": sum(s.hit_sensitivity for s in scores),
        "successes_primary": [int(s.hit_primary) for s in scores],
        "successes_sensitivity": [int(s.hit_sensitivity) for s in scores],
        "clusters": [s.gold.case_id for s in scores],
        "lags": [s.firing_lag for s in scores if s.firing_lag is not None],
        "censored": sum(not s.hit_sensitivity for s in scores),
    }


def aggregate_counts(scores: list[VariantScore]) -> dict:
    """Raw counts + per-probe success/cluster vectors for CI computation (no stats import)."""
    counts: dict = {}
    for action in ("promote", "invalidate"):
        variants = [s for s in scores if s.gold.gold_action == action]
        counts[action] = _recall_block(variants)
        if action == "invalidate":
            counts[action]["by_style"] = {
                style: _recall_block([s for s in variants if s.gold.invalidate_style == style])
                for style in ("soft", "hard")
            }

    precision: dict = {}
    for action in ("promote", "invalidate"):
        emitted = [(s, f) for s in scores for f in s.fires if f.action == action]
        precision[action] = {
            "n_fires": len(emitted),
            "x_primary": sum(fire_matched(s.gold, f, 0) for s, f in emitted),
            "x_sensitivity": sum(fire_matched(s.gold, f, SENSITIVITY_WINDOW) for s, f in emitted),
            "successes_primary": [int(fire_matched(s.gold, f, 0)) for s, f in emitted],
            "successes_sensitivity": [int(fire_matched(s.gold, f, SENSITIVITY_WINDOW))
                                      for s, f in emitted],
            "clusters": [s.gold.case_id for s, f in emitted],
            # case-level: cases with >=1 matched fire / cases with >=1 fire of this action
            "cases_with_fire": len({s.gold.case_id for s, f in emitted}),
            "cases_with_matched_fire": len({s.gold.case_id for s, f in emitted
                                            if fire_matched(s.gold, f, SENSITIVITY_WINDOW)}),
        }
    counts["precision"] = precision

    distractors = [s for s in scores if s.gold.gold_action == "none"]
    probe_successes = [int(s.fired) for s in distractors]  # any fire on any checked turn (§4.3)
    case_ids = sorted({s.gold.case_id for s in distractors})
    case_fired = [int(any(s.fired for s in distractors if s.gold.case_id == c)) for c in case_ids]
    counts["false_fire"] = {
        "probe": {"n": len(distractors), "x": sum(probe_successes),
                  "successes": probe_successes,
                  "clusters": [s.gold.case_id for s in distractors]},
        "case": {"n": len(case_ids), "x": sum(case_fired),
                 "successes": case_fired, "clusters": case_ids},
    }
    counts["total_false_fire_events"] = sum(len(s.false_fires) for s in scores)
    return counts


# ---------------------------------------------------------------------------
# Failure-audit queue (§6) — inputs only; bucket assignment happens externally
# ---------------------------------------------------------------------------

def sample_audit_queue(records: list[dict], cap: int = AUDIT_QUEUE_CAP, seed: int = SEED):
    """§6.2: if failures exceed `cap`, audit a random `cap` stratified by failure type."""
    meta = {"total_failures": len(records), "queued": min(len(records), cap),
            "sampled": len(records) > cap, "cap": cap, "seed": seed}
    if len(records) <= cap:
        return list(records), meta
    rng = random.Random(seed)
    strata: dict[str, list[dict]] = {}
    for r in records:
        strata.setdefault(r["failure_type"], []).append(r)
    # proportional allocation, largest remainder, deterministic tie-break by stratum name
    quotas = {k: len(v) * cap / len(records) for k, v in strata.items()}
    alloc = {k: int(q) for k, q in quotas.items()}
    for k in sorted(quotas, key=lambda k: (-(quotas[k] - alloc[k]), k)):
        if sum(alloc.values()) >= cap:
            break
        alloc[k] += 1
    sampled = []
    for k in sorted(strata):
        pool = sorted(strata[k], key=lambda r: r["audit_id"])
        sampled.extend(rng.sample(pool, alloc[k]))
    sampled.sort(key=lambda r: r["audit_id"])
    return sampled, meta


def build_audit_records(score: VariantScore, log_rows: list[dict], pending_item: dict) -> list[dict]:
    """§6.2 item 2: everything each external audit call receives, per failure.

    Misses use §4.2's 'no matched fire anywhere in its continuation' reading (a
    sensitivity-window hit is late, not missed); each false fire is its own record.
    """
    rows_by_pos = {r["turn_pos"]: r for r in log_rows}
    records = []
    gold = score.gold

    def base(failure_type: str, seq: int, turn_row: dict, fired: dict | None) -> dict:
        return {
            "audit_id": f"t3fa_{gold.case_id}_{gold.variant_kind}_{failure_type}_{seq:02d}",
            "case_id": gold.case_id,
            "variant_kind": gold.variant_kind,
            "invalidate_style": gold.invalidate_style,
            "failure_type": failure_type,
            "pending_item": pending_item,  # exactly the six fields the checker saw (§2.2)
            "turn": {"turn_pos": turn_row["turn_pos"], "turn_text": turn_row["turn_text"],
                     "turn_date": turn_row["turn_date"]},
            "gold": {"t_star_pos": gold.t_star_pos, "action": gold.gold_action,
                     "staged_id": gold.gold_staged_id},
            "gold_action_for_this_turn": (gold.gold_action
                                          if turn_row["turn_pos"] == gold.t_star_pos
                                          else "none"),
            "checker_output": turn_row.get("output"),
            "fired": fired,
            "bucket_definitions": BUCKET_DEFINITIONS,
            "attribution_question": ATTRIBUTION_QUESTION,
        }

    if gold.gold_action in ("promote", "invalidate") and not score.hit_sensitivity:
        t_star_row = rows_by_pos[gold.t_star_pos]
        records.append(base("miss", 0, t_star_row, None))

    for seq, fire in enumerate(score.false_fires, start=1):
        turn_row = rows_by_pos[fire.turn_pos]
        fired = next((f for f in turn_row.get("fires", [])
                      if f["staged_id"] == fire.staged_id and f["action"] == fire.action), None)
        records.append(base("false_fire", seq, turn_row, fired))
    return records


# ---------------------------------------------------------------------------
# Pipeline hooks and asserts
# ---------------------------------------------------------------------------

def assert_naturalness_gate_passed(flag_path: Path = GATE_FLAG_PATH) -> None:
    """B6 naturalness-gate hook: staging fragments must have passed the injector §10 gate
    (including any regeneration rounds) BEFORE generate_predicates runs."""
    assert flag_path.exists(), (
        f"naturalness gate flag missing: {flag_path} — Test 3 staging fragments must pass "
        "the injector §10 naturalness gate (staging strata, incl. regeneration rounds) "
        "before predicate generation (predicate_spec §3.2 B6)")


def assert_predicates_manifest_committed(path: Path = PREDICATES_MANIFEST_PATH) -> str:
    """attach_variants precondition: manifest frozen = committed and unmodified (§3.2 item 1)."""
    assert path.exists(), f"predicates manifest missing: {path}"
    log = subprocess.run(["git", "log", "-1", "--format=%H", "--", str(path)],
                         cwd=path.parent, capture_output=True, text=True, check=True)
    commit = log.stdout.strip()
    assert commit, ("predicates manifest is not git-committed — commit it before "
                    "attach_variants (predicate_spec §3.2, INTERFACES.md)")
    diff = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(path)], cwd=path.parent)
    assert diff.returncode == 0, "predicates manifest has uncommitted modifications"
    return commit


def load_t3_cases(rows: list[dict], require_variants: bool) -> dict[str, dict]:
    """Group injection-manifest rows into Test 3 cases {case_id: {staging, variants}}."""
    cases: dict[str, dict] = {}
    for row in rows:
        if row["inj_type"] == "t3_staging":
            case_id = row["t3_case_id"]
            assert case_id not in cases or "staging" not in cases[case_id], \
                f"duplicate t3_staging row for case {case_id}"
            cases.setdefault(case_id, {"variants": {}})["staging"] = row
    for row in rows:
        if row["inj_type"] == "t3_variant":
            case_id = row["t3_case_id"]
            assert case_id in cases, f"t3_variant row for unknown case {case_id}"
            kind = row["t3_variant_kind"]
            assert kind in VARIANT_KINDS, f"unknown t3_variant_kind {kind!r}"
            assert kind not in cases[case_id]["variants"], \
                f"duplicate {kind} variant for case {case_id}"
            cases[case_id]["variants"][kind] = row
    for case_id, case in cases.items():
        staging = case["staging"]
        anchor = (staging["question_id"], staging["placement"]["session_idx"],
                  staging["placement"]["turn_idx"])
        assert anchor not in WORKED_EXAMPLE_HOLDOUT, \
            f"case {case_id} stages a predicate_spec §5 worked-example turn {anchor} (ID holdout)"
        if require_variants:
            assert set(case["variants"]) == set(VARIANT_KINDS), \
                f"case {case_id} has variants {sorted(case['variants'])}, need all of {VARIANT_KINDS}"
    return cases


def invalidate_style_of(variant_row: dict) -> str | None:
    """§5.4: soft-abandonment invalidate variants tagged via edge_case_tags."""
    if variant_row["t3_variant_kind"] != "invalidate":
        return None
    tags = variant_row.get("edge_case_tags") or []
    return "soft" if any("soft" in str(t).lower() for t in tags) else "hard"


def checker_pending_item(manifest_row: dict) -> dict:
    """The six fields — and only those — the checker sees per pending item (§2.2)."""
    return {k: manifest_row[k] for k in
            ("staged_id", "entity", "attribute", "hypothesized_value",
             "promote_when", "invalidate_when")}


def build_generator_user_prompt(prefix: list[dict], entity: str, attribute: str,
                                hypothesized_value: str) -> str:
    """Deterministic function of (prefix, tuple) only — the §3.2(2) cache-key proof relies
    on this determinism."""
    lines = "\n".join(f"[{t['date']}] {t['role']}: {t['content']}" for t in prefix)
    return ("STAGED TUPLE\n"
            f"entity: {entity}\n"
            f"attribute: {attribute}\n"
            f"hypothesized_value: {hypothesized_value}\n\n"
            "TRANSCRIPT PREFIX (chronological; the final turn is the staging turn)\n"
            f"{lines}")


def build_checker_user_prompt(turn_text: str, turn_date: str, pending_items: list[dict]) -> str:
    return json.dumps({"turn_text": turn_text, "turn_date": turn_date,
                       "pending_items": pending_items}, ensure_ascii=False, indent=1)


# ---------------------------------------------------------------------------
# Stage: generate_predicates -> lint -> freeze manifest
# ---------------------------------------------------------------------------

async def _generate_case(staging_row: dict, gen_system: str, get_substrate, build_perturbed) -> dict:
    from llm import call_llm

    case_id = staging_row["t3_case_id"]
    qid = staging_row["question_id"]
    sess = staging_row["placement"]["session_idx"]
    tix = staging_row["placement"]["turn_idx"]
    entity, attribute = staging_row["entity"], staging_row["attribute"]
    hypothesized_value = staging_row["value"]

    # Clean copy of the substrate with ONLY the staging aside spliced (injector §9.1),
    # sliced through the staging turn — code-enforced prefix-only access (FM-8).
    substrate = build_perturbed(get_substrate(qid), [staging_row])
    prefix = slice_prefix(substrate, sess, tix)
    staging_text = prefix[-1]["content"]
    assert staging_row["text"] in staging_text, \
        f"case {case_id}: rendered staging aside not found in the replayed staging turn"

    user_prompt = build_generator_user_prompt(prefix, entity, attribute, hypothesized_value)
    cache_keys = [_cache_key(gen_system, user_prompt)]
    out = await call_llm(gen_system, user_prompt, trace_tag=f"t3_gen_{case_id}")

    aliases = attribute_aliases(attribute)

    def _fields(o: dict) -> tuple[str, str, dict]:
        promote, invalidate = o["promote_when"], o["invalidate_when"]
        assert isinstance(promote, str) and promote, f"case {case_id}: empty promote_when"
        assert isinstance(invalidate, str) and invalidate, f"case {case_id}: empty invalidate_when"
        return promote, invalidate, o.get("certainty_meta") or {}

    promote_when, invalidate_when, certainty_meta = _fields(out)
    first_failures = lint_predicates(promote_when, invalidate_when, hypothesized_value, aliases)
    final_failures, retried = first_failures, False
    if first_failures:
        # §3.3: exactly one regeneration with the lint feedback appended; the second
        # output is frozen as-is with the lint result logged.
        retried = True
        retry_prompt = user_prompt + lint_feedback_block(first_failures)
        cache_keys.append(_cache_key(gen_system, retry_prompt))
        out = await call_llm(gen_system, retry_prompt, trace_tag=f"t3_gen_retry_{case_id}")
        promote_when, invalidate_when, certainty_meta = _fields(out)
        final_failures = lint_predicates(promote_when, invalidate_when, hypothesized_value, aliases)

    stated = certainty_meta.get("stated_certainty")
    return {
        "staged_id": f"stg_{case_id}_01",
        "case_id": case_id,
        "question_id": qid,
        "entity": entity,
        "attribute": attribute,
        "hypothesized_value": hypothesized_value,
        "staged_at_turn": {"question_id": qid, "session_idx": sess, "turn_idx": tix,
                           "date": prefix[-1]["date"]},
        "promote_when": promote_when,
        "invalidate_when": invalidate_when,
        "certainty_meta": {
            "stated_certainty": stated,
            "stated_certainty_in_vocab": stated in CERTAINTY_VOCAB,
            "horizon_hint": certainty_meta.get("horizon_hint"),
            "secondhand": certainty_meta.get("secondhand"),
        },
        "staging_turn_sha256": _sha256(staging_text),
        "generator_cache_keys": cache_keys,
        "generator_prompt_sha256": _sha256(gen_system),
        "lint": {"first_failures": first_failures, "retried": retried,
                 "final_failures": final_failures},
        "seed": SEED,
    }


async def generate() -> list[dict]:
    """Pipeline: staging rows -> B6 gate hook -> generate_predicates -> lint -> freeze manifest."""
    import data
    import inject

    rows = _load_jsonl(INJECTION_MANIFEST_PATH)
    cases = load_t3_cases(rows, require_variants=False)
    assert cases, "no t3_staging rows in the injection manifest"
    # B6: naturalness gate (staging strata) must have passed BEFORE generate_predicates.
    assert_naturalness_gate_passed()

    gen_system = GENERATOR_PROMPT_PATH.read_text()
    manifest_rows = await asyncio.gather(*[
        _generate_case(cases[case_id]["staging"], gen_system,
                       data.get_substrate, inject.build_perturbed_substrate)
        for case_id in sorted(cases)
    ])
    manifest_rows = list(manifest_rows)
    _dump_jsonl(PREDICATES_MANIFEST_PATH, manifest_rows)
    lint_failed = [r["case_id"] for r in manifest_rows if r["lint"]["final_failures"]]
    print(f"froze {len(manifest_rows)} staged items -> {PREDICATES_MANIFEST_PATH}")
    print(f"lint: {sum(r['lint']['retried'] for r in manifest_rows)} retried, "
          f"{len(lint_failed)} frozen with residual failures {lint_failed}")
    print("NEXT: commit the manifest (orchestrator), run the 15-item hand audit (§3.4), "
          "then `python test3.py check`.")
    return manifest_rows


# ---------------------------------------------------------------------------
# attach_variants -> leak scan -> run_checker
# ---------------------------------------------------------------------------

async def _run_variant_replay(case_id: str, kind: str, pred_row: dict, cont: list[dict],
                              checker_system: str) -> list[dict]:
    """One checker call per user turn (§2.1); no cross-turn memory; harness keeps pending."""
    from llm import call_llm

    pending = [checker_pending_item(pred_row)]
    log_rows = []
    for turn in cont:
        row = {"case_id": case_id, "variant_kind": kind, "turn_pos": turn["pos"],
               "session_idx": turn["session_idx"], "turn_idx": turn["turn_idx"],
               "turn_text": turn["content"], "turn_date": turn["date"]}
        if not pending:
            log_rows.append({**row, "skipped": True, "output": None, "fires": [],
                             "malformed_output": False, "extra_keys": []})
            continue
        # M2: |pending| = 1 for every injected case by construction — asserted.
        assert len(pending) == 1, \
            f"M2 violation: |pending|={len(pending)} for case {case_id} {kind} turn {turn['pos']}"
        pending_ids = {p["staged_id"] for p in pending}
        output = await call_llm(
            checker_system,
            build_checker_user_prompt(turn["content"], turn["date"], pending),
            trace_tag=f"t3_check_{case_id}_{kind}_{turn['pos']}",
        )
        fires, malformed, extra_keys = parse_checker_fires(output, pending_ids, turn["content"])
        consumed = {f["staged_id"] for f in fires if f["staged_id"] in pending_ids}
        pending = [p for p in pending if p["staged_id"] not in consumed]
        log_rows.append({**row, "skipped": False, "output": output, "fires": fires,
                         "malformed_output": malformed, "extra_keys": extra_keys})
    return log_rows


def run_leak_scan(pred_rows: dict[str, dict], cases: dict[str, dict], get_substrate) -> dict:
    """§3.2 item 4: scan every condition for 6-gram overlap with (a) original-transcript
    turns after the staging turn and (b) all variant continuation texts. Flags are
    reported for manual review, never auto-failed."""
    all_variant_texts = [v["text"] for case in cases.values() for v in case["variants"].values()]
    flags = []
    for case_id in sorted(pred_rows):
        pred = pred_rows[case_id]
        staging = cases[case_id]["staging"]
        original = get_substrate(pred["question_id"])
        future_texts = [t["content"] for t in turns_after(
            original, staging["placement"]["session_idx"], staging["placement"]["turn_idx"])]
        for field in ("promote_when", "invalidate_when"):
            for source, texts in (("original_future_turns", future_texts),
                                  ("variant_texts", all_variant_texts)):
                for ngram in find_six_gram_overlaps(pred[field], texts):
                    flags.append({"case_id": case_id, "staged_id": pred["staged_id"],
                                  "field": field, "source": source, "ngram": " ".join(ngram)})
    return {"n_conditions_scanned": 2 * len(pred_rows), "n_flags": len(flags), "flags": flags,
            "note": "flag list is reported for manual review, not auto-failed (§3.2 item 4)"}


async def check() -> None:
    """attach_variants (freeze asserts) -> leak scan -> run_checker over all replays."""
    import data
    import inject

    manifest_commit = assert_predicates_manifest_committed()
    pred_rows = {r["case_id"]: r for r in _load_jsonl(PREDICATES_MANIFEST_PATH)}
    cases = load_t3_cases(_load_jsonl(INJECTION_MANIFEST_PATH), require_variants=True)
    assert set(pred_rows) == set(cases), \
        "predicates manifest and injection manifest disagree on the Test 3 case set"

    leak = run_leak_scan(pred_rows, cases, data.get_substrate)
    LEAK_SCAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEAK_SCAN_PATH.write_text(json.dumps(leak, ensure_ascii=False, indent=1))
    print(f"leak scan: {leak['n_flags']} flagged 6-grams -> {LEAK_SCAN_PATH}")

    checker_system = CHECKER_PROMPT_PATH.read_text()
    replay_meta, coros = [], []
    for case_id in sorted(cases):
        case, pred = cases[case_id], pred_rows[case_id]
        staging = case["staging"]
        s_sess = staging["placement"]["session_idx"]
        s_tix = staging["placement"]["turn_idx"]
        for kind in VARIANT_KINDS:
            vrow = case["variants"][kind]
            replay = inject.build_perturbed_substrate(
                data.get_substrate(staging["question_id"]), [staging, vrow])
            prefix = slice_prefix(replay, s_sess, s_tix)
            # B6: the replay builder asserts the staging hash before any checker call.
            assert _sha256(prefix[-1]["content"]) == pred["staging_turn_sha256"], \
                f"case {case_id} {kind}: staging turn hash mismatch vs frozen manifest (B6)"
            cont = continuation_user_turns(replay, s_sess, s_tix)
            t_star_pos = find_turn_pos(cont, vrow["placement"]["session_idx"],
                                       vrow["placement"]["turn_idx"])
            # §4.1: >= 2 further user turns after t* so the sensitivity window exists.
            assert len(cont) > t_star_pos + SENSITIVITY_WINDOW, \
                f"case {case_id} {kind}: continuation too short for the sensitivity window"
            assert vrow["text"] in cont[t_star_pos]["content"], \
                f"case {case_id} {kind}: variant text not found in designed turn t*"
            replay_meta.append({
                "case_id": case_id, "variant_kind": kind,
                "question_id": staging["question_id"], "staged_id": pred["staged_id"],
                "gold_action": GOLD_ACTION_BY_KIND[kind],
                "invalidate_style": invalidate_style_of(vrow),
                "t_star_pos": t_star_pos,
                "t_star_session_idx": vrow["placement"]["session_idx"],
                "t_star_turn_idx": vrow["placement"]["turn_idx"],
                "n_continuation_user_turns": len(cont),
                "staging_date": prefix[-1]["date"],
                "t_star_date": cont[t_star_pos]["date"],
                # descriptive "staged duration" (P5) — never added into TTC
                "staged_duration_user_turns": t_star_pos + 1,
                "staged_duration_days": days_between(prefix[-1]["date"], cont[t_star_pos]["date"]),
                "manifest_commit": manifest_commit,
            })
            coros.append(_run_variant_replay(case_id, kind, pred, cont, checker_system))

    all_logs = await asyncio.gather(*coros)
    checker_log = [row for rows in all_logs for row in rows]
    _dump_jsonl(VARIANTS_REPLAYED_PATH, replay_meta)
    _dump_jsonl(CHECKER_LOG_PATH, checker_log)
    n_calls = sum(not r["skipped"] for r in checker_log)
    print(f"ran checker on {len(replay_meta)} replays ({n_calls} calls, "
          f"{len(checker_log) - n_calls} turns skipped after consumption)")
    print(f"-> {VARIANTS_REPLAYED_PATH}\n-> {CHECKER_LOG_PATH}")


# ---------------------------------------------------------------------------
# Score (§4.2–§4.4, protocol §3.4/§4.1 B5) + failure-audit queue writer (§6)
# ---------------------------------------------------------------------------

def _rate_entry(x: int, n: int, successes: list[int], clusters: list[str]) -> dict:
    """Every rate carries x, n, Wilson CI, and cluster-bootstrap-by-case CI (B5/FM-15)."""
    from stats import cluster_bootstrap_rate, wilson_ci

    entry: dict = {"x": x, "n": n, "rate": (x / n) if n else None}
    entry["wilson_ci"] = list(wilson_ci(x, n)) if n else None
    entry["cluster_bootstrap"] = (
        cluster_bootstrap_rate(successes, clusters, n_boot=10000, seed=SEED) if n else None)
    return entry


def build_results(counts: dict, diagnostics: dict, provenance: dict) -> dict:
    def rate(block: dict, key: str) -> dict:
        return _rate_entry(block[f"x_{key}"], block["n"], block[f"successes_{key}"],
                           block["clusters"])

    promote, invalidate = counts["promote"], counts["invalidate"]
    prec = counts["precision"]
    ff = counts["false_fire"]

    promotion_recall_primary = rate(promote, "primary")
    false_fire_probe = _rate_entry(ff["probe"]["x"], ff["probe"]["n"],
                                   ff["probe"]["successes"], ff["probe"]["clusters"])

    def precision_block(action: str) -> dict:
        b = prec[action]
        return {
            "primary": _rate_entry(b["x_primary"], b["n_fires"], b["successes_primary"],
                                   b["clusters"]),
            "sensitivity": _rate_entry(b["x_sensitivity"], b["n_fires"],
                                       b["successes_sensitivity"], b["clusters"]),
            "case_level": {"x": b["cases_with_matched_fire"], "n": b["cases_with_fire"],
                           "rate": (b["cases_with_matched_fire"] / b["cases_with_fire"])
                                   if b["cases_with_fire"] else None,
                           "definition": "cases with >=1 matched fire / cases with >=1 "
                                         f"{action}-fire (sensitivity window)"},
        }

    def recall_section(block: dict) -> dict:
        return {
            "primary": rate(block, "primary"),
            "sensitivity": rate(block, "sensitivity"),
            "case_level_note": "1 probe per case: probe level == case level (§4.3)",
        }

    lags = promote["lags"]
    cluster_lower = (promotion_recall_primary["cluster_bootstrap"] or {}).get("lo")
    cluster_upper = (false_fire_probe["cluster_bootstrap"] or {}).get("hi")
    promotion_point = promotion_recall_primary["rate"]
    false_fire_point = false_fire_probe["rate"]

    return {
        "test": "test3_resolution_predicates",
        "n_cases": len(set(promote["clusters"]) | set(invalidate["clusters"])
                       | set(ff["case"]["clusters"])),
        "n_probes": promote["n"] + invalidate["n"] + ff["probe"]["n"],
        "promotion_recall": recall_section(promote),
        "invalidation_recall": {
            **recall_section(invalidate),
            "by_style": {style: recall_section(invalidate["by_style"][style])
                         for style in ("soft", "hard")},
        },
        "promotion_precision": precision_block("promote"),
        "invalidation_precision": precision_block("invalidate"),
        "false_fire_rate": {"probe_level": false_fire_probe,
                            "case_level": _rate_entry(ff["case"]["x"], ff["case"]["n"],
                                                      ff["case"]["successes"],
                                                      ff["case"]["clusters"])},
        "decisions": {
            "promotion_recall": {
                "rule": "point >= 0.8 AND cluster-bootstrap-by-case lower >= 0.65 "
                        "(protocol §4.1/B5, P3; primary window)",
                "point": promotion_point, "cluster_lower": cluster_lower,
                "pass": (promotion_point is not None and cluster_lower is not None
                         and promotion_point >= 0.8 and cluster_lower >= 0.65),
            },
            "false_fire": {
                "rule": "point <= 0.05 AND cluster-bootstrap-by-case upper <= 0.075 "
                        "(protocol §4.1/B5, P3; probe level, Wilson as reference)",
                "point": false_fire_point, "cluster_upper": cluster_upper,
                "pass": (false_fire_point is not None and cluster_upper is not None
                         and false_fire_point <= 0.05 and cluster_upper <= 0.075),
            },
        },
        "ttc_feed_p5": {
            "firing_lags_user_turns": lags,
            "lag_distribution": {str(k): lags.count(k) for k in (0, 1, 2)},
            "right_censored": {"x": promote["censored"], "n": promote["n"],
                               "fraction": (promote["censored"] / promote["n"])
                                           if promote["n"] else None},
            "censoring_bound": 0.2,  # <= 1 - promotion-recall target (P5)
            "censoring_within_bound": (promote["n"] > 0
                                       and promote["censored"] / promote["n"] <= 0.2),
            "note": "gated-arm TTC for staged-then-resolved items = firing lag (P5); "
                    "staged duration is descriptive and reported in variants_replayed.jsonl, "
                    "never added into TTC",
        },
        "total_false_fire_events": counts["total_false_fire_events"],
        "diagnostics": diagnostics,
        "provenance": provenance,
    }


def score() -> dict:
    replays = _load_jsonl(VARIANTS_REPLAYED_PATH)
    checker_log = _load_jsonl(CHECKER_LOG_PATH)
    pred_rows = {r["case_id"]: r for r in _load_jsonl(PREDICATES_MANIFEST_PATH)}

    log_by_variant: dict[tuple[str, str], list[dict]] = {}
    for row in checker_log:
        log_by_variant.setdefault((row["case_id"], row["variant_kind"]), []).append(row)
    for rows in log_by_variant.values():
        rows.sort(key=lambda r: r["turn_pos"])

    scores, audit_records = [], []
    for meta in replays:
        gold = VariantGold(case_id=meta["case_id"], variant_kind=meta["variant_kind"],
                           gold_action=meta["gold_action"], gold_staged_id=meta["staged_id"],
                           t_star_pos=meta["t_star_pos"],
                           invalidate_style=meta["invalidate_style"])
        log_rows = log_by_variant[(meta["case_id"], meta["variant_kind"])]
        fires = [FireEvent(turn_pos=row["turn_pos"], staged_id=f["staged_id"],
                           action=f["action"], invalid=f["invalid"])
                 for row in log_rows for f in row["fires"]]
        vscore = score_variant(gold, fires)
        scores.append(vscore)
        audit_records.extend(build_audit_records(
            vscore, log_rows, checker_pending_item(pred_rows[meta["case_id"]])))

    counts = aggregate_counts(scores)

    all_fires = [f for row in checker_log for f in row["fires"]]
    invalid_reasons: dict[str, int] = {}
    for f in all_fires:
        if f["invalid"]:
            invalid_reasons[f["invalid_reason"]] = invalid_reasons.get(f["invalid_reason"], 0) + 1
    leak = json.loads(LEAK_SCAN_PATH.read_text()) if LEAK_SCAN_PATH.exists() else None
    diagnostics = {
        "checker_calls": sum(not r["skipped"] for r in checker_log),
        "turns_skipped_after_consumption": sum(r["skipped"] for r in checker_log),
        "malformed_outputs": sum(r["malformed_output"] for r in checker_log),
        "outputs_with_extra_keys": sum(bool(r["extra_keys"]) for r in checker_log),
        "fires_total": len(all_fires),
        "fires_span_invalid": sum(not f["span_valid"] for f in all_fires),
        "invalid_fires_by_reason": invalid_reasons,
        "leak_scan_flags": leak["n_flags"] if leak else None,
        "lint_residual_failures": sorted(c for c, r in pred_rows.items()
                                         if r["lint"]["final_failures"]),
    }
    provenance = {
        "seed": SEED,
        "predicates_manifest_commit": replays[0]["manifest_commit"] if replays else None,
        "generator_prompt_sha256": _sha256(GENERATOR_PROMPT_PATH.read_text()),
        "checker_prompt_sha256": _sha256(CHECKER_PROMPT_PATH.read_text()),
    }

    results = build_results(counts, diagnostics, provenance)
    queued, audit_meta = sample_audit_queue(audit_records)
    results["failure_audit_queue"] = {**audit_meta, "path": str(AUDIT_QUEUE_PATH),
                                      "note": "bucket assignment happens externally "
                                              "(predicate_spec §6.2; Claude agents)"}

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    _dump_jsonl(AUDIT_QUEUE_PATH, queued)
    print(f"scored {len(scores)} probes -> {RESULTS_PATH}")
    print(f"failure audit queue: {audit_meta['queued']}/{audit_meta['total_failures']} "
          f"records -> {AUDIT_QUEUE_PATH}")
    dec = results["decisions"]
    print(f"promotion recall: {dec['promotion_recall']['point']} "
          f"(cluster lower {dec['promotion_recall']['cluster_lower']}) "
          f"pass={dec['promotion_recall']['pass']}")
    print(f"false-fire: {dec['false_fire']['point']} "
          f"(cluster upper {dec['false_fire']['cluster_upper']}) pass={dec['false_fire']['pass']}")
    return results


# ---------------------------------------------------------------------------
# Smoke test (<= 3 real LLM calls, tiny)
# ---------------------------------------------------------------------------

async def smoke() -> None:
    """Generate predicates for ONE synthetic staged item and run the checker on one
    synthetic resolution turn. Prints both outputs."""
    from llm import call_llm

    gen_system = GENERATOR_PROMPT_PATH.read_text()
    checker_system = CHECKER_PROMPT_PATH.read_text()

    prefix = [
        {"date": "2023/05/14 (Sun) 09:02", "role": "user",
         "content": "Can you suggest a couple of podcasts for my commute? By the way, I've "
                     "been thinking about switching my cell phone plan over to Meteor Mobile "
                     "once my current contract wraps up — still comparing coverage maps, "
                     "nothing decided yet."},
        {"date": "2023/05/14 (Sun) 09:02", "role": "assistant",
         "content": "Happy to help with podcasts! For a commute, try a mix of news briefs "
                     "and longer interview shows."},
        {"date": "2023/05/14 (Sun) 09:02", "role": "user",
         "content": "Thanks, I'll queue those up. Like I said, the Meteor Mobile switch is "
                     "still just an idea I'm weighing."},
    ]
    entity, attribute, value = "user", "cell_phone_carrier", "Meteor Mobile"
    user_prompt = build_generator_user_prompt(prefix, entity, attribute, value)
    out = await call_llm(gen_system, user_prompt, trace_tag="t3_smoke_gen")
    print("=== generator output ===")
    print(json.dumps(out, ensure_ascii=False, indent=1))

    promote_when, invalidate_when = out["promote_when"], out["invalidate_when"]
    failures = lint_predicates(promote_when, invalidate_when, value, attribute_aliases(attribute))
    print(f"lint failures: {failures or 'none'}")
    if failures:
        retry_prompt = user_prompt + lint_feedback_block(failures)
        out = await call_llm(gen_system, retry_prompt, trace_tag="t3_smoke_gen_retry")
        promote_when, invalidate_when = out["promote_when"], out["invalidate_when"]
        print("=== generator retry output ===")
        print(json.dumps(out, ensure_ascii=False, indent=1))
        print(f"lint failures after retry: "
              f"{lint_predicates(promote_when, invalidate_when, value, attribute_aliases(attribute)) or 'none'}")

    pending = [{"staged_id": "stg_smoke_01", "entity": entity, "attribute": attribute,
                "hypothesized_value": value, "promote_when": promote_when,
                "invalidate_when": invalidate_when}]
    turn_text = ("Quick update on the phone thing — I pulled the trigger and moved my "
                 "number over to Meteor Mobile on Tuesday; coverage at the house has been "
                 "great so far.")
    checker_out = await call_llm(
        checker_system,
        build_checker_user_prompt(turn_text, "2023/08/15 (Tue) 18:40", pending),
        trace_tag="t3_smoke_check",
    )
    print("=== checker output (synthetic promote turn) ===")
    print(json.dumps(checker_out, ensure_ascii=False, indent=1))
    fires, malformed, extra = parse_checker_fires(checker_out, {"stg_smoke_01"}, turn_text)
    print(f"parsed fires: {json.dumps(fires, ensure_ascii=False)} "
          f"(malformed={malformed}, extra_keys={extra})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test 3 — resolution predicates")
    parser.add_argument("stage", choices=["generate", "check", "score", "smoke"])
    args = parser.parse_args()
    if args.stage == "generate":
        asyncio.run(generate())
    elif args.stage == "check":
        asyncio.run(check())
    elif args.stage == "score":
        score()
    elif args.stage == "smoke":
        asyncio.run(smoke())


if __name__ == "__main__":
    main()
