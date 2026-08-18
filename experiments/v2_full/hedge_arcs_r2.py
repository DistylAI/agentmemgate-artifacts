"""hedge_arcs_r2.py — v2.4 M4 round-2 TEXT re-render under the corpus-derived
style contract (protocol/CHANGELOG.md, 2026-07-08 entries "M4 owner review
round 1: WHOLESALE REJECT" + "M4 r2 style contract FROZEN").

Scope: the arc PLAN is untouched — same arc ids, hosts, keys, VALUES,
placements, session/turn indices, positions, outcome labels, hedge strengths,
pending windows, seeds (results/v2.4/m4/arc_plan.jsonl + the geometry fields
of the retired arcs_r1.jsonl). ONLY the two rendered sentences per arc change,
through the frozen single-call prompt protocol/prompts/injector_render_m4r2.txt
plus the deterministic code lints below.

Deterministic per-arc directives (seed 20260708 + arc index, recorded per arc):
  * target_words sampled from HOST user-sentence stats (LME lognormal median 15
    sigma 0.42 clamp 5-30; LoCoMo median 8 sigma 0.45 clamp 4-20), floored at
    value+opener words + 4;
  * opener drawn from a per-benchmark distribution (LME "By the way," ~1/3);
  * reason quota: exactly 15% of sentences MAY carry one plain because-reason;
  * batch-level noise quotas (host-conditional, max ONE feature per sentence):
    LME exclamation 13% (abandoned resolutions ineligible), lowercase_start 6%
    (non-bare openers only), missing-apostrophe typo ~2%, no_terminal_punct 9%
    of turn_end-position sentences only (the only splice seam where a missing
    period is visible).

Lint battery (retry <= 3 with lint_feedback in the retry payload):
  * the 32 selective banned regexes from register_analysis_hosts.json VERBATIM;
  * lintable anti-patterns from register_analysis_external.json;
  * ASCII-only punctuation, no semicolon, no question, one sentence;
  * body-comma cap (opener comma exempt), word band target +/-4;
  * opener + noise-directive conformance;
  * category cue checks (hedge realizes exactly its directed strength under the
    frozen HEDGE_CUE_FAMILIES detection; resolution carries its outcome's cue
    and none of the others; LoCoMo abandonment must be setback-framed);
  * no 6-gram shared with any payload exemplar (exemplars are REAL corpus user
    sentences, payload-level only — no reference-corpus text enters stimuli);
  * still-applicable r1 checks: value verbatim exactly once, FM-4 probe
    re-assert, T3-invalidate value-inversion guard, second-person ban, template
    BLACKLIST (ROUND2_TELL_BLACKLIST deliberately NOT applied — "By the way,"
    is a directed corpus-frequent opener), relative-time lint (hardened).

Budget: every fresh call recorded under SharedCallBudget stage "m4"
(ADVISORY stages / hard 120k total envelope, 2026-07-08 amendment).

CLI (repo root, pilot/.venv/bin/python experiments/v2_full/hedge_arcs_r2.py ...):
  r2-render   [--limit K] [--fresh-cap N]   arcs_r1.jsonl plan + values ->
                                            arcs.jsonl (round-2 text) +
                                            r2_render_summary.json
  review-pack                               rebuild review_pack.md (+ template)
                                            via hedge_arcs.run_review_pack
  register-check                            hosts-analysis metrics over the 120
                                            r2 sentences vs host user vs r1 ->
                                            r2_register_check.json
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import re
import statistics
import sys

import _bootstrap  # noqa: F401

import data
import hedge_arcs as ha
import llm
from inject import (
    BLACKLIST,
    RELATIVE_TIME_PATTERNS,
    _norm,
)

from paths import PROMPTS_DIR

SEED_M4 = ha.SEED_M4                     # 20260708 — same seed family as the plan
M4_DIR = ha.M4_DIR
STYLE_PROMPT_PATH = PROMPTS_DIR / "injector_render_m4r2.txt"
ARCS_R1_PATH = M4_DIR / "arcs_r1.jsonl"
ARCS_PATH = ha.ARCS_PATH                 # results/v2.4/m4/arcs.jsonl (r2 output)
HOSTS_ANALYSIS_PATH = M4_DIR / "register_analysis_hosts.json"
EXTERNAL_ANALYSIS_PATH = M4_DIR / "register_analysis_external.json"
R2_SUMMARY_PATH = M4_DIR / "r2_render_summary.json"
R2_CHECK_PATH = M4_DIR / "r2_register_check.json"

MAX_ATTEMPTS = 4                         # 1 initial + <=3 lint-feedback retries
WORD_TOL = 4

# ---------------------------------------------------------------------------
# Host-conditional style knobs (register_analysis_{hosts,external}; the LoCoMo
# branch is contract-complete but unexercised: the M4 host pool is LME-only)
# ---------------------------------------------------------------------------

LENGTH_PARAMS = {                        # (median, sigma, lo, hi)
    "lme": (15, 0.42, 5, 30),
    "locomo": (8, 0.45, 4, 20),
}
OPENERS = ("By the way,", "Also,", "Oh,", "none")
OPENER_WEIGHTS = {
    "lme": {"By the way,": 0.33, "Also,": 0.10, "Oh,": 0.04, "none": 0.53},
    "locomo": {"By the way,": 0.12, "Also,": 0.08, "Oh,": 0.08, "none": 0.72},
}
REASON_RATE = 0.15                       # >= 85% of sentences carry none
NOISE_QUOTAS = {                         # rates; batch counts = round(rate*n)
    "lme": {"exclamation": 0.13, "lowercase_start": 0.06,
            "typo_missing_apostrophe": 0.017, "no_terminal_punct": 0.09,
            "filler": 0.0},
    "locomo": {"exclamation": 0.35, "lowercase_start": 0.0,
               "typo_missing_apostrophe": 0.01, "no_terminal_punct": 0.0,
               "filler": 0.15},
}

BARE_OPENER_BAN = ("by the way", "also,", "also ", "oh,", "oh ", "btw",
                   "speaking of", "on another note", "anyway", "well,",
                   "so,", "so ", "yeah")

# ---------------------------------------------------------------------------
# Banned patterns (32, VERBATIM from register_analysis_hosts.json) + external
# anti-patterns (lintable forms)
# ---------------------------------------------------------------------------


def load_banned_patterns() -> list[dict]:
    """The 32 selective patterns: host_user_pct < 0.6 and >= 1 r1 arc hit —
    exactly the union the hosts analysis reports flagging 97/120 r1 sentences."""
    doc = json.loads(HOSTS_ANALYSIS_PATH.read_text())
    out = [p for p in doc["banned_patterns_ranked"]
           if p["host_user_pct"] < 0.6 and p["arc_hits"] >= 1]
    n_expected = doc["banned_pattern_union_coverage"]["n_selective_patterns"]
    if len(out) != 32 or n_expected != 32:
        raise SystemExit(f"banned-pattern selection drifted: got {len(out)}, "
                         f"analysis says {n_expected}, expected 32")
    for p in out:
        p["compiled"] = re.compile(p["regex"])
    return out


EXTERNAL_ANTI_PATTERNS = [
    ("AT1_who_knows_maybe", r"(?i)\bwho knows, maybe\b"),
    ("AT4_in_the_near_future", r"(?i)\b(sometime )?in the (near )?future\b"),
    ("AT6_look_forward_to", r"(?i)\bI look forward to\b"),
    ("AC3_AA4_formal_verbs",
     r"(?i)\b(purchased|acquired|relocated|transitioned to|opted not to|"
     r"postponed|reconsidered|abandon(ed|ing)?)\b"),
    ("AC4_pleased_to_share", r"(?i)\b(pleased|excited) to (share|announce)\b"),
    ("AA2_not_the_right_fit", r"(?i)\bnot the right fit\b"),
    ("AT3_right_move_for_me", r"(?i)\bthe right (move|choice|fit) for me\b"),
]
EXTERNAL_ANTI_COMPILED = [(name, re.compile(rx))
                          for name, rx in EXTERNAL_ANTI_PATTERNS]

# ---------------------------------------------------------------------------
# Category cue regexes (resolution outcomes; hedge uses the frozen
# HEDGE_CUE_FAMILIES detection via hedge_arcs.detect_cue_family)
# ---------------------------------------------------------------------------

COMPLETED_CUE = re.compile(
    r"(?i)\b(?:(?:i|we)(?:'ve| have)? (?:just|finally) \w+"
    r"|ended up (?:get|go|buy|sign|switch|join|order|book|pick|with)\w*"
    r"|went with\b|switched (?:to|over)\b|signed up (?:for|with|at)\b"
    r"|got myself\b|just (?:got|bought|signed|switched|started|joined|picked"
    r"|ordered|booked|moved|set)\b|finally (?:got|bought|signed|switched"
    r"|started|joined|picked)\b)")
ABANDON_CUE = {
    "lme": re.compile(
        r"(?i)\b(didn'?t end up|decided against|decided not to|i'?ll pass"
        r"|changed my mind|never got around|holding off|held off|gave up on"
        r"|dropped the idea|let (?:it|that) go|scrapped"
        r"|not (?:doing|getting|going through with) (?:that|it))\b"),
    "locomo": re.compile(
        r"(?i)\b(setback|got hurt|hurt my|didn'?t (?:go|work out|pan out)"
        r"|didn'?t make it|had to (?:take a break|stop|put)|on hold"
        r"|fell through|rejection|gave up on)\b"),
}
OPEN_CUE = re.compile(
    r"(?i)\b(still|haven'?t|hasn'?t|not sure yet|might still|undecided"
    r"|we'?ll see|will see|yet)\b")
NEGATION_BEFORE = re.compile(
    r"(?i)\b(haven'?t|hasn'?t|never|not|didn'?t|don'?t)\b[\w\s,']{0,16}$")
SECOND_PERSON = re.compile(r"\byou(?:'ve|'re|r\b| have\b| said\b)", re.I)
TYPO_CONTRACTION = re.compile(
    r"(?<![\w'])(dont|didnt|im|ive|cant|wont|isnt|wasnt|thats|havent|doesnt)"
    r"(?![\w'])", re.I)
STANDALONE_LOWER_I = re.compile(r"(?<![\w'])i(?=[ ,])")
# value-inversion guard (T3 invalidate) — same logic as inject._render_ok,
# reimplemented because _render_ok also applies ROUND2_TELL_BLACKLIST, which
# would wrongly reject the directed "By the way," opener.
_KEPT_RE = re.compile(
    r"(?:stay(?:ing)?|keep(?:ing)?|stick(?:ing)?|remain(?:ing)?)\s*"
    r"(?:with|at|on)?\s+((?:\w+[ &'\-\.]*){1,7})")


def _value_inverted(text: str, value: str) -> bool:
    vn, an = _norm(value), _norm(text)
    if not vn:
        return False
    change_target = re.search(
        rf"(?:switch\w*|chang\w*|mov\w*|swap\w*)\s+(?:\w+[ &'\-\.]*){{0,6}}?"
        rf"to\s+(?:\w+[ &'\-\.]*){{0,3}}?{re.escape(vn)}", an)
    for kept in _KEPT_RE.finditer(an):
        if vn in _norm(kept.group(1)) and not change_target:
            return True
    return False


# ---------------------------------------------------------------------------
# Deterministic directives (seed 20260708 + arc index, per part)
# ---------------------------------------------------------------------------

def _rng(idx: int, part: str, what: str) -> random.Random:
    return random.Random(f"m4r2:{SEED_M4}:{idx}:{part}:{what}")


def opener_words(opener: str) -> int:
    return 0 if opener == "none" else len(opener.split())


def assign_openers(items: list[dict]) -> None:
    """Batch-level seeded opener assignment at the frozen per-benchmark rates
    (exact counts, so LME realizes 'By the way,' at ~1/3 as directed rather
    than drifting on independent draws); remainder goes to bare."""
    order = list(range(len(items)))
    random.Random(f"m4r2-opener:{SEED_M4}").shuffle(order)
    by_bm: dict[str, list[int]] = {}
    for i in order:
        by_bm.setdefault(items[i]["benchmark"], []).append(i)
    for bm, idxs in by_bm.items():
        pos = 0
        for o in OPENERS[:-1]:                      # "none" takes the rest
            count = round(OPENER_WEIGHTS[bm][o] * len(idxs))
            for i in idxs[pos:pos + count]:
                items[i]["opener"] = o
            pos += count
        for i in idxs[pos:]:
            items[i]["opener"] = "none"


def draw_target_words(idx: int, part: str, benchmark: str,
                      value: str, opener: str) -> int:
    med, sigma, lo, hi = LENGTH_PARAMS[benchmark]
    w = round(_rng(idx, part, "length").lognormvariate(math.log(med), sigma))
    w = max(lo, min(hi, w))
    floor = len(value.split()) + opener_words(opener) + 4
    return max(w, floor)


def assign_reasons(items: list[dict]) -> None:
    """Exactly round(15%) of sentences get reason=allowed (batch-level seeded);
    the rest are linted reason-free (>= 85% bare)."""
    order = list(range(len(items)))
    random.Random(f"m4r2-reason:{SEED_M4}").shuffle(order)
    n_allowed = round(REASON_RATE * len(items))
    allowed = set(order[:n_allowed])
    for i, it in enumerate(items):
        it["reason"] = ("one_short_plain_reason_allowed" if i in allowed
                        else "none")


def assign_noise(items: list[dict]) -> None:
    """Batch-level seeded noise quotas, host-conditional, max ONE feature per
    sentence. Eligibility: no_terminal_punct only at turn_end (visible seam);
    exclamation never on abandoned resolutions; lowercase_start only with a
    non-bare opener; filler is LoCoMo-only."""
    for it in items:
        it["noise_directive"] = "none"
    order = list(range(len(items)))
    random.Random(f"m4r2-noise:{SEED_M4}").shuffle(order)
    by_bm: dict[str, list[int]] = {}
    for i in order:
        by_bm.setdefault(items[i]["benchmark"], []).append(i)

    def take(bm: str, feature: str, count: int, eligible) -> None:
        for i in by_bm.get(bm, []):
            if count <= 0:
                break
            it = items[i]
            if it["noise_directive"] != "none" or not eligible(it):
                continue
            it["noise_directive"] = feature
            count -= 1

    for bm, quotas in NOISE_QUOTAS.items():
        n_bm = len(by_bm.get(bm, []))
        if not n_bm:
            continue
        n_te = sum(1 for i in by_bm[bm] if items[i]["position"] == "turn_end")
        take(bm, "no_terminal_punct", round(quotas["no_terminal_punct"] * n_te),
             lambda it: it["position"] == "turn_end")
        take(bm, "exclamation", round(quotas["exclamation"] * n_bm),
             lambda it: not (it["part"] == "resolution"
                             and it["resolution_type"] == "abandoned"))
        take(bm, "lowercase_start", round(quotas["lowercase_start"] * n_bm),
             lambda it: it["opener"] != "none")
        take(bm, "typo_missing_apostrophe",
             round(quotas["typo_missing_apostrophe"] * n_bm), lambda it: True)
        take(bm, "filler", round(quotas["filler"] * n_bm), lambda it: True)


# ---------------------------------------------------------------------------
# Exemplars — REAL corpus user sentences, matched benchmark + category,
# payload-level ONLY (no-6-gram lint keeps them out of stimuli)
# ---------------------------------------------------------------------------

HOST_CATEGORY = {"hedge": "plan", "open": "plan",
                 "resolve_true": "completed", "abandoned": "abandoned"}
EXTERNAL_CATEGORY = {"hedge": "tentative_plan", "open": "tentative_plan",
                     "resolve_true": "completed_change",
                     "abandoned": "abandonment"}


def load_exemplar_pools() -> dict:
    hosts = json.loads(HOSTS_ANALYSIS_PATH.read_text())["mined_examples"]
    ext = json.loads(EXTERNAL_ANALYSIS_PATH.read_text())["examples"]
    pools: dict[tuple[str, str], list[dict]] = {}
    for bm in ("lme", "locomo"):
        for cat in ("plan", "completed", "abandoned"):
            pools[(bm, cat)] = [
                {"id": f"hosts:{bm}:{cat}:{i}:{e['prov']}", "text": e["text"]}
                for i, e in enumerate(hosts[bm][cat])]
    for cat in ("tentative_plan", "completed_change", "abandonment"):
        pools[("ext", cat)] = [
            {"id": f"ext:{cat}:{i}:{e['corpus']}", "text": e["text"]}
            for i, e in enumerate(ext[cat])]
    return pools


def pick_exemplars(pools: dict, idx: int, part: str, benchmark: str,
                   category_key: str, n_host: int = 4,
                   n_ext: int = 3) -> list[dict]:
    rng = _rng(idx, part, "exemplars")
    host_pool = pools[(benchmark, HOST_CATEGORY[category_key])]
    ext_pool = pools[("ext", EXTERNAL_CATEGORY[category_key])]
    picked = rng.sample(host_pool, min(n_host, len(host_pool)))
    picked += rng.sample(ext_pool, min(n_ext, len(ext_pool)))
    return picked


def _ngrams(text: str, n: int = 6) -> set[tuple[str, ...]]:
    toks = re.findall(r"[a-z0-9']+", text.lower())
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


# ---------------------------------------------------------------------------
# Lint battery
# ---------------------------------------------------------------------------

def lint_aside(text: str, *, item: dict, value: str,
               banned: list[dict], exemplars: list[dict]) -> list[str]:
    """Deterministic acceptance lint. Returns [] when the aside passes; each
    failure string doubles as the retry payload's lint_feedback line."""
    fails: list[str] = []
    text = text.strip()
    if not text:
        return ["empty output"]
    opener, noise, reason = item["opener"], item["noise_directive"], item["reason"]

    # -- surface form ------------------------------------------------------
    bad_chars = sorted({c for c in text if not (32 <= ord(c) < 127)})
    if bad_chars:
        fails.append(f"non-ASCII character(s) {bad_chars!r} — use straight "
                     "apostrophes, no curly quotes, no em-dash")
    if ";" in text:
        fails.append("semicolon is forbidden")
    if "?" in text:
        fails.append("the aside must be a statement, never a question")
    if re.search(r"[.!?]\s+\S", text):
        fails.append("more than one sentence (internal sentence break)")

    # -- opener ------------------------------------------------------------
    if opener != "none":
        ok_openers = [opener]
        if noise == "lowercase_start":
            ok_openers.append(opener[0].lower() + opener[1:])
        if not any(text.startswith(o) for o in ok_openers):
            fails.append(f"must start with the directed opener {opener!r}")
        body = text
        for o in ok_openers:
            if text.startswith(o):
                body = text[len(o):]
                break
    else:
        low = text.lower()
        if any(low.startswith(b) for b in BARE_OPENER_BAN):
            fails.append("opener directive is 'none': start directly with the "
                         "sentence (no By the way/Also/Oh/Well/So/Yeah/"
                         "Speaking of/Anyway)")
        body = text
    if body.count(",") > 1:
        fails.append("more than one comma in the sentence body (opener comma "
                     "exempt)")

    # -- length ------------------------------------------------------------
    n_words = len(text.split())
    if abs(n_words - item["target_words"]) > WORD_TOL:
        fails.append(f"length {n_words} words is outside target "
                     f"{item['target_words']} +/- {WORD_TOL}")

    # -- noise directive ---------------------------------------------------
    first_alpha = next((c for c in text if c.isalpha()), "")
    if noise == "lowercase_start":
        if not first_alpha.islower():
            fails.append("noise directive lowercase_start: first letter must "
                         "be lowercase")
    elif first_alpha and first_alpha.islower():
        fails.append("start with a capital letter (no lowercase_start "
                     "directive)")
    if noise == "exclamation":
        if not text.endswith("!"):
            fails.append("noise directive exclamation: end with '!'")
    elif "!" in text:
        fails.append("no exclamation mark without the exclamation noise "
                     "directive")
    if noise == "no_terminal_punct":
        if text[-1] in ".!?":
            fails.append("noise directive no_terminal_punct: end with no "
                         "punctuation at all")
    elif noise != "exclamation" and not text.endswith("."):
        fails.append("end with a period")
    if noise == "typo_missing_apostrophe":
        if not TYPO_CONTRACTION.search(text):
            fails.append("noise directive typo_missing_apostrophe: write one "
                         "contraction without its apostrophe (dont, didnt, Im, "
                         "cant, ...)")
    else:
        if TYPO_CONTRACTION.search(text):
            fails.append("missing-apostrophe contraction without the typo "
                         "directive")
        if STANDALONE_LOWER_I.search(text):
            fails.append("standalone lowercase 'i' without the typo directive")
    if noise == "filler" and not re.search(
            r"(?i)\b(btw|haha|so yeah|honestly)\b", text):
        fails.append("noise directive filler: include one casual filler "
                     "(btw, haha, so yeah, honestly)")

    # -- reason quota ------------------------------------------------------
    n_reason = len(re.findall(r"(?i)\b(because|since)\b", text))
    if reason == "none" and n_reason:
        fails.append("no because/since reason clause allowed for this "
                     "sentence — state it bare")
    if n_reason > 1:
        fails.append("at most one reason clause")

    # -- banned patterns ---------------------------------------------------
    for p in banned:
        if p["compiled"].search(text):
            fails.append(f"banned pattern '{p['name']}' matched "
                         f"(regex {p['regex']})")
    for name, rx in EXTERNAL_ANTI_COMPILED:
        if rx.search(text):
            fails.append(f"external anti-pattern {name} matched")
    low = text.lower()
    for b in BLACKLIST:
        if b.lower() in low:
            fails.append(f"template-blacklist phrase {b!r}")
    if SECOND_PERSON.search(text):
        fails.append("second-person framing — the aside is about the "
                     "speaker only")
    for pat in RELATIVE_TIME_PATTERNS:
        if re.search(pat, text, re.I):
            fails.append(f"relative-time phrase (regex {pat}) is forbidden")

    # -- value -------------------------------------------------------------
    n_val = text.count(value)
    if n_val == 0:
        fails.append(f"the value {value!r} must appear VERBATIM")
    elif n_val > 1:
        fails.append(f"the value {value!r} must appear exactly once")

    # -- category cues -----------------------------------------------------
    part, res_type = item["part"], item["resolution_type"]
    bm = item["benchmark"]
    fam = ha.detect_cue_family(text)[0]
    if part == "hedge":
        if fam != item["hedge_strength"]:
            fails.append(
                f"hedge must read tentative at strength "
                f"{item['hedge_strength']!r} (detected {fam!r}) — use that "
                "strength's frames from the category guide")
        if ABANDON_CUE["lme"].search(text) or ABANDON_CUE["locomo"].search(text):
            fails.append("hedge must not contain abandonment language")
        if _affirmative_completed(text, value):
            fails.append("hedge must not read as a completed event")
    elif res_type == "resolve_true":
        if not COMPLETED_CUE.search(text):
            fails.append("completed resolution needs a plain completion frame "
                         "(just/finally got|switched|signed up, ended up "
                         "going with)")
        if fam is not None:
            fails.append("completed resolution must not read tentative "
                         f"(detected {fam!r} cue)")
        if ABANDON_CUE["lme"].search(text) or ABANDON_CUE["locomo"].search(text):
            fails.append("completed resolution must not contain abandonment "
                         "language")
    elif res_type == "abandoned":
        cue = ABANDON_CUE[bm]
        if not cue.search(text):
            frame = ("setback-framed (got hurt / setback / didn't go the way "
                     "I wanted / on hold)" if bm == "locomo" else
                     "decision-framed with a softener (didn't end up / "
                     "decided against it after all / I'll pass for now / "
                     "never got around to it)")
            fails.append(f"abandoned resolution must be {frame}")
        if _value_inverted(text, value):
            fails.append("abandoned resolution asserts the value as KEPT — "
                         "the value is the road not taken")
        if _affirmative_completed(text, value):
            fails.append("abandoned resolution must not read as a completed "
                         "adoption")
    elif res_type == "open":
        if fam is None and not OPEN_CUE.search(text):
            fails.append("open resolution must read still-pending (still / "
                         "haven't decided yet / might still)")
        if ABANDON_CUE["lme"].search(text) or ABANDON_CUE["locomo"].search(text):
            fails.append("open resolution must not contain abandonment "
                         "language")
        if _affirmative_completed(text, value):
            fails.append("open resolution must not read as completed")
        if _value_inverted(text, value):
            fails.append("open resolution asserts the value as adopted/kept")

    # -- exemplar 6-gram ---------------------------------------------------
    out_grams = _ngrams(text)
    for ex in exemplars:
        shared = out_grams & _ngrams(ex["text"])
        if shared:
            gram = " ".join(sorted(shared)[0])
            fails.append(f"shares the 6-gram '{gram}' with exemplar "
                         f"{ex['id']} — reword freshly")
    return fails


def _affirmative_completed(text: str, value: str | None = None) -> bool:
    """COMPLETED_CUE hit not shielded by a nearby preceding negation.

    When `value` is given the check is SCOPED TO THE VALUE: only a completed
    cue whose 80-char window contains the value counts (lint false-positive
    re-scope, frozen-lint precedent: an abandonment sentence may complete a
    DIFFERENT mundane act — "I just grabbed my usual tube" — without adopting
    the value; observed on M4-gpt4_2f8be40d)."""
    for m in COMPLETED_CUE.finditer(text):
        if NEGATION_BEFORE.search(text[:m.start()]):
            continue
        if value is not None and _norm(value) not in _norm(
                text[m.start():m.start() + 80]):
            continue
        return True
    return False


# ---------------------------------------------------------------------------
# Item construction (one item per sentence; directives all deterministic)
# ---------------------------------------------------------------------------

def build_items(arcs: list[dict]) -> list[dict]:
    items: list[dict] = []
    for idx, arc in enumerate(arcs):
        for part in ("hedge", "resolution"):
            items.append({
                "arc_id": arc["arc_id"],
                "arc_index": idx,
                "part": part,
                "benchmark": arc["benchmark"],
                "position": arc[part]["position"],
                "resolution_type": arc["resolution_type"],
                "hedge_strength": arc["hedge_strength"],
                "value": arc["value"],
            })
    assign_openers(items)                    # opener before length (floor)
    for it in items:
        it["target_words"] = draw_target_words(
            it["arc_index"], it["part"], it["benchmark"],
            it["value"], it["opener"])
    assign_reasons(items)
    assign_noise(items)
    return items


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def _category_key(item: dict) -> str:
    return "hedge" if item["part"] == "hedge" else item["resolution_type"]


def render_payload(arc: dict, item: dict, exemplars: list[dict],
                   attempt: int, lint_feedback: list[str] | None,
                   hedge_text: str | None) -> dict:
    part = arc[item["part"]]
    question = {q["question_id"]: q
                for q in data.load_oracle()}[arc["question_id"]]
    host_turn = question["haystack_sessions"][part["session_idx"]][
        part["turn_idx"]]["content"]
    payload = {
        "task": "m4_r2_render",
        "benchmark": item["benchmark"],
        "category": ("hedge" if item["part"] == "hedge"
                     else f"resolution_{item['resolution_type']}"),
        "contract": part["contract"],
        "outcome": (None if item["part"] == "hedge"
                    else item["resolution_type"]),
        "hedge_strength": (arc["hedge_strength"] if item["part"] == "hedge"
                           else None),
        "key": {k: arc["key"][k] for k in ("entity", "attribute", "description")},
        "value": arc["value"],
        "value_rule": "must appear verbatim exactly once",
        "earlier_hedge_text": hedge_text if item["part"] == "resolution" else None,
        "host_turn_excerpt": " ".join(host_turn.split()[:60]),
        "insertion_position": part["position"],
        "host_session_date": part["date"],
        "other_session_date": (arc["resolution"]["date"]
                               if item["part"] == "hedge"
                               else arc["hedge"]["date"]),
        "target_words": item["target_words"],
        "word_tolerance": WORD_TOL,
        "opener": item["opener"],
        "noise_directive": item["noise_directive"],
        "reason": item["reason"],
        "style_exemplars": exemplars,
        "exemplar_rule": ("REAL user sentences: match their register; never "
                          "reuse any 6-word sequence from them"),
        "attempt": attempt,
    }
    if lint_feedback:
        payload["lint_feedback"] = lint_feedback
        payload["fresh_wording_note"] = (
            "Previous attempt failed the lints above. Produce a freshly "
            "worded sentence that fixes EVERY listed failure while keeping "
            "all hard rules.")
    return payload


async def render_item(arc: dict, item: dict, *, render_sys: str,
                      pools: dict, banned: list[dict],
                      hedge_text: str | None) -> tuple[str, dict]:
    exemplars = pick_exemplars(pools, item["arc_index"], item["part"],
                               item["benchmark"], _category_key(item))
    feedback: list[str] | None = None
    for attempt in range(MAX_ATTEMPTS):
        payload = render_payload(arc, item, exemplars, attempt, feedback,
                                 hedge_text)
        out = await llm.call_llm(
            render_sys, json.dumps(payload, ensure_ascii=False),
            trace_tag=f"m4r2-{item['part']}-{arc['arc_id']}-a{attempt}")
        aside = str(out.get("aside", "")).strip()
        fails = lint_aside(aside, item=item, value=arc["value"],
                           banned=banned, exemplars=exemplars)
        if not fails:
            meta = {
                "opener": item["opener"],
                "target_words": item["target_words"],
                "noise_directive": item["noise_directive"],
                "reason": item["reason"],
                "exemplar_ids": [e["id"] for e in exemplars],
                "attempts": attempt + 1,
            }
            return aside, meta
        feedback = fails
    raise RuntimeError(
        f"{arc['arc_id']} {item['part']}: lint still failing after "
        f"{MAX_ATTEMPTS} attempts: {feedback}")


PLAN_GEOMETRY_FIELDS = (
    "arc_id", "row_kind", "benchmark", "question_id", "question_type",
    "resolution_type", "gold_resolution_status", "hedge_strength", "key",
    "hedge", "resolution", "pending_window_user_turns", "seed",
    "seed_draw_index")


def assert_plan_unchanged(arcs_r1: list[dict]) -> None:
    """The r2 rows must carry the frozen plan geometry byte-for-byte."""
    plan = {a["arc_id"]: a for a in ha._read_jsonl(ha.ARC_PLAN_PATH)}
    if set(plan) != {a["arc_id"] for a in arcs_r1}:
        raise SystemExit("arc id set differs from the frozen arc_plan.jsonl")
    for a in arcs_r1:
        p = plan[a["arc_id"]]
        for f in PLAN_GEOMETRY_FIELDS:
            if a[f] != p[f]:
                raise SystemExit(
                    f"{a['arc_id']}: geometry field {f!r} differs from the "
                    "frozen plan — the r2 round may not touch the plan")


async def run_r2_render(limit: int | None, fresh_cap: int | None) -> dict:
    arcs_r1 = ha._read_jsonl(ARCS_R1_PATH)
    if not arcs_r1:
        raise SystemExit(f"missing retired round-1 arcs at {ARCS_R1_PATH}")
    assert_plan_unchanged(arcs_r1)
    render_sys = STYLE_PROMPT_PATH.read_text()
    style_sha = hashlib.sha256(render_sys.encode()).hexdigest()
    banned = load_banned_patterns()
    pools = load_exemplar_pools()
    items = build_items(arcs_r1)
    by_key = {(it["arc_id"], it["part"]): it for it in items}

    existing = {a["arc_id"]: a for a in ha._read_jsonl(ARCS_PATH)
                if a.get("render_round") == 2}
    arcs_out: list[dict] = []
    todo: list[dict] = []
    for a in arcs_r1:
        prior = existing.get(a["arc_id"])
        if (prior and prior.get("hedge_text") and prior.get("resolution_text")):
            arcs_out.append(prior)           # resume: already re-rendered
        else:
            arc = dict(a)                    # plan + value carried forward
            arcs_out.append(arc)
            todo.append(arc)
    if limit is not None:
        todo = todo[:limit]

    failures: list[str] = []

    async def render_one(arc: dict) -> None:
        try:
            if not ha._fm4_probe_ok(arc["key"]["probe_question"], arc["value"]):
                raise RuntimeError("FM-4 violation: value appears in the "
                                   "probe question")
            h_item = by_key[(arc["arc_id"], "hedge")]
            r_item = by_key[(arc["arc_id"], "resolution")]
            hedge_text, h_meta = await render_item(
                arc, h_item, render_sys=render_sys, pools=pools,
                banned=banned, hedge_text=None)
            res_text, r_meta = await render_item(
                arc, r_item, render_sys=render_sys, pools=pools,
                banned=banned, hedge_text=hedge_text)
            arc["hedge_text"] = hedge_text
            arc["resolution_text"] = res_text
            arc["render_round"] = 2
            arc["style_prompt_sha"] = style_sha
            arc["r2"] = {"hedge": h_meta, "resolution": r_meta}
            arc["lints"] = {
                "hedge": ha._lint(hedge_text),
                "resolution": ha._lint(res_text),
                "fm4_probe_value_free": True,
                "r2_lint": "passed (see r2 fields; battery in hedge_arcs_r2)",
            }
            arc["render"] = {
                "pipeline": ("m4 r2 single-call corpus-styled render "
                             "(prompts/injector_render_m4r2.txt)"),
                "style_prompt_sha": style_sha,
                "style": "corpus-derived user register (M4 r2 contract)",
            }
            arc["manifest_rows"] = ha.manifest_rows_for_arc(arc)
        except Exception as exc:
            failures.append(f"{arc['arc_id']}: {exc}")

    guard = ha.m4_guard(fresh_cap)
    guard.install()
    try:
        await asyncio.gather(*(render_one(a) for a in todo))
    finally:
        guard.uninstall()

    ha._write_jsonl(ARCS_PATH, arcs_out)
    rendered = [a for a in arcs_out if a.get("render_round") == 2]
    attempts = {f"{a['arc_id']}:{p}": a["r2"][p]["attempts"]
                for a in rendered for p in ("hedge", "resolution")}
    noise_counts: dict[str, int] = {}
    for a in rendered:
        for p in ("hedge", "resolution"):
            nd = a["r2"][p]["noise_directive"]
            noise_counts[nd] = noise_counts.get(nd, 0) + 1
    summary = {
        "id": "V2.4-M4-R2-RENDER",
        "prereg": ("protocol/CHANGELOG.md 2026-07-08 'M4 r2 style contract "
                   "FROZEN' (prompt sha " + style_sha + ")"),
        "n_planned": len(arcs_r1),
        "n_rendered_r2": len(rendered),
        "plan_unchanged": True,
        "style_prompt_sha": style_sha,
        "banned_patterns_enforced": len(banned),
        "noise_directive_counts": noise_counts,
        "reason_allowed_count": sum(
            1 for a in rendered for p in ("hedge", "resolution")
            if a["r2"][p]["reason"] != "none"),
        "attempts_gt1": {k: v for k, v in attempts.items() if v > 1},
        "budget": {**ha.advisory_note(), "fresh_calls_this_run": guard.fresh},
        "failures": failures,
        "outputs": {"arcs": ha._rel(ARCS_PATH)},
    }
    ha._write_json(R2_SUMMARY_PATH, summary)
    if failures:
        for f in failures:
            print(f"R2 RENDER FAILURE: {f}", file=sys.stderr)
    return summary


# ---------------------------------------------------------------------------
# Register check — the hosts-analysis methodology over the 120 r2 sentences
# ---------------------------------------------------------------------------

_CONTRACTION = re.compile(r"\w'\w")
_FIRST_PERSON = re.compile(r"\b(?:i|we)\b", re.I)
_FILLER = re.compile(r"(?i)\b(haha|lol|btw|honestly|so yeah|you know|i mean"
                     r"|oh|well|yeah|hmm|wow|anyway)\b")
_FORMAL_CONNECTIVE = re.compile(
    r"\b(Additionally|Furthermore|Moreover|Consequently|Therefore|Notably"
    r"|Overall)\b")


def _pct(flags: list[bool]) -> float:
    return round(100 * sum(flags) / max(1, len(flags)), 2)


def register_metrics(texts: list[str]) -> dict:
    n = len(texts)
    lengths = [float(len(t.split())) for t in texts]
    sents = [max(1, len(re.findall(r"[.!?]+(?:\s|$)", t)) or 1) for t in texts]
    total_words = sum(lengths)
    return {
        "n_turns": n,
        "words": {
            "mean": round(statistics.fmean(lengths), 2),
            "median": ha._percentile(lengths, 0.5),
            "p10": ha._percentile(lengths, 0.10),
            "p90": ha._percentile(lengths, 0.90),
        },
        "words_per_sent_mean": round(
            statistics.fmean(w / s for w, s in zip(lengths, sents)), 2),
        "commas_per_sent_mean": round(
            statistics.fmean(t.count(",") / s for t, s in zip(texts, sents)), 2),
        "semicolons_per_100_turns": round(
            100 * sum(t.count(";") for t in texts) / max(1, n), 2),
        "contractions_per_100_words": round(
            100 * sum(len(_CONTRACTION.findall(t)) for t in texts)
            / max(1.0, total_words), 2),
        "pct_turns": {
            "exclamation": _pct(["!" in t for t in texts]),
            "question": _pct(["?" in t for t in texts]),
            "no_terminal_punct": _pct(
                [not t.rstrip().endswith((".", "!", "?")) for t in texts]),
            "lowercase_start": _pct(
                [bool(t) and t.lstrip()[:1].islower() for t in texts]),
            "first_person": _pct(
                [bool(_FIRST_PERSON.search(t)) for t in texts]),
            "with_contraction": _pct(
                [bool(_CONTRACTION.search(t)) for t in texts]),
            "typoish": _pct([bool(TYPO_CONTRACTION.search(t)
                                  or STANDALONE_LOWER_I.search(t)
                                  or "  " in t) for t in texts]),
            "curly_apostrophe_or_quote": _pct(
                [bool(re.search(r"[‘’“”]", t))
                 for t in texts]),
            "em_dash": _pct(["—" in t for t in texts]),
            "by_the_way_or_btw": _pct(
                [bool(re.search(r"(?i)\bby the way\b|\bbtw\b", t))
                 for t in texts]),
            "too_also_as_well": _pct(
                [bool(re.search(r"(?i)\b(too|also|as well)\b", t))
                 for t in texts]),
            "filler_or_discourse_marker": _pct(
                [bool(_FILLER.search(t)) for t in texts]),
            "formal_connective": _pct(
                [bool(_FORMAL_CONNECTIVE.search(t)) for t in texts]),
        },
    }


def banned_union_hits(texts: list[str], banned: list[dict]) -> dict:
    per_pattern = {p["name"]: sum(1 for t in texts if p["compiled"].search(t))
                   for p in banned}
    flagged = [t for t in texts
               if any(p["compiled"].search(t) for p in banned)]
    return {"n_texts": len(texts), "n_flagged": len(flagged),
            "pct_flagged": _pct([any(p["compiled"].search(t) for p in banned)
                                 for t in texts]),
            "per_pattern_nonzero": {k: v for k, v in per_pattern.items() if v},
            "flagged_texts": flagged}


def run_register_check() -> dict:
    banned = load_banned_patterns()
    r2 = [json.loads(l) for l in ARCS_PATH.open()]
    r1 = [json.loads(l) for l in ARCS_R1_PATH.open()]
    r2_texts = [a[k] for a in r2 for k in ("hedge_text", "resolution_text")]
    r1_texts = [a[k] for a in r1 for k in ("hedge_text", "resolution_text")]
    hosts = json.loads(HOSTS_ANALYSIS_PATH.read_text())["register_stats"]
    doc = {
        "id": "V2.4-M4-R2-REGISTER-CHECK",
        "method": ("register_analysis_hosts.json methodology re-run over the "
                   "120 r2 sentences; r1 recomputed with the same functions; "
                   "host rows copied from the frozen analysis for reference "
                   "(all 60 arcs are LME-hosted, so lme_user is the binding "
                   "target)"),
        "banned_regex_union": {
            "n_patterns": len(banned),
            "r2": banned_union_hits(r2_texts, banned),
            "r1": banned_union_hits(r1_texts, banned),
        },
        "register": {
            "r2_sentences": register_metrics(r2_texts),
            "r1_sentences": register_metrics(r1_texts),
            "host_lme_user": hosts["lme_user"],
            "host_locomo": hosts["locomo_user_both_speakers"],
        },
    }
    doc["banned_regex_union"]["r1"]["flagged_texts"] = "(omitted)"
    ha._write_json(R2_CHECK_PATH, doc)
    return doc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="v2.4 M4 r2 — corpus-styled TEXT re-render")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ren = sub.add_parser("r2-render", help="re-render all 60 arcs' text")
    ren.add_argument("--limit", type=int, default=None)
    ren.add_argument("--fresh-cap", type=int, default=None)
    sub.add_parser("review-pack", help="rebuild review_pack.md (r1 format, "
                                       "spliced sentence never elided)")
    sub.add_parser("register-check", help="r2 vs host vs r1 register table")
    args = ap.parse_args()

    if args.cmd == "r2-render":
        summary = asyncio.run(run_r2_render(args.limit, args.fresh_cap))
        print(json.dumps({k: summary[k] for k in (
            "n_planned", "n_rendered_r2", "noise_directive_counts",
            "reason_allowed_count", "attempts_gt1", "budget", "failures")},
            indent=1))
        print(f"-> {ARCS_PATH}\n-> {R2_SUMMARY_PATH}")
        if summary["failures"]:
            raise SystemExit(f"{len(summary['failures'])} item(s) failed "
                             "(progress persisted; rerun resumes)")
    elif args.cmd == "review-pack":
        out = ha.run_review_pack()
        print(json.dumps(out, indent=1))
    elif args.cmd == "register-check":
        doc = run_register_check()
        b = doc["banned_regex_union"]
        print(json.dumps({
            "banned_union_r2": {k: b["r2"][k] for k in
                                ("n_flagged", "pct_flagged",
                                 "per_pattern_nonzero")},
            "banned_union_r1_flagged": b["r1"]["n_flagged"],
            "r2_words": doc["register"]["r2_sentences"]["words"],
            "r1_words": doc["register"]["r1_sentences"]["words"],
        }, indent=1))
        print(f"-> {R2_CHECK_PATH}")


if __name__ == "__main__":
    main()
