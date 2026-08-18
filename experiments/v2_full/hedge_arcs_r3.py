"""hedge_arcs_r3.py — v2.4 M4 round-3 REDESIGN: 60 fully synthetic themed
multi-session conversations (protocol/CHANGELOG.md, 2026-07-09 entries
"M4 owner review round 2: REJECT" + "M4 r3 generation contract FROZEN").

Round 2 (splice-level rendering into benchmark hosts) was rejected wholesale:
the host user register is itself LLM-authored, and splicing guarantees
unmotivated asides. Round 3 replaces the aside design entirely: each arc
becomes ONE self-contained user<->assistant conversation whose THEME is the
key's domain, so the hedge is contextually motivated task content and the
resolution arrives naturally in a later session. The frozen r2 arc PLAN
carries over untouched (arc ids, keys, VALUES, value_candidates, outcome
balance 20/20/20, hedge strengths, pending-window targets, seeds); the
conversation is new and contains ZERO benchmark text.

Frozen contract (all deterministic knobs seeded 20260708, keyed by arc_id):
  * geometry: 2 sessions (3 for open arcs with pending-window target >= 6);
    3-6 user turns per session (6-12 alternating turns, user first); hedge
    beat in session 0 at user turn >= 1; resolution beat in the final
    session (open arcs: a still-pending mention); beat slots solved so the
    realized pending window approximates the plan target (clamped by
    capacity, realized value recorded); invented 2023 dates, 4-21-day
    session gaps, LME-style per-turn timestamps, strict monotonicity;
  * user-turn register: seeded word bands (lognormal median 11 sigma 0.5
    clamp 3-30, +/-5; beat turns floored at value words + 8; fragments 2-6);
    batch-level seeded noise quotas (max ONE feature per turn): lowercase
    15%, no-terminal 12%, filler 7%, fragment 7%, exclamation 6%, typo 3%;
  * lint battery per user turn: the 32 selective banned regexes VERBATIM,
    external anti-patterns, template BLACKLIST, sycophancy/apology/gratitude
    bans, no-restating (8-gram vs the preceding assistant turn), comma cap,
    no colon-lists/newlines, ASCII, word band, noise conformance;
  * arc integrity: hedge/resolution sentences verbatim in the directed beat
    turns with the VALUE exactly once; hedge realizes exactly its strength
    (frozen HEDGE_CUE_FAMILIES); value-assertion ordering (nothing before
    the hedge asserts the value adopted); the pending window stays pending;
    resolution carries its outcome's cue and none of the others; open arcs
    never resolve anywhere; FM-4 probe re-assert; no 6-gram shared with any
    payload exemplar (human-corpus exemplars are payload-level only).

Budget: every fresh call under SharedCallBudget stage "m4" (ADVISORY stages,
hard 120k total envelope). Forecast <= ~700 fresh calls.

CLI (repo root, pilot/.venv/bin/python experiments/v2_full/hedge_arcs_r3.py ...):
  r3-generate  [--limit K] [--fresh-cap N]  plan+render all 60 conversations
                                            -> arcs.jsonl (r3 schema, resume)
  register-check                            r3 user turns vs human corpora vs
                                            host LME user vs r2 + integrity
                                            relint -> r3_register_check.json
  review-pack                               rebuild review_pack.md (+ template)
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import math
import random
import re
import statistics
import sys

import _bootstrap  # noqa: F401

import hedge_arcs as ha
import hedge_arcs_r2 as r2
import llm
from inject import BLACKLIST, _norm

from paths import PROMPTS_DIR

SEED_M4 = ha.SEED_M4                     # 20260708 — same seed family
M4_DIR = ha.M4_DIR
ARCS_R2_PATH = M4_DIR / "arcs_r2.jsonl"
ARCS_PATH = ha.ARCS_PATH                 # results/v2.4/m4/arcs.jsonl (r3 out)
PLAN_PROMPT_PATH = PROMPTS_DIR / "m4_conv_plan_r3.txt"
RENDER_PROMPT_PATH = PROMPTS_DIR / "m4_conv_render_r3.txt"
R3_SUMMARY_PATH = M4_DIR / "r3_render_summary.json"
R3_CHECK_PATH = M4_DIR / "r3_register_check.json"

MAX_ATTEMPTS = 4                         # 1 initial + <=3 lint-feedback retries

# ---------------------------------------------------------------------------
# Frozen register knobs (changelog 2026-07-09 "M4 r3 generation contract")
# ---------------------------------------------------------------------------

USER_WORDS_MEDIAN = 11
USER_WORDS_SIGMA = 0.5
USER_WORDS_LO, USER_WORDS_HI = 3, 30
WORD_TOL = 5
BEAT_FLOOR_EXTRA = 8                     # beat turn floor = value words + 8
FRAG_LO, FRAG_HI = 2, 6                  # fragment-directed turns
ASSISTANT_MAX_WORDS = 120                # directed
ASSISTANT_HARD_CAP = 130                 # lint
RESTATE_NGRAM = 8                        # vs preceding assistant turn
EXEMPLAR_NGRAM = 6

NOISE_QUOTAS = {                         # rates over ALL user-turn slots
    "lowercase_start": 0.15,
    "no_terminal_punct": 0.12,
    "filler": 0.07,
    "fragment": 0.07,                    # beat turns ineligible
    "exclamation": 0.06,                 # abandoned-resolution beats ineligible
    "typo_missing_apostrophe": 0.03,
}

OPEN_3SESS_MIN_WINDOW = 6                # open arcs with target >= 6: 3 sessions
N_USER_LO, N_USER_HI = 3, 6              # user turns per session
GAP_DAYS_LO, GAP_DAYS_HI = 4, 21         # days between sessions

# ---------------------------------------------------------------------------
# User-side ban lists (owner review round 2 grounds; frozen with the contract)
# ---------------------------------------------------------------------------

SYCOPHANCY_PATTERNS = [
    ("grateful_exclaim", r"(?i)\b(thanks|thank you)\b[^.!?\n]{0,30}!"),
    ("thank_you_so_much", r"(?i)\bthank(s| you) (so|very) much\b"),
    ("thanks_for_the", r"(?i)\bthank(s| you) for (the|all|your|this|that)\b"),
    ("thats_helpful_great",
     r"(?i)\b(that|this)('s| is| was)( really| very| so| super| incredibly)?"
     r" (helpful|great|wonderful|fantastic|amazing|perfect|awesome)\b"),
    ("great_suggestion",
     r"(?i)\b(great|excellent|awesome|wonderful|perfect) "
     r"(suggestion|advice|recommendation|idea|list|breakdown|option)s?\b"),
    ("apology", r"(?i)\bi apologi[sz]e\b"),
    ("sorry_for_confusion",
     r"(?i)\bsorry for (the|any) (confusion|trouble|mix-?up|delay)\b"),
    ("would_be_great",
     r"(?i)\bthat would be (great|perfect|wonderful|amazing|awesome)\b"),
    ("youre_the_best",
     r"(?i)\byou('re| are) (the best|a lifesaver|amazing|awesome)\b"),
    ("assistant_opener", r"(?i)^(sure|certainly|absolutely|of course)[,!.]"),
    ("appreciate", r"(?i)\b(really |very much )?appreciate (your|the|all|it)\b"),
]
SYCOPHANCY_COMPILED = [(n, re.compile(p)) for n, p in SYCOPHANCY_PATTERNS]

#: clean (non-filler-directed) user turns must not carry these
FILLER_BAN = re.compile(r"(?i)(\b(btw|haha)\b|\bso yeah\b|\bhonestly\b)")
FILLER_START_BAN = re.compile(r"(?i)^(oh|well)\b")
FILLER_REQ = re.compile(r"(?i)(\b(btw|haha|honestly)\b|\bso yeah\b|^(oh|well)\b)")

#: r3-extended completion cues (the r2 set + arrival/trigger frames natural
#: to a task conversation); used value-scoped and negation-guarded
COMPLETED_CUE_R3_EXTRA = re.compile(
    r"(?i)\b(came in|arrived|showed up|got delivered|pulled the trigger"
    r"|went with|made the jump)\b")
#: r3-extended abandonment cues (alternative-adoption / drop frames)
ABANDON_CUE_R3_EXTRA = re.compile(
    r"(?i)\b(instead\b|went another (way|direction)|off the (list|table)"
    r"|not happening|bailed on|skipp(ed|ing)|pass(ed|ing) on)")


def _completed_hit(text: str, value: str) -> bool:
    """Value-scoped, negation-guarded completion detection (r2
    `_affirmative_completed` semantics — FORWARD 80-char window only, so a
    completed act whose object is something else near the value does not
    count — with the r3-extended cue set)."""
    for cue in (r2.COMPLETED_CUE, COMPLETED_CUE_R3_EXTRA):
        for m in cue.finditer(text):
            if r2.NEGATION_BEFORE.search(text[:m.start()]):
                continue
            window = text[m.start():m.start() + 80 + len(value)]
            if _norm(value) in _norm(window):
                return True
    return False


def _abandon_hit_near_value(text: str, value: str) -> bool:
    for m in re.finditer(re.escape(value), text):
        window = text[max(0, m.start() - 120):m.end() + 120]
        if (r2.ABANDON_CUE["lme"].search(window)
                or r2.ABANDON_CUE["locomo"].search(window)
                or ABANDON_CUE_R3_EXTRA.search(window)):
            return True
    return False


def _abandon_cue_any(text: str) -> bool:
    return bool(r2.ABANDON_CUE["lme"].search(text)
                or r2.ABANDON_CUE["locomo"].search(text)
                or ABANDON_CUE_R3_EXTRA.search(text))


def _adoption_regexes(value: str) -> list[re.Pattern]:
    v = re.escape(value)
    pats = [
        rf"(?i)\b(i|we)('m| am|'re| are)( now| currently| already)?"
        rf" (on|using|at|with|wearing|running in|riding|taking)"
        rf" (the |a |an |my )?{v}",
        rf"(?i)\b(i|we)( now| currently| usually| always)?"
        rf" (use|wear|take|ride|drink|shop at|go to|bank with|fly with|fly"
        rf"|run in) (the |a |an |my )?{v}",
        rf"(?i)\bmy (new|current|usual|regular) [\w \-]{{0,24}}?{v}",
        rf"(?i)\byou('re| are)?( currently| now| already)?"
        rf" (on|using|with|running in|wearing) (the |a |an |your )?{v}",
        rf"(?i)\b(i|we)('ve| have) (switched|moved|changed)"
        rf" (to|over to) (the |a |an )?{v}",
    ]
    return [re.compile(p) for p in pats]


def asserts_value_adopted(text: str, value: str) -> bool:
    """True when `text` asserts the value as adopted / in current use —
    the value-assertion ORDERING guard (mere option/recommendation
    mentions do not trip it)."""
    if value not in text and _norm(value) not in _norm(text):
        return False
    if _completed_hit(text, value):
        return True
    return any(p.search(text) for p in _adoption_regexes(value))


# ---------------------------------------------------------------------------
# Geometry (deterministic; computed for all arcs before any call)
# ---------------------------------------------------------------------------

def _rng(arc_id: str, what: str) -> random.Random:
    return random.Random(f"m4r3:{SEED_M4}:{arc_id}:{what}")


def conversation_geometry(arc: dict) -> dict:
    """Sessions/turn counts/beat slots/dates for one arc. Realized pending
    window = user turns strictly after the hedge turn up to and including
    the resolution turn (r2 `_pending_window` semantics)."""
    rng = _rng(arc["arc_id"], "geometry")
    target = arc["pending_window_user_turns"]
    n_sessions = (3 if (arc["resolution_type"] == "open"
                        and target >= OPEN_3SESS_MIN_WINDOW) else 2)
    n_user = [rng.randint(N_USER_LO, N_USER_HI) for _ in range(n_sessions)]
    for _ in range(3 * n_sessions):          # bump capacity toward the target
        mid = sum(n_user[1:-1])
        if target <= n_user[0] - 2 + mid + n_user[-1]:
            break
        i = min(range(n_sessions), key=lambda j: n_user[j])
        if n_user[i] >= N_USER_HI:
            break
        n_user[i] += 1
    mid = sum(n_user[1:-1])
    lo, hi = mid + 1, n_user[0] - 2 + mid + n_user[-1]
    realized = max(lo, min(target, hi))
    t_lo = max(1, realized - mid - (n_user[0] - 2))
    t_hi = min(n_user[-1], realized - mid)
    tail = rng.randint(t_lo, t_hi)
    rest0 = realized - mid - tail
    hedge_user_turn = n_user[0] - 1 - rest0
    resolution_user_turn = tail - 1

    drng = _rng(arc["arc_id"], "dates")
    day = dt.datetime(2023, drng.randint(2, 9), drng.randint(1, 26))
    sessions = []
    for si in range(n_sessions):
        gap = 0 if si == 0 else drng.randint(GAP_DAYS_LO, GAP_DAYS_HI)
        day = day + dt.timedelta(days=gap)
        t = day.replace(hour=drng.randint(8, 21), minute=drng.randint(0, 59))
        turn_dates = []
        for _ in range(2 * n_user[si]):
            turn_dates.append(t.strftime("%Y/%m/%d (%a) %H:%M"))
            t = t + dt.timedelta(minutes=drng.randint(1, 4))
        sessions.append({
            "session_idx": si,
            "date": turn_dates[0],
            "days_since_previous": gap if si else None,
            "n_user_turns": n_user[si],
            "turn_dates": turn_dates,
        })
    return {
        "n_sessions": n_sessions,
        "n_user_turns": n_user,
        "hedge": {"session_idx": 0, "user_turn": hedge_user_turn},
        "resolution": {"session_idx": n_sessions - 1,
                       "user_turn": resolution_user_turn},
        "target_pending_window_user_turns": target,
        "realized_pending_window_user_turns": realized,
        "sessions": sessions,
    }


# ---------------------------------------------------------------------------
# Per-turn directives (word bands + batch-level seeded noise)
# ---------------------------------------------------------------------------

def build_slots(arcs: list[dict],
                geoms: dict[str, dict]) -> list[dict]:
    """One slot per USER turn across all conversations, in deterministic
    (arc, session, turn) order — the unit of the batch noise quotas."""
    slots = []
    for arc in arcs:
        g = geoms[arc["arc_id"]]
        for si, sess in enumerate(g["sessions"]):
            for ui in range(sess["n_user_turns"]):
                slots.append({
                    "arc_id": arc["arc_id"],
                    "session_idx": si,
                    "user_turn": ui,
                    "turn_idx": 2 * ui,
                    "is_hedge": (si == g["hedge"]["session_idx"]
                                 and ui == g["hedge"]["user_turn"]),
                    "is_resolution": (si == g["resolution"]["session_idx"]
                                      and ui == g["resolution"]["user_turn"]),
                    "resolution_type": arc["resolution_type"],
                    "value": arc["value"],
                })
    return slots


def assign_noise(slots: list[dict]) -> None:
    """Batch-level seeded quotas, max ONE feature per slot, with eligibility:
    fragment never on beat turns; exclamation never on the abandoned-arc
    resolution beat."""
    for s in slots:
        s["noise"] = "none"
    order = list(range(len(slots)))
    random.Random(f"m4r3-noise:{SEED_M4}").shuffle(order)
    n = len(slots)

    def eligible(s: dict, feature: str) -> bool:
        if s["noise"] != "none":
            return False
        if feature == "fragment" and (s["is_hedge"] or s["is_resolution"]):
            return False
        if (feature == "exclamation" and s["is_resolution"]
                and s["resolution_type"] == "abandoned"):
            return False
        return True

    for feature, rate in NOISE_QUOTAS.items():
        want = round(rate * n)
        for i in order:
            if want <= 0:
                break
            if eligible(slots[i], feature):
                slots[i]["noise"] = feature
                want -= 1


def assign_word_bands(slots: list[dict]) -> None:
    for s in slots:
        rng = random.Random(f"m4r3:{SEED_M4}:{s['arc_id']}:s{s['session_idx']}"
                            f":u{s['user_turn']}:words")
        if s["noise"] == "fragment":
            t = rng.randint(FRAG_LO, FRAG_HI)
            s["target_words"] = t
            s["words_min"], s["words_max"] = FRAG_LO, FRAG_HI + 1
            continue
        t = round(rng.lognormvariate(math.log(USER_WORDS_MEDIAN),
                                     USER_WORDS_SIGMA))
        t = max(USER_WORDS_LO, min(USER_WORDS_HI, t))
        if s["is_hedge"] or s["is_resolution"]:
            t = max(t, len(s["value"].split()) + BEAT_FLOOR_EXTRA)
        s["target_words"] = t
        s["words_min"] = max(USER_WORDS_LO, t - WORD_TOL)
        s["words_max"] = t + WORD_TOL


# ---------------------------------------------------------------------------
# Exemplars — REAL human-corpus sentences, payload-level ONLY
# ---------------------------------------------------------------------------

def load_exemplar_pool() -> dict[str, list[dict]]:
    ext = json.loads(r2.EXTERNAL_ANALYSIS_PATH.read_text())["examples"]
    return {cat: [{"id": f"ext:{cat}:{i}:{e['corpus']}", "text": e["text"]}
                  for i, e in enumerate(ext[cat])]
            for cat in ("tentative_plan", "completed_change", "abandonment")}


def pick_exemplars(pool: dict, arc: dict, session_idx: int,
                   has_hedge: bool, has_resolution: bool) -> list[dict]:
    rng = _rng(arc["arc_id"], f"exemplars:s{session_idx}")
    picked: list[dict] = []
    if has_hedge:
        picked += rng.sample(pool["tentative_plan"], 3)
    if has_resolution:
        cat = {"resolve_true": "completed_change",
               "abandoned": "abandonment",
               "open": "tentative_plan"}[arc["resolution_type"]]
        pick = [e for e in rng.sample(pool[cat], 3)
                if e["id"] not in {p["id"] for p in picked}]
        picked += pick[:2 if has_hedge else 3]
    everything = [e for c in pool.values() for e in c
                  if e["id"] not in {p["id"] for p in picked}]
    picked += rng.sample(everything, max(0, 6 - len(picked)))
    return picked[:6]


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    toks = re.findall(r"[a-z0-9']+", text.lower())
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def _mask_values(text: str, values: list[str]) -> str:
    for v in values:
        text = text.replace(v, " ")
    return text


# ---------------------------------------------------------------------------
# Lint battery
# ---------------------------------------------------------------------------

_ASCII_OK = re.compile(r"^[\x20-\x7e]*$")
_ASCII_OK_NL = re.compile(r"^[\x20-\x7e\n]*$")
_COLON_NON_TIME = re.compile(r"(?<!\d):|:(?!\d)")


def lint_user_turn(text: str, slot: dict, *, banned: list[dict],
                   prev_assistant: str | None, exemplars: list[dict],
                   mask: list[str]) -> list[str]:
    """Register lints for ONE user turn (arc-integrity checks live in
    lint_session). Failure strings double as lint_feedback lines."""
    where = f"user turn {slot['user_turn']} (turn_idx {slot['turn_idx']})"
    fails: list[str] = []
    if not text.strip():
        return [f"{where}: empty"]
    if not _ASCII_OK.match(text):
        fails.append(f"{where}: non-ASCII character or newline — straight "
                     "apostrophes, single line")
    n_words = len(text.split())
    if not (slot["words_min"] <= n_words <= slot["words_max"]):
        fails.append(f"{where}: {n_words} words, band is "
                     f"[{slot['words_min']}, {slot['words_max']}]")
    if text.count(",") > 2:
        fails.append(f"{where}: more than 2 commas")
    if ";" in text:
        fails.append(f"{where}: semicolon")
    if _COLON_NON_TIME.search(text):
        fails.append(f"{where}: colon (colon-lists are assistant register)")
    if re.search(r"(^|\s)[-*•]\s", text):
        fails.append(f"{where}: bullet/list marker")
    if r2._FORMAL_CONNECTIVE.search(text):
        fails.append(f"{where}: formal connective (However/Additionally/...)")

    # -- noise directive ----------------------------------------------------
    # `fragment` licenses clipped chat style as ONE feature (the frozen
    # render prompt's own example is "ok what about socks"): lowercase
    # start and terminal punctuation are both optional on fragment turns.
    noise = slot["noise"]
    first_alpha = next((c for c in text if c.isalpha()), "")
    if noise == "lowercase_start":
        if first_alpha and not first_alpha.islower():
            fails.append(f"{where}: noise lowercase_start — start lowercase")
    elif (noise != "fragment" and first_alpha and first_alpha.islower()):
        fails.append(f"{where}: start with a capital (no lowercase_start "
                     "directive)")
    if noise == "exclamation":
        if not text.endswith("!"):
            fails.append(f"{where}: noise exclamation — end with '!'")
    elif "!" in text:
        fails.append(f"{where}: no '!' without the exclamation directive")
    if noise == "no_terminal_punct":
        if text[-1] in ".!?":
            fails.append(f"{where}: noise no_terminal_punct — no final "
                         "punctuation")
    elif (noise not in ("exclamation", "fragment")
            and text[-1] not in ".?"):
        fails.append(f"{where}: end with '.' or '?'")
    if noise == "typo_missing_apostrophe":
        if not r2.TYPO_CONTRACTION.search(text):
            fails.append(f"{where}: noise typo_missing_apostrophe — one "
                         "contraction without its apostrophe (dont, im, ...)")
    else:
        if r2.TYPO_CONTRACTION.search(text):
            fails.append(f"{where}: missing-apostrophe contraction without "
                         "the typo directive")
        if (noise not in ("lowercase_start", "fragment")
                and r2.STANDALONE_LOWER_I.search(text)):
            fails.append(f"{where}: standalone lowercase 'i' without the "
                         "typo/lowercase directive")
    if noise == "filler" and not FILLER_REQ.search(text):
        fails.append(f"{where}: noise filler — include btw/haha/honestly/"
                     "so yeah or open with oh/well")
    if noise != "filler":
        if FILLER_BAN.search(text):
            fails.append(f"{where}: filler word without the filler directive")
        if FILLER_START_BAN.search(text):
            fails.append(f"{where}: oh/well opener without the filler "
                         "directive")

    # -- banned patterns ----------------------------------------------------
    for p in banned:
        if p["compiled"].search(text):
            fails.append(f"{where}: banned pattern '{p['name']}' "
                         f"(regex {p['regex']})")
    for name, rx in r2.EXTERNAL_ANTI_COMPILED:
        if rx.search(text):
            fails.append(f"{where}: external anti-pattern {name}")
    low = text.lower()
    for b in BLACKLIST:
        if b.lower() in low:
            fails.append(f"{where}: template-blacklist phrase {b!r}")
    for name, rx in SYCOPHANCY_COMPILED:
        if rx.search(text):
            fails.append(f"{where}: sycophancy/apology ban '{name}' — users "
                         "don't thank, praise, or apologize here")

    # -- no restating the assistant ------------------------------------------
    if prev_assistant:
        shared = (_ngrams(_mask_values(text, mask), RESTATE_NGRAM)
                  & _ngrams(_mask_values(prev_assistant, mask), RESTATE_NGRAM))
        if shared:
            gram = " ".join(sorted(shared)[0])
            fails.append(f"{where}: restates the assistant (shared "
                         f"{RESTATE_NGRAM}-gram '{gram}')")

    # -- exemplar 6-gram ------------------------------------------------------
    grams = _ngrams(text, EXEMPLAR_NGRAM)
    for ex in exemplars:
        shared = grams & _ngrams(ex["text"], EXEMPLAR_NGRAM)
        if shared:
            gram = " ".join(sorted(shared)[0])
            fails.append(f"{where}: shares the 6-gram '{gram}' with exemplar "
                         f"{ex['id']} — reword freshly")
    return fails


def lint_assistant_turn(text: str, turn_idx: int) -> list[str]:
    where = f"assistant turn (turn_idx {turn_idx})"
    fails = []
    if not text.strip():
        return [f"{where}: empty"]
    if not _ASCII_OK_NL.match(text):
        fails.append(f"{where}: non-ASCII character")
    n = len(text.split())
    if n > ASSISTANT_HARD_CAP:
        fails.append(f"{where}: {n} words > {ASSISTANT_HARD_CAP} cap "
                     f"(directed max {ASSISTANT_MAX_WORDS})")
    if n < 3:
        fails.append(f"{where}: implausibly short assistant turn")
    return fails


def _beat_sentence_checks(kind: str, sentence: str, turn_text: str,
                          arc: dict) -> list[str]:
    """Shared hedge/resolution sentence integrity (value, statement,
    substring, single mention in the beat turn)."""
    fails = []
    value = arc["value"]
    if not sentence or not isinstance(sentence, str):
        return [f"{kind}_sentence missing"]
    if sentence not in turn_text:
        fails.append(f"{kind}_sentence must be a verbatim substring of the "
                     f"{kind} turn")
    if "?" in sentence:
        fails.append(f"{kind}_sentence must be a statement, not a question")
    n_val = sentence.count(value)
    if n_val != 1:
        fails.append(f"{kind}_sentence must contain the value {value!r} "
                     f"VERBATIM exactly once (found {n_val})")
    if turn_text.count(value) > 1:
        fails.append(f"the {kind} turn mentions the value more than once — "
                     "only the beat sentence carries it")
    return fails


def lint_session(arc: dict, geom: dict, session_idx: int, out: dict,
                 prior_turns: list[dict], slots: list[dict],
                 banned: list[dict], exemplars: list[dict]) -> list[str]:
    """Full deterministic acceptance lint for one rendered session, in the
    context of the previously rendered sessions (`prior_turns` = flat
    [{role, text}] so far)."""
    fails: list[str] = []
    sess = geom["sessions"][session_idx]
    n_turns = 2 * sess["n_user_turns"]
    turns = out.get("turns")
    if not isinstance(turns, list) or len(turns) != n_turns:
        return [f"need exactly {n_turns} turns (user first, alternating); "
                f"got {len(turns) if isinstance(turns, list) else 'none'}"]
    for k, t in enumerate(turns):
        want = "user" if k % 2 == 0 else "assistant"
        if not isinstance(t, dict) or t.get("role") != want or not isinstance(
                t.get("text"), str):
            return [f"turn {k} must be role={want!r} with a text string"]

    value = arc["value"]
    mask = [value] + [c for c in (arc.get("value_candidates") or [])]
    has_hedge = geom["hedge"]["session_idx"] == session_idx
    has_res = geom["resolution"]["session_idx"] == session_idx
    hedge_ti = 2 * geom["hedge"]["user_turn"]
    res_ti = 2 * geom["resolution"]["user_turn"]
    slot_by_ti = {s["turn_idx"]: s for s in slots}

    # -- per-turn register lints ---------------------------------------------
    for k, t in enumerate(turns):
        if k % 2 == 0:
            prev_assist = None
            if k >= 2:
                prev_assist = turns[k - 1]["text"]
            elif prior_turns:
                prev_assist = prior_turns[-1]["text"]
            fails += lint_user_turn(t["text"], slot_by_ti[k], banned=banned,
                                    prev_assistant=prev_assist,
                                    exemplars=exemplars, mask=mask)
        else:
            fails += lint_assistant_turn(t["text"], k)

    # -- beat sentences --------------------------------------------------------
    hedge_sentence = out.get("hedge_sentence")
    res_sentence = out.get("resolution_sentence")
    if has_hedge:
        fails += _beat_sentence_checks("hedge", hedge_sentence,
                                       turns[hedge_ti]["text"], arc)
        if isinstance(hedge_sentence, str) and hedge_sentence:
            fam = ha.detect_cue_family(hedge_sentence)[0]
            if fam != arc["hedge_strength"]:
                frames = {
                    "deliberating": "'I'm thinking about/of ...' / 'I'm "
                                    "considering ...' / 'can't decide' / "
                                    "'not sure whether'",
                    "intending": "'I might ...' / 'I may ...' / 'I'll "
                                 "probably ...' / 'I'm planning to ...' / "
                                 "'I'm hoping to ...' / 'leaning toward ...'",
                    "scheduled_but_future": "'I'm scheduled to ...' / 'I'm "
                                            "about to ...' / 'once <event> "
                                            "I'll switch ...'",
                }[arc["hedge_strength"]]
                fails.append(
                    f"hedge_sentence must read tentative at strength "
                    f"{arc['hedge_strength']!r} (detected {fam!r}) — open the "
                    f"plan clause with one of exactly these frames: {frames}; "
                    "change nothing else that already passes")
            if _abandon_cue_any(hedge_sentence):
                fails.append("hedge_sentence must not contain abandonment "
                             "language")
            if _completed_hit(hedge_sentence, value):
                fails.append("hedge_sentence must not read as completed — "
                             "nothing has happened yet")
    elif hedge_sentence:
        fails.append("hedge_sentence must be null (no hedge beat in this "
                     "session)")
    if has_res:
        kind = arc["resolution_type"]
        fails += _beat_sentence_checks("resolution", res_sentence,
                                       turns[res_ti]["text"], arc)
        if isinstance(res_sentence, str) and res_sentence:
            fam = ha.detect_cue_family(res_sentence)[0]
            if kind == "resolve_true":
                if not (r2.COMPLETED_CUE.search(res_sentence)
                        or COMPLETED_CUE_R3_EXTRA.search(res_sentence)):
                    fails.append("resolve_true resolution needs a plain "
                                 "completion frame (ended up going with / "
                                 "finally switched to / the order came in)")
                if fam is not None:
                    fails.append("resolve_true resolution must not read "
                                 f"tentative (detected {fam!r} cue)")
                if _abandon_cue_any(res_sentence):
                    fails.append("resolve_true resolution must not contain "
                                 "abandonment language")
            elif kind == "abandoned":
                if not (_abandon_cue_any(res_sentence)):
                    fails.append("abandoned resolution must read as a drop — "
                                 "didn't end up getting it / went with "
                                 "<alternative> instead / a mundane setback")
                if r2._value_inverted(res_sentence, value):
                    fails.append("abandoned resolution asserts the value as "
                                 "KEPT — the value is the road not taken")
                if _completed_hit(res_sentence, value):
                    fails.append("abandoned resolution must not read as a "
                                 "completed adoption of the value")
            else:                                            # open
                if fam is None and not r2.OPEN_CUE.search(res_sentence):
                    fails.append("open still-pending mention must read "
                                 "pending (still / haven't decided yet / "
                                 "might still)")
                if _abandon_cue_any(res_sentence):
                    fails.append("open mention must not contain abandonment "
                                 "language")
                if _completed_hit(res_sentence, value):
                    fails.append("open mention must not read as completed")
    elif res_sentence:
        fails.append("resolution_sentence must be null (no resolution beat "
                     "in this session)")

    # -- ordering / arc-consistency over every turn ----------------------------
    hedge_abs = 2 * geom["hedge"]["user_turn"]        # session 0
    res_sess = geom["resolution"]["session_idx"]
    for k, t in enumerate(turns):
        text = t["text"]
        phase_pre = (session_idx == 0 and k < hedge_abs)
        is_hedge_turn = has_hedge and k == hedge_ti
        is_res_turn = has_res and k == res_ti
        in_window = (not phase_pre and not is_res_turn
                     and (session_idx < res_sess
                          or (session_idx == res_sess and k < res_ti)))
        post = (session_idx == res_sess and k > res_ti)
        if is_hedge_turn and isinstance(hedge_sentence, str):
            text = text.replace(hedge_sentence, " ")
        if is_res_turn and isinstance(res_sentence, str):
            text = text.replace(res_sentence, " ")

        if arc["resolution_type"] == "open":
            if asserts_value_adopted(text, value):
                fails.append(f"turn {k}: open arc — the value must never be "
                             "adopted anywhere in the conversation")
            if _abandon_hit_near_value(text, value):
                fails.append(f"turn {k}: open arc — the value must never be "
                             "dropped anywhere in the conversation")
            continue
        if phase_pre or is_hedge_turn:
            if asserts_value_adopted(text, value):
                fails.append(f"turn {k}: asserts the value as adopted BEFORE "
                             "the hedge — the hedge must be a genuine "
                             "unrealized plan")
        elif in_window:
            if asserts_value_adopted(text, value):
                fails.append(f"turn {k}: adopts the value inside the pending "
                             "window — resolution happens only at the "
                             "directed beat")
            if _abandon_hit_near_value(text, value):
                fails.append(f"turn {k}: drops the value inside the pending "
                             "window — resolution happens only at the "
                             "directed beat")
        elif post or is_res_turn:
            if (arc["resolution_type"] == "abandoned"
                    and asserts_value_adopted(text, value)):
                fails.append(f"turn {k}: abandoned arc — the value must not "
                             "be adopted after the resolution")
            if (arc["resolution_type"] == "resolve_true"
                    and _abandon_hit_near_value(text, value)):
                fails.append(f"turn {k}: resolve_true arc — the value must "
                             "not be dropped after the resolution")
    return fails


def lint_plan(arc: dict, geom: dict, out: dict) -> list[str]:
    fails: list[str] = []
    sessions = out.get("sessions")
    if not isinstance(sessions, list) or len(sessions) != geom["n_sessions"]:
        return [f"need exactly {geom['n_sessions']} sessions"]
    if not isinstance(out.get("premise"), str) or not out["premise"].strip():
        fails.append("premise missing")
    for si, s in enumerate(sessions):
        beats = (s or {}).get("user_turn_beats")
        n_want = geom["sessions"][si]["n_user_turns"]
        if not isinstance(beats, list) or len(beats) != n_want:
            fails.append(f"session {si}: need exactly {n_want} "
                         "user_turn_beats")
            continue
        for b in beats:
            if not isinstance(b, str) or not b.strip():
                fails.append(f"session {si}: empty beat line")
            elif len(b.split()) > 30:
                fails.append(f"session {si}: beat line over 30 words")
            elif not _ASCII_OK.match(b):
                fails.append(f"session {si}: non-ASCII beat line")
    if fails:
        return fails
    value = arc["value"]
    h = geom["hedge"]
    r = geom["resolution"]
    if value not in sessions[h["session_idx"]]["user_turn_beats"][h["user_turn"]]:
        fails.append(f"the hedge beat (session {h['session_idx']} user turn "
                     f"{h['user_turn']}) must contain the value verbatim")
    if value not in sessions[r["session_idx"]]["user_turn_beats"][r["user_turn"]]:
        fails.append(f"the resolution beat (session {r['session_idx']} user "
                     f"turn {r['user_turn']}) must contain the value verbatim")
    return fails


# ---------------------------------------------------------------------------
# Calls
# ---------------------------------------------------------------------------

def plan_payload(arc: dict, geom: dict, attempt: int,
                 feedback: list[str] | None) -> dict:
    payload = {
        "task": "m4_r3_plan",
        "arc_id": arc["arc_id"],
        "key": {k: arc["key"][k] for k in ("entity", "attribute",
                                           "description")},
        "value": arc["value"],
        "value_candidates": arc["value_candidates"],
        "resolution_type": arc["resolution_type"],
        "hedge_strength": arc["hedge_strength"],
        "geometry": {
            "sessions": [{k: s[k] for k in ("session_idx", "date",
                                            "n_user_turns",
                                            "days_since_previous")}
                         for s in geom["sessions"]],
            "hedge_beat": geom["hedge"],
            "resolution_beat": {
                **geom["resolution"],
                "kind": ("still_pending_mention"
                         if arc["resolution_type"] == "open" else
                         "resolution")},
            "target_pending_window_user_turns":
                geom["target_pending_window_user_turns"],
            "realized_pending_window_user_turns":
                geom["realized_pending_window_user_turns"],
        },
        "attempt": attempt,
    }
    if feedback:
        payload["lint_feedback"] = feedback
    return payload


def render_payload(arc: dict, geom: dict, session_idx: int, plan: dict,
                   rendered: list[dict], slots: list[dict],
                   exemplars: list[dict], hedge_sentence: str | None,
                   attempt: int, feedback: list[str] | None) -> dict:
    sess = geom["sessions"][session_idx]
    has_hedge = geom["hedge"]["session_idx"] == session_idx
    has_res = geom["resolution"]["session_idx"] == session_idx
    directives = []
    slot_by_ti = {s["turn_idx"]: s for s in slots}
    for k in range(2 * sess["n_user_turns"]):
        if k % 2 == 0:
            s = slot_by_ti[k]
            directives.append({
                "turn_idx": k, "role": "user",
                "words_min": s["words_min"], "words_max": s["words_max"],
                "noise": s["noise"],
                "is_hedge_turn": bool(s["is_hedge"]),
                "is_resolution_turn": bool(s["is_resolution"]),
                "beat": plan["sessions"][session_idx]["user_turn_beats"][k // 2],
            })
        else:
            directives.append({"turn_idx": k, "role": "assistant",
                               "max_words": ASSISTANT_MAX_WORDS})
    payload = {
        "task": "m4_r3_render_session",
        "arc_id": arc["arc_id"],
        "session_idx": session_idx,
        "n_sessions": geom["n_sessions"],
        "session_date": sess["date"],
        "days_since_previous": sess["days_since_previous"],
        "all_session_dates": [s["date"] for s in geom["sessions"]],
        "key": {k: arc["key"][k] for k in ("entity", "attribute",
                                           "description")},
        "value": arc["value"],
        "value_candidates": arc["value_candidates"],
        "hedge_strength": arc["hedge_strength"] if has_hedge else None,
        "resolution_outcome": (arc["resolution_type"]
                               if (has_res or session_idx
                                   > geom["hedge"]["session_idx"]) else None),
        "premise": plan["premise"],
        "outline_this_session": plan["sessions"][session_idx],
        "all_session_topics": [s.get("topic") for s in plan["sessions"]],
        "previous_sessions": rendered,
        "earlier_hedge_sentence": (hedge_sentence if not has_hedge else None),
        "turn_directives": directives,
        "style_exemplars": exemplars,
        "exemplar_rule": ("REAL human chat sentences: match their register; "
                          "never reuse any 6-word sequence from them"),
        "attempt": attempt,
    }
    if feedback:
        payload["lint_feedback"] = feedback
        payload["fresh_wording_note"] = (
            "Previous attempt failed the lints above. Re-render the session "
            "fixing EVERY listed failure while keeping all rules.")
    return payload


async def plan_conversation(arc: dict, geom: dict,
                            plan_sys: str) -> tuple[dict, int]:
    feedback = None
    for attempt in range(MAX_ATTEMPTS):
        out = await llm.call_llm(
            plan_sys,
            json.dumps(plan_payload(arc, geom, attempt, feedback),
                       ensure_ascii=False),
            trace_tag=f"m4r3-plan-{arc['arc_id']}-a{attempt}")
        fails = lint_plan(arc, geom, out)
        if not fails:
            return out, attempt + 1
        feedback = fails
    raise RuntimeError(f"{arc['arc_id']} plan: lint still failing after "
                       f"{MAX_ATTEMPTS} attempts: {feedback}")


async def render_session(arc: dict, geom: dict, session_idx: int, plan: dict,
                         rendered: list[dict], prior_turns: list[dict],
                         slots: list[dict], pool: dict, banned: list[dict],
                         render_sys: str,
                         hedge_sentence: str | None) -> tuple[dict, dict]:
    has_hedge = geom["hedge"]["session_idx"] == session_idx
    has_res = geom["resolution"]["session_idx"] == session_idx
    exemplars = pick_exemplars(pool, arc, session_idx, has_hedge, has_res)
    feedback = None
    for attempt in range(MAX_ATTEMPTS):
        payload = render_payload(arc, geom, session_idx, plan, rendered,
                                 slots, exemplars, hedge_sentence, attempt,
                                 feedback)
        out = await llm.call_llm(
            render_sys, json.dumps(payload, ensure_ascii=False),
            trace_tag=f"m4r3-render-{arc['arc_id']}-s{session_idx}-a{attempt}")
        fails = lint_session(arc, geom, session_idx, out, prior_turns, slots,
                             banned, exemplars)
        if not fails:
            meta = {"attempts": attempt + 1,
                    "exemplar_ids": [e["id"] for e in exemplars]}
            return out, meta
        feedback = fails
    raise RuntimeError(f"{arc['arc_id']} session {session_idx}: lint still "
                       f"failing after {MAX_ATTEMPTS} attempts: {feedback}")


# ---------------------------------------------------------------------------
# Arc assembly
# ---------------------------------------------------------------------------

CARRIED_FIELDS = (
    "arc_id", "row_kind", "stratum", "benchmark", "question_id",
    "question_type", "resolution_type", "gold_resolution_status",
    "hedge_strength", "key", "pending_window_user_turns", "seed",
    "seed_draw_index", "value", "value_candidates")


def assert_plan_carryover(arcs_r2: list[dict]) -> None:
    """The r3 conversations inherit the frozen plan identity: ids, keys,
    values, labels, strengths, window targets, seeds — byte-equal to the
    retired r2 rows (which themselves byte-matched arc_plan.jsonl)."""
    plan = {a["arc_id"]: a for a in ha._read_jsonl(ha.ARC_PLAN_PATH)}
    if set(plan) != {a["arc_id"] for a in arcs_r2}:
        raise SystemExit("arc id set differs from the frozen arc_plan.jsonl")
    for a in arcs_r2:
        p = plan[a["arc_id"]]
        for f in ("arc_id", "row_kind", "benchmark", "question_id",
                  "question_type", "resolution_type", "gold_resolution_status",
                  "hedge_strength", "key", "pending_window_user_turns",
                  "seed", "seed_draw_index"):
            if a[f] != p[f]:
                raise SystemExit(f"{a['arc_id']}: plan field {f!r} drifted")
        if not a.get("value") or not a.get("value_candidates"):
            raise SystemExit(f"{a['arc_id']}: missing frozen value")
        if not ha._fm4_probe_ok(a["key"]["probe_question"], a["value"]):
            raise SystemExit(f"{a['arc_id']}: FM-4 violation — value in "
                             "probe question")


def assemble_arc(src: dict, geom: dict, plan: dict, sessions_out: list[dict],
                 slots: list[dict], metas: list[dict], plan_attempts: int,
                 plan_sha: str, render_sha: str) -> dict:
    arc = {k: src[k] for k in CARRIED_FIELDS}
    h, r = geom["hedge"], geom["resolution"]
    hedge_ti, res_ti = 2 * h["user_turn"], 2 * r["user_turn"]
    conversation = []
    for si, out in enumerate(sessions_out):
        sess = geom["sessions"][si]
        conversation.append({
            "session_idx": si,
            "date": sess["date"],
            "turns": [{"role": t["role"], "text": t["text"],
                       "date": sess["turn_dates"][k]}
                      for k, t in enumerate(out["turns"])],
        })
    hedge_sentence = sessions_out[h["session_idx"]]["hedge_sentence"]
    res_sentence = sessions_out[r["session_idx"]]["resolution_sentence"]
    src_res = src["resolution"]
    arc.update({
        "hedge": {
            "inj_type": "m4_hedge", "contract": "SPECULATION",
            "construction_label": "speculative",
            "session_idx": h["session_idx"], "turn_idx": hedge_ti,
            "position": "in_turn",
            "date": geom["sessions"][h["session_idx"]]["turn_dates"][hedge_ti],
        },
        "resolution": {
            "inj_type": src_res["inj_type"], "contract": src_res["contract"],
            "construction_label": src_res["construction_label"],
            "t3_variant_kind": src_res.get("t3_variant_kind"),
            "session_idx": r["session_idx"], "turn_idx": res_ti,
            "position": "in_turn",
            "date": geom["sessions"][r["session_idx"]]["turn_dates"][res_ti],
        },
        "hedge_text": hedge_sentence,
        "resolution_text": res_sentence,
        "conversation": conversation,
        "realized_pending_window_user_turns":
            geom["realized_pending_window_user_turns"],
        "render_round": 3,
        "plan_prompt_sha": plan_sha,
        "render_prompt_sha": render_sha,
        "lints": {
            "hedge": ha._lint(hedge_sentence),
            "resolution": ha._lint(res_sentence),
            "fm4_probe_value_free": True,
            "r3_lint": "passed (battery in hedge_arcs_r3)",
        },
        "render": {
            "pipeline": ("m4 r3 themed synthetic conversation "
                         "(prompts/m4_conv_plan_r3.txt + "
                         "m4_conv_render_r3.txt)"),
            "plan_prompt_sha": plan_sha,
            "render_prompt_sha": render_sha,
            "style": "human user register + bounded assistant register "
                     "(M4 r3 contract)",
        },
        "r3": {
            "geometry": {k: geom[k] for k in
                         ("n_sessions", "n_user_turns", "hedge", "resolution",
                          "target_pending_window_user_turns",
                          "realized_pending_window_user_turns")},
            "plan_outline": plan,
            "noise_ledger": [{k: s[k] for k in
                              ("session_idx", "user_turn", "turn_idx",
                               "noise", "target_words", "words_min",
                               "words_max", "is_hedge", "is_resolution")}
                             for s in slots],
            "attempts": {"plan": plan_attempts,
                         "sessions": [m["attempts"] for m in metas]},
            "exemplar_ids": {str(si): m["exemplar_ids"]
                             for si, m in enumerate(metas)},
        },
        "status": "pending_owner_review",
        "usable": False,
    })
    arc["manifest_rows"] = ha.manifest_rows_for_arc(arc)
    return arc


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------

async def run_generate(limit: int | None, fresh_cap: int | None) -> dict:
    arcs_r2 = ha._read_jsonl(ARCS_R2_PATH)
    if len(arcs_r2) != 60:
        raise SystemExit(f"expected 60 retired r2 arcs at {ARCS_R2_PATH}")
    assert_plan_carryover(arcs_r2)
    plan_sys = PLAN_PROMPT_PATH.read_text()
    render_sys = RENDER_PROMPT_PATH.read_text()
    plan_sha = hashlib.sha256(plan_sys.encode()).hexdigest()
    render_sha = hashlib.sha256(render_sys.encode()).hexdigest()
    banned = r2.load_banned_patterns()
    pool = load_exemplar_pool()

    geoms = {a["arc_id"]: conversation_geometry(a) for a in arcs_r2}
    slots = build_slots(arcs_r2, geoms)
    assign_noise(slots)
    assign_word_bands(slots)
    slots_by_arc: dict[str, dict[int, list[dict]]] = {}
    for s in slots:
        slots_by_arc.setdefault(s["arc_id"], {}).setdefault(
            s["session_idx"], []).append(s)

    existing = {a["arc_id"]: a for a in ha._read_jsonl(ARCS_PATH)
                if a.get("render_round") == 3 and a.get("conversation")}
    todo = [a for a in arcs_r2 if a["arc_id"] not in existing]
    if limit is not None:
        todo = todo[:limit]
    done: dict[str, dict] = dict(existing)
    failures: list[str] = []

    async def one(src: dict) -> None:
        try:
            geom = geoms[src["arc_id"]]
            plan, plan_attempts = await plan_conversation(src, geom, plan_sys)
            sessions_out: list[dict] = []
            metas: list[dict] = []
            rendered: list[dict] = []
            prior_turns: list[dict] = []
            hedge_sentence: str | None = None
            for si in range(geom["n_sessions"]):
                out, meta = await render_session(
                    src, geom, si, plan, rendered, prior_turns,
                    slots_by_arc[src["arc_id"]][si], pool, banned,
                    render_sys, hedge_sentence)
                sessions_out.append(out)
                metas.append(meta)
                rendered.append({"session_idx": si,
                                 "date": geom["sessions"][si]["date"],
                                 "turns": out["turns"]})
                prior_turns.extend(out["turns"])
                if geom["hedge"]["session_idx"] == si:
                    hedge_sentence = out["hedge_sentence"]
            done[src["arc_id"]] = assemble_arc(
                src, geom, plan, sessions_out,
                [s for sl in slots_by_arc[src["arc_id"]].values() for s in sl],
                metas, plan_attempts, plan_sha, render_sha)
        except Exception as exc:
            failures.append(f"{src['arc_id']}: {exc}")

    guard = ha.m4_guard(fresh_cap)
    guard.install()
    try:
        await asyncio.gather(*(one(a) for a in todo))
    finally:
        guard.uninstall()

    rows = [done[a["arc_id"]] for a in arcs_r2 if a["arc_id"] in done]
    ha._write_jsonl(ARCS_PATH, rows)
    attempts_gt1 = {}
    for a in rows:
        att = a["r3"]["attempts"]
        if att["plan"] > 1:
            attempts_gt1[f"{a['arc_id']}:plan"] = att["plan"]
        for si, n in enumerate(att["sessions"]):
            if n > 1:
                attempts_gt1[f"{a['arc_id']}:s{si}"] = n
    noise_counts: dict[str, int] = {}
    for s in slots:
        noise_counts[s["noise"]] = noise_counts.get(s["noise"], 0) + 1
    windows = [(a["pending_window_user_turns"],
                a["realized_pending_window_user_turns"]) for a in rows]
    summary = {
        "id": "V2.4-M4-R3-GENERATE",
        "prereg": ("protocol/CHANGELOG.md 2026-07-09 'M4 r3 generation "
                   "contract FROZEN' (plan sha " + plan_sha + ", render sha "
                   + render_sha + ")"),
        "n_planned": len(arcs_r2),
        "n_generated": len(rows),
        "plan_carryover_ok": True,
        "plan_prompt_sha": plan_sha,
        "render_prompt_sha": render_sha,
        "banned_patterns_enforced": len(banned),
        "n_user_turn_slots": len(slots),
        "noise_directive_counts": noise_counts,
        "session_counts": {str(k): sum(1 for a in arcs_r2
                                       if geoms[a["arc_id"]]["n_sessions"]
                                       == k) for k in (2, 3)},
        "realized_window": {
            "n_exact": sum(1 for t, r_ in windows if t == r_),
            "mean_abs_dev": (round(statistics.fmean(
                abs(t - r_) for t, r_ in windows), 3) if windows else None),
            "max_abs_dev": (max((abs(t - r_) for t, r_ in windows),
                                default=None)),
        },
        "attempts_gt1": attempts_gt1,
        "budget": {**ha.advisory_note(), "fresh_calls_this_run": guard.fresh},
        "failures": failures,
        "outputs": {"arcs": ha._rel(ARCS_PATH)},
    }
    ha._write_json(R3_SUMMARY_PATH, summary)
    if failures:
        for f in failures:
            print(f"R3 GENERATE FAILURE: {f}", file=sys.stderr)
    return summary


# ---------------------------------------------------------------------------
# Integrity relint (over the stored conversations, no LLM)
# ---------------------------------------------------------------------------

def relint_arcs(arcs: list[dict]) -> dict:
    banned = r2.load_banned_patterns()
    pool = load_exemplar_pool()
    by_id = {e["id"]: e for c in pool.values() for e in c}
    failures: dict[str, list[str]] = {}
    for arc in arcs:
        geom = conversation_geometry(arc)
        slots = build_slots([arc], {arc["arc_id"]: geom})
        # slots must be re-derived inside the full batch for noise; recover
        # the recorded ledger instead (the frozen directives of record).
        ledger = {(s["session_idx"], s["turn_idx"]): s
                  for s in arc["r3"]["noise_ledger"]}
        for s in slots:
            rec = ledger[(s["session_idx"], s["turn_idx"])]
            s.update({k: rec[k] for k in ("noise", "target_words",
                                          "words_min", "words_max")})
        by_sess: dict[int, list[dict]] = {}
        for s in slots:
            by_sess.setdefault(s["session_idx"], []).append(s)
        fails: list[str] = []
        # date monotonicity + alternation
        last = None
        for sess in arc["conversation"]:
            for k, t in enumerate(sess["turns"]):
                want = "user" if k % 2 == 0 else "assistant"
                if t["role"] != want:
                    fails.append(f"s{sess['session_idx']}t{k}: role order")
                d = re.sub(r" \(\w+\)", "", t["date"])
                stamp = dt.datetime.strptime(d, "%Y/%m/%d %H:%M")
                if last is not None and stamp <= last:
                    fails.append(f"s{sess['session_idx']}t{k}: dates not "
                                 "strictly monotonic")
                last = stamp
        prior: list[dict] = []
        for si, sess in enumerate(arc["conversation"]):
            out = {"turns": [{"role": t["role"], "text": t["text"]}
                             for t in sess["turns"]],
                   "hedge_sentence": (arc["hedge_text"]
                                      if geom["hedge"]["session_idx"] == si
                                      else None),
                   "resolution_sentence": (
                       arc["resolution_text"]
                       if geom["resolution"]["session_idx"] == si else None)}
            exemplars = [by_id[i] for i in
                         arc["r3"]["exemplar_ids"][str(si)]]
            fails += lint_session(arc, geom, si, out, prior,
                                  by_sess[si], banned, exemplars)
            prior.extend(out["turns"])
        if not ha._fm4_probe_ok(arc["key"]["probe_question"], arc["value"]):
            fails.append("FM-4: value in probe question")
        if arc["realized_pending_window_user_turns"] != geom[
                "realized_pending_window_user_turns"]:
            fails.append("realized window drifted from deterministic geometry")
        if fails:
            failures[arc["arc_id"]] = fails
    return {"n_arcs": len(arcs), "all_green": not failures,
            "failures": failures}


# ---------------------------------------------------------------------------
# Register check — r3 user turns vs human corpora vs host LME user vs r2
# ---------------------------------------------------------------------------

def _pooled_human_register(ext_stats: dict) -> dict:
    """n-weighted pooled rates over the four HUMAN corpora (cc_llm excluded:
    it is the LLM contrast point, not ground truth)."""
    human = {k: v for k, v in ext_stats.items() if k != "cc_llm"}
    n_total = sum(v["n"] for v in human.values())

    def wavg(path):
        return round(sum(v["n"] * path(v) for v in human.values()) / n_total, 2)

    return {
        "corpora": sorted(human),
        "n_utts": n_total,
        "len_median_range": [min(v["length_words"]["median"]
                                 for v in human.values()),
                             max(v["length_words"]["median"]
                                 for v in human.values())],
        "len_mean_weighted": wavg(lambda v: v["length_words"]["mean"]),
        "pct_with_contraction": wavg(lambda v: 100 * v["contraction_rate"]),
        "pct_exclamation": wavg(lambda v: 100 * v["exclamation_rate"]),
        "pct_question": wavg(lambda v: 100 * v["question_rate"]),
        "pct_lowercase_start": wavg(lambda v: 100 * v["lowercase_start_rate"]),
        "pct_no_end_punct": wavg(lambda v: 100 * v["no_end_punct_rate"]),
        "pct_typo": wavg(lambda v: 100 * v["typo_rate"]),
        "pct_fragment": wavg(lambda v: 100 * v["fragment_rate"]),
    }


def run_register_check() -> dict:
    banned = r2.load_banned_patterns()
    arcs = ha._read_jsonl(ARCS_PATH)
    if not arcs:
        raise SystemExit(f"no r3 arcs at {ARCS_PATH}")
    user_turns = [t["text"] for a in arcs for s in a["conversation"]
                  for t in s["turns"] if t["role"] == "user"]
    assistant_turns = [t["text"] for a in arcs for s in a["conversation"]
                       for t in s["turns"] if t["role"] == "assistant"]
    r2_rows = ha._read_jsonl(ARCS_R2_PATH)
    r2_texts = [a[k] for a in r2_rows
                for k in ("hedge_text", "resolution_text")]
    hosts = json.loads(r2.HOSTS_ANALYSIS_PATH.read_text())["register_stats"]
    ext = json.loads(r2.EXTERNAL_ANALYSIS_PATH.read_text())["register_stats"]
    syco_hits = [t for t in user_turns
                 if any(rx.search(t) for _, rx in SYCOPHANCY_COMPILED)]
    anti_hits = [t for t in user_turns
                 if any(rx.search(t) for _, rx in r2.EXTERNAL_ANTI_COMPILED)]
    integrity = relint_arcs(arcs)
    balance = {t: sum(1 for a in arcs if a["resolution_type"] == t)
               for t in ha.RESOLUTION_TYPES}
    windows = [(a["pending_window_user_turns"],
                a["realized_pending_window_user_turns"]) for a in arcs]
    doc = {
        "id": "V2.4-M4-R3-REGISTER-CHECK",
        "method": ("register_analysis_hosts.json metric functions "
                   "(hedge_arcs_r2.register_metrics) over ALL r3 user turns; "
                   "r2 sentences recomputed with the same functions; host "
                   "row copied from the frozen hosts analysis; human-corpus "
                   "rows copied from the frozen external analysis (plus an "
                   "n-weighted pooled summary over the four human corpora)"),
        "banned_regex_union": {
            "n_patterns": len(banned),
            "r3_user_turns": r2.banned_union_hits(user_turns, banned),
            "r2_sentences_flagged":
                r2.banned_union_hits(r2_texts, banned)["n_flagged"],
        },
        "sycophancy_union": {"n_patterns": len(SYCOPHANCY_COMPILED),
                             "n_flagged": len(syco_hits),
                             "flagged_texts": syco_hits},
        "external_anti_pattern_union": {"n_flagged": len(anti_hits),
                                        "flagged_texts": anti_hits},
        "register": {
            "r3_user_turns": r2.register_metrics(user_turns),
            "r3_assistant_turns": r2.register_metrics(assistant_turns),
            "r2_sentences": r2.register_metrics(r2_texts),
            "host_lme_user": hosts["lme_user"],
            "human_corpora": {k: ext[k] for k in sorted(ext)
                              if k != "cc_llm"},
            "human_pooled": _pooled_human_register(ext),
        },
        "balance": balance,
        "realized_window": {
            "pairs_target_realized": windows,
            "n_exact": sum(1 for t, r_ in windows if t == r_),
            "mean_abs_dev": round(statistics.fmean(abs(t - r_)
                                                   for t, r_ in windows), 3),
            "max_abs_dev": max(abs(t - r_) for t, r_ in windows),
        },
        "integrity_relint": integrity,
    }
    ha._write_json(R3_CHECK_PATH, doc)
    return doc


# ---------------------------------------------------------------------------
# Review pack (r3 — same structural shape; context from the SAME conversation)
# ---------------------------------------------------------------------------

def _turn_line(turn: dict, *, bold: str | None = None,
               limit_words: int = 40) -> str:
    text = turn["text"]
    if bold and bold in text:
        text = text.replace(bold, f"**{bold}**")
    elif not bold:
        text = ha._excerpt(text, limit_words)
    return f"> _{turn['role']}:_ {text}"


def _context_block_r3(arc: dict, session_idx: int,
                      turn_idx: int, sentence: str) -> list[str]:
    turns = arc["conversation"][session_idx]["turns"]
    lines = []
    if turn_idx > 0:
        lines.append(_turn_line(turns[turn_idx - 1]))
    lines.append(_turn_line(turns[turn_idx], bold=sentence))
    if turn_idx + 1 < len(turns):
        lines.append(_turn_line(turns[turn_idx + 1]))
    return lines


def render_review_pack_r3(arcs: list[dict]) -> str:
    lines = [
        "# v2.4 M4 — synthetic hedge-arc review pack (OWNER pass required)",
        "",
        "Round 3: each arc is a self-contained fully synthetic multi-session "
        "conversation themed on the key's domain (r3 contract, "
        "protocol/CHANGELOG.md 2026-07-09). Per the M4 preregistration this "
        "review is the sole human step: every arc needs an explicit ACCEPT "
        "or REJECT before it can enter the pool. Judge REALISM (would this "
        "user say this, here?) and VALIDITY (hedge is a real unrealized "
        "plan; resolution matches its label). Tick exactly one box per arc, "
        "then record the verdicts in "
        f"`{ha._rel(ha.OWNER_REVIEW_PATH)}` (template: "
        f"`{ha._rel(ha.REVIEW_TEMPLATE_PATH)}`).",
        "",
        f"Arcs: {len(arcs)} generated / 60 planned. Balance: "
        + ", ".join(f"{t}={sum(1 for a in arcs if a['resolution_type'] == t)}"
                    for t in ha.RESOLUTION_TYPES) + ".",
        "",
        "The arcs are UNUSABLE until this review lands: the pool emitter "
        "refuses to run without owner_review.json.",
        "",
    ]
    for a in arcs:
        h, r = a["hedge"], a["resolution"]
        flags = []
        if a["lints"]["hedge"]["realized_cue_family"] != a["hedge_strength"]:
            flags.append("hedge cue family mismatch")
        lines += [
            f"## {a['arc_id']}",
            "",
            f"- key `user/{a['key']['attribute']}`; value **{a['value']}**; "
            f"theme: {a['r3']['plan_outline']['premise']}",
            f"- labels: hedge `speculative` (strength {a['hedge_strength']}); "
            f"resolution `{r['construction_label']}` "
            f"(type **{a['resolution_type']}**, gold status "
            f"`{a['gold_resolution_status']}`)",
            f"- placement: hedge s{h['session_idx']}t{h['turn_idx']} "
            f"({h['date']}) -> resolution s{r['session_idx']}t{r['turn_idx']} "
            f"({r['date']}); window target "
            f"{a['pending_window_user_turns']} user turns, realized "
            f"{a['realized_pending_window_user_turns']}",
        ]
        if flags:
            lines.append(f"- FLAGS: {'; '.join(flags)}")
        lines += ["", "**Hedge context** (session date "
                  f"{a['conversation'][h['session_idx']]['date']}):", ""]
        lines += _context_block_r3(a, h["session_idx"], h["turn_idx"],
                                   a["hedge_text"])
        lines += ["", "**Resolution context** (session date "
                  f"{a['conversation'][r['session_idx']]['date']}):", ""]
        if a.get("resolution_text"):
            lines += _context_block_r3(a, r["session_idx"], r["turn_idx"],
                                       a["resolution_text"])
        else:                          # open arc with no pending mention
            closing = a["conversation"][-1]["turns"][-3:]
            lines += [_turn_line(t) for t in closing]
            lines.append("> (no resolution — plan left open)")
        lines += [
            "",
            f"- [ ] ACCEPT `{a['arc_id']}`",
            f"- [ ] REJECT `{a['arc_id']}` — reason: ____________",
            "",
        ]
    return "\n".join(lines)


def run_review_pack() -> dict:
    arcs = ha._read_jsonl(ARCS_PATH)
    if not arcs:
        raise SystemExit(f"no arcs at {ARCS_PATH} — run r3-generate first")
    ha.REVIEW_PACK_PATH.write_text(render_review_pack_r3(arcs))
    template = {
        "reviewer": "",
        "review_date": "",
        "note": ("copy to owner_review.json, fill reviewer/review_date and "
                 "one verdict (accept|reject) per arc; the pool emitter "
                 "requires full coverage"),
        "decisions": {a["arc_id"]: {"verdict": None, "note": ""}
                      for a in arcs},
    }
    ha._write_json(ha.REVIEW_TEMPLATE_PATH, template)
    return {"arcs": len(arcs), "pack": ha._rel(ha.REVIEW_PACK_PATH),
            "template": ha._rel(ha.REVIEW_TEMPLATE_PATH)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="v2.4 M4 r3 — themed synthetic conversations")
    sub = ap.add_subparsers(dest="cmd", required=True)
    gen = sub.add_parser("r3-generate", help="generate all 60 conversations")
    gen.add_argument("--limit", type=int, default=None)
    gen.add_argument("--fresh-cap", type=int, default=None)
    sub.add_parser("register-check",
                   help="r3 user turns vs human corpora vs host vs r2 "
                        "+ integrity relint")
    sub.add_parser("review-pack", help="rebuild review_pack.md (r3 contexts)")
    args = ap.parse_args()

    if args.cmd == "r3-generate":
        summary = asyncio.run(run_generate(args.limit, args.fresh_cap))
        print(json.dumps({k: summary[k] for k in (
            "n_planned", "n_generated", "n_user_turn_slots",
            "noise_directive_counts", "session_counts", "realized_window",
            "attempts_gt1", "budget", "failures")}, indent=1))
        print(f"-> {ARCS_PATH}\n-> {R3_SUMMARY_PATH}")
        if summary["failures"]:
            raise SystemExit(f"{len(summary['failures'])} arc(s) failed "
                             "(progress persisted; rerun resumes)")
    elif args.cmd == "register-check":
        doc = run_register_check()
        print(json.dumps({
            "banned_union_r3_user_turns": {
                k: doc["banned_regex_union"]["r3_user_turns"][k]
                for k in ("n_texts", "n_flagged", "per_pattern_nonzero")},
            "sycophancy_flagged": doc["sycophancy_union"]["n_flagged"],
            "r3_user_words": doc["register"]["r3_user_turns"]["words"],
            "balance": doc["balance"],
            "realized_window": {k: doc["realized_window"][k]
                                for k in ("n_exact", "mean_abs_dev",
                                          "max_abs_dev")},
            "integrity_all_green": doc["integrity_relint"]["all_green"],
        }, indent=1))
        print(f"-> {R3_CHECK_PATH}")
    elif args.cmd == "review-pack":
        print(json.dumps(run_review_pack(), indent=1))


if __name__ == "__main__":
    main()
