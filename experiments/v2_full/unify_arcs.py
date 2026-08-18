"""unify_arcs.py — SUBSTRATE UNIFICATION: the graded-difficulty factors of the
planted-probe suite (M2), regenerated as full conversation-benchmark-style
conversations under the M4 r4 realism contract (prereg draft:
protocol/drafts/unify_prereg.md; r4 contract: protocol/CHANGELOG.md 2026-07-09
"M4 r4 generation contract FROZEN" — followed VERBATIM wherever applicable).

TWO FACTOR LADDERS, 180 conversations:

  (1) HEDGE-STRENGTH ladder — 5 levels x 20, matching the planted gradient
      (graded_eval.GRADIENT_LEVELS semantics):
        g1_strong_hedge      explicit deliberation            outcome still_open
        g2_weak_hedge        exactly one soft hedge           outcome still_open
        g3_confident_plan    settled, NO hedge words, NO date outcome still_open
        g4_scheduled_event   locked in WITH an explicit date  outcome still_open
        g5_completed_control present-tense completed change   already_completed
      g1-g4 use still_open outcomes so the contamination endpoint (hedged
      value asserted as current in the final store) is the direct analog of
      the planted gradient; g5 is the commit-is-CORRECT control (M2 rule (d)).

  (2) CONFIRMATION-WORDING ladder — 4 distances x 20, all came_true
      (graded_eval.NEARMISS_LEVELS semantics): the hedge is a fixed
      g2-strength weak hedge; the resolution-as-opener reports the switch at
      the directed wording distance p1_verbatim / p2_paraphrase / p3_oblique /
      p4_situational, then the consequence tail carries (r4 CHANGE 1).

CARRIED r4 CONTRACT: geometry (prior-establish at s0u0, change-statement beat
at s0 u>=1, final beat = final-session user turn 0 opener, optional distractor
middle session, window clamp <= 11, 2023 dates, monotonic timestamps), the r3
register battery VERBATIM per user turn (32 banned regexes, external
anti-patterns, BLACKLIST, sycophancy bans, no-restating 8-gram, word bands,
noise-directive ledger, no-6-gram vs human-corpus exemplars), the prior-value
establishment + frozen extract/map/classify self-check, coherence lints, and
retry-with-lint-feedback <= 3. NEW: factor-MANIFESTATION lints (strength-cue
families per level, wording-distance heuristics per distance) and a
cross-level cue-contamination sweep (no hedge-cue-family hit next to the
value/candidates/prior outside the beat sentence, anywhere).

VALUES (v5 delta): no frozen plan exists — keys (frozen value-free AUX_POOL
probe questions) are registered deterministically BEFORE any call (FM-4,
results/unify/unify_keys.jsonl); the planner samples value_candidates (VALUE =
first) + prior_value per arc under code lints.

BUDGET: every fresh call under SharedCallBudget stage "unify" (ADVISORY
allocation 20,000; 120k envelope hard per the 2026-07-08 amendment) plus a
build-local HARD guard: generation refuses to run past a projected TOTAL
envelope spend of 110,000 (headroom protection for the concurrent
Claude-replication stage).

CLI (repo root, pilot/.venv/bin/python experiments/v2_full/unify_arcs.py ...):
  generate      [--limit K] [--fresh-cap N]   spec-freeze + plan + render all
                                              180 -> results/unify/arcs.jsonl
  checks                                      register + factor manifestation
                                              + prior classification
                                              -> results/unify/unify_checks.json
  write-panels  [--out DIR]                   label-free per-arc transcripts
                                              (blind-panel INPUTS; the panels
                                              themselves are maintainer-run)
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

import forecast_v22
import graded_eval as ge
import hedge_arcs as ha
import hedge_arcs_r2 as r2
import hedge_arcs_r3 as r3
import hedge_arcs_r4 as r4
import llm
from inject import AUX_POOL, _norm

from paths import PROMPTS_DIR, RESULTS_DIR

SEED_UNIFY = "unify:20260811"            # prereg: seed keyed by arc_id
STAGE = "unify"
UNIFY_ADVISORY_ALLOCATION = 20_000       # advisory (2026-07-08 amendment)
ENVELOPE_SOFT_STOP = 110_000             # build-local HARD stop (prereg draft)

UNIFY_DIR = RESULTS_DIR / "unify"
SPEC_PATH = UNIFY_DIR / "arc_spec.jsonl"
KEYS_PATH = UNIFY_DIR / "unify_keys.jsonl"
ARCS_PATH = UNIFY_DIR / "arcs.jsonl"
SUMMARY_PATH = UNIFY_DIR / "unify_render_summary.json"
CHECKS_PATH = UNIFY_DIR / "unify_checks.json"
PLAN_PROMPT_PATH = PROMPTS_DIR / "unify_conv_plan_v5.txt"
RENDER_PROMPT_PATH = PROMPTS_DIR / "unify_conv_render_v5.txt"

MAX_ATTEMPTS = 4                         # 1 initial + <=3 lint-feedback retries
N_PER_CELL = 20
RENDER_ROUND = "unify_v5"

# ---------------------------------------------------------------------------
# Factor-level tables (semantics = the frozen M2 level definitions)
# ---------------------------------------------------------------------------

HS_LEVELS = ("g1_strong_hedge", "g2_weak_hedge", "g3_confident_plan",
             "g4_scheduled_event", "g5_completed_control")
CW_LEVELS = ("p1_verbatim", "p2_paraphrase", "p3_oblique", "p4_situational")
ALL_LEVELS = HS_LEVELS + CW_LEVELS

#: outcomes: g1-g4 still_open -> "open"; g5 -> "completed_control";
#: wording ladder came_true -> "resolve_true".
OUTCOME_BY_LEVEL = {
    "g1_strong_hedge": "open", "g2_weak_hedge": "open",
    "g3_confident_plan": "open", "g4_scheduled_event": "open",
    "g5_completed_control": "completed_control",
    **{lv: "resolve_true" for lv in CW_LEVELS},
}
GOLD_STATUS = {"open": "unresolved", "resolve_true": "resolved_true",
               "completed_control": "completed_at_stimulus"}
#: frozen 3-way taxonomy strength carried as metadata (M2 GRADIENT_LEVELS
#: hedge_strength column; wording arcs stage at g2 strength).
HEDGE_STRENGTH_3WAY = {
    "g1_strong_hedge": "deliberating", "g2_weak_hedge": "intending",
    "g3_confident_plan": "intending", "g4_scheduled_event": "scheduled_but_future",
    "g5_completed_control": None, **{lv: "intending" for lv in CW_LEVELS},
}
_GE_GRADIENT = {row["level"]: row for row in ge.GRADIENT_LEVELS}
_GE_NEARMISS = {row["level"]: row for row in ge.NEARMISS_LEVELS}
M2_LEVEL_NAME = {  # this build's level name -> the M2 manifest level name
    "g5_completed_control": "g5_present_tense_change",
    **{lv: lv for lv in ALL_LEVELS if lv != "g5_completed_control"},
}


def level_gold(level: str) -> dict:
    """designed-gold fields, verbatim from the frozen M2 tables."""
    m2 = M2_LEVEL_NAME[level]
    if level in HS_LEVELS:
        row = _GE_GRADIENT[m2]
        return {k: row[k] for k in ("contract", "construction_label",
                                    "designed_gold_class",
                                    "designed_gold_route", "taxonomy_rule")}
    row = ge.NEARMISS_STAGING
    return {"contract": row["contract"],
            "construction_label": row["construction_label"],
            "designed_gold_class": row["designed_gold_class"],
            "designed_gold_route": row["designed_gold_route"],
            "taxonomy_rule": None}


LEVEL_NOTES = {
    "g1_strong_hedge": ("strong hedge: explicit open deliberation (torn "
                        "between / still mulling / can't decide / thinking "
                        "about); clearly unresolved; no might/probably/"
                        "planning-to, no date or weekday"),
    "g2_weak_hedge": ("weak hedge: exactly ONE soft prospective hedge (might "
                      "/ probably / planning to / hoping to / leaning "
                      "toward); no strong deliberation words, no date"),
    "g3_confident_plan": ("confident plan: settled and definite, NO hedging "
                          "words at all, NO calendar date / weekday / "
                          "tomorrow; the switch is still ahead"),
    "g4_scheduled_event": ("scheduled event: locked in and stated as bare "
                           "fact WITH an explicit month-day date, weekday, "
                           "or 'tomorrow'; no hedging words; NOT yet done"),
    "g5_completed_control": ("completed control: the switch already happened "
                             "— assert VALUE as current present-tense state; "
                             "do not narrate the change event; no plan or "
                             "hedge language"),
    "p1_verbatim": ("verbatim confirmation: report the switch went through "
                    "reusing the hedge sentence's own key wording, naming "
                    "VALUE"),
    "p2_paraphrase": ("paraphrase confirmation: completely fresh wording — "
                      "no 6-word sequence shared with the hedge sentence — "
                      "still naming VALUE with a plain completion frame"),
    "p3_oblique": ("oblique confirmation: asserted as done but referred back "
                   "only obliquely; VALUE appears; do NOT use the key's "
                   "domain nouns; no shared 6-word sequence"),
    "p4_situational": ("situational implication: NEVER state the change "
                       "happened, NO completion verbs; a concrete first "
                       "experience that only makes sense if the switch "
                       "already happened; VALUE at most incidental"),
}

#: resolution-slot metadata per outcome
RESOLUTION_META = {
    "open": {"inj_type": "unify_open_mention",
             "contract": "T3_DISTRACTOR_CONTINUED",
             "construction_label": "speculative"},
    "resolve_true": {"inj_type": "unify_resolution", "contract": "T3_PROMOTE",
                     "construction_label": "completed_event"},
    "completed_control": {"inj_type": "unify_continuation",
                          "contract": "STATE_CONTINUATION",
                          "construction_label": "completed_event"},
}

# ---------------------------------------------------------------------------
# Deterministic cue instruments (level manifestation + cross-level purity)
# ---------------------------------------------------------------------------

_DELIB_RES = [re.compile(p, re.I) for p in ha.HEDGE_CUE_FAMILIES["deliberating"]]
_INTEND_RES = [re.compile(p, re.I) for p in ha.HEDGE_CUE_FAMILIES["intending"]]
_SCHED_RES = [re.compile(p, re.I) for p in ha.HEDGE_CUE_FAMILIES["scheduled_but_future"]]

#: explicit date token (owner ruling 2026-07-30: seasonal/annual tokens are
#: NOT dates): month-day, (on|next|this|by|before|starting|from) weekday,
#: "the 15th", or "tomorrow".
_MONTHS = (r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?"
           r"|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?"
           r"|nov(?:ember)?|dec(?:ember)?")
_WEEKDAYS = r"monday|tuesday|wednesday|thursday|friday|saturday|sunday"
DATE_TOKEN = re.compile(
    rf"\b(?:{_MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?\b"
    rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{_MONTHS})\b"
    rf"|\b(?:on|next|this|by|before|starting|from)\s+(?:{_WEEKDAYS})\b"
    rf"|\bthe\s+\d{{1,2}}(?:st|nd|rd|th)\b"
    r"|\btomorrow\b", re.I)
_MONTH_DAY = re.compile(
    rf"\b(?:{_MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?\b", re.I)

#: M2 marker union (HEDGE_MARKERS_STRONG | HEDGE_MARKERS_WEAK), substring
#: semantics as _m2_render_ok; g4 masks month-day dates first (the M2 waiver:
#: modal "may" is never followed by a numeral).
MARKER_UNION = tuple(sorted(set(ge.HEDGE_MARKERS_STRONG)
                            | set(ge.HEDGE_MARKERS_WEAK)))
#: strong-only marker strings not already covered by the deliberating regexes
#: (bans them at g2, where only the ONE soft hedge may carry uncertainty).
L2_STRONG_BAN = ("maybe", "undecided", "still deciding", "haven't decided",
                 "havent decided", "back and forth", "haven't made up",
                 "havent made up", "torn")

#: first-person future-commitment frame (g3/g4 positive requirement — the
#: settled plan must still be a plan someone is voicing).
COMMIT_FRAME = re.compile(
    r"\b(?:i|we)(?:'ll| will)\b"
    r"|\b(?:i|we)(?:'m| am|'re| are)\s+(?:going to|gonna)\b"
    r"|\b(?:i|we)(?:'m| am|'re| are)\s+"
    r"(?:switch|mov|chang|start|sign|drop|upgrad|leav|jump|order|swapp)\w*"
    r"|\b(?:i|we)\s+(?:switch|move|start|swap|change)\b"      # 'I switch ... on May 15'
    r"|\bthe\b[^.!?]{0,40}\b(?:kicks in|goes live|starts|begins)\b",
    re.I)

#: p4 completion-phrase ban — the M2 p4 list VERBATIM (graded_eval
#: _m2_render_ok) + the frozen completion-cue regexes swept raw.
P4_COMPLETION_PHRASES = ("went ahead", "pulled the trigger", "made the switch",
                         "switched to", "went through with", "officially",
                         "finalized", "signed up", "it's done", "did it")

#: p1 positive-reuse heuristic knobs
P1_MIN_SHARED_TOKENS = 2
_STOPWORDS = frozenset(
    "a an the i we my our you your it its this that these those and or but so "
    "to of in on at for with from by as is are was were be been am do does "
    "did have has had will would might may can could should im ive ill dont "
    "didnt its thats haven havent isnt wasnt gonna going go got get still "
    "just really pretty about into over after before once when if then than "
    "now yet not no me us them they he she there here what which who how why "
    "one two next last also very much more most some any all out up down off "
    "new old good ok okay yeah well oh".split())


def _fam_hit(text: str, regexes: list[re.Pattern]) -> bool:
    return any(rx.search(text) for rx in regexes)


def _fam_count(text: str, regexes: list[re.Pattern]) -> int:
    return sum(len(rx.findall(text)) for rx in regexes)


def _mask_month_day(low: str) -> str:
    return _MONTH_DAY.sub(" ", low)


def _marker_union_hits(text: str, *, mask_dates: bool = False) -> list[str]:
    low = text.lower()
    if mask_dates:
        low = _mask_month_day(low)
    return [m for m in MARKER_UNION if m in low]


def _content_tokens(text: str, mask: list[str]) -> set[str]:
    for v in mask:
        if v:
            text = re.sub(re.escape(v), " ", text, flags=re.I)
    return {t for t in re.findall(r"[a-z0-9']+", text.lower())
            if len(t) >= 3 and t not in _STOPWORDS}


def _shared_sixgram(a: str, b: str, mask: list[str]) -> bool:
    """Shared 6-gram, both sides value-masked (the M2 waiver: the pinned value
    string is shared by construction, wording around it must stay fresh)."""
    ma, mb = a, b
    for v in mask:
        if v:
            ma = re.sub(re.escape(v), " m2valuemask ", ma, flags=re.I)
            mb = re.sub(re.escape(v), " m2valuemask ", mb, flags=re.I)
    return bool(r3._ngrams(ma, 6) & r3._ngrams(mb, 6))


def _completion_frame(text: str, value: str) -> bool:
    """Plain completion asserted for `value` (frozen cue sets + adoption
    frames + v5 extras for opener-natural phrasings)."""
    if r3._completed_hit(text, value):
        return True
    if any(p.search(text) for p in r3._adoption_regexes(value)):
        return True
    extra = re.compile(r"(?i)\b(went through|went ahead|came through"
                       r"|is (all )?set up|its official|it's official"
                       r"|swapped (over )?to|moved (everything )?(over )?to"
                       r"|(is|are) (now )?(on|with|at)\b)")
    for m in extra.finditer(text):
        lo = max(0, m.start() - 80 - len(value))
        window = text[lo:m.end() + 80 + len(value)]
        if _norm(value) in _norm(window):
            return True
    return False


# ---------------------------------------------------------------------------
# Level-manifestation lints
# ---------------------------------------------------------------------------

def hedge_level_checks(level: str, sentence: str, turn_text: str,
                       value: str) -> list[str]:
    """The change-statement beat sentence must MANIFEST its directed level and
    carry NO other level's distinguishing cues (whole beat turn swept for the
    banned families)."""
    fails: list[str] = []
    lv = level if level in HS_LEVELS else "g2_weak_hedge"  # wording-ladder hedge
    fam = ha.detect_cue_family(sentence)[0]

    if r3._abandon_cue_any(sentence):
        fails.append("FACTOR: the change sentence must not contain "
                     "abandonment language")
    if lv != "g5_completed_control":
        if r3._completed_hit(sentence, value):
            fails.append("FACTOR: the change sentence must not read as "
                         "completed — nothing has happened yet")
        if any(p.search(sentence) for p in r3._adoption_regexes(value)):
            fails.append("FACTOR: the change sentence must not assert the "
                         "value as already adopted / in current use")
    if lv in ("g1_strong_hedge", "g2_weak_hedge", "g3_confident_plan",
              "g5_completed_control"):
        if DATE_TOKEN.search(sentence):
            fails.append(f"FACTOR {lv}: no explicit date/weekday/'tomorrow' "
                         "in the change sentence (dates belong to "
                         "g4_scheduled_event only)")

    if lv == "g1_strong_hedge":
        if fam != "deliberating" or not _fam_hit(sentence, _DELIB_RES):
            fails.append("FACTOR g1: the sentence must carry an explicit "
                         "deliberation cue (torn between / thinking about / "
                         "considering / can't decide / mulling / weighing / "
                         f"on the fence / not sure whether); detected {fam!r}")
        if _fam_hit(turn_text, _INTEND_RES):
            fails.append("FACTOR g1: no soft-intent cue (might/probably/"
                         "planning to/hoping to/leaning toward) anywhere in "
                         "the beat turn — g1 is pure deliberation")
        if _fam_hit(turn_text, _SCHED_RES):
            fails.append("FACTOR g1: no scheduling cue in the beat turn")
    elif lv == "g2_weak_hedge":
        n = _fam_count(sentence, _INTEND_RES)
        if n != 1:
            fails.append(f"FACTOR g2: exactly ONE soft prospective hedge "
                         f"(might / may / I'll probably / planning to / "
                         f"hoping to / leaning toward / think I'll) — found "
                         f"{n}")
        if fam != "intending":
            fails.append(f"FACTOR g2: the sentence must read as a soft "
                         f"intent (detected {fam!r})")
        if _fam_hit(turn_text, _DELIB_RES):
            fails.append("FACTOR g2: no strong deliberation cue anywhere in "
                         "the beat turn (torn/mulling/can't decide/thinking "
                         "about/considering/weighing)")
        if _fam_hit(turn_text, _SCHED_RES):
            fails.append("FACTOR g2: no scheduling cue in the beat turn")
        low = turn_text.lower()
        strong = [m for m in L2_STRONG_BAN if m in low]
        if strong:
            fails.append(f"FACTOR g2: strong uncertainty word(s) {strong} in "
                         "the beat turn — keep the single soft hedge only")
    elif lv in ("g3_confident_plan", "g4_scheduled_event"):
        g4 = lv == "g4_scheduled_event"
        hits = _marker_union_hits(sentence, mask_dates=g4)
        if hits:
            fails.append(f"FACTOR {lv}: NO hedging words at all — remove "
                         f"{hits} (state the plan as settled bare fact)")
        if _fam_hit(turn_text, _DELIB_RES) or _fam_hit(turn_text, _INTEND_RES):
            fails.append(f"FACTOR {lv}: no deliberation/soft-intent cue "
                         "anywhere in the beat turn")
        if not (COMMIT_FRAME.search(sentence)
                or _fam_hit(sentence, _SCHED_RES)):
            fails.append(f"FACTOR {lv}: the sentence must voice a first-person "
                         "settled commitment ('I'm switching to ...', 'I'll "
                         "move ...', 'I switch ... on <date>')")
        if g4 and not DATE_TOKEN.search(sentence):
            fails.append("FACTOR g4: an explicit month-day date, weekday, or "
                         "'tomorrow' is REQUIRED in the change sentence")
        if not g4 and _fam_hit(turn_text, _SCHED_RES) and not _fam_hit(
                sentence, _SCHED_RES):
            fails.append("FACTOR g3: no scheduling cue elsewhere in the beat "
                         "turn")
    else:  # g5_completed_control
        hits = _marker_union_hits(sentence)
        if hits:
            fails.append(f"FACTOR g5: no hedging words — remove {hits}")
        if fam is not None:
            fails.append(f"FACTOR g5: no plan/hedge cue (detected {fam!r}) — "
                         "the change already happened")
        if r4._FUTURE_FRAME.search(sentence):
            fails.append("FACTOR g5: no future frame — assert the CURRENT "
                         "state, present tense")
        if r3._completed_hit(sentence, value):
            fails.append("FACTOR g5: do NOT narrate the change event (no "
                         "'switched to', 'went with', 'just signed up') — "
                         "assert the new value as current state ('i'm on "
                         "<value> now', 'it's <value> these days')")
        if not any(p.search(sentence) for p in r3._adoption_regexes(value)):
            fails.append("FACTOR g5: the sentence must assert the value as "
                         "the user's CURRENT state ('i'm on <value> now', "
                         "'i use <value> these days')")
    return fails


_WEEKDAY_IDX = {d: i for i, d in enumerate(
    ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
     "sunday"))}
_MONTH_IDX = {m[:3]: i + 1 for i, m in enumerate(
    ("january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"))}


def _parse_session_date(stamp: str) -> dt.datetime:
    return dt.datetime.strptime(re.sub(r" \(\w+\)", "", stamp),
                                "%Y/%m/%d %H:%M")


def resolve_schedule_date(sentence: str,
                          hedge_day: dt.datetime) -> dt.datetime | None:
    """The g4 sentence's explicit date token resolved to a calendar day
    (month-day in the conversation's year; weekday = next occurrence strictly
    after the hedge day; 'tomorrow' = hedge day + 1)."""
    low = sentence.lower()
    m = re.search(rf"\b({_MONTHS})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", low)
    if not m:
        m2 = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?"
                       rf"({_MONTHS})\b", low)
        if m2:
            m = None
            month = _MONTH_IDX[m2.group(2)[:3]]
            return dt.datetime(hedge_day.year, month, int(m2.group(1)))
    if m:
        return dt.datetime(hedge_day.year, _MONTH_IDX[m.group(1)[:3]],
                           int(m.group(2)))
    w = re.search(rf"\b(?:on|next|this|by|before|starting|from)\s+"
                  rf"({_WEEKDAYS})\b", low)
    if w:
        ahead = (_WEEKDAY_IDX[w.group(1)] - hedge_day.weekday()) % 7
        return hedge_day.replace(hour=0, minute=0) + dt.timedelta(
            days=ahead or 7)
    if re.search(r"\btomorrow\b", low):
        return hedge_day.replace(hour=0, minute=0) + dt.timedelta(days=1)
    d = re.search(r"\bthe\s+(\d{1,2})(?:st|nd|rd|th)\b", low)
    if d:
        day = int(d.group(1))
        cand = hedge_day.replace(day=min(day, 28), hour=0, minute=0)
        if day <= 28:
            cand = hedge_day.replace(day=day, hour=0, minute=0)
        if cand <= hedge_day:
            month = hedge_day.month % 12 + 1
            year = hedge_day.year + (1 if month == 1 else 0)
            cand = dt.datetime(year, month, day)
        return cand
    return None


def g4_schedule_checks(sentence: str, geom: dict) -> list[str]:
    """g4 still-open coherence: the scheduled switch date must fall STRICTLY
    AFTER the conversation's final session day — the switch is locked in but
    has NOT happened by the end of the last session (otherwise a later
    still-pending mention contradicts the schedule; the r4 round-3
    timeline-contradiction class)."""
    hedge_day = _parse_session_date(
        geom["sessions"][0]["turn_dates"][2 * geom["hedge"]["user_turn"]])
    final_day = _parse_session_date(geom["sessions"][-1]["turn_dates"][-1])
    sched = resolve_schedule_date(sentence, hedge_day)
    if sched is None:
        return []                       # date-token REQUIRED lint fires anyway
    if sched.date() <= final_day.date():
        return [f"FACTOR g4: the scheduled date resolves to "
                f"{sched.date().isoformat()}, which is not AFTER the final "
                f"session ({final_day.date().isoformat()}) — pick an explicit "
                "date later than every session so the switch is still ahead "
                "at the end of the conversation"]
    return []


def resolution_level_checks(level: str, outcome: str, sentence: str,
                            turn_text: str, value: str, hedge_sentence: str,
                            attribute: str,
                            candidates: list[str]) -> list[str]:
    """The final-beat opener must manifest its outcome, and for the wording
    ladder its directed distance from the hedge sentence."""
    fails: list[str] = []
    fam = ha.detect_cue_family(sentence)[0]
    mask = [value] + [c for c in candidates if c]

    if outcome == "open":
        if sentence.count(value) != 1:
            fails.append("still_open opener must contain the value verbatim "
                         "exactly once")
        if not r2.OPEN_CUE.search(sentence):
            fails.append("still_open opener must read pending (still / "
                         "haven't pulled the trigger / yet)")
        if fam is not None:
            fails.append(f"FACTOR purity: the still-open opener must carry NO "
                         f"hedge-strength cue (detected {fam!r}) — a neutral "
                         "pending nod only ('still haven't pulled the "
                         "trigger on <value>')")
        if r3._completed_hit(sentence, value) or any(
                p.search(sentence) for p in r3._adoption_regexes(value)):
            fails.append("still_open opener must not read as completed/adopted")
        if r3._abandon_cue_any(sentence):
            fails.append("still_open opener must not contain abandonment "
                         "language")
        return fails

    if outcome == "completed_control":
        if sentence.count(value) != 1:
            fails.append("continuation opener must contain the value "
                         "verbatim exactly once")
        if fam is not None:
            fails.append(f"continuation opener must carry no plan/hedge cue "
                         f"(detected {fam!r})")
        if r4._FUTURE_FRAME.search(sentence):
            fails.append("continuation opener must not read as a future plan")
        if r3._abandon_cue_any(sentence):
            fails.append("continuation opener must not contain abandonment "
                         "language")
        if r3._completed_hit(sentence, value):
            fails.append("continuation opener must PRESUME the state, not "
                         "announce a fresh switch (no 'switched to/went "
                         "with/just got')")
        subjectless = re.compile(
            rf"(?i)\b(?:been|still|now)\s+(?:on|using|with|at|running in"
            rf"|wearing|going to)\s+(?:the |a |an |my )?{re.escape(value)}")
        if not (any(p.search(sentence) for p in r3._adoption_regexes(value))
                or r4.asserts_prior_current(sentence, value)
                or subjectless.search(sentence)):
            fails.append("continuation opener must read as the value being "
                         "the user's current state ('been on <value> for a "
                         "few weeks now, ...')")
        return fails

    # resolve_true (came_true), wording ladder ------------------------------
    if r3._abandon_cue_any(sentence):
        fails.append("came_true opener must not contain abandonment language")
    if fam is not None:
        fails.append(f"came_true opener must not read tentative (detected "
                     f"{fam!r} cue)")
    n_val = sentence.count(value)
    if level == "p4_situational":
        if n_val > 1:
            fails.append("FACTOR p4: the value may appear at most once, "
                         "incidentally")
        if turn_text.count(value) > 1:
            fails.append("FACTOR p4: the opener turn mentions the value more "
                         "than once")
        low = sentence.lower()
        phr = [p for p in P4_COMPLETION_PHRASES if p in low]
        if phr:
            fails.append(f"FACTOR p4: completion phrase(s) {phr} — NEVER "
                         "state that the change happened; describe the first "
                         "experience instead")
        if (r2.COMPLETED_CUE.search(sentence)
                or r3.COMPLETED_CUE_R3_EXTRA.search(sentence)):
            fails.append("FACTOR p4: no completion verbs about the plan — the "
                         "situation alone must imply it")
        if _shared_sixgram(sentence, hedge_sentence, mask):
            fails.append("FACTOR p4: shares a 6-word sequence with the hedge "
                         "sentence — describe the situation freshly")
        if not re.search(r"(?i)\b(i|we|my|our|im|ive|first)\b", sentence):
            fails.append("FACTOR p4: the opener must read as a personal "
                         "first experience (i/we/my/first ...)")
        return fails

    if n_val != 1:
        fails.append(f"came_true opener must contain the value verbatim "
                     f"exactly once (found {n_val})")
    if not _completion_frame(sentence, value):
        fails.append("came_true opener needs a plain completion frame "
                     "(ended up going with / finally moved to / the order "
                     "came in / went through)")
    if level == "p1_verbatim":
        shared = (_content_tokens(sentence, mask)
                  & _content_tokens(hedge_sentence, mask))
        has_trigram = bool(
            r3._ngrams(re.sub(re.escape(value), " ", sentence, flags=re.I), 3)
            & r3._ngrams(re.sub(re.escape(value), " ", hedge_sentence,
                                flags=re.I), 3))
        if len(shared) < P1_MIN_SHARED_TOKENS and not has_trigram:
            fails.append("FACTOR p1: reuse the hedge sentence's own key "
                         "wording — the opener must share its verbs/nouns "
                         f"(shared content words now: {sorted(shared)})")
    elif level == "p2_paraphrase":
        if _shared_sixgram(sentence, hedge_sentence, mask):
            fails.append("FACTOR p2: shares a 6-word sequence with the hedge "
                         "sentence — reword completely freshly")
    elif level == "p3_oblique":
        if _shared_sixgram(sentence, hedge_sentence, mask):
            fails.append("FACTOR p3: shares a 6-word sequence with the hedge "
                         "sentence")
        vn = _norm(value)
        low = sentence.lower()
        for tok in ge._attr_ban_tokens(attribute):
            if tok in vn:
                continue                       # M2 waiver: value carries it
            if re.search(rf"(?<![\w]){re.escape(tok)}(?![\w])", low):
                fails.append(f"FACTOR p3: names the key's domain noun "
                             f"{tok!r} — refer back only obliquely ('went "
                             "ahead with it')")
    return fails


# ---------------------------------------------------------------------------
# Spec — deterministic; keys registered BEFORE any value exists (FM-4)
# ---------------------------------------------------------------------------

def _arc_id(level: str, i: int) -> str:
    ladder = "hs" if level in HS_LEVELS else "cw"
    return f"UNIFY-{ladder}-{level.split('_')[0]}-{i:03d}"


def build_specs() -> tuple[list[dict], list[dict]]:
    """(spec rows, key registry rows), fully deterministic from SEED_UNIFY.
    20 arcs per level cell; AUX_POOL attributes seeded-balanced 9 arcs each."""
    attrs = sorted(AUX_POOL)
    cells = [(lv, i) for lv in ALL_LEVELS for i in range(1, N_PER_CELL + 1)]
    pool = attrs * (len(cells) // len(attrs))
    assert len(pool) == len(cells), "attribute pool must tile the cells"
    random.Random(f"{SEED_UNIFY}:attrs").shuffle(pool)
    specs, keys = [], []
    for (level, i), attribute in zip(cells, pool):
        arc_id = _arc_id(level, i)
        outcome = OUTCOME_BY_LEVEL[level]
        desc, probe_q, kw = AUX_POOL[attribute]
        window = random.Random(f"{SEED_UNIFY}:{arc_id}:window").randint(2, 9)
        gold = level_gold(level)
        keys.append({
            "arc_id": arc_id, "benchmark": None, "entity": "user",
            "attribute": attribute, "description": desc,
            "probe_question": probe_q, "collision_keywords": list(kw),
            "key_source": "unify_aux_style",
            "fm4_note": ("registered at spec time, before any value was "
                         "sampled (FM-4); probe question is the frozen "
                         "value-free AUX_POOL text"),
        })
        specs.append({
            "arc_id": arc_id,
            "row_kind": "unify_synthetic_arc",
            "stratum": "unify",
            "unify_ladder": ("hedge_strength" if level in HS_LEVELS
                             else "confirmation_wording"),
            "unify_level": level,
            "m2_level": M2_LEVEL_NAME[level],
            "resolution_type": outcome,
            "gold_resolution_status": GOLD_STATUS[outcome],
            "hedge_strength": HEDGE_STRENGTH_3WAY[level],
            "designed_gold_class": gold["designed_gold_class"],
            "designed_gold_route": gold["designed_gold_route"],
            "taxonomy_rule": gold["taxonomy_rule"],
            "stimulus_contract": gold["contract"],
            "stimulus_construction_label": gold["construction_label"],
            "key": {"entity": "user", "attribute": attribute,
                    "description": desc, "probe_question": probe_q,
                    "key_source": "unify_aux_style",
                    "key_ref": f"unify/{arc_id}/user/{attribute}"},
            "pending_window_user_turns": window,
            "seed": SEED_UNIFY,
            "value": None,
            "value_candidates": None,
            "prior_value": None,
            "status": "pending_owner_review",
            "usable": False,
        })
    return specs, keys


def _assert_spec_frozen(specs: list[dict]) -> None:
    if not SPEC_PATH.exists():
        return
    frozen = ha._read_jsonl(SPEC_PATH)
    if frozen != specs:
        raise SystemExit(
            f"{SPEC_PATH} exists and differs from the recomputed spec — the "
            "spec froze before generation; delete it deliberately (and "
            "regenerate everything) to respec")


# ---------------------------------------------------------------------------
# Geometry (r4 shape, unify seed) + directives
# ---------------------------------------------------------------------------

def _rng(arc_id: str, what: str) -> random.Random:
    return random.Random(f"{SEED_UNIFY}:{arc_id}:{what}")


def conversation_geometry(arc: dict) -> dict:
    """r4 conversation_geometry VERBATIM in shape (2 or 3 sessions; prior at
    s0u0; change beat at s0 u>=1; final beat opens the final session; window
    solved toward the target, clamped by capacity <= 11), reseeded under
    SEED_UNIFY keyed by arc_id."""
    W = arc["pending_window_user_turns"]
    rng = _rng(arc["arc_id"], "geometry")
    LO, HI = r3.N_USER_LO, r3.N_USER_HI

    cands = []
    for n0 in range(LO, HI + 1):
        for n_final in range(LO, HI + 1):
            for rest0 in range(0, n0 - 1):
                cands.append((2, n0, None, n_final, rest0, rest0 + 1))
            for n_mid in range(LO, HI + 1):
                for rest0 in range(0, n0 - 1):
                    cands.append((3, n0, n_mid, n_final, rest0,
                                  rest0 + n_mid + 1))

    def key(c):
        n_sessions, _n0, _n_mid, _n_final, _rest0, window = c
        return (abs(window - W), n_sessions)

    best = min(key(c) for c in cands)
    pool = [c for c in cands if key(c) == best]
    n_sessions, n0, n_mid, n_final, rest0, realized = rng.choice(pool)
    sizes = [n0] + ([n_mid] if n_sessions == 3 else []) + [n_final]
    roles = (["hedge", "distractor", "resolution"] if n_sessions == 3
             else ["hedge", "resolution"])
    hedge_user_turn = n0 - 1 - rest0

    drng = _rng(arc["arc_id"], "dates")
    day = dt.datetime(2023, drng.randint(2, 9), drng.randint(1, 26))
    sessions = []
    for si in range(n_sessions):
        gap = 0 if si == 0 else drng.randint(r3.GAP_DAYS_LO, r3.GAP_DAYS_HI)
        day = day + dt.timedelta(days=gap)
        t = day.replace(hour=drng.randint(8, 21), minute=drng.randint(0, 59))
        turn_dates = []
        for _ in range(2 * sizes[si]):
            turn_dates.append(t.strftime("%Y/%m/%d (%a) %H:%M"))
            t = t + dt.timedelta(minutes=drng.randint(1, 4))
        sessions.append({
            "session_idx": si,
            "date": turn_dates[0],
            "days_since_previous": gap if si else None,
            "n_user_turns": sizes[si],
            "role": roles[si],
            "turn_dates": turn_dates,
        })
    return {
        "n_sessions": n_sessions,
        "n_user_turns": sizes,
        "prior_establish": {"session_idx": 0, "user_turn": 0},
        "hedge": {"session_idx": 0, "user_turn": hedge_user_turn},
        "resolution": {"session_idx": n_sessions - 1, "user_turn": 0},
        "distractor_session_idx": (1 if n_sessions == 3 else None),
        "target_pending_window_user_turns": W,
        "realized_pending_window_user_turns": realized,
        "sessions": sessions,
    }


def build_slots(arcs: list[dict], geoms: dict[str, dict]) -> list[dict]:
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
                    "is_prior": (si == 0 and ui == 0),
                    "is_hedge": (si == g["hedge"]["session_idx"]
                                 and ui == g["hedge"]["user_turn"]),
                    "is_resolution": (si == g["resolution"]["session_idx"]
                                      and ui == g["resolution"]["user_turn"]),
                    "resolution_type": arc["resolution_type"],
                })
    return slots


def assign_noise(slots: list[dict]) -> None:
    """r3 batch quotas, max ONE feature per slot; fragments never on beat
    turns (r4 eligibility; no abandoned arcs exist in this build)."""
    for s in slots:
        s["noise"] = "none"
    order = list(range(len(slots)))
    random.Random(f"unify-noise:{SEED_UNIFY}").shuffle(order)
    n = len(slots)

    def eligible(s: dict, feature: str) -> bool:
        if s["noise"] != "none":
            return False
        if (feature == "fragment"
                and (s["is_hedge"] or s["is_resolution"] or s["is_prior"])):
            return False
        return True

    for feature, rate in r3.NOISE_QUOTAS.items():
        want = round(rate * n)
        for i in order:
            if want <= 0:
                break
            if eligible(slots[i], feature):
                slots[i]["noise"] = feature
                want -= 1


def assign_word_bands(slots: list[dict]) -> None:
    """r3 lognormal bands; beat floors are applied AFTER the plan call (v5
    samples values at plan time) via apply_beat_floors."""
    for s in slots:
        rng = random.Random(f"{SEED_UNIFY}:{s['arc_id']}:s{s['session_idx']}"
                            f":u{s['user_turn']}:words")
        if s["noise"] == "fragment":
            t = rng.randint(r3.FRAG_LO, r3.FRAG_HI)
            s["target_words"] = t
            s["words_min"], s["words_max"] = r3.FRAG_LO, r3.FRAG_HI + 1
            continue
        t = round(rng.lognormvariate(math.log(r3.USER_WORDS_MEDIAN),
                                     r3.USER_WORDS_SIGMA))
        t = max(r3.USER_WORDS_LO, min(r3.USER_WORDS_HI, t))
        s["target_words"] = t
        s["words_min"] = max(r3.USER_WORDS_LO, t - r3.WORD_TOL)
        s["words_max"] = t + r3.WORD_TOL


def apply_beat_floors(arc_slots: list[dict], value: str) -> None:
    """Beat turns floored at value words + BEAT_FLOOR_EXTRA (r3 rule),
    deterministic given the sampled value; recorded in the noise ledger."""
    floor = len(value.split()) + r3.BEAT_FLOOR_EXTRA
    for s in arc_slots:
        if s["is_hedge"] or s["is_resolution"] or s["is_prior"]:
            t = max(s["target_words"], floor)
            s["target_words"] = t
            s["words_min"] = max(r3.USER_WORDS_LO, t - r3.WORD_TOL)
            s["words_max"] = t + r3.WORD_TOL


def pick_exemplars(pool: dict, arc: dict, session_idx: int,
                   has_hedge: bool, has_resolution: bool) -> list[dict]:
    """r3 exemplar picking, unify seed; category by outcome/level."""
    rng = _rng(arc["arc_id"], f"exemplars:s{session_idx}")
    hedge_cat = ("completed_change"
                 if arc["unify_level"] == "g5_completed_control"
                 else "tentative_plan")
    res_cat = {"open": "tentative_plan", "resolve_true": "completed_change",
               "completed_control": "completed_change"}[arc["resolution_type"]]
    picked: list[dict] = []
    if has_hedge:
        picked += rng.sample(pool[hedge_cat], 3)
    if has_resolution:
        pick = [e for e in rng.sample(pool[res_cat], 3)
                if e["id"] not in {p["id"] for p in picked}]
        picked += pick[:2 if has_hedge else 3]
    everything = [e for c in pool.values() for e in c
                  if e["id"] not in {p["id"] for p in picked}]
    picked += rng.sample(everything, max(0, 6 - len(picked)))
    return picked[:6]


# ---------------------------------------------------------------------------
# Plan lint (values sampled here) + session lint
# ---------------------------------------------------------------------------

def lint_plan(arc: dict, geom: dict, out: dict) -> list[str]:
    fails: list[str] = []
    sessions = out.get("sessions")
    if not isinstance(sessions, list) or len(sessions) != geom["n_sessions"]:
        return [f"need exactly {geom['n_sessions']} sessions"]
    if not isinstance(out.get("premise"), str) or not out["premise"].strip():
        fails.append("premise missing")
    prior_value = out.get("prior_value")
    if not isinstance(prior_value, str) or not prior_value.strip():
        fails.append("prior_value missing (emit the user's current value)")
    cands = out.get("value_candidates")
    if (not isinstance(cands, list) or len(cands) != 4
            or not all(isinstance(c, str) and c.strip() for c in cands)):
        fails.append("value_candidates must be exactly 4 non-empty strings "
                     "(the FIRST is VALUE, the change target)")
        cands = None
    for si, s in enumerate(sessions):
        beats = (s or {}).get("user_turn_beats")
        n_want = geom["sessions"][si]["n_user_turns"]
        if not isinstance(beats, list) or len(beats) != n_want:
            fails.append(f"session {si}: need exactly {n_want} user_turn_beats")
            continue
        for b in beats:
            if not isinstance(b, str) or not b.strip():
                fails.append(f"session {si}: empty beat line")
            elif len(b.split()) > 30:
                fails.append(f"session {si}: beat line over 30 words")
            elif not r3._ASCII_OK.match(b):
                fails.append(f"session {si}: non-ASCII beat line")
    if fails:
        return fails

    def same(a, b):
        return bool(a) and bool(b) and _norm(a) == _norm(b)

    for c in cands:
        if len(c.split()) > 5:
            fails.append(f"value candidate {c!r} over 5 words — short "
                         "concrete names only")
        if not r3._ASCII_OK.match(c):
            fails.append(f"value candidate {c!r} non-ASCII")
    for i, a in enumerate(cands):
        for b in cands[i + 1:]:
            if same(a, b):
                fails.append(f"value candidates must be mutually distinct "
                             f"({a!r} ~ {b!r})")
    for c in cands:
        if same(prior_value, c):
            fails.append(f"prior_value {prior_value!r} must be distinct from "
                         f"every candidate ({c!r} collides) — it is what the "
                         "user has NOW")
    probe = arc["key"]["probe_question"]
    for c in cands + [prior_value]:
        if not ha._fm4_probe_ok(probe, c):
            fails.append(f"FM-4: {c!r} appears in the probe question")
    if not r3._ASCII_OK.match(prior_value):
        fails.append("prior_value non-ASCII")
    if fails:
        return fails

    value = cands[0]
    pe, h, r_ = geom["prior_establish"], geom["hedge"], geom["resolution"]
    beat = lambda p: sessions[p["session_idx"]]["user_turn_beats"][p["user_turn"]]
    if prior_value not in beat(pe):
        fails.append(f"the prior-establish beat (session {pe['session_idx']} "
                     f"user turn {pe['user_turn']}) must contain prior_value "
                     "verbatim")
    if value not in beat(h):
        fails.append(f"the change-statement beat (session {h['session_idx']} "
                     f"user turn {h['user_turn']}) must contain the first "
                     "value candidate (VALUE) verbatim")
    if arc["unify_level"] != "p4_situational" and value not in beat(r_):
        fails.append(f"the final beat (session {r_['session_idx']} user turn "
                     f"{r_['user_turn']}) must contain VALUE verbatim")
    return fails


def _final_beat_sentence_checks(sentence, turn_text, level) -> list[str]:
    """Structural checks on the final-beat sentence (value-count semantics
    are level-dependent and live in resolution_level_checks)."""
    if not sentence or not isinstance(sentence, str):
        return ["resolution_sentence missing"]
    fails = []
    if sentence not in turn_text:
        fails.append("resolution_sentence must be a verbatim substring of "
                     "the final-beat turn")
    if "?" in sentence:
        fails.append("resolution_sentence must be a statement, not a question")
    return fails


def lint_session(arc: dict, geom: dict, session_idx: int, out: dict,
                 prior_turns: list[dict], slots: list[dict],
                 banned: list[dict], exemplars: list[dict],
                 prior_value: str) -> list[str]:
    """Full deterministic acceptance lint for one rendered session: the r3
    register battery per turn VERBATIM, the r4 prior/coherence checks, the
    outcome ordering sweeps, and the v5 factor-manifestation battery."""
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
    level = arc["unify_level"]
    outcome = arc["resolution_type"]
    candidates = [c for c in (arc.get("value_candidates") or [])]
    mask = [value] + candidates
    has_hedge = geom["hedge"]["session_idx"] == session_idx
    has_res = geom["resolution"]["session_idx"] == session_idx
    hedge_ti = 2 * geom["hedge"]["user_turn"]
    res_ti = 2 * geom["resolution"]["user_turn"]
    slot_by_ti = {s["turn_idx"]: s for s in slots}

    # -- per-turn register lints (r3 VERBATIM) -------------------------------
    for k, t in enumerate(turns):
        if k % 2 == 0:
            prev_assist = None
            if k >= 2:
                prev_assist = turns[k - 1]["text"]
            elif prior_turns:
                prev_assist = prior_turns[-1]["text"]
            fails += r3.lint_user_turn(t["text"], slot_by_ti[k], banned=banned,
                                       prev_assistant=prev_assist,
                                       exemplars=exemplars, mask=mask)
        else:
            fails += r3.lint_assistant_turn(t["text"], k)

    # -- prior beat (r4 VERBATIM: distinctness + settled-present-fact) -------
    if session_idx == 0:
        fails += r4._prior_checks(arc, geom, out, prior_value, None)

    # -- beat sentences + factor manifestation --------------------------------
    hedge_sentence = out.get("hedge_sentence")
    res_sentence = out.get("resolution_sentence")
    if has_hedge:
        fails += r3._beat_sentence_checks("hedge", hedge_sentence,
                                          turns[hedge_ti]["text"], arc)
        if isinstance(hedge_sentence, str) and hedge_sentence:
            fails += hedge_level_checks(level, hedge_sentence,
                                        turns[hedge_ti]["text"], value)
            if level == "g4_scheduled_event":
                fails += g4_schedule_checks(hedge_sentence, geom)
    elif hedge_sentence:
        fails.append("hedge_sentence must be null (no change-statement beat "
                     "in this session)")
    if has_res:
        fails += _final_beat_sentence_checks(res_sentence,
                                             turns[res_ti]["text"], level)
        if isinstance(res_sentence, str) and res_sentence:
            if level != "p4_situational" and turns[res_ti]["text"].count(value) > 1:
                fails.append("the final-beat turn mentions the value more "
                             "than once — only the beat sentence carries it")
            fails += resolution_level_checks(
                level, outcome, res_sentence, turns[res_ti]["text"], value,
                arc.get("hedge_text") or "", arc["key"]["attribute"],
                candidates)
    elif res_sentence:
        fails.append("resolution_sentence must be null (no final beat in "
                     "this session)")

    # -- coherence (r4 CHANGE 1) ----------------------------------------------
    res_sess = geom["resolution"]["session_idx"]
    if session_idx == res_sess and geom["resolution"]["user_turn"] != 0:
        fails.append("COHERENCE: the final beat must be the FIRST user turn "
                     "of its session (a new-conversation opener)")
    forbidden = [value] + candidates + ([prior_value] if prior_value else [])
    if geom.get("distractor_session_idx") == session_idx:
        for k, t in enumerate(turns):
            low = t["text"]
            for c in forbidden:
                if c and (c in low or _norm(c) in _norm(low)):
                    fails.append(f"COHERENCE: distractor session turn {k} "
                                 f"mentions {c!r} — a distractor session is a "
                                 "DIFFERENT sub-topic")
                    break
    if session_idx == res_sess and outcome == "resolve_true":
        for k in range(0, len(turns), 2):
            if k <= res_ti:
                continue
            t = turns[k]["text"]
            if (ha.detect_cue_family(t)[0] is not None
                    and any(c and (c in t or _norm(c) in _norm(t))
                            for c in [value] + candidates)):
                fails.append(f"COHERENCE: final-session turn {k} "
                             "re-deliberates the resolved choice — handle "
                             "the CONSEQUENCE, never re-open it")

    # -- ordering + cross-level cue sweep over every user turn ----------------
    hedge_abs_sess = geom["hedge"]["session_idx"]
    for k in range(0, len(turns), 2):
        text = turns[k]["text"]
        is_hedge_turn = has_hedge and k == hedge_ti
        is_res_turn = has_res and k == res_ti
        swept = text
        if is_hedge_turn and isinstance(hedge_sentence, str):
            swept = swept.replace(hedge_sentence, " ")
        if is_res_turn and isinstance(res_sentence, str):
            swept = swept.replace(res_sentence, " ")
        if session_idx == 0 and k == 0 and isinstance(
                out.get("prior_sentence"), str):
            swept = swept.replace(out["prior_sentence"], " ")

        # cross-level purity: no hedge-cue family next to the value world
        # outside the directed beat sentence, anywhere in the conversation.
        fam = ha.detect_cue_family(swept)[0]
        if fam is not None and any(
                c and (c in swept or _norm(c) in _norm(swept))
                for c in forbidden):
            fails.append(f"FACTOR purity: turn {k} carries a {fam!r} cue "
                         "next to the value/prior/candidates outside the "
                         "directed beat sentence — keep every other mention "
                         "cue-free")

        phase_pre = (session_idx == hedge_abs_sess and k < hedge_ti)
        in_window = (not phase_pre and not is_hedge_turn and not is_res_turn
                     and (session_idx < res_sess
                          or (session_idx == res_sess and k < res_ti)))
        post = (session_idx == res_sess and k > res_ti)

        if outcome == "open":
            if r3.asserts_value_adopted(swept, value):
                fails.append(f"turn {k}: still_open arc — the value must "
                             "never be adopted anywhere in the conversation")
            if r3._abandon_hit_near_value(swept, value):
                fails.append(f"turn {k}: still_open arc — the value must "
                             "never be dropped anywhere in the conversation")
            continue
        if outcome == "completed_control":
            if (phase_pre or is_hedge_turn) and r3.asserts_value_adopted(
                    swept, value):
                fails.append(f"turn {k}: asserts the value as adopted BEFORE "
                             "the change statement")
            if not phase_pre and r3._abandon_hit_near_value(swept, value):
                fails.append(f"turn {k}: completed-control arc — the value "
                             "must never be dropped after the change")
            continue
        # resolve_true (wording ladder)
        if phase_pre or is_hedge_turn:
            if r3.asserts_value_adopted(swept, value):
                fails.append(f"turn {k}: asserts the value as adopted BEFORE "
                             "the hedge — the hedge must be a genuine "
                             "unrealized plan")
        elif in_window:
            if r3.asserts_value_adopted(swept, value):
                fails.append(f"turn {k}: adopts the value inside the pending "
                             "window — resolution happens only at the opener")
            if r3._abandon_hit_near_value(swept, value):
                fails.append(f"turn {k}: drops the value inside the pending "
                             "window")
        elif post or is_res_turn:
            if r3._abandon_hit_near_value(swept, value):
                fails.append(f"turn {k}: came_true arc — the value must not "
                             "be dropped after the resolution")
    return fails


# ---------------------------------------------------------------------------
# Budget (stage "unify"; advisory 20k; build-local 110k soft stop)
# ---------------------------------------------------------------------------

def ensure_unify_stage(path: Path = ha.BUDGET_PATH) -> None:
    """Idempotently add the 'unify' stage (advisory 20k from unallocated
    headroom) to the shared v2.4 budget file — the same
    stage-added-from-headroom pattern as 'external' and 'claude_repl'.
    flock + atomic write (concurrent stages write this file)."""
    lock_path = path.with_suffix(".lock")
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        budget = json.loads(path.read_text())
        if budget["envelope"] != ha.ENVELOPE:
            raise SystemExit(f"{path}: envelope {budget['envelope']} != "
                             f"{ha.ENVELOPE}")
        if STAGE in budget["spent"]:
            return
        budget["spent"][STAGE] = 0
        alloc = budget.get("allocation", {})
        head = alloc.get("unallocated_headroom", 0)
        if head < UNIFY_ADVISORY_ALLOCATION:
            raise SystemExit(f"unallocated headroom {head} < unify advisory "
                             f"{UNIFY_ADVISORY_ALLOCATION}")
        alloc["unallocated_headroom"] = head - UNIFY_ADVISORY_ALLOCATION
        alloc[STAGE] = UNIFY_ADVISORY_ALLOCATION
        budget.setdefault("notes", []).append(
            "2026-08-11: unify 20000 from headroom (substrate-unification "
            "dataset generation; prereg draft protocol/drafts/"
            "unify_prereg.md). Advisory stage per the 2026-07-08 amendment; "
            "120k envelope hard; the unify build additionally refuses to "
            "push TOTAL spend past 110,000 (build-local guard).")
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(budget, indent=1, ensure_ascii=False))
        tmp.replace(path)


def unify_guard(fresh_cap: int | None = None,
                path: Path = ha.BUDGET_PATH) -> forecast_v22.SharedCallBudget:
    ensure_unify_stage(path)
    budget = json.loads(path.read_text())
    total = sum(budget["spent"].values())
    if total >= ENVELOPE_SOFT_STOP:
        raise SystemExit(
            f"UNIFY SOFT STOP: total envelope spend {total:,} >= "
            f"{ENVELOPE_SOFT_STOP:,} — generation must not push past 110k "
            "(prereg draft guard); do not run")
    cap = ENVELOPE_SOFT_STOP - total
    if fresh_cap is not None:
        cap = min(cap, fresh_cap)
    return forecast_v22.SharedCallBudget(STAGE, path=path, local_cap=cap)


def advisory_note(path: Path = ha.BUDGET_PATH) -> dict:
    budget = json.loads(path.read_text())
    spent = budget["spent"].get(STAGE, 0)
    return {
        "stage": STAGE,
        "spent": spent,
        "advisory_allocation": UNIFY_ADVISORY_ALLOCATION,
        "advisory_overrun": spent > UNIFY_ADVISORY_ALLOCATION,
        "envelope": budget["envelope"],
        "envelope_spent_total": sum(budget["spent"].values()),
        "soft_stop_total": ENVELOPE_SOFT_STOP,
        "mode": ("ADVISORY stages / hard total envelope (2026-07-08 "
                 "amendment) + build-local 110k soft stop"),
    }


# ---------------------------------------------------------------------------
# Calls
# ---------------------------------------------------------------------------

def plan_payload(arc: dict, geom: dict, attempt: int,
                 feedback: list[str] | None,
                 salt: int | None = None) -> dict:
    payload = {
        "task": "unify_v5_plan",
        "arc_id": arc["arc_id"],
        "key": {k: arc["key"][k] for k in ("entity", "attribute",
                                           "description")},
        "factor": {"ladder": arc["unify_ladder"],
                   "level": arc["unify_level"],
                   "level_note": LEVEL_NOTES[arc["unify_level"]]},
        "outcome": {"open": "still_open", "resolve_true": "came_true",
                    "completed_control": "already_completed"}[
                        arc["resolution_type"]],
        "geometry": {
            "sessions": [{k: s[k] for k in ("session_idx", "date",
                                            "n_user_turns",
                                            "days_since_previous", "role")}
                         for s in geom["sessions"]],
            "prior_establish_beat": geom["prior_establish"],
            "change_statement_beat": geom["hedge"],
            "final_beat": {
                **geom["resolution"],
                "kind": {"open": "still_pending_mention",
                         "resolve_true": "resolution",
                         "completed_control": "continuation"}[
                             arc["resolution_type"]]},
            "distractor_session_idx": geom.get("distractor_session_idx"),
            "target_pending_window_user_turns":
                geom["target_pending_window_user_turns"],
            "realized_pending_window_user_turns":
                geom["realized_pending_window_user_turns"],
        },
        "attempt": attempt,
    }
    if arc["unify_level"] == "g4_scheduled_event":
        payload["schedule_note"] = (
            "g4 scheduling coherence: any explicit switch date in the "
            "change-statement beat must fall STRICTLY AFTER the final "
            f"session date ({geom['sessions'][-1]['date']}) — the switch is "
            "locked in but has NOT happened by the end of the conversation.")
    if salt:
        payload["retry_cycle"] = salt      # post-failure rerun cycle only
    if feedback:
        payload["lint_feedback"] = feedback
    return payload


def render_payload(arc: dict, geom: dict, session_idx: int, plan: dict,
                   rendered: list[dict], slots: list[dict],
                   exemplars: list[dict], hedge_sentence: str | None,
                   prior_value: str, attempt: int,
                   feedback: list[str] | None,
                   previous_attempt: dict | None = None,
                   salt: int | None = None) -> dict:
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
                "is_prior_establish_turn": bool(s["is_prior"]),
                "is_change_statement_turn": bool(s["is_hedge"]),
                "is_final_beat_turn": bool(s["is_resolution"]),
                "beat": plan["sessions"][session_idx]["user_turn_beats"][k // 2],
            })
        else:
            directives.append({"turn_idx": k, "role": "assistant",
                               "max_words": r3.ASSISTANT_MAX_WORDS})
    payload = {
        "task": "unify_v5_render_session",
        "arc_id": arc["arc_id"],
        "session_idx": session_idx,
        "n_sessions": geom["n_sessions"],
        "session_role": sess["role"],
        "session_date": sess["date"],
        "days_since_previous": sess["days_since_previous"],
        "all_session_dates": [s["date"] for s in geom["sessions"]],
        "all_session_roles": [s["role"] for s in geom["sessions"]],
        "key": {k: arc["key"][k] for k in ("entity", "attribute",
                                           "description")},
        "value": arc["value"],
        "prior_value": prior_value,
        "value_candidates": arc["value_candidates"],
        "factor": {"ladder": arc["unify_ladder"],
                   "level": arc["unify_level"],
                   "level_note": LEVEL_NOTES[arc["unify_level"]]},
        "outcome": {"open": "still_open", "resolve_true": "came_true",
                    "completed_control": "already_completed"}[
                        arc["resolution_type"]],
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
    if arc["unify_level"] == "p3_oblique" and has_res:
        payload["banned_attribute_nouns"] = ge._attr_ban_tokens(
            arc["key"]["attribute"])
    if arc["unify_level"] == "g4_scheduled_event" and has_hedge:
        payload["schedule_note"] = (
            "g4 scheduling coherence: the change sentence's explicit switch "
            "date must fall STRICTLY AFTER the conversation's final session "
            f"date ({geom['sessions'][-1]['date']}) — the switch is locked "
            "in on the calendar but has NOT happened by the end of the last "
            "session.")
    if salt:
        payload["retry_cycle"] = salt      # post-failure rerun cycle only
    if feedback:
        payload["lint_feedback"] = feedback
        payload["fresh_wording_note"] = (
            "Previous attempt failed the lints above. Start from "
            "'previous_attempt_turns' and change ONLY the turns the "
            "lint_feedback names (a failure 'user turn N (turn_idx M): ...' is "
            "about the turn at that turn_idx) — fix EVERY listed failure, leave "
            "every passing turn byte-for-byte identical, and keep all rules and "
            "the beat sentences intact.")
        if previous_attempt is not None:
            payload["previous_attempt_turns"] = previous_attempt.get("turns")
    return payload


async def plan_conversation(arc: dict, geom: dict, plan_sys: str,
                            salt: int | None = None) -> tuple[dict, int]:
    feedback = None
    for attempt in range(MAX_ATTEMPTS):
        out = await llm.call_llm(
            plan_sys,
            json.dumps(plan_payload(arc, geom, attempt, feedback, salt=salt),
                       ensure_ascii=False),
            trace_tag=f"unify-plan-{arc['arc_id']}-a{attempt}")
        fails = lint_plan(arc, geom, out)
        if not fails:
            return out, attempt + 1
        feedback = fails
    raise RuntimeError(f"{arc['arc_id']} plan: lint still failing after "
                       f"{MAX_ATTEMPTS} attempts: {feedback}")


async def render_session(arc: dict, geom: dict, session_idx: int, plan: dict,
                         rendered: list[dict], prior_turns: list[dict],
                         slots: list[dict], pool: dict, banned: list[dict],
                         render_sys: str, hedge_sentence: str | None,
                         salt: int | None = None) -> tuple[dict, dict]:
    has_hedge = geom["hedge"]["session_idx"] == session_idx
    has_res = geom["resolution"]["session_idx"] == session_idx
    exemplars = pick_exemplars(pool, arc, session_idx, has_hedge, has_res)
    prior_value = plan["prior_value"]
    feedback = None
    last_verdict: dict = {}
    last_out: dict | None = None
    for attempt in range(MAX_ATTEMPTS):
        payload = render_payload(arc, geom, session_idx, plan, rendered,
                                 slots, exemplars, hedge_sentence,
                                 prior_value, attempt, feedback,
                                 previous_attempt=last_out, salt=salt)
        out = await llm.call_llm(
            render_sys, json.dumps(payload, ensure_ascii=False),
            trace_tag=f"unify-render-{arc['arc_id']}-s{session_idx}-a{attempt}")
        last_out = out
        if has_hedge and isinstance(out.get("hedge_sentence"), str):
            arc["hedge_text"] = out["hedge_sentence"]   # for distance checks
        fails = lint_session(arc, geom, session_idx, out, prior_turns, slots,
                             banned, exemplars, prior_value)
        selfcheck: list[str] = []
        if not fails and session_idx == 0:
            selfcheck, last_verdict = await r4.prior_selfcheck(
                arc, geom, out, prior_value)
        if not fails and not selfcheck:
            meta = {"attempts": attempt + 1,
                    "exemplar_ids": [e["id"] for e in exemplars]}
            if session_idx == 0:
                meta["prior_selfcheck"] = {"result": "passed", **last_verdict}
            return out, meta
        feedback = fails + selfcheck
    raise RuntimeError(f"{arc['arc_id']} session {session_idx}: lint still "
                       f"failing after {MAX_ATTEMPTS} attempts: {feedback}")


# ---------------------------------------------------------------------------
# Arc assembly
# ---------------------------------------------------------------------------

CARRIED_FIELDS = (
    "arc_id", "row_kind", "stratum", "unify_ladder", "unify_level",
    "m2_level", "resolution_type", "gold_resolution_status", "hedge_strength",
    "designed_gold_class", "designed_gold_route", "taxonomy_rule",
    "stimulus_contract", "stimulus_construction_label", "key",
    "pending_window_user_turns", "seed")


def assemble_arc(src: dict, geom: dict, plan: dict, sessions_out: list[dict],
                 slots: list[dict], metas: list[dict], plan_attempts: int,
                 plan_sha: str, render_sha: str) -> dict:
    arc = {k: src[k] for k in CARRIED_FIELDS}
    pe, h, r_ = geom["prior_establish"], geom["hedge"], geom["resolution"]
    prior_ti, hedge_ti, res_ti = (2 * pe["user_turn"], 2 * h["user_turn"],
                                  2 * r_["user_turn"])
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
    prior_sentence = sessions_out[0]["prior_sentence"]
    hedge_sentence = sessions_out[h["session_idx"]]["hedge_sentence"]
    res_sentence = sessions_out[r_["session_idx"]]["resolution_sentence"]
    prior_value = plan["prior_value"]
    selfcheck = metas[0].get("prior_selfcheck", {"result": "passed"})
    res_meta = RESOLUTION_META[src["resolution_type"]]
    arc.update({
        "value": src["value"],
        "value_candidates": src["value_candidates"],
        "prior_value": prior_value,
        "prior_establish": {"session_idx": pe["session_idx"],
                            "turn_idx": prior_ti, "position": "in_turn",
                            "date": geom["sessions"][0]["turn_dates"][prior_ti]},
        "prior_text": prior_sentence,
        "hedge": {
            "inj_type": "unify_change_statement",
            "contract": src["stimulus_contract"],
            "construction_label": src["stimulus_construction_label"],
            "session_idx": h["session_idx"], "turn_idx": hedge_ti,
            "position": "in_turn",
            "date": geom["sessions"][h["session_idx"]]["turn_dates"][hedge_ti],
        },
        "resolution": {
            "inj_type": res_meta["inj_type"],
            "contract": res_meta["contract"],
            "construction_label": res_meta["construction_label"],
            "t3_variant_kind": None,
            "session_idx": r_["session_idx"], "turn_idx": res_ti,
            "position": "in_turn",
            "date": geom["sessions"][r_["session_idx"]]["turn_dates"][res_ti],
        },
        "hedge_text": hedge_sentence,
        "resolution_text": res_sentence,
        "conversation": conversation,
        "realized_pending_window_user_turns":
            geom["realized_pending_window_user_turns"],
        "render_round": RENDER_ROUND,
        "plan_prompt_sha": plan_sha,
        "render_prompt_sha": render_sha,
        "lints": {
            "prior": {"n_words": len(prior_sentence.split()),
                      "realized_cue_family":
                          ha.detect_cue_family(prior_sentence)[0],
                      "current_frame_detected":
                          r4.asserts_prior_current(prior_sentence, prior_value)},
            "hedge": {**ha._lint(hedge_sentence),
                      "date_token": bool(DATE_TOKEN.search(hedge_sentence)),
                      "marker_union_hits": _marker_union_hits(
                          hedge_sentence,
                          mask_dates=(src["unify_level"]
                                      == "g4_scheduled_event"))},
            "resolution": {**ha._lint(res_sentence),
                           "value_in_sentence": res_sentence.count(
                               src["value"])},
            "fm4_probe_value_free": True,
            "prior_selfcheck": selfcheck,
            "factor_lint": "passed (battery in unify_arcs)",
        },
        "render": {
            "pipeline": ("substrate-unification v5 themed synthetic "
                         "conversation (prompts/unify_conv_plan_v5.txt + "
                         "unify_conv_render_v5.txt)"),
            "plan_prompt_sha": plan_sha,
            "render_prompt_sha": render_sha,
            "style": "human user register + bounded assistant register "
                     "(M4 r4 contract carried)",
        },
        "unify": {
            "factor": {"ladder": src["unify_ladder"],
                       "level": src["unify_level"],
                       "m2_level": src["m2_level"]},
            "geometry": {k: geom[k] for k in
                         ("n_sessions", "n_user_turns", "prior_establish",
                          "hedge", "resolution", "distractor_session_idx",
                          "target_pending_window_user_turns",
                          "realized_pending_window_user_turns")},
            "session_roles": [s["role"] for s in geom["sessions"]],
            "plan_outline": plan,
            "noise_ledger": [{k: s[k] for k in
                              ("session_idx", "user_turn", "turn_idx",
                               "noise", "target_words", "words_min",
                               "words_max", "is_prior", "is_hedge",
                               "is_resolution")}
                             for s in slots],
            "attempts": {"plan": plan_attempts,
                         "sessions": [m["attempts"] for m in metas]},
            "exemplar_ids": {str(si): m["exemplar_ids"]
                             for si, m in enumerate(metas)},
            "prior_selfcheck": selfcheck,
        },
        "status": "pending_owner_review",
        "usable": False,
    })
    return arc


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------

async def run_generate(limit: int | None, fresh_cap: int | None,
                       salt: int | None = None) -> dict:
    specs, keys = build_specs()
    _assert_spec_frozen(specs)
    if not SPEC_PATH.exists():
        ha._write_jsonl(KEYS_PATH, keys)    # FM-4: keys before any value
        ha._write_jsonl(SPEC_PATH, specs)
    plan_sys = PLAN_PROMPT_PATH.read_text()
    render_sys = RENDER_PROMPT_PATH.read_text()
    plan_sha = hashlib.sha256(plan_sys.encode()).hexdigest()
    render_sha = hashlib.sha256(render_sys.encode()).hexdigest()
    banned = r2.load_banned_patterns()
    pool = r3.load_exemplar_pool()

    geoms = {a["arc_id"]: conversation_geometry(a) for a in specs}
    slots = build_slots(specs, geoms)
    assign_noise(slots)
    assign_word_bands(slots)
    slots_by_arc: dict[str, dict[int, list[dict]]] = {}
    for s in slots:
        slots_by_arc.setdefault(s["arc_id"], {}).setdefault(
            s["session_idx"], []).append(s)

    existing = {a["arc_id"]: a for a in ha._read_jsonl(ARCS_PATH)
                if a.get("render_round") == RENDER_ROUND
                and a.get("conversation")}
    todo = [a for a in specs if a["arc_id"] not in existing]
    if limit is not None:
        todo = todo[:limit]
    done: dict[str, dict] = dict(existing)
    failures: list[str] = []

    async def one(src: dict) -> None:
        try:
            src = dict(src)
            geom = geoms[src["arc_id"]]
            plan, plan_attempts = await plan_conversation(src, geom, plan_sys,
                                                          salt=salt)
            src["value"] = plan["value_candidates"][0]
            src["value_candidates"] = plan["value_candidates"]
            arc_slots = [s for sl in slots_by_arc[src["arc_id"]].values()
                         for s in sl]
            apply_beat_floors(arc_slots, src["value"])
            sessions_out: list[dict] = []
            metas: list[dict] = []
            rendered: list[dict] = []
            prior_turns: list[dict] = []
            hedge_sentence: str | None = None
            for si in range(geom["n_sessions"]):
                out, meta = await render_session(
                    src, geom, si, plan, rendered, prior_turns,
                    slots_by_arc[src["arc_id"]][si], pool, banned,
                    render_sys, hedge_sentence, salt=salt)
                sessions_out.append(out)
                metas.append(meta)
                rendered.append({"session_idx": si,
                                 "date": geom["sessions"][si]["date"],
                                 "turns": out["turns"]})
                prior_turns.extend(out["turns"])
                if geom["hedge"]["session_idx"] == si:
                    hedge_sentence = out["hedge_sentence"]
                    src["hedge_text"] = hedge_sentence
            done[src["arc_id"]] = assemble_arc(
                src, geom, plan, sessions_out, arc_slots, metas,
                plan_attempts, plan_sha, render_sha)
        except Exception as exc:
            failures.append(f"{src['arc_id']}: {exc}")

    guard = unify_guard(fresh_cap)
    guard.install()
    try:
        await asyncio.gather(*(one(a) for a in todo))
    finally:
        guard.uninstall()

    rows = [done[a["arc_id"]] for a in specs if a["arc_id"] in done]
    ha._write_jsonl(ARCS_PATH, rows)
    attempts_gt1 = {}
    lint_retry_arcs = 0
    for a in rows:
        att = a["unify"]["attempts"]
        retried = att["plan"] > 1 or any(n > 1 for n in att["sessions"])
        lint_retry_arcs += retried
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
    balance = {lv: sum(1 for a in rows if a["unify_level"] == lv)
               for lv in ALL_LEVELS}
    summary = {
        "id": "UNIFY-V5-GENERATE",
        "prereg": ("protocol/drafts/unify_prereg.md (r4 contract carried: "
                   "protocol/CHANGELOG.md 2026-07-09 'M4 r4 generation "
                   "contract FROZEN'); plan sha " + plan_sha + ", render sha "
                   + render_sha),
        "n_planned": len(specs),
        "n_generated": len(rows),
        "plan_prompt_sha": plan_sha,
        "render_prompt_sha": render_sha,
        "banned_patterns_enforced": len(banned),
        "n_user_turn_slots": len(slots),
        "noise_directive_counts": noise_counts,
        "session_counts": {str(k): sum(1 for a in specs
                                       if geoms[a["arc_id"]]["n_sessions"] == k)
                           for k in (2, 3)},
        "balance": balance,
        "realized_window": {
            "n_exact": sum(1 for t, r_ in windows if t == r_),
            "mean_abs_dev": (round(statistics.fmean(
                abs(t - r_) for t, r_ in windows), 3) if windows else None),
            "max_abs_dev": (max((abs(t - r_) for t, r_ in windows),
                                default=None)),
        },
        "attempts_gt1": attempts_gt1,
        "n_arcs_with_lint_retries": lint_retry_arcs,
        "budget": {**advisory_note(), "fresh_calls_this_run": guard.fresh},
        "failures": failures,
        "outputs": {"arcs": ha._rel(ARCS_PATH), "spec": ha._rel(SPEC_PATH),
                    "keys": ha._rel(KEYS_PATH)},
    }
    ha._write_json(SUMMARY_PATH, summary)
    if failures:
        for f in failures:
            print(f"UNIFY GENERATE FAILURE: {f}", file=sys.stderr)
    return summary


# ---------------------------------------------------------------------------
# Integrity relint (stored conversations, no LLM)
# ---------------------------------------------------------------------------

def relint_arcs(arcs: list[dict]) -> dict:
    banned = r2.load_banned_patterns()
    pool = r3.load_exemplar_pool()
    by_id = {e["id"]: e for c in pool.values() for e in c}
    failures: dict[str, list[str]] = {}
    for arc in arcs:
        geom = conversation_geometry(arc)
        slots = build_slots([arc], {arc["arc_id"]: geom})
        ledger = {(s["session_idx"], s["turn_idx"]): s
                  for s in arc["unify"]["noise_ledger"]}
        for s in slots:
            rec = ledger[(s["session_idx"], s["turn_idx"])]
            s.update({k: rec[k] for k in ("noise", "target_words",
                                          "words_min", "words_max")})
        by_sess: dict[int, list[dict]] = {}
        for s in slots:
            by_sess.setdefault(s["session_idx"], []).append(s)
        fails: list[str] = []
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
        prior_value = arc["prior_value"]
        prior_turns: list[dict] = []
        for si, sess in enumerate(arc["conversation"]):
            out = {"turns": [{"role": t["role"], "text": t["text"]}
                             for t in sess["turns"]],
                   "prior_sentence": (arc["prior_text"] if si == 0 else None),
                   "hedge_sentence": (arc["hedge_text"]
                                      if geom["hedge"]["session_idx"] == si
                                      else None),
                   "resolution_sentence": (
                       arc["resolution_text"]
                       if geom["resolution"]["session_idx"] == si else None)}
            exemplars = [by_id[i]
                         for i in arc["unify"]["exemplar_ids"][str(si)]]
            fails += lint_session(arc, geom, si, out, prior_turns,
                                  by_sess[si], banned, exemplars, prior_value)
            prior_turns.extend(out["turns"])
        if not ha._fm4_probe_ok(arc["key"]["probe_question"], arc["value"]):
            fails.append("FM-4: value in probe question")
        if arc["realized_pending_window_user_turns"] != geom[
                "realized_pending_window_user_turns"]:
            fails.append("realized window drifted from deterministic geometry")
        sc = (arc.get("unify", {}).get("prior_selfcheck") or {})
        if sc.get("result") != "passed":
            fails.append("prior self-check not recorded as passed")
        if fails:
            failures[arc["arc_id"]] = fails
    return {"n_arcs": len(arcs), "all_green": not failures,
            "failures": failures}


# ---------------------------------------------------------------------------
# Checks — register + factor manifestation + prior classification
# ---------------------------------------------------------------------------

async def prior_classification_report(arcs: list[dict]) -> dict:
    """r4's released prior metric (frozen extractor + key_mapper + classifier
    over every prior-establish turn; cache-shared with generation)."""
    import classify
    import gated_store
    import store
    import syn_res
    extractor_sys = (PROMPTS_DIR / syn_res.LME_PROMPTS["em"]).read_text()
    rows = []

    async def one(arc):
        turn0 = arc["conversation"][0]["turns"][0]
        key = arc["key"]
        cand = store.CanonicalKey(entity=key["entity"],
                                  attribute=key["attribute"],
                                  description=key["description"],
                                  probe_question=key["probe_question"])
        tag = f"unify-priorxk-{arc['arc_id']}"
        tuples = await gated_store.extract_turn_family(
            [], turn0["text"], tag, system_prompt=extractor_sys,
            current_date=turn0["date"])
        mapped = await asyncio.gather(*[
            store._map_tuple_key(t, [cand], trace_tag=f"{tag}-map{j}")
            for j, t in enumerate(tuples)]) if tuples else []
        key_tuples = [(t, m) for t, (m, _r, _rat) in zip(tuples, mapped)
                      if m is not None and m.key == cand.key]
        cls = None
        vmatch = False
        if key_tuples:
            def vm(t):
                return (_norm(arc["prior_value"]) in _norm(t.value)
                        or _norm(t.value) in _norm(arc["prior_value"]))
            key_tuples.sort(key=lambda tm: 0 if vm(tm[0]) else 1)
            tup, _m = key_tuples[0]
            vmatch = vm(tup)
            item = {"item_id": f"unify-priorcls-{arc['arc_id']}",
                    "entity": tup.entity, "attribute": tup.attribute,
                    "asserted_value": tup.value, "stored_value": None,
                    "turn_text": turn0["text"], "preceding_turns": []}
            cls = (await classify.classify_item(item))["argmax"]
        rows.append({"arc_id": arc["arc_id"], "n_tuples": len(tuples),
                     "mapped_to_key": bool(key_tuples), "class": cls,
                     "value_match": vmatch,
                     "class_ok": cls in r4.PRIOR_OK_CLASSES})

    await asyncio.gather(*(one(a) for a in arcs))
    rows.sort(key=lambda r_: r_["arc_id"])
    class_dist: dict[str, int] = {}
    for r_ in rows:
        class_dist[str(r_["class"])] = class_dist.get(str(r_["class"]), 0) + 1
    return {
        "n_arcs": len(rows),
        "prior_established": sum(1 for a in arcs
                                 if a.get("prior_text") and a.get("prior_value")),
        "mapped_to_key": sum(1 for r_ in rows if r_["mapped_to_key"]),
        "class_completed_event_or_other": sum(1 for r_ in rows
                                              if r_["class_ok"]),
        "value_match": sum(1 for r_ in rows if r_["value_match"]),
        "class_distribution": class_dist,
        "not_ok": [r_ for r_ in rows
                   if not (r_["mapped_to_key"] and r_["class_ok"])],
        "rows": rows,
    }


def factor_manifestation_table(arcs: list[dict]) -> dict:
    """Per-level realized-cue table — the release gate's evidence that every
    assigned factor level MANIFESTS and no cross-level cue leaks."""
    table: dict[str, dict] = {}
    for lv in ALL_LEVELS:
        rows = [a for a in arcs if a["unify_level"] == lv]
        if not rows:
            table[lv] = {"n": 0}
            continue
        hs = [a["hedge_text"] for a in rows]
        rs = [a["resolution_text"] for a in rows]
        fam_counts: dict[str, int] = {}
        for h in hs:
            fam = str(ha.detect_cue_family(h)[0])
            fam_counts[fam] = fam_counts.get(fam, 0) + 1
        entry = {
            "n": len(rows),
            "hedge_realized_cue_family": fam_counts,
            "hedge_date_token": sum(1 for h in hs if DATE_TOKEN.search(h)),
            "hedge_marker_union_hits": sum(
                1 for a in rows if _marker_union_hits(
                    a["hedge_text"],
                    mask_dates=(lv == "g4_scheduled_event"))),
            "hedge_intending_cue": sum(1 for h in hs
                                       if _fam_hit(h, _INTEND_RES)),
            "hedge_deliberating_cue": sum(1 for h in hs
                                          if _fam_hit(h, _DELIB_RES)),
            "hedge_adoption_frame": sum(
                1 for a in rows if any(p.search(a["hedge_text"])
                                       for p in r3._adoption_regexes(
                                           a["value"]))),
            "resolution_open_cue": sum(1 for r_ in rs
                                       if r2.OPEN_CUE.search(r_)),
            "resolution_cue_family_none": sum(
                1 for r_ in rs if ha.detect_cue_family(r_)[0] is None),
        }
        if lv in CW_LEVELS:
            shared6 = sum(1 for a in rows if _shared_sixgram(
                a["resolution_text"], a["hedge_text"],
                [a["value"]] + a["value_candidates"]))
            shared_tokens = [len(_content_tokens(a["resolution_text"],
                                                 [a["value"]])
                                 & _content_tokens(a["hedge_text"],
                                                   [a["value"]]))
                             for a in rows]
            entry.update({
                "resolution_shared_sixgram_with_hedge": shared6,
                "resolution_shared_content_tokens_mean": round(
                    statistics.fmean(shared_tokens), 2),
                "resolution_value_named": sum(
                    1 for a in rows
                    if a["resolution_text"].count(a["value"]) >= 1),
                "resolution_completion_frame": sum(
                    1 for a in rows if _completion_frame(
                        a["resolution_text"], a["value"])),
                "resolution_attr_noun_hits": sum(
                    1 for a in rows if any(
                        tok not in _norm(a["value"])
                        and re.search(rf"(?<![\w]){re.escape(tok)}(?![\w])",
                                      a["resolution_text"].lower())
                        for tok in ge._attr_ban_tokens(
                            a["key"]["attribute"]))),
            })
        table[lv] = entry
    return table


async def run_checks() -> dict:
    banned = r2.load_banned_patterns()
    arcs = ha._read_jsonl(ARCS_PATH)
    if not arcs:
        raise SystemExit(f"no unify arcs at {ARCS_PATH}")
    user_turns = [t["text"] for a in arcs for s in a["conversation"]
                  for t in s["turns"] if t["role"] == "user"]
    assistant_turns = [t["text"] for a in arcs for s in a["conversation"]
                       for t in s["turns"] if t["role"] == "assistant"]
    hosts = json.loads(r2.HOSTS_ANALYSIS_PATH.read_text())["register_stats"]
    ext = json.loads(r2.EXTERNAL_ANALYSIS_PATH.read_text())["register_stats"]
    syco_hits = [t for t in user_turns
                 if any(rx.search(t) for _, rx in r3.SYCOPHANCY_COMPILED)]
    anti_hits = [t for t in user_turns
                 if any(rx.search(t) for _, rx in r2.EXTERNAL_ANTI_COMPILED)]
    integrity = relint_arcs(arcs)
    factor_residual = {aid: [f for f in fs if f.startswith("FACTOR")]
                       for aid, fs in integrity["failures"].items()}
    factor_residual = {k: v for k, v in factor_residual.items() if v}
    coherence_residual = {aid: [f for f in fs if f.startswith("COHERENCE")]
                          for aid, fs in integrity["failures"].items()}
    coherence_residual = {k: v for k, v in coherence_residual.items() if v}
    balance = {lv: sum(1 for a in arcs if a["unify_level"] == lv)
               for lv in ALL_LEVELS}
    windows = [(a["pending_window_user_turns"],
                a["realized_pending_window_user_turns"]) for a in arcs]
    prior_report = await prior_classification_report(arcs)
    doc = {
        "id": "UNIFY-V5-CHECKS",
        "method": ("register metrics (hedge_arcs_r2.register_metrics) over "
                   "ALL unify user turns vs the frozen human-corpus + "
                   "host-LME-user stats; deterministic integrity relint (r3 "
                   "register battery + r4 prior/coherence + v5 factor "
                   "manifestation); the frozen extractor + key_mapper + "
                   "classifier re-run over the 180 prior-establish turns"),
        "contract_shas": {"plan": hashlib.sha256(
            PLAN_PROMPT_PATH.read_bytes()).hexdigest(),
            "render": hashlib.sha256(
                RENDER_PROMPT_PATH.read_bytes()).hexdigest()},
        "banned_regex_union": {
            "n_patterns": len(banned),
            "unify_user_turns": r2.banned_union_hits(user_turns, banned),
        },
        "sycophancy_union": {"n_patterns": len(r3.SYCOPHANCY_COMPILED),
                             "n_flagged": len(syco_hits),
                             "flagged_texts": syco_hits},
        "external_anti_pattern_union": {"n_flagged": len(anti_hits),
                                        "flagged_texts": anti_hits},
        "register": {
            "unify_user_turns": r2.register_metrics(user_turns),
            "unify_assistant_turns": r2.register_metrics(assistant_turns),
            "host_lme_user": hosts["lme_user"],
            "human_corpora": {k: ext[k] for k in sorted(ext) if k != "cc_llm"},
            "human_pooled": r3._pooled_human_register(ext),
        },
        "balance": balance,
        "factor_manifestation": factor_manifestation_table(arcs),
        "factor_lint": {"residual_arcs": len(factor_residual),
                        "residual": factor_residual},
        "coherence_lint": {"residual_arcs": len(coherence_residual),
                           "residual": coherence_residual},
        "prior_value": prior_report,
        "realized_window": {
            "pairs_target_realized": windows,
            "n_exact": sum(1 for t, r_ in windows if t == r_),
            "mean_abs_dev": round(statistics.fmean(abs(t - r_)
                                                   for t, r_ in windows), 3),
            "max_abs_dev": max(abs(t - r_) for t, r_ in windows),
        },
        "integrity_relint": integrity,
    }
    ha._write_json(CHECKS_PATH, doc)
    return doc


# ---------------------------------------------------------------------------
# Blind-panel INPUTS (label-free transcripts; the panels are maintainer-run)
# ---------------------------------------------------------------------------

def write_panel_inputs(out_dir: Path) -> dict:
    arcs = ha._read_jsonl(ARCS_PATH)
    if not arcs:
        raise SystemExit(f"no unify arcs at {ARCS_PATH}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for a in arcs:
        lines = [f"# {a['arc_id']}", ""]
        for sess in a["conversation"]:
            lines.append(f"## session {sess['session_idx']} — {sess['date']}")
            lines.append("")
            for t in sess["turns"]:
                lines.append(f"[{t['date']}] {t['role']}: {t['text']}")
            lines.append("")
        (out_dir / f"{a['arc_id']}.txt").write_text("\n".join(lines))
    (out_dir / "README.md").write_text(
        "# unify blind-panel INPUTS\n\n"
        f"{len(arcs)} label-free conversation transcripts (one per arc) for "
        "the maintainer-run blind judge panels (register vs human anchors; "
        "level recovery; in-context sense). Factor levels, outcomes, beat "
        "positions, and every other label live ONLY in results/unify/"
        "arcs.jsonl — do not ship labels to judges. Written by "
        "experiments/v2_full/unify_arcs.py write-panels.\n")
    return {"n_files": len(arcs), "dir": str(out_dir)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="substrate unification — planted factor ladders as "
                    "conversations (v5)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    gen = sub.add_parser("generate", help="generate all 180 conversations")
    gen.add_argument("--limit", type=int, default=None)
    gen.add_argument("--fresh-cap", type=int, default=None)
    gen.add_argument("--salt", type=int, default=None,
                     help="retry-cycle marker for POST-FAILURE reruns only: "
                          "adds retry_cycle to the payloads (fresh cache "
                          "keys) so an arc whose <=3-retry cycle dead-ended "
                          "on cached responses gets a new cycle; never used "
                          "on first-pass generation")
    sub.add_parser("checks", help="register + factor manifestation + prior "
                                  "classification")
    pan = sub.add_parser("write-panels", help="label-free per-arc panel "
                                              "inputs")
    pan.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    if args.cmd == "generate":
        summary = asyncio.run(run_generate(args.limit, args.fresh_cap,
                                           salt=args.salt))
        print(json.dumps({k: summary[k] for k in (
            "n_planned", "n_generated", "n_user_turn_slots",
            "noise_directive_counts", "session_counts", "balance",
            "realized_window", "attempts_gt1", "n_arcs_with_lint_retries",
            "budget", "failures")}, indent=1))
        print(f"-> {ARCS_PATH}\n-> {SUMMARY_PATH}")
        if summary["failures"]:
            raise SystemExit(f"{len(summary['failures'])} arc(s) failed "
                             "(progress persisted; rerun resumes)")
    elif args.cmd == "checks":
        doc = asyncio.run(run_checks())
        print(json.dumps({
            "banned_union_unify_user_turns": {
                k: doc["banned_regex_union"]["unify_user_turns"][k]
                for k in ("n_texts", "n_flagged", "per_pattern_nonzero")},
            "sycophancy_flagged": doc["sycophancy_union"]["n_flagged"],
            "unify_user_words": doc["register"]["unify_user_turns"]["words"],
            "balance": doc["balance"],
            "factor_residual_arcs": doc["factor_lint"]["residual_arcs"],
            "coherence_residual_arcs":
                doc["coherence_lint"]["residual_arcs"],
            "prior_value": {k: doc["prior_value"][k] for k in (
                "prior_established", "mapped_to_key",
                "class_completed_event_or_other", "value_match",
                "class_distribution")},
            "realized_window": {k: doc["realized_window"][k]
                                for k in ("n_exact", "mean_abs_dev",
                                          "max_abs_dev")},
            "integrity_all_green": doc["integrity_relint"]["all_green"],
        }, indent=1))
        print(f"-> {CHECKS_PATH}")
    elif args.cmd == "write-panels":
        print(json.dumps(write_panel_inputs(Path(args.out)), indent=1))


if __name__ == "__main__":
    main()
