"""hedge_arcs_r4.py — v2.4 M4 round-4 REGENERATION: 60 themed multi-session
conversations with resolution-session COHERENCE and an established PRIOR value
(protocol/CHANGELOG.md 2026-07-09 "M4 owner review round 3 ... r4 full
regeneration authorized" + "M4 r4 generation contract FROZEN").

Round 3 was accepted on validity+register but had a structural coherence
defect: ~30/60 resolution sessions deliberated the same decision and THEN
reported it settled (e.g. request a "final matrix" at turn 6, report the
purchase at turn 8 — a contradiction). Round 4 keeps the r3 themed-conversation
design and adds TWO owner-approved changes:

  CHANGE 1 (coherence): the resolution is the FIRST user turn of its session
  (a natural new-conversation opener reporting the outcome); nothing before it
  in that session re-deliberates the decision (there is nothing before it). The
  rest of the resolution session is the CONSEQUENCE of the outcome. The
  pending-window turns live in the hedge-session tail after the hedge and/or an
  OPTIONAL distractor middle session on a DIFFERENT sub-topic (no
  value/candidate mention -> no re-litigation, so the checker's no-false-fire
  test is preserved).

  CHANGE 2 (prior value): every arc first establishes the user's CURRENT value
  for the key as a plain present-tense settled FACT in session 0 user turn 0,
  so the hedge is a switch from prior_value to VALUE (injector_spec §5.3
  v_a/v_spec). prior_value is planner-emitted and DISTINCT from VALUE, every
  value_candidate, and any adopted alternative; it must classify as
  completed_event/other and map to the arc key (a frozen extractor + key_mapper
  + classifier self-check in the session-0 render loop). abandoned arcs carry
  abandoned_mode in {kept_prior, alternative} (correct memory = prior_value or
  alternative_value); open correct memory = prior_value; resolve_true = VALUE.

The frozen PLAN carries over untouched (arc ids, keys, VALUE, value_candidates,
outcome balance 20/20/20, hedge_strength, pending-window targets, seeds); only
the conversational realization changes. Budget: every fresh call under
SharedCallBudget stage "m4" (ADVISORY stages, hard 120k total envelope).

CLI (repo root, pilot/.venv/bin/python experiments/v2_full/hedge_arcs_r4.py ...):
  r4-generate  [--limit K] [--fresh-cap N]  plan+render all 60 -> arcs.jsonl
  checks                                    r4 register + coherence + prior
                                            classification -> r4_checks.json
  review-pack                               rebuild review_pack.md (3 blocks)
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
import hedge_arcs_r3 as r3
import llm
import classify
import gated_store
import store
import syn_res
from inject import _norm

from paths import PROMPTS_DIR

SEED_M4 = ha.SEED_M4
M4_DIR = ha.M4_DIR
ARCS_R3_PATH = M4_DIR / "arcs_r3.jsonl"
ARCS_PATH = ha.ARCS_PATH                 # results/v2.4/m4/arcs.jsonl (r4 out)
PLAN_PROMPT_PATH = PROMPTS_DIR / "m4_conv_plan_r4.txt"
RENDER_PROMPT_PATH = PROMPTS_DIR / "m4_conv_render_r4.txt"
R4_SUMMARY_PATH = M4_DIR / "r4_render_summary.json"
R4_CHECK_PATH = M4_DIR / "r4_checks.json"

MAX_ATTEMPTS = 4                         # 1 initial + <=3 lint-feedback retries
MAX_SINGLE_DISTRACTOR_WINDOW = 11        # capacity cap: rest0(<=4)+n_mid(<=6)+1
PRIOR_OK_CLASSES = ("completed_event", "other")

#: future-plan frames the prior-establish sentence must NOT carry (it is a
#: settled PRESENT fact, not a plan). detect_cue_family already covers the
#: hedge-strength families; this catches bare "I will/I'm going to switch".
_FUTURE_FRAME = re.compile(
    r"(?i)\b(i|we)('ll| will| am going to|'m going to| are going to"
    r"|'re going to)\b|\bgonna\b|\babout to\b|\bplan(ning)? to\b"
    r"|\bhoping to\b|\bnext (week|month|year)\b|\bonce (i|we)\b")


# ---------------------------------------------------------------------------
# Prior-value current-state detector (deterministic "maps-to-key" proxy;
# the LLM self-check is the semantic gate)
# ---------------------------------------------------------------------------

def asserts_prior_current(text: str, prior_value: str) -> bool:
    """True when `text` asserts `prior_value` as the user's CURRENT value for
    the key — a settled present-tense fact (habit / possession / current
    provider). Reuses the r3 value-adoption regexes plus present-habit frames.
    A mere mention (comparison / recommendation) does not trip it."""
    if prior_value not in text and _norm(prior_value) not in _norm(text):
        return False
    if r3.asserts_value_adopted(text, prior_value):
        return True
    v = re.escape(prior_value)
    frames = [
        rf"(?i)\b(i|we)('ve| have) been [\w' ]{{0,24}}?"
        rf"(going to|using|getting|seeing|shopping at|buying|ordering|on)"
        rf"[\w' ]{{0,24}}?{v}",
        rf"(?i)\b(i|we)('ve| have) been [\w' ]{{0,18}}?{v}",
        rf"(?i)\bmy (current|usual|regular|go-?to)[\w' \-]{{0,24}}?{v}",
        rf"(?i){v}\b[\w' ]{{0,18}}?(is|are|has been) my "
        rf"(current|usual|regular|go-?to)\b",
        rf"(?i)\b(i|we)('ve| have) got (the |a |an |my )?{v}",
        rf"(?i)\bright now[\w', ]{{0,18}}?{v}",
        rf"(?i)\b(i|we) (still )?(use|go to|get|buy|order|see|bank with|fly"
        rf"|drive|run in|wear|take|have|stick with|stay with|shop at) "
        rf"(the |a |an |my )?{v}",
    ]
    return any(re.search(p, text) for p in frames)


# ---------------------------------------------------------------------------
# Geometry (deterministic; resolution at final-session turn 0)
# ---------------------------------------------------------------------------

def _rng(arc_id: str, what: str) -> random.Random:
    return random.Random(f"m4r4:{SEED_M4}:{arc_id}:{what}")


def conversation_geometry(arc: dict) -> dict:
    """Sessions/turn counts/roles/beat slots/dates for one arc.

    Structure: session 0 (role 'hedge') establishes the prior value at user
    turn 0 and voices the hedge at user turn h (>=1); an OPTIONAL distractor
    middle session (role 'distractor') spaces the timeline; the final session
    (role 'resolution') opens with the resolution at user turn 0. Realized
    pending window = user turns strictly after the hedge up to and including
    the resolution turn = rest0 (session-0 tail) + n_mid (distractor turns) + 1.
    The resolution-at-turn-0 rule caps a single-distractor conversation at
    window 11; larger targets clamp (realized recorded)."""
    W = arc["pending_window_user_turns"]
    rng = _rng(arc["arc_id"], "geometry")
    LO, HI = r3.N_USER_LO, r3.N_USER_HI                       # 3, 6

    # enumerate valid shapes; rest0 = session-0 tail after the hedge in
    # [0, n0-2] (hedge user turn = n0-1-rest0 >= 1; prior-establish sits at 0)
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
        return (abs(window - W), n_sessions)          # closest, then fewer

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


# ---------------------------------------------------------------------------
# abandoned_mode split (deterministic seeded batch: 10 alternative / 10 kept)
# ---------------------------------------------------------------------------

def assign_abandoned_modes(arcs: list[dict]) -> dict[str, tuple]:
    """Return {arc_id: (mode | None, alternative_value | None)}. Over the 20
    abandoned arcs, exactly half are 'alternative' (alternative_value =
    value_candidates[1], the first non-VALUE candidate); the rest 'kept_prior'.
    Non-abandoned arcs map to (None, None)."""
    out = {a["arc_id"]: (None, None) for a in arcs}
    ab = [a for a in arcs if a["resolution_type"] == "abandoned"]
    order = list(range(len(ab)))
    random.Random(f"m4r4-abmode:{SEED_M4}").shuffle(order)
    n_alt = len(ab) // 2
    alt = set(order[:n_alt])
    for i, a in enumerate(ab):
        if i in alt:
            cands = a["value_candidates"]
            assert _norm(cands[0]) == _norm(a["value"]), a["arc_id"]
            out[a["arc_id"]] = ("alternative", cands[1])
        else:
            out[a["arc_id"]] = ("kept_prior", None)
    return out


# ---------------------------------------------------------------------------
# Per-turn directives (word bands + batch noise; prior turn joins the pool)
# ---------------------------------------------------------------------------

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
                    "value": arc["value"],
                })
    return slots


def assign_noise(slots: list[dict]) -> None:
    """Batch-level seeded quotas (r3 quotas), max ONE feature per slot.
    Fragment never on prior/hedge/resolution beat turns; exclamation never on
    an abandoned-arc resolution beat."""
    for s in slots:
        s["noise"] = "none"
    order = list(range(len(slots)))
    random.Random(f"m4r4-noise:{SEED_M4}").shuffle(order)
    n = len(slots)

    def eligible(s: dict, feature: str) -> bool:
        if s["noise"] != "none":
            return False
        if (feature == "fragment"
                and (s["is_hedge"] or s["is_resolution"] or s["is_prior"])):
            return False
        if (feature == "exclamation" and s["is_resolution"]
                and s["resolution_type"] == "abandoned"):
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
    for s in slots:
        rng = random.Random(f"m4r4:{SEED_M4}:{s['arc_id']}:s{s['session_idx']}"
                            f":u{s['user_turn']}:words")
        if s["noise"] == "fragment":
            t = rng.randint(r3.FRAG_LO, r3.FRAG_HI)
            s["target_words"] = t
            s["words_min"], s["words_max"] = r3.FRAG_LO, r3.FRAG_HI + 1
            continue
        t = round(rng.lognormvariate(math.log(r3.USER_WORDS_MEDIAN),
                                     r3.USER_WORDS_SIGMA))
        t = max(r3.USER_WORDS_LO, min(r3.USER_WORDS_HI, t))
        if s["is_hedge"] or s["is_resolution"] or s["is_prior"]:
            t = max(t, len(s["value"].split()) + r3.BEAT_FLOOR_EXTRA)
        s["target_words"] = t
        s["words_min"] = max(r3.USER_WORDS_LO, t - r3.WORD_TOL)
        s["words_max"] = t + r3.WORD_TOL


# ---------------------------------------------------------------------------
# r4 lint additions (prior beat + coherence); r3 lints reused verbatim
# ---------------------------------------------------------------------------

def _prior_checks(arc: dict, geom: dict, session0_out: dict,
                  prior_value: str, alternative_value: str | None) -> list[str]:
    """Deterministic PRIOR lint on session 0 (the LLM extract/map/classify
    self-check runs separately in the render loop)."""
    fails: list[str] = []
    value = arc["value"]
    cands = arc.get("value_candidates") or []
    if not isinstance(prior_value, str) or not prior_value.strip():
        return ["prior_value missing (emit the user's current value for the key)"]

    def same(a, b):
        return bool(a) and bool(b) and _norm(a) == _norm(b)

    if same(prior_value, value):
        fails.append(f"prior_value {prior_value!r} must be DISTINCT from the "
                     f"hedge value {value!r}")
    for c in cands:
        if same(prior_value, c):
            fails.append(f"prior_value {prior_value!r} must not equal a "
                         f"value_candidate {c!r} (it is the user's CURRENT "
                         "value, not an option being weighed)")
    if same(prior_value, alternative_value):
        fails.append(f"prior_value {prior_value!r} must be DISTINCT from "
                     f"alternative_value {alternative_value!r}")

    turn0 = session0_out["turns"][0]["text"]
    prior_sentence = session0_out.get("prior_sentence")
    if not isinstance(prior_sentence, str) or not prior_sentence:
        fails.append("prior_sentence missing — session 0 user turn 0 must "
                     "establish the current value as a plain fact")
        return fails
    if prior_sentence not in turn0:
        fails.append("prior_sentence must be a verbatim substring of session 0 "
                     "user turn 0")
    if "?" in prior_sentence:
        fails.append("prior_sentence must be a statement, not a question")
    nv = prior_sentence.count(prior_value)
    if nv != 1:
        fails.append(f"prior_sentence must contain prior_value {prior_value!r} "
                     f"verbatim exactly once (found {nv})")
    if turn0.count(prior_value) > 1:
        fails.append("the prior-establish turn mentions prior_value more than "
                     "once — only the establishing sentence carries it")
    if value in turn0 or _norm(value) in _norm(turn0):
        fails.append(f"the prior-establish turn must NOT mention the hedge "
                     f"value {value!r} — it establishes the CURRENT value only")
    # present-tense / non-hedge, enforced NEGATIVELY (no hedge cue, no future
    # frame, no abandonment). The positive "reads as a current fact that maps
    # to the arc key" is the LLM extract+map+classify self-check (prior_selfcheck)
    # — the task's "(self-check) extracts+maps to the arc key under a quick
    # classify". asserts_prior_current is recorded as an informational
    # diagnostic (a brittle positive whitelist is not used as a gate).
    fam = ha.detect_cue_family(prior_sentence)[0]
    if fam is not None:
        fails.append(f"prior_sentence must NOT read as a plan/hedge (detected "
                     f"{fam!r} cue) — state the current value as a settled "
                     "present fact ('i've been going to ...', 'i run in ...')")
    if _FUTURE_FRAME.search(prior_sentence):
        fails.append("prior_sentence must be a settled PRESENT fact, not a "
                     "future plan (no 'I'll/I'm going to/about to/planning to "
                     "switch') — say what is true right now")
    if r3._abandon_cue_any(prior_sentence):
        fails.append("prior_sentence must not contain abandonment language")
    return fails


def _coherence_checks(arc: dict, geom: dict, session_idx: int, out: dict,
                      prior_value: str | None) -> list[str]:
    fails: list[str] = []
    value = arc["value"]
    forbidden = [value] + [c for c in (arc.get("value_candidates") or [])]
    if prior_value:
        forbidden.append(prior_value)
    turns = out["turns"]
    res_sess = geom["resolution"]["session_idx"]

    if session_idx == res_sess and geom["resolution"]["user_turn"] != 0:
        fails.append("COHERENCE: the resolution must be the FIRST user turn of "
                     "its session (a new-conversation opener)")

    if geom.get("distractor_session_idx") == session_idx:
        for k, t in enumerate(turns):
            low = t["text"]
            for c in forbidden:
                if c and (c in low or _norm(c) in _norm(low)):
                    fails.append(f"COHERENCE: distractor session turn {k} "
                                 f"mentions {c!r} — a distractor session is a "
                                 "DIFFERENT sub-topic (no re-litigating the "
                                 "choice, no value/prior/candidate)")
                    break

    if (session_idx == res_sess
            and arc["resolution_type"] in ("resolve_true", "abandoned")):
        res_ti = 2 * geom["resolution"]["user_turn"]
        cset = [value] + [c for c in (arc.get("value_candidates") or [])]
        for k in range(0, len(turns), 2):          # user turns after resolution
            if k <= res_ti:
                continue
            t = turns[k]["text"]
            if (ha.detect_cue_family(t)[0] is not None
                    and any(c and (c in t or _norm(c) in _norm(t))
                            for c in cset)):
                fails.append(f"COHERENCE: resolution-session turn {k} "
                             "re-deliberates the resolved decision (a hedge cue "
                             "near the value/candidate AFTER the outcome) — the "
                             "session must handle the CONSEQUENCE, not re-open "
                             "the choice")
    return fails


def _abandoned_mode_checks(arc: dict, geom: dict, out: dict,
                           abandoned_mode: str | None,
                           alternative_value: str | None) -> list[str]:
    fails: list[str] = []
    if arc["resolution_type"] != "abandoned":
        return fails
    if geom["resolution"]["session_idx"] != geom["resolution"]["session_idx"]:
        return fails
    res_sentence = out.get("resolution_sentence")
    if not isinstance(res_sentence, str) or not res_sentence:
        return fails                                    # r3 lint flags absence
    if abandoned_mode == "alternative":
        if not alternative_value or alternative_value not in res_sentence:
            fails.append(f"abandoned/alternative resolution must name the "
                         f"alternative {alternative_value!r} the user went "
                         "with instead of the value")
    return fails


def lint_session(arc: dict, geom: dict, session_idx: int, out: dict,
                 prior_turns: list[dict], slots: list[dict],
                 banned: list[dict], exemplars: list[dict],
                 prior_value: str | None, alternative_value: str | None,
                 abandoned_mode: str | None) -> list[str]:
    """r3 register + beat + ordering lints, plus the r4 prior/coherence lints.
    Deterministic (no LLM); the prior self-check runs in the render loop."""
    fails = r3.lint_session(arc, geom, session_idx, out, prior_turns, slots,
                            banned, exemplars)
    turns = out.get("turns")
    valid = (isinstance(turns, list)
             and len(turns) == 2 * geom["sessions"][session_idx]["n_user_turns"]
             and all(isinstance(t, dict) and isinstance(t.get("text"), str)
                     for t in turns))
    if not valid:
        return fails
    if session_idx == 0:
        fails += _prior_checks(arc, geom, out, prior_value, alternative_value)
    fails += _coherence_checks(arc, geom, session_idx, out, prior_value)
    if geom["resolution"]["session_idx"] == session_idx:
        fails += _abandoned_mode_checks(arc, geom, out, abandoned_mode,
                                        alternative_value)
    return fails


def lint_plan(arc: dict, geom: dict, out: dict,
              alternative_value: str | None) -> list[str]:
    fails: list[str] = []
    sessions = out.get("sessions")
    if not isinstance(sessions, list) or len(sessions) != geom["n_sessions"]:
        return [f"need exactly {geom['n_sessions']} sessions"]
    if not isinstance(out.get("premise"), str) or not out["premise"].strip():
        fails.append("premise missing")
    prior_value = out.get("prior_value")
    if not isinstance(prior_value, str) or not prior_value.strip():
        fails.append("prior_value missing (emit the user's current value)")
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
    value = arc["value"]
    cands = arc.get("value_candidates") or []

    def same(a, b):
        return bool(a) and bool(b) and _norm(a) == _norm(b)

    if same(prior_value, value):
        fails.append(f"prior_value must differ from the hedge value {value!r}")
    for c in cands:
        if same(prior_value, c):
            fails.append(f"prior_value must not be a value_candidate ({c!r})")
    if same(prior_value, alternative_value):
        fails.append(f"prior_value must differ from alternative_value "
                     f"{alternative_value!r}")
    pe, h, r = geom["prior_establish"], geom["hedge"], geom["resolution"]
    beat = lambda p: sessions[p["session_idx"]]["user_turn_beats"][p["user_turn"]]
    if prior_value and prior_value not in beat(pe):
        fails.append(f"the prior-establish beat (session {pe['session_idx']} "
                     f"user turn {pe['user_turn']}) must contain prior_value "
                     "verbatim")
    if value not in beat(h):
        fails.append(f"the hedge beat (session {h['session_idx']} user turn "
                     f"{h['user_turn']}) must contain the value verbatim")
    if value not in beat(r):
        fails.append(f"the resolution beat (session {r['session_idx']} user "
                     f"turn {r['user_turn']}) must contain the value verbatim")
    if alternative_value and alternative_value not in beat(r):
        fails.append(f"the resolution beat (abandoned/alternative) must name "
                     f"alternative_value {alternative_value!r}")
    return fails


# ---------------------------------------------------------------------------
# LLM prior self-check (frozen extractor + key_mapper + classifier)
# ---------------------------------------------------------------------------

async def prior_selfcheck(arc: dict, geom: dict, session0_out: dict,
                          prior_value: str) -> tuple[list[str], dict]:
    """Run the frozen extractor + key_mapper + classifier on the prior-establish
    turn: a tuple must map to the arc key with class argmax in
    {completed_event, other}. Returns (lint_failures, verdict_record)."""
    turn0 = session0_out["turns"][0]["text"]
    key = arc["key"]
    cand = store.CanonicalKey(entity=key["entity"], attribute=key["attribute"],
                              description=key["description"],
                              probe_question=key["probe_question"])
    date = geom["sessions"][0]["turn_dates"][0]
    tag = f"m4r4-priorxk-{arc['arc_id']}"
    extractor_sys = (PROMPTS_DIR / syn_res.LME_PROMPTS["em"]).read_text()
    tuples = await gated_store.extract_turn_family(
        [], turn0, tag, system_prompt=extractor_sys, current_date=date)
    verdict = {"extracted": len(tuples), "mapped_to_key": False,
               "class": None, "value_match": False}
    if not tuples:
        return (["PRIOR self-check: the frozen extractor found no fact in the "
                 "prior-establish turn — state the current value plainly"],
                verdict)
    mapped = await asyncio.gather(*[
        store._map_tuple_key(t, [cand], trace_tag=f"{tag}-map{j}")
        for j, t in enumerate(tuples)])
    key_tuples = [(t, m) for t, (m, _raw, _rat) in zip(tuples, mapped)
                  if m is not None and m.key == cand.key]
    if not key_tuples:
        return ([f"PRIOR self-check: no extracted tuple maps to the arc key "
                 f"{cand.name} — the establish turn must assert the user's "
                 f"current {key['attribute']} as {prior_value!r}"], verdict)
    verdict["mapped_to_key"] = True

    def vmatch(t):
        return (_norm(prior_value) in _norm(t.value)
                or _norm(t.value) in _norm(prior_value))
    key_tuples.sort(key=lambda tm: 0 if vmatch(tm[0]) else 1)
    tup, _m = key_tuples[0]
    verdict["value_match"] = vmatch(tup)
    item = {"item_id": f"m4r4-priorcls-{arc['arc_id']}", "entity": tup.entity,
            "attribute": tup.attribute, "asserted_value": tup.value,
            "stored_value": None, "turn_text": turn0, "preceding_turns": []}
    cls = await classify.classify_item(item)
    verdict["class"] = cls["argmax"]
    if cls["argmax"] not in PRIOR_OK_CLASSES:
        return ([f"PRIOR self-check: the prior-establish turn classifies as "
                 f"{cls['argmax']!r} — it must read as a settled current fact "
                 "(completed_event/other), NOT a plan/speculation; drop any "
                 "future or hedging wording"], verdict)
    return ([], verdict)


# ---------------------------------------------------------------------------
# Calls
# ---------------------------------------------------------------------------

def plan_payload(arc: dict, geom: dict, abandoned_mode: str | None,
                 alternative_value: str | None, attempt: int,
                 feedback: list[str] | None) -> dict:
    payload = {
        "task": "m4_r4_plan",
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
                                            "days_since_previous", "role")}
                         for s in geom["sessions"]],
            "prior_establish_beat": geom["prior_establish"],
            "hedge_beat": geom["hedge"],
            "resolution_beat": {
                **geom["resolution"],
                "kind": ("still_pending_mention"
                         if arc["resolution_type"] == "open" else
                         "resolution")},
            "distractor_session_idx": geom.get("distractor_session_idx"),
            "target_pending_window_user_turns":
                geom["target_pending_window_user_turns"],
            "realized_pending_window_user_turns":
                geom["realized_pending_window_user_turns"],
        },
        "attempt": attempt,
    }
    if arc["resolution_type"] == "abandoned":
        payload["abandoned_mode"] = abandoned_mode
        if abandoned_mode == "alternative":
            payload["alternative_value"] = alternative_value
    if feedback:
        payload["lint_feedback"] = feedback
    return payload


def render_payload(arc: dict, geom: dict, session_idx: int, plan: dict,
                   rendered: list[dict], slots: list[dict],
                   exemplars: list[dict], hedge_sentence: str | None,
                   prior_value: str, abandoned_mode: str | None,
                   alternative_value: str | None, attempt: int,
                   feedback: list[str] | None,
                   previous_attempt: dict | None = None) -> dict:
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
                "is_hedge_turn": bool(s["is_hedge"]),
                "is_resolution_turn": bool(s["is_resolution"]),
                "beat": plan["sessions"][session_idx]["user_turn_beats"][k // 2],
            })
        else:
            directives.append({"turn_idx": k, "role": "assistant",
                               "max_words": r3.ASSISTANT_MAX_WORDS})
    payload = {
        "task": "m4_r4_render_session",
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
        "hedge_strength": arc["hedge_strength"] if has_hedge else None,
        "resolution_outcome": arc["resolution_type"] if has_res else None,
        "abandoned_mode": (abandoned_mode
                           if arc["resolution_type"] == "abandoned" else None),
        "alternative_value": (alternative_value
                              if abandoned_mode == "alternative" else None),
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
                            abandoned_mode: str | None,
                            alternative_value: str | None) -> tuple[dict, int]:
    feedback = None
    for attempt in range(MAX_ATTEMPTS):
        out = await llm.call_llm(
            plan_sys,
            json.dumps(plan_payload(arc, geom, abandoned_mode,
                                    alternative_value, attempt, feedback),
                       ensure_ascii=False),
            trace_tag=f"m4r4-plan-{arc['arc_id']}-a{attempt}")
        fails = lint_plan(arc, geom, out, alternative_value)
        if not fails:
            return out, attempt + 1
        feedback = fails
    raise RuntimeError(f"{arc['arc_id']} plan: lint still failing after "
                       f"{MAX_ATTEMPTS} attempts: {feedback}")


async def render_session(arc: dict, geom: dict, session_idx: int, plan: dict,
                         rendered: list[dict], prior_turns: list[dict],
                         slots: list[dict], pool: dict, banned: list[dict],
                         render_sys: str, hedge_sentence: str | None,
                         abandoned_mode: str | None,
                         alternative_value: str | None) -> tuple[dict, dict]:
    has_hedge = geom["hedge"]["session_idx"] == session_idx
    has_res = geom["resolution"]["session_idx"] == session_idx
    exemplars = r3.pick_exemplars(pool, arc, session_idx, has_hedge, has_res)
    prior_value = plan["prior_value"]
    feedback = None
    last_verdict: dict = {}
    last_out: dict | None = None
    for attempt in range(MAX_ATTEMPTS):
        payload = render_payload(arc, geom, session_idx, plan, rendered, slots,
                                 exemplars, hedge_sentence, prior_value,
                                 abandoned_mode, alternative_value, attempt,
                                 feedback, previous_attempt=last_out)
        out = await llm.call_llm(
            render_sys, json.dumps(payload, ensure_ascii=False),
            trace_tag=f"m4r4-render-{arc['arc_id']}-s{session_idx}-a{attempt}")
        last_out = out
        fails = lint_session(arc, geom, session_idx, out, prior_turns, slots,
                             banned, exemplars, prior_value, alternative_value,
                             abandoned_mode)
        selfcheck: list[str] = []
        if not fails and session_idx == 0:
            selfcheck, last_verdict = await prior_selfcheck(
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

CARRIED_FIELDS = r3.CARRIED_FIELDS


def assemble_arc(src: dict, geom: dict, plan: dict, sessions_out: list[dict],
                 slots: list[dict], metas: list[dict], plan_attempts: int,
                 plan_sha: str, render_sha: str, abandoned_mode: str | None,
                 alternative_value: str | None) -> dict:
    arc = {k: src[k] for k in CARRIED_FIELDS}
    pe, h, r = geom["prior_establish"], geom["hedge"], geom["resolution"]
    prior_ti, hedge_ti, res_ti = (2 * pe["user_turn"], 2 * h["user_turn"],
                                  2 * r["user_turn"])
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
    res_sentence = sessions_out[r["session_idx"]]["resolution_sentence"]
    prior_value = plan["prior_value"]
    selfcheck = metas[0].get("prior_selfcheck", {"result": "passed"})
    src_res = src["resolution"]
    is_ab = src["resolution_type"] == "abandoned"
    arc.update({
        "prior_value": prior_value,
        "prior_establish": {"session_idx": pe["session_idx"],
                            "turn_idx": prior_ti, "position": "in_turn",
                            "date": geom["sessions"][0]["turn_dates"][prior_ti]},
        "prior_text": prior_sentence,
        "abandoned_mode": (abandoned_mode if is_ab else None),
        "alternative_value": (alternative_value
                              if (is_ab and abandoned_mode == "alternative")
                              else None),
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
        "render_round": 4,
        "plan_prompt_sha": plan_sha,
        "render_prompt_sha": render_sha,
        "lints": {
            "prior": {"n_words": len(prior_sentence.split()),
                      "realized_cue_family":
                          ha.detect_cue_family(prior_sentence)[0],
                      "current_frame_detected":
                          asserts_prior_current(prior_sentence, prior_value)},
            "hedge": ha._lint(hedge_sentence),
            "resolution": ha._lint(res_sentence),
            "fm4_probe_value_free": True,
            "prior_selfcheck": selfcheck,
            "r4_lint": "passed (battery in hedge_arcs_r4)",
        },
        "render": {
            "pipeline": ("m4 r4 themed synthetic conversation "
                         "(prompts/m4_conv_plan_r4.txt + "
                         "m4_conv_render_r4.txt)"),
            "plan_prompt_sha": plan_sha,
            "render_prompt_sha": render_sha,
            "style": "human user register + bounded assistant register "
                     "(M4 r4 contract)",
        },
        "r4": {
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
    arc["manifest_rows"] = ha.manifest_rows_for_arc(arc)
    return arc


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------

async def run_generate(limit: int | None, fresh_cap: int | None) -> dict:
    arcs_r3 = ha._read_jsonl(ARCS_R3_PATH)
    if len(arcs_r3) != 60:
        raise SystemExit(f"expected 60 retired r3 arcs at {ARCS_R3_PATH}")
    r3.assert_plan_carryover(arcs_r3)                # ids/keys/values/labels
    plan_sys = PLAN_PROMPT_PATH.read_text()
    render_sys = RENDER_PROMPT_PATH.read_text()
    plan_sha = hashlib.sha256(plan_sys.encode()).hexdigest()
    render_sha = hashlib.sha256(render_sys.encode()).hexdigest()
    banned = r2.load_banned_patterns()
    pool = r3.load_exemplar_pool()

    geoms = {a["arc_id"]: conversation_geometry(a) for a in arcs_r3}
    modes = assign_abandoned_modes(arcs_r3)
    slots = build_slots(arcs_r3, geoms)
    assign_noise(slots)
    assign_word_bands(slots)
    slots_by_arc: dict[str, dict[int, list[dict]]] = {}
    for s in slots:
        slots_by_arc.setdefault(s["arc_id"], {}).setdefault(
            s["session_idx"], []).append(s)

    existing = {a["arc_id"]: a for a in ha._read_jsonl(ARCS_PATH)
                if a.get("render_round") == 4 and a.get("conversation")}
    todo = [a for a in arcs_r3 if a["arc_id"] not in existing]
    if limit is not None:
        todo = todo[:limit]
    done: dict[str, dict] = dict(existing)
    failures: list[str] = []

    async def one(src: dict) -> None:
        try:
            geom = geoms[src["arc_id"]]
            ab_mode, alt_val = modes[src["arc_id"]]
            plan, plan_attempts = await plan_conversation(
                src, geom, plan_sys, ab_mode, alt_val)
            sessions_out: list[dict] = []
            metas: list[dict] = []
            rendered: list[dict] = []
            prior_turns: list[dict] = []
            hedge_sentence: str | None = None
            for si in range(geom["n_sessions"]):
                out, meta = await render_session(
                    src, geom, si, plan, rendered, prior_turns,
                    slots_by_arc[src["arc_id"]][si], pool, banned,
                    render_sys, hedge_sentence, ab_mode, alt_val)
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
                metas, plan_attempts, plan_sha, render_sha, ab_mode, alt_val)
        except Exception as exc:
            failures.append(f"{src['arc_id']}: {exc}")

    guard = ha.m4_guard(fresh_cap)
    guard.install()
    try:
        await asyncio.gather(*(one(a) for a in todo))
    finally:
        guard.uninstall()

    rows = [done[a["arc_id"]] for a in arcs_r3 if a["arc_id"] in done]
    ha._write_jsonl(ARCS_PATH, rows)
    attempts_gt1 = {}
    for a in rows:
        att = a["r4"]["attempts"]
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
    ab_dist: dict[str, int] = {}
    for a in rows:
        if a["resolution_type"] == "abandoned":
            ab_dist[a["abandoned_mode"]] = ab_dist.get(a["abandoned_mode"], 0) + 1
    summary = {
        "id": "V2.4-M4-R4-GENERATE",
        "prereg": ("protocol/CHANGELOG.md 2026-07-09 'M4 r4 generation "
                   "contract FROZEN' (plan sha " + plan_sha + ", render sha "
                   + render_sha + ")"),
        "n_planned": len(arcs_r3),
        "n_generated": len(rows),
        "plan_carryover_ok": True,
        "plan_prompt_sha": plan_sha,
        "render_prompt_sha": render_sha,
        "banned_patterns_enforced": len(banned),
        "n_user_turn_slots": len(slots),
        "noise_directive_counts": noise_counts,
        "session_counts": {str(k): sum(1 for a in arcs_r3
                                       if geoms[a["arc_id"]]["n_sessions"] == k)
                           for k in (2, 3)},
        "abandoned_mode_distribution": ab_dist,
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
    ha._write_json(R4_SUMMARY_PATH, summary)
    if failures:
        for f in failures:
            print(f"R4 GENERATE FAILURE: {f}", file=sys.stderr)
    return summary


# ---------------------------------------------------------------------------
# Integrity relint (over the stored conversations, no LLM)
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
                  for s in arc["r4"]["noise_ledger"]}
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
        alt = arc.get("alternative_value")
        ab_mode = arc.get("abandoned_mode")
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
            exemplars = [by_id[i] for i in arc["r4"]["exemplar_ids"][str(si)]]
            fails += lint_session(arc, geom, si, out, prior_turns,
                                  by_sess[si], banned, exemplars, prior_value,
                                  alt, ab_mode)
            prior_turns.extend(out["turns"])
        if not ha._fm4_probe_ok(arc["key"]["probe_question"], arc["value"]):
            fails.append("FM-4: value in probe question")
        if arc["realized_pending_window_user_turns"] != geom[
                "realized_pending_window_user_turns"]:
            fails.append("realized window drifted from deterministic geometry")
        sc = (arc.get("r4", {}).get("prior_selfcheck") or {})
        if sc.get("result") != "passed":
            fails.append("prior self-check not recorded as passed")
        if fails:
            failures[arc["arc_id"]] = fails
    return {"n_arcs": len(arcs), "all_green": not failures,
            "failures": failures}


# ---------------------------------------------------------------------------
# Checks — register + coherence + prior-classification -> r4_checks.json
# ---------------------------------------------------------------------------

async def prior_classification_report(arcs: list[dict]) -> dict:
    """Re-run the frozen extractor + key_mapper + classifier over the 60
    prior-establish turns (cached from generation) and tally map-to-key +
    class. A released, LLM-backed metric for 'prior maps to key 60/60'."""
    extractor_sys = (PROMPTS_DIR / syn_res.LME_PROMPTS["em"]).read_text()
    rows = []

    async def one(arc):
        geom = conversation_geometry(arc)
        turn0 = arc["conversation"][0]["turns"][0]["text"]
        key = arc["key"]
        cand = store.CanonicalKey(entity=key["entity"],
                                  attribute=key["attribute"],
                                  description=key["description"],
                                  probe_question=key["probe_question"])
        date = geom["sessions"][0]["turn_dates"][0]
        tag = f"m4r4-priorxk-{arc['arc_id']}"
        tuples = await gated_store.extract_turn_family(
            [], turn0, tag, system_prompt=extractor_sys, current_date=date)
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
            item = {"item_id": f"m4r4-priorcls-{arc['arc_id']}",
                    "entity": tup.entity, "attribute": tup.attribute,
                    "asserted_value": tup.value, "stored_value": None,
                    "turn_text": turn0, "preceding_turns": []}
            cls = (await classify.classify_item(item))["argmax"]
        rows.append({"arc_id": arc["arc_id"], "n_tuples": len(tuples),
                     "mapped_to_key": bool(key_tuples), "class": cls,
                     "value_match": vmatch,
                     "class_ok": cls in PRIOR_OK_CLASSES})

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


async def run_checks() -> dict:
    banned = r2.load_banned_patterns()
    arcs = ha._read_jsonl(ARCS_PATH)
    if not arcs:
        raise SystemExit(f"no r4 arcs at {ARCS_PATH}")
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
    coherence_residual = {aid: [f for f in fs if f.startswith("COHERENCE")]
                          for aid, fs in integrity["failures"].items()}
    coherence_residual = {k: v for k, v in coherence_residual.items() if v}
    balance = {t: sum(1 for a in arcs if a["resolution_type"] == t)
               for t in ha.RESOLUTION_TYPES}
    ab_dist: dict[str, int] = {}
    for a in arcs:
        if a["resolution_type"] == "abandoned":
            ab_dist[a["abandoned_mode"]] = ab_dist.get(a["abandoned_mode"], 0) + 1
    windows = [(a["pending_window_user_turns"],
                a["realized_pending_window_user_turns"]) for a in arcs]
    prior_report = await prior_classification_report(arcs)
    doc = {
        "id": "V2.4-M4-R4-CHECKS",
        "method": ("register metrics (hedge_arcs_r2.register_metrics) over ALL "
                   "r4 user turns vs the frozen human-corpus + host-LME-user "
                   "stats; deterministic integrity relint (r3 lints + r4 "
                   "coherence/prior lints); the frozen extractor + key_mapper "
                   "+ classifier re-run over the 60 prior-establish turns"),
        "contract_shas": {"plan": hashlib.sha256(
            PLAN_PROMPT_PATH.read_bytes()).hexdigest(),
            "render": hashlib.sha256(
                RENDER_PROMPT_PATH.read_bytes()).hexdigest()},
        "banned_regex_union": {
            "n_patterns": len(banned),
            "r4_user_turns": r2.banned_union_hits(user_turns, banned),
        },
        "sycophancy_union": {"n_patterns": len(r3.SYCOPHANCY_COMPILED),
                             "n_flagged": len(syco_hits),
                             "flagged_texts": syco_hits},
        "external_anti_pattern_union": {"n_flagged": len(anti_hits),
                                        "flagged_texts": anti_hits},
        "register": {
            "r4_user_turns": r2.register_metrics(user_turns),
            "r4_assistant_turns": r2.register_metrics(assistant_turns),
            "host_lme_user": hosts["lme_user"],
            "human_corpora": {k: ext[k] for k in sorted(ext) if k != "cc_llm"},
            "human_pooled": r3._pooled_human_register(ext),
        },
        "balance": balance,
        "abandoned_mode_distribution": ab_dist,
        "coherence_lint": {
            "residual_arcs": len(coherence_residual),
            "residual": coherence_residual,
        },
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
    ha._write_json(R4_CHECK_PATH, doc)
    return doc


# ---------------------------------------------------------------------------
# Review pack (r4 — three context blocks per arc)
# ---------------------------------------------------------------------------

def render_review_pack(arcs: list[dict]) -> str:
    lines = [
        "# v2.4 M4 — synthetic hedge-arc review pack (OWNER pass required)",
        "",
        "Round 4: each arc is a self-contained fully synthetic multi-session "
        "conversation themed on the key's domain (r4 contract, "
        "protocol/CHANGELOG.md 2026-07-09). Two changes over r3: the "
        "conversation first ESTABLISHES the user's current value as a plain "
        "fact, and the RESOLUTION is a separate later conversation that OPENS "
        "with the outcome (never after re-deliberating the decision). Per the "
        "M4 preregistration this review is the sole human step: every arc "
        "needs an explicit ACCEPT or REJECT before it can enter the pool. "
        "Judge REALISM (would this user say this, here?) and VALIDITY (prior "
        "is a settled current fact; hedge is a real unrealized switch; "
        "resolution matches its label and reads as a coherent later "
        "conversation). Tick exactly one box per arc, then record the "
        "verdicts in "
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
        pe, h, r = a["prior_establish"], a["hedge"], a["resolution"]
        flags = []
        if a["lints"]["hedge"]["realized_cue_family"] != a["hedge_strength"]:
            flags.append("hedge cue family mismatch")
        res_label = f"type **{a['resolution_type']}**"
        if a["resolution_type"] == "abandoned":
            res_label += f", mode `{a['abandoned_mode']}`"
            if a.get("alternative_value"):
                res_label += f" (-> **{a['alternative_value']}**)"
        lines += [
            f"## {a['arc_id']}",
            "",
            f"- key `user/{a['key']['attribute']}`; current value "
            f"**{a['prior_value']}** -> hedge target **{a['value']}**",
            f"- theme: {a['r4']['plan_outline']['premise']}",
            f"- labels: hedge `speculative` (strength {a['hedge_strength']}); "
            f"resolution `{r['construction_label']}` ({res_label}, gold status "
            f"`{a['gold_resolution_status']}`); correct memory: "
            f"**{_correct_memory(a)}**",
            f"- placement: prior s{pe['session_idx']}t{pe['turn_idx']} -> "
            f"hedge s{h['session_idx']}t{h['turn_idx']} ({h['date']}) -> "
            f"resolution s{r['session_idx']}t{r['turn_idx']} ({r['date']}); "
            f"window target {a['pending_window_user_turns']} user turns, "
            f"realized {a['realized_pending_window_user_turns']}; "
            f"{a['r4']['geometry']['n_sessions']} sessions "
            f"({', '.join(a['r4']['session_roles'])})",
        ]
        if flags:
            lines.append(f"- FLAGS: {'; '.join(flags)}")
        lines += ["", "**Current state** (session date "
                  f"{a['conversation'][0]['date']}):", ""]
        lines += r3._context_block_r3(a, pe["session_idx"], pe["turn_idx"],
                                      a["prior_text"])
        lines += ["", "**Hedge** (session date "
                  f"{a['conversation'][h['session_idx']]['date']}):", ""]
        lines += r3._context_block_r3(a, h["session_idx"], h["turn_idx"],
                                      a["hedge_text"])
        lines += ["", "**Resolution -- separate later conversation** (session "
                  f"date {a['conversation'][r['session_idx']]['date']}):", ""]
        lines += r3._context_block_r3(a, r["session_idx"], r["turn_idx"],
                                      a["resolution_text"])
        lines += [
            "",
            f"- [ ] ACCEPT `{a['arc_id']}`",
            f"- [ ] REJECT `{a['arc_id']}` — reason: ____________",
            "",
        ]
    return "\n".join(lines)


def _correct_memory(a: dict) -> str:
    if a["resolution_type"] == "resolve_true":
        return a["value"]
    if a["resolution_type"] == "abandoned" and a["abandoned_mode"] == "alternative":
        return a["alternative_value"]
    return a["prior_value"]


def run_review_pack() -> dict:
    arcs = ha._read_jsonl(ARCS_PATH)
    if not arcs:
        raise SystemExit(f"no arcs at {ARCS_PATH} — run r4-generate first")
    ha.REVIEW_PACK_PATH.write_text(render_review_pack(arcs))
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
        description="v2.4 M4 r4 — coherence + established prior value")
    sub = ap.add_subparsers(dest="cmd", required=True)
    gen = sub.add_parser("r4-generate", help="generate all 60 conversations")
    gen.add_argument("--limit", type=int, default=None)
    gen.add_argument("--fresh-cap", type=int, default=None)
    sub.add_parser("checks", help="register + coherence + prior-classification")
    sub.add_parser("review-pack", help="rebuild review_pack.md (3 blocks)")
    args = ap.parse_args()

    if args.cmd == "r4-generate":
        summary = asyncio.run(run_generate(args.limit, args.fresh_cap))
        print(json.dumps({k: summary[k] for k in (
            "n_planned", "n_generated", "n_user_turn_slots",
            "noise_directive_counts", "session_counts",
            "abandoned_mode_distribution", "realized_window",
            "attempts_gt1", "budget", "failures")}, indent=1))
        print(f"-> {ARCS_PATH}\n-> {R4_SUMMARY_PATH}")
        if summary["failures"]:
            raise SystemExit(f"{len(summary['failures'])} arc(s) failed "
                             "(progress persisted; rerun resumes)")
    elif args.cmd == "checks":
        doc = asyncio.run(run_checks())
        print(json.dumps({
            "banned_union_r4_user_turns": {
                k: doc["banned_regex_union"]["r4_user_turns"][k]
                for k in ("n_texts", "n_flagged", "per_pattern_nonzero")},
            "sycophancy_flagged": doc["sycophancy_union"]["n_flagged"],
            "r4_user_words": doc["register"]["r4_user_turns"]["words"],
            "balance": doc["balance"],
            "abandoned_mode_distribution": doc["abandoned_mode_distribution"],
            "coherence_residual_arcs": doc["coherence_lint"]["residual_arcs"],
            "prior_value": {k: doc["prior_value"][k] for k in (
                "prior_established", "mapped_to_key",
                "class_completed_event_or_other", "value_match",
                "class_distribution")},
            "realized_window": {k: doc["realized_window"][k]
                                for k in ("n_exact", "mean_abs_dev",
                                          "max_abs_dev")},
            "integrity_all_green": doc["integrity_relint"]["all_green"],
        }, indent=1))
        print(f"-> {R4_CHECK_PATH}")
    elif args.cmd == "review-pack":
        print(json.dumps(run_review_pack(), indent=1))


if __name__ == "__main__":
    main()
