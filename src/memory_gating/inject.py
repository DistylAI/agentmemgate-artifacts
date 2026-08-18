"""Injection manifest generator and two-call render pipeline (role I2).

Spec: research/injector_spec.md (binding semantics), research/protocol_spec.md v1.1
(sections 1, 2.1, 3.1 incl. B4), research/key_schema.json. Code contract:
pilot/INTERFACES.md (manifest path/fields, CLI verbs, build_perturbed_substrate).

Documented interpretive choices (flagged in the implementation report):
- Host slots: protected set per injector_spec 2.1 (evidence turns +/- 2). Where a
  session's strict host set is empty (3ba21379 both sessions, cf22b7bf s0,
  db467c8c s1 in core; some alternates) the spec's own arithmetic (2.2 v1.1:
  "3ba21379's 4" = its non-evidence turns; schema note "only s0 t2/t8 and s1
  t0/t10 are non-evidence -- cap injections here at 2 total") requires relaxing
  the adjacency buffer to the session's non-evidence user turns. Such
  placements are tagged "relaxed_adjacency". Evidence turns themselves are
  never hosts. The same relaxation is the last resort when a pinned count
  (60+60, 2 t3 cases/substrate) is otherwise infeasible.
- distractor naming per orchestrator instruction: distractor_a = near-miss
  event (spec 9.5 D2), distractor_b = continued speculation (spec 9.5 D1).
- plan writes the manifest without text AND without sampled values: value
  sampling is pinned to render Call 1 (task brief; injector_spec 3.3 requires
  LLM-proposed candidates). cgold_correct old_value is known at plan time.

All randomness flows through one seeded RNG (seed 20260702), draws consumed in
manifest row order (injector_spec 2.2b / 7.2); every row records
seed_draw_index = the global draw counter at its placement draw.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import random
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

SEED = 20260702

from paths import DATA_DIR, GOLD_DIR, PROMPTS_DIR, RESULTS_DIR

ORACLE_PATH = DATA_DIR / "longmemeval" / "longmemeval_oracle.json"
SCHEMA_PATH = GOLD_DIR / "key_schema.json"
INJ_DIR = RESULTS_DIR / "injections"
MANIFEST_PATH = INJ_DIR / "manifest.jsonl"
KEYS_PATH = INJ_DIR / "injected_keys.jsonl"

OPEN, MID, END = "turn_open_after_first_sentence", "turn_mid", "turn_end"
POSITIONS = (OPEN, MID, END)

# 5.2 gold-key speculation substrates (must equal speculation_compatible cores).
GOLD_SPEC_QIDS = [
    "852ce960", "945e3d21", "89941a93", "ce6d2d27", "41698283",
    "2698e78f", "e493bb7c", "18bc8abd", "3ba21379", "59524333",
]
# 2.2 cap-annotated substrates: max budget injections (spec+correction) each.
BUDGET_CAPS = {"18bc8abd": 2, "3ba21379": 2}
CGOLD_CAP = 10
# R1-B2: at most ONE C-GOLD correction per substrate — 6.3's construction ("the
# establishing assertion is the gold old-value turn itself") supports exactly one
# correction of that record; a second one at the same boundary probe manufactures
# correction residue mechanically (the later v_b overwrites the earlier).
CGOLD_PER_SUBSTRATE_CAP = 1
# predicate_spec 4.1 (R1-B1): after the designed Test 3 variant turn t*, >= 2
# further user turns must exist within the late session so the sensitivity
# window (t*, t*+1, t*+2) is checkable for every variant. Enforced at PLAN time
# so this failure class can never first surface after predicates are frozen.
T3_SENSITIVITY_WINDOW = 2
# predicate_spec 5.4 (R1-B5): soft-abandonment invalidate stratum — exactly 10
# of the 50 invalidate variants, seeded-randomly drawn, tagged
# `invalidate_style:soft` and rendered under the T3_INVALIDATE_SOFT contract.
N_SOFT_INVALIDATE = 10
SOFT_INVALIDATE_TAG = "invalidate_style:soft"
# key_schema prose notes name protected turns beyond the evidence fields
# (3ba21379: "s1 t4 mention is inside a user turn too -- treat t4 as protected").
EXTRA_PROTECTED: dict[str, set[tuple[int, int]]] = {"3ba21379": {(1, 4)}}

CONNECTIVES = ["By the way,", "Oh, and", "Speaking of which,", "Also,", ", by the way. (trailing)"]

# Naturalness-gate round (injector spec 10.5). Round 1 (stock connectives from
# the manifest, no integration guidance) failed the FM-1 gate at AUC 0.930;
# judge tells concentrated on connective formulas (59/28/29 mentions of
# "by the way"/"oh and"/"speaking of which"), topical discontinuity, and
# proper-noun over-specification. Round >= 2 rendering overrides the manifest
# connective and adds bridged-integration guidance (changelog-logged).
GATE_ROUND = 2
ROUND2_TELL_BLACKLIST = ["By the way", "Oh, and", "Speaking of which"]
ROUND2_INTEGRATION_NOTE = (
    "Integration requirements (naturalness round 2): weave the aside into the "
    "host turn so it reads as the same person continuing their own train of "
    "thought. Reuse a word, theme, or motivation already present in the host "
    "turn as a bridge into the aside (e.g. budgets->bank, schedule->class day, "
    "errands->pharmacy). Do NOT open with a stock pivot phrase; the blacklist "
    "includes the forbidden ones. Keep the aside plain and low-key: at most "
    "one proper noun (the required value itself), no extra brand names, no "
    "extra numbers, no self-narration about remembering or notes unless the "
    "contract requires corrective intent. Prefer the shorter end of the "
    "length target.")

# 4. render blacklist: template markers + every [syn] example sentence from
# protocol_spec section 1 (surface-form leakage guard, P9).
BLACKLIST = [
    "Correction:", "I misspoke", "Scratch that", "Forget what I said",
    "Actually, I misspoke earlier", "To correct myself", "just a heads up",
    "I'm moving next month — finally found a place in Seattle",
    "I start the new job Monday",
    "My wife thinks we should move closer to her parents",
    "I'll probably switch Max to a grain-free food once this bag runs out",
    "I live in Seattle now",
    "I decided not to take the job after all",
    "I don't live in Portland anymore",
    "My wife told me she wants to adopt a dog",
    "Correction — it was the 2019 Corolla I traded in, not the 2018",
    "Actually, I misspoke earlier — my personal best was 25:50, not 27:12",
    "Correction: it's a 2019 Corolla, not a 2018",
    "No, I said the appointment is Tuesday, not Thursday",
    "You've got my dog's name wrong — it's Max, not Rex",
    "Forget what I said about moving to Portland — I never ended up going, so I still live in Denver",
    "Scratch that — the race is a 10K, not a 5K",
    "I moved to Chicago",
    "No wait, let's do Italian instead of Korean for the party",
    "I was wrong to buy those shoes",
    "That's not what I meant — I wanted tips for *trail* running, not road running",
    "It's 25:50 now, not 27:12 like I said before",
]
MARKER_WORDS = ["actually", "no,", "i meant", "wait", "mixed up", "wrong"]
MARKER_FRACTION_CAP = 0.4  # of the 60 correction injections

RELATIVE_TIME_PATTERNS = [
    r"\byesterday\b", r"\btomorrow\b", r"\blast week\b", r"\blast month\b",
    r"\blast year\b", r"\bnext week\b", r"\bnext month\b", r"\ba month ago\b",
    r"\ba week ago\b", r"\bmonday\b", r"\btuesday\b", r"\bwednesday\b",
    r"\bthursday\b", r"\bfriday\b", r"\bsaturday\b", r"\bsunday\b",
]

# 5.3 aux attribute pool (volitional, persona-neutral); keyword exclusions
# approximate the no-collision / persona-fit scan purely (render Call 1 also
# re-checks against the live transcript).
AUX_POOL: dict[str, tuple[str, str, list[str]]] = {
    # attribute: (description, probe_question, collision keywords)
    "dentist_name": ("dental practice the user currently goes to",
                     "Which dentist practice does the user currently go to?", ["dentist", "dental"]),
    "hair_salon": ("hair salon the user currently uses",
                   "Which hair salon does the user currently use?", ["salon", "haircut"]),
    "phone_model": ("phone model the user currently carries",
                    "What phone model does the user currently carry?", ["phone"]),
    "coffee_shop_regular": ("coffee shop the user regularly goes to",
                            "Which coffee shop does the user regularly go to?", ["coffee"]),
    "commute_mode": ("user's usual commute mode",
                     "How does the user usually commute?", ["commute"]),
    "streaming_service": ("streaming service the user subscribes to",
                          "Which streaming service does the user subscribe to?", ["streaming", "netflix", "hulu"]),
    "bank": ("bank the user keeps their main account with",
             "Which bank does the user keep their main account with?", ["bank"]),
    "running_shoe_model": ("running shoe model the user currently runs in",
                           "What running shoe model does the user currently run in?", ["running", "sneaker"]),
    "toothpaste_brand": ("toothpaste brand the user currently uses",
                         "What toothpaste brand does the user currently use?", ["toothpaste"]),
    "car_insurance_provider": ("user's current car insurance provider",
                               "Which company currently provides the user's car insurance?",
                               ["insurance", "tokyo", "harajuku"]),
    "gym_membership_tier": ("membership tier the user holds at their gym",
                            "What membership tier does the user hold at their gym?", ["gym"]),
    "standing_desk_setup": ("desk setup the user currently works at",
                            "What desk setup does the user currently work at?", ["desk"]),
    "weekly_farmers_market": ("farmers market the user shops at each week",
                              "Which farmers market does the user shop at each week?", ["farmers market", "market"]),
    "preferred_airline": ("airline the user prefers to fly",
                          "Which airline does the user prefer to fly?", ["airline", "flight"]),
    "houseplant_species": ("houseplant species the user keeps at home",
                           "Which houseplant species does the user keep at home?", ["plant"]),
    "grocery_store_regular": ("grocery store the user usually shops at",
                              "Which grocery store does the user usually shop at?", ["grocery"]),
    "cell_carrier": ("mobile carrier the user is on",
                     "Which mobile carrier is the user on?", ["carrier", "verizon"]),
    "pharmacy_location": ("pharmacy the user fills prescriptions at",
                          "Which pharmacy does the user fill prescriptions at?", ["pharmacy"]),
    "dry_cleaner": ("dry cleaner the user takes clothes to",
                    "Which dry cleaner does the user take clothes to?", ["dry clean"]),
    "meal_kit_service": ("meal kit service the user gets deliveries from",
                         "Which meal kit service does the user get deliveries from?", ["meal kit"]),
}

HEDGE_STRENGTHS = ["deliberating", "intending", "scheduled_but_future"]

BUDGET_TYPES = {"spec_gold", "spec_aux", "caux_correct", "cgold_correct"}
CORRECTION_TYPES = {"caux_correct", "cgold_correct"}


# --------------------------------------------------------------------------- #
# seeded draws
# --------------------------------------------------------------------------- #

class Draws:
    """Single seeded RNG; every draw increments a global counter (seed_draw_index)."""

    def __init__(self, seed: int = SEED):
        self.rng = random.Random(seed)
        self.n = 0

    def choice(self, seq):
        self.n += 1
        return self.rng.choice(list(seq))

    def sample(self, seq, k):
        self.n += 1
        return self.rng.sample(list(seq), k)


# --------------------------------------------------------------------------- #
# substrate loading / host-slot geometry
# --------------------------------------------------------------------------- #

def _load_oracle() -> dict[str, dict]:
    questions = json.loads(ORACLE_PATH.read_text())
    return {q["question_id"]: q for q in questions}


def _load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def parse_session_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y/%m/%d (%a) %H:%M")


def chrono_session_order(dates: list[str]) -> list[int]:
    """Original session indices sorted chronologically (stable on same-day ties, P12)."""
    return sorted(range(len(dates)), key=lambda i: (parse_session_date(dates[i]), i))


@dataclass
class SubInfo:
    qid: str
    tier: str
    entry: dict                       # key_schema entry
    sessions: list[list[dict]]        # ORIGINAL haystack order
    dates: list[str]
    chrono: list[int]                 # original idx, chronological
    evidence: dict[int, set[int]]     # session -> evidence turn idxs
    strict_hosts: dict[int, list[int]]
    relaxed_hosts: dict[int, list[int]]  # non-evidence user turns (adjacency dropped)

    @property
    def early(self) -> int:
        return self.chrono[0]

    @property
    def late(self) -> int:
        return self.chrono[1]

    def chrono_key(self, session_idx: int, turn_idx: int) -> tuple[int, int]:
        return (self.chrono.index(session_idx), turn_idx)

    def hosts(self, session_idx: int) -> list[int]:
        """Strict hosts; falls back to non-evidence turns when strict set is empty."""
        strict = self.strict_hosts[session_idx]
        return strict if strict else self.relaxed_hosts[session_idx]

    def is_relaxed(self, session_idx: int, turn_idx: int) -> bool:
        return turn_idx not in self.strict_hosts[session_idx]

    def transcript_text(self) -> str:
        return "\n".join(t["content"] for s in self.sessions for t in s).lower()


def build_subinfo(entry: dict, question: dict) -> SubInfo:
    sessions = question["haystack_sessions"]
    dates = question["haystack_dates"]
    evidence: dict[int, set[int]] = {i: set() for i in range(len(sessions))}
    for k in ("old_evidence", "new_evidence"):
        evidence[entry[k]["session_idx"]].add(entry[k]["turn_idx"])
    for e in entry.get("extra_evidence", []):
        evidence[e["session_idx"]].add(e["turn_idx"])
    strict_hosts, relaxed_hosts = {}, {}
    extra = EXTRA_PROTECTED.get(entry["question_id"], set())
    for si, sess in enumerate(sessions):
        user = [t for t in range(len(sess)) if sess[t]["role"] == "user"
                and (si, t) not in extra]
        protected = set()
        for t in evidence[si]:
            protected |= {t - 2, t, t + 2}
        strict_hosts[si] = [t for t in user if t not in protected]
        relaxed_hosts[si] = [t for t in user if t not in evidence[si]]
    return SubInfo(
        qid=entry["question_id"], tier=entry["tier"], entry=entry,
        sessions=sessions, dates=dates, chrono=chrono_session_order(dates),
        evidence=evidence, strict_hosts=strict_hosts, relaxed_hosts=relaxed_hosts,
    )


def load_substrates() -> tuple[dict[str, SubInfo], list[str], list[str]]:
    oracle = _load_oracle()
    schema = _load_schema()
    subs: dict[str, SubInfo] = {}
    cores: list[str] = []
    alts: list[tuple[int, str]] = []
    for entry in schema["substrates"]:
        qid = entry["question_id"]
        subs[qid] = build_subinfo(entry, oracle[qid])
        if entry["tier"] == "core":
            cores.append(qid)
        else:
            alts.append((entry["rank"], qid))
    return subs, cores, [q for _, q in sorted(alts)]


# --------------------------------------------------------------------------- #
# placement ledger (per-turn load, 2.3)
# --------------------------------------------------------------------------- #

@dataclass
class Ledger:
    """Tracks asides placed per (session_idx, turn_idx) for one replay context."""
    turns: dict[tuple[int, int], list[dict]] = field(default_factory=dict)

    def entries(self, slot):
        return self.turns.get(slot, [])

    def can_take(self, slot, inj_type: str, key: tuple[str, str]) -> bool:
        cur = self.entries(slot)
        if not cur:
            return True
        if len(cur) >= 2:
            return False
        e = cur[0]
        return (e["position"] in (OPEN, END) and e["inj_type"] != inj_type
                and e["key"] != key)

    def add(self, slot, inj_type, key, position):
        self.turns.setdefault(slot, []).append(
            {"inj_type": inj_type, "key": key, "position": position})

    def position_for(self, slot, draws: Draws, tight: bool) -> str:
        cur = self.entries(slot)
        if cur:  # doubling: forced complementary position (2.3)
            return END if cur[0]["position"] == OPEN else OPEN
        return draws.choice([OPEN, END] if tight else list(POSITIONS))


def _user_turns_after(sub: SubInfo, session_idx: int, turn_idx: int) -> int:
    """User turns strictly after ``turn_idx`` within one session (R1-B1 geometry)."""
    sess = sub.sessions[session_idx]
    return sum(1 for t in range(turn_idx + 1, len(sess)) if sess[t]["role"] == "user")


def _eligible_slots(sub: SubInfo, ledger: Ledger, inj_type: str, key, sessions,
                    predicate=None) -> list[tuple[int, int]]:
    """Slots satisfying constraints; load-0 slots preferred over doubling (2.3)."""
    out = []
    for si in sessions:
        for t in sub.hosts(si):
            slot = (si, t)
            if predicate and not predicate(si, t):
                continue
            if ledger.can_take(slot, inj_type, key):
                out.append(slot)
    empty = [s for s in out if not ledger.entries(s)]
    return sorted(empty if empty else out)


# --------------------------------------------------------------------------- #
# manifest rows
# --------------------------------------------------------------------------- #

def make_row(*, injection_id, question_id, inj_type, entity, attribute,
             session_idx, turn_idx, position, probe_point, construction_label,
             edge_case_tags, seed_draw_index, t3_case_id=None,
             t3_variant_kind=None, paired_with=None, old_value=None,
             connective=None, extra=None) -> dict:
    row = {
        "injection_id": injection_id,
        "question_id": question_id,
        "inj_type": inj_type,
        "entity": entity,
        "attribute": attribute,
        "value": None,
        "old_value": old_value,
        "placement": {"session_idx": session_idx, "turn_idx": turn_idx, "position": position},
        "probe_point": probe_point,
        "construction_label": construction_label,
        "edge_case_tags": list(edge_case_tags),
        "text": None,
        "t3_case_id": t3_case_id,
        "t3_variant_kind": t3_variant_kind,
        "paired_with": paired_with,
        "connective": connective,
        "seed_draw_index": seed_draw_index,
    }
    if extra:
        row.update(extra)
    return row


# --------------------------------------------------------------------------- #
# the pure plan
# --------------------------------------------------------------------------- #

class Planner:
    def __init__(self):
        self.subs, self.cores, self.alts = load_substrates()
        self.draws = Draws()
        self.rows: list[dict] = []
        self.keys_registry: list[dict] = []
        self.ledgers: dict[str, Ledger] = {q: Ledger() for q in self.subs}
        self.aux_used: dict[str, set[str]] = {q: set() for q in self.subs}
        self.id_seq: dict[tuple[str, str], int] = {}
        self.cgold_count = 0
        self.cgold_subs: set[str] = set()  # R1-B2: 1 C-GOLD per substrate
        self.last_connective: dict[str, str] = {}

    # -- id / key helpers ---------------------------------------------------
    def next_id(self, inj_type: str, qid: str) -> str:
        k = (inj_type, qid)
        self.id_seq[k] = self.id_seq.get(k, 0) + 1
        return f"{inj_type}-{qid}-{self.id_seq[k]:02d}"

    def pick_connective(self, qid: str) -> str:
        pool = [c for c in CONNECTIVES if c != self.last_connective.get(qid)]
        c = self.draws.choice(pool)
        self.last_connective[qid] = c
        return c

    def new_aux_key(self, qid: str, created_by: str) -> tuple[str, str]:
        sub = self.subs[qid]
        transcript = sub.transcript_text()
        eligible = [a for a, (_, _, kw) in AUX_POOL.items()
                    if a not in self.aux_used[qid]
                    and not any(k in transcript for k in kw)]
        if not eligible:
            raise RuntimeError(f"aux pool exhausted for {qid}")
        attr = self.draws.choice(sorted(eligible))
        self.aux_used[qid].add(attr)
        desc, probe_q, _ = AUX_POOL[attr]
        # FM-4: key + probe question registered at plan time, before any value exists.
        self.keys_registry.append({
            "question_id": qid, "entity": "user", "attribute": attr,
            "description": desc, "probe_question": probe_q,
            "created_by": created_by,
        })
        return ("user", attr)

    # -- generic single placement --------------------------------------------
    def place(self, qid, inj_type, key, sessions, predicate=None, tight=False):
        """Returns (session_idx, turn_idx, position, draw_index, relaxed) or None."""
        sub, ledger = self.subs[qid], self.ledgers[qid]
        slots = _eligible_slots(sub, ledger, inj_type, key, sessions, predicate)
        if not slots:
            return None
        slot = self.draws.choice(slots)          # 2.2b seeded tie-break
        draw_idx = self.draws.n
        position = ledger.position_for(slot, self.draws, tight)
        ledger.add(slot, inj_type, key, position)
        return (*slot, position, draw_idx, sub.is_relaxed(*slot))

    # -- speculation placements ----------------------------------------------
    def gold_spec_slot(self, qid):
        """2.5 gold-key speculation slot; returns (slots, flagged, forced_latest)."""
        sub = self.subs[qid]
        late = sub.late
        ev_late = sub.evidence.get(late, set())
        last_ev = max(ev_late)
        after = [t for t in sub.hosts(late) if t > last_ev]
        if after:
            return after, False, False
        new_ev = sub.entry["new_evidence"]["turn_idx"]
        fallback = [t for t in sub.hosts(late) if t >= new_ev + 4]
        if fallback:
            return [max(fallback)], True, True   # "the latest host slot"
        return [], False, False

    def place_spec_gold(self, qid) -> bool:
        sub = self.subs[qid]
        slots, flagged, forced = self.gold_spec_slot(qid)
        key = (sub.entry["entity"], sub.entry["attribute"])
        pred = (lambda si, t: si == sub.late and t in slots)
        placed = self.place(qid, "spec_gold", key, [sub.late], pred,
                            tight=self.tight[qid])
        if not placed:
            return False
        si, t, pos, di, relaxed = placed
        hedge = self.draws.choice(HEDGE_STRENGTHS)
        tags = ["a", f"hedge:{hedge}"]
        if flagged:
            tags.append("gold_key_spec_not_last")
        if relaxed:
            tags.append("relaxed_adjacency")
        self.rows.append(make_row(
            injection_id=self.next_id("spec_gold", qid), question_id=qid,
            inj_type="spec_gold", entity=key[0], attribute=key[1],
            session_idx=si, turn_idx=t, position=pos,
            probe_point="provenance_timeline" if flagged else "full_ingestion",
            construction_label="speculative", edge_case_tags=tags,
            seed_draw_index=di, connective=self.pick_connective(qid),
        ))
        return True

    def place_pair(self, qid, est_type, dep_type, key, tiers, min_gap=1):
        """Place establish + dependent; tiers = list of (est_sessions, dep_sessions,
        same_session_gap_required). Returns (est_placed, dep_placed) or None."""
        sub, ledger = self.subs[qid], self.ledgers[qid]
        tight = self.tight[qid]
        for est_sessions, dep_sessions, same_gap in tiers:
            def dep_ok_after(est_slot):
                def pred(si, t):
                    if sub.chrono_key(si, t) <= sub.chrono_key(*est_slot):
                        return False
                    if si == est_slot[0]:
                        return t - est_slot[1] >= same_gap
                    return True
                return pred
            est_cands = []
            for si in est_sessions:
                for t in sub.hosts(si):
                    slot = (si, t)
                    if not ledger.can_take(slot, est_type, key):
                        continue
                    if _eligible_slots(sub, ledger, dep_type, key, dep_sessions,
                                       dep_ok_after(slot)):
                        est_cands.append(slot)
            if not est_cands:
                continue
            empty = [s for s in est_cands if not ledger.entries(s)]
            est_slot = self.draws.choice(sorted(empty if empty else est_cands))
            est_draw = self.draws.n
            est_pos = ledger.position_for(est_slot, self.draws, tight)
            ledger.add(est_slot, est_type, key, est_pos)
            dep = self.place(qid, dep_type, key, dep_sessions,
                             dep_ok_after(est_slot), tight)
            assert dep is not None  # est_cands guaranteed a dep slot
            return (est_slot, est_pos, est_draw, sub.is_relaxed(*est_slot)), dep
        return None

    def place_spec_aux(self, qid, realloc_from=None) -> bool:
        sub = self.subs[qid]
        eid = self.next_id("aux_establish", qid)
        key = self.new_aux_key(qid, eid)
        both = [sub.early, sub.late]
        pair = self.place_pair(qid, "aux_establish", "spec_aux", key,
                               tiers=[(both, both, 1)])
        if pair is None:
            self.keys_registry.pop()
            self.aux_used[qid].discard(key[1])
            self.id_seq[("aux_establish", qid)] -= 1
            return False
        (est_slot, epos, edraw, erelax), (dsi, dt, dpos, ddraw, drelax) = pair
        self.rows.append(make_row(
            injection_id=eid, question_id=qid, inj_type="aux_establish",
            entity=key[0], attribute=key[1],
            session_idx=est_slot[0], turn_idx=est_slot[1],
            position=epos, probe_point="full_ingestion",
            construction_label="other",
            edge_case_tags=["relaxed_adjacency"] if erelax else [],
            seed_draw_index=edraw, connective=self.pick_connective(qid),
        ))
        hedge = self.draws.choice(HEDGE_STRENGTHS)
        tags = ["a", f"hedge:{hedge}"]
        if drelax:
            tags.append("relaxed_adjacency")
        if realloc_from:
            tags.append(f"reallocated_from:{realloc_from}")
        self.rows.append(make_row(
            injection_id=self.next_id("spec_aux", qid), question_id=qid,
            inj_type="spec_aux", entity=key[0], attribute=key[1],
            session_idx=dsi, turn_idx=dt, position=dpos,
            probe_point="full_ingestion", construction_label="speculative",
            edge_case_tags=tags, seed_draw_index=ddraw, paired_with=eid,
            connective=self.pick_connective(qid),
        ))
        return True

    # -- correction placements -----------------------------------------------
    def place_caux(self, qid, realloc_from=None) -> bool:
        sub = self.subs[qid]
        eid = self.next_id("caux_establish", qid)
        key = self.new_aux_key(qid, eid)
        tiers = [
            ([sub.early], [sub.late], 1),          # 6.2 default: cross-session
            ([sub.early, sub.late], [sub.early, sub.late], 4),  # same session, >=2 user turns
        ]
        pair = self.place_pair(qid, "caux_establish", "caux_correct", key, tiers)
        if pair is None:
            self.keys_registry.pop()
            self.aux_used[qid].discard(key[1])
            self.id_seq[("caux_establish", qid)] -= 1
            return False
        (est_slot, epos, edraw, erelax), (dsi, dt, dpos, ddraw, drelax) = pair
        self.rows.append(make_row(
            injection_id=eid, question_id=qid, inj_type="caux_establish",
            entity=key[0], attribute=key[1],
            session_idx=est_slot[0], turn_idx=est_slot[1], position=epos,
            probe_point="full_ingestion", construction_label="other",
            edge_case_tags=["relaxed_adjacency"] if erelax else [],
            seed_draw_index=edraw, connective=self.pick_connective(qid),
        ))
        tags = ["e"]
        if drelax:
            tags.append("relaxed_adjacency")
        if realloc_from:
            tags.append(f"reallocated_from:{realloc_from}")
        self.rows.append(make_row(
            injection_id=self.next_id("caux_correct", qid), question_id=qid,
            inj_type="caux_correct", entity=key[0], attribute=key[1],
            session_idx=dsi, turn_idx=dt, position=dpos,
            probe_point="full_ingestion", construction_label="explicit_correction",
            edge_case_tags=tags, seed_draw_index=ddraw, paired_with=eid,
            connective=self.pick_connective(qid),
        ))
        return True

    def place_cgold(self, qid, realloc_from=None) -> bool:
        """6.3 C-GOLD fallback: correct v_old -> v_b after ALL old-session evidence."""
        if self.cgold_count >= CGOLD_CAP:
            return False
        if qid in self.cgold_subs:
            # R1-B2: a second C-GOLD on the same gold key would guarantee false
            # correction residue by construction; the displaced draw flows
            # through 7.3 reallocation instead.
            return False
        sub = self.subs[qid]
        old_sess = sub.early
        ev = sub.evidence.get(old_sess, set())
        if not ev:
            return False
        last_old = max(ev)
        key = (sub.entry["entity"], sub.entry["attribute"])
        pred = lambda si, t: t > last_old
        placed = self.place(qid, "cgold_correct", key, [old_sess], pred,
                            tight=self.tight[qid])
        if placed is None:
            return False  # e.g. e61a7584: old evidence is the final turn
        si, t, pos, di, relaxed = placed
        tags = ["e", "cgold"]
        if relaxed:
            tags.append("relaxed_adjacency")
        if realloc_from:
            tags.append(f"reallocated_from:{realloc_from}")
        self.rows.append(make_row(
            injection_id=self.next_id("cgold_correct", qid), question_id=qid,
            inj_type="cgold_correct", entity=key[0], attribute=key[1],
            session_idx=si, turn_idx=t, position=pos,
            probe_point="old_session_boundary",
            construction_label="explicit_correction", edge_case_tags=tags,
            seed_draw_index=di, old_value=sub.entry["old_value"],
            connective=self.pick_connective(qid),
        ))
        self.cgold_count += 1
        self.cgold_subs.add(qid)
        return True

    def place_correction(self, qid, realloc_from=None) -> bool:
        if self.place_caux(qid, realloc_from):
            return True
        return self.place_cgold(qid, realloc_from)

    # -- Test 1 top-ups (8) ----------------------------------------------------
    def place_topup(self, qid, kind, wrongval=False) -> bool:
        """kind in {d, f, c, plain, hedged}. Returns False if substrate can't host."""
        sub = self.subs[qid]
        both = [sub.early, sub.late]
        needs_establish = kind == "d" or kind == "hedged"
        inj_type = "hedged_recall" if kind == "hedged" else "completed_topup"
        if needs_establish:
            eid = self.next_id("aux_establish", qid)
            key = self.new_aux_key(qid, eid)
            gap = 4 if kind == "hedged" else 1   # 8.2: >= 2 user turns earlier
            pair = self.place_pair(qid, "aux_establish", inj_type, key,
                                   tiers=[(both, both, gap)])
            if pair is None:
                self.keys_registry.pop()
                self.aux_used[qid].discard(key[1])
                self.id_seq[("aux_establish", qid)] -= 1
                return False
            (est_slot, epos, edraw, erelax), (dsi, dt, dpos, ddraw, drelax) = pair
            self.rows.append(make_row(
                injection_id=eid, question_id=qid, inj_type="aux_establish",
                entity=key[0], attribute=key[1],
                session_idx=est_slot[0], turn_idx=est_slot[1], position=epos,
                probe_point="full_ingestion", construction_label="other",
                edge_case_tags=["relaxed_adjacency"] if erelax else [],
                seed_draw_index=edraw, connective=self.pick_connective(qid),
            ))
        else:
            eid = None
            tid = self.next_id(inj_type, qid)
            key = self.new_aux_key(qid, tid)
            placed = self.place(qid, inj_type, key, both, tight=False)
            if placed is None:
                self.keys_registry.pop()
                self.aux_used[qid].discard(key[1])
                self.id_seq[(inj_type, qid)] -= 1
                return False
            dsi, dt, dpos, ddraw, drelax = placed
        if kind == "hedged":
            label = "other"
            tags = ["hedged_recall"] + (["hedged_recall_wrongval"] if wrongval else [])
        else:
            label = "completed_event"
            tags = {"d": ["d"], "f": ["f"], "c": ["c"], "plain": []}[kind]
        if drelax:
            tags = tags + ["relaxed_adjacency"]
        iid = self.next_id(inj_type, qid) if needs_establish else tid
        self.rows.append(make_row(
            injection_id=iid, question_id=qid, inj_type=inj_type,
            entity=key[0], attribute=key[1], session_idx=dsi, turn_idx=dt,
            position=dpos, probe_point="full_ingestion",
            construction_label=label, edge_case_tags=tags,
            seed_draw_index=ddraw, paired_with=eid,
            connective=self.pick_connective(qid),
        ))
        return True

    # -- Test 3 cases (9) --------------------------------------------------------
    def place_t3_case(self, qid, case_no, used_staging, used_keys) -> None:
        sub = self.subs[qid]
        case_id = f"t3-{qid}-c{case_no}"  # lives in its own clean copy (9.1)
        sid = self.next_id("t3_staging", qid)
        key = self.new_aux_key(qid, sid)     # aux volitional key (9.2)
        assert key[1] not in used_keys
        used_keys.add(key[1])
        # staging: earlier session, mid-session preferred (9.2)
        hosts = [t for t in sub.hosts(sub.early) if t not in used_staging]
        if not hosts:  # relaxed last resort (P3 pins 2 cases/substrate)
            hosts = [t for t in sub.relaxed_hosts[sub.early] if t not in used_staging]
        assert hosts, f"no staging slot for {case_id}"
        user_turns = [t for t in range(len(sub.sessions[sub.early]))
                      if sub.sessions[sub.early][t]["role"] == "user"]
        mid = user_turns[len(user_turns) // 2]
        best = min(abs(t - mid) for t in hosts)
        stage_t = self.draws.choice(sorted(t for t in hosts if abs(t - mid) == best))
        stage_draw = self.draws.n
        stage_pos = self.draws.choice(list(POSITIONS))
        used_staging.add(stage_t)
        hedge = self.draws.choice(HEDGE_STRENGTHS)
        stags = ["a", f"hedge:{hedge}"]
        if sub.is_relaxed(sub.early, stage_t):
            stags.append("relaxed_adjacency")
        self.rows.append(make_row(
            injection_id=sid, question_id=qid, inj_type="t3_staging",
            entity=key[0], attribute=key[1], session_idx=sub.early,
            turn_idx=stage_t, position=stage_pos, probe_point="full_ingestion",
            construction_label="speculative", edge_case_tags=stags,
            seed_draw_index=stage_draw, t3_case_id=case_id,
            connective=self.pick_connective(qid),
        ))
        # variant slot: later session, fixed at case creation (9.4). R1-B1:
        # restricted to slots with >= 2 user turns after t* within the late
        # session, so the predicate_spec 4.1 sensitivity window exists for every
        # variant. Strict hosts first; tagged relaxed fallback only when no
        # strict host has the window (184da446, e493bb7c, db467c8c, 3ba21379).
        strict_ok = [t for t in sub.strict_hosts[sub.late]
                     if _user_turns_after(sub, sub.late, t) >= T3_SENSITIVITY_WINDOW]
        relaxed_ok = [t for t in sub.relaxed_hosts[sub.late]
                      if _user_turns_after(sub, sub.late, t) >= T3_SENSITIVITY_WINDOW]
        vhosts = strict_ok if strict_ok else relaxed_ok
        assert vhosts, f"no variant slot with a sensitivity window for {case_id}"
        var_t = self.draws.choice(sorted(vhosts))
        var_draw = self.draws.n
        var_pos = self.draws.choice(list(POSITIONS))
        relaxed_v = sub.is_relaxed(sub.late, var_t)
        # R1-B1 plan-time geometry assert: check()'s sensitivity-window assert
        # must never be the first place this class of error surfaces (it would
        # fire only after predicates are generated, frozen, and committed).
        assert _user_turns_after(sub, sub.late, var_t) >= T3_SENSITIVITY_WINDOW, \
            f"variant slot for {case_id} lacks the sensitivity window"
        # distractor_a near-miss subtype (9.5 D2)
        near_miss = self.draws.choice(["third_party_completion", "non_resolving_event"])
        variant_specs = [
            ("promote", "completed_event", ["t3"]),
            ("invalidate", "completed_event", ["t3", "f"]),
            ("distractor_a", "completed_event", ["t3", f"near_miss:{near_miss}"]),
            ("distractor_b", "speculative", ["t3", "continued_speculation"]),
        ]
        for kind, label, tags in variant_specs:
            if relaxed_v:
                tags = tags + ["relaxed_adjacency"]
            self.rows.append(make_row(
                injection_id=f"t3_variant-{qid}-c{case_no}-{kind}",
                question_id=qid, inj_type="t3_variant", entity=key[0],
                attribute=key[1], session_idx=sub.late, turn_idx=var_t,
                position=var_pos, probe_point="full_ingestion",
                construction_label=label, edge_case_tags=tags,
                seed_draw_index=var_draw, t3_case_id=case_id,
                t3_variant_kind=kind, paired_with=sid,
                connective=self.pick_connective(qid),
            ))

    # -- top-level ------------------------------------------------------------
    def run(self) -> tuple[list[dict], list[dict]]:
        # 7.2 allocation draws
        third_spec = self.draws.sample(self.cores, 10)
        third_corr = self.draws.sample(self.cores, 10)
        spec_target = {q: 2 + (q in third_spec) for q in self.cores}
        corr_target = {q: 2 + (q in third_corr) for q in self.cores}
        overflow: list[tuple[str, str]] = []
        for q, cap in BUDGET_CAPS.items():
            # cap-annotated substrates realize 1 spec (gold) + 1 correction
            overflow += [("spec", q)] * (spec_target[q] - 1)
            overflow += [("corr", q)] * (corr_target[q] - 1)
            spec_target[q], corr_target[q] = 1, 1
        self.tight = {}
        for q in self.cores:
            sg = 1 if q in GOLD_SPEC_QIDS else 0
            need = sg + 2 * (spec_target[q] - sg) + 2 * corr_target[q]
            n_hosts = sum(len(self.subs[q].hosts(s)) for s in (self.subs[q].early, self.subs[q].late))
            self.tight[q] = need > n_hosts
        for q in self.alts:
            self.tight[q] = False

        # first pass: per-substrate placement in schema (core-list) order
        for q in self.cores:
            spec_left, corr_left = spec_target[q], corr_target[q]
            if q in GOLD_SPEC_QIDS:
                if self.place_spec_gold(q):
                    spec_left -= 1
                else:
                    overflow.append(("spec", q))
                    spec_left -= 1
            for _ in range(corr_left):
                if not self.place_correction(q):
                    overflow.append(("corr", q))
            for _ in range(spec_left):
                if not self.place_spec_aux(q):
                    overflow.append(("spec", q))

        # 7.3 reallocation of overflow (stays within the core 25)
        for kind, from_q in overflow:
            candidates = [q for q in self.cores if q not in BUDGET_CAPS and q != from_q]
            placed = False
            while candidates and not placed:
                q = self.draws.choice(sorted(candidates))
                ok = (self.place_spec_aux(q, realloc_from=from_q) if kind == "spec"
                      else self.place_correction(q, realloc_from=from_q))
                if ok:
                    placed = True
                else:
                    candidates.remove(q)
            if not placed:
                raise RuntimeError(f"could not reallocate {kind} from {from_q}")

        # Test 1 top-ups on alternates only (8), rank order round-robin
        wrong_idx = set(self.draws.sample(range(15), 5))  # 8.2: ~1/3 wrongval
        items = ([("d", False)] * 8 + [("f", False)] * 6 + [("c", False)] * 6
                 + [("plain", False)] * 18
                 + [("hedged", i in wrong_idx) for i in range(15)])
        ptr = 0
        for kind, wv in items:
            placed = False
            for off in range(len(self.alts)):
                q = self.alts[(ptr + off) % len(self.alts)]
                if self.place_topup(q, kind, wrongval=wv):
                    placed = True
                    ptr = (ptr + off + 1) % len(self.alts)
                    break
            if not placed:
                raise RuntimeError(f"could not place top-up {kind}")

        # Test 3: 2 cases per core substrate (9.1)
        for q in self.cores:
            used_staging: set[int] = set()
            used_keys: set[str] = set()
            for case_no in (1, 2):
                self.place_t3_case(q, case_no, used_staging, used_keys)

        # predicate_spec 5.4 soft-abandonment stratum (R1-B5): tag exactly 10
        # seeded-randomly-drawn invalidate variants `invalidate_style:soft`;
        # render routes them to the T3_INVALIDATE_SOFT content contract.
        inv_ids = sorted(r["injection_id"] for r in self.rows
                         if r["inj_type"] == "t3_variant"
                         and r["t3_variant_kind"] == "invalidate")
        soft_ids = set(self.draws.sample(inv_ids, N_SOFT_INVALIDATE))
        for r in self.rows:
            if r["injection_id"] in soft_ids:
                r["edge_case_tags"].append(SOFT_INVALIDATE_TAG)

        # 4. marker allowance: <=40% of the 60 correction rows may use markers
        corr_ids = sorted(r["injection_id"] for r in self.rows
                          if r["inj_type"] in CORRECTION_TYPES)
        allowed = set(self.draws.sample(corr_ids, int(len(corr_ids) * MARKER_FRACTION_CAP)))
        for r in self.rows:
            if r["injection_id"] in allowed:
                r["edge_case_tags"].append("marker_allowed")
        return self.rows, self.keys_registry


def plan() -> tuple[list[dict], list[dict]]:
    """Pure, deterministic manifest generation (no LLM calls)."""
    return Planner().run()


# --------------------------------------------------------------------------- #
# splice: build_perturbed_substrate (pure)
# --------------------------------------------------------------------------- #

_BOUNDARY_RE = re.compile(r"[.!?][\"'\)\]]*\s+")


def _sentence_boundaries(text: str) -> list[int]:
    return [m.end() for m in _BOUNDARY_RE.finditer(text)]


def _offset_for(text: str, position: str) -> int:
    bounds = _sentence_boundaries(text)
    if position == END or not bounds:
        return len(text)
    if position == OPEN:
        return bounds[0]
    return min(bounds, key=lambda b: abs(b - len(text) / 2))  # MID


def splice_turn(text: str, insertions: list[tuple[str, str]]) -> str:
    """Insert (position, aside) fragments into one turn at sentence boundaries."""
    order = {OPEN: 0, MID: 1, END: 2}
    resolved = []
    for position, aside in insertions:
        if position not in order:
            raise ValueError(f"bad position {position}")
        resolved.append((_offset_for(text, position), order[position], aside.strip()))
    resolved.sort()
    out, prev = [], 0
    for off, _, aside in resolved:
        out.append(text[prev:off])
        chunk = aside if not text[prev:off] or text[:off].endswith((" ", "\n")) else " " + aside
        out.append(chunk if off >= len(text) else chunk + " ")
        prev = off
    out.append(text[prev:])
    return "".join(out)


def build_perturbed_substrate(substrate, manifest_rows):
    """Pure splice of rendered asides into a data.py Substrate.

    Substrate convention (data.py): ``sessions`` keeps the ORIGINAL haystack
    order, so manifest ``placement.session_idx`` indexes it directly;
    ``session_order`` is only the chronological iteration order.
    """
    by_turn: dict[tuple[int, int], list[tuple[str, str]]] = {}
    for row in manifest_rows:
        if row["question_id"] != substrate.question_id:
            continue
        if not row["text"]:
            raise ValueError(f"row {row['injection_id']} has no rendered text")
        p = row["placement"]
        by_turn.setdefault((p["session_idx"], p["turn_idx"]), []).append(
            (p["position"], row["text"]))
    new_sessions = []
    for si, session in enumerate(substrate.sessions):
        new_session = []
        for ti, turn in enumerate(session):
            ins = by_turn.get((si, ti))
            if ins:
                if turn["role"] != "user":
                    raise ValueError(f"injection into non-user turn {si}:{ti}")
                turn = {**turn, "content": splice_turn(turn["content"], ins)}
            else:
                turn = dict(turn)
            new_session.append(turn)
        new_sessions.append(new_session)
    return dataclasses.replace(substrate, sessions=new_sessions)


# --------------------------------------------------------------------------- #
# render: two-call pipeline (4)
# --------------------------------------------------------------------------- #

CONTRACT_KEYS = {
    "spec_gold": "SPECULATION", "spec_aux": "SPECULATION",
    "t3_staging": "SPECULATION",
    "aux_establish": "ESTABLISH", "caux_establish": "ESTABLISH",
    "caux_correct": "CORRECTION", "cgold_correct": "CORRECTION",
}


def contract_key(row: dict) -> str:
    t = row["inj_type"]
    if t in CONTRACT_KEYS:
        return CONTRACT_KEYS[t]
    if t == "completed_topup":
        tags = set(row["edge_case_tags"])
        for k, name in (("d", "COMPLETED_D"), ("f", "COMPLETED_F"), ("c", "COMPLETED_C")):
            if k in tags:
                return name
        return "COMPLETED_PLAIN"
    if t == "hedged_recall":
        return ("HEDGED_RECALL_WRONGVAL"
                if "hedged_recall_wrongval" in row["edge_case_tags"] else "HEDGED_RECALL")
    if t == "t3_variant":
        if (row["t3_variant_kind"] == "invalidate"
                and SOFT_INVALIDATE_TAG in row["edge_case_tags"]):
            return "T3_INVALIDATE_SOFT"  # predicate_spec 5.4 / PC-1 (R1-B5)
        return {"promote": "T3_PROMOTE", "invalidate": "T3_INVALIDATE",
                "distractor_a": "T3_DISTRACTOR_NEAR_MISS",
                "distractor_b": "T3_DISTRACTOR_CONTINUED"}[row["t3_variant_kind"]]
    raise ValueError(f"no contract for {t}")


def _norm(v: str) -> str:
    return re.sub(r"[,$]", "", v.strip().lower().rstrip("."))


def novelty_ok(value: str, sub: SubInfo, exclusions: list[str]) -> bool:
    """3.1 hard novelty constraint (code-checked).

    Values of <= 3 characters (bare counts, short numerics) match on word
    boundaries: substring containment would false-collide inside longer
    tokens ("5" in "2015"), making novelty unsatisfiable for count-valued
    keys. Longer values keep the stricter substring rule.
    """
    v = _norm(value)
    if not v:
        return False
    text = sub.transcript_text().replace(",", "").replace("$", "")
    if len(v) <= 3:
        if re.search(rf"(?<![\w]){re.escape(v)}(?![\w])", text):
            return False
    elif v in text:
        return False
    return all(v != _norm(x) for x in exclusions if x)


def gold_exclusions(sub: SubInfo) -> list[str]:
    return [sub.entry["old_value"], sub.entry["new_value"]]


def _fixed_value(row: dict, by_id: dict[str, dict]) -> str | None:
    """Rows whose value is pinned rather than sampled."""
    t = row["inj_type"]
    if t == "hedged_recall" and "hedged_recall_wrongval" not in row["edge_case_tags"]:
        return by_id[row["paired_with"]]["value"]     # 8.2: equals stored value
    if t == "t3_variant":
        return by_id[row["paired_with"]]["value"]     # variants reuse staged v_spec
    return None


def sibling_value_exclusions(row: dict, group: list[dict]) -> list[str]:
    """R1-B2 hardening: values already sampled by sibling injections on the same
    (question_id, entity, attribute) key are excluded for this row, so two
    injections on one key can never collide on independently sampled values."""
    return [x["value"] for x in group
            if x["injection_id"] != row["injection_id"] and x["value"]]


async def render_row(row: dict, subs: dict[str, SubInfo], by_id: dict[str, dict],
                     plan_sys: str, render_sys: str, keydesc: dict,
                     key_group: list[dict] = ()) -> None:
    from llm import call_llm

    sub = subs[row["question_id"]]
    p = row["placement"]
    host_turn = sub.sessions[p["session_idx"]][p["turn_idx"]]["content"]
    host_date = sub.dates[p["session_idx"]]
    paired = by_id.get(row["paired_with"]) if row["paired_with"] else None
    established = paired["value"] if paired else None
    if paired and paired["value"] is None:
        raise RuntimeError(f"{row['injection_id']}: paired row not rendered yet")
    exclusions = gold_exclusions(sub) if row["inj_type"] in ("spec_gold", "cgold_correct") else []
    if established:
        exclusions = exclusions + [established]
    # R1-B2: siblings on the same (question_id, key) — rendered sequentially in
    # manifest order (see render()) — exclude each other's sampled values.
    for v in sibling_value_exclusions(row, list(key_group)):
        if v not in exclusions:
            exclusions.append(v)
    fixed = _fixed_value(row, by_id)
    marker_allowed = "marker_allowed" in row["edge_case_tags"]

    payload = {
        "contract": contract_key(row),
        "injection_type": row["inj_type"],
        "target_key": {"entity": row["entity"], "attribute": row["attribute"]},
        "established_value": established,
        "gold_old_value": row["old_value"] if row["inj_type"] == "cgold_correct" else None,
        "fixed_value": fixed,
        "value_exclusions": exclusions,
        "host_turn_text": host_turn,
        "insertion_position": p["position"],
        "host_session_date": host_date,
        "other_session_date": sub.dates[sub.late if p["session_idx"] == sub.early else sub.early],
        "hedge_strength": next((t.split(":")[1] for t in row["edge_case_tags"]
                                if t.startswith("hedge:")), None),
        "near_miss_subtype": next((t.split(":")[1] for t in row["edge_case_tags"]
                                   if t.startswith("near_miss:")), None),
        "attribute_description": keydesc.get(
            (row["question_id"], row["attribute"]), row["attribute"].replace("_", " ")),
    }
    out1 = await call_llm(plan_sys, json.dumps(payload, ensure_ascii=False),
                          trace_tag=f"inject-plan-{row['injection_id']}")
    if fixed is not None:
        value = fixed
    else:
        candidates = [str(c) for c in out1.get("value_candidates", [])]
        value = next((c for c in candidates if novelty_ok(c, sub, exclusions)), None)
        for retry_round in range(1, 4):
            if value is not None:
                break
            retry_payload = dict(payload)
            retry_payload["value_exclusions"] = exclusions + candidates
            if row["attribute"].endswith("_count"):
                retry_payload["value_format"] = (
                    "REQUIRED: each candidate must be the count followed by its "
                    "unit noun (e.g. '4 bikes', '11 short stories'); bare "
                    "numerals are invalid and will be rejected.")
            retry_payload["retry_note"] = (
                f"Retry round {retry_round}: every previous candidate was rejected "
                "because it appears verbatim in the source transcript or collides "
                "with an excluded value. Propose distinctly different candidates "
                "(for numeric attributes, prefer values not used anywhere in the "
                "conversation, e.g. less round numbers). For count-valued "
                "attributes, phrase each candidate WITH its unit noun (e.g. "
                "'4 bikes', '11 short stories'), never as a bare numeral.")
            out1 = await call_llm(plan_sys, json.dumps(retry_payload, ensure_ascii=False),
                                  trace_tag=f"inject-plan-retry{retry_round}-{row['injection_id']}")
            retry_candidates = [str(c) for c in out1.get("value_candidates", [])]
            value = next((c for c in retry_candidates if novelty_ok(c, sub, exclusions)), None)
            candidates = candidates + retry_candidates
        if value is None:
            raise RuntimeError(
                f"{row['injection_id']}: no candidate passed novelty after retries: "
                f"{candidates}")
        row["value_candidates"] = candidates
    row["value"] = value
    if row["inj_type"] == "caux_correct":
        row["old_value"] = established

    rng = random.Random(f"{SEED}:{row['injection_id']}")
    exemplars = _style_exemplars(sub, rng)
    for attempt in range(3):
        rpayload = {
            "contract": contract_key(row),
            "content_plan": _align_plan_value(out1.get("content_plan", ""),
                                              row.get("value_candidates") or [], value),
            "value": value,
            "established_value": established,
            "old_value": row["old_value"],
            "connective": None if GATE_ROUND >= 2 else row["connective"],
            "integration_note": ROUND2_INTEGRATION_NOTE if GATE_ROUND >= 2 else None,
            "gate_round": GATE_ROUND,
            "insertion_position": p["position"],
            "host_turn_text": host_turn,
            "style_exemplars": exemplars,
            "blacklist": BLACKLIST + (ROUND2_TELL_BLACKLIST if GATE_ROUND >= 2 else []),
            "marker_words_allowed": marker_allowed,
            "marker_words": MARKER_WORDS,
            "length_target_words": "15-45",
            "attempt": attempt,
        }
        out2 = await call_llm(render_sys, json.dumps(rpayload, ensure_ascii=False),
                              trace_tag=f"inject-render-{row['injection_id']}")
        aside = str(out2.get("aside", "")).strip()
        if _render_ok(aside, value, marker_allowed, row):
            row["text"] = aside
            return
        exemplars = _style_exemplars(sub, rng)  # fresh exemplars on retry (4)
    raise RuntimeError(f"{row['injection_id']}: render failed asserts after 3 attempts")


def _align_plan_value(plan_text: str, candidates: list[str], value: str) -> str:
    """The plan call writes content_plan prose around its first candidate, but
    the code selects the first candidate that passes novelty. Rewrite mentions
    of rejected candidates to the selected value so the renderer never receives
    contradictory value instructions (root cause of the a1eacc2a mis-render)."""
    for c in candidates:
        if _norm(str(c)) != _norm(str(value)):
            plan_text = re.sub(rf"(?<![\w]){re.escape(str(c))}(?![\w])",
                               str(value), plan_text)
    return plan_text


def _render_ok(aside: str, value: str, marker_allowed: bool, row: dict) -> bool:
    if not aside:
        return False
    low = aside.lower()
    effective_blacklist = BLACKLIST + (ROUND2_TELL_BLACKLIST if GATE_ROUND >= 2 else [])
    if any(b.lower() in low for b in effective_blacklist):
        return False
    if _norm(value) not in _norm(aside):
        return False
    if (row["inj_type"] in CORRECTION_TYPES and not marker_allowed
            and any(m in low for m in MARKER_WORDS)):
        return False
    # Asides are the USER stating facts about themselves (or third parties)
    # to the assistant; second-person framing ("you've switched...") makes
    # the fact about the assistant and unextractable (completed_topup-
    # 6aeb4375-01 defect).
    if re.search(r"\byou(?:'ve|'re|r\b| have\b| said\b)", aside, re.I):
        return False
    if row.get("t3_variant_kind") == "invalidate":
        # Value-inversion defect (Test 3 failure audit, 21/21 F5): abandonment
        # variants must not assert the STAGED value as the thing being kept —
        # "decided not to switch, so I'm staying with {staged value}" reads as
        # completion, making the invalidate gold incoherent.
        vn = _norm(value)
        an = _norm(aside)
        # Exemption (v2 render, false-positive fix): when the staged value is
        # the TARGET of a negated/held change ("decided not to switch ... to
        # Twinings", "keeping my switch to Garden of Life on hold"), the value
        # is correctly mentioned as the abandoned destination, not retained.
        change_target = re.search(
            rf"(?:switch\w*|chang\w*|mov\w*|swap\w*)\s+(?:\w+[ &'\-\.]*){{0,6}}?to\s+(?:\w+[ &'\-\.]*){{0,3}}?{re.escape(vn)}",
            an) if vn else None
        for kept in re.finditer(
                r"(?:stay(?:ing)?|keep(?:ing)?|stick(?:ing)?|remain(?:ing)?)\s*(?:with|at|on)?\s+((?:\w+[ &'\-\.]*){1,7})",
                an):
            if vn and vn in _norm(kept.group(1)) and not change_target:
                return False
    if row["inj_type"] in CORRECTION_TYPES:
        # The corrected-to value must be asserted, not negated: "it should be
        # 10, not 11" with value=11 passes a bare containment check while
        # asserting the wrong number.
        ve = re.escape(_norm(value))
        if re.search(rf"\bnot\s+(be\s+)?{ve}(?![\w])|(?<![\w]){ve}\s+(was|is)\s+(wrong|off|incorrect)",
                     _norm(aside)):
            return False
    return True


def _style_exemplars(sub: SubInfo, rng: random.Random) -> list[str]:
    """3 register exemplars from the substrate's own user turns (Call 2 input)."""
    turns = [t["content"] for s in sub.sessions for t in s if t["role"] == "user"]
    turns = [" ".join(t.split()[:60]) for t in turns if len(t.split()) >= 10]
    return rng.sample(turns, min(3, len(turns)))


async def render(only: set[str] | None = None) -> None:
    rows = _read_jsonl(MANIFEST_PATH)
    subs, _, _ = load_substrates()
    by_id = {r["injection_id"]: r for r in rows}
    keydesc = {(k["question_id"], k["attribute"]): k["description"]
               for k in _read_jsonl(KEYS_PATH)}
    plan_sys = (PROMPTS_DIR / "injector_plan.txt").read_text()
    render_sys = (PROMPTS_DIR / "injector_render.txt").read_text()

    def selected(r):
        return (only is None or r["injection_id"] in only) and r["text"] is None

    # Rows sharing a (question_id, entity, attribute) key render SEQUENTIALLY in
    # manifest order (R1-B2): establish/staging rows precede their dependents
    # (paired values exist when needed), and each later sibling sees the earlier
    # siblings' sampled values in value_exclusions. Groups render concurrently.
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["question_id"], r["entity"], r["attribute"]), []).append(r)

    failures: list[str] = []

    async def render_group(group: list[dict]) -> None:
        for r in group:
            if selected(r):
                try:
                    await render_row(r, subs, by_id, plan_sys, render_sys, keydesc,
                                     key_group=group)
                except Exception as exc:
                    failures.append(f"{r['injection_id']}: {exc}")
                    return  # later siblings depend on this row; skip the group tail

    await asyncio.gather(*(render_group(g) for g in groups.values()))
    _write_jsonl(MANIFEST_PATH, rows)
    done = sum(1 for r in rows if r["text"])
    print(f"rendered: {done}/{len(rows)} rows have text")
    if failures:
        for f in failures:
            print(f"RENDER FAILURE: {f}", file=sys.stderr)
        raise SystemExit(f"{len(failures)} row(s) failed to render (progress persisted)")


# --------------------------------------------------------------------------- #
# lint (CLI)
# --------------------------------------------------------------------------- #

def lint() -> None:
    rows = _read_jsonl(MANIFEST_PATH)
    keys = _read_jsonl(KEYS_PATH)
    subs, cores, alts = load_substrates()
    errors: list[str] = []
    flags: list[str] = []

    n_spec = sum(r["inj_type"] in ("spec_gold", "spec_aux") for r in rows)
    n_corr = sum(r["inj_type"] in CORRECTION_TYPES for r in rows)
    if n_spec != 60 or n_corr != 60:
        errors.append(f"budget {n_spec}+{n_corr} != 60+60")
    if sum(r["inj_type"] == "cgold_correct" for r in rows) > CGOLD_CAP:
        errors.append("cgold cap exceeded")
    if any(r["question_id"] == "e61a7584" and r["inj_type"] == "cgold_correct" for r in rows):
        errors.append("e61a7584 has a cgold_correct row (impossible per 6.3)")
    cgold_per_sub: dict[str, int] = {}
    for r in rows:
        if r["inj_type"] == "cgold_correct":
            cgold_per_sub[r["question_id"]] = cgold_per_sub.get(r["question_id"], 0) + 1
    for q, n in sorted(cgold_per_sub.items()):
        if n > CGOLD_PER_SUBSTRATE_CAP:
            errors.append(f"{q}: {n} cgold_correct rows (max {CGOLD_PER_SUBSTRATE_CAP} "
                          "per substrate, R1-B2)")
    n_soft = sum(r["inj_type"] == "t3_variant" and r["t3_variant_kind"] == "invalidate"
                 and SOFT_INVALIDATE_TAG in r["edge_case_tags"] for r in rows)
    if n_soft != N_SOFT_INVALIDATE:
        errors.append(f"{n_soft} soft invalidate variants != {N_SOFT_INVALIDATE} "
                      "(predicate_spec 5.4, R1-B5)")
    if any(SOFT_INVALIDATE_TAG in r["edge_case_tags"] for r in rows
           if not (r["inj_type"] == "t3_variant" and r["t3_variant_kind"] == "invalidate")):
        errors.append("invalidate_style:soft tag on a non-invalidate row")

    for r in rows:
        sub = subs[r["question_id"]]
        p = r["placement"]
        turn = sub.sessions[p["session_idx"]][p["turn_idx"]]
        if turn["role"] != "user":
            errors.append(f"{r['injection_id']}: non-user host turn")
        if p["turn_idx"] in sub.evidence.get(p["session_idx"], set()):
            errors.append(f"{r['injection_id']}: placed on an evidence turn")
        if (sub.is_relaxed(p["session_idx"], p["turn_idx"])
                and "relaxed_adjacency" not in r["edge_case_tags"]):
            errors.append(f"{r['injection_id']}: adjacency-protected slot without relaxed tag")
        if (r["inj_type"] == "t3_variant"
                and _user_turns_after(sub, p["session_idx"], p["turn_idx"])
                < T3_SENSITIVITY_WINDOW):
            errors.append(f"{r['injection_id']}: < {T3_SENSITIVITY_WINDOW} user turns "
                          "after t* — no sensitivity window (predicate_spec 4.1, R1-B1)")
        if r["paired_with"]:
            est = next(x for x in rows if x["injection_id"] == r["paired_with"])
            ek, dk = (sub.chrono_key(est["placement"]["session_idx"], est["placement"]["turn_idx"]),
                      sub.chrono_key(p["session_idx"], p["turn_idx"]))
            if not ek < dk:
                errors.append(f"{r['injection_id']}: establish not chronologically before")
        if r["text"]:
            low = r["text"].lower()
            if any(b.lower() in low for b in BLACKLIST):
                errors.append(f"{r['injection_id']}: blacklist phrase in text")
            if r["value"] and _norm(r["value"]) not in _norm(r["text"]):
                errors.append(f"{r['injection_id']}: value missing from text")
            if r["value"] and not _fixed_value_safe(r) and not novelty_ok(
                    r["value"], sub, []):
                errors.append(f"{r['injection_id']}: value found in source transcript")
            for pat in RELATIVE_TIME_PATTERNS:
                if re.search(pat, low):
                    flags.append(f"{r['injection_id']}: relative-time phrase {pat!r} (manual audit)")
    # per-turn load (Test 2 world: exclude t3, which lives in clean copies)
    per_turn: dict[tuple, list[dict]] = {}
    for r in rows:
        if r["inj_type"].startswith("t3"):
            continue
        p = r["placement"]
        per_turn.setdefault((r["question_id"], p["session_idx"], p["turn_idx"]), []).append(r)
    for slot, rs in per_turn.items():
        if len(rs) > 2:
            errors.append(f"per-turn load >2 at {slot}")
        if len(rs) == 2:
            a, b = rs
            if {a["placement"]["position"], b["placement"]["position"]} != {OPEN, END}:
                errors.append(f"doubled turn {slot} positions not open+end")
            if a["inj_type"] == b["inj_type"] or (a["entity"], a["attribute"]) == (b["entity"], b["attribute"]):
                errors.append(f"doubled turn {slot} same type or key")
    # marker cap over rendered corrections
    rendered_corr = [r for r in rows if r["inj_type"] in CORRECTION_TYPES and r["text"]]
    if rendered_corr:
        with_marker = sum(any(m in r["text"].lower() for m in MARKER_WORDS)
                          for r in rendered_corr)
        if with_marker > MARKER_FRACTION_CAP * 60:
            errors.append(f"marker cap exceeded: {with_marker} corrections carry markers")
    # FM-4: no probe question contains any injected (or gold) value string
    values = [r["value"] for r in rows if r["value"] and len(str(r["value"])) >= 4]
    for k in keys:
        pq = k["probe_question"].lower()
        for v in values:
            if str(v).lower() in pq:
                errors.append(f"probe question for {k['attribute']} leaks value {v!r}")

    for f in flags:
        print("FLAG", f)
    if errors:
        for e in errors:
            print("ERROR", e)
        raise SystemExit(1)
    print(f"lint OK: {len(rows)} rows, {len(flags)} manual-audit flags")


def _fixed_value_safe(row: dict) -> bool:
    return row["inj_type"] == "hedged_recall" or row["inj_type"] == "t3_variant"


# --------------------------------------------------------------------------- #
# io + CLI
# --------------------------------------------------------------------------- #

def _read_jsonl(path: Path) -> list[dict]:
    # \n-only split: str.splitlines() also splits on U+2028/U+2029, which occur
    # unescaped inside real LongMemEval turn text and are legal in JSON strings.
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def main() -> None:
    ap = argparse.ArgumentParser(description="injection manifest tooling (role I2)")
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("plan")
    rp = sp.add_parser("render")
    rp.add_argument("--only", help="comma-separated injection_ids")
    sp.add_parser("lint")
    args = ap.parse_args()
    if args.cmd == "plan":
        rows, registry = plan()
        _write_jsonl(MANIFEST_PATH, rows)
        _write_jsonl(KEYS_PATH, registry)
        n_spec = sum(r["inj_type"] in ("spec_gold", "spec_aux") for r in rows)
        n_corr = sum(r["inj_type"] in CORRECTION_TYPES for r in rows)
        print(f"plan: {len(rows)} rows ({n_spec} speculation + {n_corr} correction "
              f"budget injections), {len(registry)} aux keys -> {MANIFEST_PATH}")
    elif args.cmd == "render":
        only = set(args.only.split(",")) if args.only else None
        asyncio.run(render(only))
    elif args.cmd == "lint":
        lint()


if __name__ == "__main__":
    main()
