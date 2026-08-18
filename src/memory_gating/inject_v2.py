"""v2 injection manifest generator + render pipeline (role W4 — injector v2).

Specs: research/protocol_spec_v2.md v2.1 (§2 budgets/substrates, §3.3 register +
naturalness-gate reruns, §7 row 8/9), research/injector_spec.md (v1 rules carry
over verbatim unless §2 of the v2 spec extends them), research/key_schema_v2_lme.json
(aux substrate list; audited_host_slots are AUTHORITATIVE over strict counts —
lme_schema_v2_notes.md §"for the injection-manifest author"),
research/key_schema_v2_locomo.json + locomo_analysis.md §3.2 (QA-evidence
protected turns). Code contract: pilot/INTERFACES.md conventions, EXTENDING
inject.py (machinery imported, never copied).

v2 deltas over inject.py (each pinned to spec text):
- Manifest at ../results/v2/injections/manifest.jsonl, v1 row schema + a
  first-class ``benchmark`` field ("lme" | "locomo").
- Seed 20260703 (the v2 plan seed; v1's 20260702 stays with v1 artifacts).
- Budgets (§2.4): LME 240 speculation + 120 correction across 65 designated KU
  clusters (67 rows minus one member of each shared-session pair — "no session
  is perturbed twice", §2.1) + 60 aux MS/TR substrates, allocated ~2 spec and
  ~1 corr per substrate, proportional between pools; LoCoMo 120 + 120
  (exactly 6 spec + 6 corr per speaker-substrate, V2-P2). Test-3-style staged
  cases: 50 cases x (1 staging + 4 variants) per benchmark.
- LME host slots: pilot-tier entries keep the v1 strict/relaxed geometry
  (inject.build_subinfo); the 23 v2 entries use audited_host_slots (clean
  primary, value_adjacent tagged fallback); the 60 aux substrates use the
  frozen host_slots_by_session MINUS a deterministic value-restatement scan
  run here (lme_schema_v2_notes.md instructs the manifest author to run the
  scan on the 60 before trusting strict counts; results logged).
- LoCoMo placement: an injection about speaker P is an aside in one of P's own
  unprotected turns. Protected set = every resolvable QA evidence turn (cats
  1-5) at {t-2, t, t+2} (locomo_analysis.md §3.2 protection rule) UNION the
  frozen schema's natural-update-key evidence turns (same window) UNION the
  censored staged hypotheses and NAT-RES pair anchors at the full [t-2, t+2]
  window (in-stream NAT-RES scoring reads resolution+2 turns, §4). Per-turn
  load 1 (LoCoMo turns are short; capacity is ample: >=78 host turns/speaker).
- LoCoMo render register (§3.3): the SAME frozen two-call prompts, with
  register guidance + in-conversation style exemplars supplied in the JSON
  payload (payload-level adaptation only; prompts untouched).
- Naturalness-gate item builder (§3.3: 150 injected / 150 natural per
  benchmark; natural comparison turns from the same substrates/conversations,
  matched to injected hosts' length deciles).

Documented interpretive choices (returned as flags in the implementation report):
- "~2/substrate, proportional" is realized as base 2 spec/substrate minus a
  seeded draw of 10 substrates (5 KU + 5 aux) that get 1; corrections base 1
  minus a seeded draw of 5 (3 KU + 2 aux) that get 0. Totals exact: 240/120.
- Shared-session pair designation (deterministic): the member with more
  primary host slots wins; ties broken by (primary+fallback count, tier rank
  core>alternate>v2_clean>v2_quirked, qid asc). Realized: 89941a93 over
  89941a94, 07741c45 over 07741c44. Shadow members host NO injections.
- F11 scarcity caps (lme_schema_v2_notes.md): 9ea5eabc, 9bbe84a2, 89941a94 at
  1 budget row; 07741c44, 07741c45 at 2; v1 caps 18bc8abd/3ba21379 carry over.
- C-GOLD global cap scaled at the v1 ratio (10/60 -> 20/120), <=1/substrate
  (spec v2 §2.4 pins only the per-substrate cap; the global figure is the v1
  fallback-mode ratio, flagged as an interpretive choice).
- F4 (lme_schema_v2_notes.md): 89941a93 s0 t6 is a known unmarked old-value
  restatement -> treated as evidence (protects t4/t6/t8) for v2 placement.
- The v2 render starts at the v1 round-2 style (bridged integration, stock
  connectives blacklisted) for BOTH benchmarks: v1 round 1 failed FM-1 at AUC
  0.930 on exactly the stock-connective tells; re-running a known-failed style
  would burn one of the <=3 preregistered gate rounds.

All randomness flows through one seeded Draws (seed 20260703), consumed in
manifest row order; every row records seed_draw_index (v1 2.2b/7.2 semantics).
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import random
import re
import sys
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

_SELF_DIR = Path(__file__).resolve().parent  # src/memory_gating (flat-import root)
if str(_SELF_DIR) not in sys.path:
    sys.path.insert(0, str(_SELF_DIR))

import inject
import locomo_data
from inject import (
    BLACKLIST,
    END,
    HEDGE_STRENGTHS,
    MARKER_WORDS,
    OPEN,
    POSITIONS,
    RELATIVE_TIME_PATTERNS,
    ROUND2_INTEGRATION_NOTE,
    ROUND2_TELL_BLACKLIST,
    Draws,
    Ledger,
    _align_plan_value,
    _eligible_slots,
    _fixed_value,
    _norm,
    _read_jsonl,
    _render_ok,
    _user_turns_after,
    _write_jsonl,
    build_subinfo,
    chrono_session_order,
    make_row,
    novelty_ok,
    parse_session_date,
    sibling_value_exclusions,
    splice_turn,
)

SEED_V2 = 20260703

from paths import DATA_DIR, GOLD_DIR, RESULTS_DIR
from paths import PROMPTS_DIR as _PROMPTS_DIR

ORACLE_PATH = DATA_DIR / "longmemeval" / "longmemeval_oracle.json"
LME_SCHEMA_PATH = GOLD_DIR / "key_schema_v2_lme.json"
NATURAL_PAIRS_PATH = GOLD_DIR / "natural_pairs_v2.jsonl"
V2_INJ_DIR = RESULTS_DIR / "v2" / "injections"
MANIFEST_V2_PATH = V2_INJ_DIR / "manifest.jsonl"
KEYS_V2_PATH = V2_INJ_DIR / "injected_keys.jsonl"
AUX_SCAN_PATH = V2_INJ_DIR / "aux_value_scan.jsonl"
GATE_ITEMS_PATH = {b: V2_INJ_DIR / f"gate_items_{b}.jsonl" for b in ("lme", "locomo")}
PROMPTS_DIR = _PROMPTS_DIR

BENCHMARKS = ("lme", "locomo")
SPEC_TYPES = {"spec_gold", "spec_aux"}
CORRECTION_TYPES = {"caux_correct", "cgold_correct"}
ESTABLISH_TYPES = {"aux_establish", "caux_establish"}
BUDGET_TYPES = SPEC_TYPES | CORRECTION_TYPES

# ---- v2 budget constants (protocol v2 §2.4) --------------------------------
LME_SPEC_BUDGET = 240
LME_CORR_BUDGET = 120
LME_SPEC_REDUCE = (5, 5)          # substrates drawn to 1 spec: (KU, aux)
LME_CORR_SKIP = (3, 2)            # substrates drawn to 0 corr: (KU, aux)
LOCO_SPEC_PER_SPEAKER = 6         # 12/conversation, ~6/speaker -> exact
LOCO_CORR_PER_SPEAKER = 6
T3_CASES_PER_BENCHMARK = 50
LOCO_T3_PER_CONV = 5
CGOLD_CAP_V2 = 20                 # v1 ratio 10/60 scaled to 120 (flagged choice)
CGOLD_PER_SUBSTRATE_CAP = 1       # carries over verbatim (§2.4)
T3_SENSITIVITY_WINDOW = 2         # predicate_spec 4.1 / R1-B1
N_SOFT_INVALIDATE = 10            # predicate_spec 5.4 / R1-B5, per benchmark
SOFT_INVALIDATE_TAG = "invalidate_style:soft"
MARKER_FRACTION_CAP = 0.4         # of each benchmark's 120 corrections
SPEC_SAME_SESSION_GAP = 2         # turn-index gap when est+dep share a session
CAUX_SAME_SESSION_GAP = 4         # = 2 user turns in the alternating layout

# F11 scarcity caps (lme_schema_v2_notes.md) + v1 caps carried over.
BUDGET_CAPS_V2 = {
    "18bc8abd": 2, "3ba21379": 2,           # v1 (injector_spec 2.2 v1.1)
    "9ea5eabc": 1, "9bbe84a2": 1, "89941a94": 1,
    "07741c44": 2, "07741c45": 2,
}
SHARED_SESSION_PAIRS = (("89941a93", "89941a94"), ("07741c44", "07741c45"))
TIER_RANK = {"core": 0, "alternate": 1, "v2_clean": 2, "v2_quirked": 3}
# F4: known unmarked old-value restatement missed by the frozen pilot entry.
EXTRA_RESTATEMENT_EVIDENCE = {"89941a93": [(0, 6)]}

VALUE_ADJACENT_TAG = "value_adjacent_host"

# ---- t3_variant contract -> construction metadata (FM-16 corrected mapping) --
# The original mapping labeled EVERY distractor_a and EVERY invalidate row
# completed_event. The Test-0 LoCoMo calibration gate (2026-07-04) caught this
# as an injector METADATA defect (texts and labelers/extractor were correct):
# - distractor_a / near_miss:non_resolving_event renders an exploratory
#   ACTIVITY about the focal speaker (taxonomy rule 1.2 near-miss) -> `other`.
# - distractor_a / near_miss:third_party_completion renders a completed event
#   about ANOTHER person -> `completed_event`, with ``expected_subject`` (the
#   third party) carried in a separate field; ``entity`` stays the focal
#   store-routing key (arms.py routes stores by row["entity"]).
# - invalidate + invalidate_style:soft ("keeping it on the back burner") is
#   continued deliberation -> `speculative` (S2); HARD abandonment stays
#   `completed_event` (rule 1.5f).
# - distractor_b (continued speculation) -> `speculative` (unchanged).
NEAR_MISS_TAG_PREFIX = "near_miss:"
THIRD_PARTY_SUBTYPE = "third_party_completion"


def t3_construction_label(kind: str, tags: list[str]) -> str:
    """Corrected construction_label for a t3_variant row (FM-16, 2026-07-04)."""
    if kind == "promote":
        return "completed_event"
    if kind == "invalidate":
        return "speculative" if SOFT_INVALIDATE_TAG in tags else "completed_event"
    if kind == "distractor_a":
        subtype = next(t[len(NEAR_MISS_TAG_PREFIX):] for t in tags
                       if t.startswith(NEAR_MISS_TAG_PREFIX))
        return "completed_event" if subtype == THIRD_PARTY_SUBTYPE else "other"
    if kind == "distractor_b":
        return "speculative"
    raise ValueError(f"unknown t3_variant kind {kind!r}")


def is_third_party_distractor(row: dict) -> bool:
    return (row.get("t3_variant_kind") == "distractor_a"
            and f"{NEAR_MISS_TAG_PREFIX}{THIRD_PARTY_SUBTYPE}" in row["edge_case_tags"])


#: relation nouns the near-miss contract uses for the unnamed third party
#: ("my sister", "my neighbor", ...); first match by text position wins.
THIRD_PARTY_RELATIONS = (
    "sister", "brother", "neighbor", "neighbour", "roommate", "coworker",
    "co-worker", "colleague", "friend", "buddy", "partner", "cousin",
    "mom", "mother", "dad", "father", "aunt", "uncle",
)
_RELATION_CANON = {"neighbour": "neighbor", "co-worker": "coworker"}


def derive_expected_subject(entity: str, text: str) -> str | None:
    """``expected_subject`` for a third-party-completion distractor_a aside.

    Deterministic derivation from the rendered text: the first relation noun
    (THIRD_PARTY_RELATIONS, word-boundary, case-insensitive) names the third
    party as ``{focal_entity}_{relation}`` — the dyadic extractor's observed
    naming convention for unnamed relations of a speaker (e.g. Dave's "my
    brother" -> ``dave_brother``). Returns None when no relation noun is
    found (render assert rejects such asides going forward; a manual mapping
    is required for any legacy row)."""
    low = text.lower()
    hits = []
    for rel in THIRD_PARTY_RELATIONS:
        # optional plural ("one of my friends" names a friend)
        m = re.search(rf"(?<![\w]){re.escape(rel)}s?(?![\w])", low)
        if m:
            hits.append((m.start(), rel))
    if not hits:
        return None
    rel = min(hits)[1]
    return f"{entity}_{_RELATION_CANON.get(rel, rel)}"

# ---- LoCoMo register adaptation (payload-level; frozen prompts untouched) ---
LOCOMO_LENGTH_TARGET = "8-28"
LME_LENGTH_TARGET = "15-45"
LOCOMO_REGISTER_NOTE = (
    "Register (dyadic chat corpus): this is a casual ongoing text conversation "
    "between two adult friends, NOT a user-assistant task chat. Turns are short "
    "(median ~20 words), warm and informal: contractions, light punctuation, "
    "exclamation marks, plain everyday wording, occasional mild slang; a minor "
    "typo is in-register but never force one. No task requests, no formal "
    "connectives, no lists. The aside must read as the same person continuing "
    "the same short message to their friend — keep it to one short sentence or "
    "two very short ones, at the low end of the length band.")
SAME_DAY_HORIZON_NOTE = (
    "Horizon constraint: this substrate's sessions span at most one day. Use "
    "same-day / short-horizon framings only (nothing that presupposes weeks or "
    "months between sessions).")


def _locomo_author_note(speaker: str) -> str:
    """First-person discipline for dyadic hosts (smoke-test defect: the model
    otherwise writes third-person asides about the message's own author)."""
    return (
        f"The host message is written by {speaker}, and the target entity "
        f"'{speaker.lower()}' IS {speaker} themselves. The aside asserts a fact "
        f"about the author in the FIRST person ('I', 'my'); it must never name "
        f"{speaker} or talk about them in the third person.")

# ---- aux attribute pool (v1 pool + dyadic-register extensions) -------------
# attribute: (description template, probe template, collision keywords)
# Templates use {who}: "the user" (LME) or the speaker's first name (LoCoMo).
# Probe questions are value-free by construction (FM-4).
AUX_POOL_V2: dict[str, tuple[str, str, list[str]]] = {
    "dentist_name": ("dental practice {who} currently goes to",
                     "Which dentist practice does {who} currently go to?",
                     ["dentist", "dental"]),
    "hair_salon": ("hair salon {who} currently uses",
                   "Which hair salon does {who} currently use?", ["salon", "haircut"]),
    "phone_model": ("phone model {who} currently carries",
                    "What phone model does {who} currently carry?", ["phone"]),
    "coffee_shop_regular": ("coffee shop {who} regularly goes to",
                            "Which coffee shop does {who} regularly go to?", ["coffee"]),
    "commute_mode": ("usual commute mode of {who}",
                     "How does {who} usually commute?", ["commute"]),
    "streaming_service": ("streaming service {who} subscribes to",
                          "Which streaming service does {who} subscribe to?",
                          ["streaming", "netflix", "hulu"]),
    "bank": ("bank {who} keeps their main account with",
             "Which bank does {who} keep their main account with?", ["bank"]),
    "running_shoe_model": ("running shoe model {who} currently runs in",
                           "What running shoe model does {who} currently run in?",
                           ["running shoe", "sneaker"]),
    "toothpaste_brand": ("toothpaste brand {who} currently uses",
                         "What toothpaste brand does {who} currently use?", ["toothpaste"]),
    "car_insurance_provider": ("current car insurance provider of {who}",
                               "Which company currently provides the car insurance of {who}?",
                               ["insurance"]),
    "gym_membership_tier": ("membership tier {who} holds at their gym",
                            "What membership tier does {who} hold at their gym?", ["gym"]),
    "standing_desk_setup": ("desk setup {who} currently works at",
                            "What desk setup does {who} currently work at?", ["desk"]),
    "weekly_farmers_market": ("farmers market {who} shops at each week",
                              "Which farmers market does {who} shop at each week?",
                              ["farmers market"]),
    "preferred_airline": ("airline {who} prefers to fly",
                          "Which airline does {who} prefer to fly?", ["airline"]),
    "houseplant_species": ("houseplant species {who} keeps at home",
                           "Which houseplant species does {who} keep at home?", ["houseplant"]),
    "grocery_store_regular": ("grocery store {who} usually shops at",
                              "Which grocery store does {who} usually shop at?", ["grocery"]),
    "cell_carrier": ("mobile carrier {who} is on",
                     "Which mobile carrier is {who} on?", ["carrier", "verizon"]),
    "pharmacy_location": ("pharmacy {who} fills prescriptions at",
                          "Which pharmacy does {who} fill prescriptions at?", ["pharmacy"]),
    "dry_cleaner": ("dry cleaner {who} takes clothes to",
                    "Which dry cleaner does {who} take clothes to?", ["dry clean"]),
    "meal_kit_service": ("meal kit service {who} gets deliveries from",
                         "Which meal kit service does {who} get deliveries from?", ["meal kit"]),
    # dyadic-register extensions (casual, volitional, persona-neutral)
    "laundry_detergent_brand": ("laundry detergent brand {who} buys",
                                "Which laundry detergent brand does {who} buy?",
                                ["detergent", "laundry"]),
    "podcast_app": ("app {who} listens to podcasts on",
                    "Which app does {who} listen to podcasts on?", ["podcast app"]),
    "pizza_takeout_place": ("pizza place {who} usually orders from",
                            "Which pizza place does {who} usually order from?", ["pizza"]),
    "barbershop": ("barbershop {who} currently goes to",
                   "Which barbershop does {who} currently go to?", ["barber"]),
    "budgeting_app": ("budgeting app {who} tracks spending with",
                      "Which budgeting app does {who} track spending with?", ["budget app", "budgeting"]),
    "mattress_brand": ("mattress brand {who} sleeps on",
                       "Which mattress brand does {who} sleep on?", ["mattress"]),
    "backpack_brand": ("everyday backpack brand {who} carries",
                       "Which backpack brand does {who} carry?", ["backpack"]),
    "sunscreen_brand": ("sunscreen brand {who} uses",
                        "Which sunscreen brand does {who} use?", ["sunscreen"]),
    "multivitamin_brand": ("multivitamin brand {who} takes",
                           "Which multivitamin brand does {who} take?", ["multivitamin", "vitamin"]),
    "internet_provider": ("home internet provider of {who}",
                          "Which company provides the home internet of {who}?",
                          ["internet provider", "wifi provider"]),
    "tea_brand": ("tea brand {who} keeps at home",
                  "Which tea brand does {who} keep at home?", ["tea brand"]),
    "note_taking_app": ("app {who} keeps their notes in",
                        "Which app does {who} keep their notes in?", ["note app", "notes app"]),
    "water_filter_brand": ("water filter brand {who} uses",
                           "Which water filter brand does {who} use?", ["water filter"]),
    "alarm_app": ("alarm app {who} wakes up to",
                  "Which alarm app does {who} wake up to?", ["alarm app"]),
    "bike_lock_brand": ("bike lock brand {who} uses",
                        "Which bike lock brand does {who} use?", ["bike lock"]),
}

_STOPWORDS = set(
    "the a an and or of to in on at for with my our your their his her its "
    "this that is are was were be been being i we you they it he she as by "
    "from about into after before over under have has had do does did not "
    "no yes than then there here when what which who how why".split())


# --------------------------------------------------------------------------- #
# data loading (cached raw parses; fresh derived objects per plan run)
# --------------------------------------------------------------------------- #

@cache
def _oracle_by_id() -> dict[str, dict]:
    questions = json.loads(ORACLE_PATH.read_text())
    return {q["question_id"]: q for q in questions}


@cache
def load_lme_schema_v2() -> dict:
    return json.loads(LME_SCHEMA_PATH.read_text())


@cache
def _natural_pairs() -> list[dict]:
    return [json.loads(line) for line in NATURAL_PAIRS_PATH.read_text().splitlines()
            if line.strip()]


# --------------------------------------------------------------------------- #
# LME substrate geometry
# --------------------------------------------------------------------------- #

@dataclass
class LmeSub:
    """Uniform host-slot geometry over the three LME substrate kinds.

    hosts(si) returns the primary host list for a session, falling back to the
    tagged fallback list only when the primary list is empty (v1 semantics);
    ``fallback_tag(si, t)`` names the edge tag a fallback placement must carry.
    """

    qid: str
    pool: str                       # "ku" | "aux"
    kind: str                       # "ku_pilot" | "ku_audited" | "aux"
    entry: dict | None              # KU schema entry (None for aux substrates)
    aux_row: dict | None
    sessions: list[list[dict]]
    dates: list[str]
    chrono: list[int]
    evidence: dict[int, set[int]]
    hosts_primary: dict[int, list[int]]
    hosts_fallback: dict[int, list[int]]
    fallback_tag_name: str | None
    scan_dropped: list[dict] = field(default_factory=list)
    horizon: str = "multi_week"

    def hosts(self, session_idx: int) -> list[int]:
        primary = self.hosts_primary.get(session_idx, [])
        if primary:
            return primary
        return self.hosts_fallback.get(session_idx, [])

    def all_host_slots(self) -> list[tuple[int, int]]:
        return [(si, t) for si in range(len(self.sessions)) for t in self.hosts(si)]

    def is_fallback(self, session_idx: int, turn_idx: int) -> bool:
        return turn_idx not in self.hosts_primary.get(session_idx, [])

    def fallback_tag(self, session_idx: int, turn_idx: int) -> str | None:
        if self.is_fallback(session_idx, turn_idx):
            return self.fallback_tag_name
        return None

    def chrono_key(self, session_idx: int, turn_idx: int) -> tuple[int, int]:
        return (self.chrono.index(session_idx), turn_idx)

    def transcript_text(self) -> str:
        return "\n".join(t["content"] for s in self.sessions for t in s).lower()

    @property
    def n_primary(self) -> int:
        return sum(len(v) for v in self.hosts_primary.values())

    @property
    def n_total(self) -> int:
        seen = {(si, t) for si, ts in self.hosts_primary.items() for t in ts}
        seen |= {(si, t) for si, ts in self.hosts_fallback.items() for t in ts}
        return len(seen)


def _horizon_from_dates(dates: list[str]) -> str:
    parsed = [parse_session_date(d) for d in dates]
    span = (max(parsed) - min(parsed)).days
    if span <= 1:
        return "same_day"
    if span <= 10:
        return "short_multi_day"
    return "multi_week"


def _build_ku_pilot(entry: dict, question: dict) -> LmeSub:
    entry = copy.deepcopy(entry)
    for si, ti in EXTRA_RESTATEMENT_EVIDENCE.get(entry["question_id"], []):
        entry.setdefault("extra_evidence", []).append(
            {"session_idx": si, "turn_idx": ti,
             "value": "F4 unmarked restatement (lme_schema_v2_notes.md)"})
    sub = build_subinfo(entry, question)      # v1 geometry incl. EXTRA_PROTECTED
    return LmeSub(
        qid=entry["question_id"], pool="ku", kind="ku_pilot", entry=entry,
        aux_row=None, sessions=sub.sessions, dates=sub.dates, chrono=sub.chrono,
        evidence=sub.evidence, hosts_primary=sub.strict_hosts,
        hosts_fallback=sub.relaxed_hosts, fallback_tag_name="relaxed_adjacency",
        horizon=_horizon_from_dates(sub.dates),
    )


def _build_ku_audited(entry: dict, question: dict) -> LmeSub:
    sessions = question["haystack_sessions"]
    dates = question["haystack_dates"]
    evidence: dict[int, set[int]] = {i: set() for i in range(len(sessions))}
    for k in ("old_evidence", "new_evidence"):
        evidence[entry[k]["session_idx"]].add(entry[k]["turn_idx"])
    for e in entry.get("extra_evidence", []) or []:
        evidence[e["session_idx"]].add(e["turn_idx"])
    audited = entry["audited_host_slots"]
    primary = {i: [] for i in range(len(sessions))}
    fallback = {i: [] for i in range(len(sessions))}
    for si_str, slots in audited.get("clean", {}).items():
        primary[int(si_str)] = sorted(slots)
    for si_str, slots in audited.get("value_adjacent", {}).items():
        fallback[int(si_str)] = sorted(slots)
    return LmeSub(
        qid=entry["question_id"], pool="ku", kind="ku_audited", entry=entry,
        aux_row=None, sessions=sessions, dates=dates,
        chrono=chrono_session_order(dates), evidence=evidence,
        hosts_primary=primary, hosts_fallback=fallback,
        fallback_tag_name=VALUE_ADJACENT_TAG,
        horizon=_horizon_from_dates(dates),
    )


def _salient_answer_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9']+", _norm(text))
    return [t for t in tokens
            if t not in _STOPWORDS and (len(t) >= 4 or any(c.isdigit() for c in t))]


def aux_value_scan(question: dict, host_slots: dict[str, list[int]]
                   ) -> tuple[dict[int, list[int]], list[dict]]:
    """Value-restatement scan over an aux substrate's strict host slots.

    Deterministic lexical scan in the pilot value_scan.py pattern
    (lme_schema_v2_notes.md: run on the 60 aux substrates before trusting the
    inventory strict counts). A slot is dropped when its turn text contains
    (a) the full normalized answer string, or (b) an answer number token plus
    at least one salient context word (from the answer, else the question —
    covering bare numeric answers), or (c) >= 2 salient answer words.
    Returns (kept slots per session, dropped-slot log rows).
    """
    answer = str(question.get("answer", ""))
    ans_tokens = _salient_answer_tokens(answer)
    nums = [t for t in ans_tokens if any(c.isdigit() for c in t)]
    words = sorted({t for t in ans_tokens if t not in nums})
    if not words:  # bare numeric answers: context words come from the question
        words = sorted(set(_salient_answer_tokens(str(question.get("question", "")))))
    ans_norm = _norm(answer)
    kept: dict[int, list[int]] = {}
    dropped: list[dict] = []
    for si_str, slots in sorted(host_slots.items(), key=lambda kv: int(kv[0])):
        si = int(si_str)
        kept_here = []
        for t in sorted(slots):
            text = _norm(question["haystack_sessions"][si][t]["content"])
            full_hit = len(ans_norm) >= 4 and ans_norm in text
            num_hit = any(re.search(rf"(?<![\w]){re.escape(n)}(?![\w])", text) for n in nums)
            word_hits = [w for w in words
                         if re.search(rf"(?<![\w]){re.escape(w)}(?![\w])", text)]
            if full_hit or (num_hit and word_hits) or len(word_hits) >= 2:
                dropped.append({
                    "question_id": question["question_id"], "session_idx": si,
                    "turn_idx": t,
                    "reason": ("full_answer_substring" if full_hit
                               else "number_plus_context" if num_hit
                               else "salient_word_pair"),
                    "matched": (word_hits or nums)[:4],
                })
            else:
                kept_here.append(t)
        kept[si] = kept_here
    return kept, dropped


def _build_aux(row: dict, question: dict) -> LmeSub:
    kept, dropped = aux_value_scan(question, row["host_slots_by_session"])
    sessions = question["haystack_sessions"]
    dates = question["haystack_dates"]
    chrono = chrono_session_order(dates)
    assert chrono == row["chronological_session_order"], row["question_id"]
    return LmeSub(
        qid=row["question_id"], pool="aux", kind="aux", entry=None, aux_row=row,
        sessions=sessions, dates=dates, chrono=chrono,
        evidence={i: set() for i in range(len(sessions))},
        hosts_primary={i: kept.get(i, []) for i in range(len(sessions))},
        hosts_fallback={i: [] for i in range(len(sessions))},
        fallback_tag_name=None, scan_dropped=dropped,
        horizon=row["injection_horizon"],
    )


def designate_pair_members(subs: dict[str, LmeSub]) -> dict[str, str]:
    """For each shared-session pair pick the single injection-hosting member.

    Deterministic: more primary slots > more total slots > better tier rank >
    qid asc. Returns {shadow_qid: designated_qid}.
    """
    shadow = {}
    for pair in SHARED_SESSION_PAIRS:
        ranked = sorted(
            pair,
            key=lambda q: (-subs[q].n_primary, -subs[q].n_total,
                           TIER_RANK[subs[q].entry["tier"]], q))
        shadow[ranked[1]] = ranked[0]
    return shadow


def load_lme_substrates() -> tuple[dict[str, LmeSub], list[str], list[str], dict[str, str]]:
    """-> (subs, designated KU order, aux order, shadow map)."""
    oracle = _oracle_by_id()
    schema = load_lme_schema_v2()
    subs: dict[str, LmeSub] = {}
    ku_order: list[str] = []
    for entry in schema["substrates"]:
        qid = entry["question_id"]
        question = oracle[qid]
        if entry["tier"] in ("core", "alternate"):
            subs[qid] = _build_ku_pilot(entry, question)
        else:
            subs[qid] = _build_ku_audited(entry, question)
        ku_order.append(qid)
    aux_order = []
    for row in sorted(schema["aux_injection_substrates"], key=lambda r: r["selection_rank"]):
        subs[row["question_id"]] = _build_aux(row, oracle[row["question_id"]])
        aux_order.append(row["question_id"])
    shadow = designate_pair_members(subs)
    ku_designated = [q for q in ku_order if q not in shadow]
    return subs, ku_designated, aux_order, shadow


# --------------------------------------------------------------------------- #
# LoCoMo substrate geometry
# --------------------------------------------------------------------------- #

class LocoLedger:
    """Per-turn load 1 for LoCoMo (short turns; capacity is ample)."""

    def __init__(self):
        self.used: set[tuple[int, int]] = set()

    def can_take(self, slot) -> bool:
        return slot not in self.used

    def add(self, slot) -> None:
        self.used.add(slot)


@dataclass
class LocoSub:
    conv: "locomo_data.LocomoConversation"
    protected: set[tuple[int, int]]
    hosts_by_speaker: dict[str, list[tuple[int, int]]]

    @property
    def sample_id(self) -> str:
        return self.conv.sample_id

    def transcript_text(self) -> str:
        parts = []
        for session in self.conv.sessions:
            for turn in session:
                parts.append(turn["text"])
                if turn.get("blip_caption"):
                    parts.append(turn["blip_caption"])
        return "\n".join(parts).lower()

    def dates(self) -> list[str]:
        return self.conv.dates

    def dia_id(self, session_idx: int, turn_idx: int) -> str:
        return locomo_data.format_dia_id(
            self.conv.session_numbers[session_idx], turn_idx + 1)


def _protect(prot: set, conv, session_idx: int, turn_idx: int, offsets) -> None:
    for dt in offsets:
        t = turn_idx + dt
        if 0 <= t < len(conv.sessions[session_idx]):
            prot.add((session_idx, t))


def locomo_protected_set(conv, schema: dict, pairs: list[dict]) -> set[tuple[int, int]]:
    """QA-evidence protected turns per the locomo schema + analysis §3.2.

    - every resolvable QA evidence pointer (cats 1-5): {t-2, t, t+2}
    - frozen update-key evidence (old/new/extra, all transitions): {t-2, t, t+2}
    - censored staged hypotheses + NAT-RES pair anchors: full [t-2, t+2] window
      (their in-stream scoring reads the resolution turn + 2 turns, §4).
    """
    prot: set[tuple[int, int]] = set()
    for qa in locomo_data.load_qa(conv.sample_id):
        evidence = [str(e) for e in qa.get("evidence", []) or []]
        for session_number, turn_number in locomo_data.resolve_evidence(conv, evidence)["resolved"]:
            si = conv.session_numbers.index(session_number)
            _protect(prot, conv, si, turn_number - 1, (-2, 0, 2))
    for key_row in schema["update_keys"]:
        if key_row["sample_id"] != conv.sample_id:
            continue
        dia_ids = []
        for tr in key_row["transitions"]:
            dia_ids += [e["dia_id"] for e in tr.get("old_evidence", []) or []]
            if tr.get("new_evidence"):
                dia_ids.append(tr["new_evidence"]["dia_id"])
            dia_ids += [e["dia_id"] for e in tr.get("extra_evidence", []) or []]
        for dia_id in dia_ids:
            try:
                si, ti, _ = locomo_data.turn_by_dia_id(conv, dia_id)
            except (KeyError, ValueError):
                continue
            _protect(prot, conv, si, ti, (-2, 0, 2))
    for row in schema["censored_staged_hypotheses"]:
        if row["sample_id"] != conv.sample_id:
            continue
        si, ti, _ = locomo_data.turn_by_dia_id(conv, row["dia_id"])
        _protect(prot, conv, si, ti, range(-2, 3))
    for pair in pairs:
        if pair["corpus"] != "locomo" or pair["source_question_id"] != conv.sample_id:
            continue
        for anchor in (pair["stage"], pair["resolution"]):
            si, ti, _ = locomo_data.turn_by_dia_id(conv, anchor["dia_id"])
            _protect(prot, conv, si, ti, range(-2, 3))
    return prot


def load_locomo_substrates() -> dict[str, LocoSub]:
    schema = locomo_data.load_locomo_schema()
    pairs = _natural_pairs()
    subs: dict[str, LocoSub] = {}
    for sample_id in locomo_data.conversation_ids():
        conv = locomo_data.get_conversation(sample_id)
        prot = locomo_protected_set(conv, schema, pairs)
        hosts = {
            speaker: [(si, ti)
                      for si in conv.session_order
                      for ti, turn in enumerate(conv.sessions[si])
                      if turn["speaker"] == speaker and (si, ti) not in prot]
            for speaker in conv.speakers
        }
        subs[sample_id] = LocoSub(conv=conv, protected=prot, hosts_by_speaker=hosts)
    return subs


# --------------------------------------------------------------------------- #
# planner
# --------------------------------------------------------------------------- #

class PlannerV2:
    def __init__(self):
        self.draws = Draws(SEED_V2)
        self.rows: list[dict] = []
        self.keys_registry: list[dict] = []
        self.lme_subs, self.ku_list, self.aux_list, self.shadow = load_lme_substrates()
        self.loco_subs = load_locomo_substrates()
        self.lme_ledgers: dict[str, Ledger] = {q: Ledger() for q in self.lme_subs}
        self.loco_ledgers: dict[str, LocoLedger] = {c: LocoLedger() for c in self.loco_subs}
        self.aux_used: dict[tuple, set[str]] = {}
        self.id_seq: dict[tuple[str, str], int] = {}
        self.cgold_count = 0
        self.cgold_subs: set[str] = set()
        self.tight: dict[str, bool] = {}

    # -- shared helpers -------------------------------------------------------
    def next_id(self, inj_type: str, qid: str) -> str:
        k = (inj_type, qid)
        self.id_seq[k] = self.id_seq.get(k, 0) + 1
        return f"{inj_type}-{qid}-{self.id_seq[k]:02d}"

    def _pop_id(self, inj_type: str, qid: str) -> None:
        self.id_seq[(inj_type, qid)] -= 1

    def new_aux_key(self, benchmark: str, qid: str, entity: str, who: str,
                    transcript: str, created_by: str) -> tuple[str, str]:
        """Register a fresh aux key (probe question BEFORE any value; FM-4)."""
        used = self.aux_used.setdefault((benchmark, qid, entity), set())
        eligible = [a for a, (_, _, keywords) in AUX_POOL_V2.items()
                    if a not in used and not any(k in transcript for k in keywords)]
        if not eligible:
            raise RuntimeError(f"aux pool exhausted for {benchmark}/{qid}/{entity}")
        attr = self.draws.choice(sorted(eligible))
        used.add(attr)
        desc_tpl, probe_tpl, _ = AUX_POOL_V2[attr]
        self.keys_registry.append({
            "benchmark": benchmark, "question_id": qid, "entity": entity,
            "attribute": attr, "description": desc_tpl.format(who=who),
            "probe_question": probe_tpl.format(who=who), "created_by": created_by,
        })
        return (entity, attr)

    def _release_aux_key(self, benchmark: str, qid: str, entity: str, attr: str) -> None:
        self.keys_registry.pop()
        self.aux_used[(benchmark, qid, entity)].discard(attr)

    def spec_tags(self, horizon: str) -> list[str]:
        strengths = (["deliberating", "intending"] if horizon == "same_day"
                     else HEDGE_STRENGTHS)
        hedge = self.draws.choice(strengths)
        return ["a", f"hedge:{hedge}"]

    def _row(self, **kwargs) -> dict:
        benchmark = kwargs.pop("benchmark")
        extra = kwargs.pop("extra", None) or {}
        extra = {"benchmark": benchmark, **extra}
        row = make_row(connective=None, extra=extra, **kwargs)
        self.rows.append(row)
        return row

    # ======================= LME =============================================
    def lme_place(self, qid, inj_type, key, sessions, predicate=None):
        sub, ledger = self.lme_subs[qid], self.lme_ledgers[qid]
        slots = _eligible_slots(sub, ledger, inj_type, key, sessions, predicate)
        if not slots:
            return None
        slot = self.draws.choice(slots)
        draw_idx = self.draws.n
        position = ledger.position_for(slot, self.draws, self.tight.get(qid, False))
        ledger.add(slot, inj_type, key, position)
        return (*slot, position, draw_idx, sub.fallback_tag(*slot))

    def lme_gold_spec_slots(self, qid) -> tuple[list[int], bool, int]:
        """v1 2.5 gold-key rule: slots after the last new-session evidence turn;
        flagged fallback = latest host >= new_evidence + 4."""
        sub = self.lme_subs[qid]
        new_si = sub.entry["new_evidence"]["session_idx"]
        ev = sub.evidence.get(new_si, set())
        last_ev = max(ev) if ev else -1
        after = [t for t in sub.hosts(new_si) if t > last_ev]
        if after:
            return after, False, new_si
        new_ev = sub.entry["new_evidence"]["turn_idx"]
        fallback = [t for t in sub.hosts(new_si) if t >= new_ev + 4]
        if fallback:
            return [max(fallback)], True, new_si
        return [], False, new_si

    def lme_place_spec_gold(self, qid) -> bool:
        sub = self.lme_subs[qid]
        if not sub.entry or not sub.entry.get("speculation_compatible"):
            return False
        slots, flagged, new_si = self.lme_gold_spec_slots(qid)
        if not slots:
            return False
        key = (sub.entry["entity"], sub.entry["attribute"])
        placed = self.lme_place(qid, "spec_gold", key, [new_si],
                                lambda si, t: t in slots)
        if not placed:
            return False
        si, t, pos, di, fb_tag = placed
        tags = self.spec_tags(sub.horizon)
        if flagged:
            tags.append("gold_key_spec_not_last")
        if fb_tag:
            tags.append(fb_tag)
        self._row(
            benchmark="lme",
            injection_id=self.next_id("spec_gold", qid), question_id=qid,
            inj_type="spec_gold", entity=key[0], attribute=key[1],
            session_idx=si, turn_idx=t, position=pos,
            probe_point="provenance_timeline" if flagged else "full_ingestion",
            construction_label="speculative", edge_case_tags=tags,
            seed_draw_index=di,
            extra={"injection_horizon": sub.horizon, "pool": sub.pool},
        )
        return True

    def lme_place_pair(self, qid, est_type, dep_type, key, same_gap):
        """Establish + dependent. Tier 1: dependent in a chronologically later
        session; tier 2: same session with turn gap >= same_gap."""
        sub, ledger = self.lme_subs[qid], self.lme_ledgers[qid]
        all_sessions = list(range(len(sub.sessions)))

        def dep_pred(est_slot, cross_only):
            e_ck = sub.chrono_key(*est_slot)

            def pred(si, t):
                ck = sub.chrono_key(si, t)
                if ck <= e_ck:
                    return False
                if si == est_slot[0]:
                    return (not cross_only) and t - est_slot[1] >= same_gap
                return ck[0] > e_ck[0]
            return pred

        for cross_only in (True, False):
            est_cands = []
            for si in all_sessions:
                for t in sub.hosts(si):
                    slot = (si, t)
                    if not ledger.can_take(slot, est_type, key):
                        continue
                    if _eligible_slots(sub, ledger, dep_type, key, all_sessions,
                                       dep_pred(slot, cross_only)):
                        est_cands.append(slot)
            if not est_cands:
                continue
            empty = [s for s in est_cands if not ledger.entries(s)]
            est_slot = self.draws.choice(sorted(empty if empty else est_cands))
            est_draw = self.draws.n
            est_pos = ledger.position_for(est_slot, self.draws, self.tight.get(qid, False))
            ledger.add(est_slot, est_type, key, est_pos)
            dep = self.lme_place(qid, dep_type, key, all_sessions,
                                 dep_pred(est_slot, cross_only))
            assert dep is not None
            return (est_slot, est_pos, est_draw,
                    sub.fallback_tag(*est_slot)), dep
        return None

    def _lme_pair_rows(self, qid, est_type, dep_type, dep_label, dep_tags,
                       same_gap, realloc_from=None) -> bool:
        sub = self.lme_subs[qid]
        eid = self.next_id(est_type, qid)
        key = self.new_aux_key("lme", qid, "user", "the user",
                               sub.transcript_text(), eid)
        pair = self.lme_place_pair(qid, est_type, dep_type, key, same_gap)
        if pair is None:
            self._release_aux_key("lme", qid, "user", key[1])
            self._pop_id(est_type, qid)
            return False
        (est_slot, epos, edraw, e_tag), (dsi, dt, dpos, ddraw, d_tag) = pair
        self._row(
            benchmark="lme", injection_id=eid, question_id=qid, inj_type=est_type,
            entity=key[0], attribute=key[1],
            session_idx=est_slot[0], turn_idx=est_slot[1], position=epos,
            probe_point="full_ingestion", construction_label="other",
            edge_case_tags=[e_tag] if e_tag else [], seed_draw_index=edraw,
            extra={"injection_horizon": sub.horizon, "pool": sub.pool},
        )
        tags = list(dep_tags)
        if d_tag:
            tags.append(d_tag)
        if realloc_from:
            tags.append(f"reallocated_from:{realloc_from}")
        self._row(
            benchmark="lme", injection_id=self.next_id(dep_type, qid),
            question_id=qid, inj_type=dep_type, entity=key[0], attribute=key[1],
            session_idx=dsi, turn_idx=dt, position=dpos,
            probe_point="full_ingestion", construction_label=dep_label,
            edge_case_tags=tags, seed_draw_index=ddraw, paired_with=eid,
            extra={"injection_horizon": sub.horizon, "pool": sub.pool},
        )
        return True

    def lme_place_spec_aux(self, qid, realloc_from=None) -> bool:
        return self._lme_pair_rows(
            qid, "aux_establish", "spec_aux", "speculative",
            self.spec_tags(self.lme_subs[qid].horizon),
            SPEC_SAME_SESSION_GAP, realloc_from)

    def lme_place_caux(self, qid, realloc_from=None) -> bool:
        return self._lme_pair_rows(
            qid, "caux_establish", "caux_correct", "explicit_correction",
            ["e"], CAUX_SAME_SESSION_GAP, realloc_from)

    def lme_place_cgold(self, qid, realloc_from=None) -> bool:
        sub = self.lme_subs[qid]
        if (sub.pool != "ku" or self.cgold_count >= CGOLD_CAP_V2
                or qid in self.cgold_subs or not sub.entry):
            return False
        old_si = sub.entry["old_evidence"]["session_idx"]
        new_si = sub.entry["new_evidence"]["session_idx"]
        if sub.chrono.index(old_si) >= sub.chrono.index(new_si):
            return False  # 6.3 needs the correction chronologically before v_new
        ev = sub.evidence.get(old_si, set())
        if not ev:
            return False
        last_old = max(ev)
        key = (sub.entry["entity"], sub.entry["attribute"])
        placed = self.lme_place(qid, "cgold_correct", key, [old_si],
                                lambda si, t: t > last_old)
        if placed is None:
            return False
        si, t, pos, di, fb_tag = placed
        tags = ["e", "cgold"]
        if fb_tag:
            tags.append(fb_tag)
        if realloc_from:
            tags.append(f"reallocated_from:{realloc_from}")
        self._row(
            benchmark="lme", injection_id=self.next_id("cgold_correct", qid),
            question_id=qid, inj_type="cgold_correct", entity=key[0],
            attribute=key[1], session_idx=si, turn_idx=t, position=pos,
            probe_point="old_session_boundary",
            construction_label="explicit_correction", edge_case_tags=tags,
            seed_draw_index=di, old_value=sub.entry["old_value"],
            extra={"injection_horizon": sub.horizon, "pool": sub.pool},
        )
        self.cgold_count += 1
        self.cgold_subs.add(qid)
        return True

    def lme_place_correction(self, qid, realloc_from=None) -> bool:
        if self.lme_place_caux(qid, realloc_from):
            return True
        return self.lme_place_cgold(qid, realloc_from)

    def plan_lme(self) -> None:
        ku, aux = self.ku_list, self.aux_list
        reduce_spec = set(self.draws.sample(ku, LME_SPEC_REDUCE[0])
                          + self.draws.sample(aux, LME_SPEC_REDUCE[1]))
        skip_corr = set(self.draws.sample(ku, LME_CORR_SKIP[0])
                        + self.draws.sample(aux, LME_CORR_SKIP[1]))
        spec_target = {q: 2 - (q in reduce_spec) for q in ku + aux}
        corr_target = {q: 1 - (q in skip_corr) for q in ku + aux}
        overflow: list[tuple[str, str]] = []
        for q in ku + aux:
            cap = BUDGET_CAPS_V2.get(q)
            if cap is None:
                continue
            while spec_target[q] + corr_target[q] > cap:
                if corr_target[q] > 0:
                    corr_target[q] -= 1
                    overflow.append(("corr", q))
                else:
                    spec_target[q] -= 1
                    overflow.append(("spec", q))
        # position-tightness heuristic (v1): worst-case aside count vs slots
        for q in ku + aux:
            sub = self.lme_subs[q]
            need = 2 * spec_target[q] + 2 * corr_target[q]
            self.tight[q] = need > len(sub.all_host_slots())

        for q in ku + aux:
            spec_left, corr_left = spec_target[q], corr_target[q]
            if self.lme_subs[q].pool == "ku" and spec_left > 0:
                if self.lme_place_spec_gold(q):
                    spec_left -= 1
                # incompatible / unplaceable gold -> the slot stays in spec_left
                # and is realized as an aux-key speculation below
            for _ in range(corr_left):
                if not self.lme_place_correction(q):
                    overflow.append(("corr", q))
            for _ in range(spec_left):
                if not self.lme_place_spec_aux(q):
                    overflow.append(("spec", q))

        # 7.3 reallocation: same pool first, then cross-pool; capped substrates
        # are never reallocation targets.
        for kind, from_q in overflow:
            from_pool = self.lme_subs[from_q].pool
            pools = [from_pool] + [p for p in ("ku", "aux") if p != from_pool]
            placed = False
            for pool in pools:
                candidates = [q for q in (ku if pool == "ku" else aux)
                              if q not in BUDGET_CAPS_V2 and q != from_q]
                while candidates and not placed:
                    q = self.draws.choice(sorted(candidates))
                    ok = (self.lme_place_spec_aux(q, realloc_from=from_q)
                          if kind == "spec"
                          else self.lme_place_correction(q, realloc_from=from_q))
                    if ok:
                        placed = True
                    else:
                        candidates.remove(q)
                if placed:
                    break
            if not placed:
                raise RuntimeError(f"could not reallocate lme {kind} from {from_q}")

    # ---- LME Test 3 ----------------------------------------------------------
    def _lme_t3_geometry(self, qid):
        """(stage_si, stage_hosts, variant_slots) for the earliest feasible
        staging session, or None."""
        sub = self.lme_subs[qid]
        for stage_pos in range(len(sub.chrono) - 1):
            stage_si = sub.chrono[stage_pos]
            stage_hosts = sub.hosts(stage_si)
            if not stage_hosts:
                continue
            variant_slots = [
                (v_si, t)
                for v_pos in range(stage_pos + 1, len(sub.chrono))
                for v_si in (sub.chrono[v_pos],)
                for t in sub.hosts(v_si)
                if _user_turns_after(sub, v_si, t) >= T3_SENSITIVITY_WINDOW
            ]
            if variant_slots:
                return stage_si, stage_hosts, variant_slots
        return None

    def _t3_variant_rows(self, *, benchmark, qid, case_id, staging_id, key,
                         var_si, var_t, var_pos, var_draw, tags_extra,
                         extra) -> None:
        near_miss = self.draws.choice(["third_party_completion", "non_resolving_event"])
        # construction_label per kind via the FM-16 corrected mapping; soft
        # invalidates are drawn later (_post_tag_draws), which re-derives the
        # invalidate labels once the invalidate_style:soft tags exist.
        variant_specs = [
            ("promote", ["t3"]),
            ("invalidate", ["t3", "f"]),
            ("distractor_a", ["t3", f"near_miss:{near_miss}"]),
            ("distractor_b", ["t3", "continued_speculation"]),
        ]
        for kind, tags in variant_specs:
            label = t3_construction_label(kind, tags)
            self._row(
                benchmark=benchmark,
                injection_id=f"t3_variant-{qid}-{case_id.rsplit('-', 1)[-1]}-{kind}",
                question_id=qid, inj_type="t3_variant", entity=key[0],
                attribute=key[1], session_idx=var_si, turn_idx=var_t,
                position=var_pos, probe_point="full_ingestion",
                construction_label=label, edge_case_tags=tags + tags_extra,
                seed_draw_index=var_draw, t3_case_id=case_id,
                t3_variant_kind=kind, paired_with=staging_id, extra=extra,
            )

    def plan_lme_t3(self) -> None:
        eligible = [q for q in self.ku_list + self.aux_list
                    if self._lme_t3_geometry(q) is not None]
        chosen = self.draws.sample(eligible, T3_CASES_PER_BENCHMARK)
        for qid in chosen:
            sub = self.lme_subs[qid]
            stage_si, stage_hosts, variant_slots = self._lme_t3_geometry(qid)
            case_id = f"t3-{qid}-c1"
            staging_id = self.next_id("t3_staging", qid)
            key = self.new_aux_key("lme", qid, "user", "the user",
                                   sub.transcript_text(), staging_id)
            user_turns = [t for t in range(len(sub.sessions[stage_si]))
                          if sub.sessions[stage_si][t]["role"] == "user"]
            mid = user_turns[len(user_turns) // 2]
            best = min(abs(t - mid) for t in stage_hosts)
            stage_t = self.draws.choice(sorted(t for t in stage_hosts
                                               if abs(t - mid) == best))
            stage_draw = self.draws.n
            stage_pos = self.draws.choice(list(POSITIONS))
            tags = self.spec_tags(sub.horizon)
            fb = sub.fallback_tag(stage_si, stage_t)
            if fb:
                tags.append(fb)
            extra = {"injection_horizon": sub.horizon, "pool": sub.pool}
            self._row(
                benchmark="lme", injection_id=staging_id, question_id=qid,
                inj_type="t3_staging", entity=key[0], attribute=key[1],
                session_idx=stage_si, turn_idx=stage_t, position=stage_pos,
                probe_point="full_ingestion", construction_label="speculative",
                edge_case_tags=tags, seed_draw_index=stage_draw,
                t3_case_id=case_id, extra=extra,
            )
            var_si, var_t = self.draws.choice(sorted(variant_slots))
            var_draw = self.draws.n
            var_pos = self.draws.choice(list(POSITIONS))
            vtags = []
            vfb = sub.fallback_tag(var_si, var_t)
            if vfb:
                vtags.append(vfb)
            self._t3_variant_rows(
                benchmark="lme", qid=qid, case_id=case_id, staging_id=staging_id,
                key=key, var_si=var_si, var_t=var_t, var_pos=var_pos,
                var_draw=var_draw, tags_extra=vtags, extra=extra)

    # ======================= LoCoMo ==========================================
    def _loco_free(self, sample_id: str, speaker: str) -> list[tuple[int, int]]:
        sub, ledger = self.loco_subs[sample_id], self.loco_ledgers[sample_id]
        return [s for s in sub.hosts_by_speaker[speaker] if ledger.can_take(s)]

    def loco_place_pair(self, sample_id, speaker, est_type, dep_type, key,
                        same_gap):
        ledger = self.loco_ledgers[sample_id]
        free = self._loco_free(sample_id, speaker)

        def deps_after(est_slot, cross_only):
            esi, eti = est_slot
            out = []
            for (si, ti) in free:
                if (si, ti) == est_slot:
                    continue
                if si > esi:
                    out.append((si, ti))
                elif si == esi and not cross_only and ti - eti >= same_gap:
                    out.append((si, ti))
            return out

        for cross_only in (True, False):
            est_cands = [s for s in free if deps_after(s, cross_only)]
            if not est_cands:
                continue
            est_slot = self.draws.choice(sorted(est_cands))
            est_draw = self.draws.n
            est_pos = self.draws.choice(list(POSITIONS))
            ledger.add(est_slot)
            dep_slot = self.draws.choice(
                sorted(d for d in deps_after(est_slot, cross_only)
                       if ledger.can_take(d)))
            dep_draw = self.draws.n
            dep_pos = self.draws.choice(list(POSITIONS))
            ledger.add(dep_slot)
            return (est_slot, est_pos, est_draw), (dep_slot, dep_pos, dep_draw)
        return None

    def _loco_pair_rows(self, sample_id, speaker, est_type, dep_type, dep_label,
                        dep_tags, same_gap) -> bool:
        sub = self.loco_subs[sample_id]
        entity = speaker.lower()
        eid = self.next_id(est_type, sample_id)
        key = self.new_aux_key("locomo", sample_id, entity, speaker,
                               sub.transcript_text(), eid)
        pair = self.loco_place_pair(sample_id, speaker, est_type, dep_type,
                                    key, same_gap)
        if pair is None:
            self._release_aux_key("locomo", sample_id, entity, key[1])
            self._pop_id(est_type, sample_id)
            return False
        (est_slot, epos, edraw), (dep_slot, dpos, ddraw) = pair
        self._row(
            benchmark="locomo", injection_id=eid, question_id=sample_id,
            inj_type=est_type, entity=key[0], attribute=key[1],
            session_idx=est_slot[0], turn_idx=est_slot[1], position=epos,
            probe_point="full_ingestion", construction_label="other",
            edge_case_tags=[], seed_draw_index=edraw,
            extra={"speaker": speaker, "dia_id": sub.dia_id(*est_slot)},
        )
        self._row(
            benchmark="locomo", injection_id=self.next_id(dep_type, sample_id),
            question_id=sample_id, inj_type=dep_type, entity=key[0],
            attribute=key[1], session_idx=dep_slot[0], turn_idx=dep_slot[1],
            position=dpos, probe_point="full_ingestion",
            construction_label=dep_label, edge_case_tags=list(dep_tags),
            seed_draw_index=ddraw, paired_with=eid,
            extra={"speaker": speaker, "dia_id": sub.dia_id(*dep_slot)},
        )
        return True

    def plan_locomo(self) -> None:
        for sample_id in locomo_data.conversation_ids():
            sub = self.loco_subs[sample_id]
            for speaker in sub.conv.speakers:
                for _ in range(LOCO_SPEC_PER_SPEAKER):
                    ok = self._loco_pair_rows(
                        sample_id, speaker, "aux_establish", "spec_aux",
                        "speculative", self.spec_tags("multi_week"),
                        SPEC_SAME_SESSION_GAP)
                    if not ok:
                        raise RuntimeError(f"locomo spec slot exhausted: "
                                           f"{sample_id}/{speaker}")
                for _ in range(LOCO_CORR_PER_SPEAKER):
                    ok = self._loco_pair_rows(
                        sample_id, speaker, "caux_establish", "caux_correct",
                        "explicit_correction", ["e"], CAUX_SAME_SESSION_GAP)
                    if not ok:
                        raise RuntimeError(f"locomo corr slot exhausted: "
                                           f"{sample_id}/{speaker}")

    # ---- LoCoMo Test 3 -------------------------------------------------------
    def _loco_t3_case(self, sample_id, speaker, case_no, used_staging) -> None:
        sub = self.loco_subs[sample_id]
        conv = sub.conv
        entity = speaker.lower()
        n_sessions = len(conv.sessions)
        hosts = sub.hosts_by_speaker[speaker]

        def window_ok(si, ti):
            return len(conv.sessions[si]) - 1 - ti >= T3_SENSITIVITY_WINDOW

        variant_pool = [(si, ti) for (si, ti) in hosts if window_ok(si, ti)]
        stage_cands = [
            (si, ti) for (si, ti) in hosts
            if si < n_sessions - 1 and (si, ti) not in used_staging
            and any(v_si > si for (v_si, _) in variant_pool)
        ]
        assert stage_cands, f"no t3 staging slot for {sample_id}/{speaker}"
        stage_slot = self.draws.choice(sorted(stage_cands))
        stage_draw = self.draws.n
        stage_pos = self.draws.choice(list(POSITIONS))
        used_staging.add(stage_slot)
        case_id = f"t3-{sample_id}-c{case_no}"
        staging_id = self.next_id("t3_staging", sample_id)
        key = self.new_aux_key("locomo", sample_id, entity, speaker,
                               sub.transcript_text(), staging_id)
        self._row(
            benchmark="locomo", injection_id=staging_id, question_id=sample_id,
            inj_type="t3_staging", entity=key[0], attribute=key[1],
            session_idx=stage_slot[0], turn_idx=stage_slot[1],
            position=stage_pos, probe_point="full_ingestion",
            construction_label="speculative",
            edge_case_tags=self.spec_tags("multi_week"),
            seed_draw_index=stage_draw, t3_case_id=case_id,
            extra={"speaker": speaker, "dia_id": sub.dia_id(*stage_slot)},
        )
        # variant slot fixed at case creation (v1 9.4); prefer a session within
        # +1..+6 chronological sessions after staging (the natural-arc lag band,
        # locomo_analysis.md §2.2), else any later session.
        preferred = [(si, ti) for (si, ti) in variant_pool
                     if stage_slot[0] < si <= stage_slot[0] + 6]
        any_later = [(si, ti) for (si, ti) in variant_pool if si > stage_slot[0]]
        var_slot = self.draws.choice(sorted(preferred if preferred else any_later))
        var_draw = self.draws.n
        var_pos = self.draws.choice(list(POSITIONS))
        self._t3_variant_rows(
            benchmark="locomo", qid=sample_id, case_id=case_id,
            staging_id=staging_id, key=key, var_si=var_slot[0],
            var_t=var_slot[1], var_pos=var_pos, var_draw=var_draw,
            tags_extra=[],
            extra={"speaker": speaker, "dia_id": sub.dia_id(*var_slot)})

    def plan_locomo_t3(self) -> None:
        for sample_id in locomo_data.conversation_ids():
            sub = self.loco_subs[sample_id]
            split = self.draws.choice([(3, 2), (2, 3)])
            used_staging: set[tuple[int, int]] = set()
            case_no = 0
            for speaker, n_cases in zip(sub.conv.speakers, split):
                for _ in range(n_cases):
                    case_no += 1
                    self._loco_t3_case(sample_id, speaker, case_no, used_staging)

    # ---- post passes ----------------------------------------------------------
    def _post_tag_draws(self) -> None:
        for benchmark in BENCHMARKS:
            inv_ids = sorted(r["injection_id"] for r in self.rows
                             if r["benchmark"] == benchmark
                             and r["inj_type"] == "t3_variant"
                             and r["t3_variant_kind"] == "invalidate")
            soft = set(self.draws.sample(inv_ids, N_SOFT_INVALIDATE))
            for r in self.rows:
                if r["injection_id"] in soft and r["benchmark"] == benchmark:
                    r["edge_case_tags"].append(SOFT_INVALIDATE_TAG)
                    # FM-16: a soft hold is continued deliberation (S2), not a
                    # completed abandonment — re-derive the label with the tag.
                    r["construction_label"] = t3_construction_label(
                        "invalidate", r["edge_case_tags"])
        for benchmark in BENCHMARKS:
            corr_ids = sorted(r["injection_id"] for r in self.rows
                              if r["benchmark"] == benchmark
                              and r["inj_type"] in CORRECTION_TYPES)
            allowed = set(self.draws.sample(
                corr_ids, int(len(corr_ids) * MARKER_FRACTION_CAP)))
            for r in self.rows:
                if r["injection_id"] in allowed and r["benchmark"] == benchmark:
                    r["edge_case_tags"].append("marker_allowed")

    def run(self) -> tuple[list[dict], list[dict]]:
        self.plan_lme()
        self.plan_lme_t3()
        self.plan_locomo()
        self.plan_locomo_t3()
        self._post_tag_draws()
        return self.rows, self.keys_registry


def plan_v2() -> tuple[list[dict], list[dict]]:
    """Pure, deterministic v2 manifest generation (no LLM calls)."""
    return PlannerV2().run()


def scan_log_rows() -> list[dict]:
    """The aux value-restatement scan log (deterministic; written at plan)."""
    subs, _, aux_list, _ = load_lme_substrates()
    out = []
    for qid in aux_list:
        sub = subs[qid]
        out.append({
            "question_id": qid,
            "kept_slots": {str(si): sub.hosts_primary[si]
                           for si in sorted(sub.hosts_primary)},
            "dropped": sub.scan_dropped,
        })
    return out


# --------------------------------------------------------------------------- #
# perturbed-world assembly (pure)
# --------------------------------------------------------------------------- #

def build_perturbed_conversation(conv, manifest_rows):
    """LoCoMo analog of inject.build_perturbed_substrate: splice rendered asides
    into speaker turns of a LocomoConversation (fresh copy returned)."""
    by_turn: dict[tuple[int, int], list[tuple[str, str]]] = {}
    for row in manifest_rows:
        if row["question_id"] != conv.sample_id:
            continue
        if not row["text"]:
            raise ValueError(f"row {row['injection_id']} has no rendered text")
        p = row["placement"]
        slot = (p["session_idx"], p["turn_idx"])
        turn = conv.sessions[slot[0]][slot[1]]
        if turn["speaker"] != row["speaker"]:
            raise ValueError(
                f"{row['injection_id']}: speaker mismatch at {slot}: "
                f"{turn['speaker']} != {row['speaker']}")
        by_turn.setdefault(slot, []).append((p["position"], row["text"]))
    new_sessions = []
    for si, session in enumerate(conv.sessions):
        new_session = []
        for ti, turn in enumerate(session):
            ins = by_turn.get((si, ti))
            turn = dict(turn)
            if ins:
                turn["content" if "content" in turn else "text"] = splice_turn(
                    turn["text"], ins)
            new_session.append(turn)
        new_sessions.append(new_session)
    import dataclasses as _dc
    return _dc.replace(conv, sessions=new_sessions)


# --------------------------------------------------------------------------- #
# render (two-call pipeline; frozen prompts; NOT executed by this role)
# --------------------------------------------------------------------------- #

def _style_exemplars_lme(sub: LmeSub, rng: random.Random) -> list[str]:
    turns = [t["content"] for s in sub.sessions for t in s if t["role"] == "user"]
    turns = [" ".join(t.split()[:60]) for t in turns if len(t.split()) >= 10]
    return rng.sample(turns, min(3, len(turns)))


def _style_exemplars_locomo(sub: LocoSub, speaker: str, rng: random.Random) -> list[str]:
    turns = [t["text"] for s in sub.conv.sessions for t in s
             if t["speaker"] == speaker]
    turns = [" ".join(t.split()[:40]) for t in turns if len(t.split()) >= 6]
    return rng.sample(turns, min(3, len(turns)))


def _lme_other_session_date(sub: LmeSub, row: dict, by_id: dict) -> str:
    p = row["placement"]
    if row["paired_with"] and by_id.get(row["paired_with"]):
        est = by_id[row["paired_with"]]
        if est["question_id"] == row["question_id"]:
            return sub.dates[est["placement"]["session_idx"]]
    others = [sub.dates[si] for si in sub.chrono if si != p["session_idx"]]
    return others[-1] if others else sub.dates[p["session_idx"]]


async def render_row_v2(row: dict, ctx: "RenderContext", by_id: dict[str, dict],
                        key_group: list[dict] = ()) -> None:
    from llm import call_llm

    benchmark = row["benchmark"]
    p = row["placement"]
    if benchmark == "lme":
        sub = ctx.lme_subs[row["question_id"]]
        host_turn = sub.sessions[p["session_idx"]][p["turn_idx"]]["content"]
        host_date = sub.dates[p["session_idx"]]
        other_date = _lme_other_session_date(sub, row, by_id)
        all_dates = [sub.dates[si] for si in sub.chrono]
        transcript_sub = sub
        horizon = row.get("injection_horizon")
        length_target = LME_LENGTH_TARGET
        register_note = None
    else:
        sub = ctx.loco_subs[row["question_id"]]
        turn = sub.conv.sessions[p["session_idx"]][p["turn_idx"]]
        host_turn = turn["text"]
        host_date = sub.conv.dates[p["session_idx"]]
        if row["paired_with"] and by_id.get(row["paired_with"]):
            est = by_id[row["paired_with"]]
            other_date = sub.conv.dates[est["placement"]["session_idx"]]
        else:
            other_date = sub.conv.dates[max(p["session_idx"] - 1, 0)]
        all_dates = None  # 19-32 sessions; the two relevant dates suffice
        transcript_sub = sub
        horizon = None
        length_target = LOCOMO_LENGTH_TARGET
        register_note = LOCOMO_REGISTER_NOTE

    paired = by_id.get(row["paired_with"]) if row["paired_with"] else None
    established = paired["value"] if paired else None
    if paired and paired["value"] is None:
        raise RuntimeError(f"{row['injection_id']}: paired row not rendered yet")
    exclusions = []
    if row["inj_type"] in ("spec_gold", "cgold_correct"):
        entry = ctx.lme_subs[row["question_id"]].entry
        exclusions += [entry["old_value"], entry["new_value"]]
    if established:
        exclusions.append(established)
    for v in sibling_value_exclusions(row, list(key_group)):
        if v not in exclusions:
            exclusions.append(v)
    fixed = _fixed_value(row, by_id)
    marker_allowed = "marker_allowed" in row["edge_case_tags"]

    payload = {
        "contract": inject.contract_key(row),
        "injection_type": row["inj_type"],
        "target_key": {"entity": row["entity"], "attribute": row["attribute"]},
        "established_value": established,
        "gold_old_value": row["old_value"] if row["inj_type"] == "cgold_correct" else None,
        "fixed_value": fixed,
        "value_exclusions": exclusions,
        "host_turn_text": host_turn,
        "insertion_position": p["position"],
        "host_session_date": host_date,
        "other_session_date": other_date,
        "hedge_strength": next((t.split(":")[1] for t in row["edge_case_tags"]
                                if t.startswith("hedge:")), None),
        "near_miss_subtype": next((t.split(":")[1] for t in row["edge_case_tags"]
                                   if t.startswith("near_miss:")), None),
        "attribute_description": ctx.keydesc.get(
            (row["question_id"], row["entity"], row["attribute"]),
            row["attribute"].replace("_", " ")),
    }
    if all_dates:
        payload["all_session_dates"] = all_dates
    if register_note:
        payload["register_note"] = register_note
        payload["speaker_name"] = row.get("speaker")
        payload["author_note"] = _locomo_author_note(row["speaker"])
    if horizon:
        payload["injection_horizon"] = horizon
        if horizon == "same_day":
            payload["horizon_note"] = SAME_DAY_HORIZON_NOTE

    out1 = await call_llm(ctx.plan_sys, json.dumps(payload, ensure_ascii=False),
                          trace_tag=f"inject-v2-plan-{row['injection_id']}")
    if fixed is not None:
        value = fixed
    else:
        candidates = [str(c) for c in out1.get("value_candidates", [])]
        value = next((c for c in candidates
                      if novelty_ok(c, transcript_sub, exclusions)), None)
        for retry_round in range(1, 4):
            if value is not None:
                break
            retry_payload = dict(payload)
            retry_payload["value_exclusions"] = exclusions + candidates
            retry_payload["retry_note"] = (
                f"Retry round {retry_round}: every previous candidate was rejected "
                "because it appears verbatim in the source transcript or collides "
                "with an excluded value. Propose distinctly different candidates.")
            if row["attribute"].endswith("_count"):
                retry_payload["value_format"] = (
                    "REQUIRED: each candidate must be the count followed by its "
                    "unit noun (e.g. '4 bikes', '11 short stories'); bare "
                    "numerals are invalid and will be rejected.")
            out1 = await call_llm(
                ctx.plan_sys, json.dumps(retry_payload, ensure_ascii=False),
                trace_tag=f"inject-v2-plan-retry{retry_round}-{row['injection_id']}")
            retry_candidates = [str(c) for c in out1.get("value_candidates", [])]
            value = next((c for c in retry_candidates
                          if novelty_ok(c, transcript_sub, exclusions)), None)
            candidates = candidates + retry_candidates
        if value is None:
            raise RuntimeError(f"{row['injection_id']}: no candidate passed "
                               f"novelty after retries: {candidates}")
        row["value_candidates"] = candidates
    row["value"] = value
    if row["inj_type"] == "caux_correct":
        row["old_value"] = established

    rng = random.Random(f"{SEED_V2}:{row['injection_id']}")
    if benchmark == "lme":
        exemplars = _style_exemplars_lme(sub, rng)
    else:
        exemplars = _style_exemplars_locomo(sub, row["speaker"], rng)
    for attempt in range(3):
        rpayload = {
            "contract": inject.contract_key(row),
            "content_plan": _align_plan_value(
                out1.get("content_plan", ""),
                row.get("value_candidates") or [], value),
            "value": value,
            "established_value": established,
            "old_value": row["old_value"],
            "connective": None,
            "integration_note": ROUND2_INTEGRATION_NOTE,
            "insertion_position": p["position"],
            "host_turn_text": host_turn,
            "style_exemplars": exemplars,
            "blacklist": BLACKLIST + ROUND2_TELL_BLACKLIST,
            "marker_words_allowed": marker_allowed,
            "marker_words": MARKER_WORDS,
            "length_target_words": length_target,
            "attempt": attempt,
        }
        if register_note:
            rpayload["register_note"] = register_note
            rpayload["author_note"] = _locomo_author_note(row["speaker"])
        out2 = await call_llm(ctx.render_sys, json.dumps(rpayload, ensure_ascii=False),
                              trace_tag=f"inject-v2-render-{row['injection_id']}")
        aside = str(out2.get("aside", "")).strip()
        if _render_ok(aside, value, marker_allowed, row) and _render_ok_v2(aside, row):
            row["text"] = aside
            if is_third_party_distractor(row):
                # FM-16: record the third party the aside is about (the
                # attribution-correct extraction subject); derivability is
                # guaranteed by _render_ok_v2.
                row["expected_subject"] = derive_expected_subject(row["entity"], aside)
            return
        if benchmark == "lme":
            exemplars = _style_exemplars_lme(sub, rng)
        else:
            exemplars = _style_exemplars_locomo(sub, row["speaker"], rng)
    raise RuntimeError(f"{row['injection_id']}: render failed asserts after 3 attempts")


def _render_ok_v2(aside: str, row: dict) -> bool:
    """v2-only render asserts on top of inject._render_ok.

    LoCoMo: the aside is spoken BY row['speaker'] about themselves — any
    mention of their own first name is a third-person mis-render (observed in
    the smoke test) and an attribution hazard for the dyadic extractor.

    Third-party-completion distractor_a rows (both benchmarks): the third
    party must be nameable by the deterministic relation lexicon, else
    ``expected_subject`` (FM-16) would be underivable."""
    if row["benchmark"] == "locomo":
        speaker = str(row.get("speaker", ""))
        if speaker and re.search(rf"(?<![\w]){re.escape(speaker)}(?![\w])",
                                 aside, re.I):
            return False
    if (is_third_party_distractor(row)
            and derive_expected_subject(row["entity"], aside) is None):
        return False
    return True


@dataclass
class RenderContext:
    lme_subs: dict[str, LmeSub]
    loco_subs: dict[str, LocoSub]
    keydesc: dict[tuple[str, str, str], str]
    plan_sys: str
    render_sys: str


def _render_context() -> RenderContext:
    lme_subs, _, _, _ = load_lme_substrates()
    return RenderContext(
        lme_subs=lme_subs,
        loco_subs=load_locomo_substrates(),
        keydesc={(k["question_id"], k["entity"], k["attribute"]): k["description"]
                 for k in _read_jsonl(KEYS_V2_PATH)},
        plan_sys=(PROMPTS_DIR / "injector_plan.txt").read_text(),
        render_sys=(PROMPTS_DIR / "injector_render.txt").read_text(),
    )


async def render_v2(only: set[str] | None = None,
                    benchmark: str | None = None) -> None:
    rows = _read_jsonl(MANIFEST_V2_PATH)
    ctx = _render_context()
    by_id = {r["injection_id"]: r for r in rows}

    def selected(r):
        return ((only is None or r["injection_id"] in only)
                and (benchmark is None or r["benchmark"] == benchmark)
                and r["text"] is None)

    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(
            (r["benchmark"], r["question_id"], r["entity"], r["attribute"]),
            []).append(r)

    failures: list[str] = []

    async def render_group(group: list[dict]) -> None:
        for r in group:
            if selected(r):
                try:
                    await render_row_v2(r, ctx, by_id, key_group=group)
                except Exception as exc:
                    failures.append(f"{r['injection_id']}: {exc}")
                    return  # later siblings depend on this row
            elif r["text"] is None and any(selected(x) for x in group):
                # unrendered ancestor of a selected sibling: dependents would
                # fail on the missing paired value; stop the group here.
                return

    await asyncio.gather(*(render_group(g) for g in groups.values()))
    _write_jsonl(MANIFEST_V2_PATH, rows)
    done = sum(1 for r in rows if r["text"])
    print(f"rendered: {done}/{len(rows)} rows have text")
    if failures:
        for f in failures:
            print(f"RENDER FAILURE: {f}", file=sys.stderr)
        raise SystemExit(f"{len(failures)} row(s) failed to render (progress persisted)")


# --------------------------------------------------------------------------- #
# naturalness-gate item builder (§3.3: 150 injected / 150 natural per benchmark)
# --------------------------------------------------------------------------- #

GATE_STRATA: list[tuple[str, set[str], int]] = [
    ("speculation", {"spec_gold", "spec_aux"}, 45),
    ("correction", {"caux_correct", "cgold_correct"}, 40),
    ("establish", {"aux_establish", "caux_establish"}, 25),
    ("t3_staging", {"t3_staging"}, 20),
    ("t3_variant", {"t3_variant"}, 20),
]
GATE_N = 150
GATE_CONTEXT_TURNS = 2


def _world_rows(row: dict, bench_rows: list[dict]) -> list[dict]:
    """Manifest rows visible in ``row``'s replay world (v1 9.1: each T3 case
    lives in its own clean copy; a variant world holds staging + that variant)."""
    if row["t3_case_id"]:
        world = [r for r in bench_rows if r["t3_case_id"] == row["t3_case_id"]
                 and r["inj_type"] == "t3_staging"]
        if row["inj_type"] == "t3_variant":
            world.append(row)
        return world
    return [r for r in bench_rows
            if r["question_id"] == row["question_id"] and not r["t3_case_id"]]


def _spliced_turn_text(raw_text: str, slot_rows: list[dict]) -> str:
    ins = [(r["placement"]["position"], r["text"]) for r in slot_rows]
    return splice_turn(raw_text, ins) if ins else raw_text


def _gate_turn_and_context(row_or_slot, benchmark: str, ctx: RenderContext,
                           world: list[dict], question_id: str,
                           session_idx: int, turn_idx: int) -> tuple[str, list[str]]:
    by_slot: dict[tuple[int, int], list[dict]] = {}
    for r in world:
        p = r["placement"]
        by_slot.setdefault((p["session_idx"], p["turn_idx"]), []).append(r)

    if benchmark == "lme":
        sessions = ctx.lme_subs[question_id].sessions
        raw = lambda si, ti: sessions[si][ti]["content"]
        n_turns = lambda si: len(sessions[si])
        speaker_of = lambda si, ti: sessions[si][ti]["role"]
    else:
        conv = ctx.loco_subs[question_id].conv
        raw = lambda si, ti: conv.sessions[si][ti]["text"]
        n_turns = lambda si: len(conv.sessions[si])
        speaker_of = lambda si, ti: conv.sessions[si][ti]["speaker"]

    def rendered(si, ti):
        return _spliced_turn_text(raw(si, ti), by_slot.get((si, ti), []))

    context = []
    si, ti = session_idx, turn_idx
    for _ in range(GATE_CONTEXT_TURNS):
        if ti > 0:
            ti -= 1
        elif benchmark == "locomo" and si > 0:
            si -= 1
            ti = n_turns(si) - 1
        else:
            break
        context.append(f"{speaker_of(si, ti)}: {rendered(si, ti)}")
    context.reverse()
    return rendered(session_idx, turn_idx), context


def build_gate_items(benchmark: str, rows: list[dict] | None = None,
                     ctx: RenderContext | None = None) -> list[dict]:
    """150 injected + 150 natural judge items for one benchmark.

    Injected: stratified over GATE_STRATA (seeded). Natural: turns from the
    same substrates/conversations hosting no injection, matched to the injected
    hosts' word-count deciles (v1 §10 packaging: full turn + up to 2 preceding
    turns, identical for both provenances)."""
    if rows is None:
        rows = _read_jsonl(MANIFEST_V2_PATH)
    if ctx is None:
        ctx = _render_context()
    bench_rows = [r for r in rows if r["benchmark"] == benchmark]
    rng = random.Random(f"{SEED_V2}:gate:{benchmark}")

    sampled: list[tuple[str, dict]] = []
    for name, types, n in GATE_STRATA:
        pool = sorted((r for r in bench_rows if r["inj_type"] in types),
                      key=lambda r: r["injection_id"])
        take = min(n, len(pool))
        sampled += [(name, r) for r in rng.sample(pool, take)]
    if len(sampled) < GATE_N:  # top up from unsampled rows, id-sorted
        chosen = {r["injection_id"] for _, r in sampled}
        rest = sorted((r for r in bench_rows if r["injection_id"] not in chosen),
                      key=lambda r: r["injection_id"])
        sampled += [("topup", r) for r in rng.sample(rest, GATE_N - len(sampled))]
    for _, r in sampled:
        if not r["text"]:
            raise ValueError(f"{r['injection_id']}: gate item requested before render")

    items = []
    injected_slots: set[tuple[str, int, int]] = set()
    for r in bench_rows:
        p = r["placement"]
        injected_slots.add((r["question_id"], p["session_idx"], p["turn_idx"]))
    for stratum, r in sampled:
        p = r["placement"]
        world = _world_rows(r, bench_rows)
        text, context = _gate_turn_and_context(
            r, benchmark, ctx, world, r["question_id"],
            p["session_idx"], p["turn_idx"])
        items.append({
            "benchmark": benchmark, "provenance": "injected", "stratum": stratum,
            "question_id": r["question_id"], "session_idx": p["session_idx"],
            "turn_idx": p["turn_idx"], "text": text, "context": context,
            "injection_ids": [x["injection_id"] for x in world
                              if x["placement"]["session_idx"] == p["session_idx"]
                              and x["placement"]["turn_idx"] == p["turn_idx"]],
            "word_count": len(text.split()),
        })

    # natural pool: same substrates, turns hosting no injection at all
    natural_pool: list[tuple[str, int, int, str]] = []
    if benchmark == "lme":
        qids = sorted({r["question_id"] for r in bench_rows})
        for qid in qids:
            sub = ctx.lme_subs[qid]
            for si, session in enumerate(sub.sessions):
                for ti, turn in enumerate(session):
                    if turn["role"] != "user":
                        continue
                    if (qid, si, ti) in injected_slots:
                        continue
                    natural_pool.append((qid, si, ti, turn["content"]))
    else:
        for sample_id in sorted({r["question_id"] for r in bench_rows}):
            conv = ctx.loco_subs[sample_id].conv
            for si, session in enumerate(conv.sessions):
                for ti, turn in enumerate(session):
                    if (sample_id, si, ti) in injected_slots:
                        continue
                    natural_pool.append((sample_id, si, ti, turn["text"]))

    counts = sorted(item["word_count"] for item in items)
    def decile(wc: int) -> int:
        import bisect
        return min(9, 10 * bisect.bisect_right(counts, wc) // max(1, len(counts)))
    pool_by_decile: dict[int, list] = {}
    for entry in natural_pool:
        pool_by_decile.setdefault(decile(len(entry[3].split())), []).append(entry)
    for d in pool_by_decile:
        pool_by_decile[d].sort()

    naturals = []
    for item in items:
        d = decile(item["word_count"])
        cand = None
        for dd in sorted(range(10), key=lambda x: (abs(x - d), x)):
            if pool_by_decile.get(dd):
                cand = pool_by_decile[dd].pop(
                    rng.randrange(len(pool_by_decile[dd])))
                break
        if cand is None:
            raise RuntimeError("natural pool exhausted")
        qid, si, ti, text = cand
        _, context = _gate_turn_and_context(
            None, benchmark, ctx, [], qid, si, ti)
        naturals.append({
            "benchmark": benchmark, "provenance": "natural", "stratum": None,
            "question_id": qid, "session_idx": si, "turn_idx": ti,
            "text": text, "context": context, "injection_ids": [],
            "word_count": len(text.split()),
        })

    all_items = items + naturals
    rng.shuffle(all_items)
    for i, item in enumerate(all_items):
        item["item_id"] = f"gate-{benchmark}-{i:03d}"
    return all_items


# --------------------------------------------------------------------------- #
# lint
# --------------------------------------------------------------------------- #

def _session_content_hash(turns: list[dict], text_key: str) -> str:
    return hashlib.sha256("\n".join(t[text_key] for t in turns).encode()).hexdigest()[:16]


def lint_v2() -> None:
    rows = _read_jsonl(MANIFEST_V2_PATH)
    keys = _read_jsonl(KEYS_V2_PATH)
    lme_subs, ku_list, aux_list, shadow = load_lme_substrates()
    loco_subs = load_locomo_substrates()
    errors: list[str] = []
    flags: list[str] = []

    by_bench = {b: [r for r in rows if r["benchmark"] == b] for b in BENCHMARKS}
    if set(r["benchmark"] for r in rows) - set(BENCHMARKS):
        errors.append("unknown benchmark value present")

    # budgets
    n = {b: {"spec": sum(r["inj_type"] in SPEC_TYPES for r in by_bench[b]),
             "corr": sum(r["inj_type"] in CORRECTION_TYPES for r in by_bench[b]),
             "stage": sum(r["inj_type"] == "t3_staging" for r in by_bench[b]),
             "var": sum(r["inj_type"] == "t3_variant" for r in by_bench[b])}
         for b in BENCHMARKS}
    if (n["lme"]["spec"], n["lme"]["corr"]) != (LME_SPEC_BUDGET, LME_CORR_BUDGET):
        errors.append(f"lme budget {n['lme']['spec']}+{n['lme']['corr']} != 240+120")
    if (n["locomo"]["spec"], n["locomo"]["corr"]) != (120, 120):
        errors.append(f"locomo budget {n['locomo']['spec']}+{n['locomo']['corr']} != 120+120")
    for b in BENCHMARKS:
        if n[b]["stage"] != T3_CASES_PER_BENCHMARK or n[b]["var"] != 4 * T3_CASES_PER_BENCHMARK:
            errors.append(f"{b} t3 counts {n[b]['stage']}/{n[b]['var']} != 50/200")
        n_soft = sum(r["inj_type"] == "t3_variant"
                     and r["t3_variant_kind"] == "invalidate"
                     and SOFT_INVALIDATE_TAG in r["edge_case_tags"]
                     for r in by_bench[b])
        if n_soft != N_SOFT_INVALIDATE:
            errors.append(f"{b}: {n_soft} soft invalidates != {N_SOFT_INVALIDATE}")

    # FM-16: t3_variant construction metadata must follow the corrected
    # contract->label mapping; rendered third-party distractor_a rows must
    # name their third party (expected_subject).
    for r in rows:
        if r["inj_type"] != "t3_variant":
            continue
        want = t3_construction_label(r["t3_variant_kind"], r["edge_case_tags"])
        if r["construction_label"] != want:
            errors.append(f"{r['injection_id']}: construction_label "
                          f"{r['construction_label']} != {want} (FM-16 mapping)")
        if is_third_party_distractor(r) and r["text"] and not r.get("expected_subject"):
            errors.append(f"{r['injection_id']}: third-party distractor_a "
                          "without expected_subject")

    if sum(r["inj_type"] == "cgold_correct" for r in rows) > CGOLD_CAP_V2:
        errors.append("cgold global cap exceeded")
    cgold_per: dict[str, int] = {}
    for r in rows:
        if r["inj_type"] == "cgold_correct":
            cgold_per[r["question_id"]] = cgold_per.get(r["question_id"], 0) + 1
    for q, c in sorted(cgold_per.items()):
        if c > CGOLD_PER_SUBSTRATE_CAP:
            errors.append(f"{q}: {c} cgold rows (max {CGOLD_PER_SUBSTRATE_CAP})")

    # shared-session discipline: shadow members must host nothing; no session
    # content hash may be perturbed under two question_ids.
    for shadow_q in shadow:
        if any(r["question_id"] == shadow_q for r in rows):
            errors.append(f"shadow pair member {shadow_q} hosts injections")
    hash_owners: dict[str, set[str]] = {}
    for r in by_bench["lme"]:
        sub = lme_subs[r["question_id"]]
        si = r["placement"]["session_idx"]
        h = _session_content_hash(sub.sessions[si], "content")
        hash_owners.setdefault(h, set()).add(r["question_id"])
    for h, owners in hash_owners.items():
        if len(owners) > 1:
            errors.append(f"session content hash {h} perturbed by {sorted(owners)}")

    # placement geometry
    for r in by_bench["lme"]:
        sub = lme_subs.get(r["question_id"])
        if sub is None:
            errors.append(f"{r['injection_id']}: unknown lme substrate")
            continue
        p = r["placement"]
        turn = sub.sessions[p["session_idx"]][p["turn_idx"]]
        if turn["role"] != "user":
            errors.append(f"{r['injection_id']}: non-user host turn")
        if p["turn_idx"] in sub.evidence.get(p["session_idx"], set()):
            errors.append(f"{r['injection_id']}: placed on an evidence turn")
        allowed = set(sub.hosts_primary.get(p["session_idx"], []))
        fb = set(sub.hosts_fallback.get(p["session_idx"], []))
        if p["turn_idx"] not in allowed:
            tag = sub.fallback_tag_name
            if p["turn_idx"] in fb:
                if tag and tag not in r["edge_case_tags"]:
                    errors.append(f"{r['injection_id']}: fallback slot without {tag} tag")
            else:
                errors.append(f"{r['injection_id']}: slot outside audited host set")
        if (r["inj_type"] == "t3_variant"
                and _user_turns_after(sub, p["session_idx"], p["turn_idx"])
                < T3_SENSITIVITY_WINDOW):
            errors.append(f"{r['injection_id']}: no t3 sensitivity window")
    for r in by_bench["locomo"]:
        sub = loco_subs.get(r["question_id"])
        if sub is None:
            errors.append(f"{r['injection_id']}: unknown conversation")
            continue
        p = r["placement"]
        slot = (p["session_idx"], p["turn_idx"])
        turn = sub.conv.sessions[slot[0]][slot[1]]
        if slot in sub.protected:
            errors.append(f"{r['injection_id']}: protected LoCoMo turn")
        if turn["speaker"] != r.get("speaker"):
            errors.append(f"{r['injection_id']}: speaker mismatch")
        if r["entity"] != str(r.get("speaker", "")).lower():
            errors.append(f"{r['injection_id']}: entity != speaker")
        if r.get("dia_id") != sub.dia_id(*slot):
            errors.append(f"{r['injection_id']}: dia_id mismatch")
        if (r["inj_type"] == "t3_variant"
                and len(sub.conv.sessions[slot[0]]) - 1 - slot[1] < T3_SENSITIVITY_WINDOW):
            errors.append(f"{r['injection_id']}: no t3 sensitivity window")

    # pair chronology
    by_id = {r["injection_id"]: r for r in rows}
    for r in rows:
        if not r["paired_with"]:
            continue
        est = by_id[r["paired_with"]]
        if r["inj_type"] == "t3_variant":
            continue  # staging/variant chronology asserted at plan
        if r["benchmark"] == "lme":
            sub = lme_subs[r["question_id"]]
            ek = sub.chrono_key(est["placement"]["session_idx"], est["placement"]["turn_idx"])
            dk = sub.chrono_key(r["placement"]["session_idx"], r["placement"]["turn_idx"])
        else:
            ek = (est["placement"]["session_idx"], est["placement"]["turn_idx"])
            dk = (r["placement"]["session_idx"], r["placement"]["turn_idx"])
        if not ek < dk:
            errors.append(f"{r['injection_id']}: establish not chronologically before")

    # per-turn load
    per_turn: dict[tuple, list[dict]] = {}
    for r in rows:
        if r["inj_type"].startswith("t3"):
            continue
        p = r["placement"]
        per_turn.setdefault((r["benchmark"], r["question_id"],
                             p["session_idx"], p["turn_idx"]), []).append(r)
    for slot, rs in per_turn.items():
        if slot[0] == "locomo" and len(rs) > 1:
            errors.append(f"locomo per-turn load >1 at {slot}")
        if len(rs) > 2:
            errors.append(f"per-turn load >2 at {slot}")
        if slot[0] == "lme" and len(rs) == 2:
            a, b = rs
            if {a["placement"]["position"], b["placement"]["position"]} != {OPEN, END}:
                errors.append(f"doubled turn {slot} positions not open+end")
            if (a["inj_type"] == b["inj_type"]
                    or (a["entity"], a["attribute"]) == (b["entity"], b["attribute"])):
                errors.append(f"doubled turn {slot} same type or key")

    # rendered-text checks (skipped rows still unrendered)
    transcripts = {}
    for r in rows:
        if not r["text"]:
            continue
        low = r["text"].lower()
        for bl in BLACKLIST + ROUND2_TELL_BLACKLIST:
            if bl.lower() in low:
                errors.append(f"{r['injection_id']}: blacklist phrase in text")
        if r["value"] and _norm(str(r["value"])) not in _norm(r["text"]):
            errors.append(f"{r['injection_id']}: value missing from text")
        key = (r["benchmark"], r["question_id"])
        if key not in transcripts:
            transcripts[key] = (lme_subs[r["question_id"]] if r["benchmark"] == "lme"
                                else loco_subs[r["question_id"]])
        if (r["value"] and r["inj_type"] != "t3_variant"
                and not novelty_ok(str(r["value"]), transcripts[key], [])):
            errors.append(f"{r['injection_id']}: value found in source transcript")
        for pat in RELATIVE_TIME_PATTERNS:
            if re.search(pat, low):
                flags.append(f"{r['injection_id']}: relative-time phrase {pat!r}")
    for b in BENCHMARKS:
        rendered_corr = [r for r in by_bench[b]
                         if r["inj_type"] in CORRECTION_TYPES and r["text"]]
        with_marker = sum(any(m in r["text"].lower() for m in MARKER_WORDS)
                          for r in rendered_corr)
        if with_marker > MARKER_FRACTION_CAP * 120:
            errors.append(f"{b}: marker cap exceeded ({with_marker})")

    # FM-4: a probe question never contains an injected value string FOR ITS
    # OWN KEY (scoped per-key: a commute-mode value 'bike' appearing inside the
    # bike_lock_brand probe text leaks nothing about either key's answer —
    # the global cross-product check false-positived on exactly that).
    values_by_key: dict[tuple, list[str]] = {}
    for r in rows:
        if r["value"] and len(str(r["value"])) >= 4:
            values_by_key.setdefault(
                (r["question_id"], r["entity"], r["attribute"]), []).append(str(r["value"]))
    for k in keys:
        pq = k["probe_question"].lower()
        kk = (k.get("question_id", k.get("substrate_qid")), k.get("entity", "user"), k["attribute"])
        for v in values_by_key.get(kk, []):
            if v.lower() in pq:
                errors.append(f"probe question {k['attribute']} leaks value {v!r}")

    for f in flags:
        print("FLAG", f)
    if errors:
        for e in errors:
            print("ERROR", e)
        raise SystemExit(1)
    print(f"lint OK: {len(rows)} rows ({len(by_bench['lme'])} lme + "
          f"{len(by_bench['locomo'])} locomo), {len(flags)} manual-audit flags")


# --------------------------------------------------------------------------- #
# FM-16 relabel pass (metadata-only; texts asserted byte-identical)
# --------------------------------------------------------------------------- #

def relabel_fm16() -> None:
    """One-shot manifest-wide t3_variant metadata relabel (FM-16, 2026-07-04).

    The Test-0 LoCoMo calibration gate caught the injector's defective
    contract->metadata mapping (see t3_construction_label). This pass rewrites
    ONLY ``construction_label`` (both benchmarks' t3_variant rows) and adds
    ``expected_subject`` to third-party-completion distractor_a rows. Texts —
    and every other field — are asserted byte-identical before/after; rows
    whose expected_subject is underivable are reported, never guessed.
    """
    raw_before = MANIFEST_V2_PATH.read_bytes()
    rows = _read_jsonl(MANIFEST_V2_PATH)
    before = copy.deepcopy(rows)
    transitions: dict[tuple, int] = {}
    relabeled: dict[tuple[str, str], int] = {}
    underivable: list[str] = []
    for r in rows:
        if r["inj_type"] != "t3_variant":
            continue
        kind = r["t3_variant_kind"]
        changed = False
        new_label = t3_construction_label(kind, r["edge_case_tags"])
        if new_label != r["construction_label"]:
            transitions[(r["benchmark"], kind, r["construction_label"],
                         new_label)] = transitions.get(
                (r["benchmark"], kind, r["construction_label"], new_label), 0) + 1
            r["construction_label"] = new_label
            changed = True
        if is_third_party_distractor(r) and r.get("expected_subject") is None:
            # never overwrites a non-null (possibly manual) mapping
            subject = derive_expected_subject(r["entity"], r["text"] or "")
            if subject is None:
                underivable.append(r["injection_id"])
            r["expected_subject"] = subject
            changed = True
        if changed:
            k = (r["benchmark"], kind)
            relabeled[k] = relabeled.get(k, 0) + 1

    metadata_fields = ("construction_label", "expected_subject")
    for old, new in zip(before, rows):
        assert old["text"] == new["text"], old["injection_id"]
        o = {k: v for k, v in old.items() if k not in metadata_fields}
        n = {k: v for k, v in new.items() if k not in metadata_fields}
        assert o == n, f"non-metadata field changed on {old['injection_id']}"

    _write_jsonl(MANIFEST_V2_PATH, rows)

    # post-write byte-level verification: text fields identical, and rows the
    # pass did not touch are byte-identical lines.
    after_lines = MANIFEST_V2_PATH.read_text().split("\n")
    before_lines = raw_before.decode().split("\n")
    assert len(after_lines) == len(before_lines)
    reparsed = _read_jsonl(MANIFEST_V2_PATH)
    changed_lines = 0
    for old, new_row, bline, aline in zip(before, reparsed, before_lines, after_lines):
        assert old["text"] == new_row["text"], old["injection_id"]
        if bline != aline:
            changed_lines += 1
            assert new_row["inj_type"] == "t3_variant", old["injection_id"]
    print(f"relabel-fm16: {sum(relabeled.values())} rows updated "
          f"({changed_lines} manifest lines changed)")
    for (bench, kind), n in sorted(relabeled.items()):
        print(f"  {bench}/{kind}: {n}")
    for (bench, kind, frm, to), n in sorted(transitions.items()):
        print(f"  label {bench}/{kind}: {frm} -> {to} ({n})")
    if underivable:
        print("EXPECTED_SUBJECT UNDERIVABLE (needs manual mapping, do not guess):")
        for iid in underivable:
            print(f"  {iid}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="v2 injection manifest tooling (role W4)")
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("plan")
    rp = sp.add_parser("render")
    rp.add_argument("--only", help="comma-separated injection_ids")
    rp.add_argument("--benchmark", choices=BENCHMARKS)
    sp.add_parser("lint")
    sp.add_parser("relabel-fm16")
    gp = sp.add_parser("gate-items")
    gp.add_argument("--benchmark", choices=BENCHMARKS)
    args = ap.parse_args()
    if args.cmd == "plan":
        rows, registry = plan_v2()
        _write_jsonl(MANIFEST_V2_PATH, rows)
        _write_jsonl(KEYS_V2_PATH, registry)
        _write_jsonl(AUX_SCAN_PATH, scan_log_rows())
        for b in BENCHMARKS:
            br = [r for r in rows if r["benchmark"] == b]
            n_spec = sum(r["inj_type"] in SPEC_TYPES for r in br)
            n_corr = sum(r["inj_type"] in CORRECTION_TYPES for r in br)
            print(f"plan[{b}]: {len(br)} rows ({n_spec} spec + {n_corr} corr "
                  f"budget injections)")
        print(f"plan: {len(rows)} rows total, {len(registry)} aux keys "
              f"-> {MANIFEST_V2_PATH}")
    elif args.cmd == "render":
        only = set(args.only.split(",")) if args.only else None
        asyncio.run(render_v2(only, args.benchmark))
    elif args.cmd == "lint":
        lint_v2()
    elif args.cmd == "relabel-fm16":
        relabel_fm16()
    elif args.cmd == "gate-items":
        benchmarks = [args.benchmark] if args.benchmark else list(BENCHMARKS)
        for b in benchmarks:
            items = build_gate_items(b)
            _write_jsonl(GATE_ITEMS_PATH[b], items)
            n_inj = sum(i["provenance"] == "injected" for i in items)
            print(f"gate-items[{b}]: {len(items)} items ({n_inj} injected) "
                  f"-> {GATE_ITEMS_PATH[b]}")


if __name__ == "__main__":
    main()
