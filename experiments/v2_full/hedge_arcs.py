"""hedge_arcs.py — Protocol v2.4 M4: human-verified synthetic at-risk hedge
arcs (protocol/CHANGELOG.md, "Protocol v2.4" M4 entry + the 2026-07-08
advisory-budget amendment; research_notes/dataset_scout_v24.md verdict).

M4 basis (dataset scout): NO published dataset carries the hedge->resolution
shape, so the arcs are SYNTHETIC, generated into existing benchmark
conversations and gated on a human (owner) realism/validity review of every
arc — the sole human step. Two scout adoptions are wired in:
  * Conversation Chronicles (HF jihyoung/ConversationChronicles, CC BY 4.0)
    as a PHRASING REFERENCE ONLY: a slice is downloaded to data/cc/ and
    ~200 hedge-candidate turns are mined for hedge-cue-family frequencies,
    length and register statistics. NO CC text is ever copied into stimuli —
    generation consumes only the aggregate statistics (hedge-strength
    sampling weights + the render length band), and the realized comparison
    is reported (generation_summary.json `calibration_comparison`).
  * PersonaMem-style persona coherence: every arc is a timestamped
    hedge->resolution event pair with an explicit reason structure (the
    plan call's content_plan), persona-fit checked against the host
    transcript (aux-attribute collision scan, style exemplars from the same
    speaker's real turns).

ARC SHAPE. One arc = ONE unused host substrate + one aux-style key +
  hedge turn        (contract SPECULATION, construction label `speculative`,
                     hedge_strength sampled from CC-calibrated weights)
  resolution turn   (chronologically later session), balanced 3 ways:
    resolve_true -> T3_PROMOTE          label completed_event  status resolved_true
    abandoned    -> T3_INVALIDATE       label completed_event  status abandoned
    open         -> T3_DISTRACTOR_CONTINUED  label speculative status unresolved
Balanced ~20/20/20 over the default 60 arcs (prereg target 40-60, balanced).

HOSTS are drawn from UNUSED LME/LoCoMo substrates: a question is "used" when
it (or any question sharing a haystack session with it) hosts any prior
artifact — v1 injections / key_schema.json, the 127 v2 substrates + v2
manifest, the v2.2 pair substrates / freeze-list sources / expansion-key
hosts, the natural-selection bought questions, or ANY harvested natural-hedge
candidate row (strict reading: no organic-hedge-bearing substrate hosts a
synthetic arc — the prereg's "never mixed with organic rows"). All 10 LoCoMo
conversations carry v2 injections, so the LoCoMo unused pool is EMPTY and the
arcs are LME-only (logged in host_scout.json and flagged, not silent).

PLACEMENT follows the frozen injector spec (protocol/specs/injector_spec.md):
  * protected turns = oracle `has_answer` evidence turns plus the nearest
    user turn on each side (§2.1 adjacency buffer); evidence never hosts.
  * host slots = unprotected user turns (>= 10 words so the aside stays an
    aside, FM-1); per-turn load 1; one arc per substrate (clean clusters).
  * hedge in the chronologically EARLIEST slot-bearing session; resolution in
    a seeded-chosen strictly-LATER-dated session (t3 staging/variant rule,
    §2.5); slot + sentence-boundary position by seeded uniform draw (§2.2b),
    seed 20260708, draws consumed in arc order, seed_draw_index recorded.
  * keys registered with value-free AUX_POOL probe questions BEFORE any value
    is sampled (FM-4), collision-scanned against the host transcript.
RENDER is the frozen two-call pipeline (prompts/injector_plan.txt +
injector_render.txt, byte-untouched): Call 1 plans content + samples values
under the code-side novelty check (inject.novelty_ok); Call 2 renders in the
round-2 bridged-integration style with same-substrate exemplars, the frozen
blacklist and inject._render_ok asserts. Payload-level adaptation only (the
CC-calibrated length band rides the existing length_target_words field).

REVIEW GATE (the prereg's sole human step): `review-pack` writes
results/v2.4/m4/review_pack.md (one section per arc, accept/reject
checkboxes) + owner_review_template.json. Arcs are UNUSABLE until reviewed:
every generated row carries status=pending_owner_review / usable=false, and
`pool` REFUSES to emit the final pool unless results/v2.4/m4/owner_review.json
exists and covers every arc. Accepted arcs land in arc_pool.jsonl as a
separately-labeled stratum (row_kind m4_synthetic_arc) with splice-ready
manifest rows (inject.build_perturbed_substrate-compatible), never mixed with
organic rows.

BUDGET (2026-07-08 amendment): stage budgets are ADVISORY. Every fresh LLM
call is recorded under stage "m4" of results/v2.4/budget.json via
forecast_v22.SharedCallBudget; the ONLY hard stop is the 120,000-call total
envelope. An m4 spend above the advisory forecast (3,000 at generation
time; raised to 12,000 by the 2026-07-09 M4-EVAL preregistration) prints an
ADVISORY OVERRUN note and continues.

CLI (repo root, pilot/.venv/bin/python experiments/v2_full/hedge_arcs.py ...):
  calibrate  [--episodes 300] [--refresh]   CC slice -> data/cc/ + mining ->
                                            results/v2.4/m4/cc_calibration.json
                                            (regex mining, zero LLM calls)
  generate   [--n 60] [--limit K] [--fresh-cap N]
                                            plan (frozen before render) ->
                                            arc_plan.jsonl + m4_keys.jsonl,
                                            then the two-call render ->
                                            arcs.jsonl + generation_summary.json
  review-pack                               arcs.jsonl -> review_pack.md +
                                            owner_review_template.json
  pool                                      owner_review.json (REQUIRED) ->
                                            arc_pool.jsonl + pool_summary.json
  smoke      [--fresh-cap 8]                2 arcs end-to-end (calibrate if
                                            needed -> plan -> render first 2)
                                            into results/v2.4/m4/smoke/;
                                            spend recorded under stage m4
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import statistics
import sys
from functools import cache
from pathlib import Path

import _bootstrap  # noqa: F401  (R1 layout: puts src/ + experiments/ on sys.path)

import data
import forecast_v22
import inject
import llm
from inject import (
    AUX_POOL,
    BLACKLIST,
    HEDGE_STRENGTHS,
    MARKER_WORDS,
    POSITIONS,
    RELATIVE_TIME_PATTERNS,
    ROUND2_INTEGRATION_NOTE,
    ROUND2_TELL_BLACKLIST,
    Draws,
    _align_plan_value,
    _norm,
    _render_ok,
    make_row,
    novelty_ok,
    splice_turn,
)

from paths import DATA_DIR, GOLD_DIR, PROMPTS_DIR, RESULTS_DIR, ROOT

SEED_M4 = 20260708         # M4 plan seed (draws consumed in arc order, §2.2b)
N_ARCS_DEFAULT = 60        # prereg target 40-60; balanced 20/20/20
MIN_HOST_TURN_WORDS = 10   # host-slot floor so the aside stays an aside (FM-1)
STAGE = "m4"
ENVELOPE = 120_000         # 2026-07-08 amendment: total-only hard limit
#: forecast, NOT a stop (advisory amendment); raised 3,000 -> 12,000 by the
#: 2026-07-09 M4-EVAL preregistration (changelog).
M4_ADVISORY_ALLOCATION = 12_000

V24_DIR = RESULTS_DIR / "v2.4"
BUDGET_PATH = V24_DIR / "budget.json"
M4_DIR = V24_DIR / "m4"
SMOKE_DIR = M4_DIR / "smoke"
CC_DIR = DATA_DIR / "cc"
CC_SLICE_PATH = CC_DIR / "cc_slice.jsonl"
CC_README_PATH = CC_DIR / "README.md"
CALIBRATION_PATH = M4_DIR / "cc_calibration.json"
HOST_SCOUT_PATH = M4_DIR / "host_scout.json"
ARC_PLAN_PATH = M4_DIR / "arc_plan.jsonl"
KEYS_PATH = M4_DIR / "m4_keys.jsonl"
ARCS_PATH = M4_DIR / "arcs.jsonl"
GEN_SUMMARY_PATH = M4_DIR / "generation_summary.json"
REVIEW_PACK_PATH = M4_DIR / "review_pack.md"
REVIEW_TEMPLATE_PATH = M4_DIR / "owner_review_template.json"
OWNER_REVIEW_PATH = M4_DIR / "owner_review.json"
POOL_PATH = M4_DIR / "arc_pool.jsonl"
POOL_SUMMARY_PATH = M4_DIR / "pool_summary.json"

CC_DATASET = "jihyoung/ConversationChronicles"
CC_LICENSE = "CC BY 4.0"
CC_ROWS_URL = "https://datasets-server.huggingface.co/rows"
CC_EPISODES_DEFAULT = 300
CC_SESSION_FIELDS = ("first", "second", "third", "fourth", "fifth")
CC_CANDIDATE_CAP = 200     # "~200 hedge-candidate turns"
CC_MIN_WORDS, CC_MAX_WORDS = 5, 80

RESOLUTION_TYPES = ("resolve_true", "abandoned", "open")
RESOLUTION_CONTRACTS = {
    "resolve_true": "T3_PROMOTE",
    "abandoned": "T3_INVALIDATE",
    "open": "T3_DISTRACTOR_CONTINUED",
}
#: FM-16-corrected construction labels (inject_v2 mapping): hard abandonment
#: is a completed decision (rule 1.5f); continued deliberation stays
#: speculative (S2).
RESOLUTION_LABELS = {
    "resolve_true": "completed_event",
    "abandoned": "completed_event",
    "open": "speculative",
}
#: v2.3 resolution-annotation vocabulary (resolution_aware_rule5): the at-risk
#: (contamination) stratum is unresolved+abandoned; resolved_true is the
#: correct-tracking stratum.
GOLD_STATUS = {
    "resolve_true": "resolved_true",
    "abandoned": "abandoned",
    "open": "unresolved",
}

#: Hedge-cue families, 1:1 with the frozen HEDGE_STRENGTHS taxonomy
#: (injector_spec §5.1). Detection order: scheduled_but_future (most
#: specific) -> deliberating -> intending.
HEDGE_CUE_FAMILIES: dict[str, tuple[str, ...]] = {
    "scheduled_but_future": (
        r"\bscheduled (?:for|to)\b",
        r"\bstarting next (?:week|month|year)\b",
        r"\babout to\b",
        r"\bnext (?:week|month|year|semester)\b[^.!?]*"
        r"\b(?:i|we)(?:'m| am|'re| are)? ?(?:going to|gonna|start|switch|mov)\w*",
        r"\b(?:i|we)(?:'m| am|'re| are) (?:going to|gonna)\b[^.!?]*"
        r"\b(?:next (?:week|month|year)|soon|tomorrow|this weekend)\b",
        r"\bonce\b[^.!?]*\b(?:i|we)(?:'ll| will)\b",
    ),
    "deliberating": (
        r"\btorn between\b",
        r"\bthinking (?:about|of)\b",
        r"\bdebating\b",
        r"\bconsidering\b",
        r"\bcan'?t (?:decide|make up my mind)\b",
        r"\bweighing\b",
        r"\bmulling\b",
        r"\btoying with\b",
        r"\bon the fence\b",
        r"\bnot sure (?:if|whether)\b",
    ),
    "intending": (
        r"\bplan(?:ning)? (?:to|on)\b",
        r"\bhoping to\b",
        r"\baiming to\b",
        r"\b(?:i|we) might\b",
        r"\b(?:i|we) may\b",
        r"\b(?:i|we)'ll probably\b",
        r"\bprobably (?:going to|gonna)\b",
        r"\bleaning towards?\b",
        r"\bgoing to try\b",
        r"\bthink (?:i|we)'ll\b",
    ),
}
FAMILY_ORDER = ("scheduled_but_future", "deliberating", "intending")
assert set(FAMILY_ORDER) == set(HEDGE_CUE_FAMILIES) == set(HEDGE_STRENGTHS)

_FIRST_PERSON_RE = re.compile(r"\b(?:i|we)\b", re.I)
_CONTRACTION_RE = re.compile(r"\w'\w")

FROZEN_LENGTH_LO, FROZEN_LENGTH_HI = 15, 45   # injector render band (frozen)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def _write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def _percentile(values: list[float], q: float) -> float:
    """Deterministic nearest-rank percentile (no numpy dependency)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, round(q * (len(ordered) - 1))))
    return float(ordered[idx])


# ---------------------------------------------------------------------------
# Budget (advisory stages, hard 120k total envelope)
# ---------------------------------------------------------------------------

def load_v24_budget(path: Path = BUDGET_PATH) -> dict:
    if not path.exists():
        raise SystemExit(
            f"missing v2.4 combined budget file {path} — run "
            "`mapper_fix.py budget` first (M1 owns file creation)")
    budget = json.loads(path.read_text())
    if budget["envelope"] != ENVELOPE:
        raise SystemExit(
            f"{path}: envelope {budget['envelope']} != amended {ENVELOPE} "
            "(2026-07-08 owner amendment)")
    if STAGE not in budget["spent"]:
        raise SystemExit(f"{path}: no '{STAGE}' stage in spent counters")
    return budget


def m4_guard(fresh_cap: int | None = None,
             path: Path = BUDGET_PATH) -> forecast_v22.SharedCallBudget:
    """The M4 spend recorder: SharedCallBudget stage 'm4' over the v2.4
    budget file. ADVISORY stages (2026-07-08 amendment): the stage
    allocation is NEVER used as a cap — the guard's only implicit limit is
    the remaining 120k total envelope; `fresh_cap` is an explicit per-run
    local cap (smoke discipline), not a stage stop."""
    load_v24_budget(path)  # existence + envelope/stage validation
    return forecast_v22.SharedCallBudget(STAGE, path=path, local_cap=fresh_cap)


def advisory_note(path: Path = BUDGET_PATH) -> dict:
    budget = load_v24_budget(path)
    spent = budget["spent"][STAGE]
    return {
        "stage": STAGE,
        "spent": spent,
        "advisory_allocation": M4_ADVISORY_ALLOCATION,
        "advisory_overrun": spent > M4_ADVISORY_ALLOCATION,
        "envelope": budget["envelope"],
        "envelope_spent_total": sum(budget["spent"].values()),
        "mode": "ADVISORY stages / hard total envelope (2026-07-08 amendment)",
    }


# ---------------------------------------------------------------------------
# calibrate — Conversation Chronicles slice + hedge-cue mining (zero LLM)
# ---------------------------------------------------------------------------

def download_cc_slice(n_episodes: int = CC_EPISODES_DEFAULT,
                      refresh: bool = False) -> list[dict]:
    """Download the first `n_episodes` train episodes of Conversation
    Chronicles via the HF datasets-server rows API into data/cc/ (CC BY 4.0;
    attribution written alongside). Reuses the existing slice unless
    `refresh`. Deterministic slice: offset 0, fixed length."""
    if CC_SLICE_PATH.exists() and not refresh:
        return _read_jsonl(CC_SLICE_PATH)
    import httpx

    rows: list[dict] = []
    offset = 0
    while offset < n_episodes:
        length = min(100, n_episodes - offset)
        resp = httpx.get(CC_ROWS_URL, params={
            "dataset": CC_DATASET, "config": "default", "split": "train",
            "offset": offset, "length": length}, timeout=60)
        resp.raise_for_status()
        for r in resp.json()["rows"]:
            row = r["row"]
            sessions = []
            for f in CC_SESSION_FIELDS:
                dialogue = row.get(f"{f}_session_dialogue") or []
                speakers = row.get(f"{f}_session_speakers") or []
                sessions.append([
                    {"speaker": (speakers[i] if i < len(speakers) else ""),
                     "text": t}
                    for i, t in enumerate(dialogue)])
            rows.append({
                "dataID": row["dataID"],
                "relationship": row.get("relationship"),
                "time_interval": row.get("time_interval"),
                "sessions": sessions,
            })
        offset += length
    _write_jsonl(CC_SLICE_PATH, rows)
    CC_README_PATH.parent.mkdir(parents=True, exist_ok=True)
    CC_README_PATH.write_text(
        "# Conversation Chronicles slice (v2.4 M4 calibration reference)\n\n"
        f"Source: https://huggingface.co/datasets/{CC_DATASET} "
        f"(license {CC_LICENSE}).\n"
        f"Slice: train split, offset 0, first {n_episodes} episodes, fetched "
        "via the HF datasets-server rows API by "
        "experiments/v2_full/hedge_arcs.py calibrate.\n\n"
        "USE: phrasing reference corpus ONLY (hedge-cue-family frequencies, "
        "length, register). Per the M4 preregistration, no text from this "
        "slice is ever copied into generated stimuli.\n")
    return rows


def detect_cue_family(text: str) -> tuple[str | None, str | None]:
    """(family, matched cue regex) for a turn, or (None, None). Fixed
    detection order FAMILY_ORDER (most specific first)."""
    for family in FAMILY_ORDER:
        for pat in HEDGE_CUE_FAMILIES[family]:
            if re.search(pat, text, re.I):
                return family, pat
    return None, None


def mine_hedge_candidates(slice_rows: list[dict],
                          cap: int = CC_CANDIDATE_CAP) -> tuple[list[dict], int]:
    """Scan the CC slice in deterministic order (episode, session, turn) for
    first-person hedge-candidate turns; stop at `cap`. Returns
    (candidates, n_turns_scanned)."""
    out: list[dict] = []
    scanned = 0
    for ep in slice_rows:
        for si, session in enumerate(ep["sessions"]):
            for ti, turn in enumerate(session):
                scanned += 1
                if len(out) >= cap:
                    continue
                text = turn["text"].strip()
                n_words = len(text.split())
                if not (CC_MIN_WORDS <= n_words <= CC_MAX_WORDS):
                    continue
                if not _FIRST_PERSON_RE.search(text):
                    continue
                family, cue = detect_cue_family(text)
                if family is None:
                    continue
                out.append({
                    "dataID": ep["dataID"],
                    "session_idx": si,
                    "turn_idx": ti,
                    "family": family,
                    "cue": cue,
                    "n_words": n_words,
                    "text": text,
                })
    return out, scanned


def text_register_stats(texts: list[str]) -> dict:
    """Register statistics of a text population (CC reference and generated
    hedges are measured with the SAME function so the comparison is fair)."""
    n = len(texts)
    if n == 0:
        return {"n": 0}
    lengths = [float(len(t.split())) for t in texts]
    return {
        "n": n,
        "length_words": {
            "mean": round(statistics.fmean(lengths), 2),
            "median": _percentile(lengths, 0.5),
            "p10": _percentile(lengths, 0.10),
            "p25": _percentile(lengths, 0.25),
            "p75": _percentile(lengths, 0.75),
            "p90": _percentile(lengths, 0.90),
        },
        "contraction_rate": round(
            sum(1 for t in texts if _CONTRACTION_RE.search(t)) / n, 3),
        "exclamation_rate": round(
            sum(1 for t in texts if "!" in t) / n, 3),
        "question_rate": round(
            sum(1 for t in texts if "?" in t) / n, 3),
        "first_person_rate": round(
            sum(1 for t in texts if _FIRST_PERSON_RE.search(t)) / n, 3),
    }


def calibration_from_candidates(cands: list[dict], scanned: int,
                                n_episodes: int) -> dict:
    fam_counts = {f: sum(1 for c in cands if c["family"] == f)
                  for f in FAMILY_ORDER}
    total = max(1, len(cands))
    return {
        "id": "V2.4-M4-CC-CALIBRATION",
        "reference": {
            "dataset": CC_DATASET, "license": CC_LICENSE,
            "split": "train", "offset": 0, "n_episodes": n_episodes,
            "use": ("phrasing reference ONLY — aggregate statistics feed "
                    "hedge-strength weights and the render length band; no "
                    "CC text enters stimuli (M4 prereg)"),
        },
        "n_turns_scanned": scanned,
        "n_candidates": len(cands),
        "candidate_cap": CC_CANDIDATE_CAP,
        "family_counts": fam_counts,
        "family_freq": {f: round(fam_counts[f] / total, 4)
                        for f in FAMILY_ORDER},
        "register": text_register_stats([c["text"] for c in cands]),
        "by_family_register": {
            f: text_register_stats([c["text"] for c in cands
                                    if c["family"] == f])
            for f in FAMILY_ORDER},
        "detection_order": list(FAMILY_ORDER),
        "cue_families": {f: list(p) for f, p in HEDGE_CUE_FAMILIES.items()},
    }


def cc_length_band(calib: dict) -> str:
    """The render length_target_words band, CC-calibrated WITHIN the frozen
    15-45 injector band (payload-level adaptation only; prompts untouched)."""
    lw = calib.get("register", {}).get("length_words")
    if not lw:
        return f"{FROZEN_LENGTH_LO}-{FROZEN_LENGTH_HI}"
    lo = int(max(FROZEN_LENGTH_LO, min(FROZEN_LENGTH_HI, lw["p25"])))
    hi = int(max(FROZEN_LENGTH_LO, min(FROZEN_LENGTH_HI, lw["p90"])))
    if hi <= lo:
        lo, hi = FROZEN_LENGTH_LO, FROZEN_LENGTH_HI
    return f"{lo}-{hi}"


def cc_strength_weights(calib: dict, floor: float = 0.10) -> dict[str, float]:
    """Hedge-strength sampling weights from the CC family frequencies, floored
    (every frozen taxonomy stratum stays sampled) and normalized."""
    freq = calib.get("family_freq") or {}
    raw = {f: max(floor, float(freq.get(f, 0.0))) for f in FAMILY_ORDER}
    z = sum(raw.values())
    return {f: raw[f] / z for f in FAMILY_ORDER}


def run_calibrate(n_episodes: int = CC_EPISODES_DEFAULT,
                  refresh: bool = False) -> dict:
    slice_rows = download_cc_slice(n_episodes, refresh=refresh)
    cands, scanned = mine_hedge_candidates(slice_rows)
    calib = calibration_from_candidates(cands, scanned, len(slice_rows))
    calib["derived"] = {
        "strength_weights": cc_strength_weights(calib),
        "length_band": cc_length_band(calib),
    }
    calib["slice_sha256"] = _sha256(CC_SLICE_PATH.read_text())
    _write_json(CALIBRATION_PATH, calib)
    return calib


def load_calibration() -> dict:
    if not CALIBRATION_PATH.exists():
        raise SystemExit(
            f"missing {CALIBRATION_PATH} — run `hedge_arcs.py calibrate` "
            "first (the M4 prereg's CC phrasing calibration precedes "
            "generation)")
    return json.loads(CALIBRATION_PATH.read_text())


# ---------------------------------------------------------------------------
# host scout — unused LME/LoCoMo substrates
# ---------------------------------------------------------------------------

V2_MANIFEST_PATH = RESULTS_DIR / "v2" / "injections" / "manifest.jsonl"
V1_MANIFEST_PATH = RESULTS_DIR / "injections" / "manifest.jsonl"
V22_EXPANSION_PATH = RESULTS_DIR / "v2.2" / "natural_stratum_expansion.jsonl"
V22_EXPANSION_KEYS_PATH = RESULTS_DIR / "v2.2" / "expansion_keys.jsonl"
NATURAL_SELECTION_PATH = RESULTS_DIR / "v2" / "natural_selection.json"
NATURAL_CANDIDATE_FILES = (
    GOLD_DIR / "natural_candidates.jsonl",
    GOLD_DIR / "natural_candidates_other.jsonl",
    GOLD_DIR / "natural_candidates_full.jsonl",
)


@cache
def used_question_ids() -> dict:
    """{'lme': frozenset, 'locomo': frozenset, 'provenance': {...}} — every
    substrate hosting any prior artifact, plus (LME) the session-sharing
    closure ("no session is perturbed twice", protocol v2 §2.1)."""
    prov: dict[str, list[str]] = {}

    def add(name: str, ids) -> set[str]:
        ids = set(ids)
        prov[name] = sorted(ids)
        return ids

    lme: set[str] = set()
    lme |= add("v1_manifest", (r["question_id"]
                               for r in _read_jsonl(V1_MANIFEST_PATH)))
    lme |= add("v1_key_schema", (e["question_id"] for e in json.loads(
        (GOLD_DIR / "key_schema.json").read_text())["substrates"]))
    ks2 = json.loads((GOLD_DIR / "key_schema_v2_lme.json").read_text())
    lme |= add("v2_ku_substrates", (e["question_id"] for e in ks2["substrates"]))
    lme |= add("v2_aux_substrates", (r["question_id"]
                                     for r in ks2["aux_injection_substrates"]))
    lme |= add("v2_pair_substrates", (e["question_id"] for e in
                                      ks2["natural_pair_keys"]["entries"]))
    v2_man = _read_jsonl(V2_MANIFEST_PATH)
    lme |= add("v2_manifest_lme", (r["question_id"] for r in v2_man
                                   if r["benchmark"] == "lme"))
    lme |= add("v22_freeze_list_sources", (r["source_question_id"]
                                           for r in _read_jsonl(V22_EXPANSION_PATH)
                                           if r.get("benchmark") == "lme"))
    exp_hosts: set[str] = set()
    for r in _read_jsonl(V22_EXPANSION_KEYS_PATH):
        if r.get("benchmark") == "lme":
            exp_hosts.add(r["question_id"])
            exp_hosts.update(r.get("host_question_ids", []))
    lme |= add("v22_expansion_key_hosts", exp_hosts)
    if NATURAL_SELECTION_PATH.exists():
        sel = json.loads(NATURAL_SELECTION_PATH.read_text())
        lme |= add("natural_selection_bought",
                   (b["question_id"] for b in sel["bought_questions"]))
    harvest: set[str] = set()
    for p in NATURAL_CANDIDATE_FILES:
        harvest.update(r["question_id"] for r in _read_jsonl(p)
                       if r.get("source") != "locomo")
    lme |= add("natural_hedge_harvest_rows", harvest)

    # session-sharing closure over the oracle
    oracle = data.load_oracle()
    by_id = {q["question_id"]: q for q in oracle}
    used_sessions: set[str] = set()
    for qid in lme:
        q = by_id.get(qid)
        if q:
            used_sessions.update(q["haystack_session_ids"])
    shares = {q["question_id"] for q in oracle
              if q["question_id"] not in lme
              and used_sessions & set(q["haystack_session_ids"])}
    lme |= add("session_sharing_closure", shares)

    locomo = add("v2_manifest_locomo", (r["question_id"] for r in v2_man
                                        if r["benchmark"] == "locomo"))
    return {"lme": frozenset(lme), "locomo": frozenset(locomo),
            "provenance": {k: len(v) for k, v in prov.items()}}


def host_geometry(question: dict) -> dict:
    """Placement geometry of one raw oracle question, per the frozen injector
    spec: protected = has_answer evidence turns + nearest user turn each side
    (§2.1); host slots = unprotected user turns >= MIN_HOST_TURN_WORDS."""
    sessions = question["haystack_sessions"]
    dates = question["haystack_dates"]
    protected: set[tuple[int, int]] = set()
    for si, session in enumerate(sessions):
        for ti, turn in enumerate(session):
            if not turn.get("has_answer"):
                continue
            protected.add((si, ti))
            for step in (-1, 1):           # nearest user turn on each side
                j = ti + step
                while 0 <= j < len(session):
                    if session[j]["role"] == "user":
                        protected.add((si, j))
                        break
                    j += step
    slots: dict[int, list[int]] = {}
    for si, session in enumerate(sessions):
        for ti, turn in enumerate(session):
            if turn["role"] != "user" or (si, ti) in protected:
                continue
            if len(turn["content"].split()) < MIN_HOST_TURN_WORDS:
                continue
            slots.setdefault(si, []).append(ti)
    return {
        "question_id": question["question_id"],
        "question_type": question["question_type"],
        "n_sessions": len(sessions),
        "dates": list(dates),
        "chrono": data.chronological_order(list(dates)),
        "protected": protected,
        "slots": slots,
    }


def _arc_sessions(geom: dict) -> tuple[int | None, list[int]]:
    """(hedge_session, resolution_session_candidates): the chronologically
    earliest slot-bearing session that has a strictly-LATER-dated
    slot-bearing session (t3 staging/variant chronology, injector spec §2.5).
    Returns (None, []) when the substrate cannot host an arc."""
    dated = [(data.parse_haystack_date(geom["dates"][si]), si)
             for si in geom["chrono"] if geom["slots"].get(si)]
    for i, (d_early, si_early) in enumerate(dated):
        later = [si for (d, si) in dated[i + 1:] if d > d_early]
        if later:
            return si_early, later
    return None, []


@cache
def eligible_hosts() -> tuple[dict, ...]:
    """Unused LME substrates that can host an arc, sorted by question_id.
    (The LoCoMo unused pool is empty — all 10 conversations carry v2
    injections; recorded by host_scout, flagged, arcs are LME-only.)"""
    used = used_question_ids()["lme"]
    out = []
    for q in data.load_oracle():
        if q["question_id"] in used:
            continue
        geom = host_geometry(q)
        hedge_si, later = _arc_sessions(geom)
        if hedge_si is None:
            continue
        out.append(geom)
    return tuple(sorted(out, key=lambda g: g["question_id"]))


def run_host_scout() -> dict:
    used = used_question_ids()
    oracle_ids = {q["question_id"] for q in data.load_oracle()}
    loco_all = sorted({r["question_id"] for r in _read_jsonl(V2_MANIFEST_PATH)
                       if r["benchmark"] == "locomo"})
    hosts = eligible_hosts()
    doc = {
        "id": "V2.4-M4-HOST-SCOUT",
        "rule": ("unused = hosts no prior artifact (v1/v2 injections, v2 "
                 "cells, v2.2 pair/freeze/expansion hosts, natural-selection "
                 "bought questions, ANY harvested natural-hedge candidate "
                 "row) and shares no haystack session with a used question"),
        "lme": {
            "oracle_questions": len(oracle_ids),
            "used": len(used["lme"] & oracle_ids),
            "unused": len(oracle_ids - used["lme"]),
            "arc_eligible_hosts": len(hosts),
            "eligibility": ("archive >= 2 sessions; a slot-bearing session "
                            "with a strictly-later-dated slot-bearing "
                            "session; host slots = unprotected user turns "
                            f">= {MIN_HOST_TURN_WORDS} words"),
            "used_provenance": used["provenance"],
        },
        "locomo": {
            "conversations": len(loco_all),
            "used": len(used["locomo"]),
            "unused": sorted(set(loco_all) - set(used["locomo"])),
            "flag": ("EMPTY unused pool — all 10 LoCoMo conversations carry "
                     "v2 injections; M4 arcs are LME-only (logged deviation, "
                     "not silent)"),
        },
    }
    _write_json(HOST_SCOUT_PATH, doc)
    return doc


# ---------------------------------------------------------------------------
# plan — deterministic arc placement (zero LLM; freezes before render)
# ---------------------------------------------------------------------------

def balanced_types(n: int) -> list[str]:
    """RESOLUTION_TYPES repeated to n, as balanced as arithmetic allows
    (n=60 -> exactly 20/20/20); remainder distributed in fixed type order."""
    base, rem = divmod(n, len(RESOLUTION_TYPES))
    out = []
    for i, t in enumerate(RESOLUTION_TYPES):
        out.extend([t] * (base + (1 if i < rem else 0)))
    return out


def _weighted_choice(draws: Draws, weights: dict[str, float]) -> str:
    """Seeded weighted draw through the shared Draws counter (§2.2b: every
    draw is reproducible from seed + draw index)."""
    draws.n += 1
    r = draws.rng.random()
    acc = 0.0
    for k in FAMILY_ORDER:
        acc += weights[k]
        if r <= acc:
            return k
    return FAMILY_ORDER[-1]


class _Transcript:
    """inject.novelty_ok adapter for a raw oracle question."""

    def __init__(self, question: dict):
        self._text = "\n".join(
            t["content"] for s in question["haystack_sessions"] for t in s)

    def transcript_text(self) -> str:
        return self._text


def _persona_fit_attributes(question: dict) -> list[str]:
    """AUX_POOL attributes whose collision keywords are absent from the host
    transcript (injector spec §5.3 no-collision / persona-fit scan)."""
    transcript = _Transcript(question).transcript_text().lower()
    return sorted(a for a, (_d, _p, kw) in AUX_POOL.items()
                  if not any(k in transcript for k in kw))


def plan_arcs(n_arcs: int = N_ARCS_DEFAULT,
              strength_weights: dict[str, float] | None = None,
              seed: int = SEED_M4) -> tuple[list[dict], list[dict]]:
    """The deterministic arc plan: (arc rows without values/text, key
    registry rows). Keys carry value-free AUX_POOL probe questions and are
    registered here — BEFORE any value exists (FM-4). One arc per unused
    host; hedge in the chronologically earliest slot-bearing session,
    resolution in a seeded strictly-later-dated session; slot/position by
    seeded uniform draw; resolution types balanced then seeded-shuffled."""
    weights = (dict(strength_weights) if strength_weights
               else {f: 1 / 3 for f in FAMILY_ORDER})
    assert abs(sum(weights.values()) - 1.0) < 1e-9, "weights must normalize"
    hosts = eligible_hosts()
    if len(hosts) < n_arcs:
        raise SystemExit(
            f"only {len(hosts)} arc-eligible unused hosts < n_arcs {n_arcs}")
    draws = Draws(seed)
    picked = sorted(draws.sample([g["question_id"] for g in hosts], n_arcs))
    types = balanced_types(n_arcs)
    draws.rng.shuffle(types)   # one shuffle; recorded via seed + draw order
    draws.n += 1

    by_qid = {g["question_id"]: g for g in hosts}
    oracle_by_id = {q["question_id"]: q for q in data.load_oracle()}
    arcs: list[dict] = []
    keys: list[dict] = []
    for i, qid in enumerate(picked):
        geom = by_qid[qid]
        question = oracle_by_id[qid]
        res_type = types[i]
        hedge_si, later = _arc_sessions(geom)
        res_si = draws.choice(later)
        hedge_ti = draws.choice(geom["slots"][hedge_si])
        res_ti = draws.choice(geom["slots"][res_si])
        hedge_pos = draws.choice(list(POSITIONS))
        res_pos = draws.choice(list(POSITIONS))
        strength = _weighted_choice(draws, weights)
        eligible_attrs = _persona_fit_attributes(question)
        if not eligible_attrs:
            raise RuntimeError(f"{qid}: aux pool exhausted (persona-fit scan)")
        attribute = draws.choice(eligible_attrs)
        desc, probe_q, kw = AUX_POOL[attribute]
        arc_id = f"M4-{qid}"
        keys.append({
            "arc_id": arc_id,
            "benchmark": "lme",
            "question_id": qid,
            "entity": "user",
            "attribute": attribute,
            "description": desc,
            "probe_question": probe_q,
            "collision_keywords": list(kw),
            "key_source": "m4_aux_style",
            "fm4_note": ("registered at plan time, before any value was "
                         "sampled (FM-4); probe question is the frozen "
                         "value-free AUX_POOL text"),
        })
        arcs.append({
            "arc_id": arc_id,
            "row_kind": "m4_synthetic_arc",
            "stratum": "m4_synthetic",
            "benchmark": "lme",
            "question_id": qid,
            "question_type": geom["question_type"],
            "resolution_type": res_type,
            "gold_resolution_status": GOLD_STATUS[res_type],
            "hedge_strength": strength,
            "key": {"entity": "user", "attribute": attribute,
                    "description": desc, "probe_question": probe_q,
                    "key_source": "m4_aux_style",
                    "key_ref": f"m4/{qid}/user/{attribute}"},
            "hedge": {
                "inj_type": "m4_hedge",
                "contract": "SPECULATION",
                "construction_label": "speculative",
                "session_idx": hedge_si, "turn_idx": hedge_ti,
                "position": hedge_pos, "date": geom["dates"][hedge_si],
            },
            "resolution": {
                "inj_type": "m4_resolution",
                "contract": RESOLUTION_CONTRACTS[res_type],
                "construction_label": RESOLUTION_LABELS[res_type],
                "t3_variant_kind": ("invalidate" if res_type == "abandoned"
                                    else None),
                "session_idx": res_si, "turn_idx": res_ti,
                "position": res_pos, "date": geom["dates"][res_si],
            },
            "pending_window_user_turns": _pending_window(
                geom, hedge_si, hedge_ti, res_si, res_ti),
            "seed": seed,
            "seed_draw_index": draws.n,
            "value": None,
            "hedge_text": None,
            "resolution_text": None,
            "status": "pending_owner_review",
            "usable": False,
        })
    return arcs, keys


def _pending_window(geom: dict, hedge_si: int, hedge_ti: int,
                    res_si: int, res_ti: int) -> int:
    """User turns strictly after the hedge turn up to AND including the
    resolution turn, walking sessions chronologically (checker-cost
    metadata for any future cell run of these hosts)."""
    count = 0
    started = False
    oracle_by_id = {q["question_id"]: q for q in data.load_oracle()}
    sessions = oracle_by_id[geom["question_id"]]["haystack_sessions"]
    for si in geom["chrono"]:
        for ti, turn in enumerate(sessions[si]):
            if turn["role"] != "user":
                continue
            if started:
                count += 1
            if (si, ti) == (hedge_si, hedge_ti):
                started = True
            if (si, ti) == (res_si, res_ti):
                return count
    return count


# ---------------------------------------------------------------------------
# render — the frozen two-call pipeline (values sampled here, after FM-4)
# ---------------------------------------------------------------------------

def _fm4_probe_ok(probe_question: str, value: str) -> bool:
    return _norm(value) not in _norm(probe_question)


def _style_exemplars(question: dict, rng) -> list[str]:
    """3 register exemplars from the host's own user turns (inject.py §4
    Call-2 convention, verbatim rule)."""
    turns = [t["content"] for s in question["haystack_sessions"] for t in s
             if t["role"] == "user"]
    turns = [" ".join(t.split()[:60]) for t in turns if len(t.split()) >= 10]
    return rng.sample(turns, min(3, len(turns)))


def _lint(text: str) -> dict:
    family, cue = detect_cue_family(text)
    return {
        "n_words": len(text.split()),
        "relative_time_hits": [p for p in RELATIVE_TIME_PATTERNS
                               if re.search(p, text, re.I)],
        "realized_cue_family": family,
        "realized_cue": cue,
    }


async def _plan_call(sys_prompt: str, payload: dict, tag: str) -> dict:
    return await llm.call_llm(sys_prompt, json.dumps(payload, ensure_ascii=False),
                              trace_tag=tag)


async def render_arc(arc: dict, *, plan_sys: str, render_sys: str,
                     length_band: str) -> None:
    """Render one arc's hedge + resolution through the frozen two-call
    pipeline (4 LLM calls when nothing retries; all cached). Mutates `arc`
    in place: value, hedge_text, resolution_text, lints, render provenance."""
    import random as _random

    question = {q["question_id"]: q for q in data.load_oracle()}[arc["question_id"]]
    sub = _Transcript(question)
    key = arc["key"]

    def host_turn(part: dict) -> str:
        return question["haystack_sessions"][part["session_idx"]][
            part["turn_idx"]]["content"]

    # ---- hedge: Call 1 (plan + value sampling under novelty) ---------------
    h = arc["hedge"]
    payload = {
        "contract": h["contract"],
        "injection_type": h["inj_type"],
        "target_key": {"entity": key["entity"], "attribute": key["attribute"]},
        "attribute_description": key["description"],
        "established_value": None,
        "gold_old_value": None,
        "fixed_value": None,
        "value_exclusions": [],
        "host_turn_text": host_turn(h),
        "insertion_position": h["position"],
        "host_session_date": h["date"],
        "other_session_date": arc["resolution"]["date"],
        "hedge_strength": arc["hedge_strength"],
        "near_miss_subtype": None,
    }
    out1 = await _plan_call(plan_sys, payload, f"m4-plan-hedge-{arc['arc_id']}")
    candidates = [str(c) for c in out1.get("value_candidates", [])]
    value = next((c for c in candidates if novelty_ok(c, sub, [])), None)
    for retry_round in range(1, 4):
        if value is not None:
            break
        retry_payload = dict(payload)
        retry_payload["value_exclusions"] = list(candidates)
        retry_payload["retry_note"] = (
            f"Retry round {retry_round}: every previous candidate was "
            "rejected because it appears verbatim in the source transcript "
            "or collides with an excluded value. Propose distinctly "
            "different candidates.")
        out1 = await _plan_call(plan_sys, retry_payload,
                                f"m4-plan-hedge-retry{retry_round}-{arc['arc_id']}")
        retry_candidates = [str(c) for c in out1.get("value_candidates", [])]
        value = next((c for c in retry_candidates if novelty_ok(c, sub, [])), None)
        candidates += retry_candidates
    if value is None:
        raise RuntimeError(f"{arc['arc_id']}: no candidate passed novelty "
                           f"after retries: {candidates}")
    if not _fm4_probe_ok(key["probe_question"], value):
        raise RuntimeError(f"{arc['arc_id']}: FM-4 violation — sampled value "
                           f"{value!r} appears in the probe question")
    arc["value"] = value
    arc["value_candidates"] = candidates

    # ---- both turns: Call 2 (render, round-2 style) ------------------------
    rng = _random.Random(f"{SEED_M4}:{arc['arc_id']}")

    async def render_part(part: dict, content_plan: str, tag: str) -> str:
        rowlike = {"inj_type": part["inj_type"],
                   "t3_variant_kind": part.get("t3_variant_kind"),
                   "edge_case_tags": []}
        exemplars = _style_exemplars(question, rng)
        for attempt in range(3):
            rpayload = {
                "contract": part["contract"],
                "content_plan": content_plan,
                "value": value,
                "established_value": None,
                "old_value": None,
                "connective": None,
                "integration_note": ROUND2_INTEGRATION_NOTE,
                "insertion_position": part["position"],
                "host_turn_text": host_turn(part),
                "style_exemplars": exemplars,
                "blacklist": BLACKLIST + ROUND2_TELL_BLACKLIST,
                "marker_words_allowed": False,
                "marker_words": MARKER_WORDS,
                "length_target_words": length_band,
                "attempt": attempt,
            }
            out2 = await llm.call_llm(
                render_sys, json.dumps(rpayload, ensure_ascii=False),
                trace_tag=f"{tag}-a{attempt}")
            aside = str(out2.get("aside", "")).strip()
            if _render_ok(aside, value, False, rowlike):
                return aside
            exemplars = _style_exemplars(question, rng)
        raise RuntimeError(f"{arc['arc_id']}: {part['inj_type']} render "
                           "failed asserts after 3 attempts")

    hedge_plan = _align_plan_value(out1.get("content_plan", ""), candidates, value)
    arc["hedge_text"] = await render_part(
        h, hedge_plan, f"m4-render-hedge-{arc['arc_id']}")

    # ---- resolution: Call 1 (fixed_value = the hedged value) ---------------
    r = arc["resolution"]
    res_payload = {
        "contract": r["contract"],
        "injection_type": r["inj_type"],
        "target_key": {"entity": key["entity"], "attribute": key["attribute"]},
        "attribute_description": key["description"],
        "established_value": None,
        "gold_old_value": None,
        "fixed_value": value,
        "value_exclusions": [],
        "host_turn_text": host_turn(r),
        "insertion_position": r["position"],
        "host_session_date": r["date"],
        "other_session_date": h["date"],
        "hedge_strength": None,
        "near_miss_subtype": None,
    }
    out3 = await _plan_call(plan_sys, res_payload,
                            f"m4-plan-res-{arc['arc_id']}")
    arc["resolution_text"] = await render_part(
        r, out3.get("content_plan", ""), f"m4-render-res-{arc['arc_id']}")

    arc["lints"] = {
        "hedge": _lint(arc["hedge_text"]),
        "resolution": _lint(arc["resolution_text"]),
        "fm4_probe_value_free": _fm4_probe_ok(key["probe_question"], value),
    }
    arc["render"] = {
        "pipeline": "frozen two-call (prompts/injector_plan.txt + "
                    "injector_render.txt, byte-untouched)",
        "length_target_words": length_band,
        "style": "round-2 bridged integration (GATE_ROUND 2 blacklist)",
    }


def manifest_rows_for_arc(arc: dict) -> list[dict]:
    """Splice-ready manifest rows (inject.make_row schema) for the arc's two
    turns — inject.build_perturbed_substrate consumes these directly."""
    rows = []
    for part_name, text, label, tags in (
            ("hedge", arc["hedge_text"], arc["hedge"]["construction_label"],
             [f"hedge:{arc['hedge_strength']}"]),
            ("resolution", arc["resolution_text"],
             arc["resolution"]["construction_label"],
             [f"m4_resolution:{arc['resolution_type']}"])):
        part = arc[part_name]
        row = make_row(
            injection_id=f"{arc['arc_id']}-{part_name}",
            question_id=arc["question_id"],
            inj_type=part["inj_type"],
            entity=arc["key"]["entity"],
            attribute=arc["key"]["attribute"],
            session_idx=part["session_idx"],
            turn_idx=part["turn_idx"],
            position=part["position"],
            probe_point="full_ingestion",
            construction_label=label,
            edge_case_tags=tags,
            seed_draw_index=arc["seed_draw_index"],
            t3_case_id=None,
            paired_with=(f"{arc['arc_id']}-hedge"
                         if part_name == "resolution" else None),
            extra={"benchmark": "lme", "arc_id": arc["arc_id"],
                   "gold_resolution_status": arc["gold_resolution_status"]},
        )
        row["value"] = arc["value"]
        row["text"] = text
        rows.append(row)
    return rows


def calibration_comparison(arcs: list[dict], calib: dict) -> dict:
    """CC reference vs the generated hedge turns, measured with the SAME
    statistics (the M4 prereg's reported calibration comparison)."""
    rendered = [a for a in arcs if a.get("hedge_text")]
    hedges = [a["hedge_text"] for a in rendered]
    intended = {f: sum(1 for a in rendered if a["hedge_strength"] == f)
                for f in FAMILY_ORDER}
    realized = {f: sum(1 for a in rendered
                       if (a.get("lints") or {}).get("hedge", {})
                       .get("realized_cue_family") == f)
                for f in FAMILY_ORDER}
    n = max(1, len(rendered))
    return {
        "cc_reference": {
            "family_freq": calib["family_freq"],
            "register": calib["register"],
            "n_candidates": calib["n_candidates"],
        },
        "generated_hedges": {
            "n": len(rendered),
            "intended_strength_counts": intended,
            "realized_cue_family_counts": realized,
            "realized_cue_family_freq": {f: round(realized[f] / n, 4)
                                         for f in FAMILY_ORDER},
            "no_cue_detected": len(rendered) - sum(realized.values()),
            "register": text_register_stats(hedges),
        },
        "derived_inputs_used": calib.get("derived"),
        "note": ("phrasing calibrated against CC via hedge-strength sampling "
                 "weights + the render length band; realized cue families "
                 "and register are compared post hoc — no CC text entered "
                 "any stimulus"),
    }


# ---------------------------------------------------------------------------
# generate — plan freeze -> render -> arcs.jsonl (+ summary)
# ---------------------------------------------------------------------------

def _assert_plan_frozen(planned: list[dict], plan_path: Path) -> None:
    """Freeze-before-render discipline: an existing arc_plan.jsonl must
    byte-match the recomputed plan (delete it DELIBERATELY to replan)."""
    if not plan_path.exists():
        return
    frozen = _read_jsonl(plan_path)
    if frozen != planned:
        raise SystemExit(
            f"{plan_path} exists and differs from the recomputed plan — the "
            "arc plan froze before rendering; delete the file deliberately "
            "(and regenerate everything) to replan")


async def run_generate(n_arcs: int, limit: int | None, out_dir: Path,
                       fresh_cap: int | None) -> dict:
    calib = load_calibration()
    weights = cc_strength_weights(calib)
    band = cc_length_band(calib)
    run_host_scout()

    planned, keys = plan_arcs(n_arcs, strength_weights=weights)
    plan_path = out_dir / ARC_PLAN_PATH.name
    keys_path = out_dir / KEYS_PATH.name
    arcs_path = out_dir / ARCS_PATH.name
    _assert_plan_frozen(planned, plan_path)
    if not plan_path.exists():
        _write_jsonl(plan_path, planned)     # freeze (values still None)
        _write_jsonl(keys_path, keys)        # FM-4: keys before values

    existing = {a["arc_id"]: a for a in _read_jsonl(arcs_path)}
    arcs: list[dict] = []
    todo: list[dict] = []
    for arc in planned:
        prior = existing.get(arc["arc_id"])
        if prior and prior.get("hedge_text") and prior.get("resolution_text"):
            arcs.append(prior)               # resume: already rendered
        else:
            arcs.append(arc)
            todo.append(arc)
    if limit is not None:
        todo = todo[:limit]

    plan_sys = (PROMPTS_DIR / "injector_plan.txt").read_text()
    render_sys = (PROMPTS_DIR / "injector_render.txt").read_text()
    failures: list[str] = []

    async def render_one(arc: dict) -> None:
        try:
            await render_arc(arc, plan_sys=plan_sys,
                             render_sys=render_sys, length_band=band)
            arc["manifest_rows"] = manifest_rows_for_arc(arc)
        except Exception as exc:
            failures.append(f"{arc['arc_id']}: {exc}")

    guard = m4_guard(fresh_cap)
    guard.install()
    try:
        # arcs are independent; per-arc calls stay sequential (dependencies),
        # global concurrency is bounded by the llm module's semaphore (10).
        await asyncio.gather(*(render_one(a) for a in todo))
    finally:
        guard.uninstall()

    _write_jsonl(arcs_path, arcs)
    rendered = [a for a in arcs if a.get("hedge_text")]
    summary = {
        "id": "V2.4-M4-GENERATION",
        "prereg": ("protocol/CHANGELOG.md Protocol v2.4 M4 + 2026-07-08 "
                   "advisory-budget amendment; "
                   "research_notes/dataset_scout_v24.md"),
        "n_planned": len(planned),
        "n_rendered": len(rendered),
        "balance_planned": {t: sum(1 for a in planned
                                   if a["resolution_type"] == t)
                            for t in RESOLUTION_TYPES},
        "balance_rendered": {t: sum(1 for a in rendered
                                    if a["resolution_type"] == t)
                             for t in RESOLUTION_TYPES},
        "hosts": {"benchmark": "lme (LoCoMo unused pool empty — see "
                               "host_scout.json)",
                  "one_arc_per_substrate": True},
        "calibration_reference": {
            "path": _rel(CALIBRATION_PATH),
            "sha256": _sha256(CALIBRATION_PATH.read_text()),
        },
        "calibration_comparison": calibration_comparison(arcs, calib),
        "budget": {**advisory_note(), "fresh_calls_this_run": guard.fresh},
        "review_gate": ("arcs are UNUSABLE until the owner review "
                        "(status pending_owner_review); the final pool "
                        "requires results/v2.4/m4/owner_review.json"),
        "failures": failures,
        "outputs": {"arc_plan": _rel(plan_path), "keys": _rel(keys_path),
                    "arcs": _rel(arcs_path)},
    }
    _write_json(out_dir / GEN_SUMMARY_PATH.name, summary)
    if failures:
        for f in failures:
            print(f"RENDER FAILURE: {f}", file=sys.stderr)
    return summary


# ---------------------------------------------------------------------------
# review-pack — the owner's fast pass (checkboxes) + machine template
# ---------------------------------------------------------------------------

def _excerpt(text: str, limit_words: int = 70) -> str:
    words = text.split()
    return (" ".join(words[:limit_words]) + " […]"
            if len(words) > limit_words else text)


def _excerpt_keep_bold(spliced: str, aside: str, limit_words: int = 110) -> str:
    """Excerpt of a spliced turn that NEVER elides the bolded aside (r1 review
    pack defect: the plain head-truncating _excerpt swallowed the spliced
    sentence in 3 context blocks). The aside is bolded, then elision happens
    only AROUND it: head words, `[…]`, the full bold span, tail words."""
    bolded = f"**{aside}**"
    text = spliced.replace(aside, bolded)
    words = text.split()
    if len(words) <= limit_words:
        return text
    b_words = bolded.split()
    start = next((i for i in range(len(words) - len(b_words) + 1)
                  if words[i:i + len(b_words)] == b_words), None)
    if start is None:                      # aside not found — never elide
        return text
    end = start + len(b_words)
    keep = max(0, limit_words - len(b_words))
    front = min(start, (2 * keep) // 3)
    back = min(len(words) - end, keep - front)
    front = min(start, keep - back)        # hand unused tail budget to front
    parts: list[str] = []
    if start - front > 0:
        parts.append("[…]")
    parts += words[start - front:end + back]
    if end + back < len(words):
        parts.append("[…]")
    return " ".join(parts)


def _context_block(question: dict, part: dict, aside: str) -> list[str]:
    """Preceding turn + the SPLICED host turn (aside bolded) — what the
    labeling pipeline would actually see. The spliced sentence is always
    present and bolded (never elided)."""
    session = question["haystack_sessions"][part["session_idx"]]
    ti = part["turn_idx"]
    lines = []
    if ti > 0:
        prev = session[ti - 1]
        lines.append(f"> _{prev['role']}:_ {_excerpt(prev['content'], 40)}")
    spliced = splice_turn(session[ti]["content"],
                          [(part["position"], aside)])
    lines.append(f"> _user (spliced):_ {_excerpt_keep_bold(spliced, aside, 110)}")
    return lines


def render_review_pack(arcs: list[dict]) -> str:
    oracle_by_id = {q["question_id"]: q for q in data.load_oracle()}
    rendered = [a for a in arcs if a.get("hedge_text")]
    lines = [
        "# v2.4 M4 — synthetic hedge-arc review pack (OWNER pass required)",
        "",
        "Per the M4 preregistration this review is the sole human step: "
        "every arc needs an explicit ACCEPT or REJECT before it can enter "
        "the pool. Judge REALISM (would this user say this, here?) and "
        "VALIDITY (hedge is a real unrealized plan; resolution matches its "
        "label). Tick exactly one box per arc, then record the verdicts in "
        f"`{_rel(OWNER_REVIEW_PATH)}` (template: "
        f"`{_rel(REVIEW_TEMPLATE_PATH)}`).",
        "",
        f"Arcs: {len(rendered)} rendered / {len(arcs)} planned. Balance: "
        + ", ".join(f"{t}={sum(1 for a in rendered if a['resolution_type'] == t)}"
                    for t in RESOLUTION_TYPES) + ".",
        "",
        "The arcs are UNUSABLE until this review lands: the pool emitter "
        "refuses to run without owner_review.json.",
        "",
    ]
    for a in rendered:
        q = oracle_by_id[a["question_id"]]
        h, r = a["hedge"], a["resolution"]
        lints = a.get("lints", {})
        flags = []
        if lints.get("hedge", {}).get("relative_time_hits"):
            flags.append("hedge relative-time lint")
        if lints.get("resolution", {}).get("relative_time_hits"):
            flags.append("resolution relative-time lint")
        if lints.get("hedge", {}).get("realized_cue_family") is None:
            flags.append("no hedge cue detected")
        lines += [
            f"## {a['arc_id']}",
            "",
            f"- host: `{a['question_id']}` ({a['question_type']}); key "
            f"`user/{a['key']['attribute']}`; value **{a['value']}**",
            f"- labels: hedge `speculative` (strength {a['hedge_strength']}); "
            f"resolution `{r['construction_label']}` "
            f"(type **{a['resolution_type']}**, gold status "
            f"`{a['gold_resolution_status']}`)",
            f"- placement: hedge s{h['session_idx']}t{h['turn_idx']} "
            f"{h['position']} ({h['date']}) -> resolution "
            f"s{r['session_idx']}t{r['turn_idx']} {r['position']} "
            f"({r['date']}); window {a['pending_window_user_turns']} user "
            "turns",
        ]
        if flags:
            lines.append(f"- FLAGS: {'; '.join(flags)}")
        lines += ["", "**Hedge context** "
                  f"(session date {h['date']}):", ""]
        lines += _context_block(q, h, a["hedge_text"])
        lines += ["", f"**Resolution context** (session date {r['date']}):", ""]
        lines += _context_block(q, r, a["resolution_text"])
        lines += [
            "",
            f"- [ ] ACCEPT `{a['arc_id']}`",
            f"- [ ] REJECT `{a['arc_id']}` — reason: ____________",
            "",
        ]
    return "\n".join(lines)


def run_review_pack(arcs_path: Path = ARCS_PATH,
                    pack_path: Path = REVIEW_PACK_PATH,
                    template_path: Path = REVIEW_TEMPLATE_PATH) -> dict:
    arcs = _read_jsonl(arcs_path)
    if not arcs:
        raise SystemExit(f"no arcs at {arcs_path} — run "
                         "`hedge_arcs.py generate` first")
    rendered = [a for a in arcs if a.get("hedge_text")]
    if not rendered:
        raise SystemExit("no rendered arcs to review")
    pack_path.parent.mkdir(parents=True, exist_ok=True)
    pack_path.write_text(render_review_pack(arcs))
    template = {
        "reviewer": "",
        "review_date": "",
        "note": ("copy to owner_review.json, fill reviewer/review_date and "
                 "one verdict (accept|reject) per arc; the pool emitter "
                 "requires full coverage"),
        "decisions": {a["arc_id"]: {"verdict": None, "note": ""}
                      for a in rendered},
    }
    _write_json(template_path, template)
    return {"arcs": len(rendered), "pack": _rel(pack_path),
            "template": _rel(template_path)}


# ---------------------------------------------------------------------------
# pool — final reviewed stratum (REFUSES without owner_review.json)
# ---------------------------------------------------------------------------

def build_pool(arcs: list[dict], review: dict | None) -> tuple[list[dict], dict]:
    """Pure pool constructor. `review` None (missing owner_review.json) or
    incomplete coverage -> SystemExit: the arcs stay UNUSABLE (M4 prereg:
    the owner reviews every generated arc)."""
    if review is None:
        raise SystemExit(
            "REFUSED: no owner review found — the M4 arcs are unusable "
            f"until the owner reviews every arc ({_rel(OWNER_REVIEW_PATH)}; "
            "template written by `hedge_arcs.py review-pack`)")
    decisions = review.get("decisions") or {}
    rendered = [a for a in arcs if a.get("hedge_text")]
    arc_ids = {a["arc_id"] for a in rendered}
    missing = sorted(arc_ids - set(decisions))
    unknown = sorted(set(decisions) - arc_ids)
    if missing:
        raise SystemExit(f"REFUSED: owner review missing decisions for "
                         f"{len(missing)} arc(s): {missing[:5]}...")
    if unknown:
        raise SystemExit(f"REFUSED: owner review names unknown arc ids: "
                         f"{unknown[:5]}...")
    bad = {aid: d for aid, d in decisions.items()
           if (d or {}).get("verdict") not in ("accept", "reject")}
    if bad:
        raise SystemExit(f"REFUSED: non accept/reject verdicts for "
                         f"{sorted(bad)[:5]}...")
    pool = []
    for a in rendered:
        d = decisions[a["arc_id"]]
        if d["verdict"] != "accept":
            continue
        pool.append({
            **{k: a[k] for k in (
                "arc_id", "row_kind", "benchmark", "question_id",
                "question_type", "resolution_type", "gold_resolution_status",
                "hedge_strength", "key", "hedge", "resolution",
                "pending_window_user_turns", "seed", "seed_draw_index",
                "value", "hedge_text", "resolution_text", "lints",
                "manifest_rows")},
            "stratum": "m4_synthetic_reviewed",
            "usable": True,
            "status": "owner_accepted",
            "review": {"verdict": "accept", "note": d.get("note", ""),
                       "reviewer": review.get("reviewer", ""),
                       "review_date": review.get("review_date", "")},
            "mixing_rule": ("separately-labeled synthetic stratum — never "
                            "pooled with organic natural-hedge rows (M4 "
                            "prereg)"),
        })
    summary = {
        "id": "V2.4-M4-POOL",
        "reviewed": len(rendered),
        "accepted": len(pool),
        "rejected": len(rendered) - len(pool),
        "balance_accepted": {t: sum(1 for p in pool
                                    if p["resolution_type"] == t)
                             for t in RESOLUTION_TYPES},
        "at_risk_accepted": sum(1 for p in pool if p["gold_resolution_status"]
                                in ("unresolved", "abandoned")),
        "resolved_true_accepted": sum(
            1 for p in pool if p["gold_resolution_status"] == "resolved_true"),
        "prereg_target": "40-60 arcs, balanced resolve-true/abandoned/open",
        "reviewer": review.get("reviewer", ""),
        "review_date": review.get("review_date", ""),
    }
    return pool, summary


def run_pool() -> dict:
    arcs = _read_jsonl(ARCS_PATH)
    if not arcs:
        raise SystemExit(f"no arcs at {ARCS_PATH}")
    review = (json.loads(OWNER_REVIEW_PATH.read_text())
              if OWNER_REVIEW_PATH.exists() else None)
    pool, summary = build_pool(arcs, review)
    _write_jsonl(POOL_PATH, pool)
    _write_json(POOL_SUMMARY_PATH, summary)
    return summary


# ---------------------------------------------------------------------------
# smoke — 2 arcs end-to-end (<= 8 fresh calls), artifacts under m4/smoke/
# ---------------------------------------------------------------------------

async def run_smoke(fresh_cap: int = 8) -> dict:
    calib = (json.loads(CALIBRATION_PATH.read_text())
             if CALIBRATION_PATH.exists() else run_calibrate())
    weights = cc_strength_weights(calib)
    band = cc_length_band(calib)
    planned, keys = plan_arcs(N_ARCS_DEFAULT, strength_weights=weights)
    picked = planned[:2]

    plan_sys = (PROMPTS_DIR / "injector_plan.txt").read_text()
    render_sys = (PROMPTS_DIR / "injector_render.txt").read_text()
    guard = m4_guard(fresh_cap)
    guard.install()
    try:
        for arc in picked:
            await render_arc(arc, plan_sys=plan_sys, render_sys=render_sys,
                             length_band=band)
            arc["manifest_rows"] = manifest_rows_for_arc(arc)
    finally:
        guard.uninstall()

    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    _write_jsonl(SMOKE_DIR / "arcs_smoke.jsonl", picked)
    _write_jsonl(SMOKE_DIR / "m4_keys_smoke.jsonl",
                 [k for k in keys if k["arc_id"] in
                  {a["arc_id"] for a in picked}])
    (SMOKE_DIR / "review_pack_smoke.md").write_text(render_review_pack(picked))
    out = {
        "arcs": [{
            "arc_id": a["arc_id"],
            "resolution_type": a["resolution_type"],
            "hedge_strength": a["hedge_strength"],
            "key": a["key"]["key_ref"],
            "value": a["value"],
            "hedge_text": a["hedge_text"],
            "resolution_text": a["resolution_text"],
            "lints": a["lints"],
        } for a in picked],
        "fresh_calls": guard.fresh,
        "fresh_cap": fresh_cap,
        "length_band": band,
        "strength_weights": weights,
        "budget": advisory_note(),
        "artifacts": _rel(SMOKE_DIR),
    }
    (SMOKE_DIR / "smoke_summary.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="v2.4 M4 — human-verified synthetic hedge arcs")
    sub = ap.add_subparsers(dest="cmd", required=True)
    cal = sub.add_parser("calibrate", help="CC slice + hedge-cue mining")
    cal.add_argument("--episodes", type=int, default=CC_EPISODES_DEFAULT)
    cal.add_argument("--refresh", action="store_true")
    gen = sub.add_parser("generate", help="plan freeze + two-call render")
    gen.add_argument("--n", type=int, default=N_ARCS_DEFAULT)
    gen.add_argument("--limit", type=int, default=None,
                     help="render at most K unrendered arcs (resume-friendly)")
    gen.add_argument("--fresh-cap", type=int, default=None,
                     help="explicit local fresh-call cap (stages are "
                          "ADVISORY; the 120k envelope is the only default "
                          "stop)")
    sub.add_parser("review-pack", help="review_pack.md + owner template")
    sub.add_parser("pool", help="final reviewed pool (REQUIRES "
                                "owner_review.json)")
    smk = sub.add_parser("smoke", help="2 arcs end-to-end (<= 8 fresh calls)")
    smk.add_argument("--fresh-cap", type=int, default=8)
    args = ap.parse_args()

    if args.cmd == "calibrate":
        calib = run_calibrate(args.episodes, refresh=args.refresh)
        print(json.dumps({
            "n_turns_scanned": calib["n_turns_scanned"],
            "n_candidates": calib["n_candidates"],
            "family_counts": calib["family_counts"],
            "length_median": calib["register"]["length_words"]["median"],
            "derived": calib["derived"],
        }, indent=1))
        print(f"-> {CALIBRATION_PATH}\n-> {CC_SLICE_PATH}")
    elif args.cmd == "generate":
        summary = asyncio.run(run_generate(
            args.n, args.limit, M4_DIR, args.fresh_cap))
        print(json.dumps({k: summary[k] for k in
                          ("n_planned", "n_rendered", "balance_rendered",
                           "budget", "failures")}, indent=1))
        print(f"-> {ARCS_PATH}\n-> {GEN_SUMMARY_PATH}")
        note = summary["budget"]
        if note["advisory_overrun"]:
            print(f"ADVISORY OVERRUN: stage m4 spent {note['spent']:,} > "
                  f"forecast {note['advisory_allocation']:,} (no stop — "
                  "2026-07-08 amendment; total envelope is the hard limit)")
        if summary["failures"]:
            raise SystemExit(f"{len(summary['failures'])} arc(s) failed "
                             "(progress persisted; rerun resumes)")
    elif args.cmd == "review-pack":
        out = run_review_pack()
        print(json.dumps(out, indent=1))
    elif args.cmd == "pool":
        summary = run_pool()
        print(json.dumps(summary, indent=1))
        print(f"-> {POOL_PATH}\n-> {POOL_SUMMARY_PATH}")
    elif args.cmd == "smoke":
        out = asyncio.run(run_smoke(args.fresh_cap))
        print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
