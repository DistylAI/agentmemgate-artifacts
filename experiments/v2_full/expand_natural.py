"""v2.2 C1 — natural-hedge stratum expansion (owner: implementer C1).

Implements the preregistered Protocol v2.2 C1 amendment
(research/protocol_changelog.md, entry "Protocol v2.2"): the §8-rule-5
estimand (Δ̂_int on the natural speculation stratum; sign-opposition +
CI-exclusion downgrade test) recomputed on an EXPANDED stratum —

  (a) the 17 LME natural-pair substrates run through all four cells as FULL
      cell substrates (they previously ran STORELESS, natres.py/V2-F3);
  (b) additional harvested natural hedges from
      research/natural_candidates_full.jsonl anchored to schema keys under
      the frozen anchoring rule: hedge turn verbatim-verified at its
      (source, question/conv, session, turn) anchor AND attribute maps to an
      existing schema key OR a newly registered aux-style key with a
      value-free probe question (FM-4-checked); the expanded row list
      FREEZES before any cell run.

Sanctioned harness corrections carried by this module (they fix the audited
defects A1/A3 of results/v2/natres/failure_audit.md, not any rule):
  A1 — pair-key preference in tuple↔span matching:
       label.match_tuple_to_span(..., preferred_key=...) (new optional
       parameter, default preserves the old behavior; natres.match_staged_tuple
       passes the authored pair key).
  A3 — pair keys registered in the key-mapper candidate lists for their
       substrates: the v2.2 CellJob candidate lists built here contain the
       natural_pair_keys entries (plus the newly registered expansion keys).

Subcommands (CLI from pilot/, .venv/bin/python expand_natural.py ...):

  anchor   The frozen anchoring pass. Scans natural_candidates_full.jsonl
           speculation rows not already in the rule-5 stratum (the 18 LoCoMo
           pair stages and the 2 schema-anchored censored hedges are IN it;
           the 17 LME pair stages enter the freeze list as row_kind
           `lme_pair_stage` per C1(a)). Emits:
             results/v2.2/natural_stratum_expansion.jsonl  (the FREEZE list)
             results/v2.2/expansion_keys.jsonl             (new aux-style keys)
             results/v2.2/anchor_summary.json
           Deterministic; seed 20260706 recorded-not-consumed (no random
           draw exists: row order is a lexicographic sort, key derivation is
           rule-based, and the only LLM stage — key mapping — is the frozen
           key_mapper prompt behind the llm.py cache, <= 60 calls, only for
           rows whose candidate-host pools contain existing schema keys).

  jobs     CellJob rows for the C1 cell runs: the 17 LME pair substrates,
           s_cleaned sources, substrate build via the natural-selection
           replay path (test1_eval.s_cleaned_substrate — the V2-F1 §2.3
           stored-value stream), UNPERTURBED (no manifest splice), candidate
           lists = pair keys + expansion keys hosted on the substrate (A3).
           Default: write results/v2.2/cell_jobs_manifest.json (no LLM).
           --run executes arms.run_cells into results/v2.2/ (orchestrator
           only). --qids may name additional freeze-listed hedge source
           questions (their candidate lists are their expansion keys).
           BUDGET GATE (review B-1 fix): --run requires the COMBINED C1+C2
           forecast's budget file (results/v2.2/budget.json, written by
           forecast_v22.py), refuses substrates outside the planned C1 set
           (17 pair substrates + the greedy-selected additions), and runs
           under forecast_v22.SharedCallBudget — every fresh LLM call
           decrements the shared 40,000-call combined envelope, and the run
           aborts loudly when it is exhausted. rule5's fresh judge calls
           decrement the same envelope (stage c1_rule5_judging) whenever the
           budget file exists.

  rule5    Extends natural_check_rule5 (imported, not copied): reproduces the
           old n=20 block UNCHANGED (hard-asserted against the frozen
           results/v2/metrics.json numbers) and computes the expanded block
           per the SAME estimand/decision rule over old rows + the frozen
           expansion rows whose four cell timelines exist; both blocks ->
           results/v2.2/rule5_expanded.json (+ per-row audit trail
           results/v2.2/rule5_expanded_rows.jsonl). Coverage stated, never
           hidden.

  smoke    Anchor 3 rows end-to-end (<= 3 key-mapper LLM calls) + build ONE
           CellJob (no cell run). Writes only results/v2.2/smoke/.

Coordinate conventions (verified, not assumed — the verbatim pass asserts
them row by row):
  * LME harvest rows: session_idx/turn_idx are ORIGINAL s_cleaned haystack
    indices — the same space the v2.2 pair-substrate replays use.
  * LoCoMo harvest rows: session_idx is the CORPUS SESSION NUMBER (1-indexed,
    dia_id space) and turn_idx is 0-based; timeline coordinates are positional
    (conv.session_numbers.index(number), turn_idx). All 13 rows resolve under
    this convention and none under the positional one.
  * v2 cell timelines replayed PERTURBED worlds, but the splice appends aside
    text into existing turns (inject.build_perturbed_substrate), so turn
    coordinates are unchanged and hedge turns stay verbatim-contained.

No frozen prompt or artifact is modified. All LLM calls go through
llm.call_llm (P10 cache).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from functools import cache
from pathlib import Path
from typing import Any, Callable, Optional

import _bootstrap  # noqa: F401  (R1 layout: puts src/ + experiments/ on sys.path)

import arms
import data
import extract
import label
import llm
import locomo_data
import metrics_v2
import natural_check_rule5 as ncr5
import driver as run_v2
import stats
import store
import eval_classifier as test1_eval

from paths import ROOT, RESEARCH_DIR, RESULTS_DIR
V2_DIR = RESULTS_DIR / "v2"
V22_DIR = RESULTS_DIR / "v2.2"
SMOKE_DIR = V22_DIR / "smoke"

CANDIDATES_PATH = RESEARCH_DIR / "natural_candidates_full.jsonl"
PAIRS_PATH = RESEARCH_DIR / "natural_pairs_v2.jsonl"
LME_SCHEMA_PATH = RESEARCH_DIR / "key_schema_v2_lme.json"

EXPANSION_PATH = V22_DIR / "natural_stratum_expansion.jsonl"
EXPANSION_KEYS_PATH = V22_DIR / "expansion_keys.jsonl"
ANCHOR_SUMMARY_PATH = V22_DIR / "anchor_summary.json"
JOBS_MANIFEST_PATH = V22_DIR / "cell_jobs_manifest.json"
RULE5_PATH = V22_DIR / "rule5_expanded.json"
RULE5_ROWS_PATH = V22_DIR / "rule5_expanded_rows.jsonl"

SEED = 20260706            # v2.2 C1 freeze seed (recorded-not-consumed)
BOOT_SEED = ncr5.SEED      # 20260702 — the SAME estimand machinery (§8 rule 5)
N_BOOT = ncr5.N_BOOT
KEY_MAPPER_CALL_CAP = 60   # C1 anchoring-pass budget line
JUDGE_CALL_CAP = 400       # equivalence-judge cap for the expanded block
TARGET_NEW_ROWS = 45       # C1: target >= 45 new rows in the freeze list

#: the two frozen censored hedges ALREADY in the rule-5 stratum, in the
#: harvest coordinate convention (corpus session NUMBER, 0-based turn) —
#: derived from ncr5.CENSORED_ANCHORED dia_ids D28:5 / D17:11.
_CENSORED_IN_STRATUM_HARVEST = {
    ("conv-41", 28, 4),   # D28:5  -> john/employment_status
    ("conv-41", 17, 10),  # D17:11 -> john/dog_ownership
}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    # \n-only iteration (label.read_jsonl rationale: U+2028/U+2029 in turns)
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Frozen-artifact loaders
# ---------------------------------------------------------------------------

@cache
def lme_pair_key_entries() -> dict[str, dict]:
    """pair_id -> natural_pair_keys entry (17, V2-F3)."""
    schema = json.loads(LME_SCHEMA_PATH.read_text())
    entries = {e["pair_id"]: e for e in schema["natural_pair_keys"]["entries"]}
    assert len(entries) == 17
    return entries


@cache
def lme_pair_qids() -> dict[str, str]:
    """question_id -> pair_id for the 17 LME pair substrates (1:1, asserted)."""
    out: dict[str, str] = {}
    for pid, e in lme_pair_key_entries().items():
        assert e["question_id"] not in out, "pair qids expected 1:1"
        out[e["question_id"]] = pid
    return out


@cache
def lme_pairs_by_id() -> dict[str, dict]:
    rows = [r for r in _read_jsonl(PAIRS_PATH) if r["corpus"] == "lme"]
    assert len(rows) == 17
    return {r["pair_id"]: r for r in rows}


def speculation_candidates() -> list[dict]:
    rows = [r for r in _read_jsonl(CANDIDATES_PATH) if r.get("kind") == "speculation"]
    assert rows, f"no speculation rows in {CANDIDATES_PATH}"
    return rows


def pair_key_as_canonical(entry: dict) -> store.CanonicalKey:
    """natural_pair_keys entry -> mapper candidate, rendered exactly like the
    gold-key branch of store.build_candidate_keys (blind: no values)."""
    return store.CanonicalKey(
        entity=entry["entity"],
        attribute=entry["attribute"],
        description=("canonical single-valued attribute "
                     f"(class: {entry['attribute_class']})"),
        probe_question=entry["probe_question"],
    )


def expansion_key_as_canonical(row: dict) -> store.CanonicalKey:
    return store.CanonicalKey(
        entity=row["entity"], attribute=row["attribute"],
        description=row.get("description", ""),
        probe_question=row["probe_question"],
    )


@cache
def _lme_v2_candidate_names(question_id: str) -> tuple[dict, ...]:
    """The FROZEN v2 candidate list of one LME cell substrate (gold KU key when
    one exists + registered v2 aux keys) — exactly run_v2.lme_cell_jobs'
    wiring through gated_store._prepare_substrate."""
    entry = run_v2.lme_ku_entries().get(question_id)
    aux = run_v2.v2_aux_keys("lme", question_id)
    cands = (store.build_candidate_keys(entry, aux) if entry is not None
             else list(aux))
    return tuple({"name": c.name, "key": c} for c in cands)


@cache
def _locomo_candidate_list(sample_id: str) -> tuple:
    """The FROZEN v2 candidate list of one LoCoMo conversation (gold update
    keys + registered v2 aux keys) — run_v2.locomo_candidates verbatim."""
    return tuple(run_v2.locomo_candidates(sample_id))


# ---------------------------------------------------------------------------
# Verbatim anchor verification (the first clause of the frozen anchoring rule)
# ---------------------------------------------------------------------------

def verify_lme_anchor(row: dict) -> dict:
    """Verbatim-verify one s_cleaned harvest row at its anchor. Raises
    AssertionError with the failure reason when the rule is not met."""
    substrate = test1_eval.s_cleaned_substrate(row["question_id"])
    si, ti = row["session_idx"], row["turn_idx"]
    turn = substrate.sessions[si][ti]
    assert turn["role"] == "user", f"anchor turn role {turn['role']!r} != user"
    assert row["text"] in turn["content"], "hedge text not verbatim in anchor turn"
    return {
        "verified": True,
        "source_file": "s_cleaned",
        "role": "user",
        "turn_sha256": _sha256(turn["content"]),
        "turn_text": turn["content"],
        "date": substrate.dates[si],
        "anchor": [si, ti],
    }


def verify_locomo_anchor(row: dict) -> dict:
    """Verbatim-verify one LoCoMo harvest row. Harvest convention (verified
    corpus-wide): session_idx = corpus session NUMBER (1-indexed, dia space),
    turn_idx 0-based; positional coords recorded for the timeline readers."""
    conv = locomo_data.get_conversation(row["question_id"])
    session_number = row["session_idx"]
    assert session_number in conv.session_numbers, \
        f"no session number {session_number} in {row['question_id']}"
    si = conv.session_numbers.index(session_number)
    ti = row["turn_idx"]
    turn = conv.sessions[si][ti]
    assert row["text"] in turn["text"], "hedge text not verbatim in anchor turn"
    return {
        "verified": True,
        "source_file": "locomo",
        "speaker": turn["speaker"],
        "dia_id": turn["dia_id"],
        "turn_sha256": _sha256(turn["text"]),
        "turn_text": turn["text"],
        "date": conv.dates[si],
        "anchor": [si, ti],  # positional (timeline) coordinates
        "anchor_harvest": [session_number, ti],
    }


# ---------------------------------------------------------------------------
# Host indexes (where a hedge turn can be read from cell timelines)
# ---------------------------------------------------------------------------

@cache
def pair_turn_index() -> dict[str, tuple]:
    """turn content -> ((qid, si, ti), ...) over the 17 pair substrates'
    s_cleaned user turns (the C1(a) scheduled cell runs)."""
    idx: dict[str, list] = {}
    for qid in sorted(lme_pair_qids()):
        substrate = test1_eval.s_cleaned_substrate(qid)
        for si, session in enumerate(substrate.sessions):
            for ti, turn in enumerate(session):
                if turn["role"] == "user":
                    idx.setdefault(turn["content"], []).append((qid, si, ti))
    return {k: tuple(v) for k, v in idx.items()}


@cache
def v2_oracle_turn_index() -> dict[str, tuple]:
    """turn content -> ((qid, si, ti), ...) over the 127 v2 LME substrates'
    ORACLE streams (the frozen v2 cell runs; splice preserves coordinates)."""
    idx: dict[str, list] = {}
    for qid in run_v2.lme_v2_qids():
        substrate = data.get_substrate(qid)
        for si, session in enumerate(substrate.sessions):
            for ti, turn in enumerate(session):
                if turn["role"] == "user":
                    idx.setdefault(turn["content"], []).append((qid, si, ti))
    return {k: tuple(v) for k, v in idx.items()}


def hedge_hosts(row: dict, verified: dict) -> list[dict]:
    """Every substrate whose four-cell timelines can (or could) contain the
    hedge anchor turn, in deterministic priority order:
      1. v22_pair_run        — the C1(a) scheduled v2.2 cell runs
      2. v2_lme_cell         — frozen v2 LME cell timelines (results/v2)
      3. v2_locomo_cell      — frozen v2 LoCoMo cell timelines (results/v2)
      4. v22_expansion_run   — the hedge's own source question, runnable via
                               `jobs --qids` (not scheduled by default)
    """
    hosts: list[dict] = []
    if verified["source_file"] == "locomo":
        hosts.append({
            "kind": "v2_locomo_cell", "question_id": row["question_id"],
            "session_idx": verified["anchor"][0],
            "turn_idx": verified["anchor"][1],
            "base": "results/v2", "scheduled": False, "exists": True,
        })
        return hosts
    turn_text = verified["turn_text"]
    seen: set[tuple] = set()
    for qid, si, ti in pair_turn_index().get(turn_text, ()):
        if (qid, si, ti) in seen:
            continue
        seen.add((qid, si, ti))
        hosts.append({
            "kind": "v22_pair_run", "question_id": qid,
            "session_idx": si, "turn_idx": ti,
            "base": "results/v2.2", "scheduled": True,
            "cross_hosted": qid != row["question_id"],
        })
    for qid, si, ti in v2_oracle_turn_index().get(turn_text, ()):
        hosts.append({
            "kind": "v2_lme_cell", "question_id": qid,
            "session_idx": si, "turn_idx": ti,
            "base": "results/v2", "scheduled": False, "exists": True,
            "note": "v2 replay world was perturbed; splice appends into "
                    "existing turns so coordinates are unchanged",
        })
    if row["question_id"] not in lme_pair_qids():
        hosts.append({
            "kind": "v22_expansion_run", "question_id": row["question_id"],
            "session_idx": row["session_idx"], "turn_idx": row["turn_idx"],
            "base": "results/v2.2", "scheduled": False,
        })
    return hosts


# ---------------------------------------------------------------------------
# New aux-style key derivation (value-free, FM-4-checked, deterministic)
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset("""
a about after all also am an and any are around as at be been being bit but by
can cant considering consider could did do does done dont down few for from get
gets getting go going gonna got had has have having how i id if ill im in into
is it its ive just like likely look looking lately may maybe me might more my
need not now of off ok okay on once one or our out over perhaps plan planning
plans pretty probably really should so some soon still sure that the them then
there these they think thinking this those though time to too try trying up us
via want wanna was we well were what when which while whether will with wont
would yeah yes you your
""".split())

_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def value_like_tokens(text: str) -> set[str]:
    """Deterministic candidate-VALUE token scan for the FM-4 check: tokens
    capitalized mid-sentence (proper nouns / brand names) and tokens containing
    digits; stopwords exempt (the article in "The Beatles" identifies no
    value). Lower-cased, apostrophes stripped. The hedge's eventual extracted
    value is unknown at anchoring time; this is the conservative surrogate the
    probe question and attribute must avoid."""
    banned: set[str] = set()
    for sentence in re.split(r"[.!?\n]+", text):
        words = _WORD_RE.findall(sentence)
        for i, w in enumerate(words):
            base = w.strip("'").replace("'", "")
            if not base or base.lower() in _STOPWORDS:
                continue
            if any(c.isdigit() for c in base):
                banned.add(base.lower())
            elif i > 0 and base[0].isupper():
                banned.add(base.lower())
    return banned


def derive_attribute_tokens(row: dict) -> tuple[list[str], set[str]]:
    """Deterministic attribute-word derivation for a new aux-style key:
    content words of `about` (falling back to the hedge text when `about`
    just repeats it), minus stopwords, minus value-like tokens."""
    about = (row.get("about") or "").strip()
    text = row["text"]
    base = about if about and about != text else text
    banned = value_like_tokens(about + "\n" + text)
    kept: list[str] = []
    for w in _WORD_RE.findall(base):
        lw = w.strip("'").lower().replace("'", "")
        if not lw or lw in _STOPWORDS or lw in banned or any(c.isdigit() for c in lw):
            continue
        if lw not in kept:
            kept.append(lw)
        if len(kept) == 4:
            break
    return kept, banned


def fm4_value_free(probe_question: str, banned: set[str]) -> bool:
    """FM-4-style check (pilot convention, key_schema natural_pair_keys
    probe_question_rule): the probe question contains no candidate-value
    token of the hedge (len >= 3), case-insensitive, apostrophes stripped."""
    probe_tokens = {w.replace("'", "").lower() for w in _WORD_RE.findall(probe_question)}
    return not any(b in probe_tokens for b in banned if len(b) >= 3)


class KeyRegistry:
    """Deterministic aux-style key registration for the anchoring pass."""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self._taken: set[tuple] = set()

    def register(self, *, benchmark: str, question_id: str, entity: str,
                 row: dict, row_id: str, host_qids: list[str]) -> dict:
        kept, banned = derive_attribute_tokens(row)
        if not kept:
            raise AssertionError("no attribute tokens derivable (all value-like/stopwords)")
        attribute = "_".join(kept)
        n = 2
        while (benchmark, question_id, entity, attribute) in self._taken:
            attribute = "_".join(kept) + f"_{n}"
            n += 1
        self._taken.add((benchmark, question_id, entity, attribute))
        who = "the user" if benchmark == "lme" else entity.capitalize()
        words = " ".join(kept)
        probe = f"What is the current status of {who}'s {words}?"
        assert fm4_value_free(probe, banned), \
            f"FM-4 violation: probe question leaks a value token ({probe!r})"
        reg = {
            "benchmark": benchmark,
            "question_id": question_id,
            "entity": entity,
            "attribute": attribute,
            "description": (f"{words} of {who} (v2.2 C1 natural-hedge expansion "
                            "aux-style key; probe authored value-free before any "
                            "cell run)"),
            "probe_question": probe,
            "created_by": row_id,
            "host_question_ids": sorted(set(host_qids)),
            "fm4_value_free": True,
        }
        self.rows.append(reg)
        return reg


# ---------------------------------------------------------------------------
# Key mapping (existing schema keys; <= 60 cached-friendly key_mapper calls)
# ---------------------------------------------------------------------------

def _existing_key_pool(hosts: list[dict]) -> list[store.CanonicalKey]:
    """Union of the FROZEN/authored candidate keys over the row's hosts, in
    host order, deduplicated by name. Only pools that could make the row
    computable are considered (v22 pair runs carry the pair key; frozen v2
    hosts carry their frozen candidate lists). Source questions that would
    only ever run with our OWN expansion keys need no existing-key mapping —
    a fresh value-free aux-style key serves the estimand identically there."""
    pool: list[store.CanonicalKey] = []
    names: set[str] = set()

    def add(cand: store.CanonicalKey) -> None:
        if cand.name not in names:
            names.add(cand.name)
            pool.append(cand)

    for host in hosts:
        if host["kind"] == "v22_pair_run":
            pid = lme_pair_qids()[host["question_id"]]
            add(pair_key_as_canonical(lme_pair_key_entries()[pid]))
        elif host["kind"] == "v2_lme_cell":
            for c in _lme_v2_candidate_names(host["question_id"]):
                add(c["key"])
        elif host["kind"] == "v2_locomo_cell":
            for c in _locomo_candidate_list(host["question_id"]):
                add(c)
    return pool


def _pseudo_tuple(row: dict, entity: str) -> extract.Tuple:
    """Deterministic key_mapper input for one hedge (cache-friendly: the
    rendered prompt depends only on frozen artifacts). Only entity/attribute/
    value are rendered by store._render_key_mapper_prompt."""
    kept, _ = derive_attribute_tokens(row)
    return extract.Tuple(
        entity=entity,
        attribute="_".join(kept) if kept else "hedged_plan",
        value=row["text"],
        statement_span=row["text"],
        confidence=50,
    )


async def default_mapper(pool: list[store.CanonicalKey], tup: extract.Tuple,
                         row_id: str) -> tuple[Optional[store.CanonicalKey], str, str]:
    return await store._map_tuple_key(tup, pool, trace_tag=f"v22-anchor-keymap-{row_id}")


# ---------------------------------------------------------------------------
# The anchoring pass (pure core; mapper and indexes injectable for tests)
# ---------------------------------------------------------------------------

def _default_entity(benchmark: str, verified: dict) -> str:
    return "user" if benchmark == "lme" else verified["speaker"].strip().lower()


def build_pair_stage_rows() -> list[dict]:
    """The 17 LME pair-stage rows (C1(a)): keys are the authored
    natural_pair_keys; anchors verbatim-verified at the stage turn."""
    rows = []
    for pid in sorted(lme_pair_key_entries()):
        entry = lme_pair_key_entries()[pid]
        pair = lme_pairs_by_id()[pid]
        stage = pair["stage"]
        verified = verify_lme_anchor({
            "question_id": pair["source_question_id"],
            "session_idx": stage["session_idx"],
            "turn_idx": stage["turn_idx"],
            "text": stage["text"],
        })
        assert entry["stage_evidence"]["quote"] in verified["turn_text"], \
            f"{pid}: authored stage evidence not verbatim"
        rows.append({
            "row_id": f"XPAIR-{pid}",
            "row_kind": "lme_pair_stage",
            "benchmark": "lme",
            "source_question_id": pair["source_question_id"],
            "pair_id": pid,
            "anchor": verified["anchor"],
            "text": stage["text"],
            "quality": stage.get("quality"),
            "date": verified["date"],
            "verbatim": {k: verified[k] for k in
                         ("verified", "source_file", "role", "turn_sha256", "date")},
            "key": {
                "entity": entry["entity"],
                "attribute": entry["attribute"],
                "key_source": "natural_pair_key",
                "key_ref": (f"{entry['question_id']}/{entry['entity']}/"
                            f"{entry['attribute']}"),
                "probe_question": entry["probe_question"],
            },
            "hosts": [{
                "kind": "v22_pair_run",
                "question_id": pair["source_question_id"],
                "session_idx": verified["anchor"][0],
                "turn_idx": verified["anchor"][1],
                "base": "results/v2.2", "scheduled": True,
                "key_in_candidates": True,
            }],
            "expected_computable": "scheduled_v22_pair_run",
        })
    assert len(rows) == 17
    return rows


async def anchor_hedge_rows(
    cand_rows: list[dict],
    *,
    mapper: Callable = default_mapper,
    mapper_cap: int = KEY_MAPPER_CALL_CAP,
    registry: Optional[KeyRegistry] = None,
) -> tuple[list[dict], KeyRegistry, dict]:
    """The C1(b) anchoring pass over harvested speculation rows (the callers
    exclude pair rows / rows already in the rule-5 stratum). Returns
    (freeze rows, key registry, summary). Deterministic given the frozen
    inputs; the only LLM stage is the (cached) key_mapper, capped."""
    registry = registry if registry is not None else KeyRegistry()
    rows_out: list[dict] = []
    failures: list[dict] = []
    mapper_calls = 0
    seen_turns: set[tuple] = set()

    ordered = sorted(cand_rows, key=lambda r: (
        r.get("source", ""), r["question_id"], r["session_idx"], r["turn_idx"]))
    for row in ordered:
        benchmark = "locomo" if row.get("source") == "locomo" else "lme"
        row_id = (f"XNAT-{benchmark}-{row['question_id']}"
                  f"-s{row['session_idx']}-t{row['turn_idx']}")
        try:
            verified = (verify_locomo_anchor(row) if benchmark == "locomo"
                        else verify_lme_anchor(row))
        except (AssertionError, KeyError, IndexError) as exc:
            failures.append({"row_id": row_id, "reason": f"verbatim: {exc}"})
            continue
        dedupe = (benchmark, verified["turn_sha256"], row["text"])
        if dedupe in seen_turns:
            failures.append({"row_id": row_id, "reason": "duplicate turn+hedge"})
            continue
        seen_turns.add(dedupe)

        hosts = hedge_hosts(row, verified)
        entity = _default_entity(benchmark, verified)
        pool = _existing_key_pool(hosts)
        key_block: dict[str, Any]
        if pool:
            if mapper_calls >= mapper_cap:
                raise RuntimeError(
                    f"key_mapper call cap {mapper_cap} exceeded at {row_id}")
            mapper_calls += 1
            cand, raw, rationale = await mapper(pool, _pseudo_tuple(row, entity), row_id)
        else:
            cand, raw, rationale = None, "(no existing-key pool)", ""
        if cand is not None:
            key_block = {
                "entity": cand.entity, "attribute": cand.attribute,
                "key_source": "existing_schema",
                "key_ref": cand.name,
                "probe_question": cand.probe_question,
                "mapper": {"called": True, "raw": raw, "rationale": rationale,
                           "pool": [c.name for c in pool]},
            }
        else:
            try:
                host_qids = [h["question_id"] for h in hosts
                             if h["kind"] in ("v22_pair_run", "v22_expansion_run")]
                reg = registry.register(
                    benchmark=benchmark, question_id=row["question_id"],
                    entity=entity, row=row, row_id=row_id, host_qids=host_qids)
            except AssertionError as exc:
                failures.append({"row_id": row_id, "reason": f"new_key: {exc}"})
                continue
            key_block = {
                "entity": reg["entity"], "attribute": reg["attribute"],
                "key_source": "new_aux_style",
                "key_ref": f"v2.2-expansion/{reg['entity']}/{reg['attribute']}",
                "probe_question": reg["probe_question"],
                "mapper": {"called": bool(pool), "raw": raw,
                           "pool": [c.name for c in pool]},
            }

        # per-host computability of THIS key
        key_name = f"{key_block['entity']}/{key_block['attribute']}"
        for host in hosts:
            if host["kind"] in ("v22_pair_run", "v22_expansion_run"):
                # v2.2 candidate lists are built FROM this freeze artifact
                host["key_in_candidates"] = True
            elif host["kind"] == "v2_lme_cell":
                host["key_in_candidates"] = key_name in {
                    c["name"] for c in _lme_v2_candidate_names(host["question_id"])}
            else:  # v2_locomo_cell
                host["key_in_candidates"] = key_name in {
                    c.name for c in _locomo_candidate_list(host["question_id"])}
        if any(h["kind"] == "v22_pair_run" for h in hosts):
            expected = "scheduled_v22_pair_run"
        elif any(h["kind"].startswith("v2_") and h["key_in_candidates"]
                 for h in hosts):
            expected = "existing_v2_timelines"
        elif any(h["kind"] == "v22_expansion_run" for h in hosts):
            expected = "if_source_question_runs"
        else:
            expected = "none"

        out = {
            "row_id": row_id,
            "row_kind": "hedge",
            "benchmark": benchmark,
            "source_question_id": row["question_id"],
            "anchor": verified["anchor"],
            "text": row["text"],
            "about": row.get("about"),
            "quality": row.get("quality"),
            "date": verified["date"],
            "verbatim": {k: v for k, v in verified.items() if k != "turn_text"},
            "key": key_block,
            "hosts": hosts,
            "expected_computable": expected,
        }
        if benchmark == "locomo":
            out["speaker"] = verified["speaker"]
            out["dia_id"] = verified["dia_id"]
        rows_out.append(out)

    summary = {
        "scanned": len(cand_rows),
        "anchored": len(rows_out),
        "failures": failures,
        "mapper_calls": mapper_calls,
        "mapper_call_cap": mapper_cap,
        "new_keys_registered": len(registry.rows),
        "existing_key_rows": sum(
            1 for r in rows_out if r["key"]["key_source"] == "existing_schema"),
        "by_expected_computable": {
            k: sum(1 for r in rows_out if r["expected_computable"] == k)
            for k in sorted({r["expected_computable"] for r in rows_out})},
    }
    return rows_out, registry, summary


def eligible_hedge_candidates() -> tuple[list[dict], dict]:
    """Speculation rows NOT already in the rule-5 stratum and not pair rows
    (LME pair stages enter as row_kind lme_pair_stage; LoCoMo pair stages and
    the 2 schema-anchored censored hedges are already IN the old stratum)."""
    excluded = {"pair_rows": 0, "censored_in_stratum": 0}
    out = []
    for row in speculation_candidates():
        if row.get("pair_id"):
            excluded["pair_rows"] += 1
            continue
        tag = (row["question_id"], row["session_idx"], row["turn_idx"])
        if row.get("source") == "locomo" and tag in _CENSORED_IN_STRATUM_HARVEST:
            excluded["censored_in_stratum"] += 1
            continue
        out.append(row)
    return out, excluded


async def run_anchor() -> dict:
    pair_rows = build_pair_stage_rows()
    hedge_cands, excluded = eligible_hedge_candidates()
    hedge_rows, registry, summary = await anchor_hedge_rows(hedge_cands)
    freeze_rows = pair_rows + hedge_rows

    n_new = len(freeze_rows)
    summary_full = {
        "id": "V2.2-C1-ANCHOR-FREEZE",
        "rule": ("protocol_changelog.md 'Protocol v2.2' C1(b): hedge turn "
                 "verbatim-verified at its (source, question/conv, session, "
                 "turn) anchor AND attribute maps to an existing schema key "
                 "or a newly registered aux-style key with a value-free "
                 "probe question (FM-4-checked); list FREEZES before any "
                 "cell run"),
        "seed": SEED,
        "seed_note": ("recorded-not-consumed: no random draw exists — row "
                      "order is a lexicographic sort, key derivation is "
                      "rule-based, key mapping is the frozen key_mapper "
                      "prompt behind the llm.py cache"),
        "excluded_already_in_stratum": excluded,
        "pair_stage_rows": len(pair_rows),
        "hedge_scan": summary,
        "freeze_rows_total": n_new,
        "target_new_rows": TARGET_NEW_ROWS,
        "target_met": n_new >= TARGET_NEW_ROWS,
        "old_stratum_n": 20,
        "inputs_sha256": {
            "natural_candidates_full.jsonl": _sha256(CANDIDATES_PATH.read_text()),
            "natural_pairs_v2.jsonl": _sha256(PAIRS_PATH.read_text()),
            "key_schema_v2_lme.json": _sha256(LME_SCHEMA_PATH.read_text()),
        },
        "outputs": {
            "freeze_list": str(EXPANSION_PATH.relative_to(ROOT)),
            "expansion_keys": str(EXPANSION_KEYS_PATH.relative_to(ROOT)),
        },
        "harness_corrections": {
            "A1": "label.match_tuple_to_span preferred_key tie-break "
                  "(natres.match_staged_tuple passes the authored pair key)",
            "A3": "pair keys + expansion keys registered in the v2.2 "
                  "key-mapper candidate lists (expand_natural.py jobs)",
        },
    }
    _write_jsonl(EXPANSION_PATH, freeze_rows)
    _write_jsonl(EXPANSION_KEYS_PATH, registry.rows)
    ANCHOR_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANCHOR_SUMMARY_PATH.write_text(
        json.dumps(summary_full, indent=1, ensure_ascii=False))
    return summary_full


# ---------------------------------------------------------------------------
# jobs — v2.2 CellJob construction (C1(a) + A3 fix)
# ---------------------------------------------------------------------------

def load_freeze_rows() -> list[dict]:
    rows = _read_jsonl(EXPANSION_PATH)
    if not rows:
        raise SystemExit(f"missing freeze list {EXPANSION_PATH} — run "
                         "`expand_natural.py anchor` first (C1: the row list "
                         "freezes before any cell run)")
    return rows


def load_expansion_keys() -> list[dict]:
    return _read_jsonl(EXPANSION_KEYS_PATH)


def candidates_for_substrate(qid: str,
                             expansion_keys: Optional[list[dict]] = None,
                             ) -> list[store.CanonicalKey]:
    """The v2.2 candidate list of one expansion substrate: the authored
    natural-pair key(s) for pair substrates (the A3 fix — the audited v2 lists
    never contained them) + every newly registered expansion key hosted on the
    substrate. Blind (no values), FM-6 unmapped logging applies unchanged."""
    keys = expansion_keys if expansion_keys is not None else load_expansion_keys()
    out: list[store.CanonicalKey] = []
    names: set[str] = set()
    pid = lme_pair_qids().get(qid)
    if pid is not None:
        cand = pair_key_as_canonical(lme_pair_key_entries()[pid])
        names.add(cand.name)
        out.append(cand)
    for row in keys:
        if qid != row["question_id"] and qid not in row.get("host_question_ids", []):
            continue
        cand = expansion_key_as_canonical(row)
        if cand.name not in names:
            names.add(cand.name)
            out.append(cand)
    return out


def build_jobs(qids: Optional[list[str]] = None) -> list[arms.CellJob]:
    """CellJob rows for the v2.2 C1 cell runs. Default: the 17 LME pair
    substrates (s_cleaned sources, UNPERTURBED, no probe rows — the estimand
    reads gate logs and store timelines, not manifest probes). Extra qids must
    be freeze-listed hedge source questions."""
    expansion_keys = load_expansion_keys() if EXPANSION_KEYS_PATH.exists() else []
    if qids is None:
        qids = sorted(lme_pair_qids())
    jobs = []
    for qid in qids:
        cands = candidates_for_substrate(qid, expansion_keys)
        if not cands:
            raise ValueError(
                f"{qid}: no v2.2 candidate keys (not a pair substrate and no "
                "expansion key hosted here) — not a C1 substrate")
        jobs.append(arms.CellJob(
            question_id=qid,
            substrate=test1_eval.s_cleaned_substrate(qid),
            key_schema_entry=None,       # candidates are the aux_keys alone
            manifest_rows=[],            # natural stream, nothing spliced
            probe_rows=[],               # deterministic reads happen in rule5
            aux_keys=cands,
        ))
    return jobs


def jobs_manifest(jobs: list[arms.CellJob]) -> dict:
    rows = []
    for job in jobs:
        sub = job.substrate
        rows.append({
            "question_id": job.question_id,
            "pair_id": lme_pair_qids().get(job.question_id),
            "source": "s_cleaned",
            "substrate_build": ("test1_eval.s_cleaned_substrate — the "
                                "natural-selection replay path (V2-F1 §2.3)"),
            "n_sessions": len(sub.sessions),
            "n_user_turns": sum(1 for s in sub.sessions for t in s
                                if t["role"] == "user"),
            "candidate_keys": [c.name for c in job.aux_keys],
            "perturbed": False,
        })
    return {
        "id": "V2.2-C1-CELL-JOBS",
        "cells": "all four (mem0/zep x ungated/gated), arms.run_cells",
        "out_dir": str(V22_DIR.relative_to(ROOT)),
        "harness_corrections": {
            "A1": "label.match_tuple_to_span preferred_key tie-break",
            "A3": "pair keys registered in the key-mapper candidate lists "
                  "for their substrates (candidate_keys below)",
        },
        "freeze_list": str(EXPANSION_PATH.relative_to(ROOT)),
        "jobs": rows,
    }


def _c1_run_guard(run_qids: list[str], call_budget: Optional[int]):
    """The B-1 budget gate for `jobs --run`: requires the COMBINED forecast's
    budget file (results/v2.2/budget.json, forecast_v22.py), refuses
    substrates outside the planned C1 set (scheduled pair substrates + the
    greedy-selected additions), and returns the shared envelope guard that
    this run decrements (fails loudly when the combined envelope is
    exhausted). Import is function-local: forecast_v22 imports this module."""
    import forecast_v22

    budget = forecast_v22.load_budget()
    allowed = set(budget["c1"]["scheduled_question_ids"]) | {
        a["question_id"] for a in budget["c1"]["additions"]}
    outside = sorted(q for q in run_qids if q not in allowed)
    if outside:
        raise SystemExit(
            f"substrates {outside} are outside the combined forecast's "
            "planned C1 set (scheduled pair runs + greedy additions) — "
            "re-run forecast_v22.py to change scope (B-1/B-2 pinned rule)")
    return forecast_v22.SharedCallBudget("c1_cell_runs",
                                         local_cap=call_budget)


def cmd_jobs(args) -> None:
    qids = [q.strip() for q in args.qids.split(",") if q.strip()] if args.qids else None
    if args.run:
        load_freeze_rows()  # freeze-before-run discipline (C1)
    jobs = build_jobs(qids)
    manifest = jobs_manifest(jobs)
    JOBS_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    JOBS_MANIFEST_PATH.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    total_turns = sum(r["n_user_turns"] for r in manifest["jobs"])
    print(f"[jobs] {len(jobs)} CellJob rows ({total_turns} user turns/family) "
          f"-> {JOBS_MANIFEST_PATH}")
    if not args.run:
        print("[jobs] dry build only (pass --run to execute the cells — "
              "orchestrator step)")
        return
    guard = _c1_run_guard([j.question_id for j in jobs], args.call_budget)
    families = tuple(f.strip() for f in args.families.split(",") if f.strip())
    policies = tuple(p.strip() for p in args.policies.split(",") if p.strip())
    guard.install()
    try:
        summary = asyncio.run(arms.run_cells(
            jobs, families=families, policies=policies, out_dir=V22_DIR))
    finally:
        guard.uninstall()
    print(json.dumps({"cells": {c: {k: v for k, v in info.items()
                                    if k != "diagnostics"}
                                for c, info in summary["cells"].items()},
                      "extraction_identity": summary["extraction_identity"],
                      "fresh_calls": guard.fresh},
                     ensure_ascii=False, indent=1))


# ---------------------------------------------------------------------------
# rule5 — the expanded §8-rule-5 check (import ncr5, don't copy)
# ---------------------------------------------------------------------------

def _base_dir(host: dict) -> Path:
    return V22_DIR if host["base"] == "results/v2.2" else V2_DIR


def _timelines_exist(host: dict) -> bool:
    base = _base_dir(host)
    return all((base / cell_dir / "timelines" / f"{host['question_id']}.json").exists()
               for cell_dir in ncr5.CELL_DIRS.values())


def resolve_row_host(freeze_row: dict) -> tuple[Optional[dict], Optional[str]]:
    """First host (freeze-list priority order) whose four cell timelines exist
    AND whose candidate list contains the row's key. Returns (host, reason)."""
    reasons = []
    for host in freeze_row["hosts"]:
        if not host.get("key_in_candidates"):
            reasons.append(f"{host['kind']}/{host['question_id']}: key not in "
                           "frozen candidate list")
            continue
        if not _timelines_exist(host):
            reasons.append(f"{host['kind']}/{host['question_id']}: cell "
                           "timelines missing")
            continue
        return host, None
    return None, "; ".join(reasons) if reasons else "no hosts"


def expansion_estimand_rows() -> tuple[list[dict], list[dict]]:
    """ncr5-shaped rows for the frozen expansion rows whose host timelines
    exist. Returns (rows, excluded)."""
    rows, excluded = [], []
    for fr in load_freeze_rows():
        host, reason = resolve_row_host(fr)
        if host is None:
            excluded.append({"row_id": fr["row_id"],
                             "expected_computable": fr["expected_computable"],
                             "reason": reason})
            continue
        row = {
            "row_id": fr["row_id"],
            "row_kind": fr["row_kind"],
            "sample_id": host["question_id"],
            "entity": fr["key"]["entity"],
            "attribute": fr["key"]["attribute"],
            "anchor": [host["session_idx"], host["turn_idx"]],
            "host_kind": host["kind"],
            "key_source": fr["key"]["key_source"],
            "_base": str(_base_dir(host)),
        }
        if fr["benchmark"] == "lme":
            # LME replays are single-store (gated_store.DEFAULT_STORE)
            row["store_id"] = "main"
        rows.append(row)
    return rows, excluded


def _boot_block(out_rows: list[dict], field: str) -> dict:
    return stats.cluster_bootstrap_mean(
        [r[field] for r in out_rows], [r["cluster"] for r in out_rows],
        n_boot=N_BOOT, seed=BOOT_SEED, bounds=(-2, 2))


def _z_distribution(out_rows: list[dict]) -> dict:
    return {str(v): sum(r["z_primary"] == v for r in out_rows)
            for v in sorted({r["z_primary"] for r in out_rows})}


async def run_rule5(judge_cap: int = JUDGE_CALL_CAP) -> dict:
    scorer = ncr5.EquivalenceScorer(cap=judge_cap)

    # --- old n=20 block, reproduced unchanged (hard-asserted) --------------
    old_rows = ncr5.build_rows()
    old_out = [await ncr5.score_row(row, scorer) for row in old_rows]
    boot_old = _boot_block(old_out, "z_primary")
    frozen = json.loads(ncr5.METRICS_PATH.read_text())[
        "natural_consistency_check_rule5"]
    reproduced = (
        boot_old["point"] == frozen["delta_nat"]
        and [boot_old["lo"], boot_old["hi"]] == frozen["ci"]
        and boot_old["n"] == frozen["n"]
        and boot_old["n_clusters"] == frozen["n_clusters"]
        and _z_distribution(old_out) == frozen["z_distribution"])
    assert reproduced, (
        "old n=20 block did NOT reproduce the frozen "
        f"results/v2/metrics.json numbers: got point={boot_old['point']}, "
        f"ci=[{boot_old['lo']}, {boot_old['hi']}], "
        f"z={_z_distribution(old_out)} vs frozen {frozen['delta_nat']}, "
        f"{frozen['ci']}, {frozen['z_distribution']}")

    # --- expanded block (same estimand, same decision rule) ----------------
    exp_rows, excluded = expansion_estimand_rows()
    exp_out = [await ncr5.score_row(row, scorer, base_dir=Path(row["_base"]))
               for row in exp_rows]
    combined = old_out + exp_out
    boot_exp = _boot_block(combined, "z_primary")
    boot_exp_spec = _boot_block(combined, "z_speculative_only")

    metrics = json.loads(ncr5.METRICS_PATH.read_text())
    injected = metrics["benchmarks"]["lme"]["H-INT"]
    check = metrics_v2.natural_consistency_check(
        {"point": injected["point"], "ci": injected["ci"]},
        {"point": boot_exp["point"], "ci": [boot_exp["lo"], boot_exp["hi"]]})

    freeze_rows = load_freeze_rows()
    result = {
        "id": "NAT-CONSISTENCY-RULE5-EXPANDED",
        "protocol": "v2.2 C1 (protocol_changelog.md); estimand/decision rule "
                    "identical to §8 rule 5",
        "old_n20": {
            "delta_nat": boot_old["point"],
            "ci": [boot_old["lo"], boot_old["hi"]],
            "n": boot_old["n"],
            "n_clusters": boot_old["n_clusters"],
            "z_distribution": _z_distribution(old_out),
            "reproduced_unchanged": True,
            "frozen_reference": "results/v2/metrics.json "
                                "natural_consistency_check_rule5",
        },
        "expanded": {
            "delta_nat": boot_exp["point"],
            "ci": [boot_exp["lo"], boot_exp["hi"]],
            "n": boot_exp["n"],
            "n_clusters": boot_exp["n_clusters"],
            "ci_kind": "cluster_bootstrap_by_conversation",
            "seed": BOOT_SEED,
            "n_boot": N_BOOT,
            "z_distribution": _z_distribution(combined),
            "injected_reference": {"id": "H-INT", "point": injected["point"],
                                   "ci": injected["ci"]},
            "sign_opposes_hint": check.get("sign_opposes", False),
            "natural_ci_excludes_injected_point":
                check.get("natural_ci_excludes_injected_point"),
            "downgrade_triggered": check.get("downgrade_to_injected_scope", False),
            "sensitivity_speculative_classified_only": {
                "delta_nat": boot_exp_spec["point"],
                "ci": [boot_exp_spec["lo"], boot_exp_spec["hi"]],
            },
            "coverage": {
                "old_stratum_rows": len(old_out),
                "frozen_expansion_rows": len(freeze_rows),
                "expansion_rows_computable": len(exp_out),
                "expansion_rows_excluded": len(excluded),
                "excluded_detail": excluded,
                "by_row_kind": {
                    k: sum(1 for r in exp_out if r["row_kind"] == k)
                    for k in sorted({r["row_kind"] for r in exp_out})},
                "target_n": 80,
                "target_met": boot_exp["n"] >= 80,
            },
        },
        "judge_calls": {"cap": judge_cap, "used": scorer.judge_calls,
                        "fresh": scorer.fresh_judge_calls,
                        "prompt": "prompts/equivalence_judge.txt (frozen)"},
        "provenance": {
            "freeze_list_sha256": _sha256(EXPANSION_PATH.read_text()),
            "anchor_seed": SEED,
            "model": llm.MODEL,
        },
    }
    all_rows = ([{**r, "block": "old_n20"} for r in old_out]
                + [{**r, "block": "expansion"} for r in exp_out])
    _write_jsonl(RULE5_ROWS_PATH, all_rows)
    RULE5_PATH.parent.mkdir(parents=True, exist_ok=True)
    RULE5_PATH.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    return result


# ---------------------------------------------------------------------------
# smoke — anchor 3 rows end-to-end + build ONE CellJob (no cell run)
# ---------------------------------------------------------------------------

def _smoke_pick_rows() -> list[dict]:
    """Deterministic 3-row smoke sample exercising all three anchoring paths:
    one LoCoMo row (existing-key pool -> mapper), one LME row with a frozen
    v2 host (pool -> mapper), one LME row with no pool (new aux-style key,
    no LLM). <= 2 mapper calls + 1 pool-less registration."""
    cands, _ = eligible_hedge_candidates()
    ordered = sorted(cands, key=lambda r: (
        r.get("source", ""), r["question_id"], r["session_idx"], r["turn_idx"]))
    picked: list[dict] = []

    def first(pred) -> Optional[dict]:
        for row in ordered:
            if any(row is p for p in picked):
                continue
            if pred(row):
                return row
        return None

    loco = first(lambda r: r.get("source") == "locomo")
    v2_hosted = first(lambda r: r.get("source") != "locomo" and (
        test1_eval.s_cleaned_substrate(r["question_id"])
        .sessions[r["session_idx"]][r["turn_idx"]]["content"]
        in v2_oracle_turn_index()))
    plain = first(lambda r: r.get("source") != "locomo" and (
        test1_eval.s_cleaned_substrate(r["question_id"])
        .sessions[r["session_idx"]][r["turn_idx"]]["content"]
        not in v2_oracle_turn_index())
        and r["question_id"] not in lme_pair_qids())
    for row in (loco, v2_hosted, plain):
        if row is not None:
            picked.append(row)
    assert len(picked) == 3
    return picked


async def run_smoke() -> None:
    picked = _smoke_pick_rows()
    rows, registry, summary = await anchor_hedge_rows(picked, mapper_cap=10)
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    _write_jsonl(SMOKE_DIR / "anchor_rows.jsonl", rows)
    _write_jsonl(SMOKE_DIR / "expansion_keys.jsonl", registry.rows)
    print("=== smoke anchor (3 rows) ===")
    for r in rows:
        print(json.dumps({
            "row_id": r["row_id"], "benchmark": r["benchmark"],
            "verbatim": r["verbatim"]["verified"],
            "key": {k: r["key"][k] for k in ("key_source", "key_ref",
                                             "probe_question")},
            "hosts": [(h["kind"], h["question_id"], h.get("key_in_candidates"))
                      for h in r["hosts"]],
            "expected_computable": r["expected_computable"],
        }, ensure_ascii=False))
    print(json.dumps({"summary": summary}, ensure_ascii=False))

    qid = sorted(lme_pair_qids())[0]
    job = build_jobs([qid])[0]
    sub = job.substrate
    print("=== smoke CellJob (no cell run) ===")
    print(json.dumps({
        "question_id": job.question_id,
        "pair_id": lme_pair_qids()[qid],
        "n_sessions": len(sub.sessions),
        "n_user_turns": sum(1 for s in sub.sessions for t in s
                            if t["role"] == "user"),
        "candidate_keys": [c.name for c in job.aux_keys],
        "key_schema_entry": job.key_schema_entry,
        "manifest_rows": len(job.manifest_rows),
        "probe_rows": job.probe_rows,
    }, ensure_ascii=False))
    print(f"smoke complete: {summary['mapper_calls']} key_mapper LLM call(s); "
          f"smoke artifacts -> {SMOKE_DIR}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="v2.2 C1 natural-hedge stratum expansion")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("anchor", help="frozen anchoring pass -> freeze list")
    jobs_p = sub.add_parser("jobs", help="v2.2 CellJob construction (C1(a))")
    jobs_p.add_argument("--qids", default=None,
                        help="comma-separated substrate qids "
                             "(default: the 17 LME pair substrates)")
    jobs_p.add_argument("--families", default=",".join(arms.FAMILIES))
    jobs_p.add_argument("--policies", default=",".join(arms.POLICIES))
    jobs_p.add_argument("--run", action="store_true",
                        help="execute arms.run_cells (orchestrator step; "
                             "requires the combined forecast_v22 budget "
                             "file and decrements the shared envelope)")
    jobs_p.add_argument("--call-budget", type=int, default=None,
                        help="additional local cap on fresh LLM calls "
                             "(the shared v2.2 envelope always applies)")
    rule5_p = sub.add_parser("rule5", help="expanded §8-rule-5 check")
    rule5_p.add_argument("--judge-cap", type=int, default=JUDGE_CALL_CAP)
    sub.add_parser("smoke", help="anchor 3 rows + build 1 CellJob (no run)")
    args = parser.parse_args()

    if args.cmd == "anchor":
        summary = asyncio.run(run_anchor())
        print(json.dumps({k: summary[k] for k in
                          ("pair_stage_rows", "hedge_scan", "freeze_rows_total",
                           "target_new_rows", "target_met")},
                         ensure_ascii=False, indent=1))
        print(f"-> {EXPANSION_PATH}\n-> {EXPANSION_KEYS_PATH}\n"
              f"-> {ANCHOR_SUMMARY_PATH}")
    elif args.cmd == "jobs":
        cmd_jobs(args)
    elif args.cmd == "rule5":
        # rule-5 judging spend is part of the C1 allocation: decrement the
        # shared v2.2 envelope when the combined budget file exists (it is
        # required for cell runs; rule5 also runs standalone pre-budget).
        import forecast_v22
        guard = None
        if forecast_v22.BUDGET_PATH.exists():
            guard = forecast_v22.SharedCallBudget("c1_rule5_judging")
            guard.install()
        try:
            result = asyncio.run(run_rule5(args.judge_cap))
        finally:
            if guard is not None:
                guard.uninstall()
        print(json.dumps({"old_n20": result["old_n20"],
                          "expanded": {k: v for k, v in result["expanded"].items()
                                       if k != "coverage"},
                          "coverage": {k: v for k, v in
                                       result["expanded"]["coverage"].items()
                                       if k != "excluded_detail"},
                          "judge_calls": result["judge_calls"]},
                         ensure_ascii=False, indent=1))
        print(f"-> {RULE5_PATH}\n-> {RULE5_ROWS_PATH}")
    elif args.cmd == "smoke":
        asyncio.run(run_smoke())


if __name__ == "__main__":
    main()
