"""Gold-labeling pipeline: item packaging, 3-effort labeler calls, majority vote,
escalation, adjudication resume, and edge-case tag aggregation.

Implements protocol_spec.md v1.1 sections 2.1-2.2 and the pinned design-review
rules B3 (stored-value provisioning), M5 (turn->tuple), M8 (tag aggregation).
Cross-module contract: INTERFACES.md "label.py + classify.py".

Pure-logic functions here take plain data structures (duck-typed Substrate-like
objects with .sessions / .dates / .session_order) so they are unit-testable
without other owners' modules; runners (test0_calibrate.py, test1_eval.py) do
the data.py / inject.py wiring.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from pathlib import Path
from typing import Any, Optional

import llm

logger = logging.getLogger(__name__)

from paths import PROMPTS_DIR, RESULTS_DIR
LABELS_DIR = RESULTS_DIR / "labels"
ESCALATION_QUEUE_PATH = LABELS_DIR / "escalation_queue.jsonl"
ESCALATION_QUEUE_META_PATH = LABELS_DIR / "escalation_queue_meta.json"
ADJUDICATIONS_PATH = LABELS_DIR / "adjudications.jsonl"

# R2-B-3 (FM-14 / protocol 2.2 blindness): the adjudicator receives ONLY the
# item, the rubric, and the three anonymized labels+rationales. Every
# bookkeeping field in the escalation queue is therefore '_'-prefixed (the
# same display-safety convention as naturalness_items.jsonl) — _item_id in
# particular encodes the injection type for injected items and the `nat-`
# prefix for natural ones, i.e. the construction label.
ESCALATION_BLINDING_NOTE = (
    "adjudicator runner: show ONLY packaged_item, labels, and escalation_reasons; "
    "every '_'-prefixed field is hidden provenance bookkeeping (protocol 2.2 / FM-14) "
    "that must never be shown to the adjudicator. Adjudicator must be Claude-family, "
    "one item per call, given the full rubric; write adjudications.jsonl rows as "
    "{item_id: <the queue row's _item_id>, class, rationale, edge_case_flags, model_id}."
)

CLASSES = ("speculative", "completed_event", "explicit_correction", "other")
COMMIT_CLASSES = ("completed_event", "explicit_correction")
RULE_FLAGS = frozenset("abcdefg")
# Protocol section 2.2 / pinned decision P6: the three labeler calls vary
# reasoning effort as the diversity lever under the standardization constraint.
LABELER_EFFORTS = ("low", "medium", "high")
SEED = 20260702

RUBRIC_PATH = PROMPTS_DIR / "labeler_rubric.txt"
ITEM_PACKAGING_PATH = PROMPTS_DIR / "item_packaging.txt"


# ---------------------------------------------------------------------------
# Rubric loading: comment stripping + machine-readable anchor header (P9/C4)
# ---------------------------------------------------------------------------

def strip_comment_lines(text: str) -> str:
    """Drop the '#'-prefixed header lines (the rubric header says the runner
    may strip them before use as a system prompt). Body text is untouched."""
    return "\n".join(line for line in text.splitlines() if not line.startswith("#"))


def parse_rubric_anchors(text: str) -> list[tuple[str, int, int]]:
    """Parse the BEGIN-ANCHORS/END-ANCHORS block of the frozen rubric.

    Returns (question_id, session_idx, turn_idx) triples of every [real]
    rubric example; eval-set construction must hold these turns out BY ID
    (after content-hash session dedupe), never by surface text.
    """
    anchors: list[tuple[str, int, int]] = []
    in_block = False
    for line in text.splitlines():
        stripped = line.lstrip("#").strip()
        if stripped == "BEGIN-ANCHORS":
            in_block = True
            continue
        if stripped == "END-ANCHORS":
            in_block = False
            continue
        if in_block and stripped:
            row = json.loads(stripped)
            anchors.append((row["question_id"], row["session_idx"], row["turn_idx"]))
    return anchors


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text()


def labeler_system_prompt() -> str:
    return strip_comment_lines(RUBRIC_PATH.read_text())


def rubric_anchors() -> list[tuple[str, int, int]]:
    return parse_rubric_anchors(RUBRIC_PATH.read_text())


# ---------------------------------------------------------------------------
# Substrate helpers (duck-typed per data.py's Substrate: .sessions keeps the
# ORIGINAL haystack order, so sessions[i] is original session_idx i;
# .session_order lists original indices sorted chronologically)
# ---------------------------------------------------------------------------

def get_turn(substrate: Any, orig_session_idx: int, turn_idx: int) -> dict:
    return substrate.sessions[orig_session_idx][turn_idx]


def preceding_turns(substrate: Any, orig_session_idx: int, turn_idx: int, k: int = 2) -> list[dict]:
    """Up to k turns immediately preceding the item turn, within its session,
    in chronological order (protocol section 2.1: 'up to 2 preceding turns')."""
    session = substrate.sessions[orig_session_idx]
    lo = max(0, turn_idx - k)
    return [dict(role=t["role"], content=t["content"]) for t in session[lo:turn_idx]]


def chronological_user_turns(substrate: Any) -> list[tuple[int, int]]:
    """(orig_session_idx, turn_idx) of every user turn in chronological order."""
    out: list[tuple[int, int]] = []
    for orig_idx in substrate.session_order:
        for ti, turn in enumerate(substrate.sessions[orig_idx]):
            if turn["role"] == "user":
                out.append((orig_idx, ti))
    return out


def session_content_hash(session: list[dict]) -> str:
    """Content hash used for the cross-question session dedupe (protocol
    section 2.3; dataset_analysis section 3). Hashes only (role, content)."""
    import hashlib

    payload = json.dumps([[t["role"], t["content"]] for t in session], ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Item packaging (protocol section 2.1) -- identical for labelers + classifier
# ---------------------------------------------------------------------------

def format_turn_context(preceding: list[dict], turn_text: str) -> str:
    """Uniform turn+context block, also used verbatim for naturalness-gate
    display items so surface formatting cannot signal provenance."""
    if preceding:
        ctx = "\n".join(f"[{t['role']}] {t['content']}" for t in preceding)
    else:
        ctx = "(none)"
    return f"Preceding turns:\n{ctx}\n\nSource user turn:\n{turn_text}"


def format_item_prompt(item: dict) -> str:
    """Package one statement tuple per protocol section 2.1: source turn
    verbatim, up to 2 preceding turns, target key, asserted value, and the
    stored value when one exists (B3). No provenance, no construction metadata.
    """
    template = ITEM_PACKAGING_PATH.read_text()
    stored = item.get("stored_value")
    fields = {
        "{turn_context}": format_turn_context(item["preceding_turns"], item["turn_text"]),
        "{entity}": item["entity"],
        "{attribute}": item["attribute"],
        "{asserted_value}": str(item["asserted_value"]),
        "{stored_value}": "(none)" if stored is None else str(stored),
    }
    out = template
    for key, val in fields.items():
        out = out.replace(key, val)
    return out.rstrip("\n")


# ---------------------------------------------------------------------------
# Turn -> tuple matching (M5) and B3 stored-value replay
# ---------------------------------------------------------------------------

def normalize_key_part(s: str) -> str:
    """snake_case normalization for (entity, attribute) matching."""
    out = "".join(c if c.isalnum() else "_" for c in s.strip().lower())
    while "__" in out:
        out = out.replace("__", "_")
    out = out.strip("_")
    if out.startswith("the_"):
        out = out[4:]
    return out


def key_matches(tup: dict, entity: str, attribute: str) -> bool:
    return (
        normalize_key_part(tup["entity"]) == normalize_key_part(entity)
        and normalize_key_part(tup["attribute"]) == normalize_key_part(attribute)
    )


def _norm_value(v: Any) -> str:
    return " ".join("".join(c.lower() if c.isalnum() else " " for c in str(v)).split())


def match_tuple_to_key(tuples: list[dict], entity: str, attribute: str,
                       *, value: Any = None, aside_text: Optional[str] = None,
                       ) -> tuple[Optional[dict], int, str]:
    """M5: only the tuple matching the target statement enters the eval set.

    Three deterministic tiers (changelog entry: M5 matcher refinement — the
    v1 exact-key matcher misclassified ~130 well-extracted injected rows as
    extraction misses because the extractor words attributes freely, e.g.
    houseplant_type vs houseplant_species, and frames speculation as
    plan-shaped attributes like planned_bike_count_after_addition):

    1. "key":        normalized (entity, attribute) equality.
    2. "span+value": statement_span overlaps the known injected aside text,
                     entity matches, and the value is equivalent (normalized
                     equality or containment either way).
    3. "span":       statement_span overlaps the aside and entity matches;
                     highest confidence (paraphrased value wording).

    Returns (matched tuple, number of tier-matches, match_method). Zero
    matches = extraction miss (caller logs and excludes/tops-up). Multiple
    tuples in the winning tier: highest confidence (count reported).
    """
    tiers: list[tuple[str, list[dict]]] = [
        ("key", [t for t in tuples if key_matches(t, entity, attribute)])]
    if aside_text:
        in_span = [
            t for t in tuples
            if span_overlap(t.get("statement_span", ""), aside_text) >= SPAN_MATCH_THRESHOLD
            and normalize_key_part(t["entity"]) == normalize_key_part(entity)]
        if value is not None:
            vn = _norm_value(value)
            tiers.append(("span+value", [
                t for t in in_span
                if _norm_value(t["value"]) == vn
                or vn in _norm_value(t["value"]) or _norm_value(t["value"]) in vn]))
        tiers.append(("span", in_span))
    for method, matches in tiers:
        if matches:
            best = max(matches, key=lambda t: t["confidence"])
            return best, len(matches), method
    return None, 0, "miss"


def _tokens(s: str) -> set[str]:
    return {w for w in "".join(c.lower() if c.isalnum() else " " for c in s).split() if w}


def span_overlap(a: str, b: str) -> float:
    """Token containment overlap in [0,1] between two text spans."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


SPAN_MATCH_THRESHOLD = 0.5


def match_tuple_to_span(tuples: list[dict], candidate_text: str,
                        *, preferred_key: Optional[tuple[str, str]] = None,
                        ) -> tuple[Optional[dict], float]:
    """M5 for natural items: harvest rows carry no formal (entity, attribute),
    so the item tuple is the extractor tuple whose statement_span overlaps the
    harvested candidate text; the tuple's own key becomes the target key.

    ``preferred_key``: optional (entity, attribute). Protocol v2.2 C1 harness
    correction (audited defect A1, results/v2/natres/failure_audit.md §2.1:
    first-tuple-wins on tied overlap scores let short background-fact spans
    beat the authored pair key's tuple). When given, a tuple whose own key
    matches ``preferred_key`` (label.key_matches normalization) WINS TIES at
    the best overlap score; scores themselves are unchanged, and a strictly
    higher-scoring tuple still wins regardless of key. Default ``None``
    preserves the pre-v2.2 first-tuple-wins behavior exactly.
    """
    best, best_score = None, 0.0
    for t in tuples:
        score = span_overlap(t["statement_span"], candidate_text)
        if score > best_score:
            best, best_score = t, score
        elif (
            preferred_key is not None
            and best is not None
            and score == best_score
            and not key_matches(best, *preferred_key)
            and key_matches(t, *preferred_key)
        ):
            best = t  # tie-break only: same score, preferred key wins
    if best_score < SPAN_MATCH_THRESHOLD:
        return None, best_score
    return best, best_score


def stored_value_replay(
    turn_order: list[tuple[int, int]],
    item_turn: tuple[int, int],
    tuples_at: dict[tuple[int, int], list[dict]],
    entity: str,
    attribute: str,
) -> Optional[str]:
    """B3: frozen-extractor replay in chronological order up to, NOT including,
    the item turn; the latest prior tuple on the target key is the stored value
    (None when no prior value maps -> rule 1.5d no-stored-value branch).

    turn_order: chronological (orig_session_idx, turn_idx) user turns.
    tuples_at: extractor tuples keyed by (orig_session_idx, turn_idx).
    """
    if item_turn not in turn_order:
        raise ValueError(f"item turn {item_turn} not found in substrate turn order")
    stored: Optional[str] = None
    for turn in turn_order:
        if turn == item_turn:
            break
        for tup in tuples_at.get(turn, []):
            if key_matches(tup, entity, attribute):
                stored = tup["value"]
    return stored


# ---------------------------------------------------------------------------
# Labeler calls, majority vote, escalation (protocol section 2.2)
# ---------------------------------------------------------------------------

def normalize_class(raw: str) -> str:
    cls = raw.strip().lower().replace("-", "_").replace(" ", "_")
    if cls not in CLASSES:
        raise ValueError(f"labeler returned invalid class {raw!r}")
    return cls


def normalize_flag(raw: str) -> str:
    flag = str(raw).strip().lower().strip("()").strip()
    return flag


def majority_vote(classes: list[str]) -> Optional[str]:
    """2/3 majority; None on a 3-way split."""
    for cls in set(classes):
        if classes.count(cls) >= 2:
            return cls
    return None


def escalation_reasons(calls: list[dict]) -> list[str]:
    """Protocol section 2.2 escalation conditions: no majority (3-way split),
    >=2 labelers uncertain, or any flag not covered by rules (a)-(g)."""
    reasons = []
    if majority_vote([c["class"] for c in calls]) is None:
        reasons.append("no_majority")
    if sum(1 for c in calls if c["uncertain"]) >= 2:
        reasons.append("uncertain_majority")
    for call in calls:
        if any(normalize_flag(f) not in RULE_FLAGS for f in call["edge_case_flags"]):
            reasons.append("uncovered_edge_case")
            break
    return reasons


def aggregate_edge_case_flags(calls: list[dict], adjudication: Optional[dict] = None) -> list[str]:
    """M8 (natural items): majority vote across the three labeler calls'
    edge_case_flags; on escalated items the adjudicator's flags override.
    (Injected items use construction tags from the manifest instead.)"""
    if adjudication is not None:
        return sorted(normalize_flag(f) for f in adjudication.get("edge_case_flags", []))
    counts: dict[str, int] = {}
    for call in calls:
        for flag in {normalize_flag(f) for f in call["edge_case_flags"]}:
            counts[flag] = counts.get(flag, 0) + 1
    return sorted(f for f, n in counts.items() if n >= 2)


async def label_item(item: dict) -> dict:
    """Three rubric calls at efforts low/medium/high (frozen labeler_rubric.txt,
    comment header stripped), majority vote, escalation reasons.

    Side-effect free with respect to the escalation queue: runners call
    append_escalations() on the results so smoke tests stay clean.
    """
    system = labeler_system_prompt()
    user = format_item_prompt(item)
    raws = await asyncio.gather(*[
        llm.call_llm(
            system,
            user,
            trace_tag=f"label-{item['item_id']}-{effort}",
            reasoning_effort=effort,
        )
        for effort in LABELER_EFFORTS
    ])
    calls = []
    for effort, raw in zip(LABELER_EFFORTS, raws):
        calls.append({
            "effort": effort,
            "class": normalize_class(raw["class"]),
            "rationale": raw["rationale"],
            "edge_case_flags": [str(f) for f in raw.get("edge_case_flags", [])],
            "uncertain": bool(raw["uncertain"]),
        })
    reasons = escalation_reasons(calls)
    return {
        "item_id": item["item_id"],
        "calls": calls,
        "vote": majority_vote([c["class"] for c in calls]),
        "escalate": bool(reasons),
        "escalation_reasons": reasons,
        "edge_case_flags": aggregate_edge_case_flags(calls),
    }


# ---------------------------------------------------------------------------
# Escalation queue + adjudication resume
# ---------------------------------------------------------------------------

def anonymized_calls(item_id: str, calls: list[dict]) -> list[dict]:
    """Shuffle deterministically and drop effort metadata so the adjudicator
    sees three anonymized labels+rationales (protocol section 2.2)."""
    rng = random.Random(f"{SEED}-{item_id}")
    order = list(range(len(calls)))
    rng.shuffle(order)
    return [
        {
            "class": calls[i]["class"],
            "rationale": calls[i]["rationale"],
            "edge_case_flags": calls[i]["edge_case_flags"],
            "uncertain": calls[i]["uncertain"],
        }
        for i in order
    ]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    # Split on \n only: str.splitlines() also splits on U+2028/U+2029, which
    # occur unescaped inside turn_text of real LongMemEval turns
    # (gpt4_4edbafa2) and are legal in JSON strings — splitlines() shatters
    # those rows into unparseable fragments.
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def append_escalations(items: list[dict], results: list[dict],
                       queue_path: Path = ESCALATION_QUEUE_PATH) -> int:
    """Append escalated items to the escalation queue (idempotent by _item_id).
    Adjudication happens OUTSIDE python via Claude subagents.

    Display-safe fields (R2-B-3): ONLY packaged_item, labels, and
    escalation_reasons may be shown to the adjudicator; every bookkeeping
    field is '_'-prefixed (hidden provenance, FM-14). A sibling meta file
    carries the judge-runner note (same convention as naturalness_items).
    """
    queued = {row["_item_id"] for row in read_jsonl(queue_path)}
    by_id = {item["item_id"]: item for item in items}
    added = 0
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    with queue_path.open("a") as fh:
        for res in results:
            if not res["escalate"] or res["item_id"] in queued:
                continue
            item = by_id[res["item_id"]]
            row = {
                "packaged_item": format_item_prompt(item),
                "labels": anonymized_calls(res["item_id"], res["calls"]),
                "escalation_reasons": res["escalation_reasons"],
                "_item_id": res["item_id"],
                "_question_id": item["question_id"],
                "_source_file": item.get("source_file", "oracle"),
                "_session_idx": item["session_idx"],
                "_turn_idx": item["turn_idx"],
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            added += 1
    meta_path = queue_path.parent / (queue_path.stem + "_meta.json")
    meta_path.write_text(json.dumps({"note": ESCALATION_BLINDING_NOTE},
                                    indent=2, ensure_ascii=False))
    return added


def read_adjudications(path: Path = ADJUDICATIONS_PATH) -> dict[str, dict]:
    """item_id -> adjudication row ({item_id, class, rationale, edge_case_flags,
    model_id}); later rows for the same item_id supersede earlier ones."""
    return {row["item_id"]: row for row in read_jsonl(path)}


def finalize_label(result: dict, adjudications: dict[str, dict]) -> dict:
    """Final label: adjudicator's on escalated items when present, else the
    2/3 vote; escalated items without an adjudication are pending (M8: the
    adjudicator's edge-case flags override on escalated items)."""
    adj = adjudications.get(result["item_id"])
    if result["escalate"]:
        if adj is None:
            return {"class": None, "label_source": "pending", "edge_case_flags": []}
        return {
            "class": normalize_class(adj["class"]),
            "label_source": "adjudication",
            "edge_case_flags": aggregate_edge_case_flags(result["calls"], adjudication=adj),
        }
    return {
        "class": result["vote"],
        "label_source": "vote",
        "edge_case_flags": result["edge_case_flags"],
    }


# ---------------------------------------------------------------------------
# Item builders (used by both runners; substrates/tuples are passed in)
# ---------------------------------------------------------------------------

def tuples_for_question(tuples_index: dict, source_file: str, question_id: str) -> dict[tuple[int, int], list[dict]]:
    """Slice a (source_file, qid, sidx, tidx) -> tuples index down to one
    question, keyed by (orig_session_idx, turn_idx)."""
    out = {}
    for (src, qid, sidx, tidx), tups in tuples_index.items():
        if src == source_file and qid == question_id:
            out[(sidx, tidx)] = tups
    return out


def build_injected_item(row: dict, perturbed: Any, tuples_index: dict) -> dict:
    """Build a Test-0/Test-1 item from one injection-manifest row (INTERFACES
    inject.py schema) against the perturbed substrate. Returns the item dict,
    or {"extraction_miss": True, ...} when no tuple matches the target key (M5).
    """
    qid = row["question_id"]
    sidx = row["placement"]["session_idx"]
    tidx = row["placement"]["turn_idx"]
    turn_tuples = tuples_for_question(tuples_index, "oracle", qid)
    matched, n_matches, match_method = match_tuple_to_key(
        turn_tuples.get((sidx, tidx), []), row["entity"], row["attribute"],
        value=row.get("value"), aside_text=row.get("text"))
    if matched is None:
        return {
            "extraction_miss": True,
            "item_id": row["injection_id"],
            "provenance": "injected",
            "question_id": qid,
            "session_idx": sidx,
            "turn_idx": tidx,
            "construction_label": row["construction_label"],
        }
    stored = stored_value_replay(
        chronological_user_turns(perturbed), (sidx, tidx), turn_tuples,
        row["entity"], row["attribute"])
    return {
        "item_id": row["injection_id"],
        "provenance": "injected",
        "source_file": "oracle",
        "question_id": qid,
        "session_idx": sidx,
        "turn_idx": tidx,
        "entity": row["entity"],
        "attribute": row["attribute"],
        "asserted_value": matched["value"],
        "statement_span": matched["statement_span"],
        "confidence": matched["confidence"],
        "tuple_match_count": n_matches,
        "tuple_match_method": match_method,
        "stored_value": stored,
        "turn_text": get_turn(perturbed, sidx, tidx)["content"],
        "preceding_turns": preceding_turns(perturbed, sidx, tidx),
        "construction_label": row["construction_label"],
        "edge_case_tags": list(row.get("edge_case_tags", [])),
    }


# R2-M-c: harvest-pool kinds that map into item tags at build time, so the
# mandatory >=15 hedged-recall stratum (protocol §4.2 P7) and the FM-10
# per-tag table see NATURAL items too (injected items carry manifest tags).
HARVEST_KIND_TAGS = {"other_hedged_recall": ("hedged_recall",)}


def build_natural_item(cand: dict, substrate: Any, tuples_index: dict, source_file: str,
                       *, expected_turn_text: Optional[str] = None) -> dict:
    """Build an item from one natural-candidate harvest row. The target key is
    the span-matched extractor tuple's own (entity, attribute) since harvest
    rows carry no formal key (M5 natural branch).

    `substrate` must be the SAME substrate the extractor saw — for candidates
    on schema substrates that is the PERTURBED substrate (all non-T3 manifest
    rows spliced, R2-M-d). When `expected_turn_text` (the extraction-time turn
    text from extractor_tuples.jsonl) is provided, a build-time assert
    enforces that packaging text == extractor stream text for the item turn.
    """
    qid = cand["question_id"]
    sidx, tidx = cand["session_idx"], cand["turn_idx"]
    item_id = f"nat-{source_file}-{qid}-s{sidx}-t{tidx}"
    turn_text = get_turn(substrate, sidx, tidx)["content"]
    if expected_turn_text is not None and turn_text != expected_turn_text:
        raise AssertionError(
            f"packaging/extraction text mismatch for {item_id}: the packaged turn text "
            "differs from the extractor stream text (R2-M-d: items must be packaged "
            "from the same substrate — perturbed for schema substrates — the extractor saw)")
    turn_tuples = tuples_for_question(tuples_index, source_file, qid)
    matched, score = match_tuple_to_span(turn_tuples.get((sidx, tidx), []), cand["text"])
    if matched is None:
        return {
            "extraction_miss": True,
            "item_id": item_id,
            "provenance": "natural",
            "question_id": qid,
            "session_idx": sidx,
            "turn_idx": tidx,
            "best_span_overlap": score,
        }
    if cand["text"] not in turn_text:
        logger.warning("candidate text not verbatim in turn for %s", item_id)
    stored = stored_value_replay(
        chronological_user_turns(substrate), (sidx, tidx), turn_tuples,
        matched["entity"], matched["attribute"])
    return {
        "item_id": item_id,
        "provenance": "natural",
        "source_file": source_file,
        "question_id": qid,
        "session_idx": sidx,
        "turn_idx": tidx,
        "entity": matched["entity"],
        "attribute": matched["attribute"],
        "asserted_value": matched["value"],
        "statement_span": matched["statement_span"],
        "confidence": matched["confidence"],
        "span_overlap": score,
        "stored_value": stored,
        "turn_text": turn_text,
        "preceding_turns": preceding_turns(substrate, sidx, tidx),
        "harvest_kind": cand["kind"],
        "harvest_tags": list(HARVEST_KIND_TAGS.get(cand["kind"], ())),
    }


def load_extractor_tuples(path: Path) -> dict[tuple[str, str, int, int], list[dict]]:
    """Load extract.py's per-turn output file (I3's module; read, never import).

    Expected row schema, one row per extractor call (= per user turn):
      {"question_id": str, "session_idx": int (ORIGINAL haystack index),
       "turn_idx": int, "source_file": "oracle"|"s_cleaned" (default "oracle"),
       "turn_text": str (optional: the extraction-time turn text — perturbed
       for schema substrates — consumed by the R2-M-d packaging assert),
       "tuples": [{"entity","attribute","value","statement_span","confidence"}]}
    Later rows for the same turn supersede earlier ones (idempotent reruns).
    The producer is `extract.py produce-tuples` (R2-B-2).
    """
    index: dict[tuple[str, str, int, int], list[dict]] = {}
    for row in read_jsonl(path):
        key = (row.get("source_file", "oracle"), row["question_id"],
               row["session_idx"], row["turn_idx"])
        index[key] = list(row["tuples"])
    return index


def load_extractor_turn_texts(path: Path) -> dict[tuple[str, str, int, int], str]:
    """(source_file, question_id, session_idx, turn_idx) -> extraction-time turn
    text, from the optional "turn_text" field of extractor_tuples rows. Feeds
    the R2-M-d build-time assert that packaging text matches the extractor
    stream text; rows lacking the field are skipped (assert not applicable)."""
    out: dict[tuple[str, str, int, int], str] = {}
    for row in read_jsonl(path):
        if "turn_text" in row:
            out[(row.get("source_file", "oracle"), row["question_id"],
                 row["session_idx"], row["turn_idx"])] = row["turn_text"]
    return out
