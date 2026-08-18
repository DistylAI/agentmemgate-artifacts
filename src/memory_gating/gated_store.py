"""Composed ONLINE evidence-gated runtime — protocol_spec_v2 commit axis (owner: W2).

Implements both commit policies of the v2 2x2 over ONE shared extraction
pathway, so the fairness constraint "the commit axis adds no extraction cost"
(protocol_spec_v2 §1.2 item 1 / §7) is structural, not procedural:

  - `run_ungated_replay` (C-W): immediate supersession — every mapped tuple
    writes through the ZepStore primitive (`zepstore.ZepStore`), always with
    `correction=False` (zep_arm_design §4: no correction/update distinction).
  - `run_gated_replay` (C-G): the online gate loop. Per user turn:
      1. checker (frozen `checker.txt`) over each substrate store's PENDING
         staged items — items staged at strictly earlier turns; ONE call per
         (turn, store) with pending != {} (protocol_spec_v2 §7 row 7). The
         pending set may exceed 1 in v2: the pilot's M2 `|pending| == 1`
         assert (test3._run_variant_replay) becomes the diagnostics counter
         `pending_gt1_checker_calls` (protocol_spec_v2 §3.3 dyadic streams /
         W2 pin), plus `max_pending_size`.
      2. extraction (family-selected system prompt; B2 rendering via
         `extract.render_extraction_prompt`, validation via
         `extract.parse_tuples` — fairness item 2, only the system prompt
         file varies across the extraction axis).
      3. key canonicalization: the existing key-mapper path
         (`store._map_tuple_key`, one call per tuple, same candidate lists,
         unmapped logged never dropped — FM-6).
      4. classifier (frozen `classifier_short.txt` via
         `classify.classify_item`) per MAPPED tuple (§7 row 5 budgets the
         classifier on the mapped-tuple fraction), packaged with the frozen
         `item_packaging.txt` (turn context + tuple + stored value). The
         stored value is the substrate store's CURRENT value for the mapped
         canonical key (the online analogue of B3 store-conditioning,
         protocol_spec v1 §1.5d: "the classifier [is] shown the current
         stored value for the target attribute when one exists").
      5. commit policy per argmax class:
           completed_event      -> commit (supersede; ZepStore write)
           explicit_correction  -> commit with retro-invalidation
                                   (`write(..., correction=True)` marks the
                                   superseded interval invalid-from-origin)
           speculative          -> stage with resolution predicates (frozen
                                   `predicate_generator.txt`; STORELESS
                                   B2-style prefix input = the transcript
                                   prefix through the staging turn, both
                                   roles, nothing after it and no store
                                   state — v1 B6 firewall; §3.3 lint with
                                   one retry via test3.lint_predicates)
           other                -> ADDITIVE commit: commit when the key has
                                   no current value (v1 §1.5d "plain first
                                   observation") or refreshes an equivalent
                                   value (Z7); when a DIFFERENT current
                                   value exists the tuple does not license
                                   canonical revision (v1 §1 taxonomy:
                                   `other` = "everything that does not
                                   license or stage a revision to canonical
                                   current state"; zep_arm_design §5
                                   "additive commit or no-op") -> logged
                                   no-op (`additive_noop`), never a
                                   supersession.
      6. checker fires resolve staged items with the pilot Test-3 consume
         semantics (test3.parse_checker_fires + the consume rule of
         test3._run_variant_replay): any fire naming a pending staged_id
         consumes it; a valid promote fire commits the hypothesized value at
         the FIRING turn; a valid invalidate fire discards it (logged);
         invalid fires (double_action / bad_action) consume without
         committing; unknown-id fires consume nothing and are logged.
      7. a per-user-turn snapshot row (store sequence numbers + pending ids)
         is retained for the TTC / probe machinery — probe points read store
         state via `ZepStore.as_of_seq`, no store copies.

Extraction and key mapping are batched (asyncio.gather) exactly like
`store.replay_write_through`: both are pure functions of the turn stream
(never of the store), so batching is identical to sequential replay and both
policies of a family render byte-identical extraction prompts — the llm.py
cache key list (`GatedTimeline.extraction_call_keys`) is recorded so
`arms.py` can hard-assert the identity per cell pair.

Multi-store streams (LoCoMo speaker-stores, protocol_spec_v2 §2.2): the
replay accepts `stores` + `route_store(key) -> store_id`; the checker runs
per (turn, store) with that store's pending set. LME substrates use the
single default store.

No prompt file is modified here; all five prompts are the frozen pilot set
(V2-P10). LLM calls go through llm.call_llm only.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import classify
import extract
import llm
import store
import run_predicates as test3
from zepstore import ZepStore, apply_sentinel

from paths import PROMPTS_DIR

#: Extraction-family system prompts (protocol_spec_v2 §1.3: the extraction
#: axis is the system prompt file, nothing else). The dyadic variants are the
#: LoCoMo renderings of the same two families (§3.3).
FAMILY_PROMPTS = {
    "mem0": "extractor.txt",
    "zep": "extractor_zep.txt",
    "mem0_dyadic": "extractor_dyadic.txt",
    "zep_dyadic": "extractor_zep_dyadic.txt",
}

GENERATOR_PROMPT_PATH = test3.GENERATOR_PROMPT_PATH
CHECKER_PROMPT_PATH = test3.CHECKER_PROMPT_PATH

DEFAULT_STORE = "main"
CLASSES_COMMIT = ("completed_event", "explicit_correction")

#: The three phase-3 dispositions of a mapped tuple (protocol v2.2 C2: the
#: classify->route step is pluggable; every router maps a tuple to one of
#: these and the branch bodies below are shared verbatim across routers).
ROUTES = ("commit", "stage", "additive")

#: The frozen v2 argmax routing (protocol_spec_v2 §7 / module docstring item
#: 5): completed_event / explicit_correction -> commit, speculative -> stage,
#: other -> additive. This is the default when no router is injected — the
#: v2.2 router points (pilot/router_sweep.py) replace ONLY this mapping.
ARGMAX_ROUTE = {
    "completed_event": "commit",
    "explicit_correction": "commit",
    "speculative": "stage",
    "other": "additive",
}


def family_system_prompt(family: str) -> str:
    """The frozen extraction system prompt for one family, read verbatim."""
    try:
        name = FAMILY_PROMPTS[family]
    except KeyError:
        raise ValueError(f"unknown extraction family {family!r}; "
                         f"expected one of {sorted(FAMILY_PROMPTS)}") from None
    return (PROMPTS_DIR / name).read_text()


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------


@dataclass
class GatedTimeline:
    """One replay's full record (either commit policy — `policy` says which).

    `snapshots` has one row per user turn: {session_idx, turn_idx, date,
    user_pos, seq: {store_id: last consumed ZepStore seq}, pending:
    {store_id: [staged_id, ...]}}. Probe points read store state through
    `probe_timeline` (ZepStore.as_of_seq — no store copies).
    """

    stream_id: str
    family: str
    policy: str  # "gated" | "ungated"
    session_order: list
    stores: dict[str, ZepStore]
    snapshots: list[dict] = field(default_factory=list)
    gate_log: list[dict] = field(default_factory=list)     # one row per mapped tuple
    staged_log: list[dict] = field(default_factory=list)   # one row per staged item
    checker_log: list[dict] = field(default_factory=list)  # one row per checker call
    mapping_log: list[dict] = field(default_factory=list)  # key_mapping rows (store.py shape)
    unmapped: list[dict] = field(default_factory=list)     # FM-6: logged, never dropped
    extraction_call_keys: list[str] = field(default_factory=list)  # llm.py cache keys
    diagnostics: dict = field(default_factory=dict)


def _new_diagnostics() -> dict:
    return {
        "n_user_turns": 0,
        "n_extraction_calls": 0,
        "n_tuples": 0,
        "n_mapped": 0,
        "n_unmapped": 0,
        "n_classifier_calls": 0,
        "class_counts": {},
        "n_committed": 0,
        "n_retro_invalidations": 0,
        "n_additive_commits": 0,
        "n_additive_noops": 0,
        "n_refreshes": 0,
        "n_staged": 0,
        "n_promoted": 0,
        "n_invalidated": 0,
        "n_discarded_invalid": 0,
        "n_pending_final": 0,  # staged-but-unresolved at end (TTC censoring feed)
        "checker_calls": 0,
        "checker_fires": 0,
        "checker_unknown_id_fires": 0,
        "checker_malformed_outputs": 0,
        # v2 counter replacing the pilot M2 |pending| == 1 assert (W2 pin):
        "pending_gt1_checker_calls": 0,
        "max_pending_size": 0,
        "generator_retries": 0,
        "generator_residual_lint_failures": 0,
    }


# ---------------------------------------------------------------------------
# Stream construction (shared with store.py's chronological walk)
# ---------------------------------------------------------------------------


def substrate_stream(substrate) -> list[dict]:
    """All turns of a Substrate chronologically, tagged with ORIGINAL session
    index and host session date — the exact walk of `store._chronological_stream`
    (reused so both policies and the pilot replay see one stream shape)."""
    return store._chronological_stream(substrate)


def _stream_session_order(stream: list[dict]) -> list:
    order: list = []
    for row in stream:
        if row["session_idx"] not in order:
            order.append(row["session_idx"])
    return order


def _preceding_turns(stream: list[dict], pos: int, k: int = 2) -> list[dict]:
    """Up to k turns immediately preceding stream[pos] WITHIN its session
    (protocol §2.1 packaging context; mirrors label.preceding_turns)."""
    session = stream[pos]["session_idx"]
    out: list[dict] = []
    for row in reversed(stream[:pos]):
        if row["session_idx"] != session or len(out) == k:
            break
        out.append({"role": row["role"], "content": row["content"]})
    return list(reversed(out))


# ---------------------------------------------------------------------------
# Phase 1 — extraction (the single per-family pathway, shared by BOTH policies)
# ---------------------------------------------------------------------------


async def extract_turn_family(
    prior_turns: list[dict],
    current_turn: str,
    trace_tag: str,
    *,
    system_prompt: str,
    current_date: str | None = None,
) -> list[extract.Tuple]:
    """One extractor call under a family-selected system prompt. Rendering
    (`extract.render_extraction_prompt`) and validation (`extract.parse_tuples`)
    are the frozen B2 contract — only the system prompt varies (§1.3)."""
    user_prompt = extract.render_extraction_prompt(prior_turns, current_turn, current_date)
    output = await llm.call_llm(system_prompt, user_prompt, trace_tag=trace_tag)
    return extract.parse_tuples(output)


async def extract_stream_tuples(
    stream: list[dict], *, stream_id: str, family: str, system_prompt: str | None = None
) -> tuple[list[dict], list[str]]:
    """ONE extractor call per user turn over the stream (B2). Returns
    (turn_rows, extraction_call_keys) where turn_rows carry
    {pos, user_pos, session_idx, turn_idx, date, turn_text, tuples} and
    extraction_call_keys are the llm.py cache keys of the calls in stream
    order — the byte-identity witness across commit policies (arms.py assert).
    """
    if system_prompt is None:
        system_prompt = family_system_prompt(family)
    user_positions = [i for i, t in enumerate(stream) if t["role"] == "user"]
    call_keys: list[str] = []
    coros = []
    for pos in user_positions:
        turn = stream[pos]
        prior = [
            {"role": t["role"], "content": t["content"], "date": t["date"]}
            for t in stream[:pos]
        ]
        user_prompt = extract.render_extraction_prompt(prior, turn["content"], turn["date"])
        call_keys.append(test3._cache_key(system_prompt, user_prompt))
        coros.append(
            extract_turn_family(
                prior,
                turn["content"],
                f"extract-{family}-{stream_id}-s{turn['session_idx']}t{turn['turn_idx']}",
                system_prompt=system_prompt,
                current_date=turn["date"],
            )
        )
    tuples_per_turn = await asyncio.gather(*coros)
    turn_rows = [
        {
            "pos": pos,
            "user_pos": upos,
            "session_idx": stream[pos]["session_idx"],
            "turn_idx": stream[pos]["turn_idx"],
            "date": stream[pos]["date"],
            "turn_text": stream[pos]["content"],
            "tuples": list(tuples),
        }
        for upos, (pos, tuples) in enumerate(zip(user_positions, tuples_per_turn))
    ]
    return turn_rows, call_keys


# ---------------------------------------------------------------------------
# Phase 2 — key canonicalization (existing key_mapper path, one call/tuple)
# ---------------------------------------------------------------------------


async def map_stream_tuples(
    turn_rows: list[dict],
    candidates: list[store.CanonicalKey],
    *,
    stream_id: str,
    family: str,
    mapper=None,
) -> list[list[tuple]]:
    """One `store._map_tuple_key` call per extracted tuple (the pilot key-mapper
    path verbatim: same prompt, same candidate lists, cache-deduplicated).
    Returns per-turn lists aligned with turn_rows[i]["tuples"].

    ``mapper`` (v2.4 M1 seam, default None = mapping mode "off"): optional
    async override with the `_map_tuple_key` calling convention
    ``(tup, candidates, *, trace_tag)``. The frozen default returns the
    3-tuple ``(cand, raw_key, rationale)`` and is passed through UNCHANGED
    (byte-preserved path — no call-site tuple shape changes); an installed
    mapper (store.make_tuple_key_mapper) returns a 4th ``meta`` element that
    only the meta-aware consumers below read. Results are passed through
    verbatim either way."""
    map_call = store._map_tuple_key if mapper is None else mapper
    coros, index = [], []
    for i, row in enumerate(turn_rows):
        for j, tup in enumerate(row["tuples"]):
            index.append((i, j))
            coros.append(
                map_call(
                    tup,
                    candidates,
                    trace_tag=f"keymap-{family}-{stream_id}-"
                              f"s{row['session_idx']}t{row['turn_idx']}-{j}",
                )
            )
    flat = list(await asyncio.gather(*coros))
    nested: list[list[tuple]] = [[] for _ in turn_rows]
    for (i, _j), res in zip(index, flat):
        nested[i].append(res)
    return nested


def _unpack_map_result(mres) -> tuple:
    """(cand, raw_key, rationale, meta) from a mapper result: the frozen
    3-tuple gets meta == {}; a v2.4 mapper 4-tuple passes through. Both
    shapes are accepted at every consumer so mode-off replays are
    byte-preserved and a mode-on mapper cannot crash the frozen unpacking
    (the first M1 build attempt's blocking defect)."""
    cand, raw_key, rationale = mres[0], mres[1], mres[2]
    meta = mres[3] if len(mres) > 3 else {}
    return cand, raw_key, rationale, (meta or {})


def _mapping_row(stream_id: str, row: dict, tup, candidates, cand, raw_key, rationale,
                 meta: dict | None = None) -> dict:
    """key_mapping.jsonl row, store.py shape (collision_audit-compatible).

    When the v2.4 plan-stem mapper touched the tuple (``meta`` non-empty) the
    row additionally records the mapping-mode provenance — PIN-1: every
    stem-mapped row is per-row auditable via (original_attribute,
    derived_stem, mapped_key, pattern_matched). Rows the wrapper never
    touched (meta empty/None — ALL rows in mode off) log byte-identically to
    the frozen shape."""
    from dataclasses import asdict

    out = {
        "substrate_qid": stream_id,
        "session_idx": row["session_idx"],
        "turn_idx": row["turn_idx"],
        "extracted_tuple": asdict(tup),
        "candidate_keys": [c.name for c in candidates],
        "mapped_key": cand.name if cand is not None else "none",
        "model_output_key": raw_key,
        "rationale": rationale,
    }
    if meta:
        out["mapping_mode"] = meta.get("mapping_mode")
        plan_stem = {k: meta[k] for k in ("original_attribute", "derived_stem",
                                          "pattern_matched", "stem_mapped")
                     if k in meta}
        for pass_key in ("first_pass", "stem_pass"):
            if pass_key in meta:
                plan_stem[pass_key] = meta[pass_key]
        out["plan_stem"] = plan_stem
        if meta.get("plan_framed"):
            out["plan_framed"] = True
    return out


# ---------------------------------------------------------------------------
# Gate-loop helpers
# ---------------------------------------------------------------------------


def _snapshot(timeline: GatedTimeline, row: dict, pending: dict[str, list[dict]]) -> None:
    timeline.snapshots.append(
        {
            "session_idx": row["session_idx"],
            "turn_idx": row["turn_idx"],
            "date": row["date"],
            "user_pos": row["user_pos"],
            "seq": {sid: st.seq for sid, st in timeline.stores.items()},
            "pending": {sid: [p["staged_id"] for p in items]
                        for sid, items in pending.items() if items},
        }
    )


async def _generate_predicates(
    stream: list[dict],
    row: dict,
    cand: store.CanonicalKey,
    tup,
    *,
    staged_id: str,
    stream_id: str,
    family: str,
    gen_system: str,
    diagnostics: dict,
) -> dict:
    """Frozen predicate generation for one speculative tuple.

    Input = the staged tuple + the STORELESS B2-style transcript prefix through
    the staging turn (both roles, session dates; nothing after the staging turn
    and no store state — v1 B6 firewall, protocol_spec_v2 §4). Lint per v1 §3.3
    with exactly one feedback retry (test3.lint_predicates), the pilot flow.
    """
    prefix = [
        {"date": t["date"], "role": t["role"], "content": t["content"]}
        for t in stream[: row["pos"] + 1]
    ]
    assert prefix[-1]["content"] == row["turn_text"], "prefix must end at the staging turn"
    user_prompt = test3.build_generator_user_prompt(
        prefix, cand.entity, cand.attribute, tup.value)
    out = await llm.call_llm(gen_system, user_prompt,
                             trace_tag=f"gate_gen-{family}-{stream_id}-{staged_id}")
    aliases = test3.attribute_aliases(cand.attribute)
    promote_when = str(out["promote_when"])
    invalidate_when = str(out["invalidate_when"])
    first_failures = test3.lint_predicates(promote_when, invalidate_when, tup.value, aliases)
    final_failures, retried = first_failures, False
    if first_failures:
        retried = True
        diagnostics["generator_retries"] += 1
        retry_prompt = user_prompt + test3.lint_feedback_block(first_failures)
        out = await llm.call_llm(gen_system, retry_prompt,
                                 trace_tag=f"gate_gen_retry-{family}-{stream_id}-{staged_id}")
        promote_when = str(out["promote_when"])
        invalidate_when = str(out["invalidate_when"])
        final_failures = test3.lint_predicates(promote_when, invalidate_when, tup.value, aliases)
        if final_failures:
            diagnostics["generator_residual_lint_failures"] += 1
    return {
        "staged_id": staged_id,
        "entity": cand.entity,
        "attribute": cand.attribute,
        "hypothesized_value": tup.value,
        "promote_when": promote_when,
        "invalidate_when": invalidate_when,
        "certainty_meta": out.get("certainty_meta") or {},
        "statement_span": tup.statement_span,
        "confidence": tup.confidence,
        "lint": {"first_failures": first_failures, "retried": retried,
                 "final_failures": final_failures},
    }


def _pending_item(staged: dict) -> dict:
    """Exactly the six fields the checker sees per pending item (predicate_spec
    §2.2 / test3.checker_pending_item)."""
    return {k: staged[k] for k in ("staged_id", "entity", "attribute",
                                   "hypothesized_value", "promote_when", "invalidate_when")}


async def _run_checker_turn(
    timeline: GatedTimeline,
    row: dict,
    store_id: str,
    pending_items: list[dict],
    staged_rows: dict[str, dict],
    *,
    checker_system: str,
) -> list[dict]:
    """ONE checker call for (turn, store); returns the surviving pending items.

    Consume semantics mirror the pilot (test3._run_variant_replay): any fire
    naming a pending staged_id consumes that item. Valid promote -> commit the
    hypothesized value at the firing turn; valid invalidate -> discard (logged);
    invalid fires (double_action / bad_action) -> consumed, discarded, logged;
    unknown-id fires consume nothing (logged in the fire list).
    """
    diag = timeline.diagnostics
    diag["checker_calls"] += 1
    n_pending = len(pending_items)
    diag["max_pending_size"] = max(diag["max_pending_size"], n_pending)
    if n_pending > 1:
        # v2: the pilot M2 assert (|pending| == 1) becomes a counter (W2 pin).
        diag["pending_gt1_checker_calls"] += 1

    pending_ids = {p["staged_id"] for p in pending_items}
    output = await llm.call_llm(
        checker_system,
        test3.build_checker_user_prompt(
            row["turn_text"], row["date"], [_pending_item(p) for p in pending_items]),
        trace_tag=(f"gate_check-{timeline.family}-{timeline.stream_id}-{store_id}-"
                   f"s{row['session_idx']}t{row['turn_idx']}"),
    )
    fires, malformed, extra_keys = test3.parse_checker_fires(
        output, pending_ids, row["turn_text"])
    diag["checker_fires"] += len(fires)
    diag["checker_malformed_outputs"] += int(malformed)
    diag["checker_unknown_id_fires"] += sum(
        1 for f in fires if f.get("invalid_reason") == "unknown_id")

    consumed = {f["staged_id"] for f in fires if f["staged_id"] in pending_ids}
    resolutions = []
    for item in pending_items:
        sid = item["staged_id"]
        if sid not in consumed:
            continue
        item_fires = [f for f in fires if f["staged_id"] == sid]
        valid = [f for f in item_fires if not f["invalid"]]
        if valid and valid[0]["action"] == "promote":
            outcome = "promoted"
            diag["n_promoted"] += 1
            timeline.stores[store_id].write(
                (item["entity"], item["attribute"]),
                item["hypothesized_value"],
                valid_from=row["date"],
                provenance={
                    "kind": "gated_promotion",
                    "staged_id": sid,
                    "staged_at": staged_rows[sid]["staged_at"],
                    "fired_at": {"session_idx": row["session_idx"],
                                 "turn_idx": row["turn_idx"], "date": row["date"],
                                 "user_pos": row["user_pos"]},
                    "evidence_span": valid[0]["evidence_span"],
                },
                correction=False,
            )
        elif valid and valid[0]["action"] == "invalidate":
            outcome = "invalidated"
            diag["n_invalidated"] += 1
        else:
            outcome = "discarded_invalid"
            diag["n_discarded_invalid"] += 1
        srow = staged_rows[sid]
        srow["status"] = outcome
        srow["resolved_at"] = {"session_idx": row["session_idx"],
                               "turn_idx": row["turn_idx"], "date": row["date"],
                               "user_pos": row["user_pos"]}
        srow["firing_lag_user_turns"] = row["user_pos"] - srow["staged_at"]["user_pos"]
        srow["resolving_fires"] = item_fires
        resolutions.append({"staged_id": sid, "outcome": outcome,
                            "invalid_reasons": sorted({f["invalid_reason"]
                                                       for f in item_fires
                                                       if f["invalid_reason"]})})

    timeline.checker_log.append(
        {
            "stream_id": timeline.stream_id,
            "store_id": store_id,
            "session_idx": row["session_idx"],
            "turn_idx": row["turn_idx"],
            "date": row["date"],
            "user_pos": row["user_pos"],
            "pending_ids": sorted(pending_ids),
            "n_pending": n_pending,
            "output": output,
            "fires": fires,
            "malformed_output": malformed,
            "extra_keys": extra_keys,
            "resolutions": resolutions,
        }
    )
    return [p for p in pending_items if p["staged_id"] not in consumed]


# ---------------------------------------------------------------------------
# The two replays
# ---------------------------------------------------------------------------


def _init_timeline(stream, *, stream_id, family, policy, stores, session_order) -> GatedTimeline:
    return GatedTimeline(
        stream_id=stream_id,
        family=family,
        policy=policy,
        session_order=(list(session_order) if session_order is not None
                       else _stream_session_order(stream)),
        stores=dict(stores) if stores else {},
        diagnostics=_new_diagnostics(),
    )


def _get_store(timeline: GatedTimeline, store_id: str) -> ZepStore:
    if store_id not in timeline.stores:
        timeline.stores[store_id] = ZepStore()
    return timeline.stores[store_id]


async def run_gated_replay(
    stream: list[dict],
    candidates: list[store.CanonicalKey],
    *,
    stream_id: str,
    family: str = "mem0",
    system_prompt: str | None = None,
    stores: dict[str, ZepStore] | None = None,
    route_store: Callable[[tuple], str] | None = None,
    session_order: list | None = None,
    extracted: tuple[list[dict], list[str]] | None = None,
    router=None,
    classify_fn=None,
    construction_label_of: Callable[[dict, store.CanonicalKey], str | None] | None = None,
    mapper=None,
) -> GatedTimeline:
    """The online gate loop (C-G) over one turn stream (module docstring).

    ``mapper`` (v2.4 M1 seam; default None = mapping mode "off", the frozen
    v2 path byte-for-byte): optional key-mapper override passed to
    `map_stream_tuples` (store.make_tuple_key_mapper). For a tuple the
    plan-stem mapper stem-MAPPED (meta.plan_framed), three behaviors are
    implemented below:

      1. its classifier item is packaged with the ORIGINAL plan-framed
         extractor attribute (the prereg's "plan-framing preserved as
         metadata for the classifier" — the store write and the predicate
         key stay CANONICAL, only the classifier sees the plan framing);
      2. the gate/mapping/staged log rows record the mapping-mode
         provenance (`mapping_mode`, `plan_framed`, `original_attribute`,
         `derived_stem`; mapping rows additionally `pattern_matched` +
         both mapper passes — PIN-1 per-row auditability);
      3. rows the wrapper never touched log byte-identically to mode-off.

    ``extracted``: optional pre-extracted phase-1 output ``(turn_rows,
    extraction_call_keys)`` in the `extract_stream_tuples` shape. This is the
    dyadic seam (protocol_spec_v2 §2.2/§3.3): LoCoMo streams render extraction
    through `locomo_render` (speaker-named turn lines, ≤2-turn context, dyadic
    prompts), which is NOT the LME B2 rendering — the caller (run_v2) supplies
    the turn rows and the llm.py cache keys of its calls, and the gate loop
    (phases 2-3) stays this single implementation for both benchmarks. The
    byte-identity assert across commit policies operates on the supplied keys
    exactly as on LME streams.

    v2.2 C2 seams (protocol_changelog "Protocol v2.2"; pilot/router_sweep.py):

    ``router``: optional classify->route override. Either an object with a
    ``route(tuple_record, classifier_output, construction_label_or_None)``
    method or a bare callable of the same signature, returning one of
    ``ROUTES``. ``None`` (default) keeps the frozen v2 argmax routing
    (``ARGMAX_ROUTE``) byte-for-byte — the branch bodies are shared, only the
    class->route mapping is replaced. When a router is supplied, each gate row
    additionally records ``route``, ``router`` and ``construction_label``.

    ``classify_fn``: optional async replacement for ``classify.classify_item``
    (same item -> {probs, s_ev, argmax, max_prob} contract). The v2.2 sweep
    injects a recorded-output provider here so re-runs consume the FROZEN v2
    classifier outputs (cache-shared spend; the amendment's "classifier calls
    are cache-shared").

    ``construction_label_of``: optional ``(turn_row, canonical_key) ->
    label | None`` lookup giving the router the injected row's construction
    label (oracle point); ``None`` for natural rows.
    """
    route = route_store if route_store is not None else (lambda key: DEFAULT_STORE)
    classify_call = classify.classify_item if classify_fn is None else classify_fn
    route_fn = (router.route if router is not None and hasattr(router, "route")
                else router)
    timeline = _init_timeline(stream, stream_id=stream_id, family=family,
                              policy="gated", stores=stores, session_order=session_order)
    diag = timeline.diagnostics

    # Phases 1-2 — batched, store-independent (identical to sequential replay).
    if extracted is not None:
        turn_rows, timeline.extraction_call_keys = extracted
    else:
        turn_rows, timeline.extraction_call_keys = await extract_stream_tuples(
            stream, stream_id=stream_id, family=family, system_prompt=system_prompt)
    map_results = await map_stream_tuples(turn_rows, candidates,
                                          stream_id=stream_id, family=family,
                                          mapper=mapper)
    diag["n_user_turns"] = diag["n_extraction_calls"] = len(turn_rows)

    gen_system = GENERATOR_PROMPT_PATH.read_text()
    checker_system = CHECKER_PROMPT_PATH.read_text()

    # Phase 3 — sequential gate loop (classifier/generator/checker are
    # store-state-dependent, so this phase is strictly chronological).
    pending: dict[str, list[dict]] = {}
    staged_rows: dict[str, dict] = {}

    for row, row_maps in zip(turn_rows, map_results):
        # 1) checker over items staged at STRICTLY earlier turns ("each
        #    subsequent turn runs the checker") — one call per (turn, store).
        for store_id in sorted(pending):
            if pending[store_id]:
                pending[store_id] = await _run_checker_turn(
                    timeline, row, store_id, pending[store_id], staged_rows,
                    checker_system=checker_system)

        # 2) this turn's tuples, in emitted order (D5).
        for j, (tup, mres) in enumerate(zip(row["tuples"], row_maps)):
            cand, raw_key, rationale, meta = _unpack_map_result(mres)
            diag["n_tuples"] += 1
            mrow = _mapping_row(stream_id, row, tup, candidates, cand, raw_key,
                                rationale, meta)
            timeline.mapping_log.append(mrow)
            if cand is None:
                diag["n_unmapped"] += 1
                timeline.unmapped.append(mrow)  # FM-6: logged, never dropped
                continue
            diag["n_mapped"] += 1
            store_id = route(cand.key)
            zstore = _get_store(timeline, store_id)
            stored_now = zstore.lookup(cand.key)

            # v2.4 M1 classifier-metadata pin: a stem-mapped tuple is
            # classified under its ORIGINAL plan-framed extractor attribute
            # (the plan-framing signal the canonical stem key would erase —
            # M1's contamination-guard linchpin); everything else about the
            # item (canonical entity, stored value for the CANONICAL key,
            # turn context) is unchanged. meta is {} in mode off.
            plan_framed = bool(meta.get("plan_framed"))
            item = {
                "item_id": f"gate-{family}-{stream_id}-"
                           f"s{row['session_idx']}t{row['turn_idx']}-{j}",
                "entity": cand.entity,
                "attribute": (meta["original_attribute"] if plan_framed
                              else cand.attribute),
                "asserted_value": tup.value,
                "stored_value": stored_now,  # sentinel passed verbatim when present
                "turn_text": row["turn_text"],
                "preceding_turns": _preceding_turns(stream, row["pos"]),
            }
            cls_out = await classify_call(item)
            diag["n_classifier_calls"] += 1
            cls = cls_out["argmax"]
            diag["class_counts"][cls] = diag["class_counts"].get(cls, 0) + 1

            gate_row = {
                "item_id": item["item_id"],
                "session_idx": row["session_idx"],
                "turn_idx": row["turn_idx"],
                "user_pos": row["user_pos"],
                "date": row["date"],
                "store_id": store_id,
                "key": cand.name,
                "value": tup.value,
                "stored_value": stored_now,
                "confidence": tup.confidence,
                "class": cls,
                "probs": cls_out["probs"],
                "s_ev": cls_out["s_ev"],
            }
            if meta:  # v2.4 M1 mapping-mode provenance (stem-path rows only)
                gate_row["mapping_mode"] = meta.get("mapping_mode")
                gate_row["plan_framed"] = plan_framed
                gate_row["original_attribute"] = meta.get("original_attribute")
                gate_row["derived_stem"] = meta.get("derived_stem")
                gate_row["classifier_attribute"] = item["attribute"]
            provenance = {"session_idx": row["session_idx"], "turn_idx": row["turn_idx"],
                          "statement_span": tup.statement_span,
                          "confidence": tup.confidence, "class": cls}

            if route_fn is None:
                route_to = ARGMAX_ROUTE[cls]  # frozen v2 argmax routing
            else:
                construction_label = (construction_label_of(row, cand)
                                      if construction_label_of is not None else None)
                route_to = route_fn(
                    {
                        "key": cand.name,
                        "entity": cand.entity,
                        "attribute": cand.attribute,
                        "raw_entity": tup.entity,
                        "raw_attribute": tup.attribute,
                        "value": tup.value,
                        "confidence": tup.confidence,
                        "stored_value": stored_now,
                        "store_id": store_id,
                        "session_idx": row["session_idx"],
                        "turn_idx": row["turn_idx"],
                        "user_pos": row["user_pos"],
                    },
                    cls_out,
                    construction_label,
                )
                assert route_to in ROUTES, (
                    f"router returned {route_to!r}; expected one of {ROUTES}")
                gate_row["route"] = route_to
                gate_row["router"] = getattr(router, "name", "custom")
                gate_row["construction_label"] = construction_label

            if route_to == "commit":
                correction = cls == "explicit_correction"
                event = zstore.write(cand.key, tup.value, valid_from=row["date"],
                                     provenance=provenance, correction=correction)
                diag["n_committed"] += 1
                diag["n_retro_invalidations"] += int(event["retro_invalidated"])
                diag["n_refreshes"] += int(event["action"] == "refresh")
                gate_row["action"] = event["action"]
                gate_row["retro_invalidated"] = event["retro_invalidated"]
            elif route_to == "stage":
                staged_id = (f"stg-{family}-{stream_id}-"
                             f"s{row['session_idx']}t{row['turn_idx']}-{j}")
                staged = await _generate_predicates(
                    stream, row, cand, tup, staged_id=staged_id, stream_id=stream_id,
                    family=family, gen_system=gen_system, diagnostics=diag)
                staged["store_id"] = store_id
                staged["staged_at"] = {"session_idx": row["session_idx"],
                                       "turn_idx": row["turn_idx"], "date": row["date"],
                                       "user_pos": row["user_pos"]}
                if meta:  # v2.4 M1 mapping-mode provenance (stem-path rows only)
                    staged["mapping_mode"] = meta.get("mapping_mode")
                    staged["plan_framed"] = plan_framed
                    staged["original_attribute"] = meta.get("original_attribute")
                    staged["derived_stem"] = meta.get("derived_stem")
                staged["status"] = "pending"
                staged["resolved_at"] = None
                staged["firing_lag_user_turns"] = None
                timeline.staged_log.append(staged)
                staged_rows[staged_id] = staged
                pending.setdefault(store_id, []).append(staged)
                diag["n_staged"] += 1
                gate_row["action"] = "staged"
                gate_row["staged_id"] = staged_id
            else:  # other -> additive commit or no-op (never a supersession)
                new_stored = apply_sentinel(tup.value)
                if stored_now is None or (
                        store.normalize_value(new_stored)
                        == store.normalize_value(stored_now)):
                    event = zstore.write(cand.key, tup.value, valid_from=row["date"],
                                         provenance=provenance, correction=False)
                    diag["n_additive_commits"] += int(event["action"] == "commit")
                    diag["n_refreshes"] += int(event["action"] == "refresh")
                    gate_row["action"] = f"additive_{event['action']}"
                else:
                    diag["n_additive_noops"] += 1
                    gate_row["action"] = "additive_noop"
            timeline.gate_log.append(gate_row)

        # 3) per-turn snapshot (TTC / probe machinery).
        _snapshot(timeline, row, pending)

    diag["n_pending_final"] = sum(len(items) for items in pending.values())
    return timeline


async def run_ungated_replay(
    stream: list[dict],
    candidates: list[store.CanonicalKey],
    *,
    stream_id: str,
    family: str = "mem0",
    system_prompt: str | None = None,
    stores: dict[str, ZepStore] | None = None,
    route_store: Callable[[tuple], str] | None = None,
    session_order: list | None = None,
    extracted: tuple[list[dict], list[str]] | None = None,
    mapper=None,
) -> GatedTimeline:
    """Immediate supersession (C-W): every mapped tuple writes through the
    ZepStore, always `correction=False` (zep_arm_design §4 — the ungated column
    has no correction/update distinction). Deterministic given extraction +
    key mapping (protocol_spec_v2 §10.2 step 8); shares the phase-1/2 pathway
    with the gated replay, so extraction calls are byte-identical.
    ``extracted`` is the same pre-extracted phase-1 seam as `run_gated_replay`
    (dyadic LoCoMo streams). ``mapper`` is the same v2.4 M1 key-mapper seam as
    `run_gated_replay` (default None = frozen mode-off path byte-for-byte;
    there is no classifier here, so stem-path provenance lands only in the
    mapping/unmapped log rows)."""
    route = route_store if route_store is not None else (lambda key: DEFAULT_STORE)
    timeline = _init_timeline(stream, stream_id=stream_id, family=family,
                              policy="ungated", stores=stores, session_order=session_order)
    diag = timeline.diagnostics

    if extracted is not None:
        turn_rows, timeline.extraction_call_keys = extracted
    else:
        turn_rows, timeline.extraction_call_keys = await extract_stream_tuples(
            stream, stream_id=stream_id, family=family, system_prompt=system_prompt)
    map_results = await map_stream_tuples(turn_rows, candidates,
                                          stream_id=stream_id, family=family,
                                          mapper=mapper)
    diag["n_user_turns"] = diag["n_extraction_calls"] = len(turn_rows)

    for row, row_maps in zip(turn_rows, map_results):
        for j, (tup, mres) in enumerate(zip(row["tuples"], row_maps)):
            cand, raw_key, rationale, meta = _unpack_map_result(mres)
            diag["n_tuples"] += 1
            mrow = _mapping_row(stream_id, row, tup, candidates, cand, raw_key,
                                rationale, meta)
            timeline.mapping_log.append(mrow)
            if cand is None:
                diag["n_unmapped"] += 1
                timeline.unmapped.append(mrow)
                continue
            diag["n_mapped"] += 1
            store_id = route(cand.key)
            event = _get_store(timeline, store_id).write(
                cand.key, tup.value, valid_from=row["date"],
                provenance={"session_idx": row["session_idx"], "turn_idx": row["turn_idx"],
                            "statement_span": tup.statement_span,
                            "confidence": tup.confidence, "class": None},
                correction=False)
            diag["n_committed"] += 1
            diag["n_refreshes"] += int(event["action"] == "refresh")
            timeline.gate_log.append(
                {
                    "item_id": f"write-{family}-{stream_id}-"
                               f"s{row['session_idx']}t{row['turn_idx']}-{j}",
                    "session_idx": row["session_idx"],
                    "turn_idx": row["turn_idx"],
                    "user_pos": row["user_pos"],
                    "date": row["date"],
                    "store_id": store_id,
                    "key": cand.name,
                    "value": tup.value,
                    "confidence": tup.confidence,
                    "class": None,
                    "action": event["action"],
                }
            )
        _snapshot(timeline, row, {})
    return timeline


# ---------------------------------------------------------------------------
# Substrate-level wrappers (INTERFACES replay_write_through conventions)
# ---------------------------------------------------------------------------


def _prepare_substrate(substrate, key_schema_entry, manifest_rows, aux_keys):
    question_id = substrate.question_id
    if manifest_rows:
        from inject import build_perturbed_substrate

        substrate = build_perturbed_substrate(substrate, manifest_rows)
    if aux_keys is None:
        aux_keys = store.load_aux_keys(question_id)
    # key_schema_entry=None: v2 aux-injection substrates (protocol_spec_v2
    # §2.1) carry NO gold KU key — the candidate list is the registered aux
    # keys alone (same mapper, same blindness; FM-6 unmapped logging applies).
    candidates = (store.build_candidate_keys(key_schema_entry, aux_keys)
                  if key_schema_entry is not None else list(aux_keys))
    return substrate, candidates


async def replay_gated(substrate, key_schema_entry: dict, manifest_rows: list[dict],
                       *, family: str = "mem0",
                       aux_keys: list[store.CanonicalKey] | None = None,
                       mapper=None) -> GatedTimeline:
    """Gated replay of one substrate (manifest rows spliced via inject, aux keys
    defaulting to the registry — the replay_write_through calling convention).
    ``mapper`` = the v2.4 M1 key-mapper seam (default None = frozen path)."""
    substrate, candidates = _prepare_substrate(substrate, key_schema_entry,
                                               manifest_rows, aux_keys)
    return await run_gated_replay(
        substrate_stream(substrate), candidates, stream_id=substrate.question_id,
        family=family, session_order=list(substrate.session_order), mapper=mapper)


async def replay_ungated(substrate, key_schema_entry: dict, manifest_rows: list[dict],
                         *, family: str = "mem0",
                         aux_keys: list[store.CanonicalKey] | None = None,
                         mapper=None) -> GatedTimeline:
    """Ungated (immediate-supersession) replay of one substrate."""
    substrate, candidates = _prepare_substrate(substrate, key_schema_entry,
                                               manifest_rows, aux_keys)
    return await run_ungated_replay(
        substrate_stream(substrate), candidates, stream_id=substrate.question_id,
        family=family, session_order=list(substrate.session_order), mapper=mapper)


# ---------------------------------------------------------------------------
# Probing (deterministic; B4 probe points, store.probe semantics on ZepStore)
# ---------------------------------------------------------------------------


def probe_timeline(
    timeline: GatedTimeline,
    key: tuple,
    probe_point: str,
    *,
    turn: tuple | None = None,
    store_id: str = DEFAULT_STORE,
) -> str | None:
    """Deterministic lookup of `key` at a manifest-designated probe point,
    mirroring store.probe: full_ingestion = after the last user turn;
    old_session_boundary = after the final turn of the chronologically earlier
    session (session_order, never dates — P12); provenance_timeline = after the
    manifest placement turn. State is read via ZepStore.as_of_seq at the
    snapshot's recorded sequence number (returns the `(unknown)` sentinel
    verbatim when stored; presence semantics belong to the caller)."""
    if probe_point == "full_ingestion":
        snap = timeline.snapshots[-1]
    elif probe_point == "old_session_boundary":
        earlier = timeline.session_order[0]
        session_snaps = [s for s in timeline.snapshots if s["session_idx"] == earlier]
        snap = session_snaps[-1]
    elif probe_point == "provenance_timeline":
        if turn is None:
            raise ValueError("provenance_timeline probe requires turn=(session_idx, turn_idx)")
        matches = [s for s in timeline.snapshots
                   if (s["session_idx"], s["turn_idx"]) == tuple(turn)]
        if not matches:
            raise ValueError(f"no user-turn snapshot at turn {turn}")
        snap = matches[0]
    else:
        raise ValueError(f"unknown probe_point {probe_point!r}; "
                         f"expected one of {store.PROBE_POINTS}")
    zstore = timeline.stores.get(store_id)
    if zstore is None:
        return None
    return zstore.as_of_seq(tuple(key), snap["seq"].get(store_id, 0))


# ---------------------------------------------------------------------------
# Serialization (per-cell store timelines, arms.py output)
# ---------------------------------------------------------------------------


def _key_name(key: tuple) -> str:
    return "/".join(key)


def timeline_to_json(timeline: GatedTimeline) -> dict:
    """JSON-serializable dump of one replay: both ZepStore timelines
    (validity + transactional) per entry, plus every gate/checker/staged log."""
    stores_out = {}
    for sid, zstore in timeline.stores.items():
        histories = {}
        for key in zstore.keys():
            histories[_key_name(key)] = [
                {
                    "value": e.value,
                    "valid_from": e.valid_from,
                    "invalid_from": e.invalid_from,
                    "created_seq": e.created_seq,
                    "expired_seq": e.expired_seq,
                    "retro_invalidated": e.retro_invalidated,
                    "provenance": e.provenance,
                }
                for e in zstore.history(key)
            ]
        stores_out[sid] = {
            "history": histories,
            "events": [{**ev, "key": _key_name(ev["key"])} for ev in zstore.events],
        }
    return {
        "stream_id": timeline.stream_id,
        "family": timeline.family,
        "policy": timeline.policy,
        "session_order": list(timeline.session_order),
        "diagnostics": timeline.diagnostics,
        "snapshots": timeline.snapshots,
        "stores": stores_out,
        "gate_log": timeline.gate_log,
        "staged_log": timeline.staged_log,
        "checker_log": timeline.checker_log,
        "unmapped": timeline.unmapped,
        "n_mapping_rows": len(timeline.mapping_log),
        "n_extraction_calls": len(timeline.extraction_call_keys),
        "extraction_call_keys": timeline.extraction_call_keys,
    }
