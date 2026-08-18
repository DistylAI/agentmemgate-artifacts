"""llm_matcher.py — FIX A: the clean LLM field matcher, preregistered in
protocol/CHANGELOG.md ("2026-08-10 — AMENDMENT (before any run call): FIX A
mechanism replaced with the clean LLM field matcher"; criterion, substrates
and disclosures per the parent entry "2026-08-10 — PREREGISTRATION: two
pre-specified fix attempts").

THE MECHANISM (frozen in the amendment; no curated vocabulary, no token
heuristics): on a frozen-key-mapper DECLINE, ONE pinned-model call decides
field-or-none. Inputs rendered per call: the tuple's entity/attribute/value,
its statement span, and the conversation's registered field names. Output
contract: exactly one registered field name verbatim, or "NONE" — anything
else is treated as NONE and counted as a parse miss. The deciding prompt is
FROZEN (protocol/prompts/llm_field_matcher_v1.txt, sha256 row in
protocol/prompts/README.md, committed before any call).

SEAM (the adopted plan-stem wrapper's pattern, mirrored from
experiments/v2_full/mapper_fix.py / store.make_tuple_key_mapper):
  1. FIRST pass = store._map_tuple_key verbatim — the rendered prompt is
     byte-identical to the frozen v2 mapper call, so the pass is cache-shared
     with the frozen runs (zero fresh spend on every tuple the frozen cells
     already mapped or declined).
  2. First pass maps -> return the first-pass result with meta == {} — the
     byte path is untouched.
  3. First pass DECLINES -> one fallback call with the frozen matcher prompt
     (trace tag suffixed `-llmmatch`), parsed STRICTLY:
       - "NONE"                          -> stay declined (decision "none");
       - a candidate name, verbatim      -> map to that candidate
                                            (decision "matched");
       - anything else                   -> stay declined, decision
                                            "parse_miss", counted on
                                            ``mapper.parse_misses``.
     On a non-match the FIRST-pass row stays the row of record (funnel
     evidence preserved, exactly like the stem wrapper's declined re-offer).

LOCAL INSTALL ONLY (assert-mode-off pattern preserved): the mapper this
module builds is passed into the arm's replay seam
(gated_store mapper=..., m4_eval._extract_and_map mapper=...) for the FIX A
arm only. It is NEVER installed over store._map_tuple_key — the runner
asserts `assert_global_mapper_untouched()` at arm entry and teardown
(m4_eval.assert_mapper_mode_off's pattern), so the four frozen cells stay
byte-untouched.

BUDGET: the fallback call goes through the project's llm.call_llm, so it is
content-address cached and instrumented by whatever SharedCallBudget guard
the CALLER has installed for its stage (the mapper_fix pattern: the runner
installs e.g. forecast_v22.SharedCallBudget(stage, ...) around the replay;
this module never installs a guard or selects a stage itself).

AUDIT (mapper_fix's audit-metadata pattern): every fallback call appends
{trace_tag, tuple, candidates, model_output_field, decision, fresh} to
``mapper.audit_rows`` for the runner to dump; ``mapper.fallback_calls`` and
``mapper.parse_misses`` are running counters. The returned 4-tuple's ``meta``
additionally carries mapping-mode provenance through
gated_store._unpack_map_result into the mapping/gate logs (meta is {} on
every first-pass-mapped tuple, so untouched rows log byte-identically).
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict

import _bootstrap  # noqa: F401  (R1 layout: puts src/ + experiments/ on sys.path)

import llm
import store

from paths import PROMPTS_DIR

MODE = "llm_field_matcher"
DECLINE = "NONE"

MATCHER_PROMPT_PATH = PROMPTS_DIR / "llm_field_matcher_v1.txt"
MATCHER_PROMPT = MATCHER_PROMPT_PATH.read_text()
MATCHER_PROMPT_SHA256 = hashlib.sha256(
    MATCHER_PROMPT_PATH.read_bytes()).hexdigest()


def render_matcher_prompt(tup, candidates) -> str:
    """The per-call USER prompt of the frozen matcher: registered field names
    (names only — no descriptions, no probe questions, no domain hints) + the
    tuple with its statement span."""
    lines = ["REGISTERED FIELD NAMES:"]
    for i, cand in enumerate(candidates, 1):
        lines.append(f"{i}. {cand.name}")
    lines += [
        "",
        "EXTRACTED TUPLE:",
        f"entity: {tup.entity}",
        f"attribute: {tup.attribute}",
        f"value: {tup.value}",
        f"statement: {tup.statement_span}",
    ]
    return "\n".join(lines)


def _fallback_cached(user_prompt: str) -> bool:
    """True when llm.py's content-addressed cache already holds this fallback
    call (the call replays free; recorded as fresh=False in the audit row)."""
    key = llm._cache_key(llm.MODEL, llm.REASONING_EFFORT,
                         MATCHER_PROMPT, user_prompt)
    return (llm.CACHE_DIR / f"{key}.json").exists()


def parse_matcher_output(output: dict, candidates) -> tuple:
    """STRICT parse of the matcher's JSON output -> (candidate | None,
    decision). The `field` value must be the string "NONE" or one of the
    candidate names VERBATIM (exact string, no normalization); everything
    else — missing key, non-string, unregistered name, case/whitespace
    variants — is decision "parse_miss" and treated as NONE."""
    field = output.get("field") if isinstance(output, dict) else None
    if field == DECLINE:
        return None, "none"
    by_name = {c.name: c for c in candidates}
    if isinstance(field, str) and field in by_name:
        return by_name[field], "matched"
    return None, "parse_miss"


def assert_global_mapper_untouched() -> None:
    """The assert-mode-off pin (m4_eval.assert_mapper_mode_off's pattern):
    this module's mapper is LOCAL-only — store._map_tuple_key must never
    carry a mapping mode (wrapper closures carry a ``mode`` attribute; the
    frozen function carries none)."""
    mode = getattr(store._map_tuple_key, "mode", "off")
    assert mode == "off", (
        f"store._map_tuple_key carries mapping mode {mode!r} — a wrapper is "
        "installed globally; the FIX A llm_field_matcher is preregistered as "
        "a LOCAL arm mapper only (frozen cells byte-untouched)")


def make_llm_fallback_mapper():
    """Build the FIX A mapper: an async callable
    ``(tup, candidates, *, trace_tag) -> (cand | None, raw_key, rationale,
    meta)`` for the local mapper seam (gated_store._unpack_map_result accepts
    the 4-tuple; meta is {} on every first-pass-mapped tuple).

    Attributes on the returned callable (mapper_fix's audit pattern):
      mode            "llm_field_matcher" (assert_global_mapper_untouched's
                      wrapper marker)
      prompt_sha256   sha256 of the frozen prompt file (freeze-table witness)
      audit_rows      one dict per fallback call: {trace_tag, tuple,
                      candidates, model_output_field, decision, fresh}
      fallback_calls  running count of fallback calls made
      parse_misses    running count of strict-parse failures (treated as NONE)
    """

    async def mapper(tup, candidates, *, trace_tag: str):
        # 1. FIRST pass — the frozen mapper verbatim (byte-identical prompt,
        #    cache-shared with the frozen runs).
        cand, raw, rationale = await store._map_tuple_key(
            tup, candidates, trace_tag=trace_tag)
        if cand is not None:
            return cand, raw, rationale, {}

        # 2. DECLINE -> one pinned-model fallback call, frozen prompt.
        user_prompt = render_matcher_prompt(tup, candidates)
        fresh = not _fallback_cached(user_prompt)
        output = await llm.call_llm(MATCHER_PROMPT, user_prompt,
                                    trace_tag=f"{trace_tag}-llmmatch")
        cand2, decision = parse_matcher_output(output, candidates)
        field = output.get("field") if isinstance(output, dict) else None

        mapper.fallback_calls += 1
        if decision == "parse_miss":
            mapper.parse_misses += 1
        mapper.audit_rows.append({
            "trace_tag": trace_tag,
            "tuple": asdict(tup),
            "candidates": [c.name for c in candidates],
            "model_output_field": field,
            "decision": decision,
            "fresh": fresh,
        })

        meta = {
            "mapping_mode": MODE,
            "first_pass": {"mapped_key": "none", "model_output_key": raw,
                           "rationale": rationale},
            "fallback": {"decision": decision, "model_output_field": field,
                         "fresh": fresh},
            "fallback_mapped": decision == "matched",
        }
        if cand2 is None:
            # first pass stays the row of record (funnel evidence preserved)
            return None, raw, rationale, meta
        return cand2, field, f"{MODE} fallback: matched", meta

    mapper.mode = MODE
    mapper.prompt_sha256 = MATCHER_PROMPT_SHA256
    mapper.audit_rows = []
    mapper.fallback_calls = 0
    mapper.parse_misses = 0
    return mapper
