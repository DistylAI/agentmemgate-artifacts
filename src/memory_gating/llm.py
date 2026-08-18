"""Standardized LLM client for all pilot experiments.

Every experiment call goes through call_llm(): OpenAI Responses API,
model gpt-5.5, reasoning effort medium, JSON object output, async with
retries and bounded concurrency. Traces are written per call for audit.

Different-family replication seam (protocol/CHANGELOG.md 2026-08-11
preregistration): call_llm carries a second provider branch keyed off the
model-name prefix ``claude-`` (Anthropic Messages API). The SYSTEM model is
overridable via the MEMGATE_SYSTEM_MODEL environment variable (default
unchanged: gpt-5.5-2026-04-23); the equivalence-judge model is pinned
separately as JUDGE_MODEL and never follows the override — the scoring
instrument stays frozen gpt-5.5 on both families (prereg deviation (d)).
The cache-key function is UNTOUCHED: the model string is already a key
component, so Claude and gpt-5.5 entries can never collide and the frozen
gpt-5.5 cache stays valid.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

#: System model for every experiment call that does not pin a model
#: explicitly. Env-overridable for the preregistered different-family
#: replication (MEMGATE_SYSTEM_MODEL=claude-opus-4-8); default unchanged.
MODEL = os.environ.get("MEMGATE_SYSTEM_MODEL", "gpt-5.5-2026-04-23")
#: The frozen equivalence-judge / scoring-instrument model. Judge call
#: sites pass this EXPLICITLY (store.values_equivalent) so the instrument
#: can never silently migrate with MEMGATE_SYSTEM_MODEL.
JUDGE_MODEL = "gpt-5.5-2026-04-23"
REASONING_EFFORT = "medium"
MAX_CONCURRENCY = 10
MAX_RETRIES = 5
#: Output cap (incl. thinking) for the claude- branch; the frozen-suite
#: outputs are small JSON objects.
CLAUDE_MAX_TOKENS = 16000

from paths import RESULTS_DIR

TRACE_DIR = RESULTS_DIR / "traces"
CACHE_DIR = RESULTS_DIR / "cache"

_client = AsyncOpenAI(timeout=600)
#: Lazy Anthropic client (created on first claude- call; llm.py owns the
#: retry loop, so the SDK's own retries are disabled).
_anthropic_client = None
_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)


def _anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import AsyncAnthropic
        _anthropic_client = AsyncAnthropic(timeout=600, max_retries=0)
    return _anthropic_client


def _strip_json_fences(text: str) -> str:
    """Remove a single markdown code fence if the model wrapped its JSON.

    The claude- branch has no API-level JSON mode (prereg deviation (b));
    the frozen prompts' explicit "Return only JSON" schema instruction does
    the real work, and a fence is the one mechanical wrapper failure mode.
    Non-fenced text passes through unchanged."""
    t = text.strip()
    if t.startswith("```"):
        t = t[t.index("\n") + 1:] if "\n" in t else ""
        t = t.rstrip()
        if t.endswith("```"):
            t = t[:-3].rstrip()
    return t


async def _provider_call(model: str, effort: str, system_prompt: str,
                         user_prompt: str) -> tuple[str, str, Optional[dict]]:
    """One raw provider call -> (output_text, served_model, usage).

    claude- prefix -> Anthropic Messages API with adaptive thinking +
    output effort mapped from the OpenAI reasoning-effort vocabulary
    (prereg deviation (c): the sanctioned depth control on Opus 4.8; fixed
    token budgets are not expressible on this model). The served
    ``response.model`` is returned so every trace and cache entry records
    it (prereg deviation (a): no dated Opus 4.8 snapshot exists — alias
    pin + response.model echo). Anything else -> the frozen OpenAI
    Responses call, byte-identical to the pre-seam body."""
    if model.startswith("claude-"):
        resp = await _anthropic().messages.create(
            model=model,
            max_tokens=CLAUDE_MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            thinking={"type": "adaptive"},
            output_config={"effort": effort},
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        usage = resp.usage.model_dump() if resp.usage else None
        return _strip_json_fences(text), resp.model, usage
    response = await _client.responses.create(
        model=model,
        reasoning={"effort": effort},
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
        ],
        text={"format": {"type": "json_object"}},
    )
    usage_r = response.usage.model_dump() if response.usage else None
    return response.output_text, response.model, usage_r


def _cache_key(model: str, effort: str, system_prompt: str, user_prompt: str) -> str:
    payload = json.dumps([model, effort, system_prompt, user_prompt], ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


async def call_llm(
    system_prompt: str,
    user_prompt: str,
    *,
    trace_tag: str,
    model: str = MODEL,
    reasoning_effort: str = REASONING_EFFORT,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Run one Responses API call and return the parsed JSON object.

    Calls are cached on (model, effort, system, user) so reruns and analysis
    re-executions are deterministic and free. Pass use_cache=False for calls
    that must be fresh (e.g. self-consistency reruns).
    """
    cache_path = CACHE_DIR / f"{_cache_key(model, reasoning_effort, system_prompt, user_prompt)}.json"
    if use_cache and cache_path.exists():
        return json.loads(cache_path.read_text())["output"]

    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with _semaphore:
                start = time.perf_counter()
                output_text, served_model, usage = await _provider_call(
                    model, reasoning_effort, system_prompt, user_prompt)
                elapsed = time.perf_counter() - start
            parsed = json.loads(output_text)
            if use_cache:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                # Atomic write (temp + rename): concurrent shard processes can
                # hit the same key (shared host-turn extractions); a plain
                # write lets another process read a torn file. pid-suffixed
                # temp so two writers never share a temp path.
                tmp = cache_path.with_suffix(f".{os.getpid()}.tmp")
                tmp.write_text(json.dumps(
                    {"model": served_model, "tag": trace_tag, "output": parsed},
                    ensure_ascii=False,
                ))
                tmp.replace(cache_path)
            _write_trace(
                tag=trace_tag,
                model=served_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_text=output_text,
                usage=usage,
                duration_seconds=round(elapsed, 3),
                attempt=attempt,
            )
            return parsed
        except Exception as exc:
            last_exc = exc
            logger.warning("[%s attempt %d/%d] %s", trace_tag, attempt, MAX_RETRIES, exc)
            if attempt == MAX_RETRIES:
                raise
            await asyncio.sleep(2 ** (attempt - 1))
    raise RuntimeError("unreachable") from last_exc


def _write_trace(*, tag: str, **payload: Any) -> None:
    try:
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        safe_tag = "".join(c for c in tag if c.isalnum() or c in ("-", "_"))
        name = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}_{safe_tag}.json"
        (TRACE_DIR / name).write_text(json.dumps({"tag": tag, **payload}, ensure_ascii=False, indent=2))
    except Exception as exc:
        logger.warning("trace write failed: %s", exc)


async def gather_llm(coros: list, *, desc: str = "") -> list:
    """Await a list of call_llm coroutines; concurrency is bounded by the shared semaphore."""
    results = await asyncio.gather(*coros, return_exceptions=True)
    failures = [r for r in results if isinstance(r, Exception)]
    if failures:
        logger.error("%s: %d/%d calls failed; first error: %s", desc, len(failures), len(results), failures[0])
    return results
