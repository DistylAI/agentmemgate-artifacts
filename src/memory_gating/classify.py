"""Test-1 evidence-type classifier (protocol_spec.md section 2.3 firewall side).

Uses the frozen SHORT operational prompt (prompts/classifier_short.txt) and the
identical item packaging as the labelers (label.format_item_prompt) so the only
asymmetry between gold labeling and the classifier is the system prompt.

s_ev = P(completed_event) + P(explicit_correction) per protocol section 3.3.
"""

from __future__ import annotations

import logging

import llm
from label import CLASSES, COMMIT_CLASSES, format_item_prompt, load_prompt

logger = logging.getLogger(__name__)

CLASSIFIER_PROMPT_NAME = "classifier_short.txt"


def validate_probs(raw_probs: dict) -> tuple[dict[str, float], bool]:
    """Validate the classifier's {probs} output: exactly the four classes,
    non-negative, sum ~1. Renormalizes (and logs) when the sum is off.

    Returns (normalized probs, renormalized flag).
    """
    if set(raw_probs) != set(CLASSES):
        raise ValueError(f"classifier probs keys {sorted(raw_probs)} != {sorted(CLASSES)}")
    vals = {cls: float(raw_probs[cls]) for cls in CLASSES}
    if any(v < 0 for v in vals.values()):
        raise ValueError(f"classifier returned negative probability: {vals}")
    total = sum(vals.values())
    if total <= 0:
        raise ValueError(f"classifier probabilities sum to {total}")
    renormalized = abs(total - 1.0) > 1e-3
    if renormalized:
        logger.warning("classifier probs sum to %.4f; renormalizing", total)
    return {cls: v / total for cls, v in vals.items()}, renormalized


def s_ev_from_probs(probs: dict[str, float]) -> float:
    return sum(probs[cls] for cls in COMMIT_CLASSES)


def argmax_class(probs: dict[str, float]) -> str:
    """Deterministic argmax (ties broken by CLASSES declaration order)."""
    return max(CLASSES, key=lambda cls: probs[cls])


async def classify_item(item: dict, *, use_cache: bool = True) -> dict:
    """One classifier call on a packaged item. Returns
    {probs, s_ev, argmax, max_prob, renormalized}."""
    raw = await llm.call_llm(
        load_prompt(CLASSIFIER_PROMPT_NAME),
        format_item_prompt(item),
        trace_tag=f"classify-{item['item_id']}",
        use_cache=use_cache,
    )
    probs, renormalized = validate_probs(raw["probs"])
    if renormalized:
        logger.warning("renormalized probs for item %s", item["item_id"])
    top = argmax_class(probs)
    return {
        "probs": probs,
        "s_ev": s_ev_from_probs(probs),
        "argmax": top,
        "max_prob": probs[top],
        "renormalized": renormalized,
    }
