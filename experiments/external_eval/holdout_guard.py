"""holdout_guard.py — HARD dev/test guard for the external dataset package
(protocol/CHANGELOG.md 2026-07-10 intake preregistration rule (2) + the
2026-07-30 "adapter phase authorized" entry: "only HOLDOUT_SPLIT dev ids may
be ingested before the one-shot test run; test ids assert-blocked").

The split is FROZEN (data/external/ai_memory_plan_handling/HOLDOUT_SPLIT.json,
seed external-intake-split:20260710): Dataset 1 = 53 dev / 147 test
conversations, Dataset 2 = 26 dev / 100 test patches. The memory systems may
ingest DEV items only, for adapter debugging and smoke; the TEST slices are
run exactly once, with the frozen system configurations, under the
preregistered one-shot authorization.

ENFORCEMENT: every adapter build path and every runner ingest path calls
``assert_ingest_allowed(dataset, item_id)``. A TEST-slice item raises
``HoldoutViolation`` unless the environment carries
``EXTERNAL_TEST_RUN_AUTHORIZED=1`` — which is set ONLY for the preregistered
one-shot test run (after the endpoints entry is committed). Unknown ids
raise too (fail closed: an id outside the frozen split is never ingestable).

The split file is read and validated at import time (module-level check):
counts must match the frozen 53/147 + 26/100 and the slices must be
disjoint. Any drift is an artifact-integrity failure -> hard ImportError,
never a silent recompute.
"""

from __future__ import annotations

import json
import os

import _bootstrap  # noqa: F401  (R1 layout: puts src/ + experiments/ on sys.path)

from paths import DATA_DIR

EXTERNAL_PKG = DATA_DIR / "external" / "ai_memory_plan_handling"
SPLIT_PATH = EXTERNAL_PKG / "HOLDOUT_SPLIT.json"

#: The frozen split seed (2026-07-10 intake prereg) — pinned, not trusted.
SPLIT_SEED = "external-intake-split:20260710"

#: The one-shot authorization flag. Set ONLY for the preregistered test run.
ENV_FLAG = "EXTERNAL_TEST_RUN_AUTHORIZED"

#: Frozen slice sizes per the 2026-07-10 intake preregistration.
EXPECTED_COUNTS = {
    "dataset1": {"dev": 53, "test": 147},
    "dataset2": {"dev": 26, "test": 100},
}


class HoldoutViolation(RuntimeError):
    """A TEST-slice (or unknown) external item was about to be built/ingested
    without the one-shot test-run authorization."""


def _load_and_validate_split() -> dict:
    split = json.loads(SPLIT_PATH.read_text())
    if split.get("seed") != SPLIT_SEED:
        raise ImportError(
            f"HOLDOUT_SPLIT.json seed {split.get('seed')!r} != frozen "
            f"{SPLIT_SEED!r} — the frozen split has drifted; hard stop")
    for ds, want in EXPECTED_COUNTS.items():
        for sl, n in want.items():
            got = len(split[ds][sl])
            if got != n:
                raise ImportError(
                    f"HOLDOUT_SPLIT.json {ds}/{sl} has {got} ids, frozen "
                    f"split says {n} — hard stop")
        overlap = set(split[ds]["dev"]) & set(split[ds]["test"])
        if overlap:
            raise ImportError(
                f"HOLDOUT_SPLIT.json {ds} dev/test overlap: {sorted(overlap)}")
    return split


#: Module-level check: the split is loaded + validated at import time.
SPLIT = _load_and_validate_split()

_SLICE_OF = {
    ds: {**{i: "dev" for i in SPLIT[ds]["dev"]},
         **{i: "test" for i in SPLIT[ds]["test"]}}
    for ds in EXPECTED_COUNTS
}


def dev_ids(dataset: str) -> list[str]:
    """The frozen DEV slice ids, sorted (the only ids ingestable pre-test-run)."""
    return sorted(SPLIT[dataset]["dev"])


def test_ids(dataset: str) -> list[str]:
    return sorted(SPLIT[dataset]["test"])


def slice_of(dataset: str, item_id: str) -> str:
    """'dev' or 'test'; raises HoldoutViolation for ids outside the frozen
    split (fail closed)."""
    if dataset not in _SLICE_OF:
        raise HoldoutViolation(f"unknown external dataset {dataset!r}")
    sl = _SLICE_OF[dataset].get(item_id)
    if sl is None:
        raise HoldoutViolation(
            f"{dataset} item {item_id!r} is not in the frozen HOLDOUT_SPLIT "
            "— refusing to build/ingest an item outside the split")
    return sl


def test_run_authorized() -> bool:
    return os.environ.get(ENV_FLAG) == "1"


def assert_ingest_allowed(dataset: str, item_id: str) -> str:
    """The HARD guard. Called by every adapter build and runner ingest path.
    Returns the item's slice on success."""
    sl = slice_of(dataset, item_id)
    if sl == "test" and not test_run_authorized():
        raise HoldoutViolation(
            f"{dataset} item {item_id!r} is in the held-out TEST slice — "
            "building/ingesting it is blocked until the preregistered "
            f"one-shot test run (set {ENV_FLAG}=1 ONLY for that run, after "
            "the endpoints entry is committed; changelog 2026-07-10 rule (2) "
            "+ 2026-07-30 authorization)")
    return sl
