"""sys.path bootstrap for the claude_repl runner (R1 split layout).

Same shim as experiments/v2_full/_bootstrap.py, extended with the
external_eval directory (the D1 replication reuses runner_external /
adapter_d1 / scorer_d1 / holdout_guard by import).
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

for _p in (
    _ROOT / "src" / "memory_gating",
    _ROOT / "experiments" / "v1_pilot",
    _ROOT / "experiments" / "v2_full",
    _ROOT / "experiments" / "external_eval",
):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
