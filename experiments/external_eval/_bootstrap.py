"""sys.path bootstrap for the external-eval runners (R1 split layout).

Same shim as experiments/v2_full/_bootstrap.py, plus this package's own
directory so the external-eval modules import each other by flat name
(``import holdout_guard``, ``import adapter_d1``, ...) no matter which entry
point was launched.
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
