"""sys.path bootstrap for experiment runners (R1 split layout).

Runners import library modules by their flat top-level names (``import
llm``, ...) and each other across the v1_pilot/v2_full split. Importing
this module (``import _bootstrap``) as the first project import makes the
library package directory and both experiment directories importable no
matter which entry point was launched. Retired in stage R2 (proper package
imports).
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

for _p in (
    _ROOT / "src" / "memory_gating",
    _ROOT / "experiments" / "v1_pilot",
    _ROOT / "experiments" / "v2_full",
):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
