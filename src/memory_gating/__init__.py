"""memory_gating — evidence-gated semantic memory experiments (R1 layout).

R1 note: modules inside this package still import each other by their flat
top-level names (``import llm``, ``import store``, ...), exactly as they did
in the pre-restructure pilot/ directory. Consumers therefore put this
package directory itself on ``sys.path`` (tests via tests/conftest.py,
experiment runners via experiments/*/_bootstrap.py) and use the same flat
imports. Do NOT mix ``import memory_gating.llm`` with ``import llm`` — that
would create two module objects. The proper-package import style (relative
imports, subpackages) lands in stage R2.
"""
