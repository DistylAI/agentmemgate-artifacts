"""Central path constants for the repository layout (R1 restructure, 2026-07).

Every module that needs a repo-level path imports it from here instead of
deriving it from ``__file__``. This is the single place that encodes where
the frozen artifacts and generated outputs live:

  ROOT/
    protocol/prompts/   frozen prompt files (formerly pilot/prompts/)
    protocol/gold/      frozen gold artifacts: key schemas, candidate pools,
                        pairs (formerly research/*.json*, the data subset)
    protocol/specs/     frozen protocol specs (never read at runtime)
    results/            generated experiment outputs (incl. cache/ + traces/)
    data/               benchmark datasets (LongMemEval gitignored, LoCoMo in)

Path relocations are recorded in protocol/CHANGELOG.md; content hashes of the
frozen artifacts are unchanged by the R1 move (verified, see the changelog
entry "Repository restructure R1").
"""

from pathlib import Path

#: Repository root (memory-gating/). This file lives at src/memory_gating/.
ROOT = Path(__file__).resolve().parents[2]

PROTOCOL_DIR = ROOT / "protocol"
PROMPTS_DIR = PROTOCOL_DIR / "prompts"   # frozen prompts + freeze table
GOLD_DIR = PROTOCOL_DIR / "gold"         # frozen gold data artifacts
SPECS_DIR = PROTOCOL_DIR / "specs"       # frozen protocol specs

RESULTS_DIR = ROOT / "results"           # generated outputs (see .gitignore)
DATA_DIR = ROOT / "data"                 # benchmark datasets

#: Legacy alias: before R1 the gold data artifacts lived in research/, and
#: modules referred to them via a RESEARCH_DIR constant. The reports/notes
#: that also lived there moved to research_notes/ (not read at runtime).
RESEARCH_DIR = GOLD_DIR
