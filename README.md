# Preventing Speculation Contamination in Conversational Assistant Memory — paper artifacts

Anonymized artifact bundle for the PALM 2026 submission "AgentMemGate: Preventing
Speculation Contamination in Conversational Assistant Memory". It contains the datasets,
prompts, and code needed to reproduce the paper's results, and nothing else.

## Contents

| Path | What it is |
|---|---|
| `results/v2.4/m4/arcs.jsonl` (+ `m4_keys.jsonl`) | **Core benchmark** — 60 plan-to-outcome conversations (20 came-true / 20 abandoned / 20 still-open) with their per-conversation store fields |
| `results/unify/arcs.jsonl` (+ `unify_keys.jsonl`, `arc_spec.jsonl`, `unify_exclusions.json`) | **Controlled-difficulty splits** — 180 arcs (179 used; the exclusions file lists the one removed during validation): hedge-strength ladder (5 levels) and confirmation-wording ladder (4 levels) |
| `results/test1/eval_items.jsonl` | Classifier evaluation items; the 163 with `provenance == "natural"` are the natural-statement set of the paper's classifier validation |
| `results/labels/natural_gold.jsonl` + `adjudications.jsonl` | Reference labels for the natural statements (model-annotated with human audit; see paper Section 4) |
| `data/locomo/` | The public LoCoMo benchmark payload (upstream distribution, unmodified) |
| `protocol/prompts/` | Every frozen prompt: extractors (`extractor.txt` cautious, `extractor_zep.txt` literal), field matcher (`key_mapper.txt`), classifier (`classifier_short.txt`), resolution predicates (`predicate_generator.txt`), checker, equivalence judge, permissive matcher (`llm_field_matcher_v1.txt`), and the dataset generation/rendering prompts |
| `src/memory_gating/` | The pipeline: extraction, field matching (+ plan-frame retry), classification, gate routing, staging/resolution, store, scoring utilities |
| `experiments/`, `results/tier1/classifier_suite.py` | The runners, mapped to paper results below |

**Not included:** the 147-conversation held-out set (third-party-provided; see the paper's
provenance paragraph), the LongMemEval payloads (fetch from upstream; the classifier items
embed the statement spans and immediate context they need), and the LLM response cache
(fresh runs re-call the pinned model).

## Setup

Python 3.11+, `pip install openai anthropic numpy`. Set `OPENAI_API_KEY` (all main results;
systems and judge run the pinned `gpt-5.5-2026-04-23`) and `ANTHROPIC_API_KEY` (second-family
replication only). Run everything from the repository root. Each runner is a CLI with
`run` and `score` subcommands; `python <runner> --help` lists options. Without the response
cache, runs re-call the model APIs; scoring re-uses recorded run outputs.

## Reproducing the paper's results

| Paper result | Runner |
|---|---|
| Tables 2–3 core-benchmark rows; correction/completed-change controls | `experiments/v2_full/m4_eval.py` |
| Figures 2–3, Tables 7–8 (hedge-strength and confirmation-wording ladders) | `experiments/v2_full/unify_eval.py` |
| Tables 2–3 held-out rows, Table 6 (requires the held-out set) | `experiments/external_eval/runner_external.py` + `scorer_d1.py` |
| Product measurements (Mem0 / Graphiti; requires the held-out set) | `experiments/baselines/runner_products.py` + `scorer_products.py` |
| 161-question public-benchmark task deltas | `experiments/v2_full/task_deltas.py` |
| Classifier validation: AUROC + CI, per-class table, stability repeats, paired prompt-objective baselines | `experiments/v1_pilot/eval_classifier.py`, then `results/tier1/classifier_suite.py` |
| Plan-typed read-time baseline (Appendix A.1) | `experiments/v2_full/plan_typed.py`; held-out extension `experiments/external_eval/plan_typed_d1_heldout.py` |
| Retry ablation (Section 6.3 paired deltas) | `experiments/external_eval/tier1_retry.py` |
| Permissive-matcher experiment (Section 6.3) | `experiments/v2_full/llm_matcher.py` (on the planted suite of `experiments/v2_full/graded_eval.py`) |
| Second-model-family replication (Appendix A.3) | `experiments/claude_repl/claude_repl.py` |

Statistical conventions: proportions carry 95% Wilson intervals; paired differences use a
cluster bootstrap (seed 20260702, 10,000 resamples, one cluster per conversation). Both are
implemented in `src/memory_gating/stats.py`.

## Dataset format

Each conversation/arc row carries the rendered multi-session conversation, the plan
statement and its outcome label (came true / abandoned / still open), the tracked store
field, the ground-truth current value over time, and the probe question. Store fields are
value-free by construction: field names and descriptions derive from attribute names only,
never from answer values.

## License

Code and constructed datasets: CC BY 4.0 upon publication. The LoCoMo payload retains its
upstream license.
