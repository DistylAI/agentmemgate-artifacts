# Preventing Speculation Contamination in Conversational Assistant Memory — paper artifacts

Datasets, prompts, and code for the PALM 2026 submission "AgentMemGate: Preventing
Speculation Contamination in Conversational Assistant Memory."

A note on layout: directory names (`results/`, `experiments/`, `protocol/`) mirror the
source repository layout that the code reads from; data files under `results/` here are
released datasets and frozen evaluation outputs, not run products.

## Datasets

| Dataset (paper name) | Files | Size |
|---|---|---|
| Core benchmark | `results/v2.4/m4/arcs.jsonl`, fields in `m4_keys.jsonl` | 60 conversations: 20 plans come true, 20 abandoned, 20 still open |
| Held-out set | `data/external/ai_memory_plan_handling/dataset1_standalone/` (`conversations.jsonl`, `labels.jsonl`, `test_questions.jsonl`); `HOLDOUT_SPLIT.json` names the 147 evaluation conversations (the remaining 53 are the development slice; the split file also covers a second dataset not used in the paper). The copies under `eval_sets/` are byte-identical inputs read by the evaluation code. | 200 conversations |
| Controlled-difficulty splits | `results/unify/arcs.jsonl`, fields in `unify_keys.jsonl`, specs in `arc_spec.jsonl` | 180 arcs; 179 evaluated (`unify_exclusions.json` lists the one removed during validation): hedge strength at 5 levels, confirmation wording at 4 |
| Classifier statement set | `results/test1/eval_items.jsonl` (the 163 items with `provenance == "natural"`, each carrying the statement, its context, the frozen classifier output, and its reference label); labels in `results/labels/natural_gold.jsonl` + `adjudications.jsonl` | 163 natural statements |

Each conversation row carries the rendered multi-session dialogue, the plan statement and
its outcome label (came true / abandoned / still open), the tracked store field, the
ground-truth current value over time, and the probe question. Store fields are value-free
by construction: field names and descriptions derive from attribute names only, never from
answer values.

`data/locomo/` is the public LoCoMo benchmark payload (upstream distribution, unmodified).
LongMemEval payloads are not redistributed; the rows below that need them say so.

## Code

- `src/memory_gating/` — the pipeline: extraction, field matching (with the plan-frame
  retry), classification, gate routing, staging and resolution, the profile store, and
  scoring utilities. All prompts are in `protocol/prompts/` (extractors `extractor.txt`
  cautious / `extractor_zep.txt` literal, field matcher `key_mapper.txt`, classifier
  `classifier_short.txt`, resolution predicates `predicate_generator.txt`, checker,
  equivalence judge, permissive matcher `llm_field_matcher_v1.txt`, and the dataset
  generation prompts).
- `experiments/` and `results/tier1/classifier_suite.py` — the runners, mapped to paper
  results below. Each is a command-line tool; `python <runner> --help` lists its
  subcommands (most have a run step and a scoring step).

## Setup

Python 3.11+ and:

```
pip install openai anthropic numpy scipy
```

(`mem0ai` and `graphiti-core` are additionally required only for the Mem0/Graphiti product
row.) Environment variables:

- `OPENAI_API_KEY` — all main results; the systems and the scoring judge run the pinned
  `gpt-5.5-2026-04-23` snapshot.
- `ANTHROPIC_API_KEY` and `MEMGATE_SYSTEM_MODEL=claude-opus-4-8` — only for the
  second-model-family replication (leave `MEMGATE_SYSTEM_MODEL` unset for everything else).
- `EXTERNAL_TEST_RUN_AUTHORIZED=1` — required for any run on the held-out 147-conversation
  evaluation slice (the code fail-closes to the development slice without it).

Run everything from the repository root. Run steps call the model APIs; scoring steps read
the run's recorded outputs, so run before score, and note the cross-runner orderings in the
table below.

## Reproducing the paper's results

| Paper result | Runner | Prerequisites |
|---|---|---|
| Tables 2–3, core-benchmark rows; correction and completed-change controls | `experiments/v2_full/m4_eval.py` | none |
| Figures 2–3 and Tables 7–8 (hedge-strength and confirmation-wording curves) | `experiments/v2_full/unify_eval.py` | none |
| Tables 2–3, held-out rows; Table 6 | `experiments/external_eval/runner_external.py`, then `scorer_d1.py` | `EXTERNAL_TEST_RUN_AUTHORIZED=1` |
| Mem0 / Graphiti product measurements (Section 1) | `experiments/baselines/runner_products.py`, then `scorer_products.py` | `pip install mem0ai graphiti-core`; `EXTERNAL_TEST_RUN_AUTHORIZED=1` |
| Plan-typed read-time baseline (Appendix A.1) | `experiments/v2_full/plan_typed.py`; held-out extension: `experiments/external_eval/plan_typed_d1_heldout.py` | run+score `m4_eval` and `unify_eval` first (its scoring compares against their outputs); held-out extension additionally as row 3 |
| Retry ablation (Section 6.3) | `experiments/external_eval/tier1_retry.py` | `unify_eval` first; the held-out arm additionally as row 3 |
| Second-model-family replication (Appendix A.3) | `experiments/claude_repl/claude_repl.py` | `ANTHROPIC_API_KEY`, `MEMGATE_SYSTEM_MODEL=claude-opus-4-8`; the held-out arm additionally as row 3 |
| Classifier validation: AUROC, per-class table (Section 4, Appendix A.5) | recompute directly from the frozen outputs and labels shipped in `results/test1/eval_items.jsonl` (fields `classifier.probs`/`s_ev` vs `gold`; interval and AUROC helpers in `src/memory_gating/stats.py`); the frozen aggregate numbers are in `results/analysis/classifier_standalone_eval.json` and `results/test1/results.json` | none |
| Classifier stability repeats and paired prompt-objective baselines (Appendix A.5) | `results/tier1/classifier_suite.py` | the frozen model-response cache of the original runs, which this bundle does not include (the suite verifies byte-identical prompt reproduction against it); full archive upon publication |
| 161-question public-benchmark task deltas (Section 6.1) | `experiments/v2_full/task_deltas.py` | LongMemEval payloads (fetch upstream) and the recorded store timelines of the original runs (134 MB, not in this bundle; full archive upon publication) |
| Permissive-matcher experiment (Section 6.3) | `experiments/v2_full/llm_matcher.py` (library; driven by the planted suite of `experiments/v2_full/graded_eval.py`) | LongMemEval payloads and recorded intermediates not in this bundle; full archive upon publication |

Statistical conventions throughout: proportions carry 95% Wilson confidence intervals;
paired differences use a cluster bootstrap (seed 20260702, 10,000 resamples, one cluster
per conversation). Both are implemented in `src/memory_gating/stats.py`.

## License

Code and the constructed datasets are released under CC BY 4.0 upon publication. The LoCoMo
payload is redistributed unmodified under its upstream CC BY-NC 4.0 license.
