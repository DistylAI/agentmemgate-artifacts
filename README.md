# Preventing Speculation Contamination in Conversational Assistant Memory — paper artifacts

Datasets, prompts, and code for the PALM 2026 submission "AgentMemGate: Preventing
Speculation Contamination in Conversational Assistant Memory."

A note on layout: directory names (`results/`, `experiments/`, `protocol/`) mirror the
source repository layout that the code reads from; files under `results/` here are released
datasets and runners, not run outputs.

## Datasets

| Dataset (paper name) | Files | Size |
|---|---|---|
| Core benchmark | `results/v2.4/m4/arcs.jsonl`, fields in `m4_keys.jsonl` | 60 conversations: 20 plans come true, 20 abandoned, 20 still open |
| Held-out set | `data/external/ai_memory_plan_handling/dataset1_standalone/` (`conversations.jsonl`, `labels.jsonl`, `test_questions.jsonl`); `HOLDOUT_SPLIT.json` names the 147 evaluation conversations (the remaining 53 are the development slice; the split file also covers a second dataset not used in the paper) | 200 conversations |
| Controlled-difficulty splits | `results/unify/arcs.jsonl`, fields in `unify_keys.jsonl`, specs in `arc_spec.jsonl` | 180 arcs; 179 evaluated (`unify_exclusions.json` lists the one removed during validation): hedge strength at 5 levels, confirmation wording at 4 |
| Classifier statement set | `results/test1/eval_items.jsonl` (the 163 items with `provenance == "natural"`); reference labels in `results/labels/natural_gold.jsonl` + `adjudications.jsonl` | 163 natural statements with evidence-type labels |

Each conversation row carries the rendered multi-session dialogue, the plan statement and
its outcome label (came true / abandoned / still open), the tracked store field, the
ground-truth current value over time, and the probe question. Store fields are value-free
by construction: field names and descriptions derive from attribute names only, never from
answer values.

`data/locomo/` is the public LoCoMo benchmark payload (upstream distribution, unmodified),
used by the public-benchmark experiments. LongMemEval payloads are not redistributed; the
classifier statement set embeds the statement spans and immediate context it needs, so no
download is required to reproduce the classifier results.

## Code

- `src/memory_gating/` — the pipeline: extraction, field matching (with the plan-frame
  retry), classification, gate routing, staging and resolution, the profile store, and
  scoring utilities. All prompts are in `protocol/prompts/` (extractors `extractor.txt`
  cautious / `extractor_zep.txt` literal, field matcher `key_mapper.txt`, classifier
  `classifier_short.txt`, resolution predicates `predicate_generator.txt`, checker,
  equivalence judge, permissive matcher `llm_field_matcher_v1.txt`, and the dataset
  generation prompts).
- `experiments/` and `results/tier1/classifier_suite.py` — the runners, mapped to paper
  results below. Each is a command-line tool with `run` and `score` subcommands
  (`python <runner> --help`).

## Setup

Python 3.11+, `pip install openai anthropic numpy`. Set `OPENAI_API_KEY` (all main results;
the systems and the scoring judge run the pinned `gpt-5.5-2026-04-23` snapshot) and
`ANTHROPIC_API_KEY` (second-model-family replication only). Run from the repository root.
Runs call the model APIs; scoring re-uses recorded run outputs.

## Reproducing the paper's results

| Paper result | Runner |
|---|---|
| Tables 2–3, core-benchmark rows; correction and completed-change controls | `experiments/v2_full/m4_eval.py` |
| Figures 2–3 and Tables 7–8 (hedge-strength and confirmation-wording curves) | `experiments/v2_full/unify_eval.py` |
| Tables 2–3, held-out rows; Table 6 | `experiments/external_eval/runner_external.py`, then `scorer_d1.py` |
| Mem0 / Graphiti product measurements (Section 1) | `experiments/baselines/runner_products.py`, then `scorer_products.py` |
| 161-question public-benchmark task deltas (Section 6.1) | `experiments/v2_full/task_deltas.py` |
| Classifier validation: AUROC with CI, per-class table, repeated-run stability, paired prompt-objective baselines (Section 4, Appendix A.5) | `experiments/v1_pilot/eval_classifier.py`, then `results/tier1/classifier_suite.py` |
| Plan-typed read-time baseline (Appendix A.1) | `experiments/v2_full/plan_typed.py`; held-out extension: `experiments/external_eval/plan_typed_d1_heldout.py` |
| Retry ablation (Section 6.3) | `experiments/external_eval/tier1_retry.py` |
| Permissive-matcher experiment (Section 6.3) | `experiments/v2_full/llm_matcher.py`, on the planted suite built by `experiments/v2_full/graded_eval.py` |
| Second-model-family replication (Appendix A.3) | `experiments/claude_repl/claude_repl.py` |

Statistical conventions throughout: proportions carry 95% Wilson confidence intervals;
paired differences use a cluster bootstrap (seed 20260702, 10,000 resamples, one cluster
per conversation). Both are implemented in `src/memory_gating/stats.py`.

## License

Code and the constructed datasets are released under CC BY 4.0 upon publication. The LoCoMo
payload retains its upstream license.
