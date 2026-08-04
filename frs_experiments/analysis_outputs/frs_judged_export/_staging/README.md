# FRS judged trace export

## Contents

- `checkpoints/judged_<Model>__<Benchmark>.json` — 54 files, 250 judged traces each (13,500 total)
- `all_judged_traces.csv` — flat table with all traces and 4 rubric dimensions
- `per_pair_summary.csv` — row counts per model×benchmark
- `reasoning_by_confidence_bin.csv` — aggregated bin-level FRS statistics
- `reasoning_sampling_metadata.json` — bin definitions and sampling parameters

## Sampling protocol

For each model×benchmark pair:
1. Pool all pass@16 traces; rank by low-prob-tail confidence
2. Keep top 50% of traces by confidence
3. Split into 5 equal-count bins (0–10%, …, 40–50% within pool)
4. Judge 50 traces per bin → **250 judged traces per pair**

## Rubric dimensions (1–5 each)

- faithfulness, utility, coherence, factuality
- reasoning_score = normalized mean in [0, 1]

## Joining to raw CoT text

Use pass@16 JSONL (`source_pass16_jsonl_by_model*/**/*.jsonl`):
- question_idx = JSONL field `idx`
- trace_idx = index into `code[]`, `score[]`, `pred[]`

Judge checkpoints do **not** include raw chain-of-thought text.
