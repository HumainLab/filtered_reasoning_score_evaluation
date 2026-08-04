# Reasoning by confidence bins — methods summary

This document describes `reasoning_confidence_bins.py`, which implements a **slice-based reasoning quality** analysis aligned with the pooled top-K **accuracy** setup in `topk_ablation.py`, while using the same **LLM judge** as `topk_judge_eval.py` (GPT-4o-mini, four 1–5 subscores, normalized to \([0,1]\) via `reasoning_score_from_judge`).

## Data source

- Input: pass@16 JSONL under `source_pass16_jsonl_by_model*/**/*.jsonl`, discovered with `build_file_map()` from `topk_ablation.py` (no hardcoded model/dataset list).
- Per trace: `question`, `gt`, `code[]` (CoT), `score[]`, `chosen_token_probs_per_path.epoch_0[]`.
- Confidence: same definition as elsewhere — mean of the lowest 10% of token probabilities in the trace (`compute_trace_confidence`).

## Pooling and ranking

1. For each `(model, dataset)`, concatenate **all** traces from all questions into one pool.
2. Sort the pool by **descending** confidence (highest first).

## Top-50% restriction

3. Keep only the **top half** of the ranked pool by count: `n_keep = floor(N * 0.5)` (at least one trace if `N ≥ 1`).

## Disjoint bins (within the top-50% slice)

4. Split this top-50% slice into **five equal-count, disjoint bins** along rank order:
   - `0-10`, `10-20`, `20-30`, `30-40`, `40-50` — percent labels refer to the **deciles of the top-50% slice** (not the global pool).
5. Implementation: `bin_id = min(4, (i * 5) // m)` for rank `i ∈ {0,…,m-1}` in the sorted top-50% slice of length `m`.

## Sampling for judging

6. Within each bin, draw up to **50** traces **without replacement** (or all traces if fewer than 50).
7. RNG: `numpy.random.default_rng` with a deterministic seed derived from SHA-256 of `model`, `dataset`, bin label, and a user `--seed` base.
8. **Resume:** judged `(idx, trace_idx, bin_label)` keys are stored in `judging_checkpoints/judged_<model>__<dataset>.json`; re-runs skip already-judged traces.

## Judge

9. Same `Judge` class and prompt as `topk_judge_eval.py` (faithfulness, utility, coherence, factuality; temperature 0).
10. Output per trace: four integers + normalized reasoning score in \([0,1]\).

## Aggregates

**Per-bin CSV (`reasoning_by_confidence_bin.csv`):**

- Mean / std / stderr of judged reasoning scores in the bin.
- 95% CI: Student’s *t* on the judged sample when \(n≥2\); bootstrap percentiles also stored.
- `mean_accuracy_population`: fraction correct on **all** traces in the bin (not only judged) for calibration vs accuracy.

**Cumulative CSV (`reasoning_cumulative_topk.csv`):**

- Labels `top10` … `top50`: combine judged traces from bins 1..1, 1..2, …, 1..5.
- **Mean:** unweighted mean of **all** judged scores in the union of bins (equivalent to weighting by judged counts per bin).
- **CI:** bootstrap on the **concatenated** judged scores (same weighting principle).

## Plots

- Per-dataset: reasoning vs bin; cumulative reasoning; twin-axis reasoning vs population accuracy on the **same** bin structure; optional overlay of **global** top-K% accuracy from `topk_ablation_results.csv` (different definition — labeled in figure titles).

## What this is not

- Not per-question binning; bins are **global** over the pooled trace distribution.
- Not the same slice as `topk_ablation` top-10% / top-20% over the **full** pool — those are referenced only in the optional comparison figure.

## Caveats

- API failures yield missing scores; they are excluded from means.
- When \(n=1\) judged trace in a bin, CI is one-sided / bootstrap only.
- Global accuracy curves and slice-based reasoning curves answer related but **not identical** scientific questions; compare with care.
