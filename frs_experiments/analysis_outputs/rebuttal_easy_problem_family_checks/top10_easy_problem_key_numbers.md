# Top-10% problem concentration — key numbers (Reviewer yweD)

_Generated: 2026-05-22T16:18:00.881207+00:00_

## Method

Top-10% traces = same rule as `topk_ablation.py`: all pass@16 traces pooled per model×benchmark, keep traces with confidence ≥ the **90th percentile** (global top 10% by token-confidence).

## Headline (54 pairs)

- Mean **% of problems** with ≥1 selected trace: **63.79%** (median **65.28%**).
- Mean **max selected traces from one problem**: **11.06** (median **11.0**).
- Mean fraction of selected traces from **pass@16 = 16/16** problems: **0.164**.
- Mean fraction from problems with **≥12/16** correct: **0.4695**.
- Mean fraction from **≤4/16** correct (hard) problems: **0.2228**.
- Mean within-pair Spearman(**#selected**, **pass@16 acc**): **0.0522**.
- Pairs with **<50%** problem coverage: **9**.
- Pairs with **≥10** selected traces on one problem: **37**.

## Direct answers

### Does top-10% collapse onto only a few problems?

**Partially, on some pairs.** A non-trivial subset of pairs show heavy tail concentration (37 pairs with max≥10 traces from one problem). On average, problem coverage is broader than a handful of items (mean % problems represented ≈ 63.79%).

### How broad is problem coverage on average?

Roughly **63.79%** of problems contribute at least one top-10% trace (median **65.28%**). This is **not** a single-problem collapse for most pairs.

### Are selected traces overwhelmingly from pass@16=16/16 easy problems?

**Not overwhelmingly** (mean **16.4%** from 16/16 problems). Harder problems still contribute selected traces.

### Pairs with high concentration (acknowledge in paper)

- Qwen3-4B × SVAMP: max=16 selected, 65.2% problems represented
- Qwen3-4B × MATH500: max=16 selected, 45.2% problems represented
- Phi-4 × GPQA: max=16 selected, 41.3% problems represented
- Qwen3-4B × GPQA: max=15 selected, 42.6% problems represented
- Qwen3-4B × AQuA: max=15 selected, 42.5% problems represented
- Qwen2.5-Math × GPQA: max=15 selected, 38.6% problems represented
- DS-R1-7B × SVAMP: max=15 selected, 58.5% problems represented
- DS-R1-1.5B × GPQA: max=14 selected, 40.0% problems represented

### Low problem-coverage pairs

- Qwen2.5-Math × GPQA: 38.6% problems
- DS-R1-1.5B × GPQA: 40.0% problems
- Phi-4 × GPQA: 41.3% problems
- Qwen3-4B × AQuA: 42.5% problems
- Qwen3-4B × GPQA: 42.6% problems
- Qwen2.5-Math × MATH500: 44.8% problems
- Qwen3-4B × MATH500: 45.2% problems
- Qwen2.5-7B × MATH500: 48.8% problems

## Safe rebuttal claims

- Top-10% is computed over **traces**, not by cherry-picking a few problem IDs; a majority of problems typically contribute at least one high-confidence trace.
- Selection is **associated** with higher per-problem pass@16 (positive Spearman in many pairs); state this as correlation, not causation.
- FRS still uses **reasoning judge** scores within the filtered pool — not identical to pass@16 accuracy alone.

## Avoid overclaiming

- Do not say the filter is uniform over problems on every pair.
- Do not deny that high pass@16 problems supply many top-confidence traces on some benchmarks.
