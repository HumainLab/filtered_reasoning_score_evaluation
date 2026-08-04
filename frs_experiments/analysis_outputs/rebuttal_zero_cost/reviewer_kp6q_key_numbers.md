# Reviewer kp6q — key numbers (zero-cost)

_Generated: 2026-05-22T15:42:51.903866+00:00_

## One-sentence answer

FRS is not redundant with pass@1 or a single unfiltered trace sample: at similar pass@1, FRS still separates models (larger |ΔFRS| when |Δpass@1|≤2 pp), rankings differ from unfiltered RS, and all four rubric dimensions contribute (LORO re-score from cached judges).

## Headline statistics

- **FRS vs pass@1 model-rank Spearman:** 0.15 (pass@1 alone is not a substitute ranking).
- **FRS vs unfiltered RS model-rank Spearman:** 0.45 (single-trace-per-question unfiltered sample differs materially).
- **Median |ΔFRS| when |Δpass@1|≤2 pp:** 9.9 pp (similar-accuracy amplification).
- **Median |Δpass@1| when |Δpass@1|≤2 pp (baseline):** 1.0 pp.
- **Median |Δunfiltered RS| when |Δpass@1|≤2 pp:** 5.38 pp.
- **FRS LOBO transfer (precomputed):** mean Spearman ≈ 0.712 (cross-benchmark).
- **FRS vs selection-gain Pearson:** 0.4906 (deployment proxy; modest).
- **Trace-0 RS coverage (judged):** mean 6.4% of judged traces — **do not claim full-corpus single-trace RS.**

## Leave-one-rubric-out (Spearman vs full 4D, top bin)

- **drop_coherence:** 0.9929
- **drop_factuality:** 0.9861
- **drop_faithfulness:** 0.9964
- **drop_utility:** 0.9972

## What we can claim

- Confidence filtering changes accuracy monotonically (top-10% vs full pool) and concentrates traces (`confidence_filtering_necessity_summary.csv`).
- FRS captures reasoning quality in high-confidence strata beyond pass@1 and beyond unfiltered one-trace-per-question judging.
- Rubric dimensions are correlated but not interchangeable (LORO ρ < 1).

## What we cannot claim

- That one cheap trace (trace 0) reproduces FRS trends on full benchmarks (trace-0 RS is ~6% judged coverage).
- That FRS is independent of the judge or confidence (same judge family; selection policy uses confidence).
- Causal deployment gains from selection-gain alone (mean gain ≈ 0 at macro level).

## Incomplete analyses (trace-0 RS)

- Judged trace-0 pairs: 867 trace scores across 54 checkpoints.
- Use `trace0_accuracy_pct` for full-corpus **accuracy**; use `trace0_rs_judged_pct` only with coverage caveat.
