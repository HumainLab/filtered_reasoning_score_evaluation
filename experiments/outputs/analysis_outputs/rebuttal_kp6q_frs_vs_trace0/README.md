# kp6q FRS vs trace-0 rebuttal analysis

Generated: 2026-05-23T01:22:20.100521+00:00

## Question

Does FRS provide more signal than single-trace (trace-0) or unfiltered/all-traces reasoning score,
especially among models with similar pass@1?

## Inputs

| Source | Path | Status |
|--------|------|--------|
| FRS + pass@1 | `/Users/manaspathak11/Desktop/Everything/research paper/threshold/global_pass1_frs_analysis/merged_pass1_frs_per_benchmark.csv` | 54 rows |
| Trace-0 RS | `/Users/manaspathak11/Desktop/Everything/research paper/threshold/analysis_outputs/trace0_k1_judging/trace0_k1_vs_frs_comparison.csv` | 54 rows |
| Unfiltered RS | `/Users/manaspathak11/Desktop/Everything/research paper/threshold/analysis_outputs/unfiltered_reasoning/per_pair_scores.csv` | available |

Panel coverage: **54/54** model×benchmark pairs.

## Methods

1. **Near-equal accuracy:** Within each benchmark, all model pairs with |Δpass@1| ≤ {2,3,5} pp.
   Compare mean |ΔFRS|, |Δtrace-0|, |Δunfiltered| and amplification ratios.

2. **LOBO transfer:** Macro mean over 5 train benchmarks → held-out benchmark (6 folds).
   Spearman and Pearson for each train→test metric pair.

3. **Rank reversals:** Pairs where FRS and baseline disagree on winner (≥2.0 pp both sides).
   LOBO FRS from other benchmarks validates reversal direction.

4. **Bootstrap CIs:** 2000 resamples, seed 123.

## Interpretation

- **Amplification > 1** near equal pass@1 → metric separates models beyond accuracy.
- **FRS→FRS LOBO ρ >> trace-0→FRS** → FRS ranking transfers; trace-0 macro does not predict held-out FRS.
- **Rank reversals with LOBO FRS agreement** → FRS re-ranking is cross-benchmark signal, not noise.

## Outputs

See `key_numbers.md`, `rebuttal_paragraph.md`, and CSV summaries in this directory.
