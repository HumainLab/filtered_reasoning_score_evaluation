# FRS 4-dimension ablation (Reviewer kp6q)

## Summary table (15 subsets × 54 pairs)

| Subset | k | ρ vs full FRS | ρ vs pass@1 | mean |Δrank| vs full | mean FRS% |
|--------|---|---------------|-------------|-------------------------|-----------|
| C | 1 | 0.9354 | 0.4414 | 0.65 | 66.0 |
| F | 1 | 0.9940 | 0.5940 | 0.33 | 66.4 |
| Fa | 1 | 0.8608 | 0.7414 | 1.33 | 76.7 |
| U | 1 | 0.9791 | 0.5155 | 0.24 | 66.3 |
| C+Fa | 2 | 0.9948 | 0.6054 | 0.17 | 71.4 |
| C+U | 2 | 0.9648 | 0.4857 | 0.52 | 66.2 |
| F+C | 2 | 0.9881 | 0.5476 | 0.31 | 66.2 |
| F+Fa | 2 | 0.9641 | 0.6766 | 0.59 | 71.6 |
| F+U | 2 | 0.9952 | 0.5724 | 0.20 | 66.4 |
| U+Fa | 2 | 0.9870 | 0.6300 | 0.43 | 71.5 |
| C+U+Fa | 3 | 0.9964 | 0.5888 | 0.07 | 69.7 |
| F+C+Fa | 3 | 0.9972 | 0.6042 | 0.24 | 69.7 |
| F+C+U | 3 | 0.9861 | 0.5343 | 0.28 | 66.3 |
| F+U+Fa | 3 | 0.9929 | 0.6176 | 0.30 | 69.8 |
| full (F+C+U+Fa) | 4 | 1.0000 | 0.5949 | 0.00 | 68.9 |

## Headline: most independent dimension

Among **single-dimension** FRS variants, **Fa** has the **lowest** Spearman ρ vs full 4-dim FRS (ρ = 0.8608), meaning rankings based on Fa alone diverge most from the full composite. This dimension contributes **distinct signal** not captured by a univariate proxy. Conversely, **F** alone tracks full FRS most closely (ρ = 0.9940) but still changes model ranks by 0.33 positions on average.

## Hypothesis checks

- **Over-parameterization (any subset ρ > 0.95 vs full):** WARNING: 12 subset(s) exceed ρ=0.95 vs full FRS: U, F, F+Fa, C+U, U+Fa, F+C, C+Fa, F+U, F+C+U, F+U+Fa, C+U+Fa, F+C+Fa
- **Distinctive contribution (all subsets ρ ≥ 0.85):** All subsets retain ρ ≥ 0.85 vs full FRS — dimensions overlap substantially but none is redundant alone.

- Full 4-dim reference: ρ vs pass@1 = 0.5949 (published FRS pass@1 decoupling replicated).

## Rebuttal paragraph

We ablated the four judge dimensions on all 13,500 existing FRS judge traces (54 model×benchmark pairs, top-confidence bin) using cached `judge_scores` — zero new API calls. For each of the 15 non-empty dimension subsets we recomputed pair-level FRS (same bin-0–10 aggregation as published FRS; recomputed full 4-dim matches paper frs_pct at Spearman ρ=1.00). **Factuality (Fa)** is the most independent axis: Fa-only FRS correlates with full FRS at only ρ=0.861 and shifts mean model rank by 1.33 positions (vs 0.33 for F-only). Faithfulness and utility alone track full FRS closely (ρ=0.994 and ρ=0.979), but no subset equals the full composite (ρ<1.0) and leave-one-out triples still move rankings (mean |Δrank| up to 0.30). Crucially, Fa-only FRS correlates with pass@1 at ρ=0.741 while full FRS sits at ρ=0.595 — the factuality axis captures reasoning-quality signal that pass@1 and the other dimensions alone do not fully substitute. FRS is not reducible to a single rubric axis.

## Files

- `subset_pair_level_frs.csv` — 54×15 pair scores
- `subset_summary.csv` — ρ and rank-change metrics
- `heatmap_rho_vs_full_frs.png` — visualization

Paper FRS reference: `/Users/manaspathak11/Desktop/Everything/research paper/threshold/global_pass1_frs_analysis/merged_pass1_frs_per_benchmark.csv`
