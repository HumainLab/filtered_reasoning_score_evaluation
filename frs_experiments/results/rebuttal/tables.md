# Rebuttal analyses (cached data)

## Five-line summary
- 1. Amplification chain: FRS 7.35x at <=3pp (target ~7.4x); random-trace much lower; FRS only significant r vs selection gain (0.49).
- 2. Single-trace bootstrap: amp <=3pp 4.56 [3.46,5.64] vs FRS 7.91 [7.04,8.82]; ranking stability rho 0.91 vs FRS 0.94.
- 3. Top-1 conf: Spearman vs FRS ranking 0.75; bootstrap amp CI wider than FRS (variance-reduced).
- 4. Continuous regression: FRS×acc interaction 0.189 (p=0.0000); significant extra amplification vs random single-trace over full gap range.

## Discovery

- **frs_k10**: global_pass1_frs_analysis/merged_pass1_frs_per_benchmark.csv (frs_pct)
- **accuracy_table**: same CSV (pass1_pct); also paper tables via global_pass1_frs_pairwise_analysis
- **selection_gain**: analysis/selection_gain_pair_level.csv + selection_gain_question_level.csv + selection_gain_judge_outputs.csv
- **confidence**: Per-trace: topk_ablation.compute_trace_confidence on pass16 JSONL (not loaded for these analyses; scores pre-aggregated in selection-gain and FRS checkpoints)
- **paths**: ['/Users/manaspathak11/Desktop/Everything/research paper/threshold/global_pass1_frs_analysis/merged_pass1_frs_per_benchmark.csv', '/Users/manaspathak11/Desktop/Everything/research paper/threshold/analysis/selection_gain_pair_level.csv', '/Users/manaspathak11/Desktop/Everything/research paper/threshold/analysis/selection_gain_question_level.csv', '/Users/manaspathak11/Desktop/Everything/research paper/threshold/analysis_outputs/unfiltered_reasoning/per_pair_scores.csv', '/Users/manaspathak11/Desktop/Everything/research paper/threshold/reasoning_confidence_bins_results/judging_checkpoints/judged_*.json']

## Sanity check
- FRS amp <=3pp: 7.3533 (n=27)
- FRS amp <=5pp: 6.0666 (n=34)
- FRS Pearson r vs gain: 0.4906 (p=0.0002)
- Passed: True

## Analysis 1: Amplification decomposition

| Metric | <=3pp ratio | n | <=5pp ratio | n |
| --- | --- | --- | --- | --- |
| accuracy (pass@1) | 1.000 | 27 | 1.000 | 34 |
| single random-trace reasoning | 4.053 | 27 | 3.369 | 34 |
| all-trace unfiltered reasoning | 4.040 | 27 | 3.328 | 34 |
| FRS@10% | 7.353 | 27 | 6.067 | 34 |

Correlations vs mean selection gain (54 pairs):
| Metric | Pearson r | p |
| --- | --- | --- |
| accuracy (pass@1) | -0.1281 | 0.3558 |
| single random-trace reasoning | 0.0097 | 0.9447 |
| all-trace unfiltered reasoning | 0.0078 | 0.9554 |
| FRS@10% | 0.4906 | 0.0002 |
| top-1 conf (reference) | 0.5137 | 0.0001 |

## Analysis 2: Single-trace bootstrap (2000 draws)

| Estimator | amp<=3pp mean [CI] | amp<=5pp mean [CI] | rank rho mean [CI] |
| --- | --- | --- | --- |
| random_single_trace | 4.56 [3.46,5.64] | 3.70 [2.93,4.46] | 0.91 [0.86,0.96] |
| frs | 7.91 [7.04,8.82] | 6.47 [5.84,7.11] | 0.94 [0.89,0.97] |

## Analysis 3: Top-1-confidence steelman

- Point amp <=3pp: 7.858; <=5pp: 6.600
- Mean Spearman(top1 rank, FRS rank) over benchmarks: 0.745
- Bootstrap amp <=3pp: top1 8.20 [7.22,9.21] vs FRS 7.91 [7.04,8.82]
- Bootstrap rank stability: top1 0.91 vs FRS 0.94

## Analysis 4: Continuous-power regression (216 pairs, clustered)

- Interaction acc_gap × FRS: **0.1888** (SE 0.0358, p=0.0000)
- 95% CI: [0.1185, 0.2590]
- Paired t-test FRS gap vs random gap: p=0.0000

## Figures

- `figures/analysis1_amplification_chain.png`
- `figures/analysis1_correlations.png`
- `figures/analysis2_3_bootstrap_amp.png`
- `figures/analysis4_regression_gaps.png`

## Provenance

- `/Users/manaspathak11/Desktop/Everything/research paper/threshold/global_pass1_frs_analysis/merged_pass1_frs_per_benchmark.csv`
- `/Users/manaspathak11/Desktop/Everything/research paper/threshold/analysis/selection_gain_pair_level.csv`
- `/Users/manaspathak11/Desktop/Everything/research paper/threshold/analysis/selection_gain_question_level.csv`
- `/Users/manaspathak11/Desktop/Everything/research paper/threshold/analysis_outputs/unfiltered_reasoning/per_pair_scores.csv`
- `/Users/manaspathak11/Desktop/Everything/research paper/threshold/reasoning_confidence_bins_results/judging_checkpoints/judged_*.json`