# Selection-gain decisive analyses (COLM rebuttal)

## STEP 0 — Discovery

**Status:** cached — 54-cell means available
**54-cell selection gain:** `/Users/manaspathak11/Desktop/Everything/research paper/threshold/analysis/selection_gain_pair_level.csv` → `mean_selection_gain`
- mean over 50 questions of (RS(top_conf trace) - RS(random trace)), 0-1 scale
- n_pairs=54, n_questions/pair=50

**Supporting caches:**
- pair_level_54: `/Users/manaspathak11/Desktop/Everything/research paper/threshold/analysis/selection_gain_pair_level.csv`
- question_level: `/Users/manaspathak11/Desktop/Everything/research paper/threshold/analysis/selection_gain_question_level.csv`
- judge_outputs: `/Users/manaspathak11/Desktop/Everything/research paper/threshold/analysis/selection_gain_judge_outputs.csv`
- predictor_panel: `/Users/manaspathak11/Desktop/Everything/research paper/threshold/analysis/selection_gain_predictor_panel_merged.csv`
- prior_table14: `/Users/manaspathak11/Desktop/Everything/research paper/threshold/analysis/selection_gain_predictor_results.csv`
- frs_predictor_panel: `/Users/manaspathak11/Desktop/Everything/research paper/threshold/analysis/frs_predictor_panel.csv`
- trace0: `/Users/manaspathak11/Desktop/Everything/research paper/threshold/analysis_outputs/trace0_k1_judging/per_pair_scores.csv`
- frs_pass1: `/Users/manaspathak11/Desktop/Everything/research paper/threshold/global_pass1_frs_analysis/merged_pass1_frs_per_benchmark.csv`
- unfiltered: `/Users/manaspathak11/Desktop/Everything/research paper/threshold/analysis_outputs/unfiltered_reasoning/per_pair_scores.csv`
- frs_judge_bins: `/Users/manaspathak11/Desktop/Everything/research paper/threshold/reasoning_confidence_bins_results/judging_checkpoints/judged_*.json`
- trace0_checkpoints: `/Users/manaspathak11/Desktop/Everything/research paper/threshold/analysis_outputs/trace0_k1_judging/judging_checkpoints/trace0_judged_*.json`

## Experiment 1 — Table 14 predictors vs mean selection gain (54 pairs)

Sanity: FRS Pearson r ≈ 0.49; unfiltered ≈ 0.008.
| Predictor | Pearson r | p | Spearman ρ | p | n |
| --- | --- | --- | --- | --- | --- |
| FRS (%) | 0.4906 | 0.0002 | 0.4010 | 0.0027 | 54 |
| high-conf accuracy (%) | 0.2009 | 0.1452 | 0.1857 | 0.1788 | 54 |
| SNR | 0.1031 | 0.4582 | 0.1292 | 0.3516 | 54 |
| unfiltered reasoning (0-1) | 0.0078 | 0.9554 | -0.0647 | 0.6421 | 54 |
| pass@16 (%) | -0.0824 | 0.5537 | -0.1484 | 0.2841 | 54 |
| pass@1 (%) | -0.1281 | 0.3558 | -0.1445 | 0.2972 | 54 |
| single-trace (trace-0, 0-1) | -0.1044 | 0.4523 | -0.1159 | 0.4041 | 54 |
| top-1 conf. single trace (0-1 mean) | 0.5137 | 0.0001 | 0.4395 | 0.0009 | 54 |

*Predictor (b): `mean_top_conf_reasoning` from pair_level — mean RS of the top-confidence trace per question (50/50 judged in selection_gain_judge_outputs.csv, selection_type=top_conf).*

## Experiment 2 — Dependent correlation: FRS vs trace-0 / unfiltered

Paired bootstrap over 54 cells: Δr = r(predictor, gain) each resample.
| Test | r(A,gain) | r(B,gain) | r(A,B) | Δr | 95% CI | p(Δr≤0) |
| --- | --- | --- | --- | --- | --- | --- |
| FRS vs trace0 | 0.4906 | -0.1044 | 0.6700 | 0.5951 | [0.380, 0.801] | 0.0000 |
| FRS vs unfiltered | 0.4906 | 0.0078 | 0.8151 | 0.4828 | [0.326, 0.658] | 0.0000 |

## Experiment 3 — Paired amplification difference (shared denominator)

Δamp = [mean(FRS_gap) − mean(comparator_gap)] / mean(acc_gap) on the same close-accuracy pairs.
| Comparison | thresh (pp) | n | Δamp point | 95% CI | p(FRS≤comp) |
| --- | --- | --- | --- | --- | --- |
| FRS vs trace0 | 3.0 | 27 | 1.892 | [-0.907, 5.132] | 0.0944 |
| FRS vs unfiltered | 3.0 | 27 | 3.313 | [1.182, 6.041] | 0.0012 |
| FRS vs trace0 | 5.0 | 34 | 2.066 | [0.170, 4.144] | 0.0180 |
| FRS vs unfiltered | 5.0 | 34 | 2.739 | [1.290, 4.554] | 0.0001 |

## Experiment 4 — Estimator reliability (bootstrap SD of mean)

Median per-pair bootstrap SD: **trace-0 = 0.0370**, **FRS = 0.0305** (ratio trace0/FRS ≈ 1.21×)

## Provenance

- `/Users/manaspathak11/Desktop/Everything/research paper/threshold/analysis/selection_gain_pair_level.csv`
- `/Users/manaspathak11/Desktop/Everything/research paper/threshold/analysis/selection_gain_question_level.csv`
- `/Users/manaspathak11/Desktop/Everything/research paper/threshold/analysis/selection_gain_judge_outputs.csv`
- `/Users/manaspathak11/Desktop/Everything/research paper/threshold/analysis/selection_gain_predictor_panel_merged.csv`
- `/Users/manaspathak11/Desktop/Everything/research paper/threshold/analysis/selection_gain_predictor_results.csv`
- `/Users/manaspathak11/Desktop/Everything/research paper/threshold/analysis/frs_predictor_panel.csv`
- `/Users/manaspathak11/Desktop/Everything/research paper/threshold/analysis_outputs/trace0_k1_judging/per_pair_scores.csv`
- `/Users/manaspathak11/Desktop/Everything/research paper/threshold/global_pass1_frs_analysis/merged_pass1_frs_per_benchmark.csv`
- `/Users/manaspathak11/Desktop/Everything/research paper/threshold/analysis_outputs/unfiltered_reasoning/per_pair_scores.csv`
- `/Users/manaspathak11/Desktop/Everything/research paper/threshold/reasoning_confidence_bins_results/judging_checkpoints/judged_*.json`
- `/Users/manaspathak11/Desktop/Everything/research paper/threshold/analysis_outputs/trace0_k1_judging/judging_checkpoints/trace0_judged_*.json`