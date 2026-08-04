# FRS predictor analysis — results

Generated: 2026-03-31T03:44:58.458013
Panel rows: 54 (full coverage: merged, unfiltered, top-10% high-conf, cumulative top-10 optional).

Pass@16: fraction of problems with ≥1 correct trace in each JSONL line’s `score` list (length may vary; lines with `len(score)≠16` are counted in `n_lines_with_trace_count_neq_16`).

## Source coverage summary

See `analysis/coverage_summary.csv` for artifact row counts and join keys.

## Outcome and circularity labels

| Variable | Role | Circularity risk |
|----------|------|------------------|
| `unfiltered_reasoning_mean` | **Primary outcome** | **Low–moderate** — same judge family as FRS; sampling differs from confidence-binned traces. |
| `pass1_pct`, `pass16_pct`, `high_conf_accuracy_pct` | Predictors / benchmarks | **Low** for pass@k; high-conf accuracy is **moderate** (accuracy on a confidence slice, not the outcome). |
| `snr` | Control from merged table | **Low–moderate** (derived from reasoning signal). |
| `cum_top10_mean_reasoning_0_1` | Not used as outcome here | **High** vs FRS (both bin/top-k style reasoning). |

## Key correlations with outcome (unfiltered)

| predictor              |   pearson_r |   pearson_p |   spearman_rho |   spearman_p |   n |
|:-----------------------|------------:|------------:|---------------:|-------------:|----:|
| frs_pct                |    0.815141 | 6.25394e-14 |       0.744177 |  1.12639e-10 |  54 |
| pass1_pct              |    0.846962 | 6.97532e-16 |       0.8667   |  2.48257e-17 |  54 |
| pass16_pct             |    0.327997 | 0.0154661   |       0.473725 |  0.000296648 |  54 |
| high_conf_accuracy_pct |    0.731412 | 3.35446e-10 |       0.666997 |  3.64881e-08 |  54 |
| snr                    |    0.287033 | 0.0353432   |       0.208885 |  0.129558    |  54 |

## OLS ladder (incremental validity; outcome = unfiltered reasoning)

| model_name        | predictors                                              |   n |      r2 |   r2_adj |
|:------------------|:--------------------------------------------------------|----:|--------:|---------:|
| m1_pass1_only     | pass1_pct                                               |  54 | 0.71734 |  0.71191 |
| m2_pass1_highconf | pass1_pct+high_conf_accuracy_pct                        |  54 | 0.77439 |  0.76554 |
| m3_add_pass16     | pass1_pct+high_conf_accuracy_pct+pass16_pct             |  54 | 0.77471 |  0.76119 |
| m4_add_snr        | pass1_pct+high_conf_accuracy_pct+pass16_pct+snr         |  54 | 0.77952 |  0.76153 |
| m5_add_frs        | pass1_pct+high_conf_accuracy_pct+pass16_pct+snr+frs_pct |  54 | 0.85048 |  0.8349  |

## Incremental R² (add FRS last)

- **ΔR² (m5 − m4):** adding `frs_pct` after `pass1_pct` + `high_conf_accuracy_pct` + `pass16_pct` + `snr` → **0.07095455947871632**
- **R² (m4, no FRS):** 0.7795241081040135
- **R² (m5, with FRS):** 0.8504786675827298
- **Residual view:** `figures/residual_vs_frs.png` plots FRS vs residual of outcome after m4 predictors; non-flat pattern indicates FRS aligns with variation not explained by those metrics.

## Pairwise near-ties on pass@1 (|Δpass@1| ≤ 2 pp)

- Qualifying pairs: **15** (sparse; interpret cautiously).
- Spearman(|ΔFRS|, |Δunfiltered|): ρ ≈ **0.121**, p ≈ **0.6664** (does not support strong separation in this small set).
- Spearman(|Δhighconf|, |Δunfiltered|): ρ ≈ **0.293**, p ≈ **0.2895**.

## Files

- `analysis/frs_predictor_panel.csv`
- `analysis/pass16_recomputed.csv`
- `/Users/manaspathak11/Desktop/Everything/research paper/threshold/analysis/coverage_summary.csv`
- `/Users/manaspathak11/Desktop/Everything/research paper/threshold/analysis/frs_predictor_correlations.csv`
- `/Users/manaspathak11/Desktop/Everything/research paper/threshold/analysis/frs_predictor_regressions.csv`
- `/Users/manaspathak11/Desktop/Everything/research paper/threshold/analysis/frs_predictor_pairwise_discrimination.csv`
- `/Users/manaspathak11/Desktop/Everything/research paper/threshold/analysis/figures/scatter_*.png`, `residual_vs_frs.png`
- `/Users/manaspathak11/Desktop/Everything/research paper/threshold/analysis/logs/frs_predictor_analysis.log`

## Answers (conservative, non-salesy)

1. **Strongest result in this run:** In the full panel (n=54), adding FRS to pass@1, high-confidence accuracy, pass@16, and SNR raises R² by a clear margin (see ΔR² above). Pass@1 alone already explains most variance (R²≈0.72); FRS is strongly correlated with the outcome (r≈0.82) but that alone does not prove incremental validity — the ΔR² step addresses that.
2. **Does FRS add predictive value beyond the other metrics?** **On this panel, yes in the sense of ΔR²:** the m4→m5 gap is positive and sizeable. Caveats: n=54 aggregate pairs; **judge overlap** between FRS and unfiltered outcome inflates shared variance; **multicollinearity** means individual coefficients (including FRS) are not stable for causal reading.
3. **Circularity risk of the main claim:** **Moderate.** The outcome is judge-based like FRS. Claims should be framed as **incremental association with another judge-derived reasoning aggregate**, not as independence from judgment. Pass@1 / pass@16 / high-conf accuracy ground part of the story in non-FRS signals.
4. **What we can say in the paper now:** We can report that **after controlling for standard accuracy-style metrics available in the repo**, FRS still accounts for **additional** variance in mean unfiltered reasoning scores across model×benchmark pairs, and show the ladder and ΔR². We should **not** claim FRS is the best single metric or that it subsumes pass@1.
5. **What would strengthen the claim:** Held-out questions or cross-validation at the **question** level; reporting **VIF/partial correlations**; an outcome less tied to the same judge pipeline; or trace-level selection-gain tests if clean joins exist.

## Bonus: trace-level joins

Not fully audited here. If `per_pair_scores` `jsonl_path` / `sampled_traces_path` align with confidence-bin artifacts on `(model, benchmark, question_idx)`, a selection-gain test could be sketched; requires verifying keys and no duplicate question mapping.