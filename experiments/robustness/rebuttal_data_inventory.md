# Rebuttal data inventory — FRS codebase audit

**Audit date:** 2026-05-22  
**Scope:** Read-only inspection of `analysis/`, `analysis_outputs/`, `analysis_exports/`, `reasoning_confidence_bins_results/`, judging checkpoints, logs, and related CSV/JSON/parquet. **No new experiments were run.**

**Scan progress (logged):**
1. Directory manifests for `analysis/` (excluding `cache/`), `analysis_outputs/`, `analysis_exports/`, `reasoning_confidence_bins_results/`
2. Schema sampling via pandas/json on representative artifacts
3. Aggregate counts: 54 FRS judge checkpoints (13,500 traces), 54 unfiltered (5,400), 5,400 selection-gain cache entries, 54 parquets, 27 top-k judge JSONs (math only)
4. Cross-reference to five reviewer concern areas

---

## Executive summary

| Reviewer concern | Partially answered? | Strongest existing artifacts | Main gap |
|:---|:---:|:---|:---|
| **1.** Single-trace vs unfiltered vs FRS | **Yes (pair-level)** | `per_pair_scores.csv`, `paper_frs_by_benchmark.csv`, `merged_pass1_frs_per_benchmark.csv`, `frs_predictor_panel.csv` | **Trace-0 RS** not systematically judged; `base_reasoning` is a different pass@1-era metric |
| **2.** Is confidence filtering necessary? | **Yes** | `topk_ablation_results.csv`, `correctness_conditioned.csv`, proxy/SC robustness exports | “No filter” RS on **all** traces not judged; filter vs no-filter **same judge** |
| **3.** Rubric component ablations | **Partial** | Per-dimension scores in all judge JSON; `diagnostics/dimension_correlations.csv` | **Leave-one-rubric-out FRS** not precomputed |
| **4.** Selection-gain prediction | **Yes** | Full `selection_gain_*` tables + 5,400 cached judge outputs | FRS↔gain correlation modest; macro mean gain ≈ 0 |
| **5.** Cross-benchmark transfer | **Yes** | `frs_cross_benchmark_results.csv`, LOCO/LOMO `generalization_results.csv` | n=9 models per fold — illustrative only |

**Raw data sufficient without new generation:** Yes — 54× pass@16 JSONL (~18 GB across `source_pass16_jsonl_by_model*`).  
**Sufficient without new judge calls for:** confidence rankings, pass@1/trace-0 **accuracy**, top-K coverage, LORO-rubric FRS (re-score from stored dimensions), recomputation of bin aggregates from checkpoints.  
**Requires new judge API for:** full-corpus single-trace-0 RS, all-trace RS, second judge, judging traces never sampled.

---

## 1. Existing reusable analyses (by reviewer concern)

### Concern 1 — Single-trace vs unfiltered vs FRS

| Path | Contains | Schema (key columns) | Rebuttal use | Reusable? | Recompute? |
|:---|:---|:---|:---:|:---:|:---:|
| `global_pass1_frs_analysis/paper_frs_by_benchmark.csv` | Wide FRS + `FRS_Avg` / `Acc_Avg` | `model` + 6 benchmark FRS columns | **FRS** leaderboard | **Yes** | No |
| `global_pass1_frs_analysis/merged_pass1_frs_per_benchmark.csv` | FRS + pass@1 + SNR | `model`, `benchmark`, `frs_pct`, `pass1_pct`, `base_reasoning`, `snr` | Three-metric panel | **Yes** | No |
| `global_pass1_frs_analysis/paper_pass1_reasoning_by_benchmark.csv` | Pass@1-era reasoning | `model`, `benchmark`, `base_reasoning`, `base_acc`, `snr` | **Single-trace-style** reasoning (external eval; not GPT judge) | **Yes** | No |
| `analysis_outputs/unfiltered_reasoning/per_pair_scores.csv` | Unfiltered judge baseline | `model`, `dataset`, `mean_reasoning_score` (0–1), `n_judged_traces`=100, paths | **Unfiltered RS** (100 q/pair, 1 trace/q) | **Yes** | No |
| `analysis_outputs/unfiltered_reasoning/ranking_comparison_vs_frs.csv` | Model ranks | `unfiltered_reasoning_avg`, `frs_avg`, ranks, `delta_rank` | Rank separation narrative | **Yes** | No |
| `analysis/frs_predictor_panel.csv` | Merged predictors + outcome | 13 cols incl. `frs_pct`, `pass1_pct`, `unfiltered_reasoning_mean`, `pass16_pct`, `high_conf_accuracy_pct` | Regression / correlation table | **Yes** | No |
| `analysis/frs_predictor_results.md` + `frs_predictor_correlations.csv` | OLS ladder, ΔR² | predictor ↔ `unfiltered_reasoning_mean` | FRS incremental validity vs pass@1 | **Yes** | No |
| `reasoning_confidence_bins_results/reasoning_by_confidence_bin.csv` | Binned judged RS | `model`, `dataset`, `bin_label`, `mean_reasoning_score`, CIs, `mean_accuracy_population` | FRS bin structure | **Yes** | No |
| `reasoning_confidence_bins_results/frs_by_model_benchmark_threshold.csv` | FRS @ K threshold | `model`, `benchmark`, `K_threshold`, `frs_score`, `ci_lower`, `ci_upper` | Threshold-style FRS | **Yes** | No |
| `analysis_exports/reasoning_converges_faster_csv/per_problem_accuracy_pass1_trace0_from_pass16.csv` | **Accuracy** trace 0 | `model`, `benchmark`, `problem_idx`, `pass1_trace0` | Single-trace **correctness** only | **Yes** | No (from JSONL) |
| `analysis_exports/reasoning_converges_faster_csv/per_problem_reasoning_judge_sparse_aggregated.csv` | Sparse judged RS by problem | `problem_idx`, `mean_reasoning_score_0_1`, `n_judged_traces_this_problem` | Partial trace-level RS | **Partial** | Join only |
| `temp0_analysis_results/analysis1_single_trace_frs.csv` | T=0 k=1 vs T=0.7 k=16 gaps | `gap_t0_k1`, `gap_t07_k16` | Single-trace **regime** (not standard RS table) | **Partial** | No |
| `topk_judge_results/*_topk_judge.json` (27 files) | Most vs least conf per **question** | `results[].most_confident/least_confident` + `judge_scores` | Per-question conf extremes (GSM8K/MATH500/SVAMP only) | **Partial** | No |
| `reasoning_confidence_bins_results/judging_checkpoints/judged_*.json` (54) | **Per-trace** RS + dims | `idx`, `trace_idx`, `confidence`, `correct`, `judge_scores`, `reasoning_score` | Trace-level RS; only **6.4%** are `trace_idx==0` | **Partial** | Aggregate only |
| `analysis_outputs/unfiltered_reasoning/judging_checkpoints/unfiltered_judged_*.json` (54) | 100 traces/pair | `judged_samples["idx:trace_idx"]` + `judge_raw` dims | Unfiltered trace-level; **6.8%** trace 0 | **Partial** | Aggregate only |

**Verdict (concern 1):** Pair-level **FRS vs unfiltered vs pass@1** is rebuttal-ready. A clean **three-way RS table** on the **same trace definition** (e.g. trace 0 only) is **not** fully judged — only accuracy for trace 0 exists at full problem coverage.

---

### Concern 2 — Is confidence filtering necessary?

| Path | Contains | Schema | Rebuttal use | Reusable? | Recompute? |
|:---|:---|:---|:---:|:---:|:---:|
| `topk_ablation_results/topk_ablation_results.csv` | Accuracy in top-K% by confidence | `model`, `dataset`, `top_k_pct`, `accuracy`, `n_traces`, `mean_confidence` | Filter **helps accuracy** monotonicity | **Yes** | No |
| `topk_ablation_results/topk_accuracy_k10_50.csv` | Paper-style top-K slice | same family | Same | **Yes** | No |
| `correctness_conditioned_results/correctness_conditioned.csv` | Median split | `acc_high_conf_half`, `acc_low_conf_half`, `gap_pp` | High vs low conf **accuracy** gap | **Yes** | No |
| `analysis_exports/confidence_proxy_robustness/proxy_benchmark_results_k10.csv` | FRS@bin0 under 3 proxies | `proxy_name`, `frs_k10_bin010` | Ranking robust to confidence **definition** | **Yes** | No |
| `analysis/confidence_proxy_robustness_report.md` | Narrative + rank stability | — | Text for rebuttal | **Yes** | No |
| `analysis_exports/self_consistency_proxy_robustness/sc_proxy_benchmark_results_k10.csv` | SC-vote FRS proxy | `sc_frs_k10`, coverage cols | Non-logit confidence ablation | **Yes** | No |
| `analysis/sc_proxy_robustness_results.csv` | SC vs FRS incremental validity | `metric`, `delta_r2` style | Filter alternative | **Yes** | No |
| `diagnostics/top10_concentration.csv` | Problem contribution to top-10% pool | `fraction_contributing`, `concentration_ratio_top20pct`, `n_top10_traces` | **Coverage** / concentration | **Yes** | No |
| `reasoning_confidence_bins_results/reasoning_sampling_metadata.json` | Pool design | `top_pool_fraction`=0.5, `n_bins`=5, per-pair `n_pooled_traces` | Documents filter | **Yes** | No |
| `sample_count_ablation_results/ablation_rankings.csv` | k-subsample stability | `k_sub`, `frs_acc_mean`, `spearman_vs_k16` | Sampling vs filter interaction | **Yes** | No |

**Verdict (concern 2):** Strong for **accuracy** and **ranking robustness**; weaker for “unfiltered **reasoning** mean equals filtered” — unfiltered RS uses a **different sample** (100 random questions, not full pool).

---

### Concern 3 — Rubric component ablations

| Path | Contains | Schema | Rebuttal use | Reusable? | Recompute? |
|:---|:---|:---|:---:|:---:|:---:|
| `reasoning_confidence_bins_results/judging_checkpoints/judged_*.json` | 4 rubric dims per trace | `judge_scores`: faithfulness, utility, coherence, factuality (1–5) | **LORO-rubric** source | **Yes** | **Re-aggregate** only |
| `analysis_outputs/unfiltered_reasoning/judging_checkpoints/unfiltered_judged_*.json` | Same dims in `judge_raw` | 4 ints | Unfiltered LORO | **Yes** | Re-aggregate |
| `analysis/selection_gain_judge_outputs.csv` | Policy traces | `reasoning_score`; `raw_json` may hold dims | Selection-gain LORO | **Partial** | Parse `raw_json` |
| `diagnostics/dimension_correlations.csv` | Global Spearman across dims | 4×4 matrix | Dimension redundancy | **Yes** | Optional refresh |
| `dimension_correlation_analysis.py` | Script (no new judges) | — | Regenerate matrix | — | CPU seconds |

**Formula (in `topk_judge_eval.reasoning_score_from_judge`):** RS = (sum of 4 scores − 4) / 16 → [0,1]. LORO = drop one dimension, renormalize with divisor 12 instead of 16 (script not checked in — **needs small new analysis script**, zero API).

**Verdict (concern 3):** **Per-dimension scores exist** for 13,500 FRS + 5,400 unfiltered + 5,400 selection-gain judged traces. **No precomputed LORO-FRS table** — lowest-cost gap.

---

### Concern 4 — Selection-gain prediction comparisons

| Path | Contains | Schema | Rebuttal use | Reusable? | Recompute? |
|:---|:---|:---|:---:|:---:|:---:|
| `analysis/selection_gain_pair_level.csv` | Pair means | `mean_selection_gain`, `mean_top_conf_reasoning`, `mean_random_reasoning`, `n_questions`=50 | Primary outcome | **Yes** | No |
| `analysis/selection_gain_question_level.csv` | Per-question | `selection_gain_question`, top vs random RS | Distribution / variance | **Yes** | No |
| `analysis/selection_gain_predictor_results.csv` | Correlations | `predictor`, `pearson_r`, `spearman_rho`, n=54 | FRS ρ≈0.49 vs gain | **Yes** | No |
| `analysis/selection_gain_predictor_panel_merged.csv` | Full panel | predictors + gain | Regression inputs | **Yes** | No |
| `analysis/selection_gain_regression_incremental.csv` | OLS | `m4_bases` R²=0.17, `m5_plus_frs` R²=0.75, ΔR²≈0.58 | Incremental (check outcome def.) | **Yes** | No |
| `analysis/selection_gain_loco_generalization.csv` | LOCO on gain | fold-level R² | OOS gain prediction | **Yes** | No |
| `analysis/selection_gain_judge_outputs.csv` | 5,400 trace rows | `selection_type` ∈ {top_conf, random}, `confidence`, `reasoning_score` | Audit trail | **Yes** | No |
| `analysis/cache/selection_gain_judging/*.json` (5400) | Cached judge responses | `reasoning_score`, `judge_raw` | Re-parse dims | **Yes** | No |
| `analysis/selection_gain_worklist.csv` | Planned calls | full worklist | Repro | **Yes** | No |
| `analysis/selection_gain_run_metadata.json` | Design | 50 q/pair, seed 42, GPT-4o-mini | Methods | **Yes** | No |
| `analysis/selection_gain_followup_report.md` | Interpretation | — | Wording | **Yes** | No |
| `analysis/figures/selection_gain/scatter_gain_vs_frs.png` | Figure | — | Paper figure | **Yes** | No |

**Note:** Macro mean selection gain ≈ **−0.026** (top-conf not uniformly better on judge RS). FRS still correlates with gain (Spearman ≈ 0.40).

**Verdict (concern 4):** **Complete** for the designed experiment (50 questions × 54 pairs). Not a second judge or multi-random-draw per question.

---

### Concern 5 — Cross-benchmark transfer

| Path | Contains | Schema | Rebuttal use | Reusable? | Recompute? |
|:---|:---|:---|:---:|:---:|:---:|
| `analysis/cross_benchmark_generalization/frs_cross_benchmark_results.csv` | LOBO rank corr | `held_out_benchmark`, `train_metric`, `test_target`, `corr_kind`, `r`, `ci_lo_95`, `ci_hi_95`, `p_permutation` | **Transfer matrix** (144 rows) | **Yes** | No |
| `analysis/cross_benchmark_generalization/frs_cross_benchmark_summary.csv` | Aggregated | mean ρ across 6 folds | Headline stats | **Yes** | No |
| `analysis/test_frs_cross_benchmark_generalization.py` | Regenerator | — | Refresh | — | CPU ~seconds |
| `analysis/generalization_results.csv` | LOCO + LOMO ΔR² | `fold_type`, `held_out_group`, `delta_r2_test_add_frs` | Incremental validity OOS | **Yes** | No |
| `analysis/generalization_results_lodo_only.csv` | Benchmark folds only | subset | Same | **Yes** | No |
| `analysis/generalization_results_lomo_only.csv` | Model folds only | subset | Same | **Yes** | No |
| `analysis/heldout_predictor_results.csv` | 8 seeds, question holdout | `y_test_mean_reasoning` | Alternative outcome | **Yes** | No |
| `analysis/heldout_question_outcomes_merged.csv` | Per-fold outcomes | train/test means | Detail | **Yes** | No |
| `analysis/figures/high_impact_followup/loco_*.png` | LOCO plots | — | Figures | **Yes** | No |

**Headline (existing):** Mean LOBO Spearman(train FRS aggregate → held-out FRS) ≈ **0.71** (`frs_cross_benchmark_summary.csv`); FRS vs pass@1 transfer weaker.

**Verdict (concern 5):** **Rebuttal-ready** with small-n caveat (9 models).

---

## 2. Supporting artifacts (bootstrap, ranking, downstream)

| Path | Contains | Concerns | Reusable? |
|:---|:---|:---|:---:|
| `sample_count_ablation_results/ablation_per_bootstrap_detail.csv` | Bootstrap FRS gaps per seed | 2, k-sensitivity | **Yes** |
| `sample_count_ablation_results/ablation_rankings_global_spearman.csv` | Rank stability vs k=16 | 1, 2 | **Yes** |
| `reasoning_confidence_bins_results/reasoning_cumulative_topk.csv` | Cumulative RS + `cumulative_accuracy_population` | 1, 2 | **Yes** |
| `results/*.parquet` (54) | 250 judged traces/pair intersection | downstream | **Yes** |
| `downstream_results/per_split_metrics.csv` | Split-half BoN/random | deployment | **Yes** |
| `analysis/pairwise_winner_results.csv` | Tie-break accuracy | ranking | **Yes** |
| `global_pass1_frs_analysis/pairwise_pass1_frs_gaps.csv` | \|Δpass@1\| vs \|ΔFRS\| | 1 | **Yes** |
| `analysis/pass16_recomputed.csv` | pass@16 from JSONL | panel | **Yes** |
| `analysis/coverage_summary.csv` | Row counts per artifact | audit | **Yes** |
| `analysis/frs_predictor_artifact_inventory.csv` | Prior audit | index | **Yes** |
| `logs/unfiltered_reasoning_baseline_*.log` | 5,400 traces, 1,550 new calls | cost audit | **Yes** |
| `analysis/logs/selection_gain_20260331_094812.log` | Completed 5,400 calls | cost audit | **Yes** |

**Bootstrap CIs:** Present in `reasoning_by_confidence_bin.csv` (`bootstrap_ci_lower/upper`), `frs_by_model_benchmark_threshold.csv`, and cross-benchmark `ci_lo_95`/`ci_hi_95`. **No** dedicated bootstrap CI table for FRS vs unfiltered vs single-trace trio.

---

## 3. Checklist — do we ALREADY have X?

| Item | Status | Where / notes |
|:---|:---:|:---|
| Per-trace reasoning scores | **Yes** | 13,500 FRS + 5,400 unfiltered + 5,400 selection-gain; not all 16×N traces |
| Per-question trace confidence rankings | **Yes (compute)** | From JSONL + `compute_trace_confidence`; not stored as one CSV |
| Unfiltered reasoning aggregates | **Yes** | `per_pair_scores.csv` (100 traces/pair) |
| Random / single-trace RS baselines | **Partial** | Unfiltered = 1 random trace/q; `base_reasoning` = pass@1 metric; trace-0 RS **sparse** in judges |
| Per-dimension rubric scores | **Yes** | All judge checkpoints |
| Held-out **judge** experiments | **No** | Held-out **questions** yes (`heldout_*`); same GPT-4o-mini |
| Ranking separation metrics | **Yes** | `ranking_comparison_vs_frs.csv`, proxy/SC rank comparisons, pairwise winner |
| Cross-benchmark transfer matrices | **Yes** | `frs_cross_benchmark_results.csv` |
| Bootstrap CI outputs | **Partial** | Bin-level + LOBO; not all comparisons |
| Top-K coverage statistics | **Yes** | `topk_ablation_results.csv`, `diagnostics/top10_concentration.csv` |
| Problem-level selection distributions | **Yes** | `selection_gain_question_level.csv` (50 q/pair) |

---

## 4. Raw-data computability (no new generation / no new judges)

| Target | From existing data? | Notes |
|:---|:---:|:---|
| **Single-trace RS** (trace 0) | **Accuracy: yes; RS: partial** | `per_problem_accuracy_pass1_trace0_from_pass16.csv`; RS needs judge on trace 0 (~94% of problems **unjudged** at trace 0) |
| **All-trace RS** (full pool mean) | **No** without judges | Pool sizes 4k–21k traces/pair; only ~250–500 judged |
| **Leave-one-rubric-out FRS** | **Yes** | Re-score from `judge_scores` in checkpoints; re-bin if claiming full FRS pipeline |
| **Top-K coverage analysis** | **Yes** | JSONL + existing `topk_ablation` / `top10_concentration` |
| Per-question confidence ranks | **Yes** | JSONL only, CPU |
| pass@1 / pass@16 | **Yes** | `merged_pass1_frs`, `pass16_recomputed.csv`, JSONL |

---

## 5. Missing analyses (gaps for rebuttal)

| Gap | Priority | Blocker |
|:---|:---:|:---|
| Unified **RS table**: FRS vs unfiltered vs trace-0 RS (same judge, same scale) | High | Trace-0 not fully judged |
| **Leave-one-rubric-out** FRS curves / rank changes | High | Not scripted (data ready) |
| **No-confidence-filter** mean RS over full pool | Medium | Needs judge or extrapolation from biased sample |
| **Second judge** / judge stability | High | No data |
| **All-trace** mean RS | Low | Prohibitively many API calls |
| FRS vs unfiltered vs **base_reasoning** three-way figure | Medium | `base_reasoning` incompatible scale/source |
| Selection-gain with **multiple random draws** / more questions | Medium | Design choice; cache is 1 draw |
| `figures/selection_gain/` committed vs report path | Low | PNG exists under `analysis/figures/selection_gain/` |

---

## 6. Lowest-cost experiments (recommended order)

| # | Experiment | GPU? | Est. runtime | Est. API cost | Addresses |
|:---:|:---|:---:|:---:|:---:|:---|
| 1 | **LORO-rubric FRS** from existing `judge_scores` (4 variants + full) | No | 1–5 min | **$0** | 3 |
| 2 | **Three-way pair table**: merge `frs_pct`, `unfiltered_reasoning_mean`, `pass1_pct`, `high_conf_accuracy_pct` + Spearman/Δrank | No | <1 min | **$0** | 1, 2 |
| 3 | **Trace-0 RS** from judged subset + bootstrap CI; report coverage % | No | 5 min | **$0** (sparse) | 1 |
| 4 | **Confidence-necessity figure**: overlay `topk_ablation` accuracy + unfiltered RS gap vs FRS | No | 5 min | **$0** | 2 |
| 5 | Re-run / export **`dimension_correlation_analysis.py`** + per-bin dimension means | No | 1 min | **$0** | 3 |
| 6 | **Selection-gain** appendix table (already done); add partial correlation controlling pass@1 | No | 1 min | **$0** | 4 |
| 7 | Refresh **`test_frs_cross_benchmark_generalization.py`** PDF/plot for paper | No | <1 min | **$0** | 5 |
| 8 | Judge **trace 0 only** for all problems (54 pairs × ~500–1300 q) | No | hours | **~$500–2,000+** | 1 |
| 9 | **Second judge** on 500–1000 cached traces | No | 1–2 h | **~$50–200** | 3, 4 |
| 10 | Extend selection-gain to **100 q** or 3 random draws | No | 2×–6× | **~$5k–15k** (5400× multiplier) | 4 |

**API cost assumptions:** GPT-4o-mini via Portkey on long CoT; unfiltered run logged **5,400** calls with **1,550** new (resume). Scale linearly with judge calls. Generation (pass@16) **not** required for rows 1–7.

**GPU:** None of the listed analyses require local GPU; all are aggregation or API judging.

---

## 7. Master artifact table (paths → concerns)

| Path | n (typical) | Concerns |
|:---|---:|:---|
| `reasoning_confidence_bins_results/judging_checkpoints/judged_*.json` | 54 × 250 traces | 1, 3 |
| `analysis_outputs/unfiltered_reasoning/judging_checkpoints/unfiltered_judged_*.json` | 54 × 100 traces | 1, 3 |
| `analysis/cache/selection_gain_judging/*.json` | 5400 | 4 |
| `source_pass16_jsonl_by_model*/**/*.jsonl` | 54 files | 1, 2, 5 (compute) |
| `analysis/selection_gain_*.csv` | 54–2700 rows | 4 |
| `analysis/cross_benchmark_generalization/*.csv` | 144 / summary | 5 |
| `analysis/generalization_results*.csv` | 15 folds | 5 |
| `topk_ablation_results/topk_ablation_results.csv` | 324 | 2 |
| `analysis_exports/confidence_proxy_robustness/*` | 54×3 proxies | 2 |
| `analysis_exports/self_consistency_proxy_robustness/*` | 54 | 2 |
| `global_pass1_frs_analysis/*.csv` | 54 | 1, 5 |
| `diagnostics/dimension_correlations.csv` | 4×4 | 3 |
| `diagnostics/top10_concentration.csv` | 54 | 2 |
| `topk_judge_results/*.json` | 27 (3 benchmarks) | 1, 2 |
| `results/*.parquet` | 54 × 250 rows | 1 (partial) |

---

## 8. Logs & reproducibility

| Log | Notes |
|:---|:---|
| `logs/unfiltered_reasoning_baseline_20260331_051742.log` | Full 54-pair unfiltered run |
| `analysis/logs/selection_gain_20260331_094812.log` | Successful 5,400-call completion |
| `analysis/logs/frs_predictor_analysis.log` | Panel merge |
| `analysis/logs/high_impact_followup_*.log` | LOCO/LOMO/held-out |

---

## 9. Recommended rebuttal packaging (no new spend)

1. **Table A:** `merged_pass1_frs_per_benchmark.csv` + `per_pair_scores.csv` (FRS vs unfiltered RS vs pass@1).
2. **Table B:** `selection_gain_predictor_results.csv` + `selection_gain_pair_level.csv`.
3. **Table C:** `frs_cross_benchmark_summary.csv` (LOBO transfer).
4. **Table D:** `proxy_benchmark_results_k10.csv` + `sc_proxy_benchmark_results_k10.csv` (filter necessity / robustness).
5. **Appendix figure:** `analysis/figures/selection_gain/scatter_gain_vs_frs.png`, `loco_LODO-benchmark_test_r2.png`.
6. **New script (cheap):** LORO-rubric + dimension dominance paragraph citing `dimension_correlations.csv`.

---

*End of inventory. For machine-readable row counts see `analysis/coverage_summary.csv` and `analysis/frs_predictor_artifact_inventory.csv`.*
