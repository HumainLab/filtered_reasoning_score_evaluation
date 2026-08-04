# High-impact follow-up analyses

Generated: 2026-03-31T03:53:52.391526
Log: `analysis/logs/high_impact_followup_20260331_035351.log`

## STEP 1 — Feasibility

See `analysis/high_impact_feasibility_table.csv` for A–H verdicts.

## STEP 2 — Priority (executed: 1–4)


| Rank | Analysis | Why | Feasibility | Circularity | Reviewer impact | Runtime | Action |
|------|----------|-----|---------------|-------------|-----------------|---------|--------|
| 1 | LOCO + LOMO incremental validity | Direct OOS generalization | Full | Moderate | Very high | Seconds | **Run** |
| 2 | Held-out question mean reasoning | Reduces overlap in outcome vs full mean | Full | Moderate | High | ~1–2 min | **Run** |
| 3 | Pairwise winner (multi-tolerance) | Tie-break narrative | Full | Moderate | Medium | Seconds | **Run** |
| 4 | SC proxy vs FRS incremental ΔR² | Robustness to confidence definition | Full | Moderate–high | Medium | Seconds | **Run** |
| — | Selection-gain on held-out traces | Needs multi-trace judge / question | **Not feasible** | — | High | — | Skip |
| — | Second-judge robustness | Single judge in metadata | **Not feasible** | — | High | — | Skip |


## STEP 3–4 — Results summary

### Leave-one-dataset-out (benchmark)

- **Note:** Test R² can be **negative** on small held-out groups (n_test=9 here); folds are **illustrative**, not calibrated forecasts.

- Folds: 6 | Positive ΔR²_test folds: **5/6** | Mean ΔR² (test, add FRS): **0.1296**
- Std: 0.1289
- Detail: `analysis/generalization_results.csv` (filter `fold_type=="LODO-benchmark"`)

### Leave-one-model-out

- Folds: 9 | Positive ΔR²_test folds: **6/9** | Mean ΔR² (test, add FRS): **0.1349**
- Std: 0.5919

### Held-out question mean (y_test)

- Per-seed OLS and correlations: `analysis/heldout_predictor_results.csv`
- Merged long table: `analysis/heldout_question_outcomes_merged.csv`

### Pairwise winner (direction accuracy)

- Aggregated: `analysis/pairwise_winner_results.csv`
- Pair counts: `analysis/pairwise_winner_pair_counts.csv`

### SC proxy (alternative confidence)

- `analysis/selection_gain_results.csv` (despite filename: SC incremental validity, not trace selection gain)

## STEP 5 — Interpretation (conservative)

### STEP 6 — Direct answers

1. **Highest-impact analyses run:** LOCO/LOMO (generalization), held-out question outcomes (reduced overlap in y), pairwise winner at multiple tie tolerances, SC-proxy incremental validity (alternative confidence definition).
2. **Feasible now:** A,C,D,E,F fully; B (trace-level selection gain) not; G not; H exploratory (SNR×fold table in `regime_snr_loco_fold.csv`).
3. **Strongest empirical results for the paper:** (a) LOCO/LOMO **positive mean ΔR²_test** when adding FRS; (b) full-panel ΔR² from prior analysis as reference; (c) SC-proxy `sc_frs_k10` can show **comparable or higher** ΔR² than FRS after the same base predictors—interpret cautiously (different construct, often **higher circularity** vs unfiltered outcome; see `sc_proxy_metadata.json`).
4. **Implication for FRS:** Under OOS group splits, FRS often **adds** test-set explained variance beyond pass@1/pass@16/high-conf/SNR; it is **not** redundant in that sense on this panel. Pairwise tie-break accuracy is **mixed** and often **dominated by pass@1** when many pairs qualify.
5. **Supported more strongly:** Generalization-style incremental validity tables; **unsupported** as a universal tie-breaker vs pass@1.
6. **Weak / unsupported:** Per-question selection gain; second-judge agreement; **high** pairwise n when pass@1 tolerance is tight (see pair counts).
7. **Circularity risk:** **Moderate** throughout (judge-based outcome). Held-out **questions** lower overlap with the aggregate mean but same judge. LOCO/LOMO **do not** remove shared judge family.
8. **Next single best analysis:** Judge **multiple traces per question** (or a second judge on a subset) to enable selection-gain and reduce circularity.

## STEP 7 — Paper-facing recommendations

- **Best result to add to main paper:** LOCO/LOMO `generalization_results.csv` + one sentence with mean ΔR²_test and fold positivity counts.
- **Best for appendix:** Held-out question correlations (`heldout_predictor_results.csv`); pairwise table with tolerances; SC proxy robustness.
- **Result to avoid over-emphasizing:** Pairwise winner when **n** is tiny (e.g. pass@1 tolerance 1 pp) or when predictors tie.
- **Suggested table:** `generalization_results.csv` (fold-level).
- **Suggested figure:** `figures/high_impact_followup/loco_LODO-benchmark_test_r2.png` and `loco_LOMO-model_test_r2.png`.
- **One-paragraph wording (draft):**
  > We assessed incremental validity of FRS beyond pass@1, pass@16, high-confidence accuracy, and SNR using leave-one-benchmark-out and leave-one-model-out OLS, predicting mean unfiltered reasoning on the held-out group's model×benchmark pairs. Adding FRS improved test-set R² in 5/6 benchmark folds (mean ΔR²_test=0.130) and 6/9 model folds (mean ΔR²_test=0.135). Held-out question splits (8 seeds) provide an alternative outcome construction; see `heldout_predictor_results.csv`.
