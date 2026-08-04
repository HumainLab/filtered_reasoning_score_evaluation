# Selection-gain judging experiment

Generated: 2026-03-31T09:54:04.125336+00:00

Log: `analysis/logs/selection_gain_20260331_094812.log`

## Feasibility (STEP 1)

- Raw generations: `source_pass16_jsonl_by_model*/**/*.jsonl` via `discover_jsonl_groups`.
- Traces per question: up to 16; valid traces = those with non-NaN `compute_trace_confidence` on `chosen_token_probs_per_path.epoch_0`.
- Question id: JSONL field `idx`.
- Trace id: index into `code` / `score` / token prob lists.
- Judge: same `Judge` + rubric as `topk_judge_eval` / unfiltered baseline.

## Design

- questions_per_pair=50, seed=42, exclude_top_from_random=True
- Worklist rows: 5400 (= judge calls attempted).
- Judge calls succeeded (rows with score): 5400
- Cache directory: `analysis/cache/selection_gain_judging`

## Circularity

- **Lower** than summary-only analyses: outcome is **policy contrast** (top-conf vs random) on **fresh** judge calls.
- **Residual coupling:** same judge model as elsewhere; confidence for selection uses the **same** token-prob formula as FRS-related work — FRS may correlate mechanically with selection gain.

## Selection gain summary

- Pairs with gain: 54
- Mean gain (macro over pairs): -0.02597

## Predictor correlations

See `analysis/selection_gain_predictor_results.csv` (import into paper).

## Paper-facing (STEP 12)

- **Best headline claim:** Deployment-style **selection gain** (mean judge score of top-confidence trace minus random trace), reported per model×benchmark and correlated with FRS vs baselines.
- **This experiment does NOT support:** That FRS is optimal for all routing rules; causal production impact; independence from confidence (policy uses confidence).
- **Best figure:** `figures/selection_gain/scatter_gain_vs_frs.png`
- **Best table:** `selection_gain_pair_level.csv` + `selection_gain_predictor_results.csv` + `selection_gain_regression_incremental.csv`
- **Appendix-only:** `selection_gain_judge_outputs.csv`, full worklist, per-trace cache JSONs.
- **Next best experiment:** Second judge on the **same** (question, trace) pairs to quantify judge variance.
