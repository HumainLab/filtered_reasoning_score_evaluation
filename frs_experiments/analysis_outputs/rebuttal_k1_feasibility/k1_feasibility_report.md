# k=1 (trace_idx=0) baseline feasibility audit

_Generated: 2026-05-22T16:00:54.843341+00:00_

## Critical distinction (read first)

Our pass@16 JSONL stores **16 stochastic traces per question** at temperature 0.7. **`trace_idx=0` is the first of those 16 samples**, not a separate k=1 generation run. A reviewer-requested **true k=1 baseline** would require generating **one** trace per question (typically its own run). Using trace_idx=0 from the k=16 pool is a **conservative proxy**: it answers “what if we only looked at the first sample in a multi-sample run?” but **does not** replicate a dedicated k=1 model call distribution.

We also have **unfiltered RS** = one **random** trace per question (100 judged/pair), which is another approximate k=1 judge baseline, usually **not** trace_idx=0.

## Executive totals

- Expected model×benchmark pairs: **54** (found JSONL: **54**)
- Total questions (trace-0 slots): **42,678**
- Trace-0 correctness available (JSONL): **42,678** slots, accuracy definable for **54** pairs
- FRS judge trace-0 RS available: **725** / 42,678 (**1.7%** global)
- Pairs with ≥50 trace-0 judged: **0**
- Pairs with ≥75 trace-0 judged: **0**
- Pairs with ≥100 trace-0 judged: **0**
- Pairs with 100% trace-0 judged (all questions): **0**

### Per-question availability (JSONL)

- Every inspected row has `score[0]` and `code[0]`: **39** / 54 pairs with full code
- Token probs for trace 0: **54** pairs with complete prob lists

### Judge bias check (trace-0 among FRS judged)

- Of **11250** FRS judged (idx, trace) records, **725** are trace_idx=0 (**6.4%**).
- Among trace-0 judged records, in top bin `0-10`: **138**; other bins: **587** (judge subsample is **not** uniform over questions; trace-0 judged set is a **sparse, bin-enriched** subset).

## Feasible baselines without new API

| Baseline | Feasible? | Coverage | Notes |
|:---|:---:|:---|:---|
| k1_trace0_accuracy_jsonl | yes | 54/54 pairs, 42,678 questions | Full correctness from score[0]; no judge API |
| k1_trace0_rs_frs_judge_cache | partial | 725/42,678 (1.7%) | Only questions that happened to be judged at trace_idx=0 in FRS sample |
| k1_trace0_rs_new_judge | requires_api | 41,953 missing trace-0 judges | See missing_trace0_judge_worklist.csv |
| k1_random_trace_unfiltered | yes | 54 pairs × 100 judged (random trace, usually not trace 0) | Existing unfiltered baseline; approximate k=1 judge story |
| k1_bootstrap_simulated_accuracy | yes | 54 pairs, 200 draws/pair | Random one-of-16 trace; not trace_idx=0 specific |
| frs_top10_reference | yes | paper_frs / merged table 54 pairs | From global_pass1_frs_analysis |

## Preliminary k=1 proxy metrics (trace_idx=0, cached only)

- spearman_model_rank_k1_accuracy_vs_frs: **0.55**
- spearman_model_rank_k1_rs_vs_frs: **0.8167**
- median_abs_delta_frs_at_pass1_tie_2pp: **10.25**
- median_abs_delta_k1_acc_at_pass1_tie_2pp: **12.15**
- median_abs_delta_k1_rs_at_pass1_tie_2pp: **12.92**

## Additional judge calls (planning estimates)

| Scenario | Calls | Est. cost (@ $0.03/call) | Est. wall-clock (@ 2.5s/call) |
| 50 questions / pair × 54 pairs | 2,700 | $81 | 1.9 h |
| 100 questions / pair × 54 pairs | 5,400 | $162 | 3.8 h |
| All missing trace-0 (full N per pair) | 41,953 | $1,259 | 29.1 h |

Worklists: `missing_trace0_judge_worklist.csv` (full), capped variants in report metadata.

## Recommendation

**3. **Full judge-call run needed** for rebuttal-worthy k=1 RS (or rely on accuracy-only k=1 + unfiltered random-trace RS).**

### Rationale

- Global trace-0 judge coverage is <5–10%; pair-level FRS vs k=1 RS comparisons are not representative without substantial judging.


## Files

- `trace0_coverage_by_pair.csv`
- `feasible_k1_baselines.csv`
- `trace0_preliminary_metrics.csv`
- `missing_trace0_judge_worklist.csv`
