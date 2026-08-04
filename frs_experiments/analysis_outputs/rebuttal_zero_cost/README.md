# Zero-cost rebuttal outputs

Produced by `analysis/run_rebuttal_zero_cost_ablations.py` from cached artifacts only.

## Files

| File | Description |
|:---|:---|
| `unified_metric_comparison.csv` | Main comparison table for Reviewer kp6q |
| `unified_metric_comparison.md` | Markdown table |
| `loro_rubric_*.csv` | Leave-one-rubric-out FRS from cached `judge_scores` |
| `confidence_filtering_necessity_summary.csv` | Top-K spread, median-split gaps, concentration |
| `reviewer_kp6q_key_numbers.md` | Copy-paste headline stats |
| `figures/` | Optional bar charts |

## Rebuttal claims (safe)

1. **Why not one trace?** Unfiltered judging uses one random trace per question (100 q/pair) and yields a **different model ranking** than FRS; pass@1 ties still show **large FRS gaps**.
2. **Why confidence filter?** Top-10% accuracy exceeds full-pool accuracy on average; median-split confidence gaps are positive for most pairs.
3. **All rubric dims?** LORO shows dropping any dimension changes the top-bin score (ρ < 1 vs full 4D).

## Rebuttal claims (avoid)

- Full-benchmark **trace-0 reasoning score** parity with FRS (sparse judge coverage).
- FRS optimizes selection-gain (weak macro correlation / near-zero mean gain).
- Second-judge robustness (not in repo).

## Artifact coverage

- `/Users/manaspathak11/Desktop/Everything/research paper/threshold/global_pass1_frs_analysis/merged_pass1_frs_per_benchmark.csv`: exists=True, rows=54, ok
- `/Users/manaspathak11/Desktop/Everything/research paper/threshold/analysis_outputs/unfiltered_reasoning/per_pair_scores.csv`: exists=True, rows=54, ok
- `/Users/manaspathak11/Desktop/Everything/research paper/threshold/topk_ablation_results/topk_ablation_results.csv`: exists=True, rows=324, ok
- `/Users/manaspathak11/Desktop/Everything/research paper/threshold/analysis/pass16_recomputed.csv`: exists=True, rows=54, ok
- `/Users/manaspathak11/Desktop/Everything/research paper/threshold/global_pass1_frs_analysis/paper_pass1_reasoning_by_benchmark.csv`: exists=True, rows=54, ok
- `/Users/manaspathak11/Desktop/Everything/research paper/threshold/analysis/selection_gain_pair_level.csv`: exists=True, rows=54, ok
- `/Users/manaspathak11/Desktop/Everything/research paper/threshold/reasoning_confidence_bins_results/reasoning_cumulative_topk.csv`: exists=True, rows=270, ok
