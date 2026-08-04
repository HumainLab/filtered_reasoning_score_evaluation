# Coverage-aware FRS (reviewer-facing analysis)

This folder can hold outputs from `analysis/generate_frs_coverage_analysis.py`.

## What it does

1. **Threshold sweep**  
   For each model, pool all samples (across approved datasets) that have **both**:
   - `answer_confidence` in `filtered-cot-*/<stem>_filtered_p1_only.jsonl`
   - `fused_scores.overall` in `filtered-cot-*/results/<stem>_filtered_p1_only_results.json`  

   For each confidence threshold `τ ∈ [0, 1]`, keep samples with `conf ≥ τ`, then:
   - **Coverage** = fraction of the *pooled judged set* retained (same as “fraction of dataset with conf ≥ τ” when judge coverage is complete).
   - **FRS** = mean fused overall × 100 on the retained set.
   - **Accuracy** = mean correctness on the retained set (uses `evidence.final_correct` when present).

2. **Equal-coverage table**  
   Sort the pooled set by confidence **descending** and take the top `⌊c·N⌋` points for  
   `c ∈ {0.40, 0.60, 0.80}`.  
   This matches models at **the same effective coverage** without letting a model “win” by answering fewer, easier items.

## Outputs

| File | Purpose |
|------|--------|
| `frs_coverage_curves_pooled.png` | FRS vs coverage + accuracy vs coverage (one line per model, pooled across datasets). |
| `frs_coverage_curves_long.csv` | Long-form curve data (`tau`, `coverage`, `frs_percent`, `accuracy_percent`). |
| `frs_at_fixed_coverage.csv` | FRS and accuracy at ~40% / 60% / 80% coverage (sorted-confidence / top-fraction method). |
| `frs_coverage_metadata.json` | Per-model pool size and merge warnings. |

## Regenerate

```bash
cd reasoning-models-eval
python3 analysis/generate_frs_coverage_analysis.py
```

## Caveats (important for the paper)

- **Pooled curve** aggregates GSM8K, MATH500, SVAMP, AQuA, GPQA, CommonSense when judge JSONs exist. Macro-average **per dataset** is easy to add if you want dataset-fair weighting.
- Some **results JSONs are incomplete** vs the JSONL (e.g. partial `math500` runs). Those splits contribute only the intersecting indices; `frs_coverage_metadata.json` lists warnings. For publication, complete the missing `*_filtered_p1_only_results.json` runs so coverage is not silently biased.
- This analysis uses **existing** fused judge scores; it does **not** re-run the judge at each τ. It only **filters** which judged samples are included at each confidence level.

## Suggested paper sentence

> We report FRS and accuracy as a function of confidence-threshold coverage (fraction of samples with model confidence above τ) and at fixed coverage levels (top-fraction by confidence), so improvements cannot be explained solely by selective answering.
