# Unified metric comparison (zero-cost rebuttal)

Discrimination at similar accuracy: among model pairs with |Δpass@1| ≤ 2 pp, median |Δmetric| (higher = metric still separates models when accuracy is tied). Low Spearman(|Δpass@1|, |Δmetric|) also indicates separation beyond accuracy.

| Metric | n pairs | Coverage | Tie median |Δ| (pp) | Rank ρ vs FRS | Sel. gain r | LOBO ρ | Incomplete? |
|:---|---:|---:|---:|---:|---:|---:|:---|
| pass@1 accuracy | 54 | 100% | 1.0 | 0.15 | -0.1281 | 0.5489 | no |
| pass@16 accuracy | 54 | 100% | 0.45 | -0.15 | -0.0824 | 0.2385 | no |
| Top-10% pool accuracy (confidence filter) | 54 | 100% | 18.68 | 0.5333 | 0.2009 | 0.6505 | no |
| Unfiltered mean reasoning score (100× judge 0–1) | 54 | 100% | 5.38 | 0.45 | 0.0078 | 0.7083 | no |
| FRS (paper top-confidence bin, %) | 54 | 100% | 9.9 | 1.0 | 0.4906 | 0.7117 | no |
| Cumulative top-10% bin mean RS (filtered) | 54 | 100% | 9.88 | 1.0 | 0.4907 | 0.7117 | no |
| base_reasoning (pass@1-era eval API) | 54 | 100% | 5.25 | 0.3833 | 0.0583 | 0.5972 | yes |
| Trace-0 mean RS (judged subset only) | 54 | 100% | 11.91 | 0.6833 | 0.1757 | 0.7667 | yes |
| Trace-0 accuracy (all problems, pass16 JSONL) | 45 | 83% | 12.15 | 0.4333 | -0.0672 | 0.6173 | no |
