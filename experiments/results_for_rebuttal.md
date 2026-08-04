# COLM rebuttal — cached-data results

## STEP 0 — Discovery (schemas & paths)

- **confidence_per_trace**: NOT pre-exported as a flat array. Computed from pass@16 JSONL via topk_ablation.compute_trace_confidence (mean of lowest 10% token probs). Published FRS uses bin-sampled judged traces instead.
- **judge_4dim_subscores**: reasoning_confidence_bins_results/judging_checkpoints/judged_*.json → judged_samples[].judge_scores.{faithfulness,coherence,utility,factuality} (int 1–5)
- **frs_pass1_table5**: global_pass1_frs_analysis/merged_pass1_frs_per_benchmark.csv (frs_pct, pass1_pct)
- **trace0_single**: analysis_outputs/trace0_k1_judging/per_pair_scores.csv (mean_reasoning_score_pct)
- **unfiltered**: analysis_outputs/unfiltered_reasoning/per_pair_scores.csv (mean_reasoning_score ×100)

**Models (n=9)**: DS-R1-1.5B, DS-R1-7B, Gemma-7B, LLaMA-3.1-8B, Phi-4, Phi-4-Reas., Qwen2.5-7B, Qwen2.5-Math, Qwen3-4B
**Benchmarks (n=6)**: GSM8K, MATH500, SVAMP, AQuA, GPQA, CSQA
**Pairs**: 9 × 6 = 54

## (1) Reversal denominator (FRS vs single-trace / trace-0)

### Aggregate model-level (macro mean over 6 benchmarks, C(9,2)=36)
| Quantity | Count |
| --- | --- |
| Total pairs | 36 |
| Ties (either metric |Δ|<ε) | 0 |
| Non-tied pairs | 36 |
| Strict flips (opposite sign, neither tied) | 13 |
| Qualifying (both ≥2.0 pp) | 29 |
| Reversals among qualifying | 9 |

### Per-benchmark (C(9,2)=36 each; 216 total)
| Benchmark | Pairs | Ties | Non-tied | Flips | Qual≥2.0pp | Rev@qual |
| --- | --- | --- | --- | --- | --- | --- |
| GSM8K | 36 | 0 | 36 | 12 | 28 | 7 |
| MATH500 | 36 | 0 | 36 | 10 | 32 | 8 |
| SVAMP | 36 | 2 | 34 | 13 | 29 | 9 |
| AQuA | 36 | 1 | 35 | 15 | 31 | 14 |
| GPQA | 36 | 2 | 34 | 11 | 30 | 10 |
| CSQA | 36 | 0 | 36 | 18 | 31 | 14 |

**Totals across 6 benchmarks**: pairs=216, ties=5, flips=79

**Denominator = 181** (pooled qualifying pairs); **reversals = 62**. Denominator 181 = total per-benchmark model pairs (9 choose 2 = 36 per bench) where BOTH |ΔFRS| and |Δtrace-0| ≥ 2.0 pp, pooled over 6 benchmarks. Ties (either metric exactly equal) are excluded from flip counts but included in n_pairs=216 per bench.

**Tie / flip rule**: Strict ordering flip: sign(FRS_a−FRS_b) ≠ sign(trace0_a−trace0_b) with neither diff ≈ 0 (|diff|<1e-9). Paper-style reversal: same sign opposition among pairs with both gaps ≥ 2.0 pp.

**Spearman ρ (54 pair-level trace-0 vs FRS)**: 0.6582 (bootstrap 95% CI [0.456, 0.795])

## (2) LOBO transfer (mean ρ, std across 6 folds, per-fold permutation p)

### FRS → FRS
Mean ρ = **0.7117**, std = 0.1585
| Held-out | ρ | p_perm |
| --- | --- | --- |
| GSM8K | 0.9333 | 0.0008 |
| MATH500 | 0.7167 | 0.0198 |
| SVAMP | 0.6833 | 0.0254 |
| AQuA | 0.8201 | 0.0050 |
| GPQA | 0.6500 | 0.0278 |
| CSQA | 0.4667 | 0.1024 |

### trace-0 → trace-0
Mean ρ = **0.6545**, std = 0.1615
| Held-out | ρ | p_perm |
| --- | --- | --- |
| GSM8K | 0.8000 | 0.0062 |
| MATH500 | 0.8667 | 0.0032 |
| SVAMP | 0.5714 | 0.0530 |
| AQuA | 0.6333 | 0.0336 |
| GPQA | 0.6387 | 0.0410 |
| CSQA | 0.4167 | 0.1360 |

### trace-0 macro → held-out FRS
Mean ρ = **0.3155**, std = 0.2036
| Held-out | ρ | p_perm |
| --- | --- | --- |
| GSM8K | 0.1500 | 0.3610 |
| MATH500 | 0.4333 | 0.1216 |
| SVAMP | 0.0667 | 0.4426 |
| AQuA | 0.3431 | 0.1892 |
| GPQA | 0.6333 | 0.0352 |
| CSQA | 0.2667 | 0.2440 |

### unfiltered → unfiltered
Mean ρ = **0.7083**, std = 0.1109
| Held-out | ρ | p_perm |
| --- | --- | --- |
| GSM8K | 0.7500 | 0.0122 |
| MATH500 | 0.7167 | 0.0172 |
| SVAMP | 0.7667 | 0.0124 |
| AQuA | 0.8500 | 0.0028 |
| GPQA | 0.6333 | 0.0388 |
| CSQA | 0.5333 | 0.0742 |

### unfiltered macro → held-out FRS
Mean ρ = **0.4657**, std = 0.1830
| Held-out | ρ | p_perm |
| --- | --- | --- |
| GSM8K | 0.3167 | 0.2046 |
| MATH500 | 0.6000 | 0.0478 |
| SVAMP | 0.1667 | 0.3388 |
| AQuA | 0.5774 | 0.0502 |
| GPQA | 0.6167 | 0.0428 |
| CSQA | 0.5167 | 0.0776 |

### pass@1 → pass@1
Mean ρ = **0.5489**, std = 0.2536
| Held-out | ρ | p_perm |
| --- | --- | --- |
| GSM8K | 0.7333 | 0.0154 |
| MATH500 | 0.4268 | 0.1280 |
| SVAMP | 0.8333 | 0.0046 |
| AQuA | 0.7167 | 0.0158 |
| GPQA | 0.4167 | 0.1336 |
| CSQA | 0.1667 | 0.3342 |

### pass@1 macro → held-out FRS
Mean ρ = **0.2766**, std = 0.1836
| Held-out | ρ | p_perm |
| --- | --- | --- |
| GSM8K | 0.0500 | 0.4576 |
| MATH500 | 0.2667 | 0.2470 |
| SVAMP | 0.0833 | 0.4176 |
| AQuA | 0.3264 | 0.2078 |
| GPQA | 0.5167 | 0.0794 |
| CSQA | 0.4167 | 0.1334 |

## (3) Amplification at 3 pp and 5 pp

### |Δpass@1| ≤ 3.0 pp
| Metric | n_pairs | ratio | win_frac | bootstrap 95% CI |
| --- | --- | --- | --- | --- |
| frs | 27 | 7.353 | 0.815 | [4.789, 10.783] |
| trace0 | 27 | 5.461 | 0.926 | [3.739, 7.707] |
| unfiltered | 27 | 4.040 | 0.963 | [3.059, 5.515] |

### |Δpass@1| ≤ 5.0 pp
| Metric | n_pairs | ratio | win_frac | bootstrap 95% CI |
| --- | --- | --- | --- | --- |
| frs | 34 | 6.067 | 0.824 | [4.296, 8.315] |
| trace0 | 34 | 4.001 | 0.824 | [2.782, 5.568] |
| unfiltered | 34 | 3.328 | 0.882 | [2.539, 4.365] |

## (4) Leave-one-dimension-out FRS (bin 0–10 judged traces)

### |Δpass@1| ≤ 3.0 pp
| Subset | n_pairs | amp ratio | win_frac |
| --- | --- | --- | --- |
| full | 27 | 7.356 | 0.815 |
| drop-faithfulness | 27 | 7.544 | 0.852 |
| drop-coherence | 27 | 6.589 | 0.889 |
| drop-utility | 27 | 6.533 | 0.778 |
| drop-factuality | 27 | 9.519 | 0.889 |

### |Δpass@1| ≤ 5.0 pp
| Subset | n_pairs | amp ratio | win_frac |
| --- | --- | --- | --- |
| full | 34 | 6.067 | 0.824 |
| drop-faithfulness | 34 | 6.186 | 0.853 |
| drop-coherence | 34 | 5.487 | 0.912 |
| drop-utility | 34 | 5.420 | 0.794 |
| drop-factuality | 34 | 7.874 | 0.912 |

## (5) Inter-dimension redundancy (macro model rankings, 4×4 Spearman)

|  | faithfulness | coherence | utility | factuality |
| --- | --- | --- | --- | --- |
| faithfulness | 1.000 | 0.904 | 0.979 | 0.613 |
| coherence | 0.904 | 1.000 | 0.933 | 0.351 |
| utility | 0.979 | 0.933 | 1.000 | 0.569 |
| factuality | 0.613 | 0.351 | 0.569 | 1.000 |

## (6) LOBO transfer ρ by drop-one-dimension subset (optional)

| Subset | mean ρ | std |
| --- | --- | --- |
| full | 0.7117 | 0.1585 |
| drop-faithfulness | 0.7089 | 0.1434 |
| drop-coherence | 0.6158 | 0.1793 |
| drop-utility | 0.6349 | 0.1932 |
| drop-factuality | 0.7518 | 0.0954 |
| only-faithfulness | 0.6189 | 0.2035 |
| only-coherence | 0.8222 | 0.1639 |
| only-utility | 0.7433 | 0.1513 |
| only-factuality | 0.7345 | 0.2096 |

## Provenance (files read)

- `/Users/manaspathak11/Desktop/Everything/research paper/threshold/global_pass1_frs_analysis/merged_pass1_frs_per_benchmark.csv`
- `/Users/manaspathak11/Desktop/Everything/research paper/threshold/analysis_outputs/trace0_k1_judging/per_pair_scores.csv`
- `/Users/manaspathak11/Desktop/Everything/research paper/threshold/analysis_outputs/unfiltered_reasoning/per_pair_scores.csv`
- `/Users/manaspathak11/Desktop/Everything/research paper/threshold/reasoning_confidence_bins_results/judging_checkpoints/judged_*.json`