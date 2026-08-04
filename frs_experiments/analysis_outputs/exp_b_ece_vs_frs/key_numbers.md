# Experiment B: ECE vs FRS

Generated: 2026-05-23T05:57:12.345135+00:00

## Headline

| Metric | Value | p-value | 95% CI | N |
|---|---:|---:|---|---:|
| Pair-level Pearson r (ECE ↔ FRS) | 0.490 | 0.0002 | [0.268, 0.674] | 54 |
| Pair-level Spearman ρ (ECE ↔ FRS) | 0.585 | 0.0000 | — | 54 |
| Model-level Spearman ρ (macro-avg rankings) | 0.650 | 0.0581 | — | 9 |

## Data quality

- Traces skipped (NaN confidence): **0**
- Confidence values clipped to [0, 1]: **0**
- ECE bins: 10 equal-width on [0, 1]

## Outliers: ECE > 0.3

- DS-R1-1.5B × CommonsenseQA: ECE = 0.3259
- DS-R1-1.5B × GSM8K: ECE = 0.5426
- DS-R1-1.5B × MATH500: ECE = 0.3498
- DS-R1-1.5B × SVAMP: ECE = 0.6374
- DS-R1-7B × AQuA: ECE = 0.4904
- DS-R1-7B × CommonsenseQA: ECE = 0.3653
- DS-R1-7B × GPQA: ECE = 0.3156
- DS-R1-7B × GSM8K: ECE = 0.6085
- DS-R1-7B × MATH500: ECE = 0.3360
- DS-R1-7B × SVAMP: ECE = 0.6639
- Gemma-7B × SVAMP: ECE = 0.3503
- LLaMA-3.1-8B × GSM8K: ECE = 0.4514
- LLaMA-3.1-8B × SVAMP: ECE = 0.5346
- Phi-4 × CommonsenseQA: ECE = 0.3315
- Phi-4 × GSM8K: ECE = 0.4230
- Phi-4 × SVAMP: ECE = 0.3502
- Phi-4-Reas. × AQuA: ECE = 0.4158
- Phi-4-Reas. × CommonsenseQA: ECE = 0.6494
- Phi-4-Reas. × GPQA: ECE = 0.3720
- Phi-4-Reas. × GSM8K: ECE = 0.7666
- Phi-4-Reas. × MATH500: ECE = 0.6424
- Phi-4-Reas. × SVAMP: ECE = 0.7807
- Qwen2.5-7B × CommonsenseQA: ECE = 0.6265
- Qwen2.5-7B × GSM8K: ECE = 0.3984
- Qwen2.5-7B × SVAMP: ECE = 0.4426
- Qwen2.5-Math × CommonsenseQA: ECE = 0.3585
- Qwen2.5-Math × GSM8K: ECE = 0.3868
- Qwen2.5-Math × SVAMP: ECE = 0.4793
- Qwen3-4B × CommonsenseQA: ECE = 0.4647
- Qwen3-4B × GPQA: ECE = 0.3526
- Qwen3-4B × GSM8K: ECE = 0.4775
- Qwen3-4B × SVAMP: ECE = 0.5123

## Interpretation

Pair-level expected calibration error (ECE) and FRS show a positive Pearson correlation of r = 0.490 (p = 0.0002, 95% bootstrap CI [0.268, 0.674], N = 54 model×benchmark pairs). This association is statistically significant at α = 0.05. Model-level macro-averaged rankings yield Spearman ρ = 0.650 (p = 0.0581, N = 9 models).
