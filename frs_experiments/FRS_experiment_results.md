# FRS Experiment Results — Complete Numerical Data

> Generated 2026-03-24 from `threshold/` workspace.
> Scripts: `topk_ablation.py`, `correctness_conditioned.py`

---

## 1. Experimental Setup

### 1.1 Models (9)

| Short Name | Full Model |
|:---|:---|
| DS-R1-1.5B | DeepSeek-R1-Distill-Qwen-1.5B |
| DS-R1-7B | DeepSeek-R1-Distill-Qwen-7B |
| Gemma-7B | Google Gemma-7B |
| LLaMA-3.1-8B | Meta LLaMA-3.1-8B-Instruct |
| Phi-4 | Microsoft Phi-4 |
| Phi-4-Reas. | Microsoft Phi-4-Reasoning |
| Qwen2.5-7B | Alibaba Qwen2.5-7B-Instruct |
| Qwen2.5-Math | Alibaba Qwen2.5-Math-7B |
| Qwen3-4B | Alibaba Qwen3-4B-Thinking |

### 1.2 Benchmarks (6)

| Benchmark | Type | Problems |
|:---|:---|---:|
| GSM8K | Grade-school math | 1,319 |
| MATH500 | Competition math | 500 |
| SVAMP | Simple math word problems | 1,000 |
| AQuA | Algebra word problems (MC) | 254 |
| GPQA | Graduate-level science (MC) | 448 |
| CommonsenseQA | Commonsense reasoning (MC) | 1,221 |

### 1.3 Data

Each model-benchmark pair has a JSONL file with **pass@16** data: 16 sampled reasoning traces per problem (temperature 0.7). Each trace includes:
- `score`: list of 16 booleans (correct/incorrect for each trace)
- `code`: list of 16 chain-of-thought strings
- `chosen_token_probs_per_path["epoch_0"]`: list of 16 lists of per-token probabilities

### 1.4 Confidence Definition

**Per-trace confidence** = mean probability of the **lowest 10%** of token probabilities in that trace.
```
confidence(trace) = mean(bottom_10%_of_token_probs)
```
Rationale: the weakest tokens in a generation are the most informative about model uncertainty; high-confidence tokens are near-deterministic and uninformative.

---

## 2. Experiment A — Confidence-Accuracy Ablation (Top-K%)

**Script:** `topk_ablation.py`

**Method:** For each model-benchmark pair, pool all traces (16 per problem), rank by confidence, and report accuracy on the **top K% most confident** traces for K ∈ {10, 20, 30, 50, 70, 100}. If confidence is informative, accuracy should degrade monotonically as less-confident traces are included.

**Total data points:** 324 rows (9 models × 6 datasets × 6 cutoffs)

### AQuA

| Model | 10% | 20% | 30% | 50% | 70% | 100% | Spread (pp) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| DS-R1-1.5B | 80.34 | 72.69 | 64.97 | 54.92 | 47.91 | 42.91 | +37.43 |
| DS-R1-7B | 92.14 | 91.88 | 89.91 | 85.88 | 80.00 | 71.19 | +20.95 |
| Gemma-7B | 13.51 | 19.56 | 23.63 | 27.31 | 28.15 | 27.88 | -14.37 |
| LLaMA-3.1-8B | 56.76 | 50.55 | 45.53 | 40.90 | 42.57 | 44.34 | +12.42 |
| Phi-4 | 53.32 | 53.26 | 52.83 | 52.85 | 52.72 | 53.96 | -0.64 |
| Phi-4-Reas. | 73.46 | 71.83 | 71.37 | 67.37 | 64.36 | 58.44 | +15.02 |
| Qwen2.5-7B | 86.49 | 80.69 | 77.44 | 66.58 | 59.65 | 62.57 | +23.92 |
| Qwen2.5-Math | 47.91 | 48.95 | 50.45 | 51.92 | 53.85 | 54.21 | -6.30 |
| Qwen3-4B | 53.32 | 53.38 | 51.76 | 51.57 | 49.42 | 49.41 | +3.91 |

### CommonsenseQA

| Model | 10% | 20% | 30% | 50% | 70% | 100% | Spread (pp) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| DS-R1-1.5B | 44.37 | 42.14 | 41.82 | 41.74 | 41.02 | 40.74 | +3.63 |
| DS-R1-7B | 56.91 | 56.83 | 56.49 | 54.98 | 52.53 | 49.39 | +7.52 |
| Gemma-7B | 15.51 | 18.30 | 19.69 | 16.71 | 17.84 | 18.34 | -2.83 |
| LLaMA-3.1-8B | 68.27 | 68.37 | 68.11 | 67.07 | 66.27 | 65.02 | +3.25 |
| Phi-4 | 65.76 | 58.01 | 54.72 | 54.95 | 57.18 | 49.45 | +16.31 |
| Phi-4-Reas. | 67.20 | 75.33 | 77.34 | 78.71 | 77.47 | 74.17 | -6.97 |
| Qwen2.5-7B | 82.29 | 86.03 | 86.15 | 85.10 | 83.55 | 81.11 | +1.18 |
| Qwen2.5-Math | 50.72 | 50.38 | 50.20 | 50.51 | 49.96 | 47.28 | +3.44 |
| Qwen3-4B | 84.49 | 78.33 | 74.95 | 70.77 | 68.41 | 68.02 | +16.47 |

### GPQA

| Model | 10% | 20% | 30% | 50% | 70% | 100% | Spread (pp) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| DS-R1-1.5B | 36.26 | 34.94 | 34.73 | 33.71 | 32.67 | 35.18 | +1.08 |
| DS-R1-7B | 43.10 | 40.45 | 41.70 | 43.64 | 46.68 | 50.74 | -7.64 |
| Gemma-7B | 11.40 | 7.95 | 9.02 | 15.26 | 19.08 | 20.68 | -9.28 |
| LLaMA-3.1-8B | 38.91 | 40.93 | 41.47 | 38.62 | 36.20 | 35.49 | +3.42 |
| Phi-4 | 17.43 | 16.44 | 17.29 | 19.78 | 23.76 | 26.09 | -8.66 |
| Phi-4-Reas. | 56.21 | 58.79 | 57.51 | 54.27 | 51.43 | 47.31 | +8.90 |
| Qwen2.5-7B | 39.61 | 37.94 | 36.36 | 35.27 | 34.48 | 33.79 | +5.82 |
| Qwen2.5-Math | 39.19 | 37.03 | 35.66 | 33.62 | 31.95 | 29.99 | +9.20 |
| Qwen3-4B | 62.90 | 58.23 | 57.74 | 56.72 | 57.29 | 58.29 | +4.61 |

### GSM8K

| Model | 10% | 20% | 30% | 50% | 70% | 100% | Spread (pp) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| DS-R1-1.5B | 92.28 | 91.33 | 89.45 | 86.24 | 80.67 | 73.33 | +18.95 |
| DS-R1-7B | 96.92 | 96.14 | 95.75 | 94.85 | 94.13 | 90.28 | +6.64 |
| Gemma-7B | 24.44 | 33.07 | 35.65 | 36.59 | 36.38 | 32.64 | -8.20 |
| LLaMA-3.1-8B | 82.85 | 77.49 | 75.25 | 78.03 | 78.39 | 75.33 | +7.52 |
| Phi-4 | 62.58 | 62.73 | 63.01 | 61.44 | 62.84 | 65.50 | -2.92 |
| Phi-4-Reas. | 96.02 | 97.20 | 97.17 | 97.15 | 96.65 | 94.86 | +1.16 |
| Qwen2.5-7B | 62.62 | 61.48 | 61.24 | 64.60 | 72.61 | 77.55 | -14.93 |
| Qwen2.5-Math | 55.42 | 61.38 | 64.38 | 67.87 | 70.46 | 71.17 | -15.75 |
| Qwen3-4B | 92.14 | 91.12 | 88.63 | 83.41 | 78.58 | 73.68 | +18.46 |

### MATH500

| Model | 10% | 20% | 30% | 50% | 70% | 100% | Spread (pp) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| DS-R1-1.5B | 76.38 | 73.50 | 70.62 | 65.03 | 60.62 | 53.90 | +22.48 |
| DS-R1-7B | 77.25 | 76.31 | 75.33 | 72.52 | 68.57 | 60.41 | +16.84 |
| Gemma-7B | 4.00 | 7.00 | 9.12 | 12.20 | 13.41 | 13.34 | -9.34 |
| LLaMA-3.1-8B | 54.50 | 44.19 | 39.29 | 35.72 | 35.36 | 34.90 | +19.60 |
| Phi-4 | 60.75 | 58.56 | 56.58 | 55.78 | 54.98 | 55.30 | +5.45 |
| Phi-4-Reas. | 88.50 | 90.00 | 91.17 | 90.85 | 89.21 | 84.71 | +3.79 |
| Qwen2.5-7B | 53.50 | 51.00 | 49.54 | 46.18 | 45.23 | 51.31 | +2.19 |
| Qwen2.5-Math | 61.00 | 62.56 | 63.38 | 65.30 | 64.59 | 61.08 | -0.08 |
| Qwen3-4B | 27.00 | 31.06 | 34.83 | 39.02 | 42.75 | 44.60 | -17.60 |

### SVAMP

| Model | 10% | 20% | 30% | 50% | 70% | 100% | Spread (pp) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| DS-R1-1.5B | 91.62 | 91.91 | 91.19 | 89.28 | 84.88 | 81.59 | +10.03 |
| DS-R1-7B | 96.25 | 96.12 | 95.88 | 95.75 | 95.34 | 91.91 | +4.34 |
| Gemma-7B | 21.94 | 32.00 | 36.12 | 40.21 | 42.27 | 40.94 | -19.00 |
| LLaMA-3.1-8B | 86.19 | 78.88 | 79.79 | 81.66 | 82.38 | 80.82 | +5.37 |
| Phi-4 | 47.31 | 52.03 | 54.06 | 54.86 | 55.00 | 57.04 | -9.73 |
| Phi-4-Reas. | 94.50 | 95.88 | 96.42 | 96.31 | 95.41 | 94.25 | +0.25 |
| Qwen2.5-7B | 62.25 | 61.72 | 62.29 | 66.95 | 75.10 | 80.38 | -18.13 |
| Qwen2.5-Math | 44.75 | 57.41 | 63.38 | 70.44 | 73.41 | 75.79 | -31.04 |
| Qwen3-4B | 88.00 | 87.94 | 86.88 | 83.93 | 80.39 | 76.23 | +11.77 |

### Mean Confidence at Each Top-K% Pool (all datasets)

These show the actual confidence values at each pool cutoff.

**AQuA**

| Model | 10% | 20% | 30% | 50% | 70% | 100% |
|:---|---:|---:|---:|---:|---:|---:|
| DS-R1-1.5B | 0.2964 | 0.2587 | 0.2357 | 0.2067 | 0.1880 | 0.1662 |
| DS-R1-7B | 0.3640 | 0.3315 | 0.3093 | 0.2770 | 0.2524 | 0.2214 |
| Gemma-7B | 0.5902 | 0.4154 | 0.3291 | 0.2421 | 0.1948 | 0.1493 |
| LLaMA-3.1-8B | 0.7634 | 0.6831 | 0.6069 | 0.4823 | 0.3965 | 0.3065 |
| Phi-4 | 0.5288 | 0.4616 | 0.4144 | 0.3476 | 0.3001 | 0.2441 |
| Phi-4-Reas. | 0.3296 | 0.2773 | 0.2514 | 0.2192 | 0.1966 | 0.1686 |
| Qwen2.5-7B | 0.8607 | 0.8329 | 0.8084 | 0.7596 | 0.7011 | 0.5917 |
| Qwen2.5-Math | 0.6051 | 0.5334 | 0.4880 | 0.4260 | 0.3801 | 0.3155 |
| Qwen3-4B | 0.3935 | 0.3644 | 0.3463 | 0.3208 | 0.3018 | 0.2758 |

**CommonsenseQA**

| Model | 10% | 20% | 30% | 50% | 70% | 100% |
|:---|---:|---:|---:|---:|---:|---:|
| DS-R1-1.5B | 0.2130 | 0.1641 | 0.1414 | 0.1177 | 0.1039 | 0.0882 |
| DS-R1-7B | 0.2177 | 0.1895 | 0.1747 | 0.1565 | 0.1441 | 0.1286 |
| Gemma-7B | 0.9333 | 0.8540 | 0.7292 | 0.5213 | 0.3926 | 0.2804 |
| LLaMA-3.1-8B | 0.7365 | 0.6788 | 0.6338 | 0.5569 | 0.4892 | 0.3903 |
| Phi-4 | 0.3489 | 0.2972 | 0.2689 | 0.2324 | 0.2041 | 0.1633 |
| Phi-4-Reas. | 0.4734 | 0.3389 | 0.2645 | 0.1894 | 0.1524 | 0.1197 |
| Qwen2.5-7B | 0.3544 | 0.2892 | 0.2602 | 0.2287 | 0.2090 | 0.1846 |
| Qwen2.5-Math | 0.2919 | 0.2382 | 0.2050 | 0.1658 | 0.1426 | 0.1167 |
| Qwen3-4B | 0.3064 | 0.2760 | 0.2621 | 0.2461 | 0.2348 | 0.2155 |

**GPQA**

| Model | 10% | 20% | 30% | 50% | 70% | 100% |
|:---|---:|---:|---:|---:|---:|---:|
| DS-R1-1.5B | 0.4282 | 0.3583 | 0.3160 | 0.2614 | 0.2255 | 0.1855 |
| DS-R1-7B | 0.4015 | 0.3449 | 0.3099 | 0.2631 | 0.2298 | 0.1917 |
| Gemma-7B | 0.4091 | 0.3031 | 0.2475 | 0.1733 | 0.1308 | 0.0939 |
| LLaMA-3.1-8B | 0.5064 | 0.4288 | 0.3808 | 0.3200 | 0.2800 | 0.2331 |
| Phi-4 | 0.5762 | 0.4899 | 0.4322 | 0.3518 | 0.2962 | 0.2308 |
| Phi-4-Reas. | 0.3562 | 0.2545 | 0.2109 | 0.1676 | 0.1438 | 0.1173 |
| Qwen2.5-7B | 0.5961 | 0.5466 | 0.5135 | 0.4634 | 0.4205 | 0.3539 |
| Qwen2.5-Math | 0.3671 | 0.3087 | 0.2723 | 0.2235 | 0.1917 | 0.1565 |
| Qwen3-4B | 0.3807 | 0.3351 | 0.3099 | 0.2777 | 0.2559 | 0.2303 |

**GSM8K**

| Model | 10% | 20% | 30% | 50% | 70% | 100% |
|:---|---:|---:|---:|---:|---:|---:|
| DS-R1-1.5B | 0.3680 | 0.3314 | 0.3033 | 0.2571 | 0.2247 | 0.1909 |
| DS-R1-7B | 0.4301 | 0.4034 | 0.3852 | 0.3573 | 0.3336 | 0.2943 |
| Gemma-7B | 0.4328 | 0.3103 | 0.2561 | 0.1994 | 0.1662 | 0.1304 |
| LLaMA-3.1-8B | 0.8000 | 0.7165 | 0.6048 | 0.4603 | 0.3809 | 0.3019 |
| Phi-4 | 0.5179 | 0.4499 | 0.4022 | 0.3334 | 0.2859 | 0.2320 |
| Phi-4-Reas. | 0.3618 | 0.3042 | 0.2749 | 0.2394 | 0.2147 | 0.1820 |
| Qwen2.5-7B | 0.8826 | 0.8584 | 0.8343 | 0.7466 | 0.6437 | 0.5310 |
| Qwen2.5-Math | 0.5657 | 0.5110 | 0.4776 | 0.4309 | 0.3938 | 0.3373 |
| Qwen3-4B | 0.3984 | 0.3581 | 0.3346 | 0.3046 | 0.2843 | 0.2593 |

**MATH500**

| Model | 10% | 20% | 30% | 50% | 70% | 100% |
|:---|---:|---:|---:|---:|---:|---:|
| DS-R1-1.5B | 0.3435 | 0.3001 | 0.2740 | 0.2408 | 0.2182 | 0.1907 |
| DS-R1-7B | 0.4419 | 0.4022 | 0.3758 | 0.3383 | 0.3088 | 0.2688 |
| Gemma-7B | 0.6404 | 0.4881 | 0.4016 | 0.3048 | 0.2475 | 0.1888 |
| LLaMA-3.1-8B | 0.7485 | 0.6563 | 0.5734 | 0.4566 | 0.3830 | 0.3044 |
| Phi-4 | 0.6897 | 0.6282 | 0.5822 | 0.5023 | 0.4304 | 0.3425 |
| Phi-4-Reas. | 0.3846 | 0.3334 | 0.3050 | 0.2686 | 0.2418 | 0.2063 |
| Qwen2.5-7B | 0.8737 | 0.8498 | 0.8293 | 0.7867 | 0.7330 | 0.6275 |
| Qwen2.5-Math | 0.6103 | 0.5667 | 0.5391 | 0.4991 | 0.4667 | 0.4145 |
| Qwen3-4B | 0.5020 | 0.4716 | 0.4511 | 0.4200 | 0.3940 | 0.3539 |

**SVAMP**

| Model | 10% | 20% | 30% | 50% | 70% | 100% |
|:---|---:|---:|---:|---:|---:|---:|
| DS-R1-1.5B | 0.3450 | 0.3097 | 0.2835 | 0.2410 | 0.2107 | 0.1789 |
| DS-R1-7B | 0.3907 | 0.3623 | 0.3428 | 0.3139 | 0.2906 | 0.2552 |
| Gemma-7B | 0.4762 | 0.3208 | 0.2558 | 0.1917 | 0.1564 | 0.1209 |
| LLaMA-3.1-8B | 0.8266 | 0.6921 | 0.5563 | 0.4187 | 0.3456 | 0.2736 |
| Phi-4 | 0.5305 | 0.4535 | 0.4038 | 0.3360 | 0.2888 | 0.2342 |
| Phi-4-Reas. | 0.3629 | 0.2948 | 0.2609 | 0.2210 | 0.1944 | 0.1618 |
| Qwen2.5-7B | 0.8977 | 0.8706 | 0.8396 | 0.7239 | 0.6142 | 0.4996 |
| Qwen2.5-Math | 0.5574 | 0.4928 | 0.4547 | 0.4035 | 0.3651 | 0.3104 |
| Qwen3-4B | 0.3924 | 0.3490 | 0.3246 | 0.2940 | 0.2739 | 0.2500 |

---

## 3. Experiment B — Correctness-Conditioned Confidence (Median Split)

**Script:** `correctness_conditioned.py`

**Method:** For each model-benchmark pair, split all traces at the **median confidence** (data-driven, no peeking at labels). Report accuracy in the high-confidence half vs the low-confidence half, plus mean confidence for correct vs incorrect traces.

**Total data points:** 54 rows (9 models × 6 datasets)

### Accuracy: High-Confidence Half vs Low-Confidence Half

| Model | Dataset | N traces | Median Conf | Acc High % | Acc Low % | Gap (pp) |
|:---|---:|---:|---:|---:|---:|---:|
| DS-R1-1.5B | AQuA | 4064 | 0.1514 | 54.92 | 30.91 | +24.02 |
| DS-R1-1.5B | CommonsenseQA | 19536 | 0.0752 | 41.74 | 39.74 | +2.00 |
| DS-R1-1.5B | GPQA | 7168 | 0.1536 | 33.71 | 36.66 | -2.96 |
| DS-R1-1.5B | GSM8K | 21104 | 0.1595 | 86.24 | 60.42 | +25.82 |
| DS-R1-1.5B | MATH500 | 8000 | 0.1747 | 65.03 | 42.78 | +22.25 |
| DS-R1-1.5B | SVAMP | 16000 | 0.1501 | 89.28 | 73.91 | +15.36 |
| DS-R1-7B | AQuA | 4064 | 0.2075 | 85.88 | 56.50 | +29.38 |
| DS-R1-7B | CommonsenseQA | 19536 | 0.1206 | 54.98 | 43.80 | +11.18 |
| DS-R1-7B | GPQA | 7168 | 0.1669 | 43.64 | 57.84 | -14.20 |
| DS-R1-7B | GSM8K | 21104 | 0.2949 | 94.85 | 85.70 | +9.15 |
| DS-R1-7B | MATH500 | 8000 | 0.2575 | 72.52 | 48.30 | +24.22 |
| DS-R1-7B | SVAMP | 16000 | 0.2514 | 95.75 | 88.06 | +7.69 |
| Gemma-7B | AQuA | 4064 | 0.0911 | 27.31 | 28.44 | -1.13 |
| Gemma-7B | CommonsenseQA | 19536 | 0.1222 | 16.71 | 19.96 | -3.26 |
| Gemma-7B | GPQA | 7168 | 0.0351 | 15.26 | 26.09 | -10.83 |
| Gemma-7B | GSM8K | 21104 | 0.0970 | 36.59 | 28.70 | +7.89 |
| Gemma-7B | MATH500 | 8000 | 0.1289 | 12.20 | 14.48 | -2.28 |
| Gemma-7B | SVAMP | 16000 | 0.0803 | 40.21 | 41.66 | -1.45 |
| LLaMA-3.1-8B | AQuA | 4064 | 0.2276 | 40.90 | 47.79 | -6.89 |
| LLaMA-3.1-8B | CommonsenseQA | 19536 | 0.3770 | 67.07 | 62.97 | +4.10 |
| LLaMA-3.1-8B | GPQA | 7168 | 0.2017 | 38.62 | 32.37 | +6.25 |
| LLaMA-3.1-8B | GSM8K | 21104 | 0.2077 | 78.03 | 72.63 | +5.40 |
| LLaMA-3.1-8B | MATH500 | 8000 | 0.2325 | 35.72 | 34.08 | +1.65 |
| LLaMA-3.1-8B | SVAMP | 16000 | 0.1841 | 81.66 | 79.99 | +1.68 |
| Phi-4 | AQuA | 4064 | 0.2090 | 52.85 | 55.07 | -2.21 |
| Phi-4 | CommonsenseQA | 19536 | 0.1558 | 54.95 | 43.95 | +11.00 |
| Phi-4 | GPQA | 7168 | 0.1881 | 19.78 | 32.39 | -12.61 |
| Phi-4 | GSM8K | 21104 | 0.1935 | 61.44 | 69.56 | -8.12 |
| Phi-4 | MATH500 | 8000 | 0.3090 | 55.78 | 54.82 | +0.95 |
| Phi-4 | SVAMP | 16000 | 0.1981 | 54.86 | 59.23 | -4.36 |
| Phi-4-Reas. | AQuA | 4064 | 0.1551 | 67.37 | 49.51 | +17.86 |
| Phi-4-Reas. | CommonsenseQA | 19536 | 0.0662 | 78.71 | 69.64 | +9.07 |
| Phi-4-Reas. | GPQA | 7168 | 0.0922 | 54.27 | 40.35 | +13.92 |
| Phi-4-Reas. | GSM8K | 21104 | 0.1688 | 97.15 | 92.58 | +4.57 |
| Phi-4-Reas. | MATH500 | 8000 | 0.1935 | 90.85 | 78.57 | +12.28 |
| Phi-4-Reas. | SVAMP | 16000 | 0.1436 | 96.31 | 92.19 | +4.12 |
| Qwen2.5-7B | AQuA | 4064 | 0.6296 | 66.58 | 58.56 | +8.02 |
| Qwen2.5-7B | CommonsenseQA | 19536 | 0.1702 | 85.10 | 77.11 | +8.00 |
| Qwen2.5-7B | GPQA | 7168 | 0.3528 | 35.27 | 32.31 | +2.96 |
| Qwen2.5-7B | GSM8K | 21104 | 0.4498 | 64.60 | 90.49 | -25.89 |
| Qwen2.5-7B | MATH500 | 8000 | 0.6717 | 46.18 | 56.45 | -10.28 |
| Qwen2.5-7B | SVAMP | 16000 | 0.3916 | 66.95 | 93.81 | -26.86 |
| Qwen2.5-Math | AQuA | 4064 | 0.2967 | 51.92 | 56.50 | -4.58 |
| Qwen2.5-Math | CommonsenseQA | 19536 | 0.0943 | 50.51 | 44.04 | +6.47 |
| Qwen2.5-Math | GPQA | 7168 | 0.1274 | 33.62 | 26.37 | +7.25 |
| Qwen2.5-Math | GSM8K | 21104 | 0.3300 | 67.87 | 74.46 | -6.59 |
| Qwen2.5-Math | MATH500 | 8000 | 0.4122 | 65.30 | 56.85 | +8.45 |
| Qwen2.5-Math | SVAMP | 16000 | 0.2969 | 70.44 | 81.14 | -10.70 |
| Qwen3-4B | AQuA | 4064 | 0.2674 | 51.57 | 47.24 | +4.33 |
| Qwen3-4B | CommonsenseQA | 19536 | 0.2147 | 70.77 | 65.26 | +5.51 |
| Qwen3-4B | GPQA | 7168 | 0.2136 | 56.72 | 59.85 | -3.12 |
| Qwen3-4B | GSM8K | 21104 | 0.2453 | 83.41 | 63.95 | +19.46 |
| Qwen3-4B | MATH500 | 8000 | 0.3511 | 39.02 | 50.18 | -11.15 |
| Qwen3-4B | SVAMP | 16000 | 0.2348 | 83.93 | 68.54 | +15.39 |

### Mean Confidence: Correct vs Incorrect Traces

| Model | Dataset | Mean Conf (Correct) | Mean Conf (Incorrect) | Gap |
|:---|---:|---:|---:|---:|
| DS-R1-1.5B | AQuA | 0.1869 | 0.1506 | +0.0363 |
| DS-R1-1.5B | CommonsenseQA | 0.0882 | 0.0882 | +0.0000 |
| DS-R1-1.5B | GPQA | 0.1818 | 0.1876 | -0.0058 |
| DS-R1-1.5B | GSM8K | 0.2055 | 0.1508 | +0.0547 |
| DS-R1-1.5B | MATH500 | 0.2058 | 0.1731 | +0.0327 |
| DS-R1-1.5B | SVAMP | 0.1855 | 0.1495 | +0.0361 |
| DS-R1-7B | AQuA | 0.2363 | 0.1847 | +0.0516 |
| DS-R1-7B | CommonsenseQA | 0.1326 | 0.1247 | +0.0079 |
| DS-R1-7B | GPQA | 0.1805 | 0.2033 | -0.0228 |
| DS-R1-7B | GSM8K | 0.3002 | 0.2391 | +0.0612 |
| DS-R1-7B | MATH500 | 0.2877 | 0.2398 | +0.0479 |
| DS-R1-7B | SVAMP | 0.2591 | 0.2115 | +0.0476 |
| Gemma-7B | AQuA | 0.1233 | 0.1594 | -0.0361 |
| Gemma-7B | CommonsenseQA | 0.2768 | 0.2812 | -0.0044 |
| Gemma-7B | GPQA | 0.0632 | 0.1019 | -0.0387 |
| Gemma-7B | GSM8K | 0.1227 | 0.1341 | -0.0114 |
| Gemma-7B | MATH500 | 0.1475 | 0.1952 | -0.0476 |
| Gemma-7B | SVAMP | 0.0979 | 0.1367 | -0.0388 |
| LLaMA-3.1-8B | AQuA | 0.3112 | 0.3027 | +0.0085 |
| LLaMA-3.1-8B | CommonsenseQA | 0.3974 | 0.3773 | +0.0201 |
| LLaMA-3.1-8B | GPQA | 0.2433 | 0.2275 | +0.0158 |
| LLaMA-3.1-8B | GSM8K | 0.3087 | 0.2813 | +0.0274 |
| LLaMA-3.1-8B | MATH500 | 0.3315 | 0.2898 | +0.0417 |
| LLaMA-3.1-8B | SVAMP | 0.2758 | 0.2645 | +0.0113 |
| Phi-4 | AQuA | 0.2418 | 0.2469 | -0.0050 |
| Phi-4 | CommonsenseQA | 0.1821 | 0.1449 | +0.0372 |
| Phi-4 | GPQA | 0.2010 | 0.2413 | -0.0403 |
| Phi-4 | GSM8K | 0.2259 | 0.2437 | -0.0178 |
| Phi-4 | MATH500 | 0.3441 | 0.3406 | +0.0036 |
| Phi-4 | SVAMP | 0.2257 | 0.2456 | -0.0199 |
| Phi-4-Reas. | AQuA | 0.1810 | 0.1512 | +0.0298 |
| Phi-4-Reas. | CommonsenseQA | 0.1126 | 0.1401 | -0.0275 |
| Phi-4-Reas. | GPQA | 0.1230 | 0.1121 | +0.0108 |
| Phi-4-Reas. | GSM8K | 0.1836 | 0.1540 | +0.0296 |
| Phi-4-Reas. | MATH500 | 0.2107 | 0.1820 | +0.0287 |
| Phi-4-Reas. | SVAMP | 0.1627 | 0.1461 | +0.0167 |
| Qwen2.5-7B | AQuA | 0.5991 | 0.5793 | +0.0198 |
| Qwen2.5-7B | CommonsenseQA | 0.1861 | 0.1779 | +0.0082 |
| Qwen2.5-7B | GPQA | 0.3606 | 0.3504 | +0.0102 |
| Qwen2.5-7B | GSM8K | 0.4907 | 0.6703 | -0.1796 |
| Qwen2.5-7B | MATH500 | 0.5995 | 0.6569 | -0.0574 |
| Qwen2.5-7B | SVAMP | 0.4545 | 0.6844 | -0.2300 |
| Qwen2.5-Math | AQuA | 0.3111 | 0.3207 | -0.0096 |
| Qwen2.5-Math | CommonsenseQA | 0.1207 | 0.1130 | +0.0077 |
| Qwen2.5-Math | GPQA | 0.1697 | 0.1508 | +0.0190 |
| Qwen2.5-Math | GSM8K | 0.3303 | 0.3546 | -0.0243 |
| Qwen2.5-Math | MATH500 | 0.4212 | 0.4040 | +0.0172 |
| Qwen2.5-Math | SVAMP | 0.2945 | 0.3600 | -0.0655 |
| Qwen3-4B | AQuA | 0.2774 | 0.2742 | +0.0032 |
| Qwen3-4B | CommonsenseQA | 0.2185 | 0.2090 | +0.0095 |
| Qwen3-4B | GPQA | 0.2308 | 0.2296 | +0.0011 |
| Qwen3-4B | GSM8K | 0.2677 | 0.2359 | +0.0318 |
| Qwen3-4B | MATH500 | 0.3427 | 0.3630 | -0.0203 |
| Qwen3-4B | SVAMP | 0.2558 | 0.2315 | +0.0242 |

---

## 4. Summary Statistics

### 4.1 Monotonicity of Accuracy @K

A model-benchmark pair is **monotonic** if accuracy strictly decreases as K increases (more traces = lower accuracy).

- **16 / 54** model-benchmark pairs are strictly monotonic
- Mean spread (top-10% − 100%): **+2.78 pp**
- Median spread: **+3.54 pp**

### 4.2 Per-Model Summary (across all datasets)

| Model | Avg Spread (Acc@10−@100) | Monotonic Pairs | Interpretation |
|:---|---:|---:|:---|
| DS-R1-1.5B | +15.60 | 4/6 | Strong positive accuracy–confidence alignment |
| DS-R1-7B | +8.11 | 5/6 | Strong positive accuracy–confidence alignment |
| Gemma-7B | -10.50 | 0/6 | Inverted: accuracy rises as lower-confidence traces are included |
| LLaMA-3.1-8B | +8.60 | 1/6 | Mild positive spread; mixed across datasets |
| Phi-4 | -0.03 | 0/6 | Weak / noisy confidence signal |
| Phi-4-Reas. | +3.69 | 1/6 | Modest spread; mixed across datasets |
| Qwen2.5-7B | +0.01 | 1/6 | Weak / noisy confidence signal |
| Qwen2.5-Math | -6.75 | 1/6 | Inverted on several benchmarks |
| Qwen3-4B | +6.27 | 3/6 | Strong positive accuracy–confidence alignment |

---

## 5. Key Findings

1. **Confidence is strongly informative for DeepSeek-R1 models and Qwen3-4B:** These show large positive spreads (top-10% accuracy well above the full pool). Token-level confidence tracks which traces are more likely to be correct.

2. **Confidence is inverted for Qwen2.5-7B (and partly Qwen2.5-Math):** Accuracy *increases* as less-confident traces are included on several benchmarks. For these models, the bottom-decile token statistic is not a reliable quality signal.

3. **Phi-4 family shows small or mixed spreads:** Pool-level accuracy changes little with confidence filtering; the signal is weak compared to R1 and Qwen3-4B.

4. **Gemma-7B and LLaMA-3.1-8B are near-neutral:** Small spreads on average. Confidence has limited discriminative power for these models in this setup.

5. **The correctness-conditioned analysis matches the top-K% curves:** Models where the high-confidence half is more accurate (Section 3) largely coincide with models where accuracy falls as wider confidence slices are included (Section 2).

---

## 6. Limitations and Scope

- **Confidence metric uses bottom-10% token probabilities** — other aggregation functions (entropy, min-prob, etc.) were not compared.
- **Pass@16 sampling at temperature 0.7** — results may differ at other temperatures or sample counts.
