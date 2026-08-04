# Master experiment results — FRS / confidence / accuracy

> **Generated:** 2026-03-25
> **Workspace:** `threshold/` (paths below are relative to this folder unless noted)

This document indexes **accuracy** ablations, **median‑split** analyses, **confidence‑bin** population accuracy (disjoint and cumulative), sample‑count ablations, and judge‑free top‑K% trace accuracy.  
**Part II** reproduces the numerical tables from `FRS_experiment_results.md`.  
**Part III** embeds CSV data in **wide tables** where noted (regenerate after re-running experiments).

---

## Quick index — where everything lives

| Topic | Script(s) | Main outputs |
|:---|:---|:---|
| **Top‑K% accuracy** (pooled traces, confidence filter) | `topk_ablation.py` | `topk_ablation_results/topk_ablation_results.csv`, `topk_ablation_*.png`, `topk_ablation_all_datasets_accuracy.png` |
| **Median split** (high vs low confidence half) | `correctness_conditioned.py` | `correctness_conditioned_results/correctness_conditioned.csv`, plots |
| **Population accuracy by confidence bins** | `reasoning_confidence_bins.py` | `reasoning_by_confidence_bin.csv`, `reasoning_cumulative_topk.csv`, PNGs |
| **Judge‑free top‑K% trace accuracy** | `reasoning_confidence_bins.py topk-accuracy` | `reasoning_confidence_bins_results/topk_trace_accuracy.csv` (if generated) |
| **Sample‑count ablation** (k = 4, 8, 12, 16 traces) | `sample_count_ablation.py` | `sample_count_ablation_results/ablation_*.csv`, `checkpoints/chunks/`, PDFs |

---

## Shared experimental setup

### Models & benchmarks

- **9 models** and **6 benchmarks** (AQuA, CommonsenseQA, GPQA, GSM8K, MATH500, SVAMP) — see Part II §1.1–1.2 for the full table.
- **Data:** pass@**16** JSONL per problem: 16 stochastic traces (temperature **0.7**), each with `score[]`, CoT `code[]`, and token probabilities for confidence.

### Confidence definition (used almost everywhere)

**Per‑trace confidence** = mean probability of the **lowest 10%** of token probabilities in that trace (mean over the bottom decile of token probs).  
Rationale: weakest tokens carry more information about uncertainty than near‑deterministic tokens.

---

## 1. Top‑K% accuracy ablation (`topk_ablation.py`)

**Question:** If we pool all traces, rank by confidence, and take **only the top K%** of traces (by a **confidence threshold** on the pooled distribution), how does **accuracy** change?

**Cutoffs:** `K ∈ {10, 20, 30, 50, 70, 100}` (see `PERCENTILES` in `topk_ablation.py`).

**Outputs:** `topk_ablation_results/topk_ablation_results.csv` (accuracy %, n traces, mean confidence per row).  
**Figures:** per‑dataset line charts, `topk_ablation_spread_heatmap.png`, combined grid `topk_ablation_all_datasets_accuracy.png`.

**Note:** The implementation uses a **percentile threshold** on the pooled `confidence` column (not always exactly `⌊n·K/100⌋` traces). For **count‑based** top‑K% aligned with the reasoning‑bins pipeline, use `python reasoning_confidence_bins.py topk-accuracy` (see §4).

---

## 2. Correctness‑conditioned / median split (`correctness_conditioned.py`)

**Question:** Split traces at the **median confidence** (per model×dataset). Is accuracy higher in the **high‑confidence half** than the **low‑confidence half**?

**Outputs:** `correctness_conditioned_results/correctness_conditioned.csv`, plots such as `median_split_gap_by_dataset.png`.

**Full tables:** Part II §3 and Part III (CSV dump).

---

## 3. Population accuracy by confidence bins (`reasoning_confidence_bins.py`)

**Question:** After ranking traces by confidence, split the pool into **equal‑count percentile bins** and report **mean terminal correctness** over **all traces in each bin** (`mean_accuracy_population`) and over **cumulative top‑K% slices** (`cumulative_accuracy_population` in `reasoning_cumulative_topk.csv`).

**Wide tables (Experiment A–style columns):** Part III formats **disjoint** bin accuracy and **cumulative** top‑K% accuracy with columns **10% | 20% | …** aligned with the accuracy ablation where the run supports it. Disjoint bands in the current CSV cover the **top 50%** of the pool in five steps; use `--n-bins 10` and `--top-pool-frac 1.0` to obtain ten disjoint deciles through **100%**.

**Key CSVs:**

| File | Contents |
|:---|:---|
| `reasoning_by_confidence_bin.csv` | Per bin: `mean_accuracy_population` (plus optional judge metadata from the same run) |
| `reasoning_cumulative_topk.csv` | Cumulative: `cumulative_accuracy_population` at each `topk_pct` |
| `reasoning_sampling_metadata.json` | `top_pool_fraction`, `n_bins`, bin definitions |

**Plots:** `reasoning_bins_*.png`, `reasoning_cumulative_*.png`.

**Methods detail:** `reasoning_confidence_bins_methods.md`.

---

## 4. Judge‑free top‑K% trace accuracy (`topk-accuracy` subcommand)

**Command:**

```bash
python reasoning_confidence_bins.py topk-accuracy --data-root . --output-dir ./reasoning_confidence_bins_results \
  --top-pool-frac 0.5 --k-pcts 10,20,30,40,50
```

**What it does:** After optional **pool slice** (`--top-pool-frac`), sort remaining traces by confidence, take the **first ⌊n·K/100⌋** traces for each K, report **mean(correct)** — no API calls.

**Output:** `reasoning_confidence_bins_results/topk_trace_accuracy.csv` (embedded in Part III when present).

---

## 5. Sample‑count ablation — why k ∈ {4, 8, 12, 16}? (`sample_count_ablation.py`)

### Motivation

- The production dataset is **pass@16**: up to **16** traces per problem.
- Many analyses (median split, FRS‑style “gap”) need **enough** traces to split into high/low confidence halves **with** variance across bootstrap seeds.
- **k = 16** is the **full** sample: **deterministic** median split (no subsampling noise for that setting) — used as the **reference** when comparing Spearman correlation of rankings vs k=16.
- **k ∈ {4, 8, 12}** are **strict subsets**: for each problem we **subsample k traces without replacement** (with multiple RNG seeds), compute median split on that subset, and measure **FRS‑style** metrics (high‑confidence minus low‑confidence accuracy, etc.).

### What the script reports

- Per `(model, dataset, k_sub)`: mean/std of the FRS metric across bootstrap seeds.
- **Spearman vs k=16:** how much **ranking of models** (by dataset) agrees with the full k=16 reference when only k traces are available.

### Numerical outputs (all embedded in Part III)

| File | Description |
|:---|:---|
| `ablation_frs_variance_table.csv` | Mean bootstrap std of FRS (across models) by dataset × `k_sub` |
| `ablation_rankings.csv` | Per model×dataset×`k_sub`: mean/std FRS, Spearman vs k=16 |
| `ablation_rankings_global_spearman.csv` | Global Spearman vs k=16 (single row per `k_sub`) |
| `ablation_regime_separation.csv` | Per `k_sub`, model, dataset: mean FRS gap vs reference gap at k=16, sign agreement |
| `ablation_per_bootstrap_detail.csv` | **Full log:** one row per bootstrap × model × dataset × `k_sub` (`frs_accuracy_proxy_pct`, `acc_unconfident_half_pct`, `gap_pp`) |

The shard files `checkpoints/chunks/<Model>__<Dataset>.csv` are **splits** of the same per‑bootstrap rows as `ablation_per_bootstrap_detail.csv` (one shard per model×dataset); Part III embeds **only** the consolidated file so nothing is duplicated.

### Figures

- `ablation_ranking_stability.pdf`, `ablation_regime_heatmap.pdf`, `ablation_frs_variance.pdf`

### Why not only {4, 2, 8}?

The code uses **`K_VALUES = [4, 8, 12, 16]`** (see `sample_count_ablation.py`): **12** bridges small subsamples and full pass@16; **16** is the **no‑subsampling** baseline (variance 0 in `ablation_frs_variance_table` for k=16).

---

## 6. Figures (examples)

| Pattern | Example path |
|:---|:---|
| All‑datasets top‑K accuracy grid | `topk_ablation_results/topk_ablation_all_datasets_accuracy.png` |
| Sample ablation | `sample_count_ablation_results/ablation_*.pdf` |
| Bins / cumulative | `reasoning_confidence_bins_results/reasoning_bins_*.png`, `reasoning_cumulative_*.png` |

---

## How to regenerate this markdown

```bash
cd threshold
python build_master_results_doc.py
```

## How to regenerate experiments

```bash
# Top-K ablation (from JSONL)
python topk_ablation.py --data_root . --output_dir ./topk_ablation_results

# Correctness conditioned
python correctness_conditioned.py --data_root . --output_dir ./correctness_conditioned_results

# Reasoning bins (needs PORTKEY_API_KEY if using judge path inside the script)
python reasoning_confidence_bins.py run --data-root . --output-dir ./reasoning_confidence_bins_results

# Judge-free top-K accuracy only
python reasoning_confidence_bins.py topk-accuracy --data-root . --output-dir ./reasoning_confidence_bins_results

# Sample count ablation
python sample_count_ablation.py --data_root . --output_dir ./sample_count_ablation_results
```

---

# Part II — Full numerical tables (from `FRS_experiment_results.md`)

The following section is the **complete** prior document (Experiments A–B, summary stats, findings, limitations).

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


---

# Part III — Tables (CSV-backed)

Sections below are **generated** from CSVs under `threshold/`. If a file is missing, that subsection is omitted.


### Top‑K ablation — `topk_ablation_results/topk_ablation_results.csv`

| model | dataset | top_k_pct | accuracy | n_traces | n_total | mean_confidence |
| --- | --- | --- | --- | --- | --- | --- |
| DS-R1-1.5B | AQuA | 10 | 80.34 | 407 | 4064 | 0.2964 |
| DS-R1-1.5B | AQuA | 20 | 72.69 | 813 | 4064 | 0.2587 |
| DS-R1-1.5B | AQuA | 30 | 64.97 | 1219 | 4064 | 0.2357 |
| DS-R1-1.5B | AQuA | 50 | 54.92 | 2032 | 4064 | 0.2067 |
| DS-R1-1.5B | AQuA | 70 | 47.91 | 2845 | 4064 | 0.188 |
| DS-R1-1.5B | AQuA | 100 | 42.91 | 4064 | 4064 | 0.1662 |
| DS-R1-1.5B | CommonsenseQA | 10 | 44.37 | 1954 | 19536 | 0.213 |
| DS-R1-1.5B | CommonsenseQA | 20 | 42.14 | 3908 | 19536 | 0.1641 |
| DS-R1-1.5B | CommonsenseQA | 30 | 41.82 | 5861 | 19536 | 0.1414 |
| DS-R1-1.5B | CommonsenseQA | 50 | 41.74 | 9768 | 19536 | 0.1177 |
| DS-R1-1.5B | CommonsenseQA | 70 | 41.02 | 13675 | 19536 | 0.1039 |
| DS-R1-1.5B | CommonsenseQA | 100 | 40.74 | 19536 | 19536 | 0.0882 |
| DS-R1-1.5B | GPQA | 10 | 36.26 | 717 | 7168 | 0.4282 |
| DS-R1-1.5B | GPQA | 20 | 34.94 | 1434 | 7168 | 0.3583 |
| DS-R1-1.5B | GPQA | 30 | 34.73 | 2151 | 7168 | 0.316 |
| DS-R1-1.5B | GPQA | 50 | 33.71 | 3584 | 7168 | 0.2614 |
| DS-R1-1.5B | GPQA | 70 | 32.67 | 5017 | 7168 | 0.2255 |
| DS-R1-1.5B | GPQA | 100 | 35.18 | 7168 | 7168 | 0.1855 |
| DS-R1-1.5B | GSM8K | 10 | 92.28 | 2111 | 21104 | 0.368 |
| DS-R1-1.5B | GSM8K | 20 | 91.33 | 4221 | 21104 | 0.3314 |
| DS-R1-1.5B | GSM8K | 30 | 89.45 | 6331 | 21104 | 0.3033 |
| DS-R1-1.5B | GSM8K | 50 | 86.24 | 10552 | 21104 | 0.2571 |
| DS-R1-1.5B | GSM8K | 70 | 80.67 | 14773 | 21104 | 0.2247 |
| DS-R1-1.5B | GSM8K | 100 | 73.33 | 21104 | 21104 | 0.1909 |
| DS-R1-1.5B | MATH500 | 10 | 76.38 | 800 | 8000 | 0.3435 |
| DS-R1-1.5B | MATH500 | 20 | 73.5 | 1600 | 8000 | 0.3001 |
| DS-R1-1.5B | MATH500 | 30 | 70.62 | 2400 | 8000 | 0.274 |
| DS-R1-1.5B | MATH500 | 50 | 65.03 | 4000 | 8000 | 0.2408 |
| DS-R1-1.5B | MATH500 | 70 | 60.62 | 5600 | 8000 | 0.2182 |
| DS-R1-1.5B | MATH500 | 100 | 53.9 | 8000 | 8000 | 0.1907 |
| DS-R1-1.5B | SVAMP | 10 | 91.62 | 1600 | 16000 | 0.345 |
| DS-R1-1.5B | SVAMP | 20 | 91.91 | 3200 | 16000 | 0.3097 |
| DS-R1-1.5B | SVAMP | 30 | 91.19 | 4800 | 16000 | 0.2835 |
| DS-R1-1.5B | SVAMP | 50 | 89.28 | 8000 | 16000 | 0.241 |
| DS-R1-1.5B | SVAMP | 70 | 84.88 | 11200 | 16000 | 0.2107 |
| DS-R1-1.5B | SVAMP | 100 | 81.59 | 16000 | 16000 | 0.1789 |
| DS-R1-7B | AQuA | 10 | 92.14 | 407 | 4064 | 0.364 |
| DS-R1-7B | AQuA | 20 | 91.88 | 813 | 4064 | 0.3315 |
| DS-R1-7B | AQuA | 30 | 89.91 | 1219 | 4064 | 0.3093 |
| DS-R1-7B | AQuA | 50 | 85.88 | 2032 | 4064 | 0.277 |
| DS-R1-7B | AQuA | 70 | 80 | 2845 | 4064 | 0.2524 |
| DS-R1-7B | AQuA | 100 | 71.19 | 4064 | 4064 | 0.2214 |
| DS-R1-7B | CommonsenseQA | 10 | 56.91 | 1954 | 19536 | 0.2177 |
| DS-R1-7B | CommonsenseQA | 20 | 56.83 | 3908 | 19536 | 0.1895 |
| DS-R1-7B | CommonsenseQA | 30 | 56.49 | 5861 | 19536 | 0.1747 |
| DS-R1-7B | CommonsenseQA | 50 | 54.98 | 9768 | 19536 | 0.1565 |
| DS-R1-7B | CommonsenseQA | 70 | 52.53 | 13675 | 19536 | 0.1441 |
| DS-R1-7B | CommonsenseQA | 100 | 49.39 | 19536 | 19536 | 0.1286 |
| DS-R1-7B | GPQA | 10 | 43.1 | 717 | 7168 | 0.4015 |
| DS-R1-7B | GPQA | 20 | 40.45 | 1434 | 7168 | 0.3449 |
| DS-R1-7B | GPQA | 30 | 41.7 | 2151 | 7168 | 0.3099 |
| DS-R1-7B | GPQA | 50 | 43.64 | 3584 | 7168 | 0.2631 |
| DS-R1-7B | GPQA | 70 | 46.68 | 5017 | 7168 | 0.2298 |
| DS-R1-7B | GPQA | 100 | 50.74 | 7168 | 7168 | 0.1917 |
| DS-R1-7B | GSM8K | 10 | 96.92 | 2111 | 21104 | 0.4301 |
| DS-R1-7B | GSM8K | 20 | 96.14 | 4221 | 21104 | 0.4034 |
| DS-R1-7B | GSM8K | 30 | 95.75 | 6331 | 21104 | 0.3852 |
| DS-R1-7B | GSM8K | 50 | 94.85 | 10552 | 21104 | 0.3573 |
| DS-R1-7B | GSM8K | 70 | 94.13 | 14773 | 21104 | 0.3336 |
| DS-R1-7B | GSM8K | 100 | 90.28 | 21104 | 21104 | 0.2943 |
| DS-R1-7B | MATH500 | 10 | 77.25 | 800 | 8000 | 0.4419 |
| DS-R1-7B | MATH500 | 20 | 76.31 | 1600 | 8000 | 0.4022 |
| DS-R1-7B | MATH500 | 30 | 75.33 | 2400 | 8000 | 0.3758 |
| DS-R1-7B | MATH500 | 50 | 72.52 | 4000 | 8000 | 0.3383 |
| DS-R1-7B | MATH500 | 70 | 68.57 | 5600 | 8000 | 0.3088 |
| DS-R1-7B | MATH500 | 100 | 60.41 | 8000 | 8000 | 0.2688 |
| DS-R1-7B | SVAMP | 10 | 96.25 | 1600 | 16000 | 0.3907 |
| DS-R1-7B | SVAMP | 20 | 96.12 | 3200 | 16000 | 0.3623 |
| DS-R1-7B | SVAMP | 30 | 95.88 | 4800 | 16000 | 0.3428 |
| DS-R1-7B | SVAMP | 50 | 95.75 | 8000 | 16000 | 0.3139 |
| DS-R1-7B | SVAMP | 70 | 95.34 | 11200 | 16000 | 0.2906 |
| DS-R1-7B | SVAMP | 100 | 91.91 | 16000 | 16000 | 0.2552 |
| Gemma-7B | AQuA | 10 | 13.51 | 407 | 4064 | 0.5902 |
| Gemma-7B | AQuA | 20 | 19.56 | 813 | 4064 | 0.4154 |
| Gemma-7B | AQuA | 30 | 23.63 | 1219 | 4064 | 0.3291 |
| Gemma-7B | AQuA | 50 | 27.31 | 2032 | 4064 | 0.2421 |
| Gemma-7B | AQuA | 70 | 28.15 | 2845 | 4064 | 0.1948 |
| Gemma-7B | AQuA | 100 | 27.88 | 4064 | 4064 | 0.1493 |
| Gemma-7B | CommonsenseQA | 10 | 15.51 | 1954 | 19536 | 0.9333 |
| Gemma-7B | CommonsenseQA | 20 | 18.3 | 3908 | 19536 | 0.854 |
| Gemma-7B | CommonsenseQA | 30 | 19.69 | 5861 | 19536 | 0.7292 |
| Gemma-7B | CommonsenseQA | 50 | 16.71 | 9768 | 19536 | 0.5213 |
| Gemma-7B | CommonsenseQA | 70 | 17.84 | 13675 | 19536 | 0.3926 |
| Gemma-7B | CommonsenseQA | 100 | 18.34 | 19536 | 19536 | 0.2804 |
| Gemma-7B | GPQA | 10 | 11.4 | 719 | 7168 | 0.4091 |
| Gemma-7B | GPQA | 20 | 7.95 | 1434 | 7168 | 0.3031 |
| Gemma-7B | GPQA | 30 | 9.02 | 2151 | 7168 | 0.2475 |
| Gemma-7B | GPQA | 50 | 15.26 | 3584 | 7168 | 0.1733 |
| Gemma-7B | GPQA | 70 | 19.08 | 5017 | 7168 | 0.1308 |
| Gemma-7B | GPQA | 100 | 20.68 | 7168 | 7168 | 0.0939 |
| Gemma-7B | GSM8K | 10 | 24.44 | 2111 | 21104 | 0.4328 |
| Gemma-7B | GSM8K | 20 | 33.07 | 4221 | 21104 | 0.3103 |
| Gemma-7B | GSM8K | 30 | 35.65 | 6331 | 21104 | 0.2561 |
| Gemma-7B | GSM8K | 50 | 36.59 | 10552 | 21104 | 0.1994 |
| Gemma-7B | GSM8K | 70 | 36.38 | 14773 | 21104 | 0.1662 |
| Gemma-7B | GSM8K | 100 | 32.64 | 21104 | 21104 | 0.1304 |
| Gemma-7B | MATH500 | 10 | 4 | 800 | 8000 | 0.6404 |
| Gemma-7B | MATH500 | 20 | 7 | 1600 | 8000 | 0.4881 |
| Gemma-7B | MATH500 | 30 | 9.12 | 2400 | 8000 | 0.4016 |
| Gemma-7B | MATH500 | 50 | 12.2 | 4000 | 8000 | 0.3048 |
| Gemma-7B | MATH500 | 70 | 13.41 | 5600 | 8000 | 0.2475 |
| Gemma-7B | MATH500 | 100 | 13.34 | 8000 | 8000 | 0.1888 |
| Gemma-7B | SVAMP | 10 | 21.94 | 1600 | 16000 | 0.4762 |
| Gemma-7B | SVAMP | 20 | 32 | 3200 | 16000 | 0.3208 |
| Gemma-7B | SVAMP | 30 | 36.12 | 4800 | 16000 | 0.2558 |
| Gemma-7B | SVAMP | 50 | 40.21 | 8000 | 16000 | 0.1917 |
| Gemma-7B | SVAMP | 70 | 42.27 | 11200 | 16000 | 0.1564 |
| Gemma-7B | SVAMP | 100 | 40.94 | 16000 | 16000 | 0.1209 |
| LLaMA-3.1-8B | AQuA | 10 | 56.76 | 407 | 4064 | 0.7634 |
| LLaMA-3.1-8B | AQuA | 20 | 50.55 | 813 | 4064 | 0.6831 |
| LLaMA-3.1-8B | AQuA | 30 | 45.53 | 1219 | 4064 | 0.6069 |
| LLaMA-3.1-8B | AQuA | 50 | 40.9 | 2032 | 4064 | 0.4823 |
| LLaMA-3.1-8B | AQuA | 70 | 42.57 | 2845 | 4064 | 0.3965 |
| LLaMA-3.1-8B | AQuA | 100 | 44.34 | 4064 | 4064 | 0.3065 |
| LLaMA-3.1-8B | CommonsenseQA | 10 | 68.27 | 1954 | 19536 | 0.7365 |
| LLaMA-3.1-8B | CommonsenseQA | 20 | 68.37 | 3908 | 19536 | 0.6788 |
| LLaMA-3.1-8B | CommonsenseQA | 30 | 68.11 | 5861 | 19536 | 0.6338 |
| LLaMA-3.1-8B | CommonsenseQA | 50 | 67.07 | 9768 | 19536 | 0.5569 |
| LLaMA-3.1-8B | CommonsenseQA | 70 | 66.27 | 13675 | 19536 | 0.4892 |
| LLaMA-3.1-8B | CommonsenseQA | 100 | 65.02 | 19536 | 19536 | 0.3903 |
| LLaMA-3.1-8B | GPQA | 10 | 38.91 | 717 | 7168 | 0.5064 |
| LLaMA-3.1-8B | GPQA | 20 | 40.93 | 1434 | 7168 | 0.4288 |
| LLaMA-3.1-8B | GPQA | 30 | 41.47 | 2151 | 7168 | 0.3808 |
| LLaMA-3.1-8B | GPQA | 50 | 38.62 | 3584 | 7168 | 0.32 |
| LLaMA-3.1-8B | GPQA | 70 | 36.2 | 5017 | 7168 | 0.28 |
| LLaMA-3.1-8B | GPQA | 100 | 35.49 | 7168 | 7168 | 0.2331 |
| LLaMA-3.1-8B | GSM8K | 10 | 82.85 | 2111 | 21104 | 0.8 |
| LLaMA-3.1-8B | GSM8K | 20 | 77.49 | 4221 | 21104 | 0.7165 |
| LLaMA-3.1-8B | GSM8K | 30 | 75.25 | 6331 | 21104 | 0.6048 |
| LLaMA-3.1-8B | GSM8K | 50 | 78.03 | 10552 | 21104 | 0.4603 |
| LLaMA-3.1-8B | GSM8K | 70 | 78.39 | 14773 | 21104 | 0.3809 |
| LLaMA-3.1-8B | GSM8K | 100 | 75.33 | 21104 | 21104 | 0.3019 |
| LLaMA-3.1-8B | MATH500 | 10 | 54.5 | 800 | 8000 | 0.7485 |
| LLaMA-3.1-8B | MATH500 | 20 | 44.19 | 1600 | 8000 | 0.6563 |
| LLaMA-3.1-8B | MATH500 | 30 | 39.29 | 2400 | 8000 | 0.5734 |
| LLaMA-3.1-8B | MATH500 | 50 | 35.72 | 4000 | 8000 | 0.4566 |
| LLaMA-3.1-8B | MATH500 | 70 | 35.36 | 5600 | 8000 | 0.383 |
| LLaMA-3.1-8B | MATH500 | 100 | 34.9 | 8000 | 8000 | 0.3044 |
| LLaMA-3.1-8B | SVAMP | 10 | 86.19 | 1600 | 16000 | 0.8266 |
| LLaMA-3.1-8B | SVAMP | 20 | 78.88 | 3200 | 16000 | 0.6921 |
| LLaMA-3.1-8B | SVAMP | 30 | 79.79 | 4800 | 16000 | 0.5563 |
| LLaMA-3.1-8B | SVAMP | 50 | 81.66 | 8000 | 16000 | 0.4187 |
| LLaMA-3.1-8B | SVAMP | 70 | 82.38 | 11200 | 16000 | 0.3456 |
| LLaMA-3.1-8B | SVAMP | 100 | 80.82 | 16000 | 16000 | 0.2736 |
| Phi-4 | AQuA | 10 | 53.32 | 407 | 4064 | 0.5288 |
| Phi-4 | AQuA | 20 | 53.26 | 813 | 4064 | 0.4616 |
| Phi-4 | AQuA | 30 | 52.83 | 1219 | 4064 | 0.4144 |
| Phi-4 | AQuA | 50 | 52.85 | 2032 | 4064 | 0.3476 |
| Phi-4 | AQuA | 70 | 52.72 | 2845 | 4064 | 0.3001 |
| Phi-4 | AQuA | 100 | 53.96 | 4064 | 4064 | 0.2441 |
| Phi-4 | CommonsenseQA | 10 | 65.76 | 1954 | 19536 | 0.3489 |
| Phi-4 | CommonsenseQA | 20 | 58.01 | 3908 | 19536 | 0.2972 |
| Phi-4 | CommonsenseQA | 30 | 54.72 | 5861 | 19536 | 0.2689 |
| Phi-4 | CommonsenseQA | 50 | 54.95 | 9769 | 19536 | 0.2324 |
| Phi-4 | CommonsenseQA | 70 | 57.18 | 13675 | 19536 | 0.2041 |
| Phi-4 | CommonsenseQA | 100 | 49.45 | 19536 | 19536 | 0.1633 |
| Phi-4 | GPQA | 10 | 17.43 | 717 | 7168 | 0.5762 |
| Phi-4 | GPQA | 20 | 16.44 | 1442 | 7168 | 0.4899 |
| Phi-4 | GPQA | 30 | 17.29 | 2151 | 7168 | 0.4322 |
| Phi-4 | GPQA | 50 | 19.78 | 3584 | 7168 | 0.3518 |
| Phi-4 | GPQA | 70 | 23.76 | 5017 | 7168 | 0.2962 |
| Phi-4 | GPQA | 100 | 26.09 | 7168 | 7168 | 0.2308 |
| Phi-4 | GSM8K | 10 | 62.58 | 2111 | 21104 | 0.5179 |
| Phi-4 | GSM8K | 20 | 62.73 | 4221 | 21104 | 0.4499 |
| Phi-4 | GSM8K | 30 | 63.01 | 6331 | 21104 | 0.4022 |
| Phi-4 | GSM8K | 50 | 61.44 | 10552 | 21104 | 0.3334 |
| Phi-4 | GSM8K | 70 | 62.84 | 14773 | 21104 | 0.2859 |
| Phi-4 | GSM8K | 100 | 65.5 | 21104 | 21104 | 0.232 |
| Phi-4 | MATH500 | 10 | 60.75 | 800 | 8000 | 0.6897 |
| Phi-4 | MATH500 | 20 | 58.56 | 1600 | 8000 | 0.6282 |
| Phi-4 | MATH500 | 30 | 56.58 | 2400 | 8000 | 0.5822 |
| Phi-4 | MATH500 | 50 | 55.78 | 4000 | 8000 | 0.5023 |
| Phi-4 | MATH500 | 70 | 54.98 | 5600 | 8000 | 0.4304 |
| Phi-4 | MATH500 | 100 | 55.3 | 8000 | 8000 | 0.3425 |
| Phi-4 | SVAMP | 10 | 47.31 | 1600 | 16000 | 0.5305 |
| Phi-4 | SVAMP | 20 | 52.03 | 3200 | 16000 | 0.4535 |
| Phi-4 | SVAMP | 30 | 54.06 | 4800 | 16000 | 0.4038 |
| Phi-4 | SVAMP | 50 | 54.86 | 8000 | 16000 | 0.336 |
| Phi-4 | SVAMP | 70 | 55 | 11200 | 16000 | 0.2888 |
| Phi-4 | SVAMP | 100 | 57.04 | 16000 | 16000 | 0.2342 |
| Phi-4-Reas. | AQuA | 10 | 73.46 | 407 | 4064 | 0.3296 |
| Phi-4-Reas. | AQuA | 20 | 71.83 | 813 | 4064 | 0.2773 |
| Phi-4-Reas. | AQuA | 30 | 71.37 | 1219 | 4064 | 0.2514 |
| Phi-4-Reas. | AQuA | 50 | 67.37 | 2032 | 4064 | 0.2192 |
| Phi-4-Reas. | AQuA | 70 | 64.36 | 2845 | 4064 | 0.1966 |
| Phi-4-Reas. | AQuA | 100 | 58.44 | 4064 | 4064 | 0.1686 |
| Phi-4-Reas. | CommonsenseQA | 10 | 67.2 | 1954 | 19536 | 0.4734 |
| Phi-4-Reas. | CommonsenseQA | 20 | 75.33 | 3908 | 19536 | 0.3389 |
| Phi-4-Reas. | CommonsenseQA | 30 | 77.34 | 5861 | 19536 | 0.2645 |
| Phi-4-Reas. | CommonsenseQA | 50 | 78.71 | 9768 | 19536 | 0.1894 |
| Phi-4-Reas. | CommonsenseQA | 70 | 77.47 | 13675 | 19536 | 0.1524 |
| Phi-4-Reas. | CommonsenseQA | 100 | 74.17 | 19536 | 19536 | 0.1197 |
| Phi-4-Reas. | GPQA | 10 | 56.21 | 717 | 7168 | 0.3562 |
| Phi-4-Reas. | GPQA | 20 | 58.79 | 1434 | 7168 | 0.2545 |
| Phi-4-Reas. | GPQA | 30 | 57.51 | 2151 | 7168 | 0.2109 |
| Phi-4-Reas. | GPQA | 50 | 54.27 | 3584 | 7168 | 0.1676 |
| Phi-4-Reas. | GPQA | 70 | 51.43 | 5017 | 7168 | 0.1438 |
| Phi-4-Reas. | GPQA | 100 | 47.31 | 7168 | 7168 | 0.1173 |
| Phi-4-Reas. | GSM8K | 10 | 96.02 | 2111 | 21104 | 0.3618 |
| Phi-4-Reas. | GSM8K | 20 | 97.2 | 4221 | 21104 | 0.3042 |
| Phi-4-Reas. | GSM8K | 30 | 97.17 | 6331 | 21104 | 0.2749 |
| Phi-4-Reas. | GSM8K | 50 | 97.15 | 10552 | 21104 | 0.2394 |
| Phi-4-Reas. | GSM8K | 70 | 96.65 | 14773 | 21104 | 0.2147 |
| Phi-4-Reas. | GSM8K | 100 | 94.86 | 21104 | 21104 | 0.182 |
| Phi-4-Reas. | MATH500 | 10 | 88.5 | 800 | 8000 | 0.3846 |
| Phi-4-Reas. | MATH500 | 20 | 90 | 1600 | 8000 | 0.3334 |
| Phi-4-Reas. | MATH500 | 30 | 91.17 | 2400 | 8000 | 0.305 |
| Phi-4-Reas. | MATH500 | 50 | 90.85 | 4000 | 8000 | 0.2686 |
| Phi-4-Reas. | MATH500 | 70 | 89.21 | 5600 | 8000 | 0.2418 |
| Phi-4-Reas. | MATH500 | 100 | 84.71 | 8000 | 8000 | 0.2063 |
| Phi-4-Reas. | SVAMP | 10 | 94.5 | 1600 | 16000 | 0.3629 |
| Phi-4-Reas. | SVAMP | 20 | 95.88 | 3200 | 16000 | 0.2948 |
| Phi-4-Reas. | SVAMP | 30 | 96.42 | 4800 | 16000 | 0.2609 |
| Phi-4-Reas. | SVAMP | 50 | 96.31 | 8000 | 16000 | 0.221 |
| Phi-4-Reas. | SVAMP | 70 | 95.41 | 11200 | 16000 | 0.1944 |
| Phi-4-Reas. | SVAMP | 100 | 94.25 | 16000 | 16000 | 0.1618 |
| Qwen2.5-7B | AQuA | 10 | 86.49 | 407 | 4064 | 0.8607 |
| Qwen2.5-7B | AQuA | 20 | 80.69 | 813 | 4064 | 0.8329 |
| Qwen2.5-7B | AQuA | 30 | 77.44 | 1219 | 4064 | 0.8084 |
| Qwen2.5-7B | AQuA | 50 | 66.58 | 2032 | 4064 | 0.7596 |
| Qwen2.5-7B | AQuA | 70 | 59.65 | 2845 | 4064 | 0.7011 |
| Qwen2.5-7B | AQuA | 100 | 62.57 | 4064 | 4064 | 0.5917 |
| Qwen2.5-7B | CommonsenseQA | 10 | 82.29 | 1954 | 19536 | 0.3544 |
| Qwen2.5-7B | CommonsenseQA | 20 | 86.03 | 3908 | 19536 | 0.2892 |
| Qwen2.5-7B | CommonsenseQA | 30 | 86.15 | 5861 | 19536 | 0.2602 |
| Qwen2.5-7B | CommonsenseQA | 50 | 85.1 | 9768 | 19536 | 0.2287 |
| Qwen2.5-7B | CommonsenseQA | 70 | 83.55 | 13675 | 19536 | 0.209 |
| Qwen2.5-7B | CommonsenseQA | 100 | 81.11 | 19536 | 19536 | 0.1846 |
| Qwen2.5-7B | GPQA | 10 | 39.61 | 717 | 7168 | 0.5961 |
| Qwen2.5-7B | GPQA | 20 | 37.94 | 1434 | 7168 | 0.5466 |
| Qwen2.5-7B | GPQA | 30 | 36.36 | 2151 | 7168 | 0.5135 |
| Qwen2.5-7B | GPQA | 50 | 35.27 | 3584 | 7168 | 0.4634 |
| Qwen2.5-7B | GPQA | 70 | 34.48 | 5017 | 7168 | 0.4205 |
| Qwen2.5-7B | GPQA | 100 | 33.79 | 7168 | 7168 | 0.3539 |
| Qwen2.5-7B | GSM8K | 10 | 62.62 | 2111 | 21104 | 0.8826 |
| Qwen2.5-7B | GSM8K | 20 | 61.48 | 4221 | 21104 | 0.8584 |
| Qwen2.5-7B | GSM8K | 30 | 61.24 | 6331 | 21104 | 0.8343 |
| Qwen2.5-7B | GSM8K | 50 | 64.6 | 10552 | 21104 | 0.7466 |
| Qwen2.5-7B | GSM8K | 70 | 72.61 | 14773 | 21104 | 0.6437 |
| Qwen2.5-7B | GSM8K | 100 | 77.55 | 21104 | 21104 | 0.531 |
| Qwen2.5-7B | MATH500 | 10 | 53.5 | 800 | 8000 | 0.8737 |
| Qwen2.5-7B | MATH500 | 20 | 51 | 1600 | 8000 | 0.8498 |
| Qwen2.5-7B | MATH500 | 30 | 49.54 | 2400 | 8000 | 0.8293 |
| Qwen2.5-7B | MATH500 | 50 | 46.18 | 4000 | 8000 | 0.7867 |
| Qwen2.5-7B | MATH500 | 70 | 45.23 | 5600 | 8000 | 0.733 |
| Qwen2.5-7B | MATH500 | 100 | 51.31 | 8000 | 8000 | 0.6275 |
| Qwen2.5-7B | SVAMP | 10 | 62.25 | 1600 | 16000 | 0.8977 |
| Qwen2.5-7B | SVAMP | 20 | 61.72 | 3200 | 16000 | 0.8706 |
| Qwen2.5-7B | SVAMP | 30 | 62.29 | 4800 | 16000 | 0.8396 |
| Qwen2.5-7B | SVAMP | 50 | 66.95 | 8000 | 16000 | 0.7239 |
| Qwen2.5-7B | SVAMP | 70 | 75.1 | 11200 | 16000 | 0.6142 |
| Qwen2.5-7B | SVAMP | 100 | 80.38 | 16000 | 16000 | 0.4996 |
| Qwen2.5-Math | AQuA | 10 | 47.91 | 407 | 4064 | 0.6051 |
| Qwen2.5-Math | AQuA | 20 | 48.95 | 813 | 4064 | 0.5334 |
| Qwen2.5-Math | AQuA | 30 | 50.45 | 1219 | 4064 | 0.488 |
| Qwen2.5-Math | AQuA | 50 | 51.92 | 2032 | 4064 | 0.426 |
| Qwen2.5-Math | AQuA | 70 | 53.85 | 2845 | 4064 | 0.3801 |
| Qwen2.5-Math | AQuA | 100 | 54.21 | 4064 | 4064 | 0.3155 |
| Qwen2.5-Math | CommonsenseQA | 10 | 50.72 | 1954 | 19536 | 0.2919 |
| Qwen2.5-Math | CommonsenseQA | 20 | 50.38 | 3908 | 19536 | 0.2382 |
| Qwen2.5-Math | CommonsenseQA | 30 | 50.2 | 5861 | 19536 | 0.205 |
| Qwen2.5-Math | CommonsenseQA | 50 | 50.51 | 9768 | 19536 | 0.1658 |
| Qwen2.5-Math | CommonsenseQA | 70 | 49.96 | 13675 | 19536 | 0.1426 |
| Qwen2.5-Math | CommonsenseQA | 100 | 47.28 | 19536 | 19536 | 0.1167 |
| Qwen2.5-Math | GPQA | 10 | 39.19 | 717 | 7168 | 0.3671 |
| Qwen2.5-Math | GPQA | 20 | 37.03 | 1434 | 7168 | 0.3087 |
| Qwen2.5-Math | GPQA | 30 | 35.66 | 2151 | 7168 | 0.2723 |
| Qwen2.5-Math | GPQA | 50 | 33.62 | 3584 | 7168 | 0.2235 |
| Qwen2.5-Math | GPQA | 70 | 31.95 | 5017 | 7168 | 0.1917 |
| Qwen2.5-Math | GPQA | 100 | 29.99 | 7168 | 7168 | 0.1565 |
| Qwen2.5-Math | GSM8K | 10 | 55.42 | 2111 | 21104 | 0.5657 |
| Qwen2.5-Math | GSM8K | 20 | 61.38 | 4221 | 21104 | 0.511 |
| Qwen2.5-Math | GSM8K | 30 | 64.38 | 6331 | 21104 | 0.4776 |
| Qwen2.5-Math | GSM8K | 50 | 67.87 | 10552 | 21104 | 0.4309 |
| Qwen2.5-Math | GSM8K | 70 | 70.46 | 14773 | 21104 | 0.3938 |
| Qwen2.5-Math | GSM8K | 100 | 71.17 | 21104 | 21104 | 0.3373 |
| Qwen2.5-Math | MATH500 | 10 | 61 | 800 | 8000 | 0.6103 |
| Qwen2.5-Math | MATH500 | 20 | 62.56 | 1600 | 8000 | 0.5667 |
| Qwen2.5-Math | MATH500 | 30 | 63.38 | 2400 | 8000 | 0.5391 |
| Qwen2.5-Math | MATH500 | 50 | 65.3 | 4000 | 8000 | 0.4991 |
| Qwen2.5-Math | MATH500 | 70 | 64.59 | 5600 | 8000 | 0.4667 |
| Qwen2.5-Math | MATH500 | 100 | 61.08 | 8000 | 8000 | 0.4145 |
| Qwen2.5-Math | SVAMP | 10 | 44.75 | 1600 | 16000 | 0.5574 |
| Qwen2.5-Math | SVAMP | 20 | 57.41 | 3200 | 16000 | 0.4928 |
| Qwen2.5-Math | SVAMP | 30 | 63.38 | 4800 | 16000 | 0.4547 |
| Qwen2.5-Math | SVAMP | 50 | 70.44 | 8000 | 16000 | 0.4035 |
| Qwen2.5-Math | SVAMP | 70 | 73.41 | 11200 | 16000 | 0.3651 |
| Qwen2.5-Math | SVAMP | 100 | 75.79 | 16000 | 16000 | 0.3104 |
| Qwen3-4B | AQuA | 10 | 53.32 | 407 | 4064 | 0.3935 |
| Qwen3-4B | AQuA | 20 | 53.38 | 813 | 4064 | 0.3644 |
| Qwen3-4B | AQuA | 30 | 51.76 | 1219 | 4064 | 0.3463 |
| Qwen3-4B | AQuA | 50 | 51.57 | 2032 | 4064 | 0.3208 |
| Qwen3-4B | AQuA | 70 | 49.42 | 2845 | 4064 | 0.3018 |
| Qwen3-4B | AQuA | 100 | 49.41 | 4064 | 4064 | 0.2758 |
| Qwen3-4B | CommonsenseQA | 10 | 84.49 | 1954 | 19536 | 0.3064 |
| Qwen3-4B | CommonsenseQA | 20 | 78.33 | 3908 | 19536 | 0.276 |
| Qwen3-4B | CommonsenseQA | 30 | 74.95 | 5861 | 19536 | 0.2621 |
| Qwen3-4B | CommonsenseQA | 50 | 70.77 | 9768 | 19536 | 0.2461 |
| Qwen3-4B | CommonsenseQA | 70 | 68.41 | 13675 | 19536 | 0.2348 |
| Qwen3-4B | CommonsenseQA | 100 | 68.02 | 19536 | 19536 | 0.2155 |
| Qwen3-4B | GPQA | 10 | 62.9 | 717 | 7168 | 0.3807 |
| Qwen3-4B | GPQA | 20 | 58.23 | 1434 | 7168 | 0.3351 |
| Qwen3-4B | GPQA | 30 | 57.74 | 2151 | 7168 | 0.3099 |
| Qwen3-4B | GPQA | 50 | 56.72 | 3584 | 7168 | 0.2777 |
| Qwen3-4B | GPQA | 70 | 57.29 | 5017 | 7168 | 0.2559 |
| Qwen3-4B | GPQA | 100 | 58.29 | 7168 | 7168 | 0.2303 |
| Qwen3-4B | GSM8K | 10 | 92.14 | 2111 | 21104 | 0.3984 |
| Qwen3-4B | GSM8K | 20 | 91.12 | 4221 | 21104 | 0.3581 |
| Qwen3-4B | GSM8K | 30 | 88.63 | 6331 | 21104 | 0.3346 |
| Qwen3-4B | GSM8K | 50 | 83.41 | 10552 | 21104 | 0.3046 |
| Qwen3-4B | GSM8K | 70 | 78.58 | 14773 | 21104 | 0.2843 |
| Qwen3-4B | GSM8K | 100 | 73.68 | 21104 | 21104 | 0.2593 |
| Qwen3-4B | MATH500 | 10 | 27 | 800 | 8000 | 0.502 |
| Qwen3-4B | MATH500 | 20 | 31.06 | 1600 | 8000 | 0.4716 |
| Qwen3-4B | MATH500 | 30 | 34.83 | 2400 | 8000 | 0.4511 |
| Qwen3-4B | MATH500 | 50 | 39.02 | 4000 | 8000 | 0.42 |
| Qwen3-4B | MATH500 | 70 | 42.75 | 5600 | 8000 | 0.394 |
| Qwen3-4B | MATH500 | 100 | 44.6 | 8000 | 8000 | 0.3539 |
| Qwen3-4B | SVAMP | 10 | 88 | 1600 | 16000 | 0.3924 |
| Qwen3-4B | SVAMP | 20 | 87.94 | 3200 | 16000 | 0.349 |
| Qwen3-4B | SVAMP | 30 | 86.88 | 4800 | 16000 | 0.3246 |
| Qwen3-4B | SVAMP | 50 | 83.93 | 8000 | 16000 | 0.294 |
| Qwen3-4B | SVAMP | 70 | 80.39 | 11200 | 16000 | 0.2739 |
| Qwen3-4B | SVAMP | 100 | 76.23 | 16000 | 16000 | 0.25 |

### Correctness conditioned — `correctness_conditioned_results/correctness_conditioned.csv`

| model | dataset | n_traces | median_conf | acc_high_conf_half | acc_low_conf_half | gap_pp | mean_conf_correct | mean_conf_incorrect | mean_conf_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DS-R1-1.5B | AQuA | 4064 | 0.151364 | 54.92 | 30.91 | 24.02 | 0.1869 | 0.1506 | 0.0363 |
| DS-R1-1.5B | CommonsenseQA | 19536 | 0.075196 | 41.74 | 39.74 | 2 | 0.0882 | 0.0882 | 0 |
| DS-R1-1.5B | GPQA | 7168 | 0.153629 | 33.71 | 36.66 | -2.96 | 0.1818 | 0.1876 | -0.0058 |
| DS-R1-1.5B | GSM8K | 21104 | 0.15951 | 86.24 | 60.42 | 25.82 | 0.2055 | 0.1508 | 0.0547 |
| DS-R1-1.5B | MATH500 | 8000 | 0.174691 | 65.03 | 42.78 | 22.25 | 0.2058 | 0.1731 | 0.0327 |
| DS-R1-1.5B | SVAMP | 16000 | 0.150052 | 89.28 | 73.91 | 15.36 | 0.1855 | 0.1495 | 0.0361 |
| DS-R1-7B | AQuA | 4064 | 0.207458 | 85.88 | 56.5 | 29.38 | 0.2363 | 0.1847 | 0.0516 |
| DS-R1-7B | CommonsenseQA | 19536 | 0.120632 | 54.98 | 43.8 | 11.18 | 0.1326 | 0.1247 | 0.0079 |
| DS-R1-7B | GPQA | 7168 | 0.166869 | 43.64 | 57.84 | -14.2 | 0.1805 | 0.2033 | -0.0228 |
| DS-R1-7B | GSM8K | 21104 | 0.294944 | 94.85 | 85.7 | 9.15 | 0.3002 | 0.2391 | 0.0612 |
| DS-R1-7B | MATH500 | 8000 | 0.257452 | 72.52 | 48.3 | 24.22 | 0.2877 | 0.2398 | 0.0479 |
| DS-R1-7B | SVAMP | 16000 | 0.251427 | 95.75 | 88.06 | 7.69 | 0.2591 | 0.2115 | 0.0476 |
| Gemma-7B | AQuA | 4064 | 0.091056 | 27.31 | 28.44 | -1.13 | 0.1233 | 0.1594 | -0.0361 |
| Gemma-7B | CommonsenseQA | 19536 | 0.122194 | 16.71 | 19.96 | -3.26 | 0.2768 | 0.2812 | -0.0044 |
| Gemma-7B | GPQA | 7168 | 0.035126 | 15.26 | 26.09 | -10.83 | 0.0632 | 0.1019 | -0.0387 |
| Gemma-7B | GSM8K | 21104 | 0.097017 | 36.59 | 28.7 | 7.89 | 0.1227 | 0.1341 | -0.0114 |
| Gemma-7B | MATH500 | 8000 | 0.128866 | 12.2 | 14.48 | -2.28 | 0.1475 | 0.1952 | -0.0476 |
| Gemma-7B | SVAMP | 16000 | 0.080253 | 40.21 | 41.66 | -1.45 | 0.0979 | 0.1367 | -0.0388 |
| LLaMA-3.1-8B | AQuA | 4064 | 0.227607 | 40.9 | 47.79 | -6.89 | 0.3112 | 0.3027 | 0.0085 |
| LLaMA-3.1-8B | CommonsenseQA | 19536 | 0.377034 | 67.07 | 62.97 | 4.1 | 0.3974 | 0.3773 | 0.0201 |
| LLaMA-3.1-8B | GPQA | 7168 | 0.201709 | 38.62 | 32.37 | 6.25 | 0.2433 | 0.2275 | 0.0158 |
| LLaMA-3.1-8B | GSM8K | 21104 | 0.207713 | 78.03 | 72.63 | 5.4 | 0.3087 | 0.2813 | 0.0274 |
| LLaMA-3.1-8B | MATH500 | 8000 | 0.232534 | 35.72 | 34.08 | 1.65 | 0.3315 | 0.2898 | 0.0417 |
| LLaMA-3.1-8B | SVAMP | 16000 | 0.184135 | 81.66 | 79.99 | 1.68 | 0.2758 | 0.2645 | 0.0113 |
| Phi-4 | AQuA | 4064 | 0.209021 | 52.85 | 55.07 | -2.21 | 0.2418 | 0.2469 | -0.005 |
| Phi-4 | CommonsenseQA | 19536 | 0.155801 | 54.95 | 43.95 | 11 | 0.1821 | 0.1449 | 0.0372 |
| Phi-4 | GPQA | 7168 | 0.188148 | 19.78 | 32.39 | -12.61 | 0.201 | 0.2413 | -0.0403 |
| Phi-4 | GSM8K | 21104 | 0.193494 | 61.44 | 69.56 | -8.12 | 0.2259 | 0.2437 | -0.0178 |
| Phi-4 | MATH500 | 8000 | 0.308979 | 55.78 | 54.82 | 0.95 | 0.3441 | 0.3406 | 0.0036 |
| Phi-4 | SVAMP | 16000 | 0.19814 | 54.86 | 59.23 | -4.36 | 0.2257 | 0.2456 | -0.0199 |
| Phi-4-Reas. | AQuA | 4064 | 0.155075 | 67.37 | 49.51 | 17.86 | 0.181 | 0.1512 | 0.0298 |
| Phi-4-Reas. | CommonsenseQA | 19536 | 0.066169 | 78.71 | 69.64 | 9.07 | 0.1126 | 0.1401 | -0.0275 |
| Phi-4-Reas. | GPQA | 7168 | 0.092186 | 54.27 | 40.35 | 13.92 | 0.123 | 0.1121 | 0.0108 |
| Phi-4-Reas. | GSM8K | 21104 | 0.168825 | 97.15 | 92.58 | 4.57 | 0.1836 | 0.154 | 0.0296 |
| Phi-4-Reas. | MATH500 | 8000 | 0.193457 | 90.85 | 78.57 | 12.28 | 0.2107 | 0.182 | 0.0287 |
| Phi-4-Reas. | SVAMP | 16000 | 0.143635 | 96.31 | 92.19 | 4.12 | 0.1627 | 0.1461 | 0.0167 |
| Qwen2.5-7B | AQuA | 4064 | 0.629621 | 66.58 | 58.56 | 8.02 | 0.5991 | 0.5793 | 0.0198 |
| Qwen2.5-7B | CommonsenseQA | 19536 | 0.17017 | 85.1 | 77.11 | 8 | 0.1861 | 0.1779 | 0.0082 |
| Qwen2.5-7B | GPQA | 7168 | 0.352794 | 35.27 | 32.31 | 2.96 | 0.3606 | 0.3504 | 0.0102 |
| Qwen2.5-7B | GSM8K | 21104 | 0.449792 | 64.6 | 90.49 | -25.89 | 0.4907 | 0.6703 | -0.1796 |
| Qwen2.5-7B | MATH500 | 8000 | 0.671676 | 46.18 | 56.45 | -10.28 | 0.5995 | 0.6569 | -0.0574 |
| Qwen2.5-7B | SVAMP | 16000 | 0.391561 | 66.95 | 93.81 | -26.86 | 0.4545 | 0.6844 | -0.23 |
| Qwen2.5-Math | AQuA | 4064 | 0.296729 | 51.92 | 56.5 | -4.58 | 0.3111 | 0.3207 | -0.0096 |
| Qwen2.5-Math | CommonsenseQA | 19536 | 0.094321 | 50.51 | 44.04 | 6.47 | 0.1207 | 0.113 | 0.0077 |
| Qwen2.5-Math | GPQA | 7168 | 0.127398 | 33.62 | 26.37 | 7.25 | 0.1697 | 0.1508 | 0.019 |
| Qwen2.5-Math | GSM8K | 21104 | 0.330028 | 67.87 | 74.46 | -6.59 | 0.3303 | 0.3546 | -0.0243 |
| Qwen2.5-Math | MATH500 | 8000 | 0.41219 | 65.3 | 56.85 | 8.45 | 0.4212 | 0.404 | 0.0172 |
| Qwen2.5-Math | SVAMP | 16000 | 0.296938 | 70.44 | 81.14 | -10.7 | 0.2945 | 0.36 | -0.0655 |
| Qwen3-4B | AQuA | 4064 | 0.267385 | 51.57 | 47.24 | 4.33 | 0.2774 | 0.2742 | 0.0032 |
| Qwen3-4B | CommonsenseQA | 19536 | 0.214731 | 70.77 | 65.26 | 5.51 | 0.2185 | 0.209 | 0.0095 |
| Qwen3-4B | GPQA | 7168 | 0.2136 | 56.72 | 59.85 | -3.12 | 0.2308 | 0.2296 | 0.0011 |
| Qwen3-4B | GSM8K | 21104 | 0.245311 | 83.41 | 63.95 | 19.46 | 0.2677 | 0.2359 | 0.0318 |
| Qwen3-4B | MATH500 | 8000 | 0.351094 | 39.02 | 50.18 | -11.15 | 0.3427 | 0.363 | -0.0203 |
| Qwen3-4B | SVAMP | 16000 | 0.234829 | 83.93 | 68.54 | 15.39 | 0.2558 | 0.2315 | 0.0242 |

### Disjoint bins — population accuracy (wide, Experiment A–style columns)

*Disjoint bins: each column is **population accuracy** (terminal correctness) within that confidence percentile band of the ranked pool — **not** cumulative top‑K. Column labels match the **right edge** of each band (same style as §1 top‑K% columns where they overlap). This run uses five bands covering the top 50% of the pool (see `bin_label` in the CSV).*


#### AQuA

| Model | 10% | 20% | 30% | 40% | 50% | Spread (pp) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| DS-R1-1.5B | 80.34 | 65.02 | 49.63 | 44.09 | 35.47 | +44.88 |
| DS-R1-7B | 92.14 | 91.63 | 85.75 | 83.00 | 76.85 | +15.29 |
| Gemma-7B | 13.51 | 25.62 | 31.94 | 32.02 | 33.50 | -19.98 |
| LLaMA-3.1-8B | 56.76 | 44.33 | 35.38 | 34.73 | 33.25 | +23.51 |
| Phi-4 | 53.32 | 53.20 | 52.09 | 51.97 | 53.69 | -0.38 |
| Phi-4-Reas. | 73.46 | 70.20 | 70.27 | 61.58 | 61.33 | +12.13 |
| Qwen2.5-7B | 86.49 | 74.88 | 71.01 | 58.62 | 41.87 | +44.61 |
| Qwen2.5-Math | 47.91 | 50.00 | 53.32 | 54.43 | 53.94 | -6.03 |
| Qwen3-4B | 53.32 | 53.45 | 48.40 | 51.48 | 51.23 | +2.09 |


#### CommonsenseQA

| Model | 10% | 20% | 30% | 40% | 50% | Spread (pp) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| DS-R1-1.5B | 44.37 | 39.92 | 41.17 | 40.94 | 42.29 | +2.08 |
| DS-R1-7B | 56.91 | 56.76 | 55.81 | 54.71 | 50.69 | +6.22 |
| Gemma-7B | 15.51 | 21.08 | 22.48 | 11.41 | 13.06 | +2.45 |
| LLaMA-3.1-8B | 68.27 | 68.47 | 67.59 | 65.56 | 65.44 | +2.83 |
| Phi-4 | 65.76 | 50.26 | 48.13 | 52.10 | 58.53 | +7.24 |
| Phi-4-Reas. | 67.20 | 83.47 | 81.36 | 81.73 | 79.77 | -12.58 |
| Qwen2.5-7B | 82.29 | 89.76 | 86.38 | 85.16 | 81.93 | +0.37 |
| Qwen2.5-Math | 50.72 | 50.05 | 49.82 | 51.54 | 50.44 | +0.28 |
| Qwen3-4B | 84.49 | 72.16 | 68.20 | 66.48 | 62.52 | +21.97 |


#### GPQA

| Model | 10% | 20% | 30% | 40% | 50% | Spread (pp) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| DS-R1-1.5B | 36.26 | 33.61 | 34.31 | 34.45 | 29.89 | +6.37 |
| DS-R1-7B | 43.10 | 37.80 | 44.21 | 44.77 | 48.32 | -5.23 |
| Gemma-7B | 11.44 | 4.46 | 11.16 | 22.45 | 26.82 | -15.38 |
| LLaMA-3.1-8B | 38.91 | 42.96 | 42.54 | 35.43 | 33.24 | +5.67 |
| Phi-4 | 17.43 | 15.62 | 18.83 | 23.15 | 23.88 | -6.45 |
| Phi-4-Reas. | 56.21 | 61.37 | 54.95 | 52.72 | 46.09 | +10.12 |
| Qwen2.5-7B | 39.61 | 36.26 | 33.19 | 34.31 | 32.96 | +6.65 |
| Qwen2.5-Math | 39.19 | 34.87 | 32.91 | 31.66 | 29.47 | +9.72 |
| Qwen3-4B | 62.90 | 53.56 | 56.76 | 55.93 | 54.47 | +8.43 |


#### GSM8K

| Model | 10% | 20% | 30% | 40% | 50% | Spread (pp) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| DS-R1-1.5B | 92.28 | 90.38 | 85.69 | 84.41 | 78.44 | +13.84 |
| DS-R1-7B | 96.92 | 95.36 | 94.98 | 93.98 | 93.03 | +3.89 |
| Gemma-7B | 24.44 | 41.71 | 40.83 | 39.05 | 36.92 | -12.48 |
| LLaMA-3.1-8B | 82.85 | 72.13 | 70.77 | 82.75 | 81.66 | +1.19 |
| Phi-4 | 62.58 | 62.89 | 63.57 | 56.59 | 61.56 | +1.01 |
| Phi-4-Reas. | 96.02 | 98.39 | 97.11 | 97.49 | 96.73 | -0.71 |
| Qwen2.5-7B | 62.62 | 60.33 | 60.78 | 59.81 | 79.48 | -16.85 |
| Qwen2.5-Math | 55.42 | 67.35 | 70.39 | 72.13 | 74.08 | -18.65 |
| Qwen3-4B | 92.14 | 90.09 | 83.61 | 77.63 | 73.55 | +18.58 |


#### MATH500

| Model | 10% | 20% | 30% | 40% | 50% | Spread (pp) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| DS-R1-1.5B | 76.38 | 70.62 | 64.88 | 60.62 | 52.62 | +23.75 |
| DS-R1-7B | 77.25 | 75.38 | 73.38 | 71.88 | 64.75 | +12.50 |
| Gemma-7B | 4.00 | 10.00 | 13.38 | 16.25 | 17.38 | -13.37 |
| LLaMA-3.1-8B | 54.50 | 33.88 | 29.50 | 28.50 | 32.25 | +22.25 |
| Phi-4 | 60.75 | 56.38 | 52.62 | 54.00 | 55.12 | +5.63 |
| Phi-4-Reas. | 88.50 | 91.50 | 93.50 | 92.00 | 88.75 | -0.25 |
| Qwen2.5-7B | 53.50 | 48.50 | 46.62 | 43.50 | 38.75 | +14.75 |
| Qwen2.5-Math | 61.00 | 64.12 | 65.00 | 68.00 | 68.38 | -7.37 |
| Qwen3-4B | 27.00 | 35.12 | 42.38 | 42.00 | 48.62 | -21.62 |


#### SVAMP

| Model | 10% | 20% | 30% | 40% | 50% | Spread (pp) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| DS-R1-1.5B | 91.62 | 92.19 | 89.75 | 88.75 | 84.06 | +7.56 |
| DS-R1-7B | 96.25 | 96.00 | 95.38 | 95.81 | 95.31 | +0.94 |
| Gemma-7B | 21.94 | 42.06 | 44.38 | 46.50 | 46.19 | -24.25 |
| LLaMA-3.1-8B | 86.19 | 71.56 | 81.62 | 84.38 | 84.56 | +1.62 |
| Phi-4 | 47.31 | 56.75 | 58.13 | 55.94 | 56.19 | -8.88 |
| Phi-4-Reas. | 94.50 | 97.25 | 97.50 | 96.56 | 95.75 | -1.25 |
| Qwen2.5-7B | 62.25 | 61.19 | 63.44 | 62.12 | 85.75 | -23.50 |
| Qwen2.5-Math | 44.75 | 70.06 | 75.31 | 80.75 | 81.31 | -36.56 |
| Qwen3-4B | 88.00 | 87.88 | 84.75 | 81.94 | 77.06 | +10.94 |

### Cumulative top‑K% — population accuracy (wide, same column style as §1 where overlapping)

*Cumulative top‑K% (population accuracy): column **K%** is mean terminal correctness over **all traces in the top K%** of the confidence-ranked pool (same cumulative semantics as the accuracy ablation’s top‑K% slices, modulo implementation details in each script).*


#### AQuA

| Model | 10% | 20% | 30% | 40% | 50% | Spread (pp) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| DS-R1-1.5B | 80.34 | 72.69 | 65.00 | 59.78 | 54.92 | +25.42 |
| DS-R1-7B | 92.14 | 91.88 | 89.84 | 88.13 | 85.88 | +6.26 |
| Gemma-7B | 13.51 | 19.56 | 23.69 | 25.77 | 27.31 | -13.80 |
| LLaMA-3.1-8B | 56.76 | 50.55 | 45.49 | 42.80 | 40.90 | +15.86 |
| Phi-4 | 53.32 | 53.26 | 52.87 | 52.64 | 52.85 | +0.46 |
| Phi-4-Reas. | 73.46 | 71.83 | 71.31 | 68.88 | 67.37 | +6.09 |
| Qwen2.5-7B | 86.49 | 80.69 | 77.46 | 72.76 | 66.58 | +19.90 |
| Qwen2.5-Math | 47.91 | 48.95 | 50.41 | 51.41 | 51.92 | -4.01 |
| Qwen3-4B | 53.32 | 53.38 | 51.72 | 51.66 | 51.57 | +1.74 |


#### CommonsenseQA

| Model | 10% | 20% | 30% | 40% | 50% | Spread (pp) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| DS-R1-1.5B | 44.37 | 42.14 | 41.82 | 41.60 | 41.74 | +2.63 |
| DS-R1-7B | 56.91 | 56.83 | 56.49 | 56.05 | 54.98 | +1.93 |
| Gemma-7B | 15.51 | 18.30 | 19.69 | 17.62 | 16.71 | -1.20 |
| LLaMA-3.1-8B | 68.27 | 68.37 | 68.11 | 67.47 | 67.07 | +1.20 |
| Phi-4 | 65.76 | 58.01 | 54.72 | 54.06 | 54.95 | +10.81 |
| Phi-4-Reas. | 67.20 | 75.33 | 77.34 | 78.44 | 78.71 | -11.51 |
| Qwen2.5-7B | 82.29 | 86.03 | 86.15 | 85.90 | 85.10 | -2.81 |
| Qwen2.5-Math | 50.72 | 50.38 | 50.20 | 50.53 | 50.51 | +0.20 |
| Qwen3-4B | 84.49 | 78.33 | 74.95 | 72.83 | 70.77 | +13.72 |


#### GPQA

| Model | 10% | 20% | 30% | 40% | 50% | Spread (pp) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| DS-R1-1.5B | 36.26 | 34.94 | 34.73 | 34.66 | 33.71 | +2.56 |
| DS-R1-7B | 43.10 | 40.45 | 41.70 | 42.47 | 43.64 | -0.54 |
| Gemma-7B | 11.44 | 7.95 | 9.02 | 12.38 | 15.26 | -3.83 |
| LLaMA-3.1-8B | 38.91 | 40.93 | 41.47 | 39.96 | 38.62 | +0.30 |
| Phi-4 | 17.43 | 16.53 | 17.29 | 18.76 | 19.78 | -2.35 |
| Phi-4-Reas. | 56.21 | 58.79 | 57.51 | 56.31 | 54.27 | +1.94 |
| Qwen2.5-7B | 39.61 | 37.94 | 36.36 | 35.84 | 35.27 | +4.34 |
| Qwen2.5-Math | 39.19 | 37.03 | 35.66 | 34.66 | 33.62 | +5.57 |
| Qwen3-4B | 62.90 | 58.23 | 57.74 | 57.29 | 56.72 | +6.18 |


#### GSM8K

| Model | 10% | 20% | 30% | 40% | 50% | Spread (pp) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| DS-R1-1.5B | 92.28 | 91.33 | 89.45 | 88.19 | 86.24 | +6.04 |
| DS-R1-7B | 96.92 | 96.14 | 95.75 | 95.31 | 94.85 | +2.07 |
| Gemma-7B | 24.44 | 33.07 | 35.66 | 36.51 | 36.59 | -12.15 |
| LLaMA-3.1-8B | 82.85 | 77.49 | 75.25 | 77.13 | 78.03 | +4.82 |
| Phi-4 | 62.58 | 62.73 | 63.01 | 61.41 | 61.44 | +1.14 |
| Phi-4-Reas. | 96.02 | 97.20 | 97.17 | 97.25 | 97.15 | -1.13 |
| Qwen2.5-7B | 62.62 | 61.48 | 61.24 | 60.89 | 64.60 | -1.98 |
| Qwen2.5-Math | 55.42 | 61.38 | 64.39 | 66.32 | 67.87 | -12.45 |
| Qwen3-4B | 92.14 | 91.12 | 88.61 | 85.87 | 83.41 | +8.73 |


#### MATH500

| Model | 10% | 20% | 30% | 40% | 50% | Spread (pp) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| DS-R1-1.5B | 76.38 | 73.50 | 70.62 | 68.12 | 65.03 | +11.35 |
| DS-R1-7B | 77.25 | 76.31 | 75.33 | 74.47 | 72.52 | +4.73 |
| Gemma-7B | 4.00 | 7.00 | 9.12 | 10.91 | 12.20 | -8.20 |
| LLaMA-3.1-8B | 54.50 | 44.19 | 39.29 | 36.59 | 35.73 | +18.78 |
| Phi-4 | 60.75 | 58.56 | 56.58 | 55.94 | 55.77 | +4.98 |
| Phi-4-Reas. | 88.50 | 90.00 | 91.17 | 91.38 | 90.85 | -2.35 |
| Qwen2.5-7B | 53.50 | 51.00 | 49.54 | 48.03 | 46.17 | +7.33 |
| Qwen2.5-Math | 61.00 | 62.56 | 63.38 | 64.53 | 65.30 | -4.30 |
| Qwen3-4B | 27.00 | 31.06 | 34.83 | 36.62 | 39.02 | -12.02 |


#### SVAMP

| Model | 10% | 20% | 30% | 40% | 50% | Spread (pp) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| DS-R1-1.5B | 91.62 | 91.91 | 91.19 | 90.58 | 89.28 | +2.35 |
| DS-R1-7B | 96.25 | 96.12 | 95.88 | 95.86 | 95.75 | +0.50 |
| Gemma-7B | 21.94 | 32.00 | 36.12 | 38.72 | 40.21 | -18.28 |
| LLaMA-3.1-8B | 86.19 | 78.88 | 79.79 | 80.94 | 81.66 | +4.52 |
| Phi-4 | 47.31 | 52.03 | 54.06 | 54.53 | 54.86 | -7.55 |
| Phi-4-Reas. | 94.50 | 95.88 | 96.42 | 96.45 | 96.31 | -1.81 |
| Qwen2.5-7B | 62.25 | 61.72 | 62.29 | 62.25 | 66.95 | -4.70 |
| Qwen2.5-Math | 44.75 | 57.41 | 63.38 | 67.72 | 70.44 | -25.69 |
| Qwen3-4B | 88.00 | 87.94 | 86.88 | 85.64 | 83.93 | +4.07 |

### §5 Sample‑count ablation — all numerical outputs

These tables match **§5** in Part I. Shard files under `sample_count_ablation_results/checkpoints/chunks/` are not repeated here (they partition `ablation_per_bootstrap_detail.csv` by model×dataset).


### FRS variance by dataset — `sample_count_ablation_results/ablation_frs_variance_table.csv`

| dataset | k_sub | mean_bootstrap_std_across_models |
| --- | --- | --- |
| AQuA | 4 | 1.6465 |
| AQuA | 8 | 1.042853 |
| AQuA | 12 | 0.602305 |
| AQuA | 16 | 0 |
| CommonsenseQA | 4 | 0.650381 |
| CommonsenseQA | 8 | 0.418093 |
| CommonsenseQA | 12 | 0.236566 |
| CommonsenseQA | 16 | 0 |
| GPQA | 4 | 1.304642 |
| GPQA | 8 | 0.769915 |
| GPQA | 12 | 0.438869 |
| GPQA | 16 | 0 |
| GSM8K | 4 | 0.617468 |
| GSM8K | 8 | 0.349722 |
| GSM8K | 12 | 0.209556 |
| GSM8K | 16 | 0 |
| MATH500 | 4 | 0.913827 |
| MATH500 | 8 | 0.518677 |
| MATH500 | 12 | 0.307327 |
| MATH500 | 16 | 0 |
| SVAMP | 4 | 0.646962 |
| SVAMP | 8 | 0.377616 |
| SVAMP | 12 | 0.22799 |
| SVAMP | 16 | 0 |

### Rankings vs k=16 (mean/std FRS, Spearman) — `sample_count_ablation_results/ablation_rankings.csv`

| model | dataset | k_sub | frs_acc_mean | frs_acc_std | spearman_vs_k16_mean | spearman_vs_k16_std |
| --- | --- | --- | --- | --- | --- | --- |
| DS-R1-1.5B | AQuA | 4 | 50.003942 | 1.503649 | 0.980582 | 0.018853 |
| DS-R1-1.5B | AQuA | 8 | 50.458664 | 1.024142 | 0.985234 | 0.019273 |
| DS-R1-1.5B | AQuA | 12 | 50.670608 | 0.577125 | 0.993749 | 0.009862 |
| DS-R1-1.5B | AQuA | 16 | 50.8858 | 0 |  |  |
| DS-R1-1.5B | CommonsenseQA | 4 | 40.49221 | 0.571906 | 0.990333 | 0.008309 |
| DS-R1-1.5B | CommonsenseQA | 8 | 40.427512 | 0.452721 | 0.990667 | 0.008357 |
| DS-R1-1.5B | CommonsenseQA | 12 | 40.428866 | 0.227389 | 0.993667 | 0.008172 |
| DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 0 |  |  |
| DS-R1-1.5B | GPQA | 4 | 32.082588 | 1.419483 | 0.989829 | 0.010255 |
| DS-R1-1.5B | GPQA | 8 | 31.675216 | 0.73739 | 0.991833 | 0.008289 |
| DS-R1-1.5B | GPQA | 12 | 31.719484 | 0.419342 | 0.993583 | 0.008127 |
| DS-R1-1.5B | GPQA | 16 | 31.7243 | 0 |  |  |
| DS-R1-1.5B | GSM8K | 4 | 78.162244 | 0.638044 | 0.988 | 0.007559 |
| DS-R1-1.5B | GSM8K | 8 | 78.67514 | 0.305891 | 0.994333 | 0.007975 |
| DS-R1-1.5B | GSM8K | 12 | 78.773056 | 0.195305 | 0.998667 | 0.004567 |
| DS-R1-1.5B | GSM8K | 16 | 78.8476 | 0 |  |  |
| DS-R1-1.5B | MATH500 | 4 | 60.05 | 0.810455 | 0.992249 | 0.008748 |
| DS-R1-1.5B | MATH500 | 8 | 60.828 | 0.472203 | 0.9935 | 0.008081 |
| DS-R1-1.5B | MATH500 | 12 | 61.006004 | 0.280483 | 0.992749 | 0.008153 |
| DS-R1-1.5B | MATH500 | 16 | 61.05 | 0 |  |  |
| DS-R1-1.5B | SVAMP | 4 | 83.546 | 0.52759 | 0.983582 | 0.007916 |
| DS-R1-1.5B | SVAMP | 8 | 83.935 | 0.313066 | 0.987832 | 0.008243 |
| DS-R1-1.5B | SVAMP | 12 | 84.047664 | 0.182386 | 0.993667 | 0.008172 |
| DS-R1-1.5B | SVAMP | 16 | 84.15 | 0 |  |  |
| DS-R1-7B | AQuA | 4 | 74.885826 | 1.179839 | 0.980582 | 0.018853 |
| DS-R1-7B | AQuA | 8 | 75.30119 | 0.84398 | 0.985234 | 0.019273 |
| DS-R1-7B | AQuA | 12 | 75.476382 | 0.461866 | 0.993749 | 0.009862 |
| DS-R1-7B | AQuA | 16 | 75.2461 | 0 |  |  |
| DS-R1-7B | CommonsenseQA | 4 | 49.152312 | 0.702605 | 0.990333 | 0.008309 |
| DS-R1-7B | CommonsenseQA | 8 | 49.19081 | 0.451614 | 0.990667 | 0.008357 |
| DS-R1-7B | CommonsenseQA | 12 | 49.232574 | 0.249176 | 0.993667 | 0.008172 |
| DS-R1-7B | CommonsenseQA | 16 | 49.396 | 0 |  |  |
| DS-R1-7B | GPQA | 4 | 44.957594 | 1.346797 | 0.989829 | 0.010255 |
| DS-R1-7B | GPQA | 8 | 43.919638 | 0.84012 | 0.991833 | 0.008289 |
| DS-R1-7B | GPQA | 12 | 43.614582 | 0.468566 | 0.993583 | 0.008127 |
| DS-R1-7B | GPQA | 16 | 43.5547 | 0 |  |  |
| DS-R1-7B | GSM8K | 4 | 90.877938 | 0.366548 | 0.988 | 0.007559 |
| DS-R1-7B | GSM8K | 8 | 90.882482 | 0.208468 | 0.994333 | 0.007975 |
| DS-R1-7B | GSM8K | 12 | 90.920896 | 0.128661 | 0.998667 | 0.004567 |
| DS-R1-7B | GSM8K | 16 | 90.9496 | 0 |  |  |
| DS-R1-7B | MATH500 | 4 | 62.978 | 0.721475 | 0.992249 | 0.008748 |
| DS-R1-7B | MATH500 | 8 | 63.394 | 0.40351 | 0.9935 | 0.008081 |
| DS-R1-7B | MATH500 | 12 | 63.477336 | 0.187167 | 0.992749 | 0.008153 |
| DS-R1-7B | MATH500 | 16 | 63.475 | 0 |  |  |
| DS-R1-7B | SVAMP | 4 | 92.183 | 0.367924 | 0.983582 | 0.007916 |
| DS-R1-7B | SVAMP | 8 | 92.188 | 0.213046 | 0.987832 | 0.008243 |
| DS-R1-7B | SVAMP | 12 | 92.164998 | 0.108081 | 0.993667 | 0.008172 |
| DS-R1-7B | SVAMP | 16 | 92.1875 | 0 |  |  |
| Gemma-7B | AQuA | 4 | 27.661434 | 1.48125 | 0.980582 | 0.018853 |
| Gemma-7B | AQuA | 8 | 27.76969 | 1.045421 | 0.985234 | 0.019273 |
| Gemma-7B | AQuA | 12 | 27.930448 | 0.55228 | 0.993749 | 0.009862 |
| Gemma-7B | AQuA | 16 | 27.6575 | 0 |  |  |
| Gemma-7B | CommonsenseQA | 4 | 17.989366 | 0.676852 | 0.990333 | 0.008309 |
| Gemma-7B | CommonsenseQA | 8 | 17.591722 | 0.405865 | 0.990667 | 0.008357 |
| Gemma-7B | CommonsenseQA | 12 | 17.473666 | 0.233639 | 0.993667 | 0.008172 |
| Gemma-7B | CommonsenseQA | 16 | 17.3423 | 0 |  |  |
| Gemma-7B | GPQA | 4 | 16.74554 | 1.196739 | 0.989829 | 0.010255 |
| Gemma-7B | GPQA | 8 | 15.717638 | 0.756247 | 0.991833 | 0.008289 |
| Gemma-7B | GPQA | 12 | 15.54093 | 0.333564 | 0.993583 | 0.008127 |
| Gemma-7B | GPQA | 16 | 15.4297 | 0 |  |  |
| Gemma-7B | GSM8K | 4 | 36.162244 | 0.796401 | 0.988 | 0.007559 |
| Gemma-7B | GSM8K | 8 | 36.740712 | 0.487174 | 0.994333 | 0.007975 |
| Gemma-7B | GSM8K | 12 | 36.958814 | 0.266423 | 0.998667 | 0.004567 |
| Gemma-7B | GSM8K | 16 | 37.083 | 0 |  |  |
| Gemma-7B | MATH500 | 4 | 14.234 | 0.774494 | 0.992249 | 0.008748 |
| Gemma-7B | MATH500 | 8 | 14.448 | 0.378499 | 0.9935 | 0.008081 |
| Gemma-7B | MATH500 | 12 | 14.478672 | 0.285292 | 0.992749 | 0.008153 |
| Gemma-7B | MATH500 | 16 | 14.475 | 0 |  |  |
| Gemma-7B | SVAMP | 4 | 42.061 | 0.713663 | 0.983582 | 0.007916 |
| Gemma-7B | SVAMP | 8 | 42.1535 | 0.484373 | 0.987832 | 0.008243 |
| Gemma-7B | SVAMP | 12 | 42.326004 | 0.322952 | 0.993667 | 0.008172 |
| Gemma-7B | SVAMP | 16 | 42.425 | 0 |  |  |
| LLaMA-3.1-8B | AQuA | 4 | 42.531484 | 1.706581 | 0.980582 | 0.018853 |
| LLaMA-3.1-8B | AQuA | 8 | 41.866144 | 1.137925 | 0.985234 | 0.019273 |
| LLaMA-3.1-8B | AQuA | 12 | 41.818904 | 0.700783 | 0.993749 | 0.009862 |
| LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 0 |  |  |
| LLaMA-3.1-8B | CommonsenseQA | 4 | 64.922204 | 0.673742 | 0.990333 | 0.008309 |
| LLaMA-3.1-8B | CommonsenseQA | 8 | 64.906636 | 0.429858 | 0.990667 | 0.008357 |
| LLaMA-3.1-8B | CommonsenseQA | 12 | 64.762774 | 0.240339 | 0.993667 | 0.008172 |
| LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 0 |  |  |
| LLaMA-3.1-8B | GPQA | 4 | 37.220988 | 1.293958 | 0.989829 | 0.010255 |
| LLaMA-3.1-8B | GPQA | 8 | 37.42634 | 0.880988 | 0.991833 | 0.008289 |
| LLaMA-3.1-8B | GPQA | 12 | 37.492558 | 0.447605 | 0.993583 | 0.008127 |
| LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 0 |  |  |
| LLaMA-3.1-8B | GSM8K | 4 | 75.978012 | 0.51158 | 0.988 | 0.007559 |
| LLaMA-3.1-8B | GSM8K | 8 | 75.893478 | 0.317969 | 0.994333 | 0.007975 |
| LLaMA-3.1-8B | GSM8K | 12 | 75.776098 | 0.206397 | 0.998667 | 0.004567 |
| LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 0 |  |  |
| LLaMA-3.1-8B | MATH500 | 4 | 34.464 | 1.189883 | 0.992249 | 0.008748 |
| LLaMA-3.1-8B | MATH500 | 8 | 34.424 | 0.729987 | 0.9935 | 0.008081 |
| LLaMA-3.1-8B | MATH500 | 12 | 34.303334 | 0.39043 | 0.992749 | 0.008153 |
| LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 0 |  |  |
| LLaMA-3.1-8B | SVAMP | 4 | 80.712 | 0.599112 | 0.983582 | 0.007916 |
| LLaMA-3.1-8B | SVAMP | 8 | 80.639 | 0.375974 | 0.987832 | 0.008243 |
| LLaMA-3.1-8B | SVAMP | 12 | 80.629336 | 0.213187 | 0.993667 | 0.008172 |
| LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 0 |  |  |
| Phi-4 | AQuA | 4 | 54.81891 | 1.857396 | 0.980582 | 0.018853 |
| Phi-4 | AQuA | 8 | 54.338584 | 1.052744 | 0.985234 | 0.019273 |
| Phi-4 | AQuA | 12 | 54.34383 | 0.565854 | 0.993749 | 0.009862 |
| Phi-4 | AQuA | 16 | 54.4783 | 0 |  |  |
| Phi-4 | CommonsenseQA | 4 | 55.223608 | 0.928002 | 0.990333 | 0.008309 |
| Phi-4 | CommonsenseQA | 8 | 54.948816 | 0.525025 | 0.990667 | 0.008357 |
| Phi-4 | CommonsenseQA | 12 | 54.607448 | 0.34362 | 0.993667 | 0.008172 |
| Phi-4 | CommonsenseQA | 16 | 54.3305 | 0 |  |  |
| Phi-4 | GPQA | 4 | 25.41517 | 1.367468 | 0.989829 | 0.010255 |
| Phi-4 | GPQA | 8 | 24.595984 | 0.739252 | 0.991833 | 0.008289 |
| Phi-4 | GPQA | 12 | 24.26339 | 0.447767 | 0.993583 | 0.008127 |
| Phi-4 | GPQA | 16 | 23.9955 | 0 |  |  |
| Phi-4 | GSM8K | 4 | 62.89158 | 0.802007 | 0.988 | 0.007559 |
| Phi-4 | GSM8K | 8 | 62.529196 | 0.41258 | 0.994333 | 0.007975 |
| Phi-4 | GSM8K | 12 | 62.501132 | 0.305712 | 0.998667 | 0.004567 |
| Phi-4 | GSM8K | 16 | 62.5095 | 0 |  |  |
| Phi-4 | MATH500 | 4 | 57.186 | 1.433564 | 0.992249 | 0.008748 |
| Phi-4 | MATH500 | 8 | 57.369 | 0.714992 | 0.9935 | 0.008081 |
| Phi-4 | MATH500 | 12 | 57.443332 | 0.402903 | 0.992749 | 0.008153 |
| Phi-4 | MATH500 | 16 | 57.475 | 0 |  |  |
| Phi-4 | SVAMP | 4 | 54.767 | 0.831621 | 0.983582 | 0.007916 |
| Phi-4 | SVAMP | 8 | 54.707 | 0.447215 | 0.987832 | 0.008243 |
| Phi-4 | SVAMP | 12 | 54.648332 | 0.2756 | 0.993667 | 0.008172 |
| Phi-4 | SVAMP | 16 | 54.525 | 0 |  |  |
| Phi-4-Reas. | AQuA | 4 | 59.842526 | 1.607104 | 0.980582 | 0.018853 |
| Phi-4-Reas. | AQuA | 8 | 59.970472 | 1.098219 | 0.985234 | 0.019273 |
| Phi-4-Reas. | AQuA | 12 | 60.308402 | 0.640069 | 0.993749 | 0.009862 |
| Phi-4-Reas. | AQuA | 16 | 60.3346 | 0 |  |  |
| Phi-4-Reas. | CommonsenseQA | 4 | 74.555284 | 0.707331 | 0.990333 | 0.008309 |
| Phi-4-Reas. | CommonsenseQA | 8 | 74.613416 | 0.415101 | 0.990667 | 0.008357 |
| Phi-4-Reas. | CommonsenseQA | 12 | 74.604972 | 0.22519 | 0.993667 | 0.008172 |
| Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 0 |  |  |
| Phi-4-Reas. | GPQA | 4 | 49.73214 | 1.33529 | 0.989829 | 0.010255 |
| Phi-4-Reas. | GPQA | 8 | 49.936388 | 0.811966 | 0.991833 | 0.008289 |
| Phi-4-Reas. | GPQA | 12 | 50.001484 | 0.464828 | 0.993583 | 0.008127 |
| Phi-4-Reas. | GPQA | 16 | 50.1116 | 0 |  |  |
| Phi-4-Reas. | GSM8K | 4 | 95.237298 | 0.233016 | 0.988 | 0.007559 |
| Phi-4-Reas. | GSM8K | 8 | 95.21797 | 0.137708 | 0.994333 | 0.007975 |
| Phi-4-Reas. | GSM8K | 12 | 95.206472 | 0.073444 | 0.998667 | 0.004567 |
| Phi-4-Reas. | GSM8K | 16 | 95.2331 | 0 |  |  |
| Phi-4-Reas. | MATH500 | 4 | 85.122 | 0.574737 | 0.992249 | 0.008748 |
| Phi-4-Reas. | MATH500 | 8 | 85.278 | 0.324848 | 0.9935 | 0.008081 |
| Phi-4-Reas. | MATH500 | 12 | 85.39733 | 0.20547 | 0.992749 | 0.008153 |
| Phi-4-Reas. | MATH500 | 16 | 85.475 | 0 |  |  |
| Phi-4-Reas. | SVAMP | 4 | 94.094 | 0.280786 | 0.983582 | 0.007916 |
| Phi-4-Reas. | SVAMP | 8 | 94.0485 | 0.178243 | 0.987832 | 0.008243 |
| Phi-4-Reas. | SVAMP | 12 | 94.055662 | 0.104637 | 0.993667 | 0.008172 |
| Phi-4-Reas. | SVAMP | 16 | 94.025 | 0 |  |  |
| Qwen2.5-7B | AQuA | 4 | 63.48819 | 1.590406 | 0.980582 | 0.018853 |
| Qwen2.5-7B | AQuA | 8 | 63.564964 | 1.011529 | 0.985234 | 0.019273 |
| Qwen2.5-7B | AQuA | 12 | 63.363512 | 0.653706 | 0.993749 | 0.009862 |
| Qwen2.5-7B | AQuA | 16 | 63.1398 | 0 |  |  |
| Qwen2.5-7B | CommonsenseQA | 4 | 81.30794 | 0.392969 | 0.990333 | 0.008309 |
| Qwen2.5-7B | CommonsenseQA | 8 | 81.396816 | 0.256306 | 0.990667 | 0.008357 |
| Qwen2.5-7B | CommonsenseQA | 12 | 81.390656 | 0.164343 | 0.993667 | 0.008172 |
| Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 0 |  |  |
| Qwen2.5-7B | GPQA | 4 | 34.819194 | 1.099196 | 0.989829 | 0.010255 |
| Qwen2.5-7B | GPQA | 8 | 34.687494 | 0.658348 | 0.991833 | 0.008289 |
| Qwen2.5-7B | GPQA | 12 | 34.70535 | 0.458387 | 0.993583 | 0.008127 |
| Qwen2.5-7B | GPQA | 16 | 34.8772 | 0 |  |  |
| Qwen2.5-7B | GSM8K | 4 | 68.818046 | 0.761087 | 0.988 | 0.007559 |
| Qwen2.5-7B | GSM8K | 8 | 67.674754 | 0.480418 | 0.994333 | 0.007975 |
| Qwen2.5-7B | GSM8K | 12 | 67.14683 | 0.259548 | 0.998667 | 0.004567 |
| Qwen2.5-7B | GSM8K | 16 | 66.8309 | 0 |  |  |
| Qwen2.5-7B | MATH500 | 4 | 44.224 | 1.204339 | 0.992249 | 0.008748 |
| Qwen2.5-7B | MATH500 | 8 | 43.161 | 0.746986 | 0.9935 | 0.008081 |
| Qwen2.5-7B | MATH500 | 12 | 42.786668 | 0.469278 | 0.992749 | 0.008153 |
| Qwen2.5-7B | MATH500 | 16 | 42.575 | 0 |  |  |
| Qwen2.5-7B | SVAMP | 4 | 71.744 | 0.913987 | 0.983582 | 0.007916 |
| Qwen2.5-7B | SVAMP | 8 | 70.489 | 0.46451 | 0.987832 | 0.008243 |
| Qwen2.5-7B | SVAMP | 12 | 69.928004 | 0.299626 | 0.993667 | 0.008172 |
| Qwen2.5-7B | SVAMP | 16 | 69.6375 | 0 |  |  |
| Qwen2.5-Math | AQuA | 4 | 53.011828 | 1.79637 | 0.980582 | 0.018853 |
| Qwen2.5-Math | AQuA | 8 | 52.51969 | 1.171863 | 0.985234 | 0.019273 |
| Qwen2.5-Math | AQuA | 12 | 52.38714 | 0.615097 | 0.993749 | 0.009862 |
| Qwen2.5-Math | AQuA | 16 | 52.5098 | 0 |  |  |
| Qwen2.5-Math | CommonsenseQA | 4 | 49.135118 | 0.666729 | 0.990333 | 0.008309 |
| Qwen2.5-Math | CommonsenseQA | 8 | 49.247324 | 0.486759 | 0.990667 | 0.008357 |
| Qwen2.5-Math | CommonsenseQA | 12 | 49.34942 | 0.245667 | 0.993667 | 0.008172 |
| Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 0 |  |  |
| Qwen2.5-Math | GPQA | 4 | 31.368302 | 1.33423 | 0.989829 | 0.010255 |
| Qwen2.5-Math | GPQA | 8 | 31.462048 | 0.778643 | 0.991833 | 0.008289 |
| Qwen2.5-Math | GPQA | 12 | 31.553564 | 0.431778 | 0.993583 | 0.008127 |
| Qwen2.5-Math | GPQA | 16 | 31.5569 | 0 |  |  |
| Qwen2.5-Math | GSM8K | 4 | 68.216078 | 0.794642 | 0.988 | 0.007559 |
| Qwen2.5-Math | GSM8K | 8 | 67.93859 | 0.438798 | 0.994333 | 0.007975 |
| Qwen2.5-Math | GSM8K | 12 | 67.760932 | 0.245218 | 0.998667 | 0.004567 |
| Qwen2.5-Math | GSM8K | 16 | 67.6933 | 0 |  |  |
| Qwen2.5-Math | MATH500 | 4 | 63.16 | 0.668535 | 0.992249 | 0.008748 |
| Qwen2.5-Math | MATH500 | 8 | 63.425 | 0.384688 | 0.9935 | 0.008081 |
| Qwen2.5-Math | MATH500 | 12 | 63.515334 | 0.239638 | 0.992749 | 0.008153 |
| Qwen2.5-Math | MATH500 | 16 | 63.5 | 0 |  |  |
| Qwen2.5-Math | SVAMP | 4 | 70.312 | 0.858366 | 0.983582 | 0.007916 |
| Qwen2.5-Math | SVAMP | 8 | 70.171 | 0.499652 | 0.987832 | 0.008243 |
| Qwen2.5-Math | SVAMP | 12 | 70.114002 | 0.289999 | 0.993667 | 0.008172 |
| Qwen2.5-Math | SVAMP | 16 | 70.1375 | 0 |  |  |
| Qwen3-4B | AQuA | 4 | 51.21261 | 2.095904 | 0.980582 | 0.018853 |
| Qwen3-4B | AQuA | 8 | 51.466532 | 0.999853 | 0.985234 | 0.019273 |
| Qwen3-4B | AQuA | 12 | 51.656174 | 0.65396 | 0.993749 | 0.009862 |
| Qwen3-4B | AQuA | 16 | 51.378 | 0 |  |  |
| Qwen3-4B | CommonsenseQA | 4 | 66.756766 | 0.533288 | 0.990333 | 0.008309 |
| Qwen3-4B | CommonsenseQA | 8 | 66.69778 | 0.339586 | 0.990667 | 0.008357 |
| Qwen3-4B | CommonsenseQA | 12 | 66.600336 | 0.199735 | 0.993667 | 0.008172 |
| Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 0 |  |  |
| Qwen3-4B | GPQA | 4 | 56.225442 | 1.348614 | 0.989829 | 0.010255 |
| Qwen3-4B | GPQA | 8 | 56.050224 | 0.726279 | 0.991833 | 0.008289 |
| Qwen3-4B | GPQA | 12 | 56.06325 | 0.477982 | 0.993583 | 0.008127 |
| Qwen3-4B | GPQA | 16 | 56.1384 | 0 |  |  |
| Qwen3-4B | GSM8K | 4 | 80.626238 | 0.653888 | 0.988 | 0.007559 |
| Qwen3-4B | GSM8K | 8 | 81.432528 | 0.358494 | 0.994333 | 0.007975 |
| Qwen3-4B | GSM8K | 12 | 81.842304 | 0.205295 | 0.998667 | 0.004567 |
| Qwen3-4B | GSM8K | 16 | 81.9845 | 0 |  |  |
| Qwen3-4B | MATH500 | 4 | 47.998 | 0.846961 | 0.992249 | 0.008748 |
| Qwen3-4B | MATH500 | 8 | 48.313 | 0.512378 | 0.9935 | 0.008081 |
| Qwen3-4B | MATH500 | 12 | 48.50466 | 0.305281 | 0.992749 | 0.008153 |
| Qwen3-4B | MATH500 | 16 | 48.425 | 0 |  |  |
| Qwen3-4B | SVAMP | 4 | 81.969 | 0.729613 | 0.983582 | 0.007916 |
| Qwen3-4B | SVAMP | 8 | 82.954 | 0.422467 | 0.987832 | 0.008243 |
| Qwen3-4B | SVAMP | 12 | 83.240672 | 0.255444 | 0.993667 | 0.008172 |
| Qwen3-4B | SVAMP | 16 | 83.4 | 0 |  |  |

### Global Spearman vs k=16 — `sample_count_ablation_results/ablation_rankings_global_spearman.csv`

| k_sub | spearman_mean_all_ds | spearman_std_all_ds |
| --- | --- | --- |
| 4 | 0.987429 | 0.011645 |
| 8 | 0.990566 | 0.011222 |
| 12 | 0.994347 | 0.008174 |

### Regime separation — `sample_count_ablation_results/ablation_regime_separation.csv`

| k_sub | model | dataset | mean_gap_pp | ref_gap_pp_k16 | sign_k | sign_k16 | sign_agrees |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | DS-R1-1.5B | AQuA | 14.1339 | 15.9449 | 1 | 1 | 1 |
| 4 | DS-R1-1.5B | CommonsenseQA | -0.5766 | -0.5016 | 0 | 0 | 1 |
| 4 | DS-R1-1.5B | GPQA | -5.8013 | -6.9196 | -1 | -1 | 1 |
| 4 | DS-R1-1.5B | GSM8K | 9.5315 | 11.0406 | 1 | 1 | 1 |
| 4 | DS-R1-1.5B | MATH500 | 12.454 | 14.3 | 1 | 1 | 1 |
| 4 | DS-R1-1.5B | SVAMP | 4.094 | 5.1125 | 1 | 1 | 1 |
| 4 | DS-R1-7B | AQuA | 7.3976 | 8.1201 | 1 | 1 | 1 |
| 4 | DS-R1-7B | CommonsenseQA | -0.4013 | 0.0205 | 0 | 0 | 1 |
| 4 | DS-R1-7B | GPQA | -11.8817 | -14.3694 | -1 | -1 | 1 |
| 4 | DS-R1-7B | GSM8K | 1.1046 | 1.3457 | 0 | 0 | 1 |
| 4 | DS-R1-7B | MATH500 | 5.196 | 6.125 | 1 | 1 | 1 |
| 4 | DS-R1-7B | SVAMP | 0.462 | 0.5625 | 0 | 0 | 1 |
| 4 | Gemma-7B | AQuA | -0.8307 | -0.4429 | 0 | 0 | 1 |
| 4 | Gemma-7B | CommonsenseQA | -0.8575 | -1.9861 | 0 | 0 | 1 |
| 4 | Gemma-7B | GPQA | -8.1495 | -10.4911 | -1 | -1 | 1 |
| 4 | Gemma-7B | GSM8K | 7.2995 | 8.8798 | 1 | 1 | 1 |
| 4 | Gemma-7B | MATH500 | 1.976 | 2.275 | 0 | 1 | 0 |
| 4 | Gemma-7B | SVAMP | 2.43 | 2.975 | 1 | 1 | 1 |
| 4 | LLaMA-3.1-8B | AQuA | -3.2559 | -5.4134 | -1 | -1 | 1 |
| 4 | LLaMA-3.1-8B | CommonsenseQA | -0.1769 | -0.4095 | 0 | 0 | 1 |
| 4 | LLaMA-3.1-8B | GPQA | 3.3594 | 4.1853 | 1 | 1 | 1 |
| 4 | LLaMA-3.1-8B | GSM8K | 1.1698 | 1.0235 | 0 | 0 | 1 |
| 4 | LLaMA-3.1-8B | MATH500 | -0.89 | -1.1 | 0 | 0 | 1 |
| 4 | LLaMA-3.1-8B | SVAMP | -0.245 | -0.25 | 0 | 0 | 1 |
| 4 | Phi-4 | AQuA | 1.3583 | 1.0335 | 0 | 0 | 1 |
| 4 | Phi-4 | CommonsenseQA | 11.8518 | 9.7563 | 1 | 1 | 1 |
| 4 | Phi-4 | GPQA | -1.4196 | -4.1853 | 0 | -1 | 0 |
| 4 | Phi-4 | GSM8K | -5.3397 | -5.9799 | -1 | -1 | 1 |
| 4 | Phi-4 | MATH500 | 3.552 | 4.35 | 1 | 1 | 1 |
| 4 | Phi-4 | SVAMP | -4.643 | -5.0375 | -1 | -1 | 1 |
| 4 | Phi-4-Reas. | AQuA | 2.8504 | 3.7894 | 1 | 1 | 1 |
| 4 | Phi-4-Reas. | CommonsenseQA | 0.8469 | 1.1261 | 0 | 0 | 1 |
| 4 | Phi-4-Reas. | GPQA | 5.3326 | 5.6083 | 1 | 1 | 1 |
| 4 | Phi-4-Reas. | GSM8K | 0.771 | 0.7392 | 0 | 0 | 1 |
| 4 | Phi-4-Reas. | MATH500 | 0.692 | 1.525 | 0 | 0 | 1 |
| 4 | Phi-4-Reas. | SVAMP | -0.333 | -0.45 | 0 | 0 | 1 |
| 4 | Qwen2.5-7B | AQuA | 1.8662 | 1.1319 | 0 | 0 | 1 |
| 4 | Qwen2.5-7B | CommonsenseQA | 0.4112 | 0.5426 | 0 | 0 | 1 |
| 4 | Qwen2.5-7B | GPQA | 1.8013 | 2.1763 | 0 | 1 | 0 |
| 4 | Qwen2.5-7B | GSM8K | -17.4776 | -21.4367 | -1 | -1 | 1 |
| 4 | Qwen2.5-7B | MATH500 | -14.246 | -17.475 | -1 | -1 | 1 |
| 4 | Qwen2.5-7B | SVAMP | -17.352 | -21.4875 | -1 | -1 | 1 |
| 4 | Qwen2.5-Math | AQuA | -2.0748 | -3.3957 | -1 | -1 | 1 |
| 4 | Qwen2.5-Math | CommonsenseQA | 3.7388 | 4.3817 | 1 | 1 | 1 |
| 4 | Qwen2.5-Math | GPQA | 3.1161 | 3.125 | 1 | 1 | 1 |
| 4 | Qwen2.5-Math | GSM8K | -6.1327 | -6.9466 | -1 | -1 | 1 |
| 4 | Qwen2.5-Math | MATH500 | 4.476 | 4.85 | 1 | 1 | 1 |
| 4 | Qwen2.5-Math | SVAMP | -10.943 | -11.3 | -1 | -1 | 1 |
| 4 | Qwen3-4B | AQuA | 3.1457 | 3.937 | 1 | 1 | 1 |
| 4 | Qwen3-4B | CommonsenseQA | -2.3604 | -2.9484 | -1 | -1 | 1 |
| 4 | Qwen3-4B | GPQA | -3.808 | -4.2969 | -1 | -1 | 1 |
| 4 | Qwen3-4B | GSM8K | 13.9128 | 16.613 | 1 | 1 | 1 |
| 4 | Qwen3-4B | MATH500 | 6.652 | 7.65 | 1 | 1 | 1 |
| 4 | Qwen3-4B | SVAMP | 11.779 | 14.3375 | 1 | 1 | 1 |
| 8 | DS-R1-1.5B | AQuA | 15.0413 | 15.9449 | 1 | 1 | 1 |
| 8 | DS-R1-1.5B | CommonsenseQA | -0.5811 | -0.5016 | 0 | 0 | 1 |
| 8 | DS-R1-1.5B | GPQA | -6.6641 | -6.9196 | -1 | -1 | 1 |
| 8 | DS-R1-1.5B | GSM8K | 10.6626 | 11.0406 | 1 | 1 | 1 |
| 8 | DS-R1-1.5B | MATH500 | 13.912 | 14.3 | 1 | 1 | 1 |
| 8 | DS-R1-1.5B | SVAMP | 4.6445 | 5.1125 | 1 | 1 | 1 |
| 8 | DS-R1-7B | AQuA | 8.2933 | 8.1201 | 1 | 1 | 1 |
| 8 | DS-R1-7B | CommonsenseQA | -0.4034 | 0.0205 | 0 | 0 | 1 |
| 8 | DS-R1-7B | GPQA | -13.7288 | -14.3694 | -1 | -1 | 1 |
| 8 | DS-R1-7B | GSM8K | 1.1797 | 1.3457 | 0 | 0 | 1 |
| 8 | DS-R1-7B | MATH500 | 5.99 | 6.125 | 1 | 1 | 1 |
| 8 | DS-R1-7B | SVAMP | 0.4835 | 0.5625 | 0 | 0 | 1 |
| 8 | Gemma-7B | AQuA | -0.5709 | -0.4429 | 0 | 0 | 1 |
| 8 | Gemma-7B | CommonsenseQA | -1.5451 | -1.9861 | 0 | 0 | 1 |
| 8 | Gemma-7B | GPQA | -9.8672 | -10.4911 | -1 | -1 | 1 |
| 8 | Gemma-7B | GSM8K | 8.3772 | 8.8798 | 1 | 1 | 1 |
| 8 | Gemma-7B | MATH500 | 2.225 | 2.275 | 1 | 1 | 1 |
| 8 | Gemma-7B | SVAMP | 2.5265 | 2.975 | 1 | 1 | 1 |
| 8 | LLaMA-3.1-8B | AQuA | -4.6693 | -5.4134 | -1 | -1 | 1 |
| 8 | LLaMA-3.1-8B | CommonsenseQA | -0.2518 | -0.4095 | 0 | 0 | 1 |
| 8 | LLaMA-3.1-8B | GPQA | 3.8795 | 4.1853 | 1 | 1 | 1 |
| 8 | LLaMA-3.1-8B | GSM8K | 1.0322 | 1.0235 | 0 | 0 | 1 |
| 8 | LLaMA-3.1-8B | MATH500 | -0.941 | -1.1 | 0 | 0 | 1 |
| 8 | LLaMA-3.1-8B | SVAMP | -0.365 | -0.25 | 0 | 0 | 1 |
| 8 | Phi-4 | AQuA | 0.939 | 1.0335 | 0 | 0 | 1 |
| 8 | Phi-4 | CommonsenseQA | 11.2088 | 9.7563 | 1 | 1 | 1 |
| 8 | Phi-4 | GPQA | -3.029 | -4.1853 | -1 | -1 | 1 |
| 8 | Phi-4 | GSM8K | -5.8715 | -5.9799 | -1 | -1 | 1 |
| 8 | Phi-4 | MATH500 | 3.997 | 4.35 | 1 | 1 | 1 |
| 8 | Phi-4 | SVAMP | -4.721 | -5.0375 | -1 | -1 | 1 |
| 8 | Phi-4-Reas. | AQuA | 3.3957 | 3.7894 | 1 | 1 | 1 |
| 8 | Phi-4-Reas. | CommonsenseQA | 0.921 | 1.1261 | 0 | 0 | 1 |
| 8 | Phi-4-Reas. | GPQA | 5.4308 | 5.6083 | 1 | 1 | 1 |
| 8 | Phi-4-Reas. | GSM8K | 0.7259 | 0.7392 | 0 | 0 | 1 |
| 8 | Phi-4-Reas. | MATH500 | 1.089 | 1.525 | 0 | 0 | 1 |
| 8 | Phi-4-Reas. | SVAMP | -0.3965 | -0.45 | 0 | 0 | 1 |
| 8 | Qwen2.5-7B | AQuA | 1.7815 | 1.1319 | 0 | 0 | 1 |
| 8 | Qwen2.5-7B | CommonsenseQA | 0.593 | 0.5426 | 0 | 0 | 1 |
| 8 | Qwen2.5-7B | GPQA | 1.8862 | 2.1763 | 0 | 1 | 0 |
| 8 | Qwen2.5-7B | GSM8K | -19.757 | -21.4367 | -1 | -1 | 1 |
| 8 | Qwen2.5-7B | MATH500 | -16.242 | -17.475 | -1 | -1 | 1 |
| 8 | Qwen2.5-7B | SVAMP | -19.7965 | -21.4875 | -1 | -1 | 1 |
| 8 | Qwen2.5-Math | AQuA | -3.1339 | -3.3957 | -1 | -1 | 1 |
| 8 | Qwen2.5-Math | CommonsenseQA | 4.0483 | 4.3817 | 1 | 1 | 1 |
| 8 | Qwen2.5-Math | GPQA | 3.1551 | 3.125 | 1 | 1 | 1 |
| 8 | Qwen2.5-Math | GSM8K | -6.6236 | -6.9466 | -1 | -1 | 1 |
| 8 | Qwen2.5-Math | MATH500 | 4.8 | 4.85 | 1 | 1 | 1 |
| 8 | Qwen2.5-Math | SVAMP | -11.265 | -11.3 | -1 | -1 | 1 |
| 8 | Qwen3-4B | AQuA | 3.9114 | 3.937 | 1 | 1 | 1 |
| 8 | Qwen3-4B | CommonsenseQA | -2.6245 | -2.9484 | -1 | -1 | 1 |
| 8 | Qwen3-4B | GPQA | -4.3382 | -4.2969 | -1 | -1 | 1 |
| 8 | Qwen3-4B | GSM8K | 15.5876 | 16.613 | 1 | 1 | 1 |
| 8 | Qwen3-4B | MATH500 | 7.278 | 7.65 | 1 | 1 | 1 |
| 8 | Qwen3-4B | SVAMP | 13.5225 | 14.3375 | 1 | 1 | 1 |
| 12 | DS-R1-1.5B | AQuA | 15.5879 | 15.9449 | 1 | 1 | 1 |
| 12 | DS-R1-1.5B | CommonsenseQA | -0.5389 | -0.5016 | 0 | 0 | 1 |
| 12 | DS-R1-1.5B | GPQA | -6.7768 | -6.9196 | -1 | -1 | 1 |
| 12 | DS-R1-1.5B | GSM8K | 10.9133 | 11.0406 | 1 | 1 | 1 |
| 12 | DS-R1-1.5B | MATH500 | 14.202 | 14.3 | 1 | 1 | 1 |
| 12 | DS-R1-1.5B | SVAMP | 4.9327 | 5.1125 | 1 | 1 | 1 |
| 12 | DS-R1-7B | AQuA | 8.6102 | 8.1201 | 1 | 1 | 1 |
| 12 | DS-R1-7B | CommonsenseQA | -0.2959 | 0.0205 | 0 | 0 | 1 |
| 12 | DS-R1-7B | GPQA | -14.2366 | -14.3694 | -1 | -1 | 1 |
| 12 | DS-R1-7B | GSM8K | 1.2792 | 1.3457 | 0 | 0 | 1 |
| 12 | DS-R1-7B | MATH500 | 6.1587 | 6.125 | 1 | 1 | 1 |
| 12 | DS-R1-7B | SVAMP | 0.4933 | 0.5625 | 0 | 0 | 1 |
| 12 | Gemma-7B | AQuA | 0.0656 | -0.4429 | 0 | 0 | 1 |
| 12 | Gemma-7B | CommonsenseQA | -1.7751 | -1.9861 | 0 | 0 | 1 |
| 12 | Gemma-7B | GPQA | -10.279 | -10.4911 | -1 | -1 | 1 |
| 12 | Gemma-7B | GSM8K | 8.6884 | 8.8798 | 1 | 1 | 1 |
| 12 | Gemma-7B | MATH500 | 2.3 | 2.275 | 1 | 1 | 1 |
| 12 | Gemma-7B | SVAMP | 2.775 | 2.975 | 1 | 1 | 1 |
| 12 | LLaMA-3.1-8B | AQuA | -5.0459 | -5.4134 | -1 | -1 | 1 |
| 12 | LLaMA-3.1-8B | CommonsenseQA | -0.5201 | -0.4095 | 0 | 0 | 1 |
| 12 | LLaMA-3.1-8B | GPQA | 3.9903 | 4.1853 | 1 | 1 | 1 |
| 12 | LLaMA-3.1-8B | GSM8K | 0.8635 | 1.0235 | 0 | 0 | 1 |
| 12 | LLaMA-3.1-8B | MATH500 | -1.1613 | -1.1 | 0 | 0 | 1 |
| 12 | LLaMA-3.1-8B | SVAMP | -0.3607 | -0.25 | 0 | 0 | 1 |
| 12 | Phi-4 | AQuA | 0.7441 | 1.0335 | 0 | 0 | 1 |
| 12 | Phi-4 | CommonsenseQA | 10.3639 | 9.7563 | 1 | 1 | 1 |
| 12 | Phi-4 | GPQA | -3.7507 | -4.1853 | -1 | -1 | 1 |
| 12 | Phi-4 | GSM8K | -6.0015 | -5.9799 | -1 | -1 | 1 |
| 12 | Phi-4 | MATH500 | 4.2147 | 4.35 | 1 | 1 | 1 |
| 12 | Phi-4 | SVAMP | -4.7823 | -5.0375 | -1 | -1 | 1 |
| 12 | Phi-4-Reas. | AQuA | 3.7926 | 3.7894 | 1 | 1 | 1 |
| 12 | Phi-4-Reas. | CommonsenseQA | 0.8665 | 1.1261 | 0 | 0 | 1 |
| 12 | Phi-4-Reas. | GPQA | 5.4695 | 5.6083 | 1 | 1 | 1 |
| 12 | Phi-4-Reas. | GSM8K | 0.6849 | 0.7392 | 0 | 0 | 1 |
| 12 | Phi-4-Reas. | MATH500 | 1.3847 | 1.525 | 0 | 0 | 1 |
| 12 | Phi-4-Reas. | SVAMP | -0.409 | -0.45 | 0 | 0 | 1 |
| 12 | Qwen2.5-7B | AQuA | 1.4974 | 1.1319 | 0 | 0 | 1 |
| 12 | Qwen2.5-7B | CommonsenseQA | 0.635 | 0.5426 | 0 | 0 | 1 |
| 12 | Qwen2.5-7B | GPQA | 1.8713 | 2.1763 | 0 | 1 | 0 |
| 12 | Qwen2.5-7B | GSM8K | -20.8102 | -21.4367 | -1 | -1 | 1 |
| 12 | Qwen2.5-7B | MATH500 | -17 | -17.475 | -1 | -1 | 1 |
| 12 | Qwen2.5-7B | SVAMP | -20.8463 | -21.4875 | -1 | -1 | 1 |
| 12 | Qwen2.5-Math | AQuA | -3.4619 | -3.3957 | -1 | -1 | 1 |
| 12 | Qwen2.5-Math | CommonsenseQA | 4.154 | 4.3817 | 1 | 1 | 1 |
| 12 | Qwen2.5-Math | GPQA | 3.1079 | 3.125 | 1 | 1 | 1 |
| 12 | Qwen2.5-Math | GSM8K | -6.8479 | -6.9466 | -1 | -1 | 1 |
| 12 | Qwen2.5-Math | MATH500 | 4.952 | 4.85 | 1 | 1 | 1 |
| 12 | Qwen2.5-Math | SVAMP | -11.4007 | -11.3 | -1 | -1 | 1 |
| 12 | Qwen3-4B | AQuA | 4.3543 | 3.937 | 1 | 1 | 1 |
| 12 | Qwen3-4B | CommonsenseQA | -2.7691 | -2.9484 | -1 | -1 | 1 |
| 12 | Qwen3-4B | GPQA | -4.372 | -4.2969 | -1 | -1 | 1 |
| 12 | Qwen3-4B | GSM8K | 16.3371 | 16.613 | 1 | 1 | 1 |
| 12 | Qwen3-4B | MATH500 | 7.7327 | 7.65 | 1 | 1 | 1 |
| 12 | Qwen3-4B | SVAMP | 14.0963 | 14.3375 | 1 | 1 | 1 |
| 16 | DS-R1-1.5B | AQuA | 15.9449 | 15.9449 | 1 | 1 | 1 |
| 16 | DS-R1-1.5B | CommonsenseQA | -0.5016 | -0.5016 | 0 | 0 | 1 |
| 16 | DS-R1-1.5B | GPQA | -6.9196 | -6.9196 | -1 | -1 | 1 |
| 16 | DS-R1-1.5B | GSM8K | 11.0406 | 11.0406 | 1 | 1 | 1 |
| 16 | DS-R1-1.5B | MATH500 | 14.3 | 14.3 | 1 | 1 | 1 |
| 16 | DS-R1-1.5B | SVAMP | 5.1125 | 5.1125 | 1 | 1 | 1 |
| 16 | DS-R1-7B | AQuA | 8.1201 | 8.1201 | 1 | 1 | 1 |
| 16 | DS-R1-7B | CommonsenseQA | 0.0205 | 0.0205 | 0 | 0 | 1 |
| 16 | DS-R1-7B | GPQA | -14.3694 | -14.3694 | -1 | -1 | 1 |
| 16 | DS-R1-7B | GSM8K | 1.3457 | 1.3457 | 0 | 0 | 1 |
| 16 | DS-R1-7B | MATH500 | 6.125 | 6.125 | 1 | 1 | 1 |
| 16 | DS-R1-7B | SVAMP | 0.5625 | 0.5625 | 0 | 0 | 1 |
| 16 | Gemma-7B | AQuA | -0.4429 | -0.4429 | 0 | 0 | 1 |
| 16 | Gemma-7B | CommonsenseQA | -1.9861 | -1.9861 | 0 | 0 | 1 |
| 16 | Gemma-7B | GPQA | -10.4911 | -10.4911 | -1 | -1 | 1 |
| 16 | Gemma-7B | GSM8K | 8.8798 | 8.8798 | 1 | 1 | 1 |
| 16 | Gemma-7B | MATH500 | 2.275 | 2.275 | 1 | 1 | 1 |
| 16 | Gemma-7B | SVAMP | 2.975 | 2.975 | 1 | 1 | 1 |
| 16 | LLaMA-3.1-8B | AQuA | -5.4134 | -5.4134 | -1 | -1 | 1 |
| 16 | LLaMA-3.1-8B | CommonsenseQA | -0.4095 | -0.4095 | 0 | 0 | 1 |
| 16 | LLaMA-3.1-8B | GPQA | 4.1853 | 4.1853 | 1 | 1 | 1 |
| 16 | LLaMA-3.1-8B | GSM8K | 1.0235 | 1.0235 | 0 | 0 | 1 |
| 16 | LLaMA-3.1-8B | MATH500 | -1.1 | -1.1 | 0 | 0 | 1 |
| 16 | LLaMA-3.1-8B | SVAMP | -0.25 | -0.25 | 0 | 0 | 1 |
| 16 | Phi-4 | AQuA | 1.0335 | 1.0335 | 0 | 0 | 1 |
| 16 | Phi-4 | CommonsenseQA | 9.7563 | 9.7563 | 1 | 1 | 1 |
| 16 | Phi-4 | GPQA | -4.1853 | -4.1853 | -1 | -1 | 1 |
| 16 | Phi-4 | GSM8K | -5.9799 | -5.9799 | -1 | -1 | 1 |
| 16 | Phi-4 | MATH500 | 4.35 | 4.35 | 1 | 1 | 1 |
| 16 | Phi-4 | SVAMP | -5.0375 | -5.0375 | -1 | -1 | 1 |
| 16 | Phi-4-Reas. | AQuA | 3.7894 | 3.7894 | 1 | 1 | 1 |
| 16 | Phi-4-Reas. | CommonsenseQA | 1.1261 | 1.1261 | 0 | 0 | 1 |
| 16 | Phi-4-Reas. | GPQA | 5.6083 | 5.6083 | 1 | 1 | 1 |
| 16 | Phi-4-Reas. | GSM8K | 0.7392 | 0.7392 | 0 | 0 | 1 |
| 16 | Phi-4-Reas. | MATH500 | 1.525 | 1.525 | 0 | 0 | 1 |
| 16 | Phi-4-Reas. | SVAMP | -0.45 | -0.45 | 0 | 0 | 1 |
| 16 | Qwen2.5-7B | AQuA | 1.1319 | 1.1319 | 0 | 0 | 1 |
| 16 | Qwen2.5-7B | CommonsenseQA | 0.5426 | 0.5426 | 0 | 0 | 1 |
| 16 | Qwen2.5-7B | GPQA | 2.1763 | 2.1763 | 1 | 1 | 1 |
| 16 | Qwen2.5-7B | GSM8K | -21.4367 | -21.4367 | -1 | -1 | 1 |
| 16 | Qwen2.5-7B | MATH500 | -17.475 | -17.475 | -1 | -1 | 1 |
| 16 | Qwen2.5-7B | SVAMP | -21.4875 | -21.4875 | -1 | -1 | 1 |
| 16 | Qwen2.5-Math | AQuA | -3.3957 | -3.3957 | -1 | -1 | 1 |
| 16 | Qwen2.5-Math | CommonsenseQA | 4.3817 | 4.3817 | 1 | 1 | 1 |
| 16 | Qwen2.5-Math | GPQA | 3.125 | 3.125 | 1 | 1 | 1 |
| 16 | Qwen2.5-Math | GSM8K | -6.9466 | -6.9466 | -1 | -1 | 1 |
| 16 | Qwen2.5-Math | MATH500 | 4.85 | 4.85 | 1 | 1 | 1 |
| 16 | Qwen2.5-Math | SVAMP | -11.3 | -11.3 | -1 | -1 | 1 |
| 16 | Qwen3-4B | AQuA | 3.937 | 3.937 | 1 | 1 | 1 |
| 16 | Qwen3-4B | CommonsenseQA | -2.9484 | -2.9484 | -1 | -1 | 1 |
| 16 | Qwen3-4B | GPQA | -4.2969 | -4.2969 | -1 | -1 | 1 |
| 16 | Qwen3-4B | GSM8K | 16.613 | 16.613 | 1 | 1 | 1 |
| 16 | Qwen3-4B | MATH500 | 7.65 | 7.65 | 1 | 1 | 1 |
| 16 | Qwen3-4B | SVAMP | 14.3375 | 14.3375 | 1 | 1 | 1 |

### Per-bootstrap detail (full log) — `sample_count_ablation_results/ablation_per_bootstrap_detail.csv`

| bootstrap_idx | seed | model | dataset | k_sub | frs_accuracy_proxy_pct | acc_unconfident_half_pct | gap_pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 42 | DS-R1-1.5B | AQuA | 4 | 48.4252 | 37.2047 | 11.2205 |
| 1 | 43 | DS-R1-1.5B | AQuA | 4 | 52.1654 | 35.2362 | 16.9291 |
| 2 | 44 | DS-R1-1.5B | AQuA | 4 | 49.8031 | 36.0236 | 13.7795 |
| 3 | 45 | DS-R1-1.5B | AQuA | 4 | 48.8189 | 35.8268 | 12.9921 |
| 4 | 46 | DS-R1-1.5B | AQuA | 4 | 50.1969 | 33.4646 | 16.7323 |
| 5 | 47 | DS-R1-1.5B | AQuA | 4 | 48.0315 | 35.6299 | 12.4016 |
| 6 | 48 | DS-R1-1.5B | AQuA | 4 | 53.937 | 32.0866 | 21.8504 |
| 7 | 49 | DS-R1-1.5B | AQuA | 4 | 50.7874 | 35.8268 | 14.9606 |
| 8 | 50 | DS-R1-1.5B | AQuA | 4 | 51.1811 | 38.7795 | 12.4016 |
| 9 | 51 | DS-R1-1.5B | AQuA | 4 | 50 | 35.8268 | 14.1732 |
| 10 | 52 | DS-R1-1.5B | AQuA | 4 | 51.5748 | 36.2205 | 15.3543 |
| 11 | 53 | DS-R1-1.5B | AQuA | 4 | 50.7874 | 35.6299 | 15.1575 |
| 12 | 54 | DS-R1-1.5B | AQuA | 4 | 50 | 35.2362 | 14.7638 |
| 13 | 55 | DS-R1-1.5B | AQuA | 4 | 47.8346 | 34.8425 | 12.9921 |
| 14 | 56 | DS-R1-1.5B | AQuA | 4 | 51.1811 | 38.189 | 12.9921 |
| 15 | 57 | DS-R1-1.5B | AQuA | 4 | 51.1811 | 33.8583 | 17.3228 |
| 16 | 58 | DS-R1-1.5B | AQuA | 4 | 50.5906 | 34.6457 | 15.9449 |
| 17 | 59 | DS-R1-1.5B | AQuA | 4 | 51.1811 | 35.2362 | 15.9449 |
| 18 | 60 | DS-R1-1.5B | AQuA | 4 | 47.2441 | 34.6457 | 12.5984 |
| 19 | 61 | DS-R1-1.5B | AQuA | 4 | 49.6063 | 38.7795 | 10.8268 |
| 20 | 62 | DS-R1-1.5B | AQuA | 4 | 51.7717 | 32.4803 | 19.2913 |
| 21 | 63 | DS-R1-1.5B | AQuA | 4 | 48.8189 | 32.874 | 15.9449 |
| 22 | 64 | DS-R1-1.5B | AQuA | 4 | 48.4252 | 33.6614 | 14.7638 |
| 23 | 65 | DS-R1-1.5B | AQuA | 4 | 47.6378 | 37.0079 | 10.6299 |
| 24 | 66 | DS-R1-1.5B | AQuA | 4 | 50.9843 | 34.6457 | 16.3386 |
| 25 | 67 | DS-R1-1.5B | AQuA | 4 | 50.1969 | 39.3701 | 10.8268 |
| 26 | 68 | DS-R1-1.5B | AQuA | 4 | 48.622 | 34.4488 | 14.1732 |
| 27 | 69 | DS-R1-1.5B | AQuA | 4 | 50.9843 | 34.6457 | 16.3386 |
| 28 | 70 | DS-R1-1.5B | AQuA | 4 | 51.9685 | 35.8268 | 16.1417 |
| 29 | 71 | DS-R1-1.5B | AQuA | 4 | 49.2126 | 36.6142 | 12.5984 |
| 30 | 72 | DS-R1-1.5B | AQuA | 4 | 49.0157 | 36.811 | 12.2047 |
| 31 | 73 | DS-R1-1.5B | AQuA | 4 | 50 | 36.811 | 13.189 |
| 32 | 74 | DS-R1-1.5B | AQuA | 4 | 51.7717 | 35.0394 | 16.7323 |
| 33 | 75 | DS-R1-1.5B | AQuA | 4 | 50.1969 | 36.6142 | 13.5827 |
| 34 | 76 | DS-R1-1.5B | AQuA | 4 | 48.0315 | 36.811 | 11.2205 |
| 35 | 77 | DS-R1-1.5B | AQuA | 4 | 48.4252 | 37.4016 | 11.0236 |
| 36 | 78 | DS-R1-1.5B | AQuA | 4 | 51.5748 | 38.7795 | 12.7953 |
| 37 | 79 | DS-R1-1.5B | AQuA | 4 | 50.7874 | 38.9764 | 11.811 |
| 38 | 80 | DS-R1-1.5B | AQuA | 4 | 47.6378 | 37.0079 | 10.6299 |
| 39 | 81 | DS-R1-1.5B | AQuA | 4 | 50.7874 | 35.0394 | 15.748 |
| 40 | 82 | DS-R1-1.5B | AQuA | 4 | 51.5748 | 34.252 | 17.3228 |
| 41 | 83 | DS-R1-1.5B | AQuA | 4 | 51.5748 | 37.2047 | 14.3701 |
| 42 | 84 | DS-R1-1.5B | AQuA | 4 | 50.3937 | 31.4961 | 18.8976 |
| 43 | 85 | DS-R1-1.5B | AQuA | 4 | 50.7874 | 37.0079 | 13.7795 |
| 44 | 86 | DS-R1-1.5B | AQuA | 4 | 50.9843 | 36.0236 | 14.9606 |
| 45 | 87 | DS-R1-1.5B | AQuA | 4 | 50 | 36.6142 | 13.3858 |
| 46 | 88 | DS-R1-1.5B | AQuA | 4 | 49.2126 | 37.9921 | 11.2205 |
| 47 | 89 | DS-R1-1.5B | AQuA | 4 | 48.8189 | 35.2362 | 13.5827 |
| 48 | 90 | DS-R1-1.5B | AQuA | 4 | 48.4252 | 36.0236 | 12.4016 |
| 49 | 91 | DS-R1-1.5B | AQuA | 4 | 47.0472 | 37.5984 | 9.4488 |
| 0 | 42 | DS-R1-1.5B | AQuA | 8 | 49.5079 | 36.811 | 12.6969 |
| 1 | 43 | DS-R1-1.5B | AQuA | 8 | 51.1811 | 34.8425 | 16.3386 |
| 2 | 44 | DS-R1-1.5B | AQuA | 8 | 50 | 36.6142 | 13.3858 |
| 3 | 45 | DS-R1-1.5B | AQuA | 8 | 50 | 34.8425 | 15.1575 |
| 4 | 46 | DS-R1-1.5B | AQuA | 8 | 50.0984 | 35.2362 | 14.8622 |
| 5 | 47 | DS-R1-1.5B | AQuA | 8 | 50.0984 | 35.6299 | 14.4685 |
| 6 | 48 | DS-R1-1.5B | AQuA | 8 | 52.0669 | 35.6299 | 16.437 |
| 7 | 49 | DS-R1-1.5B | AQuA | 8 | 51.0827 | 36.122 | 14.9606 |
| 8 | 50 | DS-R1-1.5B | AQuA | 8 | 52.6575 | 36.0236 | 16.6339 |
| 9 | 51 | DS-R1-1.5B | AQuA | 8 | 51.6732 | 35.6299 | 16.0433 |
| 10 | 52 | DS-R1-1.5B | AQuA | 8 | 50.0984 | 35.1378 | 14.9606 |
| 11 | 53 | DS-R1-1.5B | AQuA | 8 | 50 | 35.7283 | 14.2717 |
| 12 | 54 | DS-R1-1.5B | AQuA | 8 | 51.1811 | 34.0551 | 17.126 |
| 13 | 55 | DS-R1-1.5B | AQuA | 8 | 49.9016 | 34.9409 | 14.9606 |
| 14 | 56 | DS-R1-1.5B | AQuA | 8 | 49.1142 | 36.4173 | 12.6969 |
| 15 | 57 | DS-R1-1.5B | AQuA | 8 | 51.1811 | 35.5315 | 15.6496 |
| 16 | 58 | DS-R1-1.5B | AQuA | 8 | 49.7047 | 34.4488 | 15.2559 |
| 17 | 59 | DS-R1-1.5B | AQuA | 8 | 51.8701 | 33.6614 | 18.2087 |
| 18 | 60 | DS-R1-1.5B | AQuA | 8 | 49.2126 | 35.5315 | 13.6811 |
| 19 | 61 | DS-R1-1.5B | AQuA | 8 | 48.7205 | 38.4843 | 10.2362 |
| 20 | 62 | DS-R1-1.5B | AQuA | 8 | 51.378 | 33.8583 | 17.5197 |
| 21 | 63 | DS-R1-1.5B | AQuA | 8 | 51.0827 | 35.1378 | 15.9449 |
| 22 | 64 | DS-R1-1.5B | AQuA | 8 | 50.8858 | 34.8425 | 16.0433 |
| 23 | 65 | DS-R1-1.5B | AQuA | 8 | 50.1969 | 36.6142 | 13.5827 |
| 24 | 66 | DS-R1-1.5B | AQuA | 8 | 50.2953 | 34.9409 | 15.3543 |
| 25 | 67 | DS-R1-1.5B | AQuA | 8 | 49.1142 | 34.7441 | 14.3701 |
| 26 | 68 | DS-R1-1.5B | AQuA | 8 | 49.7047 | 34.9409 | 14.7638 |
| 27 | 69 | DS-R1-1.5B | AQuA | 8 | 50.7874 | 34.252 | 16.5354 |
| 28 | 70 | DS-R1-1.5B | AQuA | 8 | 51.1811 | 36.2205 | 14.9606 |
| 29 | 71 | DS-R1-1.5B | AQuA | 8 | 50.1969 | 35.8268 | 14.3701 |
| 30 | 72 | DS-R1-1.5B | AQuA | 8 | 50.1969 | 35.7283 | 14.4685 |
| 31 | 73 | DS-R1-1.5B | AQuA | 8 | 51.2795 | 35.5315 | 15.748 |
| 32 | 74 | DS-R1-1.5B | AQuA | 8 | 51.0827 | 36.122 | 14.9606 |
| 33 | 75 | DS-R1-1.5B | AQuA | 8 | 51.1811 | 34.7441 | 16.437 |
| 34 | 76 | DS-R1-1.5B | AQuA | 8 | 48.622 | 35.9252 | 12.6969 |
| 35 | 77 | DS-R1-1.5B | AQuA | 8 | 50.1969 | 35.1378 | 15.0591 |
| 36 | 78 | DS-R1-1.5B | AQuA | 8 | 52.9528 | 36.6142 | 16.3386 |
| 37 | 79 | DS-R1-1.5B | AQuA | 8 | 49.5079 | 36.0236 | 13.4843 |
| 38 | 80 | DS-R1-1.5B | AQuA | 8 | 50.3937 | 35.3346 | 15.0591 |
| 39 | 81 | DS-R1-1.5B | AQuA | 8 | 49.7047 | 36.2205 | 13.4843 |
| 40 | 82 | DS-R1-1.5B | AQuA | 8 | 51.5748 | 34.6457 | 16.9291 |
| 41 | 83 | DS-R1-1.5B | AQuA | 8 | 51.5748 | 35.6299 | 15.9449 |
| 42 | 84 | DS-R1-1.5B | AQuA | 8 | 50.0984 | 33.6614 | 16.437 |
| 43 | 85 | DS-R1-1.5B | AQuA | 8 | 51.2795 | 36.2205 | 15.0591 |
| 44 | 86 | DS-R1-1.5B | AQuA | 8 | 50.7874 | 34.3504 | 16.437 |
| 45 | 87 | DS-R1-1.5B | AQuA | 8 | 48.4252 | 34.6457 | 13.7795 |
| 46 | 88 | DS-R1-1.5B | AQuA | 8 | 50.4921 | 36.0236 | 14.4685 |
| 47 | 89 | DS-R1-1.5B | AQuA | 8 | 50.0984 | 35.3346 | 14.7638 |
| 48 | 90 | DS-R1-1.5B | AQuA | 8 | 50.8858 | 34.3504 | 16.5354 |
| 49 | 91 | DS-R1-1.5B | AQuA | 8 | 48.4252 | 35.9252 | 12.5 |
| 0 | 42 | DS-R1-1.5B | AQuA | 12 | 50.7218 | 35.5643 | 15.1575 |
| 1 | 43 | DS-R1-1.5B | AQuA | 12 | 51.3123 | 34.6457 | 16.6667 |
| 2 | 44 | DS-R1-1.5B | AQuA | 12 | 49.8688 | 35.5643 | 14.3045 |
| 3 | 45 | DS-R1-1.5B | AQuA | 12 | 50.5249 | 34.9081 | 15.6168 |
| 4 | 46 | DS-R1-1.5B | AQuA | 12 | 50 | 35.3675 | 14.6325 |
| 5 | 47 | DS-R1-1.5B | AQuA | 12 | 50.5906 | 34.7113 | 15.8793 |
| 6 | 48 | DS-R1-1.5B | AQuA | 12 | 50.7874 | 35.5643 | 15.2231 |
| 7 | 49 | DS-R1-1.5B | AQuA | 12 | 51.1155 | 35.7612 | 15.3543 |
| 8 | 50 | DS-R1-1.5B | AQuA | 12 | 51.0499 | 35.6955 | 15.3543 |
| 9 | 51 | DS-R1-1.5B | AQuA | 12 | 51.0499 | 34.6457 | 16.4042 |
| 10 | 52 | DS-R1-1.5B | AQuA | 12 | 50.853 | 34.9081 | 15.9449 |
| 11 | 53 | DS-R1-1.5B | AQuA | 12 | 50.3937 | 34.9738 | 15.4199 |
| 12 | 54 | DS-R1-1.5B | AQuA | 12 | 50.5906 | 34.4488 | 16.1417 |
| 13 | 55 | DS-R1-1.5B | AQuA | 12 | 50.9186 | 35.2362 | 15.6824 |
| 14 | 56 | DS-R1-1.5B | AQuA | 12 | 50.853 | 34.7769 | 16.0761 |
| 15 | 57 | DS-R1-1.5B | AQuA | 12 | 50.5906 | 35.4331 | 15.1575 |
| 16 | 58 | DS-R1-1.5B | AQuA | 12 | 50.3281 | 34.8425 | 15.4856 |
| 17 | 59 | DS-R1-1.5B | AQuA | 12 | 51.6404 | 34.3832 | 17.2572 |
| 18 | 60 | DS-R1-1.5B | AQuA | 12 | 50.1312 | 35.3675 | 14.7638 |
| 19 | 61 | DS-R1-1.5B | AQuA | 12 | 50.4593 | 35.5643 | 14.895 |
| 20 | 62 | DS-R1-1.5B | AQuA | 12 | 51.1155 | 34.3176 | 16.7979 |
| 21 | 63 | DS-R1-1.5B | AQuA | 12 | 51.1155 | 34.7769 | 16.3386 |
| 22 | 64 | DS-R1-1.5B | AQuA | 12 | 50.7218 | 34.3176 | 16.4042 |
| 23 | 65 | DS-R1-1.5B | AQuA | 12 | 50.2625 | 36.2205 | 14.042 |
| 24 | 66 | DS-R1-1.5B | AQuA | 12 | 51.1155 | 34.7113 | 16.4042 |
| 25 | 67 | DS-R1-1.5B | AQuA | 12 | 50.1312 | 34.3832 | 15.748 |
| 26 | 68 | DS-R1-1.5B | AQuA | 12 | 50.1312 | 34.3176 | 15.8136 |
| 27 | 69 | DS-R1-1.5B | AQuA | 12 | 50.6562 | 35.2362 | 15.4199 |
| 28 | 70 | DS-R1-1.5B | AQuA | 12 | 50.7874 | 35.7612 | 15.0262 |
| 29 | 71 | DS-R1-1.5B | AQuA | 12 | 50.9843 | 34.1207 | 16.8635 |
| 30 | 72 | DS-R1-1.5B | AQuA | 12 | 50.5906 | 34.6457 | 15.9449 |
| 31 | 73 | DS-R1-1.5B | AQuA | 12 | 50.6562 | 35.5643 | 15.0919 |
| 32 | 74 | DS-R1-1.5B | AQuA | 12 | 51.6404 | 35.0394 | 16.601 |
| 33 | 75 | DS-R1-1.5B | AQuA | 12 | 50.7874 | 34.3176 | 16.4698 |
| 34 | 76 | DS-R1-1.5B | AQuA | 12 | 49.6063 | 35.4987 | 14.1076 |
| 35 | 77 | DS-R1-1.5B | AQuA | 12 | 50.7218 | 35.1706 | 15.5512 |
| 36 | 78 | DS-R1-1.5B | AQuA | 12 | 52.0997 | 35.5643 | 16.5354 |
| 37 | 79 | DS-R1-1.5B | AQuA | 12 | 49.8031 | 35.105 | 14.6982 |
| 38 | 80 | DS-R1-1.5B | AQuA | 12 | 50.7218 | 35.4331 | 15.2887 |
| 39 | 81 | DS-R1-1.5B | AQuA | 12 | 49.6063 | 35.6955 | 13.9108 |
| 40 | 82 | DS-R1-1.5B | AQuA | 12 | 50.3281 | 35.958 | 14.3701 |
| 41 | 83 | DS-R1-1.5B | AQuA | 12 | 51.378 | 35.1706 | 16.2073 |
| 42 | 84 | DS-R1-1.5B | AQuA | 12 | 51.3123 | 34.0551 | 17.2572 |
| 43 | 85 | DS-R1-1.5B | AQuA | 12 | 51.6404 | 35.5643 | 16.0761 |
| 44 | 86 | DS-R1-1.5B | AQuA | 12 | 50.5906 | 35.105 | 15.4856 |
| 45 | 87 | DS-R1-1.5B | AQuA | 12 | 51.0499 | 34.9081 | 16.1417 |
| 46 | 88 | DS-R1-1.5B | AQuA | 12 | 49.5407 | 35.1706 | 14.3701 |
| 47 | 89 | DS-R1-1.5B | AQuA | 12 | 50.7874 | 35.4987 | 15.2887 |
| 48 | 90 | DS-R1-1.5B | AQuA | 12 | 50.5249 | 35.4331 | 15.0919 |
| 49 | 91 | DS-R1-1.5B | AQuA | 12 | 49.3438 | 34.7113 | 14.6325 |
| 0 | 42 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 1 | 43 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 2 | 44 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 3 | 45 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 4 | 46 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 5 | 47 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 6 | 48 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 7 | 49 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 8 | 50 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 9 | 51 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 10 | 52 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 11 | 53 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 12 | 54 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 13 | 55 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 14 | 56 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 15 | 57 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 16 | 58 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 17 | 59 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 18 | 60 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 19 | 61 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 20 | 62 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 21 | 63 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 22 | 64 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 23 | 65 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 24 | 66 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 25 | 67 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 26 | 68 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 27 | 69 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 28 | 70 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 29 | 71 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 30 | 72 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 31 | 73 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 32 | 74 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 33 | 75 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 34 | 76 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 35 | 77 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 36 | 78 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 37 | 79 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 38 | 80 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 39 | 81 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 40 | 82 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 41 | 83 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 42 | 84 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 43 | 85 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 44 | 86 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 45 | 87 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 46 | 88 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 47 | 89 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 48 | 90 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 49 | 91 | DS-R1-1.5B | AQuA | 16 | 50.8858 | 34.9409 | 15.9449 |
| 0 | 42 | DS-R1-1.5B | CommonsenseQA | 4 | 40.7453 | 40.172 | 0.5733 |
| 1 | 43 | DS-R1-1.5B | CommonsenseQA | 4 | 40.3358 | 42.0147 | -1.679 |
| 2 | 44 | DS-R1-1.5B | CommonsenseQA | 4 | 39.6396 | 40.6634 | -1.0238 |
| 3 | 45 | DS-R1-1.5B | CommonsenseQA | 4 | 40.6634 | 40.6634 | 0 |
| 4 | 46 | DS-R1-1.5B | CommonsenseQA | 4 | 40.0082 | 41.2367 | -1.2285 |
| 5 | 47 | DS-R1-1.5B | CommonsenseQA | 4 | 40.4996 | 41.5643 | -1.0647 |
| 6 | 48 | DS-R1-1.5B | CommonsenseQA | 4 | 40.4177 | 41.6462 | -1.2285 |
| 7 | 49 | DS-R1-1.5B | CommonsenseQA | 4 | 40.2539 | 40.7043 | -0.4505 |
| 8 | 50 | DS-R1-1.5B | CommonsenseQA | 4 | 41.0729 | 41.4005 | -0.3276 |
| 9 | 51 | DS-R1-1.5B | CommonsenseQA | 4 | 41.0729 | 41.4824 | -0.4095 |
| 10 | 52 | DS-R1-1.5B | CommonsenseQA | 4 | 41.6871 | 41.0729 | 0.6143 |
| 11 | 53 | DS-R1-1.5B | CommonsenseQA | 4 | 39.8034 | 40.8681 | -1.0647 |
| 12 | 54 | DS-R1-1.5B | CommonsenseQA | 4 | 40.4586 | 41.3595 | -0.9009 |
| 13 | 55 | DS-R1-1.5B | CommonsenseQA | 4 | 40.5815 | 40.2948 | 0.2867 |
| 14 | 56 | DS-R1-1.5B | CommonsenseQA | 4 | 40.7453 | 40.8272 | -0.0819 |
| 15 | 57 | DS-R1-1.5B | CommonsenseQA | 4 | 41.5643 | 41.0319 | 0.5324 |
| 16 | 58 | DS-R1-1.5B | CommonsenseQA | 4 | 39.8034 | 40.5815 | -0.7781 |
| 17 | 59 | DS-R1-1.5B | CommonsenseQA | 4 | 39.5577 | 41.5233 | -1.9656 |
| 18 | 60 | DS-R1-1.5B | CommonsenseQA | 4 | 41.8509 | 42.3833 | -0.5324 |
| 19 | 61 | DS-R1-1.5B | CommonsenseQA | 4 | 39.9263 | 39.8853 | 0.041 |
| 20 | 62 | DS-R1-1.5B | CommonsenseQA | 4 | 39.8853 | 41.769 | -1.8837 |
| 21 | 63 | DS-R1-1.5B | CommonsenseQA | 4 | 41.1548 | 40.95 | 0.2048 |
| 22 | 64 | DS-R1-1.5B | CommonsenseQA | 4 | 40.172 | 41.1138 | -0.9419 |
| 23 | 65 | DS-R1-1.5B | CommonsenseQA | 4 | 40.4586 | 41.1548 | -0.6962 |
| 24 | 66 | DS-R1-1.5B | CommonsenseQA | 4 | 40.2948 | 39.3939 | 0.9009 |
| 25 | 67 | DS-R1-1.5B | CommonsenseQA | 4 | 40.7862 | 41.0729 | -0.2867 |
| 26 | 68 | DS-R1-1.5B | CommonsenseQA | 4 | 40.7043 | 41.4824 | -0.7781 |
| 27 | 69 | DS-R1-1.5B | CommonsenseQA | 4 | 40.0082 | 42.4242 | -2.4161 |
| 28 | 70 | DS-R1-1.5B | CommonsenseQA | 4 | 40.4996 | 41.0729 | -0.5733 |
| 29 | 71 | DS-R1-1.5B | CommonsenseQA | 4 | 39.353 | 39.7625 | -0.4095 |
| 30 | 72 | DS-R1-1.5B | CommonsenseQA | 4 | 39.9263 | 41.3186 | -1.3923 |
| 31 | 73 | DS-R1-1.5B | CommonsenseQA | 4 | 40.3767 | 41.1957 | -0.819 |
| 32 | 74 | DS-R1-1.5B | CommonsenseQA | 4 | 41.4824 | 40.3767 | 1.1057 |
| 33 | 75 | DS-R1-1.5B | CommonsenseQA | 4 | 39.5987 | 40.6634 | -1.0647 |
| 34 | 76 | DS-R1-1.5B | CommonsenseQA | 4 | 41.1957 | 39.353 | 1.8428 |
| 35 | 77 | DS-R1-1.5B | CommonsenseQA | 4 | 41.0319 | 41.3595 | -0.3276 |
| 36 | 78 | DS-R1-1.5B | CommonsenseQA | 4 | 39.9672 | 40.5815 | -0.6143 |
| 37 | 79 | DS-R1-1.5B | CommonsenseQA | 4 | 40.4177 | 42.2604 | -1.8428 |
| 38 | 80 | DS-R1-1.5B | CommonsenseQA | 4 | 40.3358 | 40.3358 | 0 |
| 39 | 81 | DS-R1-1.5B | CommonsenseQA | 4 | 41.0729 | 40.991 | 0.0819 |
| 40 | 82 | DS-R1-1.5B | CommonsenseQA | 4 | 40.172 | 42.0966 | -1.9247 |
| 41 | 83 | DS-R1-1.5B | CommonsenseQA | 4 | 40.9091 | 41.5643 | -0.6552 |
| 42 | 84 | DS-R1-1.5B | CommonsenseQA | 4 | 39.7625 | 42.3833 | -2.6208 |
| 43 | 85 | DS-R1-1.5B | CommonsenseQA | 4 | 40.95 | 41.2776 | -0.3276 |
| 44 | 86 | DS-R1-1.5B | CommonsenseQA | 4 | 40.2129 | 41.9738 | -1.7609 |
| 45 | 87 | DS-R1-1.5B | CommonsenseQA | 4 | 40.5815 | 40.3767 | 0.2048 |
| 46 | 88 | DS-R1-1.5B | CommonsenseQA | 4 | 40.5815 | 39.5577 | 1.0238 |
| 47 | 89 | DS-R1-1.5B | CommonsenseQA | 4 | 40.4177 | 41.1957 | -0.7781 |
| 48 | 90 | DS-R1-1.5B | CommonsenseQA | 4 | 41.0319 | 41.1548 | -0.1229 |
| 49 | 91 | DS-R1-1.5B | CommonsenseQA | 4 | 40.5815 | 41.8509 | -1.2695 |
| 0 | 42 | DS-R1-1.5B | CommonsenseQA | 8 | 41.4414 | 40.8272 | 0.6143 |
| 1 | 43 | DS-R1-1.5B | CommonsenseQA | 8 | 40.6429 | 41.38 | -0.7371 |
| 2 | 44 | DS-R1-1.5B | CommonsenseQA | 8 | 40.3767 | 40.7043 | -0.3276 |
| 3 | 45 | DS-R1-1.5B | CommonsenseQA | 8 | 39.8239 | 41.8714 | -2.0475 |
| 4 | 46 | DS-R1-1.5B | CommonsenseQA | 8 | 40.3767 | 40.0082 | 0.3686 |
| 5 | 47 | DS-R1-1.5B | CommonsenseQA | 8 | 40.4177 | 41.0319 | -0.6143 |
| 6 | 48 | DS-R1-1.5B | CommonsenseQA | 8 | 39.7625 | 41.769 | -2.0066 |
| 7 | 49 | DS-R1-1.5B | CommonsenseQA | 8 | 40.5201 | 41.4619 | -0.9419 |
| 8 | 50 | DS-R1-1.5B | CommonsenseQA | 8 | 41.1138 | 41.1753 | -0.0614 |
| 9 | 51 | DS-R1-1.5B | CommonsenseQA | 8 | 40.2744 | 40.9705 | -0.6962 |
| 10 | 52 | DS-R1-1.5B | CommonsenseQA | 8 | 40.7658 | 41.4824 | -0.7166 |
| 11 | 53 | DS-R1-1.5B | CommonsenseQA | 8 | 39.1892 | 41.3186 | -2.1294 |
| 12 | 54 | DS-R1-1.5B | CommonsenseQA | 8 | 40.0082 | 41.0729 | -1.0647 |
| 13 | 55 | DS-R1-1.5B | CommonsenseQA | 8 | 40.7043 | 39.9468 | 0.7576 |
| 14 | 56 | DS-R1-1.5B | CommonsenseQA | 8 | 39.6192 | 41.5233 | -1.9042 |
| 15 | 57 | DS-R1-1.5B | CommonsenseQA | 8 | 40.95 | 40.9296 | 0.0205 |
| 16 | 58 | DS-R1-1.5B | CommonsenseQA | 8 | 40.5405 | 41.6871 | -1.1466 |
| 17 | 59 | DS-R1-1.5B | CommonsenseQA | 8 | 40.2948 | 40.8272 | -0.5324 |
| 18 | 60 | DS-R1-1.5B | CommonsenseQA | 8 | 41.0319 | 40.8067 | 0.2252 |
| 19 | 61 | DS-R1-1.5B | CommonsenseQA | 8 | 41.0524 | 40.4791 | 0.5733 |
| 20 | 62 | DS-R1-1.5B | CommonsenseQA | 8 | 39.6192 | 40.9705 | -1.3514 |
| 21 | 63 | DS-R1-1.5B | CommonsenseQA | 8 | 40.0901 | 41.6871 | -1.5971 |
| 22 | 64 | DS-R1-1.5B | CommonsenseQA | 8 | 40.6839 | 40.7862 | -0.1024 |
| 23 | 65 | DS-R1-1.5B | CommonsenseQA | 8 | 40.3972 | 40.3153 | 0.0819 |
| 24 | 66 | DS-R1-1.5B | CommonsenseQA | 8 | 40.5201 | 40.4586 | 0.0614 |
| 25 | 67 | DS-R1-1.5B | CommonsenseQA | 8 | 40.2948 | 41.6667 | -1.3718 |
| 26 | 68 | DS-R1-1.5B | CommonsenseQA | 8 | 40.602 | 40.9091 | -0.3071 |
| 27 | 69 | DS-R1-1.5B | CommonsenseQA | 8 | 40.3153 | 40.602 | -0.2867 |
| 28 | 70 | DS-R1-1.5B | CommonsenseQA | 8 | 40.9091 | 40.9296 | -0.0205 |
| 29 | 71 | DS-R1-1.5B | CommonsenseQA | 8 | 39.5577 | 40.5405 | -0.9828 |
| 30 | 72 | DS-R1-1.5B | CommonsenseQA | 8 | 40.0082 | 41.0319 | -1.0238 |
| 31 | 73 | DS-R1-1.5B | CommonsenseQA | 8 | 40.2744 | 40.6839 | -0.4095 |
| 32 | 74 | DS-R1-1.5B | CommonsenseQA | 8 | 40.3153 | 40.4996 | -0.1843 |
| 33 | 75 | DS-R1-1.5B | CommonsenseQA | 8 | 40.5405 | 40.7248 | -0.1843 |
| 34 | 76 | DS-R1-1.5B | CommonsenseQA | 8 | 40.6429 | 40.8477 | -0.2048 |
| 35 | 77 | DS-R1-1.5B | CommonsenseQA | 8 | 40.131 | 40.7862 | -0.6552 |
| 36 | 78 | DS-R1-1.5B | CommonsenseQA | 8 | 40.8067 | 40.6634 | 0.1433 |
| 37 | 79 | DS-R1-1.5B | CommonsenseQA | 8 | 40.3358 | 41.7486 | -1.4128 |
| 38 | 80 | DS-R1-1.5B | CommonsenseQA | 8 | 40.0082 | 41.1138 | -1.1057 |
| 39 | 81 | DS-R1-1.5B | CommonsenseQA | 8 | 41.1548 | 40.991 | 0.1638 |
| 40 | 82 | DS-R1-1.5B | CommonsenseQA | 8 | 39.9468 | 41.7076 | -1.7609 |
| 41 | 83 | DS-R1-1.5B | CommonsenseQA | 8 | 40.6224 | 41.421 | -0.7985 |
| 42 | 84 | DS-R1-1.5B | CommonsenseQA | 8 | 40.3767 | 41.5233 | -1.1466 |
| 43 | 85 | DS-R1-1.5B | CommonsenseQA | 8 | 40.3153 | 41.0524 | -0.7371 |
| 44 | 86 | DS-R1-1.5B | CommonsenseQA | 8 | 40.7453 | 41.1753 | -0.43 |
| 45 | 87 | DS-R1-1.5B | CommonsenseQA | 8 | 40.3767 | 41.0319 | -0.6552 |
| 46 | 88 | DS-R1-1.5B | CommonsenseQA | 8 | 40.4996 | 40.6429 | -0.1433 |
| 47 | 89 | DS-R1-1.5B | CommonsenseQA | 8 | 40.4996 | 40.991 | -0.4914 |
| 48 | 90 | DS-R1-1.5B | CommonsenseQA | 8 | 40.8681 | 40.95 | -0.0819 |
| 49 | 91 | DS-R1-1.5B | CommonsenseQA | 8 | 41.0115 | 40.7043 | 0.3071 |
| 0 | 42 | DS-R1-1.5B | CommonsenseQA | 12 | 40.7453 | 40.8954 | -0.1502 |
| 1 | 43 | DS-R1-1.5B | CommonsenseQA | 12 | 40.2948 | 40.991 | -0.6962 |
| 2 | 44 | DS-R1-1.5B | CommonsenseQA | 12 | 40.1856 | 41.1548 | -0.9692 |
| 3 | 45 | DS-R1-1.5B | CommonsenseQA | 12 | 40.3221 | 41.6462 | -1.3241 |
| 4 | 46 | DS-R1-1.5B | CommonsenseQA | 12 | 40.3494 | 40.6224 | -0.273 |
| 5 | 47 | DS-R1-1.5B | CommonsenseQA | 12 | 40.5815 | 40.6088 | -0.0273 |
| 6 | 48 | DS-R1-1.5B | CommonsenseQA | 12 | 40.1037 | 41.0729 | -0.9692 |
| 7 | 49 | DS-R1-1.5B | CommonsenseQA | 12 | 40.3358 | 41.1548 | -0.819 |
| 8 | 50 | DS-R1-1.5B | CommonsenseQA | 12 | 40.7589 | 41.1957 | -0.4368 |
| 9 | 51 | DS-R1-1.5B | CommonsenseQA | 12 | 40.4859 | 41.1138 | -0.6279 |
| 10 | 52 | DS-R1-1.5B | CommonsenseQA | 12 | 40.2948 | 40.95 | -0.6552 |
| 11 | 53 | DS-R1-1.5B | CommonsenseQA | 12 | 39.9809 | 41.1684 | -1.1876 |
| 12 | 54 | DS-R1-1.5B | CommonsenseQA | 12 | 40.1447 | 40.9773 | -0.8327 |
| 13 | 55 | DS-R1-1.5B | CommonsenseQA | 12 | 40.6088 | 40.5542 | 0.0546 |
| 14 | 56 | DS-R1-1.5B | CommonsenseQA | 12 | 40.2675 | 41.0456 | -0.7781 |
| 15 | 57 | DS-R1-1.5B | CommonsenseQA | 12 | 40.5132 | 40.6224 | -0.1092 |
| 16 | 58 | DS-R1-1.5B | CommonsenseQA | 12 | 40.3767 | 40.991 | -0.6143 |
| 17 | 59 | DS-R1-1.5B | CommonsenseQA | 12 | 40.5678 | 41.1684 | -0.6006 |
| 18 | 60 | DS-R1-1.5B | CommonsenseQA | 12 | 40.6361 | 41.264 | -0.6279 |
| 19 | 61 | DS-R1-1.5B | CommonsenseQA | 12 | 40.6907 | 40.6361 | 0.0546 |
| 20 | 62 | DS-R1-1.5B | CommonsenseQA | 12 | 40.2266 | 41.0046 | -0.7781 |
| 21 | 63 | DS-R1-1.5B | CommonsenseQA | 12 | 40.5132 | 40.9091 | -0.3959 |
| 22 | 64 | DS-R1-1.5B | CommonsenseQA | 12 | 40.3358 | 41.1275 | -0.7917 |
| 23 | 65 | DS-R1-1.5B | CommonsenseQA | 12 | 40.6497 | 40.5951 | 0.0546 |
| 24 | 66 | DS-R1-1.5B | CommonsenseQA | 12 | 40.5815 | 40.4313 | 0.1502 |
| 25 | 67 | DS-R1-1.5B | CommonsenseQA | 12 | 40.1993 | 40.9227 | -0.7235 |
| 26 | 68 | DS-R1-1.5B | CommonsenseQA | 12 | 40.4723 | 41.0456 | -0.5733 |
| 27 | 69 | DS-R1-1.5B | CommonsenseQA | 12 | 40.7862 | 40.9227 | -0.1365 |
| 28 | 70 | DS-R1-1.5B | CommonsenseQA | 12 | 40.3221 | 40.8818 | -0.5597 |
| 29 | 71 | DS-R1-1.5B | CommonsenseQA | 12 | 40.3221 | 40.5815 | -0.2594 |
| 30 | 72 | DS-R1-1.5B | CommonsenseQA | 12 | 40.3494 | 40.8681 | -0.5187 |
| 31 | 73 | DS-R1-1.5B | CommonsenseQA | 12 | 40.445 | 41.1275 | -0.6825 |
| 32 | 74 | DS-R1-1.5B | CommonsenseQA | 12 | 40.4723 | 41.0456 | -0.5733 |
| 33 | 75 | DS-R1-1.5B | CommonsenseQA | 12 | 40.2402 | 41.3459 | -1.1057 |
| 34 | 76 | DS-R1-1.5B | CommonsenseQA | 12 | 40.5405 | 40.9637 | -0.4232 |
| 35 | 77 | DS-R1-1.5B | CommonsenseQA | 12 | 39.9672 | 41.2094 | -1.2422 |
| 36 | 78 | DS-R1-1.5B | CommonsenseQA | 12 | 40.7862 | 40.404 | 0.3822 |
| 37 | 79 | DS-R1-1.5B | CommonsenseQA | 12 | 40.3904 | 41.4414 | -1.0511 |
| 38 | 80 | DS-R1-1.5B | CommonsenseQA | 12 | 40.2812 | 41.1957 | -0.9146 |
| 39 | 81 | DS-R1-1.5B | CommonsenseQA | 12 | 41.0729 | 40.8954 | 0.1775 |
| 40 | 82 | DS-R1-1.5B | CommonsenseQA | 12 | 40.3358 | 40.5951 | -0.2594 |
| 41 | 83 | DS-R1-1.5B | CommonsenseQA | 12 | 40.1037 | 41.0456 | -0.9419 |
| 42 | 84 | DS-R1-1.5B | CommonsenseQA | 12 | 40.677 | 41.1411 | -0.4641 |
| 43 | 85 | DS-R1-1.5B | CommonsenseQA | 12 | 40.5815 | 41.1684 | -0.587 |
| 44 | 86 | DS-R1-1.5B | CommonsenseQA | 12 | 40.2948 | 41.1138 | -0.819 |
| 45 | 87 | DS-R1-1.5B | CommonsenseQA | 12 | 40.1993 | 40.7862 | -0.587 |
| 46 | 88 | DS-R1-1.5B | CommonsenseQA | 12 | 40.7453 | 40.991 | -0.2457 |
| 47 | 89 | DS-R1-1.5B | CommonsenseQA | 12 | 40.2948 | 40.7726 | -0.4778 |
| 48 | 90 | DS-R1-1.5B | CommonsenseQA | 12 | 40.404 | 41.1275 | -0.7235 |
| 49 | 91 | DS-R1-1.5B | CommonsenseQA | 12 | 40.6088 | 40.8954 | -0.2867 |
| 0 | 42 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 1 | 43 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 2 | 44 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 3 | 45 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 4 | 46 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 5 | 47 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 6 | 48 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 7 | 49 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 8 | 50 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 9 | 51 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 10 | 52 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 11 | 53 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 12 | 54 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 13 | 55 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 14 | 56 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 15 | 57 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 16 | 58 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 17 | 59 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 18 | 60 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 19 | 61 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 20 | 62 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 21 | 63 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 22 | 64 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 23 | 65 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 24 | 66 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 25 | 67 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 26 | 68 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 27 | 69 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 28 | 70 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 29 | 71 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 30 | 72 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 31 | 73 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 32 | 74 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 33 | 75 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 34 | 76 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 35 | 77 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 36 | 78 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 37 | 79 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 38 | 80 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 39 | 81 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 40 | 82 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 41 | 83 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 42 | 84 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 43 | 85 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 44 | 86 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 45 | 87 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 46 | 88 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 47 | 89 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 48 | 90 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 49 | 91 | DS-R1-1.5B | CommonsenseQA | 16 | 40.4894 | 40.991 | -0.5016 |
| 0 | 42 | DS-R1-1.5B | GPQA | 4 | 31.6964 | 35.9375 | -4.2411 |
| 1 | 43 | DS-R1-1.5B | GPQA | 4 | 33.0357 | 37.3884 | -4.3527 |
| 2 | 44 | DS-R1-1.5B | GPQA | 4 | 34.2634 | 38.7277 | -4.4643 |
| 3 | 45 | DS-R1-1.5B | GPQA | 4 | 30.5804 | 37.6116 | -7.0312 |
| 4 | 46 | DS-R1-1.5B | GPQA | 4 | 29.7991 | 37.5 | -7.7009 |
| 5 | 47 | DS-R1-1.5B | GPQA | 4 | 33.1473 | 36.4955 | -3.3482 |
| 6 | 48 | DS-R1-1.5B | GPQA | 4 | 30.0223 | 38.7277 | -8.7054 |
| 7 | 49 | DS-R1-1.5B | GPQA | 4 | 31.5848 | 36.6071 | -5.0223 |
| 8 | 50 | DS-R1-1.5B | GPQA | 4 | 30.8036 | 39.5089 | -8.7054 |
| 9 | 51 | DS-R1-1.5B | GPQA | 4 | 31.808 | 35.8259 | -4.0179 |
| 10 | 52 | DS-R1-1.5B | GPQA | 4 | 30.8036 | 36.4955 | -5.692 |
| 11 | 53 | DS-R1-1.5B | GPQA | 4 | 29.7991 | 39.3973 | -9.5982 |
| 12 | 54 | DS-R1-1.5B | GPQA | 4 | 33.0357 | 39.2857 | -6.25 |
| 13 | 55 | DS-R1-1.5B | GPQA | 4 | 32.3661 | 38.9509 | -6.5848 |
| 14 | 56 | DS-R1-1.5B | GPQA | 4 | 35.4911 | 38.3929 | -2.9018 |
| 15 | 57 | DS-R1-1.5B | GPQA | 4 | 31.5848 | 37.2768 | -5.692 |
| 16 | 58 | DS-R1-1.5B | GPQA | 4 | 30.9152 | 37.8348 | -6.9196 |
| 17 | 59 | DS-R1-1.5B | GPQA | 4 | 31.808 | 37.5 | -5.692 |
| 18 | 60 | DS-R1-1.5B | GPQA | 4 | 31.3616 | 36.6071 | -5.2455 |
| 19 | 61 | DS-R1-1.5B | GPQA | 4 | 31.6964 | 37.0536 | -5.3571 |
| 20 | 62 | DS-R1-1.5B | GPQA | 4 | 32.3661 | 38.058 | -5.692 |
| 21 | 63 | DS-R1-1.5B | GPQA | 4 | 34.0402 | 39.3973 | -5.3571 |
| 22 | 64 | DS-R1-1.5B | GPQA | 4 | 32.1429 | 37.2768 | -5.1339 |
| 23 | 65 | DS-R1-1.5B | GPQA | 4 | 32.5893 | 39.2857 | -6.6964 |
| 24 | 66 | DS-R1-1.5B | GPQA | 4 | 30.2455 | 38.5045 | -8.2589 |
| 25 | 67 | DS-R1-1.5B | GPQA | 4 | 32.2545 | 37.5 | -5.2455 |
| 26 | 68 | DS-R1-1.5B | GPQA | 4 | 31.0268 | 36.8304 | -5.8036 |
| 27 | 69 | DS-R1-1.5B | GPQA | 4 | 33.2589 | 36.1607 | -2.9018 |
| 28 | 70 | DS-R1-1.5B | GPQA | 4 | 33.1473 | 38.8393 | -5.692 |
| 29 | 71 | DS-R1-1.5B | GPQA | 4 | 31.4732 | 37.7232 | -6.25 |
| 30 | 72 | DS-R1-1.5B | GPQA | 4 | 32.3661 | 36.4955 | -4.1295 |
| 31 | 73 | DS-R1-1.5B | GPQA | 4 | 32.5893 | 38.8393 | -6.25 |
| 32 | 74 | DS-R1-1.5B | GPQA | 4 | 30.4688 | 38.058 | -7.5893 |
| 33 | 75 | DS-R1-1.5B | GPQA | 4 | 33.0357 | 38.058 | -5.0223 |
| 34 | 76 | DS-R1-1.5B | GPQA | 4 | 34.933 | 41.2946 | -6.3616 |
| 35 | 77 | DS-R1-1.5B | GPQA | 4 | 32.0312 | 35.4911 | -3.4598 |
| 36 | 78 | DS-R1-1.5B | GPQA | 4 | 32.7009 | 36.7188 | -4.0179 |
| 37 | 79 | DS-R1-1.5B | GPQA | 4 | 30.5804 | 38.1696 | -7.5893 |
| 38 | 80 | DS-R1-1.5B | GPQA | 4 | 30.4688 | 37.1652 | -6.6964 |
| 39 | 81 | DS-R1-1.5B | GPQA | 4 | 30.692 | 36.8304 | -6.1384 |
| 40 | 82 | DS-R1-1.5B | GPQA | 4 | 32.5893 | 37.7232 | -5.1339 |
| 41 | 83 | DS-R1-1.5B | GPQA | 4 | 35.0446 | 35.3795 | -0.3348 |
| 42 | 84 | DS-R1-1.5B | GPQA | 4 | 35.0446 | 37.9464 | -2.9018 |
| 43 | 85 | DS-R1-1.5B | GPQA | 4 | 31.4732 | 40.7366 | -9.2634 |
| 44 | 86 | DS-R1-1.5B | GPQA | 4 | 34.2634 | 37.9464 | -3.683 |
| 45 | 87 | DS-R1-1.5B | GPQA | 4 | 30.5804 | 38.3929 | -7.8125 |
| 46 | 88 | DS-R1-1.5B | GPQA | 4 | 31.6964 | 39.0625 | -7.3661 |
| 47 | 89 | DS-R1-1.5B | GPQA | 4 | 31.808 | 38.2812 | -6.4732 |
| 48 | 90 | DS-R1-1.5B | GPQA | 4 | 31.808 | 39.5089 | -7.7009 |
| 49 | 91 | DS-R1-1.5B | GPQA | 4 | 31.808 | 39.3973 | -7.5893 |
| 0 | 42 | DS-R1-1.5B | GPQA | 8 | 31.808 | 36.2723 | -4.4643 |
| 1 | 43 | DS-R1-1.5B | GPQA | 8 | 32.1429 | 37.2768 | -5.1339 |
| 2 | 44 | DS-R1-1.5B | GPQA | 8 | 31.4732 | 39.4531 | -7.9799 |
| 3 | 45 | DS-R1-1.5B | GPQA | 8 | 30.692 | 38.2812 | -7.5893 |
| 4 | 46 | DS-R1-1.5B | GPQA | 8 | 30.971 | 37.6674 | -6.6964 |
| 5 | 47 | DS-R1-1.5B | GPQA | 8 | 33.3705 | 37.8348 | -4.4643 |
| 6 | 48 | DS-R1-1.5B | GPQA | 8 | 32.0312 | 38.2812 | -6.25 |
| 7 | 49 | DS-R1-1.5B | GPQA | 8 | 32.0312 | 36.942 | -4.9107 |
| 8 | 50 | DS-R1-1.5B | GPQA | 8 | 30.9152 | 39.2857 | -8.3705 |
| 9 | 51 | DS-R1-1.5B | GPQA | 8 | 30.3571 | 36.4955 | -6.1384 |
| 10 | 52 | DS-R1-1.5B | GPQA | 8 | 31.529 | 38.4487 | -6.9196 |
| 11 | 53 | DS-R1-1.5B | GPQA | 8 | 30.2455 | 37.4442 | -7.1987 |
| 12 | 54 | DS-R1-1.5B | GPQA | 8 | 32.1987 | 39.1183 | -6.9196 |
| 13 | 55 | DS-R1-1.5B | GPQA | 8 | 31.5848 | 38.9509 | -7.3661 |
| 14 | 56 | DS-R1-1.5B | GPQA | 8 | 32.5893 | 39.6205 | -7.0312 |
| 15 | 57 | DS-R1-1.5B | GPQA | 8 | 31.6406 | 39.3415 | -7.7009 |
| 16 | 58 | DS-R1-1.5B | GPQA | 8 | 30.2455 | 39.5089 | -9.2634 |
| 17 | 59 | DS-R1-1.5B | GPQA | 8 | 31.4174 | 38.7835 | -7.3661 |
| 18 | 60 | DS-R1-1.5B | GPQA | 8 | 32.0312 | 37.4442 | -5.4129 |
| 19 | 61 | DS-R1-1.5B | GPQA | 8 | 31.1942 | 37.8906 | -6.6964 |
| 20 | 62 | DS-R1-1.5B | GPQA | 8 | 31.9196 | 38.6161 | -6.6964 |
| 21 | 63 | DS-R1-1.5B | GPQA | 8 | 31.6964 | 38.5603 | -6.8638 |
| 22 | 64 | DS-R1-1.5B | GPQA | 8 | 31.6964 | 38.1138 | -6.4174 |
| 23 | 65 | DS-R1-1.5B | GPQA | 8 | 32.6451 | 38.6719 | -6.0268 |
| 24 | 66 | DS-R1-1.5B | GPQA | 8 | 30.4129 | 38.9509 | -8.5379 |
| 25 | 67 | DS-R1-1.5B | GPQA | 8 | 32.3661 | 38.5045 | -6.1384 |
| 26 | 68 | DS-R1-1.5B | GPQA | 8 | 31.529 | 38.7277 | -7.1987 |
| 27 | 69 | DS-R1-1.5B | GPQA | 8 | 32.7567 | 37.5 | -4.7433 |
| 28 | 70 | DS-R1-1.5B | GPQA | 8 | 32.4219 | 39.4531 | -7.0312 |
| 29 | 71 | DS-R1-1.5B | GPQA | 8 | 32.1987 | 37.5558 | -5.3571 |
| 30 | 72 | DS-R1-1.5B | GPQA | 8 | 31.6406 | 37.1652 | -5.5246 |
| 31 | 73 | DS-R1-1.5B | GPQA | 8 | 31.0268 | 38.8393 | -7.8125 |
| 32 | 74 | DS-R1-1.5B | GPQA | 8 | 32.3103 | 36.7746 | -4.4643 |
| 33 | 75 | DS-R1-1.5B | GPQA | 8 | 31.9754 | 38.4487 | -6.4732 |
| 34 | 76 | DS-R1-1.5B | GPQA | 8 | 31.6964 | 39.9554 | -8.2589 |
| 35 | 77 | DS-R1-1.5B | GPQA | 8 | 31.8638 | 38.4487 | -6.5848 |
| 36 | 78 | DS-R1-1.5B | GPQA | 8 | 32.2545 | 38.3371 | -6.0826 |
| 37 | 79 | DS-R1-1.5B | GPQA | 8 | 31.0268 | 39.1183 | -8.0915 |
| 38 | 80 | DS-R1-1.5B | GPQA | 8 | 31.9196 | 37.3326 | -5.4129 |
| 39 | 81 | DS-R1-1.5B | GPQA | 8 | 30.9152 | 39.0625 | -8.1473 |
| 40 | 82 | DS-R1-1.5B | GPQA | 8 | 31.1384 | 39.0067 | -7.8683 |
| 41 | 83 | DS-R1-1.5B | GPQA | 8 | 32.1987 | 36.942 | -4.7433 |
| 42 | 84 | DS-R1-1.5B | GPQA | 8 | 33.0357 | 38.4487 | -5.4129 |
| 43 | 85 | DS-R1-1.5B | GPQA | 8 | 31.6406 | 39.4531 | -7.8125 |
| 44 | 86 | DS-R1-1.5B | GPQA | 8 | 32.5335 | 37.5558 | -5.0223 |
| 45 | 87 | DS-R1-1.5B | GPQA | 8 | 31.4174 | 38.9509 | -7.5335 |
| 46 | 88 | DS-R1-1.5B | GPQA | 8 | 32.1987 | 38.6719 | -6.4732 |
| 47 | 89 | DS-R1-1.5B | GPQA | 8 | 30.971 | 37.8906 | -6.9196 |
| 48 | 90 | DS-R1-1.5B | GPQA | 8 | 30.1339 | 39.1183 | -8.9844 |
| 49 | 91 | DS-R1-1.5B | GPQA | 8 | 31.7522 | 38.4487 | -6.6964 |
| 0 | 42 | DS-R1-1.5B | GPQA | 12 | 31.3616 | 37.7976 | -6.436 |
| 1 | 43 | DS-R1-1.5B | GPQA | 12 | 31.9568 | 37.9464 | -5.9896 |
| 2 | 44 | DS-R1-1.5B | GPQA | 12 | 31.6592 | 39.1369 | -7.4777 |
| 3 | 45 | DS-R1-1.5B | GPQA | 12 | 30.7292 | 39.2485 | -8.5193 |
| 4 | 46 | DS-R1-1.5B | GPQA | 12 | 31.436 | 38.8393 | -7.4033 |
| 5 | 47 | DS-R1-1.5B | GPQA | 12 | 32.2917 | 38.8021 | -6.5104 |
| 6 | 48 | DS-R1-1.5B | GPQA | 12 | 32.2917 | 38.5045 | -6.2128 |
| 7 | 49 | DS-R1-1.5B | GPQA | 12 | 31.622 | 37.3884 | -5.7664 |
| 8 | 50 | DS-R1-1.5B | GPQA | 12 | 30.7664 | 38.7649 | -7.9985 |
| 9 | 51 | DS-R1-1.5B | GPQA | 12 | 31.5476 | 38.1324 | -6.5848 |
| 10 | 52 | DS-R1-1.5B | GPQA | 12 | 31.4732 | 38.4301 | -6.9568 |
| 11 | 53 | DS-R1-1.5B | GPQA | 12 | 30.9524 | 38.7649 | -7.8125 |
| 12 | 54 | DS-R1-1.5B | GPQA | 12 | 31.5104 | 38.3557 | -6.8452 |
| 13 | 55 | DS-R1-1.5B | GPQA | 12 | 31.9568 | 38.8765 | -6.9196 |
| 14 | 56 | DS-R1-1.5B | GPQA | 12 | 32.6265 | 38.1324 | -5.506 |
| 15 | 57 | DS-R1-1.5B | GPQA | 12 | 32.0312 | 38.9881 | -6.9568 |
| 16 | 58 | DS-R1-1.5B | GPQA | 12 | 31.1756 | 39.0625 | -7.8869 |
| 17 | 59 | DS-R1-1.5B | GPQA | 12 | 31.5104 | 38.4673 | -6.9568 |
| 18 | 60 | DS-R1-1.5B | GPQA | 12 | 31.8452 | 38.4301 | -6.5848 |
| 19 | 61 | DS-R1-1.5B | GPQA | 12 | 31.5848 | 38.5417 | -6.9568 |
| 20 | 62 | DS-R1-1.5B | GPQA | 12 | 30.9896 | 39.4717 | -8.4821 |
| 21 | 63 | DS-R1-1.5B | GPQA | 12 | 31.9568 | 39.0997 | -7.1429 |
| 22 | 64 | DS-R1-1.5B | GPQA | 12 | 32.1429 | 37.4256 | -5.2827 |
| 23 | 65 | DS-R1-1.5B | GPQA | 12 | 32.0312 | 38.6905 | -6.6592 |
| 24 | 66 | DS-R1-1.5B | GPQA | 12 | 31.1756 | 37.9464 | -6.7708 |
| 25 | 67 | DS-R1-1.5B | GPQA | 12 | 32.1057 | 38.2068 | -6.1012 |
| 26 | 68 | DS-R1-1.5B | GPQA | 12 | 32.1429 | 38.8765 | -6.7336 |
| 27 | 69 | DS-R1-1.5B | GPQA | 12 | 32.0312 | 38.5789 | -6.5476 |
| 28 | 70 | DS-R1-1.5B | GPQA | 12 | 32.2173 | 38.8393 | -6.622 |
| 29 | 71 | DS-R1-1.5B | GPQA | 12 | 31.3244 | 38.6533 | -7.3289 |
| 30 | 72 | DS-R1-1.5B | GPQA | 12 | 31.5848 | 38.2068 | -6.622 |
| 31 | 73 | DS-R1-1.5B | GPQA | 12 | 31.0268 | 38.0208 | -6.994 |
| 32 | 74 | DS-R1-1.5B | GPQA | 12 | 31.808 | 37.4256 | -5.6176 |
| 33 | 75 | DS-R1-1.5B | GPQA | 12 | 31.5104 | 38.8021 | -7.2917 |
| 34 | 76 | DS-R1-1.5B | GPQA | 12 | 31.6592 | 38.9509 | -7.2917 |
| 35 | 77 | DS-R1-1.5B | GPQA | 12 | 31.9568 | 38.6161 | -6.6592 |
| 36 | 78 | DS-R1-1.5B | GPQA | 12 | 32.2173 | 38.9137 | -6.6964 |
| 37 | 79 | DS-R1-1.5B | GPQA | 12 | 31.994 | 38.5417 | -6.5476 |
| 38 | 80 | DS-R1-1.5B | GPQA | 12 | 31.7708 | 38.3557 | -6.5848 |
| 39 | 81 | DS-R1-1.5B | GPQA | 12 | 32.1429 | 38.8765 | -6.7336 |
| 40 | 82 | DS-R1-1.5B | GPQA | 12 | 31.5476 | 38.9137 | -7.3661 |
| 41 | 83 | DS-R1-1.5B | GPQA | 12 | 31.6592 | 38.2812 | -6.622 |
| 42 | 84 | DS-R1-1.5B | GPQA | 12 | 31.6592 | 38.2068 | -6.5476 |
| 43 | 85 | DS-R1-1.5B | GPQA | 12 | 31.9568 | 38.6905 | -6.7336 |
| 44 | 86 | DS-R1-1.5B | GPQA | 12 | 31.9196 | 37.8348 | -5.9152 |
| 45 | 87 | DS-R1-1.5B | GPQA | 12 | 31.7708 | 38.4673 | -6.6964 |
| 46 | 88 | DS-R1-1.5B | GPQA | 12 | 32.3289 | 38.6161 | -6.2872 |
| 47 | 89 | DS-R1-1.5B | GPQA | 12 | 31.622 | 38.3929 | -6.7708 |
| 48 | 90 | DS-R1-1.5B | GPQA | 12 | 31.4732 | 38.7277 | -7.2545 |
| 49 | 91 | DS-R1-1.5B | GPQA | 12 | 31.9196 | 37.5744 | -5.6548 |
| 0 | 42 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 1 | 43 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 2 | 44 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 3 | 45 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 4 | 46 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 5 | 47 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 6 | 48 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 7 | 49 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 8 | 50 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 9 | 51 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 10 | 52 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 11 | 53 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 12 | 54 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 13 | 55 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 14 | 56 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 15 | 57 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 16 | 58 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 17 | 59 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 18 | 60 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 19 | 61 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 20 | 62 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 21 | 63 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 22 | 64 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 23 | 65 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 24 | 66 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 25 | 67 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 26 | 68 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 27 | 69 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 28 | 70 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 29 | 71 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 30 | 72 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 31 | 73 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 32 | 74 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 33 | 75 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 34 | 76 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 35 | 77 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 36 | 78 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 37 | 79 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 38 | 80 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 39 | 81 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 40 | 82 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 41 | 83 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 42 | 84 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 43 | 85 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 44 | 86 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 45 | 87 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 46 | 88 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 47 | 89 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 48 | 90 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 49 | 91 | DS-R1-1.5B | GPQA | 16 | 31.7243 | 38.644 | -6.9196 |
| 0 | 42 | DS-R1-1.5B | GSM8K | 4 | 78.1274 | 68.9158 | 9.2115 |
| 1 | 43 | DS-R1-1.5B | GSM8K | 4 | 77.4829 | 68.044 | 9.439 |
| 2 | 44 | DS-R1-1.5B | GSM8K | 4 | 78.3548 | 68.5368 | 9.818 |
| 3 | 45 | DS-R1-1.5B | GSM8K | 4 | 77.9757 | 68.423 | 9.5527 |
| 4 | 46 | DS-R1-1.5B | GSM8K | 4 | 77.8241 | 69.4466 | 8.3776 |
| 5 | 47 | DS-R1-1.5B | GSM8K | 4 | 77.7862 | 68.3851 | 9.4011 |
| 6 | 48 | DS-R1-1.5B | GSM8K | 4 | 78.1653 | 67.9682 | 10.1971 |
| 7 | 49 | DS-R1-1.5B | GSM8K | 4 | 78.8476 | 68.2714 | 10.5762 |
| 8 | 50 | DS-R1-1.5B | GSM8K | 4 | 77.7104 | 69.5982 | 8.1122 |
| 9 | 51 | DS-R1-1.5B | GSM8K | 4 | 78.0895 | 69.0675 | 9.022 |
| 10 | 52 | DS-R1-1.5B | GSM8K | 4 | 78.5064 | 68.0819 | 10.4246 |
| 11 | 53 | DS-R1-1.5B | GSM8K | 4 | 78.4685 | 68.7263 | 9.7422 |
| 12 | 54 | DS-R1-1.5B | GSM8K | 4 | 76.7627 | 69.4086 | 7.3541 |
| 13 | 55 | DS-R1-1.5B | GSM8K | 4 | 79.3025 | 68.6126 | 10.6899 |
| 14 | 56 | DS-R1-1.5B | GSM8K | 4 | 77.1039 | 68.8779 | 8.2259 |
| 15 | 57 | DS-R1-1.5B | GSM8K | 4 | 77.862 | 68.6126 | 9.2494 |
| 16 | 58 | DS-R1-1.5B | GSM8K | 4 | 78.696 | 68.5747 | 10.1213 |
| 17 | 59 | DS-R1-1.5B | GSM8K | 4 | 78.8855 | 68.3093 | 10.5762 |
| 18 | 60 | DS-R1-1.5B | GSM8K | 4 | 78.8097 | 68.6505 | 10.1592 |
| 19 | 61 | DS-R1-1.5B | GSM8K | 4 | 78.6581 | 68.7263 | 9.9318 |
| 20 | 62 | DS-R1-1.5B | GSM8K | 4 | 79.0371 | 67.8544 | 11.1827 |
| 21 | 63 | DS-R1-1.5B | GSM8K | 4 | 76.8385 | 68.9158 | 7.9227 |
| 22 | 64 | DS-R1-1.5B | GSM8K | 4 | 78.5823 | 68.7642 | 9.818 |
| 23 | 65 | DS-R1-1.5B | GSM8K | 4 | 77.9378 | 68.8021 | 9.1357 |
| 24 | 66 | DS-R1-1.5B | GSM8K | 4 | 78.3927 | 68.4989 | 9.8939 |
| 25 | 67 | DS-R1-1.5B | GSM8K | 4 | 77.3313 | 69.1433 | 8.188 |
| 26 | 68 | DS-R1-1.5B | GSM8K | 4 | 78.0516 | 70.6975 | 7.3541 |
| 27 | 69 | DS-R1-1.5B | GSM8K | 4 | 77.4071 | 69.257 | 8.1501 |
| 28 | 70 | DS-R1-1.5B | GSM8K | 4 | 77.8241 | 68.423 | 9.4011 |
| 29 | 71 | DS-R1-1.5B | GSM8K | 4 | 77.3313 | 68.5368 | 8.7945 |
| 30 | 72 | DS-R1-1.5B | GSM8K | 4 | 79.0751 | 69.3328 | 9.7422 |
| 31 | 73 | DS-R1-1.5B | GSM8K | 4 | 78.3169 | 68.1956 | 10.1213 |
| 32 | 74 | DS-R1-1.5B | GSM8K | 4 | 77.5967 | 68.6126 | 8.9841 |
| 33 | 75 | DS-R1-1.5B | GSM8K | 4 | 78.8476 | 67.8923 | 10.9553 |
| 34 | 76 | DS-R1-1.5B | GSM8K | 4 | 78.5823 | 68.8779 | 9.7043 |
| 35 | 77 | DS-R1-1.5B | GSM8K | 4 | 78.1653 | 69.2191 | 8.9462 |
| 36 | 78 | DS-R1-1.5B | GSM8K | 4 | 79.1509 | 68.8779 | 10.2729 |
| 37 | 79 | DS-R1-1.5B | GSM8K | 4 | 78.1274 | 68.6126 | 9.5148 |
| 38 | 80 | DS-R1-1.5B | GSM8K | 4 | 78.5444 | 68.1956 | 10.3487 |
| 39 | 81 | DS-R1-1.5B | GSM8K | 4 | 78.1653 | 68.0061 | 10.1592 |
| 40 | 82 | DS-R1-1.5B | GSM8K | 4 | 78.8476 | 69.1433 | 9.7043 |
| 41 | 83 | DS-R1-1.5B | GSM8K | 4 | 77.9757 | 67.4375 | 10.5383 |
| 42 | 84 | DS-R1-1.5B | GSM8K | 4 | 78.2032 | 68.423 | 9.7801 |
| 43 | 85 | DS-R1-1.5B | GSM8K | 4 | 78.8855 | 68.1577 | 10.7278 |
| 44 | 86 | DS-R1-1.5B | GSM8K | 4 | 76.9522 | 68.2335 | 8.7187 |
| 45 | 87 | DS-R1-1.5B | GSM8K | 4 | 77.4071 | 68.1198 | 9.2873 |
| 46 | 88 | DS-R1-1.5B | GSM8K | 4 | 78.6202 | 68.84 | 9.7801 |
| 47 | 89 | DS-R1-1.5B | GSM8K | 4 | 77.445 | 69.7877 | 7.6573 |
| 48 | 90 | DS-R1-1.5B | GSM8K | 4 | 78.9992 | 68.0819 | 10.9174 |
| 49 | 91 | DS-R1-1.5B | GSM8K | 4 | 78.0516 | 67.3616 | 10.6899 |
| 0 | 42 | DS-R1-1.5B | GSM8K | 8 | 78.4117 | 68.6884 | 9.7233 |
| 1 | 43 | DS-R1-1.5B | GSM8K | 8 | 78.6202 | 67.8544 | 10.7657 |
| 2 | 44 | DS-R1-1.5B | GSM8K | 8 | 78.298 | 67.8165 | 10.4814 |
| 3 | 45 | DS-R1-1.5B | GSM8K | 8 | 78.5064 | 67.7597 | 10.7468 |
| 4 | 46 | DS-R1-1.5B | GSM8K | 8 | 78.9045 | 68.0819 | 10.8226 |
| 5 | 47 | DS-R1-1.5B | GSM8K | 8 | 78.4117 | 67.7028 | 10.7089 |
| 6 | 48 | DS-R1-1.5B | GSM8K | 8 | 78.7908 | 67.9871 | 10.8036 |
| 7 | 49 | DS-R1-1.5B | GSM8K | 8 | 78.9045 | 68.1577 | 10.7468 |
| 8 | 50 | DS-R1-1.5B | GSM8K | 8 | 78.8666 | 68.7074 | 10.1592 |
| 9 | 51 | DS-R1-1.5B | GSM8K | 8 | 78.8666 | 68.8211 | 10.0455 |
| 10 | 52 | DS-R1-1.5B | GSM8K | 8 | 79.0751 | 67.3995 | 11.6755 |
| 11 | 53 | DS-R1-1.5B | GSM8K | 8 | 78.6012 | 68.1766 | 10.4246 |
| 12 | 54 | DS-R1-1.5B | GSM8K | 8 | 77.7862 | 68.2904 | 9.4958 |
| 13 | 55 | DS-R1-1.5B | GSM8K | 8 | 79.3025 | 67.9871 | 11.3154 |
| 14 | 56 | DS-R1-1.5B | GSM8K | 8 | 78.6202 | 68.0819 | 10.5383 |
| 15 | 57 | DS-R1-1.5B | GSM8K | 8 | 78.696 | 67.5133 | 11.1827 |
| 16 | 58 | DS-R1-1.5B | GSM8K | 8 | 78.5444 | 68.2525 | 10.2919 |
| 17 | 59 | DS-R1-1.5B | GSM8K | 8 | 78.677 | 67.9113 | 10.7657 |
| 18 | 60 | DS-R1-1.5B | GSM8K | 8 | 78.6202 | 68.442 | 10.1782 |
| 19 | 61 | DS-R1-1.5B | GSM8K | 8 | 79.0561 | 67.3616 | 11.6945 |
| 20 | 62 | DS-R1-1.5B | GSM8K | 8 | 79.2077 | 68.1577 | 11.05 |
| 21 | 63 | DS-R1-1.5B | GSM8K | 8 | 78.1842 | 68.044 | 10.1403 |
| 22 | 64 | DS-R1-1.5B | GSM8K | 8 | 78.7149 | 68.2335 | 10.4814 |
| 23 | 65 | DS-R1-1.5B | GSM8K | 8 | 79.0182 | 67.627 | 11.3912 |
| 24 | 66 | DS-R1-1.5B | GSM8K | 8 | 78.7339 | 67.4754 | 11.2585 |
| 25 | 67 | DS-R1-1.5B | GSM8K | 8 | 78.6391 | 68.1008 | 10.5383 |
| 26 | 68 | DS-R1-1.5B | GSM8K | 8 | 78.8476 | 68.859 | 9.9886 |
| 27 | 69 | DS-R1-1.5B | GSM8K | 8 | 78.3548 | 67.8923 | 10.4625 |
| 28 | 70 | DS-R1-1.5B | GSM8K | 8 | 78.3169 | 68.4799 | 9.837 |
| 29 | 71 | DS-R1-1.5B | GSM8K | 8 | 78.9234 | 67.9492 | 10.9742 |
| 30 | 72 | DS-R1-1.5B | GSM8K | 8 | 79.0751 | 68.025 | 11.05 |
| 31 | 73 | DS-R1-1.5B | GSM8K | 8 | 78.298 | 67.9492 | 10.3487 |
| 32 | 74 | DS-R1-1.5B | GSM8K | 8 | 78.6391 | 68.2904 | 10.3487 |
| 33 | 75 | DS-R1-1.5B | GSM8K | 8 | 79.0182 | 68.1577 | 10.8605 |
| 34 | 76 | DS-R1-1.5B | GSM8K | 8 | 79.2077 | 68.044 | 11.1638 |
| 35 | 77 | DS-R1-1.5B | GSM8K | 8 | 78.677 | 67.7218 | 10.9553 |
| 36 | 78 | DS-R1-1.5B | GSM8K | 8 | 79.113 | 67.8734 | 11.2396 |
| 37 | 79 | DS-R1-1.5B | GSM8K | 8 | 78.7908 | 69.0864 | 9.7043 |
| 38 | 80 | DS-R1-1.5B | GSM8K | 8 | 78.7718 | 68.044 | 10.7278 |
| 39 | 81 | DS-R1-1.5B | GSM8K | 8 | 78.3548 | 67.9492 | 10.4056 |
| 40 | 82 | DS-R1-1.5B | GSM8K | 8 | 78.7718 | 67.4754 | 11.2964 |
| 41 | 83 | DS-R1-1.5B | GSM8K | 8 | 78.4496 | 67.5891 | 10.8605 |
| 42 | 84 | DS-R1-1.5B | GSM8K | 8 | 78.6012 | 67.8734 | 10.7278 |
| 43 | 85 | DS-R1-1.5B | GSM8K | 8 | 78.7528 | 67.6459 | 11.1069 |
| 44 | 86 | DS-R1-1.5B | GSM8K | 8 | 78.7908 | 67.5701 | 11.2206 |
| 45 | 87 | DS-R1-1.5B | GSM8K | 8 | 78.3548 | 68.0819 | 10.2729 |
| 46 | 88 | DS-R1-1.5B | GSM8K | 8 | 78.279 | 68.1766 | 10.1024 |
| 47 | 89 | DS-R1-1.5B | GSM8K | 8 | 78.3927 | 67.8923 | 10.5004 |
| 48 | 90 | DS-R1-1.5B | GSM8K | 8 | 78.6202 | 67.8923 | 10.7278 |
| 49 | 91 | DS-R1-1.5B | GSM8K | 8 | 78.298 | 67.4754 | 10.8226 |
| 0 | 42 | DS-R1-1.5B | GSM8K | 12 | 78.6454 | 68.1956 | 10.4498 |
| 1 | 43 | DS-R1-1.5B | GSM8K | 12 | 78.4812 | 67.7407 | 10.7405 |
| 2 | 44 | DS-R1-1.5B | GSM8K | 12 | 78.9992 | 67.8039 | 11.1954 |
| 3 | 45 | DS-R1-1.5B | GSM8K | 12 | 78.4054 | 67.7533 | 10.652 |
| 4 | 46 | DS-R1-1.5B | GSM8K | 12 | 78.8855 | 67.7028 | 11.1827 |
| 5 | 47 | DS-R1-1.5B | GSM8K | 12 | 78.6328 | 67.6144 | 11.0184 |
| 6 | 48 | DS-R1-1.5B | GSM8K | 12 | 78.9866 | 67.7281 | 11.2585 |
| 7 | 49 | DS-R1-1.5B | GSM8K | 12 | 78.974 | 67.905 | 11.069 |
| 8 | 50 | DS-R1-1.5B | GSM8K | 12 | 79.0498 | 67.8418 | 11.208 |
| 9 | 51 | DS-R1-1.5B | GSM8K | 12 | 78.4685 | 68.3093 | 10.1592 |
| 10 | 52 | DS-R1-1.5B | GSM8K | 12 | 78.6454 | 67.7281 | 10.9174 |
| 11 | 53 | DS-R1-1.5B | GSM8K | 12 | 78.835 | 68.1324 | 10.7026 |
| 12 | 54 | DS-R1-1.5B | GSM8K | 12 | 78.5949 | 67.7533 | 10.8415 |
| 13 | 55 | DS-R1-1.5B | GSM8K | 12 | 78.9234 | 67.8544 | 11.069 |
| 14 | 56 | DS-R1-1.5B | GSM8K | 12 | 78.5823 | 67.8418 | 10.7405 |
| 15 | 57 | DS-R1-1.5B | GSM8K | 12 | 78.8602 | 67.5512 | 11.3091 |
| 16 | 58 | DS-R1-1.5B | GSM8K | 12 | 79.0371 | 68.1703 | 10.8668 |
| 17 | 59 | DS-R1-1.5B | GSM8K | 12 | 78.9108 | 67.7028 | 11.208 |
| 18 | 60 | DS-R1-1.5B | GSM8K | 12 | 78.557 | 68.3346 | 10.2224 |
| 19 | 61 | DS-R1-1.5B | GSM8K | 12 | 79.0498 | 67.6396 | 11.4102 |
| 20 | 62 | DS-R1-1.5B | GSM8K | 12 | 78.9108 | 67.7154 | 11.1954 |
| 21 | 63 | DS-R1-1.5B | GSM8K | 12 | 78.4938 | 67.8671 | 10.6267 |
| 22 | 64 | DS-R1-1.5B | GSM8K | 12 | 78.6707 | 67.8292 | 10.8415 |
| 23 | 65 | DS-R1-1.5B | GSM8K | 12 | 79.0877 | 67.6775 | 11.4102 |
| 24 | 66 | DS-R1-1.5B | GSM8K | 12 | 78.6707 | 67.8544 | 10.8163 |
| 25 | 67 | DS-R1-1.5B | GSM8K | 12 | 78.8729 | 67.6396 | 11.2333 |
| 26 | 68 | DS-R1-1.5B | GSM8K | 12 | 78.8729 | 67.9934 | 10.8795 |
| 27 | 69 | DS-R1-1.5B | GSM8K | 12 | 78.6454 | 67.7028 | 10.9426 |
| 28 | 70 | DS-R1-1.5B | GSM8K | 12 | 78.4938 | 68.322 | 10.1718 |
| 29 | 71 | DS-R1-1.5B | GSM8K | 12 | 79.0119 | 67.9555 | 11.0564 |
| 30 | 72 | DS-R1-1.5B | GSM8K | 12 | 78.8223 | 67.9682 | 10.8542 |
| 31 | 73 | DS-R1-1.5B | GSM8K | 12 | 78.8476 | 67.6902 | 11.1574 |
| 32 | 74 | DS-R1-1.5B | GSM8K | 12 | 78.5444 | 67.5891 | 10.9553 |
| 33 | 75 | DS-R1-1.5B | GSM8K | 12 | 79.1509 | 67.9555 | 11.1954 |
| 34 | 76 | DS-R1-1.5B | GSM8K | 12 | 78.7465 | 68.0566 | 10.6899 |
| 35 | 77 | DS-R1-1.5B | GSM8K | 12 | 78.8223 | 67.627 | 11.1954 |
| 36 | 78 | DS-R1-1.5B | GSM8K | 12 | 78.7465 | 67.905 | 10.8415 |
| 37 | 79 | DS-R1-1.5B | GSM8K | 12 | 79.0245 | 68.3346 | 10.6899 |
| 38 | 80 | DS-R1-1.5B | GSM8K | 12 | 78.9487 | 67.9555 | 10.9932 |
| 39 | 81 | DS-R1-1.5B | GSM8K | 12 | 78.6328 | 67.6775 | 10.9553 |
| 40 | 82 | DS-R1-1.5B | GSM8K | 12 | 78.8602 | 67.5638 | 11.2964 |
| 41 | 83 | DS-R1-1.5B | GSM8K | 12 | 78.557 | 67.8039 | 10.7531 |
| 42 | 84 | DS-R1-1.5B | GSM8K | 12 | 78.9866 | 67.7913 | 11.1954 |
| 43 | 85 | DS-R1-1.5B | GSM8K | 12 | 78.8729 | 67.9682 | 10.9047 |
| 44 | 86 | DS-R1-1.5B | GSM8K | 12 | 78.6328 | 67.8797 | 10.7531 |
| 45 | 87 | DS-R1-1.5B | GSM8K | 12 | 78.6833 | 68.1198 | 10.5636 |
| 46 | 88 | DS-R1-1.5B | GSM8K | 12 | 78.4812 | 67.7154 | 10.7657 |
| 47 | 89 | DS-R1-1.5B | GSM8K | 12 | 78.6328 | 67.7786 | 10.8542 |
| 48 | 90 | DS-R1-1.5B | GSM8K | 12 | 78.7339 | 68.0566 | 10.6773 |
| 49 | 91 | DS-R1-1.5B | GSM8K | 12 | 78.6707 | 67.6902 | 10.9805 |
| 0 | 42 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 1 | 43 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 2 | 44 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 3 | 45 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 4 | 46 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 5 | 47 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 6 | 48 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 7 | 49 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 8 | 50 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 9 | 51 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 10 | 52 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 11 | 53 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 12 | 54 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 13 | 55 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 14 | 56 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 15 | 57 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 16 | 58 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 17 | 59 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 18 | 60 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 19 | 61 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 20 | 62 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 21 | 63 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 22 | 64 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 23 | 65 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 24 | 66 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 25 | 67 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 26 | 68 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 27 | 69 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 28 | 70 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 29 | 71 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 30 | 72 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 31 | 73 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 32 | 74 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 33 | 75 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 34 | 76 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 35 | 77 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 36 | 78 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 37 | 79 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 38 | 80 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 39 | 81 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 40 | 82 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 41 | 83 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 42 | 84 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 43 | 85 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 44 | 86 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 45 | 87 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 46 | 88 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 47 | 89 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 48 | 90 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 49 | 91 | DS-R1-1.5B | GSM8K | 16 | 78.8476 | 67.8071 | 11.0406 |
| 0 | 42 | DS-R1-1.5B | MATH500 | 4 | 59.9 | 48.3 | 11.6 |
| 1 | 43 | DS-R1-1.5B | MATH500 | 4 | 60.1 | 45.9 | 14.2 |
| 2 | 44 | DS-R1-1.5B | MATH500 | 4 | 59.5 | 47 | 12.5 |
| 3 | 45 | DS-R1-1.5B | MATH500 | 4 | 60 | 47.1 | 12.9 |
| 4 | 46 | DS-R1-1.5B | MATH500 | 4 | 60.9 | 47.4 | 13.5 |
| 5 | 47 | DS-R1-1.5B | MATH500 | 4 | 61.6 | 49.6 | 12 |
| 6 | 48 | DS-R1-1.5B | MATH500 | 4 | 60 | 48.5 | 11.5 |
| 7 | 49 | DS-R1-1.5B | MATH500 | 4 | 58.4 | 47.2 | 11.2 |
| 8 | 50 | DS-R1-1.5B | MATH500 | 4 | 61.7 | 48.9 | 12.8 |
| 9 | 51 | DS-R1-1.5B | MATH500 | 4 | 59.7 | 49.2 | 10.5 |
| 10 | 52 | DS-R1-1.5B | MATH500 | 4 | 59.4 | 46.3 | 13.1 |
| 11 | 53 | DS-R1-1.5B | MATH500 | 4 | 61 | 48.5 | 12.5 |
| 12 | 54 | DS-R1-1.5B | MATH500 | 4 | 59.3 | 45.5 | 13.8 |
| 13 | 55 | DS-R1-1.5B | MATH500 | 4 | 59.5 | 47.6 | 11.9 |
| 14 | 56 | DS-R1-1.5B | MATH500 | 4 | 59.1 | 46.6 | 12.5 |
| 15 | 57 | DS-R1-1.5B | MATH500 | 4 | 59.6 | 46.8 | 12.8 |
| 16 | 58 | DS-R1-1.5B | MATH500 | 4 | 59.1 | 48.8 | 10.3 |
| 17 | 59 | DS-R1-1.5B | MATH500 | 4 | 58.5 | 46.6 | 11.9 |
| 18 | 60 | DS-R1-1.5B | MATH500 | 4 | 60.1 | 47.1 | 13 |
| 19 | 61 | DS-R1-1.5B | MATH500 | 4 | 60.5 | 48.9 | 11.6 |
| 20 | 62 | DS-R1-1.5B | MATH500 | 4 | 59.3 | 48.9 | 10.4 |
| 21 | 63 | DS-R1-1.5B | MATH500 | 4 | 59.6 | 48.2 | 11.4 |
| 22 | 64 | DS-R1-1.5B | MATH500 | 4 | 60.7 | 47.5 | 13.2 |
| 23 | 65 | DS-R1-1.5B | MATH500 | 4 | 59.5 | 48.3 | 11.2 |
| 24 | 66 | DS-R1-1.5B | MATH500 | 4 | 59.8 | 48.7 | 11.1 |
| 25 | 67 | DS-R1-1.5B | MATH500 | 4 | 59.2 | 47.8 | 11.4 |
| 26 | 68 | DS-R1-1.5B | MATH500 | 4 | 60.3 | 46.9 | 13.4 |
| 27 | 69 | DS-R1-1.5B | MATH500 | 4 | 61.3 | 49.4 | 11.9 |
| 28 | 70 | DS-R1-1.5B | MATH500 | 4 | 59.6 | 47 | 12.6 |
| 29 | 71 | DS-R1-1.5B | MATH500 | 4 | 60.6 | 47.7 | 12.9 |
| 30 | 72 | DS-R1-1.5B | MATH500 | 4 | 59.1 | 47.9 | 11.2 |
| 31 | 73 | DS-R1-1.5B | MATH500 | 4 | 59.7 | 48.6 | 11.1 |
| 32 | 74 | DS-R1-1.5B | MATH500 | 4 | 59.4 | 46.7 | 12.7 |
| 33 | 75 | DS-R1-1.5B | MATH500 | 4 | 59.6 | 47.7 | 11.9 |
| 34 | 76 | DS-R1-1.5B | MATH500 | 4 | 60.9 | 45.9 | 15 |
| 35 | 77 | DS-R1-1.5B | MATH500 | 4 | 60.5 | 48.2 | 12.3 |
| 36 | 78 | DS-R1-1.5B | MATH500 | 4 | 60.7 | 47.6 | 13.1 |
| 37 | 79 | DS-R1-1.5B | MATH500 | 4 | 61.2 | 46.9 | 14.3 |
| 38 | 80 | DS-R1-1.5B | MATH500 | 4 | 60.3 | 47.1 | 13.2 |
| 39 | 81 | DS-R1-1.5B | MATH500 | 4 | 60.3 | 45.7 | 14.6 |
| 40 | 82 | DS-R1-1.5B | MATH500 | 4 | 60.7 | 46.7 | 14 |
| 41 | 83 | DS-R1-1.5B | MATH500 | 4 | 60.2 | 47.6 | 12.6 |
| 42 | 84 | DS-R1-1.5B | MATH500 | 4 | 58.8 | 48.3 | 10.5 |
| 43 | 85 | DS-R1-1.5B | MATH500 | 4 | 60.9 | 47.3 | 13.6 |
| 44 | 86 | DS-R1-1.5B | MATH500 | 4 | 59.5 | 49 | 10.5 |
| 45 | 87 | DS-R1-1.5B | MATH500 | 4 | 60.4 | 46.3 | 14.1 |
| 46 | 88 | DS-R1-1.5B | MATH500 | 4 | 59.7 | 48.1 | 11.6 |
| 47 | 89 | DS-R1-1.5B | MATH500 | 4 | 60.7 | 46.6 | 14.1 |
| 48 | 90 | DS-R1-1.5B | MATH500 | 4 | 60.1 | 48.6 | 11.5 |
| 49 | 91 | DS-R1-1.5B | MATH500 | 4 | 62 | 46.8 | 15.2 |
| 0 | 42 | DS-R1-1.5B | MATH500 | 8 | 61.1 | 47 | 14.1 |
| 1 | 43 | DS-R1-1.5B | MATH500 | 8 | 61.2 | 47.25 | 13.95 |
| 2 | 44 | DS-R1-1.5B | MATH500 | 8 | 60.05 | 46.9 | 13.15 |
| 3 | 45 | DS-R1-1.5B | MATH500 | 8 | 60.5 | 46.95 | 13.55 |
| 4 | 46 | DS-R1-1.5B | MATH500 | 8 | 60.65 | 46.15 | 14.5 |
| 5 | 47 | DS-R1-1.5B | MATH500 | 8 | 61.6 | 47.55 | 14.05 |
| 6 | 48 | DS-R1-1.5B | MATH500 | 8 | 61.15 | 46.75 | 14.4 |
| 7 | 49 | DS-R1-1.5B | MATH500 | 8 | 60.15 | 46.65 | 13.5 |
| 8 | 50 | DS-R1-1.5B | MATH500 | 8 | 61.2 | 47.55 | 13.65 |
| 9 | 51 | DS-R1-1.5B | MATH500 | 8 | 60.8 | 47.9 | 12.9 |
| 10 | 52 | DS-R1-1.5B | MATH500 | 8 | 60.85 | 47.1 | 13.75 |
| 11 | 53 | DS-R1-1.5B | MATH500 | 8 | 61.8 | 46.35 | 15.45 |
| 12 | 54 | DS-R1-1.5B | MATH500 | 8 | 60.25 | 45.1 | 15.15 |
| 13 | 55 | DS-R1-1.5B | MATH500 | 8 | 61 | 46.5 | 14.5 |
| 14 | 56 | DS-R1-1.5B | MATH500 | 8 | 61 | 46.55 | 14.45 |
| 15 | 57 | DS-R1-1.5B | MATH500 | 8 | 60.7 | 45.9 | 14.8 |
| 16 | 58 | DS-R1-1.5B | MATH500 | 8 | 60 | 47.1 | 12.9 |
| 17 | 59 | DS-R1-1.5B | MATH500 | 8 | 60.55 | 47.6 | 12.95 |
| 18 | 60 | DS-R1-1.5B | MATH500 | 8 | 61.6 | 46.25 | 15.35 |
| 19 | 61 | DS-R1-1.5B | MATH500 | 8 | 60.3 | 47.6 | 12.7 |
| 20 | 62 | DS-R1-1.5B | MATH500 | 8 | 59.85 | 47.95 | 11.9 |
| 21 | 63 | DS-R1-1.5B | MATH500 | 8 | 60.65 | 47.4 | 13.25 |
| 22 | 64 | DS-R1-1.5B | MATH500 | 8 | 60.9 | 47.25 | 13.65 |
| 23 | 65 | DS-R1-1.5B | MATH500 | 8 | 61.1 | 47.1 | 14 |
| 24 | 66 | DS-R1-1.5B | MATH500 | 8 | 60.65 | 46.45 | 14.2 |
| 25 | 67 | DS-R1-1.5B | MATH500 | 8 | 61.25 | 45.45 | 15.8 |
| 26 | 68 | DS-R1-1.5B | MATH500 | 8 | 61.65 | 46.5 | 15.15 |
| 27 | 69 | DS-R1-1.5B | MATH500 | 8 | 60.55 | 48.15 | 12.4 |
| 28 | 70 | DS-R1-1.5B | MATH500 | 8 | 61.1 | 47.45 | 13.65 |
| 29 | 71 | DS-R1-1.5B | MATH500 | 8 | 60.7 | 46.9 | 13.8 |
| 30 | 72 | DS-R1-1.5B | MATH500 | 8 | 60.1 | 46.85 | 13.25 |
| 31 | 73 | DS-R1-1.5B | MATH500 | 8 | 60.5 | 47.6 | 12.9 |
| 32 | 74 | DS-R1-1.5B | MATH500 | 8 | 60.7 | 46.65 | 14.05 |
| 33 | 75 | DS-R1-1.5B | MATH500 | 8 | 60.9 | 46.7 | 14.2 |
| 34 | 76 | DS-R1-1.5B | MATH500 | 8 | 60.85 | 46.8 | 14.05 |
| 35 | 77 | DS-R1-1.5B | MATH500 | 8 | 60.3 | 46.95 | 13.35 |
| 36 | 78 | DS-R1-1.5B | MATH500 | 8 | 61.15 | 47.4 | 13.75 |
| 37 | 79 | DS-R1-1.5B | MATH500 | 8 | 61.65 | 46.8 | 14.85 |
| 38 | 80 | DS-R1-1.5B | MATH500 | 8 | 60.7 | 46.15 | 14.55 |
| 39 | 81 | DS-R1-1.5B | MATH500 | 8 | 60.45 | 46.85 | 13.6 |
| 40 | 82 | DS-R1-1.5B | MATH500 | 8 | 61.2 | 47.05 | 14.15 |
| 41 | 83 | DS-R1-1.5B | MATH500 | 8 | 61.85 | 46.35 | 15.5 |
| 42 | 84 | DS-R1-1.5B | MATH500 | 8 | 60.5 | 46.95 | 13.55 |
| 43 | 85 | DS-R1-1.5B | MATH500 | 8 | 60.7 | 46.25 | 14.45 |
| 44 | 86 | DS-R1-1.5B | MATH500 | 8 | 60.6 | 46.65 | 13.95 |
| 45 | 87 | DS-R1-1.5B | MATH500 | 8 | 60.55 | 46.5 | 14.05 |
| 46 | 88 | DS-R1-1.5B | MATH500 | 8 | 60.85 | 47.5 | 13.35 |
| 47 | 89 | DS-R1-1.5B | MATH500 | 8 | 61.25 | 47.3 | 13.95 |
| 48 | 90 | DS-R1-1.5B | MATH500 | 8 | 60.7 | 47.75 | 12.95 |
| 49 | 91 | DS-R1-1.5B | MATH500 | 8 | 61.05 | 47.45 | 13.6 |
| 0 | 42 | DS-R1-1.5B | MATH500 | 12 | 60.9667 | 46.9333 | 14.0333 |
| 1 | 43 | DS-R1-1.5B | MATH500 | 12 | 61.3 | 47.3333 | 13.9667 |
| 2 | 44 | DS-R1-1.5B | MATH500 | 12 | 60.5667 | 46.4667 | 14.1 |
| 3 | 45 | DS-R1-1.5B | MATH500 | 12 | 60.9667 | 46.9333 | 14.0333 |
| 4 | 46 | DS-R1-1.5B | MATH500 | 12 | 60.6333 | 46.7333 | 13.9 |
| 5 | 47 | DS-R1-1.5B | MATH500 | 12 | 61.4333 | 47.1667 | 14.2667 |
| 6 | 48 | DS-R1-1.5B | MATH500 | 12 | 61.3 | 46.3333 | 14.9667 |
| 7 | 49 | DS-R1-1.5B | MATH500 | 12 | 60.6667 | 46.7 | 13.9667 |
| 8 | 50 | DS-R1-1.5B | MATH500 | 12 | 60.7333 | 47.2333 | 13.5 |
| 9 | 51 | DS-R1-1.5B | MATH500 | 12 | 60.8667 | 47.0333 | 13.8333 |
| 10 | 52 | DS-R1-1.5B | MATH500 | 12 | 60.8 | 46.3333 | 14.4667 |
| 11 | 53 | DS-R1-1.5B | MATH500 | 12 | 61.5 | 47.2667 | 14.2333 |
| 12 | 54 | DS-R1-1.5B | MATH500 | 12 | 61.4667 | 46 | 15.4667 |
| 13 | 55 | DS-R1-1.5B | MATH500 | 12 | 60.9 | 46.5 | 14.4 |
| 14 | 56 | DS-R1-1.5B | MATH500 | 12 | 61.3667 | 46.6333 | 14.7333 |
| 15 | 57 | DS-R1-1.5B | MATH500 | 12 | 60.8667 | 46.7667 | 14.1 |
| 16 | 58 | DS-R1-1.5B | MATH500 | 12 | 60.6333 | 47.2 | 13.4333 |
| 17 | 59 | DS-R1-1.5B | MATH500 | 12 | 60.9333 | 46.9667 | 13.9667 |
| 18 | 60 | DS-R1-1.5B | MATH500 | 12 | 61.2667 | 46.5 | 14.7667 |
| 19 | 61 | DS-R1-1.5B | MATH500 | 12 | 60.7667 | 46.9667 | 13.8 |
| 20 | 62 | DS-R1-1.5B | MATH500 | 12 | 60.6667 | 47.3333 | 13.3333 |
| 21 | 63 | DS-R1-1.5B | MATH500 | 12 | 60.8667 | 46.7667 | 14.1 |
| 22 | 64 | DS-R1-1.5B | MATH500 | 12 | 60.8667 | 46.8 | 14.0667 |
| 23 | 65 | DS-R1-1.5B | MATH500 | 12 | 60.6667 | 46.9667 | 13.7 |
| 24 | 66 | DS-R1-1.5B | MATH500 | 12 | 61.0333 | 46.9333 | 14.1 |
| 25 | 67 | DS-R1-1.5B | MATH500 | 12 | 61 | 46.4333 | 14.5667 |
| 26 | 68 | DS-R1-1.5B | MATH500 | 12 | 61.2 | 46.9667 | 14.2333 |
| 27 | 69 | DS-R1-1.5B | MATH500 | 12 | 60.6667 | 47.2333 | 13.4333 |
| 28 | 70 | DS-R1-1.5B | MATH500 | 12 | 61.2 | 47.2 | 14 |
| 29 | 71 | DS-R1-1.5B | MATH500 | 12 | 61.0667 | 46.3667 | 14.7 |
| 30 | 72 | DS-R1-1.5B | MATH500 | 12 | 61.1667 | 46.4667 | 14.7 |
| 31 | 73 | DS-R1-1.5B | MATH500 | 12 | 60.5 | 47.2 | 13.3 |
| 32 | 74 | DS-R1-1.5B | MATH500 | 12 | 61.1333 | 46.7333 | 14.4 |
| 33 | 75 | DS-R1-1.5B | MATH500 | 12 | 60.7 | 46.8 | 13.9 |
| 34 | 76 | DS-R1-1.5B | MATH500 | 12 | 60.6333 | 47 | 13.6333 |
| 35 | 77 | DS-R1-1.5B | MATH500 | 12 | 60.9333 | 46.8 | 14.1333 |
| 36 | 78 | DS-R1-1.5B | MATH500 | 12 | 61.2667 | 46.5667 | 14.7 |
| 37 | 79 | DS-R1-1.5B | MATH500 | 12 | 61.6 | 46.9667 | 14.6333 |
| 38 | 80 | DS-R1-1.5B | MATH500 | 12 | 61.2333 | 46.5667 | 14.6667 |
| 39 | 81 | DS-R1-1.5B | MATH500 | 12 | 61.2 | 46.9 | 14.3 |
| 40 | 82 | DS-R1-1.5B | MATH500 | 12 | 61.0333 | 46.9667 | 14.0667 |
| 41 | 83 | DS-R1-1.5B | MATH500 | 12 | 61.1333 | 46.4333 | 14.7 |
| 42 | 84 | DS-R1-1.5B | MATH500 | 12 | 61.1667 | 46.3333 | 14.8333 |
| 43 | 85 | DS-R1-1.5B | MATH500 | 12 | 60.8333 | 46.8667 | 13.9667 |
| 44 | 86 | DS-R1-1.5B | MATH500 | 12 | 60.6333 | 46.9667 | 13.6667 |
| 45 | 87 | DS-R1-1.5B | MATH500 | 12 | 61.0667 | 46.4 | 14.6667 |
| 46 | 88 | DS-R1-1.5B | MATH500 | 12 | 61.5 | 46.9667 | 14.5333 |
| 47 | 89 | DS-R1-1.5B | MATH500 | 12 | 61.2 | 46.7 | 14.5 |
| 48 | 90 | DS-R1-1.5B | MATH500 | 12 | 61.1333 | 46.7333 | 14.4 |
| 49 | 91 | DS-R1-1.5B | MATH500 | 12 | 61.0667 | 46.8333 | 14.2333 |
| 0 | 42 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 1 | 43 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 2 | 44 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 3 | 45 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 4 | 46 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 5 | 47 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 6 | 48 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 7 | 49 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 8 | 50 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 9 | 51 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 10 | 52 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 11 | 53 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 12 | 54 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 13 | 55 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 14 | 56 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 15 | 57 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 16 | 58 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 17 | 59 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 18 | 60 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 19 | 61 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 20 | 62 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 21 | 63 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 22 | 64 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 23 | 65 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 24 | 66 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 25 | 67 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 26 | 68 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 27 | 69 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 28 | 70 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 29 | 71 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 30 | 72 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 31 | 73 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 32 | 74 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 33 | 75 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 34 | 76 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 35 | 77 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 36 | 78 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 37 | 79 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 38 | 80 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 39 | 81 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 40 | 82 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 41 | 83 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 42 | 84 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 43 | 85 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 44 | 86 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 45 | 87 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 46 | 88 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 47 | 89 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 48 | 90 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 49 | 91 | DS-R1-1.5B | MATH500 | 16 | 61.05 | 46.75 | 14.3 |
| 0 | 42 | DS-R1-1.5B | SVAMP | 4 | 84.55 | 79.4 | 5.15 |
| 1 | 43 | DS-R1-1.5B | SVAMP | 4 | 83.05 | 79.15 | 3.9 |
| 2 | 44 | DS-R1-1.5B | SVAMP | 4 | 84.2 | 80.75 | 3.45 |
| 3 | 45 | DS-R1-1.5B | SVAMP | 4 | 84 | 78.5 | 5.5 |
| 4 | 46 | DS-R1-1.5B | SVAMP | 4 | 83.3 | 79 | 4.3 |
| 5 | 47 | DS-R1-1.5B | SVAMP | 4 | 82.95 | 80.4 | 2.55 |
| 6 | 48 | DS-R1-1.5B | SVAMP | 4 | 83.05 | 79.5 | 3.55 |
| 7 | 49 | DS-R1-1.5B | SVAMP | 4 | 83.6 | 80.1 | 3.5 |
| 8 | 50 | DS-R1-1.5B | SVAMP | 4 | 83.8 | 79.35 | 4.45 |
| 9 | 51 | DS-R1-1.5B | SVAMP | 4 | 83.85 | 79.95 | 3.9 |
| 10 | 52 | DS-R1-1.5B | SVAMP | 4 | 84.65 | 79.05 | 5.6 |
| 11 | 53 | DS-R1-1.5B | SVAMP | 4 | 83.7 | 78.9 | 4.8 |
| 12 | 54 | DS-R1-1.5B | SVAMP | 4 | 82.9 | 80.35 | 2.55 |
| 13 | 55 | DS-R1-1.5B | SVAMP | 4 | 84.05 | 80.05 | 4 |
| 14 | 56 | DS-R1-1.5B | SVAMP | 4 | 84.1 | 79.05 | 5.05 |
| 15 | 57 | DS-R1-1.5B | SVAMP | 4 | 82.65 | 79.45 | 3.2 |
| 16 | 58 | DS-R1-1.5B | SVAMP | 4 | 83.25 | 79.4 | 3.85 |
| 17 | 59 | DS-R1-1.5B | SVAMP | 4 | 84.1 | 79.15 | 4.95 |
| 18 | 60 | DS-R1-1.5B | SVAMP | 4 | 83.75 | 79.6 | 4.15 |
| 19 | 61 | DS-R1-1.5B | SVAMP | 4 | 83.8 | 79.4 | 4.4 |
| 20 | 62 | DS-R1-1.5B | SVAMP | 4 | 83.75 | 79.85 | 3.9 |
| 21 | 63 | DS-R1-1.5B | SVAMP | 4 | 82.95 | 78.9 | 4.05 |
| 22 | 64 | DS-R1-1.5B | SVAMP | 4 | 84.25 | 79.6 | 4.65 |
| 23 | 65 | DS-R1-1.5B | SVAMP | 4 | 82.65 | 79.5 | 3.15 |
| 24 | 66 | DS-R1-1.5B | SVAMP | 4 | 83.9 | 80.2 | 3.7 |
| 25 | 67 | DS-R1-1.5B | SVAMP | 4 | 83.7 | 78.6 | 5.1 |
| 26 | 68 | DS-R1-1.5B | SVAMP | 4 | 84.1 | 79.4 | 4.7 |
| 27 | 69 | DS-R1-1.5B | SVAMP | 4 | 83.5 | 78.65 | 4.85 |
| 28 | 70 | DS-R1-1.5B | SVAMP | 4 | 83.95 | 78.9 | 5.05 |
| 29 | 71 | DS-R1-1.5B | SVAMP | 4 | 83.65 | 79.1 | 4.55 |
| 30 | 72 | DS-R1-1.5B | SVAMP | 4 | 83.2 | 79.55 | 3.65 |
| 31 | 73 | DS-R1-1.5B | SVAMP | 4 | 84.4 | 80 | 4.4 |
| 32 | 74 | DS-R1-1.5B | SVAMP | 4 | 83.3 | 80.1 | 3.2 |
| 33 | 75 | DS-R1-1.5B | SVAMP | 4 | 84.1 | 78.95 | 5.15 |
| 34 | 76 | DS-R1-1.5B | SVAMP | 4 | 82.95 | 78.7 | 4.25 |
| 35 | 77 | DS-R1-1.5B | SVAMP | 4 | 84.15 | 79.4 | 4.75 |
| 36 | 78 | DS-R1-1.5B | SVAMP | 4 | 83.6 | 79.9 | 3.7 |
| 37 | 79 | DS-R1-1.5B | SVAMP | 4 | 83.25 | 79.75 | 3.5 |
| 38 | 80 | DS-R1-1.5B | SVAMP | 4 | 83.15 | 78.8 | 4.35 |
| 39 | 81 | DS-R1-1.5B | SVAMP | 4 | 83.1 | 79.85 | 3.25 |
| 40 | 82 | DS-R1-1.5B | SVAMP | 4 | 83.3 | 79.45 | 3.85 |
| 41 | 83 | DS-R1-1.5B | SVAMP | 4 | 82.25 | 80.35 | 1.9 |
| 42 | 84 | DS-R1-1.5B | SVAMP | 4 | 83.65 | 78.65 | 5 |
| 43 | 85 | DS-R1-1.5B | SVAMP | 4 | 83.45 | 79.6 | 3.85 |
| 44 | 86 | DS-R1-1.5B | SVAMP | 4 | 83.7 | 79.35 | 4.35 |
| 45 | 87 | DS-R1-1.5B | SVAMP | 4 | 83.5 | 80 | 3.5 |
| 46 | 88 | DS-R1-1.5B | SVAMP | 4 | 83.1 | 79.15 | 3.95 |
| 47 | 89 | DS-R1-1.5B | SVAMP | 4 | 83.35 | 78.45 | 4.9 |
| 48 | 90 | DS-R1-1.5B | SVAMP | 4 | 82.75 | 79.35 | 3.4 |
| 49 | 91 | DS-R1-1.5B | SVAMP | 4 | 83.35 | 80.05 | 3.3 |
| 0 | 42 | DS-R1-1.5B | SVAMP | 8 | 84.425 | 79.25 | 5.175 |
| 1 | 43 | DS-R1-1.5B | SVAMP | 8 | 83.075 | 79.3 | 3.775 |
| 2 | 44 | DS-R1-1.5B | SVAMP | 8 | 84.05 | 80.2 | 3.85 |
| 3 | 45 | DS-R1-1.5B | SVAMP | 8 | 84.175 | 78.975 | 5.2 |
| 4 | 46 | DS-R1-1.5B | SVAMP | 8 | 83.725 | 79.475 | 4.25 |
| 5 | 47 | DS-R1-1.5B | SVAMP | 8 | 84 | 79.4 | 4.6 |
| 6 | 48 | DS-R1-1.5B | SVAMP | 8 | 83.675 | 79.575 | 4.1 |
| 7 | 49 | DS-R1-1.5B | SVAMP | 8 | 83.575 | 79.625 | 3.95 |
| 8 | 50 | DS-R1-1.5B | SVAMP | 8 | 84 | 79.275 | 4.725 |
| 9 | 51 | DS-R1-1.5B | SVAMP | 8 | 83.875 | 79.375 | 4.5 |
| 10 | 52 | DS-R1-1.5B | SVAMP | 8 | 84.275 | 78.85 | 5.425 |
| 11 | 53 | DS-R1-1.5B | SVAMP | 8 | 83.925 | 79.6 | 4.325 |
| 12 | 54 | DS-R1-1.5B | SVAMP | 8 | 83.425 | 79.8 | 3.625 |
| 13 | 55 | DS-R1-1.5B | SVAMP | 8 | 83.35 | 80.1 | 3.25 |
| 14 | 56 | DS-R1-1.5B | SVAMP | 8 | 84.15 | 78.725 | 5.425 |
| 15 | 57 | DS-R1-1.5B | SVAMP | 8 | 83.525 | 79.275 | 4.25 |
| 16 | 58 | DS-R1-1.5B | SVAMP | 8 | 84.05 | 79.15 | 4.9 |
| 17 | 59 | DS-R1-1.5B | SVAMP | 8 | 84.375 | 79.075 | 5.3 |
| 18 | 60 | DS-R1-1.5B | SVAMP | 8 | 83.95 | 79.525 | 4.425 |
| 19 | 61 | DS-R1-1.5B | SVAMP | 8 | 84.275 | 78.675 | 5.6 |
| 20 | 62 | DS-R1-1.5B | SVAMP | 8 | 83.9 | 79.275 | 4.625 |
| 21 | 63 | DS-R1-1.5B | SVAMP | 8 | 83.825 | 78.775 | 5.05 |
| 22 | 64 | DS-R1-1.5B | SVAMP | 8 | 84.275 | 79.45 | 4.825 |
| 23 | 65 | DS-R1-1.5B | SVAMP | 8 | 83.475 | 78.525 | 4.95 |
| 24 | 66 | DS-R1-1.5B | SVAMP | 8 | 84.2 | 79.275 | 4.925 |
| 25 | 67 | DS-R1-1.5B | SVAMP | 8 | 84.2 | 79.025 | 5.175 |
| 26 | 68 | DS-R1-1.5B | SVAMP | 8 | 84.3 | 79.575 | 4.725 |
| 27 | 69 | DS-R1-1.5B | SVAMP | 8 | 83.725 | 78.975 | 4.75 |
| 28 | 70 | DS-R1-1.5B | SVAMP | 8 | 83.975 | 79.25 | 4.725 |
| 29 | 71 | DS-R1-1.5B | SVAMP | 8 | 84.175 | 79.1 | 5.075 |
| 30 | 72 | DS-R1-1.5B | SVAMP | 8 | 84.05 | 79.425 | 4.625 |
| 31 | 73 | DS-R1-1.5B | SVAMP | 8 | 84 | 79.375 | 4.625 |
| 32 | 74 | DS-R1-1.5B | SVAMP | 8 | 83.75 | 79.475 | 4.275 |
| 33 | 75 | DS-R1-1.5B | SVAMP | 8 | 84.5 | 79.275 | 5.225 |
| 34 | 76 | DS-R1-1.5B | SVAMP | 8 | 83.975 | 78.975 | 5 |
| 35 | 77 | DS-R1-1.5B | SVAMP | 8 | 84.1 | 79.15 | 4.95 |
| 36 | 78 | DS-R1-1.5B | SVAMP | 8 | 83.9 | 78.85 | 5.05 |
| 37 | 79 | DS-R1-1.5B | SVAMP | 8 | 83.675 | 79.425 | 4.25 |
| 38 | 80 | DS-R1-1.5B | SVAMP | 8 | 84.125 | 78.9 | 5.225 |
| 39 | 81 | DS-R1-1.5B | SVAMP | 8 | 84.175 | 79.225 | 4.95 |
| 40 | 82 | DS-R1-1.5B | SVAMP | 8 | 83.875 | 79.475 | 4.4 |
| 41 | 83 | DS-R1-1.5B | SVAMP | 8 | 83.375 | 80.05 | 3.325 |
| 42 | 84 | DS-R1-1.5B | SVAMP | 8 | 83.9 | 79.35 | 4.55 |
| 43 | 85 | DS-R1-1.5B | SVAMP | 8 | 84.05 | 79.325 | 4.725 |
| 44 | 86 | DS-R1-1.5B | SVAMP | 8 | 84.025 | 78.7 | 5.325 |
| 45 | 87 | DS-R1-1.5B | SVAMP | 8 | 84 | 79.4 | 4.6 |
| 46 | 88 | DS-R1-1.5B | SVAMP | 8 | 84.15 | 78.975 | 5.175 |
| 47 | 89 | DS-R1-1.5B | SVAMP | 8 | 83.75 | 79.8 | 3.95 |
| 48 | 90 | DS-R1-1.5B | SVAMP | 8 | 83.25 | 79.625 | 3.625 |
| 49 | 91 | DS-R1-1.5B | SVAMP | 8 | 84.2 | 79.3 | 4.9 |
| 0 | 42 | DS-R1-1.5B | SVAMP | 12 | 84.15 | 79.0167 | 5.1333 |
| 1 | 43 | DS-R1-1.5B | SVAMP | 12 | 83.8667 | 79.05 | 4.8167 |
| 2 | 44 | DS-R1-1.5B | SVAMP | 12 | 84.3 | 79.3667 | 4.9333 |
| 3 | 45 | DS-R1-1.5B | SVAMP | 12 | 84.0667 | 78.9667 | 5.1 |
| 4 | 46 | DS-R1-1.5B | SVAMP | 12 | 84.0833 | 79.5333 | 4.55 |
| 5 | 47 | DS-R1-1.5B | SVAMP | 12 | 84.2 | 78.9833 | 5.2167 |
| 6 | 48 | DS-R1-1.5B | SVAMP | 12 | 84.1167 | 79.1667 | 4.95 |
| 7 | 49 | DS-R1-1.5B | SVAMP | 12 | 83.5667 | 79.1667 | 4.4 |
| 8 | 50 | DS-R1-1.5B | SVAMP | 12 | 84.1833 | 78.9333 | 5.25 |
| 9 | 51 | DS-R1-1.5B | SVAMP | 12 | 83.9833 | 79.05 | 4.9333 |
| 10 | 52 | DS-R1-1.5B | SVAMP | 12 | 84 | 79.2 | 4.8 |
| 11 | 53 | DS-R1-1.5B | SVAMP | 12 | 84.0167 | 79.0667 | 4.95 |
| 12 | 54 | DS-R1-1.5B | SVAMP | 12 | 84 | 79.5167 | 4.4833 |
| 13 | 55 | DS-R1-1.5B | SVAMP | 12 | 83.7333 | 79.4667 | 4.2667 |
| 14 | 56 | DS-R1-1.5B | SVAMP | 12 | 84.4833 | 78.6833 | 5.8 |
| 15 | 57 | DS-R1-1.5B | SVAMP | 12 | 83.85 | 79.1833 | 4.6667 |
| 16 | 58 | DS-R1-1.5B | SVAMP | 12 | 83.9667 | 79.05 | 4.9167 |
| 17 | 59 | DS-R1-1.5B | SVAMP | 12 | 84.2833 | 79.1833 | 5.1 |
| 18 | 60 | DS-R1-1.5B | SVAMP | 12 | 84.2 | 79.2833 | 4.9167 |
| 19 | 61 | DS-R1-1.5B | SVAMP | 12 | 84.1833 | 78.7 | 5.4833 |
| 20 | 62 | DS-R1-1.5B | SVAMP | 12 | 83.95 | 79.15 | 4.8 |
| 21 | 63 | DS-R1-1.5B | SVAMP | 12 | 83.9833 | 79.0333 | 4.95 |
| 22 | 64 | DS-R1-1.5B | SVAMP | 12 | 84.05 | 79.45 | 4.6 |
| 23 | 65 | DS-R1-1.5B | SVAMP | 12 | 84.1167 | 78.9833 | 5.1333 |
| 24 | 66 | DS-R1-1.5B | SVAMP | 12 | 83.7333 | 79.2 | 4.5333 |
| 25 | 67 | DS-R1-1.5B | SVAMP | 12 | 84.2 | 79.0833 | 5.1167 |
| 26 | 68 | DS-R1-1.5B | SVAMP | 12 | 84.0667 | 78.9167 | 5.15 |
| 27 | 69 | DS-R1-1.5B | SVAMP | 12 | 83.8833 | 78.9167 | 4.9667 |
| 28 | 70 | DS-R1-1.5B | SVAMP | 12 | 84.0833 | 79.15 | 4.9333 |
| 29 | 71 | DS-R1-1.5B | SVAMP | 12 | 84.0833 | 79.1667 | 4.9167 |
| 30 | 72 | DS-R1-1.5B | SVAMP | 12 | 84.1167 | 79.45 | 4.6667 |
| 31 | 73 | DS-R1-1.5B | SVAMP | 12 | 84.35 | 79.0333 | 5.3167 |
| 32 | 74 | DS-R1-1.5B | SVAMP | 12 | 83.8 | 79.2333 | 4.5667 |
| 33 | 75 | DS-R1-1.5B | SVAMP | 12 | 84.3 | 79.3333 | 4.9667 |
| 34 | 76 | DS-R1-1.5B | SVAMP | 12 | 84.0333 | 78.9 | 5.1333 |
| 35 | 77 | DS-R1-1.5B | SVAMP | 12 | 84 | 78.8667 | 5.1333 |
| 36 | 78 | DS-R1-1.5B | SVAMP | 12 | 84.25 | 78.3333 | 5.9167 |
| 37 | 79 | DS-R1-1.5B | SVAMP | 12 | 83.8833 | 78.85 | 5.0333 |
| 38 | 80 | DS-R1-1.5B | SVAMP | 12 | 84.25 | 79 | 5.25 |
| 39 | 81 | DS-R1-1.5B | SVAMP | 12 | 84.2667 | 78.9167 | 5.35 |
| 40 | 82 | DS-R1-1.5B | SVAMP | 12 | 83.9667 | 79.3 | 4.6667 |
| 41 | 83 | DS-R1-1.5B | SVAMP | 12 | 83.8 | 79.7667 | 4.0333 |
| 42 | 84 | DS-R1-1.5B | SVAMP | 12 | 84.05 | 79.0667 | 4.9833 |
| 43 | 85 | DS-R1-1.5B | SVAMP | 12 | 83.95 | 79.3167 | 4.6333 |
| 44 | 86 | DS-R1-1.5B | SVAMP | 12 | 83.9833 | 78.85 | 5.1333 |
| 45 | 87 | DS-R1-1.5B | SVAMP | 12 | 83.85 | 79.1167 | 4.7333 |
| 46 | 88 | DS-R1-1.5B | SVAMP | 12 | 83.95 | 78.9333 | 5.0167 |
| 47 | 89 | DS-R1-1.5B | SVAMP | 12 | 84.3333 | 79.1833 | 5.15 |
| 48 | 90 | DS-R1-1.5B | SVAMP | 12 | 83.85 | 79.2833 | 4.5667 |
| 49 | 91 | DS-R1-1.5B | SVAMP | 12 | 84.0167 | 79.4 | 4.6167 |
| 0 | 42 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 1 | 43 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 2 | 44 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 3 | 45 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 4 | 46 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 5 | 47 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 6 | 48 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 7 | 49 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 8 | 50 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 9 | 51 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 10 | 52 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 11 | 53 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 12 | 54 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 13 | 55 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 14 | 56 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 15 | 57 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 16 | 58 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 17 | 59 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 18 | 60 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 19 | 61 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 20 | 62 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 21 | 63 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 22 | 64 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 23 | 65 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 24 | 66 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 25 | 67 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 26 | 68 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 27 | 69 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 28 | 70 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 29 | 71 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 30 | 72 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 31 | 73 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 32 | 74 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 33 | 75 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 34 | 76 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 35 | 77 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 36 | 78 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 37 | 79 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 38 | 80 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 39 | 81 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 40 | 82 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 41 | 83 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 42 | 84 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 43 | 85 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 44 | 86 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 45 | 87 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 46 | 88 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 47 | 89 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 48 | 90 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 49 | 91 | DS-R1-1.5B | SVAMP | 16 | 84.15 | 79.0375 | 5.1125 |
| 0 | 42 | DS-R1-7B | AQuA | 4 | 77.7559 | 66.1417 | 11.6142 |
| 1 | 43 | DS-R1-7B | AQuA | 4 | 76.9685 | 68.3071 | 8.6614 |
| 2 | 44 | DS-R1-7B | AQuA | 4 | 75.3937 | 66.5354 | 8.8583 |
| 3 | 45 | DS-R1-7B | AQuA | 4 | 74.2126 | 65.5512 | 8.6614 |
| 4 | 46 | DS-R1-7B | AQuA | 4 | 74.4094 | 67.3228 | 7.0866 |
| 5 | 47 | DS-R1-7B | AQuA | 4 | 75.3937 | 67.3228 | 8.0709 |
| 6 | 48 | DS-R1-7B | AQuA | 4 | 74.4094 | 69.2913 | 5.1181 |
| 7 | 49 | DS-R1-7B | AQuA | 4 | 72.8346 | 63.7795 | 9.0551 |
| 8 | 50 | DS-R1-7B | AQuA | 4 | 75.3937 | 67.7165 | 7.6772 |
| 9 | 51 | DS-R1-7B | AQuA | 4 | 74.6063 | 65.9449 | 8.6614 |
| 10 | 52 | DS-R1-7B | AQuA | 4 | 75.5906 | 66.3386 | 9.252 |
| 11 | 53 | DS-R1-7B | AQuA | 4 | 73.8189 | 68.7008 | 5.1181 |
| 12 | 54 | DS-R1-7B | AQuA | 4 | 74.4094 | 69.2913 | 5.1181 |
| 13 | 55 | DS-R1-7B | AQuA | 4 | 75.7874 | 66.1417 | 9.6457 |
| 14 | 56 | DS-R1-7B | AQuA | 4 | 77.1654 | 65.748 | 11.4173 |
| 15 | 57 | DS-R1-7B | AQuA | 4 | 75.3937 | 69.8819 | 5.5118 |
| 16 | 58 | DS-R1-7B | AQuA | 4 | 73.2283 | 66.7323 | 6.4961 |
| 17 | 59 | DS-R1-7B | AQuA | 4 | 73.622 | 67.5197 | 6.1024 |
| 18 | 60 | DS-R1-7B | AQuA | 4 | 75.7874 | 66.1417 | 9.6457 |
| 19 | 61 | DS-R1-7B | AQuA | 4 | 75.1969 | 68.1102 | 7.0866 |
| 20 | 62 | DS-R1-7B | AQuA | 4 | 73.0315 | 66.5354 | 6.4961 |
| 21 | 63 | DS-R1-7B | AQuA | 4 | 75.5906 | 65.5512 | 10.0394 |
| 22 | 64 | DS-R1-7B | AQuA | 4 | 74.2126 | 67.9134 | 6.2992 |
| 23 | 65 | DS-R1-7B | AQuA | 4 | 75.1969 | 64.9606 | 10.2362 |
| 24 | 66 | DS-R1-7B | AQuA | 4 | 75 | 68.3071 | 6.6929 |
| 25 | 67 | DS-R1-7B | AQuA | 4 | 74.0157 | 66.1417 | 7.874 |
| 26 | 68 | DS-R1-7B | AQuA | 4 | 73.0315 | 69.4882 | 3.5433 |
| 27 | 69 | DS-R1-7B | AQuA | 4 | 75.3937 | 68.8976 | 6.4961 |
| 28 | 70 | DS-R1-7B | AQuA | 4 | 75.3937 | 66.1417 | 9.252 |
| 29 | 71 | DS-R1-7B | AQuA | 4 | 73.4252 | 66.9291 | 6.4961 |
| 30 | 72 | DS-R1-7B | AQuA | 4 | 73.4252 | 68.7008 | 4.7244 |
| 31 | 73 | DS-R1-7B | AQuA | 4 | 74.4094 | 69.685 | 4.7244 |
| 32 | 74 | DS-R1-7B | AQuA | 4 | 76.7717 | 65.9449 | 10.8268 |
| 33 | 75 | DS-R1-7B | AQuA | 4 | 73.4252 | 67.7165 | 5.7087 |
| 34 | 76 | DS-R1-7B | AQuA | 4 | 75.9843 | 69.685 | 6.2992 |
| 35 | 77 | DS-R1-7B | AQuA | 4 | 74.2126 | 66.1417 | 8.0709 |
| 36 | 78 | DS-R1-7B | AQuA | 4 | 74.4094 | 67.5197 | 6.8898 |
| 37 | 79 | DS-R1-7B | AQuA | 4 | 76.1811 | 69.2913 | 6.8898 |
| 38 | 80 | DS-R1-7B | AQuA | 4 | 75 | 69.4882 | 5.5118 |
| 39 | 81 | DS-R1-7B | AQuA | 4 | 76.1811 | 66.9291 | 9.252 |
| 40 | 82 | DS-R1-7B | AQuA | 4 | 75.1969 | 67.7165 | 7.4803 |
| 41 | 83 | DS-R1-7B | AQuA | 4 | 72.8346 | 69.2913 | 3.5433 |
| 42 | 84 | DS-R1-7B | AQuA | 4 | 75.3937 | 68.5039 | 6.8898 |
| 43 | 85 | DS-R1-7B | AQuA | 4 | 75.1969 | 65.748 | 9.4488 |
| 44 | 86 | DS-R1-7B | AQuA | 4 | 75.3937 | 68.1102 | 7.2835 |
| 45 | 87 | DS-R1-7B | AQuA | 4 | 76.1811 | 67.126 | 9.0551 |
| 46 | 88 | DS-R1-7B | AQuA | 4 | 74.6063 | 65.9449 | 8.6614 |
| 47 | 89 | DS-R1-7B | AQuA | 4 | 73.622 | 67.3228 | 6.2992 |
| 48 | 90 | DS-R1-7B | AQuA | 4 | 73.8189 | 69.685 | 4.1339 |
| 49 | 91 | DS-R1-7B | AQuA | 4 | 76.378 | 70.4724 | 5.9055 |
| 0 | 42 | DS-R1-7B | AQuA | 8 | 76.9685 | 64.8622 | 12.1063 |
| 1 | 43 | DS-R1-7B | AQuA | 8 | 76.378 | 67.4213 | 8.9567 |
| 2 | 44 | DS-R1-7B | AQuA | 8 | 75.4921 | 65.9449 | 9.5472 |
| 3 | 45 | DS-R1-7B | AQuA | 8 | 75.689 | 65.4528 | 10.2362 |
| 4 | 46 | DS-R1-7B | AQuA | 8 | 75.4921 | 68.4055 | 7.0866 |
| 5 | 47 | DS-R1-7B | AQuA | 8 | 75.5906 | 66.6339 | 8.9567 |
| 6 | 48 | DS-R1-7B | AQuA | 8 | 74.7047 | 66.9291 | 7.7756 |
| 7 | 49 | DS-R1-7B | AQuA | 8 | 73.5236 | 66.437 | 7.0866 |
| 8 | 50 | DS-R1-7B | AQuA | 8 | 76.378 | 67.2244 | 9.1535 |
| 9 | 51 | DS-R1-7B | AQuA | 8 | 75.8858 | 66.3386 | 9.5472 |
| 10 | 52 | DS-R1-7B | AQuA | 8 | 75.8858 | 66.0433 | 9.8425 |
| 11 | 53 | DS-R1-7B | AQuA | 8 | 75.1969 | 67.126 | 8.0709 |
| 12 | 54 | DS-R1-7B | AQuA | 8 | 75.2953 | 68.1102 | 7.185 |
| 13 | 55 | DS-R1-7B | AQuA | 8 | 76.0827 | 65.9449 | 10.1378 |
| 14 | 56 | DS-R1-7B | AQuA | 8 | 76.6732 | 66.7323 | 9.9409 |
| 15 | 57 | DS-R1-7B | AQuA | 8 | 75.689 | 67.5197 | 8.1693 |
| 16 | 58 | DS-R1-7B | AQuA | 8 | 75.689 | 66.9291 | 8.7598 |
| 17 | 59 | DS-R1-7B | AQuA | 8 | 74.0157 | 67.7165 | 6.2992 |
| 18 | 60 | DS-R1-7B | AQuA | 8 | 75.1969 | 66.1417 | 9.0551 |
| 19 | 61 | DS-R1-7B | AQuA | 8 | 75.8858 | 67.0276 | 8.8583 |
| 20 | 62 | DS-R1-7B | AQuA | 8 | 74.5079 | 65.9449 | 8.563 |
| 21 | 63 | DS-R1-7B | AQuA | 8 | 74.9016 | 66.2402 | 8.6614 |
| 22 | 64 | DS-R1-7B | AQuA | 8 | 75.3937 | 67.7165 | 7.6772 |
| 23 | 65 | DS-R1-7B | AQuA | 8 | 74.8031 | 66.0433 | 8.7598 |
| 24 | 66 | DS-R1-7B | AQuA | 8 | 74.5079 | 67.7165 | 6.7913 |
| 25 | 67 | DS-R1-7B | AQuA | 8 | 74.8031 | 66.3386 | 8.4646 |
| 26 | 68 | DS-R1-7B | AQuA | 8 | 74.5079 | 68.2087 | 6.2992 |
| 27 | 69 | DS-R1-7B | AQuA | 8 | 75.689 | 68.5039 | 7.185 |
| 28 | 70 | DS-R1-7B | AQuA | 8 | 75.3937 | 66.437 | 8.9567 |
| 29 | 71 | DS-R1-7B | AQuA | 8 | 75 | 66.5354 | 8.4646 |
| 30 | 72 | DS-R1-7B | AQuA | 8 | 75.5906 | 67.0276 | 8.563 |
| 31 | 73 | DS-R1-7B | AQuA | 8 | 73.3268 | 68.6024 | 4.7244 |
| 32 | 74 | DS-R1-7B | AQuA | 8 | 76.1811 | 66.1417 | 10.0394 |
| 33 | 75 | DS-R1-7B | AQuA | 8 | 75.1969 | 66.8307 | 8.3661 |
| 34 | 76 | DS-R1-7B | AQuA | 8 | 76.0827 | 68.3071 | 7.7756 |
| 35 | 77 | DS-R1-7B | AQuA | 8 | 74.0157 | 66.437 | 7.5787 |
| 36 | 78 | DS-R1-7B | AQuA | 8 | 75.2953 | 67.6181 | 7.6772 |
| 37 | 79 | DS-R1-7B | AQuA | 8 | 76.2795 | 67.126 | 9.1535 |
| 38 | 80 | DS-R1-7B | AQuA | 8 | 75.9843 | 67.0276 | 8.9567 |
| 39 | 81 | DS-R1-7B | AQuA | 8 | 75.4921 | 66.8307 | 8.6614 |
| 40 | 82 | DS-R1-7B | AQuA | 8 | 74.1142 | 67.4213 | 6.6929 |
| 41 | 83 | DS-R1-7B | AQuA | 8 | 74.1142 | 67.9134 | 6.2008 |
| 42 | 84 | DS-R1-7B | AQuA | 8 | 74.9016 | 67.3228 | 7.5787 |
| 43 | 85 | DS-R1-7B | AQuA | 8 | 76.4764 | 66.0433 | 10.4331 |
| 44 | 86 | DS-R1-7B | AQuA | 8 | 75.5906 | 67.5197 | 8.0709 |
| 45 | 87 | DS-R1-7B | AQuA | 8 | 75.689 | 67.2244 | 8.4646 |
| 46 | 88 | DS-R1-7B | AQuA | 8 | 74.0157 | 66.2402 | 7.7756 |
| 47 | 89 | DS-R1-7B | AQuA | 8 | 74.9016 | 66.9291 | 7.9724 |
| 48 | 90 | DS-R1-7B | AQuA | 8 | 74.1142 | 68.7008 | 5.4134 |
| 49 | 91 | DS-R1-7B | AQuA | 8 | 76.4764 | 68.5039 | 7.9724 |
| 0 | 42 | DS-R1-7B | AQuA | 12 | 76.1811 | 66.0761 | 10.105 |
| 1 | 43 | DS-R1-7B | AQuA | 12 | 75.3937 | 66.4042 | 8.9895 |
| 2 | 44 | DS-R1-7B | AQuA | 12 | 74.9344 | 66.7979 | 8.1365 |
| 3 | 45 | DS-R1-7B | AQuA | 12 | 75.853 | 66.0761 | 9.7769 |
| 4 | 46 | DS-R1-7B | AQuA | 12 | 75.5249 | 67.5197 | 8.0052 |
| 5 | 47 | DS-R1-7B | AQuA | 12 | 75.7874 | 65.9449 | 9.8425 |
| 6 | 48 | DS-R1-7B | AQuA | 12 | 75.6562 | 67.7822 | 7.874 |
| 7 | 49 | DS-R1-7B | AQuA | 12 | 74.4094 | 67.126 | 7.2835 |
| 8 | 50 | DS-R1-7B | AQuA | 12 | 75.5906 | 67.0604 | 8.5302 |
| 9 | 51 | DS-R1-7B | AQuA | 12 | 75.2625 | 66.6667 | 8.5958 |
| 10 | 52 | DS-R1-7B | AQuA | 12 | 75.5249 | 66.0761 | 9.4488 |
| 11 | 53 | DS-R1-7B | AQuA | 12 | 75.2625 | 66.8635 | 8.399 |
| 12 | 54 | DS-R1-7B | AQuA | 12 | 75.4593 | 67.2572 | 8.2021 |
| 13 | 55 | DS-R1-7B | AQuA | 12 | 75.2625 | 66.3386 | 8.9239 |
| 14 | 56 | DS-R1-7B | AQuA | 12 | 76.378 | 66.0761 | 10.3018 |
| 15 | 57 | DS-R1-7B | AQuA | 12 | 76.1811 | 67.126 | 9.0551 |
| 16 | 58 | DS-R1-7B | AQuA | 12 | 75.3281 | 66.5354 | 8.7927 |
| 17 | 59 | DS-R1-7B | AQuA | 12 | 75.2625 | 67.126 | 8.1365 |
| 18 | 60 | DS-R1-7B | AQuA | 12 | 75.3281 | 66.273 | 9.0551 |
| 19 | 61 | DS-R1-7B | AQuA | 12 | 75.6562 | 67.1916 | 8.4646 |
| 20 | 62 | DS-R1-7B | AQuA | 12 | 75.1969 | 66.0105 | 9.1864 |
| 21 | 63 | DS-R1-7B | AQuA | 12 | 75.4593 | 66.8635 | 8.5958 |
| 22 | 64 | DS-R1-7B | AQuA | 12 | 75 | 67.5853 | 7.4147 |
| 23 | 65 | DS-R1-7B | AQuA | 12 | 75.7218 | 66.5354 | 9.1864 |
| 24 | 66 | DS-R1-7B | AQuA | 12 | 74.2782 | 67.2572 | 7.021 |
| 25 | 67 | DS-R1-7B | AQuA | 12 | 75.5906 | 66.6667 | 8.9239 |
| 26 | 68 | DS-R1-7B | AQuA | 12 | 74.6719 | 67.3228 | 7.3491 |
| 27 | 69 | DS-R1-7B | AQuA | 12 | 75.5249 | 66.4042 | 9.1207 |
| 28 | 70 | DS-R1-7B | AQuA | 12 | 75.7874 | 66.9291 | 8.8583 |
| 29 | 71 | DS-R1-7B | AQuA | 12 | 75.5906 | 66.9291 | 8.6614 |
| 30 | 72 | DS-R1-7B | AQuA | 12 | 75.5249 | 67.0604 | 8.4646 |
| 31 | 73 | DS-R1-7B | AQuA | 12 | 74.8031 | 67.5197 | 7.2835 |
| 32 | 74 | DS-R1-7B | AQuA | 12 | 75.7218 | 66.4042 | 9.3176 |
| 33 | 75 | DS-R1-7B | AQuA | 12 | 76.1155 | 67.8478 | 8.2677 |
| 34 | 76 | DS-R1-7B | AQuA | 12 | 75.7874 | 67.5197 | 8.2677 |
| 35 | 77 | DS-R1-7B | AQuA | 12 | 75.4593 | 66.7979 | 8.6614 |
| 36 | 78 | DS-R1-7B | AQuA | 12 | 75.5906 | 66.7979 | 8.7927 |
| 37 | 79 | DS-R1-7B | AQuA | 12 | 75.9843 | 67.4541 | 8.5302 |
| 38 | 80 | DS-R1-7B | AQuA | 12 | 75.0656 | 67.5197 | 7.5459 |
| 39 | 81 | DS-R1-7B | AQuA | 12 | 75.4593 | 66.8635 | 8.5958 |
| 40 | 82 | DS-R1-7B | AQuA | 12 | 75 | 67.5197 | 7.4803 |
| 41 | 83 | DS-R1-7B | AQuA | 12 | 75.1312 | 66.9291 | 8.2021 |
| 42 | 84 | DS-R1-7B | AQuA | 12 | 75.0656 | 67.2572 | 7.8084 |
| 43 | 85 | DS-R1-7B | AQuA | 12 | 75.4593 | 66.8635 | 8.5958 |
| 44 | 86 | DS-R1-7B | AQuA | 12 | 76.2467 | 66.4042 | 9.8425 |
| 45 | 87 | DS-R1-7B | AQuA | 12 | 76.1155 | 66.601 | 9.5144 |
| 46 | 88 | DS-R1-7B | AQuA | 12 | 75.7218 | 65.5512 | 10.1706 |
| 47 | 89 | DS-R1-7B | AQuA | 12 | 75.9843 | 67.0604 | 8.9239 |
| 48 | 90 | DS-R1-7B | AQuA | 12 | 74.6719 | 67.4541 | 7.2178 |
| 49 | 91 | DS-R1-7B | AQuA | 12 | 75.853 | 67.0604 | 8.7927 |
| 0 | 42 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 1 | 43 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 2 | 44 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 3 | 45 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 4 | 46 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 5 | 47 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 6 | 48 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 7 | 49 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 8 | 50 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 9 | 51 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 10 | 52 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 11 | 53 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 12 | 54 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 13 | 55 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 14 | 56 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 15 | 57 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 16 | 58 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 17 | 59 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 18 | 60 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 19 | 61 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 20 | 62 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 21 | 63 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 22 | 64 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 23 | 65 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 24 | 66 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 25 | 67 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 26 | 68 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 27 | 69 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 28 | 70 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 29 | 71 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 30 | 72 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 31 | 73 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 32 | 74 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 33 | 75 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 34 | 76 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 35 | 77 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 36 | 78 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 37 | 79 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 38 | 80 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 39 | 81 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 40 | 82 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 41 | 83 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 42 | 84 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 43 | 85 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 44 | 86 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 45 | 87 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 46 | 88 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 47 | 89 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 48 | 90 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 49 | 91 | DS-R1-7B | AQuA | 16 | 75.2461 | 67.126 | 8.1201 |
| 0 | 42 | DS-R1-7B | CommonsenseQA | 4 | 49.3448 | 49.14 | 0.2048 |
| 1 | 43 | DS-R1-7B | CommonsenseQA | 4 | 49.181 | 48.9762 | 0.2048 |
| 2 | 44 | DS-R1-7B | CommonsenseQA | 4 | 49.6724 | 49.8362 | -0.1638 |
| 3 | 45 | DS-R1-7B | CommonsenseQA | 4 | 48.7715 | 49.5905 | -0.819 |
| 4 | 46 | DS-R1-7B | CommonsenseQA | 4 | 49.5495 | 49.6314 | -0.0819 |
| 5 | 47 | DS-R1-7B | CommonsenseQA | 4 | 48.8943 | 49.9181 | -1.0238 |
| 6 | 48 | DS-R1-7B | CommonsenseQA | 4 | 49.5495 | 48.0344 | 1.5152 |
| 7 | 49 | DS-R1-7B | CommonsenseQA | 4 | 48.1163 | 50.6962 | -2.5799 |
| 8 | 50 | DS-R1-7B | CommonsenseQA | 4 | 49.181 | 51.1876 | -2.0066 |
| 9 | 51 | DS-R1-7B | CommonsenseQA | 4 | 48.9353 | 48.8534 | 0.0819 |
| 10 | 52 | DS-R1-7B | CommonsenseQA | 4 | 49.7543 | 50.1229 | -0.3686 |
| 11 | 53 | DS-R1-7B | CommonsenseQA | 4 | 49.4676 | 50.2048 | -0.7371 |
| 12 | 54 | DS-R1-7B | CommonsenseQA | 4 | 49.3038 | 50.041 | -0.7371 |
| 13 | 55 | DS-R1-7B | CommonsenseQA | 4 | 49.8771 | 49.7952 | 0.0819 |
| 14 | 56 | DS-R1-7B | CommonsenseQA | 4 | 48.1572 | 49.4676 | -1.3104 |
| 15 | 57 | DS-R1-7B | CommonsenseQA | 4 | 48.6486 | 50.5324 | -1.8837 |
| 16 | 58 | DS-R1-7B | CommonsenseQA | 4 | 49.6724 | 49.6314 | 0.041 |
| 17 | 59 | DS-R1-7B | CommonsenseQA | 4 | 49.14 | 49.6314 | -0.4914 |
| 18 | 60 | DS-R1-7B | CommonsenseQA | 4 | 48.5667 | 48.2801 | 0.2867 |
| 19 | 61 | DS-R1-7B | CommonsenseQA | 4 | 48.7305 | 49.9181 | -1.1876 |
| 20 | 62 | DS-R1-7B | CommonsenseQA | 4 | 49.6724 | 48.9762 | 0.6962 |
| 21 | 63 | DS-R1-7B | CommonsenseQA | 4 | 48.9762 | 50.1229 | -1.1466 |
| 22 | 64 | DS-R1-7B | CommonsenseQA | 4 | 49.7952 | 49.4267 | 0.3686 |
| 23 | 65 | DS-R1-7B | CommonsenseQA | 4 | 50.3276 | 48.321 | 2.0066 |
| 24 | 66 | DS-R1-7B | CommonsenseQA | 4 | 47.4201 | 48.9762 | -1.5561 |
| 25 | 67 | DS-R1-7B | CommonsenseQA | 4 | 50 | 47.5839 | 2.4161 |
| 26 | 68 | DS-R1-7B | CommonsenseQA | 4 | 49.3038 | 49.0991 | 0.2048 |
| 27 | 69 | DS-R1-7B | CommonsenseQA | 4 | 48.4848 | 50.3276 | -1.8428 |
| 28 | 70 | DS-R1-7B | CommonsenseQA | 4 | 49.4267 | 49.0172 | 0.4095 |
| 29 | 71 | DS-R1-7B | CommonsenseQA | 4 | 48.9353 | 50.1229 | -1.1876 |
| 30 | 72 | DS-R1-7B | CommonsenseQA | 4 | 48.9353 | 50.4914 | -1.5561 |
| 31 | 73 | DS-R1-7B | CommonsenseQA | 4 | 49.7543 | 49.3857 | 0.3686 |
| 32 | 74 | DS-R1-7B | CommonsenseQA | 4 | 49.7133 | 48.7305 | 0.9828 |
| 33 | 75 | DS-R1-7B | CommonsenseQA | 4 | 48.5258 | 50.1638 | -1.638 |
| 34 | 76 | DS-R1-7B | CommonsenseQA | 4 | 49.7543 | 49.8771 | -0.1229 |
| 35 | 77 | DS-R1-7B | CommonsenseQA | 4 | 48.7715 | 48.8124 | -0.041 |
| 36 | 78 | DS-R1-7B | CommonsenseQA | 4 | 49.4267 | 49.3448 | 0.0819 |
| 37 | 79 | DS-R1-7B | CommonsenseQA | 4 | 50.6552 | 49.6724 | 0.9828 |
| 38 | 80 | DS-R1-7B | CommonsenseQA | 4 | 49.0581 | 50.1638 | -1.1057 |
| 39 | 81 | DS-R1-7B | CommonsenseQA | 4 | 48.1163 | 50.4914 | -2.3751 |
| 40 | 82 | DS-R1-7B | CommonsenseQA | 4 | 47.8706 | 49.9181 | -2.0475 |
| 41 | 83 | DS-R1-7B | CommonsenseQA | 4 | 49.5905 | 48.362 | 1.2285 |
| 42 | 84 | DS-R1-7B | CommonsenseQA | 4 | 48.5667 | 50.4914 | -1.9247 |
| 43 | 85 | DS-R1-7B | CommonsenseQA | 4 | 49.8771 | 48.1163 | 1.7609 |
| 44 | 86 | DS-R1-7B | CommonsenseQA | 4 | 50.2867 | 48.9762 | 1.3104 |
| 45 | 87 | DS-R1-7B | CommonsenseQA | 4 | 48.1572 | 49.14 | -0.9828 |
| 46 | 88 | DS-R1-7B | CommonsenseQA | 4 | 49.3038 | 49.5086 | -0.2048 |
| 47 | 89 | DS-R1-7B | CommonsenseQA | 4 | 49.9181 | 49.6724 | 0.2457 |
| 48 | 90 | DS-R1-7B | CommonsenseQA | 4 | 47.7477 | 49.4267 | -1.679 |
| 49 | 91 | DS-R1-7B | CommonsenseQA | 4 | 48.7305 | 51.4742 | -2.7437 |
| 0 | 42 | DS-R1-7B | CommonsenseQA | 8 | 49.3038 | 49.4062 | -0.1024 |
| 1 | 43 | DS-R1-7B | CommonsenseQA | 8 | 49.8976 | 49.8157 | 0.0819 |
| 2 | 44 | DS-R1-7B | CommonsenseQA | 8 | 48.6691 | 49.9181 | -1.249 |
| 3 | 45 | DS-R1-7B | CommonsenseQA | 8 | 49.611 | 49.181 | 0.43 |
| 4 | 46 | DS-R1-7B | CommonsenseQA | 8 | 49.14 | 49.7543 | -0.6143 |
| 5 | 47 | DS-R1-7B | CommonsenseQA | 8 | 49.4881 | 49.7748 | -0.2867 |
| 6 | 48 | DS-R1-7B | CommonsenseQA | 8 | 49.3243 | 48.8943 | 0.43 |
| 7 | 49 | DS-R1-7B | CommonsenseQA | 8 | 48.2391 | 50.2867 | -2.0475 |
| 8 | 50 | DS-R1-7B | CommonsenseQA | 8 | 49.2629 | 49.7952 | -0.5324 |
| 9 | 51 | DS-R1-7B | CommonsenseQA | 8 | 48.9353 | 49.3243 | -0.389 |
| 10 | 52 | DS-R1-7B | CommonsenseQA | 8 | 49.7543 | 49.9181 | -0.1638 |
| 11 | 53 | DS-R1-7B | CommonsenseQA | 8 | 49.7338 | 49.5905 | 0.1433 |
| 12 | 54 | DS-R1-7B | CommonsenseQA | 8 | 49.4267 | 49.5086 | -0.0819 |
| 13 | 55 | DS-R1-7B | CommonsenseQA | 8 | 48.9148 | 49.8976 | -0.9828 |
| 14 | 56 | DS-R1-7B | CommonsenseQA | 8 | 49.3448 | 50.1843 | -0.8395 |
| 15 | 57 | DS-R1-7B | CommonsenseQA | 8 | 49.2834 | 50.1638 | -0.8804 |
| 16 | 58 | DS-R1-7B | CommonsenseQA | 8 | 49.0581 | 49.9181 | -0.86 |
| 17 | 59 | DS-R1-7B | CommonsenseQA | 8 | 49.5291 | 49.4676 | 0.0614 |
| 18 | 60 | DS-R1-7B | CommonsenseQA | 8 | 48.9558 | 49.1605 | -0.2048 |
| 19 | 61 | DS-R1-7B | CommonsenseQA | 8 | 48.4644 | 49.6314 | -1.1671 |
| 20 | 62 | DS-R1-7B | CommonsenseQA | 8 | 49.0991 | 49.0786 | 0.0205 |
| 21 | 63 | DS-R1-7B | CommonsenseQA | 8 | 48.9148 | 49.14 | -0.2252 |
| 22 | 64 | DS-R1-7B | CommonsenseQA | 8 | 49.7543 | 49.959 | -0.2048 |
| 23 | 65 | DS-R1-7B | CommonsenseQA | 8 | 49.4676 | 48.7101 | 0.7576 |
| 24 | 66 | DS-R1-7B | CommonsenseQA | 8 | 48.2391 | 49.3857 | -1.1466 |
| 25 | 67 | DS-R1-7B | CommonsenseQA | 8 | 49.5291 | 49.2629 | 0.2662 |
| 26 | 68 | DS-R1-7B | CommonsenseQA | 8 | 49.3243 | 49.611 | -0.2867 |
| 27 | 69 | DS-R1-7B | CommonsenseQA | 8 | 48.9762 | 49.0991 | -0.1229 |
| 28 | 70 | DS-R1-7B | CommonsenseQA | 8 | 49.0991 | 49.4472 | -0.3481 |
| 29 | 71 | DS-R1-7B | CommonsenseQA | 8 | 48.8124 | 49.959 | -1.1466 |
| 30 | 72 | DS-R1-7B | CommonsenseQA | 8 | 48.4029 | 50.5528 | -2.1499 |
| 31 | 73 | DS-R1-7B | CommonsenseQA | 8 | 49.2219 | 49.4062 | -0.1843 |
| 32 | 74 | DS-R1-7B | CommonsenseQA | 8 | 49.3038 | 49.8157 | -0.5119 |
| 33 | 75 | DS-R1-7B | CommonsenseQA | 8 | 49.4472 | 49.6314 | -0.1843 |
| 34 | 76 | DS-R1-7B | CommonsenseQA | 8 | 48.9148 | 49.8771 | -0.9623 |
| 35 | 77 | DS-R1-7B | CommonsenseQA | 8 | 49.14 | 48.9353 | 0.2048 |
| 36 | 78 | DS-R1-7B | CommonsenseQA | 8 | 49.3038 | 48.9967 | 0.3071 |
| 37 | 79 | DS-R1-7B | CommonsenseQA | 8 | 49.4472 | 49.7952 | -0.3481 |
| 38 | 80 | DS-R1-7B | CommonsenseQA | 8 | 48.6486 | 50.2048 | -1.5561 |
| 39 | 81 | DS-R1-7B | CommonsenseQA | 8 | 48.792 | 49.7952 | -1.0033 |
| 40 | 82 | DS-R1-7B | CommonsenseQA | 8 | 49.14 | 50.0205 | -0.8804 |
| 41 | 83 | DS-R1-7B | CommonsenseQA | 8 | 49.959 | 48.9353 | 1.0238 |
| 42 | 84 | DS-R1-7B | CommonsenseQA | 8 | 49.4676 | 50.1638 | -0.6962 |
| 43 | 85 | DS-R1-7B | CommonsenseQA | 8 | 49.9386 | 48.8329 | 1.1057 |
| 44 | 86 | DS-R1-7B | CommonsenseQA | 8 | 49.3243 | 49.3448 | -0.0205 |
| 45 | 87 | DS-R1-7B | CommonsenseQA | 8 | 49.2015 | 48.6691 | 0.5324 |
| 46 | 88 | DS-R1-7B | CommonsenseQA | 8 | 49.5495 | 49.959 | -0.4095 |
| 47 | 89 | DS-R1-7B | CommonsenseQA | 8 | 49.959 | 49.2834 | 0.6757 |
| 48 | 90 | DS-R1-7B | CommonsenseQA | 8 | 48.7715 | 49.4676 | -0.6962 |
| 49 | 91 | DS-R1-7B | CommonsenseQA | 8 | 48.0549 | 50.7781 | -2.7232 |
| 0 | 42 | DS-R1-7B | CommonsenseQA | 12 | 48.9626 | 49.3175 | -0.3549 |
| 1 | 43 | DS-R1-7B | CommonsenseQA | 12 | 49.181 | 49.727 | -0.546 |
| 2 | 44 | DS-R1-7B | CommonsenseQA | 12 | 48.9899 | 49.3584 | -0.3686 |
| 3 | 45 | DS-R1-7B | CommonsenseQA | 12 | 48.9626 | 49.3994 | -0.4368 |
| 4 | 46 | DS-R1-7B | CommonsenseQA | 12 | 49.3857 | 49.8635 | -0.4778 |
| 5 | 47 | DS-R1-7B | CommonsenseQA | 12 | 49.0445 | 49.6451 | -0.6006 |
| 6 | 48 | DS-R1-7B | CommonsenseQA | 12 | 49.4813 | 49.0718 | 0.4095 |
| 7 | 49 | DS-R1-7B | CommonsenseQA | 12 | 49.1673 | 49.3857 | -0.2184 |
| 8 | 50 | DS-R1-7B | CommonsenseQA | 12 | 49.3448 | 49.4403 | -0.0956 |
| 9 | 51 | DS-R1-7B | CommonsenseQA | 12 | 49.0991 | 49.413 | -0.314 |
| 10 | 52 | DS-R1-7B | CommonsenseQA | 12 | 49.7816 | 49.5768 | 0.2048 |
| 11 | 53 | DS-R1-7B | CommonsenseQA | 12 | 49.5768 | 49.5632 | 0.0137 |
| 12 | 54 | DS-R1-7B | CommonsenseQA | 12 | 49.6314 | 49.4949 | 0.1365 |
| 13 | 55 | DS-R1-7B | CommonsenseQA | 12 | 49.0172 | 49.5632 | -0.546 |
| 14 | 56 | DS-R1-7B | CommonsenseQA | 12 | 48.7578 | 49.8635 | -1.1057 |
| 15 | 57 | DS-R1-7B | CommonsenseQA | 12 | 49.3448 | 49.4949 | -0.1502 |
| 16 | 58 | DS-R1-7B | CommonsenseQA | 12 | 49.181 | 49.8089 | -0.6279 |
| 17 | 59 | DS-R1-7B | CommonsenseQA | 12 | 49.1673 | 49.5086 | -0.3413 |
| 18 | 60 | DS-R1-7B | CommonsenseQA | 12 | 49.0854 | 49.6041 | -0.5187 |
| 19 | 61 | DS-R1-7B | CommonsenseQA | 12 | 48.9626 | 49.3584 | -0.3959 |
| 20 | 62 | DS-R1-7B | CommonsenseQA | 12 | 49.2629 | 49.2219 | 0.041 |
| 21 | 63 | DS-R1-7B | CommonsenseQA | 12 | 49.3448 | 49.3311 | 0.0137 |
| 22 | 64 | DS-R1-7B | CommonsenseQA | 12 | 49.2902 | 50 | -0.7098 |
| 23 | 65 | DS-R1-7B | CommonsenseQA | 12 | 49.6314 | 49.14 | 0.4914 |
| 24 | 66 | DS-R1-7B | CommonsenseQA | 12 | 49.2219 | 49.4813 | -0.2594 |
| 25 | 67 | DS-R1-7B | CommonsenseQA | 12 | 49.2765 | 49.3584 | -0.0819 |
| 26 | 68 | DS-R1-7B | CommonsenseQA | 12 | 49.7679 | 49.3721 | 0.3959 |
| 27 | 69 | DS-R1-7B | CommonsenseQA | 12 | 49.0718 | 49.6178 | -0.546 |
| 28 | 70 | DS-R1-7B | CommonsenseQA | 12 | 49.1127 | 49.5086 | -0.3959 |
| 29 | 71 | DS-R1-7B | CommonsenseQA | 12 | 49.2629 | 49.7406 | -0.4778 |
| 30 | 72 | DS-R1-7B | CommonsenseQA | 12 | 48.908 | 49.9044 | -0.9965 |
| 31 | 73 | DS-R1-7B | CommonsenseQA | 12 | 49.2219 | 49.7816 | -0.5597 |
| 32 | 74 | DS-R1-7B | CommonsenseQA | 12 | 49.0581 | 49.7133 | -0.6552 |
| 33 | 75 | DS-R1-7B | CommonsenseQA | 12 | 49.2356 | 49.6314 | -0.3959 |
| 34 | 76 | DS-R1-7B | CommonsenseQA | 12 | 49.2356 | 49.5086 | -0.273 |
| 35 | 77 | DS-R1-7B | CommonsenseQA | 12 | 49.181 | 49.413 | -0.2321 |
| 36 | 78 | DS-R1-7B | CommonsenseQA | 12 | 49.6178 | 49.413 | 0.2048 |
| 37 | 79 | DS-R1-7B | CommonsenseQA | 12 | 49.3175 | 49.5086 | -0.1911 |
| 38 | 80 | DS-R1-7B | CommonsenseQA | 12 | 49.413 | 49.5086 | -0.0956 |
| 39 | 81 | DS-R1-7B | CommonsenseQA | 12 | 49.1264 | 49.4267 | -0.3003 |
| 40 | 82 | DS-R1-7B | CommonsenseQA | 12 | 49.6587 | 49.3448 | 0.314 |
| 41 | 83 | DS-R1-7B | CommonsenseQA | 12 | 48.9489 | 49.3994 | -0.4505 |
| 42 | 84 | DS-R1-7B | CommonsenseQA | 12 | 49.413 | 49.8908 | -0.4778 |
| 43 | 85 | DS-R1-7B | CommonsenseQA | 12 | 49.3448 | 49.1946 | 0.1502 |
| 44 | 86 | DS-R1-7B | CommonsenseQA | 12 | 49.0854 | 49.6587 | -0.5733 |
| 45 | 87 | DS-R1-7B | CommonsenseQA | 12 | 49.2765 | 49.413 | -0.1365 |
| 46 | 88 | DS-R1-7B | CommonsenseQA | 12 | 49.413 | 49.5086 | -0.0956 |
| 47 | 89 | DS-R1-7B | CommonsenseQA | 12 | 49.2902 | 49.5905 | -0.3003 |
| 48 | 90 | DS-R1-7B | CommonsenseQA | 12 | 48.594 | 49.7543 | -1.1603 |
| 49 | 91 | DS-R1-7B | CommonsenseQA | 12 | 48.9216 | 49.6314 | -0.7098 |
| 0 | 42 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 1 | 43 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 2 | 44 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 3 | 45 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 4 | 46 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 5 | 47 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 6 | 48 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 7 | 49 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 8 | 50 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 9 | 51 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 10 | 52 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 11 | 53 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 12 | 54 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 13 | 55 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 14 | 56 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 15 | 57 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 16 | 58 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 17 | 59 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 18 | 60 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 19 | 61 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 20 | 62 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 21 | 63 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 22 | 64 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 23 | 65 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 24 | 66 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 25 | 67 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 26 | 68 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 27 | 69 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 28 | 70 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 29 | 71 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 30 | 72 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 31 | 73 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 32 | 74 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 33 | 75 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 34 | 76 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 35 | 77 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 36 | 78 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 37 | 79 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 38 | 80 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 39 | 81 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 40 | 82 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 41 | 83 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 42 | 84 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 43 | 85 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 44 | 86 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 45 | 87 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 46 | 88 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 47 | 89 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 48 | 90 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 49 | 91 | DS-R1-7B | CommonsenseQA | 16 | 49.396 | 49.3755 | 0.0205 |
| 0 | 42 | DS-R1-7B | GPQA | 4 | 44.8661 | 57.5893 | -12.7232 |
| 1 | 43 | DS-R1-7B | GPQA | 4 | 44.308 | 56.808 | -12.5 |
| 2 | 44 | DS-R1-7B | GPQA | 4 | 45.4241 | 56.808 | -11.3839 |
| 3 | 45 | DS-R1-7B | GPQA | 4 | 45.0893 | 58.4821 | -13.3929 |
| 4 | 46 | DS-R1-7B | GPQA | 4 | 44.7545 | 56.0268 | -11.2723 |
| 5 | 47 | DS-R1-7B | GPQA | 4 | 44.5312 | 55.9152 | -11.3839 |
| 6 | 48 | DS-R1-7B | GPQA | 4 | 45.4241 | 58.3705 | -12.9464 |
| 7 | 49 | DS-R1-7B | GPQA | 4 | 45.0893 | 56.25 | -11.1607 |
| 8 | 50 | DS-R1-7B | GPQA | 4 | 45.7589 | 55.9152 | -10.1562 |
| 9 | 51 | DS-R1-7B | GPQA | 4 | 45.7589 | 57.3661 | -11.6071 |
| 10 | 52 | DS-R1-7B | GPQA | 4 | 43.6384 | 57.4777 | -13.8393 |
| 11 | 53 | DS-R1-7B | GPQA | 4 | 47.3214 | 55.1339 | -7.8125 |
| 12 | 54 | DS-R1-7B | GPQA | 4 | 45.7589 | 58.2589 | -12.5 |
| 13 | 55 | DS-R1-7B | GPQA | 4 | 46.5402 | 55.8036 | -9.2634 |
| 14 | 56 | DS-R1-7B | GPQA | 4 | 43.6384 | 56.6964 | -13.058 |
| 15 | 57 | DS-R1-7B | GPQA | 4 | 46.2054 | 56.808 | -10.6027 |
| 16 | 58 | DS-R1-7B | GPQA | 4 | 44.0848 | 55.2455 | -11.1607 |
| 17 | 59 | DS-R1-7B | GPQA | 4 | 45.4241 | 58.0357 | -12.6116 |
| 18 | 60 | DS-R1-7B | GPQA | 4 | 44.0848 | 55.9152 | -11.8304 |
| 19 | 61 | DS-R1-7B | GPQA | 4 | 45.5357 | 56.808 | -11.2723 |
| 20 | 62 | DS-R1-7B | GPQA | 4 | 44.8661 | 55.3571 | -10.4911 |
| 21 | 63 | DS-R1-7B | GPQA | 4 | 43.4152 | 57.5893 | -14.1741 |
| 22 | 64 | DS-R1-7B | GPQA | 4 | 44.6429 | 57.0312 | -12.3884 |
| 23 | 65 | DS-R1-7B | GPQA | 4 | 43.0804 | 55.692 | -12.6116 |
| 24 | 66 | DS-R1-7B | GPQA | 4 | 43.9732 | 54.5759 | -10.6027 |
| 25 | 67 | DS-R1-7B | GPQA | 4 | 43.5268 | 56.4732 | -12.9464 |
| 26 | 68 | DS-R1-7B | GPQA | 4 | 46.317 | 58.1473 | -11.8304 |
| 27 | 69 | DS-R1-7B | GPQA | 4 | 42.2991 | 55.5804 | -13.2812 |
| 28 | 70 | DS-R1-7B | GPQA | 4 | 47.8795 | 58.7054 | -10.8259 |
| 29 | 71 | DS-R1-7B | GPQA | 4 | 47.5446 | 55.9152 | -8.3705 |
| 30 | 72 | DS-R1-7B | GPQA | 4 | 46.5402 | 57.2545 | -10.7143 |
| 31 | 73 | DS-R1-7B | GPQA | 4 | 44.6429 | 55.3571 | -10.7143 |
| 32 | 74 | DS-R1-7B | GPQA | 4 | 42.9688 | 55.3571 | -12.3884 |
| 33 | 75 | DS-R1-7B | GPQA | 4 | 46.4286 | 57.8125 | -11.3839 |
| 34 | 76 | DS-R1-7B | GPQA | 4 | 44.0848 | 56.4732 | -12.3884 |
| 35 | 77 | DS-R1-7B | GPQA | 4 | 43.9732 | 57.7009 | -13.7277 |
| 36 | 78 | DS-R1-7B | GPQA | 4 | 44.8661 | 55.8036 | -10.9375 |
| 37 | 79 | DS-R1-7B | GPQA | 4 | 42.6339 | 58.2589 | -15.625 |
| 38 | 80 | DS-R1-7B | GPQA | 4 | 45.3125 | 57.9241 | -12.6116 |
| 39 | 81 | DS-R1-7B | GPQA | 4 | 42.6339 | 56.3616 | -13.7277 |
| 40 | 82 | DS-R1-7B | GPQA | 4 | 43.6384 | 57.4777 | -13.8393 |
| 41 | 83 | DS-R1-7B | GPQA | 4 | 43.192 | 57.5893 | -14.3973 |
| 42 | 84 | DS-R1-7B | GPQA | 4 | 45.5357 | 54.4643 | -8.9286 |
| 43 | 85 | DS-R1-7B | GPQA | 4 | 46.317 | 58.4821 | -12.1652 |
| 44 | 86 | DS-R1-7B | GPQA | 4 | 46.875 | 58.3705 | -11.4955 |
| 45 | 87 | DS-R1-7B | GPQA | 4 | 44.9777 | 55.9152 | -10.9375 |
| 46 | 88 | DS-R1-7B | GPQA | 4 | 47.2098 | 56.6964 | -9.4866 |
| 47 | 89 | DS-R1-7B | GPQA | 4 | 45.0893 | 57.9241 | -12.8348 |
| 48 | 90 | DS-R1-7B | GPQA | 4 | 44.7545 | 57.9241 | -13.1696 |
| 49 | 91 | DS-R1-7B | GPQA | 4 | 45.4241 | 58.0357 | -12.6116 |
| 0 | 42 | DS-R1-7B | GPQA | 8 | 43.3594 | 57.6451 | -14.2857 |
| 1 | 43 | DS-R1-7B | GPQA | 8 | 43.8058 | 57.2545 | -13.4487 |
| 2 | 44 | DS-R1-7B | GPQA | 8 | 44.308 | 58.8728 | -14.5647 |
| 3 | 45 | DS-R1-7B | GPQA | 8 | 44.9777 | 58.0915 | -13.1138 |
| 4 | 46 | DS-R1-7B | GPQA | 8 | 44.0848 | 57.1987 | -13.1138 |
| 5 | 47 | DS-R1-7B | GPQA | 8 | 42.8571 | 58.0357 | -15.1786 |
| 6 | 48 | DS-R1-7B | GPQA | 8 | 45.0335 | 59.0402 | -14.0067 |
| 7 | 49 | DS-R1-7B | GPQA | 8 | 43.3036 | 58.0357 | -14.7321 |
| 8 | 50 | DS-R1-7B | GPQA | 8 | 45.1451 | 57.3661 | -12.221 |
| 9 | 51 | DS-R1-7B | GPQA | 8 | 44.029 | 57.9799 | -13.9509 |
| 10 | 52 | DS-R1-7B | GPQA | 8 | 44.5312 | 57.7009 | -13.1696 |
| 11 | 53 | DS-R1-7B | GPQA | 8 | 44.6987 | 56.4732 | -11.7746 |
| 12 | 54 | DS-R1-7B | GPQA | 8 | 43.2478 | 58.9286 | -15.6808 |
| 13 | 55 | DS-R1-7B | GPQA | 8 | 43.471 | 57.5335 | -14.0625 |
| 14 | 56 | DS-R1-7B | GPQA | 8 | 43.3594 | 57.8125 | -14.4531 |
| 15 | 57 | DS-R1-7B | GPQA | 8 | 44.9777 | 58.0357 | -13.058 |
| 16 | 58 | DS-R1-7B | GPQA | 8 | 42.2991 | 56.9196 | -14.6205 |
| 17 | 59 | DS-R1-7B | GPQA | 8 | 44.0848 | 57.5893 | -13.5045 |
| 18 | 60 | DS-R1-7B | GPQA | 8 | 44.5871 | 58.4821 | -13.8951 |
| 19 | 61 | DS-R1-7B | GPQA | 8 | 42.5781 | 57.2545 | -14.6763 |
| 20 | 62 | DS-R1-7B | GPQA | 8 | 43.471 | 56.8638 | -13.3929 |
| 21 | 63 | DS-R1-7B | GPQA | 8 | 42.0759 | 57.4777 | -15.4018 |
| 22 | 64 | DS-R1-7B | GPQA | 8 | 44.0848 | 58.2031 | -14.1183 |
| 23 | 65 | DS-R1-7B | GPQA | 8 | 44.3638 | 56.6964 | -12.3326 |
| 24 | 66 | DS-R1-7B | GPQA | 8 | 44.4196 | 56.1384 | -11.7188 |
| 25 | 67 | DS-R1-7B | GPQA | 8 | 44.1406 | 57.5893 | -13.4487 |
| 26 | 68 | DS-R1-7B | GPQA | 8 | 44.0848 | 58.2589 | -14.1741 |
| 27 | 69 | DS-R1-7B | GPQA | 8 | 43.192 | 56.9196 | -13.7277 |
| 28 | 70 | DS-R1-7B | GPQA | 8 | 45.1451 | 59.0402 | -13.8951 |
| 29 | 71 | DS-R1-7B | GPQA | 8 | 44.6429 | 58.3147 | -13.6719 |
| 30 | 72 | DS-R1-7B | GPQA | 8 | 44.029 | 57.4219 | -13.3929 |
| 31 | 73 | DS-R1-7B | GPQA | 8 | 43.0804 | 57.9241 | -14.8437 |
| 32 | 74 | DS-R1-7B | GPQA | 8 | 44.3638 | 54.3527 | -9.9888 |
| 33 | 75 | DS-R1-7B | GPQA | 8 | 45.3683 | 56.9754 | -11.6071 |
| 34 | 76 | DS-R1-7B | GPQA | 8 | 42.6339 | 56.6964 | -14.0625 |
| 35 | 77 | DS-R1-7B | GPQA | 8 | 44.5312 | 57.8125 | -13.2812 |
| 36 | 78 | DS-R1-7B | GPQA | 8 | 43.8058 | 57.5893 | -13.7835 |
| 37 | 79 | DS-R1-7B | GPQA | 8 | 42.9129 | 58.2589 | -15.346 |
| 38 | 80 | DS-R1-7B | GPQA | 8 | 44.5312 | 57.4219 | -12.8906 |
| 39 | 81 | DS-R1-7B | GPQA | 8 | 44.2522 | 57.0871 | -12.8348 |
| 40 | 82 | DS-R1-7B | GPQA | 8 | 43.5826 | 57.3103 | -13.7277 |
| 41 | 83 | DS-R1-7B | GPQA | 8 | 42.8013 | 58.4263 | -15.625 |
| 42 | 84 | DS-R1-7B | GPQA | 8 | 42.9129 | 58.3705 | -15.4576 |
| 43 | 85 | DS-R1-7B | GPQA | 8 | 44.4754 | 58.2031 | -13.7277 |
| 44 | 86 | DS-R1-7B | GPQA | 8 | 45.4241 | 58.6496 | -13.2254 |
| 45 | 87 | DS-R1-7B | GPQA | 8 | 42.9688 | 58.6496 | -15.6808 |
| 46 | 88 | DS-R1-7B | GPQA | 8 | 44.9219 | 57.8683 | -12.9464 |
| 47 | 89 | DS-R1-7B | GPQA | 8 | 43.9174 | 57.6451 | -13.7277 |
| 48 | 90 | DS-R1-7B | GPQA | 8 | 43.1362 | 56.9754 | -13.8393 |
| 49 | 91 | DS-R1-7B | GPQA | 8 | 43.9732 | 57.0312 | -13.058 |
| 0 | 42 | DS-R1-7B | GPQA | 12 | 43.0432 | 57.5521 | -14.5089 |
| 1 | 43 | DS-R1-7B | GPQA | 12 | 44.5312 | 57.5149 | -12.9836 |
| 2 | 44 | DS-R1-7B | GPQA | 12 | 43.1176 | 57.9985 | -14.881 |
| 3 | 45 | DS-R1-7B | GPQA | 12 | 43.8616 | 57.7753 | -13.9137 |
| 4 | 46 | DS-R1-7B | GPQA | 12 | 43.378 | 57.5893 | -14.2113 |
| 5 | 47 | DS-R1-7B | GPQA | 12 | 42.8571 | 58.9286 | -16.0714 |
| 6 | 48 | DS-R1-7B | GPQA | 12 | 44.1592 | 58.4449 | -14.2857 |
| 7 | 49 | DS-R1-7B | GPQA | 12 | 43.2292 | 57.7009 | -14.4717 |
| 8 | 50 | DS-R1-7B | GPQA | 12 | 44.122 | 57.7009 | -13.5789 |
| 9 | 51 | DS-R1-7B | GPQA | 12 | 43.3036 | 57.4777 | -14.1741 |
| 10 | 52 | DS-R1-7B | GPQA | 12 | 43.8616 | 57.6637 | -13.8021 |
| 11 | 53 | DS-R1-7B | GPQA | 12 | 43.7872 | 57.0685 | -13.2813 |
| 12 | 54 | DS-R1-7B | GPQA | 12 | 42.7455 | 58.1845 | -15.439 |
| 13 | 55 | DS-R1-7B | GPQA | 12 | 43.9732 | 57.5149 | -13.5417 |
| 14 | 56 | DS-R1-7B | GPQA | 12 | 43.6384 | 58.2961 | -14.6577 |
| 15 | 57 | DS-R1-7B | GPQA | 12 | 44.122 | 58.1101 | -13.9881 |
| 16 | 58 | DS-R1-7B | GPQA | 12 | 43.4152 | 57.5893 | -14.1741 |
| 17 | 59 | DS-R1-7B | GPQA | 12 | 43.4896 | 57.9241 | -14.4345 |
| 18 | 60 | DS-R1-7B | GPQA | 12 | 44.0104 | 58.5938 | -14.5833 |
| 19 | 61 | DS-R1-7B | GPQA | 12 | 43.9732 | 57.4405 | -13.4673 |
| 20 | 62 | DS-R1-7B | GPQA | 12 | 43.1548 | 58.0357 | -14.881 |
| 21 | 63 | DS-R1-7B | GPQA | 12 | 43.1176 | 58.5193 | -15.4018 |
| 22 | 64 | DS-R1-7B | GPQA | 12 | 43.8244 | 57.4777 | -13.6533 |
| 23 | 65 | DS-R1-7B | GPQA | 12 | 44.0476 | 57.2917 | -13.244 |
| 24 | 66 | DS-R1-7B | GPQA | 12 | 43.6384 | 56.8824 | -13.244 |
| 25 | 67 | DS-R1-7B | GPQA | 12 | 43.192 | 58.1845 | -14.9926 |
| 26 | 68 | DS-R1-7B | GPQA | 12 | 43.6756 | 57.6265 | -13.9509 |
| 27 | 69 | DS-R1-7B | GPQA | 12 | 43.6384 | 57.3289 | -13.6905 |
| 28 | 70 | DS-R1-7B | GPQA | 12 | 44.2708 | 58.1101 | -13.8393 |
| 29 | 71 | DS-R1-7B | GPQA | 12 | 44.5312 | 57.5893 | -13.058 |
| 30 | 72 | DS-R1-7B | GPQA | 12 | 43.6384 | 58.2217 | -14.5833 |
| 31 | 73 | DS-R1-7B | GPQA | 12 | 43.4896 | 58.3705 | -14.881 |
| 32 | 74 | DS-R1-7B | GPQA | 12 | 43.0804 | 57.1057 | -14.0253 |
| 33 | 75 | DS-R1-7B | GPQA | 12 | 44.0476 | 57.2173 | -13.1696 |
| 34 | 76 | DS-R1-7B | GPQA | 12 | 42.7083 | 57.2917 | -14.5833 |
| 35 | 77 | DS-R1-7B | GPQA | 12 | 43.1176 | 57.6637 | -14.5461 |
| 36 | 78 | DS-R1-7B | GPQA | 12 | 43.7128 | 57.5521 | -13.8393 |
| 37 | 79 | DS-R1-7B | GPQA | 12 | 43.75 | 57.3661 | -13.6161 |
| 38 | 80 | DS-R1-7B | GPQA | 12 | 44.0476 | 58.2961 | -14.2485 |
| 39 | 81 | DS-R1-7B | GPQA | 12 | 43.8616 | 57.9613 | -14.0997 |
| 40 | 82 | DS-R1-7B | GPQA | 12 | 43.5268 | 57.2917 | -13.7649 |
| 41 | 83 | DS-R1-7B | GPQA | 12 | 42.7455 | 59.0774 | -16.3318 |
| 42 | 84 | DS-R1-7B | GPQA | 12 | 42.8199 | 58.5193 | -15.6994 |
| 43 | 85 | DS-R1-7B | GPQA | 12 | 43.7872 | 58.0357 | -14.2485 |
| 44 | 86 | DS-R1-7B | GPQA | 12 | 43.4896 | 58.3705 | -14.881 |
| 45 | 87 | DS-R1-7B | GPQA | 12 | 44.3452 | 57.5149 | -13.1696 |
| 46 | 88 | DS-R1-7B | GPQA | 12 | 44.1592 | 58.1845 | -14.0253 |
| 47 | 89 | DS-R1-7B | GPQA | 12 | 43.564 | 58.4821 | -14.9182 |
| 48 | 90 | DS-R1-7B | GPQA | 12 | 43.6756 | 58.1473 | -14.4717 |
| 49 | 91 | DS-R1-7B | GPQA | 12 | 43.4524 | 57.7753 | -14.3229 |
| 0 | 42 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 1 | 43 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 2 | 44 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 3 | 45 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 4 | 46 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 5 | 47 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 6 | 48 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 7 | 49 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 8 | 50 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 9 | 51 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 10 | 52 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 11 | 53 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 12 | 54 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 13 | 55 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 14 | 56 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 15 | 57 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 16 | 58 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 17 | 59 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 18 | 60 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 19 | 61 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 20 | 62 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 21 | 63 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 22 | 64 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 23 | 65 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 24 | 66 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 25 | 67 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 26 | 68 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 27 | 69 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 28 | 70 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 29 | 71 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 30 | 72 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 31 | 73 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 32 | 74 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 33 | 75 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 34 | 76 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 35 | 77 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 36 | 78 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 37 | 79 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 38 | 80 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 39 | 81 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 40 | 82 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 41 | 83 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 42 | 84 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 43 | 85 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 44 | 86 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 45 | 87 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 46 | 88 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 47 | 89 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 48 | 90 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 49 | 91 | DS-R1-7B | GPQA | 16 | 43.5547 | 57.9241 | -14.3694 |
| 0 | 42 | DS-R1-7B | GSM8K | 4 | 90.5989 | 89.348 | 1.2509 |
| 1 | 43 | DS-R1-7B | GSM8K | 4 | 90.7506 | 90.182 | 0.5686 |
| 2 | 44 | DS-R1-7B | GSM8K | 4 | 90.7127 | 89.7271 | 0.9856 |
| 3 | 45 | DS-R1-7B | GSM8K | 4 | 90.8264 | 89.348 | 1.4784 |
| 4 | 46 | DS-R1-7B | GSM8K | 4 | 91.6603 | 89.8408 | 1.8196 |
| 5 | 47 | DS-R1-7B | GSM8K | 4 | 91.0538 | 90.2199 | 0.834 |
| 6 | 48 | DS-R1-7B | GSM8K | 4 | 90.9401 | 90.561 | 0.3791 |
| 7 | 49 | DS-R1-7B | GSM8K | 4 | 91.3192 | 89.9924 | 1.3268 |
| 8 | 50 | DS-R1-7B | GSM8K | 4 | 91.0159 | 89.9166 | 1.0993 |
| 9 | 51 | DS-R1-7B | GSM8K | 4 | 91.5087 | 90.144 | 1.3647 |
| 10 | 52 | DS-R1-7B | GSM8K | 4 | 90.6748 | 89.9924 | 0.6823 |
| 11 | 53 | DS-R1-7B | GSM8K | 4 | 90.5231 | 89.9166 | 0.6065 |
| 12 | 54 | DS-R1-7B | GSM8K | 4 | 90.8643 | 89.6892 | 1.1751 |
| 13 | 55 | DS-R1-7B | GSM8K | 4 | 90.8264 | 89.3859 | 1.4405 |
| 14 | 56 | DS-R1-7B | GSM8K | 4 | 91.0538 | 90.0682 | 0.9856 |
| 15 | 57 | DS-R1-7B | GSM8K | 4 | 90.182 | 89.2343 | 0.9477 |
| 16 | 58 | DS-R1-7B | GSM8K | 4 | 91.3192 | 89.2722 | 2.047 |
| 17 | 59 | DS-R1-7B | GSM8K | 4 | 91.1296 | 89.7271 | 1.4026 |
| 18 | 60 | DS-R1-7B | GSM8K | 4 | 90.8643 | 90.2578 | 0.6065 |
| 19 | 61 | DS-R1-7B | GSM8K | 4 | 90.3336 | 90.1061 | 0.2274 |
| 20 | 62 | DS-R1-7B | GSM8K | 4 | 90.144 | 89.7271 | 0.417 |
| 21 | 63 | DS-R1-7B | GSM8K | 4 | 90.8264 | 90.6368 | 0.1895 |
| 22 | 64 | DS-R1-7B | GSM8K | 4 | 90.9022 | 89.9924 | 0.9098 |
| 23 | 65 | DS-R1-7B | GSM8K | 4 | 91.3192 | 89.8408 | 1.4784 |
| 24 | 66 | DS-R1-7B | GSM8K | 4 | 90.6368 | 89.4617 | 1.1751 |
| 25 | 67 | DS-R1-7B | GSM8K | 4 | 91.0159 | 89.6513 | 1.3647 |
| 26 | 68 | DS-R1-7B | GSM8K | 4 | 90.9022 | 88.4382 | 2.464 |
| 27 | 69 | DS-R1-7B | GSM8K | 4 | 90.3336 | 88.9689 | 1.3647 |
| 28 | 70 | DS-R1-7B | GSM8K | 4 | 90.8643 | 89.8029 | 1.0614 |
| 29 | 71 | DS-R1-7B | GSM8K | 4 | 90.4094 | 90.3715 | 0.0379 |
| 30 | 72 | DS-R1-7B | GSM8K | 4 | 90.8643 | 90.3715 | 0.4928 |
| 31 | 73 | DS-R1-7B | GSM8K | 4 | 91.2055 | 89.9166 | 1.2889 |
| 32 | 74 | DS-R1-7B | GSM8K | 4 | 91.2434 | 90.0682 | 1.1751 |
| 33 | 75 | DS-R1-7B | GSM8K | 4 | 91.0917 | 89.6133 | 1.4784 |
| 34 | 76 | DS-R1-7B | GSM8K | 4 | 90.3715 | 89.765 | 0.6065 |
| 35 | 77 | DS-R1-7B | GSM8K | 4 | 90.3336 | 89.6513 | 0.6823 |
| 36 | 78 | DS-R1-7B | GSM8K | 4 | 90.2957 | 90.182 | 0.1137 |
| 37 | 79 | DS-R1-7B | GSM8K | 4 | 90.6368 | 89.7271 | 0.9098 |
| 38 | 80 | DS-R1-7B | GSM8K | 4 | 90.3715 | 89.3859 | 0.9856 |
| 39 | 81 | DS-R1-7B | GSM8K | 4 | 90.978 | 89.5375 | 1.4405 |
| 40 | 82 | DS-R1-7B | GSM8K | 4 | 90.7506 | 89.6513 | 1.0993 |
| 41 | 83 | DS-R1-7B | GSM8K | 4 | 91.4329 | 90.0303 | 1.4026 |
| 42 | 84 | DS-R1-7B | GSM8K | 4 | 91.0538 | 89.9924 | 1.0614 |
| 43 | 85 | DS-R1-7B | GSM8K | 4 | 90.5989 | 89.9545 | 0.6444 |
| 44 | 86 | DS-R1-7B | GSM8K | 4 | 91.395 | 89.5375 | 1.8575 |
| 45 | 87 | DS-R1-7B | GSM8K | 4 | 91.1296 | 89.5754 | 1.5542 |
| 46 | 88 | DS-R1-7B | GSM8K | 4 | 90.978 | 89.2343 | 1.7437 |
| 47 | 89 | DS-R1-7B | GSM8K | 4 | 91.2055 | 90.182 | 1.0235 |
| 48 | 90 | DS-R1-7B | GSM8K | 4 | 91.1676 | 89.1205 | 2.047 |
| 49 | 91 | DS-R1-7B | GSM8K | 4 | 91.2813 | 89.348 | 1.9333 |
| 0 | 42 | DS-R1-7B | GSM8K | 8 | 90.8264 | 89.8597 | 0.9666 |
| 1 | 43 | DS-R1-7B | GSM8K | 8 | 90.9212 | 89.9924 | 0.9287 |
| 2 | 44 | DS-R1-7B | GSM8K | 8 | 90.5421 | 89.5754 | 0.9666 |
| 3 | 45 | DS-R1-7B | GSM8K | 8 | 90.8453 | 89.7081 | 1.1372 |
| 4 | 46 | DS-R1-7B | GSM8K | 8 | 91.0917 | 89.8029 | 1.2889 |
| 5 | 47 | DS-R1-7B | GSM8K | 8 | 91.2434 | 89.8597 | 1.3836 |
| 6 | 48 | DS-R1-7B | GSM8K | 8 | 90.9022 | 89.8408 | 1.0614 |
| 7 | 49 | DS-R1-7B | GSM8K | 8 | 90.8264 | 89.8597 | 0.9666 |
| 8 | 50 | DS-R1-7B | GSM8K | 8 | 90.9212 | 89.6323 | 1.2889 |
| 9 | 51 | DS-R1-7B | GSM8K | 8 | 90.9401 | 90.144 | 0.7961 |
| 10 | 52 | DS-R1-7B | GSM8K | 8 | 90.8453 | 89.8976 | 0.9477 |
| 11 | 53 | DS-R1-7B | GSM8K | 8 | 90.7885 | 89.6892 | 1.0993 |
| 12 | 54 | DS-R1-7B | GSM8K | 8 | 91.0159 | 89.6323 | 1.3836 |
| 13 | 55 | DS-R1-7B | GSM8K | 8 | 90.5421 | 89.8218 | 0.7202 |
| 14 | 56 | DS-R1-7B | GSM8K | 8 | 91.0728 | 89.8029 | 1.2699 |
| 15 | 57 | DS-R1-7B | GSM8K | 8 | 90.2388 | 89.7839 | 0.4549 |
| 16 | 58 | DS-R1-7B | GSM8K | 8 | 91.0159 | 89.4049 | 1.6111 |
| 17 | 59 | DS-R1-7B | GSM8K | 8 | 91.1865 | 89.6133 | 1.5732 |
| 18 | 60 | DS-R1-7B | GSM8K | 8 | 90.7885 | 89.7081 | 1.0804 |
| 19 | 61 | DS-R1-7B | GSM8K | 8 | 90.8453 | 89.9166 | 0.9287 |
| 20 | 62 | DS-R1-7B | GSM8K | 8 | 90.5042 | 89.6892 | 0.815 |
| 21 | 63 | DS-R1-7B | GSM8K | 8 | 90.978 | 90.163 | 0.815 |
| 22 | 64 | DS-R1-7B | GSM8K | 8 | 90.7695 | 89.8787 | 0.8908 |
| 23 | 65 | DS-R1-7B | GSM8K | 8 | 91.376 | 89.5944 | 1.7817 |
| 24 | 66 | DS-R1-7B | GSM8K | 8 | 90.9022 | 89.4996 | 1.4026 |
| 25 | 67 | DS-R1-7B | GSM8K | 8 | 91.0159 | 89.4049 | 1.6111 |
| 26 | 68 | DS-R1-7B | GSM8K | 8 | 91.0159 | 89.6892 | 1.3268 |
| 27 | 69 | DS-R1-7B | GSM8K | 8 | 90.5421 | 89.5944 | 0.9477 |
| 28 | 70 | DS-R1-7B | GSM8K | 8 | 91.0728 | 89.4807 | 1.5921 |
| 29 | 71 | DS-R1-7B | GSM8K | 8 | 90.8832 | 89.7271 | 1.1562 |
| 30 | 72 | DS-R1-7B | GSM8K | 8 | 91.0728 | 89.746 | 1.3268 |
| 31 | 73 | DS-R1-7B | GSM8K | 8 | 91.0538 | 89.7271 | 1.3268 |
| 32 | 74 | DS-R1-7B | GSM8K | 8 | 90.8264 | 89.746 | 1.0804 |
| 33 | 75 | DS-R1-7B | GSM8K | 8 | 90.9022 | 89.5375 | 1.3647 |
| 34 | 76 | DS-R1-7B | GSM8K | 8 | 90.8074 | 89.4996 | 1.3078 |
| 35 | 77 | DS-R1-7B | GSM8K | 8 | 90.8074 | 89.6892 | 1.1183 |
| 36 | 78 | DS-R1-7B | GSM8K | 8 | 90.6368 | 89.8029 | 0.834 |
| 37 | 79 | DS-R1-7B | GSM8K | 8 | 90.8074 | 89.4238 | 1.3836 |
| 38 | 80 | DS-R1-7B | GSM8K | 8 | 90.8832 | 89.5186 | 1.3647 |
| 39 | 81 | DS-R1-7B | GSM8K | 8 | 90.8264 | 89.5375 | 1.2889 |
| 40 | 82 | DS-R1-7B | GSM8K | 8 | 91.0159 | 89.8976 | 1.1183 |
| 41 | 83 | DS-R1-7B | GSM8K | 8 | 91.1865 | 89.7839 | 1.4026 |
| 42 | 84 | DS-R1-7B | GSM8K | 8 | 90.7885 | 89.9545 | 0.834 |
| 43 | 85 | DS-R1-7B | GSM8K | 8 | 90.4663 | 89.8408 | 0.6255 |
| 44 | 86 | DS-R1-7B | GSM8K | 8 | 90.9022 | 89.765 | 1.1372 |
| 45 | 87 | DS-R1-7B | GSM8K | 8 | 91.1107 | 89.4238 | 1.6869 |
| 46 | 88 | DS-R1-7B | GSM8K | 8 | 90.978 | 89.2532 | 1.7248 |
| 47 | 89 | DS-R1-7B | GSM8K | 8 | 90.8453 | 89.746 | 1.0993 |
| 48 | 90 | DS-R1-7B | GSM8K | 8 | 90.8074 | 89.4049 | 1.4026 |
| 49 | 91 | DS-R1-7B | GSM8K | 8 | 90.9401 | 89.5754 | 1.3647 |
| 0 | 42 | DS-R1-7B | GSM8K | 12 | 91.0665 | 89.5123 | 1.5542 |
| 1 | 43 | DS-R1-7B | GSM8K | 12 | 91.117 | 89.5881 | 1.5289 |
| 2 | 44 | DS-R1-7B | GSM8K | 12 | 90.6874 | 89.4743 | 1.213 |
| 3 | 45 | DS-R1-7B | GSM8K | 12 | 90.8011 | 89.765 | 1.0361 |
| 4 | 46 | DS-R1-7B | GSM8K | 12 | 91.117 | 89.8787 | 1.2383 |
| 5 | 47 | DS-R1-7B | GSM8K | 12 | 90.9654 | 89.765 | 1.2004 |
| 6 | 48 | DS-R1-7B | GSM8K | 12 | 90.9275 | 89.7523 | 1.1751 |
| 7 | 49 | DS-R1-7B | GSM8K | 12 | 90.9527 | 89.4996 | 1.4531 |
| 8 | 50 | DS-R1-7B | GSM8K | 12 | 91.1423 | 89.626 | 1.5163 |
| 9 | 51 | DS-R1-7B | GSM8K | 12 | 90.9022 | 89.8661 | 1.0361 |
| 10 | 52 | DS-R1-7B | GSM8K | 12 | 90.7632 | 89.7523 | 1.0109 |
| 11 | 53 | DS-R1-7B | GSM8K | 12 | 90.9654 | 89.6892 | 1.2762 |
| 12 | 54 | DS-R1-7B | GSM8K | 12 | 91.0033 | 89.5881 | 1.4152 |
| 13 | 55 | DS-R1-7B | GSM8K | 12 | 90.8643 | 89.8029 | 1.0614 |
| 14 | 56 | DS-R1-7B | GSM8K | 12 | 90.978 | 89.6133 | 1.3647 |
| 15 | 57 | DS-R1-7B | GSM8K | 12 | 90.6621 | 89.8155 | 0.8466 |
| 16 | 58 | DS-R1-7B | GSM8K | 12 | 90.978 | 89.4238 | 1.5542 |
| 17 | 59 | DS-R1-7B | GSM8K | 12 | 90.8137 | 89.7144 | 1.0993 |
| 18 | 60 | DS-R1-7B | GSM8K | 12 | 90.8643 | 89.4743 | 1.3899 |
| 19 | 61 | DS-R1-7B | GSM8K | 12 | 90.8769 | 89.7271 | 1.1499 |
| 20 | 62 | DS-R1-7B | GSM8K | 12 | 90.9148 | 89.6007 | 1.3141 |
| 21 | 63 | DS-R1-7B | GSM8K | 12 | 90.9906 | 89.8534 | 1.1372 |
| 22 | 64 | DS-R1-7B | GSM8K | 12 | 90.7758 | 89.7018 | 1.074 |
| 23 | 65 | DS-R1-7B | GSM8K | 12 | 91.1296 | 89.5754 | 1.5542 |
| 24 | 66 | DS-R1-7B | GSM8K | 12 | 90.9148 | 89.5249 | 1.3899 |
| 25 | 67 | DS-R1-7B | GSM8K | 12 | 91.0033 | 89.7144 | 1.2889 |
| 26 | 68 | DS-R1-7B | GSM8K | 12 | 90.9148 | 89.6386 | 1.2762 |
| 27 | 69 | DS-R1-7B | GSM8K | 12 | 90.8643 | 89.6133 | 1.2509 |
| 28 | 70 | DS-R1-7B | GSM8K | 12 | 91.0033 | 89.5249 | 1.4784 |
| 29 | 71 | DS-R1-7B | GSM8K | 12 | 90.8769 | 89.7018 | 1.1751 |
| 30 | 72 | DS-R1-7B | GSM8K | 12 | 90.9401 | 89.8029 | 1.1372 |
| 31 | 73 | DS-R1-7B | GSM8K | 12 | 90.8517 | 89.7018 | 1.1499 |
| 32 | 74 | DS-R1-7B | GSM8K | 12 | 90.8264 | 89.6765 | 1.1499 |
| 33 | 75 | DS-R1-7B | GSM8K | 12 | 90.8264 | 89.8787 | 0.9477 |
| 34 | 76 | DS-R1-7B | GSM8K | 12 | 90.7885 | 89.4617 | 1.3268 |
| 35 | 77 | DS-R1-7B | GSM8K | 12 | 91.0286 | 89.4491 | 1.5795 |
| 36 | 78 | DS-R1-7B | GSM8K | 12 | 90.9275 | 89.626 | 1.3015 |
| 37 | 79 | DS-R1-7B | GSM8K | 12 | 90.839 | 89.7397 | 1.0993 |
| 38 | 80 | DS-R1-7B | GSM8K | 12 | 90.9022 | 89.765 | 1.1372 |
| 39 | 81 | DS-R1-7B | GSM8K | 12 | 90.6748 | 89.5754 | 1.0993 |
| 40 | 82 | DS-R1-7B | GSM8K | 12 | 90.8769 | 89.7523 | 1.1246 |
| 41 | 83 | DS-R1-7B | GSM8K | 12 | 91.1802 | 89.4491 | 1.7311 |
| 42 | 84 | DS-R1-7B | GSM8K | 12 | 90.8769 | 89.5502 | 1.3268 |
| 43 | 85 | DS-R1-7B | GSM8K | 12 | 90.6495 | 89.5249 | 1.1246 |
| 44 | 86 | DS-R1-7B | GSM8K | 12 | 90.7885 | 89.7144 | 1.074 |
| 45 | 87 | DS-R1-7B | GSM8K | 12 | 91.0538 | 89.4743 | 1.5795 |
| 46 | 88 | DS-R1-7B | GSM8K | 12 | 90.978 | 89.5754 | 1.4026 |
| 47 | 89 | DS-R1-7B | GSM8K | 12 | 91.0791 | 89.487 | 1.5921 |
| 48 | 90 | DS-R1-7B | GSM8K | 12 | 90.9906 | 89.5123 | 1.4784 |
| 49 | 91 | DS-R1-7B | GSM8K | 12 | 91.1296 | 89.5881 | 1.5416 |
| 0 | 42 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 1 | 43 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 2 | 44 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 3 | 45 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 4 | 46 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 5 | 47 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 6 | 48 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 7 | 49 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 8 | 50 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 9 | 51 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 10 | 52 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 11 | 53 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 12 | 54 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 13 | 55 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 14 | 56 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 15 | 57 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 16 | 58 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 17 | 59 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 18 | 60 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 19 | 61 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 20 | 62 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 21 | 63 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 22 | 64 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 23 | 65 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 24 | 66 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 25 | 67 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 26 | 68 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 27 | 69 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 28 | 70 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 29 | 71 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 30 | 72 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 31 | 73 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 32 | 74 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 33 | 75 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 34 | 76 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 35 | 77 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 36 | 78 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 37 | 79 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 38 | 80 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 39 | 81 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 40 | 82 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 41 | 83 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 42 | 84 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 43 | 85 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 44 | 86 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 45 | 87 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 46 | 88 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 47 | 89 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 48 | 90 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 49 | 91 | DS-R1-7B | GSM8K | 16 | 90.9496 | 89.6039 | 1.3457 |
| 0 | 42 | DS-R1-7B | MATH500 | 4 | 63.1 | 57 | 6.1 |
| 1 | 43 | DS-R1-7B | MATH500 | 4 | 64.2 | 58.3 | 5.9 |
| 2 | 44 | DS-R1-7B | MATH500 | 4 | 62.9 | 57.1 | 5.8 |
| 3 | 45 | DS-R1-7B | MATH500 | 4 | 62.4 | 57.8 | 4.6 |
| 4 | 46 | DS-R1-7B | MATH500 | 4 | 64 | 56.6 | 7.4 |
| 5 | 47 | DS-R1-7B | MATH500 | 4 | 63.3 | 57.7 | 5.6 |
| 6 | 48 | DS-R1-7B | MATH500 | 4 | 63.2 | 58.8 | 4.4 |
| 7 | 49 | DS-R1-7B | MATH500 | 4 | 62.9 | 58.7 | 4.2 |
| 8 | 50 | DS-R1-7B | MATH500 | 4 | 63.1 | 56.7 | 6.4 |
| 9 | 51 | DS-R1-7B | MATH500 | 4 | 63.1 | 58.6 | 4.5 |
| 10 | 52 | DS-R1-7B | MATH500 | 4 | 63.9 | 57.8 | 6.1 |
| 11 | 53 | DS-R1-7B | MATH500 | 4 | 63.1 | 59 | 4.1 |
| 12 | 54 | DS-R1-7B | MATH500 | 4 | 61.9 | 57.1 | 4.8 |
| 13 | 55 | DS-R1-7B | MATH500 | 4 | 63.1 | 57.1 | 6 |
| 14 | 56 | DS-R1-7B | MATH500 | 4 | 62.3 | 57.6 | 4.7 |
| 15 | 57 | DS-R1-7B | MATH500 | 4 | 63.1 | 57.1 | 6 |
| 16 | 58 | DS-R1-7B | MATH500 | 4 | 63.3 | 58.5 | 4.8 |
| 17 | 59 | DS-R1-7B | MATH500 | 4 | 63.2 | 56.2 | 7 |
| 18 | 60 | DS-R1-7B | MATH500 | 4 | 62.5 | 58.1 | 4.4 |
| 19 | 61 | DS-R1-7B | MATH500 | 4 | 63.9 | 58.6 | 5.3 |
| 20 | 62 | DS-R1-7B | MATH500 | 4 | 61.8 | 58.5 | 3.3 |
| 21 | 63 | DS-R1-7B | MATH500 | 4 | 62.2 | 58.5 | 3.7 |
| 22 | 64 | DS-R1-7B | MATH500 | 4 | 63.3 | 58.5 | 4.8 |
| 23 | 65 | DS-R1-7B | MATH500 | 4 | 61.9 | 57.9 | 4 |
| 24 | 66 | DS-R1-7B | MATH500 | 4 | 62.8 | 56.6 | 6.2 |
| 25 | 67 | DS-R1-7B | MATH500 | 4 | 63.8 | 57.4 | 6.4 |
| 26 | 68 | DS-R1-7B | MATH500 | 4 | 63.8 | 57.1 | 6.7 |
| 27 | 69 | DS-R1-7B | MATH500 | 4 | 62.9 | 59.1 | 3.8 |
| 28 | 70 | DS-R1-7B | MATH500 | 4 | 63.3 | 58 | 5.3 |
| 29 | 71 | DS-R1-7B | MATH500 | 4 | 62.8 | 58.1 | 4.7 |
| 30 | 72 | DS-R1-7B | MATH500 | 4 | 63.2 | 57.6 | 5.6 |
| 31 | 73 | DS-R1-7B | MATH500 | 4 | 63.9 | 58.4 | 5.5 |
| 32 | 74 | DS-R1-7B | MATH500 | 4 | 63.2 | 57.8 | 5.4 |
| 33 | 75 | DS-R1-7B | MATH500 | 4 | 64.3 | 56.9 | 7.4 |
| 34 | 76 | DS-R1-7B | MATH500 | 4 | 63.9 | 56.8 | 7.1 |
| 35 | 77 | DS-R1-7B | MATH500 | 4 | 61.8 | 57.9 | 3.9 |
| 36 | 78 | DS-R1-7B | MATH500 | 4 | 63.8 | 58.8 | 5 |
| 37 | 79 | DS-R1-7B | MATH500 | 4 | 62.7 | 57.8 | 4.9 |
| 38 | 80 | DS-R1-7B | MATH500 | 4 | 63.5 | 57.8 | 5.7 |
| 39 | 81 | DS-R1-7B | MATH500 | 4 | 62.8 | 57.9 | 4.9 |
| 40 | 82 | DS-R1-7B | MATH500 | 4 | 61.7 | 58.1 | 3.6 |
| 41 | 83 | DS-R1-7B | MATH500 | 4 | 63.7 | 57.5 | 6.2 |
| 42 | 84 | DS-R1-7B | MATH500 | 4 | 63.2 | 57.2 | 6 |
| 43 | 85 | DS-R1-7B | MATH500 | 4 | 61.3 | 58.6 | 2.7 |
| 44 | 86 | DS-R1-7B | MATH500 | 4 | 61.8 | 58.1 | 3.7 |
| 45 | 87 | DS-R1-7B | MATH500 | 4 | 62.9 | 57.4 | 5.5 |
| 46 | 88 | DS-R1-7B | MATH500 | 4 | 63.2 | 57.4 | 5.8 |
| 47 | 89 | DS-R1-7B | MATH500 | 4 | 62.1 | 58.1 | 4 |
| 48 | 90 | DS-R1-7B | MATH500 | 4 | 62.4 | 56.8 | 5.6 |
| 49 | 91 | DS-R1-7B | MATH500 | 4 | 62.4 | 58.1 | 4.3 |
| 0 | 42 | DS-R1-7B | MATH500 | 8 | 63.5 | 57.35 | 6.15 |
| 1 | 43 | DS-R1-7B | MATH500 | 8 | 63.9 | 57.4 | 6.5 |
| 2 | 44 | DS-R1-7B | MATH500 | 8 | 63.3 | 57.05 | 6.25 |
| 3 | 45 | DS-R1-7B | MATH500 | 8 | 63.25 | 57.25 | 6 |
| 4 | 46 | DS-R1-7B | MATH500 | 8 | 63.9 | 57.35 | 6.55 |
| 5 | 47 | DS-R1-7B | MATH500 | 8 | 63.45 | 57.25 | 6.2 |
| 6 | 48 | DS-R1-7B | MATH500 | 8 | 63.35 | 58 | 5.35 |
| 7 | 49 | DS-R1-7B | MATH500 | 8 | 62.9 | 57.85 | 5.05 |
| 8 | 50 | DS-R1-7B | MATH500 | 8 | 63.35 | 57.2 | 6.15 |
| 9 | 51 | DS-R1-7B | MATH500 | 8 | 63.35 | 57.65 | 5.7 |
| 10 | 52 | DS-R1-7B | MATH500 | 8 | 64.1 | 57.65 | 6.45 |
| 11 | 53 | DS-R1-7B | MATH500 | 8 | 63.75 | 57.15 | 6.6 |
| 12 | 54 | DS-R1-7B | MATH500 | 8 | 62.95 | 56.9 | 6.05 |
| 13 | 55 | DS-R1-7B | MATH500 | 8 | 63.6 | 57.3 | 6.3 |
| 14 | 56 | DS-R1-7B | MATH500 | 8 | 62.95 | 57.2 | 5.75 |
| 15 | 57 | DS-R1-7B | MATH500 | 8 | 63.3 | 57.15 | 6.15 |
| 16 | 58 | DS-R1-7B | MATH500 | 8 | 63.65 | 57.5 | 6.15 |
| 17 | 59 | DS-R1-7B | MATH500 | 8 | 62.95 | 56.8 | 6.15 |
| 18 | 60 | DS-R1-7B | MATH500 | 8 | 63.3 | 57.95 | 5.35 |
| 19 | 61 | DS-R1-7B | MATH500 | 8 | 64 | 57.9 | 6.1 |
| 20 | 62 | DS-R1-7B | MATH500 | 8 | 63 | 57.4 | 5.6 |
| 21 | 63 | DS-R1-7B | MATH500 | 8 | 63 | 58.25 | 4.75 |
| 22 | 64 | DS-R1-7B | MATH500 | 8 | 63.65 | 57.35 | 6.3 |
| 23 | 65 | DS-R1-7B | MATH500 | 8 | 63.5 | 56.8 | 6.7 |
| 24 | 66 | DS-R1-7B | MATH500 | 8 | 62.75 | 57.25 | 5.5 |
| 25 | 67 | DS-R1-7B | MATH500 | 8 | 63.55 | 57.1 | 6.45 |
| 26 | 68 | DS-R1-7B | MATH500 | 8 | 64.1 | 57.15 | 6.95 |
| 27 | 69 | DS-R1-7B | MATH500 | 8 | 62.75 | 58.15 | 4.6 |
| 28 | 70 | DS-R1-7B | MATH500 | 8 | 63.05 | 57.65 | 5.4 |
| 29 | 71 | DS-R1-7B | MATH500 | 8 | 63.3 | 57.4 | 5.9 |
| 30 | 72 | DS-R1-7B | MATH500 | 8 | 62.85 | 57.55 | 5.3 |
| 31 | 73 | DS-R1-7B | MATH500 | 8 | 63.8 | 56.95 | 6.85 |
| 32 | 74 | DS-R1-7B | MATH500 | 8 | 63.9 | 57.15 | 6.75 |
| 33 | 75 | DS-R1-7B | MATH500 | 8 | 63.7 | 57.7 | 6 |
| 34 | 76 | DS-R1-7B | MATH500 | 8 | 63.8 | 57.55 | 6.25 |
| 35 | 77 | DS-R1-7B | MATH500 | 8 | 63.35 | 57.45 | 5.9 |
| 36 | 78 | DS-R1-7B | MATH500 | 8 | 63.4 | 57.95 | 5.45 |
| 37 | 79 | DS-R1-7B | MATH500 | 8 | 63.5 | 57.3 | 6.2 |
| 38 | 80 | DS-R1-7B | MATH500 | 8 | 64.35 | 57.05 | 7.3 |
| 39 | 81 | DS-R1-7B | MATH500 | 8 | 63.6 | 57.35 | 6.25 |
| 40 | 82 | DS-R1-7B | MATH500 | 8 | 63.4 | 57.25 | 6.15 |
| 41 | 83 | DS-R1-7B | MATH500 | 8 | 63.9 | 56.9 | 7 |
| 42 | 84 | DS-R1-7B | MATH500 | 8 | 62.95 | 57.3 | 5.65 |
| 43 | 85 | DS-R1-7B | MATH500 | 8 | 62.65 | 58.2 | 4.45 |
| 44 | 86 | DS-R1-7B | MATH500 | 8 | 62.75 | 57.85 | 4.9 |
| 45 | 87 | DS-R1-7B | MATH500 | 8 | 63.45 | 57 | 6.45 |
| 46 | 88 | DS-R1-7B | MATH500 | 8 | 63.05 | 57.15 | 5.9 |
| 47 | 89 | DS-R1-7B | MATH500 | 8 | 63.15 | 57.85 | 5.3 |
| 48 | 90 | DS-R1-7B | MATH500 | 8 | 63.2 | 56.95 | 6.25 |
| 49 | 91 | DS-R1-7B | MATH500 | 8 | 63.5 | 57.4 | 6.1 |
| 0 | 42 | DS-R1-7B | MATH500 | 12 | 63.2333 | 57.6333 | 5.6 |
| 1 | 43 | DS-R1-7B | MATH500 | 12 | 63.3 | 57.0667 | 6.2333 |
| 2 | 44 | DS-R1-7B | MATH500 | 12 | 63.2333 | 57.2333 | 6 |
| 3 | 45 | DS-R1-7B | MATH500 | 12 | 63.5667 | 57.3 | 6.2667 |
| 4 | 46 | DS-R1-7B | MATH500 | 12 | 63.5333 | 57.3667 | 6.1667 |
| 5 | 47 | DS-R1-7B | MATH500 | 12 | 63.5333 | 57.3667 | 6.1667 |
| 6 | 48 | DS-R1-7B | MATH500 | 12 | 63.0667 | 57.7 | 5.3667 |
| 7 | 49 | DS-R1-7B | MATH500 | 12 | 63.4333 | 57.4333 | 6 |
| 8 | 50 | DS-R1-7B | MATH500 | 12 | 63.4 | 57.2333 | 6.1667 |
| 9 | 51 | DS-R1-7B | MATH500 | 12 | 63.8333 | 57.4667 | 6.3667 |
| 10 | 52 | DS-R1-7B | MATH500 | 12 | 63.7333 | 57.5333 | 6.2 |
| 11 | 53 | DS-R1-7B | MATH500 | 12 | 63.6667 | 57.2667 | 6.4 |
| 12 | 54 | DS-R1-7B | MATH500 | 12 | 63.3667 | 57.1667 | 6.2 |
| 13 | 55 | DS-R1-7B | MATH500 | 12 | 63.6333 | 57.0333 | 6.6 |
| 14 | 56 | DS-R1-7B | MATH500 | 12 | 63.7 | 57.0333 | 6.6667 |
| 15 | 57 | DS-R1-7B | MATH500 | 12 | 63.5 | 57.0667 | 6.4333 |
| 16 | 58 | DS-R1-7B | MATH500 | 12 | 63.6 | 57.4333 | 6.1667 |
| 17 | 59 | DS-R1-7B | MATH500 | 12 | 63.3333 | 57.1333 | 6.2 |
| 18 | 60 | DS-R1-7B | MATH500 | 12 | 62.9667 | 57.3 | 5.6667 |
| 19 | 61 | DS-R1-7B | MATH500 | 12 | 63.4333 | 57.5667 | 5.8667 |
| 20 | 62 | DS-R1-7B | MATH500 | 12 | 63.3667 | 57.2 | 6.1667 |
| 21 | 63 | DS-R1-7B | MATH500 | 12 | 63.3333 | 57.5667 | 5.7667 |
| 22 | 64 | DS-R1-7B | MATH500 | 12 | 63.4667 | 57.1 | 6.3667 |
| 23 | 65 | DS-R1-7B | MATH500 | 12 | 63.4 | 57.3667 | 6.0333 |
| 24 | 66 | DS-R1-7B | MATH500 | 12 | 63.4667 | 57.2 | 6.2667 |
| 25 | 67 | DS-R1-7B | MATH500 | 12 | 63.3667 | 57 | 6.3667 |
| 26 | 68 | DS-R1-7B | MATH500 | 12 | 63.8 | 57.3 | 6.5 |
| 27 | 69 | DS-R1-7B | MATH500 | 12 | 63.3333 | 57.6333 | 5.7 |
| 28 | 70 | DS-R1-7B | MATH500 | 12 | 63.2333 | 57.1 | 6.1333 |
| 29 | 71 | DS-R1-7B | MATH500 | 12 | 63.5667 | 57.6 | 5.9667 |
| 30 | 72 | DS-R1-7B | MATH500 | 12 | 63.4333 | 56.8667 | 6.5667 |
| 31 | 73 | DS-R1-7B | MATH500 | 12 | 63.6 | 57.4667 | 6.1333 |
| 32 | 74 | DS-R1-7B | MATH500 | 12 | 63.5333 | 57.3667 | 6.1667 |
| 33 | 75 | DS-R1-7B | MATH500 | 12 | 63.5 | 57.5333 | 5.9667 |
| 34 | 76 | DS-R1-7B | MATH500 | 12 | 63.7667 | 57.6 | 6.1667 |
| 35 | 77 | DS-R1-7B | MATH500 | 12 | 63.4667 | 57.1667 | 6.3 |
| 36 | 78 | DS-R1-7B | MATH500 | 12 | 63.5 | 57.3 | 6.2 |
| 37 | 79 | DS-R1-7B | MATH500 | 12 | 63.4667 | 57.1333 | 6.3333 |
| 38 | 80 | DS-R1-7B | MATH500 | 12 | 63.6667 | 57.2333 | 6.4333 |
| 39 | 81 | DS-R1-7B | MATH500 | 12 | 63.5667 | 57.5333 | 6.0333 |
| 40 | 82 | DS-R1-7B | MATH500 | 12 | 63.7333 | 57.2 | 6.5333 |
| 41 | 83 | DS-R1-7B | MATH500 | 12 | 63.6667 | 57.2333 | 6.4333 |
| 42 | 84 | DS-R1-7B | MATH500 | 12 | 63.2667 | 57.2 | 6.0667 |
| 43 | 85 | DS-R1-7B | MATH500 | 12 | 63.2333 | 57.6333 | 5.6 |
| 44 | 86 | DS-R1-7B | MATH500 | 12 | 63.5667 | 57.6333 | 5.9333 |
| 45 | 87 | DS-R1-7B | MATH500 | 12 | 63.3667 | 57.5667 | 5.8 |
| 46 | 88 | DS-R1-7B | MATH500 | 12 | 63.7667 | 57.0333 | 6.7333 |
| 47 | 89 | DS-R1-7B | MATH500 | 12 | 63.2667 | 57.3667 | 5.9 |
| 48 | 90 | DS-R1-7B | MATH500 | 12 | 63.5 | 57.3 | 6.2 |
| 49 | 91 | DS-R1-7B | MATH500 | 12 | 63.6 | 57.1667 | 6.4333 |
| 0 | 42 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 1 | 43 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 2 | 44 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 3 | 45 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 4 | 46 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 5 | 47 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 6 | 48 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 7 | 49 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 8 | 50 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 9 | 51 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 10 | 52 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 11 | 53 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 12 | 54 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 13 | 55 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 14 | 56 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 15 | 57 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 16 | 58 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 17 | 59 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 18 | 60 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 19 | 61 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 20 | 62 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 21 | 63 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 22 | 64 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 23 | 65 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 24 | 66 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 25 | 67 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 26 | 68 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 27 | 69 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 28 | 70 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 29 | 71 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 30 | 72 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 31 | 73 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 32 | 74 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 33 | 75 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 34 | 76 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 35 | 77 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 36 | 78 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 37 | 79 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 38 | 80 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 39 | 81 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 40 | 82 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 41 | 83 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 42 | 84 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 43 | 85 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 44 | 86 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 45 | 87 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 46 | 88 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 47 | 89 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 48 | 90 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 49 | 91 | DS-R1-7B | MATH500 | 16 | 63.475 | 57.35 | 6.125 |
| 0 | 42 | DS-R1-7B | SVAMP | 4 | 92.15 | 92.15 | 0 |
| 1 | 43 | DS-R1-7B | SVAMP | 4 | 91.8 | 90.8 | 1 |
| 2 | 44 | DS-R1-7B | SVAMP | 4 | 92.55 | 91.7 | 0.85 |
| 3 | 45 | DS-R1-7B | SVAMP | 4 | 92.55 | 92.45 | 0.1 |
| 4 | 46 | DS-R1-7B | SVAMP | 4 | 92.35 | 91.65 | 0.7 |
| 5 | 47 | DS-R1-7B | SVAMP | 4 | 92.1 | 91.9 | 0.2 |
| 6 | 48 | DS-R1-7B | SVAMP | 4 | 92.65 | 91.2 | 1.45 |
| 7 | 49 | DS-R1-7B | SVAMP | 4 | 91.75 | 92 | -0.25 |
| 8 | 50 | DS-R1-7B | SVAMP | 4 | 92.45 | 92 | 0.45 |
| 9 | 51 | DS-R1-7B | SVAMP | 4 | 92.35 | 92 | 0.35 |
| 10 | 52 | DS-R1-7B | SVAMP | 4 | 91.85 | 91.45 | 0.4 |
| 11 | 53 | DS-R1-7B | SVAMP | 4 | 90.85 | 92.35 | -1.5 |
| 12 | 54 | DS-R1-7B | SVAMP | 4 | 92.45 | 91.85 | 0.6 |
| 13 | 55 | DS-R1-7B | SVAMP | 4 | 92.05 | 92.15 | -0.1 |
| 14 | 56 | DS-R1-7B | SVAMP | 4 | 91.95 | 91.45 | 0.5 |
| 15 | 57 | DS-R1-7B | SVAMP | 4 | 92.45 | 90.7 | 1.75 |
| 16 | 58 | DS-R1-7B | SVAMP | 4 | 92.5 | 91.65 | 0.85 |
| 17 | 59 | DS-R1-7B | SVAMP | 4 | 91.8 | 91.95 | -0.15 |
| 18 | 60 | DS-R1-7B | SVAMP | 4 | 91.95 | 92.1 | -0.15 |
| 19 | 61 | DS-R1-7B | SVAMP | 4 | 92.15 | 92.3 | -0.15 |
| 20 | 62 | DS-R1-7B | SVAMP | 4 | 92.9 | 92 | 0.9 |
| 21 | 63 | DS-R1-7B | SVAMP | 4 | 92.65 | 92.25 | 0.4 |
| 22 | 64 | DS-R1-7B | SVAMP | 4 | 92.3 | 90.85 | 1.45 |
| 23 | 65 | DS-R1-7B | SVAMP | 4 | 91.9 | 91.65 | 0.25 |
| 24 | 66 | DS-R1-7B | SVAMP | 4 | 91.95 | 91.65 | 0.3 |
| 25 | 67 | DS-R1-7B | SVAMP | 4 | 92.2 | 91.8 | 0.4 |
| 26 | 68 | DS-R1-7B | SVAMP | 4 | 92.4 | 91.9 | 0.5 |
| 27 | 69 | DS-R1-7B | SVAMP | 4 | 92.25 | 92.05 | 0.2 |
| 28 | 70 | DS-R1-7B | SVAMP | 4 | 91.95 | 92 | -0.05 |
| 29 | 71 | DS-R1-7B | SVAMP | 4 | 91.7 | 91.3 | 0.4 |
| 30 | 72 | DS-R1-7B | SVAMP | 4 | 92.25 | 91.45 | 0.8 |
| 31 | 73 | DS-R1-7B | SVAMP | 4 | 92.3 | 91.7 | 0.6 |
| 32 | 74 | DS-R1-7B | SVAMP | 4 | 92.7 | 92.1 | 0.6 |
| 33 | 75 | DS-R1-7B | SVAMP | 4 | 91.9 | 91.65 | 0.25 |
| 34 | 76 | DS-R1-7B | SVAMP | 4 | 91.85 | 91.3 | 0.55 |
| 35 | 77 | DS-R1-7B | SVAMP | 4 | 92.5 | 91.45 | 1.05 |
| 36 | 78 | DS-R1-7B | SVAMP | 4 | 92.2 | 91.6 | 0.6 |
| 37 | 79 | DS-R1-7B | SVAMP | 4 | 92.05 | 91.9 | 0.15 |
| 38 | 80 | DS-R1-7B | SVAMP | 4 | 91.9 | 91.9 | 0 |
| 39 | 81 | DS-R1-7B | SVAMP | 4 | 92.35 | 92.05 | 0.3 |
| 40 | 82 | DS-R1-7B | SVAMP | 4 | 91.95 | 91.2 | 0.75 |
| 41 | 83 | DS-R1-7B | SVAMP | 4 | 92.75 | 91.3 | 1.45 |
| 42 | 84 | DS-R1-7B | SVAMP | 4 | 92.15 | 91.9 | 0.25 |
| 43 | 85 | DS-R1-7B | SVAMP | 4 | 92.4 | 91.7 | 0.7 |
| 44 | 86 | DS-R1-7B | SVAMP | 4 | 92.75 | 91.55 | 1.2 |
| 45 | 87 | DS-R1-7B | SVAMP | 4 | 92.3 | 91.6 | 0.7 |
| 46 | 88 | DS-R1-7B | SVAMP | 4 | 91.7 | 91.45 | 0.25 |
| 47 | 89 | DS-R1-7B | SVAMP | 4 | 92.55 | 92.05 | 0.5 |
| 48 | 90 | DS-R1-7B | SVAMP | 4 | 91.85 | 91.4 | 0.45 |
| 49 | 91 | DS-R1-7B | SVAMP | 4 | 91.85 | 91.55 | 0.3 |
| 0 | 42 | DS-R1-7B | SVAMP | 8 | 92.3 | 91.825 | 0.475 |
| 1 | 43 | DS-R1-7B | SVAMP | 8 | 92.175 | 91.625 | 0.55 |
| 2 | 44 | DS-R1-7B | SVAMP | 8 | 92.475 | 91.85 | 0.625 |
| 3 | 45 | DS-R1-7B | SVAMP | 8 | 92.225 | 92.15 | 0.075 |
| 4 | 46 | DS-R1-7B | SVAMP | 8 | 92.225 | 91.9 | 0.325 |
| 5 | 47 | DS-R1-7B | SVAMP | 8 | 92.25 | 91.475 | 0.775 |
| 6 | 48 | DS-R1-7B | SVAMP | 8 | 92.25 | 91.725 | 0.525 |
| 7 | 49 | DS-R1-7B | SVAMP | 8 | 92.05 | 91.775 | 0.275 |
| 8 | 50 | DS-R1-7B | SVAMP | 8 | 92.375 | 91.775 | 0.6 |
| 9 | 51 | DS-R1-7B | SVAMP | 8 | 92.375 | 91.275 | 1.1 |
| 10 | 52 | DS-R1-7B | SVAMP | 8 | 92.325 | 91.75 | 0.575 |
| 11 | 53 | DS-R1-7B | SVAMP | 8 | 91.725 | 91.875 | -0.15 |
| 12 | 54 | DS-R1-7B | SVAMP | 8 | 92.375 | 91.85 | 0.525 |
| 13 | 55 | DS-R1-7B | SVAMP | 8 | 92.075 | 91.725 | 0.35 |
| 14 | 56 | DS-R1-7B | SVAMP | 8 | 91.875 | 91.8 | 0.075 |
| 15 | 57 | DS-R1-7B | SVAMP | 8 | 91.8 | 91.475 | 0.325 |
| 16 | 58 | DS-R1-7B | SVAMP | 8 | 92.4 | 91.6 | 0.8 |
| 17 | 59 | DS-R1-7B | SVAMP | 8 | 92.125 | 91.925 | 0.2 |
| 18 | 60 | DS-R1-7B | SVAMP | 8 | 91.825 | 91.75 | 0.075 |
| 19 | 61 | DS-R1-7B | SVAMP | 8 | 92.375 | 91.975 | 0.4 |
| 20 | 62 | DS-R1-7B | SVAMP | 8 | 92.425 | 91.975 | 0.45 |
| 21 | 63 | DS-R1-7B | SVAMP | 8 | 92.225 | 91.625 | 0.6 |
| 22 | 64 | DS-R1-7B | SVAMP | 8 | 92 | 91.75 | 0.25 |
| 23 | 65 | DS-R1-7B | SVAMP | 8 | 92.05 | 91.425 | 0.625 |
| 24 | 66 | DS-R1-7B | SVAMP | 8 | 92.225 | 91.825 | 0.4 |
| 25 | 67 | DS-R1-7B | SVAMP | 8 | 92.3 | 91.4 | 0.9 |
| 26 | 68 | DS-R1-7B | SVAMP | 8 | 92.15 | 91.65 | 0.5 |
| 27 | 69 | DS-R1-7B | SVAMP | 8 | 92.225 | 91.7 | 0.525 |
| 28 | 70 | DS-R1-7B | SVAMP | 8 | 92.25 | 91.85 | 0.4 |
| 29 | 71 | DS-R1-7B | SVAMP | 8 | 91.8 | 91.475 | 0.325 |
| 30 | 72 | DS-R1-7B | SVAMP | 8 | 92.125 | 91.825 | 0.3 |
| 31 | 73 | DS-R1-7B | SVAMP | 8 | 92.375 | 91.5 | 0.875 |
| 32 | 74 | DS-R1-7B | SVAMP | 8 | 92.675 | 91.85 | 0.825 |
| 33 | 75 | DS-R1-7B | SVAMP | 8 | 92.3 | 91.9 | 0.4 |
| 34 | 76 | DS-R1-7B | SVAMP | 8 | 92.175 | 91.65 | 0.525 |
| 35 | 77 | DS-R1-7B | SVAMP | 8 | 92.4 | 91.375 | 1.025 |
| 36 | 78 | DS-R1-7B | SVAMP | 8 | 92.25 | 91.525 | 0.725 |
| 37 | 79 | DS-R1-7B | SVAMP | 8 | 92.225 | 91.7 | 0.525 |
| 38 | 80 | DS-R1-7B | SVAMP | 8 | 92.25 | 91.675 | 0.575 |
| 39 | 81 | DS-R1-7B | SVAMP | 8 | 92.15 | 91.625 | 0.525 |
| 40 | 82 | DS-R1-7B | SVAMP | 8 | 91.925 | 91.525 | 0.4 |
| 41 | 83 | DS-R1-7B | SVAMP | 8 | 92.475 | 91.775 | 0.7 |
| 42 | 84 | DS-R1-7B | SVAMP | 8 | 92.375 | 91.725 | 0.65 |
| 43 | 85 | DS-R1-7B | SVAMP | 8 | 92.3 | 91.3 | 1 |
| 44 | 86 | DS-R1-7B | SVAMP | 8 | 92.15 | 91.725 | 0.425 |
| 45 | 87 | DS-R1-7B | SVAMP | 8 | 92.075 | 91.75 | 0.325 |
| 46 | 88 | DS-R1-7B | SVAMP | 8 | 92.125 | 91.6 | 0.525 |
| 47 | 89 | DS-R1-7B | SVAMP | 8 | 92.325 | 92.05 | 0.275 |
| 48 | 90 | DS-R1-7B | SVAMP | 8 | 91.825 | 91.55 | 0.275 |
| 49 | 91 | DS-R1-7B | SVAMP | 8 | 91.65 | 91.825 | -0.175 |
| 0 | 42 | DS-R1-7B | SVAMP | 12 | 91.9833 | 91.65 | 0.3333 |
| 1 | 43 | DS-R1-7B | SVAMP | 12 | 92.1167 | 91.6167 | 0.5 |
| 2 | 44 | DS-R1-7B | SVAMP | 12 | 92.2667 | 91.7167 | 0.55 |
| 3 | 45 | DS-R1-7B | SVAMP | 12 | 92.2 | 91.9 | 0.3 |
| 4 | 46 | DS-R1-7B | SVAMP | 12 | 92.2167 | 91.55 | 0.6667 |
| 5 | 47 | DS-R1-7B | SVAMP | 12 | 92.1667 | 91.5667 | 0.6 |
| 6 | 48 | DS-R1-7B | SVAMP | 12 | 92.2167 | 91.6333 | 0.5833 |
| 7 | 49 | DS-R1-7B | SVAMP | 12 | 92.1333 | 91.6833 | 0.45 |
| 8 | 50 | DS-R1-7B | SVAMP | 12 | 92.3 | 91.65 | 0.65 |
| 9 | 51 | DS-R1-7B | SVAMP | 12 | 92.0333 | 91.5 | 0.5333 |
| 10 | 52 | DS-R1-7B | SVAMP | 12 | 92.35 | 91.55 | 0.8 |
| 11 | 53 | DS-R1-7B | SVAMP | 12 | 92.0167 | 91.65 | 0.3667 |
| 12 | 54 | DS-R1-7B | SVAMP | 12 | 92.0833 | 91.8333 | 0.25 |
| 13 | 55 | DS-R1-7B | SVAMP | 12 | 92.1667 | 91.7833 | 0.3833 |
| 14 | 56 | DS-R1-7B | SVAMP | 12 | 92.2333 | 91.7167 | 0.5167 |
| 15 | 57 | DS-R1-7B | SVAMP | 12 | 91.9167 | 91.6 | 0.3167 |
| 16 | 58 | DS-R1-7B | SVAMP | 12 | 92.25 | 91.6 | 0.65 |
| 17 | 59 | DS-R1-7B | SVAMP | 12 | 92.0833 | 91.6333 | 0.45 |
| 18 | 60 | DS-R1-7B | SVAMP | 12 | 92 | 91.65 | 0.35 |
| 19 | 61 | DS-R1-7B | SVAMP | 12 | 92.2167 | 91.8667 | 0.35 |
| 20 | 62 | DS-R1-7B | SVAMP | 12 | 92.1833 | 91.7333 | 0.45 |
| 21 | 63 | DS-R1-7B | SVAMP | 12 | 92 | 91.8 | 0.2 |
| 22 | 64 | DS-R1-7B | SVAMP | 12 | 92.2333 | 91.6 | 0.6333 |
| 23 | 65 | DS-R1-7B | SVAMP | 12 | 92.2 | 91.5 | 0.7 |
| 24 | 66 | DS-R1-7B | SVAMP | 12 | 92.15 | 91.7333 | 0.4167 |
| 25 | 67 | DS-R1-7B | SVAMP | 12 | 92.15 | 91.4667 | 0.6833 |
| 26 | 68 | DS-R1-7B | SVAMP | 12 | 92.1333 | 91.6667 | 0.4667 |
| 27 | 69 | DS-R1-7B | SVAMP | 12 | 92.1833 | 91.6833 | 0.5 |
| 28 | 70 | DS-R1-7B | SVAMP | 12 | 92.1167 | 91.9333 | 0.1833 |
| 29 | 71 | DS-R1-7B | SVAMP | 12 | 92.05 | 91.3667 | 0.6833 |
| 30 | 72 | DS-R1-7B | SVAMP | 12 | 92.1667 | 91.8833 | 0.2833 |
| 31 | 73 | DS-R1-7B | SVAMP | 12 | 92.3167 | 91.6 | 0.7167 |
| 32 | 74 | DS-R1-7B | SVAMP | 12 | 92.3333 | 91.6667 | 0.6667 |
| 33 | 75 | DS-R1-7B | SVAMP | 12 | 91.8833 | 91.7667 | 0.1167 |
| 34 | 76 | DS-R1-7B | SVAMP | 12 | 92.15 | 91.8167 | 0.3333 |
| 35 | 77 | DS-R1-7B | SVAMP | 12 | 92.3167 | 91.6333 | 0.6833 |
| 36 | 78 | DS-R1-7B | SVAMP | 12 | 92.2667 | 91.7833 | 0.4833 |
| 37 | 79 | DS-R1-7B | SVAMP | 12 | 92.0833 | 91.7167 | 0.3667 |
| 38 | 80 | DS-R1-7B | SVAMP | 12 | 92.3333 | 91.7 | 0.6333 |
| 39 | 81 | DS-R1-7B | SVAMP | 12 | 92.1333 | 91.7833 | 0.35 |
| 40 | 82 | DS-R1-7B | SVAMP | 12 | 92.1333 | 91.6 | 0.5333 |
| 41 | 83 | DS-R1-7B | SVAMP | 12 | 92.3 | 91.6 | 0.7 |
| 42 | 84 | DS-R1-7B | SVAMP | 12 | 92.2833 | 91.6167 | 0.6667 |
| 43 | 85 | DS-R1-7B | SVAMP | 12 | 92.1833 | 91.5833 | 0.6 |
| 44 | 86 | DS-R1-7B | SVAMP | 12 | 92.15 | 91.7667 | 0.3833 |
| 45 | 87 | DS-R1-7B | SVAMP | 12 | 92.1167 | 91.6167 | 0.5 |
| 46 | 88 | DS-R1-7B | SVAMP | 12 | 92.1667 | 91.7333 | 0.4333 |
| 47 | 89 | DS-R1-7B | SVAMP | 12 | 92.2833 | 91.7 | 0.5833 |
| 48 | 90 | DS-R1-7B | SVAMP | 12 | 92.0833 | 91.55 | 0.5333 |
| 49 | 91 | DS-R1-7B | SVAMP | 12 | 92.2167 | 91.6333 | 0.5833 |
| 0 | 42 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 1 | 43 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 2 | 44 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 3 | 45 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 4 | 46 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 5 | 47 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 6 | 48 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 7 | 49 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 8 | 50 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 9 | 51 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 10 | 52 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 11 | 53 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 12 | 54 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 13 | 55 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 14 | 56 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 15 | 57 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 16 | 58 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 17 | 59 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 18 | 60 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 19 | 61 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 20 | 62 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 21 | 63 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 22 | 64 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 23 | 65 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 24 | 66 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 25 | 67 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 26 | 68 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 27 | 69 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 28 | 70 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 29 | 71 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 30 | 72 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 31 | 73 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 32 | 74 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 33 | 75 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 34 | 76 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 35 | 77 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 36 | 78 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 37 | 79 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 38 | 80 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 39 | 81 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 40 | 82 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 41 | 83 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 42 | 84 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 43 | 85 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 44 | 86 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 45 | 87 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 46 | 88 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 47 | 89 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 48 | 90 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 49 | 91 | DS-R1-7B | SVAMP | 16 | 92.1875 | 91.625 | 0.5625 |
| 0 | 42 | Gemma-7B | AQuA | 4 | 30.315 | 27.3622 | 2.9528 |
| 1 | 43 | Gemma-7B | AQuA | 4 | 27.5591 | 27.3622 | 0.1969 |
| 2 | 44 | Gemma-7B | AQuA | 4 | 25.5906 | 29.3307 | -3.7402 |
| 3 | 45 | Gemma-7B | AQuA | 4 | 26.1811 | 27.9528 | -1.7717 |
| 4 | 46 | Gemma-7B | AQuA | 4 | 25.7874 | 25.7874 | 0 |
| 5 | 47 | Gemma-7B | AQuA | 4 | 28.1496 | 28.1496 | 0 |
| 6 | 48 | Gemma-7B | AQuA | 4 | 26.5748 | 30.9055 | -4.3307 |
| 7 | 49 | Gemma-7B | AQuA | 4 | 27.7559 | 30.5118 | -2.7559 |
| 8 | 50 | Gemma-7B | AQuA | 4 | 26.378 | 28.3465 | -1.9685 |
| 9 | 51 | Gemma-7B | AQuA | 4 | 26.1811 | 27.5591 | -1.378 |
| 10 | 52 | Gemma-7B | AQuA | 4 | 27.5591 | 26.7717 | 0.7874 |
| 11 | 53 | Gemma-7B | AQuA | 4 | 28.937 | 32.874 | -3.937 |
| 12 | 54 | Gemma-7B | AQuA | 4 | 29.3307 | 26.5748 | 2.7559 |
| 13 | 55 | Gemma-7B | AQuA | 4 | 31.2992 | 30.9055 | 0.3937 |
| 14 | 56 | Gemma-7B | AQuA | 4 | 29.5276 | 28.1496 | 1.378 |
| 15 | 57 | Gemma-7B | AQuA | 4 | 26.5748 | 26.9685 | -0.3937 |
| 16 | 58 | Gemma-7B | AQuA | 4 | 25.1969 | 28.937 | -3.7402 |
| 17 | 59 | Gemma-7B | AQuA | 4 | 29.1339 | 31.8898 | -2.7559 |
| 18 | 60 | Gemma-7B | AQuA | 4 | 28.1496 | 29.5276 | -1.378 |
| 19 | 61 | Gemma-7B | AQuA | 4 | 26.5748 | 27.5591 | -0.9843 |
| 20 | 62 | Gemma-7B | AQuA | 4 | 26.378 | 28.1496 | -1.7717 |
| 21 | 63 | Gemma-7B | AQuA | 4 | 29.1339 | 26.378 | 2.7559 |
| 22 | 64 | Gemma-7B | AQuA | 4 | 25.1969 | 29.5276 | -4.3307 |
| 23 | 65 | Gemma-7B | AQuA | 4 | 25 | 26.378 | -1.378 |
| 24 | 66 | Gemma-7B | AQuA | 4 | 28.3465 | 25.7874 | 2.5591 |
| 25 | 67 | Gemma-7B | AQuA | 4 | 28.5433 | 29.9213 | -1.378 |
| 26 | 68 | Gemma-7B | AQuA | 4 | 28.3465 | 30.7087 | -2.3622 |
| 27 | 69 | Gemma-7B | AQuA | 4 | 27.7559 | 26.9685 | 0.7874 |
| 28 | 70 | Gemma-7B | AQuA | 4 | 28.1496 | 26.378 | 1.7717 |
| 29 | 71 | Gemma-7B | AQuA | 4 | 28.3465 | 29.1339 | -0.7874 |
| 30 | 72 | Gemma-7B | AQuA | 4 | 28.937 | 27.9528 | 0.9843 |
| 31 | 73 | Gemma-7B | AQuA | 4 | 25.3937 | 27.5591 | -2.1654 |
| 32 | 74 | Gemma-7B | AQuA | 4 | 28.3465 | 28.1496 | 0.1969 |
| 33 | 75 | Gemma-7B | AQuA | 4 | 28.1496 | 28.7402 | -0.5906 |
| 34 | 76 | Gemma-7B | AQuA | 4 | 25.9843 | 28.1496 | -2.1654 |
| 35 | 77 | Gemma-7B | AQuA | 4 | 27.7559 | 30.1181 | -2.3622 |
| 36 | 78 | Gemma-7B | AQuA | 4 | 25.3937 | 28.5433 | -3.1496 |
| 37 | 79 | Gemma-7B | AQuA | 4 | 27.5591 | 29.9213 | -2.3622 |
| 38 | 80 | Gemma-7B | AQuA | 4 | 29.5276 | 27.7559 | 1.7717 |
| 39 | 81 | Gemma-7B | AQuA | 4 | 28.7402 | 26.5748 | 2.1654 |
| 40 | 82 | Gemma-7B | AQuA | 4 | 26.7717 | 28.3465 | -1.5748 |
| 41 | 83 | Gemma-7B | AQuA | 4 | 25.7874 | 28.937 | -3.1496 |
| 42 | 84 | Gemma-7B | AQuA | 4 | 28.5433 | 27.5591 | 0.9843 |
| 43 | 85 | Gemma-7B | AQuA | 4 | 27.3622 | 30.5118 | -3.1496 |
| 44 | 86 | Gemma-7B | AQuA | 4 | 28.3465 | 29.5276 | -1.1811 |
| 45 | 87 | Gemma-7B | AQuA | 4 | 29.7244 | 26.5748 | 3.1496 |
| 46 | 88 | Gemma-7B | AQuA | 4 | 27.7559 | 29.5276 | -1.7717 |
| 47 | 89 | Gemma-7B | AQuA | 4 | 29.3307 | 30.1181 | -0.7874 |
| 48 | 90 | Gemma-7B | AQuA | 4 | 27.1654 | 29.3307 | -2.1654 |
| 49 | 91 | Gemma-7B | AQuA | 4 | 28.5433 | 27.9528 | 0.5906 |
| 0 | 42 | Gemma-7B | AQuA | 8 | 28.937 | 27.7559 | 1.1811 |
| 1 | 43 | Gemma-7B | AQuA | 8 | 26.5748 | 27.8543 | -1.2795 |
| 2 | 44 | Gemma-7B | AQuA | 8 | 27.1654 | 29.2323 | -2.0669 |
| 3 | 45 | Gemma-7B | AQuA | 8 | 26.2795 | 29.3307 | -3.0512 |
| 4 | 46 | Gemma-7B | AQuA | 8 | 26.0827 | 27.2638 | -1.1811 |
| 5 | 47 | Gemma-7B | AQuA | 8 | 29.3307 | 28.6417 | 0.689 |
| 6 | 48 | Gemma-7B | AQuA | 8 | 28.5433 | 28.5433 | 0 |
| 7 | 49 | Gemma-7B | AQuA | 8 | 26.1811 | 30.0197 | -3.8386 |
| 8 | 50 | Gemma-7B | AQuA | 8 | 27.1654 | 28.6417 | -1.4764 |
| 9 | 51 | Gemma-7B | AQuA | 8 | 27.0669 | 27.8543 | -0.7874 |
| 10 | 52 | Gemma-7B | AQuA | 8 | 26.6732 | 28.6417 | -1.9685 |
| 11 | 53 | Gemma-7B | AQuA | 8 | 29.2323 | 29.9213 | -0.689 |
| 12 | 54 | Gemma-7B | AQuA | 8 | 29.4291 | 28.8386 | 0.5906 |
| 13 | 55 | Gemma-7B | AQuA | 8 | 28.8386 | 28.3465 | 0.4921 |
| 14 | 56 | Gemma-7B | AQuA | 8 | 28.1496 | 28.248 | -0.0984 |
| 15 | 57 | Gemma-7B | AQuA | 8 | 27.3622 | 27.1654 | 0.1969 |
| 16 | 58 | Gemma-7B | AQuA | 8 | 26.8701 | 27.3622 | -0.4921 |
| 17 | 59 | Gemma-7B | AQuA | 8 | 27.4606 | 29.2323 | -1.7717 |
| 18 | 60 | Gemma-7B | AQuA | 8 | 28.6417 | 26.8701 | 1.7717 |
| 19 | 61 | Gemma-7B | AQuA | 8 | 26.378 | 27.3622 | -0.9843 |
| 20 | 62 | Gemma-7B | AQuA | 8 | 26.7717 | 28.8386 | -2.0669 |
| 21 | 63 | Gemma-7B | AQuA | 8 | 28.248 | 27.3622 | 0.8858 |
| 22 | 64 | Gemma-7B | AQuA | 8 | 27.8543 | 27.1654 | 0.689 |
| 23 | 65 | Gemma-7B | AQuA | 8 | 29.1339 | 28.3465 | 0.7874 |
| 24 | 66 | Gemma-7B | AQuA | 8 | 27.7559 | 26.9685 | 0.7874 |
| 25 | 67 | Gemma-7B | AQuA | 8 | 27.6575 | 28.937 | -1.2795 |
| 26 | 68 | Gemma-7B | AQuA | 8 | 28.1496 | 28.937 | -0.7874 |
| 27 | 69 | Gemma-7B | AQuA | 8 | 28.8386 | 26.1811 | 2.6575 |
| 28 | 70 | Gemma-7B | AQuA | 8 | 27.9528 | 28.248 | -0.2953 |
| 29 | 71 | Gemma-7B | AQuA | 8 | 27.7559 | 28.8386 | -1.0827 |
| 30 | 72 | Gemma-7B | AQuA | 8 | 27.8543 | 27.4606 | 0.3937 |
| 31 | 73 | Gemma-7B | AQuA | 8 | 26.1811 | 27.8543 | -1.6732 |
| 32 | 74 | Gemma-7B | AQuA | 8 | 28.3465 | 29.1339 | -0.7874 |
| 33 | 75 | Gemma-7B | AQuA | 8 | 27.7559 | 28.1496 | -0.3937 |
| 34 | 76 | Gemma-7B | AQuA | 8 | 28.248 | 27.2638 | 0.9843 |
| 35 | 77 | Gemma-7B | AQuA | 8 | 27.7559 | 27.8543 | -0.0984 |
| 36 | 78 | Gemma-7B | AQuA | 8 | 25.689 | 27.5591 | -1.8701 |
| 37 | 79 | Gemma-7B | AQuA | 8 | 28.1496 | 29.1339 | -0.9843 |
| 38 | 80 | Gemma-7B | AQuA | 8 | 29.3307 | 28.5433 | 0.7874 |
| 39 | 81 | Gemma-7B | AQuA | 8 | 28.7402 | 27.8543 | 0.8858 |
| 40 | 82 | Gemma-7B | AQuA | 8 | 26.378 | 28.1496 | -1.7717 |
| 41 | 83 | Gemma-7B | AQuA | 8 | 26.5748 | 30.2165 | -3.6417 |
| 42 | 84 | Gemma-7B | AQuA | 8 | 27.9528 | 28.4449 | -0.4921 |
| 43 | 85 | Gemma-7B | AQuA | 8 | 29.626 | 29.1339 | 0.4921 |
| 44 | 86 | Gemma-7B | AQuA | 8 | 27.1654 | 30.4134 | -3.248 |
| 45 | 87 | Gemma-7B | AQuA | 8 | 29.0354 | 28.3465 | 0.689 |
| 46 | 88 | Gemma-7B | AQuA | 8 | 29.2323 | 28.7402 | 0.4921 |
| 47 | 89 | Gemma-7B | AQuA | 8 | 26.5748 | 28.8386 | -2.2638 |
| 48 | 90 | Gemma-7B | AQuA | 8 | 27.6575 | 28.7402 | -1.0827 |
| 49 | 91 | Gemma-7B | AQuA | 8 | 27.7559 | 28.248 | -0.4921 |
| 0 | 42 | Gemma-7B | AQuA | 12 | 27.6903 | 28.2808 | -0.5906 |
| 1 | 43 | Gemma-7B | AQuA | 12 | 27.6903 | 28.084 | -0.3937 |
| 2 | 44 | Gemma-7B | AQuA | 12 | 27.1654 | 28.4121 | -1.2467 |
| 3 | 45 | Gemma-7B | AQuA | 12 | 27.6247 | 28.084 | -0.4593 |
| 4 | 46 | Gemma-7B | AQuA | 12 | 26.706 | 28.084 | -1.378 |
| 5 | 47 | Gemma-7B | AQuA | 12 | 27.9528 | 28.1496 | -0.1969 |
| 6 | 48 | Gemma-7B | AQuA | 12 | 28.4777 | 26.9029 | 1.5748 |
| 7 | 49 | Gemma-7B | AQuA | 12 | 27.6247 | 28.3465 | -0.7218 |
| 8 | 50 | Gemma-7B | AQuA | 12 | 28.6089 | 27.6903 | 0.9186 |
| 9 | 51 | Gemma-7B | AQuA | 12 | 27.4934 | 27.6903 | -0.1969 |
| 10 | 52 | Gemma-7B | AQuA | 12 | 27.6903 | 27.8215 | -0.1312 |
| 11 | 53 | Gemma-7B | AQuA | 12 | 28.6089 | 27.8871 | 0.7218 |
| 12 | 54 | Gemma-7B | AQuA | 12 | 29.1995 | 28.3465 | 0.853 |
| 13 | 55 | Gemma-7B | AQuA | 12 | 28.2808 | 27.4934 | 0.7874 |
| 14 | 56 | Gemma-7B | AQuA | 12 | 28.3465 | 28.8714 | -0.5249 |
| 15 | 57 | Gemma-7B | AQuA | 12 | 28.2152 | 27.9528 | 0.2625 |
| 16 | 58 | Gemma-7B | AQuA | 12 | 28.084 | 26.706 | 1.378 |
| 17 | 59 | Gemma-7B | AQuA | 12 | 27.7559 | 28.5433 | -0.7874 |
| 18 | 60 | Gemma-7B | AQuA | 12 | 28.1496 | 26.6404 | 1.5092 |
| 19 | 61 | Gemma-7B | AQuA | 12 | 27.0341 | 27.2966 | -0.2625 |
| 20 | 62 | Gemma-7B | AQuA | 12 | 27.8215 | 28.5433 | -0.7218 |
| 21 | 63 | Gemma-7B | AQuA | 12 | 27.5591 | 27.8215 | -0.2625 |
| 22 | 64 | Gemma-7B | AQuA | 12 | 27.6903 | 27.5591 | 0.1312 |
| 23 | 65 | Gemma-7B | AQuA | 12 | 27.4278 | 28.0184 | -0.5906 |
| 24 | 66 | Gemma-7B | AQuA | 12 | 27.0997 | 27.1654 | -0.0656 |
| 25 | 67 | Gemma-7B | AQuA | 12 | 28.1496 | 27.2966 | 0.853 |
| 26 | 68 | Gemma-7B | AQuA | 12 | 27.2966 | 28.0184 | -0.7218 |
| 27 | 69 | Gemma-7B | AQuA | 12 | 27.8215 | 27.0341 | 0.7874 |
| 28 | 70 | Gemma-7B | AQuA | 12 | 28.084 | 27.5591 | 0.5249 |
| 29 | 71 | Gemma-7B | AQuA | 12 | 27.9528 | 27.5591 | 0.3937 |
| 30 | 72 | Gemma-7B | AQuA | 12 | 28.2808 | 27.5591 | 0.7218 |
| 31 | 73 | Gemma-7B | AQuA | 12 | 27.3622 | 27.5591 | -0.1969 |
| 32 | 74 | Gemma-7B | AQuA | 12 | 27.4278 | 28.937 | -1.5092 |
| 33 | 75 | Gemma-7B | AQuA | 12 | 28.5433 | 27.6247 | 0.9186 |
| 34 | 76 | Gemma-7B | AQuA | 12 | 27.8871 | 28.1496 | -0.2625 |
| 35 | 77 | Gemma-7B | AQuA | 12 | 28.6745 | 27.3622 | 1.3123 |
| 36 | 78 | Gemma-7B | AQuA | 12 | 27.1654 | 27.4278 | -0.2625 |
| 37 | 79 | Gemma-7B | AQuA | 12 | 28.1496 | 28.6089 | -0.4593 |
| 38 | 80 | Gemma-7B | AQuA | 12 | 29.2651 | 28.1496 | 1.1155 |
| 39 | 81 | Gemma-7B | AQuA | 12 | 28.3465 | 28.4121 | -0.0656 |
| 40 | 82 | Gemma-7B | AQuA | 12 | 27.8871 | 28.1496 | -0.2625 |
| 41 | 83 | Gemma-7B | AQuA | 12 | 28.3465 | 27.3622 | 0.9843 |
| 42 | 84 | Gemma-7B | AQuA | 12 | 27.8871 | 27.4278 | 0.4593 |
| 43 | 85 | Gemma-7B | AQuA | 12 | 27.6247 | 29.1339 | -1.5092 |
| 44 | 86 | Gemma-7B | AQuA | 12 | 28.3465 | 29.3307 | -0.9843 |
| 45 | 87 | Gemma-7B | AQuA | 12 | 27.9528 | 27.8871 | 0.0656 |
| 46 | 88 | Gemma-7B | AQuA | 12 | 28.2152 | 27.7559 | 0.4593 |
| 47 | 89 | Gemma-7B | AQuA | 12 | 28.0184 | 27.4278 | 0.5906 |
| 48 | 90 | Gemma-7B | AQuA | 12 | 28.8714 | 27.8871 | 0.9843 |
| 49 | 91 | Gemma-7B | AQuA | 12 | 26.9685 | 27.231 | -0.2625 |
| 0 | 42 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 1 | 43 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 2 | 44 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 3 | 45 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 4 | 46 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 5 | 47 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 6 | 48 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 7 | 49 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 8 | 50 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 9 | 51 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 10 | 52 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 11 | 53 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 12 | 54 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 13 | 55 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 14 | 56 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 15 | 57 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 16 | 58 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 17 | 59 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 18 | 60 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 19 | 61 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 20 | 62 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 21 | 63 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 22 | 64 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 23 | 65 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 24 | 66 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 25 | 67 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 26 | 68 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 27 | 69 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 28 | 70 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 29 | 71 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 30 | 72 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 31 | 73 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 32 | 74 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 33 | 75 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 34 | 76 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 35 | 77 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 36 | 78 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 37 | 79 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 38 | 80 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 39 | 81 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 40 | 82 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 41 | 83 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 42 | 84 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 43 | 85 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 44 | 86 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 45 | 87 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 46 | 88 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 47 | 89 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 48 | 90 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 49 | 91 | Gemma-7B | AQuA | 16 | 27.6575 | 28.1004 | -0.4429 |
| 0 | 42 | Gemma-7B | CommonsenseQA | 4 | 18.5504 | 19.3284 | -0.7781 |
| 1 | 43 | Gemma-7B | CommonsenseQA | 4 | 17.7314 | 18.4275 | -0.6962 |
| 2 | 44 | Gemma-7B | CommonsenseQA | 4 | 17.4857 | 19.0008 | -1.5152 |
| 3 | 45 | Gemma-7B | CommonsenseQA | 4 | 19.4513 | 18.5094 | 0.9419 |
| 4 | 46 | Gemma-7B | CommonsenseQA | 4 | 18.2228 | 19.4103 | -1.1876 |
| 5 | 47 | Gemma-7B | CommonsenseQA | 4 | 17.9361 | 17.24 | 0.6962 |
| 6 | 48 | Gemma-7B | CommonsenseQA | 4 | 18.1409 | 18.6732 | -0.5324 |
| 7 | 49 | Gemma-7B | CommonsenseQA | 4 | 17.1171 | 18.878 | -1.7609 |
| 8 | 50 | Gemma-7B | CommonsenseQA | 4 | 17.7723 | 18.5504 | -0.7781 |
| 9 | 51 | Gemma-7B | CommonsenseQA | 4 | 18.9599 | 18.6323 | 0.3276 |
| 10 | 52 | Gemma-7B | CommonsenseQA | 4 | 17.4038 | 18.2637 | -0.86 |
| 11 | 53 | Gemma-7B | CommonsenseQA | 4 | 18.3866 | 18.5913 | -0.2048 |
| 12 | 54 | Gemma-7B | CommonsenseQA | 4 | 17.0352 | 18.018 | -0.9828 |
| 13 | 55 | Gemma-7B | CommonsenseQA | 4 | 17.6495 | 18.9189 | -1.2695 |
| 14 | 56 | Gemma-7B | CommonsenseQA | 4 | 17.5676 | 18.3866 | -0.819 |
| 15 | 57 | Gemma-7B | CommonsenseQA | 4 | 18.4685 | 17.5676 | 0.9009 |
| 16 | 58 | Gemma-7B | CommonsenseQA | 4 | 18.1818 | 18.5913 | -0.4095 |
| 17 | 59 | Gemma-7B | CommonsenseQA | 4 | 17.4447 | 18.2228 | -0.7781 |
| 18 | 60 | Gemma-7B | CommonsenseQA | 4 | 17.199 | 19.5741 | -2.3751 |
| 19 | 61 | Gemma-7B | CommonsenseQA | 4 | 18.9189 | 19.7379 | -0.819 |
| 20 | 62 | Gemma-7B | CommonsenseQA | 4 | 18.5094 | 18.9189 | -0.4095 |
| 21 | 63 | Gemma-7B | CommonsenseQA | 4 | 17.7723 | 19.5332 | -1.7609 |
| 22 | 64 | Gemma-7B | CommonsenseQA | 4 | 18.5913 | 20.0246 | -1.4333 |
| 23 | 65 | Gemma-7B | CommonsenseQA | 4 | 18.6323 | 18.878 | -0.2457 |
| 24 | 66 | Gemma-7B | CommonsenseQA | 4 | 17.7314 | 18.878 | -1.1466 |
| 25 | 67 | Gemma-7B | CommonsenseQA | 4 | 18.5504 | 18.2228 | 0.3276 |
| 26 | 68 | Gemma-7B | CommonsenseQA | 4 | 17.6904 | 19.2875 | -1.5971 |
| 27 | 69 | Gemma-7B | CommonsenseQA | 4 | 18.5504 | 20.1884 | -1.638 |
| 28 | 70 | Gemma-7B | CommonsenseQA | 4 | 16.5848 | 18.1818 | -1.5971 |
| 29 | 71 | Gemma-7B | CommonsenseQA | 4 | 17.8952 | 19.656 | -1.7609 |
| 30 | 72 | Gemma-7B | CommonsenseQA | 4 | 17.6495 | 19.2465 | -1.5971 |
| 31 | 73 | Gemma-7B | CommonsenseQA | 4 | 16.7076 | 19.6151 | -2.9075 |
| 32 | 74 | Gemma-7B | CommonsenseQA | 4 | 18.3456 | 19.6151 | -1.2695 |
| 33 | 75 | Gemma-7B | CommonsenseQA | 4 | 17.7723 | 19.2465 | -1.4742 |
| 34 | 76 | Gemma-7B | CommonsenseQA | 4 | 18.3047 | 18.9189 | -0.6143 |
| 35 | 77 | Gemma-7B | CommonsenseQA | 4 | 17.8133 | 18.1818 | -0.3686 |
| 36 | 78 | Gemma-7B | CommonsenseQA | 4 | 16.8714 | 19.0418 | -2.1704 |
| 37 | 79 | Gemma-7B | CommonsenseQA | 4 | 16.4619 | 18.9189 | -2.457 |
| 38 | 80 | Gemma-7B | CommonsenseQA | 4 | 17.8952 | 18.5504 | -0.6552 |
| 39 | 81 | Gemma-7B | CommonsenseQA | 4 | 19.4513 | 19.1237 | 0.3276 |
| 40 | 82 | Gemma-7B | CommonsenseQA | 4 | 18.3456 | 18.5094 | -0.1638 |
| 41 | 83 | Gemma-7B | CommonsenseQA | 4 | 17.1581 | 18.4685 | -1.3104 |
| 42 | 84 | Gemma-7B | CommonsenseQA | 4 | 18.6323 | 18.7961 | -0.1638 |
| 43 | 85 | Gemma-7B | CommonsenseQA | 4 | 18.3456 | 19.7789 | -1.4333 |
| 44 | 86 | Gemma-7B | CommonsenseQA | 4 | 17.7314 | 18.9599 | -1.2285 |
| 45 | 87 | Gemma-7B | CommonsenseQA | 4 | 18.4685 | 18.2228 | 0.2457 |
| 46 | 88 | Gemma-7B | CommonsenseQA | 4 | 18.6323 | 17.8952 | 0.7371 |
| 47 | 89 | Gemma-7B | CommonsenseQA | 4 | 17.9771 | 18.837 | -0.86 |
| 48 | 90 | Gemma-7B | CommonsenseQA | 4 | 18.5504 | 19.4513 | -0.9009 |
| 49 | 91 | Gemma-7B | CommonsenseQA | 4 | 18.2228 | 18.6732 | -0.4505 |
| 0 | 42 | Gemma-7B | CommonsenseQA | 8 | 18.4275 | 18.8575 | -0.43 |
| 1 | 43 | Gemma-7B | CommonsenseQA | 8 | 17.9361 | 19.1646 | -1.2285 |
| 2 | 44 | Gemma-7B | CommonsenseQA | 8 | 17.8747 | 19.3898 | -1.5152 |
| 3 | 45 | Gemma-7B | CommonsenseQA | 8 | 17.3833 | 18.837 | -1.4537 |
| 4 | 46 | Gemma-7B | CommonsenseQA | 8 | 18.4685 | 19.697 | -1.2285 |
| 5 | 47 | Gemma-7B | CommonsenseQA | 8 | 17.7928 | 19.0213 | -1.2285 |
| 6 | 48 | Gemma-7B | CommonsenseQA | 8 | 18.059 | 19.1441 | -1.0852 |
| 7 | 49 | Gemma-7B | CommonsenseQA | 8 | 17.5676 | 19.4103 | -1.8428 |
| 8 | 50 | Gemma-7B | CommonsenseQA | 8 | 17.6904 | 18.878 | -1.1876 |
| 9 | 51 | Gemma-7B | CommonsenseQA | 8 | 18.018 | 19.3898 | -1.3718 |
| 10 | 52 | Gemma-7B | CommonsenseQA | 8 | 17.7109 | 19.0418 | -1.3309 |
| 11 | 53 | Gemma-7B | CommonsenseQA | 8 | 17.5676 | 19.3284 | -1.7609 |
| 12 | 54 | Gemma-7B | CommonsenseQA | 8 | 17.5471 | 18.3456 | -0.7985 |
| 13 | 55 | Gemma-7B | CommonsenseQA | 8 | 17.2195 | 18.7346 | -1.5152 |
| 14 | 56 | Gemma-7B | CommonsenseQA | 8 | 17.588 | 18.5504 | -0.9623 |
| 15 | 57 | Gemma-7B | CommonsenseQA | 8 | 18.1204 | 18.0385 | 0.0819 |
| 16 | 58 | Gemma-7B | CommonsenseQA | 8 | 17.5061 | 18.8575 | -1.3514 |
| 17 | 59 | Gemma-7B | CommonsenseQA | 8 | 17.0352 | 18.9599 | -1.9247 |
| 18 | 60 | Gemma-7B | CommonsenseQA | 8 | 17.5266 | 19.2875 | -1.7609 |
| 19 | 61 | Gemma-7B | CommonsenseQA | 8 | 18.2432 | 19.5127 | -1.2695 |
| 20 | 62 | Gemma-7B | CommonsenseQA | 8 | 17.199 | 18.9803 | -1.7813 |
| 21 | 63 | Gemma-7B | CommonsenseQA | 8 | 17.6085 | 19.5127 | -1.9042 |
| 22 | 64 | Gemma-7B | CommonsenseQA | 8 | 17.7928 | 19.3694 | -1.5766 |
| 23 | 65 | Gemma-7B | CommonsenseQA | 8 | 17.6495 | 19.226 | -1.5766 |
| 24 | 66 | Gemma-7B | CommonsenseQA | 8 | 17.0966 | 19.4513 | -2.3546 |
| 25 | 67 | Gemma-7B | CommonsenseQA | 8 | 17.3219 | 18.6937 | -1.3718 |
| 26 | 68 | Gemma-7B | CommonsenseQA | 8 | 16.8305 | 19.4513 | -2.6208 |
| 27 | 69 | Gemma-7B | CommonsenseQA | 8 | 17.6904 | 20.3112 | -2.6208 |
| 28 | 70 | Gemma-7B | CommonsenseQA | 8 | 17.2604 | 18.9599 | -1.6994 |
| 29 | 71 | Gemma-7B | CommonsenseQA | 8 | 18.3456 | 19.4922 | -1.1466 |
| 30 | 72 | Gemma-7B | CommonsenseQA | 8 | 17.5266 | 19.3694 | -1.8428 |
| 31 | 73 | Gemma-7B | CommonsenseQA | 8 | 17.4242 | 19.5741 | -2.1499 |
| 32 | 74 | Gemma-7B | CommonsenseQA | 8 | 17.2604 | 19.267 | -2.0066 |
| 33 | 75 | Gemma-7B | CommonsenseQA | 8 | 16.9124 | 18.5504 | -1.638 |
| 34 | 76 | Gemma-7B | CommonsenseQA | 8 | 17.7314 | 19.3284 | -1.5971 |
| 35 | 77 | Gemma-7B | CommonsenseQA | 8 | 17.5061 | 18.8165 | -1.3104 |
| 36 | 78 | Gemma-7B | CommonsenseQA | 8 | 17.199 | 18.9189 | -1.7199 |
| 37 | 79 | Gemma-7B | CommonsenseQA | 8 | 17.0762 | 19.656 | -2.5799 |
| 38 | 80 | Gemma-7B | CommonsenseQA | 8 | 17.5061 | 19.6355 | -2.1294 |
| 39 | 81 | Gemma-7B | CommonsenseQA | 8 | 18.448 | 19.1646 | -0.7166 |
| 40 | 82 | Gemma-7B | CommonsenseQA | 8 | 17.8542 | 18.7551 | -0.9009 |
| 41 | 83 | Gemma-7B | CommonsenseQA | 8 | 17.2604 | 19.0827 | -1.8223 |
| 42 | 84 | Gemma-7B | CommonsenseQA | 8 | 17.5676 | 19.4513 | -1.8837 |
| 43 | 85 | Gemma-7B | CommonsenseQA | 8 | 17.0966 | 19.5332 | -2.4365 |
| 44 | 86 | Gemma-7B | CommonsenseQA | 8 | 17.8337 | 18.7961 | -0.9623 |
| 45 | 87 | Gemma-7B | CommonsenseQA | 8 | 17.9566 | 19.3284 | -1.3718 |
| 46 | 88 | Gemma-7B | CommonsenseQA | 8 | 17.1171 | 19.0008 | -1.8837 |
| 47 | 89 | Gemma-7B | CommonsenseQA | 8 | 17.6495 | 18.7961 | -1.1466 |
| 48 | 90 | Gemma-7B | CommonsenseQA | 8 | 17.0352 | 19.5127 | -2.4775 |
| 49 | 91 | Gemma-7B | CommonsenseQA | 8 | 17.5471 | 18.407 | -0.86 |
| 0 | 42 | Gemma-7B | CommonsenseQA | 12 | 17.6495 | 19.4103 | -1.7609 |
| 1 | 43 | Gemma-7B | CommonsenseQA | 12 | 17.5266 | 19.1919 | -1.6653 |
| 2 | 44 | Gemma-7B | CommonsenseQA | 12 | 17.7314 | 19.424 | -1.6926 |
| 3 | 45 | Gemma-7B | CommonsenseQA | 12 | 17.7041 | 18.9462 | -1.2422 |
| 4 | 46 | Gemma-7B | CommonsenseQA | 12 | 17.513 | 19.6424 | -2.1294 |
| 5 | 47 | Gemma-7B | CommonsenseQA | 12 | 17.5539 | 19.2738 | -1.7199 |
| 6 | 48 | Gemma-7B | CommonsenseQA | 12 | 17.9225 | 19.3421 | -1.4196 |
| 7 | 49 | Gemma-7B | CommonsenseQA | 12 | 17.4174 | 19.5195 | -2.1021 |
| 8 | 50 | Gemma-7B | CommonsenseQA | 12 | 17.4447 | 19.0554 | -1.6107 |
| 9 | 51 | Gemma-7B | CommonsenseQA | 12 | 17.5266 | 19.1783 | -1.6517 |
| 10 | 52 | Gemma-7B | CommonsenseQA | 12 | 18.1682 | 18.7415 | -0.5733 |
| 11 | 53 | Gemma-7B | CommonsenseQA | 12 | 17.2263 | 19.4376 | -2.2113 |
| 12 | 54 | Gemma-7B | CommonsenseQA | 12 | 17.3492 | 18.9735 | -1.6244 |
| 13 | 55 | Gemma-7B | CommonsenseQA | 12 | 17.5676 | 19.2875 | -1.7199 |
| 14 | 56 | Gemma-7B | CommonsenseQA | 12 | 17.513 | 19.0008 | -1.4879 |
| 15 | 57 | Gemma-7B | CommonsenseQA | 12 | 17.6631 | 18.8097 | -1.1466 |
| 16 | 58 | Gemma-7B | CommonsenseQA | 12 | 17.5539 | 19.0964 | -1.5425 |
| 17 | 59 | Gemma-7B | CommonsenseQA | 12 | 17.3765 | 19.1237 | -1.7472 |
| 18 | 60 | Gemma-7B | CommonsenseQA | 12 | 17.6222 | 19.3148 | -1.6926 |
| 19 | 61 | Gemma-7B | CommonsenseQA | 12 | 17.6222 | 19.2056 | -1.5834 |
| 20 | 62 | Gemma-7B | CommonsenseQA | 12 | 17.4174 | 19.3148 | -1.8974 |
| 21 | 63 | Gemma-7B | CommonsenseQA | 12 | 17.513 | 19.2738 | -1.7609 |
| 22 | 64 | Gemma-7B | CommonsenseQA | 12 | 17.745 | 19.2329 | -1.4879 |
| 23 | 65 | Gemma-7B | CommonsenseQA | 12 | 17.2946 | 19.2192 | -1.9247 |
| 24 | 66 | Gemma-7B | CommonsenseQA | 12 | 17.1308 | 19.3011 | -2.1704 |
| 25 | 67 | Gemma-7B | CommonsenseQA | 12 | 17.6085 | 18.9872 | -1.3787 |
| 26 | 68 | Gemma-7B | CommonsenseQA | 12 | 17.3492 | 19.7379 | -2.3888 |
| 27 | 69 | Gemma-7B | CommonsenseQA | 12 | 17.2946 | 20.0246 | -2.73 |
| 28 | 70 | Gemma-7B | CommonsenseQA | 12 | 17.4311 | 19.0145 | -1.5834 |
| 29 | 71 | Gemma-7B | CommonsenseQA | 12 | 17.5539 | 19.2465 | -1.6926 |
| 30 | 72 | Gemma-7B | CommonsenseQA | 12 | 17.1717 | 19.4103 | -2.2386 |
| 31 | 73 | Gemma-7B | CommonsenseQA | 12 | 17.8542 | 19.4103 | -1.5561 |
| 32 | 74 | Gemma-7B | CommonsenseQA | 12 | 16.9533 | 19.3557 | -2.4024 |
| 33 | 75 | Gemma-7B | CommonsenseQA | 12 | 17.2946 | 19.0008 | -1.7063 |
| 34 | 76 | Gemma-7B | CommonsenseQA | 12 | 17.0898 | 19.4376 | -2.3478 |
| 35 | 77 | Gemma-7B | CommonsenseQA | 12 | 17.3219 | 19.383 | -2.0612 |
| 36 | 78 | Gemma-7B | CommonsenseQA | 12 | 17.7041 | 19.3694 | -1.6653 |
| 37 | 79 | Gemma-7B | CommonsenseQA | 12 | 17.3219 | 19.6014 | -2.2796 |
| 38 | 80 | Gemma-7B | CommonsenseQA | 12 | 17.0762 | 19.0827 | -2.0066 |
| 39 | 81 | Gemma-7B | CommonsenseQA | 12 | 17.6631 | 19.2056 | -1.5425 |
| 40 | 82 | Gemma-7B | CommonsenseQA | 12 | 17.1171 | 19.0418 | -1.9247 |
| 41 | 83 | Gemma-7B | CommonsenseQA | 12 | 17.2536 | 19.11 | -1.8564 |
| 42 | 84 | Gemma-7B | CommonsenseQA | 12 | 17.4584 | 19.3284 | -1.8701 |
| 43 | 85 | Gemma-7B | CommonsenseQA | 12 | 17.4857 | 19.5741 | -2.0885 |
| 44 | 86 | Gemma-7B | CommonsenseQA | 12 | 17.7723 | 19.1237 | -1.3514 |
| 45 | 87 | Gemma-7B | CommonsenseQA | 12 | 17.5266 | 19.424 | -1.8974 |
| 46 | 88 | Gemma-7B | CommonsenseQA | 12 | 17.2946 | 19.0008 | -1.7063 |
| 47 | 89 | Gemma-7B | CommonsenseQA | 12 | 17.2536 | 19.1373 | -1.8837 |
| 48 | 90 | Gemma-7B | CommonsenseQA | 12 | 17.513 | 19.0418 | -1.5288 |
| 49 | 91 | Gemma-7B | CommonsenseQA | 12 | 17.5676 | 19.0691 | -1.5015 |
| 0 | 42 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 1 | 43 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 2 | 44 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 3 | 45 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 4 | 46 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 5 | 47 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 6 | 48 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 7 | 49 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 8 | 50 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 9 | 51 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 10 | 52 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 11 | 53 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 12 | 54 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 13 | 55 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 14 | 56 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 15 | 57 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 16 | 58 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 17 | 59 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 18 | 60 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 19 | 61 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 20 | 62 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 21 | 63 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 22 | 64 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 23 | 65 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 24 | 66 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 25 | 67 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 26 | 68 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 27 | 69 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 28 | 70 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 29 | 71 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 30 | 72 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 31 | 73 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 32 | 74 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 33 | 75 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 34 | 76 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 35 | 77 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 36 | 78 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 37 | 79 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 38 | 80 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 39 | 81 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 40 | 82 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 41 | 83 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 42 | 84 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 43 | 85 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 44 | 86 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 45 | 87 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 46 | 88 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 47 | 89 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 48 | 90 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 49 | 91 | Gemma-7B | CommonsenseQA | 16 | 17.3423 | 19.3284 | -1.9861 |
| 0 | 42 | Gemma-7B | GPQA | 4 | 16.2946 | 25.558 | -9.2634 |
| 1 | 43 | Gemma-7B | GPQA | 4 | 18.5268 | 23.4375 | -4.9107 |
| 2 | 44 | Gemma-7B | GPQA | 4 | 16.8527 | 25.3348 | -8.4821 |
| 3 | 45 | Gemma-7B | GPQA | 4 | 16.8527 | 22.7679 | -5.9152 |
| 4 | 46 | Gemma-7B | GPQA | 4 | 16.0714 | 25 | -8.9286 |
| 5 | 47 | Gemma-7B | GPQA | 4 | 18.0804 | 27.0089 | -8.9286 |
| 6 | 48 | Gemma-7B | GPQA | 4 | 16.5179 | 23.3259 | -6.808 |
| 7 | 49 | Gemma-7B | GPQA | 4 | 14.2857 | 25.1116 | -10.8259 |
| 8 | 50 | Gemma-7B | GPQA | 4 | 15.4018 | 26.5625 | -11.1607 |
| 9 | 51 | Gemma-7B | GPQA | 4 | 17.0759 | 25.558 | -8.4821 |
| 10 | 52 | Gemma-7B | GPQA | 4 | 17.8571 | 26.4509 | -8.5937 |
| 11 | 53 | Gemma-7B | GPQA | 4 | 16.0714 | 25.4464 | -9.375 |
| 12 | 54 | Gemma-7B | GPQA | 4 | 15.9598 | 25.7812 | -9.8214 |
| 13 | 55 | Gemma-7B | GPQA | 4 | 16.2946 | 25.7812 | -9.4866 |
| 14 | 56 | Gemma-7B | GPQA | 4 | 18.192 | 25.2232 | -7.0312 |
| 15 | 57 | Gemma-7B | GPQA | 4 | 17.0759 | 25.7812 | -8.7054 |
| 16 | 58 | Gemma-7B | GPQA | 4 | 17.4107 | 24.3304 | -6.9196 |
| 17 | 59 | Gemma-7B | GPQA | 4 | 14.9554 | 26.5625 | -11.6071 |
| 18 | 60 | Gemma-7B | GPQA | 4 | 16.5179 | 24.6652 | -8.1473 |
| 19 | 61 | Gemma-7B | GPQA | 4 | 16.4062 | 23.6607 | -7.2545 |
| 20 | 62 | Gemma-7B | GPQA | 4 | 18.6384 | 25.1116 | -6.4732 |
| 21 | 63 | Gemma-7B | GPQA | 4 | 16.5179 | 22.8795 | -6.3616 |
| 22 | 64 | Gemma-7B | GPQA | 4 | 16.8527 | 26.5625 | -9.7098 |
| 23 | 65 | Gemma-7B | GPQA | 4 | 17.1875 | 24.2188 | -7.0312 |
| 24 | 66 | Gemma-7B | GPQA | 4 | 18.0804 | 23.9955 | -5.9152 |
| 25 | 67 | Gemma-7B | GPQA | 4 | 18.192 | 27.1205 | -8.9286 |
| 26 | 68 | Gemma-7B | GPQA | 4 | 16.9643 | 24.442 | -7.4777 |
| 27 | 69 | Gemma-7B | GPQA | 4 | 18.4152 | 25.1116 | -6.6964 |
| 28 | 70 | Gemma-7B | GPQA | 4 | 16.9643 | 23.3259 | -6.3616 |
| 29 | 71 | Gemma-7B | GPQA | 4 | 15.8482 | 24.2188 | -8.3705 |
| 30 | 72 | Gemma-7B | GPQA | 4 | 18.8616 | 25.4464 | -6.5848 |
| 31 | 73 | Gemma-7B | GPQA | 4 | 14.9554 | 21.9866 | -7.0312 |
| 32 | 74 | Gemma-7B | GPQA | 4 | 15.7366 | 23.3259 | -7.5893 |
| 33 | 75 | Gemma-7B | GPQA | 4 | 15.1786 | 25.558 | -10.3795 |
| 34 | 76 | Gemma-7B | GPQA | 4 | 17.6339 | 25.1116 | -7.4777 |
| 35 | 77 | Gemma-7B | GPQA | 4 | 17.4107 | 24.8884 | -7.4777 |
| 36 | 78 | Gemma-7B | GPQA | 4 | 17.7455 | 25.2232 | -7.4777 |
| 37 | 79 | Gemma-7B | GPQA | 4 | 17.0759 | 23.1027 | -6.0268 |
| 38 | 80 | Gemma-7B | GPQA | 4 | 14.7321 | 24.2188 | -9.4866 |
| 39 | 81 | Gemma-7B | GPQA | 4 | 15.2902 | 23.9955 | -8.7054 |
| 40 | 82 | Gemma-7B | GPQA | 4 | 15.4018 | 24.8884 | -9.4866 |
| 41 | 83 | Gemma-7B | GPQA | 4 | 17.7455 | 24.1071 | -6.3616 |
| 42 | 84 | Gemma-7B | GPQA | 4 | 17.9688 | 25 | -7.0312 |
| 43 | 85 | Gemma-7B | GPQA | 4 | 16.9643 | 25.2232 | -8.2589 |
| 44 | 86 | Gemma-7B | GPQA | 4 | 17.5223 | 23.6607 | -6.1384 |
| 45 | 87 | Gemma-7B | GPQA | 4 | 14.0625 | 23.6607 | -9.5982 |
| 46 | 88 | Gemma-7B | GPQA | 4 | 16.9643 | 24.1071 | -7.1429 |
| 47 | 89 | Gemma-7B | GPQA | 4 | 14.8438 | 27.0089 | -12.1652 |
| 48 | 90 | Gemma-7B | GPQA | 4 | 17.7455 | 25.7812 | -8.0357 |
| 49 | 91 | Gemma-7B | GPQA | 4 | 17.0759 | 28.125 | -11.0491 |
| 0 | 42 | Gemma-7B | GPQA | 8 | 16.2946 | 24.442 | -8.1473 |
| 1 | 43 | Gemma-7B | GPQA | 8 | 15.4576 | 26.1719 | -10.7143 |
| 2 | 44 | Gemma-7B | GPQA | 8 | 16.7969 | 24.7768 | -7.9799 |
| 3 | 45 | Gemma-7B | GPQA | 8 | 14.6763 | 24.4978 | -9.8214 |
| 4 | 46 | Gemma-7B | GPQA | 8 | 15.4576 | 25.1116 | -9.654 |
| 5 | 47 | Gemma-7B | GPQA | 8 | 16.7411 | 25.4464 | -8.7054 |
| 6 | 48 | Gemma-7B | GPQA | 8 | 15.0112 | 25.7254 | -10.7143 |
| 7 | 49 | Gemma-7B | GPQA | 8 | 15.2902 | 25.8371 | -10.5469 |
| 8 | 50 | Gemma-7B | GPQA | 8 | 15.5692 | 25.8929 | -10.3237 |
| 9 | 51 | Gemma-7B | GPQA | 8 | 15.067 | 25.6138 | -10.5469 |
| 10 | 52 | Gemma-7B | GPQA | 8 | 16.7969 | 25.279 | -8.4821 |
| 11 | 53 | Gemma-7B | GPQA | 8 | 15.4576 | 26.2277 | -10.7701 |
| 12 | 54 | Gemma-7B | GPQA | 8 | 14.8996 | 26.7299 | -11.8304 |
| 13 | 55 | Gemma-7B | GPQA | 8 | 14.5089 | 26.6741 | -12.1652 |
| 14 | 56 | Gemma-7B | GPQA | 8 | 17.5223 | 25.5022 | -7.9799 |
| 15 | 57 | Gemma-7B | GPQA | 8 | 17.4107 | 25.279 | -7.8683 |
| 16 | 58 | Gemma-7B | GPQA | 8 | 15.4576 | 25.4464 | -9.9888 |
| 17 | 59 | Gemma-7B | GPQA | 8 | 16.3504 | 25.6138 | -9.2634 |
| 18 | 60 | Gemma-7B | GPQA | 8 | 15.2344 | 26.3951 | -11.1607 |
| 19 | 61 | Gemma-7B | GPQA | 8 | 15.8482 | 24.442 | -8.5938 |
| 20 | 62 | Gemma-7B | GPQA | 8 | 16.9085 | 25 | -8.0915 |
| 21 | 63 | Gemma-7B | GPQA | 8 | 15.067 | 24.721 | -9.654 |
| 22 | 64 | Gemma-7B | GPQA | 8 | 15.5692 | 26.2277 | -10.6585 |
| 23 | 65 | Gemma-7B | GPQA | 8 | 15.625 | 24.442 | -8.817 |
| 24 | 66 | Gemma-7B | GPQA | 8 | 15.5692 | 26.3393 | -10.7701 |
| 25 | 67 | Gemma-7B | GPQA | 8 | 16.8527 | 26.0045 | -9.1518 |
| 26 | 68 | Gemma-7B | GPQA | 8 | 15.7924 | 25.4464 | -9.654 |
| 27 | 69 | Gemma-7B | GPQA | 8 | 15.2902 | 25.9487 | -10.6585 |
| 28 | 70 | Gemma-7B | GPQA | 8 | 16.5179 | 24.8884 | -8.3705 |
| 29 | 71 | Gemma-7B | GPQA | 8 | 16.0156 | 24.9442 | -8.9286 |
| 30 | 72 | Gemma-7B | GPQA | 8 | 16.183 | 25.1116 | -8.9286 |
| 31 | 73 | Gemma-7B | GPQA | 8 | 15.1228 | 24.1071 | -8.9844 |
| 32 | 74 | Gemma-7B | GPQA | 8 | 15.5134 | 26.1719 | -10.6585 |
| 33 | 75 | Gemma-7B | GPQA | 8 | 15.067 | 25.6696 | -10.6027 |
| 34 | 76 | Gemma-7B | GPQA | 8 | 15.8482 | 25.8371 | -9.9888 |
| 35 | 77 | Gemma-7B | GPQA | 8 | 15.1228 | 26.1161 | -10.9933 |
| 36 | 78 | Gemma-7B | GPQA | 8 | 15.6808 | 25.279 | -9.5982 |
| 37 | 79 | Gemma-7B | GPQA | 8 | 15.7366 | 24.5536 | -8.817 |
| 38 | 80 | Gemma-7B | GPQA | 8 | 15.7924 | 26.6741 | -10.8817 |
| 39 | 81 | Gemma-7B | GPQA | 8 | 15.346 | 24.7768 | -9.4308 |
| 40 | 82 | Gemma-7B | GPQA | 8 | 15.1228 | 26.2277 | -11.1049 |
| 41 | 83 | Gemma-7B | GPQA | 8 | 16.4621 | 26.3951 | -9.933 |
| 42 | 84 | Gemma-7B | GPQA | 8 | 15.9598 | 24.8884 | -8.9286 |
| 43 | 85 | Gemma-7B | GPQA | 8 | 15.7924 | 26.0045 | -10.2121 |
| 44 | 86 | Gemma-7B | GPQA | 8 | 16.2388 | 26.0603 | -9.8214 |
| 45 | 87 | Gemma-7B | GPQA | 8 | 14.1183 | 25.5022 | -11.3839 |
| 46 | 88 | Gemma-7B | GPQA | 8 | 16.2946 | 24.6652 | -8.3705 |
| 47 | 89 | Gemma-7B | GPQA | 8 | 14.0625 | 27.3996 | -13.3371 |
| 48 | 90 | Gemma-7B | GPQA | 8 | 16.1272 | 25.558 | -9.4308 |
| 49 | 91 | Gemma-7B | GPQA | 8 | 15.2344 | 27.1763 | -11.942 |
| 0 | 42 | Gemma-7B | GPQA | 12 | 16.183 | 24.5536 | -8.3705 |
| 1 | 43 | Gemma-7B | GPQA | 12 | 15.1042 | 26.6369 | -11.5327 |
| 2 | 44 | Gemma-7B | GPQA | 12 | 15.7738 | 25.6696 | -9.8958 |
| 3 | 45 | Gemma-7B | GPQA | 12 | 15.2902 | 25.2232 | -9.933 |
| 4 | 46 | Gemma-7B | GPQA | 12 | 15.5134 | 26.5997 | -11.0863 |
| 5 | 47 | Gemma-7B | GPQA | 12 | 15.9226 | 25.7068 | -9.7842 |
| 6 | 48 | Gemma-7B | GPQA | 12 | 15.1414 | 25.8929 | -10.7515 |
| 7 | 49 | Gemma-7B | GPQA | 12 | 15.067 | 26.3021 | -11.2351 |
| 8 | 50 | Gemma-7B | GPQA | 12 | 15.6622 | 26.2649 | -10.6027 |
| 9 | 51 | Gemma-7B | GPQA | 12 | 15.5506 | 26.3393 | -10.7887 |
| 10 | 52 | Gemma-7B | GPQA | 12 | 15.8854 | 25.5208 | -9.6354 |
| 11 | 53 | Gemma-7B | GPQA | 12 | 14.9926 | 26.3393 | -11.3467 |
| 12 | 54 | Gemma-7B | GPQA | 12 | 15.2902 | 25.9301 | -10.6399 |
| 13 | 55 | Gemma-7B | GPQA | 12 | 15.3646 | 26.0045 | -10.6399 |
| 14 | 56 | Gemma-7B | GPQA | 12 | 15.8482 | 25.4464 | -9.5982 |
| 15 | 57 | Gemma-7B | GPQA | 12 | 16.2946 | 25.2604 | -8.9658 |
| 16 | 58 | Gemma-7B | GPQA | 12 | 15.5878 | 25.8557 | -10.2679 |
| 17 | 59 | Gemma-7B | GPQA | 12 | 15.5134 | 26.6741 | -11.1607 |
| 18 | 60 | Gemma-7B | GPQA | 12 | 15.4762 | 25.8185 | -10.3423 |
| 19 | 61 | Gemma-7B | GPQA | 12 | 15.625 | 25.1488 | -9.5238 |
| 20 | 62 | Gemma-7B | GPQA | 12 | 15.811 | 25.0372 | -9.2262 |
| 21 | 63 | Gemma-7B | GPQA | 12 | 15.5134 | 25.1488 | -9.6354 |
| 22 | 64 | Gemma-7B | GPQA | 12 | 15.6994 | 26.0045 | -10.3051 |
| 23 | 65 | Gemma-7B | GPQA | 12 | 15.2902 | 25.5952 | -10.3051 |
| 24 | 66 | Gemma-7B | GPQA | 12 | 15.5134 | 25.744 | -10.2307 |
| 25 | 67 | Gemma-7B | GPQA | 12 | 15.5134 | 25.9301 | -10.4167 |
| 26 | 68 | Gemma-7B | GPQA | 12 | 15.7738 | 25.3348 | -9.561 |
| 27 | 69 | Gemma-7B | GPQA | 12 | 15.4762 | 26.2649 | -10.7887 |
| 28 | 70 | Gemma-7B | GPQA | 12 | 15.2158 | 26.2277 | -11.0119 |
| 29 | 71 | Gemma-7B | GPQA | 12 | 15.8482 | 25.6696 | -9.8214 |
| 30 | 72 | Gemma-7B | GPQA | 12 | 15.5506 | 24.9628 | -9.4122 |
| 31 | 73 | Gemma-7B | GPQA | 12 | 14.9926 | 25.6696 | -10.6771 |
| 32 | 74 | Gemma-7B | GPQA | 12 | 15.1042 | 26.5997 | -11.4955 |
| 33 | 75 | Gemma-7B | GPQA | 12 | 15.0298 | 26.5253 | -11.4955 |
| 34 | 76 | Gemma-7B | GPQA | 12 | 15.439 | 25.7068 | -10.2679 |
| 35 | 77 | Gemma-7B | GPQA | 12 | 15.4762 | 25.7812 | -10.3051 |
| 36 | 78 | Gemma-7B | GPQA | 12 | 15.811 | 25.5952 | -9.7842 |
| 37 | 79 | Gemma-7B | GPQA | 12 | 15.5134 | 25.8929 | -10.3795 |
| 38 | 80 | Gemma-7B | GPQA | 12 | 15.6994 | 25.7812 | -10.0818 |
| 39 | 81 | Gemma-7B | GPQA | 12 | 15.5134 | 25.7068 | -10.1935 |
| 40 | 82 | Gemma-7B | GPQA | 12 | 15.2902 | 26.1161 | -10.8259 |
| 41 | 83 | Gemma-7B | GPQA | 12 | 15.6622 | 26.3021 | -10.6399 |
| 42 | 84 | Gemma-7B | GPQA | 12 | 16.4435 | 24.9256 | -8.4821 |
| 43 | 85 | Gemma-7B | GPQA | 12 | 15.1786 | 26.4881 | -11.3095 |
| 44 | 86 | Gemma-7B | GPQA | 12 | 15.5506 | 26.2277 | -10.6771 |
| 45 | 87 | Gemma-7B | GPQA | 12 | 15.2158 | 25.2976 | -10.0818 |
| 46 | 88 | Gemma-7B | GPQA | 12 | 16.2202 | 25.2976 | -9.0774 |
| 47 | 89 | Gemma-7B | GPQA | 12 | 15.5506 | 25.744 | -10.1935 |
| 48 | 90 | Gemma-7B | GPQA | 12 | 15.7738 | 25.9301 | -10.1563 |
| 49 | 91 | Gemma-7B | GPQA | 12 | 15.2902 | 26.3021 | -11.0119 |
| 0 | 42 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 1 | 43 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 2 | 44 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 3 | 45 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 4 | 46 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 5 | 47 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 6 | 48 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 7 | 49 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 8 | 50 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 9 | 51 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 10 | 52 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 11 | 53 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 12 | 54 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 13 | 55 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 14 | 56 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 15 | 57 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 16 | 58 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 17 | 59 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 18 | 60 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 19 | 61 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 20 | 62 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 21 | 63 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 22 | 64 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 23 | 65 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 24 | 66 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 25 | 67 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 26 | 68 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 27 | 69 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 28 | 70 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 29 | 71 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 30 | 72 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 31 | 73 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 32 | 74 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 33 | 75 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 34 | 76 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 35 | 77 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 36 | 78 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 37 | 79 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 38 | 80 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 39 | 81 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 40 | 82 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 41 | 83 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 42 | 84 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 43 | 85 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 44 | 86 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 45 | 87 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 46 | 88 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 47 | 89 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 48 | 90 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 49 | 91 | Gemma-7B | GPQA | 16 | 15.4297 | 25.9208 | -10.4911 |
| 0 | 42 | Gemma-7B | GSM8K | 4 | 35.7468 | 28.0516 | 7.6952 |
| 1 | 43 | Gemma-7B | GSM8K | 4 | 36.3533 | 27.8241 | 8.5292 |
| 2 | 44 | Gemma-7B | GSM8K | 4 | 35.9742 | 28.9992 | 6.975 |
| 3 | 45 | Gemma-7B | GSM8K | 4 | 36.5049 | 27.5588 | 8.9462 |
| 4 | 46 | Gemma-7B | GSM8K | 4 | 36.6187 | 29.5299 | 7.0887 |
| 5 | 47 | Gemma-7B | GSM8K | 4 | 35.254 | 28.9234 | 6.3306 |
| 6 | 48 | Gemma-7B | GSM8K | 4 | 36.8082 | 29.1888 | 7.6194 |
| 7 | 49 | Gemma-7B | GSM8K | 4 | 34.9128 | 29.3783 | 5.5345 |
| 8 | 50 | Gemma-7B | GSM8K | 4 | 35.5572 | 29.6437 | 5.9136 |
| 9 | 51 | Gemma-7B | GSM8K | 4 | 35.2161 | 29.2267 | 5.9894 |
| 10 | 52 | Gemma-7B | GSM8K | 4 | 36.4291 | 28.8855 | 7.5436 |
| 11 | 53 | Gemma-7B | GSM8K | 4 | 36.6566 | 29.9469 | 6.7096 |
| 12 | 54 | Gemma-7B | GSM8K | 4 | 37.7938 | 28.8476 | 8.9462 |
| 13 | 55 | Gemma-7B | GSM8K | 4 | 35.8984 | 28.3548 | 7.5436 |
| 14 | 56 | Gemma-7B | GSM8K | 4 | 35.4056 | 27.1797 | 8.2259 |
| 15 | 57 | Gemma-7B | GSM8K | 4 | 35.7089 | 30.5914 | 5.1175 |
| 16 | 58 | Gemma-7B | GSM8K | 4 | 36.9977 | 29.6437 | 7.3541 |
| 17 | 59 | Gemma-7B | GSM8K | 4 | 36.6566 | 29.6058 | 7.0508 |
| 18 | 60 | Gemma-7B | GSM8K | 4 | 37.2252 | 28.279 | 8.9462 |
| 19 | 61 | Gemma-7B | GSM8K | 4 | 36.9598 | 28.8855 | 8.0743 |
| 20 | 62 | Gemma-7B | GSM8K | 4 | 37.1114 | 29.0371 | 8.0743 |
| 21 | 63 | Gemma-7B | GSM8K | 4 | 34.6854 | 29.4541 | 5.2312 |
| 22 | 64 | Gemma-7B | GSM8K | 4 | 36.467 | 27.6346 | 8.8324 |
| 23 | 65 | Gemma-7B | GSM8K | 4 | 35.2161 | 29.6058 | 5.6103 |
| 24 | 66 | Gemma-7B | GSM8K | 4 | 35.0644 | 28.3169 | 6.7475 |
| 25 | 67 | Gemma-7B | GSM8K | 4 | 35.254 | 28.0895 | 7.1645 |
| 26 | 68 | Gemma-7B | GSM8K | 4 | 37.0356 | 28.9992 | 8.0364 |
| 27 | 69 | Gemma-7B | GSM8K | 4 | 36.0879 | 28.2411 | 7.8469 |
| 28 | 70 | Gemma-7B | GSM8K | 4 | 35.4056 | 29.2267 | 6.1789 |
| 29 | 71 | Gemma-7B | GSM8K | 4 | 36.9219 | 28.6202 | 8.3017 |
| 30 | 72 | Gemma-7B | GSM8K | 4 | 36.7703 | 29.8711 | 6.8992 |
| 31 | 73 | Gemma-7B | GSM8K | 4 | 36.05 | 28.0136 | 8.0364 |
| 32 | 74 | Gemma-7B | GSM8K | 4 | 36.3912 | 28.8855 | 7.5057 |
| 33 | 75 | Gemma-7B | GSM8K | 4 | 36.9977 | 29.1888 | 7.8089 |
| 34 | 76 | Gemma-7B | GSM8K | 4 | 36.4291 | 28.1274 | 8.3017 |
| 35 | 77 | Gemma-7B | GSM8K | 4 | 34.6096 | 29.0751 | 5.5345 |
| 36 | 78 | Gemma-7B | GSM8K | 4 | 36.7703 | 28.3548 | 8.4155 |
| 37 | 79 | Gemma-7B | GSM8K | 4 | 35.7847 | 28.9234 | 6.8613 |
| 38 | 80 | Gemma-7B | GSM8K | 4 | 36.6945 | 29.6058 | 7.0887 |
| 39 | 81 | Gemma-7B | GSM8K | 4 | 37.4147 | 29.5299 | 7.8848 |
| 40 | 82 | Gemma-7B | GSM8K | 4 | 37.0735 | 29.4541 | 7.6194 |
| 41 | 83 | Gemma-7B | GSM8K | 4 | 35.5572 | 29.1888 | 6.3685 |
| 42 | 84 | Gemma-7B | GSM8K | 4 | 35.5193 | 28.9992 | 6.5201 |
| 43 | 85 | Gemma-7B | GSM8K | 4 | 35.7847 | 28.9234 | 6.8613 |
| 44 | 86 | Gemma-7B | GSM8K | 4 | 35.9363 | 27.862 | 8.0743 |
| 45 | 87 | Gemma-7B | GSM8K | 4 | 35.6331 | 29.0371 | 6.5959 |
| 46 | 88 | Gemma-7B | GSM8K | 4 | 36.5428 | 27.6725 | 8.8704 |
| 47 | 89 | Gemma-7B | GSM8K | 4 | 34.6475 | 29.909 | 4.7384 |
| 48 | 90 | Gemma-7B | GSM8K | 4 | 36.2396 | 28.6581 | 7.5815 |
| 49 | 91 | Gemma-7B | GSM8K | 4 | 37.3389 | 28.0895 | 9.2494 |
| 0 | 42 | Gemma-7B | GSM8K | 8 | 36.1259 | 27.8052 | 8.3207 |
| 1 | 43 | Gemma-7B | GSM8K | 8 | 37.4337 | 27.5019 | 9.9318 |
| 2 | 44 | Gemma-7B | GSM8K | 8 | 36.5807 | 29.1698 | 7.4109 |
| 3 | 45 | Gemma-7B | GSM8K | 8 | 36.4481 | 28.4306 | 8.0174 |
| 4 | 46 | Gemma-7B | GSM8K | 8 | 36.9219 | 28.5444 | 8.3776 |
| 5 | 47 | Gemma-7B | GSM8K | 8 | 36.3533 | 28.0136 | 8.3397 |
| 6 | 48 | Gemma-7B | GSM8K | 8 | 36.7134 | 29.0182 | 7.6952 |
| 7 | 49 | Gemma-7B | GSM8K | 8 | 36.9409 | 27.7862 | 9.1547 |
| 8 | 50 | Gemma-7B | GSM8K | 8 | 36.3343 | 29.3025 | 7.0318 |
| 9 | 51 | Gemma-7B | GSM8K | 8 | 36.8082 | 28.4496 | 8.3586 |
| 10 | 52 | Gemma-7B | GSM8K | 8 | 36.9788 | 27.7483 | 9.2305 |
| 11 | 53 | Gemma-7B | GSM8K | 8 | 37.2252 | 28.4875 | 8.7377 |
| 12 | 54 | Gemma-7B | GSM8K | 8 | 37.3199 | 28.1463 | 9.1736 |
| 13 | 55 | Gemma-7B | GSM8K | 8 | 37.0356 | 28.298 | 8.7377 |
| 14 | 56 | Gemma-7B | GSM8K | 8 | 36.3912 | 27.445 | 8.9462 |
| 15 | 57 | Gemma-7B | GSM8K | 8 | 36.5049 | 29.0371 | 7.4678 |
| 16 | 58 | Gemma-7B | GSM8K | 8 | 37.0356 | 29.1698 | 7.8658 |
| 17 | 59 | Gemma-7B | GSM8K | 8 | 36.4481 | 28.1084 | 8.3397 |
| 18 | 60 | Gemma-7B | GSM8K | 8 | 37.1304 | 27.9947 | 9.1357 |
| 19 | 61 | Gemma-7B | GSM8K | 8 | 37.6611 | 28.1653 | 9.4958 |
| 20 | 62 | Gemma-7B | GSM8K | 8 | 37.0735 | 28.0326 | 9.0409 |
| 21 | 63 | Gemma-7B | GSM8K | 8 | 35.8415 | 28.6012 | 7.2403 |
| 22 | 64 | Gemma-7B | GSM8K | 8 | 36.2017 | 28.5633 | 7.6384 |
| 23 | 65 | Gemma-7B | GSM8K | 8 | 36.6945 | 28.6581 | 8.0364 |
| 24 | 66 | Gemma-7B | GSM8K | 8 | 36.6187 | 27.5967 | 9.022 |
| 25 | 67 | Gemma-7B | GSM8K | 8 | 35.652 | 28.5444 | 7.1077 |
| 26 | 68 | Gemma-7B | GSM8K | 8 | 37.1494 | 27.9568 | 9.1926 |
| 27 | 69 | Gemma-7B | GSM8K | 8 | 36.4481 | 28.7528 | 7.6952 |
| 28 | 70 | Gemma-7B | GSM8K | 8 | 36.6376 | 28.5064 | 8.1312 |
| 29 | 71 | Gemma-7B | GSM8K | 8 | 37.5663 | 28.0136 | 9.5527 |
| 30 | 72 | Gemma-7B | GSM8K | 8 | 36.9788 | 28.9045 | 8.0743 |
| 31 | 73 | Gemma-7B | GSM8K | 8 | 36.05 | 27.5398 | 8.5102 |
| 32 | 74 | Gemma-7B | GSM8K | 8 | 36.8082 | 28.3738 | 8.4344 |
| 33 | 75 | Gemma-7B | GSM8K | 8 | 36.6566 | 28.5254 | 8.1312 |
| 34 | 76 | Gemma-7B | GSM8K | 8 | 36.5997 | 28.6581 | 7.9416 |
| 35 | 77 | Gemma-7B | GSM8K | 8 | 35.7657 | 27.9757 | 7.79 |
| 36 | 78 | Gemma-7B | GSM8K | 8 | 36.6376 | 28.4496 | 8.188 |
| 37 | 79 | Gemma-7B | GSM8K | 8 | 36.7324 | 28.3738 | 8.3586 |
| 38 | 80 | Gemma-7B | GSM8K | 8 | 37.8696 | 29.0751 | 8.7945 |
| 39 | 81 | Gemma-7B | GSM8K | 8 | 37.4526 | 28.4306 | 9.022 |
| 40 | 82 | Gemma-7B | GSM8K | 8 | 37.1304 | 28.5633 | 8.5671 |
| 41 | 83 | Gemma-7B | GSM8K | 8 | 36.7703 | 28.3927 | 8.3776 |
| 42 | 84 | Gemma-7B | GSM8K | 8 | 36.5049 | 28.7718 | 7.7331 |
| 43 | 85 | Gemma-7B | GSM8K | 8 | 36.7892 | 28.7528 | 8.0364 |
| 44 | 86 | Gemma-7B | GSM8K | 8 | 36.467 | 28.1653 | 8.3017 |
| 45 | 87 | Gemma-7B | GSM8K | 8 | 36.6755 | 27.8999 | 8.7756 |
| 46 | 88 | Gemma-7B | GSM8K | 8 | 37.282 | 27.4829 | 9.7991 |
| 47 | 89 | Gemma-7B | GSM8K | 8 | 35.8226 | 29.1509 | 6.6717 |
| 48 | 90 | Gemma-7B | GSM8K | 8 | 36.6945 | 28.4496 | 8.2449 |
| 49 | 91 | Gemma-7B | GSM8K | 8 | 37.0735 | 28.3927 | 8.6808 |
| 0 | 42 | Gemma-7B | GSM8K | 12 | 36.8461 | 27.7862 | 9.0599 |
| 1 | 43 | Gemma-7B | GSM8K | 12 | 37.2883 | 27.9252 | 9.3632 |
| 2 | 44 | Gemma-7B | GSM8K | 12 | 37.1999 | 28.1779 | 9.022 |
| 3 | 45 | Gemma-7B | GSM8K | 12 | 36.9219 | 28.2916 | 8.6303 |
| 4 | 46 | Gemma-7B | GSM8K | 12 | 37.0862 | 28.1653 | 8.9209 |
| 5 | 47 | Gemma-7B | GSM8K | 12 | 36.6692 | 28.3295 | 8.3397 |
| 6 | 48 | Gemma-7B | GSM8K | 12 | 36.8335 | 28.3422 | 8.4913 |
| 7 | 49 | Gemma-7B | GSM8K | 12 | 37.0483 | 27.9884 | 9.0599 |
| 8 | 50 | Gemma-7B | GSM8K | 12 | 36.7071 | 28.7339 | 7.9732 |
| 9 | 51 | Gemma-7B | GSM8K | 12 | 37.1367 | 28.2032 | 8.9335 |
| 10 | 52 | Gemma-7B | GSM8K | 12 | 36.8966 | 28.1274 | 8.7693 |
| 11 | 53 | Gemma-7B | GSM8K | 12 | 36.8966 | 28.2916 | 8.605 |
| 12 | 54 | Gemma-7B | GSM8K | 12 | 36.9851 | 28.2411 | 8.744 |
| 13 | 55 | Gemma-7B | GSM8K | 12 | 36.9725 | 27.723 | 9.2494 |
| 14 | 56 | Gemma-7B | GSM8K | 12 | 36.9851 | 27.9884 | 8.9967 |
| 15 | 57 | Gemma-7B | GSM8K | 12 | 36.7829 | 28.0389 | 8.744 |
| 16 | 58 | Gemma-7B | GSM8K | 12 | 37.0988 | 28.5696 | 8.5292 |
| 17 | 59 | Gemma-7B | GSM8K | 12 | 36.9093 | 28.5317 | 8.3776 |
| 18 | 60 | Gemma-7B | GSM8K | 12 | 37.301 | 28.2664 | 9.0346 |
| 19 | 61 | Gemma-7B | GSM8K | 12 | 37.0483 | 28.1274 | 8.9209 |
| 20 | 62 | Gemma-7B | GSM8K | 12 | 36.8335 | 28.3169 | 8.5166 |
| 21 | 63 | Gemma-7B | GSM8K | 12 | 36.7197 | 28.5949 | 8.1248 |
| 22 | 64 | Gemma-7B | GSM8K | 12 | 36.8714 | 28.4433 | 8.4281 |
| 23 | 65 | Gemma-7B | GSM8K | 12 | 36.3154 | 28.7086 | 7.6068 |
| 24 | 66 | Gemma-7B | GSM8K | 12 | 36.6818 | 27.9378 | 8.744 |
| 25 | 67 | Gemma-7B | GSM8K | 12 | 36.8082 | 28.279 | 8.5292 |
| 26 | 68 | Gemma-7B | GSM8K | 12 | 37.2631 | 28.5191 | 8.744 |
| 27 | 69 | Gemma-7B | GSM8K | 12 | 36.8461 | 28.6075 | 8.2386 |
| 28 | 70 | Gemma-7B | GSM8K | 12 | 36.5681 | 28.7086 | 7.8595 |
| 29 | 71 | Gemma-7B | GSM8K | 12 | 37.1494 | 28.2285 | 8.9209 |
| 30 | 72 | Gemma-7B | GSM8K | 12 | 37.0862 | 28.4433 | 8.6429 |
| 31 | 73 | Gemma-7B | GSM8K | 12 | 36.9472 | 27.9252 | 9.022 |
| 32 | 74 | Gemma-7B | GSM8K | 12 | 36.7956 | 28.6075 | 8.188 |
| 33 | 75 | Gemma-7B | GSM8K | 12 | 36.9725 | 28.7213 | 8.2512 |
| 34 | 76 | Gemma-7B | GSM8K | 12 | 37.3894 | 28.1147 | 9.2747 |
| 35 | 77 | Gemma-7B | GSM8K | 12 | 36.4797 | 27.7357 | 8.744 |
| 36 | 78 | Gemma-7B | GSM8K | 12 | 37.0104 | 28.5444 | 8.466 |
| 37 | 79 | Gemma-7B | GSM8K | 12 | 36.3659 | 28.5949 | 7.771 |
| 38 | 80 | Gemma-7B | GSM8K | 12 | 37.5916 | 28.6707 | 8.9209 |
| 39 | 81 | Gemma-7B | GSM8K | 12 | 36.9472 | 28.2032 | 8.744 |
| 40 | 82 | Gemma-7B | GSM8K | 12 | 37.5916 | 27.9757 | 9.6159 |
| 41 | 83 | Gemma-7B | GSM8K | 12 | 37.1999 | 28.2537 | 8.9462 |
| 42 | 84 | Gemma-7B | GSM8K | 12 | 36.9977 | 28.1021 | 8.8956 |
| 43 | 85 | Gemma-7B | GSM8K | 12 | 36.9093 | 28.0516 | 8.8577 |
| 44 | 86 | Gemma-7B | GSM8K | 12 | 36.606 | 28.6202 | 7.9858 |
| 45 | 87 | Gemma-7B | GSM8K | 12 | 37.2252 | 27.9126 | 9.3126 |
| 46 | 88 | Gemma-7B | GSM8K | 12 | 37.0483 | 28.4685 | 8.5797 |
| 47 | 89 | Gemma-7B | GSM8K | 12 | 36.884 | 28.001 | 8.883 |
| 48 | 90 | Gemma-7B | GSM8K | 12 | 36.9219 | 28.2916 | 8.6303 |
| 49 | 91 | Gemma-7B | GSM8K | 12 | 37.301 | 28.0895 | 9.2115 |
| 0 | 42 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 1 | 43 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 2 | 44 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 3 | 45 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 4 | 46 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 5 | 47 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 6 | 48 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 7 | 49 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 8 | 50 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 9 | 51 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 10 | 52 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 11 | 53 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 12 | 54 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 13 | 55 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 14 | 56 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 15 | 57 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 16 | 58 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 17 | 59 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 18 | 60 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 19 | 61 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 20 | 62 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 21 | 63 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 22 | 64 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 23 | 65 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 24 | 66 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 25 | 67 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 26 | 68 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 27 | 69 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 28 | 70 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 29 | 71 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 30 | 72 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 31 | 73 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 32 | 74 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 33 | 75 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 34 | 76 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 35 | 77 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 36 | 78 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 37 | 79 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 38 | 80 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 39 | 81 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 40 | 82 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 41 | 83 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 42 | 84 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 43 | 85 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 44 | 86 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 45 | 87 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 46 | 88 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 47 | 89 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 48 | 90 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 49 | 91 | Gemma-7B | GSM8K | 16 | 37.083 | 28.2032 | 8.8798 |
| 0 | 42 | Gemma-7B | MATH500 | 4 | 13.6 | 12.4 | 1.2 |
| 1 | 43 | Gemma-7B | MATH500 | 4 | 13.1 | 12 | 1.1 |
| 2 | 44 | Gemma-7B | MATH500 | 4 | 13.1 | 12.6 | 0.5 |
| 3 | 45 | Gemma-7B | MATH500 | 4 | 13.5 | 11 | 2.5 |
| 4 | 46 | Gemma-7B | MATH500 | 4 | 15.1 | 11.8 | 3.3 |
| 5 | 47 | Gemma-7B | MATH500 | 4 | 14.6 | 12.5 | 2.1 |
| 6 | 48 | Gemma-7B | MATH500 | 4 | 14.4 | 12.4 | 2 |
| 7 | 49 | Gemma-7B | MATH500 | 4 | 14.7 | 12.6 | 2.1 |
| 8 | 50 | Gemma-7B | MATH500 | 4 | 15.7 | 11.6 | 4.1 |
| 9 | 51 | Gemma-7B | MATH500 | 4 | 12.9 | 11.8 | 1.1 |
| 10 | 52 | Gemma-7B | MATH500 | 4 | 15.4 | 11.4 | 4 |
| 11 | 53 | Gemma-7B | MATH500 | 4 | 13.9 | 12.3 | 1.6 |
| 12 | 54 | Gemma-7B | MATH500 | 4 | 14 | 14 | 0 |
| 13 | 55 | Gemma-7B | MATH500 | 4 | 14.4 | 12.5 | 1.9 |
| 14 | 56 | Gemma-7B | MATH500 | 4 | 14.5 | 13 | 1.5 |
| 15 | 57 | Gemma-7B | MATH500 | 4 | 15.4 | 11 | 4.4 |
| 16 | 58 | Gemma-7B | MATH500 | 4 | 15.7 | 12.2 | 3.5 |
| 17 | 59 | Gemma-7B | MATH500 | 4 | 15.6 | 12 | 3.6 |
| 18 | 60 | Gemma-7B | MATH500 | 4 | 13.9 | 11.4 | 2.5 |
| 19 | 61 | Gemma-7B | MATH500 | 4 | 13 | 11.6 | 1.4 |
| 20 | 62 | Gemma-7B | MATH500 | 4 | 13.9 | 12.7 | 1.2 |
| 21 | 63 | Gemma-7B | MATH500 | 4 | 13.4 | 12.3 | 1.1 |
| 22 | 64 | Gemma-7B | MATH500 | 4 | 13.5 | 12 | 1.5 |
| 23 | 65 | Gemma-7B | MATH500 | 4 | 13.6 | 13.5 | 0.1 |
| 24 | 66 | Gemma-7B | MATH500 | 4 | 14.3 | 12.9 | 1.4 |
| 25 | 67 | Gemma-7B | MATH500 | 4 | 14.5 | 11.9 | 2.6 |
| 26 | 68 | Gemma-7B | MATH500 | 4 | 14.1 | 13.1 | 1 |
| 27 | 69 | Gemma-7B | MATH500 | 4 | 14.1 | 12.9 | 1.2 |
| 28 | 70 | Gemma-7B | MATH500 | 4 | 13.7 | 12.8 | 0.9 |
| 29 | 71 | Gemma-7B | MATH500 | 4 | 14.1 | 12.6 | 1.5 |
| 30 | 72 | Gemma-7B | MATH500 | 4 | 15.8 | 14 | 1.8 |
| 31 | 73 | Gemma-7B | MATH500 | 4 | 15.3 | 10.8 | 4.5 |
| 32 | 74 | Gemma-7B | MATH500 | 4 | 14.3 | 11.6 | 2.7 |
| 33 | 75 | Gemma-7B | MATH500 | 4 | 14.6 | 11.4 | 3.2 |
| 34 | 76 | Gemma-7B | MATH500 | 4 | 14.4 | 10.1 | 4.3 |
| 35 | 77 | Gemma-7B | MATH500 | 4 | 13.7 | 12.2 | 1.5 |
| 36 | 78 | Gemma-7B | MATH500 | 4 | 13.8 | 13.3 | 0.5 |
| 37 | 79 | Gemma-7B | MATH500 | 4 | 14.7 | 12.9 | 1.8 |
| 38 | 80 | Gemma-7B | MATH500 | 4 | 14.5 | 12.5 | 2 |
| 39 | 81 | Gemma-7B | MATH500 | 4 | 13.4 | 12.8 | 0.6 |
| 40 | 82 | Gemma-7B | MATH500 | 4 | 13.9 | 12.3 | 1.6 |
| 41 | 83 | Gemma-7B | MATH500 | 4 | 12.9 | 12.7 | 0.2 |
| 42 | 84 | Gemma-7B | MATH500 | 4 | 14.3 | 12 | 2.3 |
| 43 | 85 | Gemma-7B | MATH500 | 4 | 13.9 | 13.3 | 0.6 |
| 44 | 86 | Gemma-7B | MATH500 | 4 | 15 | 11.9 | 3.1 |
| 45 | 87 | Gemma-7B | MATH500 | 4 | 13.6 | 11.2 | 2.4 |
| 46 | 88 | Gemma-7B | MATH500 | 4 | 13.8 | 12.3 | 1.5 |
| 47 | 89 | Gemma-7B | MATH500 | 4 | 14.7 | 11.4 | 3.3 |
| 48 | 90 | Gemma-7B | MATH500 | 4 | 15.4 | 13.6 | 1.8 |
| 49 | 91 | Gemma-7B | MATH500 | 4 | 14 | 11.8 | 2.2 |
| 0 | 42 | Gemma-7B | MATH500 | 8 | 14.45 | 12.2 | 2.25 |
| 1 | 43 | Gemma-7B | MATH500 | 8 | 13.95 | 11.7 | 2.25 |
| 2 | 44 | Gemma-7B | MATH500 | 8 | 14.15 | 11.85 | 2.3 |
| 3 | 45 | Gemma-7B | MATH500 | 8 | 13.45 | 12.1 | 1.35 |
| 4 | 46 | Gemma-7B | MATH500 | 8 | 14.4 | 11.8 | 2.6 |
| 5 | 47 | Gemma-7B | MATH500 | 8 | 14.55 | 12.5 | 2.05 |
| 6 | 48 | Gemma-7B | MATH500 | 8 | 13.9 | 12.3 | 1.6 |
| 7 | 49 | Gemma-7B | MATH500 | 8 | 14.5 | 11.95 | 2.55 |
| 8 | 50 | Gemma-7B | MATH500 | 8 | 14.5 | 12.2 | 2.3 |
| 9 | 51 | Gemma-7B | MATH500 | 8 | 14.85 | 11.8 | 3.05 |
| 10 | 52 | Gemma-7B | MATH500 | 8 | 14.7 | 11.85 | 2.85 |
| 11 | 53 | Gemma-7B | MATH500 | 8 | 14.35 | 13.1 | 1.25 |
| 12 | 54 | Gemma-7B | MATH500 | 8 | 14.25 | 12.45 | 1.8 |
| 13 | 55 | Gemma-7B | MATH500 | 8 | 14.85 | 12.55 | 2.3 |
| 14 | 56 | Gemma-7B | MATH500 | 8 | 14.5 | 12.9 | 1.6 |
| 15 | 57 | Gemma-7B | MATH500 | 8 | 14.8 | 12.1 | 2.7 |
| 16 | 58 | Gemma-7B | MATH500 | 8 | 14.2 | 12.15 | 2.05 |
| 17 | 59 | Gemma-7B | MATH500 | 8 | 14.1 | 12.35 | 1.75 |
| 18 | 60 | Gemma-7B | MATH500 | 8 | 14.05 | 12.05 | 2 |
| 19 | 61 | Gemma-7B | MATH500 | 8 | 14.35 | 12 | 2.35 |
| 20 | 62 | Gemma-7B | MATH500 | 8 | 13.95 | 12.9 | 1.05 |
| 21 | 63 | Gemma-7B | MATH500 | 8 | 14.45 | 11.7 | 2.75 |
| 22 | 64 | Gemma-7B | MATH500 | 8 | 14.25 | 12.3 | 1.95 |
| 23 | 65 | Gemma-7B | MATH500 | 8 | 14.9 | 12.75 | 2.15 |
| 24 | 66 | Gemma-7B | MATH500 | 8 | 14.35 | 12.6 | 1.75 |
| 25 | 67 | Gemma-7B | MATH500 | 8 | 14.55 | 12.55 | 2 |
| 26 | 68 | Gemma-7B | MATH500 | 8 | 14.45 | 12.2 | 2.25 |
| 27 | 69 | Gemma-7B | MATH500 | 8 | 14.5 | 12.6 | 1.9 |
| 28 | 70 | Gemma-7B | MATH500 | 8 | 14.6 | 11.9 | 2.7 |
| 29 | 71 | Gemma-7B | MATH500 | 8 | 14.15 | 11.6 | 2.55 |
| 30 | 72 | Gemma-7B | MATH500 | 8 | 14.75 | 13.1 | 1.65 |
| 31 | 73 | Gemma-7B | MATH500 | 8 | 14.65 | 11.5 | 3.15 |
| 32 | 74 | Gemma-7B | MATH500 | 8 | 14.15 | 11.65 | 2.5 |
| 33 | 75 | Gemma-7B | MATH500 | 8 | 13.55 | 12.7 | 0.85 |
| 34 | 76 | Gemma-7B | MATH500 | 8 | 14.4 | 11.6 | 2.8 |
| 35 | 77 | Gemma-7B | MATH500 | 8 | 14.7 | 11.5 | 3.2 |
| 36 | 78 | Gemma-7B | MATH500 | 8 | 14.55 | 12.7 | 1.85 |
| 37 | 79 | Gemma-7B | MATH500 | 8 | 15.05 | 12.35 | 2.7 |
| 38 | 80 | Gemma-7B | MATH500 | 8 | 14.25 | 12.5 | 1.75 |
| 39 | 81 | Gemma-7B | MATH500 | 8 | 14.6 | 12.75 | 1.85 |
| 40 | 82 | Gemma-7B | MATH500 | 8 | 14.1 | 12.5 | 1.6 |
| 41 | 83 | Gemma-7B | MATH500 | 8 | 15.4 | 12.05 | 3.35 |
| 42 | 84 | Gemma-7B | MATH500 | 8 | 14.75 | 12.15 | 2.6 |
| 43 | 85 | Gemma-7B | MATH500 | 8 | 15.2 | 11.9 | 3.3 |
| 44 | 86 | Gemma-7B | MATH500 | 8 | 14.8 | 12 | 2.8 |
| 45 | 87 | Gemma-7B | MATH500 | 8 | 14.05 | 12.4 | 1.65 |
| 46 | 88 | Gemma-7B | MATH500 | 8 | 14.8 | 11.95 | 2.85 |
| 47 | 89 | Gemma-7B | MATH500 | 8 | 14.55 | 11.9 | 2.65 |
| 48 | 90 | Gemma-7B | MATH500 | 8 | 14.85 | 12.5 | 2.35 |
| 49 | 91 | Gemma-7B | MATH500 | 8 | 14.25 | 12.45 | 1.8 |
| 0 | 42 | Gemma-7B | MATH500 | 12 | 14.6333 | 12.1 | 2.5333 |
| 1 | 43 | Gemma-7B | MATH500 | 12 | 14.1 | 12 | 2.1 |
| 2 | 44 | Gemma-7B | MATH500 | 12 | 14.4333 | 11.9333 | 2.5 |
| 3 | 45 | Gemma-7B | MATH500 | 12 | 14.4 | 12 | 2.4 |
| 4 | 46 | Gemma-7B | MATH500 | 12 | 14.1667 | 12.4667 | 1.7 |
| 5 | 47 | Gemma-7B | MATH500 | 12 | 14.5333 | 12.4667 | 2.0667 |
| 6 | 48 | Gemma-7B | MATH500 | 12 | 14.4333 | 11.5333 | 2.9 |
| 7 | 49 | Gemma-7B | MATH500 | 12 | 14.2667 | 12.2667 | 2 |
| 8 | 50 | Gemma-7B | MATH500 | 12 | 14.4667 | 12.4 | 2.0667 |
| 9 | 51 | Gemma-7B | MATH500 | 12 | 14.7333 | 11.6333 | 3.1 |
| 10 | 52 | Gemma-7B | MATH500 | 12 | 14.4333 | 11.9333 | 2.5 |
| 11 | 53 | Gemma-7B | MATH500 | 12 | 14.6 | 12.4 | 2.2 |
| 12 | 54 | Gemma-7B | MATH500 | 12 | 14.2333 | 12.2333 | 2 |
| 13 | 55 | Gemma-7B | MATH500 | 12 | 15.0667 | 12.0333 | 3.0333 |
| 14 | 56 | Gemma-7B | MATH500 | 12 | 14.9333 | 12.0333 | 2.9 |
| 15 | 57 | Gemma-7B | MATH500 | 12 | 14.7667 | 12 | 2.7667 |
| 16 | 58 | Gemma-7B | MATH500 | 12 | 14.1333 | 12.6333 | 1.5 |
| 17 | 59 | Gemma-7B | MATH500 | 12 | 14.3667 | 12.4 | 1.9667 |
| 18 | 60 | Gemma-7B | MATH500 | 12 | 13.8667 | 12.3 | 1.5667 |
| 19 | 61 | Gemma-7B | MATH500 | 12 | 14.4 | 12.3333 | 2.0667 |
| 20 | 62 | Gemma-7B | MATH500 | 12 | 14.2667 | 12.3333 | 1.9333 |
| 21 | 63 | Gemma-7B | MATH500 | 12 | 14.1667 | 11.7667 | 2.4 |
| 22 | 64 | Gemma-7B | MATH500 | 12 | 14.4667 | 11.9667 | 2.5 |
| 23 | 65 | Gemma-7B | MATH500 | 12 | 14.5333 | 12.4 | 2.1333 |
| 24 | 66 | Gemma-7B | MATH500 | 12 | 15.1333 | 12.5 | 2.6333 |
| 25 | 67 | Gemma-7B | MATH500 | 12 | 14.2 | 12.6667 | 1.5333 |
| 26 | 68 | Gemma-7B | MATH500 | 12 | 15.2333 | 11.8 | 3.4333 |
| 27 | 69 | Gemma-7B | MATH500 | 12 | 14.3667 | 12.1 | 2.2667 |
| 28 | 70 | Gemma-7B | MATH500 | 12 | 14.3667 | 11.7667 | 2.6 |
| 29 | 71 | Gemma-7B | MATH500 | 12 | 14.1667 | 12.3 | 1.8667 |
| 30 | 72 | Gemma-7B | MATH500 | 12 | 14.4667 | 12.6 | 1.8667 |
| 31 | 73 | Gemma-7B | MATH500 | 12 | 14.4 | 12.2 | 2.2 |
| 32 | 74 | Gemma-7B | MATH500 | 12 | 14.8 | 12.4 | 2.4 |
| 33 | 75 | Gemma-7B | MATH500 | 12 | 14.3333 | 12.4667 | 1.8667 |
| 34 | 76 | Gemma-7B | MATH500 | 12 | 14.2667 | 11.8 | 2.4667 |
| 35 | 77 | Gemma-7B | MATH500 | 12 | 14.2667 | 11.4 | 2.8667 |
| 36 | 78 | Gemma-7B | MATH500 | 12 | 14.8667 | 12.1667 | 2.7 |
| 37 | 79 | Gemma-7B | MATH500 | 12 | 14.4667 | 12.4667 | 2 |
| 38 | 80 | Gemma-7B | MATH500 | 12 | 14.4667 | 12.2667 | 2.2 |
| 39 | 81 | Gemma-7B | MATH500 | 12 | 14.3333 | 13.1333 | 1.2 |
| 40 | 82 | Gemma-7B | MATH500 | 12 | 14.4 | 12.1667 | 2.2333 |
| 41 | 83 | Gemma-7B | MATH500 | 12 | 14.5667 | 12.2 | 2.3667 |
| 42 | 84 | Gemma-7B | MATH500 | 12 | 14.5667 | 12.0333 | 2.5333 |
| 43 | 85 | Gemma-7B | MATH500 | 12 | 14.5667 | 12.5333 | 2.0333 |
| 44 | 86 | Gemma-7B | MATH500 | 12 | 14.7 | 12.1 | 2.6 |
| 45 | 87 | Gemma-7B | MATH500 | 12 | 14.7333 | 12.1 | 2.6333 |
| 46 | 88 | Gemma-7B | MATH500 | 12 | 14.3 | 12.4667 | 1.8333 |
| 47 | 89 | Gemma-7B | MATH500 | 12 | 14.1667 | 11.7 | 2.4667 |
| 48 | 90 | Gemma-7B | MATH500 | 12 | 15.0667 | 11.8 | 3.2667 |
| 49 | 91 | Gemma-7B | MATH500 | 12 | 14.3333 | 12.2333 | 2.1 |
| 0 | 42 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 1 | 43 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 2 | 44 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 3 | 45 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 4 | 46 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 5 | 47 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 6 | 48 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 7 | 49 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 8 | 50 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 9 | 51 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 10 | 52 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 11 | 53 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 12 | 54 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 13 | 55 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 14 | 56 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 15 | 57 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 16 | 58 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 17 | 59 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 18 | 60 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 19 | 61 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 20 | 62 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 21 | 63 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 22 | 64 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 23 | 65 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 24 | 66 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 25 | 67 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 26 | 68 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 27 | 69 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 28 | 70 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 29 | 71 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 30 | 72 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 31 | 73 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 32 | 74 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 33 | 75 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 34 | 76 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 35 | 77 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 36 | 78 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 37 | 79 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 38 | 80 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 39 | 81 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 40 | 82 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 41 | 83 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 42 | 84 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 43 | 85 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 44 | 86 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 45 | 87 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 46 | 88 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 47 | 89 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 48 | 90 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 49 | 91 | Gemma-7B | MATH500 | 16 | 14.475 | 12.2 | 2.275 |
| 0 | 42 | Gemma-7B | SVAMP | 4 | 41.4 | 39.05 | 2.35 |
| 1 | 43 | Gemma-7B | SVAMP | 4 | 41.9 | 39.05 | 2.85 |
| 2 | 44 | Gemma-7B | SVAMP | 4 | 42.95 | 38.95 | 4 |
| 3 | 45 | Gemma-7B | SVAMP | 4 | 41.15 | 39.2 | 1.95 |
| 4 | 46 | Gemma-7B | SVAMP | 4 | 41.6 | 39.85 | 1.75 |
| 5 | 47 | Gemma-7B | SVAMP | 4 | 42.65 | 39.85 | 2.8 |
| 6 | 48 | Gemma-7B | SVAMP | 4 | 42.2 | 39.9 | 2.3 |
| 7 | 49 | Gemma-7B | SVAMP | 4 | 43.6 | 41.85 | 1.75 |
| 8 | 50 | Gemma-7B | SVAMP | 4 | 41.85 | 39.4 | 2.45 |
| 9 | 51 | Gemma-7B | SVAMP | 4 | 41.7 | 39.25 | 2.45 |
| 10 | 52 | Gemma-7B | SVAMP | 4 | 41.6 | 39.6 | 2 |
| 11 | 53 | Gemma-7B | SVAMP | 4 | 41.4 | 41.45 | -0.05 |
| 12 | 54 | Gemma-7B | SVAMP | 4 | 41.8 | 39.2 | 2.6 |
| 13 | 55 | Gemma-7B | SVAMP | 4 | 42.4 | 37.75 | 4.65 |
| 14 | 56 | Gemma-7B | SVAMP | 4 | 40.95 | 38.6 | 2.35 |
| 15 | 57 | Gemma-7B | SVAMP | 4 | 42.35 | 40.1 | 2.25 |
| 16 | 58 | Gemma-7B | SVAMP | 4 | 43.15 | 40.3 | 2.85 |
| 17 | 59 | Gemma-7B | SVAMP | 4 | 40.7 | 37.65 | 3.05 |
| 18 | 60 | Gemma-7B | SVAMP | 4 | 41.9 | 40.35 | 1.55 |
| 19 | 61 | Gemma-7B | SVAMP | 4 | 41.85 | 40.1 | 1.75 |
| 20 | 62 | Gemma-7B | SVAMP | 4 | 42.6 | 39.4 | 3.2 |
| 21 | 63 | Gemma-7B | SVAMP | 4 | 41.25 | 39.85 | 1.4 |
| 22 | 64 | Gemma-7B | SVAMP | 4 | 42.85 | 38.75 | 4.1 |
| 23 | 65 | Gemma-7B | SVAMP | 4 | 41.6 | 40.95 | 0.65 |
| 24 | 66 | Gemma-7B | SVAMP | 4 | 42.1 | 38.95 | 3.15 |
| 25 | 67 | Gemma-7B | SVAMP | 4 | 40.85 | 40 | 0.85 |
| 26 | 68 | Gemma-7B | SVAMP | 4 | 42.5 | 40.2 | 2.3 |
| 27 | 69 | Gemma-7B | SVAMP | 4 | 42.35 | 38.3 | 4.05 |
| 28 | 70 | Gemma-7B | SVAMP | 4 | 43.25 | 40.1 | 3.15 |
| 29 | 71 | Gemma-7B | SVAMP | 4 | 42.7 | 40.35 | 2.35 |
| 30 | 72 | Gemma-7B | SVAMP | 4 | 41.85 | 39.7 | 2.15 |
| 31 | 73 | Gemma-7B | SVAMP | 4 | 41.9 | 39.4 | 2.5 |
| 32 | 74 | Gemma-7B | SVAMP | 4 | 42.8 | 38.45 | 4.35 |
| 33 | 75 | Gemma-7B | SVAMP | 4 | 43.3 | 38.8 | 4.5 |
| 34 | 76 | Gemma-7B | SVAMP | 4 | 41.9 | 40.25 | 1.65 |
| 35 | 77 | Gemma-7B | SVAMP | 4 | 43.35 | 39.8 | 3.55 |
| 36 | 78 | Gemma-7B | SVAMP | 4 | 42.95 | 38.85 | 4.1 |
| 37 | 79 | Gemma-7B | SVAMP | 4 | 41.75 | 39.9 | 1.85 |
| 38 | 80 | Gemma-7B | SVAMP | 4 | 42.2 | 39.2 | 3 |
| 39 | 81 | Gemma-7B | SVAMP | 4 | 42.35 | 38.95 | 3.4 |
| 40 | 82 | Gemma-7B | SVAMP | 4 | 41.5 | 40.05 | 1.45 |
| 41 | 83 | Gemma-7B | SVAMP | 4 | 42.2 | 41.45 | 0.75 |
| 42 | 84 | Gemma-7B | SVAMP | 4 | 42.6 | 39.5 | 3.1 |
| 43 | 85 | Gemma-7B | SVAMP | 4 | 41.95 | 40.95 | 1 |
| 44 | 86 | Gemma-7B | SVAMP | 4 | 42.65 | 39.25 | 3.4 |
| 45 | 87 | Gemma-7B | SVAMP | 4 | 41.3 | 40.1 | 1.2 |
| 46 | 88 | Gemma-7B | SVAMP | 4 | 41.65 | 40.1 | 1.55 |
| 47 | 89 | Gemma-7B | SVAMP | 4 | 40.8 | 40.5 | 0.3 |
| 48 | 90 | Gemma-7B | SVAMP | 4 | 41.1 | 39.85 | 1.25 |
| 49 | 91 | Gemma-7B | SVAMP | 4 | 41.8 | 38.2 | 3.6 |
| 0 | 42 | Gemma-7B | SVAMP | 8 | 42.65 | 38.125 | 4.525 |
| 1 | 43 | Gemma-7B | SVAMP | 8 | 42.075 | 39.725 | 2.35 |
| 2 | 44 | Gemma-7B | SVAMP | 8 | 43.25 | 39.525 | 3.725 |
| 3 | 45 | Gemma-7B | SVAMP | 8 | 41.9 | 39.65 | 2.25 |
| 4 | 46 | Gemma-7B | SVAMP | 8 | 41.475 | 39.95 | 1.525 |
| 5 | 47 | Gemma-7B | SVAMP | 8 | 42.125 | 39.95 | 2.175 |
| 6 | 48 | Gemma-7B | SVAMP | 8 | 42.4 | 39.975 | 2.425 |
| 7 | 49 | Gemma-7B | SVAMP | 8 | 43.075 | 39.6 | 3.475 |
| 8 | 50 | Gemma-7B | SVAMP | 8 | 43.075 | 38.875 | 4.2 |
| 9 | 51 | Gemma-7B | SVAMP | 8 | 41.875 | 39.4 | 2.475 |
| 10 | 52 | Gemma-7B | SVAMP | 8 | 41.65 | 39.2 | 2.45 |
| 11 | 53 | Gemma-7B | SVAMP | 8 | 41.6 | 40.15 | 1.45 |
| 12 | 54 | Gemma-7B | SVAMP | 8 | 41.6 | 39.3 | 2.3 |
| 13 | 55 | Gemma-7B | SVAMP | 8 | 42.35 | 39.025 | 3.325 |
| 14 | 56 | Gemma-7B | SVAMP | 8 | 41.675 | 39.55 | 2.125 |
| 15 | 57 | Gemma-7B | SVAMP | 8 | 42.25 | 39.85 | 2.4 |
| 16 | 58 | Gemma-7B | SVAMP | 8 | 42.975 | 40.375 | 2.6 |
| 17 | 59 | Gemma-7B | SVAMP | 8 | 41.525 | 38.725 | 2.8 |
| 18 | 60 | Gemma-7B | SVAMP | 8 | 42.85 | 40.45 | 2.4 |
| 19 | 61 | Gemma-7B | SVAMP | 8 | 42.1 | 39.85 | 2.25 |
| 20 | 62 | Gemma-7B | SVAMP | 8 | 42.225 | 38.325 | 3.9 |
| 21 | 63 | Gemma-7B | SVAMP | 8 | 42.2 | 39.8 | 2.4 |
| 22 | 64 | Gemma-7B | SVAMP | 8 | 42.05 | 39.475 | 2.575 |
| 23 | 65 | Gemma-7B | SVAMP | 8 | 41.775 | 39.35 | 2.425 |
| 24 | 66 | Gemma-7B | SVAMP | 8 | 41.275 | 39.675 | 1.6 |
| 25 | 67 | Gemma-7B | SVAMP | 8 | 41.85 | 40.275 | 1.575 |
| 26 | 68 | Gemma-7B | SVAMP | 8 | 42.325 | 39.7 | 2.625 |
| 27 | 69 | Gemma-7B | SVAMP | 8 | 41.775 | 39 | 2.775 |
| 28 | 70 | Gemma-7B | SVAMP | 8 | 42.625 | 40.15 | 2.475 |
| 29 | 71 | Gemma-7B | SVAMP | 8 | 42.05 | 40.1 | 1.95 |
| 30 | 72 | Gemma-7B | SVAMP | 8 | 42.375 | 39.75 | 2.625 |
| 31 | 73 | Gemma-7B | SVAMP | 8 | 41.425 | 39.95 | 1.475 |
| 32 | 74 | Gemma-7B | SVAMP | 8 | 41.975 | 39.525 | 2.45 |
| 33 | 75 | Gemma-7B | SVAMP | 8 | 42.225 | 39.5 | 2.725 |
| 34 | 76 | Gemma-7B | SVAMP | 8 | 42 | 39.975 | 2.025 |
| 35 | 77 | Gemma-7B | SVAMP | 8 | 42.375 | 40.075 | 2.3 |
| 36 | 78 | Gemma-7B | SVAMP | 8 | 42.725 | 40.025 | 2.7 |
| 37 | 79 | Gemma-7B | SVAMP | 8 | 42.55 | 39.425 | 3.125 |
| 38 | 80 | Gemma-7B | SVAMP | 8 | 42.675 | 39.35 | 3.325 |
| 39 | 81 | Gemma-7B | SVAMP | 8 | 42.325 | 39.3 | 3.025 |
| 40 | 82 | Gemma-7B | SVAMP | 8 | 42.1 | 39.575 | 2.525 |
| 41 | 83 | Gemma-7B | SVAMP | 8 | 41.9 | 40.2 | 1.7 |
| 42 | 84 | Gemma-7B | SVAMP | 8 | 41.525 | 39.5 | 2.025 |
| 43 | 85 | Gemma-7B | SVAMP | 8 | 42.725 | 40.275 | 2.45 |
| 44 | 86 | Gemma-7B | SVAMP | 8 | 42.35 | 39.05 | 3.3 |
| 45 | 87 | Gemma-7B | SVAMP | 8 | 41.725 | 39.65 | 2.075 |
| 46 | 88 | Gemma-7B | SVAMP | 8 | 42.875 | 39.525 | 3.35 |
| 47 | 89 | Gemma-7B | SVAMP | 8 | 41.75 | 39.975 | 1.775 |
| 48 | 90 | Gemma-7B | SVAMP | 8 | 41.8 | 40.525 | 1.275 |
| 49 | 91 | Gemma-7B | SVAMP | 8 | 41.65 | 39.1 | 2.55 |
| 0 | 42 | Gemma-7B | SVAMP | 12 | 41.95 | 39.15 | 2.8 |
| 1 | 43 | Gemma-7B | SVAMP | 12 | 42.4667 | 39.05 | 3.4167 |
| 2 | 44 | Gemma-7B | SVAMP | 12 | 42.6667 | 39.4333 | 3.2333 |
| 3 | 45 | Gemma-7B | SVAMP | 12 | 42.2167 | 39.3833 | 2.8333 |
| 4 | 46 | Gemma-7B | SVAMP | 12 | 42 | 39.9333 | 2.0667 |
| 5 | 47 | Gemma-7B | SVAMP | 12 | 41.7667 | 39.4167 | 2.35 |
| 6 | 48 | Gemma-7B | SVAMP | 12 | 42.8833 | 39.5833 | 3.3 |
| 7 | 49 | Gemma-7B | SVAMP | 12 | 42.45 | 39.7167 | 2.7333 |
| 8 | 50 | Gemma-7B | SVAMP | 12 | 43.0333 | 38.8833 | 4.15 |
| 9 | 51 | Gemma-7B | SVAMP | 12 | 42.4667 | 38.9667 | 3.5 |
| 10 | 52 | Gemma-7B | SVAMP | 12 | 42.0833 | 39.5333 | 2.55 |
| 11 | 53 | Gemma-7B | SVAMP | 12 | 42.1167 | 39.2333 | 2.8833 |
| 12 | 54 | Gemma-7B | SVAMP | 12 | 42.3 | 38.8833 | 3.4167 |
| 13 | 55 | Gemma-7B | SVAMP | 12 | 42.2167 | 39.5333 | 2.6833 |
| 14 | 56 | Gemma-7B | SVAMP | 12 | 42.4 | 39.5 | 2.9 |
| 15 | 57 | Gemma-7B | SVAMP | 12 | 42.8333 | 39.25 | 3.5833 |
| 16 | 58 | Gemma-7B | SVAMP | 12 | 42.6833 | 39.9333 | 2.75 |
| 17 | 59 | Gemma-7B | SVAMP | 12 | 41.9333 | 39.15 | 2.7833 |
| 18 | 60 | Gemma-7B | SVAMP | 12 | 42.7667 | 39.7833 | 2.9833 |
| 19 | 61 | Gemma-7B | SVAMP | 12 | 42.1833 | 39.9 | 2.2833 |
| 20 | 62 | Gemma-7B | SVAMP | 12 | 42.2667 | 39.4167 | 2.85 |
| 21 | 63 | Gemma-7B | SVAMP | 12 | 41.9833 | 40.05 | 1.9333 |
| 22 | 64 | Gemma-7B | SVAMP | 12 | 42.25 | 39.2667 | 2.9833 |
| 23 | 65 | Gemma-7B | SVAMP | 12 | 42.0833 | 39.2333 | 2.85 |
| 24 | 66 | Gemma-7B | SVAMP | 12 | 42.2167 | 39.3333 | 2.8833 |
| 25 | 67 | Gemma-7B | SVAMP | 12 | 42.3667 | 39.8333 | 2.5333 |
| 26 | 68 | Gemma-7B | SVAMP | 12 | 42.0833 | 39.8 | 2.2833 |
| 27 | 69 | Gemma-7B | SVAMP | 12 | 41.65 | 39.75 | 1.9 |
| 28 | 70 | Gemma-7B | SVAMP | 12 | 42.45 | 39.9667 | 2.4833 |
| 29 | 71 | Gemma-7B | SVAMP | 12 | 42.85 | 39.35 | 3.5 |
| 30 | 72 | Gemma-7B | SVAMP | 12 | 42.2333 | 39.6667 | 2.5667 |
| 31 | 73 | Gemma-7B | SVAMP | 12 | 42.5167 | 39.6 | 2.9167 |
| 32 | 74 | Gemma-7B | SVAMP | 12 | 42.3 | 39.5833 | 2.7167 |
| 33 | 75 | Gemma-7B | SVAMP | 12 | 42.2167 | 39.4667 | 2.75 |
| 34 | 76 | Gemma-7B | SVAMP | 12 | 42.2667 | 39.4167 | 2.85 |
| 35 | 77 | Gemma-7B | SVAMP | 12 | 42.8667 | 39.7333 | 3.1333 |
| 36 | 78 | Gemma-7B | SVAMP | 12 | 42.7 | 39.6667 | 3.0333 |
| 37 | 79 | Gemma-7B | SVAMP | 12 | 42.1833 | 39.9167 | 2.2667 |
| 38 | 80 | Gemma-7B | SVAMP | 12 | 42.5 | 39.3833 | 3.1167 |
| 39 | 81 | Gemma-7B | SVAMP | 12 | 42.4167 | 39.6 | 2.8167 |
| 40 | 82 | Gemma-7B | SVAMP | 12 | 42.0667 | 39.7167 | 2.35 |
| 41 | 83 | Gemma-7B | SVAMP | 12 | 41.7667 | 39.9 | 1.8667 |
| 42 | 84 | Gemma-7B | SVAMP | 12 | 42.2833 | 39.4833 | 2.8 |
| 43 | 85 | Gemma-7B | SVAMP | 12 | 42.6667 | 39.4667 | 3.2 |
| 44 | 86 | Gemma-7B | SVAMP | 12 | 42.5833 | 39.7 | 2.8833 |
| 45 | 87 | Gemma-7B | SVAMP | 12 | 42.2167 | 39.55 | 2.6667 |
| 46 | 88 | Gemma-7B | SVAMP | 12 | 42.6 | 39.65 | 2.95 |
| 47 | 89 | Gemma-7B | SVAMP | 12 | 41.7 | 40.4 | 1.3 |
| 48 | 90 | Gemma-7B | SVAMP | 12 | 42.5 | 39.7833 | 2.7167 |
| 49 | 91 | Gemma-7B | SVAMP | 12 | 42.1 | 39.65 | 2.45 |
| 0 | 42 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 1 | 43 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 2 | 44 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 3 | 45 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 4 | 46 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 5 | 47 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 6 | 48 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 7 | 49 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 8 | 50 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 9 | 51 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 10 | 52 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 11 | 53 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 12 | 54 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 13 | 55 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 14 | 56 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 15 | 57 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 16 | 58 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 17 | 59 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 18 | 60 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 19 | 61 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 20 | 62 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 21 | 63 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 22 | 64 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 23 | 65 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 24 | 66 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 25 | 67 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 26 | 68 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 27 | 69 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 28 | 70 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 29 | 71 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 30 | 72 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 31 | 73 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 32 | 74 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 33 | 75 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 34 | 76 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 35 | 77 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 36 | 78 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 37 | 79 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 38 | 80 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 39 | 81 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 40 | 82 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 41 | 83 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 42 | 84 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 43 | 85 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 44 | 86 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 45 | 87 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 46 | 88 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 47 | 89 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 48 | 90 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 49 | 91 | Gemma-7B | SVAMP | 16 | 42.425 | 39.45 | 2.975 |
| 0 | 42 | LLaMA-3.1-8B | AQuA | 4 | 43.1102 | 46.4567 | -3.3465 |
| 1 | 43 | LLaMA-3.1-8B | AQuA | 4 | 44.685 | 42.126 | 2.5591 |
| 2 | 44 | LLaMA-3.1-8B | AQuA | 4 | 41.7323 | 46.2598 | -4.5276 |
| 3 | 45 | LLaMA-3.1-8B | AQuA | 4 | 39.7638 | 44.8819 | -5.1181 |
| 4 | 46 | LLaMA-3.1-8B | AQuA | 4 | 39.1732 | 45.4724 | -6.2992 |
| 5 | 47 | LLaMA-3.1-8B | AQuA | 4 | 41.7323 | 47.0472 | -5.315 |
| 6 | 48 | LLaMA-3.1-8B | AQuA | 4 | 43.7008 | 44.685 | -0.9843 |
| 7 | 49 | LLaMA-3.1-8B | AQuA | 4 | 43.8976 | 45.6693 | -1.7717 |
| 8 | 50 | LLaMA-3.1-8B | AQuA | 4 | 43.3071 | 45.2756 | -1.9685 |
| 9 | 51 | LLaMA-3.1-8B | AQuA | 4 | 42.9134 | 47.0472 | -4.1339 |
| 10 | 52 | LLaMA-3.1-8B | AQuA | 4 | 42.7165 | 47.0472 | -4.3307 |
| 11 | 53 | LLaMA-3.1-8B | AQuA | 4 | 41.7323 | 47.2441 | -5.5118 |
| 12 | 54 | LLaMA-3.1-8B | AQuA | 4 | 43.1102 | 46.063 | -2.9528 |
| 13 | 55 | LLaMA-3.1-8B | AQuA | 4 | 43.8976 | 48.0315 | -4.1339 |
| 14 | 56 | LLaMA-3.1-8B | AQuA | 4 | 41.3386 | 42.7165 | -1.378 |
| 15 | 57 | LLaMA-3.1-8B | AQuA | 4 | 43.8976 | 46.4567 | -2.5591 |
| 16 | 58 | LLaMA-3.1-8B | AQuA | 4 | 44.685 | 48.2283 | -3.5433 |
| 17 | 59 | LLaMA-3.1-8B | AQuA | 4 | 43.7008 | 47.4409 | -3.7402 |
| 18 | 60 | LLaMA-3.1-8B | AQuA | 4 | 41.5354 | 44.4882 | -2.9528 |
| 19 | 61 | LLaMA-3.1-8B | AQuA | 4 | 43.5039 | 46.063 | -2.5591 |
| 20 | 62 | LLaMA-3.1-8B | AQuA | 4 | 43.5039 | 44.8819 | -1.378 |
| 21 | 63 | LLaMA-3.1-8B | AQuA | 4 | 41.7323 | 44.8819 | -3.1496 |
| 22 | 64 | LLaMA-3.1-8B | AQuA | 4 | 41.3386 | 45.8661 | -4.5276 |
| 23 | 65 | LLaMA-3.1-8B | AQuA | 4 | 43.1102 | 46.6535 | -3.5433 |
| 24 | 66 | LLaMA-3.1-8B | AQuA | 4 | 42.7165 | 45.8661 | -3.1496 |
| 25 | 67 | LLaMA-3.1-8B | AQuA | 4 | 42.9134 | 47.6378 | -4.7244 |
| 26 | 68 | LLaMA-3.1-8B | AQuA | 4 | 44.0945 | 45.4724 | -1.378 |
| 27 | 69 | LLaMA-3.1-8B | AQuA | 4 | 40.5512 | 47.4409 | -6.8898 |
| 28 | 70 | LLaMA-3.1-8B | AQuA | 4 | 41.7323 | 45.4724 | -3.7402 |
| 29 | 71 | LLaMA-3.1-8B | AQuA | 4 | 40.9449 | 45.6693 | -4.7244 |
| 30 | 72 | LLaMA-3.1-8B | AQuA | 4 | 43.8976 | 45.8661 | -1.9685 |
| 31 | 73 | LLaMA-3.1-8B | AQuA | 4 | 43.5039 | 45.6693 | -2.1654 |
| 32 | 74 | LLaMA-3.1-8B | AQuA | 4 | 45.6693 | 46.2598 | -0.5906 |
| 33 | 75 | LLaMA-3.1-8B | AQuA | 4 | 42.7165 | 44.0945 | -1.378 |
| 34 | 76 | LLaMA-3.1-8B | AQuA | 4 | 43.1102 | 43.5039 | -0.3937 |
| 35 | 77 | LLaMA-3.1-8B | AQuA | 4 | 42.9134 | 44.0945 | -1.1811 |
| 36 | 78 | LLaMA-3.1-8B | AQuA | 4 | 44.4882 | 46.6535 | -2.1654 |
| 37 | 79 | LLaMA-3.1-8B | AQuA | 4 | 43.1102 | 47.6378 | -4.5276 |
| 38 | 80 | LLaMA-3.1-8B | AQuA | 4 | 43.5039 | 46.2598 | -2.7559 |
| 39 | 81 | LLaMA-3.1-8B | AQuA | 4 | 39.5669 | 44.8819 | -5.315 |
| 40 | 82 | LLaMA-3.1-8B | AQuA | 4 | 43.8976 | 48.0315 | -4.1339 |
| 41 | 83 | LLaMA-3.1-8B | AQuA | 4 | 39.3701 | 46.4567 | -7.0866 |
| 42 | 84 | LLaMA-3.1-8B | AQuA | 4 | 40.5512 | 45.8661 | -5.315 |
| 43 | 85 | LLaMA-3.1-8B | AQuA | 4 | 41.7323 | 44.8819 | -3.1496 |
| 44 | 86 | LLaMA-3.1-8B | AQuA | 4 | 45.8661 | 46.8504 | -0.9843 |
| 45 | 87 | LLaMA-3.1-8B | AQuA | 4 | 40.3543 | 44.4882 | -4.1339 |
| 46 | 88 | LLaMA-3.1-8B | AQuA | 4 | 45.2756 | 44.8819 | 0.3937 |
| 47 | 89 | LLaMA-3.1-8B | AQuA | 4 | 39.5669 | 44.8819 | -5.315 |
| 48 | 90 | LLaMA-3.1-8B | AQuA | 4 | 39.5669 | 45.6693 | -6.1024 |
| 49 | 91 | LLaMA-3.1-8B | AQuA | 4 | 41.1417 | 43.8976 | -2.7559 |
| 0 | 42 | LLaMA-3.1-8B | AQuA | 8 | 43.3071 | 47.2441 | -3.937 |
| 1 | 43 | LLaMA-3.1-8B | AQuA | 8 | 41.9291 | 45.4724 | -3.5433 |
| 2 | 44 | LLaMA-3.1-8B | AQuA | 8 | 41.437 | 47.2441 | -5.8071 |
| 3 | 45 | LLaMA-3.1-8B | AQuA | 8 | 38.878 | 46.8504 | -7.9724 |
| 4 | 46 | LLaMA-3.1-8B | AQuA | 8 | 41.0433 | 44.2913 | -3.248 |
| 5 | 47 | LLaMA-3.1-8B | AQuA | 8 | 43.1102 | 44.5866 | -1.4764 |
| 6 | 48 | LLaMA-3.1-8B | AQuA | 8 | 42.6181 | 44.5866 | -1.9685 |
| 7 | 49 | LLaMA-3.1-8B | AQuA | 8 | 42.3228 | 46.8504 | -4.5276 |
| 8 | 50 | LLaMA-3.1-8B | AQuA | 8 | 44.1929 | 47.5394 | -3.3465 |
| 9 | 51 | LLaMA-3.1-8B | AQuA | 8 | 42.9134 | 46.752 | -3.8386 |
| 10 | 52 | LLaMA-3.1-8B | AQuA | 8 | 42.0276 | 47.2441 | -5.2165 |
| 11 | 53 | LLaMA-3.1-8B | AQuA | 8 | 41.437 | 47.3425 | -5.9055 |
| 12 | 54 | LLaMA-3.1-8B | AQuA | 8 | 41.6339 | 46.1614 | -4.5276 |
| 13 | 55 | LLaMA-3.1-8B | AQuA | 8 | 42.3228 | 46.063 | -3.7402 |
| 14 | 56 | LLaMA-3.1-8B | AQuA | 8 | 39.4685 | 46.2598 | -6.7913 |
| 15 | 57 | LLaMA-3.1-8B | AQuA | 8 | 42.5197 | 46.063 | -3.5433 |
| 16 | 58 | LLaMA-3.1-8B | AQuA | 8 | 44.0945 | 47.5394 | -3.4449 |
| 17 | 59 | LLaMA-3.1-8B | AQuA | 8 | 42.0276 | 46.2598 | -4.2323 |
| 18 | 60 | LLaMA-3.1-8B | AQuA | 8 | 42.126 | 45.9646 | -3.8386 |
| 19 | 61 | LLaMA-3.1-8B | AQuA | 8 | 42.815 | 47.1457 | -4.3307 |
| 20 | 62 | LLaMA-3.1-8B | AQuA | 8 | 42.4213 | 46.3583 | -3.937 |
| 21 | 63 | LLaMA-3.1-8B | AQuA | 8 | 41.6339 | 45.7677 | -4.1339 |
| 22 | 64 | LLaMA-3.1-8B | AQuA | 8 | 41.1417 | 46.2598 | -5.1181 |
| 23 | 65 | LLaMA-3.1-8B | AQuA | 8 | 42.0276 | 46.063 | -4.0354 |
| 24 | 66 | LLaMA-3.1-8B | AQuA | 8 | 41.1417 | 47.3425 | -6.2008 |
| 25 | 67 | LLaMA-3.1-8B | AQuA | 8 | 40.3543 | 48.4252 | -8.0709 |
| 26 | 68 | LLaMA-3.1-8B | AQuA | 8 | 43.5039 | 48.8189 | -5.315 |
| 27 | 69 | LLaMA-3.1-8B | AQuA | 8 | 40.6496 | 48.8189 | -8.1693 |
| 28 | 70 | LLaMA-3.1-8B | AQuA | 8 | 43.7008 | 45.6693 | -1.9685 |
| 29 | 71 | LLaMA-3.1-8B | AQuA | 8 | 41.1417 | 46.9488 | -5.8071 |
| 30 | 72 | LLaMA-3.1-8B | AQuA | 8 | 41.1417 | 48.2283 | -7.0866 |
| 31 | 73 | LLaMA-3.1-8B | AQuA | 8 | 42.6181 | 44.4882 | -1.8701 |
| 32 | 74 | LLaMA-3.1-8B | AQuA | 8 | 42.0276 | 46.4567 | -4.4291 |
| 33 | 75 | LLaMA-3.1-8B | AQuA | 8 | 41.8307 | 45.0787 | -3.248 |
| 34 | 76 | LLaMA-3.1-8B | AQuA | 8 | 42.815 | 44.2913 | -1.4764 |
| 35 | 77 | LLaMA-3.1-8B | AQuA | 8 | 41.6339 | 47.3425 | -5.7087 |
| 36 | 78 | LLaMA-3.1-8B | AQuA | 8 | 40.8465 | 47.0472 | -6.2008 |
| 37 | 79 | LLaMA-3.1-8B | AQuA | 8 | 42.5197 | 47.3425 | -4.8228 |
| 38 | 80 | LLaMA-3.1-8B | AQuA | 8 | 41.5354 | 47.3425 | -5.8071 |
| 39 | 81 | LLaMA-3.1-8B | AQuA | 8 | 42.126 | 45.6693 | -3.5433 |
| 40 | 82 | LLaMA-3.1-8B | AQuA | 8 | 42.815 | 47.3425 | -4.5276 |
| 41 | 83 | LLaMA-3.1-8B | AQuA | 8 | 42.126 | 46.9488 | -4.8228 |
| 42 | 84 | LLaMA-3.1-8B | AQuA | 8 | 41.1417 | 46.4567 | -5.315 |
| 43 | 85 | LLaMA-3.1-8B | AQuA | 8 | 39.5669 | 46.8504 | -7.2835 |
| 44 | 86 | LLaMA-3.1-8B | AQuA | 8 | 43.1102 | 46.5551 | -3.4449 |
| 45 | 87 | LLaMA-3.1-8B | AQuA | 8 | 41.8307 | 46.1614 | -4.3307 |
| 46 | 88 | LLaMA-3.1-8B | AQuA | 8 | 42.2244 | 45.0787 | -2.8543 |
| 47 | 89 | LLaMA-3.1-8B | AQuA | 8 | 40.1575 | 47.8346 | -7.6772 |
| 48 | 90 | LLaMA-3.1-8B | AQuA | 8 | 40.2559 | 46.752 | -6.4961 |
| 49 | 91 | LLaMA-3.1-8B | AQuA | 8 | 41.0433 | 45.5709 | -4.5276 |
| 0 | 42 | LLaMA-3.1-8B | AQuA | 12 | 41.273 | 47.9003 | -6.6273 |
| 1 | 43 | LLaMA-3.1-8B | AQuA | 12 | 40.6824 | 45.9318 | -5.2493 |
| 2 | 44 | LLaMA-3.1-8B | AQuA | 12 | 41.6667 | 46.2598 | -4.5932 |
| 3 | 45 | LLaMA-3.1-8B | AQuA | 12 | 40.8136 | 45.9974 | -5.1837 |
| 4 | 46 | LLaMA-3.1-8B | AQuA | 12 | 40.3543 | 46.063 | -5.7087 |
| 5 | 47 | LLaMA-3.1-8B | AQuA | 12 | 42.126 | 46.8504 | -4.7244 |
| 6 | 48 | LLaMA-3.1-8B | AQuA | 12 | 41.0105 | 46.5223 | -5.5118 |
| 7 | 49 | LLaMA-3.1-8B | AQuA | 12 | 41.601 | 46.916 | -5.315 |
| 8 | 50 | LLaMA-3.1-8B | AQuA | 12 | 41.9948 | 47.1129 | -5.1181 |
| 9 | 51 | LLaMA-3.1-8B | AQuA | 12 | 43.1102 | 46.916 | -3.8058 |
| 10 | 52 | LLaMA-3.1-8B | AQuA | 12 | 41.5354 | 47.6378 | -6.1024 |
| 11 | 53 | LLaMA-3.1-8B | AQuA | 12 | 41.273 | 46.8504 | -5.5774 |
| 12 | 54 | LLaMA-3.1-8B | AQuA | 12 | 41.3386 | 46.6535 | -5.315 |
| 13 | 55 | LLaMA-3.1-8B | AQuA | 12 | 42.7822 | 46.5223 | -3.7402 |
| 14 | 56 | LLaMA-3.1-8B | AQuA | 12 | 41.601 | 47.1785 | -5.5774 |
| 15 | 57 | LLaMA-3.1-8B | AQuA | 12 | 42.1916 | 47.1129 | -4.9213 |
| 16 | 58 | LLaMA-3.1-8B | AQuA | 12 | 43.1759 | 48.0971 | -4.9213 |
| 17 | 59 | LLaMA-3.1-8B | AQuA | 12 | 42.9134 | 46.916 | -4.0026 |
| 18 | 60 | LLaMA-3.1-8B | AQuA | 12 | 41.9948 | 47.1785 | -5.1837 |
| 19 | 61 | LLaMA-3.1-8B | AQuA | 12 | 42.126 | 46.3255 | -4.1995 |
| 20 | 62 | LLaMA-3.1-8B | AQuA | 12 | 42.7165 | 47.1129 | -4.3963 |
| 21 | 63 | LLaMA-3.1-8B | AQuA | 12 | 42.6509 | 46.5223 | -3.8714 |
| 22 | 64 | LLaMA-3.1-8B | AQuA | 12 | 40.9449 | 46.6535 | -5.7087 |
| 23 | 65 | LLaMA-3.1-8B | AQuA | 12 | 42.3885 | 45.9974 | -3.6089 |
| 24 | 66 | LLaMA-3.1-8B | AQuA | 12 | 41.9291 | 46.916 | -4.9869 |
| 25 | 67 | LLaMA-3.1-8B | AQuA | 12 | 40.4199 | 47.3097 | -6.8898 |
| 26 | 68 | LLaMA-3.1-8B | AQuA | 12 | 42.5197 | 46.916 | -4.3963 |
| 27 | 69 | LLaMA-3.1-8B | AQuA | 12 | 41.3386 | 47.8346 | -6.4961 |
| 28 | 70 | LLaMA-3.1-8B | AQuA | 12 | 42.3885 | 47.1785 | -4.79 |
| 29 | 71 | LLaMA-3.1-8B | AQuA | 12 | 41.6667 | 47.3097 | -5.643 |
| 30 | 72 | LLaMA-3.1-8B | AQuA | 12 | 42.126 | 46.916 | -4.79 |
| 31 | 73 | LLaMA-3.1-8B | AQuA | 12 | 42.9134 | 46.3255 | -3.4121 |
| 32 | 74 | LLaMA-3.1-8B | AQuA | 12 | 41.273 | 47.1785 | -5.9055 |
| 33 | 75 | LLaMA-3.1-8B | AQuA | 12 | 41.9291 | 46.4567 | -4.5276 |
| 34 | 76 | LLaMA-3.1-8B | AQuA | 12 | 41.6667 | 46.063 | -4.3963 |
| 35 | 77 | LLaMA-3.1-8B | AQuA | 12 | 42.0604 | 46.7848 | -4.7244 |
| 36 | 78 | LLaMA-3.1-8B | AQuA | 12 | 41.3386 | 48.4252 | -7.0866 |
| 37 | 79 | LLaMA-3.1-8B | AQuA | 12 | 42.1916 | 46.9816 | -4.79 |
| 38 | 80 | LLaMA-3.1-8B | AQuA | 12 | 42.126 | 47.0472 | -4.9213 |
| 39 | 81 | LLaMA-3.1-8B | AQuA | 12 | 42.6509 | 45.8661 | -3.2152 |
| 40 | 82 | LLaMA-3.1-8B | AQuA | 12 | 41.273 | 47.769 | -6.4961 |
| 41 | 83 | LLaMA-3.1-8B | AQuA | 12 | 42.2572 | 46.7848 | -4.5276 |
| 42 | 84 | LLaMA-3.1-8B | AQuA | 12 | 42.126 | 45.5381 | -3.4121 |
| 43 | 85 | LLaMA-3.1-8B | AQuA | 12 | 41.4042 | 46.1286 | -4.7244 |
| 44 | 86 | LLaMA-3.1-8B | AQuA | 12 | 42.4541 | 46.7848 | -4.3307 |
| 45 | 87 | LLaMA-3.1-8B | AQuA | 12 | 41.4698 | 46.5223 | -5.0525 |
| 46 | 88 | LLaMA-3.1-8B | AQuA | 12 | 41.0761 | 47.2441 | -6.168 |
| 47 | 89 | LLaMA-3.1-8B | AQuA | 12 | 41.7979 | 47.4409 | -5.643 |
| 48 | 90 | LLaMA-3.1-8B | AQuA | 12 | 41.7323 | 46.5223 | -4.79 |
| 49 | 91 | LLaMA-3.1-8B | AQuA | 12 | 40.5512 | 47.769 | -7.2178 |
| 0 | 42 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 1 | 43 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 2 | 44 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 3 | 45 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 4 | 46 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 5 | 47 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 6 | 48 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 7 | 49 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 8 | 50 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 9 | 51 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 10 | 52 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 11 | 53 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 12 | 54 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 13 | 55 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 14 | 56 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 15 | 57 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 16 | 58 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 17 | 59 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 18 | 60 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 19 | 61 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 20 | 62 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 21 | 63 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 22 | 64 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 23 | 65 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 24 | 66 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 25 | 67 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 26 | 68 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 27 | 69 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 28 | 70 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 29 | 71 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 30 | 72 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 31 | 73 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 32 | 74 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 33 | 75 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 34 | 76 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 35 | 77 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 36 | 78 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 37 | 79 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 38 | 80 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 39 | 81 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 40 | 82 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 41 | 83 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 42 | 84 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 43 | 85 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 44 | 86 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 45 | 87 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 46 | 88 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 47 | 89 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 48 | 90 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 49 | 91 | LLaMA-3.1-8B | AQuA | 16 | 41.6339 | 47.0472 | -5.4134 |
| 0 | 42 | LLaMA-3.1-8B | CommonsenseQA | 4 | 65.7248 | 64.5373 | 1.1876 |
| 1 | 43 | LLaMA-3.1-8B | CommonsenseQA | 4 | 65.4791 | 65.0696 | 0.4095 |
| 2 | 44 | LLaMA-3.1-8B | CommonsenseQA | 4 | 65.3563 | 64.9877 | 0.3686 |
| 3 | 45 | LLaMA-3.1-8B | CommonsenseQA | 4 | 63.8002 | 64.2097 | -0.4095 |
| 4 | 46 | LLaMA-3.1-8B | CommonsenseQA | 4 | 63.4316 | 64.783 | -1.3514 |
| 5 | 47 | LLaMA-3.1-8B | CommonsenseQA | 4 | 64.9058 | 65.561 | -0.6552 |
| 6 | 48 | LLaMA-3.1-8B | CommonsenseQA | 4 | 63.3088 | 65.561 | -2.2523 |
| 7 | 49 | LLaMA-3.1-8B | CommonsenseQA | 4 | 65.2334 | 65.3153 | -0.0819 |
| 8 | 50 | LLaMA-3.1-8B | CommonsenseQA | 4 | 63.964 | 65.5201 | -1.5561 |
| 9 | 51 | LLaMA-3.1-8B | CommonsenseQA | 4 | 65.7658 | 63.8002 | 1.9656 |
| 10 | 52 | LLaMA-3.1-8B | CommonsenseQA | 4 | 66.2162 | 64.3325 | 1.8837 |
| 11 | 53 | LLaMA-3.1-8B | CommonsenseQA | 4 | 64.3735 | 65.8477 | -1.4742 |
| 12 | 54 | LLaMA-3.1-8B | CommonsenseQA | 4 | 64.5782 | 64.6601 | -0.0819 |
| 13 | 55 | LLaMA-3.1-8B | CommonsenseQA | 4 | 63.5954 | 65.9705 | -2.3751 |
| 14 | 56 | LLaMA-3.1-8B | CommonsenseQA | 4 | 65.602 | 65.3153 | 0.2867 |
| 15 | 57 | LLaMA-3.1-8B | CommonsenseQA | 4 | 65.7658 | 64.6192 | 1.1466 |
| 16 | 58 | LLaMA-3.1-8B | CommonsenseQA | 4 | 65.1925 | 65.3563 | -0.1638 |
| 17 | 59 | LLaMA-3.1-8B | CommonsenseQA | 4 | 65.4382 | 64.9058 | 0.5324 |
| 18 | 60 | LLaMA-3.1-8B | CommonsenseQA | 4 | 64.3735 | 64.3325 | 0.041 |
| 19 | 61 | LLaMA-3.1-8B | CommonsenseQA | 4 | 65.0696 | 64.1687 | 0.9009 |
| 20 | 62 | LLaMA-3.1-8B | CommonsenseQA | 4 | 64.783 | 65.5201 | -0.7371 |
| 21 | 63 | LLaMA-3.1-8B | CommonsenseQA | 4 | 64.6601 | 65.2744 | -0.6143 |
| 22 | 64 | LLaMA-3.1-8B | CommonsenseQA | 4 | 65.1106 | 66.0934 | -0.9828 |
| 23 | 65 | LLaMA-3.1-8B | CommonsenseQA | 4 | 64.5782 | 64.5782 | 0 |
| 24 | 66 | LLaMA-3.1-8B | CommonsenseQA | 4 | 64.9877 | 64.783 | 0.2048 |
| 25 | 67 | LLaMA-3.1-8B | CommonsenseQA | 4 | 64.8649 | 64.7011 | 0.1638 |
| 26 | 68 | LLaMA-3.1-8B | CommonsenseQA | 4 | 65.5201 | 66.1753 | -0.6552 |
| 27 | 69 | LLaMA-3.1-8B | CommonsenseQA | 4 | 64.783 | 65.3563 | -0.5733 |
| 28 | 70 | LLaMA-3.1-8B | CommonsenseQA | 4 | 65.7658 | 64.783 | 0.9828 |
| 29 | 71 | LLaMA-3.1-8B | CommonsenseQA | 4 | 64.783 | 64.6192 | 0.1638 |
| 30 | 72 | LLaMA-3.1-8B | CommonsenseQA | 4 | 65.4382 | 65.2334 | 0.2048 |
| 31 | 73 | LLaMA-3.1-8B | CommonsenseQA | 4 | 65.6429 | 64.4144 | 1.2285 |
| 32 | 74 | LLaMA-3.1-8B | CommonsenseQA | 4 | 65.0696 | 65.561 | -0.4914 |
| 33 | 75 | LLaMA-3.1-8B | CommonsenseQA | 4 | 64.8239 | 65.8067 | -0.9828 |
| 34 | 76 | LLaMA-3.1-8B | CommonsenseQA | 4 | 63.964 | 65.7248 | -1.7609 |
| 35 | 77 | LLaMA-3.1-8B | CommonsenseQA | 4 | 65.8067 | 64.2097 | 1.5971 |
| 36 | 78 | LLaMA-3.1-8B | CommonsenseQA | 4 | 64.9877 | 65.561 | -0.5733 |
| 37 | 79 | LLaMA-3.1-8B | CommonsenseQA | 4 | 65.2334 | 64.7011 | 0.5324 |
| 38 | 80 | LLaMA-3.1-8B | CommonsenseQA | 4 | 65.0696 | 65.0287 | 0.041 |
| 39 | 81 | LLaMA-3.1-8B | CommonsenseQA | 4 | 65.4382 | 65.0696 | 0.3686 |
| 40 | 82 | LLaMA-3.1-8B | CommonsenseQA | 4 | 64.5782 | 65.5201 | -0.9419 |
| 41 | 83 | LLaMA-3.1-8B | CommonsenseQA | 4 | 64.9058 | 66.2572 | -1.3514 |
| 42 | 84 | LLaMA-3.1-8B | CommonsenseQA | 4 | 65.2744 | 64.0049 | 1.2695 |
| 43 | 85 | LLaMA-3.1-8B | CommonsenseQA | 4 | 64.8239 | 64.9468 | -0.1229 |
| 44 | 86 | LLaMA-3.1-8B | CommonsenseQA | 4 | 63.8411 | 65.5201 | -1.679 |
| 45 | 87 | LLaMA-3.1-8B | CommonsenseQA | 4 | 63.923 | 65.1106 | -1.1876 |
| 46 | 88 | LLaMA-3.1-8B | CommonsenseQA | 4 | 65.3972 | 65.1106 | 0.2867 |
| 47 | 89 | LLaMA-3.1-8B | CommonsenseQA | 4 | 64.4554 | 65.1925 | -0.7371 |
| 48 | 90 | LLaMA-3.1-8B | CommonsenseQA | 4 | 64.8239 | 65.1106 | -0.2867 |
| 49 | 91 | LLaMA-3.1-8B | CommonsenseQA | 4 | 65.602 | 66.1343 | -0.5324 |
| 0 | 42 | LLaMA-3.1-8B | CommonsenseQA | 8 | 65.4791 | 65.2539 | 0.2252 |
| 1 | 43 | LLaMA-3.1-8B | CommonsenseQA | 8 | 64.783 | 64.742 | 0.041 |
| 2 | 44 | LLaMA-3.1-8B | CommonsenseQA | 8 | 64.7625 | 65.3153 | -0.5528 |
| 3 | 45 | LLaMA-3.1-8B | CommonsenseQA | 8 | 65.2744 | 64.9058 | 0.3686 |
| 4 | 46 | LLaMA-3.1-8B | CommonsenseQA | 8 | 63.8616 | 65.5815 | -1.7199 |
| 5 | 47 | LLaMA-3.1-8B | CommonsenseQA | 8 | 64.9468 | 64.6601 | 0.2867 |
| 6 | 48 | LLaMA-3.1-8B | CommonsenseQA | 8 | 64.4554 | 65.2334 | -0.7781 |
| 7 | 49 | LLaMA-3.1-8B | CommonsenseQA | 8 | 64.9263 | 65.0287 | -0.1024 |
| 8 | 50 | LLaMA-3.1-8B | CommonsenseQA | 8 | 64.8649 | 65.3358 | -0.4709 |
| 9 | 51 | LLaMA-3.1-8B | CommonsenseQA | 8 | 65.2744 | 64.6601 | 0.6143 |
| 10 | 52 | LLaMA-3.1-8B | CommonsenseQA | 8 | 65.5405 | 65.0287 | 0.5119 |
| 11 | 53 | LLaMA-3.1-8B | CommonsenseQA | 8 | 64.9877 | 65.2334 | -0.2457 |
| 12 | 54 | LLaMA-3.1-8B | CommonsenseQA | 8 | 64.8649 | 65.2129 | -0.3481 |
| 13 | 55 | LLaMA-3.1-8B | CommonsenseQA | 8 | 64.5987 | 65.4382 | -0.8395 |
| 14 | 56 | LLaMA-3.1-8B | CommonsenseQA | 8 | 65.602 | 65.4177 | 0.1843 |
| 15 | 57 | LLaMA-3.1-8B | CommonsenseQA | 8 | 64.7215 | 65.3153 | -0.5938 |
| 16 | 58 | LLaMA-3.1-8B | CommonsenseQA | 8 | 64.9058 | 64.9263 | -0.0205 |
| 17 | 59 | LLaMA-3.1-8B | CommonsenseQA | 8 | 65.2744 | 65.4586 | -0.1843 |
| 18 | 60 | LLaMA-3.1-8B | CommonsenseQA | 8 | 65.0491 | 65.3358 | -0.2867 |
| 19 | 61 | LLaMA-3.1-8B | CommonsenseQA | 8 | 65.0082 | 65.3153 | -0.3071 |
| 20 | 62 | LLaMA-3.1-8B | CommonsenseQA | 8 | 64.3735 | 65.4382 | -1.0647 |
| 21 | 63 | LLaMA-3.1-8B | CommonsenseQA | 8 | 64.8649 | 65.3767 | -0.5119 |
| 22 | 64 | LLaMA-3.1-8B | CommonsenseQA | 8 | 64.4963 | 65.1515 | -0.6552 |
| 23 | 65 | LLaMA-3.1-8B | CommonsenseQA | 8 | 64.7625 | 65.4791 | -0.7166 |
| 24 | 66 | LLaMA-3.1-8B | CommonsenseQA | 8 | 64.6601 | 64.312 | 0.3481 |
| 25 | 67 | LLaMA-3.1-8B | CommonsenseQA | 8 | 64.2916 | 65.0287 | -0.7371 |
| 26 | 68 | LLaMA-3.1-8B | CommonsenseQA | 8 | 65.2129 | 65.2948 | -0.0819 |
| 27 | 69 | LLaMA-3.1-8B | CommonsenseQA | 8 | 65.172 | 64.7215 | 0.4505 |
| 28 | 70 | LLaMA-3.1-8B | CommonsenseQA | 8 | 65.3358 | 64.7215 | 0.6143 |
| 29 | 71 | LLaMA-3.1-8B | CommonsenseQA | 8 | 64.8649 | 65.561 | -0.6962 |
| 30 | 72 | LLaMA-3.1-8B | CommonsenseQA | 8 | 64.4144 | 64.9468 | -0.5324 |
| 31 | 73 | LLaMA-3.1-8B | CommonsenseQA | 8 | 64.8649 | 65.131 | -0.2662 |
| 32 | 74 | LLaMA-3.1-8B | CommonsenseQA | 8 | 65.3972 | 64.9877 | 0.4095 |
| 33 | 75 | LLaMA-3.1-8B | CommonsenseQA | 8 | 65.2948 | 65.8477 | -0.5528 |
| 34 | 76 | LLaMA-3.1-8B | CommonsenseQA | 8 | 64.9058 | 65.2744 | -0.3686 |
| 35 | 77 | LLaMA-3.1-8B | CommonsenseQA | 8 | 65.2948 | 65.2948 | 0 |
| 36 | 78 | LLaMA-3.1-8B | CommonsenseQA | 8 | 65.2334 | 65.3563 | -0.1229 |
| 37 | 79 | LLaMA-3.1-8B | CommonsenseQA | 8 | 65.0696 | 64.5577 | 0.5119 |
| 38 | 80 | LLaMA-3.1-8B | CommonsenseQA | 8 | 64.7625 | 65.1925 | -0.43 |
| 39 | 81 | LLaMA-3.1-8B | CommonsenseQA | 8 | 64.9468 | 64.5782 | 0.3686 |
| 40 | 82 | LLaMA-3.1-8B | CommonsenseQA | 8 | 64.312 | 65.4791 | -1.1671 |
| 41 | 83 | LLaMA-3.1-8B | CommonsenseQA | 8 | 65.8067 | 65.3358 | 0.4709 |
| 42 | 84 | LLaMA-3.1-8B | CommonsenseQA | 8 | 64.5577 | 65.0491 | -0.4914 |
| 43 | 85 | LLaMA-3.1-8B | CommonsenseQA | 8 | 64.353 | 64.7625 | -0.4095 |
| 44 | 86 | LLaMA-3.1-8B | CommonsenseQA | 8 | 64.1892 | 65.0901 | -0.9009 |
| 45 | 87 | LLaMA-3.1-8B | CommonsenseQA | 8 | 64.0459 | 65.6634 | -1.6175 |
| 46 | 88 | LLaMA-3.1-8B | CommonsenseQA | 8 | 65.7862 | 64.9672 | 0.819 |
| 47 | 89 | LLaMA-3.1-8B | CommonsenseQA | 8 | 65.2129 | 65.0082 | 0.2048 |
| 48 | 90 | LLaMA-3.1-8B | CommonsenseQA | 8 | 64.8444 | 65.1925 | -0.3481 |
| 49 | 91 | LLaMA-3.1-8B | CommonsenseQA | 8 | 64.8239 | 65.7248 | -0.9009 |
| 0 | 42 | LLaMA-3.1-8B | CommonsenseQA | 12 | 64.742 | 65.5474 | -0.8054 |
| 1 | 43 | LLaMA-3.1-8B | CommonsenseQA | 12 | 64.8512 | 65.1515 | -0.3003 |
| 2 | 44 | LLaMA-3.1-8B | CommonsenseQA | 12 | 64.6055 | 65.6429 | -1.0374 |
| 3 | 45 | LLaMA-3.1-8B | CommonsenseQA | 12 | 64.7284 | 65.0969 | -0.3686 |
| 4 | 46 | LLaMA-3.1-8B | CommonsenseQA | 12 | 64.3598 | 65.4928 | -1.133 |
| 5 | 47 | LLaMA-3.1-8B | CommonsenseQA | 12 | 64.6465 | 65.3153 | -0.6689 |
| 6 | 48 | LLaMA-3.1-8B | CommonsenseQA | 12 | 64.6192 | 65.2334 | -0.6143 |
| 7 | 49 | LLaMA-3.1-8B | CommonsenseQA | 12 | 64.5919 | 65.4245 | -0.8327 |
| 8 | 50 | LLaMA-3.1-8B | CommonsenseQA | 12 | 65.0287 | 65.2061 | -0.1775 |
| 9 | 51 | LLaMA-3.1-8B | CommonsenseQA | 12 | 64.3189 | 65.1652 | -0.8463 |
| 10 | 52 | LLaMA-3.1-8B | CommonsenseQA | 12 | 64.8649 | 65.2198 | -0.3549 |
| 11 | 53 | LLaMA-3.1-8B | CommonsenseQA | 12 | 64.742 | 65.2061 | -0.4641 |
| 12 | 54 | LLaMA-3.1-8B | CommonsenseQA | 12 | 65.0287 | 65.2061 | -0.1775 |
| 13 | 55 | LLaMA-3.1-8B | CommonsenseQA | 12 | 64.4827 | 65.3836 | -0.9009 |
| 14 | 56 | LLaMA-3.1-8B | CommonsenseQA | 12 | 64.9331 | 65.4382 | -0.5051 |
| 15 | 57 | LLaMA-3.1-8B | CommonsenseQA | 12 | 64.8103 | 65.0833 | -0.273 |
| 16 | 58 | LLaMA-3.1-8B | CommonsenseQA | 12 | 64.8785 | 65.056 | -0.1775 |
| 17 | 59 | LLaMA-3.1-8B | CommonsenseQA | 12 | 64.5373 | 65.4245 | -0.8873 |
| 18 | 60 | LLaMA-3.1-8B | CommonsenseQA | 12 | 64.9195 | 65.4518 | -0.5324 |
| 19 | 61 | LLaMA-3.1-8B | CommonsenseQA | 12 | 64.7966 | 65.3836 | -0.587 |
| 20 | 62 | LLaMA-3.1-8B | CommonsenseQA | 12 | 64.8239 | 65.6429 | -0.819 |
| 21 | 63 | LLaMA-3.1-8B | CommonsenseQA | 12 | 64.51 | 65.2607 | -0.7508 |
| 22 | 64 | LLaMA-3.1-8B | CommonsenseQA | 12 | 65.015 | 64.8785 | 0.1365 |
| 23 | 65 | LLaMA-3.1-8B | CommonsenseQA | 12 | 64.9058 | 65.1652 | -0.2594 |
| 24 | 66 | LLaMA-3.1-8B | CommonsenseQA | 12 | 64.6601 | 64.7557 | -0.0956 |
| 25 | 67 | LLaMA-3.1-8B | CommonsenseQA | 12 | 64.5919 | 65.2744 | -0.6825 |
| 26 | 68 | LLaMA-3.1-8B | CommonsenseQA | 12 | 65.3836 | 65.1242 | 0.2594 |
| 27 | 69 | LLaMA-3.1-8B | CommonsenseQA | 12 | 64.9058 | 65.1242 | -0.2184 |
| 28 | 70 | LLaMA-3.1-8B | CommonsenseQA | 12 | 64.6192 | 65.0423 | -0.4232 |
| 29 | 71 | LLaMA-3.1-8B | CommonsenseQA | 12 | 64.6192 | 65.9978 | -1.3787 |
| 30 | 72 | LLaMA-3.1-8B | CommonsenseQA | 12 | 64.5782 | 65.1925 | -0.6143 |
| 31 | 73 | LLaMA-3.1-8B | CommonsenseQA | 12 | 64.7966 | 65.1925 | -0.3959 |
| 32 | 74 | LLaMA-3.1-8B | CommonsenseQA | 12 | 64.9468 | 65.1106 | -0.1638 |
| 33 | 75 | LLaMA-3.1-8B | CommonsenseQA | 12 | 64.8376 | 65.9159 | -1.0784 |
| 34 | 76 | LLaMA-3.1-8B | CommonsenseQA | 12 | 64.9604 | 65.4245 | -0.4641 |
| 35 | 77 | LLaMA-3.1-8B | CommonsenseQA | 12 | 64.9331 | 65.1379 | -0.2048 |
| 36 | 78 | LLaMA-3.1-8B | CommonsenseQA | 12 | 64.3871 | 65.288 | -0.9009 |
| 37 | 79 | LLaMA-3.1-8B | CommonsenseQA | 12 | 64.9741 | 65.0696 | -0.0956 |
| 38 | 80 | LLaMA-3.1-8B | CommonsenseQA | 12 | 64.9741 | 65.2607 | -0.2867 |
| 39 | 81 | LLaMA-3.1-8B | CommonsenseQA | 12 | 64.3052 | 65.1925 | -0.8873 |
| 40 | 82 | LLaMA-3.1-8B | CommonsenseQA | 12 | 64.2506 | 65.3563 | -1.1057 |
| 41 | 83 | LLaMA-3.1-8B | CommonsenseQA | 12 | 65.2471 | 65.2471 | 0 |
| 42 | 84 | LLaMA-3.1-8B | CommonsenseQA | 12 | 64.8376 | 65.2471 | -0.4095 |
| 43 | 85 | LLaMA-3.1-8B | CommonsenseQA | 12 | 64.51 | 65.3563 | -0.8463 |
| 44 | 86 | LLaMA-3.1-8B | CommonsenseQA | 12 | 64.5236 | 65.2198 | -0.6962 |
| 45 | 87 | LLaMA-3.1-8B | CommonsenseQA | 12 | 64.8922 | 65.4109 | -0.5187 |
| 46 | 88 | LLaMA-3.1-8B | CommonsenseQA | 12 | 65.1242 | 65.2061 | -0.0819 |
| 47 | 89 | LLaMA-3.1-8B | CommonsenseQA | 12 | 64.8239 | 65.0833 | -0.2594 |
| 48 | 90 | LLaMA-3.1-8B | CommonsenseQA | 12 | 64.7966 | 65.2061 | -0.4095 |
| 49 | 91 | LLaMA-3.1-8B | CommonsenseQA | 12 | 64.9195 | 65.6293 | -0.7098 |
| 0 | 42 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 1 | 43 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 2 | 44 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 3 | 45 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 4 | 46 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 5 | 47 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 6 | 48 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 7 | 49 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 8 | 50 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 9 | 51 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 10 | 52 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 11 | 53 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 12 | 54 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 13 | 55 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 14 | 56 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 15 | 57 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 16 | 58 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 17 | 59 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 18 | 60 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 19 | 61 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 20 | 62 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 21 | 63 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 22 | 64 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 23 | 65 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 24 | 66 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 25 | 67 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 26 | 68 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 27 | 69 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 28 | 70 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 29 | 71 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 30 | 72 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 31 | 73 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 32 | 74 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 33 | 75 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 34 | 76 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 35 | 77 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 36 | 78 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 37 | 79 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 38 | 80 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 39 | 81 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 40 | 82 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 41 | 83 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 42 | 84 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 43 | 85 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 44 | 86 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 45 | 87 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 46 | 88 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 47 | 89 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 48 | 90 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 49 | 91 | LLaMA-3.1-8B | CommonsenseQA | 16 | 64.8137 | 65.2232 | -0.4095 |
| 0 | 42 | LLaMA-3.1-8B | GPQA | 4 | 38.2812 | 32.7009 | 5.5804 |
| 1 | 43 | LLaMA-3.1-8B | GPQA | 4 | 37.2768 | 33.4821 | 3.7946 |
| 2 | 44 | LLaMA-3.1-8B | GPQA | 4 | 36.2723 | 34.8214 | 1.4509 |
| 3 | 45 | LLaMA-3.1-8B | GPQA | 4 | 38.3929 | 33.3705 | 5.0223 |
| 4 | 46 | LLaMA-3.1-8B | GPQA | 4 | 38.6161 | 34.7098 | 3.9062 |
| 5 | 47 | LLaMA-3.1-8B | GPQA | 4 | 36.3839 | 33.7054 | 2.6786 |
| 6 | 48 | LLaMA-3.1-8B | GPQA | 4 | 37.1652 | 34.933 | 2.2321 |
| 7 | 49 | LLaMA-3.1-8B | GPQA | 4 | 35.3795 | 34.4866 | 0.8929 |
| 8 | 50 | LLaMA-3.1-8B | GPQA | 4 | 36.7188 | 34.8214 | 1.8973 |
| 9 | 51 | LLaMA-3.1-8B | GPQA | 4 | 37.7232 | 34.8214 | 2.9018 |
| 10 | 52 | LLaMA-3.1-8B | GPQA | 4 | 37.2768 | 32.3661 | 4.9107 |
| 11 | 53 | LLaMA-3.1-8B | GPQA | 4 | 37.8348 | 33.7054 | 4.1295 |
| 12 | 54 | LLaMA-3.1-8B | GPQA | 4 | 38.5045 | 34.1518 | 4.3527 |
| 13 | 55 | LLaMA-3.1-8B | GPQA | 4 | 36.6071 | 35.9375 | 0.6696 |
| 14 | 56 | LLaMA-3.1-8B | GPQA | 4 | 36.4955 | 34.933 | 1.5625 |
| 15 | 57 | LLaMA-3.1-8B | GPQA | 4 | 37.1652 | 33.0357 | 4.1295 |
| 16 | 58 | LLaMA-3.1-8B | GPQA | 4 | 39.1741 | 35.7143 | 3.4598 |
| 17 | 59 | LLaMA-3.1-8B | GPQA | 4 | 38.6161 | 32.7009 | 5.9152 |
| 18 | 60 | LLaMA-3.1-8B | GPQA | 4 | 37.0536 | 33.5938 | 3.4598 |
| 19 | 61 | LLaMA-3.1-8B | GPQA | 4 | 34.2634 | 33.5938 | 0.6696 |
| 20 | 62 | LLaMA-3.1-8B | GPQA | 4 | 37.3884 | 34.375 | 3.0134 |
| 21 | 63 | LLaMA-3.1-8B | GPQA | 4 | 35.7143 | 35.3795 | 0.3348 |
| 22 | 64 | LLaMA-3.1-8B | GPQA | 4 | 36.7188 | 33.2589 | 3.4598 |
| 23 | 65 | LLaMA-3.1-8B | GPQA | 4 | 38.2812 | 33.3705 | 4.9107 |
| 24 | 66 | LLaMA-3.1-8B | GPQA | 4 | 38.3929 | 36.0491 | 2.3438 |
| 25 | 67 | LLaMA-3.1-8B | GPQA | 4 | 38.5045 | 34.4866 | 4.0179 |
| 26 | 68 | LLaMA-3.1-8B | GPQA | 4 | 35.3795 | 33.0357 | 2.3438 |
| 27 | 69 | LLaMA-3.1-8B | GPQA | 4 | 38.6161 | 35.7143 | 2.9018 |
| 28 | 70 | LLaMA-3.1-8B | GPQA | 4 | 36.942 | 33.2589 | 3.683 |
| 29 | 71 | LLaMA-3.1-8B | GPQA | 4 | 36.8304 | 33.5938 | 3.2366 |
| 30 | 72 | LLaMA-3.1-8B | GPQA | 4 | 39.6205 | 33.817 | 5.8036 |
| 31 | 73 | LLaMA-3.1-8B | GPQA | 4 | 38.8393 | 30.1339 | 8.7054 |
| 32 | 74 | LLaMA-3.1-8B | GPQA | 4 | 36.4955 | 33.9286 | 2.567 |
| 33 | 75 | LLaMA-3.1-8B | GPQA | 4 | 35.2679 | 34.933 | 0.3348 |
| 34 | 76 | LLaMA-3.1-8B | GPQA | 4 | 36.6071 | 36.1607 | 0.4464 |
| 35 | 77 | LLaMA-3.1-8B | GPQA | 4 | 36.942 | 31.3616 | 5.5804 |
| 36 | 78 | LLaMA-3.1-8B | GPQA | 4 | 38.5045 | 32.5893 | 5.9152 |
| 37 | 79 | LLaMA-3.1-8B | GPQA | 4 | 37.1652 | 33.5938 | 3.5714 |
| 38 | 80 | LLaMA-3.1-8B | GPQA | 4 | 37.3884 | 33.0357 | 4.3527 |
| 39 | 81 | LLaMA-3.1-8B | GPQA | 4 | 35.6027 | 31.25 | 4.3527 |
| 40 | 82 | LLaMA-3.1-8B | GPQA | 4 | 34.2634 | 35.0446 | -0.7812 |
| 41 | 83 | LLaMA-3.1-8B | GPQA | 4 | 37.8348 | 33.4821 | 4.3527 |
| 42 | 84 | LLaMA-3.1-8B | GPQA | 4 | 37.2768 | 33.9286 | 3.3482 |
| 43 | 85 | LLaMA-3.1-8B | GPQA | 4 | 38.058 | 33.9286 | 4.1295 |
| 44 | 86 | LLaMA-3.1-8B | GPQA | 4 | 37.8348 | 33.0357 | 4.7991 |
| 45 | 87 | LLaMA-3.1-8B | GPQA | 4 | 38.2812 | 35.9375 | 2.3438 |
| 46 | 88 | LLaMA-3.1-8B | GPQA | 4 | 37.6116 | 34.0402 | 3.5714 |
| 47 | 89 | LLaMA-3.1-8B | GPQA | 4 | 34.8214 | 32.9241 | 1.8973 |
| 48 | 90 | LLaMA-3.1-8B | GPQA | 4 | 39.2857 | 31.5848 | 7.7009 |
| 49 | 91 | LLaMA-3.1-8B | GPQA | 4 | 35.3795 | 34.2634 | 1.1161 |
| 0 | 42 | LLaMA-3.1-8B | GPQA | 8 | 37.1094 | 33.2031 | 3.9062 |
| 1 | 43 | LLaMA-3.1-8B | GPQA | 8 | 37.3884 | 34.096 | 3.2924 |
| 2 | 44 | LLaMA-3.1-8B | GPQA | 8 | 36.7746 | 33.7054 | 3.0692 |
| 3 | 45 | LLaMA-3.1-8B | GPQA | 8 | 38.2812 | 33.4821 | 4.7991 |
| 4 | 46 | LLaMA-3.1-8B | GPQA | 8 | 37.1094 | 33.1473 | 3.9621 |
| 5 | 47 | LLaMA-3.1-8B | GPQA | 8 | 36.7746 | 33.3147 | 3.4598 |
| 6 | 48 | LLaMA-3.1-8B | GPQA | 8 | 36.0491 | 34.4308 | 1.6183 |
| 7 | 49 | LLaMA-3.1-8B | GPQA | 8 | 35.1562 | 33.3705 | 1.7857 |
| 8 | 50 | LLaMA-3.1-8B | GPQA | 8 | 37.3326 | 33.7054 | 3.6272 |
| 9 | 51 | LLaMA-3.1-8B | GPQA | 8 | 37.7232 | 33.8728 | 3.8504 |
| 10 | 52 | LLaMA-3.1-8B | GPQA | 8 | 36.8304 | 32.9799 | 3.8504 |
| 11 | 53 | LLaMA-3.1-8B | GPQA | 8 | 37.1652 | 34.5424 | 2.6228 |
| 12 | 54 | LLaMA-3.1-8B | GPQA | 8 | 39.0625 | 33.5379 | 5.5246 |
| 13 | 55 | LLaMA-3.1-8B | GPQA | 8 | 37.6116 | 34.0402 | 3.5714 |
| 14 | 56 | LLaMA-3.1-8B | GPQA | 8 | 37.2768 | 34.0402 | 3.2366 |
| 15 | 57 | LLaMA-3.1-8B | GPQA | 8 | 37.5558 | 33.7612 | 3.7946 |
| 16 | 58 | LLaMA-3.1-8B | GPQA | 8 | 38.2254 | 34.096 | 4.1295 |
| 17 | 59 | LLaMA-3.1-8B | GPQA | 8 | 37.2768 | 33.9286 | 3.3482 |
| 18 | 60 | LLaMA-3.1-8B | GPQA | 8 | 39.3973 | 33.2589 | 6.1384 |
| 19 | 61 | LLaMA-3.1-8B | GPQA | 8 | 36.4955 | 32.7009 | 3.7946 |
| 20 | 62 | LLaMA-3.1-8B | GPQA | 8 | 37.8906 | 33.817 | 4.0737 |
| 21 | 63 | LLaMA-3.1-8B | GPQA | 8 | 37.4442 | 33.5379 | 3.9062 |
| 22 | 64 | LLaMA-3.1-8B | GPQA | 8 | 37.2768 | 33.3147 | 3.9621 |
| 23 | 65 | LLaMA-3.1-8B | GPQA | 8 | 37.3884 | 33.0357 | 4.3527 |
| 24 | 66 | LLaMA-3.1-8B | GPQA | 8 | 37.5558 | 35.0446 | 2.5112 |
| 25 | 67 | LLaMA-3.1-8B | GPQA | 8 | 38.9509 | 33.5379 | 5.4129 |
| 26 | 68 | LLaMA-3.1-8B | GPQA | 8 | 35.4353 | 33.8728 | 1.5625 |
| 27 | 69 | LLaMA-3.1-8B | GPQA | 8 | 39.1741 | 33.3705 | 5.8036 |
| 28 | 70 | LLaMA-3.1-8B | GPQA | 8 | 37.2768 | 32.7009 | 4.5759 |
| 29 | 71 | LLaMA-3.1-8B | GPQA | 8 | 37.7232 | 32.9799 | 4.7433 |
| 30 | 72 | LLaMA-3.1-8B | GPQA | 8 | 37.2768 | 34.0402 | 3.2366 |
| 31 | 73 | LLaMA-3.1-8B | GPQA | 8 | 37.8348 | 32.3103 | 5.5246 |
| 32 | 74 | LLaMA-3.1-8B | GPQA | 8 | 37.221 | 33.2031 | 4.0179 |
| 33 | 75 | LLaMA-3.1-8B | GPQA | 8 | 37.7232 | 33.4821 | 4.2411 |
| 34 | 76 | LLaMA-3.1-8B | GPQA | 8 | 35.6585 | 34.5424 | 1.1161 |
| 35 | 77 | LLaMA-3.1-8B | GPQA | 8 | 37.779 | 32.3661 | 5.4129 |
| 36 | 78 | LLaMA-3.1-8B | GPQA | 8 | 38.3371 | 33.2031 | 5.1339 |
| 37 | 79 | LLaMA-3.1-8B | GPQA | 8 | 37.3326 | 32.7009 | 4.6317 |
| 38 | 80 | LLaMA-3.1-8B | GPQA | 8 | 38.3929 | 33.3705 | 5.0223 |
| 39 | 81 | LLaMA-3.1-8B | GPQA | 8 | 37.6674 | 33.3705 | 4.2969 |
| 40 | 82 | LLaMA-3.1-8B | GPQA | 8 | 37.221 | 33.8728 | 3.3482 |
| 41 | 83 | LLaMA-3.1-8B | GPQA | 8 | 38.0022 | 32.4777 | 5.5246 |
| 42 | 84 | LLaMA-3.1-8B | GPQA | 8 | 36.1049 | 33.1473 | 2.9576 |
| 43 | 85 | LLaMA-3.1-8B | GPQA | 8 | 38.2254 | 33.3147 | 4.9107 |
| 44 | 86 | LLaMA-3.1-8B | GPQA | 8 | 38.058 | 33.817 | 4.2411 |
| 45 | 87 | LLaMA-3.1-8B | GPQA | 8 | 37.779 | 33.9286 | 3.8504 |
| 46 | 88 | LLaMA-3.1-8B | GPQA | 8 | 36.8862 | 33.9844 | 2.9018 |
| 47 | 89 | LLaMA-3.1-8B | GPQA | 8 | 36.8304 | 34.654 | 2.1763 |
| 48 | 90 | LLaMA-3.1-8B | GPQA | 8 | 37.9464 | 33.3705 | 4.5759 |
| 49 | 91 | LLaMA-3.1-8B | GPQA | 8 | 36.3281 | 33.7612 | 2.567 |
| 0 | 42 | LLaMA-3.1-8B | GPQA | 12 | 37.2024 | 33.631 | 3.5714 |
| 1 | 43 | LLaMA-3.1-8B | GPQA | 12 | 37.7976 | 33.3333 | 4.4643 |
| 2 | 44 | LLaMA-3.1-8B | GPQA | 12 | 36.7188 | 34.0774 | 2.6414 |
| 3 | 45 | LLaMA-3.1-8B | GPQA | 12 | 37.872 | 33.7054 | 4.1667 |
| 4 | 46 | LLaMA-3.1-8B | GPQA | 12 | 37.4256 | 32.8869 | 4.5387 |
| 5 | 47 | LLaMA-3.1-8B | GPQA | 12 | 37.0908 | 33.0729 | 4.0179 |
| 6 | 48 | LLaMA-3.1-8B | GPQA | 12 | 36.756 | 33.9286 | 2.8274 |
| 7 | 49 | LLaMA-3.1-8B | GPQA | 12 | 37.0164 | 33.0729 | 3.9435 |
| 8 | 50 | LLaMA-3.1-8B | GPQA | 12 | 37.5 | 33.5565 | 3.9435 |
| 9 | 51 | LLaMA-3.1-8B | GPQA | 12 | 37.7976 | 33.5193 | 4.2783 |
| 10 | 52 | LLaMA-3.1-8B | GPQA | 12 | 36.756 | 34.003 | 2.753 |
| 11 | 53 | LLaMA-3.1-8B | GPQA | 12 | 37.0908 | 33.7054 | 3.3854 |
| 12 | 54 | LLaMA-3.1-8B | GPQA | 12 | 37.7604 | 33.4449 | 4.3155 |
| 13 | 55 | LLaMA-3.1-8B | GPQA | 12 | 37.2396 | 33.5565 | 3.683 |
| 14 | 56 | LLaMA-3.1-8B | GPQA | 12 | 37.0908 | 34.5982 | 2.4926 |
| 15 | 57 | LLaMA-3.1-8B | GPQA | 12 | 37.7604 | 33.6682 | 4.0923 |
| 16 | 58 | LLaMA-3.1-8B | GPQA | 12 | 37.3512 | 33.7798 | 3.5714 |
| 17 | 59 | LLaMA-3.1-8B | GPQA | 12 | 37.8348 | 34.0402 | 3.7946 |
| 18 | 60 | LLaMA-3.1-8B | GPQA | 12 | 38.2068 | 33.5565 | 4.6503 |
| 19 | 61 | LLaMA-3.1-8B | GPQA | 12 | 37.4628 | 32.7009 | 4.7619 |
| 20 | 62 | LLaMA-3.1-8B | GPQA | 12 | 37.7976 | 33.4077 | 4.3899 |
| 21 | 63 | LLaMA-3.1-8B | GPQA | 12 | 37.0164 | 34.003 | 3.0134 |
| 22 | 64 | LLaMA-3.1-8B | GPQA | 12 | 37.1652 | 33.8542 | 3.311 |
| 23 | 65 | LLaMA-3.1-8B | GPQA | 12 | 37.7232 | 33.2217 | 4.5015 |
| 24 | 66 | LLaMA-3.1-8B | GPQA | 12 | 37.6116 | 34.003 | 3.6086 |
| 25 | 67 | LLaMA-3.1-8B | GPQA | 12 | 37.7976 | 33.631 | 4.1667 |
| 26 | 68 | LLaMA-3.1-8B | GPQA | 12 | 36.8676 | 33.5938 | 3.2738 |
| 27 | 69 | LLaMA-3.1-8B | GPQA | 12 | 38.2812 | 33.3705 | 4.9107 |
| 28 | 70 | LLaMA-3.1-8B | GPQA | 12 | 37.4628 | 32.1801 | 5.2827 |
| 29 | 71 | LLaMA-3.1-8B | GPQA | 12 | 37.686 | 32.7753 | 4.9107 |
| 30 | 72 | LLaMA-3.1-8B | GPQA | 12 | 37.1652 | 33.817 | 3.3482 |
| 31 | 73 | LLaMA-3.1-8B | GPQA | 12 | 38.2812 | 32.5893 | 5.692 |
| 32 | 74 | LLaMA-3.1-8B | GPQA | 12 | 37.6116 | 33.4077 | 4.2039 |
| 33 | 75 | LLaMA-3.1-8B | GPQA | 12 | 38.0208 | 33.1473 | 4.8735 |
| 34 | 76 | LLaMA-3.1-8B | GPQA | 12 | 37.5 | 33.7054 | 3.7946 |
| 35 | 77 | LLaMA-3.1-8B | GPQA | 12 | 38.244 | 32.7381 | 5.506 |
| 36 | 78 | LLaMA-3.1-8B | GPQA | 12 | 37.314 | 33.7054 | 3.6086 |
| 37 | 79 | LLaMA-3.1-8B | GPQA | 12 | 37.2396 | 32.9985 | 4.2411 |
| 38 | 80 | LLaMA-3.1-8B | GPQA | 12 | 38.058 | 33.1473 | 4.9107 |
| 39 | 81 | LLaMA-3.1-8B | GPQA | 12 | 37.6116 | 33.3705 | 4.2411 |
| 40 | 82 | LLaMA-3.1-8B | GPQA | 12 | 37.7604 | 33.1473 | 4.6131 |
| 41 | 83 | LLaMA-3.1-8B | GPQA | 12 | 37.0908 | 32.6265 | 4.4643 |
| 42 | 84 | LLaMA-3.1-8B | GPQA | 12 | 36.942 | 33.817 | 3.125 |
| 43 | 85 | LLaMA-3.1-8B | GPQA | 12 | 37.5372 | 33.2589 | 4.2783 |
| 44 | 86 | LLaMA-3.1-8B | GPQA | 12 | 38.058 | 34.003 | 4.0551 |
| 45 | 87 | LLaMA-3.1-8B | GPQA | 12 | 37.7604 | 34.1518 | 3.6086 |
| 46 | 88 | LLaMA-3.1-8B | GPQA | 12 | 37.2024 | 34.3378 | 2.8646 |
| 47 | 89 | LLaMA-3.1-8B | GPQA | 12 | 37.5 | 33.7798 | 3.7202 |
| 48 | 90 | LLaMA-3.1-8B | GPQA | 12 | 38.1324 | 33.9286 | 4.2039 |
| 49 | 91 | LLaMA-3.1-8B | GPQA | 12 | 36.4583 | 33.5565 | 2.9018 |
| 0 | 42 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 1 | 43 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 2 | 44 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 3 | 45 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 4 | 46 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 5 | 47 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 6 | 48 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 7 | 49 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 8 | 50 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 9 | 51 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 10 | 52 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 11 | 53 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 12 | 54 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 13 | 55 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 14 | 56 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 15 | 57 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 16 | 58 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 17 | 59 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 18 | 60 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 19 | 61 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 20 | 62 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 21 | 63 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 22 | 64 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 23 | 65 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 24 | 66 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 25 | 67 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 26 | 68 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 27 | 69 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 28 | 70 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 29 | 71 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 30 | 72 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 31 | 73 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 32 | 74 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 33 | 75 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 34 | 76 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 35 | 77 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 36 | 78 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 37 | 79 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 38 | 80 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 39 | 81 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 40 | 82 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 41 | 83 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 42 | 84 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 43 | 85 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 44 | 86 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 45 | 87 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 46 | 88 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 47 | 89 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 48 | 90 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 49 | 91 | LLaMA-3.1-8B | GPQA | 16 | 37.5837 | 33.3984 | 4.1853 |
| 0 | 42 | LLaMA-3.1-8B | GSM8K | 4 | 76.8006 | 74.7536 | 2.047 |
| 1 | 43 | LLaMA-3.1-8B | GSM8K | 4 | 75.8908 | 74.7536 | 1.1372 |
| 2 | 44 | LLaMA-3.1-8B | GSM8K | 4 | 75.7771 | 75.5118 | 0.2654 |
| 3 | 45 | LLaMA-3.1-8B | GSM8K | 4 | 75.7013 | 73.6922 | 2.0091 |
| 4 | 46 | LLaMA-3.1-8B | GSM8K | 4 | 75.7392 | 74.3366 | 1.4026 |
| 5 | 47 | LLaMA-3.1-8B | GSM8K | 4 | 75.7771 | 74.8673 | 0.9098 |
| 6 | 48 | LLaMA-3.1-8B | GSM8K | 4 | 75.7392 | 74.9431 | 0.7961 |
| 7 | 49 | LLaMA-3.1-8B | GSM8K | 4 | 76.0425 | 74.8673 | 1.1751 |
| 8 | 50 | LLaMA-3.1-8B | GSM8K | 4 | 75.0948 | 74.4124 | 0.6823 |
| 9 | 51 | LLaMA-3.1-8B | GSM8K | 4 | 76.1562 | 74.7536 | 1.4026 |
| 10 | 52 | LLaMA-3.1-8B | GSM8K | 4 | 75.4359 | 74.6778 | 0.7582 |
| 11 | 53 | LLaMA-3.1-8B | GSM8K | 4 | 76.1941 | 75.6634 | 0.5307 |
| 12 | 54 | LLaMA-3.1-8B | GSM8K | 4 | 76.3078 | 74.602 | 1.7058 |
| 13 | 55 | LLaMA-3.1-8B | GSM8K | 4 | 76.0804 | 73.9955 | 2.0849 |
| 14 | 56 | LLaMA-3.1-8B | GSM8K | 4 | 75.1706 | 74.5262 | 0.6444 |
| 15 | 57 | LLaMA-3.1-8B | GSM8K | 4 | 75.5497 | 73.5406 | 2.0091 |
| 16 | 58 | LLaMA-3.1-8B | GSM8K | 4 | 75.7771 | 75.0569 | 0.7202 |
| 17 | 59 | LLaMA-3.1-8B | GSM8K | 4 | 75.1327 | 75.019 | 0.1137 |
| 18 | 60 | LLaMA-3.1-8B | GSM8K | 4 | 76.3457 | 74.0713 | 2.2745 |
| 19 | 61 | LLaMA-3.1-8B | GSM8K | 4 | 75.9287 | 74.6778 | 1.2509 |
| 20 | 62 | LLaMA-3.1-8B | GSM8K | 4 | 76.3078 | 73.8059 | 2.5019 |
| 21 | 63 | LLaMA-3.1-8B | GSM8K | 4 | 76.649 | 75.398 | 1.2509 |
| 22 | 64 | LLaMA-3.1-8B | GSM8K | 4 | 76.3836 | 73.9575 | 2.4261 |
| 23 | 65 | LLaMA-3.1-8B | GSM8K | 4 | 75.5118 | 74.0713 | 1.4405 |
| 24 | 66 | LLaMA-3.1-8B | GSM8K | 4 | 76.2699 | 74.5641 | 1.7058 |
| 25 | 67 | LLaMA-3.1-8B | GSM8K | 4 | 75.8529 | 74.9052 | 0.9477 |
| 26 | 68 | LLaMA-3.1-8B | GSM8K | 4 | 76.1941 | 74.7157 | 1.4784 |
| 27 | 69 | LLaMA-3.1-8B | GSM8K | 4 | 75.8908 | 74.7536 | 1.1372 |
| 28 | 70 | LLaMA-3.1-8B | GSM8K | 4 | 75.5876 | 74.7915 | 0.7961 |
| 29 | 71 | LLaMA-3.1-8B | GSM8K | 4 | 76.4973 | 74.981 | 1.5163 |
| 30 | 72 | LLaMA-3.1-8B | GSM8K | 4 | 76.3457 | 74.5641 | 1.7817 |
| 31 | 73 | LLaMA-3.1-8B | GSM8K | 4 | 76.0045 | 75.2843 | 0.7202 |
| 32 | 74 | LLaMA-3.1-8B | GSM8K | 4 | 75.3222 | 74.7536 | 0.5686 |
| 33 | 75 | LLaMA-3.1-8B | GSM8K | 4 | 75.8529 | 74.981 | 0.8719 |
| 34 | 76 | LLaMA-3.1-8B | GSM8K | 4 | 76.1941 | 74.6399 | 1.5542 |
| 35 | 77 | LLaMA-3.1-8B | GSM8K | 4 | 76.232 | 75.9666 | 0.2654 |
| 36 | 78 | LLaMA-3.1-8B | GSM8K | 4 | 77.1039 | 75.7392 | 1.3647 |
| 37 | 79 | LLaMA-3.1-8B | GSM8K | 4 | 75.9666 | 75.5118 | 0.4549 |
| 38 | 80 | LLaMA-3.1-8B | GSM8K | 4 | 75.3601 | 75.2843 | 0.0758 |
| 39 | 81 | LLaMA-3.1-8B | GSM8K | 4 | 75.4359 | 74.6778 | 0.7582 |
| 40 | 82 | LLaMA-3.1-8B | GSM8K | 4 | 76.0045 | 75.0569 | 0.9477 |
| 41 | 83 | LLaMA-3.1-8B | GSM8K | 4 | 76.649 | 75.2464 | 1.4026 |
| 42 | 84 | LLaMA-3.1-8B | GSM8K | 4 | 75.6634 | 74.4882 | 1.1751 |
| 43 | 85 | LLaMA-3.1-8B | GSM8K | 4 | 76.3078 | 75.019 | 1.2889 |
| 44 | 86 | LLaMA-3.1-8B | GSM8K | 4 | 75.0948 | 74.1471 | 0.9477 |
| 45 | 87 | LLaMA-3.1-8B | GSM8K | 4 | 76.9143 | 74.6399 | 2.2745 |
| 46 | 88 | LLaMA-3.1-8B | GSM8K | 4 | 76.0045 | 74.8673 | 1.1372 |
| 47 | 89 | LLaMA-3.1-8B | GSM8K | 4 | 76.9901 | 76.3078 | 0.6823 |
| 48 | 90 | LLaMA-3.1-8B | GSM8K | 4 | 75.019 | 75.5118 | -0.4928 |
| 49 | 91 | LLaMA-3.1-8B | GSM8K | 4 | 76.649 | 75.0569 | 1.5921 |
| 0 | 42 | LLaMA-3.1-8B | GSM8K | 8 | 76.2889 | 74.7536 | 1.5353 |
| 1 | 43 | LLaMA-3.1-8B | GSM8K | 8 | 75.9856 | 74.8484 | 1.1372 |
| 2 | 44 | LLaMA-3.1-8B | GSM8K | 8 | 75.4928 | 74.602 | 0.8908 |
| 3 | 45 | LLaMA-3.1-8B | GSM8K | 8 | 75.7013 | 74.5451 | 1.1562 |
| 4 | 46 | LLaMA-3.1-8B | GSM8K | 8 | 76.0425 | 74.981 | 1.0614 |
| 5 | 47 | LLaMA-3.1-8B | GSM8K | 8 | 75.5307 | 74.4124 | 1.1183 |
| 6 | 48 | LLaMA-3.1-8B | GSM8K | 8 | 76.1372 | 75.019 | 1.1183 |
| 7 | 49 | LLaMA-3.1-8B | GSM8K | 8 | 75.6065 | 74.9621 | 0.6444 |
| 8 | 50 | LLaMA-3.1-8B | GSM8K | 8 | 76.0993 | 74.7157 | 1.3836 |
| 9 | 51 | LLaMA-3.1-8B | GSM8K | 8 | 75.7202 | 74.6967 | 1.0235 |
| 10 | 52 | LLaMA-3.1-8B | GSM8K | 8 | 75.7202 | 74.602 | 1.1183 |
| 11 | 53 | LLaMA-3.1-8B | GSM8K | 8 | 76.4215 | 74.8863 | 1.5353 |
| 12 | 54 | LLaMA-3.1-8B | GSM8K | 8 | 75.9856 | 75.2274 | 0.7582 |
| 13 | 55 | LLaMA-3.1-8B | GSM8K | 8 | 75.7013 | 74.185 | 1.5163 |
| 14 | 56 | LLaMA-3.1-8B | GSM8K | 8 | 76.0425 | 74.7915 | 1.2509 |
| 15 | 57 | LLaMA-3.1-8B | GSM8K | 8 | 76.0235 | 74.7726 | 1.2509 |
| 16 | 58 | LLaMA-3.1-8B | GSM8K | 8 | 75.8908 | 75.1137 | 0.7771 |
| 17 | 59 | LLaMA-3.1-8B | GSM8K | 8 | 75.7013 | 74.8673 | 0.834 |
| 18 | 60 | LLaMA-3.1-8B | GSM8K | 8 | 75.2843 | 74.602 | 0.6823 |
| 19 | 61 | LLaMA-3.1-8B | GSM8K | 8 | 75.834 | 74.6967 | 1.1372 |
| 20 | 62 | LLaMA-3.1-8B | GSM8K | 8 | 76.1372 | 74.5451 | 1.5921 |
| 21 | 63 | LLaMA-3.1-8B | GSM8K | 8 | 75.6823 | 75.3601 | 0.3222 |
| 22 | 64 | LLaMA-3.1-8B | GSM8K | 8 | 76.0614 | 74.7346 | 1.3268 |
| 23 | 65 | LLaMA-3.1-8B | GSM8K | 8 | 75.5497 | 74.4882 | 1.0614 |
| 24 | 66 | LLaMA-3.1-8B | GSM8K | 8 | 75.7771 | 74.4693 | 1.3078 |
| 25 | 67 | LLaMA-3.1-8B | GSM8K | 8 | 75.9666 | 74.7536 | 1.213 |
| 26 | 68 | LLaMA-3.1-8B | GSM8K | 8 | 76.3078 | 74.8105 | 1.4973 |
| 27 | 69 | LLaMA-3.1-8B | GSM8K | 8 | 75.7961 | 75.0948 | 0.7013 |
| 28 | 70 | LLaMA-3.1-8B | GSM8K | 8 | 75.9098 | 74.8105 | 1.0993 |
| 29 | 71 | LLaMA-3.1-8B | GSM8K | 8 | 75.6634 | 75.5876 | 0.0758 |
| 30 | 72 | LLaMA-3.1-8B | GSM8K | 8 | 75.5876 | 75 | 0.5876 |
| 31 | 73 | LLaMA-3.1-8B | GSM8K | 8 | 76.1751 | 75.1895 | 0.9856 |
| 32 | 74 | LLaMA-3.1-8B | GSM8K | 8 | 75.4738 | 74.981 | 0.4928 |
| 33 | 75 | LLaMA-3.1-8B | GSM8K | 8 | 76.0045 | 74.7536 | 1.2509 |
| 34 | 76 | LLaMA-3.1-8B | GSM8K | 8 | 75.815 | 74.8294 | 0.9856 |
| 35 | 77 | LLaMA-3.1-8B | GSM8K | 8 | 76.2889 | 75.0948 | 1.1941 |
| 36 | 78 | LLaMA-3.1-8B | GSM8K | 8 | 76.1372 | 75.1516 | 0.9856 |
| 37 | 79 | LLaMA-3.1-8B | GSM8K | 8 | 75.1516 | 74.7915 | 0.3601 |
| 38 | 80 | LLaMA-3.1-8B | GSM8K | 8 | 76.2889 | 75.1137 | 1.1751 |
| 39 | 81 | LLaMA-3.1-8B | GSM8K | 8 | 75.7771 | 74.3935 | 1.3836 |
| 40 | 82 | LLaMA-3.1-8B | GSM8K | 8 | 76.6111 | 74.4124 | 2.1986 |
| 41 | 83 | LLaMA-3.1-8B | GSM8K | 8 | 76.3457 | 74.7536 | 1.5921 |
| 42 | 84 | LLaMA-3.1-8B | GSM8K | 8 | 76.0235 | 74.8673 | 1.1562 |
| 43 | 85 | LLaMA-3.1-8B | GSM8K | 8 | 76.3078 | 74.8863 | 1.4215 |
| 44 | 86 | LLaMA-3.1-8B | GSM8K | 8 | 75.5686 | 75.0569 | 0.5118 |
| 45 | 87 | LLaMA-3.1-8B | GSM8K | 8 | 75.4928 | 74.4882 | 1.0045 |
| 46 | 88 | LLaMA-3.1-8B | GSM8K | 8 | 76.1183 | 75.3601 | 0.7582 |
| 47 | 89 | LLaMA-3.1-8B | GSM8K | 8 | 75.8719 | 75.5497 | 0.3222 |
| 48 | 90 | LLaMA-3.1-8B | GSM8K | 8 | 75.3601 | 75.1137 | 0.2464 |
| 49 | 91 | LLaMA-3.1-8B | GSM8K | 8 | 76.213 | 75.3412 | 0.8719 |
| 0 | 42 | LLaMA-3.1-8B | GSM8K | 12 | 75.7392 | 74.8926 | 0.8466 |
| 1 | 43 | LLaMA-3.1-8B | GSM8K | 12 | 75.8529 | 75.1074 | 0.7455 |
| 2 | 44 | LLaMA-3.1-8B | GSM8K | 12 | 75.5497 | 74.9937 | 0.556 |
| 3 | 45 | LLaMA-3.1-8B | GSM8K | 12 | 75.5497 | 74.9684 | 0.5812 |
| 4 | 46 | LLaMA-3.1-8B | GSM8K | 12 | 75.5497 | 75.0695 | 0.4802 |
| 5 | 47 | LLaMA-3.1-8B | GSM8K | 12 | 75.9414 | 74.3113 | 1.63 |
| 6 | 48 | LLaMA-3.1-8B | GSM8K | 12 | 75.9035 | 75.0442 | 0.8592 |
| 7 | 49 | LLaMA-3.1-8B | GSM8K | 12 | 76.0045 | 74.9431 | 1.0614 |
| 8 | 50 | LLaMA-3.1-8B | GSM8K | 12 | 75.7392 | 74.9179 | 0.8213 |
| 9 | 51 | LLaMA-3.1-8B | GSM8K | 12 | 75.9919 | 74.8421 | 1.1499 |
| 10 | 52 | LLaMA-3.1-8B | GSM8K | 12 | 75.2969 | 74.6652 | 0.6318 |
| 11 | 53 | LLaMA-3.1-8B | GSM8K | 12 | 75.8656 | 74.8294 | 1.0361 |
| 12 | 54 | LLaMA-3.1-8B | GSM8K | 12 | 75.8276 | 74.9179 | 0.9098 |
| 13 | 55 | LLaMA-3.1-8B | GSM8K | 12 | 75.6128 | 74.5514 | 1.0614 |
| 14 | 56 | LLaMA-3.1-8B | GSM8K | 12 | 75.6634 | 74.8673 | 0.7961 |
| 15 | 57 | LLaMA-3.1-8B | GSM8K | 12 | 75.9414 | 75.0063 | 0.9351 |
| 16 | 58 | LLaMA-3.1-8B | GSM8K | 12 | 75.4865 | 74.8421 | 0.6444 |
| 17 | 59 | LLaMA-3.1-8B | GSM8K | 12 | 75.6002 | 74.8168 | 0.7834 |
| 18 | 60 | LLaMA-3.1-8B | GSM8K | 12 | 75.7139 | 74.88 | 0.834 |
| 19 | 61 | LLaMA-3.1-8B | GSM8K | 12 | 75.7013 | 75.1579 | 0.5433 |
| 20 | 62 | LLaMA-3.1-8B | GSM8K | 12 | 75.7897 | 74.6904 | 1.0993 |
| 21 | 63 | LLaMA-3.1-8B | GSM8K | 12 | 76.0298 | 75.2338 | 0.7961 |
| 22 | 64 | LLaMA-3.1-8B | GSM8K | 12 | 75.9666 | 75.12 | 0.8466 |
| 23 | 65 | LLaMA-3.1-8B | GSM8K | 12 | 75.6634 | 74.9431 | 0.7202 |
| 24 | 66 | LLaMA-3.1-8B | GSM8K | 12 | 75.815 | 74.5641 | 1.2509 |
| 25 | 67 | LLaMA-3.1-8B | GSM8K | 12 | 75.815 | 75.0316 | 0.7834 |
| 26 | 68 | LLaMA-3.1-8B | GSM8K | 12 | 76.1815 | 74.7662 | 1.4152 |
| 27 | 69 | LLaMA-3.1-8B | GSM8K | 12 | 75.8529 | 74.9431 | 0.9098 |
| 28 | 70 | LLaMA-3.1-8B | GSM8K | 12 | 76.1183 | 74.7536 | 1.3647 |
| 29 | 71 | LLaMA-3.1-8B | GSM8K | 12 | 75.7897 | 74.8673 | 0.9224 |
| 30 | 72 | LLaMA-3.1-8B | GSM8K | 12 | 75.3222 | 74.981 | 0.3412 |
| 31 | 73 | LLaMA-3.1-8B | GSM8K | 12 | 75.954 | 75.3854 | 0.5686 |
| 32 | 74 | LLaMA-3.1-8B | GSM8K | 12 | 75.7013 | 74.8673 | 0.834 |
| 33 | 75 | LLaMA-3.1-8B | GSM8K | 12 | 75.4865 | 74.5514 | 0.9351 |
| 34 | 76 | LLaMA-3.1-8B | GSM8K | 12 | 75.4612 | 75.0821 | 0.3791 |
| 35 | 77 | LLaMA-3.1-8B | GSM8K | 12 | 75.7645 | 75.2211 | 0.5433 |
| 36 | 78 | LLaMA-3.1-8B | GSM8K | 12 | 75.9161 | 74.9558 | 0.9603 |
| 37 | 79 | LLaMA-3.1-8B | GSM8K | 12 | 75.7645 | 75.0695 | 0.695 |
| 38 | 80 | LLaMA-3.1-8B | GSM8K | 12 | 76.0677 | 74.9937 | 1.074 |
| 39 | 81 | LLaMA-3.1-8B | GSM8K | 12 | 75.815 | 74.7536 | 1.0614 |
| 40 | 82 | LLaMA-3.1-8B | GSM8K | 12 | 75.9035 | 74.602 | 1.3015 |
| 41 | 83 | LLaMA-3.1-8B | GSM8K | 12 | 76.1183 | 74.9305 | 1.1878 |
| 42 | 84 | LLaMA-3.1-8B | GSM8K | 12 | 75.6002 | 74.88 | 0.7202 |
| 43 | 85 | LLaMA-3.1-8B | GSM8K | 12 | 75.9161 | 75.0569 | 0.8592 |
| 44 | 86 | LLaMA-3.1-8B | GSM8K | 12 | 75.9414 | 74.7536 | 1.1878 |
| 45 | 87 | LLaMA-3.1-8B | GSM8K | 12 | 75.7266 | 74.7662 | 0.9603 |
| 46 | 88 | LLaMA-3.1-8B | GSM8K | 12 | 75.4865 | 74.9305 | 0.556 |
| 47 | 89 | LLaMA-3.1-8B | GSM8K | 12 | 75.9919 | 75.4738 | 0.5181 |
| 48 | 90 | LLaMA-3.1-8B | GSM8K | 12 | 75.5118 | 74.8673 | 0.6444 |
| 49 | 91 | LLaMA-3.1-8B | GSM8K | 12 | 75.8024 | 74.9684 | 0.834 |
| 0 | 42 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 1 | 43 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 2 | 44 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 3 | 45 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 4 | 46 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 5 | 47 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 6 | 48 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 7 | 49 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 8 | 50 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 9 | 51 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 10 | 52 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 11 | 53 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 12 | 54 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 13 | 55 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 14 | 56 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 15 | 57 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 16 | 58 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 17 | 59 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 18 | 60 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 19 | 61 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 20 | 62 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 21 | 63 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 22 | 64 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 23 | 65 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 24 | 66 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 25 | 67 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 26 | 68 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 27 | 69 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 28 | 70 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 29 | 71 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 30 | 72 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 31 | 73 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 32 | 74 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 33 | 75 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 34 | 76 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 35 | 77 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 36 | 78 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 37 | 79 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 38 | 80 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 39 | 81 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 40 | 82 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 41 | 83 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 42 | 84 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 43 | 85 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 44 | 86 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 45 | 87 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 46 | 88 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 47 | 89 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 48 | 90 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 49 | 91 | LLaMA-3.1-8B | GSM8K | 16 | 75.8434 | 74.8199 | 1.0235 |
| 0 | 42 | LLaMA-3.1-8B | MATH500 | 4 | 34.1 | 35.2 | -1.1 |
| 1 | 43 | LLaMA-3.1-8B | MATH500 | 4 | 35.4 | 35.3 | 0.1 |
| 2 | 44 | LLaMA-3.1-8B | MATH500 | 4 | 32.8 | 35.2 | -2.4 |
| 3 | 45 | LLaMA-3.1-8B | MATH500 | 4 | 34.8 | 36.7 | -1.9 |
| 4 | 46 | LLaMA-3.1-8B | MATH500 | 4 | 35.6 | 35.8 | -0.2 |
| 5 | 47 | LLaMA-3.1-8B | MATH500 | 4 | 35.7 | 36.2 | -0.5 |
| 6 | 48 | LLaMA-3.1-8B | MATH500 | 4 | 34.3 | 38.3 | -4 |
| 7 | 49 | LLaMA-3.1-8B | MATH500 | 4 | 34.4 | 33.7 | 0.7 |
| 8 | 50 | LLaMA-3.1-8B | MATH500 | 4 | 34.3 | 35.8 | -1.5 |
| 9 | 51 | LLaMA-3.1-8B | MATH500 | 4 | 33.2 | 36.2 | -3 |
| 10 | 52 | LLaMA-3.1-8B | MATH500 | 4 | 34.7 | 36 | -1.3 |
| 11 | 53 | LLaMA-3.1-8B | MATH500 | 4 | 33.1 | 35.2 | -2.1 |
| 12 | 54 | LLaMA-3.1-8B | MATH500 | 4 | 34.3 | 36.1 | -1.8 |
| 13 | 55 | LLaMA-3.1-8B | MATH500 | 4 | 33.3 | 34.6 | -1.3 |
| 14 | 56 | LLaMA-3.1-8B | MATH500 | 4 | 35.5 | 36 | -0.5 |
| 15 | 57 | LLaMA-3.1-8B | MATH500 | 4 | 33.9 | 33.4 | 0.5 |
| 16 | 58 | LLaMA-3.1-8B | MATH500 | 4 | 34.8 | 36.4 | -1.6 |
| 17 | 59 | LLaMA-3.1-8B | MATH500 | 4 | 33.3 | 34.7 | -1.4 |
| 18 | 60 | LLaMA-3.1-8B | MATH500 | 4 | 34 | 35.4 | -1.4 |
| 19 | 61 | LLaMA-3.1-8B | MATH500 | 4 | 34.5 | 36.1 | -1.6 |
| 20 | 62 | LLaMA-3.1-8B | MATH500 | 4 | 34.9 | 35.3 | -0.4 |
| 21 | 63 | LLaMA-3.1-8B | MATH500 | 4 | 37.2 | 34.8 | 2.4 |
| 22 | 64 | LLaMA-3.1-8B | MATH500 | 4 | 35 | 35.2 | -0.2 |
| 23 | 65 | LLaMA-3.1-8B | MATH500 | 4 | 32.7 | 35.5 | -2.8 |
| 24 | 66 | LLaMA-3.1-8B | MATH500 | 4 | 33.4 | 34.8 | -1.4 |
| 25 | 67 | LLaMA-3.1-8B | MATH500 | 4 | 33.8 | 35 | -1.2 |
| 26 | 68 | LLaMA-3.1-8B | MATH500 | 4 | 33.7 | 34.1 | -0.4 |
| 27 | 69 | LLaMA-3.1-8B | MATH500 | 4 | 32.5 | 35.2 | -2.7 |
| 28 | 70 | LLaMA-3.1-8B | MATH500 | 4 | 35.1 | 35.9 | -0.8 |
| 29 | 71 | LLaMA-3.1-8B | MATH500 | 4 | 33.7 | 35.4 | -1.7 |
| 30 | 72 | LLaMA-3.1-8B | MATH500 | 4 | 34.1 | 33.4 | 0.7 |
| 31 | 73 | LLaMA-3.1-8B | MATH500 | 4 | 33.2 | 36 | -2.8 |
| 32 | 74 | LLaMA-3.1-8B | MATH500 | 4 | 34.7 | 35.2 | -0.5 |
| 33 | 75 | LLaMA-3.1-8B | MATH500 | 4 | 33.2 | 34.8 | -1.6 |
| 34 | 76 | LLaMA-3.1-8B | MATH500 | 4 | 35.8 | 35 | 0.8 |
| 35 | 77 | LLaMA-3.1-8B | MATH500 | 4 | 34 | 35.2 | -1.2 |
| 36 | 78 | LLaMA-3.1-8B | MATH500 | 4 | 35.4 | 36.6 | -1.2 |
| 37 | 79 | LLaMA-3.1-8B | MATH500 | 4 | 35.7 | 37.9 | -2.2 |
| 38 | 80 | LLaMA-3.1-8B | MATH500 | 4 | 33.8 | 35.9 | -2.1 |
| 39 | 81 | LLaMA-3.1-8B | MATH500 | 4 | 35.1 | 35.2 | -0.1 |
| 40 | 82 | LLaMA-3.1-8B | MATH500 | 4 | 34.4 | 34.5 | -0.1 |
| 41 | 83 | LLaMA-3.1-8B | MATH500 | 4 | 35.5 | 34.6 | 0.9 |
| 42 | 84 | LLaMA-3.1-8B | MATH500 | 4 | 35 | 36.6 | -1.6 |
| 43 | 85 | LLaMA-3.1-8B | MATH500 | 4 | 33.8 | 34.2 | -0.4 |
| 44 | 86 | LLaMA-3.1-8B | MATH500 | 4 | 35 | 37.2 | -2.2 |
| 45 | 87 | LLaMA-3.1-8B | MATH500 | 4 | 34.7 | 33.4 | 1.3 |
| 46 | 88 | LLaMA-3.1-8B | MATH500 | 4 | 38.1 | 33.5 | 4.6 |
| 47 | 89 | LLaMA-3.1-8B | MATH500 | 4 | 34.7 | 35.3 | -0.6 |
| 48 | 90 | LLaMA-3.1-8B | MATH500 | 4 | 37 | 35.8 | 1.2 |
| 49 | 91 | LLaMA-3.1-8B | MATH500 | 4 | 32 | 33.9 | -1.9 |
| 0 | 42 | LLaMA-3.1-8B | MATH500 | 8 | 34.75 | 35.3 | -0.55 |
| 1 | 43 | LLaMA-3.1-8B | MATH500 | 8 | 33.85 | 35.5 | -1.65 |
| 2 | 44 | LLaMA-3.1-8B | MATH500 | 8 | 33.1 | 36.2 | -3.1 |
| 3 | 45 | LLaMA-3.1-8B | MATH500 | 8 | 34.5 | 35.4 | -0.9 |
| 4 | 46 | LLaMA-3.1-8B | MATH500 | 8 | 34.6 | 36.2 | -1.6 |
| 5 | 47 | LLaMA-3.1-8B | MATH500 | 8 | 35.35 | 35.15 | 0.2 |
| 6 | 48 | LLaMA-3.1-8B | MATH500 | 8 | 34.35 | 36.4 | -2.05 |
| 7 | 49 | LLaMA-3.1-8B | MATH500 | 8 | 34.8 | 34.05 | 0.75 |
| 8 | 50 | LLaMA-3.1-8B | MATH500 | 8 | 34.15 | 35.7 | -1.55 |
| 9 | 51 | LLaMA-3.1-8B | MATH500 | 8 | 34.3 | 35.7 | -1.4 |
| 10 | 52 | LLaMA-3.1-8B | MATH500 | 8 | 34.6 | 35.55 | -0.95 |
| 11 | 53 | LLaMA-3.1-8B | MATH500 | 8 | 33.15 | 34.55 | -1.4 |
| 12 | 54 | LLaMA-3.1-8B | MATH500 | 8 | 34.7 | 35.7 | -1 |
| 13 | 55 | LLaMA-3.1-8B | MATH500 | 8 | 34.45 | 35.25 | -0.8 |
| 14 | 56 | LLaMA-3.1-8B | MATH500 | 8 | 35 | 35 | 0 |
| 15 | 57 | LLaMA-3.1-8B | MATH500 | 8 | 34.45 | 35.25 | -0.8 |
| 16 | 58 | LLaMA-3.1-8B | MATH500 | 8 | 35.55 | 36 | -0.45 |
| 17 | 59 | LLaMA-3.1-8B | MATH500 | 8 | 35.05 | 34.05 | 1 |
| 18 | 60 | LLaMA-3.1-8B | MATH500 | 8 | 34.3 | 35.45 | -1.15 |
| 19 | 61 | LLaMA-3.1-8B | MATH500 | 8 | 33.85 | 36.15 | -2.3 |
| 20 | 62 | LLaMA-3.1-8B | MATH500 | 8 | 34.65 | 35.05 | -0.4 |
| 21 | 63 | LLaMA-3.1-8B | MATH500 | 8 | 35.85 | 35.35 | 0.5 |
| 22 | 64 | LLaMA-3.1-8B | MATH500 | 8 | 35.6 | 35.4 | 0.2 |
| 23 | 65 | LLaMA-3.1-8B | MATH500 | 8 | 33.15 | 35.4 | -2.25 |
| 24 | 66 | LLaMA-3.1-8B | MATH500 | 8 | 34.8 | 34.95 | -0.15 |
| 25 | 67 | LLaMA-3.1-8B | MATH500 | 8 | 34.4 | 35.05 | -0.65 |
| 26 | 68 | LLaMA-3.1-8B | MATH500 | 8 | 33.5 | 35.55 | -2.05 |
| 27 | 69 | LLaMA-3.1-8B | MATH500 | 8 | 33.75 | 34.55 | -0.8 |
| 28 | 70 | LLaMA-3.1-8B | MATH500 | 8 | 35.1 | 35.15 | -0.05 |
| 29 | 71 | LLaMA-3.1-8B | MATH500 | 8 | 34.5 | 36.1 | -1.6 |
| 30 | 72 | LLaMA-3.1-8B | MATH500 | 8 | 33.25 | 34.5 | -1.25 |
| 31 | 73 | LLaMA-3.1-8B | MATH500 | 8 | 33.8 | 36.2 | -2.4 |
| 32 | 74 | LLaMA-3.1-8B | MATH500 | 8 | 35.25 | 35.8 | -0.55 |
| 33 | 75 | LLaMA-3.1-8B | MATH500 | 8 | 33 | 34.45 | -1.45 |
| 34 | 76 | LLaMA-3.1-8B | MATH500 | 8 | 35 | 35.7 | -0.7 |
| 35 | 77 | LLaMA-3.1-8B | MATH500 | 8 | 34.3 | 34.75 | -0.45 |
| 36 | 78 | LLaMA-3.1-8B | MATH500 | 8 | 34.7 | 35.25 | -0.55 |
| 37 | 79 | LLaMA-3.1-8B | MATH500 | 8 | 34.25 | 36.8 | -2.55 |
| 38 | 80 | LLaMA-3.1-8B | MATH500 | 8 | 34.2 | 35.5 | -1.3 |
| 39 | 81 | LLaMA-3.1-8B | MATH500 | 8 | 34.75 | 35.3 | -0.55 |
| 40 | 82 | LLaMA-3.1-8B | MATH500 | 8 | 34.05 | 35.05 | -1 |
| 41 | 83 | LLaMA-3.1-8B | MATH500 | 8 | 34.45 | 34.95 | -0.5 |
| 42 | 84 | LLaMA-3.1-8B | MATH500 | 8 | 33.8 | 35 | -1.2 |
| 43 | 85 | LLaMA-3.1-8B | MATH500 | 8 | 34.55 | 35.3 | -0.75 |
| 44 | 86 | LLaMA-3.1-8B | MATH500 | 8 | 35.8 | 36 | -0.2 |
| 45 | 87 | LLaMA-3.1-8B | MATH500 | 8 | 33.65 | 34.6 | -0.95 |
| 46 | 88 | LLaMA-3.1-8B | MATH500 | 8 | 36.1 | 35.8 | 0.3 |
| 47 | 89 | LLaMA-3.1-8B | MATH500 | 8 | 34 | 36.6 | -2.6 |
| 48 | 90 | LLaMA-3.1-8B | MATH500 | 8 | 34.5 | 35.35 | -0.85 |
| 49 | 91 | LLaMA-3.1-8B | MATH500 | 8 | 33.65 | 34.25 | -0.6 |
| 0 | 42 | LLaMA-3.1-8B | MATH500 | 12 | 33.6 | 35.8333 | -2.2333 |
| 1 | 43 | LLaMA-3.1-8B | MATH500 | 12 | 34.6333 | 35.5667 | -0.9333 |
| 2 | 44 | LLaMA-3.1-8B | MATH500 | 12 | 33.7 | 36.0667 | -2.3667 |
| 3 | 45 | LLaMA-3.1-8B | MATH500 | 12 | 34.5333 | 35.4667 | -0.9333 |
| 4 | 46 | LLaMA-3.1-8B | MATH500 | 12 | 34.6333 | 35.6333 | -1 |
| 5 | 47 | LLaMA-3.1-8B | MATH500 | 12 | 34 | 35.4 | -1.4 |
| 6 | 48 | LLaMA-3.1-8B | MATH500 | 12 | 34.5 | 35.8667 | -1.3667 |
| 7 | 49 | LLaMA-3.1-8B | MATH500 | 12 | 33.9333 | 35.3667 | -1.4333 |
| 8 | 50 | LLaMA-3.1-8B | MATH500 | 12 | 33.9333 | 35.5 | -1.5667 |
| 9 | 51 | LLaMA-3.1-8B | MATH500 | 12 | 34.5667 | 35.9333 | -1.3667 |
| 10 | 52 | LLaMA-3.1-8B | MATH500 | 12 | 34.6 | 35.4667 | -0.8667 |
| 11 | 53 | LLaMA-3.1-8B | MATH500 | 12 | 33.5667 | 35.0333 | -1.4667 |
| 12 | 54 | LLaMA-3.1-8B | MATH500 | 12 | 34.5 | 36.2667 | -1.7667 |
| 13 | 55 | LLaMA-3.1-8B | MATH500 | 12 | 33.9 | 34.8 | -0.9 |
| 14 | 56 | LLaMA-3.1-8B | MATH500 | 12 | 34.5 | 35.2333 | -0.7333 |
| 15 | 57 | LLaMA-3.1-8B | MATH500 | 12 | 33.8 | 35.8 | -2 |
| 16 | 58 | LLaMA-3.1-8B | MATH500 | 12 | 34.4333 | 36.1333 | -1.7 |
| 17 | 59 | LLaMA-3.1-8B | MATH500 | 12 | 34.8 | 35.1 | -0.3 |
| 18 | 60 | LLaMA-3.1-8B | MATH500 | 12 | 34.1 | 35.5 | -1.4 |
| 19 | 61 | LLaMA-3.1-8B | MATH500 | 12 | 34.5667 | 35.9667 | -1.4 |
| 20 | 62 | LLaMA-3.1-8B | MATH500 | 12 | 34.6667 | 34.8667 | -0.2 |
| 21 | 63 | LLaMA-3.1-8B | MATH500 | 12 | 35.1333 | 35.3667 | -0.2333 |
| 22 | 64 | LLaMA-3.1-8B | MATH500 | 12 | 34.4 | 35.9 | -1.5 |
| 23 | 65 | LLaMA-3.1-8B | MATH500 | 12 | 33.4667 | 35.3667 | -1.9 |
| 24 | 66 | LLaMA-3.1-8B | MATH500 | 12 | 34.5333 | 35.1667 | -0.6333 |
| 25 | 67 | LLaMA-3.1-8B | MATH500 | 12 | 33.8 | 34.7667 | -0.9667 |
| 26 | 68 | LLaMA-3.1-8B | MATH500 | 12 | 34.3667 | 35.7 | -1.3333 |
| 27 | 69 | LLaMA-3.1-8B | MATH500 | 12 | 34.1667 | 35.4333 | -1.2667 |
| 28 | 70 | LLaMA-3.1-8B | MATH500 | 12 | 33.4333 | 35.6 | -2.1667 |
| 29 | 71 | LLaMA-3.1-8B | MATH500 | 12 | 34.6667 | 35.1333 | -0.4667 |
| 30 | 72 | LLaMA-3.1-8B | MATH500 | 12 | 33.5667 | 34.8 | -1.2333 |
| 31 | 73 | LLaMA-3.1-8B | MATH500 | 12 | 34.6 | 35.4 | -0.8 |
| 32 | 74 | LLaMA-3.1-8B | MATH500 | 12 | 34.3333 | 35.0667 | -0.7333 |
| 33 | 75 | LLaMA-3.1-8B | MATH500 | 12 | 34.1333 | 35.6333 | -1.5 |
| 34 | 76 | LLaMA-3.1-8B | MATH500 | 12 | 34.3667 | 35.5667 | -1.2 |
| 35 | 77 | LLaMA-3.1-8B | MATH500 | 12 | 34.6333 | 35.1333 | -0.5 |
| 36 | 78 | LLaMA-3.1-8B | MATH500 | 12 | 34.5333 | 35.4667 | -0.9333 |
| 37 | 79 | LLaMA-3.1-8B | MATH500 | 12 | 34.5667 | 35.7333 | -1.1667 |
| 38 | 80 | LLaMA-3.1-8B | MATH500 | 12 | 34.4 | 35.1667 | -0.7667 |
| 39 | 81 | LLaMA-3.1-8B | MATH500 | 12 | 34.1333 | 35.7 | -1.5667 |
| 40 | 82 | LLaMA-3.1-8B | MATH500 | 12 | 34.5667 | 34.9667 | -0.4 |
| 41 | 83 | LLaMA-3.1-8B | MATH500 | 12 | 34.1667 | 35.6 | -1.4333 |
| 42 | 84 | LLaMA-3.1-8B | MATH500 | 12 | 34.2667 | 35.2 | -0.9333 |
| 43 | 85 | LLaMA-3.1-8B | MATH500 | 12 | 34.7667 | 35.3 | -0.5333 |
| 44 | 86 | LLaMA-3.1-8B | MATH500 | 12 | 34.4 | 36.1333 | -1.7333 |
| 45 | 87 | LLaMA-3.1-8B | MATH500 | 12 | 34.4667 | 35.2 | -0.7333 |
| 46 | 88 | LLaMA-3.1-8B | MATH500 | 12 | 34.8333 | 35.1667 | -0.3333 |
| 47 | 89 | LLaMA-3.1-8B | MATH500 | 12 | 34.2 | 35.9 | -1.7 |
| 48 | 90 | LLaMA-3.1-8B | MATH500 | 12 | 34.5333 | 35.2667 | -0.7333 |
| 49 | 91 | LLaMA-3.1-8B | MATH500 | 12 | 34.2667 | 35.6 | -1.3333 |
| 0 | 42 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 1 | 43 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 2 | 44 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 3 | 45 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 4 | 46 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 5 | 47 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 6 | 48 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 7 | 49 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 8 | 50 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 9 | 51 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 10 | 52 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 11 | 53 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 12 | 54 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 13 | 55 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 14 | 56 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 15 | 57 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 16 | 58 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 17 | 59 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 18 | 60 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 19 | 61 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 20 | 62 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 21 | 63 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 22 | 64 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 23 | 65 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 24 | 66 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 25 | 67 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 26 | 68 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 27 | 69 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 28 | 70 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 29 | 71 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 30 | 72 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 31 | 73 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 32 | 74 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 33 | 75 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 34 | 76 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 35 | 77 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 36 | 78 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 37 | 79 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 38 | 80 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 39 | 81 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 40 | 82 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 41 | 83 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 42 | 84 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 43 | 85 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 44 | 86 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 45 | 87 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 46 | 88 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 47 | 89 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 48 | 90 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 49 | 91 | LLaMA-3.1-8B | MATH500 | 16 | 34.35 | 35.45 | -1.1 |
| 0 | 42 | LLaMA-3.1-8B | SVAMP | 4 | 80.45 | 80.15 | 0.3 |
| 1 | 43 | LLaMA-3.1-8B | SVAMP | 4 | 80.7 | 81.1 | -0.4 |
| 2 | 44 | LLaMA-3.1-8B | SVAMP | 4 | 81.35 | 82.4 | -1.05 |
| 3 | 45 | LLaMA-3.1-8B | SVAMP | 4 | 79.7 | 80.1 | -0.4 |
| 4 | 46 | LLaMA-3.1-8B | SVAMP | 4 | 80.9 | 80.2 | 0.7 |
| 5 | 47 | LLaMA-3.1-8B | SVAMP | 4 | 82.15 | 81.1 | 1.05 |
| 6 | 48 | LLaMA-3.1-8B | SVAMP | 4 | 80.5 | 81.3 | -0.8 |
| 7 | 49 | LLaMA-3.1-8B | SVAMP | 4 | 80.75 | 81.7 | -0.95 |
| 8 | 50 | LLaMA-3.1-8B | SVAMP | 4 | 81.4 | 80 | 1.4 |
| 9 | 51 | LLaMA-3.1-8B | SVAMP | 4 | 80 | 81.25 | -1.25 |
| 10 | 52 | LLaMA-3.1-8B | SVAMP | 4 | 80.75 | 80.65 | 0.1 |
| 11 | 53 | LLaMA-3.1-8B | SVAMP | 4 | 82.1 | 80.9 | 1.2 |
| 12 | 54 | LLaMA-3.1-8B | SVAMP | 4 | 80.9 | 81.2 | -0.3 |
| 13 | 55 | LLaMA-3.1-8B | SVAMP | 4 | 81.25 | 81.05 | 0.2 |
| 14 | 56 | LLaMA-3.1-8B | SVAMP | 4 | 80.9 | 80.85 | 0.05 |
| 15 | 57 | LLaMA-3.1-8B | SVAMP | 4 | 79.95 | 79.85 | 0.1 |
| 16 | 58 | LLaMA-3.1-8B | SVAMP | 4 | 81.15 | 80.35 | 0.8 |
| 17 | 59 | LLaMA-3.1-8B | SVAMP | 4 | 80.25 | 80.05 | 0.2 |
| 18 | 60 | LLaMA-3.1-8B | SVAMP | 4 | 81.1 | 81.75 | -0.65 |
| 19 | 61 | LLaMA-3.1-8B | SVAMP | 4 | 80.65 | 82.9 | -2.25 |
| 20 | 62 | LLaMA-3.1-8B | SVAMP | 4 | 80.3 | 81.25 | -0.95 |
| 21 | 63 | LLaMA-3.1-8B | SVAMP | 4 | 80.2 | 80.75 | -0.55 |
| 22 | 64 | LLaMA-3.1-8B | SVAMP | 4 | 80.7 | 80.75 | -0.05 |
| 23 | 65 | LLaMA-3.1-8B | SVAMP | 4 | 81.35 | 81.9 | -0.55 |
| 24 | 66 | LLaMA-3.1-8B | SVAMP | 4 | 79.85 | 81.5 | -1.65 |
| 25 | 67 | LLaMA-3.1-8B | SVAMP | 4 | 80.65 | 81.85 | -1.2 |
| 26 | 68 | LLaMA-3.1-8B | SVAMP | 4 | 80.65 | 82.05 | -1.4 |
| 27 | 69 | LLaMA-3.1-8B | SVAMP | 4 | 80.9 | 80.25 | 0.65 |
| 28 | 70 | LLaMA-3.1-8B | SVAMP | 4 | 80.6 | 80.55 | 0.05 |
| 29 | 71 | LLaMA-3.1-8B | SVAMP | 4 | 80.05 | 81.15 | -1.1 |
| 30 | 72 | LLaMA-3.1-8B | SVAMP | 4 | 80.4 | 80.25 | 0.15 |
| 31 | 73 | LLaMA-3.1-8B | SVAMP | 4 | 81.15 | 81.5 | -0.35 |
| 32 | 74 | LLaMA-3.1-8B | SVAMP | 4 | 81.6 | 80.75 | 0.85 |
| 33 | 75 | LLaMA-3.1-8B | SVAMP | 4 | 81 | 80.6 | 0.4 |
| 34 | 76 | LLaMA-3.1-8B | SVAMP | 4 | 81.4 | 79.9 | 1.5 |
| 35 | 77 | LLaMA-3.1-8B | SVAMP | 4 | 80.6 | 81.35 | -0.75 |
| 36 | 78 | LLaMA-3.1-8B | SVAMP | 4 | 80.15 | 80.35 | -0.2 |
| 37 | 79 | LLaMA-3.1-8B | SVAMP | 4 | 80 | 80.4 | -0.4 |
| 38 | 80 | LLaMA-3.1-8B | SVAMP | 4 | 80.1 | 81.3 | -1.2 |
| 39 | 81 | LLaMA-3.1-8B | SVAMP | 4 | 80.4 | 81.85 | -1.45 |
| 40 | 82 | LLaMA-3.1-8B | SVAMP | 4 | 81.7 | 82.65 | -0.95 |
| 41 | 83 | LLaMA-3.1-8B | SVAMP | 4 | 80.85 | 81.1 | -0.25 |
| 42 | 84 | LLaMA-3.1-8B | SVAMP | 4 | 80.45 | 80.6 | -0.15 |
| 43 | 85 | LLaMA-3.1-8B | SVAMP | 4 | 81.25 | 81 | 0.25 |
| 44 | 86 | LLaMA-3.1-8B | SVAMP | 4 | 79.7 | 79.85 | -0.15 |
| 45 | 87 | LLaMA-3.1-8B | SVAMP | 4 | 80.7 | 81 | -0.3 |
| 46 | 88 | LLaMA-3.1-8B | SVAMP | 4 | 80.7 | 81.4 | -0.7 |
| 47 | 89 | LLaMA-3.1-8B | SVAMP | 4 | 80.45 | 80 | 0.45 |
| 48 | 90 | LLaMA-3.1-8B | SVAMP | 4 | 81.4 | 80.2 | 1.2 |
| 49 | 91 | LLaMA-3.1-8B | SVAMP | 4 | 79.45 | 80.95 | -1.5 |
| 0 | 42 | LLaMA-3.1-8B | SVAMP | 8 | 80.7 | 81.075 | -0.375 |
| 1 | 43 | LLaMA-3.1-8B | SVAMP | 8 | 80.525 | 81.025 | -0.5 |
| 2 | 44 | LLaMA-3.1-8B | SVAMP | 8 | 81.225 | 81.325 | -0.1 |
| 3 | 45 | LLaMA-3.1-8B | SVAMP | 8 | 79.975 | 80.85 | -0.875 |
| 4 | 46 | LLaMA-3.1-8B | SVAMP | 8 | 80.4 | 80.55 | -0.15 |
| 5 | 47 | LLaMA-3.1-8B | SVAMP | 8 | 81.4 | 80.925 | 0.475 |
| 6 | 48 | LLaMA-3.1-8B | SVAMP | 8 | 80.15 | 81.025 | -0.875 |
| 7 | 49 | LLaMA-3.1-8B | SVAMP | 8 | 80.575 | 81.325 | -0.75 |
| 8 | 50 | LLaMA-3.1-8B | SVAMP | 8 | 80.5 | 81.125 | -0.625 |
| 9 | 51 | LLaMA-3.1-8B | SVAMP | 8 | 80.075 | 81.375 | -1.3 |
| 10 | 52 | LLaMA-3.1-8B | SVAMP | 8 | 81.125 | 80.875 | 0.25 |
| 11 | 53 | LLaMA-3.1-8B | SVAMP | 8 | 81.2 | 80.7 | 0.5 |
| 12 | 54 | LLaMA-3.1-8B | SVAMP | 8 | 80.8 | 81.1 | -0.3 |
| 13 | 55 | LLaMA-3.1-8B | SVAMP | 8 | 80.225 | 81.35 | -1.125 |
| 14 | 56 | LLaMA-3.1-8B | SVAMP | 8 | 80.75 | 80.975 | -0.225 |
| 15 | 57 | LLaMA-3.1-8B | SVAMP | 8 | 80.125 | 80.925 | -0.8 |
| 16 | 58 | LLaMA-3.1-8B | SVAMP | 8 | 79.975 | 80.85 | -0.875 |
| 17 | 59 | LLaMA-3.1-8B | SVAMP | 8 | 80.45 | 80.875 | -0.425 |
| 18 | 60 | LLaMA-3.1-8B | SVAMP | 8 | 80.4 | 80.975 | -0.575 |
| 19 | 61 | LLaMA-3.1-8B | SVAMP | 8 | 80.75 | 81.35 | -0.6 |
| 20 | 62 | LLaMA-3.1-8B | SVAMP | 8 | 80.225 | 81.425 | -1.2 |
| 21 | 63 | LLaMA-3.1-8B | SVAMP | 8 | 80.925 | 80.4 | 0.525 |
| 22 | 64 | LLaMA-3.1-8B | SVAMP | 8 | 80.75 | 80.95 | -0.2 |
| 23 | 65 | LLaMA-3.1-8B | SVAMP | 8 | 81 | 81.55 | -0.55 |
| 24 | 66 | LLaMA-3.1-8B | SVAMP | 8 | 80.925 | 80.85 | 0.075 |
| 25 | 67 | LLaMA-3.1-8B | SVAMP | 8 | 80.925 | 81.175 | -0.25 |
| 26 | 68 | LLaMA-3.1-8B | SVAMP | 8 | 80.725 | 81.35 | -0.625 |
| 27 | 69 | LLaMA-3.1-8B | SVAMP | 8 | 80.85 | 81.175 | -0.325 |
| 28 | 70 | LLaMA-3.1-8B | SVAMP | 8 | 80.95 | 80.775 | 0.175 |
| 29 | 71 | LLaMA-3.1-8B | SVAMP | 8 | 80.75 | 81.05 | -0.3 |
| 30 | 72 | LLaMA-3.1-8B | SVAMP | 8 | 81.1 | 80.525 | 0.575 |
| 31 | 73 | LLaMA-3.1-8B | SVAMP | 8 | 81.225 | 81.125 | 0.1 |
| 32 | 74 | LLaMA-3.1-8B | SVAMP | 8 | 80.9 | 80.725 | 0.175 |
| 33 | 75 | LLaMA-3.1-8B | SVAMP | 8 | 79.925 | 80.525 | -0.6 |
| 34 | 76 | LLaMA-3.1-8B | SVAMP | 8 | 80.7 | 80.675 | 0.025 |
| 35 | 77 | LLaMA-3.1-8B | SVAMP | 8 | 80.625 | 81.225 | -0.6 |
| 36 | 78 | LLaMA-3.1-8B | SVAMP | 8 | 79.925 | 81.15 | -1.225 |
| 37 | 79 | LLaMA-3.1-8B | SVAMP | 8 | 80.45 | 81.15 | -0.7 |
| 38 | 80 | LLaMA-3.1-8B | SVAMP | 8 | 80.7 | 81.425 | -0.725 |
| 39 | 81 | LLaMA-3.1-8B | SVAMP | 8 | 80.2 | 80.85 | -0.65 |
| 40 | 82 | LLaMA-3.1-8B | SVAMP | 8 | 80.95 | 82.225 | -1.275 |
| 41 | 83 | LLaMA-3.1-8B | SVAMP | 8 | 80.625 | 81.175 | -0.55 |
| 42 | 84 | LLaMA-3.1-8B | SVAMP | 8 | 80.7 | 80.825 | -0.125 |
| 43 | 85 | LLaMA-3.1-8B | SVAMP | 8 | 80.35 | 80.375 | -0.025 |
| 44 | 86 | LLaMA-3.1-8B | SVAMP | 8 | 80.525 | 80.45 | 0.075 |
| 45 | 87 | LLaMA-3.1-8B | SVAMP | 8 | 80.725 | 81.3 | -0.575 |
| 46 | 88 | LLaMA-3.1-8B | SVAMP | 8 | 80.4 | 81.05 | -0.65 |
| 47 | 89 | LLaMA-3.1-8B | SVAMP | 8 | 80.575 | 80.575 | 0 |
| 48 | 90 | LLaMA-3.1-8B | SVAMP | 8 | 81.325 | 80.925 | 0.4 |
| 49 | 91 | LLaMA-3.1-8B | SVAMP | 8 | 80.675 | 80.65 | 0.025 |
| 0 | 42 | LLaMA-3.1-8B | SVAMP | 12 | 80.4333 | 81.05 | -0.6167 |
| 1 | 43 | LLaMA-3.1-8B | SVAMP | 12 | 80.3 | 81.0167 | -0.7167 |
| 2 | 44 | LLaMA-3.1-8B | SVAMP | 12 | 80.9833 | 80.9833 | 0 |
| 3 | 45 | LLaMA-3.1-8B | SVAMP | 12 | 80.4167 | 81.4667 | -1.05 |
| 4 | 46 | LLaMA-3.1-8B | SVAMP | 12 | 80.4833 | 80.95 | -0.4667 |
| 5 | 47 | LLaMA-3.1-8B | SVAMP | 12 | 80.8667 | 81 | -0.1333 |
| 6 | 48 | LLaMA-3.1-8B | SVAMP | 12 | 80.4 | 81.0167 | -0.6167 |
| 7 | 49 | LLaMA-3.1-8B | SVAMP | 12 | 80.7833 | 81.35 | -0.5667 |
| 8 | 50 | LLaMA-3.1-8B | SVAMP | 12 | 80.6333 | 80.85 | -0.2167 |
| 9 | 51 | LLaMA-3.1-8B | SVAMP | 12 | 80.5 | 81.3667 | -0.8667 |
| 10 | 52 | LLaMA-3.1-8B | SVAMP | 12 | 80.6667 | 81.0333 | -0.3667 |
| 11 | 53 | LLaMA-3.1-8B | SVAMP | 12 | 80.6 | 80.9167 | -0.3167 |
| 12 | 54 | LLaMA-3.1-8B | SVAMP | 12 | 80.65 | 80.8667 | -0.2167 |
| 13 | 55 | LLaMA-3.1-8B | SVAMP | 12 | 80.3833 | 80.7833 | -0.4 |
| 14 | 56 | LLaMA-3.1-8B | SVAMP | 12 | 80.75 | 81.1 | -0.35 |
| 15 | 57 | LLaMA-3.1-8B | SVAMP | 12 | 80.5167 | 80.95 | -0.4333 |
| 16 | 58 | LLaMA-3.1-8B | SVAMP | 12 | 80.3667 | 80.65 | -0.2833 |
| 17 | 59 | LLaMA-3.1-8B | SVAMP | 12 | 80.65 | 80.9667 | -0.3167 |
| 18 | 60 | LLaMA-3.1-8B | SVAMP | 12 | 80.4667 | 80.9833 | -0.5167 |
| 19 | 61 | LLaMA-3.1-8B | SVAMP | 12 | 80.6833 | 81.0833 | -0.4 |
| 20 | 62 | LLaMA-3.1-8B | SVAMP | 12 | 80.2833 | 81.0667 | -0.7833 |
| 21 | 63 | LLaMA-3.1-8B | SVAMP | 12 | 80.45 | 80.8167 | -0.3667 |
| 22 | 64 | LLaMA-3.1-8B | SVAMP | 12 | 80.6667 | 81.0167 | -0.35 |
| 23 | 65 | LLaMA-3.1-8B | SVAMP | 12 | 80.9333 | 81.55 | -0.6167 |
| 24 | 66 | LLaMA-3.1-8B | SVAMP | 12 | 81.0167 | 81.1333 | -0.1167 |
| 25 | 67 | LLaMA-3.1-8B | SVAMP | 12 | 80.45 | 80.8667 | -0.4167 |
| 26 | 68 | LLaMA-3.1-8B | SVAMP | 12 | 80.8 | 81.1167 | -0.3167 |
| 27 | 69 | LLaMA-3.1-8B | SVAMP | 12 | 80.5667 | 81.3333 | -0.7667 |
| 28 | 70 | LLaMA-3.1-8B | SVAMP | 12 | 80.55 | 80.85 | -0.3 |
| 29 | 71 | LLaMA-3.1-8B | SVAMP | 12 | 80.6667 | 81.1333 | -0.4667 |
| 30 | 72 | LLaMA-3.1-8B | SVAMP | 12 | 80.9667 | 80.8 | 0.1667 |
| 31 | 73 | LLaMA-3.1-8B | SVAMP | 12 | 80.85 | 81.05 | -0.2 |
| 32 | 74 | LLaMA-3.1-8B | SVAMP | 12 | 80.4833 | 80.9333 | -0.45 |
| 33 | 75 | LLaMA-3.1-8B | SVAMP | 12 | 80.6167 | 80.7833 | -0.1667 |
| 34 | 76 | LLaMA-3.1-8B | SVAMP | 12 | 80.7833 | 80.9167 | -0.1333 |
| 35 | 77 | LLaMA-3.1-8B | SVAMP | 12 | 80.75 | 80.9167 | -0.1667 |
| 36 | 78 | LLaMA-3.1-8B | SVAMP | 12 | 80.3667 | 81.0333 | -0.6667 |
| 37 | 79 | LLaMA-3.1-8B | SVAMP | 12 | 80.25 | 80.6833 | -0.4333 |
| 38 | 80 | LLaMA-3.1-8B | SVAMP | 12 | 80.65 | 80.9833 | -0.3333 |
| 39 | 81 | LLaMA-3.1-8B | SVAMP | 12 | 80.55 | 81 | -0.45 |
| 40 | 82 | LLaMA-3.1-8B | SVAMP | 12 | 80.85 | 81.4833 | -0.6333 |
| 41 | 83 | LLaMA-3.1-8B | SVAMP | 12 | 81.05 | 80.8167 | 0.2333 |
| 42 | 84 | LLaMA-3.1-8B | SVAMP | 12 | 81.0167 | 80.7 | 0.3167 |
| 43 | 85 | LLaMA-3.1-8B | SVAMP | 12 | 80.4 | 81.0167 | -0.6167 |
| 44 | 86 | LLaMA-3.1-8B | SVAMP | 12 | 80.5667 | 80.55 | 0.0167 |
| 45 | 87 | LLaMA-3.1-8B | SVAMP | 12 | 80.8167 | 80.9 | -0.0833 |
| 46 | 88 | LLaMA-3.1-8B | SVAMP | 12 | 80.3667 | 80.9667 | -0.6 |
| 47 | 89 | LLaMA-3.1-8B | SVAMP | 12 | 80.5333 | 80.9 | -0.3667 |
| 48 | 90 | LLaMA-3.1-8B | SVAMP | 12 | 80.8 | 81.1167 | -0.3167 |
| 49 | 91 | LLaMA-3.1-8B | SVAMP | 12 | 80.8833 | 80.6833 | 0.2 |
| 0 | 42 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 1 | 43 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 2 | 44 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 3 | 45 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 4 | 46 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 5 | 47 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 6 | 48 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 7 | 49 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 8 | 50 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 9 | 51 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 10 | 52 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 11 | 53 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 12 | 54 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 13 | 55 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 14 | 56 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 15 | 57 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 16 | 58 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 17 | 59 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 18 | 60 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 19 | 61 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 20 | 62 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 21 | 63 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 22 | 64 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 23 | 65 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 24 | 66 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 25 | 67 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 26 | 68 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 27 | 69 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 28 | 70 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 29 | 71 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 30 | 72 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 31 | 73 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 32 | 74 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 33 | 75 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 34 | 76 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 35 | 77 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 36 | 78 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 37 | 79 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 38 | 80 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 39 | 81 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 40 | 82 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 41 | 83 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 42 | 84 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 43 | 85 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 44 | 86 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 45 | 87 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 46 | 88 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 47 | 89 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 48 | 90 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 49 | 91 | LLaMA-3.1-8B | SVAMP | 16 | 80.7 | 80.95 | -0.25 |
| 0 | 42 | Phi-4-Reas. | AQuA | 4 | 58.4646 | 60.0394 | -1.5748 |
| 1 | 43 | Phi-4-Reas. | AQuA | 4 | 60.4331 | 57.874 | 2.5591 |
| 2 | 44 | Phi-4-Reas. | AQuA | 4 | 57.874 | 57.0866 | 0.7874 |
| 3 | 45 | Phi-4-Reas. | AQuA | 4 | 59.4488 | 57.874 | 1.5748 |
| 4 | 46 | Phi-4-Reas. | AQuA | 4 | 61.2205 | 57.874 | 3.3465 |
| 5 | 47 | Phi-4-Reas. | AQuA | 4 | 57.4803 | 56.6929 | 0.7874 |
| 6 | 48 | Phi-4-Reas. | AQuA | 4 | 60.0394 | 56.2992 | 3.7402 |
| 7 | 49 | Phi-4-Reas. | AQuA | 4 | 59.252 | 56.8898 | 2.3622 |
| 8 | 50 | Phi-4-Reas. | AQuA | 4 | 62.2047 | 53.1496 | 9.0551 |
| 9 | 51 | Phi-4-Reas. | AQuA | 4 | 58.0709 | 56.6929 | 1.378 |
| 10 | 52 | Phi-4-Reas. | AQuA | 4 | 59.6457 | 60.0394 | -0.3937 |
| 11 | 53 | Phi-4-Reas. | AQuA | 4 | 59.252 | 59.252 | 0 |
| 12 | 54 | Phi-4-Reas. | AQuA | 4 | 61.2205 | 61.4173 | -0.1969 |
| 13 | 55 | Phi-4-Reas. | AQuA | 4 | 58.8583 | 55.1181 | 3.7402 |
| 14 | 56 | Phi-4-Reas. | AQuA | 4 | 63.7795 | 57.874 | 5.9055 |
| 15 | 57 | Phi-4-Reas. | AQuA | 4 | 59.4488 | 59.4488 | 0 |
| 16 | 58 | Phi-4-Reas. | AQuA | 4 | 60.6299 | 56.6929 | 3.937 |
| 17 | 59 | Phi-4-Reas. | AQuA | 4 | 61.6142 | 57.4803 | 4.1339 |
| 18 | 60 | Phi-4-Reas. | AQuA | 4 | 61.6142 | 58.0709 | 3.5433 |
| 19 | 61 | Phi-4-Reas. | AQuA | 4 | 61.811 | 57.6772 | 4.1339 |
| 20 | 62 | Phi-4-Reas. | AQuA | 4 | 60.2362 | 55.1181 | 5.1181 |
| 21 | 63 | Phi-4-Reas. | AQuA | 4 | 60.6299 | 56.1024 | 4.5276 |
| 22 | 64 | Phi-4-Reas. | AQuA | 4 | 60.4331 | 53.5433 | 6.8898 |
| 23 | 65 | Phi-4-Reas. | AQuA | 4 | 61.2205 | 54.7244 | 6.4961 |
| 24 | 66 | Phi-4-Reas. | AQuA | 4 | 61.811 | 57.4803 | 4.3307 |
| 25 | 67 | Phi-4-Reas. | AQuA | 4 | 60.6299 | 54.9213 | 5.7087 |
| 26 | 68 | Phi-4-Reas. | AQuA | 4 | 59.6457 | 61.0236 | -1.378 |
| 27 | 69 | Phi-4-Reas. | AQuA | 4 | 62.2047 | 55.315 | 6.8898 |
| 28 | 70 | Phi-4-Reas. | AQuA | 4 | 59.8425 | 58.6614 | 1.1811 |
| 29 | 71 | Phi-4-Reas. | AQuA | 4 | 58.4646 | 56.8898 | 1.5748 |
| 30 | 72 | Phi-4-Reas. | AQuA | 4 | 60.4331 | 58.0709 | 2.3622 |
| 31 | 73 | Phi-4-Reas. | AQuA | 4 | 56.8898 | 59.6457 | -2.7559 |
| 32 | 74 | Phi-4-Reas. | AQuA | 4 | 58.8583 | 56.2992 | 2.5591 |
| 33 | 75 | Phi-4-Reas. | AQuA | 4 | 58.4646 | 55.5118 | 2.9528 |
| 34 | 76 | Phi-4-Reas. | AQuA | 4 | 62.2047 | 60.0394 | 2.1654 |
| 35 | 77 | Phi-4-Reas. | AQuA | 4 | 58.8583 | 58.0709 | 0.7874 |
| 36 | 78 | Phi-4-Reas. | AQuA | 4 | 61.4173 | 56.4961 | 4.9213 |
| 37 | 79 | Phi-4-Reas. | AQuA | 4 | 60.0394 | 58.0709 | 1.9685 |
| 38 | 80 | Phi-4-Reas. | AQuA | 4 | 59.252 | 51.5748 | 7.6772 |
| 39 | 81 | Phi-4-Reas. | AQuA | 4 | 56.4961 | 56.2992 | 0.1969 |
| 40 | 82 | Phi-4-Reas. | AQuA | 4 | 59.4488 | 56.6929 | 2.7559 |
| 41 | 83 | Phi-4-Reas. | AQuA | 4 | 57.874 | 58.0709 | -0.1969 |
| 42 | 84 | Phi-4-Reas. | AQuA | 4 | 61.0236 | 55.5118 | 5.5118 |
| 43 | 85 | Phi-4-Reas. | AQuA | 4 | 60.4331 | 55.7087 | 4.7244 |
| 44 | 86 | Phi-4-Reas. | AQuA | 4 | 58.2677 | 53.5433 | 4.7244 |
| 45 | 87 | Phi-4-Reas. | AQuA | 4 | 61.811 | 53.7402 | 8.0709 |
| 46 | 88 | Phi-4-Reas. | AQuA | 4 | 57.874 | 57.6772 | 0.1969 |
| 47 | 89 | Phi-4-Reas. | AQuA | 4 | 57.874 | 57.2835 | 0.5906 |
| 48 | 90 | Phi-4-Reas. | AQuA | 4 | 57.2835 | 55.9055 | 1.378 |
| 49 | 91 | Phi-4-Reas. | AQuA | 4 | 59.8425 | 58.0709 | 1.7717 |
| 0 | 42 | Phi-4-Reas. | AQuA | 8 | 58.4646 | 57.9724 | 0.4921 |
| 1 | 43 | Phi-4-Reas. | AQuA | 8 | 58.8583 | 56.1024 | 2.7559 |
| 2 | 44 | Phi-4-Reas. | AQuA | 8 | 59.3504 | 55.9055 | 3.4449 |
| 3 | 45 | Phi-4-Reas. | AQuA | 8 | 58.7598 | 56.9882 | 1.7717 |
| 4 | 46 | Phi-4-Reas. | AQuA | 8 | 58.6614 | 57.2835 | 1.378 |
| 5 | 47 | Phi-4-Reas. | AQuA | 8 | 60.1378 | 55.6102 | 4.5276 |
| 6 | 48 | Phi-4-Reas. | AQuA | 8 | 59.4488 | 56.3976 | 3.0512 |
| 7 | 49 | Phi-4-Reas. | AQuA | 8 | 58.8583 | 57.0866 | 1.7717 |
| 8 | 50 | Phi-4-Reas. | AQuA | 8 | 60.1378 | 55.7087 | 4.4291 |
| 9 | 51 | Phi-4-Reas. | AQuA | 8 | 58.9567 | 56.2992 | 2.6575 |
| 10 | 52 | Phi-4-Reas. | AQuA | 8 | 58.4646 | 56.9882 | 1.4764 |
| 11 | 53 | Phi-4-Reas. | AQuA | 8 | 59.9409 | 56.9882 | 2.9528 |
| 12 | 54 | Phi-4-Reas. | AQuA | 8 | 61.122 | 58.1693 | 2.9528 |
| 13 | 55 | Phi-4-Reas. | AQuA | 8 | 60.7283 | 54.5276 | 6.2008 |
| 14 | 56 | Phi-4-Reas. | AQuA | 8 | 61.2205 | 56.5945 | 4.626 |
| 15 | 57 | Phi-4-Reas. | AQuA | 8 | 58.9567 | 55.9055 | 3.0512 |
| 16 | 58 | Phi-4-Reas. | AQuA | 8 | 58.1693 | 57.185 | 0.9843 |
| 17 | 59 | Phi-4-Reas. | AQuA | 8 | 60.9252 | 56.3976 | 4.5276 |
| 18 | 60 | Phi-4-Reas. | AQuA | 8 | 59.0551 | 57.7756 | 1.2795 |
| 19 | 61 | Phi-4-Reas. | AQuA | 8 | 60.6299 | 57.6772 | 2.9528 |
| 20 | 62 | Phi-4-Reas. | AQuA | 8 | 61.122 | 57.3819 | 3.7402 |
| 21 | 63 | Phi-4-Reas. | AQuA | 8 | 60.2362 | 56.7913 | 3.4449 |
| 22 | 64 | Phi-4-Reas. | AQuA | 8 | 59.6457 | 56.6929 | 2.9528 |
| 23 | 65 | Phi-4-Reas. | AQuA | 8 | 59.252 | 55.5118 | 3.7402 |
| 24 | 66 | Phi-4-Reas. | AQuA | 8 | 63.0906 | 57.4803 | 5.6102 |
| 25 | 67 | Phi-4-Reas. | AQuA | 8 | 61.811 | 55.0197 | 6.7913 |
| 26 | 68 | Phi-4-Reas. | AQuA | 8 | 60.6299 | 56.9882 | 3.6417 |
| 27 | 69 | Phi-4-Reas. | AQuA | 8 | 62.1063 | 56.4961 | 5.6102 |
| 28 | 70 | Phi-4-Reas. | AQuA | 8 | 59.6457 | 56.2992 | 3.3465 |
| 29 | 71 | Phi-4-Reas. | AQuA | 8 | 60.1378 | 56.9882 | 3.1496 |
| 30 | 72 | Phi-4-Reas. | AQuA | 8 | 60.7283 | 56.4961 | 4.2323 |
| 31 | 73 | Phi-4-Reas. | AQuA | 8 | 60.6299 | 55.8071 | 4.8228 |
| 32 | 74 | Phi-4-Reas. | AQuA | 8 | 61.2205 | 56.1024 | 5.1181 |
| 33 | 75 | Phi-4-Reas. | AQuA | 8 | 60.7283 | 55.8071 | 4.9213 |
| 34 | 76 | Phi-4-Reas. | AQuA | 8 | 60.7283 | 58.3661 | 2.3622 |
| 35 | 77 | Phi-4-Reas. | AQuA | 8 | 59.5472 | 57.4803 | 2.0669 |
| 36 | 78 | Phi-4-Reas. | AQuA | 8 | 60.4331 | 56.6929 | 3.7402 |
| 37 | 79 | Phi-4-Reas. | AQuA | 8 | 60.3346 | 56.7913 | 3.5433 |
| 38 | 80 | Phi-4-Reas. | AQuA | 8 | 59.3504 | 55.6102 | 3.7402 |
| 39 | 81 | Phi-4-Reas. | AQuA | 8 | 59.6457 | 55.2165 | 4.4291 |
| 40 | 82 | Phi-4-Reas. | AQuA | 8 | 58.1693 | 57.0866 | 1.0827 |
| 41 | 83 | Phi-4-Reas. | AQuA | 8 | 58.8583 | 58.8583 | 0 |
| 42 | 84 | Phi-4-Reas. | AQuA | 8 | 59.7441 | 55.7087 | 4.0354 |
| 43 | 85 | Phi-4-Reas. | AQuA | 8 | 61.2205 | 56.0039 | 5.2165 |
| 44 | 86 | Phi-4-Reas. | AQuA | 8 | 61.2205 | 55.6102 | 5.6102 |
| 45 | 87 | Phi-4-Reas. | AQuA | 8 | 60.9252 | 55.4134 | 5.5118 |
| 46 | 88 | Phi-4-Reas. | AQuA | 8 | 59.7441 | 56.8898 | 2.8543 |
| 47 | 89 | Phi-4-Reas. | AQuA | 8 | 58.4646 | 56.2008 | 2.2638 |
| 48 | 90 | Phi-4-Reas. | AQuA | 8 | 59.252 | 55.315 | 3.937 |
| 49 | 91 | Phi-4-Reas. | AQuA | 8 | 59.0551 | 58.0709 | 0.9843 |
| 0 | 42 | Phi-4-Reas. | AQuA | 12 | 60.8924 | 57.6115 | 3.2808 |
| 1 | 43 | Phi-4-Reas. | AQuA | 12 | 59.8425 | 56.6273 | 3.2152 |
| 2 | 44 | Phi-4-Reas. | AQuA | 12 | 59.9081 | 56.4961 | 3.4121 |
| 3 | 45 | Phi-4-Reas. | AQuA | 12 | 60.3018 | 56.6929 | 3.6089 |
| 4 | 46 | Phi-4-Reas. | AQuA | 12 | 59.9738 | 56.6929 | 3.2808 |
| 5 | 47 | Phi-4-Reas. | AQuA | 12 | 59.252 | 55.643 | 3.6089 |
| 6 | 48 | Phi-4-Reas. | AQuA | 12 | 59.9738 | 56.8241 | 3.1496 |
| 7 | 49 | Phi-4-Reas. | AQuA | 12 | 60.0394 | 56.5617 | 3.4777 |
| 8 | 50 | Phi-4-Reas. | AQuA | 12 | 60.105 | 55.7743 | 4.3307 |
| 9 | 51 | Phi-4-Reas. | AQuA | 12 | 60.2362 | 55.7743 | 4.4619 |
| 10 | 52 | Phi-4-Reas. | AQuA | 12 | 59.7769 | 56.6929 | 3.084 |
| 11 | 53 | Phi-4-Reas. | AQuA | 12 | 60.3018 | 56.1024 | 4.1995 |
| 12 | 54 | Phi-4-Reas. | AQuA | 12 | 60.8268 | 56.2992 | 4.5276 |
| 13 | 55 | Phi-4-Reas. | AQuA | 12 | 60.0394 | 56.3648 | 3.6745 |
| 14 | 56 | Phi-4-Reas. | AQuA | 12 | 61.2861 | 55.8399 | 5.4462 |
| 15 | 57 | Phi-4-Reas. | AQuA | 12 | 59.1207 | 56.5617 | 2.5591 |
| 16 | 58 | Phi-4-Reas. | AQuA | 12 | 59.7769 | 56.168 | 3.6089 |
| 17 | 59 | Phi-4-Reas. | AQuA | 12 | 60.3018 | 56.4961 | 3.8058 |
| 18 | 60 | Phi-4-Reas. | AQuA | 12 | 60.7612 | 56.4304 | 4.3307 |
| 19 | 61 | Phi-4-Reas. | AQuA | 12 | 61.0892 | 57.1522 | 3.937 |
| 20 | 62 | Phi-4-Reas. | AQuA | 12 | 60.7612 | 56.3648 | 4.3963 |
| 21 | 63 | Phi-4-Reas. | AQuA | 12 | 61.3517 | 56.2992 | 5.0525 |
| 22 | 64 | Phi-4-Reas. | AQuA | 12 | 60.105 | 56.8898 | 3.2152 |
| 23 | 65 | Phi-4-Reas. | AQuA | 12 | 59.7769 | 56.4304 | 3.3465 |
| 24 | 66 | Phi-4-Reas. | AQuA | 12 | 61.4829 | 57.0866 | 4.3963 |
| 25 | 67 | Phi-4-Reas. | AQuA | 12 | 60.3018 | 55.7743 | 4.5276 |
| 26 | 68 | Phi-4-Reas. | AQuA | 12 | 60.5643 | 57.021 | 3.5433 |
| 27 | 69 | Phi-4-Reas. | AQuA | 12 | 61.0236 | 56.8241 | 4.1995 |
| 28 | 70 | Phi-4-Reas. | AQuA | 12 | 59.6457 | 57.2835 | 2.3622 |
| 29 | 71 | Phi-4-Reas. | AQuA | 12 | 60.4331 | 56.7585 | 3.6745 |
| 30 | 72 | Phi-4-Reas. | AQuA | 12 | 60.7612 | 56.2992 | 4.4619 |
| 31 | 73 | Phi-4-Reas. | AQuA | 12 | 60.7612 | 56.4304 | 4.3307 |
| 32 | 74 | Phi-4-Reas. | AQuA | 12 | 59.252 | 56.3648 | 2.8871 |
| 33 | 75 | Phi-4-Reas. | AQuA | 12 | 60.0394 | 56.8898 | 3.1496 |
| 34 | 76 | Phi-4-Reas. | AQuA | 12 | 60.958 | 56.7585 | 4.1995 |
| 35 | 77 | Phi-4-Reas. | AQuA | 12 | 60.8268 | 56.4961 | 4.3307 |
| 36 | 78 | Phi-4-Reas. | AQuA | 12 | 62.4016 | 56.168 | 6.2336 |
| 37 | 79 | Phi-4-Reas. | AQuA | 12 | 60.5643 | 56.1024 | 4.4619 |
| 38 | 80 | Phi-4-Reas. | AQuA | 12 | 59.7113 | 56.168 | 3.5433 |
| 39 | 81 | Phi-4-Reas. | AQuA | 12 | 60.2362 | 55.7743 | 4.4619 |
| 40 | 82 | Phi-4-Reas. | AQuA | 12 | 59.6457 | 56.6273 | 3.0184 |
| 41 | 83 | Phi-4-Reas. | AQuA | 12 | 60.2362 | 57.4803 | 2.7559 |
| 42 | 84 | Phi-4-Reas. | AQuA | 12 | 60.105 | 56.3648 | 3.7402 |
| 43 | 85 | Phi-4-Reas. | AQuA | 12 | 59.9081 | 56.9554 | 2.9528 |
| 44 | 86 | Phi-4-Reas. | AQuA | 12 | 60.2362 | 56.9554 | 3.2808 |
| 45 | 87 | Phi-4-Reas. | AQuA | 12 | 60.958 | 56.4961 | 4.4619 |
| 46 | 88 | Phi-4-Reas. | AQuA | 12 | 60.3018 | 56.8241 | 3.4777 |
| 47 | 89 | Phi-4-Reas. | AQuA | 12 | 59.3832 | 55.7743 | 3.6089 |
| 48 | 90 | Phi-4-Reas. | AQuA | 12 | 59.4488 | 56.4304 | 3.0184 |
| 49 | 91 | Phi-4-Reas. | AQuA | 12 | 60.4331 | 56.8898 | 3.5433 |
| 0 | 42 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 1 | 43 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 2 | 44 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 3 | 45 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 4 | 46 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 5 | 47 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 6 | 48 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 7 | 49 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 8 | 50 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 9 | 51 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 10 | 52 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 11 | 53 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 12 | 54 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 13 | 55 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 14 | 56 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 15 | 57 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 16 | 58 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 17 | 59 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 18 | 60 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 19 | 61 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 20 | 62 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 21 | 63 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 22 | 64 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 23 | 65 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 24 | 66 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 25 | 67 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 26 | 68 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 27 | 69 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 28 | 70 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 29 | 71 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 30 | 72 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 31 | 73 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 32 | 74 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 33 | 75 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 34 | 76 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 35 | 77 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 36 | 78 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 37 | 79 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 38 | 80 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 39 | 81 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 40 | 82 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 41 | 83 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 42 | 84 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 43 | 85 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 44 | 86 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 45 | 87 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 46 | 88 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 47 | 89 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 48 | 90 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 49 | 91 | Phi-4-Reas. | AQuA | 16 | 60.3346 | 56.5453 | 3.7894 |
| 0 | 42 | Phi-4-Reas. | CommonsenseQA | 4 | 73.3006 | 73.0139 | 0.2867 |
| 1 | 43 | Phi-4-Reas. | CommonsenseQA | 4 | 73.6282 | 74.3653 | -0.7371 |
| 2 | 44 | Phi-4-Reas. | CommonsenseQA | 4 | 74.3653 | 74.1196 | 0.2457 |
| 3 | 45 | Phi-4-Reas. | CommonsenseQA | 4 | 74.8157 | 73.792 | 1.0238 |
| 4 | 46 | Phi-4-Reas. | CommonsenseQA | 4 | 75.6347 | 74.1196 | 1.5152 |
| 5 | 47 | Phi-4-Reas. | CommonsenseQA | 4 | 74.9386 | 73.792 | 1.1466 |
| 6 | 48 | Phi-4-Reas. | CommonsenseQA | 4 | 74.1196 | 73.0549 | 1.0647 |
| 7 | 49 | Phi-4-Reas. | CommonsenseQA | 4 | 74.2424 | 74.4881 | -0.2457 |
| 8 | 50 | Phi-4-Reas. | CommonsenseQA | 4 | 74.57 | 73.9558 | 0.6143 |
| 9 | 51 | Phi-4-Reas. | CommonsenseQA | 4 | 75.5119 | 73.6691 | 1.8428 |
| 10 | 52 | Phi-4-Reas. | CommonsenseQA | 4 | 74.3653 | 74.1196 | 0.2457 |
| 11 | 53 | Phi-4-Reas. | CommonsenseQA | 4 | 74.2424 | 73.6282 | 0.6143 |
| 12 | 54 | Phi-4-Reas. | CommonsenseQA | 4 | 74.8157 | 74.2424 | 0.5733 |
| 13 | 55 | Phi-4-Reas. | CommonsenseQA | 4 | 74.2834 | 74.1605 | 0.1229 |
| 14 | 56 | Phi-4-Reas. | CommonsenseQA | 4 | 75.1843 | 74.4062 | 0.7781 |
| 15 | 57 | Phi-4-Reas. | CommonsenseQA | 4 | 74.611 | 73.3006 | 1.3104 |
| 16 | 58 | Phi-4-Reas. | CommonsenseQA | 4 | 73.5053 | 74.9795 | -1.4742 |
| 17 | 59 | Phi-4-Reas. | CommonsenseQA | 4 | 75.2252 | 73.5463 | 1.679 |
| 18 | 60 | Phi-4-Reas. | CommonsenseQA | 4 | 73.2596 | 73.7101 | -0.4505 |
| 19 | 61 | Phi-4-Reas. | CommonsenseQA | 4 | 75.2662 | 72.4406 | 2.8256 |
| 20 | 62 | Phi-4-Reas. | CommonsenseQA | 4 | 74.5291 | 74.1196 | 0.4095 |
| 21 | 63 | Phi-4-Reas. | CommonsenseQA | 4 | 74.5291 | 73.8329 | 0.6962 |
| 22 | 64 | Phi-4-Reas. | CommonsenseQA | 4 | 75.1843 | 73.3415 | 1.8428 |
| 23 | 65 | Phi-4-Reas. | CommonsenseQA | 4 | 73.1777 | 73.4644 | -0.2867 |
| 24 | 66 | Phi-4-Reas. | CommonsenseQA | 4 | 74.611 | 73.7101 | 0.9009 |
| 25 | 67 | Phi-4-Reas. | CommonsenseQA | 4 | 75.1024 | 74.4472 | 0.6552 |
| 26 | 68 | Phi-4-Reas. | CommonsenseQA | 4 | 75.1843 | 74.8567 | 0.3276 |
| 27 | 69 | Phi-4-Reas. | CommonsenseQA | 4 | 75.0205 | 73.2187 | 1.8018 |
| 28 | 70 | Phi-4-Reas. | CommonsenseQA | 4 | 73.1368 | 73.3825 | -0.2457 |
| 29 | 71 | Phi-4-Reas. | CommonsenseQA | 4 | 74.7338 | 73.3006 | 1.4333 |
| 30 | 72 | Phi-4-Reas. | CommonsenseQA | 4 | 76.0442 | 73.3415 | 2.7027 |
| 31 | 73 | Phi-4-Reas. | CommonsenseQA | 4 | 74.1605 | 73.6691 | 0.4914 |
| 32 | 74 | Phi-4-Reas. | CommonsenseQA | 4 | 74.9795 | 75.1843 | -0.2048 |
| 33 | 75 | Phi-4-Reas. | CommonsenseQA | 4 | 73.4234 | 73.3825 | 0.041 |
| 34 | 76 | Phi-4-Reas. | CommonsenseQA | 4 | 74.3243 | 73.2187 | 1.1057 |
| 35 | 77 | Phi-4-Reas. | CommonsenseQA | 4 | 74.3653 | 72.6454 | 1.7199 |
| 36 | 78 | Phi-4-Reas. | CommonsenseQA | 4 | 74.4472 | 72.6863 | 1.7609 |
| 37 | 79 | Phi-4-Reas. | CommonsenseQA | 4 | 74.6929 | 74.1196 | 0.5733 |
| 38 | 80 | Phi-4-Reas. | CommonsenseQA | 4 | 76.6175 | 74.2424 | 2.3751 |
| 39 | 81 | Phi-4-Reas. | CommonsenseQA | 4 | 74.4881 | 73.0549 | 1.4333 |
| 40 | 82 | Phi-4-Reas. | CommonsenseQA | 4 | 74.9386 | 73.2187 | 1.7199 |
| 41 | 83 | Phi-4-Reas. | CommonsenseQA | 4 | 74.57 | 73.2596 | 1.3104 |
| 42 | 84 | Phi-4-Reas. | CommonsenseQA | 4 | 74.3653 | 73.3006 | 1.0647 |
| 43 | 85 | Phi-4-Reas. | CommonsenseQA | 4 | 74.1605 | 73.8329 | 0.3276 |
| 44 | 86 | Phi-4-Reas. | CommonsenseQA | 4 | 74.57 | 73.792 | 0.7781 |
| 45 | 87 | Phi-4-Reas. | CommonsenseQA | 4 | 75.0614 | 73.1368 | 1.9247 |
| 46 | 88 | Phi-4-Reas. | CommonsenseQA | 4 | 74.4881 | 73.4644 | 1.0238 |
| 47 | 89 | Phi-4-Reas. | CommonsenseQA | 4 | 75.0205 | 73.0139 | 2.0066 |
| 48 | 90 | Phi-4-Reas. | CommonsenseQA | 4 | 73.751 | 74.2424 | -0.4914 |
| 49 | 91 | Phi-4-Reas. | CommonsenseQA | 4 | 74.2015 | 74.0377 | 0.1638 |
| 0 | 42 | Phi-4-Reas. | CommonsenseQA | 8 | 74.14 | 74.14 | 0 |
| 1 | 43 | Phi-4-Reas. | CommonsenseQA | 8 | 74.0786 | 74.0786 | 0 |
| 2 | 44 | Phi-4-Reas. | CommonsenseQA | 8 | 74.0786 | 74.4676 | -0.389 |
| 3 | 45 | Phi-4-Reas. | CommonsenseQA | 8 | 74.8567 | 73.6282 | 1.2285 |
| 4 | 46 | Phi-4-Reas. | CommonsenseQA | 8 | 74.8157 | 73.6691 | 1.1466 |
| 5 | 47 | Phi-4-Reas. | CommonsenseQA | 8 | 74.4472 | 73.751 | 0.6962 |
| 6 | 48 | Phi-4-Reas. | CommonsenseQA | 8 | 74.3038 | 73.2187 | 1.0852 |
| 7 | 49 | Phi-4-Reas. | CommonsenseQA | 8 | 74.4062 | 73.8943 | 0.5119 |
| 8 | 50 | Phi-4-Reas. | CommonsenseQA | 8 | 75.1433 | 73.7305 | 1.4128 |
| 9 | 51 | Phi-4-Reas. | CommonsenseQA | 8 | 74.6314 | 73.9353 | 0.6962 |
| 10 | 52 | Phi-4-Reas. | CommonsenseQA | 8 | 74.7748 | 73.7305 | 1.0442 |
| 11 | 53 | Phi-4-Reas. | CommonsenseQA | 8 | 74.4062 | 73.4644 | 0.9419 |
| 12 | 54 | Phi-4-Reas. | CommonsenseQA | 8 | 75 | 73.9353 | 1.0647 |
| 13 | 55 | Phi-4-Reas. | CommonsenseQA | 8 | 74.1605 | 73.5872 | 0.5733 |
| 14 | 56 | Phi-4-Reas. | CommonsenseQA | 8 | 74.8362 | 73.751 | 1.0852 |
| 15 | 57 | Phi-4-Reas. | CommonsenseQA | 8 | 74.57 | 73.6077 | 0.9623 |
| 16 | 58 | Phi-4-Reas. | CommonsenseQA | 8 | 73.751 | 74.2219 | -0.4709 |
| 17 | 59 | Phi-4-Reas. | CommonsenseQA | 8 | 74.9181 | 73.7101 | 1.208 |
| 18 | 60 | Phi-4-Reas. | CommonsenseQA | 8 | 74.0172 | 74.0991 | -0.0819 |
| 19 | 61 | Phi-4-Reas. | CommonsenseQA | 8 | 75.0205 | 73.0549 | 1.9656 |
| 20 | 62 | Phi-4-Reas. | CommonsenseQA | 8 | 75.3276 | 74.2015 | 1.1261 |
| 21 | 63 | Phi-4-Reas. | CommonsenseQA | 8 | 74.8567 | 74.181 | 0.6757 |
| 22 | 64 | Phi-4-Reas. | CommonsenseQA | 8 | 75.2252 | 73.0549 | 2.1704 |
| 23 | 65 | Phi-4-Reas. | CommonsenseQA | 8 | 74.2424 | 73.9967 | 0.2457 |
| 24 | 66 | Phi-4-Reas. | CommonsenseQA | 8 | 74.3448 | 73.5053 | 0.8395 |
| 25 | 67 | Phi-4-Reas. | CommonsenseQA | 8 | 74.5905 | 74.3038 | 0.2867 |
| 26 | 68 | Phi-4-Reas. | CommonsenseQA | 8 | 74.7748 | 73.5667 | 1.208 |
| 27 | 69 | Phi-4-Reas. | CommonsenseQA | 8 | 74.7952 | 73.6691 | 1.1261 |
| 28 | 70 | Phi-4-Reas. | CommonsenseQA | 8 | 74.14 | 73.6486 | 0.4914 |
| 29 | 71 | Phi-4-Reas. | CommonsenseQA | 8 | 75.2252 | 73.7101 | 1.5152 |
| 30 | 72 | Phi-4-Reas. | CommonsenseQA | 8 | 74.959 | 73.792 | 1.1671 |
| 31 | 73 | Phi-4-Reas. | CommonsenseQA | 8 | 74.5495 | 73.0344 | 1.5152 |
| 32 | 74 | Phi-4-Reas. | CommonsenseQA | 8 | 74.7952 | 74.0377 | 0.7576 |
| 33 | 75 | Phi-4-Reas. | CommonsenseQA | 8 | 74.2219 | 73.7305 | 0.4914 |
| 34 | 76 | Phi-4-Reas. | CommonsenseQA | 8 | 74.57 | 73.4439 | 1.1261 |
| 35 | 77 | Phi-4-Reas. | CommonsenseQA | 8 | 74.5291 | 72.6863 | 1.8428 |
| 36 | 78 | Phi-4-Reas. | CommonsenseQA | 8 | 74.1605 | 72.9115 | 1.249 |
| 37 | 79 | Phi-4-Reas. | CommonsenseQA | 8 | 74.4881 | 73.792 | 0.6962 |
| 38 | 80 | Phi-4-Reas. | CommonsenseQA | 8 | 75.7985 | 73.9148 | 1.8837 |
| 39 | 81 | Phi-4-Reas. | CommonsenseQA | 8 | 73.7305 | 73.5667 | 0.1638 |
| 40 | 82 | Phi-4-Reas. | CommonsenseQA | 8 | 75 | 73.4644 | 1.5356 |
| 41 | 83 | Phi-4-Reas. | CommonsenseQA | 8 | 74.4062 | 73.2801 | 1.1261 |
| 42 | 84 | Phi-4-Reas. | CommonsenseQA | 8 | 74.7338 | 73.4234 | 1.3104 |
| 43 | 85 | Phi-4-Reas. | CommonsenseQA | 8 | 74.4267 | 73.9353 | 0.4914 |
| 44 | 86 | Phi-4-Reas. | CommonsenseQA | 8 | 75.1024 | 73.5872 | 1.5152 |
| 45 | 87 | Phi-4-Reas. | CommonsenseQA | 8 | 74.7338 | 73.5258 | 1.208 |
| 46 | 88 | Phi-4-Reas. | CommonsenseQA | 8 | 74.2834 | 73.9967 | 0.2867 |
| 47 | 89 | Phi-4-Reas. | CommonsenseQA | 8 | 74.6519 | 73.6486 | 1.0033 |
| 48 | 90 | Phi-4-Reas. | CommonsenseQA | 8 | 74.959 | 73.321 | 1.638 |
| 49 | 91 | Phi-4-Reas. | CommonsenseQA | 8 | 74.6929 | 74.0172 | 0.6757 |
| 0 | 42 | Phi-4-Reas. | CommonsenseQA | 12 | 74.7884 | 73.9012 | 0.8873 |
| 1 | 43 | Phi-4-Reas. | CommonsenseQA | 12 | 74.5154 | 73.6828 | 0.8327 |
| 2 | 44 | Phi-4-Reas. | CommonsenseQA | 12 | 74.6656 | 73.7374 | 0.9282 |
| 3 | 45 | Phi-4-Reas. | CommonsenseQA | 12 | 74.7748 | 73.4234 | 1.3514 |
| 4 | 46 | Phi-4-Reas. | CommonsenseQA | 12 | 74.5837 | 73.5599 | 1.0238 |
| 5 | 47 | Phi-4-Reas. | CommonsenseQA | 12 | 74.5837 | 73.9148 | 0.6689 |
| 6 | 48 | Phi-4-Reas. | CommonsenseQA | 12 | 74.2424 | 73.9421 | 0.3003 |
| 7 | 49 | Phi-4-Reas. | CommonsenseQA | 12 | 74.6792 | 73.9421 | 0.7371 |
| 8 | 50 | Phi-4-Reas. | CommonsenseQA | 12 | 74.4199 | 73.9012 | 0.5187 |
| 9 | 51 | Phi-4-Reas. | CommonsenseQA | 12 | 74.4335 | 73.9831 | 0.4505 |
| 10 | 52 | Phi-4-Reas. | CommonsenseQA | 12 | 75.0751 | 73.5872 | 1.4879 |
| 11 | 53 | Phi-4-Reas. | CommonsenseQA | 12 | 74.611 | 73.8466 | 0.7644 |
| 12 | 54 | Phi-4-Reas. | CommonsenseQA | 12 | 74.6383 | 73.9831 | 0.6552 |
| 13 | 55 | Phi-4-Reas. | CommonsenseQA | 12 | 74.2015 | 73.5599 | 0.6416 |
| 14 | 56 | Phi-4-Reas. | CommonsenseQA | 12 | 74.7202 | 73.7374 | 0.9828 |
| 15 | 57 | Phi-4-Reas. | CommonsenseQA | 12 | 74.611 | 73.6964 | 0.9146 |
| 16 | 58 | Phi-4-Reas. | CommonsenseQA | 12 | 74.5154 | 73.7783 | 0.7371 |
| 17 | 59 | Phi-4-Reas. | CommonsenseQA | 12 | 74.5291 | 73.5736 | 0.9555 |
| 18 | 60 | Phi-4-Reas. | CommonsenseQA | 12 | 74.1332 | 74.1332 | 0 |
| 19 | 61 | Phi-4-Reas. | CommonsenseQA | 12 | 74.8021 | 73.6145 | 1.1876 |
| 20 | 62 | Phi-4-Reas. | CommonsenseQA | 12 | 74.7202 | 73.8739 | 0.8463 |
| 21 | 63 | Phi-4-Reas. | CommonsenseQA | 12 | 74.57 | 73.7783 | 0.7917 |
| 22 | 64 | Phi-4-Reas. | CommonsenseQA | 12 | 74.884 | 73.3279 | 1.5561 |
| 23 | 65 | Phi-4-Reas. | CommonsenseQA | 12 | 74.3243 | 73.8875 | 0.4368 |
| 24 | 66 | Phi-4-Reas. | CommonsenseQA | 12 | 74.6246 | 73.478 | 1.1466 |
| 25 | 67 | Phi-4-Reas. | CommonsenseQA | 12 | 74.6246 | 74.1196 | 0.5051 |
| 26 | 68 | Phi-4-Reas. | CommonsenseQA | 12 | 74.5973 | 73.8193 | 0.7781 |
| 27 | 69 | Phi-4-Reas. | CommonsenseQA | 12 | 74.57 | 73.751 | 0.819 |
| 28 | 70 | Phi-4-Reas. | CommonsenseQA | 12 | 74.6656 | 73.6691 | 0.9965 |
| 29 | 71 | Phi-4-Reas. | CommonsenseQA | 12 | 74.9386 | 73.6009 | 1.3377 |
| 30 | 72 | Phi-4-Reas. | CommonsenseQA | 12 | 74.4745 | 73.8739 | 0.6006 |
| 31 | 73 | Phi-4-Reas. | CommonsenseQA | 12 | 74.5837 | 73.792 | 0.7917 |
| 32 | 74 | Phi-4-Reas. | CommonsenseQA | 12 | 74.8157 | 74.065 | 0.7508 |
| 33 | 75 | Phi-4-Reas. | CommonsenseQA | 12 | 74.6656 | 73.7237 | 0.9419 |
| 34 | 76 | Phi-4-Reas. | CommonsenseQA | 12 | 74.2834 | 73.5872 | 0.6962 |
| 35 | 77 | Phi-4-Reas. | CommonsenseQA | 12 | 74.3926 | 73.792 | 0.6006 |
| 36 | 78 | Phi-4-Reas. | CommonsenseQA | 12 | 74.5291 | 73.3415 | 1.1876 |
| 37 | 79 | Phi-4-Reas. | CommonsenseQA | 12 | 74.4062 | 73.751 | 0.6552 |
| 38 | 80 | Phi-4-Reas. | CommonsenseQA | 12 | 75.0614 | 73.8466 | 1.2149 |
| 39 | 81 | Phi-4-Reas. | CommonsenseQA | 12 | 74.4335 | 73.6964 | 0.7371 |
| 40 | 82 | Phi-4-Reas. | CommonsenseQA | 12 | 74.5564 | 73.8602 | 0.6962 |
| 41 | 83 | Phi-4-Reas. | CommonsenseQA | 12 | 74.6246 | 73.519 | 1.1057 |
| 42 | 84 | Phi-4-Reas. | CommonsenseQA | 12 | 74.7884 | 73.4234 | 1.365 |
| 43 | 85 | Phi-4-Reas. | CommonsenseQA | 12 | 74.5837 | 73.6555 | 0.9282 |
| 44 | 86 | Phi-4-Reas. | CommonsenseQA | 12 | 75.3071 | 73.3006 | 2.0066 |
| 45 | 87 | Phi-4-Reas. | CommonsenseQA | 12 | 74.7611 | 73.8875 | 0.8736 |
| 46 | 88 | Phi-4-Reas. | CommonsenseQA | 12 | 74.4472 | 73.8602 | 0.587 |
| 47 | 89 | Phi-4-Reas. | CommonsenseQA | 12 | 74.2834 | 73.6418 | 0.6416 |
| 48 | 90 | Phi-4-Reas. | CommonsenseQA | 12 | 74.7884 | 73.3552 | 1.4333 |
| 49 | 91 | Phi-4-Reas. | CommonsenseQA | 12 | 74.4199 | 74.1469 | 0.273 |
| 0 | 42 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 1 | 43 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 2 | 44 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 3 | 45 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 4 | 46 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 5 | 47 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 6 | 48 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 7 | 49 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 8 | 50 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 9 | 51 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 10 | 52 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 11 | 53 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 12 | 54 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 13 | 55 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 14 | 56 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 15 | 57 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 16 | 58 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 17 | 59 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 18 | 60 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 19 | 61 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 20 | 62 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 21 | 63 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 22 | 64 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 23 | 65 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 24 | 66 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 25 | 67 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 26 | 68 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 27 | 69 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 28 | 70 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 29 | 71 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 30 | 72 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 31 | 73 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 32 | 74 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 33 | 75 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 34 | 76 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 35 | 77 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 36 | 78 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 37 | 79 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 38 | 80 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 39 | 81 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 40 | 82 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 41 | 83 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 42 | 84 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 43 | 85 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 44 | 86 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 45 | 87 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 46 | 88 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 47 | 89 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 48 | 90 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 49 | 91 | Phi-4-Reas. | CommonsenseQA | 16 | 74.7338 | 73.6077 | 1.1261 |
| 0 | 42 | Phi-4-Reas. | GPQA | 4 | 50.7812 | 42.7455 | 8.0357 |
| 1 | 43 | Phi-4-Reas. | GPQA | 4 | 48.4375 | 43.0804 | 5.3571 |
| 2 | 44 | Phi-4-Reas. | GPQA | 4 | 48.7723 | 43.0804 | 5.692 |
| 3 | 45 | Phi-4-Reas. | GPQA | 4 | 48.3259 | 42.1875 | 6.1384 |
| 4 | 46 | Phi-4-Reas. | GPQA | 4 | 47.9911 | 46.7634 | 1.2277 |
| 5 | 47 | Phi-4-Reas. | GPQA | 4 | 51.3393 | 45.3125 | 6.0268 |
| 6 | 48 | Phi-4-Reas. | GPQA | 4 | 50.8929 | 42.4107 | 8.4821 |
| 7 | 49 | Phi-4-Reas. | GPQA | 4 | 49.6652 | 43.5268 | 6.1384 |
| 8 | 50 | Phi-4-Reas. | GPQA | 4 | 50.3348 | 43.6384 | 6.6964 |
| 9 | 51 | Phi-4-Reas. | GPQA | 4 | 49.442 | 44.308 | 5.1339 |
| 10 | 52 | Phi-4-Reas. | GPQA | 4 | 49.2188 | 44.308 | 4.9107 |
| 11 | 53 | Phi-4-Reas. | GPQA | 4 | 46.875 | 44.8661 | 2.0089 |
| 12 | 54 | Phi-4-Reas. | GPQA | 4 | 48.5491 | 45.7589 | 2.7902 |
| 13 | 55 | Phi-4-Reas. | GPQA | 4 | 49.8884 | 45.3125 | 4.5759 |
| 14 | 56 | Phi-4-Reas. | GPQA | 4 | 48.7723 | 45.6473 | 3.125 |
| 15 | 57 | Phi-4-Reas. | GPQA | 4 | 48.2143 | 45.6473 | 2.567 |
| 16 | 58 | Phi-4-Reas. | GPQA | 4 | 49.8884 | 42.6339 | 7.2545 |
| 17 | 59 | Phi-4-Reas. | GPQA | 4 | 46.9866 | 44.4196 | 2.567 |
| 18 | 60 | Phi-4-Reas. | GPQA | 4 | 50.6696 | 44.8661 | 5.8036 |
| 19 | 61 | Phi-4-Reas. | GPQA | 4 | 50.6696 | 44.7545 | 5.9152 |
| 20 | 62 | Phi-4-Reas. | GPQA | 4 | 52.0089 | 44.9777 | 7.0312 |
| 21 | 63 | Phi-4-Reas. | GPQA | 4 | 50.7812 | 42.9688 | 7.8125 |
| 22 | 64 | Phi-4-Reas. | GPQA | 4 | 52.2321 | 44.1964 | 8.0357 |
| 23 | 65 | Phi-4-Reas. | GPQA | 4 | 49.6652 | 46.5402 | 3.125 |
| 24 | 66 | Phi-4-Reas. | GPQA | 4 | 51.5625 | 46.4286 | 5.1339 |
| 25 | 67 | Phi-4-Reas. | GPQA | 4 | 50.8929 | 43.0804 | 7.8125 |
| 26 | 68 | Phi-4-Reas. | GPQA | 4 | 50.2232 | 43.8616 | 6.3616 |
| 27 | 69 | Phi-4-Reas. | GPQA | 4 | 48.6607 | 44.8661 | 3.7946 |
| 28 | 70 | Phi-4-Reas. | GPQA | 4 | 50.3348 | 46.6518 | 3.683 |
| 29 | 71 | Phi-4-Reas. | GPQA | 4 | 48.6607 | 43.6384 | 5.0223 |
| 30 | 72 | Phi-4-Reas. | GPQA | 4 | 49.3304 | 45.4241 | 3.9062 |
| 31 | 73 | Phi-4-Reas. | GPQA | 4 | 48.2143 | 44.7545 | 3.4598 |
| 32 | 74 | Phi-4-Reas. | GPQA | 4 | 50.2232 | 44.5312 | 5.692 |
| 33 | 75 | Phi-4-Reas. | GPQA | 4 | 48.5491 | 44.1964 | 4.3527 |
| 34 | 76 | Phi-4-Reas. | GPQA | 4 | 49.8884 | 42.7455 | 7.1429 |
| 35 | 77 | Phi-4-Reas. | GPQA | 4 | 48.5491 | 45.6473 | 2.9018 |
| 36 | 78 | Phi-4-Reas. | GPQA | 4 | 50.3348 | 45.6473 | 4.6875 |
| 37 | 79 | Phi-4-Reas. | GPQA | 4 | 48.6607 | 46.5402 | 2.1205 |
| 38 | 80 | Phi-4-Reas. | GPQA | 4 | 50.2232 | 42.4107 | 7.8125 |
| 39 | 81 | Phi-4-Reas. | GPQA | 4 | 49.5536 | 45.2009 | 4.3527 |
| 40 | 82 | Phi-4-Reas. | GPQA | 4 | 50.4464 | 44.4196 | 6.0268 |
| 41 | 83 | Phi-4-Reas. | GPQA | 4 | 50.1116 | 43.9732 | 6.1384 |
| 42 | 84 | Phi-4-Reas. | GPQA | 4 | 52.6786 | 47.0982 | 5.5804 |
| 43 | 85 | Phi-4-Reas. | GPQA | 4 | 50.3348 | 44.4196 | 5.9152 |
| 44 | 86 | Phi-4-Reas. | GPQA | 4 | 48.8839 | 42.8571 | 6.0268 |
| 45 | 87 | Phi-4-Reas. | GPQA | 4 | 47.5446 | 40.9598 | 6.5848 |
| 46 | 88 | Phi-4-Reas. | GPQA | 4 | 49.3304 | 45.7589 | 3.5714 |
| 47 | 89 | Phi-4-Reas. | GPQA | 4 | 49.3304 | 43.9732 | 5.3571 |
| 48 | 90 | Phi-4-Reas. | GPQA | 4 | 51.8973 | 43.75 | 8.1473 |
| 49 | 91 | Phi-4-Reas. | GPQA | 4 | 51.7857 | 43.192 | 8.5938 |
| 0 | 42 | Phi-4-Reas. | GPQA | 8 | 50.3906 | 45.0893 | 5.3013 |
| 1 | 43 | Phi-4-Reas. | GPQA | 8 | 49.442 | 43.5268 | 5.9152 |
| 2 | 44 | Phi-4-Reas. | GPQA | 8 | 49.9442 | 43.5268 | 6.4174 |
| 3 | 45 | Phi-4-Reas. | GPQA | 8 | 49.3862 | 43.471 | 5.9152 |
| 4 | 46 | Phi-4-Reas. | GPQA | 8 | 48.8839 | 44.4754 | 4.4085 |
| 5 | 47 | Phi-4-Reas. | GPQA | 8 | 50.9487 | 45.1451 | 5.8036 |
| 6 | 48 | Phi-4-Reas. | GPQA | 8 | 51.1161 | 44.1964 | 6.9196 |
| 7 | 49 | Phi-4-Reas. | GPQA | 8 | 49.3862 | 44.6429 | 4.7433 |
| 8 | 50 | Phi-4-Reas. | GPQA | 8 | 50.8929 | 43.192 | 7.7009 |
| 9 | 51 | Phi-4-Reas. | GPQA | 8 | 48.9397 | 45.6473 | 3.2924 |
| 10 | 52 | Phi-4-Reas. | GPQA | 8 | 49.5536 | 45.0893 | 4.4643 |
| 11 | 53 | Phi-4-Reas. | GPQA | 8 | 49.1629 | 44.5312 | 4.6317 |
| 12 | 54 | Phi-4-Reas. | GPQA | 8 | 49.8884 | 46.0379 | 3.8504 |
| 13 | 55 | Phi-4-Reas. | GPQA | 8 | 50.3348 | 43.192 | 7.1429 |
| 14 | 56 | Phi-4-Reas. | GPQA | 8 | 48.6049 | 45.5357 | 3.0692 |
| 15 | 57 | Phi-4-Reas. | GPQA | 8 | 50.8371 | 44.5312 | 6.3058 |
| 16 | 58 | Phi-4-Reas. | GPQA | 8 | 50.558 | 43.9174 | 6.6406 |
| 17 | 59 | Phi-4-Reas. | GPQA | 8 | 49.0513 | 44.9777 | 4.0737 |
| 18 | 60 | Phi-4-Reas. | GPQA | 8 | 51.0603 | 44.5871 | 6.4732 |
| 19 | 61 | Phi-4-Reas. | GPQA | 8 | 50.3906 | 44.5312 | 5.8594 |
| 20 | 62 | Phi-4-Reas. | GPQA | 8 | 50.9487 | 44.9219 | 6.0268 |
| 21 | 63 | Phi-4-Reas. | GPQA | 8 | 50.7812 | 43.9732 | 6.808 |
| 22 | 64 | Phi-4-Reas. | GPQA | 8 | 51.5067 | 43.6384 | 7.8683 |
| 23 | 65 | Phi-4-Reas. | GPQA | 8 | 50 | 45.3125 | 4.6875 |
| 24 | 66 | Phi-4-Reas. | GPQA | 8 | 50.6138 | 44.6429 | 5.971 |
| 25 | 67 | Phi-4-Reas. | GPQA | 8 | 50.3906 | 44.1964 | 6.1942 |
| 26 | 68 | Phi-4-Reas. | GPQA | 8 | 50 | 44.8103 | 5.1897 |
| 27 | 69 | Phi-4-Reas. | GPQA | 8 | 49.9442 | 44.9219 | 5.0223 |
| 28 | 70 | Phi-4-Reas. | GPQA | 8 | 49.3862 | 44.9777 | 4.4085 |
| 29 | 71 | Phi-4-Reas. | GPQA | 8 | 48.1585 | 44.0848 | 4.0737 |
| 30 | 72 | Phi-4-Reas. | GPQA | 8 | 49.9442 | 43.8616 | 6.0826 |
| 31 | 73 | Phi-4-Reas. | GPQA | 8 | 49.7768 | 45.0335 | 4.7433 |
| 32 | 74 | Phi-4-Reas. | GPQA | 8 | 51.5067 | 44.8661 | 6.6406 |
| 33 | 75 | Phi-4-Reas. | GPQA | 8 | 49.1071 | 44.3638 | 4.7433 |
| 34 | 76 | Phi-4-Reas. | GPQA | 8 | 49.3862 | 43.8616 | 5.5246 |
| 35 | 77 | Phi-4-Reas. | GPQA | 8 | 48.8839 | 44.9777 | 3.9062 |
| 36 | 78 | Phi-4-Reas. | GPQA | 8 | 49.4978 | 43.9732 | 5.5246 |
| 37 | 79 | Phi-4-Reas. | GPQA | 8 | 50.279 | 45.0335 | 5.2455 |
| 38 | 80 | Phi-4-Reas. | GPQA | 8 | 50 | 43.471 | 6.529 |
| 39 | 81 | Phi-4-Reas. | GPQA | 8 | 49.6094 | 44.8661 | 4.7433 |
| 40 | 82 | Phi-4-Reas. | GPQA | 8 | 49.2188 | 44.9777 | 4.2411 |
| 41 | 83 | Phi-4-Reas. | GPQA | 8 | 49.8326 | 44.0848 | 5.7478 |
| 42 | 84 | Phi-4-Reas. | GPQA | 8 | 50.3348 | 45.5357 | 4.7991 |
| 43 | 85 | Phi-4-Reas. | GPQA | 8 | 50.6696 | 44.1406 | 6.529 |
| 44 | 86 | Phi-4-Reas. | GPQA | 8 | 49.6652 | 45.4241 | 4.2411 |
| 45 | 87 | Phi-4-Reas. | GPQA | 8 | 48.0469 | 43.0804 | 4.9665 |
| 46 | 88 | Phi-4-Reas. | GPQA | 8 | 49.4978 | 45.9821 | 3.5156 |
| 47 | 89 | Phi-4-Reas. | GPQA | 8 | 50.3906 | 43.3036 | 7.0871 |
| 48 | 90 | Phi-4-Reas. | GPQA | 8 | 49.721 | 44.9219 | 4.7991 |
| 49 | 91 | Phi-4-Reas. | GPQA | 8 | 50.9487 | 44.1964 | 6.7522 |
| 0 | 42 | Phi-4-Reas. | GPQA | 12 | 50 | 44.6429 | 5.3571 |
| 1 | 43 | Phi-4-Reas. | GPQA | 12 | 50.4092 | 43.6756 | 6.7336 |
| 2 | 44 | Phi-4-Reas. | GPQA | 12 | 50.2232 | 44.0848 | 6.1384 |
| 3 | 45 | Phi-4-Reas. | GPQA | 12 | 50.0372 | 45.0521 | 4.9851 |
| 4 | 46 | Phi-4-Reas. | GPQA | 12 | 48.9583 | 45.1637 | 3.7946 |
| 5 | 47 | Phi-4-Reas. | GPQA | 12 | 50 | 44.494 | 5.506 |
| 6 | 48 | Phi-4-Reas. | GPQA | 12 | 50 | 44.1964 | 5.8036 |
| 7 | 49 | Phi-4-Reas. | GPQA | 12 | 50.3348 | 44.6429 | 5.692 |
| 8 | 50 | Phi-4-Reas. | GPQA | 12 | 50.3348 | 43.9732 | 6.3616 |
| 9 | 51 | Phi-4-Reas. | GPQA | 12 | 49.9628 | 44.2336 | 5.7292 |
| 10 | 52 | Phi-4-Reas. | GPQA | 12 | 50.558 | 44.7917 | 5.7664 |
| 11 | 53 | Phi-4-Reas. | GPQA | 12 | 50.1116 | 44.2708 | 5.8408 |
| 12 | 54 | Phi-4-Reas. | GPQA | 12 | 49.7768 | 45.0893 | 4.6875 |
| 13 | 55 | Phi-4-Reas. | GPQA | 12 | 50.2604 | 44.4568 | 5.8036 |
| 14 | 56 | Phi-4-Reas. | GPQA | 12 | 49.6652 | 43.8244 | 5.8408 |
| 15 | 57 | Phi-4-Reas. | GPQA | 12 | 50.4092 | 44.7545 | 5.6548 |
| 16 | 58 | Phi-4-Reas. | GPQA | 12 | 50.186 | 44.7173 | 5.4687 |
| 17 | 59 | Phi-4-Reas. | GPQA | 12 | 49.628 | 44.8661 | 4.7619 |
| 18 | 60 | Phi-4-Reas. | GPQA | 12 | 50.4092 | 44.494 | 5.9152 |
| 19 | 61 | Phi-4-Reas. | GPQA | 12 | 50.0372 | 44.7173 | 5.3199 |
| 20 | 62 | Phi-4-Reas. | GPQA | 12 | 50.9673 | 44.6429 | 6.3244 |
| 21 | 63 | Phi-4-Reas. | GPQA | 12 | 49.7396 | 44.9405 | 4.7991 |
| 22 | 64 | Phi-4-Reas. | GPQA | 12 | 50.5208 | 44.6429 | 5.878 |
| 23 | 65 | Phi-4-Reas. | GPQA | 12 | 49.5536 | 44.9033 | 4.6503 |
| 24 | 66 | Phi-4-Reas. | GPQA | 12 | 50.3348 | 44.6801 | 5.6548 |
| 25 | 67 | Phi-4-Reas. | GPQA | 12 | 49.9256 | 44.9405 | 4.9851 |
| 26 | 68 | Phi-4-Reas. | GPQA | 12 | 50.4464 | 44.5312 | 5.9152 |
| 27 | 69 | Phi-4-Reas. | GPQA | 12 | 50.5208 | 44.494 | 6.0268 |
| 28 | 70 | Phi-4-Reas. | GPQA | 12 | 50.1116 | 44.2336 | 5.878 |
| 29 | 71 | Phi-4-Reas. | GPQA | 12 | 48.5491 | 44.3824 | 4.1667 |
| 30 | 72 | Phi-4-Reas. | GPQA | 12 | 50.0372 | 45.0149 | 5.0223 |
| 31 | 73 | Phi-4-Reas. | GPQA | 12 | 50.2232 | 44.7173 | 5.506 |
| 32 | 74 | Phi-4-Reas. | GPQA | 12 | 50.6324 | 44.7917 | 5.8408 |
| 33 | 75 | Phi-4-Reas. | GPQA | 12 | 49.3304 | 45.0893 | 4.2411 |
| 34 | 76 | Phi-4-Reas. | GPQA | 12 | 50.0372 | 43.6384 | 6.3988 |
| 35 | 77 | Phi-4-Reas. | GPQA | 12 | 49.3676 | 44.6801 | 4.6875 |
| 36 | 78 | Phi-4-Reas. | GPQA | 12 | 49.8884 | 43.8244 | 6.064 |
| 37 | 79 | Phi-4-Reas. | GPQA | 12 | 50.6324 | 44.6429 | 5.9896 |
| 38 | 80 | Phi-4-Reas. | GPQA | 12 | 49.8512 | 44.3824 | 5.4688 |
| 39 | 81 | Phi-4-Reas. | GPQA | 12 | 50.558 | 44.122 | 6.436 |
| 40 | 82 | Phi-4-Reas. | GPQA | 12 | 49.4792 | 44.3824 | 5.0967 |
| 41 | 83 | Phi-4-Reas. | GPQA | 12 | 48.9211 | 44.6801 | 4.2411 |
| 42 | 84 | Phi-4-Reas. | GPQA | 12 | 49.7396 | 44.8289 | 4.9107 |
| 43 | 85 | Phi-4-Reas. | GPQA | 12 | 49.9256 | 44.4196 | 5.506 |
| 44 | 86 | Phi-4-Reas. | GPQA | 12 | 50.1116 | 45.5729 | 4.5387 |
| 45 | 87 | Phi-4-Reas. | GPQA | 12 | 49.5164 | 43.8244 | 5.692 |
| 46 | 88 | Phi-4-Reas. | GPQA | 12 | 49.9256 | 44.7917 | 5.1339 |
| 47 | 89 | Phi-4-Reas. | GPQA | 12 | 49.9256 | 43.936 | 5.9896 |
| 48 | 90 | Phi-4-Reas. | GPQA | 12 | 49.9628 | 44.122 | 5.8408 |
| 49 | 91 | Phi-4-Reas. | GPQA | 12 | 50.0372 | 44.6057 | 5.4315 |
| 0 | 42 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 1 | 43 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 2 | 44 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 3 | 45 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 4 | 46 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 5 | 47 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 6 | 48 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 7 | 49 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 8 | 50 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 9 | 51 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 10 | 52 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 11 | 53 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 12 | 54 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 13 | 55 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 14 | 56 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 15 | 57 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 16 | 58 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 17 | 59 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 18 | 60 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 19 | 61 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 20 | 62 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 21 | 63 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 22 | 64 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 23 | 65 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 24 | 66 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 25 | 67 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 26 | 68 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 27 | 69 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 28 | 70 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 29 | 71 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 30 | 72 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 31 | 73 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 32 | 74 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 33 | 75 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 34 | 76 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 35 | 77 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 36 | 78 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 37 | 79 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 38 | 80 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 39 | 81 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 40 | 82 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 41 | 83 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 42 | 84 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 43 | 85 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 44 | 86 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 45 | 87 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 46 | 88 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 47 | 89 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 48 | 90 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 49 | 91 | Phi-4-Reas. | GPQA | 16 | 50.1116 | 44.5033 | 5.6083 |
| 0 | 42 | Phi-4-Reas. | GSM8K | 4 | 94.9583 | 94.4655 | 0.4928 |
| 1 | 43 | Phi-4-Reas. | GSM8K | 4 | 95.489 | 94.3897 | 1.0993 |
| 2 | 44 | Phi-4-Reas. | GSM8K | 4 | 95.2616 | 94.655 | 0.6065 |
| 3 | 45 | Phi-4-Reas. | GSM8K | 4 | 95.4132 | 94.0106 | 1.4026 |
| 4 | 46 | Phi-4-Reas. | GSM8K | 4 | 95.072 | 94.1243 | 0.9477 |
| 5 | 47 | Phi-4-Reas. | GSM8K | 4 | 95.5269 | 94.655 | 0.8719 |
| 6 | 48 | Phi-4-Reas. | GSM8K | 4 | 94.8446 | 94.2381 | 0.6065 |
| 7 | 49 | Phi-4-Reas. | GSM8K | 4 | 94.9583 | 94.655 | 0.3033 |
| 8 | 50 | Phi-4-Reas. | GSM8K | 4 | 95.072 | 94.4655 | 0.6065 |
| 9 | 51 | Phi-4-Reas. | GSM8K | 4 | 95.6785 | 94.5792 | 1.0993 |
| 10 | 52 | Phi-4-Reas. | GSM8K | 4 | 95.4132 | 94.276 | 1.1372 |
| 11 | 53 | Phi-4-Reas. | GSM8K | 4 | 94.9583 | 94.3518 | 0.6065 |
| 12 | 54 | Phi-4-Reas. | GSM8K | 4 | 94.8825 | 94.1622 | 0.7202 |
| 13 | 55 | Phi-4-Reas. | GSM8K | 4 | 95.5269 | 94.4276 | 1.0993 |
| 14 | 56 | Phi-4-Reas. | GSM8K | 4 | 94.7309 | 94.5034 | 0.2274 |
| 15 | 57 | Phi-4-Reas. | GSM8K | 4 | 95.3753 | 94.5034 | 0.8719 |
| 16 | 58 | Phi-4-Reas. | GSM8K | 4 | 95.4132 | 94.3139 | 1.0993 |
| 17 | 59 | Phi-4-Reas. | GSM8K | 4 | 95.1099 | 94.4276 | 0.6823 |
| 18 | 60 | Phi-4-Reas. | GSM8K | 4 | 95.6027 | 94.5413 | 1.0614 |
| 19 | 61 | Phi-4-Reas. | GSM8K | 4 | 95.1857 | 94.6171 | 0.5686 |
| 20 | 62 | Phi-4-Reas. | GSM8K | 4 | 95.4511 | 94.3139 | 1.1372 |
| 21 | 63 | Phi-4-Reas. | GSM8K | 4 | 95.489 | 94.4655 | 1.0235 |
| 22 | 64 | Phi-4-Reas. | GSM8K | 4 | 95.1478 | 94.3518 | 0.7961 |
| 23 | 65 | Phi-4-Reas. | GSM8K | 4 | 94.9962 | 94.5792 | 0.417 |
| 24 | 66 | Phi-4-Reas. | GSM8K | 4 | 95.3753 | 94.7688 | 0.6065 |
| 25 | 67 | Phi-4-Reas. | GSM8K | 4 | 94.9962 | 94.9962 | 0 |
| 26 | 68 | Phi-4-Reas. | GSM8K | 4 | 95.489 | 94.2002 | 1.2889 |
| 27 | 69 | Phi-4-Reas. | GSM8K | 4 | 95.1478 | 94.3518 | 0.7961 |
| 28 | 70 | Phi-4-Reas. | GSM8K | 4 | 95.5269 | 94.8825 | 0.6444 |
| 29 | 71 | Phi-4-Reas. | GSM8K | 4 | 95.4132 | 94.3139 | 1.0993 |
| 30 | 72 | Phi-4-Reas. | GSM8K | 4 | 95.0341 | 94.5792 | 0.4549 |
| 31 | 73 | Phi-4-Reas. | GSM8K | 4 | 95.0341 | 94.8067 | 0.2274 |
| 32 | 74 | Phi-4-Reas. | GSM8K | 4 | 95.2237 | 94.655 | 0.5686 |
| 33 | 75 | Phi-4-Reas. | GSM8K | 4 | 95.2616 | 94.5034 | 0.7582 |
| 34 | 76 | Phi-4-Reas. | GSM8K | 4 | 95.2616 | 94.0864 | 1.1751 |
| 35 | 77 | Phi-4-Reas. | GSM8K | 4 | 95.2616 | 94.7688 | 0.4928 |
| 36 | 78 | Phi-4-Reas. | GSM8K | 4 | 95.2995 | 94.3897 | 0.9098 |
| 37 | 79 | Phi-4-Reas. | GSM8K | 4 | 95.5269 | 94.5413 | 0.9856 |
| 38 | 80 | Phi-4-Reas. | GSM8K | 4 | 94.8067 | 94.3897 | 0.417 |
| 39 | 81 | Phi-4-Reas. | GSM8K | 4 | 95.1857 | 94.1243 | 1.0614 |
| 40 | 82 | Phi-4-Reas. | GSM8K | 4 | 95.2237 | 94.8067 | 0.417 |
| 41 | 83 | Phi-4-Reas. | GSM8K | 4 | 95.4511 | 94.5792 | 0.8719 |
| 42 | 84 | Phi-4-Reas. | GSM8K | 4 | 95.0341 | 94.5413 | 0.4928 |
| 43 | 85 | Phi-4-Reas. | GSM8K | 4 | 95.1478 | 93.8211 | 1.3268 |
| 44 | 86 | Phi-4-Reas. | GSM8K | 4 | 95.1478 | 94.6929 | 0.4549 |
| 45 | 87 | Phi-4-Reas. | GSM8K | 4 | 95.3374 | 93.9348 | 1.4026 |
| 46 | 88 | Phi-4-Reas. | GSM8K | 4 | 95.489 | 94.655 | 0.834 |
| 47 | 89 | Phi-4-Reas. | GSM8K | 4 | 94.9583 | 94.6929 | 0.2654 |
| 48 | 90 | Phi-4-Reas. | GSM8K | 4 | 95.5269 | 94.3897 | 1.1372 |
| 49 | 91 | Phi-4-Reas. | GSM8K | 4 | 95.1478 | 94.7688 | 0.3791 |
| 0 | 42 | Phi-4-Reas. | GSM8K | 8 | 95.2616 | 94.674 | 0.5876 |
| 1 | 43 | Phi-4-Reas. | GSM8K | 8 | 95.1289 | 94.5982 | 0.5307 |
| 2 | 44 | Phi-4-Reas. | GSM8K | 8 | 95.1099 | 94.5224 | 0.5876 |
| 3 | 45 | Phi-4-Reas. | GSM8K | 8 | 95.2805 | 94.5034 | 0.7771 |
| 4 | 46 | Phi-4-Reas. | GSM8K | 8 | 95.2047 | 94.3139 | 0.8908 |
| 5 | 47 | Phi-4-Reas. | GSM8K | 8 | 95.3563 | 94.6361 | 0.7202 |
| 6 | 48 | Phi-4-Reas. | GSM8K | 8 | 95.072 | 94.5413 | 0.5307 |
| 7 | 49 | Phi-4-Reas. | GSM8K | 8 | 95.2995 | 94.5224 | 0.7771 |
| 8 | 50 | Phi-4-Reas. | GSM8K | 8 | 94.8825 | 94.5224 | 0.3601 |
| 9 | 51 | Phi-4-Reas. | GSM8K | 8 | 95.2995 | 94.6171 | 0.6823 |
| 10 | 52 | Phi-4-Reas. | GSM8K | 8 | 95.2995 | 94.3139 | 0.9856 |
| 11 | 53 | Phi-4-Reas. | GSM8K | 8 | 95.1099 | 94.7119 | 0.398 |
| 12 | 54 | Phi-4-Reas. | GSM8K | 8 | 95.1289 | 94.4086 | 0.7202 |
| 13 | 55 | Phi-4-Reas. | GSM8K | 8 | 95.4701 | 94.5224 | 0.9477 |
| 14 | 56 | Phi-4-Reas. | GSM8K | 8 | 95.2237 | 94.3897 | 0.834 |
| 15 | 57 | Phi-4-Reas. | GSM8K | 8 | 95.2805 | 94.5792 | 0.7013 |
| 16 | 58 | Phi-4-Reas. | GSM8K | 8 | 95.2616 | 94.4086 | 0.8529 |
| 17 | 59 | Phi-4-Reas. | GSM8K | 8 | 95.1857 | 94.5034 | 0.6823 |
| 18 | 60 | Phi-4-Reas. | GSM8K | 8 | 95.4511 | 94.3707 | 1.0804 |
| 19 | 61 | Phi-4-Reas. | GSM8K | 8 | 95.4132 | 94.5224 | 0.8908 |
| 20 | 62 | Phi-4-Reas. | GSM8K | 8 | 95.3563 | 94.5224 | 0.834 |
| 21 | 63 | Phi-4-Reas. | GSM8K | 8 | 95.091 | 94.5224 | 0.5686 |
| 22 | 64 | Phi-4-Reas. | GSM8K | 8 | 95.1099 | 94.4086 | 0.7013 |
| 23 | 65 | Phi-4-Reas. | GSM8K | 8 | 95.1857 | 94.2949 | 0.8908 |
| 24 | 66 | Phi-4-Reas. | GSM8K | 8 | 95.2047 | 94.5982 | 0.6065 |
| 25 | 67 | Phi-4-Reas. | GSM8K | 8 | 95.091 | 94.6929 | 0.398 |
| 26 | 68 | Phi-4-Reas. | GSM8K | 8 | 95.0531 | 94.5603 | 0.4928 |
| 27 | 69 | Phi-4-Reas. | GSM8K | 8 | 95.3563 | 94.2002 | 1.1562 |
| 28 | 70 | Phi-4-Reas. | GSM8K | 8 | 95.2237 | 94.7119 | 0.5118 |
| 29 | 71 | Phi-4-Reas. | GSM8K | 8 | 95.1289 | 94.4845 | 0.6444 |
| 30 | 72 | Phi-4-Reas. | GSM8K | 8 | 95.1478 | 94.6929 | 0.4549 |
| 31 | 73 | Phi-4-Reas. | GSM8K | 8 | 95.1478 | 94.5224 | 0.6255 |
| 32 | 74 | Phi-4-Reas. | GSM8K | 8 | 95.2805 | 94.4466 | 0.834 |
| 33 | 75 | Phi-4-Reas. | GSM8K | 8 | 95.489 | 94.4845 | 1.0045 |
| 34 | 76 | Phi-4-Reas. | GSM8K | 8 | 95.2616 | 94.4276 | 0.834 |
| 35 | 77 | Phi-4-Reas. | GSM8K | 8 | 94.9773 | 94.5224 | 0.4549 |
| 36 | 78 | Phi-4-Reas. | GSM8K | 8 | 95.5459 | 94.257 | 1.2889 |
| 37 | 79 | Phi-4-Reas. | GSM8K | 8 | 95.1289 | 94.5034 | 0.6255 |
| 38 | 80 | Phi-4-Reas. | GSM8K | 8 | 95.2047 | 94.5224 | 0.6823 |
| 39 | 81 | Phi-4-Reas. | GSM8K | 8 | 95.3374 | 94.3897 | 0.9477 |
| 40 | 82 | Phi-4-Reas. | GSM8K | 8 | 95.1668 | 94.5792 | 0.5876 |
| 41 | 83 | Phi-4-Reas. | GSM8K | 8 | 95.3563 | 94.5034 | 0.8529 |
| 42 | 84 | Phi-4-Reas. | GSM8K | 8 | 94.9962 | 94.4276 | 0.5686 |
| 43 | 85 | Phi-4-Reas. | GSM8K | 8 | 95.072 | 94.2949 | 0.7771 |
| 44 | 86 | Phi-4-Reas. | GSM8K | 8 | 95.1857 | 94.6171 | 0.5686 |
| 45 | 87 | Phi-4-Reas. | GSM8K | 8 | 95.2995 | 94.2381 | 1.0614 |
| 46 | 88 | Phi-4-Reas. | GSM8K | 8 | 95.1857 | 94.5224 | 0.6634 |
| 47 | 89 | Phi-4-Reas. | GSM8K | 8 | 95.0152 | 94.5224 | 0.4928 |
| 48 | 90 | Phi-4-Reas. | GSM8K | 8 | 95.2805 | 94.4845 | 0.7961 |
| 49 | 91 | Phi-4-Reas. | GSM8K | 8 | 95.2995 | 94.4655 | 0.834 |
| 0 | 42 | Phi-4-Reas. | GSM8K | 12 | 95.2995 | 94.4276 | 0.8719 |
| 1 | 43 | Phi-4-Reas. | GSM8K | 12 | 95.211 | 94.5413 | 0.6697 |
| 2 | 44 | Phi-4-Reas. | GSM8K | 12 | 95.1478 | 94.6045 | 0.5433 |
| 3 | 45 | Phi-4-Reas. | GSM8K | 12 | 95.2489 | 94.4402 | 0.8087 |
| 4 | 46 | Phi-4-Reas. | GSM8K | 12 | 95.1605 | 94.5034 | 0.6571 |
| 5 | 47 | Phi-4-Reas. | GSM8K | 12 | 95.3121 | 94.5666 | 0.7455 |
| 6 | 48 | Phi-4-Reas. | GSM8K | 12 | 95.1731 | 94.4276 | 0.7455 |
| 7 | 49 | Phi-4-Reas. | GSM8K | 12 | 95.211 | 94.6298 | 0.5812 |
| 8 | 50 | Phi-4-Reas. | GSM8K | 12 | 94.9836 | 94.6045 | 0.3791 |
| 9 | 51 | Phi-4-Reas. | GSM8K | 12 | 95.2489 | 94.554 | 0.695 |
| 10 | 52 | Phi-4-Reas. | GSM8K | 12 | 95.0847 | 94.5413 | 0.5433 |
| 11 | 53 | Phi-4-Reas. | GSM8K | 12 | 95.1352 | 94.554 | 0.5812 |
| 12 | 54 | Phi-4-Reas. | GSM8K | 12 | 95.2363 | 94.5413 | 0.695 |
| 13 | 55 | Phi-4-Reas. | GSM8K | 12 | 95.3879 | 94.5919 | 0.7961 |
| 14 | 56 | Phi-4-Reas. | GSM8K | 12 | 95.1352 | 94.6045 | 0.5307 |
| 15 | 57 | Phi-4-Reas. | GSM8K | 12 | 95.1731 | 94.6424 | 0.5307 |
| 16 | 58 | Phi-4-Reas. | GSM8K | 12 | 95.211 | 94.5792 | 0.6318 |
| 17 | 59 | Phi-4-Reas. | GSM8K | 12 | 95.1605 | 94.5034 | 0.6571 |
| 18 | 60 | Phi-4-Reas. | GSM8K | 12 | 95.2995 | 94.4781 | 0.8213 |
| 19 | 61 | Phi-4-Reas. | GSM8K | 12 | 95.2489 | 94.5792 | 0.6697 |
| 20 | 62 | Phi-4-Reas. | GSM8K | 12 | 95.1857 | 94.4908 | 0.695 |
| 21 | 63 | Phi-4-Reas. | GSM8K | 12 | 95.1984 | 94.6298 | 0.5686 |
| 22 | 64 | Phi-4-Reas. | GSM8K | 12 | 95.1478 | 94.4276 | 0.7202 |
| 23 | 65 | Phi-4-Reas. | GSM8K | 12 | 95.2489 | 94.5034 | 0.7455 |
| 24 | 66 | Phi-4-Reas. | GSM8K | 12 | 95.2237 | 94.5413 | 0.6823 |
| 25 | 67 | Phi-4-Reas. | GSM8K | 12 | 95.1605 | 94.6045 | 0.556 |
| 26 | 68 | Phi-4-Reas. | GSM8K | 12 | 95.1605 | 94.415 | 0.7455 |
| 27 | 69 | Phi-4-Reas. | GSM8K | 12 | 95.2868 | 94.2886 | 0.9982 |
| 28 | 70 | Phi-4-Reas. | GSM8K | 12 | 95.2363 | 94.6424 | 0.5939 |
| 29 | 71 | Phi-4-Reas. | GSM8K | 12 | 95.1984 | 94.6424 | 0.556 |
| 30 | 72 | Phi-4-Reas. | GSM8K | 12 | 95.1984 | 94.5792 | 0.6192 |
| 31 | 73 | Phi-4-Reas. | GSM8K | 12 | 95.2237 | 94.415 | 0.8087 |
| 32 | 74 | Phi-4-Reas. | GSM8K | 12 | 95.2237 | 94.4023 | 0.8213 |
| 33 | 75 | Phi-4-Reas. | GSM8K | 12 | 95.3121 | 94.554 | 0.7582 |
| 34 | 76 | Phi-4-Reas. | GSM8K | 12 | 95.2237 | 94.4908 | 0.7329 |
| 35 | 77 | Phi-4-Reas. | GSM8K | 12 | 95.1352 | 94.5792 | 0.556 |
| 36 | 78 | Phi-4-Reas. | GSM8K | 12 | 95.211 | 94.4529 | 0.7582 |
| 37 | 79 | Phi-4-Reas. | GSM8K | 12 | 95.0594 | 94.4655 | 0.5939 |
| 38 | 80 | Phi-4-Reas. | GSM8K | 12 | 95.0847 | 94.4402 | 0.6444 |
| 39 | 81 | Phi-4-Reas. | GSM8K | 12 | 95.1478 | 94.516 | 0.6318 |
| 40 | 82 | Phi-4-Reas. | GSM8K | 12 | 95.2742 | 94.6298 | 0.6444 |
| 41 | 83 | Phi-4-Reas. | GSM8K | 12 | 95.1731 | 94.516 | 0.6571 |
| 42 | 84 | Phi-4-Reas. | GSM8K | 12 | 95.2363 | 94.5792 | 0.6571 |
| 43 | 85 | Phi-4-Reas. | GSM8K | 12 | 95.2237 | 94.3771 | 0.8466 |
| 44 | 86 | Phi-4-Reas. | GSM8K | 12 | 95.2742 | 94.5287 | 0.7455 |
| 45 | 87 | Phi-4-Reas. | GSM8K | 12 | 95.2742 | 94.5287 | 0.7455 |
| 46 | 88 | Phi-4-Reas. | GSM8K | 12 | 95.2489 | 94.4529 | 0.7961 |
| 47 | 89 | Phi-4-Reas. | GSM8K | 12 | 95.1731 | 94.4276 | 0.7455 |
| 48 | 90 | Phi-4-Reas. | GSM8K | 12 | 95.3374 | 94.5287 | 0.8087 |
| 49 | 91 | Phi-4-Reas. | GSM8K | 12 | 95.1731 | 94.516 | 0.6571 |
| 0 | 42 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 1 | 43 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 2 | 44 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 3 | 45 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 4 | 46 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 5 | 47 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 6 | 48 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 7 | 49 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 8 | 50 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 9 | 51 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 10 | 52 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 11 | 53 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 12 | 54 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 13 | 55 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 14 | 56 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 15 | 57 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 16 | 58 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 17 | 59 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 18 | 60 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 19 | 61 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 20 | 62 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 21 | 63 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 22 | 64 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 23 | 65 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 24 | 66 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 25 | 67 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 26 | 68 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 27 | 69 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 28 | 70 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 29 | 71 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 30 | 72 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 31 | 73 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 32 | 74 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 33 | 75 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 34 | 76 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 35 | 77 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 36 | 78 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 37 | 79 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 38 | 80 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 39 | 81 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 40 | 82 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 41 | 83 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 42 | 84 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 43 | 85 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 44 | 86 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 45 | 87 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 46 | 88 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 47 | 89 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 48 | 90 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 49 | 91 | Phi-4-Reas. | GSM8K | 16 | 95.2331 | 94.4939 | 0.7392 |
| 0 | 42 | Phi-4-Reas. | MATH500 | 4 | 85.2 | 85.6 | -0.4 |
| 1 | 43 | Phi-4-Reas. | MATH500 | 4 | 85.6 | 83 | 2.6 |
| 2 | 44 | Phi-4-Reas. | MATH500 | 4 | 85.6 | 84.9 | 0.7 |
| 3 | 45 | Phi-4-Reas. | MATH500 | 4 | 85.6 | 84.3 | 1.3 |
| 4 | 46 | Phi-4-Reas. | MATH500 | 4 | 84.3 | 84.1 | 0.2 |
| 5 | 47 | Phi-4-Reas. | MATH500 | 4 | 84.8 | 84.8 | 0 |
| 6 | 48 | Phi-4-Reas. | MATH500 | 4 | 84.4 | 85 | -0.6 |
| 7 | 49 | Phi-4-Reas. | MATH500 | 4 | 85.4 | 84.1 | 1.3 |
| 8 | 50 | Phi-4-Reas. | MATH500 | 4 | 85 | 84.3 | 0.7 |
| 9 | 51 | Phi-4-Reas. | MATH500 | 4 | 86.3 | 85.3 | 1 |
| 10 | 52 | Phi-4-Reas. | MATH500 | 4 | 85.6 | 84.3 | 1.3 |
| 11 | 53 | Phi-4-Reas. | MATH500 | 4 | 84.4 | 84.6 | -0.2 |
| 12 | 54 | Phi-4-Reas. | MATH500 | 4 | 85.3 | 84.4 | 0.9 |
| 13 | 55 | Phi-4-Reas. | MATH500 | 4 | 85.2 | 85.4 | -0.2 |
| 14 | 56 | Phi-4-Reas. | MATH500 | 4 | 85.1 | 84.4 | 0.7 |
| 15 | 57 | Phi-4-Reas. | MATH500 | 4 | 85.8 | 84.8 | 1 |
| 16 | 58 | Phi-4-Reas. | MATH500 | 4 | 85.4 | 84.8 | 0.6 |
| 17 | 59 | Phi-4-Reas. | MATH500 | 4 | 85.5 | 83.9 | 1.6 |
| 18 | 60 | Phi-4-Reas. | MATH500 | 4 | 83.8 | 85.7 | -1.9 |
| 19 | 61 | Phi-4-Reas. | MATH500 | 4 | 84.6 | 85.7 | -1.1 |
| 20 | 62 | Phi-4-Reas. | MATH500 | 4 | 84.7 | 84 | 0.7 |
| 21 | 63 | Phi-4-Reas. | MATH500 | 4 | 85.3 | 83 | 2.3 |
| 22 | 64 | Phi-4-Reas. | MATH500 | 4 | 84.9 | 85 | -0.1 |
| 23 | 65 | Phi-4-Reas. | MATH500 | 4 | 85.7 | 83.9 | 1.8 |
| 24 | 66 | Phi-4-Reas. | MATH500 | 4 | 84.7 | 84.5 | 0.2 |
| 25 | 67 | Phi-4-Reas. | MATH500 | 4 | 85.4 | 83.9 | 1.5 |
| 26 | 68 | Phi-4-Reas. | MATH500 | 4 | 84.4 | 83.8 | 0.6 |
| 27 | 69 | Phi-4-Reas. | MATH500 | 4 | 85.9 | 84.3 | 1.6 |
| 28 | 70 | Phi-4-Reas. | MATH500 | 4 | 85.1 | 83.1 | 2 |
| 29 | 71 | Phi-4-Reas. | MATH500 | 4 | 84.9 | 85 | -0.1 |
| 30 | 72 | Phi-4-Reas. | MATH500 | 4 | 85.7 | 84.8 | 0.9 |
| 31 | 73 | Phi-4-Reas. | MATH500 | 4 | 84.8 | 84.5 | 0.3 |
| 32 | 74 | Phi-4-Reas. | MATH500 | 4 | 85.2 | 83.5 | 1.7 |
| 33 | 75 | Phi-4-Reas. | MATH500 | 4 | 84.2 | 84.3 | -0.1 |
| 34 | 76 | Phi-4-Reas. | MATH500 | 4 | 85.2 | 84.5 | 0.7 |
| 35 | 77 | Phi-4-Reas. | MATH500 | 4 | 85.7 | 84.3 | 1.4 |
| 36 | 78 | Phi-4-Reas. | MATH500 | 4 | 84.1 | 85.3 | -1.2 |
| 37 | 79 | Phi-4-Reas. | MATH500 | 4 | 84.9 | 83.5 | 1.4 |
| 38 | 80 | Phi-4-Reas. | MATH500 | 4 | 84.9 | 83.5 | 1.4 |
| 39 | 81 | Phi-4-Reas. | MATH500 | 4 | 86.2 | 84.6 | 1.6 |
| 40 | 82 | Phi-4-Reas. | MATH500 | 4 | 85.1 | 84.6 | 0.5 |
| 41 | 83 | Phi-4-Reas. | MATH500 | 4 | 85 | 83.4 | 1.6 |
| 42 | 84 | Phi-4-Reas. | MATH500 | 4 | 85.5 | 83.6 | 1.9 |
| 43 | 85 | Phi-4-Reas. | MATH500 | 4 | 84.1 | 85.8 | -1.7 |
| 44 | 86 | Phi-4-Reas. | MATH500 | 4 | 84.8 | 84 | 0.8 |
| 45 | 87 | Phi-4-Reas. | MATH500 | 4 | 85.7 | 84.5 | 1.2 |
| 46 | 88 | Phi-4-Reas. | MATH500 | 4 | 84.7 | 84.1 | 0.6 |
| 47 | 89 | Phi-4-Reas. | MATH500 | 4 | 84.9 | 85.2 | -0.3 |
| 48 | 90 | Phi-4-Reas. | MATH500 | 4 | 85.2 | 84.5 | 0.7 |
| 49 | 91 | Phi-4-Reas. | MATH500 | 4 | 86.3 | 85.1 | 1.2 |
| 0 | 42 | Phi-4-Reas. | MATH500 | 8 | 85.4 | 84.8 | 0.6 |
| 1 | 43 | Phi-4-Reas. | MATH500 | 8 | 85.6 | 83.15 | 2.45 |
| 2 | 44 | Phi-4-Reas. | MATH500 | 8 | 84.9 | 84.3 | 0.6 |
| 3 | 45 | Phi-4-Reas. | MATH500 | 8 | 85.4 | 84.05 | 1.35 |
| 4 | 46 | Phi-4-Reas. | MATH500 | 8 | 85.2 | 84.4 | 0.8 |
| 5 | 47 | Phi-4-Reas. | MATH500 | 8 | 85.7 | 83.7 | 2 |
| 6 | 48 | Phi-4-Reas. | MATH500 | 8 | 85.15 | 84.2 | 0.95 |
| 7 | 49 | Phi-4-Reas. | MATH500 | 8 | 85 | 84.05 | 0.95 |
| 8 | 50 | Phi-4-Reas. | MATH500 | 8 | 85.4 | 84.5 | 0.9 |
| 9 | 51 | Phi-4-Reas. | MATH500 | 8 | 85.55 | 84.9 | 0.65 |
| 10 | 52 | Phi-4-Reas. | MATH500 | 8 | 85.5 | 84.45 | 1.05 |
| 11 | 53 | Phi-4-Reas. | MATH500 | 8 | 85.3 | 83.95 | 1.35 |
| 12 | 54 | Phi-4-Reas. | MATH500 | 8 | 85.1 | 84.2 | 0.9 |
| 13 | 55 | Phi-4-Reas. | MATH500 | 8 | 86 | 84.35 | 1.65 |
| 14 | 56 | Phi-4-Reas. | MATH500 | 8 | 85.5 | 83.95 | 1.55 |
| 15 | 57 | Phi-4-Reas. | MATH500 | 8 | 85.45 | 84.5 | 0.95 |
| 16 | 58 | Phi-4-Reas. | MATH500 | 8 | 85.55 | 84.25 | 1.3 |
| 17 | 59 | Phi-4-Reas. | MATH500 | 8 | 84.9 | 84.3 | 0.6 |
| 18 | 60 | Phi-4-Reas. | MATH500 | 8 | 84.7 | 84.4 | 0.3 |
| 19 | 61 | Phi-4-Reas. | MATH500 | 8 | 85.3 | 84.8 | 0.5 |
| 20 | 62 | Phi-4-Reas. | MATH500 | 8 | 84.65 | 84.15 | 0.5 |
| 21 | 63 | Phi-4-Reas. | MATH500 | 8 | 85.55 | 83.8 | 1.75 |
| 22 | 64 | Phi-4-Reas. | MATH500 | 8 | 85.15 | 84.45 | 0.7 |
| 23 | 65 | Phi-4-Reas. | MATH500 | 8 | 85.25 | 84.35 | 0.9 |
| 24 | 66 | Phi-4-Reas. | MATH500 | 8 | 84.8 | 84.1 | 0.7 |
| 25 | 67 | Phi-4-Reas. | MATH500 | 8 | 85.25 | 84.55 | 0.7 |
| 26 | 68 | Phi-4-Reas. | MATH500 | 8 | 84.95 | 84.1 | 0.85 |
| 27 | 69 | Phi-4-Reas. | MATH500 | 8 | 85.3 | 83.9 | 1.4 |
| 28 | 70 | Phi-4-Reas. | MATH500 | 8 | 84.7 | 83.8 | 0.9 |
| 29 | 71 | Phi-4-Reas. | MATH500 | 8 | 84.45 | 84.7 | -0.25 |
| 30 | 72 | Phi-4-Reas. | MATH500 | 8 | 85.6 | 84.8 | 0.8 |
| 31 | 73 | Phi-4-Reas. | MATH500 | 8 | 85.25 | 84.15 | 1.1 |
| 32 | 74 | Phi-4-Reas. | MATH500 | 8 | 85.4 | 84.25 | 1.15 |
| 33 | 75 | Phi-4-Reas. | MATH500 | 8 | 84.9 | 83.6 | 1.3 |
| 34 | 76 | Phi-4-Reas. | MATH500 | 8 | 85.15 | 84.95 | 0.2 |
| 35 | 77 | Phi-4-Reas. | MATH500 | 8 | 85.9 | 84.05 | 1.85 |
| 36 | 78 | Phi-4-Reas. | MATH500 | 8 | 85.25 | 84.2 | 1.05 |
| 37 | 79 | Phi-4-Reas. | MATH500 | 8 | 85.55 | 83.55 | 2 |
| 38 | 80 | Phi-4-Reas. | MATH500 | 8 | 85.05 | 83.7 | 1.35 |
| 39 | 81 | Phi-4-Reas. | MATH500 | 8 | 85.65 | 84.45 | 1.2 |
| 40 | 82 | Phi-4-Reas. | MATH500 | 8 | 85.5 | 84 | 1.5 |
| 41 | 83 | Phi-4-Reas. | MATH500 | 8 | 85.2 | 83.8 | 1.4 |
| 42 | 84 | Phi-4-Reas. | MATH500 | 8 | 85.45 | 83.9 | 1.55 |
| 43 | 85 | Phi-4-Reas. | MATH500 | 8 | 85.25 | 84 | 1.25 |
| 44 | 86 | Phi-4-Reas. | MATH500 | 8 | 85.4 | 83.85 | 1.55 |
| 45 | 87 | Phi-4-Reas. | MATH500 | 8 | 84.85 | 84.45 | 0.4 |
| 46 | 88 | Phi-4-Reas. | MATH500 | 8 | 85.3 | 83.95 | 1.35 |
| 47 | 89 | Phi-4-Reas. | MATH500 | 8 | 85.45 | 84.65 | 0.8 |
| 48 | 90 | Phi-4-Reas. | MATH500 | 8 | 85.45 | 84.25 | 1.2 |
| 49 | 91 | Phi-4-Reas. | MATH500 | 8 | 85.7 | 83.8 | 1.9 |
| 0 | 42 | Phi-4-Reas. | MATH500 | 12 | 85.2667 | 84.2 | 1.0667 |
| 1 | 43 | Phi-4-Reas. | MATH500 | 12 | 85.5 | 83.7667 | 1.7333 |
| 2 | 44 | Phi-4-Reas. | MATH500 | 12 | 85.2333 | 84.0667 | 1.1667 |
| 3 | 45 | Phi-4-Reas. | MATH500 | 12 | 85.2333 | 83.7667 | 1.4667 |
| 4 | 46 | Phi-4-Reas. | MATH500 | 12 | 85.6 | 84.0333 | 1.5667 |
| 5 | 47 | Phi-4-Reas. | MATH500 | 12 | 85.4 | 83.6667 | 1.7333 |
| 6 | 48 | Phi-4-Reas. | MATH500 | 12 | 85.4667 | 84.0667 | 1.4 |
| 7 | 49 | Phi-4-Reas. | MATH500 | 12 | 85.2 | 84.1333 | 1.0667 |
| 8 | 50 | Phi-4-Reas. | MATH500 | 12 | 85.4333 | 84.1667 | 1.2667 |
| 9 | 51 | Phi-4-Reas. | MATH500 | 12 | 85.3667 | 84.2 | 1.1667 |
| 10 | 52 | Phi-4-Reas. | MATH500 | 12 | 85.4333 | 84.3333 | 1.1 |
| 11 | 53 | Phi-4-Reas. | MATH500 | 12 | 84.8333 | 84 | 0.8333 |
| 12 | 54 | Phi-4-Reas. | MATH500 | 12 | 85.0667 | 84.2667 | 0.8 |
| 13 | 55 | Phi-4-Reas. | MATH500 | 12 | 85.7667 | 83.7 | 2.0667 |
| 14 | 56 | Phi-4-Reas. | MATH500 | 12 | 85.5333 | 83.8667 | 1.6667 |
| 15 | 57 | Phi-4-Reas. | MATH500 | 12 | 85.3 | 84.3667 | 0.9333 |
| 16 | 58 | Phi-4-Reas. | MATH500 | 12 | 85.3333 | 84.3333 | 1 |
| 17 | 59 | Phi-4-Reas. | MATH500 | 12 | 85.5333 | 84 | 1.5333 |
| 18 | 60 | Phi-4-Reas. | MATH500 | 12 | 84.9333 | 84.5 | 0.4333 |
| 19 | 61 | Phi-4-Reas. | MATH500 | 12 | 85.2333 | 84 | 1.2333 |
| 20 | 62 | Phi-4-Reas. | MATH500 | 12 | 85.1333 | 84.1 | 1.0333 |
| 21 | 63 | Phi-4-Reas. | MATH500 | 12 | 85.1333 | 84 | 1.1333 |
| 22 | 64 | Phi-4-Reas. | MATH500 | 12 | 85.3667 | 83.9 | 1.4667 |
| 23 | 65 | Phi-4-Reas. | MATH500 | 12 | 85.5333 | 83.9 | 1.6333 |
| 24 | 66 | Phi-4-Reas. | MATH500 | 12 | 85.5667 | 83.9333 | 1.6333 |
| 25 | 67 | Phi-4-Reas. | MATH500 | 12 | 85.3 | 83.7 | 1.6 |
| 26 | 68 | Phi-4-Reas. | MATH500 | 12 | 85.3 | 84.2667 | 1.0333 |
| 27 | 69 | Phi-4-Reas. | MATH500 | 12 | 85.2333 | 83.7667 | 1.4667 |
| 28 | 70 | Phi-4-Reas. | MATH500 | 12 | 85.4667 | 83.9 | 1.5667 |
| 29 | 71 | Phi-4-Reas. | MATH500 | 12 | 85.4333 | 84.0667 | 1.3667 |
| 30 | 72 | Phi-4-Reas. | MATH500 | 12 | 85.9 | 84.1 | 1.8 |
| 31 | 73 | Phi-4-Reas. | MATH500 | 12 | 85.4333 | 84.1 | 1.3333 |
| 32 | 74 | Phi-4-Reas. | MATH500 | 12 | 85.4 | 84 | 1.4 |
| 33 | 75 | Phi-4-Reas. | MATH500 | 12 | 85.2333 | 83.8667 | 1.3667 |
| 34 | 76 | Phi-4-Reas. | MATH500 | 12 | 85.4667 | 84.2 | 1.2667 |
| 35 | 77 | Phi-4-Reas. | MATH500 | 12 | 85.6333 | 83.6333 | 2 |
| 36 | 78 | Phi-4-Reas. | MATH500 | 12 | 85.5 | 83.9667 | 1.5333 |
| 37 | 79 | Phi-4-Reas. | MATH500 | 12 | 85.4 | 84.1333 | 1.2667 |
| 38 | 80 | Phi-4-Reas. | MATH500 | 12 | 85.0667 | 83.9667 | 1.1 |
| 39 | 81 | Phi-4-Reas. | MATH500 | 12 | 85.7333 | 84 | 1.7333 |
| 40 | 82 | Phi-4-Reas. | MATH500 | 12 | 85.5667 | 84.1333 | 1.4333 |
| 41 | 83 | Phi-4-Reas. | MATH500 | 12 | 85.5 | 83.9 | 1.6 |
| 42 | 84 | Phi-4-Reas. | MATH500 | 12 | 85.5667 | 83.8 | 1.7667 |
| 43 | 85 | Phi-4-Reas. | MATH500 | 12 | 85.4667 | 83.9 | 1.5667 |
| 44 | 86 | Phi-4-Reas. | MATH500 | 12 | 85.3 | 84.1 | 1.2 |
| 45 | 87 | Phi-4-Reas. | MATH500 | 12 | 85.4333 | 84.0333 | 1.4 |
| 46 | 88 | Phi-4-Reas. | MATH500 | 12 | 85.4 | 83.7 | 1.7 |
| 47 | 89 | Phi-4-Reas. | MATH500 | 12 | 85.4 | 84.1667 | 1.2333 |
| 48 | 90 | Phi-4-Reas. | MATH500 | 12 | 85.6667 | 84.0333 | 1.6333 |
| 49 | 91 | Phi-4-Reas. | MATH500 | 12 | 85.6667 | 83.9333 | 1.7333 |
| 0 | 42 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 1 | 43 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 2 | 44 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 3 | 45 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 4 | 46 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 5 | 47 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 6 | 48 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 7 | 49 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 8 | 50 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 9 | 51 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 10 | 52 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 11 | 53 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 12 | 54 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 13 | 55 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 14 | 56 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 15 | 57 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 16 | 58 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 17 | 59 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 18 | 60 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 19 | 61 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 20 | 62 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 21 | 63 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 22 | 64 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 23 | 65 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 24 | 66 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 25 | 67 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 26 | 68 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 27 | 69 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 28 | 70 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 29 | 71 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 30 | 72 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 31 | 73 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 32 | 74 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 33 | 75 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 34 | 76 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 35 | 77 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 36 | 78 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 37 | 79 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 38 | 80 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 39 | 81 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 40 | 82 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 41 | 83 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 42 | 84 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 43 | 85 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 44 | 86 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 45 | 87 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 46 | 88 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 47 | 89 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 48 | 90 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 49 | 91 | Phi-4-Reas. | MATH500 | 16 | 85.475 | 83.95 | 1.525 |
| 0 | 42 | Phi-4-Reas. | SVAMP | 4 | 94.05 | 94 | 0.05 |
| 1 | 43 | Phi-4-Reas. | SVAMP | 4 | 94.05 | 94.3 | -0.25 |
| 2 | 44 | Phi-4-Reas. | SVAMP | 4 | 94.05 | 94.3 | -0.25 |
| 3 | 45 | Phi-4-Reas. | SVAMP | 4 | 93.95 | 94.45 | -0.5 |
| 4 | 46 | Phi-4-Reas. | SVAMP | 4 | 94.25 | 94.6 | -0.35 |
| 5 | 47 | Phi-4-Reas. | SVAMP | 4 | 94.3 | 94.05 | 0.25 |
| 6 | 48 | Phi-4-Reas. | SVAMP | 4 | 94.35 | 94.15 | 0.2 |
| 7 | 49 | Phi-4-Reas. | SVAMP | 4 | 94 | 94.8 | -0.8 |
| 8 | 50 | Phi-4-Reas. | SVAMP | 4 | 94.1 | 94.2 | -0.1 |
| 9 | 51 | Phi-4-Reas. | SVAMP | 4 | 94.4 | 94.75 | -0.35 |
| 10 | 52 | Phi-4-Reas. | SVAMP | 4 | 94.25 | 94.35 | -0.1 |
| 11 | 53 | Phi-4-Reas. | SVAMP | 4 | 94 | 94.55 | -0.55 |
| 12 | 54 | Phi-4-Reas. | SVAMP | 4 | 93.6 | 94.85 | -1.25 |
| 13 | 55 | Phi-4-Reas. | SVAMP | 4 | 93.8 | 94.35 | -0.55 |
| 14 | 56 | Phi-4-Reas. | SVAMP | 4 | 93.8 | 94.45 | -0.65 |
| 15 | 57 | Phi-4-Reas. | SVAMP | 4 | 94.15 | 94.2 | -0.05 |
| 16 | 58 | Phi-4-Reas. | SVAMP | 4 | 93.85 | 94.35 | -0.5 |
| 17 | 59 | Phi-4-Reas. | SVAMP | 4 | 93.85 | 94.75 | -0.9 |
| 18 | 60 | Phi-4-Reas. | SVAMP | 4 | 94 | 94.75 | -0.75 |
| 19 | 61 | Phi-4-Reas. | SVAMP | 4 | 94.05 | 94.75 | -0.7 |
| 20 | 62 | Phi-4-Reas. | SVAMP | 4 | 94.2 | 94.55 | -0.35 |
| 21 | 63 | Phi-4-Reas. | SVAMP | 4 | 94.25 | 93.85 | 0.4 |
| 22 | 64 | Phi-4-Reas. | SVAMP | 4 | 94.2 | 94.25 | -0.05 |
| 23 | 65 | Phi-4-Reas. | SVAMP | 4 | 94.3 | 94 | 0.3 |
| 24 | 66 | Phi-4-Reas. | SVAMP | 4 | 93.9 | 94.55 | -0.65 |
| 25 | 67 | Phi-4-Reas. | SVAMP | 4 | 94.3 | 94.45 | -0.15 |
| 26 | 68 | Phi-4-Reas. | SVAMP | 4 | 94.25 | 94.3 | -0.05 |
| 27 | 69 | Phi-4-Reas. | SVAMP | 4 | 94.45 | 94.65 | -0.2 |
| 28 | 70 | Phi-4-Reas. | SVAMP | 4 | 93.9 | 94.9 | -1 |
| 29 | 71 | Phi-4-Reas. | SVAMP | 4 | 94.15 | 94.3 | -0.15 |
| 30 | 72 | Phi-4-Reas. | SVAMP | 4 | 94.25 | 93.8 | 0.45 |
| 31 | 73 | Phi-4-Reas. | SVAMP | 4 | 93.5 | 94.65 | -1.15 |
| 32 | 74 | Phi-4-Reas. | SVAMP | 4 | 93.9 | 94.25 | -0.35 |
| 33 | 75 | Phi-4-Reas. | SVAMP | 4 | 94.2 | 94.6 | -0.4 |
| 34 | 76 | Phi-4-Reas. | SVAMP | 4 | 93.6 | 94.7 | -1.1 |
| 35 | 77 | Phi-4-Reas. | SVAMP | 4 | 94.15 | 94.3 | -0.15 |
| 36 | 78 | Phi-4-Reas. | SVAMP | 4 | 94.15 | 94.6 | -0.45 |
| 37 | 79 | Phi-4-Reas. | SVAMP | 4 | 94.55 | 94.65 | -0.1 |
| 38 | 80 | Phi-4-Reas. | SVAMP | 4 | 94.35 | 94.55 | -0.2 |
| 39 | 81 | Phi-4-Reas. | SVAMP | 4 | 94.05 | 94.3 | -0.25 |
| 40 | 82 | Phi-4-Reas. | SVAMP | 4 | 94.1 | 94.15 | -0.05 |
| 41 | 83 | Phi-4-Reas. | SVAMP | 4 | 94.4 | 94.25 | 0.15 |
| 42 | 84 | Phi-4-Reas. | SVAMP | 4 | 94.35 | 93.85 | 0.5 |
| 43 | 85 | Phi-4-Reas. | SVAMP | 4 | 94.15 | 94.2 | -0.05 |
| 44 | 86 | Phi-4-Reas. | SVAMP | 4 | 94.5 | 94.8 | -0.3 |
| 45 | 87 | Phi-4-Reas. | SVAMP | 4 | 93.75 | 94.75 | -1 |
| 46 | 88 | Phi-4-Reas. | SVAMP | 4 | 94.1 | 95.05 | -0.95 |
| 47 | 89 | Phi-4-Reas. | SVAMP | 4 | 94.7 | 94.05 | 0.65 |
| 48 | 90 | Phi-4-Reas. | SVAMP | 4 | 93.15 | 94.95 | -1.8 |
| 49 | 91 | Phi-4-Reas. | SVAMP | 4 | 94.05 | 94.15 | -0.1 |
| 0 | 42 | Phi-4-Reas. | SVAMP | 8 | 94.1 | 94.075 | 0.025 |
| 1 | 43 | Phi-4-Reas. | SVAMP | 8 | 93.925 | 94.475 | -0.55 |
| 2 | 44 | Phi-4-Reas. | SVAMP | 8 | 94.275 | 94.5 | -0.225 |
| 3 | 45 | Phi-4-Reas. | SVAMP | 8 | 93.9 | 94.45 | -0.55 |
| 4 | 46 | Phi-4-Reas. | SVAMP | 8 | 94.125 | 94.525 | -0.4 |
| 5 | 47 | Phi-4-Reas. | SVAMP | 8 | 93.9 | 94.4 | -0.5 |
| 6 | 48 | Phi-4-Reas. | SVAMP | 8 | 94.15 | 94.65 | -0.5 |
| 7 | 49 | Phi-4-Reas. | SVAMP | 8 | 94.075 | 94.85 | -0.775 |
| 8 | 50 | Phi-4-Reas. | SVAMP | 8 | 94 | 94.425 | -0.425 |
| 9 | 51 | Phi-4-Reas. | SVAMP | 8 | 94.4 | 94.6 | -0.2 |
| 10 | 52 | Phi-4-Reas. | SVAMP | 8 | 94.25 | 94.45 | -0.2 |
| 11 | 53 | Phi-4-Reas. | SVAMP | 8 | 93.95 | 94.575 | -0.625 |
| 12 | 54 | Phi-4-Reas. | SVAMP | 8 | 93.95 | 94.525 | -0.575 |
| 13 | 55 | Phi-4-Reas. | SVAMP | 8 | 93.925 | 94.4 | -0.475 |
| 14 | 56 | Phi-4-Reas. | SVAMP | 8 | 94.275 | 94.5 | -0.225 |
| 15 | 57 | Phi-4-Reas. | SVAMP | 8 | 93.8 | 94.6 | -0.8 |
| 16 | 58 | Phi-4-Reas. | SVAMP | 8 | 93.975 | 94.65 | -0.675 |
| 17 | 59 | Phi-4-Reas. | SVAMP | 8 | 94.025 | 94.55 | -0.525 |
| 18 | 60 | Phi-4-Reas. | SVAMP | 8 | 94.075 | 94.475 | -0.4 |
| 19 | 61 | Phi-4-Reas. | SVAMP | 8 | 93.925 | 94.5 | -0.575 |
| 20 | 62 | Phi-4-Reas. | SVAMP | 8 | 94.15 | 94.475 | -0.325 |
| 21 | 63 | Phi-4-Reas. | SVAMP | 8 | 94 | 94.05 | -0.05 |
| 22 | 64 | Phi-4-Reas. | SVAMP | 8 | 94.125 | 94.35 | -0.225 |
| 23 | 65 | Phi-4-Reas. | SVAMP | 8 | 94.175 | 94.475 | -0.3 |
| 24 | 66 | Phi-4-Reas. | SVAMP | 8 | 93.875 | 94.5 | -0.625 |
| 25 | 67 | Phi-4-Reas. | SVAMP | 8 | 94.1 | 94.45 | -0.35 |
| 26 | 68 | Phi-4-Reas. | SVAMP | 8 | 94.075 | 94.45 | -0.375 |
| 27 | 69 | Phi-4-Reas. | SVAMP | 8 | 94.25 | 94.425 | -0.175 |
| 28 | 70 | Phi-4-Reas. | SVAMP | 8 | 94.1 | 94.6 | -0.5 |
| 29 | 71 | Phi-4-Reas. | SVAMP | 8 | 94.325 | 94.175 | 0.15 |
| 30 | 72 | Phi-4-Reas. | SVAMP | 8 | 93.9 | 94.25 | -0.35 |
| 31 | 73 | Phi-4-Reas. | SVAMP | 8 | 93.875 | 94.375 | -0.5 |
| 32 | 74 | Phi-4-Reas. | SVAMP | 8 | 93.85 | 94.25 | -0.4 |
| 33 | 75 | Phi-4-Reas. | SVAMP | 8 | 94.05 | 94.575 | -0.525 |
| 34 | 76 | Phi-4-Reas. | SVAMP | 8 | 93.925 | 94.525 | -0.6 |
| 35 | 77 | Phi-4-Reas. | SVAMP | 8 | 94.125 | 94.025 | 0.1 |
| 36 | 78 | Phi-4-Reas. | SVAMP | 8 | 94.35 | 94.425 | -0.075 |
| 37 | 79 | Phi-4-Reas. | SVAMP | 8 | 94.6 | 94.55 | 0.05 |
| 38 | 80 | Phi-4-Reas. | SVAMP | 8 | 94.05 | 94.325 | -0.275 |
| 39 | 81 | Phi-4-Reas. | SVAMP | 8 | 94.275 | 94.225 | 0.05 |
| 40 | 82 | Phi-4-Reas. | SVAMP | 8 | 94.1 | 94.4 | -0.3 |
| 41 | 83 | Phi-4-Reas. | SVAMP | 8 | 93.85 | 94.375 | -0.525 |
| 42 | 84 | Phi-4-Reas. | SVAMP | 8 | 94.025 | 94.15 | -0.125 |
| 43 | 85 | Phi-4-Reas. | SVAMP | 8 | 93.825 | 94.375 | -0.55 |
| 44 | 86 | Phi-4-Reas. | SVAMP | 8 | 94.15 | 94.625 | -0.475 |
| 45 | 87 | Phi-4-Reas. | SVAMP | 8 | 93.8 | 94.6 | -0.8 |
| 46 | 88 | Phi-4-Reas. | SVAMP | 8 | 94 | 94.9 | -0.9 |
| 47 | 89 | Phi-4-Reas. | SVAMP | 8 | 93.975 | 94.375 | -0.4 |
| 48 | 90 | Phi-4-Reas. | SVAMP | 8 | 93.7 | 94.425 | -0.725 |
| 49 | 91 | Phi-4-Reas. | SVAMP | 8 | 93.825 | 94.375 | -0.55 |
| 0 | 42 | Phi-4-Reas. | SVAMP | 12 | 94.1833 | 94.3 | -0.1167 |
| 1 | 43 | Phi-4-Reas. | SVAMP | 12 | 94.0167 | 94.5167 | -0.5 |
| 2 | 44 | Phi-4-Reas. | SVAMP | 12 | 94.1 | 94.45 | -0.35 |
| 3 | 45 | Phi-4-Reas. | SVAMP | 12 | 94.0333 | 94.5 | -0.4667 |
| 4 | 46 | Phi-4-Reas. | SVAMP | 12 | 94.0667 | 94.6833 | -0.6167 |
| 5 | 47 | Phi-4-Reas. | SVAMP | 12 | 93.9167 | 94.5333 | -0.6167 |
| 6 | 48 | Phi-4-Reas. | SVAMP | 12 | 94.1667 | 94.5667 | -0.4 |
| 7 | 49 | Phi-4-Reas. | SVAMP | 12 | 93.8833 | 94.5333 | -0.65 |
| 8 | 50 | Phi-4-Reas. | SVAMP | 12 | 94.1333 | 94.3833 | -0.25 |
| 9 | 51 | Phi-4-Reas. | SVAMP | 12 | 94.2 | 94.5333 | -0.3333 |
| 10 | 52 | Phi-4-Reas. | SVAMP | 12 | 94.0833 | 94.3833 | -0.3 |
| 11 | 53 | Phi-4-Reas. | SVAMP | 12 | 93.95 | 94.5833 | -0.6333 |
| 12 | 54 | Phi-4-Reas. | SVAMP | 12 | 94.05 | 94.6 | -0.55 |
| 13 | 55 | Phi-4-Reas. | SVAMP | 12 | 94.1 | 94.3333 | -0.2333 |
| 14 | 56 | Phi-4-Reas. | SVAMP | 12 | 94.25 | 94.2833 | -0.0333 |
| 15 | 57 | Phi-4-Reas. | SVAMP | 12 | 94 | 94.5833 | -0.5833 |
| 16 | 58 | Phi-4-Reas. | SVAMP | 12 | 94.15 | 94.6167 | -0.4667 |
| 17 | 59 | Phi-4-Reas. | SVAMP | 12 | 94.05 | 94.3833 | -0.3333 |
| 18 | 60 | Phi-4-Reas. | SVAMP | 12 | 94 | 94.6 | -0.6 |
| 19 | 61 | Phi-4-Reas. | SVAMP | 12 | 94.0167 | 94.4333 | -0.4167 |
| 20 | 62 | Phi-4-Reas. | SVAMP | 12 | 94.2167 | 94.4833 | -0.2667 |
| 21 | 63 | Phi-4-Reas. | SVAMP | 12 | 94.1167 | 94.4 | -0.2833 |
| 22 | 64 | Phi-4-Reas. | SVAMP | 12 | 94.1667 | 94.3667 | -0.2 |
| 23 | 65 | Phi-4-Reas. | SVAMP | 12 | 94.1833 | 94.5167 | -0.3333 |
| 24 | 66 | Phi-4-Reas. | SVAMP | 12 | 93.9833 | 94.4833 | -0.5 |
| 25 | 67 | Phi-4-Reas. | SVAMP | 12 | 94.1667 | 94.4167 | -0.25 |
| 26 | 68 | Phi-4-Reas. | SVAMP | 12 | 94.05 | 94.5667 | -0.5167 |
| 27 | 69 | Phi-4-Reas. | SVAMP | 12 | 93.9833 | 94.4333 | -0.45 |
| 28 | 70 | Phi-4-Reas. | SVAMP | 12 | 93.9833 | 94.5833 | -0.6 |
| 29 | 71 | Phi-4-Reas. | SVAMP | 12 | 94.05 | 94.25 | -0.2 |
| 30 | 72 | Phi-4-Reas. | SVAMP | 12 | 93.95 | 94.4 | -0.45 |
| 31 | 73 | Phi-4-Reas. | SVAMP | 12 | 94.0833 | 94.3833 | -0.3 |
| 32 | 74 | Phi-4-Reas. | SVAMP | 12 | 93.9833 | 94.4 | -0.4167 |
| 33 | 75 | Phi-4-Reas. | SVAMP | 12 | 94.0333 | 94.55 | -0.5167 |
| 34 | 76 | Phi-4-Reas. | SVAMP | 12 | 94.1 | 94.5667 | -0.4667 |
| 35 | 77 | Phi-4-Reas. | SVAMP | 12 | 94.1 | 94.3167 | -0.2167 |
| 36 | 78 | Phi-4-Reas. | SVAMP | 12 | 94.2333 | 94.4167 | -0.1833 |
| 37 | 79 | Phi-4-Reas. | SVAMP | 12 | 94.1 | 94.5333 | -0.4333 |
| 38 | 80 | Phi-4-Reas. | SVAMP | 12 | 94.2 | 94.5 | -0.3 |
| 39 | 81 | Phi-4-Reas. | SVAMP | 12 | 93.9833 | 94.2833 | -0.3 |
| 40 | 82 | Phi-4-Reas. | SVAMP | 12 | 93.9 | 94.5167 | -0.6167 |
| 41 | 83 | Phi-4-Reas. | SVAMP | 12 | 93.8333 | 94.5333 | -0.7 |
| 42 | 84 | Phi-4-Reas. | SVAMP | 12 | 94.1 | 94.35 | -0.25 |
| 43 | 85 | Phi-4-Reas. | SVAMP | 12 | 94.0833 | 94.25 | -0.1667 |
| 44 | 86 | Phi-4-Reas. | SVAMP | 12 | 94.15 | 94.5667 | -0.4167 |
| 45 | 87 | Phi-4-Reas. | SVAMP | 12 | 93.9667 | 94.5 | -0.5333 |
| 46 | 88 | Phi-4-Reas. | SVAMP | 12 | 94.05 | 94.6 | -0.55 |
| 47 | 89 | Phi-4-Reas. | SVAMP | 12 | 93.9833 | 94.4 | -0.4167 |
| 48 | 90 | Phi-4-Reas. | SVAMP | 12 | 93.9167 | 94.4 | -0.4833 |
| 49 | 91 | Phi-4-Reas. | SVAMP | 12 | 93.7833 | 94.4667 | -0.6833 |
| 0 | 42 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 1 | 43 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 2 | 44 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 3 | 45 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 4 | 46 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 5 | 47 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 6 | 48 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 7 | 49 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 8 | 50 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 9 | 51 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 10 | 52 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 11 | 53 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 12 | 54 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 13 | 55 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 14 | 56 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 15 | 57 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 16 | 58 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 17 | 59 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 18 | 60 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 19 | 61 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 20 | 62 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 21 | 63 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 22 | 64 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 23 | 65 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 24 | 66 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 25 | 67 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 26 | 68 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 27 | 69 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 28 | 70 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 29 | 71 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 30 | 72 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 31 | 73 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 32 | 74 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 33 | 75 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 34 | 76 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 35 | 77 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 36 | 78 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 37 | 79 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 38 | 80 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 39 | 81 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 40 | 82 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 41 | 83 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 42 | 84 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 43 | 85 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 44 | 86 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 45 | 87 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 46 | 88 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 47 | 89 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 48 | 90 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 49 | 91 | Phi-4-Reas. | SVAMP | 16 | 94.025 | 94.475 | -0.45 |
| 0 | 42 | Phi-4 | AQuA | 4 | 56.2992 | 54.7244 | 1.5748 |
| 1 | 43 | Phi-4 | AQuA | 4 | 53.937 | 56.8898 | -2.9528 |
| 2 | 44 | Phi-4 | AQuA | 4 | 53.5433 | 52.7559 | 0.7874 |
| 3 | 45 | Phi-4 | AQuA | 4 | 50.7874 | 55.315 | -4.5276 |
| 4 | 46 | Phi-4 | AQuA | 4 | 53.7402 | 53.5433 | 0.1969 |
| 5 | 47 | Phi-4 | AQuA | 4 | 55.9055 | 53.5433 | 2.3622 |
| 6 | 48 | Phi-4 | AQuA | 4 | 55.5118 | 50.7874 | 4.7244 |
| 7 | 49 | Phi-4 | AQuA | 4 | 51.7717 | 54.3307 | -2.5591 |
| 8 | 50 | Phi-4 | AQuA | 4 | 52.1654 | 54.7244 | -2.5591 |
| 9 | 51 | Phi-4 | AQuA | 4 | 52.3622 | 53.3465 | -0.9843 |
| 10 | 52 | Phi-4 | AQuA | 4 | 55.5118 | 52.1654 | 3.3465 |
| 11 | 53 | Phi-4 | AQuA | 4 | 53.1496 | 53.3465 | -0.1969 |
| 12 | 54 | Phi-4 | AQuA | 4 | 55.9055 | 50.9843 | 4.9213 |
| 13 | 55 | Phi-4 | AQuA | 4 | 53.937 | 51.7717 | 2.1654 |
| 14 | 56 | Phi-4 | AQuA | 4 | 57.0866 | 51.1811 | 5.9055 |
| 15 | 57 | Phi-4 | AQuA | 4 | 57.6772 | 53.3465 | 4.3307 |
| 16 | 58 | Phi-4 | AQuA | 4 | 56.8898 | 54.1339 | 2.7559 |
| 17 | 59 | Phi-4 | AQuA | 4 | 56.8898 | 54.9213 | 1.9685 |
| 18 | 60 | Phi-4 | AQuA | 4 | 55.9055 | 54.1339 | 1.7717 |
| 19 | 61 | Phi-4 | AQuA | 4 | 51.5748 | 52.3622 | -0.7874 |
| 20 | 62 | Phi-4 | AQuA | 4 | 59.0551 | 53.3465 | 5.7087 |
| 21 | 63 | Phi-4 | AQuA | 4 | 56.6929 | 50.9843 | 5.7087 |
| 22 | 64 | Phi-4 | AQuA | 4 | 52.3622 | 55.9055 | -3.5433 |
| 23 | 65 | Phi-4 | AQuA | 4 | 54.1339 | 54.5276 | -0.3937 |
| 24 | 66 | Phi-4 | AQuA | 4 | 54.9213 | 53.3465 | 1.5748 |
| 25 | 67 | Phi-4 | AQuA | 4 | 53.1496 | 50 | 3.1496 |
| 26 | 68 | Phi-4 | AQuA | 4 | 56.4961 | 54.7244 | 1.7717 |
| 27 | 69 | Phi-4 | AQuA | 4 | 53.937 | 54.3307 | -0.3937 |
| 28 | 70 | Phi-4 | AQuA | 4 | 55.9055 | 55.9055 | 0 |
| 29 | 71 | Phi-4 | AQuA | 4 | 53.7402 | 53.1496 | 0.5906 |
| 30 | 72 | Phi-4 | AQuA | 4 | 55.7087 | 52.1654 | 3.5433 |
| 31 | 73 | Phi-4 | AQuA | 4 | 54.1339 | 52.5591 | 1.5748 |
| 32 | 74 | Phi-4 | AQuA | 4 | 57.2835 | 53.5433 | 3.7402 |
| 33 | 75 | Phi-4 | AQuA | 4 | 55.1181 | 51.5748 | 3.5433 |
| 34 | 76 | Phi-4 | AQuA | 4 | 51.378 | 55.1181 | -3.7402 |
| 35 | 77 | Phi-4 | AQuA | 4 | 56.2992 | 51.378 | 4.9213 |
| 36 | 78 | Phi-4 | AQuA | 4 | 55.315 | 54.1339 | 1.1811 |
| 37 | 79 | Phi-4 | AQuA | 4 | 56.4961 | 53.1496 | 3.3465 |
| 38 | 80 | Phi-4 | AQuA | 4 | 54.5276 | 52.5591 | 1.9685 |
| 39 | 81 | Phi-4 | AQuA | 4 | 55.1181 | 53.7402 | 1.378 |
| 40 | 82 | Phi-4 | AQuA | 4 | 55.9055 | 51.5748 | 4.3307 |
| 41 | 83 | Phi-4 | AQuA | 4 | 54.5276 | 52.1654 | 2.3622 |
| 42 | 84 | Phi-4 | AQuA | 4 | 55.1181 | 56.1024 | -0.9843 |
| 43 | 85 | Phi-4 | AQuA | 4 | 54.7244 | 53.5433 | 1.1811 |
| 44 | 86 | Phi-4 | AQuA | 4 | 57.2835 | 55.9055 | 1.378 |
| 45 | 87 | Phi-4 | AQuA | 4 | 54.9213 | 55.315 | -0.3937 |
| 46 | 88 | Phi-4 | AQuA | 4 | 56.1024 | 54.1339 | 1.9685 |
| 47 | 89 | Phi-4 | AQuA | 4 | 55.5118 | 53.1496 | 2.3622 |
| 48 | 90 | Phi-4 | AQuA | 4 | 52.5591 | 53.3465 | -0.7874 |
| 49 | 91 | Phi-4 | AQuA | 4 | 51.9685 | 53.3465 | -1.378 |
| 0 | 42 | Phi-4 | AQuA | 8 | 54.3307 | 54.7244 | -0.3937 |
| 1 | 43 | Phi-4 | AQuA | 8 | 53.3465 | 54.1339 | -0.7874 |
| 2 | 44 | Phi-4 | AQuA | 8 | 53.0512 | 53.7402 | -0.689 |
| 3 | 45 | Phi-4 | AQuA | 8 | 52.6575 | 52.9528 | -0.2953 |
| 4 | 46 | Phi-4 | AQuA | 8 | 53.7402 | 53.3465 | 0.3937 |
| 5 | 47 | Phi-4 | AQuA | 8 | 55.6102 | 52.9528 | 2.6575 |
| 6 | 48 | Phi-4 | AQuA | 8 | 53.937 | 52.5591 | 1.378 |
| 7 | 49 | Phi-4 | AQuA | 8 | 53.937 | 52.6575 | 1.2795 |
| 8 | 50 | Phi-4 | AQuA | 8 | 52.0669 | 55.8071 | -3.7402 |
| 9 | 51 | Phi-4 | AQuA | 8 | 52.8543 | 53.8386 | -0.9843 |
| 10 | 52 | Phi-4 | AQuA | 8 | 54.4291 | 52.3622 | 2.0669 |
| 11 | 53 | Phi-4 | AQuA | 8 | 52.6575 | 53.3465 | -0.689 |
| 12 | 54 | Phi-4 | AQuA | 8 | 55.4134 | 51.9685 | 3.4449 |
| 13 | 55 | Phi-4 | AQuA | 8 | 54.5276 | 51.7717 | 2.7559 |
| 14 | 56 | Phi-4 | AQuA | 8 | 55.315 | 53.8386 | 1.4764 |
| 15 | 57 | Phi-4 | AQuA | 8 | 56.1024 | 53.1496 | 2.9528 |
| 16 | 58 | Phi-4 | AQuA | 8 | 55.6102 | 53.7402 | 1.8701 |
| 17 | 59 | Phi-4 | AQuA | 8 | 55.9055 | 55.315 | 0.5906 |
| 18 | 60 | Phi-4 | AQuA | 8 | 56.6929 | 51.7717 | 4.9213 |
| 19 | 61 | Phi-4 | AQuA | 8 | 54.3307 | 51.7717 | 2.5591 |
| 20 | 62 | Phi-4 | AQuA | 8 | 55.9055 | 52.8543 | 3.0512 |
| 21 | 63 | Phi-4 | AQuA | 8 | 55.7087 | 53.8386 | 1.8701 |
| 22 | 64 | Phi-4 | AQuA | 8 | 53.7402 | 54.7244 | -0.9843 |
| 23 | 65 | Phi-4 | AQuA | 8 | 54.8228 | 54.3307 | 0.4921 |
| 24 | 66 | Phi-4 | AQuA | 8 | 52.4606 | 53.5433 | -1.0827 |
| 25 | 67 | Phi-4 | AQuA | 8 | 52.8543 | 51.0827 | 1.7717 |
| 26 | 68 | Phi-4 | AQuA | 8 | 55.7087 | 53.248 | 2.4606 |
| 27 | 69 | Phi-4 | AQuA | 8 | 54.4291 | 52.3622 | 2.0669 |
| 28 | 70 | Phi-4 | AQuA | 8 | 54.2323 | 54.3307 | -0.0984 |
| 29 | 71 | Phi-4 | AQuA | 8 | 54.5276 | 53.937 | 0.5906 |
| 30 | 72 | Phi-4 | AQuA | 8 | 54.4291 | 52.4606 | 1.9685 |
| 31 | 73 | Phi-4 | AQuA | 8 | 53.1496 | 52.8543 | 0.2953 |
| 32 | 74 | Phi-4 | AQuA | 8 | 55.6102 | 53.1496 | 2.4606 |
| 33 | 75 | Phi-4 | AQuA | 8 | 53.937 | 53.1496 | 0.7874 |
| 34 | 76 | Phi-4 | AQuA | 8 | 54.4291 | 54.4291 | 0 |
| 35 | 77 | Phi-4 | AQuA | 8 | 53.937 | 55.0197 | -1.0827 |
| 36 | 78 | Phi-4 | AQuA | 8 | 55.1181 | 53.8386 | 1.2795 |
| 37 | 79 | Phi-4 | AQuA | 8 | 53.7402 | 52.8543 | 0.8858 |
| 38 | 80 | Phi-4 | AQuA | 8 | 54.4291 | 53.0512 | 1.378 |
| 39 | 81 | Phi-4 | AQuA | 8 | 54.1339 | 54.4291 | -0.2953 |
| 40 | 82 | Phi-4 | AQuA | 8 | 54.0354 | 52.9528 | 1.0827 |
| 41 | 83 | Phi-4 | AQuA | 8 | 54.1339 | 53.4449 | 0.689 |
| 42 | 84 | Phi-4 | AQuA | 8 | 54.5276 | 53.937 | 0.5906 |
| 43 | 85 | Phi-4 | AQuA | 8 | 54.3307 | 54.5276 | -0.1969 |
| 44 | 86 | Phi-4 | AQuA | 8 | 55.4134 | 54.2323 | 1.1811 |
| 45 | 87 | Phi-4 | AQuA | 8 | 54.3307 | 54.0354 | 0.2953 |
| 46 | 88 | Phi-4 | AQuA | 8 | 54.7244 | 53.4449 | 1.2795 |
| 47 | 89 | Phi-4 | AQuA | 8 | 55.0197 | 52.9528 | 2.0669 |
| 48 | 90 | Phi-4 | AQuA | 8 | 52.8543 | 53.0512 | -0.1969 |
| 49 | 91 | Phi-4 | AQuA | 8 | 53.7402 | 52.1654 | 1.5748 |
| 0 | 42 | Phi-4 | AQuA | 12 | 54.9213 | 53.8058 | 1.1155 |
| 1 | 43 | Phi-4 | AQuA | 12 | 54.4619 | 53.2152 | 1.2467 |
| 2 | 44 | Phi-4 | AQuA | 12 | 54.4619 | 54.3963 | 0.0656 |
| 3 | 45 | Phi-4 | AQuA | 12 | 54.0682 | 53.937 | 0.1312 |
| 4 | 46 | Phi-4 | AQuA | 12 | 54.4619 | 53.2152 | 1.2467 |
| 5 | 47 | Phi-4 | AQuA | 12 | 55.1837 | 52.5591 | 2.6247 |
| 6 | 48 | Phi-4 | AQuA | 12 | 53.7402 | 52.6903 | 1.0499 |
| 7 | 49 | Phi-4 | AQuA | 12 | 54.7244 | 53.6089 | 1.1155 |
| 8 | 50 | Phi-4 | AQuA | 12 | 53.6089 | 53.2152 | 0.3937 |
| 9 | 51 | Phi-4 | AQuA | 12 | 53.2808 | 53.1496 | 0.1312 |
| 10 | 52 | Phi-4 | AQuA | 12 | 55.1181 | 53.1496 | 1.9685 |
| 11 | 53 | Phi-4 | AQuA | 12 | 54.9213 | 52.9528 | 1.9685 |
| 12 | 54 | Phi-4 | AQuA | 12 | 54.4619 | 53.6089 | 0.853 |
| 13 | 55 | Phi-4 | AQuA | 12 | 53.6745 | 53.2152 | 0.4593 |
| 14 | 56 | Phi-4 | AQuA | 12 | 54.79 | 53.084 | 1.706 |
| 15 | 57 | Phi-4 | AQuA | 12 | 55.1181 | 53.2808 | 1.8373 |
| 16 | 58 | Phi-4 | AQuA | 12 | 53.937 | 54.1339 | -0.1969 |
| 17 | 59 | Phi-4 | AQuA | 12 | 54.6588 | 54.3307 | 0.3281 |
| 18 | 60 | Phi-4 | AQuA | 12 | 54.6588 | 54.0682 | 0.5906 |
| 19 | 61 | Phi-4 | AQuA | 12 | 54.1339 | 53.2808 | 0.853 |
| 20 | 62 | Phi-4 | AQuA | 12 | 54.2651 | 54.0026 | 0.2625 |
| 21 | 63 | Phi-4 | AQuA | 12 | 54.7244 | 54.1995 | 0.5249 |
| 22 | 64 | Phi-4 | AQuA | 12 | 54.5276 | 54.4619 | 0.0656 |
| 23 | 65 | Phi-4 | AQuA | 12 | 54.8556 | 53.4121 | 1.4436 |
| 24 | 66 | Phi-4 | AQuA | 12 | 53.8058 | 54.1339 | -0.3281 |
| 25 | 67 | Phi-4 | AQuA | 12 | 53.084 | 52.6903 | 0.3937 |
| 26 | 68 | Phi-4 | AQuA | 12 | 54.7244 | 53.4777 | 1.2467 |
| 27 | 69 | Phi-4 | AQuA | 12 | 54.6588 | 54.0682 | 0.5906 |
| 28 | 70 | Phi-4 | AQuA | 12 | 53.6089 | 53.4121 | 0.1969 |
| 29 | 71 | Phi-4 | AQuA | 12 | 55.0525 | 53.4777 | 1.5748 |
| 30 | 72 | Phi-4 | AQuA | 12 | 54.1339 | 52.8871 | 1.2467 |
| 31 | 73 | Phi-4 | AQuA | 12 | 53.7402 | 53.5433 | 0.1969 |
| 32 | 74 | Phi-4 | AQuA | 12 | 55.7743 | 52.4934 | 3.2808 |
| 33 | 75 | Phi-4 | AQuA | 12 | 54.3307 | 53.4121 | 0.9186 |
| 34 | 76 | Phi-4 | AQuA | 12 | 54.3307 | 54.5932 | -0.2625 |
| 35 | 77 | Phi-4 | AQuA | 12 | 54.2651 | 54.6588 | -0.3937 |
| 36 | 78 | Phi-4 | AQuA | 12 | 55.2493 | 53.937 | 1.3123 |
| 37 | 79 | Phi-4 | AQuA | 12 | 54.1339 | 53.8058 | 0.3281 |
| 38 | 80 | Phi-4 | AQuA | 12 | 54.4619 | 53.6089 | 0.853 |
| 39 | 81 | Phi-4 | AQuA | 12 | 54.5276 | 53.7402 | 0.7874 |
| 40 | 82 | Phi-4 | AQuA | 12 | 53.5433 | 52.8871 | 0.6562 |
| 41 | 83 | Phi-4 | AQuA | 12 | 53.5433 | 53.937 | -0.3937 |
| 42 | 84 | Phi-4 | AQuA | 12 | 53.6089 | 54.0026 | -0.3937 |
| 43 | 85 | Phi-4 | AQuA | 12 | 54.2651 | 53.8058 | 0.4593 |
| 44 | 86 | Phi-4 | AQuA | 12 | 54.2651 | 53.8714 | 0.3937 |
| 45 | 87 | Phi-4 | AQuA | 12 | 54.6588 | 53.2152 | 1.4436 |
| 46 | 88 | Phi-4 | AQuA | 12 | 54.1339 | 55.0525 | -0.9186 |
| 47 | 89 | Phi-4 | AQuA | 12 | 54.8556 | 53.1496 | 1.706 |
| 48 | 90 | Phi-4 | AQuA | 12 | 53.937 | 53.2808 | 0.6562 |
| 49 | 91 | Phi-4 | AQuA | 12 | 53.7402 | 53.8714 | -0.1312 |
| 0 | 42 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 1 | 43 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 2 | 44 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 3 | 45 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 4 | 46 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 5 | 47 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 6 | 48 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 7 | 49 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 8 | 50 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 9 | 51 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 10 | 52 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 11 | 53 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 12 | 54 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 13 | 55 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 14 | 56 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 15 | 57 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 16 | 58 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 17 | 59 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 18 | 60 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 19 | 61 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 20 | 62 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 21 | 63 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 22 | 64 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 23 | 65 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 24 | 66 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 25 | 67 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 26 | 68 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 27 | 69 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 28 | 70 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 29 | 71 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 30 | 72 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 31 | 73 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 32 | 74 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 33 | 75 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 34 | 76 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 35 | 77 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 36 | 78 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 37 | 79 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 38 | 80 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 39 | 81 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 40 | 82 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 41 | 83 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 42 | 84 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 43 | 85 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 44 | 86 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 45 | 87 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 46 | 88 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 47 | 89 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 48 | 90 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 49 | 91 | Phi-4 | AQuA | 16 | 54.4783 | 53.4449 | 1.0335 |
| 0 | 42 | Phi-4 | CommonsenseQA | 4 | 55.9378 | 42.9156 | 13.0221 |
| 1 | 43 | Phi-4 | CommonsenseQA | 4 | 55.2007 | 43.6527 | 11.5479 |
| 2 | 44 | Phi-4 | CommonsenseQA | 4 | 54.4226 | 42.9975 | 11.4251 |
| 3 | 45 | Phi-4 | CommonsenseQA | 4 | 54.8731 | 43.407 | 11.466 |
| 4 | 46 | Phi-4 | CommonsenseQA | 4 | 53.6855 | 43.1613 | 10.5242 |
| 5 | 47 | Phi-4 | CommonsenseQA | 4 | 57.4939 | 42.2604 | 15.2334 |
| 6 | 48 | Phi-4 | CommonsenseQA | 4 | 54.4636 | 41.3186 | 13.145 |
| 7 | 49 | Phi-4 | CommonsenseQA | 4 | 54.7912 | 44.4717 | 10.3194 |
| 8 | 50 | Phi-4 | CommonsenseQA | 4 | 53.4808 | 43.7756 | 9.7052 |
| 9 | 51 | Phi-4 | CommonsenseQA | 4 | 55.4873 | 44.1851 | 11.3022 |
| 10 | 52 | Phi-4 | CommonsenseQA | 4 | 55.3235 | 43.6527 | 11.6708 |
| 11 | 53 | Phi-4 | CommonsenseQA | 4 | 55.2007 | 42.629 | 12.5717 |
| 12 | 54 | Phi-4 | CommonsenseQA | 4 | 55.3645 | 43.2023 | 12.1622 |
| 13 | 55 | Phi-4 | CommonsenseQA | 4 | 54.6683 | 42.9566 | 11.7117 |
| 14 | 56 | Phi-4 | CommonsenseQA | 4 | 55.1597 | 43.0794 | 12.0803 |
| 15 | 57 | Phi-4 | CommonsenseQA | 4 | 56.4701 | 42.9975 | 13.4726 |
| 16 | 58 | Phi-4 | CommonsenseQA | 4 | 53.6036 | 43.5708 | 10.0328 |
| 17 | 59 | Phi-4 | CommonsenseQA | 4 | 54.8731 | 43.8984 | 10.9746 |
| 18 | 60 | Phi-4 | CommonsenseQA | 4 | 55.2826 | 43.4889 | 11.7936 |
| 19 | 61 | Phi-4 | CommonsenseQA | 4 | 54.5455 | 43.7756 | 10.7699 |
| 20 | 62 | Phi-4 | CommonsenseQA | 4 | 55.6921 | 42.4652 | 13.2269 |
| 21 | 63 | Phi-4 | CommonsenseQA | 4 | 57.0434 | 42.8747 | 14.1687 |
| 22 | 64 | Phi-4 | CommonsenseQA | 4 | 55.6511 | 42.8337 | 12.8174 |
| 23 | 65 | Phi-4 | CommonsenseQA | 4 | 55.4873 | 43.6118 | 11.8755 |
| 24 | 66 | Phi-4 | CommonsenseQA | 4 | 54.4636 | 43.9803 | 10.4832 |
| 25 | 67 | Phi-4 | CommonsenseQA | 4 | 54.5864 | 43.1204 | 11.466 |
| 26 | 68 | Phi-4 | CommonsenseQA | 4 | 54.955 | 43.407 | 11.5479 |
| 27 | 69 | Phi-4 | CommonsenseQA | 4 | 55.3235 | 43.2023 | 12.1212 |
| 28 | 70 | Phi-4 | CommonsenseQA | 4 | 56.6749 | 44.0622 | 12.6126 |
| 29 | 71 | Phi-4 | CommonsenseQA | 4 | 54.0541 | 42.7518 | 11.3022 |
| 30 | 72 | Phi-4 | CommonsenseQA | 4 | 54.6683 | 43.2432 | 11.4251 |
| 31 | 73 | Phi-4 | CommonsenseQA | 4 | 55.2416 | 43.5708 | 11.6708 |
| 32 | 74 | Phi-4 | CommonsenseQA | 4 | 55.4464 | 42.6699 | 12.7764 |
| 33 | 75 | Phi-4 | CommonsenseQA | 4 | 56.4292 | 44.0622 | 12.3669 |
| 34 | 76 | Phi-4 | CommonsenseQA | 4 | 55.1597 | 43.6118 | 11.5479 |
| 35 | 77 | Phi-4 | CommonsenseQA | 4 | 55.0778 | 43.4889 | 11.5889 |
| 36 | 78 | Phi-4 | CommonsenseQA | 4 | 55.2007 | 43.6118 | 11.5889 |
| 37 | 79 | Phi-4 | CommonsenseQA | 4 | 56.1425 | 44.1851 | 11.9574 |
| 38 | 80 | Phi-4 | CommonsenseQA | 4 | 55.5692 | 43.1204 | 12.4488 |
| 39 | 81 | Phi-4 | CommonsenseQA | 4 | 54.7093 | 43.8165 | 10.8927 |
| 40 | 82 | Phi-4 | CommonsenseQA | 4 | 56.593 | 45.2088 | 11.3841 |
| 41 | 83 | Phi-4 | CommonsenseQA | 4 | 53.7674 | 43.8575 | 9.9099 |
| 42 | 84 | Phi-4 | CommonsenseQA | 4 | 55.774 | 44.3898 | 11.3841 |
| 43 | 85 | Phi-4 | CommonsenseQA | 4 | 55.1188 | 42.5471 | 12.5717 |
| 44 | 86 | Phi-4 | CommonsenseQA | 4 | 56.593 | 43.2023 | 13.3907 |
| 45 | 87 | Phi-4 | CommonsenseQA | 4 | 55.2416 | 44.0622 | 11.1794 |
| 46 | 88 | Phi-4 | CommonsenseQA | 4 | 56.4701 | 43.5299 | 12.9402 |
| 47 | 89 | Phi-4 | CommonsenseQA | 4 | 56.0606 | 42.4652 | 13.5954 |
| 48 | 90 | Phi-4 | CommonsenseQA | 4 | 54.4226 | 42.5061 | 11.9165 |
| 49 | 91 | Phi-4 | CommonsenseQA | 4 | 53.2351 | 43.7346 | 9.5004 |
| 0 | 42 | Phi-4 | CommonsenseQA | 8 | 54.9959 | 43.9599 | 11.036 |
| 1 | 43 | Phi-4 | CommonsenseQA | 8 | 54.6888 | 43.4685 | 11.2203 |
| 2 | 44 | Phi-4 | CommonsenseQA | 8 | 54.6274 | 43.3047 | 11.3227 |
| 3 | 45 | Phi-4 | CommonsenseQA | 8 | 54.1564 | 44.0213 | 10.1351 |
| 4 | 46 | Phi-4 | CommonsenseQA | 8 | 54.4431 | 44.3079 | 10.1351 |
| 5 | 47 | Phi-4 | CommonsenseQA | 8 | 56.2244 | 42.9361 | 13.2883 |
| 6 | 48 | Phi-4 | CommonsenseQA | 8 | 54.3817 | 43.2842 | 11.0975 |
| 7 | 49 | Phi-4 | CommonsenseQA | 8 | 54.8935 | 44.0827 | 10.8108 |
| 8 | 50 | Phi-4 | CommonsenseQA | 8 | 54.5045 | 43.4889 | 11.0156 |
| 9 | 51 | Phi-4 | CommonsenseQA | 8 | 55.2826 | 44.0418 | 11.2408 |
| 10 | 52 | Phi-4 | CommonsenseQA | 8 | 55.2826 | 43.6937 | 11.5889 |
| 11 | 53 | Phi-4 | CommonsenseQA | 8 | 54.6478 | 42.9771 | 11.6708 |
| 12 | 54 | Phi-4 | CommonsenseQA | 8 | 54.7297 | 43.3047 | 11.4251 |
| 13 | 55 | Phi-4 | CommonsenseQA | 8 | 54.955 | 43.4889 | 11.466 |
| 14 | 56 | Phi-4 | CommonsenseQA | 8 | 53.9926 | 44.0622 | 9.9304 |
| 15 | 57 | Phi-4 | CommonsenseQA | 8 | 54.7093 | 44.0622 | 10.647 |
| 16 | 58 | Phi-4 | CommonsenseQA | 8 | 54.5864 | 43.6323 | 10.9541 |
| 17 | 59 | Phi-4 | CommonsenseQA | 8 | 55.733 | 43.6937 | 12.0393 |
| 18 | 60 | Phi-4 | CommonsenseQA | 8 | 55.3235 | 43.6527 | 11.6708 |
| 19 | 61 | Phi-4 | CommonsenseQA | 8 | 54.525 | 44.9631 | 9.5618 |
| 20 | 62 | Phi-4 | CommonsenseQA | 8 | 55.3645 | 43.0385 | 12.326 |
| 21 | 63 | Phi-4 | CommonsenseQA | 8 | 55.4259 | 44.4308 | 10.9951 |
| 22 | 64 | Phi-4 | CommonsenseQA | 8 | 54.6478 | 43.0385 | 11.6093 |
| 23 | 65 | Phi-4 | CommonsenseQA | 8 | 54.8935 | 44.5536 | 10.3399 |
| 24 | 66 | Phi-4 | CommonsenseQA | 8 | 54.8731 | 43.6118 | 11.2613 |
| 25 | 67 | Phi-4 | CommonsenseQA | 8 | 55.5078 | 43.407 | 12.1007 |
| 26 | 68 | Phi-4 | CommonsenseQA | 8 | 55.1802 | 43.7961 | 11.3841 |
| 27 | 69 | Phi-4 | CommonsenseQA | 8 | 55.4259 | 43.4275 | 11.9984 |
| 28 | 70 | Phi-4 | CommonsenseQA | 8 | 55.1597 | 44.1646 | 10.9951 |
| 29 | 71 | Phi-4 | CommonsenseQA | 8 | 53.9722 | 43.2023 | 10.7699 |
| 30 | 72 | Phi-4 | CommonsenseQA | 8 | 54.5455 | 44.0418 | 10.5037 |
| 31 | 73 | Phi-4 | CommonsenseQA | 8 | 55.4259 | 43.4889 | 11.9369 |
| 32 | 74 | Phi-4 | CommonsenseQA | 8 | 54.6478 | 44.2056 | 10.4423 |
| 33 | 75 | Phi-4 | CommonsenseQA | 8 | 55.7944 | 44.3284 | 11.466 |
| 34 | 76 | Phi-4 | CommonsenseQA | 8 | 55.3235 | 44.2465 | 11.077 |
| 35 | 77 | Phi-4 | CommonsenseQA | 8 | 54.5659 | 44.6765 | 9.8894 |
| 36 | 78 | Phi-4 | CommonsenseQA | 8 | 54.6478 | 43.9803 | 10.6675 |
| 37 | 79 | Phi-4 | CommonsenseQA | 8 | 55.1392 | 44.0827 | 11.0565 |
| 38 | 80 | Phi-4 | CommonsenseQA | 8 | 54.8935 | 43.6527 | 11.2408 |
| 39 | 81 | Phi-4 | CommonsenseQA | 8 | 54.7502 | 44.1237 | 10.6265 |
| 40 | 82 | Phi-4 | CommonsenseQA | 8 | 55.303 | 44.3489 | 10.9541 |
| 41 | 83 | Phi-4 | CommonsenseQA | 8 | 53.5831 | 44.697 | 8.8862 |
| 42 | 84 | Phi-4 | CommonsenseQA | 8 | 54.7707 | 43.6527 | 11.1179 |
| 43 | 85 | Phi-4 | CommonsenseQA | 8 | 54.914 | 42.8133 | 12.1007 |
| 44 | 86 | Phi-4 | CommonsenseQA | 8 | 56.163 | 43.1818 | 12.9812 |
| 45 | 87 | Phi-4 | CommonsenseQA | 8 | 55.2007 | 43.5094 | 11.6912 |
| 46 | 88 | Phi-4 | CommonsenseQA | 8 | 55.4668 | 43.448 | 12.0188 |
| 47 | 89 | Phi-4 | CommonsenseQA | 8 | 55.4668 | 44.3694 | 11.0975 |
| 48 | 90 | Phi-4 | CommonsenseQA | 8 | 54.9959 | 42.1581 | 12.8378 |
| 49 | 91 | Phi-4 | CommonsenseQA | 8 | 54.7093 | 42.8952 | 11.8141 |
| 0 | 42 | Phi-4 | CommonsenseQA | 12 | 54.8867 | 44.0486 | 10.8381 |
| 1 | 43 | Phi-4 | CommonsenseQA | 12 | 54.2998 | 44.1168 | 10.1829 |
| 2 | 44 | Phi-4 | CommonsenseQA | 12 | 54.4499 | 44.1578 | 10.2921 |
| 3 | 45 | Phi-4 | CommonsenseQA | 12 | 54.0404 | 44.5127 | 9.5277 |
| 4 | 46 | Phi-4 | CommonsenseQA | 12 | 54.6001 | 44.4171 | 10.1829 |
| 5 | 47 | Phi-4 | CommonsenseQA | 12 | 55.1324 | 43.8848 | 11.2476 |
| 6 | 48 | Phi-4 | CommonsenseQA | 12 | 54.1633 | 44.5673 | 9.596 |
| 7 | 49 | Phi-4 | CommonsenseQA | 12 | 54.7912 | 43.5162 | 11.2749 |
| 8 | 50 | Phi-4 | CommonsenseQA | 12 | 54.8731 | 43.8302 | 11.0429 |
| 9 | 51 | Phi-4 | CommonsenseQA | 12 | 54.3817 | 44.1714 | 10.2102 |
| 10 | 52 | Phi-4 | CommonsenseQA | 12 | 54.2725 | 44.3079 | 9.9645 |
| 11 | 53 | Phi-4 | CommonsenseQA | 12 | 54.5591 | 43.9257 | 10.6334 |
| 12 | 54 | Phi-4 | CommonsenseQA | 12 | 55.0232 | 44.1851 | 10.8381 |
| 13 | 55 | Phi-4 | CommonsenseQA | 12 | 55.2007 | 44.0213 | 11.1794 |
| 14 | 56 | Phi-4 | CommonsenseQA | 12 | 54.2725 | 44.7584 | 9.5141 |
| 15 | 57 | Phi-4 | CommonsenseQA | 12 | 54.6956 | 43.7483 | 10.9473 |
| 16 | 58 | Phi-4 | CommonsenseQA | 12 | 54.368 | 44.4035 | 9.9645 |
| 17 | 59 | Phi-4 | CommonsenseQA | 12 | 54.8594 | 44.4581 | 10.4013 |
| 18 | 60 | Phi-4 | CommonsenseQA | 12 | 54.7093 | 44.1578 | 10.5515 |
| 19 | 61 | Phi-4 | CommonsenseQA | 12 | 54.1087 | 44.4854 | 9.6233 |
| 20 | 62 | Phi-4 | CommonsenseQA | 12 | 54.5182 | 44.4308 | 10.0874 |
| 21 | 63 | Phi-4 | CommonsenseQA | 12 | 54.9004 | 44.4171 | 10.4832 |
| 22 | 64 | Phi-4 | CommonsenseQA | 12 | 54.3271 | 43.5845 | 10.7426 |
| 23 | 65 | Phi-4 | CommonsenseQA | 12 | 55.0915 | 44.4717 | 10.6197 |
| 24 | 66 | Phi-4 | CommonsenseQA | 12 | 54.5455 | 44.2943 | 10.2512 |
| 25 | 67 | Phi-4 | CommonsenseQA | 12 | 54.5455 | 44.499 | 10.0464 |
| 26 | 68 | Phi-4 | CommonsenseQA | 12 | 54.5182 | 44.4717 | 10.0464 |
| 27 | 69 | Phi-4 | CommonsenseQA | 12 | 54.7639 | 43.9121 | 10.8518 |
| 28 | 70 | Phi-4 | CommonsenseQA | 12 | 54.5864 | 44.1851 | 10.4013 |
| 29 | 71 | Phi-4 | CommonsenseQA | 12 | 54.2042 | 44.1305 | 10.0737 |
| 30 | 72 | Phi-4 | CommonsenseQA | 12 | 54.4226 | 44.4854 | 9.9372 |
| 31 | 73 | Phi-4 | CommonsenseQA | 12 | 55.0642 | 43.8302 | 11.234 |
| 32 | 74 | Phi-4 | CommonsenseQA | 12 | 54.136 | 44.7584 | 9.3776 |
| 33 | 75 | Phi-4 | CommonsenseQA | 12 | 54.6547 | 44.2806 | 10.374 |
| 34 | 76 | Phi-4 | CommonsenseQA | 12 | 54.7639 | 44.4308 | 10.3331 |
| 35 | 77 | Phi-4 | CommonsenseQA | 12 | 54.2588 | 44.2124 | 10.0464 |
| 36 | 78 | Phi-4 | CommonsenseQA | 12 | 54.3953 | 44.8403 | 9.555 |
| 37 | 79 | Phi-4 | CommonsenseQA | 12 | 55.0232 | 44.2806 | 10.7426 |
| 38 | 80 | Phi-4 | CommonsenseQA | 12 | 54.2452 | 44.2124 | 10.0328 |
| 39 | 81 | Phi-4 | CommonsenseQA | 12 | 54.4363 | 44.0349 | 10.4013 |
| 40 | 82 | Phi-4 | CommonsenseQA | 12 | 54.7229 | 44.4717 | 10.2512 |
| 41 | 83 | Phi-4 | CommonsenseQA | 12 | 54.0404 | 44.4854 | 9.555 |
| 42 | 84 | Phi-4 | CommonsenseQA | 12 | 54.7229 | 44.1168 | 10.6061 |
| 43 | 85 | Phi-4 | CommonsenseQA | 12 | 54.3953 | 43.9257 | 10.4696 |
| 44 | 86 | Phi-4 | CommonsenseQA | 12 | 55.7603 | 44.0759 | 11.6844 |
| 45 | 87 | Phi-4 | CommonsenseQA | 12 | 54.5864 | 44.7993 | 9.7871 |
| 46 | 88 | Phi-4 | CommonsenseQA | 12 | 54.682 | 44.7993 | 9.8826 |
| 47 | 89 | Phi-4 | CommonsenseQA | 12 | 54.8048 | 44.2533 | 10.5515 |
| 48 | 90 | Phi-4 | CommonsenseQA | 12 | 54.6001 | 43.7892 | 10.8108 |
| 49 | 91 | Phi-4 | CommonsenseQA | 12 | 54.9686 | 44.0213 | 10.9473 |
| 0 | 42 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 1 | 43 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 2 | 44 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 3 | 45 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 4 | 46 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 5 | 47 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 6 | 48 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 7 | 49 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 8 | 50 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 9 | 51 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 10 | 52 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 11 | 53 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 12 | 54 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 13 | 55 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 14 | 56 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 15 | 57 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 16 | 58 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 17 | 59 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 18 | 60 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 19 | 61 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 20 | 62 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 21 | 63 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 22 | 64 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 23 | 65 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 24 | 66 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 25 | 67 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 26 | 68 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 27 | 69 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 28 | 70 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 29 | 71 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 30 | 72 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 31 | 73 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 32 | 74 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 33 | 75 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 34 | 76 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 35 | 77 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 36 | 78 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 37 | 79 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 38 | 80 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 39 | 81 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 40 | 82 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 41 | 83 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 42 | 84 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 43 | 85 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 44 | 86 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 45 | 87 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 46 | 88 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 47 | 89 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 48 | 90 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 49 | 91 | Phi-4 | CommonsenseQA | 16 | 54.3305 | 44.5741 | 9.7563 |
| 0 | 42 | Phi-4 | GPQA | 4 | 25.3348 | 27.1205 | -1.7857 |
| 1 | 43 | Phi-4 | GPQA | 4 | 25.7812 | 26.7857 | -1.0045 |
| 2 | 44 | Phi-4 | GPQA | 4 | 25.4464 | 24.442 | 1.0045 |
| 3 | 45 | Phi-4 | GPQA | 4 | 26.0045 | 26.6741 | -0.6696 |
| 4 | 46 | Phi-4 | GPQA | 4 | 28.683 | 27.4554 | 1.2277 |
| 5 | 47 | Phi-4 | GPQA | 4 | 24.442 | 26.1161 | -1.6741 |
| 6 | 48 | Phi-4 | GPQA | 4 | 25.8929 | 27.1205 | -1.2277 |
| 7 | 49 | Phi-4 | GPQA | 4 | 22.6562 | 26.1161 | -3.4598 |
| 8 | 50 | Phi-4 | GPQA | 4 | 24.7768 | 25.1116 | -0.3348 |
| 9 | 51 | Phi-4 | GPQA | 4 | 25.7812 | 29.5759 | -3.7946 |
| 10 | 52 | Phi-4 | GPQA | 4 | 23.8839 | 25.4464 | -1.5625 |
| 11 | 53 | Phi-4 | GPQA | 4 | 25.6696 | 28.2366 | -2.567 |
| 12 | 54 | Phi-4 | GPQA | 4 | 25.1116 | 28.0134 | -2.9018 |
| 13 | 55 | Phi-4 | GPQA | 4 | 23.1027 | 28.3482 | -5.2455 |
| 14 | 56 | Phi-4 | GPQA | 4 | 25.4464 | 23.9955 | 1.4509 |
| 15 | 57 | Phi-4 | GPQA | 4 | 26.8973 | 28.0134 | -1.1161 |
| 16 | 58 | Phi-4 | GPQA | 4 | 24.7768 | 25.6696 | -0.8929 |
| 17 | 59 | Phi-4 | GPQA | 4 | 26.3393 | 27.9018 | -1.5625 |
| 18 | 60 | Phi-4 | GPQA | 4 | 23.1027 | 27.3438 | -4.2411 |
| 19 | 61 | Phi-4 | GPQA | 4 | 25.7812 | 27.6786 | -1.8973 |
| 20 | 62 | Phi-4 | GPQA | 4 | 23.9955 | 26.2277 | -2.2321 |
| 21 | 63 | Phi-4 | GPQA | 4 | 24.8884 | 26.6741 | -1.7857 |
| 22 | 64 | Phi-4 | GPQA | 4 | 22.6562 | 28.0134 | -5.3571 |
| 23 | 65 | Phi-4 | GPQA | 4 | 27.0089 | 27.4554 | -0.4464 |
| 24 | 66 | Phi-4 | GPQA | 4 | 23.6607 | 26.8973 | -3.2366 |
| 25 | 67 | Phi-4 | GPQA | 4 | 26.6741 | 26.1161 | 0.558 |
| 26 | 68 | Phi-4 | GPQA | 4 | 25.6696 | 28.4598 | -2.7902 |
| 27 | 69 | Phi-4 | GPQA | 4 | 26.5625 | 27.2321 | -0.6696 |
| 28 | 70 | Phi-4 | GPQA | 4 | 26.2277 | 27.2321 | -1.0045 |
| 29 | 71 | Phi-4 | GPQA | 4 | 26.1161 | 26.2277 | -0.1116 |
| 30 | 72 | Phi-4 | GPQA | 4 | 25.6696 | 25.6696 | 0 |
| 31 | 73 | Phi-4 | GPQA | 4 | 22.6562 | 27.567 | -4.9107 |
| 32 | 74 | Phi-4 | GPQA | 4 | 24.6652 | 25.4464 | -0.7812 |
| 33 | 75 | Phi-4 | GPQA | 4 | 25.4464 | 26.4509 | -1.0045 |
| 34 | 76 | Phi-4 | GPQA | 4 | 24.5536 | 24.8884 | -0.3348 |
| 35 | 77 | Phi-4 | GPQA | 4 | 25 | 27.567 | -2.567 |
| 36 | 78 | Phi-4 | GPQA | 4 | 24.8884 | 26.1161 | -1.2277 |
| 37 | 79 | Phi-4 | GPQA | 4 | 28.4598 | 28.2366 | 0.2232 |
| 38 | 80 | Phi-4 | GPQA | 4 | 24.2188 | 26.8973 | -2.6786 |
| 39 | 81 | Phi-4 | GPQA | 4 | 25 | 26.0045 | -1.0045 |
| 40 | 82 | Phi-4 | GPQA | 4 | 25 | 26.2277 | -1.2277 |
| 41 | 83 | Phi-4 | GPQA | 4 | 26.8973 | 25.2232 | 1.6741 |
| 42 | 84 | Phi-4 | GPQA | 4 | 25.8929 | 24.5536 | 1.3393 |
| 43 | 85 | Phi-4 | GPQA | 4 | 25.6696 | 28.683 | -3.0134 |
| 44 | 86 | Phi-4 | GPQA | 4 | 27.7902 | 28.5714 | -0.7812 |
| 45 | 87 | Phi-4 | GPQA | 4 | 27.1205 | 25.7812 | 1.3393 |
| 46 | 88 | Phi-4 | GPQA | 4 | 24.8884 | 29.5759 | -4.6875 |
| 47 | 89 | Phi-4 | GPQA | 4 | 26.5625 | 26.1161 | 0.4464 |
| 48 | 90 | Phi-4 | GPQA | 4 | 25.3348 | 27.567 | -2.2321 |
| 49 | 91 | Phi-4 | GPQA | 4 | 26.6741 | 26.8973 | -0.2232 |
| 0 | 42 | Phi-4 | GPQA | 8 | 25 | 28.2924 | -3.2924 |
| 1 | 43 | Phi-4 | GPQA | 8 | 25 | 27.2879 | -2.2879 |
| 2 | 44 | Phi-4 | GPQA | 8 | 23.8839 | 26.6183 | -2.7344 |
| 3 | 45 | Phi-4 | GPQA | 8 | 24.2746 | 27.2321 | -2.9576 |
| 4 | 46 | Phi-4 | GPQA | 8 | 26.0045 | 27.0647 | -1.0603 |
| 5 | 47 | Phi-4 | GPQA | 8 | 24.2188 | 27.0089 | -2.7902 |
| 6 | 48 | Phi-4 | GPQA | 8 | 24.2188 | 27.3438 | -3.125 |
| 7 | 49 | Phi-4 | GPQA | 8 | 23.9397 | 27.6228 | -3.683 |
| 8 | 50 | Phi-4 | GPQA | 8 | 24.442 | 27.4554 | -3.0134 |
| 9 | 51 | Phi-4 | GPQA | 8 | 24.721 | 28.5714 | -3.8504 |
| 10 | 52 | Phi-4 | GPQA | 8 | 23.3259 | 26.6741 | -3.3482 |
| 11 | 53 | Phi-4 | GPQA | 8 | 24.5536 | 29.8549 | -5.3013 |
| 12 | 54 | Phi-4 | GPQA | 8 | 24.0513 | 28.404 | -4.3527 |
| 13 | 55 | Phi-4 | GPQA | 8 | 22.6004 | 28.0134 | -5.4129 |
| 14 | 56 | Phi-4 | GPQA | 8 | 23.9397 | 27.1763 | -3.2366 |
| 15 | 57 | Phi-4 | GPQA | 8 | 26.1161 | 27.4554 | -1.3393 |
| 16 | 58 | Phi-4 | GPQA | 8 | 24.6652 | 27.7344 | -3.0692 |
| 17 | 59 | Phi-4 | GPQA | 8 | 24.8884 | 27.9576 | -3.0692 |
| 18 | 60 | Phi-4 | GPQA | 8 | 23.6049 | 27.846 | -4.2411 |
| 19 | 61 | Phi-4 | GPQA | 8 | 25.1116 | 27.2879 | -2.1763 |
| 20 | 62 | Phi-4 | GPQA | 8 | 24.442 | 27.1763 | -2.7344 |
| 21 | 63 | Phi-4 | GPQA | 8 | 25.1674 | 26.8973 | -1.7299 |
| 22 | 64 | Phi-4 | GPQA | 8 | 23.8281 | 28.2924 | -4.4643 |
| 23 | 65 | Phi-4 | GPQA | 8 | 24.442 | 28.5714 | -4.1295 |
| 24 | 66 | Phi-4 | GPQA | 8 | 24.2746 | 29.5201 | -5.2455 |
| 25 | 67 | Phi-4 | GPQA | 8 | 24.8884 | 27.4554 | -2.567 |
| 26 | 68 | Phi-4 | GPQA | 8 | 23.9955 | 28.0692 | -4.0737 |
| 27 | 69 | Phi-4 | GPQA | 8 | 25.6138 | 28.3482 | -2.7344 |
| 28 | 70 | Phi-4 | GPQA | 8 | 24.442 | 27.1763 | -2.7344 |
| 29 | 71 | Phi-4 | GPQA | 8 | 24.1629 | 28.2924 | -4.1295 |
| 30 | 72 | Phi-4 | GPQA | 8 | 24.442 | 28.404 | -3.9621 |
| 31 | 73 | Phi-4 | GPQA | 8 | 24.5536 | 27.846 | -3.2924 |
| 32 | 74 | Phi-4 | GPQA | 8 | 24.7768 | 27.2321 | -2.4554 |
| 33 | 75 | Phi-4 | GPQA | 8 | 24.6094 | 27.4554 | -2.846 |
| 34 | 76 | Phi-4 | GPQA | 8 | 24.7768 | 26.6183 | -1.8415 |
| 35 | 77 | Phi-4 | GPQA | 8 | 25.6696 | 26.4509 | -0.7812 |
| 36 | 78 | Phi-4 | GPQA | 8 | 24.7768 | 27.0647 | -2.2879 |
| 37 | 79 | Phi-4 | GPQA | 8 | 25.4464 | 29.7433 | -4.2969 |
| 38 | 80 | Phi-4 | GPQA | 8 | 22.8795 | 27.7344 | -4.8549 |
| 39 | 81 | Phi-4 | GPQA | 8 | 25.8371 | 26.7857 | -0.9487 |
| 40 | 82 | Phi-4 | GPQA | 8 | 25.4464 | 26.3393 | -0.8929 |
| 41 | 83 | Phi-4 | GPQA | 8 | 25.4464 | 28.125 | -2.6786 |
| 42 | 84 | Phi-4 | GPQA | 8 | 24.721 | 25.8929 | -1.1719 |
| 43 | 85 | Phi-4 | GPQA | 8 | 25.279 | 28.125 | -2.846 |
| 44 | 86 | Phi-4 | GPQA | 8 | 25.4464 | 27.7344 | -2.2879 |
| 45 | 87 | Phi-4 | GPQA | 8 | 24.8884 | 27.0647 | -2.1763 |
| 46 | 88 | Phi-4 | GPQA | 8 | 24.0513 | 28.2924 | -4.2411 |
| 47 | 89 | Phi-4 | GPQA | 8 | 24.9442 | 27.1763 | -2.2321 |
| 48 | 90 | Phi-4 | GPQA | 8 | 23.9397 | 27.3996 | -3.4598 |
| 49 | 91 | Phi-4 | GPQA | 8 | 24.0513 | 27.0647 | -3.0134 |
| 0 | 42 | Phi-4 | GPQA | 12 | 24.6652 | 28.7574 | -4.0923 |
| 1 | 43 | Phi-4 | GPQA | 12 | 23.9583 | 28.3482 | -4.3899 |
| 2 | 44 | Phi-4 | GPQA | 12 | 23.8095 | 27.1205 | -3.311 |
| 3 | 45 | Phi-4 | GPQA | 12 | 23.8095 | 27.753 | -3.9435 |
| 4 | 46 | Phi-4 | GPQA | 12 | 24.814 | 28.2738 | -3.4598 |
| 5 | 47 | Phi-4 | GPQA | 12 | 23.6979 | 27.2321 | -3.5342 |
| 6 | 48 | Phi-4 | GPQA | 12 | 24.1443 | 28.125 | -3.9807 |
| 7 | 49 | Phi-4 | GPQA | 12 | 24.0327 | 28.0878 | -4.0551 |
| 8 | 50 | Phi-4 | GPQA | 12 | 24.2932 | 28.4226 | -4.1295 |
| 9 | 51 | Phi-4 | GPQA | 12 | 24.5536 | 27.2321 | -2.6786 |
| 10 | 52 | Phi-4 | GPQA | 12 | 23.6235 | 27.7158 | -4.0923 |
| 11 | 53 | Phi-4 | GPQA | 12 | 23.6607 | 28.869 | -5.2083 |
| 12 | 54 | Phi-4 | GPQA | 12 | 23.9211 | 28.9062 | -4.9851 |
| 13 | 55 | Phi-4 | GPQA | 12 | 24.1815 | 28.0134 | -3.8318 |
| 14 | 56 | Phi-4 | GPQA | 12 | 24.6652 | 27.6786 | -3.0134 |
| 15 | 57 | Phi-4 | GPQA | 12 | 24.5164 | 28.0506 | -3.5342 |
| 16 | 58 | Phi-4 | GPQA | 12 | 24.5164 | 27.8646 | -3.3482 |
| 17 | 59 | Phi-4 | GPQA | 12 | 24.4048 | 27.9018 | -3.497 |
| 18 | 60 | Phi-4 | GPQA | 12 | 23.8095 | 27.7158 | -3.9062 |
| 19 | 61 | Phi-4 | GPQA | 12 | 24.3304 | 28.0506 | -3.7202 |
| 20 | 62 | Phi-4 | GPQA | 12 | 23.2887 | 28.0506 | -4.7619 |
| 21 | 63 | Phi-4 | GPQA | 12 | 24.1815 | 27.8646 | -3.683 |
| 22 | 64 | Phi-4 | GPQA | 12 | 24.1443 | 28.9062 | -4.7619 |
| 23 | 65 | Phi-4 | GPQA | 12 | 24.442 | 28.6458 | -4.2039 |
| 24 | 66 | Phi-4 | GPQA | 12 | 23.8839 | 29.3899 | -5.506 |
| 25 | 67 | Phi-4 | GPQA | 12 | 24.2932 | 27.3065 | -3.0134 |
| 26 | 68 | Phi-4 | GPQA | 12 | 24.1815 | 28.3854 | -4.2039 |
| 27 | 69 | Phi-4 | GPQA | 12 | 23.9955 | 28.497 | -4.5015 |
| 28 | 70 | Phi-4 | GPQA | 12 | 23.8467 | 28.0134 | -4.1667 |
| 29 | 71 | Phi-4 | GPQA | 12 | 24.5164 | 28.3854 | -3.869 |
| 30 | 72 | Phi-4 | GPQA | 12 | 25.0744 | 27.939 | -2.8646 |
| 31 | 73 | Phi-4 | GPQA | 12 | 24.8512 | 27.7158 | -2.8646 |
| 32 | 74 | Phi-4 | GPQA | 12 | 24.7024 | 28.2366 | -3.5342 |
| 33 | 75 | Phi-4 | GPQA | 12 | 24.1815 | 27.567 | -3.3854 |
| 34 | 76 | Phi-4 | GPQA | 12 | 24.3304 | 27.753 | -3.4226 |
| 35 | 77 | Phi-4 | GPQA | 12 | 23.6607 | 27.6414 | -3.9807 |
| 36 | 78 | Phi-4 | GPQA | 12 | 24.814 | 27.1949 | -2.381 |
| 37 | 79 | Phi-4 | GPQA | 12 | 24.628 | 28.9062 | -4.2783 |
| 38 | 80 | Phi-4 | GPQA | 12 | 24.1071 | 27.567 | -3.4598 |
| 39 | 81 | Phi-4 | GPQA | 12 | 25.2232 | 27.0833 | -1.8601 |
| 40 | 82 | Phi-4 | GPQA | 12 | 25.372 | 26.9345 | -1.5625 |
| 41 | 83 | Phi-4 | GPQA | 12 | 24.4792 | 27.4554 | -2.9762 |
| 42 | 84 | Phi-4 | GPQA | 12 | 24.1815 | 27.567 | -3.3854 |
| 43 | 85 | Phi-4 | GPQA | 12 | 24.5536 | 28.5342 | -3.9807 |
| 44 | 86 | Phi-4 | GPQA | 12 | 24.7396 | 28.0506 | -3.311 |
| 45 | 87 | Phi-4 | GPQA | 12 | 24.4792 | 28.311 | -3.8318 |
| 46 | 88 | Phi-4 | GPQA | 12 | 23.9583 | 28.6458 | -4.6875 |
| 47 | 89 | Phi-4 | GPQA | 12 | 24.256 | 27.9762 | -3.7202 |
| 48 | 90 | Phi-4 | GPQA | 12 | 23.4003 | 28.2366 | -4.8363 |
| 49 | 91 | Phi-4 | GPQA | 12 | 23.9955 | 27.8274 | -3.8318 |
| 0 | 42 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 1 | 43 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 2 | 44 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 3 | 45 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 4 | 46 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 5 | 47 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 6 | 48 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 7 | 49 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 8 | 50 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 9 | 51 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 10 | 52 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 11 | 53 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 12 | 54 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 13 | 55 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 14 | 56 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 15 | 57 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 16 | 58 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 17 | 59 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 18 | 60 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 19 | 61 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 20 | 62 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 21 | 63 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 22 | 64 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 23 | 65 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 24 | 66 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 25 | 67 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 26 | 68 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 27 | 69 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 28 | 70 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 29 | 71 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 30 | 72 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 31 | 73 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 32 | 74 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 33 | 75 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 34 | 76 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 35 | 77 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 36 | 78 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 37 | 79 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 38 | 80 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 39 | 81 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 40 | 82 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 41 | 83 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 42 | 84 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 43 | 85 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 44 | 86 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 45 | 87 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 46 | 88 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 47 | 89 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 48 | 90 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 49 | 91 | Phi-4 | GPQA | 16 | 23.9955 | 28.1808 | -4.1853 |
| 0 | 42 | Phi-4 | GSM8K | 4 | 63.7225 | 67.3995 | -3.677 |
| 1 | 43 | Phi-4 | GSM8K | 4 | 62.7748 | 67.21 | -4.4352 |
| 2 | 44 | Phi-4 | GSM8K | 4 | 62.4337 | 67.21 | -4.7763 |
| 3 | 45 | Phi-4 | GSM8K | 4 | 61.865 | 67.0584 | -5.1933 |
| 4 | 46 | Phi-4 | GSM8K | 4 | 62.6232 | 70.1668 | -7.5436 |
| 5 | 47 | Phi-4 | GSM8K | 4 | 63.8741 | 68.3472 | -4.4731 |
| 6 | 48 | Phi-4 | GSM8K | 4 | 63.8741 | 68.461 | -4.5868 |
| 7 | 49 | Phi-4 | GSM8K | 4 | 62.8886 | 68.8021 | -5.9136 |
| 8 | 50 | Phi-4 | GSM8K | 4 | 62.6611 | 69.6361 | -6.975 |
| 9 | 51 | Phi-4 | GSM8K | 4 | 63.4572 | 68.461 | -5.0038 |
| 10 | 52 | Phi-4 | GSM8K | 4 | 62.0546 | 68.6884 | -6.6338 |
| 11 | 53 | Phi-4 | GSM8K | 4 | 62.5474 | 69.4086 | -6.8613 |
| 12 | 54 | Phi-4 | GSM8K | 4 | 62.9644 | 66.5656 | -3.6012 |
| 13 | 55 | Phi-4 | GSM8K | 4 | 63.8741 | 67.5891 | -3.7149 |
| 14 | 56 | Phi-4 | GSM8K | 4 | 63.5709 | 70.7354 | -7.1645 |
| 15 | 57 | Phi-4 | GSM8K | 4 | 63.116 | 67.5512 | -4.4352 |
| 16 | 58 | Phi-4 | GSM8K | 4 | 62.282 | 68.5368 | -6.2547 |
| 17 | 59 | Phi-4 | GSM8K | 4 | 62.7748 | 68.4989 | -5.724 |
| 18 | 60 | Phi-4 | GSM8K | 4 | 63.7983 | 67.8165 | -4.0182 |
| 19 | 61 | Phi-4 | GSM8K | 4 | 62.1304 | 68.461 | -6.3306 |
| 20 | 62 | Phi-4 | GSM8K | 4 | 62.2062 | 67.8923 | -5.6861 |
| 21 | 63 | Phi-4 | GSM8K | 4 | 62.282 | 68.0819 | -5.7998 |
| 22 | 64 | Phi-4 | GSM8K | 4 | 63.116 | 69.0296 | -5.9136 |
| 23 | 65 | Phi-4 | GSM8K | 4 | 61.4102 | 67.3616 | -5.9515 |
| 24 | 66 | Phi-4 | GSM8K | 4 | 63.1918 | 68.3093 | -5.1175 |
| 25 | 67 | Phi-4 | GSM8K | 4 | 63.7983 | 67.5133 | -3.7149 |
| 26 | 68 | Phi-4 | GSM8K | 4 | 61.069 | 66.9447 | -5.8757 |
| 27 | 69 | Phi-4 | GSM8K | 4 | 62.9644 | 67.4754 | -4.511 |
| 28 | 70 | Phi-4 | GSM8K | 4 | 63.7604 | 68.1198 | -4.3594 |
| 29 | 71 | Phi-4 | GSM8K | 4 | 63.6088 | 68.3472 | -4.7384 |
| 30 | 72 | Phi-4 | GSM8K | 4 | 62.3958 | 70.0152 | -7.6194 |
| 31 | 73 | Phi-4 | GSM8K | 4 | 62.7748 | 67.5512 | -4.7763 |
| 32 | 74 | Phi-4 | GSM8K | 4 | 63.2297 | 69.3707 | -6.141 |
| 33 | 75 | Phi-4 | GSM8K | 4 | 63.2676 | 67.8165 | -4.5489 |
| 34 | 76 | Phi-4 | GSM8K | 4 | 64.2153 | 67.6649 | -3.4496 |
| 35 | 77 | Phi-4 | GSM8K | 4 | 63.6467 | 67.0584 | -3.4117 |
| 36 | 78 | Phi-4 | GSM8K | 4 | 62.4337 | 68.6505 | -6.2168 |
| 37 | 79 | Phi-4 | GSM8K | 4 | 63.4951 | 68.9538 | -5.4587 |
| 38 | 80 | Phi-4 | GSM8K | 4 | 62.282 | 69.0675 | -6.7854 |
| 39 | 81 | Phi-4 | GSM8K | 4 | 62.2062 | 67.7407 | -5.5345 |
| 40 | 82 | Phi-4 | GSM8K | 4 | 62.8127 | 67.6649 | -4.8522 |
| 41 | 83 | Phi-4 | GSM8K | 4 | 63.1539 | 68.3093 | -5.1554 |
| 42 | 84 | Phi-4 | GSM8K | 4 | 62.5095 | 68.4989 | -5.9894 |
| 43 | 85 | Phi-4 | GSM8K | 4 | 63.0023 | 68.3093 | -5.3071 |
| 44 | 86 | Phi-4 | GSM8K | 4 | 63.3813 | 67.6649 | -4.2835 |
| 45 | 87 | Phi-4 | GSM8K | 4 | 62.3199 | 68.3472 | -6.0273 |
| 46 | 88 | Phi-4 | GSM8K | 4 | 62.5853 | 68.8779 | -6.2926 |
| 47 | 89 | Phi-4 | GSM8K | 4 | 64.746 | 68.0819 | -3.3359 |
| 48 | 90 | Phi-4 | GSM8K | 4 | 60.4625 | 68.3093 | -7.8469 |
| 49 | 91 | Phi-4 | GSM8K | 4 | 62.9644 | 67.9303 | -4.9659 |
| 0 | 42 | Phi-4 | GSM8K | 8 | 63.4003 | 67.7786 | -4.3783 |
| 1 | 43 | Phi-4 | GSM8K | 8 | 63.0781 | 68.1387 | -5.0607 |
| 2 | 44 | Phi-4 | GSM8K | 8 | 62.301 | 67.7028 | -5.4018 |
| 3 | 45 | Phi-4 | GSM8K | 8 | 62.2252 | 68.5368 | -6.3116 |
| 4 | 46 | Phi-4 | GSM8K | 8 | 62.301 | 68.6694 | -6.3685 |
| 5 | 47 | Phi-4 | GSM8K | 8 | 63.2676 | 68.5368 | -5.2691 |
| 6 | 48 | Phi-4 | GSM8K | 8 | 62.0546 | 68.4989 | -6.4443 |
| 7 | 49 | Phi-4 | GSM8K | 8 | 63.0402 | 68.1198 | -5.0796 |
| 8 | 50 | Phi-4 | GSM8K | 8 | 62.0925 | 68.5936 | -6.5011 |
| 9 | 51 | Phi-4 | GSM8K | 8 | 62.1494 | 68.0629 | -5.9136 |
| 10 | 52 | Phi-4 | GSM8K | 8 | 62.0167 | 69.2949 | -7.2782 |
| 11 | 53 | Phi-4 | GSM8K | 8 | 62.7938 | 68.3093 | -5.5155 |
| 12 | 54 | Phi-4 | GSM8K | 8 | 62.1494 | 67.5891 | -5.4397 |
| 13 | 55 | Phi-4 | GSM8K | 8 | 62.8317 | 68.1577 | -5.326 |
| 14 | 56 | Phi-4 | GSM8K | 8 | 63.1539 | 69.2381 | -6.0842 |
| 15 | 57 | Phi-4 | GSM8K | 8 | 62.1873 | 68.84 | -6.6528 |
| 16 | 58 | Phi-4 | GSM8K | 8 | 62.699 | 69.2949 | -6.5959 |
| 17 | 59 | Phi-4 | GSM8K | 8 | 62.6801 | 68.6694 | -5.9894 |
| 18 | 60 | Phi-4 | GSM8K | 8 | 62.6232 | 68.461 | -5.8378 |
| 19 | 61 | Phi-4 | GSM8K | 8 | 62.3578 | 68.423 | -6.0652 |
| 20 | 62 | Phi-4 | GSM8K | 8 | 62.7559 | 68.6694 | -5.9136 |
| 21 | 63 | Phi-4 | GSM8K | 8 | 62.5663 | 68.7832 | -6.2168 |
| 22 | 64 | Phi-4 | GSM8K | 8 | 62.7938 | 68.5936 | -5.7998 |
| 23 | 65 | Phi-4 | GSM8K | 8 | 63.135 | 67.9303 | -4.7953 |
| 24 | 66 | Phi-4 | GSM8K | 8 | 62.6042 | 69.2191 | -6.6149 |
| 25 | 67 | Phi-4 | GSM8K | 8 | 63.1539 | 68.1956 | -5.0417 |
| 26 | 68 | Phi-4 | GSM8K | 8 | 61.6566 | 67.9871 | -6.3306 |
| 27 | 69 | Phi-4 | GSM8K | 8 | 62.1494 | 68.7832 | -6.6338 |
| 28 | 70 | Phi-4 | GSM8K | 8 | 63.135 | 68.0819 | -4.9469 |
| 29 | 71 | Phi-4 | GSM8K | 8 | 63.0212 | 67.7976 | -4.7763 |
| 30 | 72 | Phi-4 | GSM8K | 8 | 62.1683 | 68.7074 | -6.539 |
| 31 | 73 | Phi-4 | GSM8K | 8 | 62.2441 | 67.9871 | -5.743 |
| 32 | 74 | Phi-4 | GSM8K | 8 | 62.5095 | 68.6884 | -6.1789 |
| 33 | 75 | Phi-4 | GSM8K | 8 | 62.282 | 67.3427 | -5.0607 |
| 34 | 76 | Phi-4 | GSM8K | 8 | 62.5095 | 68.4041 | -5.8946 |
| 35 | 77 | Phi-4 | GSM8K | 8 | 62.5474 | 68.1198 | -5.5724 |
| 36 | 78 | Phi-4 | GSM8K | 8 | 62.6611 | 68.1766 | -5.5155 |
| 37 | 79 | Phi-4 | GSM8K | 8 | 62.4716 | 68.6694 | -6.1979 |
| 38 | 80 | Phi-4 | GSM8K | 8 | 62.282 | 68.2904 | -6.0083 |
| 39 | 81 | Phi-4 | GSM8K | 8 | 62.718 | 68.4799 | -5.7619 |
| 40 | 82 | Phi-4 | GSM8K | 8 | 62.5663 | 68.3472 | -5.7809 |
| 41 | 83 | Phi-4 | GSM8K | 8 | 62.1494 | 68.7263 | -6.577 |
| 42 | 84 | Phi-4 | GSM8K | 8 | 62.9265 | 67.9871 | -5.0607 |
| 43 | 85 | Phi-4 | GSM8K | 8 | 62.4147 | 68.4989 | -6.0842 |
| 44 | 86 | Phi-4 | GSM8K | 8 | 62.5663 | 68.2525 | -5.6861 |
| 45 | 87 | Phi-4 | GSM8K | 8 | 62.0356 | 68.6126 | -6.577 |
| 46 | 88 | Phi-4 | GSM8K | 8 | 61.5428 | 68.84 | -7.2972 |
| 47 | 89 | Phi-4 | GSM8K | 8 | 62.8696 | 68.0629 | -5.1933 |
| 48 | 90 | Phi-4 | GSM8K | 8 | 62.3958 | 68.8021 | -6.4064 |
| 49 | 91 | Phi-4 | GSM8K | 8 | 62.2252 | 68.0819 | -5.8567 |
| 0 | 42 | Phi-4 | GSM8K | 12 | 62.7875 | 68.1577 | -5.3702 |
| 1 | 43 | Phi-4 | GSM8K | 12 | 62.9265 | 68.1577 | -5.2312 |
| 2 | 44 | Phi-4 | GSM8K | 12 | 62.5853 | 68.6884 | -6.1031 |
| 3 | 45 | Phi-4 | GSM8K | 12 | 62.143 | 68.701 | -6.558 |
| 4 | 46 | Phi-4 | GSM8K | 12 | 62.4589 | 68.7895 | -6.3306 |
| 5 | 47 | Phi-4 | GSM8K | 12 | 62.3199 | 68.2461 | -5.9262 |
| 6 | 48 | Phi-4 | GSM8K | 12 | 62.0167 | 68.4989 | -6.4822 |
| 7 | 49 | Phi-4 | GSM8K | 12 | 63.0781 | 68.0187 | -4.9406 |
| 8 | 50 | Phi-4 | GSM8K | 12 | 62.2189 | 68.3599 | -6.141 |
| 9 | 51 | Phi-4 | GSM8K | 12 | 62.421 | 68.3725 | -5.9515 |
| 10 | 52 | Phi-4 | GSM8K | 12 | 61.9282 | 68.7895 | -6.8613 |
| 11 | 53 | Phi-4 | GSM8K | 12 | 62.8506 | 68.5747 | -5.724 |
| 12 | 54 | Phi-4 | GSM8K | 12 | 62.4842 | 68.6126 | -6.1284 |
| 13 | 55 | Phi-4 | GSM8K | 12 | 62.838 | 68.6379 | -5.7998 |
| 14 | 56 | Phi-4 | GSM8K | 12 | 63.0528 | 68.9411 | -5.8883 |
| 15 | 57 | Phi-4 | GSM8K | 12 | 62.0799 | 69.0296 | -6.9497 |
| 16 | 58 | Phi-4 | GSM8K | 12 | 62.9265 | 68.3978 | -5.4713 |
| 17 | 59 | Phi-4 | GSM8K | 12 | 62.1683 | 68.8274 | -6.6591 |
| 18 | 60 | Phi-4 | GSM8K | 12 | 62.9391 | 68.3599 | -5.4208 |
| 19 | 61 | Phi-4 | GSM8K | 12 | 62.1809 | 68.5494 | -6.3685 |
| 20 | 62 | Phi-4 | GSM8K | 12 | 62.5221 | 68.5999 | -6.0778 |
| 21 | 63 | Phi-4 | GSM8K | 12 | 62.4968 | 68.6758 | -6.1789 |
| 22 | 64 | Phi-4 | GSM8K | 12 | 62.699 | 68.2588 | -5.5598 |
| 23 | 65 | Phi-4 | GSM8K | 12 | 62.56 | 68.2082 | -5.6482 |
| 24 | 66 | Phi-4 | GSM8K | 12 | 62.4337 | 69.3328 | -6.8992 |
| 25 | 67 | Phi-4 | GSM8K | 12 | 62.3958 | 68.7389 | -6.3432 |
| 26 | 68 | Phi-4 | GSM8K | 12 | 62.3578 | 68.3599 | -6.002 |
| 27 | 69 | Phi-4 | GSM8K | 12 | 62.0925 | 68.562 | -6.4695 |
| 28 | 70 | Phi-4 | GSM8K | 12 | 62.5853 | 68.5368 | -5.9515 |
| 29 | 71 | Phi-4 | GSM8K | 12 | 62.2315 | 68.562 | -6.3306 |
| 30 | 72 | Phi-4 | GSM8K | 12 | 62.1809 | 68.9538 | -6.7728 |
| 31 | 73 | Phi-4 | GSM8K | 12 | 62.9265 | 68.0692 | -5.1428 |
| 32 | 74 | Phi-4 | GSM8K | 12 | 62.4463 | 68.4736 | -6.0273 |
| 33 | 75 | Phi-4 | GSM8K | 12 | 62.3705 | 68.5368 | -6.1663 |
| 34 | 76 | Phi-4 | GSM8K | 12 | 61.9661 | 68.4736 | -6.5075 |
| 35 | 77 | Phi-4 | GSM8K | 12 | 63.0528 | 68.2335 | -5.1807 |
| 36 | 78 | Phi-4 | GSM8K | 12 | 62.6611 | 68.6379 | -5.9768 |
| 37 | 79 | Phi-4 | GSM8K | 12 | 62.4463 | 68.2335 | -5.7872 |
| 38 | 80 | Phi-4 | GSM8K | 12 | 62.6106 | 67.8165 | -5.206 |
| 39 | 81 | Phi-4 | GSM8K | 12 | 62.8506 | 68.3346 | -5.484 |
| 40 | 82 | Phi-4 | GSM8K | 12 | 62.3831 | 68.4104 | -6.0273 |
| 41 | 83 | Phi-4 | GSM8K | 12 | 62.699 | 68.5368 | -5.8378 |
| 42 | 84 | Phi-4 | GSM8K | 12 | 62.2441 | 68.6884 | -6.4443 |
| 43 | 85 | Phi-4 | GSM8K | 12 | 62.4968 | 68.5999 | -6.1031 |
| 44 | 86 | Phi-4 | GSM8K | 12 | 62.5347 | 68.4104 | -5.8757 |
| 45 | 87 | Phi-4 | GSM8K | 12 | 62.3326 | 68.2967 | -5.9641 |
| 46 | 88 | Phi-4 | GSM8K | 12 | 62.282 | 68.7263 | -6.4443 |
| 47 | 89 | Phi-4 | GSM8K | 12 | 63.0023 | 68.1577 | -5.1554 |
| 48 | 90 | Phi-4 | GSM8K | 12 | 62.3326 | 68.4483 | -6.1157 |
| 49 | 91 | Phi-4 | GSM8K | 12 | 62.4589 | 68.5494 | -6.0905 |
| 0 | 42 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 1 | 43 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 2 | 44 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 3 | 45 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 4 | 46 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 5 | 47 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 6 | 48 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 7 | 49 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 8 | 50 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 9 | 51 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 10 | 52 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 11 | 53 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 12 | 54 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 13 | 55 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 14 | 56 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 15 | 57 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 16 | 58 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 17 | 59 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 18 | 60 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 19 | 61 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 20 | 62 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 21 | 63 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 22 | 64 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 23 | 65 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 24 | 66 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 25 | 67 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 26 | 68 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 27 | 69 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 28 | 70 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 29 | 71 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 30 | 72 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 31 | 73 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 32 | 74 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 33 | 75 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 34 | 76 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 35 | 77 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 36 | 78 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 37 | 79 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 38 | 80 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 39 | 81 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 40 | 82 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 41 | 83 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 42 | 84 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 43 | 85 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 44 | 86 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 45 | 87 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 46 | 88 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 47 | 89 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 48 | 90 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 49 | 91 | Phi-4 | GSM8K | 16 | 62.5095 | 68.4894 | -5.9799 |
| 0 | 42 | Phi-4 | MATH500 | 4 | 56.3 | 53.2 | 3.1 |
| 1 | 43 | Phi-4 | MATH500 | 4 | 58.1 | 52.3 | 5.8 |
| 2 | 44 | Phi-4 | MATH500 | 4 | 55.6 | 52.2 | 3.4 |
| 3 | 45 | Phi-4 | MATH500 | 4 | 58.3 | 53.5 | 4.8 |
| 4 | 46 | Phi-4 | MATH500 | 4 | 58 | 52.5 | 5.5 |
| 5 | 47 | Phi-4 | MATH500 | 4 | 56.6 | 53.4 | 3.2 |
| 6 | 48 | Phi-4 | MATH500 | 4 | 56.3 | 53 | 3.3 |
| 7 | 49 | Phi-4 | MATH500 | 4 | 55.9 | 53.3 | 2.6 |
| 8 | 50 | Phi-4 | MATH500 | 4 | 55.6 | 53 | 2.6 |
| 9 | 51 | Phi-4 | MATH500 | 4 | 56.7 | 55 | 1.7 |
| 10 | 52 | Phi-4 | MATH500 | 4 | 57.3 | 53.3 | 4 |
| 11 | 53 | Phi-4 | MATH500 | 4 | 57.8 | 55.3 | 2.5 |
| 12 | 54 | Phi-4 | MATH500 | 4 | 55.8 | 54.6 | 1.2 |
| 13 | 55 | Phi-4 | MATH500 | 4 | 59.8 | 49.4 | 10.4 |
| 14 | 56 | Phi-4 | MATH500 | 4 | 56 | 54.1 | 1.9 |
| 15 | 57 | Phi-4 | MATH500 | 4 | 55.4 | 55 | 0.4 |
| 16 | 58 | Phi-4 | MATH500 | 4 | 57.2 | 54 | 3.2 |
| 17 | 59 | Phi-4 | MATH500 | 4 | 55.9 | 52.3 | 3.6 |
| 18 | 60 | Phi-4 | MATH500 | 4 | 58.6 | 55.3 | 3.3 |
| 19 | 61 | Phi-4 | MATH500 | 4 | 55.5 | 54.8 | 0.7 |
| 20 | 62 | Phi-4 | MATH500 | 4 | 57.4 | 51.7 | 5.7 |
| 21 | 63 | Phi-4 | MATH500 | 4 | 56.8 | 53.6 | 3.2 |
| 22 | 64 | Phi-4 | MATH500 | 4 | 56.8 | 53.3 | 3.5 |
| 23 | 65 | Phi-4 | MATH500 | 4 | 55.7 | 54.5 | 1.2 |
| 24 | 66 | Phi-4 | MATH500 | 4 | 56.7 | 56.2 | 0.5 |
| 25 | 67 | Phi-4 | MATH500 | 4 | 57.2 | 53.4 | 3.8 |
| 26 | 68 | Phi-4 | MATH500 | 4 | 57.9 | 52.9 | 5 |
| 27 | 69 | Phi-4 | MATH500 | 4 | 56.9 | 53.2 | 3.7 |
| 28 | 70 | Phi-4 | MATH500 | 4 | 58.4 | 53.6 | 4.8 |
| 29 | 71 | Phi-4 | MATH500 | 4 | 55.5 | 54 | 1.5 |
| 30 | 72 | Phi-4 | MATH500 | 4 | 57.4 | 52.8 | 4.6 |
| 31 | 73 | Phi-4 | MATH500 | 4 | 58 | 53.3 | 4.7 |
| 32 | 74 | Phi-4 | MATH500 | 4 | 57.3 | 53.3 | 4 |
| 33 | 75 | Phi-4 | MATH500 | 4 | 57.1 | 54.8 | 2.3 |
| 34 | 76 | Phi-4 | MATH500 | 4 | 53.3 | 52.8 | 0.5 |
| 35 | 77 | Phi-4 | MATH500 | 4 | 60.7 | 54.2 | 6.5 |
| 36 | 78 | Phi-4 | MATH500 | 4 | 57.1 | 53.5 | 3.6 |
| 37 | 79 | Phi-4 | MATH500 | 4 | 60.5 | 52.3 | 8.2 |
| 38 | 80 | Phi-4 | MATH500 | 4 | 60.1 | 53.7 | 6.4 |
| 39 | 81 | Phi-4 | MATH500 | 4 | 58.8 | 53.9 | 4.9 |
| 40 | 82 | Phi-4 | MATH500 | 4 | 58.8 | 51 | 7.8 |
| 41 | 83 | Phi-4 | MATH500 | 4 | 56.4 | 55.1 | 1.3 |
| 42 | 84 | Phi-4 | MATH500 | 4 | 58.4 | 52.6 | 5.8 |
| 43 | 85 | Phi-4 | MATH500 | 4 | 56.8 | 54.2 | 2.6 |
| 44 | 86 | Phi-4 | MATH500 | 4 | 59 | 52.7 | 6.3 |
| 45 | 87 | Phi-4 | MATH500 | 4 | 55.8 | 54.4 | 1.4 |
| 46 | 88 | Phi-4 | MATH500 | 4 | 57.4 | 56.6 | 0.8 |
| 47 | 89 | Phi-4 | MATH500 | 4 | 56.5 | 54.1 | 2.4 |
| 48 | 90 | Phi-4 | MATH500 | 4 | 56.6 | 55.8 | 0.8 |
| 49 | 91 | Phi-4 | MATH500 | 4 | 57.3 | 54.7 | 2.6 |
| 0 | 42 | Phi-4 | MATH500 | 8 | 57.95 | 53.35 | 4.6 |
| 1 | 43 | Phi-4 | MATH500 | 8 | 57.85 | 53.25 | 4.6 |
| 2 | 44 | Phi-4 | MATH500 | 8 | 56.45 | 52.9 | 3.55 |
| 3 | 45 | Phi-4 | MATH500 | 8 | 57.15 | 54.05 | 3.1 |
| 4 | 46 | Phi-4 | MATH500 | 8 | 58.4 | 53.8 | 4.6 |
| 5 | 47 | Phi-4 | MATH500 | 8 | 57.7 | 53.65 | 4.05 |
| 6 | 48 | Phi-4 | MATH500 | 8 | 57.65 | 53.4 | 4.25 |
| 7 | 49 | Phi-4 | MATH500 | 8 | 57.65 | 53.5 | 4.15 |
| 8 | 50 | Phi-4 | MATH500 | 8 | 57.45 | 55.15 | 2.3 |
| 9 | 51 | Phi-4 | MATH500 | 8 | 57.05 | 53.55 | 3.5 |
| 10 | 52 | Phi-4 | MATH500 | 8 | 56.7 | 53 | 3.7 |
| 11 | 53 | Phi-4 | MATH500 | 8 | 57.05 | 53.3 | 3.75 |
| 12 | 54 | Phi-4 | MATH500 | 8 | 57.15 | 53.65 | 3.5 |
| 13 | 55 | Phi-4 | MATH500 | 8 | 58.7 | 51.35 | 7.35 |
| 14 | 56 | Phi-4 | MATH500 | 8 | 56.2 | 55.05 | 1.15 |
| 15 | 57 | Phi-4 | MATH500 | 8 | 57.15 | 53.5 | 3.65 |
| 16 | 58 | Phi-4 | MATH500 | 8 | 56.75 | 53.85 | 2.9 |
| 17 | 59 | Phi-4 | MATH500 | 8 | 57.25 | 52.65 | 4.6 |
| 18 | 60 | Phi-4 | MATH500 | 8 | 57.85 | 53.2 | 4.65 |
| 19 | 61 | Phi-4 | MATH500 | 8 | 56.8 | 54.1 | 2.7 |
| 20 | 62 | Phi-4 | MATH500 | 8 | 56.65 | 52.4 | 4.25 |
| 21 | 63 | Phi-4 | MATH500 | 8 | 56.45 | 52.8 | 3.65 |
| 22 | 64 | Phi-4 | MATH500 | 8 | 57.1 | 52.4 | 4.7 |
| 23 | 65 | Phi-4 | MATH500 | 8 | 57.25 | 53.3 | 3.95 |
| 24 | 66 | Phi-4 | MATH500 | 8 | 56.75 | 54.1 | 2.65 |
| 25 | 67 | Phi-4 | MATH500 | 8 | 57.4 | 53 | 4.4 |
| 26 | 68 | Phi-4 | MATH500 | 8 | 56.9 | 53.2 | 3.7 |
| 27 | 69 | Phi-4 | MATH500 | 8 | 57.95 | 53.4 | 4.55 |
| 28 | 70 | Phi-4 | MATH500 | 8 | 57.15 | 52.75 | 4.4 |
| 29 | 71 | Phi-4 | MATH500 | 8 | 55.9 | 53.55 | 2.35 |
| 30 | 72 | Phi-4 | MATH500 | 8 | 56.85 | 53.45 | 3.4 |
| 31 | 73 | Phi-4 | MATH500 | 8 | 57.2 | 53.7 | 3.5 |
| 32 | 74 | Phi-4 | MATH500 | 8 | 57 | 53.3 | 3.7 |
| 33 | 75 | Phi-4 | MATH500 | 8 | 56.85 | 53.4 | 3.45 |
| 34 | 76 | Phi-4 | MATH500 | 8 | 55.75 | 52.75 | 3 |
| 35 | 77 | Phi-4 | MATH500 | 8 | 59.1 | 52.5 | 6.6 |
| 36 | 78 | Phi-4 | MATH500 | 8 | 57.55 | 53.4 | 4.15 |
| 37 | 79 | Phi-4 | MATH500 | 8 | 58.8 | 52.5 | 6.3 |
| 38 | 80 | Phi-4 | MATH500 | 8 | 58.2 | 54.75 | 3.45 |
| 39 | 81 | Phi-4 | MATH500 | 8 | 57.85 | 53.2 | 4.65 |
| 40 | 82 | Phi-4 | MATH500 | 8 | 58.75 | 52.45 | 6.3 |
| 41 | 83 | Phi-4 | MATH500 | 8 | 56.95 | 53.55 | 3.4 |
| 42 | 84 | Phi-4 | MATH500 | 8 | 58.2 | 53.2 | 5 |
| 43 | 85 | Phi-4 | MATH500 | 8 | 57.4 | 52.75 | 4.65 |
| 44 | 86 | Phi-4 | MATH500 | 8 | 58.05 | 53.45 | 4.6 |
| 45 | 87 | Phi-4 | MATH500 | 8 | 57.5 | 52.5 | 5 |
| 46 | 88 | Phi-4 | MATH500 | 8 | 57.25 | 54.45 | 2.8 |
| 47 | 89 | Phi-4 | MATH500 | 8 | 57.55 | 53.75 | 3.8 |
| 48 | 90 | Phi-4 | MATH500 | 8 | 57.45 | 54.25 | 3.2 |
| 49 | 91 | Phi-4 | MATH500 | 8 | 57.8 | 54.15 | 3.65 |
| 0 | 42 | Phi-4 | MATH500 | 12 | 58 | 53 | 5 |
| 1 | 43 | Phi-4 | MATH500 | 12 | 57.7667 | 53.2667 | 4.5 |
| 2 | 44 | Phi-4 | MATH500 | 12 | 57.2 | 53.2667 | 3.9333 |
| 3 | 45 | Phi-4 | MATH500 | 12 | 57.5667 | 52.9667 | 4.6 |
| 4 | 46 | Phi-4 | MATH500 | 12 | 57.8333 | 53.8667 | 3.9667 |
| 5 | 47 | Phi-4 | MATH500 | 12 | 57.3667 | 52.8333 | 4.5333 |
| 6 | 48 | Phi-4 | MATH500 | 12 | 57.4667 | 54.4333 | 3.0333 |
| 7 | 49 | Phi-4 | MATH500 | 12 | 58.0667 | 53.2333 | 4.8333 |
| 8 | 50 | Phi-4 | MATH500 | 12 | 57.8 | 53.4 | 4.4 |
| 9 | 51 | Phi-4 | MATH500 | 12 | 56.8 | 53.6 | 3.2 |
| 10 | 52 | Phi-4 | MATH500 | 12 | 57.2 | 53.0667 | 4.1333 |
| 11 | 53 | Phi-4 | MATH500 | 12 | 57.4667 | 52.9667 | 4.5 |
| 12 | 54 | Phi-4 | MATH500 | 12 | 57.5333 | 52.9333 | 4.6 |
| 13 | 55 | Phi-4 | MATH500 | 12 | 57.8 | 52.8667 | 4.9333 |
| 14 | 56 | Phi-4 | MATH500 | 12 | 56.9667 | 53.9333 | 3.0333 |
| 15 | 57 | Phi-4 | MATH500 | 12 | 57.4333 | 53.6 | 3.8333 |
| 16 | 58 | Phi-4 | MATH500 | 12 | 57.0333 | 52.9333 | 4.1 |
| 17 | 59 | Phi-4 | MATH500 | 12 | 57.1667 | 53.3 | 3.8667 |
| 18 | 60 | Phi-4 | MATH500 | 12 | 57.3667 | 52.9667 | 4.4 |
| 19 | 61 | Phi-4 | MATH500 | 12 | 57.7333 | 53.5 | 4.2333 |
| 20 | 62 | Phi-4 | MATH500 | 12 | 57.3333 | 52.6 | 4.7333 |
| 21 | 63 | Phi-4 | MATH500 | 12 | 57.2667 | 53.6 | 3.6667 |
| 22 | 64 | Phi-4 | MATH500 | 12 | 57.1 | 52.7 | 4.4 |
| 23 | 65 | Phi-4 | MATH500 | 12 | 57.7333 | 52.9667 | 4.7667 |
| 24 | 66 | Phi-4 | MATH500 | 12 | 57.0333 | 52.7667 | 4.2667 |
| 25 | 67 | Phi-4 | MATH500 | 12 | 57.2333 | 52.8 | 4.4333 |
| 26 | 68 | Phi-4 | MATH500 | 12 | 57.3333 | 53.2333 | 4.1 |
| 27 | 69 | Phi-4 | MATH500 | 12 | 57.3333 | 53.6333 | 3.7 |
| 28 | 70 | Phi-4 | MATH500 | 12 | 57.3 | 53.0333 | 4.2667 |
| 29 | 71 | Phi-4 | MATH500 | 12 | 56.2333 | 53.4667 | 2.7667 |
| 30 | 72 | Phi-4 | MATH500 | 12 | 57.6667 | 53.4667 | 4.2 |
| 31 | 73 | Phi-4 | MATH500 | 12 | 57.3333 | 53.4333 | 3.9 |
| 32 | 74 | Phi-4 | MATH500 | 12 | 57.9333 | 52.6333 | 5.3 |
| 33 | 75 | Phi-4 | MATH500 | 12 | 57.2 | 53.3667 | 3.8333 |
| 34 | 76 | Phi-4 | MATH500 | 12 | 57.3667 | 52.5333 | 4.8333 |
| 35 | 77 | Phi-4 | MATH500 | 12 | 57.3333 | 53.2667 | 4.0667 |
| 36 | 78 | Phi-4 | MATH500 | 12 | 57.3 | 53.7 | 3.6 |
| 37 | 79 | Phi-4 | MATH500 | 12 | 58.2667 | 52.5 | 5.7667 |
| 38 | 80 | Phi-4 | MATH500 | 12 | 58.3667 | 53.6667 | 4.7 |
| 39 | 81 | Phi-4 | MATH500 | 12 | 57.6333 | 53.9 | 3.7333 |
| 40 | 82 | Phi-4 | MATH500 | 12 | 58.0667 | 52.9333 | 5.1333 |
| 41 | 83 | Phi-4 | MATH500 | 12 | 57.7 | 53.0333 | 4.6667 |
| 42 | 84 | Phi-4 | MATH500 | 12 | 57.6 | 53.6333 | 3.9667 |
| 43 | 85 | Phi-4 | MATH500 | 12 | 57.7 | 52.8 | 4.9 |
| 44 | 86 | Phi-4 | MATH500 | 12 | 57.5 | 52.9 | 4.6 |
| 45 | 87 | Phi-4 | MATH500 | 12 | 57.1667 | 52.7667 | 4.4 |
| 46 | 88 | Phi-4 | MATH500 | 12 | 56.7333 | 53.7 | 3.0333 |
| 47 | 89 | Phi-4 | MATH500 | 12 | 57.2 | 53.6667 | 3.5333 |
| 48 | 90 | Phi-4 | MATH500 | 12 | 56.8333 | 53.7333 | 3.1 |
| 49 | 91 | Phi-4 | MATH500 | 12 | 57.8 | 53.0667 | 4.7333 |
| 0 | 42 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 1 | 43 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 2 | 44 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 3 | 45 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 4 | 46 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 5 | 47 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 6 | 48 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 7 | 49 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 8 | 50 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 9 | 51 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 10 | 52 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 11 | 53 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 12 | 54 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 13 | 55 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 14 | 56 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 15 | 57 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 16 | 58 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 17 | 59 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 18 | 60 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 19 | 61 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 20 | 62 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 21 | 63 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 22 | 64 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 23 | 65 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 24 | 66 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 25 | 67 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 26 | 68 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 27 | 69 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 28 | 70 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 29 | 71 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 30 | 72 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 31 | 73 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 32 | 74 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 33 | 75 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 34 | 76 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 35 | 77 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 36 | 78 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 37 | 79 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 38 | 80 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 39 | 81 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 40 | 82 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 41 | 83 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 42 | 84 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 43 | 85 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 44 | 86 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 45 | 87 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 46 | 88 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 47 | 89 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 48 | 90 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 49 | 91 | Phi-4 | MATH500 | 16 | 57.475 | 53.125 | 4.35 |
| 0 | 42 | Phi-4 | SVAMP | 4 | 54.65 | 58.75 | -4.1 |
| 1 | 43 | Phi-4 | SVAMP | 4 | 55.6 | 59.45 | -3.85 |
| 2 | 44 | Phi-4 | SVAMP | 4 | 54.6 | 59.8 | -5.2 |
| 3 | 45 | Phi-4 | SVAMP | 4 | 54.25 | 58.85 | -4.6 |
| 4 | 46 | Phi-4 | SVAMP | 4 | 54.25 | 59.85 | -5.6 |
| 5 | 47 | Phi-4 | SVAMP | 4 | 55.15 | 58.1 | -2.95 |
| 6 | 48 | Phi-4 | SVAMP | 4 | 56.65 | 59 | -2.35 |
| 7 | 49 | Phi-4 | SVAMP | 4 | 54.6 | 60.15 | -5.55 |
| 8 | 50 | Phi-4 | SVAMP | 4 | 53.25 | 60.7 | -7.45 |
| 9 | 51 | Phi-4 | SVAMP | 4 | 55.55 | 59.2 | -3.65 |
| 10 | 52 | Phi-4 | SVAMP | 4 | 54.75 | 58.05 | -3.3 |
| 11 | 53 | Phi-4 | SVAMP | 4 | 55.45 | 60.05 | -4.6 |
| 12 | 54 | Phi-4 | SVAMP | 4 | 55.75 | 59.4 | -3.65 |
| 13 | 55 | Phi-4 | SVAMP | 4 | 54.2 | 60.2 | -6 |
| 14 | 56 | Phi-4 | SVAMP | 4 | 55.45 | 60.7 | -5.25 |
| 15 | 57 | Phi-4 | SVAMP | 4 | 53.75 | 60.35 | -6.6 |
| 16 | 58 | Phi-4 | SVAMP | 4 | 54.6 | 59.95 | -5.35 |
| 17 | 59 | Phi-4 | SVAMP | 4 | 54.1 | 59.3 | -5.2 |
| 18 | 60 | Phi-4 | SVAMP | 4 | 54.45 | 59.65 | -5.2 |
| 19 | 61 | Phi-4 | SVAMP | 4 | 54.65 | 59.8 | -5.15 |
| 20 | 62 | Phi-4 | SVAMP | 4 | 55.15 | 58.85 | -3.7 |
| 21 | 63 | Phi-4 | SVAMP | 4 | 56.3 | 60.35 | -4.05 |
| 22 | 64 | Phi-4 | SVAMP | 4 | 54.1 | 60.2 | -6.1 |
| 23 | 65 | Phi-4 | SVAMP | 4 | 52.4 | 59.15 | -6.75 |
| 24 | 66 | Phi-4 | SVAMP | 4 | 55.45 | 59.4 | -3.95 |
| 25 | 67 | Phi-4 | SVAMP | 4 | 53.65 | 60.25 | -6.6 |
| 26 | 68 | Phi-4 | SVAMP | 4 | 55.05 | 58.5 | -3.45 |
| 27 | 69 | Phi-4 | SVAMP | 4 | 54.7 | 58.55 | -3.85 |
| 28 | 70 | Phi-4 | SVAMP | 4 | 53.4 | 59.45 | -6.05 |
| 29 | 71 | Phi-4 | SVAMP | 4 | 55.65 | 59.55 | -3.9 |
| 30 | 72 | Phi-4 | SVAMP | 4 | 54.3 | 58.9 | -4.6 |
| 31 | 73 | Phi-4 | SVAMP | 4 | 55.2 | 61.3 | -6.1 |
| 32 | 74 | Phi-4 | SVAMP | 4 | 54.35 | 59 | -4.65 |
| 33 | 75 | Phi-4 | SVAMP | 4 | 56.65 | 58.4 | -1.75 |
| 34 | 76 | Phi-4 | SVAMP | 4 | 54.1 | 60.15 | -6.05 |
| 35 | 77 | Phi-4 | SVAMP | 4 | 54.65 | 58.95 | -4.3 |
| 36 | 78 | Phi-4 | SVAMP | 4 | 54.8 | 59.5 | -4.7 |
| 37 | 79 | Phi-4 | SVAMP | 4 | 54.3 | 58.35 | -4.05 |
| 38 | 80 | Phi-4 | SVAMP | 4 | 55 | 59.5 | -4.5 |
| 39 | 81 | Phi-4 | SVAMP | 4 | 54.3 | 58.85 | -4.55 |
| 40 | 82 | Phi-4 | SVAMP | 4 | 54.55 | 58.7 | -4.15 |
| 41 | 83 | Phi-4 | SVAMP | 4 | 54.85 | 58.4 | -3.55 |
| 42 | 84 | Phi-4 | SVAMP | 4 | 55 | 59.05 | -4.05 |
| 43 | 85 | Phi-4 | SVAMP | 4 | 55.45 | 59.45 | -4 |
| 44 | 86 | Phi-4 | SVAMP | 4 | 55.35 | 59.85 | -4.5 |
| 45 | 87 | Phi-4 | SVAMP | 4 | 54.2 | 59.45 | -5.25 |
| 46 | 88 | Phi-4 | SVAMP | 4 | 55.55 | 59.6 | -4.05 |
| 47 | 89 | Phi-4 | SVAMP | 4 | 54.1 | 60.05 | -5.95 |
| 48 | 90 | Phi-4 | SVAMP | 4 | 55.95 | 58.95 | -3 |
| 49 | 91 | Phi-4 | SVAMP | 4 | 54.15 | 58.55 | -4.4 |
| 0 | 42 | Phi-4 | SVAMP | 8 | 53.75 | 58.95 | -5.2 |
| 1 | 43 | Phi-4 | SVAMP | 8 | 54.575 | 59.85 | -5.275 |
| 2 | 44 | Phi-4 | SVAMP | 8 | 54.975 | 59.95 | -4.975 |
| 3 | 45 | Phi-4 | SVAMP | 8 | 54.15 | 59.625 | -5.475 |
| 4 | 46 | Phi-4 | SVAMP | 8 | 55.075 | 59.9 | -4.825 |
| 5 | 47 | Phi-4 | SVAMP | 8 | 53.975 | 58.625 | -4.65 |
| 6 | 48 | Phi-4 | SVAMP | 8 | 55.125 | 60 | -4.875 |
| 7 | 49 | Phi-4 | SVAMP | 8 | 54.425 | 59.825 | -5.4 |
| 8 | 50 | Phi-4 | SVAMP | 8 | 54.525 | 60.475 | -5.95 |
| 9 | 51 | Phi-4 | SVAMP | 8 | 54.575 | 59.45 | -4.875 |
| 10 | 52 | Phi-4 | SVAMP | 8 | 55.425 | 58.525 | -3.1 |
| 11 | 53 | Phi-4 | SVAMP | 8 | 54.925 | 59.6 | -4.675 |
| 12 | 54 | Phi-4 | SVAMP | 8 | 54.875 | 59.025 | -4.15 |
| 13 | 55 | Phi-4 | SVAMP | 8 | 55.2 | 59.8 | -4.6 |
| 14 | 56 | Phi-4 | SVAMP | 8 | 55.3 | 58.975 | -3.675 |
| 15 | 57 | Phi-4 | SVAMP | 8 | 55.45 | 59.625 | -4.175 |
| 16 | 58 | Phi-4 | SVAMP | 8 | 54.35 | 59.325 | -4.975 |
| 17 | 59 | Phi-4 | SVAMP | 8 | 55.1 | 58.925 | -3.825 |
| 18 | 60 | Phi-4 | SVAMP | 8 | 54.25 | 59.5 | -5.25 |
| 19 | 61 | Phi-4 | SVAMP | 8 | 55.125 | 59.05 | -3.925 |
| 20 | 62 | Phi-4 | SVAMP | 8 | 54.8 | 59.425 | -4.625 |
| 21 | 63 | Phi-4 | SVAMP | 8 | 54.95 | 59.975 | -5.025 |
| 22 | 64 | Phi-4 | SVAMP | 8 | 54.375 | 60.4 | -6.025 |
| 23 | 65 | Phi-4 | SVAMP | 8 | 54.275 | 58.925 | -4.65 |
| 24 | 66 | Phi-4 | SVAMP | 8 | 54.45 | 59.8 | -5.35 |
| 25 | 67 | Phi-4 | SVAMP | 8 | 53.425 | 59.9 | -6.475 |
| 26 | 68 | Phi-4 | SVAMP | 8 | 54.175 | 59.4 | -5.225 |
| 27 | 69 | Phi-4 | SVAMP | 8 | 54.7 | 59.675 | -4.975 |
| 28 | 70 | Phi-4 | SVAMP | 8 | 54.6 | 58.825 | -4.225 |
| 29 | 71 | Phi-4 | SVAMP | 8 | 55.25 | 58.325 | -3.075 |
| 30 | 72 | Phi-4 | SVAMP | 8 | 54.65 | 59.425 | -4.775 |
| 31 | 73 | Phi-4 | SVAMP | 8 | 54.475 | 60.7 | -6.225 |
| 32 | 74 | Phi-4 | SVAMP | 8 | 55.25 | 58.875 | -3.625 |
| 33 | 75 | Phi-4 | SVAMP | 8 | 55.075 | 59.45 | -4.375 |
| 34 | 76 | Phi-4 | SVAMP | 8 | 55 | 59.125 | -4.125 |
| 35 | 77 | Phi-4 | SVAMP | 8 | 54.825 | 60 | -5.175 |
| 36 | 78 | Phi-4 | SVAMP | 8 | 54.65 | 60.25 | -5.6 |
| 37 | 79 | Phi-4 | SVAMP | 8 | 54.65 | 58.975 | -4.325 |
| 38 | 80 | Phi-4 | SVAMP | 8 | 54.75 | 58.775 | -4.025 |
| 39 | 81 | Phi-4 | SVAMP | 8 | 54.375 | 59.175 | -4.8 |
| 40 | 82 | Phi-4 | SVAMP | 8 | 54.225 | 59.225 | -5 |
| 41 | 83 | Phi-4 | SVAMP | 8 | 54.35 | 59.4 | -5.05 |
| 42 | 84 | Phi-4 | SVAMP | 8 | 55.225 | 58.725 | -3.5 |
| 43 | 85 | Phi-4 | SVAMP | 8 | 54.825 | 58.85 | -4.025 |
| 44 | 86 | Phi-4 | SVAMP | 8 | 55.375 | 59.4 | -4.025 |
| 45 | 87 | Phi-4 | SVAMP | 8 | 54.075 | 59.825 | -5.75 |
| 46 | 88 | Phi-4 | SVAMP | 8 | 55.075 | 60.2 | -5.125 |
| 47 | 89 | Phi-4 | SVAMP | 8 | 54.6 | 59.375 | -4.775 |
| 48 | 90 | Phi-4 | SVAMP | 8 | 55.025 | 59.775 | -4.75 |
| 49 | 91 | Phi-4 | SVAMP | 8 | 54.725 | 58.225 | -3.5 |
| 0 | 42 | Phi-4 | SVAMP | 12 | 54.65 | 59.15 | -4.5 |
| 1 | 43 | Phi-4 | SVAMP | 12 | 54.6833 | 59.4833 | -4.8 |
| 2 | 44 | Phi-4 | SVAMP | 12 | 54.6833 | 60.1167 | -5.4333 |
| 3 | 45 | Phi-4 | SVAMP | 12 | 54.3167 | 59.5 | -5.1833 |
| 4 | 46 | Phi-4 | SVAMP | 12 | 54.4333 | 59.25 | -4.8167 |
| 5 | 47 | Phi-4 | SVAMP | 12 | 53.9167 | 59.2833 | -5.3667 |
| 6 | 48 | Phi-4 | SVAMP | 12 | 54.45 | 59.7667 | -5.3167 |
| 7 | 49 | Phi-4 | SVAMP | 12 | 54.45 | 59.7333 | -5.2833 |
| 8 | 50 | Phi-4 | SVAMP | 12 | 54.85 | 60.0333 | -5.1833 |
| 9 | 51 | Phi-4 | SVAMP | 12 | 54.5167 | 59.7333 | -5.2167 |
| 10 | 52 | Phi-4 | SVAMP | 12 | 54.7833 | 59.3833 | -4.6 |
| 11 | 53 | Phi-4 | SVAMP | 12 | 54.6667 | 59.2833 | -4.6167 |
| 12 | 54 | Phi-4 | SVAMP | 12 | 54.8333 | 59.85 | -5.0167 |
| 13 | 55 | Phi-4 | SVAMP | 12 | 54.3833 | 59.9 | -5.5167 |
| 14 | 56 | Phi-4 | SVAMP | 12 | 54.55 | 59.2333 | -4.6833 |
| 15 | 57 | Phi-4 | SVAMP | 12 | 55.0667 | 59.35 | -4.2833 |
| 16 | 58 | Phi-4 | SVAMP | 12 | 55.15 | 59.4 | -4.25 |
| 17 | 59 | Phi-4 | SVAMP | 12 | 55.0167 | 59 | -3.9833 |
| 18 | 60 | Phi-4 | SVAMP | 12 | 54.5333 | 59.6 | -5.0667 |
| 19 | 61 | Phi-4 | SVAMP | 12 | 54.8833 | 59.0833 | -4.2 |
| 20 | 62 | Phi-4 | SVAMP | 12 | 55.25 | 59.4667 | -4.2167 |
| 21 | 63 | Phi-4 | SVAMP | 12 | 54.5 | 59.65 | -5.15 |
| 22 | 64 | Phi-4 | SVAMP | 12 | 54.4667 | 60.1 | -5.6333 |
| 23 | 65 | Phi-4 | SVAMP | 12 | 54.55 | 59.1667 | -4.6167 |
| 24 | 66 | Phi-4 | SVAMP | 12 | 55.2 | 59.3667 | -4.1667 |
| 25 | 67 | Phi-4 | SVAMP | 12 | 53.9667 | 59.2167 | -5.25 |
| 26 | 68 | Phi-4 | SVAMP | 12 | 54.8 | 58.9 | -4.1 |
| 27 | 69 | Phi-4 | SVAMP | 12 | 54.6167 | 59.7 | -5.0833 |
| 28 | 70 | Phi-4 | SVAMP | 12 | 54.4833 | 59.25 | -4.7667 |
| 29 | 71 | Phi-4 | SVAMP | 12 | 54.9167 | 58.8333 | -3.9167 |
| 30 | 72 | Phi-4 | SVAMP | 12 | 54.3833 | 59.7833 | -5.4 |
| 31 | 73 | Phi-4 | SVAMP | 12 | 54.7833 | 59.7167 | -4.9333 |
| 32 | 74 | Phi-4 | SVAMP | 12 | 54.7667 | 58.6833 | -3.9167 |
| 33 | 75 | Phi-4 | SVAMP | 12 | 54.8667 | 59.6 | -4.7333 |
| 34 | 76 | Phi-4 | SVAMP | 12 | 55 | 59.5 | -4.5 |
| 35 | 77 | Phi-4 | SVAMP | 12 | 54.6 | 59.8 | -5.2 |
| 36 | 78 | Phi-4 | SVAMP | 12 | 54.3 | 59.45 | -5.15 |
| 37 | 79 | Phi-4 | SVAMP | 12 | 54.4833 | 59.65 | -5.1667 |
| 38 | 80 | Phi-4 | SVAMP | 12 | 54.6167 | 59.1833 | -4.5667 |
| 39 | 81 | Phi-4 | SVAMP | 12 | 54.7167 | 59.0833 | -4.3667 |
| 40 | 82 | Phi-4 | SVAMP | 12 | 54.4833 | 59.2667 | -4.7833 |
| 41 | 83 | Phi-4 | SVAMP | 12 | 54.4333 | 59.8333 | -5.4 |
| 42 | 84 | Phi-4 | SVAMP | 12 | 54.45 | 59.4 | -4.95 |
| 43 | 85 | Phi-4 | SVAMP | 12 | 54.6833 | 58.9833 | -4.3 |
| 44 | 86 | Phi-4 | SVAMP | 12 | 54.4333 | 59.6833 | -5.25 |
| 45 | 87 | Phi-4 | SVAMP | 12 | 54.8167 | 59.75 | -4.9333 |
| 46 | 88 | Phi-4 | SVAMP | 12 | 54.6667 | 59.5833 | -4.9167 |
| 47 | 89 | Phi-4 | SVAMP | 12 | 54.5 | 58.8833 | -4.3833 |
| 48 | 90 | Phi-4 | SVAMP | 12 | 54.8833 | 59.5667 | -4.6833 |
| 49 | 91 | Phi-4 | SVAMP | 12 | 54.9833 | 58.35 | -3.3667 |
| 0 | 42 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 1 | 43 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 2 | 44 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 3 | 45 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 4 | 46 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 5 | 47 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 6 | 48 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 7 | 49 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 8 | 50 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 9 | 51 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 10 | 52 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 11 | 53 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 12 | 54 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 13 | 55 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 14 | 56 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 15 | 57 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 16 | 58 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 17 | 59 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 18 | 60 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 19 | 61 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 20 | 62 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 21 | 63 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 22 | 64 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 23 | 65 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 24 | 66 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 25 | 67 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 26 | 68 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 27 | 69 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 28 | 70 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 29 | 71 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 30 | 72 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 31 | 73 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 32 | 74 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 33 | 75 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 34 | 76 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 35 | 77 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 36 | 78 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 37 | 79 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 38 | 80 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 39 | 81 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 40 | 82 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 41 | 83 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 42 | 84 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 43 | 85 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 44 | 86 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 45 | 87 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 46 | 88 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 47 | 89 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 48 | 90 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 49 | 91 | Phi-4 | SVAMP | 16 | 54.525 | 59.5625 | -5.0375 |
| 0 | 42 | Qwen2.5-7B | AQuA | 4 | 62.5984 | 60.4331 | 2.1654 |
| 1 | 43 | Qwen2.5-7B | AQuA | 4 | 64.1732 | 62.4016 | 1.7717 |
| 2 | 44 | Qwen2.5-7B | AQuA | 4 | 65.1575 | 61.811 | 3.3465 |
| 3 | 45 | Qwen2.5-7B | AQuA | 4 | 66.1417 | 60.6299 | 5.5118 |
| 4 | 46 | Qwen2.5-7B | AQuA | 4 | 62.9921 | 63.3858 | -0.3937 |
| 5 | 47 | Qwen2.5-7B | AQuA | 4 | 63.189 | 60.0394 | 3.1496 |
| 6 | 48 | Qwen2.5-7B | AQuA | 4 | 60.4331 | 64.3701 | -3.937 |
| 7 | 49 | Qwen2.5-7B | AQuA | 4 | 63.7795 | 59.252 | 4.5276 |
| 8 | 50 | Qwen2.5-7B | AQuA | 4 | 61.2205 | 61.0236 | 0.1969 |
| 9 | 51 | Qwen2.5-7B | AQuA | 4 | 62.4016 | 61.4173 | 0.9843 |
| 10 | 52 | Qwen2.5-7B | AQuA | 4 | 62.4016 | 61.2205 | 1.1811 |
| 11 | 53 | Qwen2.5-7B | AQuA | 4 | 62.5984 | 63.5827 | -0.9843 |
| 12 | 54 | Qwen2.5-7B | AQuA | 4 | 62.4016 | 62.4016 | 0 |
| 13 | 55 | Qwen2.5-7B | AQuA | 4 | 63.189 | 62.2047 | 0.9843 |
| 14 | 56 | Qwen2.5-7B | AQuA | 4 | 67.3228 | 63.7795 | 3.5433 |
| 15 | 57 | Qwen2.5-7B | AQuA | 4 | 60.8268 | 62.4016 | -1.5748 |
| 16 | 58 | Qwen2.5-7B | AQuA | 4 | 62.9921 | 59.252 | 3.7402 |
| 17 | 59 | Qwen2.5-7B | AQuA | 4 | 66.7323 | 60.6299 | 6.1024 |
| 18 | 60 | Qwen2.5-7B | AQuA | 4 | 62.9921 | 60.4331 | 2.5591 |
| 19 | 61 | Qwen2.5-7B | AQuA | 4 | 63.189 | 64.1732 | -0.9843 |
| 20 | 62 | Qwen2.5-7B | AQuA | 4 | 62.4016 | 61.0236 | 1.378 |
| 21 | 63 | Qwen2.5-7B | AQuA | 4 | 63.189 | 60.8268 | 2.3622 |
| 22 | 64 | Qwen2.5-7B | AQuA | 4 | 64.3701 | 64.5669 | -0.1969 |
| 23 | 65 | Qwen2.5-7B | AQuA | 4 | 61.811 | 63.189 | -1.378 |
| 24 | 66 | Qwen2.5-7B | AQuA | 4 | 62.5984 | 61.811 | 0.7874 |
| 25 | 67 | Qwen2.5-7B | AQuA | 4 | 63.5827 | 63.7795 | -0.1969 |
| 26 | 68 | Qwen2.5-7B | AQuA | 4 | 65.1575 | 59.252 | 5.9055 |
| 27 | 69 | Qwen2.5-7B | AQuA | 4 | 63.5827 | 62.9921 | 0.5906 |
| 28 | 70 | Qwen2.5-7B | AQuA | 4 | 62.9921 | 61.2205 | 1.7717 |
| 29 | 71 | Qwen2.5-7B | AQuA | 4 | 63.9764 | 60.2362 | 3.7402 |
| 30 | 72 | Qwen2.5-7B | AQuA | 4 | 64.3701 | 62.5984 | 1.7717 |
| 31 | 73 | Qwen2.5-7B | AQuA | 4 | 62.4016 | 61.0236 | 1.378 |
| 32 | 74 | Qwen2.5-7B | AQuA | 4 | 65.3543 | 60.4331 | 4.9213 |
| 33 | 75 | Qwen2.5-7B | AQuA | 4 | 60.8268 | 62.2047 | -1.378 |
| 34 | 76 | Qwen2.5-7B | AQuA | 4 | 62.4016 | 58.6614 | 3.7402 |
| 35 | 77 | Qwen2.5-7B | AQuA | 4 | 63.3858 | 60.6299 | 2.7559 |
| 36 | 78 | Qwen2.5-7B | AQuA | 4 | 62.0079 | 60.4331 | 1.5748 |
| 37 | 79 | Qwen2.5-7B | AQuA | 4 | 66.1417 | 63.189 | 2.9528 |
| 38 | 80 | Qwen2.5-7B | AQuA | 4 | 62.9921 | 62.0079 | 0.9843 |
| 39 | 81 | Qwen2.5-7B | AQuA | 4 | 63.9764 | 62.0079 | 1.9685 |
| 40 | 82 | Qwen2.5-7B | AQuA | 4 | 64.1732 | 59.252 | 4.9213 |
| 41 | 83 | Qwen2.5-7B | AQuA | 4 | 64.7638 | 63.7795 | 0.9843 |
| 42 | 84 | Qwen2.5-7B | AQuA | 4 | 63.3858 | 61.811 | 1.5748 |
| 43 | 85 | Qwen2.5-7B | AQuA | 4 | 65.1575 | 58.8583 | 6.2992 |
| 44 | 86 | Qwen2.5-7B | AQuA | 4 | 61.6142 | 60.0394 | 1.5748 |
| 45 | 87 | Qwen2.5-7B | AQuA | 4 | 64.9606 | 64.1732 | 0.7874 |
| 46 | 88 | Qwen2.5-7B | AQuA | 4 | 66.3386 | 63.9764 | 2.3622 |
| 47 | 89 | Qwen2.5-7B | AQuA | 4 | 61.811 | 58.0709 | 3.7402 |
| 48 | 90 | Qwen2.5-7B | AQuA | 4 | 62.2047 | 61.4173 | 0.7874 |
| 49 | 91 | Qwen2.5-7B | AQuA | 4 | 65.748 | 62.7953 | 2.9528 |
| 0 | 42 | Qwen2.5-7B | AQuA | 8 | 63.189 | 61.7126 | 1.4764 |
| 1 | 43 | Qwen2.5-7B | AQuA | 8 | 63.878 | 61.2205 | 2.6575 |
| 2 | 44 | Qwen2.5-7B | AQuA | 8 | 63.5827 | 62.7953 | 0.7874 |
| 3 | 45 | Qwen2.5-7B | AQuA | 8 | 62.5984 | 62.5 | 0.0984 |
| 4 | 46 | Qwen2.5-7B | AQuA | 8 | 62.8937 | 61.3189 | 1.5748 |
| 5 | 47 | Qwen2.5-7B | AQuA | 8 | 64.1732 | 60.7283 | 3.4449 |
| 6 | 48 | Qwen2.5-7B | AQuA | 8 | 62.2047 | 62.8937 | -0.689 |
| 7 | 49 | Qwen2.5-7B | AQuA | 8 | 63.189 | 60.0394 | 3.1496 |
| 8 | 50 | Qwen2.5-7B | AQuA | 8 | 61.811 | 62.3031 | -0.4921 |
| 9 | 51 | Qwen2.5-7B | AQuA | 8 | 62.6969 | 61.9094 | 0.7874 |
| 10 | 52 | Qwen2.5-7B | AQuA | 8 | 63.5827 | 62.6969 | 0.8858 |
| 11 | 53 | Qwen2.5-7B | AQuA | 8 | 64.3701 | 62.2047 | 2.1654 |
| 12 | 54 | Qwen2.5-7B | AQuA | 8 | 62.6969 | 62.9921 | -0.2953 |
| 13 | 55 | Qwen2.5-7B | AQuA | 8 | 64.9606 | 61.3189 | 3.6417 |
| 14 | 56 | Qwen2.5-7B | AQuA | 8 | 63.7795 | 63.7795 | 0 |
| 15 | 57 | Qwen2.5-7B | AQuA | 8 | 63.3858 | 62.3031 | 1.0827 |
| 16 | 58 | Qwen2.5-7B | AQuA | 8 | 64.9606 | 61.6142 | 3.3465 |
| 17 | 59 | Qwen2.5-7B | AQuA | 8 | 63.878 | 62.3031 | 1.5748 |
| 18 | 60 | Qwen2.5-7B | AQuA | 8 | 62.8937 | 61.2205 | 1.6732 |
| 19 | 61 | Qwen2.5-7B | AQuA | 8 | 62.9921 | 62.6969 | 0.2953 |
| 20 | 62 | Qwen2.5-7B | AQuA | 8 | 63.189 | 60.0394 | 3.1496 |
| 21 | 63 | Qwen2.5-7B | AQuA | 8 | 63.2874 | 61.2205 | 2.0669 |
| 22 | 64 | Qwen2.5-7B | AQuA | 8 | 64.7638 | 62.4016 | 2.3622 |
| 23 | 65 | Qwen2.5-7B | AQuA | 8 | 63.3858 | 62.0079 | 1.378 |
| 24 | 66 | Qwen2.5-7B | AQuA | 8 | 62.8937 | 60.1378 | 2.7559 |
| 25 | 67 | Qwen2.5-7B | AQuA | 8 | 64.2717 | 63.2874 | 0.9843 |
| 26 | 68 | Qwen2.5-7B | AQuA | 8 | 62.0079 | 61.9094 | 0.0984 |
| 27 | 69 | Qwen2.5-7B | AQuA | 8 | 62.9921 | 62.5 | 0.4921 |
| 28 | 70 | Qwen2.5-7B | AQuA | 8 | 62.9921 | 61.122 | 1.8701 |
| 29 | 71 | Qwen2.5-7B | AQuA | 8 | 64.8622 | 59.7441 | 5.1181 |
| 30 | 72 | Qwen2.5-7B | AQuA | 8 | 62.5 | 62.7953 | -0.2953 |
| 31 | 73 | Qwen2.5-7B | AQuA | 8 | 64.2717 | 61.5157 | 2.7559 |
| 32 | 74 | Qwen2.5-7B | AQuA | 8 | 64.4685 | 60.4331 | 4.0354 |
| 33 | 75 | Qwen2.5-7B | AQuA | 8 | 61.6142 | 61.4173 | 0.1969 |
| 34 | 76 | Qwen2.5-7B | AQuA | 8 | 65.6496 | 60.5315 | 5.1181 |
| 35 | 77 | Qwen2.5-7B | AQuA | 8 | 65.3543 | 60.1378 | 5.2165 |
| 36 | 78 | Qwen2.5-7B | AQuA | 8 | 63.6811 | 61.5157 | 2.1654 |
| 37 | 79 | Qwen2.5-7B | AQuA | 8 | 63.2874 | 63.3858 | -0.0984 |
| 38 | 80 | Qwen2.5-7B | AQuA | 8 | 63.9764 | 60.4331 | 3.5433 |
| 39 | 81 | Qwen2.5-7B | AQuA | 8 | 63.2874 | 62.8937 | 0.3937 |
| 40 | 82 | Qwen2.5-7B | AQuA | 8 | 63.7795 | 61.811 | 1.9685 |
| 41 | 83 | Qwen2.5-7B | AQuA | 8 | 62.6969 | 62.6969 | 0 |
| 42 | 84 | Qwen2.5-7B | AQuA | 8 | 63.4843 | 61.9094 | 1.5748 |
| 43 | 85 | Qwen2.5-7B | AQuA | 8 | 63.6811 | 60.6299 | 3.0512 |
| 44 | 86 | Qwen2.5-7B | AQuA | 8 | 63.5827 | 61.122 | 2.4606 |
| 45 | 87 | Qwen2.5-7B | AQuA | 8 | 65.1575 | 62.4016 | 2.7559 |
| 46 | 88 | Qwen2.5-7B | AQuA | 8 | 63.7795 | 63.7795 | 0 |
| 47 | 89 | Qwen2.5-7B | AQuA | 8 | 64.1732 | 61.0236 | 3.1496 |
| 48 | 90 | Qwen2.5-7B | AQuA | 8 | 61.4173 | 62.3031 | -0.8858 |
| 49 | 91 | Qwen2.5-7B | AQuA | 8 | 66.0433 | 61.5157 | 4.5276 |
| 0 | 42 | Qwen2.5-7B | AQuA | 12 | 63.189 | 62.0079 | 1.1811 |
| 1 | 43 | Qwen2.5-7B | AQuA | 12 | 63.0577 | 62.4672 | 0.5906 |
| 2 | 44 | Qwen2.5-7B | AQuA | 12 | 63.3858 | 62.4672 | 0.9186 |
| 3 | 45 | Qwen2.5-7B | AQuA | 12 | 63.5171 | 62.2703 | 1.2467 |
| 4 | 46 | Qwen2.5-7B | AQuA | 12 | 62.5328 | 60.8268 | 1.706 |
| 5 | 47 | Qwen2.5-7B | AQuA | 12 | 62.7953 | 61.4173 | 1.378 |
| 6 | 48 | Qwen2.5-7B | AQuA | 12 | 62.9921 | 62.664 | 0.3281 |
| 7 | 49 | Qwen2.5-7B | AQuA | 12 | 63.6483 | 61.4829 | 2.1654 |
| 8 | 50 | Qwen2.5-7B | AQuA | 12 | 62.7297 | 61.6798 | 1.0499 |
| 9 | 51 | Qwen2.5-7B | AQuA | 12 | 63.5827 | 60.8924 | 2.6903 |
| 10 | 52 | Qwen2.5-7B | AQuA | 12 | 63.5171 | 62.8609 | 0.6562 |
| 11 | 53 | Qwen2.5-7B | AQuA | 12 | 62.9921 | 62.5984 | 0.3937 |
| 12 | 54 | Qwen2.5-7B | AQuA | 12 | 63.5827 | 62.2703 | 1.3123 |
| 13 | 55 | Qwen2.5-7B | AQuA | 12 | 63.2546 | 61.9423 | 1.3123 |
| 14 | 56 | Qwen2.5-7B | AQuA | 12 | 64.3045 | 62.9265 | 1.378 |
| 15 | 57 | Qwen2.5-7B | AQuA | 12 | 63.189 | 61.8766 | 1.3123 |
| 16 | 58 | Qwen2.5-7B | AQuA | 12 | 64.1732 | 62.1391 | 2.0341 |
| 17 | 59 | Qwen2.5-7B | AQuA | 12 | 63.7139 | 61.811 | 1.9029 |
| 18 | 60 | Qwen2.5-7B | AQuA | 12 | 63.4514 | 60.6955 | 2.7559 |
| 19 | 61 | Qwen2.5-7B | AQuA | 12 | 63.5827 | 62.1391 | 1.4436 |
| 20 | 62 | Qwen2.5-7B | AQuA | 12 | 63.0577 | 61.7454 | 1.3123 |
| 21 | 63 | Qwen2.5-7B | AQuA | 12 | 62.664 | 61.5486 | 1.1155 |
| 22 | 64 | Qwen2.5-7B | AQuA | 12 | 63.4514 | 62.4672 | 0.9843 |
| 23 | 65 | Qwen2.5-7B | AQuA | 12 | 62.7953 | 60.7612 | 2.0341 |
| 24 | 66 | Qwen2.5-7B | AQuA | 12 | 62.2703 | 61.6798 | 0.5906 |
| 25 | 67 | Qwen2.5-7B | AQuA | 12 | 63.7795 | 61.9423 | 1.8373 |
| 26 | 68 | Qwen2.5-7B | AQuA | 12 | 62.664 | 62.5984 | 0.0656 |
| 27 | 69 | Qwen2.5-7B | AQuA | 12 | 63.6483 | 62.7297 | 0.9186 |
| 28 | 70 | Qwen2.5-7B | AQuA | 12 | 63.7795 | 62.2047 | 1.5748 |
| 29 | 71 | Qwen2.5-7B | AQuA | 12 | 64.7638 | 60.2362 | 4.5276 |
| 30 | 72 | Qwen2.5-7B | AQuA | 12 | 63.2546 | 61.7454 | 1.5092 |
| 31 | 73 | Qwen2.5-7B | AQuA | 12 | 62.5328 | 61.2205 | 1.3123 |
| 32 | 74 | Qwen2.5-7B | AQuA | 12 | 65.2231 | 61.2205 | 4.0026 |
| 33 | 75 | Qwen2.5-7B | AQuA | 12 | 61.811 | 62.0735 | -0.2625 |
| 34 | 76 | Qwen2.5-7B | AQuA | 12 | 64.042 | 61.5486 | 2.4934 |
| 35 | 77 | Qwen2.5-7B | AQuA | 12 | 63.3202 | 62.1391 | 1.1811 |
| 36 | 78 | Qwen2.5-7B | AQuA | 12 | 64.2388 | 62.0079 | 2.231 |
| 37 | 79 | Qwen2.5-7B | AQuA | 12 | 62.9265 | 61.7454 | 1.1811 |
| 38 | 80 | Qwen2.5-7B | AQuA | 12 | 62.9921 | 61.1549 | 1.8373 |
| 39 | 81 | Qwen2.5-7B | AQuA | 12 | 64.2388 | 62.2703 | 1.9685 |
| 40 | 82 | Qwen2.5-7B | AQuA | 12 | 62.664 | 62.336 | 0.3281 |
| 41 | 83 | Qwen2.5-7B | AQuA | 12 | 63.1234 | 62.0079 | 1.1155 |
| 42 | 84 | Qwen2.5-7B | AQuA | 12 | 63.5171 | 62.0079 | 1.5092 |
| 43 | 85 | Qwen2.5-7B | AQuA | 12 | 62.7953 | 62.4016 | 0.3937 |
| 44 | 86 | Qwen2.5-7B | AQuA | 12 | 62.9921 | 61.4173 | 1.5748 |
| 45 | 87 | Qwen2.5-7B | AQuA | 12 | 64.7638 | 61.6142 | 3.1496 |
| 46 | 88 | Qwen2.5-7B | AQuA | 12 | 63.0577 | 61.6798 | 1.378 |
| 47 | 89 | Qwen2.5-7B | AQuA | 12 | 63.3202 | 62.336 | 0.9843 |
| 48 | 90 | Qwen2.5-7B | AQuA | 12 | 63.7139 | 61.6142 | 2.0997 |
| 49 | 91 | Qwen2.5-7B | AQuA | 12 | 63.5827 | 61.4173 | 2.1654 |
| 0 | 42 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 1 | 43 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 2 | 44 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 3 | 45 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 4 | 46 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 5 | 47 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 6 | 48 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 7 | 49 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 8 | 50 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 9 | 51 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 10 | 52 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 11 | 53 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 12 | 54 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 13 | 55 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 14 | 56 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 15 | 57 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 16 | 58 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 17 | 59 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 18 | 60 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 19 | 61 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 20 | 62 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 21 | 63 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 22 | 64 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 23 | 65 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 24 | 66 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 25 | 67 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 26 | 68 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 27 | 69 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 28 | 70 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 29 | 71 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 30 | 72 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 31 | 73 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 32 | 74 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 33 | 75 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 34 | 76 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 35 | 77 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 36 | 78 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 37 | 79 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 38 | 80 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 39 | 81 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 40 | 82 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 41 | 83 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 42 | 84 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 43 | 85 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 44 | 86 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 45 | 87 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 46 | 88 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 47 | 89 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 48 | 90 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 49 | 91 | Qwen2.5-7B | AQuA | 16 | 63.1398 | 62.0079 | 1.1319 |
| 0 | 42 | Qwen2.5-7B | CommonsenseQA | 4 | 81.8182 | 80.8763 | 0.9419 |
| 1 | 43 | Qwen2.5-7B | CommonsenseQA | 4 | 82.1048 | 80.6716 | 1.4333 |
| 2 | 44 | Qwen2.5-7B | CommonsenseQA | 4 | 81.7363 | 81.163 | 0.5733 |
| 3 | 45 | Qwen2.5-7B | CommonsenseQA | 4 | 81.982 | 81.4496 | 0.5324 |
| 4 | 46 | Qwen2.5-7B | CommonsenseQA | 4 | 81.2449 | 80.9992 | 0.2457 |
| 5 | 47 | Qwen2.5-7B | CommonsenseQA | 4 | 81.122 | 80.9992 | 0.1229 |
| 6 | 48 | Qwen2.5-7B | CommonsenseQA | 4 | 81.3677 | 81.0401 | 0.3276 |
| 7 | 49 | Qwen2.5-7B | CommonsenseQA | 4 | 81.2858 | 80.303 | 0.9828 |
| 8 | 50 | Qwen2.5-7B | CommonsenseQA | 4 | 81.3677 | 80.7535 | 0.6143 |
| 9 | 51 | Qwen2.5-7B | CommonsenseQA | 4 | 81.2858 | 80.4668 | 0.819 |
| 10 | 52 | Qwen2.5-7B | CommonsenseQA | 4 | 81.3268 | 80.344 | 0.9828 |
| 11 | 53 | Qwen2.5-7B | CommonsenseQA | 4 | 81.5315 | 80.6306 | 0.9009 |
| 12 | 54 | Qwen2.5-7B | CommonsenseQA | 4 | 81.4087 | 80.3849 | 1.0238 |
| 13 | 55 | Qwen2.5-7B | CommonsenseQA | 4 | 81.3268 | 81.2039 | 0.1229 |
| 14 | 56 | Qwen2.5-7B | CommonsenseQA | 4 | 81.2858 | 80.4668 | 0.819 |
| 15 | 57 | Qwen2.5-7B | CommonsenseQA | 4 | 81.8182 | 80.9992 | 0.819 |
| 16 | 58 | Qwen2.5-7B | CommonsenseQA | 4 | 80.9582 | 80.303 | 0.6552 |
| 17 | 59 | Qwen2.5-7B | CommonsenseQA | 4 | 81.3268 | 80.7944 | 0.5324 |
| 18 | 60 | Qwen2.5-7B | CommonsenseQA | 4 | 81.9001 | 79.7707 | 2.1294 |
| 19 | 61 | Qwen2.5-7B | CommonsenseQA | 4 | 80.7125 | 81.163 | -0.4505 |
| 20 | 62 | Qwen2.5-7B | CommonsenseQA | 4 | 80.8354 | 81.3677 | -0.5324 |
| 21 | 63 | Qwen2.5-7B | CommonsenseQA | 4 | 80.9582 | 81.163 | -0.2048 |
| 22 | 64 | Qwen2.5-7B | CommonsenseQA | 4 | 80.9173 | 81.163 | -0.2457 |
| 23 | 65 | Qwen2.5-7B | CommonsenseQA | 4 | 81.163 | 80.7535 | 0.4095 |
| 24 | 66 | Qwen2.5-7B | CommonsenseQA | 4 | 80.6716 | 81.2858 | -0.6143 |
| 25 | 67 | Qwen2.5-7B | CommonsenseQA | 4 | 80.9992 | 81.6134 | -0.6143 |
| 26 | 68 | Qwen2.5-7B | CommonsenseQA | 4 | 80.6306 | 80.4259 | 0.2048 |
| 27 | 69 | Qwen2.5-7B | CommonsenseQA | 4 | 81.0401 | 81.4496 | -0.4095 |
| 28 | 70 | Qwen2.5-7B | CommonsenseQA | 4 | 81.0401 | 80.6306 | 0.4095 |
| 29 | 71 | Qwen2.5-7B | CommonsenseQA | 4 | 81.4496 | 80.5897 | 0.86 |
| 30 | 72 | Qwen2.5-7B | CommonsenseQA | 4 | 81.8591 | 80.8354 | 1.0238 |
| 31 | 73 | Qwen2.5-7B | CommonsenseQA | 4 | 81.2449 | 80.1802 | 1.0647 |
| 32 | 74 | Qwen2.5-7B | CommonsenseQA | 4 | 81.0811 | 81.4906 | -0.4095 |
| 33 | 75 | Qwen2.5-7B | CommonsenseQA | 4 | 81.4087 | 80.7125 | 0.6962 |
| 34 | 76 | Qwen2.5-7B | CommonsenseQA | 4 | 80.9173 | 81.4496 | -0.5324 |
| 35 | 77 | Qwen2.5-7B | CommonsenseQA | 4 | 81.6134 | 80.7535 | 0.86 |
| 36 | 78 | Qwen2.5-7B | CommonsenseQA | 4 | 81.6953 | 80.1802 | 1.5152 |
| 37 | 79 | Qwen2.5-7B | CommonsenseQA | 4 | 81.6544 | 81.4087 | 0.2457 |
| 38 | 80 | Qwen2.5-7B | CommonsenseQA | 4 | 81.3677 | 80.9173 | 0.4505 |
| 39 | 81 | Qwen2.5-7B | CommonsenseQA | 4 | 81.6134 | 81.2449 | 0.3686 |
| 40 | 82 | Qwen2.5-7B | CommonsenseQA | 4 | 81.163 | 80.9582 | 0.2048 |
| 41 | 83 | Qwen2.5-7B | CommonsenseQA | 4 | 80.2621 | 81.3268 | -1.0647 |
| 42 | 84 | Qwen2.5-7B | CommonsenseQA | 4 | 81.7772 | 81.6134 | 0.1638 |
| 43 | 85 | Qwen2.5-7B | CommonsenseQA | 4 | 81.6544 | 80.9582 | 0.6962 |
| 44 | 86 | Qwen2.5-7B | CommonsenseQA | 4 | 81.0811 | 81.163 | -0.0819 |
| 45 | 87 | Qwen2.5-7B | CommonsenseQA | 4 | 80.9992 | 80.9992 | 0 |
| 46 | 88 | Qwen2.5-7B | CommonsenseQA | 4 | 81.8591 | 80.2621 | 1.5971 |
| 47 | 89 | Qwen2.5-7B | CommonsenseQA | 4 | 81.0811 | 80.7125 | 0.3686 |
| 48 | 90 | Qwen2.5-7B | CommonsenseQA | 4 | 80.8763 | 81.0401 | -0.1638 |
| 49 | 91 | Qwen2.5-7B | CommonsenseQA | 4 | 81.5725 | 81.4087 | 0.1638 |
| 0 | 42 | Qwen2.5-7B | CommonsenseQA | 8 | 81.7158 | 80.8968 | 0.819 |
| 1 | 43 | Qwen2.5-7B | CommonsenseQA | 8 | 81.6134 | 80.5078 | 1.1057 |
| 2 | 44 | Qwen2.5-7B | CommonsenseQA | 8 | 81.593 | 81.0401 | 0.5528 |
| 3 | 45 | Qwen2.5-7B | CommonsenseQA | 8 | 81.5111 | 81.3473 | 0.1638 |
| 4 | 46 | Qwen2.5-7B | CommonsenseQA | 8 | 81.8387 | 80.7535 | 1.0852 |
| 5 | 47 | Qwen2.5-7B | CommonsenseQA | 8 | 81.3268 | 80.6921 | 0.6347 |
| 6 | 48 | Qwen2.5-7B | CommonsenseQA | 8 | 81.7363 | 80.9582 | 0.7781 |
| 7 | 49 | Qwen2.5-7B | CommonsenseQA | 8 | 81.2244 | 80.5283 | 0.6962 |
| 8 | 50 | Qwen2.5-7B | CommonsenseQA | 8 | 81.3063 | 80.7125 | 0.5938 |
| 9 | 51 | Qwen2.5-7B | CommonsenseQA | 8 | 81.3677 | 80.4668 | 0.9009 |
| 10 | 52 | Qwen2.5-7B | CommonsenseQA | 8 | 81.3473 | 80.6306 | 0.7166 |
| 11 | 53 | Qwen2.5-7B | CommonsenseQA | 8 | 81.4292 | 80.6306 | 0.7985 |
| 12 | 54 | Qwen2.5-7B | CommonsenseQA | 8 | 81.3268 | 80.8354 | 0.4914 |
| 13 | 55 | Qwen2.5-7B | CommonsenseQA | 8 | 81.5725 | 81.2449 | 0.3276 |
| 14 | 56 | Qwen2.5-7B | CommonsenseQA | 8 | 81.3882 | 80.3235 | 1.0647 |
| 15 | 57 | Qwen2.5-7B | CommonsenseQA | 8 | 81.6339 | 81.122 | 0.5119 |
| 16 | 58 | Qwen2.5-7B | CommonsenseQA | 8 | 81.7568 | 80.9378 | 0.819 |
| 17 | 59 | Qwen2.5-7B | CommonsenseQA | 8 | 81.4087 | 80.6921 | 0.7166 |
| 18 | 60 | Qwen2.5-7B | CommonsenseQA | 8 | 81.3268 | 80.4054 | 0.9214 |
| 19 | 61 | Qwen2.5-7B | CommonsenseQA | 8 | 81.2858 | 81.3473 | -0.0614 |
| 20 | 62 | Qwen2.5-7B | CommonsenseQA | 8 | 81.2244 | 80.9787 | 0.2457 |
| 21 | 63 | Qwen2.5-7B | CommonsenseQA | 8 | 81.122 | 80.9173 | 0.2048 |
| 22 | 64 | Qwen2.5-7B | CommonsenseQA | 8 | 81.4701 | 80.4259 | 1.0442 |
| 23 | 65 | Qwen2.5-7B | CommonsenseQA | 8 | 81.2654 | 80.8149 | 0.4505 |
| 24 | 66 | Qwen2.5-7B | CommonsenseQA | 8 | 81.1835 | 81.163 | 0.0205 |
| 25 | 67 | Qwen2.5-7B | CommonsenseQA | 8 | 80.9787 | 80.9787 | 0 |
| 26 | 68 | Qwen2.5-7B | CommonsenseQA | 8 | 81.2858 | 80.5283 | 0.7576 |
| 27 | 69 | Qwen2.5-7B | CommonsenseQA | 8 | 80.9787 | 81.163 | -0.1843 |
| 28 | 70 | Qwen2.5-7B | CommonsenseQA | 8 | 80.6716 | 80.9378 | -0.2662 |
| 29 | 71 | Qwen2.5-7B | CommonsenseQA | 8 | 81.0401 | 80.5283 | 0.5119 |
| 30 | 72 | Qwen2.5-7B | CommonsenseQA | 8 | 81.5111 | 80.8968 | 0.6143 |
| 31 | 73 | Qwen2.5-7B | CommonsenseQA | 8 | 81.5315 | 80.4464 | 1.0852 |
| 32 | 74 | Qwen2.5-7B | CommonsenseQA | 8 | 81.3063 | 80.9992 | 0.3071 |
| 33 | 75 | Qwen2.5-7B | CommonsenseQA | 8 | 81.2449 | 80.9173 | 0.3276 |
| 34 | 76 | Qwen2.5-7B | CommonsenseQA | 8 | 81.7158 | 80.6921 | 1.0238 |
| 35 | 77 | Qwen2.5-7B | CommonsenseQA | 8 | 81.5111 | 80.4054 | 1.1057 |
| 36 | 78 | Qwen2.5-7B | CommonsenseQA | 8 | 81.2654 | 80.4464 | 0.819 |
| 37 | 79 | Qwen2.5-7B | CommonsenseQA | 8 | 81.7772 | 81.122 | 0.6552 |
| 38 | 80 | Qwen2.5-7B | CommonsenseQA | 8 | 81.2039 | 80.6511 | 0.5528 |
| 39 | 81 | Qwen2.5-7B | CommonsenseQA | 8 | 81.6749 | 80.774 | 0.9009 |
| 40 | 82 | Qwen2.5-7B | CommonsenseQA | 8 | 81.2449 | 80.733 | 0.5119 |
| 41 | 83 | Qwen2.5-7B | CommonsenseQA | 8 | 80.8559 | 80.9992 | -0.1433 |
| 42 | 84 | Qwen2.5-7B | CommonsenseQA | 8 | 81.8591 | 80.5692 | 1.2899 |
| 43 | 85 | Qwen2.5-7B | CommonsenseQA | 8 | 81.593 | 80.8354 | 0.7576 |
| 44 | 86 | Qwen2.5-7B | CommonsenseQA | 8 | 81.2858 | 81.2858 | 0 |
| 45 | 87 | Qwen2.5-7B | CommonsenseQA | 8 | 81.5111 | 81.0197 | 0.4914 |
| 46 | 88 | Qwen2.5-7B | CommonsenseQA | 8 | 81.7363 | 80.4464 | 1.2899 |
| 47 | 89 | Qwen2.5-7B | CommonsenseQA | 8 | 81.2654 | 80.7125 | 0.5528 |
| 48 | 90 | Qwen2.5-7B | CommonsenseQA | 8 | 81.2449 | 80.6306 | 0.6143 |
| 49 | 91 | Qwen2.5-7B | CommonsenseQA | 8 | 81.5725 | 81.1016 | 0.4709 |
| 0 | 42 | Qwen2.5-7B | CommonsenseQA | 12 | 81.6134 | 80.8354 | 0.7781 |
| 1 | 43 | Qwen2.5-7B | CommonsenseQA | 12 | 81.5042 | 80.6716 | 0.8327 |
| 2 | 44 | Qwen2.5-7B | CommonsenseQA | 12 | 81.3404 | 81.0538 | 0.2867 |
| 3 | 45 | Qwen2.5-7B | CommonsenseQA | 12 | 81.2585 | 80.9309 | 0.3276 |
| 4 | 46 | Qwen2.5-7B | CommonsenseQA | 12 | 81.5315 | 80.8354 | 0.6962 |
| 5 | 47 | Qwen2.5-7B | CommonsenseQA | 12 | 81.4906 | 80.4668 | 1.0238 |
| 6 | 48 | Qwen2.5-7B | CommonsenseQA | 12 | 81.395 | 80.7262 | 0.6689 |
| 7 | 49 | Qwen2.5-7B | CommonsenseQA | 12 | 81.1903 | 80.6716 | 0.5187 |
| 8 | 50 | Qwen2.5-7B | CommonsenseQA | 12 | 81.1766 | 80.7671 | 0.4095 |
| 9 | 51 | Qwen2.5-7B | CommonsenseQA | 12 | 81.1357 | 80.7262 | 0.4095 |
| 10 | 52 | Qwen2.5-7B | CommonsenseQA | 12 | 81.3268 | 80.89 | 0.4368 |
| 11 | 53 | Qwen2.5-7B | CommonsenseQA | 12 | 81.4496 | 80.4532 | 0.9965 |
| 12 | 54 | Qwen2.5-7B | CommonsenseQA | 12 | 81.163 | 80.7808 | 0.3822 |
| 13 | 55 | Qwen2.5-7B | CommonsenseQA | 12 | 81.4087 | 80.89 | 0.5187 |
| 14 | 56 | Qwen2.5-7B | CommonsenseQA | 12 | 81.4087 | 80.576 | 0.8327 |
| 15 | 57 | Qwen2.5-7B | CommonsenseQA | 12 | 81.5861 | 80.9036 | 0.6825 |
| 16 | 58 | Qwen2.5-7B | CommonsenseQA | 12 | 81.7636 | 80.7262 | 1.0374 |
| 17 | 59 | Qwen2.5-7B | CommonsenseQA | 12 | 81.3541 | 80.6989 | 0.6552 |
| 18 | 60 | Qwen2.5-7B | CommonsenseQA | 12 | 81.8455 | 80.5351 | 1.3104 |
| 19 | 61 | Qwen2.5-7B | CommonsenseQA | 12 | 81.2995 | 80.849 | 0.4505 |
| 20 | 62 | Qwen2.5-7B | CommonsenseQA | 12 | 81.4769 | 80.7944 | 0.6825 |
| 21 | 63 | Qwen2.5-7B | CommonsenseQA | 12 | 81.2039 | 80.9309 | 0.273 |
| 22 | 64 | Qwen2.5-7B | CommonsenseQA | 12 | 81.3268 | 80.4395 | 0.8873 |
| 23 | 65 | Qwen2.5-7B | CommonsenseQA | 12 | 81.436 | 80.6579 | 0.7781 |
| 24 | 66 | Qwen2.5-7B | CommonsenseQA | 12 | 81.2722 | 80.849 | 0.4232 |
| 25 | 67 | Qwen2.5-7B | CommonsenseQA | 12 | 81.3677 | 80.6716 | 0.6962 |
| 26 | 68 | Qwen2.5-7B | CommonsenseQA | 12 | 81.3131 | 80.6852 | 0.6279 |
| 27 | 69 | Qwen2.5-7B | CommonsenseQA | 12 | 81.1903 | 81.0674 | 0.1229 |
| 28 | 70 | Qwen2.5-7B | CommonsenseQA | 12 | 81.0947 | 80.6306 | 0.4641 |
| 29 | 71 | Qwen2.5-7B | CommonsenseQA | 12 | 81.4633 | 80.5351 | 0.9282 |
| 30 | 72 | Qwen2.5-7B | CommonsenseQA | 12 | 81.6407 | 80.9036 | 0.7371 |
| 31 | 73 | Qwen2.5-7B | CommonsenseQA | 12 | 81.395 | 80.7535 | 0.6416 |
| 32 | 74 | Qwen2.5-7B | CommonsenseQA | 12 | 81.1903 | 80.9446 | 0.2457 |
| 33 | 75 | Qwen2.5-7B | CommonsenseQA | 12 | 81.4087 | 80.6716 | 0.7371 |
| 34 | 76 | Qwen2.5-7B | CommonsenseQA | 12 | 81.6271 | 80.8081 | 0.819 |
| 35 | 77 | Qwen2.5-7B | CommonsenseQA | 12 | 81.3677 | 80.7398 | 0.6279 |
| 36 | 78 | Qwen2.5-7B | CommonsenseQA | 12 | 81.3404 | 80.5078 | 0.8327 |
| 37 | 79 | Qwen2.5-7B | CommonsenseQA | 12 | 81.6407 | 80.7944 | 0.8463 |
| 38 | 80 | Qwen2.5-7B | CommonsenseQA | 12 | 81.3677 | 80.7671 | 0.6006 |
| 39 | 81 | Qwen2.5-7B | CommonsenseQA | 12 | 81.5725 | 80.7125 | 0.86 |
| 40 | 82 | Qwen2.5-7B | CommonsenseQA | 12 | 81.1766 | 80.7262 | 0.4505 |
| 41 | 83 | Qwen2.5-7B | CommonsenseQA | 12 | 81.2722 | 80.6033 | 0.6689 |
| 42 | 84 | Qwen2.5-7B | CommonsenseQA | 12 | 81.5042 | 80.7808 | 0.7235 |
| 43 | 85 | Qwen2.5-7B | CommonsenseQA | 12 | 81.3677 | 80.6443 | 0.7235 |
| 44 | 86 | Qwen2.5-7B | CommonsenseQA | 12 | 81.3677 | 81.0811 | 0.2867 |
| 45 | 87 | Qwen2.5-7B | CommonsenseQA | 12 | 81.3404 | 80.9855 | 0.3549 |
| 46 | 88 | Qwen2.5-7B | CommonsenseQA | 12 | 81.5452 | 80.6989 | 0.8463 |
| 47 | 89 | Qwen2.5-7B | CommonsenseQA | 12 | 81.1903 | 80.849 | 0.3413 |
| 48 | 90 | Qwen2.5-7B | CommonsenseQA | 12 | 81.4633 | 80.6989 | 0.7644 |
| 49 | 91 | Qwen2.5-7B | CommonsenseQA | 12 | 81.3677 | 80.8627 | 0.5051 |
| 0 | 42 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 1 | 43 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 2 | 44 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 3 | 45 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 4 | 46 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 5 | 47 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 6 | 48 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 7 | 49 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 8 | 50 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 9 | 51 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 10 | 52 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 11 | 53 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 12 | 54 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 13 | 55 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 14 | 56 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 15 | 57 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 16 | 58 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 17 | 59 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 18 | 60 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 19 | 61 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 20 | 62 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 21 | 63 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 22 | 64 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 23 | 65 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 24 | 66 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 25 | 67 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 26 | 68 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 27 | 69 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 28 | 70 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 29 | 71 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 30 | 72 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 31 | 73 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 32 | 74 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 33 | 75 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 34 | 76 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 35 | 77 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 36 | 78 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 37 | 79 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 38 | 80 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 39 | 81 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 40 | 82 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 41 | 83 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 42 | 84 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 43 | 85 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 44 | 86 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 45 | 87 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 46 | 88 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 47 | 89 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 48 | 90 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 49 | 91 | Qwen2.5-7B | CommonsenseQA | 16 | 81.378 | 80.8354 | 0.5426 |
| 0 | 42 | Qwen2.5-7B | GPQA | 4 | 35.7143 | 32.5893 | 3.125 |
| 1 | 43 | Qwen2.5-7B | GPQA | 4 | 36.0491 | 31.4732 | 4.5759 |
| 2 | 44 | Qwen2.5-7B | GPQA | 4 | 33.3705 | 32.7009 | 0.6696 |
| 3 | 45 | Qwen2.5-7B | GPQA | 4 | 37.5 | 34.933 | 2.567 |
| 4 | 46 | Qwen2.5-7B | GPQA | 4 | 34.4866 | 33.9286 | 0.558 |
| 5 | 47 | Qwen2.5-7B | GPQA | 4 | 32.7009 | 33.7054 | -1.0045 |
| 6 | 48 | Qwen2.5-7B | GPQA | 4 | 34.7098 | 31.3616 | 3.3482 |
| 7 | 49 | Qwen2.5-7B | GPQA | 4 | 34.8214 | 32.2545 | 2.567 |
| 8 | 50 | Qwen2.5-7B | GPQA | 4 | 35.9375 | 34.2634 | 1.6741 |
| 9 | 51 | Qwen2.5-7B | GPQA | 4 | 34.8214 | 34.1518 | 0.6696 |
| 10 | 52 | Qwen2.5-7B | GPQA | 4 | 37.5 | 32.9241 | 4.5759 |
| 11 | 53 | Qwen2.5-7B | GPQA | 4 | 36.2723 | 32.1429 | 4.1295 |
| 12 | 54 | Qwen2.5-7B | GPQA | 4 | 34.2634 | 32.8125 | 1.4509 |
| 13 | 55 | Qwen2.5-7B | GPQA | 4 | 35.7143 | 31.4732 | 4.2411 |
| 14 | 56 | Qwen2.5-7B | GPQA | 4 | 34.5982 | 31.808 | 2.7902 |
| 15 | 57 | Qwen2.5-7B | GPQA | 4 | 34.933 | 36.1607 | -1.2277 |
| 16 | 58 | Qwen2.5-7B | GPQA | 4 | 34.933 | 31.1384 | 3.7946 |
| 17 | 59 | Qwen2.5-7B | GPQA | 4 | 34.2634 | 32.9241 | 1.3393 |
| 18 | 60 | Qwen2.5-7B | GPQA | 4 | 33.817 | 31.808 | 2.0089 |
| 19 | 61 | Qwen2.5-7B | GPQA | 4 | 33.5938 | 35.1562 | -1.5625 |
| 20 | 62 | Qwen2.5-7B | GPQA | 4 | 35.6027 | 34.4866 | 1.1161 |
| 21 | 63 | Qwen2.5-7B | GPQA | 4 | 34.1518 | 34.5982 | -0.4464 |
| 22 | 64 | Qwen2.5-7B | GPQA | 4 | 35.0446 | 31.4732 | 3.5714 |
| 23 | 65 | Qwen2.5-7B | GPQA | 4 | 34.5982 | 31.6964 | 2.9018 |
| 24 | 66 | Qwen2.5-7B | GPQA | 4 | 36.942 | 33.817 | 3.125 |
| 25 | 67 | Qwen2.5-7B | GPQA | 4 | 33.5938 | 32.8125 | 0.7812 |
| 26 | 68 | Qwen2.5-7B | GPQA | 4 | 33.9286 | 35.1562 | -1.2277 |
| 27 | 69 | Qwen2.5-7B | GPQA | 4 | 34.0402 | 32.8125 | 1.2277 |
| 28 | 70 | Qwen2.5-7B | GPQA | 4 | 35.2679 | 31.6964 | 3.5714 |
| 29 | 71 | Qwen2.5-7B | GPQA | 4 | 36.2723 | 32.9241 | 3.3482 |
| 30 | 72 | Qwen2.5-7B | GPQA | 4 | 34.8214 | 32.3661 | 2.4554 |
| 31 | 73 | Qwen2.5-7B | GPQA | 4 | 33.4821 | 31.5848 | 1.8973 |
| 32 | 74 | Qwen2.5-7B | GPQA | 4 | 36.2723 | 34.375 | 1.8973 |
| 33 | 75 | Qwen2.5-7B | GPQA | 4 | 34.7098 | 33.0357 | 1.6741 |
| 34 | 76 | Qwen2.5-7B | GPQA | 4 | 34.8214 | 32.2545 | 2.567 |
| 35 | 77 | Qwen2.5-7B | GPQA | 4 | 35.0446 | 33.1473 | 1.8973 |
| 36 | 78 | Qwen2.5-7B | GPQA | 4 | 33.7054 | 32.5893 | 1.1161 |
| 37 | 79 | Qwen2.5-7B | GPQA | 4 | 34.5982 | 34.4866 | 0.1116 |
| 38 | 80 | Qwen2.5-7B | GPQA | 4 | 33.5938 | 33.0357 | 0.558 |
| 39 | 81 | Qwen2.5-7B | GPQA | 4 | 33.7054 | 32.0312 | 1.6741 |
| 40 | 82 | Qwen2.5-7B | GPQA | 4 | 33.9286 | 33.3705 | 0.558 |
| 41 | 83 | Qwen2.5-7B | GPQA | 4 | 34.5982 | 32.0312 | 2.567 |
| 42 | 84 | Qwen2.5-7B | GPQA | 4 | 33.4821 | 32.3661 | 1.1161 |
| 43 | 85 | Qwen2.5-7B | GPQA | 4 | 36.1607 | 33.3705 | 2.7902 |
| 44 | 86 | Qwen2.5-7B | GPQA | 4 | 33.3705 | 35.3795 | -2.0089 |
| 45 | 87 | Qwen2.5-7B | GPQA | 4 | 35.4911 | 31.9196 | 3.5714 |
| 46 | 88 | Qwen2.5-7B | GPQA | 4 | 35.0446 | 32.3661 | 2.6786 |
| 47 | 89 | Qwen2.5-7B | GPQA | 4 | 35.9375 | 34.0402 | 1.8973 |
| 48 | 90 | Qwen2.5-7B | GPQA | 4 | 33.7054 | 33.4821 | 0.2232 |
| 49 | 91 | Qwen2.5-7B | GPQA | 4 | 35.0446 | 32.4777 | 2.567 |
| 0 | 42 | Qwen2.5-7B | GPQA | 8 | 34.2634 | 32.8125 | 1.4509 |
| 1 | 43 | Qwen2.5-7B | GPQA | 8 | 35.3237 | 31.9754 | 3.3482 |
| 2 | 44 | Qwen2.5-7B | GPQA | 8 | 34.4866 | 32.5893 | 1.8973 |
| 3 | 45 | Qwen2.5-7B | GPQA | 8 | 36.1607 | 33.6496 | 2.5112 |
| 4 | 46 | Qwen2.5-7B | GPQA | 8 | 35.3237 | 32.5335 | 2.7902 |
| 5 | 47 | Qwen2.5-7B | GPQA | 8 | 33.4821 | 33.3147 | 0.1674 |
| 6 | 48 | Qwen2.5-7B | GPQA | 8 | 33.3705 | 32.5893 | 0.7812 |
| 7 | 49 | Qwen2.5-7B | GPQA | 8 | 34.5424 | 32.4219 | 2.1205 |
| 8 | 50 | Qwen2.5-7B | GPQA | 8 | 33.8728 | 33.3147 | 0.558 |
| 9 | 51 | Qwen2.5-7B | GPQA | 8 | 34.9888 | 32.8683 | 2.1205 |
| 10 | 52 | Qwen2.5-7B | GPQA | 8 | 35.8259 | 33.2031 | 2.6228 |
| 11 | 53 | Qwen2.5-7B | GPQA | 8 | 35.7143 | 31.3616 | 4.3527 |
| 12 | 54 | Qwen2.5-7B | GPQA | 8 | 34.7656 | 32.8125 | 1.9531 |
| 13 | 55 | Qwen2.5-7B | GPQA | 8 | 35.1004 | 31.1942 | 3.9062 |
| 14 | 56 | Qwen2.5-7B | GPQA | 8 | 34.5982 | 32.9799 | 1.6183 |
| 15 | 57 | Qwen2.5-7B | GPQA | 8 | 34.654 | 33.7612 | 0.8929 |
| 16 | 58 | Qwen2.5-7B | GPQA | 8 | 34.8214 | 32.1987 | 2.6228 |
| 17 | 59 | Qwen2.5-7B | GPQA | 8 | 34.5424 | 33.0357 | 1.5067 |
| 18 | 60 | Qwen2.5-7B | GPQA | 8 | 34.096 | 32.3103 | 1.7857 |
| 19 | 61 | Qwen2.5-7B | GPQA | 8 | 34.7098 | 33.8728 | 0.8371 |
| 20 | 62 | Qwen2.5-7B | GPQA | 8 | 34.933 | 33.7054 | 1.2277 |
| 21 | 63 | Qwen2.5-7B | GPQA | 8 | 33.8728 | 33.5379 | 0.3348 |
| 22 | 64 | Qwen2.5-7B | GPQA | 8 | 34.7098 | 32.3103 | 2.3996 |
| 23 | 65 | Qwen2.5-7B | GPQA | 8 | 34.654 | 31.3058 | 3.3482 |
| 24 | 66 | Qwen2.5-7B | GPQA | 8 | 34.7656 | 33.2589 | 1.5067 |
| 25 | 67 | Qwen2.5-7B | GPQA | 8 | 34.375 | 33.3147 | 1.0603 |
| 26 | 68 | Qwen2.5-7B | GPQA | 8 | 34.5982 | 33.1473 | 1.4509 |
| 27 | 69 | Qwen2.5-7B | GPQA | 8 | 34.3192 | 33.2589 | 1.0603 |
| 28 | 70 | Qwen2.5-7B | GPQA | 8 | 34.9888 | 32.1429 | 2.846 |
| 29 | 71 | Qwen2.5-7B | GPQA | 8 | 34.2076 | 33.5938 | 0.6138 |
| 30 | 72 | Qwen2.5-7B | GPQA | 8 | 34.4308 | 32.7009 | 1.7299 |
| 31 | 73 | Qwen2.5-7B | GPQA | 8 | 33.5938 | 32.3661 | 1.2277 |
| 32 | 74 | Qwen2.5-7B | GPQA | 8 | 34.933 | 32.2545 | 2.6786 |
| 33 | 75 | Qwen2.5-7B | GPQA | 8 | 34.8214 | 33.0915 | 1.7299 |
| 34 | 76 | Qwen2.5-7B | GPQA | 8 | 33.8728 | 33.4821 | 0.3906 |
| 35 | 77 | Qwen2.5-7B | GPQA | 8 | 36.2165 | 32.9799 | 3.2366 |
| 36 | 78 | Qwen2.5-7B | GPQA | 8 | 35.2121 | 32.5335 | 2.6786 |
| 37 | 79 | Qwen2.5-7B | GPQA | 8 | 35.9375 | 32.9799 | 2.9576 |
| 38 | 80 | Qwen2.5-7B | GPQA | 8 | 35.1004 | 32.2545 | 2.846 |
| 39 | 81 | Qwen2.5-7B | GPQA | 8 | 34.2634 | 33.4821 | 0.7812 |
| 40 | 82 | Qwen2.5-7B | GPQA | 8 | 33.3147 | 33.5938 | -0.279 |
| 41 | 83 | Qwen2.5-7B | GPQA | 8 | 34.654 | 32.4777 | 2.1763 |
| 42 | 84 | Qwen2.5-7B | GPQA | 8 | 33.8728 | 33.0915 | 0.7812 |
| 43 | 85 | Qwen2.5-7B | GPQA | 8 | 35.2121 | 32.0312 | 3.1808 |
| 44 | 86 | Qwen2.5-7B | GPQA | 8 | 34.7656 | 34.3192 | 0.4464 |
| 45 | 87 | Qwen2.5-7B | GPQA | 8 | 35.3237 | 32.9241 | 2.3996 |
| 46 | 88 | Qwen2.5-7B | GPQA | 8 | 34.5982 | 32.9241 | 1.6741 |
| 47 | 89 | Qwen2.5-7B | GPQA | 8 | 34.933 | 32.5335 | 2.3996 |
| 48 | 90 | Qwen2.5-7B | GPQA | 8 | 34.7098 | 31.4174 | 3.2924 |
| 49 | 91 | Qwen2.5-7B | GPQA | 8 | 34.5424 | 32.2545 | 2.2879 |
| 0 | 42 | Qwen2.5-7B | GPQA | 12 | 34.747 | 32.8497 | 1.8973 |
| 1 | 43 | Qwen2.5-7B | GPQA | 12 | 34.8214 | 32.5149 | 2.3065 |
| 2 | 44 | Qwen2.5-7B | GPQA | 12 | 34.2634 | 32.5893 | 1.6741 |
| 3 | 45 | Qwen2.5-7B | GPQA | 12 | 35.3051 | 33.3705 | 1.9345 |
| 4 | 46 | Qwen2.5-7B | GPQA | 12 | 34.189 | 32.4777 | 1.7113 |
| 5 | 47 | Qwen2.5-7B | GPQA | 12 | 34.3006 | 33.2589 | 1.0417 |
| 6 | 48 | Qwen2.5-7B | GPQA | 12 | 34.561 | 32.5893 | 1.9717 |
| 7 | 49 | Qwen2.5-7B | GPQA | 12 | 35.0074 | 31.6964 | 3.311 |
| 8 | 50 | Qwen2.5-7B | GPQA | 12 | 33.9286 | 32.7381 | 1.1905 |
| 9 | 51 | Qwen2.5-7B | GPQA | 12 | 34.375 | 32.7381 | 1.6369 |
| 10 | 52 | Qwen2.5-7B | GPQA | 12 | 35.7515 | 33.4449 | 2.3065 |
| 11 | 53 | Qwen2.5-7B | GPQA | 12 | 35.0818 | 32.5893 | 2.4926 |
| 12 | 54 | Qwen2.5-7B | GPQA | 12 | 34.2634 | 33.4449 | 0.8185 |
| 13 | 55 | Qwen2.5-7B | GPQA | 12 | 35.3423 | 32.1801 | 3.1622 |
| 14 | 56 | Qwen2.5-7B | GPQA | 12 | 35.1935 | 32.8125 | 2.381 |
| 15 | 57 | Qwen2.5-7B | GPQA | 12 | 34.4494 | 33.0357 | 1.4137 |
| 16 | 58 | Qwen2.5-7B | GPQA | 12 | 34.375 | 32.5893 | 1.7857 |
| 17 | 59 | Qwen2.5-7B | GPQA | 12 | 34.6354 | 32.8497 | 1.7857 |
| 18 | 60 | Qwen2.5-7B | GPQA | 12 | 34.933 | 32.3289 | 2.6042 |
| 19 | 61 | Qwen2.5-7B | GPQA | 12 | 35.0074 | 33.7426 | 1.2649 |
| 20 | 62 | Qwen2.5-7B | GPQA | 12 | 34.003 | 33.5938 | 0.4092 |
| 21 | 63 | Qwen2.5-7B | GPQA | 12 | 34.5982 | 33.7054 | 0.8929 |
| 22 | 64 | Qwen2.5-7B | GPQA | 12 | 35.3795 | 32.1801 | 3.1994 |
| 23 | 65 | Qwen2.5-7B | GPQA | 12 | 34.375 | 32.8497 | 1.5253 |
| 24 | 66 | Qwen2.5-7B | GPQA | 12 | 34.8586 | 32.8497 | 2.0089 |
| 25 | 67 | Qwen2.5-7B | GPQA | 12 | 34.7842 | 32.6265 | 2.1577 |
| 26 | 68 | Qwen2.5-7B | GPQA | 12 | 35.0074 | 32.5893 | 2.4182 |
| 27 | 69 | Qwen2.5-7B | GPQA | 12 | 34.5982 | 32.8869 | 1.7113 |
| 28 | 70 | Qwen2.5-7B | GPQA | 12 | 35.4167 | 32.4777 | 2.939 |
| 29 | 71 | Qwen2.5-7B | GPQA | 12 | 34.7842 | 33.2589 | 1.5253 |
| 30 | 72 | Qwen2.5-7B | GPQA | 12 | 34.3378 | 32.4405 | 1.8973 |
| 31 | 73 | Qwen2.5-7B | GPQA | 12 | 33.9658 | 32.8125 | 1.1533 |
| 32 | 74 | Qwen2.5-7B | GPQA | 12 | 34.9702 | 31.9196 | 3.0506 |
| 33 | 75 | Qwen2.5-7B | GPQA | 12 | 34.8214 | 32.8869 | 1.9345 |
| 34 | 76 | Qwen2.5-7B | GPQA | 12 | 34.8586 | 33.1473 | 1.7113 |
| 35 | 77 | Qwen2.5-7B | GPQA | 12 | 35.0074 | 32.7009 | 2.3065 |
| 36 | 78 | Qwen2.5-7B | GPQA | 12 | 34.4866 | 32.5149 | 1.9717 |
| 37 | 79 | Qwen2.5-7B | GPQA | 12 | 34.561 | 33.1101 | 1.4509 |
| 38 | 80 | Qwen2.5-7B | GPQA | 12 | 34.7098 | 32.8125 | 1.8973 |
| 39 | 81 | Qwen2.5-7B | GPQA | 12 | 34.3006 | 33.0729 | 1.2277 |
| 40 | 82 | Qwen2.5-7B | GPQA | 12 | 33.7426 | 32.8125 | 0.9301 |
| 41 | 83 | Qwen2.5-7B | GPQA | 12 | 34.3378 | 33.0357 | 1.3021 |
| 42 | 84 | Qwen2.5-7B | GPQA | 12 | 34.8214 | 32.7753 | 2.0461 |
| 43 | 85 | Qwen2.5-7B | GPQA | 12 | 35.8259 | 32.0312 | 3.7946 |
| 44 | 86 | Qwen2.5-7B | GPQA | 12 | 35.0818 | 33.2961 | 1.7857 |
| 45 | 87 | Qwen2.5-7B | GPQA | 12 | 35.119 | 33.1473 | 1.9717 |
| 46 | 88 | Qwen2.5-7B | GPQA | 12 | 34.8586 | 33.1845 | 1.6741 |
| 47 | 89 | Qwen2.5-7B | GPQA | 12 | 34.5982 | 33.2589 | 1.3393 |
| 48 | 90 | Qwen2.5-7B | GPQA | 12 | 33.9658 | 32.9241 | 1.0417 |
| 49 | 91 | Qwen2.5-7B | GPQA | 12 | 34.561 | 32.9613 | 1.5997 |
| 0 | 42 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 1 | 43 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 2 | 44 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 3 | 45 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 4 | 46 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 5 | 47 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 6 | 48 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 7 | 49 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 8 | 50 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 9 | 51 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 10 | 52 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 11 | 53 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 12 | 54 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 13 | 55 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 14 | 56 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 15 | 57 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 16 | 58 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 17 | 59 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 18 | 60 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 19 | 61 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 20 | 62 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 21 | 63 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 22 | 64 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 23 | 65 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 24 | 66 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 25 | 67 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 26 | 68 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 27 | 69 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 28 | 70 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 29 | 71 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 30 | 72 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 31 | 73 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 32 | 74 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 33 | 75 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 34 | 76 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 35 | 77 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 36 | 78 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 37 | 79 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 38 | 80 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 39 | 81 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 40 | 82 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 41 | 83 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 42 | 84 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 43 | 85 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 44 | 86 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 45 | 87 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 46 | 88 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 47 | 89 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 48 | 90 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 49 | 91 | Qwen2.5-7B | GPQA | 16 | 34.8772 | 32.7009 | 2.1763 |
| 0 | 42 | Qwen2.5-7B | GSM8K | 4 | 68.9538 | 87.6801 | -18.7263 |
| 1 | 43 | Qwen2.5-7B | GSM8K | 4 | 68.9538 | 86.467 | -17.5133 |
| 2 | 44 | Qwen2.5-7B | GSM8K | 4 | 68.423 | 85.9363 | -17.5133 |
| 3 | 45 | Qwen2.5-7B | GSM8K | 4 | 67.9303 | 85.9742 | -18.044 |
| 4 | 46 | Qwen2.5-7B | GSM8K | 4 | 69.3328 | 86.5428 | -17.21 |
| 5 | 47 | Qwen2.5-7B | GSM8K | 4 | 69.7877 | 87.4526 | -17.6649 |
| 6 | 48 | Qwen2.5-7B | GSM8K | 4 | 69.5982 | 86.4291 | -16.8309 |
| 7 | 49 | Qwen2.5-7B | GSM8K | 4 | 70.2047 | 86.2396 | -16.0349 |
| 8 | 50 | Qwen2.5-7B | GSM8K | 4 | 68.0061 | 86.3533 | -18.3472 |
| 9 | 51 | Qwen2.5-7B | GSM8K | 4 | 67.5512 | 86.5049 | -18.9538 |
| 10 | 52 | Qwen2.5-7B | GSM8K | 4 | 68.1577 | 86.3533 | -18.1956 |
| 11 | 53 | Qwen2.5-7B | GSM8K | 4 | 68.4989 | 86.3154 | -17.8165 |
| 12 | 54 | Qwen2.5-7B | GSM8K | 4 | 67.3995 | 86.467 | -19.0675 |
| 13 | 55 | Qwen2.5-7B | GSM8K | 4 | 68.6505 | 85.5572 | -16.9067 |
| 14 | 56 | Qwen2.5-7B | GSM8K | 4 | 68.9917 | 86.6945 | -17.7028 |
| 15 | 57 | Qwen2.5-7B | GSM8K | 4 | 69.257 | 86.8082 | -17.5512 |
| 16 | 58 | Qwen2.5-7B | GSM8K | 4 | 69.0296 | 85.5193 | -16.4898 |
| 17 | 59 | Qwen2.5-7B | GSM8K | 4 | 68.0061 | 85.7468 | -17.7407 |
| 18 | 60 | Qwen2.5-7B | GSM8K | 4 | 69.674 | 86.7703 | -17.0963 |
| 19 | 61 | Qwen2.5-7B | GSM8K | 4 | 69.0296 | 85.6331 | -16.6035 |
| 20 | 62 | Qwen2.5-7B | GSM8K | 4 | 68.3472 | 86.2775 | -17.9303 |
| 21 | 63 | Qwen2.5-7B | GSM8K | 4 | 69.9773 | 86.2775 | -16.3002 |
| 22 | 64 | Qwen2.5-7B | GSM8K | 4 | 68.9917 | 85.5572 | -16.5656 |
| 23 | 65 | Qwen2.5-7B | GSM8K | 4 | 68.7642 | 87.0735 | -18.3093 |
| 24 | 66 | Qwen2.5-7B | GSM8K | 4 | 68.1198 | 86.5807 | -18.461 |
| 25 | 67 | Qwen2.5-7B | GSM8K | 4 | 69.7877 | 87.0356 | -17.2479 |
| 26 | 68 | Qwen2.5-7B | GSM8K | 4 | 67.7028 | 86.3912 | -18.6884 |
| 27 | 69 | Qwen2.5-7B | GSM8K | 4 | 68.1577 | 85.7468 | -17.5891 |
| 28 | 70 | Qwen2.5-7B | GSM8K | 4 | 68.3472 | 85.7089 | -17.3616 |
| 29 | 71 | Qwen2.5-7B | GSM8K | 4 | 68.4989 | 85.254 | -16.7551 |
| 30 | 72 | Qwen2.5-7B | GSM8K | 4 | 69.1054 | 85.7847 | -16.6793 |
| 31 | 73 | Qwen2.5-7B | GSM8K | 4 | 69.7119 | 86.3912 | -16.6793 |
| 32 | 74 | Qwen2.5-7B | GSM8K | 4 | 68.2714 | 86.9219 | -18.6505 |
| 33 | 75 | Qwen2.5-7B | GSM8K | 4 | 69.4086 | 86.0879 | -16.6793 |
| 34 | 76 | Qwen2.5-7B | GSM8K | 4 | 70.1289 | 85.2919 | -15.163 |
| 35 | 77 | Qwen2.5-7B | GSM8K | 4 | 67.8923 | 85.9742 | -18.0819 |
| 36 | 78 | Qwen2.5-7B | GSM8K | 4 | 69.674 | 85.8605 | -16.1865 |
| 37 | 79 | Qwen2.5-7B | GSM8K | 4 | 69.3328 | 86.8082 | -17.4754 |
| 38 | 80 | Qwen2.5-7B | GSM8K | 4 | 68.6505 | 85.7847 | -17.1342 |
| 39 | 81 | Qwen2.5-7B | GSM8K | 4 | 67.8165 | 87.2631 | -19.4466 |
| 40 | 82 | Qwen2.5-7B | GSM8K | 4 | 67.9682 | 85.7847 | -17.8165 |
| 41 | 83 | Qwen2.5-7B | GSM8K | 4 | 68.7263 | 87.0356 | -18.3093 |
| 42 | 84 | Qwen2.5-7B | GSM8K | 4 | 68.3093 | 86.2775 | -17.9682 |
| 43 | 85 | Qwen2.5-7B | GSM8K | 4 | 68.461 | 85.8984 | -17.4375 |
| 44 | 86 | Qwen2.5-7B | GSM8K | 4 | 68.84 | 87.0356 | -18.1956 |
| 45 | 87 | Qwen2.5-7B | GSM8K | 4 | 69.1054 | 86.1638 | -17.0584 |
| 46 | 88 | Qwen2.5-7B | GSM8K | 4 | 68.423 | 85.4814 | -17.0584 |
| 47 | 89 | Qwen2.5-7B | GSM8K | 4 | 68.8021 | 86.2396 | -17.4375 |
| 48 | 90 | Qwen2.5-7B | GSM8K | 4 | 69.2191 | 86.5807 | -17.3616 |
| 49 | 91 | Qwen2.5-7B | GSM8K | 4 | 70.9249 | 86.7703 | -15.8453 |
| 0 | 42 | Qwen2.5-7B | GSM8K | 8 | 68.1198 | 88.0212 | -19.9014 |
| 1 | 43 | Qwen2.5-7B | GSM8K | 8 | 68.2714 | 87.3958 | -19.1243 |
| 2 | 44 | Qwen2.5-7B | GSM8K | 8 | 67.1911 | 87.1304 | -19.9393 |
| 3 | 45 | Qwen2.5-7B | GSM8K | 8 | 67.3616 | 87.4337 | -20.072 |
| 4 | 46 | Qwen2.5-7B | GSM8K | 8 | 67.7786 | 87.5853 | -19.8067 |
| 5 | 47 | Qwen2.5-7B | GSM8K | 8 | 68.044 | 87.5474 | -19.5034 |
| 6 | 48 | Qwen2.5-7B | GSM8K | 8 | 67.7028 | 87.5284 | -19.8256 |
| 7 | 49 | Qwen2.5-7B | GSM8K | 8 | 67.9113 | 87.6042 | -19.6929 |
| 8 | 50 | Qwen2.5-7B | GSM8K | 8 | 67.4185 | 87.8696 | -20.4511 |
| 9 | 51 | Qwen2.5-7B | GSM8K | 8 | 67.1911 | 87.4526 | -20.2616 |
| 10 | 52 | Qwen2.5-7B | GSM8K | 8 | 67.5322 | 87.1683 | -19.6361 |
| 11 | 53 | Qwen2.5-7B | GSM8K | 8 | 66.8309 | 87.2062 | -20.3753 |
| 12 | 54 | Qwen2.5-7B | GSM8K | 8 | 67.5701 | 87.2252 | -19.655 |
| 13 | 55 | Qwen2.5-7B | GSM8K | 8 | 67.8355 | 87.1683 | -19.3328 |
| 14 | 56 | Qwen2.5-7B | GSM8K | 8 | 67.0584 | 87.6422 | -20.5838 |
| 15 | 57 | Qwen2.5-7B | GSM8K | 8 | 67.5891 | 87.301 | -19.7119 |
| 16 | 58 | Qwen2.5-7B | GSM8K | 8 | 67.627 | 87.1494 | -19.5224 |
| 17 | 59 | Qwen2.5-7B | GSM8K | 8 | 66.9067 | 87.5095 | -20.6027 |
| 18 | 60 | Qwen2.5-7B | GSM8K | 8 | 68.044 | 87.7938 | -19.7498 |
| 19 | 61 | Qwen2.5-7B | GSM8K | 8 | 68.1577 | 87.4905 | -19.3328 |
| 20 | 62 | Qwen2.5-7B | GSM8K | 8 | 67.4564 | 87.2631 | -19.8067 |
| 21 | 63 | Qwen2.5-7B | GSM8K | 8 | 68.0629 | 87.3768 | -19.3139 |
| 22 | 64 | Qwen2.5-7B | GSM8K | 8 | 67.9303 | 87.4716 | -19.5413 |
| 23 | 65 | Qwen2.5-7B | GSM8K | 8 | 67.9682 | 87.6422 | -19.674 |
| 24 | 66 | Qwen2.5-7B | GSM8K | 8 | 67.3616 | 87.282 | -19.9204 |
| 25 | 67 | Qwen2.5-7B | GSM8K | 8 | 68.6126 | 87.718 | -19.1054 |
| 26 | 68 | Qwen2.5-7B | GSM8K | 8 | 66.8878 | 87.4716 | -20.5838 |
| 27 | 69 | Qwen2.5-7B | GSM8K | 8 | 67.3237 | 87.301 | -19.9773 |
| 28 | 70 | Qwen2.5-7B | GSM8K | 8 | 67.5133 | 87.4337 | -19.9204 |
| 29 | 71 | Qwen2.5-7B | GSM8K | 8 | 67.6459 | 87.4716 | -19.8256 |
| 30 | 72 | Qwen2.5-7B | GSM8K | 8 | 67.7407 | 87.4526 | -19.7119 |
| 31 | 73 | Qwen2.5-7B | GSM8K | 8 | 68.4041 | 87.301 | -18.8969 |
| 32 | 74 | Qwen2.5-7B | GSM8K | 8 | 67.5322 | 87.4905 | -19.9583 |
| 33 | 75 | Qwen2.5-7B | GSM8K | 8 | 67.5322 | 87.3578 | -19.8256 |
| 34 | 76 | Qwen2.5-7B | GSM8K | 8 | 67.4943 | 87.5095 | -20.0152 |
| 35 | 77 | Qwen2.5-7B | GSM8K | 8 | 67.1721 | 87.3958 | -20.2237 |
| 36 | 78 | Qwen2.5-7B | GSM8K | 8 | 68.0061 | 87.0167 | -19.0106 |
| 37 | 79 | Qwen2.5-7B | GSM8K | 8 | 67.9492 | 87.8127 | -19.8635 |
| 38 | 80 | Qwen2.5-7B | GSM8K | 8 | 67.2479 | 86.7134 | -19.4655 |
| 39 | 81 | Qwen2.5-7B | GSM8K | 8 | 67.5512 | 87.5095 | -19.9583 |
| 40 | 82 | Qwen2.5-7B | GSM8K | 8 | 68.1766 | 87.2631 | -19.0864 |
| 41 | 83 | Qwen2.5-7B | GSM8K | 8 | 67.1342 | 87.4905 | -20.3563 |
| 42 | 84 | Qwen2.5-7B | GSM8K | 8 | 66.8878 | 87.699 | -20.8112 |
| 43 | 85 | Qwen2.5-7B | GSM8K | 8 | 67.7976 | 87.4147 | -19.6171 |
| 44 | 86 | Qwen2.5-7B | GSM8K | 8 | 67.8544 | 87.4337 | -19.5792 |
| 45 | 87 | Qwen2.5-7B | GSM8K | 8 | 67.8734 | 87.5095 | -19.6361 |
| 46 | 88 | Qwen2.5-7B | GSM8K | 8 | 67.1911 | 87.6611 | -20.4701 |
| 47 | 89 | Qwen2.5-7B | GSM8K | 8 | 67.627 | 87.301 | -19.674 |
| 48 | 90 | Qwen2.5-7B | GSM8K | 8 | 68.442 | 87.6042 | -19.1622 |
| 49 | 91 | Qwen2.5-7B | GSM8K | 8 | 69.2191 | 86.9977 | -17.7786 |
| 0 | 42 | Qwen2.5-7B | GSM8K | 12 | 67.1089 | 88.1097 | -21.0008 |
| 1 | 43 | Qwen2.5-7B | GSM8K | 12 | 67.5006 | 87.9328 | -20.4321 |
| 2 | 44 | Qwen2.5-7B | GSM8K | 12 | 66.9699 | 87.7685 | -20.7986 |
| 3 | 45 | Qwen2.5-7B | GSM8K | 12 | 66.9573 | 87.857 | -20.8997 |
| 4 | 46 | Qwen2.5-7B | GSM8K | 12 | 67.0457 | 88.0465 | -21.0008 |
| 5 | 47 | Qwen2.5-7B | GSM8K | 12 | 67.6396 | 88.1476 | -20.508 |
| 6 | 48 | Qwen2.5-7B | GSM8K | 12 | 67.0078 | 87.5916 | -20.5838 |
| 7 | 49 | Qwen2.5-7B | GSM8K | 12 | 67.5385 | 88.1476 | -20.609 |
| 8 | 50 | Qwen2.5-7B | GSM8K | 12 | 67.1216 | 88.097 | -20.9755 |
| 9 | 51 | Qwen2.5-7B | GSM8K | 12 | 67.1595 | 87.9075 | -20.748 |
| 10 | 52 | Qwen2.5-7B | GSM8K | 12 | 67.3995 | 87.8822 | -20.4827 |
| 11 | 53 | Qwen2.5-7B | GSM8K | 12 | 66.5403 | 88.0086 | -21.4683 |
| 12 | 54 | Qwen2.5-7B | GSM8K | 12 | 67.1595 | 87.7559 | -20.5964 |
| 13 | 55 | Qwen2.5-7B | GSM8K | 12 | 67.349 | 87.9454 | -20.5964 |
| 14 | 56 | Qwen2.5-7B | GSM8K | 12 | 67.1089 | 88.097 | -20.9881 |
| 15 | 57 | Qwen2.5-7B | GSM8K | 12 | 67.0584 | 87.996 | -20.9376 |
| 16 | 58 | Qwen2.5-7B | GSM8K | 12 | 66.9194 | 87.8949 | -20.9755 |
| 17 | 59 | Qwen2.5-7B | GSM8K | 12 | 67.0457 | 87.9075 | -20.8618 |
| 18 | 60 | Qwen2.5-7B | GSM8K | 12 | 67.2985 | 87.8822 | -20.5838 |
| 19 | 61 | Qwen2.5-7B | GSM8K | 12 | 67.7913 | 87.8317 | -20.0404 |
| 20 | 62 | Qwen2.5-7B | GSM8K | 12 | 67.2226 | 87.8696 | -20.647 |
| 21 | 63 | Qwen2.5-7B | GSM8K | 12 | 66.8309 | 87.7811 | -20.9502 |
| 22 | 64 | Qwen2.5-7B | GSM8K | 12 | 67.4501 | 87.6927 | -20.2426 |
| 23 | 65 | Qwen2.5-7B | GSM8K | 12 | 67.1721 | 88.0591 | -20.887 |
| 24 | 66 | Qwen2.5-7B | GSM8K | 12 | 67.4248 | 87.8949 | -20.4701 |
| 25 | 67 | Qwen2.5-7B | GSM8K | 12 | 67.2732 | 88.097 | -20.8239 |
| 26 | 68 | Qwen2.5-7B | GSM8K | 12 | 66.8941 | 87.958 | -21.0639 |
| 27 | 69 | Qwen2.5-7B | GSM8K | 12 | 67.3111 | 87.7559 | -20.4448 |
| 28 | 70 | Qwen2.5-7B | GSM8K | 12 | 67.1847 | 87.8822 | -20.6975 |
| 29 | 71 | Qwen2.5-7B | GSM8K | 12 | 67.1089 | 88.097 | -20.9881 |
| 30 | 72 | Qwen2.5-7B | GSM8K | 12 | 67.0963 | 87.9454 | -20.8491 |
| 31 | 73 | Qwen2.5-7B | GSM8K | 12 | 67.1721 | 87.9075 | -20.7354 |
| 32 | 74 | Qwen2.5-7B | GSM8K | 12 | 66.9699 | 87.9707 | -21.0008 |
| 33 | 75 | Qwen2.5-7B | GSM8K | 12 | 66.9447 | 88.0465 | -21.1018 |
| 34 | 76 | Qwen2.5-7B | GSM8K | 12 | 67.1216 | 88.0591 | -20.9376 |
| 35 | 77 | Qwen2.5-7B | GSM8K | 12 | 67.0836 | 88.0591 | -20.9755 |
| 36 | 78 | Qwen2.5-7B | GSM8K | 12 | 67.3111 | 88.0718 | -20.7607 |
| 37 | 79 | Qwen2.5-7B | GSM8K | 12 | 67.5259 | 88.135 | -20.609 |
| 38 | 80 | Qwen2.5-7B | GSM8K | 12 | 66.7678 | 87.6674 | -20.8997 |
| 39 | 81 | Qwen2.5-7B | GSM8K | 12 | 66.8309 | 88.1223 | -21.2914 |
| 40 | 82 | Qwen2.5-7B | GSM8K | 12 | 67.2985 | 87.8317 | -20.5332 |
| 41 | 83 | Qwen2.5-7B | GSM8K | 12 | 67.1595 | 87.9707 | -20.8112 |
| 42 | 84 | Qwen2.5-7B | GSM8K | 12 | 66.8436 | 88.2739 | -21.4304 |
| 43 | 85 | Qwen2.5-7B | GSM8K | 12 | 67.2606 | 87.8443 | -20.5838 |
| 44 | 86 | Qwen2.5-7B | GSM8K | 12 | 66.9194 | 87.8696 | -20.9502 |
| 45 | 87 | Qwen2.5-7B | GSM8K | 12 | 66.9826 | 88.0086 | -21.026 |
| 46 | 88 | Qwen2.5-7B | GSM8K | 12 | 66.5656 | 88.1602 | -21.5946 |
| 47 | 89 | Qwen2.5-7B | GSM8K | 12 | 66.932 | 88.2866 | -21.3546 |
| 48 | 90 | Qwen2.5-7B | GSM8K | 12 | 67.4375 | 87.7306 | -20.2932 |
| 49 | 91 | Qwen2.5-7B | GSM8K | 12 | 67.5259 | 87.996 | -20.4701 |
| 0 | 42 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 1 | 43 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 2 | 44 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 3 | 45 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 4 | 46 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 5 | 47 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 6 | 48 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 7 | 49 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 8 | 50 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 9 | 51 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 10 | 52 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 11 | 53 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 12 | 54 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 13 | 55 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 14 | 56 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 15 | 57 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 16 | 58 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 17 | 59 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 18 | 60 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 19 | 61 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 20 | 62 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 21 | 63 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 22 | 64 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 23 | 65 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 24 | 66 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 25 | 67 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 26 | 68 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 27 | 69 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 28 | 70 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 29 | 71 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 30 | 72 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 31 | 73 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 32 | 74 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 33 | 75 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 34 | 76 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 35 | 77 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 36 | 78 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 37 | 79 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 38 | 80 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 39 | 81 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 40 | 82 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 41 | 83 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 42 | 84 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 43 | 85 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 44 | 86 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 45 | 87 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 46 | 88 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 47 | 89 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 48 | 90 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 49 | 91 | Qwen2.5-7B | GSM8K | 16 | 66.8309 | 88.2676 | -21.4367 |
| 0 | 42 | Qwen2.5-7B | MATH500 | 4 | 43.6 | 59 | -15.4 |
| 1 | 43 | Qwen2.5-7B | MATH500 | 4 | 43.7 | 58.3 | -14.6 |
| 2 | 44 | Qwen2.5-7B | MATH500 | 4 | 44.7 | 57.3 | -12.6 |
| 3 | 45 | Qwen2.5-7B | MATH500 | 4 | 45.7 | 56.9 | -11.2 |
| 4 | 46 | Qwen2.5-7B | MATH500 | 4 | 43.5 | 58.8 | -15.3 |
| 5 | 47 | Qwen2.5-7B | MATH500 | 4 | 43.8 | 60 | -16.2 |
| 6 | 48 | Qwen2.5-7B | MATH500 | 4 | 45.9 | 57.3 | -11.4 |
| 7 | 49 | Qwen2.5-7B | MATH500 | 4 | 44.5 | 59.1 | -14.6 |
| 8 | 50 | Qwen2.5-7B | MATH500 | 4 | 41.8 | 58.5 | -16.7 |
| 9 | 51 | Qwen2.5-7B | MATH500 | 4 | 45 | 57.3 | -12.3 |
| 10 | 52 | Qwen2.5-7B | MATH500 | 4 | 42.1 | 57.3 | -15.2 |
| 11 | 53 | Qwen2.5-7B | MATH500 | 4 | 41.8 | 61.1 | -19.3 |
| 12 | 54 | Qwen2.5-7B | MATH500 | 4 | 44.4 | 58.5 | -14.1 |
| 13 | 55 | Qwen2.5-7B | MATH500 | 4 | 43.4 | 60.5 | -17.1 |
| 14 | 56 | Qwen2.5-7B | MATH500 | 4 | 44.3 | 57 | -12.7 |
| 15 | 57 | Qwen2.5-7B | MATH500 | 4 | 42.8 | 59.8 | -17 |
| 16 | 58 | Qwen2.5-7B | MATH500 | 4 | 42.4 | 58.7 | -16.3 |
| 17 | 59 | Qwen2.5-7B | MATH500 | 4 | 45.3 | 59.7 | -14.4 |
| 18 | 60 | Qwen2.5-7B | MATH500 | 4 | 44.9 | 59.1 | -14.2 |
| 19 | 61 | Qwen2.5-7B | MATH500 | 4 | 45.2 | 56.6 | -11.4 |
| 20 | 62 | Qwen2.5-7B | MATH500 | 4 | 45 | 58.3 | -13.3 |
| 21 | 63 | Qwen2.5-7B | MATH500 | 4 | 44.8 | 57.6 | -12.8 |
| 22 | 64 | Qwen2.5-7B | MATH500 | 4 | 42.9 | 60.2 | -17.3 |
| 23 | 65 | Qwen2.5-7B | MATH500 | 4 | 45.4 | 59.7 | -14.3 |
| 24 | 66 | Qwen2.5-7B | MATH500 | 4 | 45.5 | 60.8 | -15.3 |
| 25 | 67 | Qwen2.5-7B | MATH500 | 4 | 45 | 56.7 | -11.7 |
| 26 | 68 | Qwen2.5-7B | MATH500 | 4 | 42.5 | 60.4 | -17.9 |
| 27 | 69 | Qwen2.5-7B | MATH500 | 4 | 43.2 | 58.6 | -15.4 |
| 28 | 70 | Qwen2.5-7B | MATH500 | 4 | 43.8 | 60.2 | -16.4 |
| 29 | 71 | Qwen2.5-7B | MATH500 | 4 | 44.2 | 56 | -11.8 |
| 30 | 72 | Qwen2.5-7B | MATH500 | 4 | 44 | 55.8 | -11.8 |
| 31 | 73 | Qwen2.5-7B | MATH500 | 4 | 44.1 | 59.9 | -15.8 |
| 32 | 74 | Qwen2.5-7B | MATH500 | 4 | 44.6 | 58.4 | -13.8 |
| 33 | 75 | Qwen2.5-7B | MATH500 | 4 | 47.5 | 57.2 | -9.7 |
| 34 | 76 | Qwen2.5-7B | MATH500 | 4 | 43 | 57.7 | -14.7 |
| 35 | 77 | Qwen2.5-7B | MATH500 | 4 | 42 | 59.8 | -17.8 |
| 36 | 78 | Qwen2.5-7B | MATH500 | 4 | 44 | 57.7 | -13.7 |
| 37 | 79 | Qwen2.5-7B | MATH500 | 4 | 43.3 | 59.2 | -15.9 |
| 38 | 80 | Qwen2.5-7B | MATH500 | 4 | 43.9 | 58.5 | -14.6 |
| 39 | 81 | Qwen2.5-7B | MATH500 | 4 | 44.4 | 57.9 | -13.5 |
| 40 | 82 | Qwen2.5-7B | MATH500 | 4 | 45.3 | 58.4 | -13.1 |
| 41 | 83 | Qwen2.5-7B | MATH500 | 4 | 44.2 | 58.8 | -14.6 |
| 42 | 84 | Qwen2.5-7B | MATH500 | 4 | 45.2 | 58.4 | -13.2 |
| 43 | 85 | Qwen2.5-7B | MATH500 | 4 | 45.1 | 56.8 | -11.7 |
| 44 | 86 | Qwen2.5-7B | MATH500 | 4 | 43.3 | 57.2 | -13.9 |
| 45 | 87 | Qwen2.5-7B | MATH500 | 4 | 44.9 | 57.6 | -12.7 |
| 46 | 88 | Qwen2.5-7B | MATH500 | 4 | 45.6 | 59.2 | -13.6 |
| 47 | 89 | Qwen2.5-7B | MATH500 | 4 | 45.2 | 58.1 | -12.9 |
| 48 | 90 | Qwen2.5-7B | MATH500 | 4 | 44.6 | 58.6 | -14 |
| 49 | 91 | Qwen2.5-7B | MATH500 | 4 | 45.9 | 59 | -13.1 |
| 0 | 42 | Qwen2.5-7B | MATH500 | 8 | 42.4 | 60.15 | -17.75 |
| 1 | 43 | Qwen2.5-7B | MATH500 | 8 | 42.25 | 58.85 | -16.6 |
| 2 | 44 | Qwen2.5-7B | MATH500 | 8 | 43.1 | 59.25 | -16.15 |
| 3 | 45 | Qwen2.5-7B | MATH500 | 8 | 44.75 | 58.7 | -13.95 |
| 4 | 46 | Qwen2.5-7B | MATH500 | 8 | 42.6 | 58.45 | -15.85 |
| 5 | 47 | Qwen2.5-7B | MATH500 | 8 | 42.95 | 61.35 | -18.4 |
| 6 | 48 | Qwen2.5-7B | MATH500 | 8 | 43.5 | 59 | -15.5 |
| 7 | 49 | Qwen2.5-7B | MATH500 | 8 | 43.25 | 60.2 | -16.95 |
| 8 | 50 | Qwen2.5-7B | MATH500 | 8 | 41.3 | 60.3 | -19 |
| 9 | 51 | Qwen2.5-7B | MATH500 | 8 | 43.6 | 59 | -15.4 |
| 10 | 52 | Qwen2.5-7B | MATH500 | 8 | 43 | 59.65 | -16.65 |
| 11 | 53 | Qwen2.5-7B | MATH500 | 8 | 42.55 | 60.75 | -18.2 |
| 12 | 54 | Qwen2.5-7B | MATH500 | 8 | 43.6 | 58.95 | -15.35 |
| 13 | 55 | Qwen2.5-7B | MATH500 | 8 | 42.7 | 60.75 | -18.05 |
| 14 | 56 | Qwen2.5-7B | MATH500 | 8 | 43.45 | 59.3 | -15.85 |
| 15 | 57 | Qwen2.5-7B | MATH500 | 8 | 43.05 | 59.5 | -16.45 |
| 16 | 58 | Qwen2.5-7B | MATH500 | 8 | 43.4 | 59 | -15.6 |
| 17 | 59 | Qwen2.5-7B | MATH500 | 8 | 43.65 | 59.8 | -16.15 |
| 18 | 60 | Qwen2.5-7B | MATH500 | 8 | 43.35 | 60 | -16.65 |
| 19 | 61 | Qwen2.5-7B | MATH500 | 8 | 43.2 | 59.35 | -16.15 |
| 20 | 62 | Qwen2.5-7B | MATH500 | 8 | 43.95 | 59.65 | -15.7 |
| 21 | 63 | Qwen2.5-7B | MATH500 | 8 | 43.25 | 58.45 | -15.2 |
| 22 | 64 | Qwen2.5-7B | MATH500 | 8 | 42.8 | 59.55 | -16.75 |
| 23 | 65 | Qwen2.5-7B | MATH500 | 8 | 44.3 | 60.15 | -15.85 |
| 24 | 66 | Qwen2.5-7B | MATH500 | 8 | 43.75 | 58.65 | -14.9 |
| 25 | 67 | Qwen2.5-7B | MATH500 | 8 | 42.75 | 59.2 | -16.45 |
| 26 | 68 | Qwen2.5-7B | MATH500 | 8 | 42.85 | 59.2 | -16.35 |
| 27 | 69 | Qwen2.5-7B | MATH500 | 8 | 42.25 | 59.4 | -17.15 |
| 28 | 70 | Qwen2.5-7B | MATH500 | 8 | 43.1 | 60.9 | -17.8 |
| 29 | 71 | Qwen2.5-7B | MATH500 | 8 | 44.85 | 57.65 | -12.8 |
| 30 | 72 | Qwen2.5-7B | MATH500 | 8 | 42.5 | 58.25 | -15.75 |
| 31 | 73 | Qwen2.5-7B | MATH500 | 8 | 42.15 | 60.3 | -18.15 |
| 32 | 74 | Qwen2.5-7B | MATH500 | 8 | 43.25 | 60.1 | -16.85 |
| 33 | 75 | Qwen2.5-7B | MATH500 | 8 | 44.05 | 59.75 | -15.7 |
| 34 | 76 | Qwen2.5-7B | MATH500 | 8 | 42.1 | 59.15 | -17.05 |
| 35 | 77 | Qwen2.5-7B | MATH500 | 8 | 42.3 | 59.55 | -17.25 |
| 36 | 78 | Qwen2.5-7B | MATH500 | 8 | 44 | 59.45 | -15.45 |
| 37 | 79 | Qwen2.5-7B | MATH500 | 8 | 42.6 | 58.3 | -15.7 |
| 38 | 80 | Qwen2.5-7B | MATH500 | 8 | 43.5 | 59.15 | -15.65 |
| 39 | 81 | Qwen2.5-7B | MATH500 | 8 | 42.95 | 58.9 | -15.95 |
| 40 | 82 | Qwen2.5-7B | MATH500 | 8 | 43.3 | 58.95 | -15.65 |
| 41 | 83 | Qwen2.5-7B | MATH500 | 8 | 43.55 | 58.55 | -15 |
| 42 | 84 | Qwen2.5-7B | MATH500 | 8 | 43.85 | 59.1 | -15.25 |
| 43 | 85 | Qwen2.5-7B | MATH500 | 8 | 43.7 | 58.85 | -15.15 |
| 44 | 86 | Qwen2.5-7B | MATH500 | 8 | 41.75 | 58.95 | -17.2 |
| 45 | 87 | Qwen2.5-7B | MATH500 | 8 | 42.15 | 59.45 | -17.3 |
| 46 | 88 | Qwen2.5-7B | MATH500 | 8 | 43.15 | 59.6 | -16.45 |
| 47 | 89 | Qwen2.5-7B | MATH500 | 8 | 43.3 | 59.95 | -16.65 |
| 48 | 90 | Qwen2.5-7B | MATH500 | 8 | 43.85 | 60.05 | -16.2 |
| 49 | 91 | Qwen2.5-7B | MATH500 | 8 | 44.55 | 58.7 | -14.15 |
| 0 | 42 | Qwen2.5-7B | MATH500 | 12 | 42.9333 | 59.9 | -16.9667 |
| 1 | 43 | Qwen2.5-7B | MATH500 | 12 | 42.2667 | 59.8 | -17.5333 |
| 2 | 44 | Qwen2.5-7B | MATH500 | 12 | 42.2 | 59.6 | -17.4 |
| 3 | 45 | Qwen2.5-7B | MATH500 | 12 | 42.6667 | 59 | -16.3333 |
| 4 | 46 | Qwen2.5-7B | MATH500 | 12 | 43.2 | 59.3333 | -16.1333 |
| 5 | 47 | Qwen2.5-7B | MATH500 | 12 | 42.1333 | 60.8333 | -18.7 |
| 6 | 48 | Qwen2.5-7B | MATH500 | 12 | 42.9 | 59.9 | -17 |
| 7 | 49 | Qwen2.5-7B | MATH500 | 12 | 42.9333 | 59.8 | -16.8667 |
| 8 | 50 | Qwen2.5-7B | MATH500 | 12 | 42.1667 | 59.9333 | -17.7667 |
| 9 | 51 | Qwen2.5-7B | MATH500 | 12 | 41.6667 | 60 | -18.3333 |
| 10 | 52 | Qwen2.5-7B | MATH500 | 12 | 42.8667 | 59.5 | -16.6333 |
| 11 | 53 | Qwen2.5-7B | MATH500 | 12 | 43.0667 | 60.4667 | -17.4 |
| 12 | 54 | Qwen2.5-7B | MATH500 | 12 | 42.7333 | 59.7 | -16.9667 |
| 13 | 55 | Qwen2.5-7B | MATH500 | 12 | 42.8333 | 60.6333 | -17.8 |
| 14 | 56 | Qwen2.5-7B | MATH500 | 12 | 43.1 | 60 | -16.9 |
| 15 | 57 | Qwen2.5-7B | MATH500 | 12 | 42.5667 | 59.3 | -16.7333 |
| 16 | 58 | Qwen2.5-7B | MATH500 | 12 | 44.0333 | 59.4333 | -15.4 |
| 17 | 59 | Qwen2.5-7B | MATH500 | 12 | 42.5 | 59.7667 | -17.2667 |
| 18 | 60 | Qwen2.5-7B | MATH500 | 12 | 43.1333 | 59.4667 | -16.3333 |
| 19 | 61 | Qwen2.5-7B | MATH500 | 12 | 43.1667 | 59.4333 | -16.2667 |
| 20 | 62 | Qwen2.5-7B | MATH500 | 12 | 42.5667 | 60 | -17.4333 |
| 21 | 63 | Qwen2.5-7B | MATH500 | 12 | 42.3667 | 59.2333 | -16.8667 |
| 22 | 64 | Qwen2.5-7B | MATH500 | 12 | 43.4333 | 59.4333 | -16 |
| 23 | 65 | Qwen2.5-7B | MATH500 | 12 | 42.9 | 59.4 | -16.5 |
| 24 | 66 | Qwen2.5-7B | MATH500 | 12 | 42.9333 | 59.6333 | -16.7 |
| 25 | 67 | Qwen2.5-7B | MATH500 | 12 | 42.8333 | 59.9333 | -17.1 |
| 26 | 68 | Qwen2.5-7B | MATH500 | 12 | 42.0667 | 59.7 | -17.6333 |
| 27 | 69 | Qwen2.5-7B | MATH500 | 12 | 42.3 | 59.8333 | -17.5333 |
| 28 | 70 | Qwen2.5-7B | MATH500 | 12 | 42.6333 | 60.6 | -17.9667 |
| 29 | 71 | Qwen2.5-7B | MATH500 | 12 | 43.2333 | 58.4667 | -15.2333 |
| 30 | 72 | Qwen2.5-7B | MATH500 | 12 | 42.6 | 60.2 | -17.6 |
| 31 | 73 | Qwen2.5-7B | MATH500 | 12 | 42.9333 | 59.8333 | -16.9 |
| 32 | 74 | Qwen2.5-7B | MATH500 | 12 | 43.0667 | 59.9 | -16.8333 |
| 33 | 75 | Qwen2.5-7B | MATH500 | 12 | 43.3667 | 59.6333 | -16.2667 |
| 34 | 76 | Qwen2.5-7B | MATH500 | 12 | 41.9 | 60.1 | -18.2 |
| 35 | 77 | Qwen2.5-7B | MATH500 | 12 | 42.1333 | 60.7 | -18.5667 |
| 36 | 78 | Qwen2.5-7B | MATH500 | 12 | 43.0667 | 59.7 | -16.6333 |
| 37 | 79 | Qwen2.5-7B | MATH500 | 12 | 43.3 | 59.1333 | -15.8333 |
| 38 | 80 | Qwen2.5-7B | MATH500 | 12 | 43.2 | 59.8333 | -16.6333 |
| 39 | 81 | Qwen2.5-7B | MATH500 | 12 | 43.5333 | 59.8333 | -16.3 |
| 40 | 82 | Qwen2.5-7B | MATH500 | 12 | 42.7667 | 60.2667 | -17.5 |
| 41 | 83 | Qwen2.5-7B | MATH500 | 12 | 43.0667 | 59.2667 | -16.2 |
| 42 | 84 | Qwen2.5-7B | MATH500 | 12 | 42.3333 | 59.4333 | -17.1 |
| 43 | 85 | Qwen2.5-7B | MATH500 | 12 | 42.6 | 60.0667 | -17.4667 |
| 44 | 86 | Qwen2.5-7B | MATH500 | 12 | 42.3667 | 59.4333 | -17.0667 |
| 45 | 87 | Qwen2.5-7B | MATH500 | 12 | 42.8333 | 59.8667 | -17.0333 |
| 46 | 88 | Qwen2.5-7B | MATH500 | 12 | 43.3667 | 59.8 | -16.4333 |
| 47 | 89 | Qwen2.5-7B | MATH500 | 12 | 43.2 | 59.9 | -16.7 |
| 48 | 90 | Qwen2.5-7B | MATH500 | 12 | 42.3 | 60.7667 | -18.4667 |
| 49 | 91 | Qwen2.5-7B | MATH500 | 12 | 43.0667 | 59.6333 | -16.5667 |
| 0 | 42 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 1 | 43 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 2 | 44 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 3 | 45 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 4 | 46 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 5 | 47 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 6 | 48 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 7 | 49 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 8 | 50 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 9 | 51 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 10 | 52 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 11 | 53 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 12 | 54 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 13 | 55 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 14 | 56 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 15 | 57 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 16 | 58 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 17 | 59 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 18 | 60 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 19 | 61 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 20 | 62 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 21 | 63 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 22 | 64 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 23 | 65 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 24 | 66 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 25 | 67 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 26 | 68 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 27 | 69 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 28 | 70 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 29 | 71 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 30 | 72 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 31 | 73 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 32 | 74 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 33 | 75 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 34 | 76 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 35 | 77 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 36 | 78 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 37 | 79 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 38 | 80 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 39 | 81 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 40 | 82 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 41 | 83 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 42 | 84 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 43 | 85 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 44 | 86 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 45 | 87 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 46 | 88 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 47 | 89 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 48 | 90 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 49 | 91 | Qwen2.5-7B | MATH500 | 16 | 42.575 | 60.05 | -17.475 |
| 0 | 42 | Qwen2.5-7B | SVAMP | 4 | 72.8 | 88.5 | -15.7 |
| 1 | 43 | Qwen2.5-7B | SVAMP | 4 | 71.25 | 88.6 | -17.35 |
| 2 | 44 | Qwen2.5-7B | SVAMP | 4 | 69.75 | 89.1 | -19.35 |
| 3 | 45 | Qwen2.5-7B | SVAMP | 4 | 70.65 | 90 | -19.35 |
| 4 | 46 | Qwen2.5-7B | SVAMP | 4 | 72.4 | 89 | -16.6 |
| 5 | 47 | Qwen2.5-7B | SVAMP | 4 | 71.35 | 89.9 | -18.55 |
| 6 | 48 | Qwen2.5-7B | SVAMP | 4 | 72.15 | 88.4 | -16.25 |
| 7 | 49 | Qwen2.5-7B | SVAMP | 4 | 71.85 | 89.4 | -17.55 |
| 8 | 50 | Qwen2.5-7B | SVAMP | 4 | 72.15 | 88.5 | -16.35 |
| 9 | 51 | Qwen2.5-7B | SVAMP | 4 | 72.8 | 88.7 | -15.9 |
| 10 | 52 | Qwen2.5-7B | SVAMP | 4 | 72.05 | 89.1 | -17.05 |
| 11 | 53 | Qwen2.5-7B | SVAMP | 4 | 71.75 | 89.7 | -17.95 |
| 12 | 54 | Qwen2.5-7B | SVAMP | 4 | 72.3 | 88.25 | -15.95 |
| 13 | 55 | Qwen2.5-7B | SVAMP | 4 | 71.55 | 89.2 | -17.65 |
| 14 | 56 | Qwen2.5-7B | SVAMP | 4 | 72.65 | 89.2 | -16.55 |
| 15 | 57 | Qwen2.5-7B | SVAMP | 4 | 71.05 | 89.5 | -18.45 |
| 16 | 58 | Qwen2.5-7B | SVAMP | 4 | 71.8 | 89.4 | -17.6 |
| 17 | 59 | Qwen2.5-7B | SVAMP | 4 | 71.5 | 89.15 | -17.65 |
| 18 | 60 | Qwen2.5-7B | SVAMP | 4 | 71.65 | 88.65 | -17 |
| 19 | 61 | Qwen2.5-7B | SVAMP | 4 | 71.45 | 88.5 | -17.05 |
| 20 | 62 | Qwen2.5-7B | SVAMP | 4 | 71.1 | 89.2 | -18.1 |
| 21 | 63 | Qwen2.5-7B | SVAMP | 4 | 71.25 | 89.6 | -18.35 |
| 22 | 64 | Qwen2.5-7B | SVAMP | 4 | 70.35 | 89.55 | -19.2 |
| 23 | 65 | Qwen2.5-7B | SVAMP | 4 | 72.25 | 89.3 | -17.05 |
| 24 | 66 | Qwen2.5-7B | SVAMP | 4 | 70.5 | 87.85 | -17.35 |
| 25 | 67 | Qwen2.5-7B | SVAMP | 4 | 72.55 | 89.5 | -16.95 |
| 26 | 68 | Qwen2.5-7B | SVAMP | 4 | 72 | 89.3 | -17.3 |
| 27 | 69 | Qwen2.5-7B | SVAMP | 4 | 72.5 | 88.9 | -16.4 |
| 28 | 70 | Qwen2.5-7B | SVAMP | 4 | 72.1 | 88.95 | -16.85 |
| 29 | 71 | Qwen2.5-7B | SVAMP | 4 | 71.4 | 89.35 | -17.95 |
| 30 | 72 | Qwen2.5-7B | SVAMP | 4 | 72.15 | 90.4 | -18.25 |
| 31 | 73 | Qwen2.5-7B | SVAMP | 4 | 73.45 | 88.85 | -15.4 |
| 32 | 74 | Qwen2.5-7B | SVAMP | 4 | 71 | 89.2 | -18.2 |
| 33 | 75 | Qwen2.5-7B | SVAMP | 4 | 72.45 | 88.85 | -16.4 |
| 34 | 76 | Qwen2.5-7B | SVAMP | 4 | 69.9 | 89.35 | -19.45 |
| 35 | 77 | Qwen2.5-7B | SVAMP | 4 | 73.65 | 89.9 | -16.25 |
| 36 | 78 | Qwen2.5-7B | SVAMP | 4 | 71.5 | 88.7 | -17.2 |
| 37 | 79 | Qwen2.5-7B | SVAMP | 4 | 71.5 | 89.2 | -17.7 |
| 38 | 80 | Qwen2.5-7B | SVAMP | 4 | 72.15 | 89.35 | -17.2 |
| 39 | 81 | Qwen2.5-7B | SVAMP | 4 | 74.05 | 88.4 | -14.35 |
| 40 | 82 | Qwen2.5-7B | SVAMP | 4 | 71.95 | 89.05 | -17.1 |
| 41 | 83 | Qwen2.5-7B | SVAMP | 4 | 70.75 | 88.6 | -17.85 |
| 42 | 84 | Qwen2.5-7B | SVAMP | 4 | 70.9 | 88.55 | -17.65 |
| 43 | 85 | Qwen2.5-7B | SVAMP | 4 | 71.75 | 88.75 | -17 |
| 44 | 86 | Qwen2.5-7B | SVAMP | 4 | 69.35 | 89.25 | -19.9 |
| 45 | 87 | Qwen2.5-7B | SVAMP | 4 | 71.85 | 89 | -17.15 |
| 46 | 88 | Qwen2.5-7B | SVAMP | 4 | 72.1 | 88.75 | -16.65 |
| 47 | 89 | Qwen2.5-7B | SVAMP | 4 | 71.95 | 89.6 | -17.65 |
| 48 | 90 | Qwen2.5-7B | SVAMP | 4 | 71.85 | 89.55 | -17.7 |
| 49 | 91 | Qwen2.5-7B | SVAMP | 4 | 72.05 | 89.25 | -17.2 |
| 0 | 42 | Qwen2.5-7B | SVAMP | 8 | 71.4 | 89.9 | -18.5 |
| 1 | 43 | Qwen2.5-7B | SVAMP | 8 | 69.625 | 90.325 | -20.7 |
| 2 | 44 | Qwen2.5-7B | SVAMP | 8 | 70.025 | 90.125 | -20.1 |
| 3 | 45 | Qwen2.5-7B | SVAMP | 8 | 69.975 | 91 | -21.025 |
| 4 | 46 | Qwen2.5-7B | SVAMP | 8 | 70.65 | 90.175 | -19.525 |
| 5 | 47 | Qwen2.5-7B | SVAMP | 8 | 69.925 | 90.7 | -20.775 |
| 6 | 48 | Qwen2.5-7B | SVAMP | 8 | 70.175 | 89.725 | -19.55 |
| 7 | 49 | Qwen2.5-7B | SVAMP | 8 | 70.725 | 90.125 | -19.4 |
| 8 | 50 | Qwen2.5-7B | SVAMP | 8 | 71.125 | 90.35 | -19.225 |
| 9 | 51 | Qwen2.5-7B | SVAMP | 8 | 70.875 | 90.175 | -19.3 |
| 10 | 52 | Qwen2.5-7B | SVAMP | 8 | 70.45 | 90.4 | -19.95 |
| 11 | 53 | Qwen2.5-7B | SVAMP | 8 | 71.275 | 90.825 | -19.55 |
| 12 | 54 | Qwen2.5-7B | SVAMP | 8 | 70.675 | 89.95 | -19.275 |
| 13 | 55 | Qwen2.5-7B | SVAMP | 8 | 70.7 | 89.9 | -19.2 |
| 14 | 56 | Qwen2.5-7B | SVAMP | 8 | 70.275 | 89.925 | -19.65 |
| 15 | 57 | Qwen2.5-7B | SVAMP | 8 | 70.375 | 90.625 | -20.25 |
| 16 | 58 | Qwen2.5-7B | SVAMP | 8 | 70.95 | 90.125 | -19.175 |
| 17 | 59 | Qwen2.5-7B | SVAMP | 8 | 70.575 | 90.7 | -20.125 |
| 18 | 60 | Qwen2.5-7B | SVAMP | 8 | 70.575 | 89.525 | -18.95 |
| 19 | 61 | Qwen2.5-7B | SVAMP | 8 | 70.2 | 90.025 | -19.825 |
| 20 | 62 | Qwen2.5-7B | SVAMP | 8 | 70.65 | 90 | -19.35 |
| 21 | 63 | Qwen2.5-7B | SVAMP | 8 | 69.7 | 90.6 | -20.9 |
| 22 | 64 | Qwen2.5-7B | SVAMP | 8 | 70.25 | 90.325 | -20.075 |
| 23 | 65 | Qwen2.5-7B | SVAMP | 8 | 70.15 | 90.3 | -20.15 |
| 24 | 66 | Qwen2.5-7B | SVAMP | 8 | 70.025 | 90.05 | -20.025 |
| 25 | 67 | Qwen2.5-7B | SVAMP | 8 | 70.875 | 90.225 | -19.35 |
| 26 | 68 | Qwen2.5-7B | SVAMP | 8 | 70.125 | 90.35 | -20.225 |
| 27 | 69 | Qwen2.5-7B | SVAMP | 8 | 69.925 | 90.6 | -20.675 |
| 28 | 70 | Qwen2.5-7B | SVAMP | 8 | 70.45 | 90.4 | -19.95 |
| 29 | 71 | Qwen2.5-7B | SVAMP | 8 | 70.475 | 90.55 | -20.075 |
| 30 | 72 | Qwen2.5-7B | SVAMP | 8 | 70.3 | 91.25 | -20.95 |
| 31 | 73 | Qwen2.5-7B | SVAMP | 8 | 71.25 | 90.425 | -19.175 |
| 32 | 74 | Qwen2.5-7B | SVAMP | 8 | 70.8 | 90.425 | -19.625 |
| 33 | 75 | Qwen2.5-7B | SVAMP | 8 | 71.475 | 90.2 | -18.725 |
| 34 | 76 | Qwen2.5-7B | SVAMP | 8 | 69.725 | 90.45 | -20.725 |
| 35 | 77 | Qwen2.5-7B | SVAMP | 8 | 71 | 91 | -20 |
| 36 | 78 | Qwen2.5-7B | SVAMP | 8 | 70.225 | 90.575 | -20.35 |
| 37 | 79 | Qwen2.5-7B | SVAMP | 8 | 70.275 | 90.525 | -20.25 |
| 38 | 80 | Qwen2.5-7B | SVAMP | 8 | 71.275 | 89.775 | -18.5 |
| 39 | 81 | Qwen2.5-7B | SVAMP | 8 | 70.925 | 90.1 | -19.175 |
| 40 | 82 | Qwen2.5-7B | SVAMP | 8 | 70.9 | 90.05 | -19.15 |
| 41 | 83 | Qwen2.5-7B | SVAMP | 8 | 70.075 | 89.425 | -19.35 |
| 42 | 84 | Qwen2.5-7B | SVAMP | 8 | 70.85 | 90.4 | -19.55 |
| 43 | 85 | Qwen2.5-7B | SVAMP | 8 | 69.975 | 90 | -20.025 |
| 44 | 86 | Qwen2.5-7B | SVAMP | 8 | 69.925 | 90.425 | -20.5 |
| 45 | 87 | Qwen2.5-7B | SVAMP | 8 | 70.125 | 90.375 | -20.25 |
| 46 | 88 | Qwen2.5-7B | SVAMP | 8 | 70.3 | 89.75 | -19.45 |
| 47 | 89 | Qwen2.5-7B | SVAMP | 8 | 70.75 | 90.2 | -19.45 |
| 48 | 90 | Qwen2.5-7B | SVAMP | 8 | 70.5 | 90.8 | -20.3 |
| 49 | 91 | Qwen2.5-7B | SVAMP | 8 | 70.625 | 90.125 | -19.5 |
| 0 | 42 | Qwen2.5-7B | SVAMP | 12 | 70.2333 | 90.8333 | -20.6 |
| 1 | 43 | Qwen2.5-7B | SVAMP | 12 | 69.5667 | 90.7 | -21.1333 |
| 2 | 44 | Qwen2.5-7B | SVAMP | 12 | 70 | 90.6 | -20.6 |
| 3 | 45 | Qwen2.5-7B | SVAMP | 12 | 70.1167 | 90.9833 | -20.8667 |
| 4 | 46 | Qwen2.5-7B | SVAMP | 12 | 69.9667 | 90.6 | -20.6333 |
| 5 | 47 | Qwen2.5-7B | SVAMP | 12 | 69.4167 | 90.8333 | -21.4167 |
| 6 | 48 | Qwen2.5-7B | SVAMP | 12 | 69.9667 | 90.4667 | -20.5 |
| 7 | 49 | Qwen2.5-7B | SVAMP | 12 | 70.0167 | 90.8333 | -20.8167 |
| 8 | 50 | Qwen2.5-7B | SVAMP | 12 | 70.35 | 90.6667 | -20.3167 |
| 9 | 51 | Qwen2.5-7B | SVAMP | 12 | 70.2167 | 90.9333 | -20.7167 |
| 10 | 52 | Qwen2.5-7B | SVAMP | 12 | 69.6667 | 90.5 | -20.8333 |
| 11 | 53 | Qwen2.5-7B | SVAMP | 12 | 70.25 | 90.8 | -20.55 |
| 12 | 54 | Qwen2.5-7B | SVAMP | 12 | 69.9167 | 90.6833 | -20.7667 |
| 13 | 55 | Qwen2.5-7B | SVAMP | 12 | 69.65 | 90.5833 | -20.9333 |
| 14 | 56 | Qwen2.5-7B | SVAMP | 12 | 69.8667 | 90.6 | -20.7333 |
| 15 | 57 | Qwen2.5-7B | SVAMP | 12 | 69.9667 | 91.15 | -21.1833 |
| 16 | 58 | Qwen2.5-7B | SVAMP | 12 | 70.0333 | 90.6833 | -20.65 |
| 17 | 59 | Qwen2.5-7B | SVAMP | 12 | 69.9833 | 90.9667 | -20.9833 |
| 18 | 60 | Qwen2.5-7B | SVAMP | 12 | 70.1 | 90.6667 | -20.5667 |
| 19 | 61 | Qwen2.5-7B | SVAMP | 12 | 70.2833 | 90.5833 | -20.3 |
| 20 | 62 | Qwen2.5-7B | SVAMP | 12 | 70.3333 | 90.65 | -20.3167 |
| 21 | 63 | Qwen2.5-7B | SVAMP | 12 | 69.8 | 90.8833 | -21.0833 |
| 22 | 64 | Qwen2.5-7B | SVAMP | 12 | 70.0667 | 90.45 | -20.3833 |
| 23 | 65 | Qwen2.5-7B | SVAMP | 12 | 69.35 | 90.9833 | -21.6333 |
| 24 | 66 | Qwen2.5-7B | SVAMP | 12 | 69.6167 | 90.8667 | -21.25 |
| 25 | 67 | Qwen2.5-7B | SVAMP | 12 | 70.25 | 90.8667 | -20.6167 |
| 26 | 68 | Qwen2.5-7B | SVAMP | 12 | 69.55 | 90.5 | -20.95 |
| 27 | 69 | Qwen2.5-7B | SVAMP | 12 | 69.9667 | 90.7333 | -20.7667 |
| 28 | 70 | Qwen2.5-7B | SVAMP | 12 | 70.2333 | 90.7833 | -20.55 |
| 29 | 71 | Qwen2.5-7B | SVAMP | 12 | 69.8833 | 90.7167 | -20.8333 |
| 30 | 72 | Qwen2.5-7B | SVAMP | 12 | 70.1333 | 91.1 | -20.9667 |
| 31 | 73 | Qwen2.5-7B | SVAMP | 12 | 70.4833 | 90.95 | -20.4667 |
| 32 | 74 | Qwen2.5-7B | SVAMP | 12 | 70 | 90.85 | -20.85 |
| 33 | 75 | Qwen2.5-7B | SVAMP | 12 | 69.85 | 91.0833 | -21.2333 |
| 34 | 76 | Qwen2.5-7B | SVAMP | 12 | 69.1 | 90.6333 | -21.5333 |
| 35 | 77 | Qwen2.5-7B | SVAMP | 12 | 70.2667 | 91.05 | -20.7833 |
| 36 | 78 | Qwen2.5-7B | SVAMP | 12 | 70.2167 | 90.8167 | -20.6 |
| 37 | 79 | Qwen2.5-7B | SVAMP | 12 | 69.8833 | 90.5333 | -20.65 |
| 38 | 80 | Qwen2.5-7B | SVAMP | 12 | 70.3667 | 90.6833 | -20.3167 |
| 39 | 81 | Qwen2.5-7B | SVAMP | 12 | 70.15 | 90.6167 | -20.4667 |
| 40 | 82 | Qwen2.5-7B | SVAMP | 12 | 69.8667 | 90.6833 | -20.8167 |
| 41 | 83 | Qwen2.5-7B | SVAMP | 12 | 69.5333 | 90.8167 | -21.2833 |
| 42 | 84 | Qwen2.5-7B | SVAMP | 12 | 69.9833 | 90.9667 | -20.9833 |
| 43 | 85 | Qwen2.5-7B | SVAMP | 12 | 69.8667 | 90.6833 | -20.8167 |
| 44 | 86 | Qwen2.5-7B | SVAMP | 12 | 69.6333 | 91.0667 | -21.4333 |
| 45 | 87 | Qwen2.5-7B | SVAMP | 12 | 69.5833 | 90.7167 | -21.1333 |
| 46 | 88 | Qwen2.5-7B | SVAMP | 12 | 69.5167 | 90.6 | -21.0833 |
| 47 | 89 | Qwen2.5-7B | SVAMP | 12 | 69.95 | 90.7833 | -20.8333 |
| 48 | 90 | Qwen2.5-7B | SVAMP | 12 | 69.5167 | 91.2167 | -21.7 |
| 49 | 91 | Qwen2.5-7B | SVAMP | 12 | 69.8833 | 90.7667 | -20.8833 |
| 0 | 42 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 1 | 43 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 2 | 44 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 3 | 45 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 4 | 46 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 5 | 47 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 6 | 48 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 7 | 49 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 8 | 50 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 9 | 51 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 10 | 52 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 11 | 53 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 12 | 54 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 13 | 55 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 14 | 56 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 15 | 57 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 16 | 58 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 17 | 59 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 18 | 60 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 19 | 61 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 20 | 62 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 21 | 63 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 22 | 64 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 23 | 65 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 24 | 66 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 25 | 67 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 26 | 68 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 27 | 69 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 28 | 70 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 29 | 71 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 30 | 72 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 31 | 73 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 32 | 74 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 33 | 75 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 34 | 76 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 35 | 77 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 36 | 78 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 37 | 79 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 38 | 80 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 39 | 81 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 40 | 82 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 41 | 83 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 42 | 84 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 43 | 85 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 44 | 86 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 45 | 87 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 46 | 88 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 47 | 89 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 48 | 90 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 49 | 91 | Qwen2.5-7B | SVAMP | 16 | 69.6375 | 91.125 | -21.4875 |
| 0 | 42 | Qwen2.5-Math | AQuA | 4 | 54.9213 | 55.7087 | -0.7874 |
| 1 | 43 | Qwen2.5-Math | AQuA | 4 | 51.1811 | 53.1496 | -1.9685 |
| 2 | 44 | Qwen2.5-Math | AQuA | 4 | 53.937 | 58.6614 | -4.7244 |
| 3 | 45 | Qwen2.5-Math | AQuA | 4 | 51.5748 | 56.1024 | -4.5276 |
| 4 | 46 | Qwen2.5-Math | AQuA | 4 | 53.937 | 57.6772 | -3.7402 |
| 5 | 47 | Qwen2.5-Math | AQuA | 4 | 51.1811 | 56.8898 | -5.7087 |
| 6 | 48 | Qwen2.5-Math | AQuA | 4 | 55.1181 | 54.9213 | 0.1969 |
| 7 | 49 | Qwen2.5-Math | AQuA | 4 | 53.1496 | 58.8583 | -5.7087 |
| 8 | 50 | Qwen2.5-Math | AQuA | 4 | 50.5906 | 56.4961 | -5.9055 |
| 9 | 51 | Qwen2.5-Math | AQuA | 4 | 52.9528 | 52.3622 | 0.5906 |
| 10 | 52 | Qwen2.5-Math | AQuA | 4 | 51.1811 | 54.7244 | -3.5433 |
| 11 | 53 | Qwen2.5-Math | AQuA | 4 | 51.7717 | 56.6929 | -4.9213 |
| 12 | 54 | Qwen2.5-Math | AQuA | 4 | 56.4961 | 55.315 | 1.1811 |
| 13 | 55 | Qwen2.5-Math | AQuA | 4 | 54.5276 | 58.6614 | -4.1339 |
| 14 | 56 | Qwen2.5-Math | AQuA | 4 | 53.1496 | 54.9213 | -1.7717 |
| 15 | 57 | Qwen2.5-Math | AQuA | 4 | 53.1496 | 55.5118 | -2.3622 |
| 16 | 58 | Qwen2.5-Math | AQuA | 4 | 51.7717 | 55.7087 | -3.937 |
| 17 | 59 | Qwen2.5-Math | AQuA | 4 | 52.1654 | 53.3465 | -1.1811 |
| 18 | 60 | Qwen2.5-Math | AQuA | 4 | 55.315 | 53.937 | 1.378 |
| 19 | 61 | Qwen2.5-Math | AQuA | 4 | 52.5591 | 57.4803 | -4.9213 |
| 20 | 62 | Qwen2.5-Math | AQuA | 4 | 51.9685 | 52.9528 | -0.9843 |
| 21 | 63 | Qwen2.5-Math | AQuA | 4 | 53.5433 | 55.9055 | -2.3622 |
| 22 | 64 | Qwen2.5-Math | AQuA | 4 | 51.378 | 54.1339 | -2.7559 |
| 23 | 65 | Qwen2.5-Math | AQuA | 4 | 53.937 | 54.9213 | -0.9843 |
| 24 | 66 | Qwen2.5-Math | AQuA | 4 | 56.4961 | 55.5118 | 0.9843 |
| 25 | 67 | Qwen2.5-Math | AQuA | 4 | 51.7717 | 53.1496 | -1.378 |
| 26 | 68 | Qwen2.5-Math | AQuA | 4 | 51.9685 | 55.7087 | -3.7402 |
| 27 | 69 | Qwen2.5-Math | AQuA | 4 | 52.1654 | 55.7087 | -3.5433 |
| 28 | 70 | Qwen2.5-Math | AQuA | 4 | 52.5591 | 54.3307 | -1.7717 |
| 29 | 71 | Qwen2.5-Math | AQuA | 4 | 56.8898 | 58.2677 | -1.378 |
| 30 | 72 | Qwen2.5-Math | AQuA | 4 | 52.3622 | 53.7402 | -1.378 |
| 31 | 73 | Qwen2.5-Math | AQuA | 4 | 53.7402 | 53.3465 | 0.3937 |
| 32 | 74 | Qwen2.5-Math | AQuA | 4 | 54.9213 | 57.0866 | -2.1654 |
| 33 | 75 | Qwen2.5-Math | AQuA | 4 | 52.7559 | 53.937 | -1.1811 |
| 34 | 76 | Qwen2.5-Math | AQuA | 4 | 52.1654 | 52.9528 | -0.7874 |
| 35 | 77 | Qwen2.5-Math | AQuA | 4 | 53.5433 | 55.1181 | -1.5748 |
| 36 | 78 | Qwen2.5-Math | AQuA | 4 | 52.5591 | 54.9213 | -2.3622 |
| 37 | 79 | Qwen2.5-Math | AQuA | 4 | 51.9685 | 55.1181 | -3.1496 |
| 38 | 80 | Qwen2.5-Math | AQuA | 4 | 54.9213 | 50.1969 | 4.7244 |
| 39 | 81 | Qwen2.5-Math | AQuA | 4 | 51.1811 | 56.8898 | -5.7087 |
| 40 | 82 | Qwen2.5-Math | AQuA | 4 | 53.1496 | 54.1339 | -0.9843 |
| 41 | 83 | Qwen2.5-Math | AQuA | 4 | 51.7717 | 57.0866 | -5.315 |
| 42 | 84 | Qwen2.5-Math | AQuA | 4 | 52.1654 | 52.5591 | -0.3937 |
| 43 | 85 | Qwen2.5-Math | AQuA | 4 | 49.2126 | 53.7402 | -4.5276 |
| 44 | 86 | Qwen2.5-Math | AQuA | 4 | 54.9213 | 52.3622 | 2.5591 |
| 45 | 87 | Qwen2.5-Math | AQuA | 4 | 51.1811 | 58.2677 | -7.0866 |
| 46 | 88 | Qwen2.5-Math | AQuA | 4 | 54.7244 | 52.7559 | 1.9685 |
| 47 | 89 | Qwen2.5-Math | AQuA | 4 | 49.8031 | 55.9055 | -6.1024 |
| 48 | 90 | Qwen2.5-Math | AQuA | 4 | 52.7559 | 53.7402 | -0.9843 |
| 49 | 91 | Qwen2.5-Math | AQuA | 4 | 57.4803 | 52.7559 | 4.7244 |
| 0 | 42 | Qwen2.5-Math | AQuA | 8 | 52.3622 | 55.9055 | -3.5433 |
| 1 | 43 | Qwen2.5-Math | AQuA | 8 | 51.1811 | 55.5118 | -4.3307 |
| 2 | 44 | Qwen2.5-Math | AQuA | 8 | 55.7087 | 57.0866 | -1.378 |
| 3 | 45 | Qwen2.5-Math | AQuA | 8 | 52.8543 | 54.9213 | -2.0669 |
| 4 | 46 | Qwen2.5-Math | AQuA | 8 | 51.8701 | 55.4134 | -3.5433 |
| 5 | 47 | Qwen2.5-Math | AQuA | 8 | 51.2795 | 56.9882 | -5.7087 |
| 6 | 48 | Qwen2.5-Math | AQuA | 8 | 54.0354 | 54.1339 | -0.0984 |
| 7 | 49 | Qwen2.5-Math | AQuA | 8 | 52.6575 | 57.0866 | -4.4291 |
| 8 | 50 | Qwen2.5-Math | AQuA | 8 | 52.0669 | 56.1024 | -4.0354 |
| 9 | 51 | Qwen2.5-Math | AQuA | 8 | 50.7874 | 55.8071 | -5.0197 |
| 10 | 52 | Qwen2.5-Math | AQuA | 8 | 52.0669 | 55.6102 | -3.5433 |
| 11 | 53 | Qwen2.5-Math | AQuA | 8 | 50.4921 | 55.8071 | -5.315 |
| 12 | 54 | Qwen2.5-Math | AQuA | 8 | 53.937 | 55.4134 | -1.4764 |
| 13 | 55 | Qwen2.5-Math | AQuA | 8 | 54.9213 | 56.7913 | -1.8701 |
| 14 | 56 | Qwen2.5-Math | AQuA | 8 | 52.9528 | 55.9055 | -2.9528 |
| 15 | 57 | Qwen2.5-Math | AQuA | 8 | 51.1811 | 56.2992 | -5.1181 |
| 16 | 58 | Qwen2.5-Math | AQuA | 8 | 50.7874 | 54.3307 | -3.5433 |
| 17 | 59 | Qwen2.5-Math | AQuA | 8 | 52.2638 | 55.8071 | -3.5433 |
| 18 | 60 | Qwen2.5-Math | AQuA | 8 | 52.9528 | 55.9055 | -2.9528 |
| 19 | 61 | Qwen2.5-Math | AQuA | 8 | 52.1654 | 56.2008 | -4.0354 |
| 20 | 62 | Qwen2.5-Math | AQuA | 8 | 51.5748 | 54.4291 | -2.8543 |
| 21 | 63 | Qwen2.5-Math | AQuA | 8 | 52.3622 | 56.1024 | -3.7402 |
| 22 | 64 | Qwen2.5-Math | AQuA | 8 | 51.2795 | 54.9213 | -3.6417 |
| 23 | 65 | Qwen2.5-Math | AQuA | 8 | 51.8701 | 57.0866 | -5.2165 |
| 24 | 66 | Qwen2.5-Math | AQuA | 8 | 53.3465 | 58.1693 | -4.8228 |
| 25 | 67 | Qwen2.5-Math | AQuA | 8 | 51.6732 | 56.0039 | -4.3307 |
| 26 | 68 | Qwen2.5-Math | AQuA | 8 | 51.7717 | 55.6102 | -3.8386 |
| 27 | 69 | Qwen2.5-Math | AQuA | 8 | 52.8543 | 54.2323 | -1.378 |
| 28 | 70 | Qwen2.5-Math | AQuA | 8 | 53.5433 | 54.5276 | -0.9843 |
| 29 | 71 | Qwen2.5-Math | AQuA | 8 | 53.1496 | 55.0197 | -1.8701 |
| 30 | 72 | Qwen2.5-Math | AQuA | 8 | 53.8386 | 54.7244 | -0.8858 |
| 31 | 73 | Qwen2.5-Math | AQuA | 8 | 53.4449 | 55.2165 | -1.7717 |
| 32 | 74 | Qwen2.5-Math | AQuA | 8 | 53.8386 | 57.185 | -3.3465 |
| 33 | 75 | Qwen2.5-Math | AQuA | 8 | 51.9685 | 56.1024 | -4.1339 |
| 34 | 76 | Qwen2.5-Math | AQuA | 8 | 52.0669 | 55.315 | -3.248 |
| 35 | 77 | Qwen2.5-Math | AQuA | 8 | 54.3307 | 55.6102 | -1.2795 |
| 36 | 78 | Qwen2.5-Math | AQuA | 8 | 52.2638 | 54.9213 | -2.6575 |
| 37 | 79 | Qwen2.5-Math | AQuA | 8 | 53.0512 | 55.4134 | -2.3622 |
| 38 | 80 | Qwen2.5-Math | AQuA | 8 | 51.6732 | 54.1339 | -2.4606 |
| 39 | 81 | Qwen2.5-Math | AQuA | 8 | 52.7559 | 56.9882 | -4.2323 |
| 40 | 82 | Qwen2.5-Math | AQuA | 8 | 51.8701 | 56.3976 | -4.5276 |
| 41 | 83 | Qwen2.5-Math | AQuA | 8 | 52.6575 | 56.1024 | -3.4449 |
| 42 | 84 | Qwen2.5-Math | AQuA | 8 | 51.378 | 53.6417 | -2.2638 |
| 43 | 85 | Qwen2.5-Math | AQuA | 8 | 52.2638 | 54.3307 | -2.0669 |
| 44 | 86 | Qwen2.5-Math | AQuA | 8 | 53.7402 | 56.1024 | -2.3622 |
| 45 | 87 | Qwen2.5-Math | AQuA | 8 | 50.2953 | 57.3819 | -7.0866 |
| 46 | 88 | Qwen2.5-Math | AQuA | 8 | 54.0354 | 55.0197 | -0.9843 |
| 47 | 89 | Qwen2.5-Math | AQuA | 8 | 51.9685 | 55.2165 | -3.248 |
| 48 | 90 | Qwen2.5-Math | AQuA | 8 | 52.1654 | 56.0039 | -3.8386 |
| 49 | 91 | Qwen2.5-Math | AQuA | 8 | 54.4291 | 53.7402 | 0.689 |
| 0 | 42 | Qwen2.5-Math | AQuA | 12 | 52.5591 | 56.2336 | -3.6745 |
| 1 | 43 | Qwen2.5-Math | AQuA | 12 | 51.706 | 55.8399 | -4.1339 |
| 2 | 44 | Qwen2.5-Math | AQuA | 12 | 52.9528 | 56.4961 | -3.5433 |
| 3 | 45 | Qwen2.5-Math | AQuA | 12 | 52.3622 | 55.315 | -2.9528 |
| 4 | 46 | Qwen2.5-Math | AQuA | 12 | 52.0997 | 55.5118 | -3.4121 |
| 5 | 47 | Qwen2.5-Math | AQuA | 12 | 50.9843 | 56.1024 | -5.1181 |
| 6 | 48 | Qwen2.5-Math | AQuA | 12 | 52.4278 | 54.1339 | -1.706 |
| 7 | 49 | Qwen2.5-Math | AQuA | 12 | 52.8215 | 56.9554 | -4.1339 |
| 8 | 50 | Qwen2.5-Math | AQuA | 12 | 52.3622 | 56.3648 | -4.0026 |
| 9 | 51 | Qwen2.5-Math | AQuA | 12 | 50.9186 | 55.1181 | -4.1995 |
| 10 | 52 | Qwen2.5-Math | AQuA | 12 | 52.1654 | 55.7743 | -3.6089 |
| 11 | 53 | Qwen2.5-Math | AQuA | 12 | 51.3123 | 55.643 | -4.3307 |
| 12 | 54 | Qwen2.5-Math | AQuA | 12 | 52.3622 | 56.2992 | -3.937 |
| 13 | 55 | Qwen2.5-Math | AQuA | 12 | 53.084 | 56.1024 | -3.0184 |
| 14 | 56 | Qwen2.5-Math | AQuA | 12 | 53.0184 | 55.4462 | -2.4278 |
| 15 | 57 | Qwen2.5-Math | AQuA | 12 | 52.7559 | 55.9055 | -3.1496 |
| 16 | 58 | Qwen2.5-Math | AQuA | 12 | 52.6903 | 55.1181 | -2.4278 |
| 17 | 59 | Qwen2.5-Math | AQuA | 12 | 52.7559 | 55.9711 | -3.2152 |
| 18 | 60 | Qwen2.5-Math | AQuA | 12 | 51.9029 | 55.8399 | -3.937 |
| 19 | 61 | Qwen2.5-Math | AQuA | 12 | 51.3123 | 56.4304 | -5.1181 |
| 20 | 62 | Qwen2.5-Math | AQuA | 12 | 53.4121 | 55.2493 | -1.8373 |
| 21 | 63 | Qwen2.5-Math | AQuA | 12 | 52.1654 | 55.3806 | -3.2152 |
| 22 | 64 | Qwen2.5-Math | AQuA | 12 | 51.9029 | 56.4304 | -4.5276 |
| 23 | 65 | Qwen2.5-Math | AQuA | 12 | 51.5748 | 56.168 | -4.5932 |
| 24 | 66 | Qwen2.5-Math | AQuA | 12 | 52.6247 | 56.5617 | -3.937 |
| 25 | 67 | Qwen2.5-Math | AQuA | 12 | 52.6903 | 55.5774 | -2.8871 |
| 26 | 68 | Qwen2.5-Math | AQuA | 12 | 52.4934 | 56.6929 | -4.1995 |
| 27 | 69 | Qwen2.5-Math | AQuA | 12 | 52.231 | 56.2336 | -4.0026 |
| 28 | 70 | Qwen2.5-Math | AQuA | 12 | 52.5591 | 55.2493 | -2.6903 |
| 29 | 71 | Qwen2.5-Math | AQuA | 12 | 52.8871 | 55.8399 | -2.9528 |
| 30 | 72 | Qwen2.5-Math | AQuA | 12 | 52.8215 | 55.3806 | -2.5591 |
| 31 | 73 | Qwen2.5-Math | AQuA | 12 | 54.0682 | 55.643 | -1.5748 |
| 32 | 74 | Qwen2.5-Math | AQuA | 12 | 52.8871 | 56.1024 | -3.2152 |
| 33 | 75 | Qwen2.5-Math | AQuA | 12 | 51.8373 | 56.7585 | -4.9213 |
| 34 | 76 | Qwen2.5-Math | AQuA | 12 | 52.8215 | 55.7743 | -2.9528 |
| 35 | 77 | Qwen2.5-Math | AQuA | 12 | 52.7559 | 55.9055 | -3.1496 |
| 36 | 78 | Qwen2.5-Math | AQuA | 12 | 52.6903 | 56.4304 | -3.7402 |
| 37 | 79 | Qwen2.5-Math | AQuA | 12 | 52.6247 | 56.168 | -3.5433 |
| 38 | 80 | Qwen2.5-Math | AQuA | 12 | 52.6247 | 54.6588 | -2.0341 |
| 39 | 81 | Qwen2.5-Math | AQuA | 12 | 52.7559 | 56.8241 | -4.0682 |
| 40 | 82 | Qwen2.5-Math | AQuA | 12 | 52.0341 | 56.2992 | -4.2651 |
| 41 | 83 | Qwen2.5-Math | AQuA | 12 | 51.6404 | 56.168 | -4.5276 |
| 42 | 84 | Qwen2.5-Math | AQuA | 12 | 51.9685 | 54.79 | -2.8215 |
| 43 | 85 | Qwen2.5-Math | AQuA | 12 | 52.4934 | 54.5932 | -2.0997 |
| 44 | 86 | Qwen2.5-Math | AQuA | 12 | 52.231 | 56.3648 | -4.1339 |
| 45 | 87 | Qwen2.5-Math | AQuA | 12 | 51.9029 | 56.4304 | -4.5276 |
| 46 | 88 | Qwen2.5-Math | AQuA | 12 | 53.4121 | 55.1837 | -1.7717 |
| 47 | 89 | Qwen2.5-Math | AQuA | 12 | 51.8373 | 55.643 | -3.8058 |
| 48 | 90 | Qwen2.5-Math | AQuA | 12 | 52.8215 | 55.9055 | -3.084 |
| 49 | 91 | Qwen2.5-Math | AQuA | 12 | 52.0341 | 55.4462 | -3.4121 |
| 0 | 42 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 1 | 43 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 2 | 44 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 3 | 45 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 4 | 46 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 5 | 47 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 6 | 48 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 7 | 49 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 8 | 50 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 9 | 51 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 10 | 52 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 11 | 53 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 12 | 54 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 13 | 55 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 14 | 56 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 15 | 57 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 16 | 58 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 17 | 59 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 18 | 60 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 19 | 61 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 20 | 62 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 21 | 63 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 22 | 64 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 23 | 65 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 24 | 66 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 25 | 67 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 26 | 68 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 27 | 69 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 28 | 70 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 29 | 71 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 30 | 72 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 31 | 73 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 32 | 74 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 33 | 75 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 34 | 76 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 35 | 77 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 36 | 78 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 37 | 79 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 38 | 80 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 39 | 81 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 40 | 82 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 41 | 83 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 42 | 84 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 43 | 85 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 44 | 86 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 45 | 87 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 46 | 88 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 47 | 89 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 48 | 90 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 49 | 91 | Qwen2.5-Math | AQuA | 16 | 52.5098 | 55.9055 | -3.3957 |
| 0 | 42 | Qwen2.5-Math | CommonsenseQA | 4 | 49.0991 | 45.3726 | 3.7265 |
| 1 | 43 | Qwen2.5-Math | CommonsenseQA | 4 | 49.959 | 45.2088 | 4.7502 |
| 2 | 44 | Qwen2.5-Math | CommonsenseQA | 4 | 49.2219 | 45.7412 | 3.4808 |
| 3 | 45 | Qwen2.5-Math | CommonsenseQA | 4 | 49.3038 | 45.1679 | 4.136 |
| 4 | 46 | Qwen2.5-Math | CommonsenseQA | 4 | 48.9353 | 44.7584 | 4.1769 |
| 5 | 47 | Qwen2.5-Math | CommonsenseQA | 4 | 49.3857 | 45.045 | 4.3407 |
| 6 | 48 | Qwen2.5-Math | CommonsenseQA | 4 | 49.5905 | 45.045 | 4.5455 |
| 7 | 49 | Qwen2.5-Math | CommonsenseQA | 4 | 48.8124 | 46.2735 | 2.5389 |
| 8 | 50 | Qwen2.5-Math | CommonsenseQA | 4 | 48.9353 | 44.5127 | 4.4226 |
| 9 | 51 | Qwen2.5-Math | CommonsenseQA | 4 | 48.7305 | 44.7174 | 4.0131 |
| 10 | 52 | Qwen2.5-Math | CommonsenseQA | 4 | 49.7133 | 45.7002 | 4.0131 |
| 11 | 53 | Qwen2.5-Math | CommonsenseQA | 4 | 49.0991 | 45.3726 | 3.7265 |
| 12 | 54 | Qwen2.5-Math | CommonsenseQA | 4 | 48.8534 | 44.1032 | 4.7502 |
| 13 | 55 | Qwen2.5-Math | CommonsenseQA | 4 | 50.1229 | 46.4373 | 3.6855 |
| 14 | 56 | Qwen2.5-Math | CommonsenseQA | 4 | 48.1163 | 43.9803 | 4.136 |
| 15 | 57 | Qwen2.5-Math | CommonsenseQA | 4 | 49.5495 | 45.8231 | 3.7265 |
| 16 | 58 | Qwen2.5-Math | CommonsenseQA | 4 | 47.543 | 44.1441 | 3.3989 |
| 17 | 59 | Qwen2.5-Math | CommonsenseQA | 4 | 48.6077 | 44.6765 | 3.9312 |
| 18 | 60 | Qwen2.5-Math | CommonsenseQA | 4 | 49.8771 | 45.4955 | 4.3817 |
| 19 | 61 | Qwen2.5-Math | CommonsenseQA | 4 | 50.1229 | 44.8812 | 5.2416 |
| 20 | 62 | Qwen2.5-Math | CommonsenseQA | 4 | 49.2629 | 46.3145 | 2.9484 |
| 21 | 63 | Qwen2.5-Math | CommonsenseQA | 4 | 49.4267 | 45.4545 | 3.9722 |
| 22 | 64 | Qwen2.5-Math | CommonsenseQA | 4 | 49.181 | 44.3898 | 4.7912 |
| 23 | 65 | Qwen2.5-Math | CommonsenseQA | 4 | 49.3857 | 45.7002 | 3.6855 |
| 24 | 66 | Qwen2.5-Math | CommonsenseQA | 4 | 49.3448 | 44.8812 | 4.4636 |
| 25 | 67 | Qwen2.5-Math | CommonsenseQA | 4 | 49.181 | 45.9459 | 3.2351 |
| 26 | 68 | Qwen2.5-Math | CommonsenseQA | 4 | 49.8771 | 45.4136 | 4.4636 |
| 27 | 69 | Qwen2.5-Math | CommonsenseQA | 4 | 47.9115 | 43.6937 | 4.2179 |
| 28 | 70 | Qwen2.5-Math | CommonsenseQA | 4 | 49.3857 | 45.905 | 3.4808 |
| 29 | 71 | Qwen2.5-Math | CommonsenseQA | 4 | 48.1572 | 46.0688 | 2.0885 |
| 30 | 72 | Qwen2.5-Math | CommonsenseQA | 4 | 48.1982 | 45.6593 | 2.5389 |
| 31 | 73 | Qwen2.5-Math | CommonsenseQA | 4 | 49.2629 | 45.4545 | 3.8084 |
| 32 | 74 | Qwen2.5-Math | CommonsenseQA | 4 | 48.4439 | 44.3898 | 4.0541 |
| 33 | 75 | Qwen2.5-Math | CommonsenseQA | 4 | 47.6249 | 45.3317 | 2.2932 |
| 34 | 76 | Qwen2.5-Math | CommonsenseQA | 4 | 49.8771 | 45.2088 | 4.6683 |
| 35 | 77 | Qwen2.5-Math | CommonsenseQA | 4 | 49.4267 | 45.1679 | 4.2588 |
| 36 | 78 | Qwen2.5-Math | CommonsenseQA | 4 | 48.7305 | 46.724 | 2.0066 |
| 37 | 79 | Qwen2.5-Math | CommonsenseQA | 4 | 48.4439 | 46.1916 | 2.2523 |
| 38 | 80 | Qwen2.5-Math | CommonsenseQA | 4 | 48.6486 | 45.8231 | 2.8256 |
| 39 | 81 | Qwen2.5-Math | CommonsenseQA | 4 | 49.5495 | 45.9459 | 3.6036 |
| 40 | 82 | Qwen2.5-Math | CommonsenseQA | 4 | 50.1229 | 45.9869 | 4.136 |
| 41 | 83 | Qwen2.5-Math | CommonsenseQA | 4 | 50.1638 | 46.6011 | 3.5627 |
| 42 | 84 | Qwen2.5-Math | CommonsenseQA | 4 | 49.181 | 45.864 | 3.317 |
| 43 | 85 | Qwen2.5-Math | CommonsenseQA | 4 | 48.9353 | 46.6011 | 2.3342 |
| 44 | 86 | Qwen2.5-Math | CommonsenseQA | 4 | 48.7715 | 45.3317 | 3.4398 |
| 45 | 87 | Qwen2.5-Math | CommonsenseQA | 4 | 49.3038 | 45.864 | 3.4398 |
| 46 | 88 | Qwen2.5-Math | CommonsenseQA | 4 | 50.5733 | 45.6183 | 4.955 |
| 47 | 89 | Qwen2.5-Math | CommonsenseQA | 4 | 49.4676 | 45.7821 | 3.6855 |
| 48 | 90 | Qwen2.5-Math | CommonsenseQA | 4 | 48.8124 | 44.7993 | 4.0131 |
| 49 | 91 | Qwen2.5-Math | CommonsenseQA | 4 | 48.5258 | 45.2498 | 3.276 |
| 0 | 42 | Qwen2.5-Math | CommonsenseQA | 8 | 49.1196 | 44.697 | 4.4226 |
| 1 | 43 | Qwen2.5-Math | CommonsenseQA | 8 | 49.2834 | 45.6798 | 3.6036 |
| 2 | 44 | Qwen2.5-Math | CommonsenseQA | 8 | 49.4267 | 44.9427 | 4.484 |
| 3 | 45 | Qwen2.5-Math | CommonsenseQA | 8 | 49.959 | 44.7789 | 5.1802 |
| 4 | 46 | Qwen2.5-Math | CommonsenseQA | 8 | 49.1196 | 45.6388 | 3.4808 |
| 5 | 47 | Qwen2.5-Math | CommonsenseQA | 8 | 49.959 | 45.516 | 4.4431 |
| 6 | 48 | Qwen2.5-Math | CommonsenseQA | 8 | 48.6691 | 44.8608 | 3.8084 |
| 7 | 49 | Qwen2.5-Math | CommonsenseQA | 8 | 49.4267 | 44.9222 | 4.5045 |
| 8 | 50 | Qwen2.5-Math | CommonsenseQA | 8 | 48.8943 | 45.2703 | 3.6241 |
| 9 | 51 | Qwen2.5-Math | CommonsenseQA | 8 | 49.7543 | 45.4136 | 4.3407 |
| 10 | 52 | Qwen2.5-Math | CommonsenseQA | 8 | 49.3448 | 45.4136 | 3.9312 |
| 11 | 53 | Qwen2.5-Math | CommonsenseQA | 8 | 49.2834 | 45.1065 | 4.1769 |
| 12 | 54 | Qwen2.5-Math | CommonsenseQA | 8 | 48.9353 | 44.9631 | 3.9722 |
| 13 | 55 | Qwen2.5-Math | CommonsenseQA | 8 | 49.7338 | 45.3726 | 4.3612 |
| 14 | 56 | Qwen2.5-Math | CommonsenseQA | 8 | 48.9762 | 44.9427 | 4.0336 |
| 15 | 57 | Qwen2.5-Math | CommonsenseQA | 8 | 48.7101 | 44.9631 | 3.7469 |
| 16 | 58 | Qwen2.5-Math | CommonsenseQA | 8 | 48.0139 | 44.9427 | 3.0713 |
| 17 | 59 | Qwen2.5-Math | CommonsenseQA | 8 | 49.2629 | 44.4922 | 4.7707 |
| 18 | 60 | Qwen2.5-Math | CommonsenseQA | 8 | 49.57 | 45.3522 | 4.2179 |
| 19 | 61 | Qwen2.5-Math | CommonsenseQA | 8 | 49.4676 | 44.3284 | 5.1392 |
| 20 | 62 | Qwen2.5-Math | CommonsenseQA | 8 | 49.0172 | 46.7854 | 2.2318 |
| 21 | 63 | Qwen2.5-Math | CommonsenseQA | 8 | 49.2424 | 45.475 | 3.7674 |
| 22 | 64 | Qwen2.5-Math | CommonsenseQA | 8 | 49.2219 | 44.4513 | 4.7707 |
| 23 | 65 | Qwen2.5-Math | CommonsenseQA | 8 | 48.7305 | 45.864 | 2.8665 |
| 24 | 66 | Qwen2.5-Math | CommonsenseQA | 8 | 49.2629 | 44.9222 | 4.3407 |
| 25 | 67 | Qwen2.5-Math | CommonsenseQA | 8 | 50.5528 | 44.9836 | 5.5692 |
| 26 | 68 | Qwen2.5-Math | CommonsenseQA | 8 | 48.7305 | 45.1884 | 3.5422 |
| 27 | 69 | Qwen2.5-Math | CommonsenseQA | 8 | 48.4848 | 44.8403 | 3.6446 |
| 28 | 70 | Qwen2.5-Math | CommonsenseQA | 8 | 49.4062 | 45.3317 | 4.0745 |
| 29 | 71 | Qwen2.5-Math | CommonsenseQA | 8 | 48.5258 | 45.3112 | 3.2146 |
| 30 | 72 | Qwen2.5-Math | CommonsenseQA | 8 | 49.2015 | 45.2293 | 3.9722 |
| 31 | 73 | Qwen2.5-Math | CommonsenseQA | 8 | 49.2629 | 45.5569 | 3.706 |
| 32 | 74 | Qwen2.5-Math | CommonsenseQA | 8 | 48.8124 | 45.1679 | 3.6446 |
| 33 | 75 | Qwen2.5-Math | CommonsenseQA | 8 | 48.5053 | 45.0246 | 3.4808 |
| 34 | 76 | Qwen2.5-Math | CommonsenseQA | 8 | 49.9181 | 45.7207 | 4.1974 |
| 35 | 77 | Qwen2.5-Math | CommonsenseQA | 8 | 49.5495 | 45.4955 | 4.0541 |
| 36 | 78 | Qwen2.5-Math | CommonsenseQA | 8 | 49.1196 | 45.4136 | 3.706 |
| 37 | 79 | Qwen2.5-Math | CommonsenseQA | 8 | 49.2834 | 45.4955 | 3.7879 |
| 38 | 80 | Qwen2.5-Math | CommonsenseQA | 8 | 48.9967 | 45.2498 | 3.7469 |
| 39 | 81 | Qwen2.5-Math | CommonsenseQA | 8 | 49.4881 | 44.9222 | 4.5659 |
| 40 | 82 | Qwen2.5-Math | CommonsenseQA | 8 | 50.0819 | 45.3317 | 4.7502 |
| 41 | 83 | Qwen2.5-Math | CommonsenseQA | 8 | 49.3653 | 45.7821 | 3.5831 |
| 42 | 84 | Qwen2.5-Math | CommonsenseQA | 8 | 49.9795 | 44.6765 | 5.303 |
| 43 | 85 | Qwen2.5-Math | CommonsenseQA | 8 | 49.7338 | 45.1065 | 4.6274 |
| 44 | 86 | Qwen2.5-Math | CommonsenseQA | 8 | 48.8329 | 45.0246 | 3.8084 |
| 45 | 87 | Qwen2.5-Math | CommonsenseQA | 8 | 48.9353 | 45.8436 | 3.0917 |
| 46 | 88 | Qwen2.5-Math | CommonsenseQA | 8 | 49.6724 | 45.4341 | 4.2383 |
| 47 | 89 | Qwen2.5-Math | CommonsenseQA | 8 | 49.7543 | 45.0041 | 4.7502 |
| 48 | 90 | Qwen2.5-Math | CommonsenseQA | 8 | 48.6486 | 44.6355 | 4.0131 |
| 49 | 91 | Qwen2.5-Math | CommonsenseQA | 8 | 49.14 | 45.086 | 4.0541 |
| 0 | 42 | Qwen2.5-Math | CommonsenseQA | 12 | 49.4949 | 45.0041 | 4.4909 |
| 1 | 43 | Qwen2.5-Math | CommonsenseQA | 12 | 49.1127 | 45.1815 | 3.9312 |
| 2 | 44 | Qwen2.5-Math | CommonsenseQA | 12 | 49.4949 | 45.0723 | 4.4226 |
| 3 | 45 | Qwen2.5-Math | CommonsenseQA | 12 | 49.5359 | 44.9768 | 4.5591 |
| 4 | 46 | Qwen2.5-Math | CommonsenseQA | 12 | 49.0035 | 45.4818 | 3.5217 |
| 5 | 47 | Qwen2.5-Math | CommonsenseQA | 12 | 49.454 | 45.4955 | 3.9585 |
| 6 | 48 | Qwen2.5-Math | CommonsenseQA | 12 | 49.3175 | 44.9768 | 4.3407 |
| 7 | 49 | Qwen2.5-Math | CommonsenseQA | 12 | 49.2629 | 45.0723 | 4.1906 |
| 8 | 50 | Qwen2.5-Math | CommonsenseQA | 12 | 49.1946 | 45.318 | 3.8766 |
| 9 | 51 | Qwen2.5-Math | CommonsenseQA | 12 | 49.6451 | 45.0041 | 4.641 |
| 10 | 52 | Qwen2.5-Math | CommonsenseQA | 12 | 49.5086 | 45.2361 | 4.2725 |
| 11 | 53 | Qwen2.5-Math | CommonsenseQA | 12 | 49.3175 | 44.9085 | 4.409 |
| 12 | 54 | Qwen2.5-Math | CommonsenseQA | 12 | 49.14 | 45.0723 | 4.0677 |
| 13 | 55 | Qwen2.5-Math | CommonsenseQA | 12 | 49.6178 | 45.3999 | 4.2179 |
| 14 | 56 | Qwen2.5-Math | CommonsenseQA | 12 | 49.3311 | 44.8539 | 4.4772 |
| 15 | 57 | Qwen2.5-Math | CommonsenseQA | 12 | 49.2492 | 44.9222 | 4.3271 |
| 16 | 58 | Qwen2.5-Math | CommonsenseQA | 12 | 48.7851 | 45.1269 | 3.6582 |
| 17 | 59 | Qwen2.5-Math | CommonsenseQA | 12 | 49.2083 | 44.9904 | 4.2179 |
| 18 | 60 | Qwen2.5-Math | CommonsenseQA | 12 | 49.454 | 45.2771 | 4.1769 |
| 19 | 61 | Qwen2.5-Math | CommonsenseQA | 12 | 49.686 | 44.8812 | 4.8048 |
| 20 | 62 | Qwen2.5-Math | CommonsenseQA | 12 | 49.3721 | 45.7139 | 3.6582 |
| 21 | 63 | Qwen2.5-Math | CommonsenseQA | 12 | 49.2356 | 44.772 | 4.4636 |
| 22 | 64 | Qwen2.5-Math | CommonsenseQA | 12 | 49.727 | 45.0041 | 4.7229 |
| 23 | 65 | Qwen2.5-Math | CommonsenseQA | 12 | 49.3721 | 45.2907 | 4.0814 |
| 24 | 66 | Qwen2.5-Math | CommonsenseQA | 12 | 49.2356 | 45.0041 | 4.2315 |
| 25 | 67 | Qwen2.5-Math | CommonsenseQA | 12 | 49.959 | 45.4818 | 4.4772 |
| 26 | 68 | Qwen2.5-Math | CommonsenseQA | 12 | 49.181 | 45.2907 | 3.8903 |
| 27 | 69 | Qwen2.5-Math | CommonsenseQA | 12 | 48.7851 | 44.8949 | 3.8903 |
| 28 | 70 | Qwen2.5-Math | CommonsenseQA | 12 | 49.0445 | 45.7821 | 3.2624 |
| 29 | 71 | Qwen2.5-Math | CommonsenseQA | 12 | 48.9216 | 45.2498 | 3.6719 |
| 30 | 72 | Qwen2.5-Math | CommonsenseQA | 12 | 49.2083 | 45.0314 | 4.1769 |
| 31 | 73 | Qwen2.5-Math | CommonsenseQA | 12 | 49.2356 | 45.6183 | 3.6173 |
| 32 | 74 | Qwen2.5-Math | CommonsenseQA | 12 | 49.6178 | 45.2907 | 4.3271 |
| 33 | 75 | Qwen2.5-Math | CommonsenseQA | 12 | 48.9489 | 45.3044 | 3.6446 |
| 34 | 76 | Qwen2.5-Math | CommonsenseQA | 12 | 49.3857 | 45.4682 | 3.9176 |
| 35 | 77 | Qwen2.5-Math | CommonsenseQA | 12 | 49.2902 | 45.5364 | 3.7538 |
| 36 | 78 | Qwen2.5-Math | CommonsenseQA | 12 | 49.3721 | 45.2361 | 4.136 |
| 37 | 79 | Qwen2.5-Math | CommonsenseQA | 12 | 49.3584 | 45.4955 | 3.863 |
| 38 | 80 | Qwen2.5-Math | CommonsenseQA | 12 | 49.5222 | 45.2907 | 4.2315 |
| 39 | 81 | Qwen2.5-Math | CommonsenseQA | 12 | 49.3448 | 45.1406 | 4.2042 |
| 40 | 82 | Qwen2.5-Math | CommonsenseQA | 12 | 49.9044 | 44.8403 | 5.0642 |
| 41 | 83 | Qwen2.5-Math | CommonsenseQA | 12 | 49.3994 | 45.6729 | 3.7265 |
| 42 | 84 | Qwen2.5-Math | CommonsenseQA | 12 | 49.686 | 45.0587 | 4.6274 |
| 43 | 85 | Qwen2.5-Math | CommonsenseQA | 12 | 49.3994 | 44.7038 | 4.6956 |
| 44 | 86 | Qwen2.5-Math | CommonsenseQA | 12 | 49.2902 | 45.0177 | 4.2725 |
| 45 | 87 | Qwen2.5-Math | CommonsenseQA | 12 | 49.3038 | 45.045 | 4.2588 |
| 46 | 88 | Qwen2.5-Math | CommonsenseQA | 12 | 49.3448 | 45.0587 | 4.2861 |
| 47 | 89 | Qwen2.5-Math | CommonsenseQA | 12 | 49.2219 | 45.7958 | 3.4262 |
| 48 | 90 | Qwen2.5-Math | CommonsenseQA | 12 | 49.454 | 45.4682 | 3.9858 |
| 49 | 91 | Qwen2.5-Math | CommonsenseQA | 12 | 49.5359 | 44.9631 | 4.5728 |
| 0 | 42 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 1 | 43 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 2 | 44 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 3 | 45 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 4 | 46 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 5 | 47 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 6 | 48 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 7 | 49 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 8 | 50 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 9 | 51 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 10 | 52 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 11 | 53 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 12 | 54 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 13 | 55 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 14 | 56 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 15 | 57 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 16 | 58 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 17 | 59 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 18 | 60 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 19 | 61 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 20 | 62 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 21 | 63 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 22 | 64 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 23 | 65 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 24 | 66 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 25 | 67 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 26 | 68 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 27 | 69 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 28 | 70 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 29 | 71 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 30 | 72 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 31 | 73 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 32 | 74 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 33 | 75 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 34 | 76 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 35 | 77 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 36 | 78 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 37 | 79 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 38 | 80 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 39 | 81 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 40 | 82 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 41 | 83 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 42 | 84 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 43 | 85 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 44 | 86 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 45 | 87 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 46 | 88 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 47 | 89 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 48 | 90 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 49 | 91 | Qwen2.5-Math | CommonsenseQA | 16 | 49.4676 | 45.086 | 4.3817 |
| 0 | 42 | Qwen2.5-Math | GPQA | 4 | 32.3661 | 27.2321 | 5.1339 |
| 1 | 43 | Qwen2.5-Math | GPQA | 4 | 32.4777 | 28.2366 | 4.2411 |
| 2 | 44 | Qwen2.5-Math | GPQA | 4 | 30.8036 | 28.3482 | 2.4554 |
| 3 | 45 | Qwen2.5-Math | GPQA | 4 | 33.1473 | 27.7902 | 5.3571 |
| 4 | 46 | Qwen2.5-Math | GPQA | 4 | 30.3571 | 27.7902 | 2.567 |
| 5 | 47 | Qwen2.5-Math | GPQA | 4 | 30.1339 | 27.567 | 2.567 |
| 6 | 48 | Qwen2.5-Math | GPQA | 4 | 28.683 | 28.9062 | -0.2232 |
| 7 | 49 | Qwen2.5-Math | GPQA | 4 | 30.0223 | 25.8929 | 4.1295 |
| 8 | 50 | Qwen2.5-Math | GPQA | 4 | 31.5848 | 30.0223 | 1.5625 |
| 9 | 51 | Qwen2.5-Math | GPQA | 4 | 30.5804 | 28.7946 | 1.7857 |
| 10 | 52 | Qwen2.5-Math | GPQA | 4 | 32.4777 | 27.4554 | 5.0223 |
| 11 | 53 | Qwen2.5-Math | GPQA | 4 | 31.4732 | 28.3482 | 3.125 |
| 12 | 54 | Qwen2.5-Math | GPQA | 4 | 29.4643 | 27.3438 | 2.1205 |
| 13 | 55 | Qwen2.5-Math | GPQA | 4 | 31.4732 | 28.4598 | 3.0134 |
| 14 | 56 | Qwen2.5-Math | GPQA | 4 | 29.4643 | 25.7812 | 3.683 |
| 15 | 57 | Qwen2.5-Math | GPQA | 4 | 31.6964 | 27.0089 | 4.6875 |
| 16 | 58 | Qwen2.5-Math | GPQA | 4 | 33.4821 | 29.3527 | 4.1295 |
| 17 | 59 | Qwen2.5-Math | GPQA | 4 | 34.1518 | 27.2321 | 6.9196 |
| 18 | 60 | Qwen2.5-Math | GPQA | 4 | 32.8125 | 28.5714 | 4.2411 |
| 19 | 61 | Qwen2.5-Math | GPQA | 4 | 31.0268 | 27.567 | 3.4598 |
| 20 | 62 | Qwen2.5-Math | GPQA | 4 | 31.3616 | 28.7946 | 2.567 |
| 21 | 63 | Qwen2.5-Math | GPQA | 4 | 31.9196 | 31.1384 | 0.7812 |
| 22 | 64 | Qwen2.5-Math | GPQA | 4 | 31.808 | 29.4643 | 2.3438 |
| 23 | 65 | Qwen2.5-Math | GPQA | 4 | 30.2455 | 27.567 | 2.6786 |
| 24 | 66 | Qwen2.5-Math | GPQA | 4 | 31.5848 | 29.1295 | 2.4554 |
| 25 | 67 | Qwen2.5-Math | GPQA | 4 | 30.8036 | 29.2411 | 1.5625 |
| 26 | 68 | Qwen2.5-Math | GPQA | 4 | 33.2589 | 27.7902 | 5.4688 |
| 27 | 69 | Qwen2.5-Math | GPQA | 4 | 29.7991 | 28.125 | 1.6741 |
| 28 | 70 | Qwen2.5-Math | GPQA | 4 | 32.2545 | 29.0179 | 3.2366 |
| 29 | 71 | Qwen2.5-Math | GPQA | 4 | 29.9107 | 31.5848 | -1.6741 |
| 30 | 72 | Qwen2.5-Math | GPQA | 4 | 31.25 | 30.1339 | 1.1161 |
| 31 | 73 | Qwen2.5-Math | GPQA | 4 | 31.5848 | 27.9018 | 3.683 |
| 32 | 74 | Qwen2.5-Math | GPQA | 4 | 30.0223 | 28.125 | 1.8973 |
| 33 | 75 | Qwen2.5-Math | GPQA | 4 | 30.2455 | 28.125 | 2.1205 |
| 34 | 76 | Qwen2.5-Math | GPQA | 4 | 32.4777 | 28.2366 | 4.2411 |
| 35 | 77 | Qwen2.5-Math | GPQA | 4 | 32.7009 | 30.4688 | 2.2321 |
| 36 | 78 | Qwen2.5-Math | GPQA | 4 | 31.808 | 28.125 | 3.683 |
| 37 | 79 | Qwen2.5-Math | GPQA | 4 | 31.0268 | 30.2455 | 0.7812 |
| 38 | 80 | Qwen2.5-Math | GPQA | 4 | 29.2411 | 28.0134 | 1.2277 |
| 39 | 81 | Qwen2.5-Math | GPQA | 4 | 32.1429 | 27.9018 | 4.2411 |
| 40 | 82 | Qwen2.5-Math | GPQA | 4 | 33.9286 | 28.5714 | 5.3571 |
| 41 | 83 | Qwen2.5-Math | GPQA | 4 | 30.4688 | 26.3393 | 4.1295 |
| 42 | 84 | Qwen2.5-Math | GPQA | 4 | 33.4821 | 27.7902 | 5.692 |
| 43 | 85 | Qwen2.5-Math | GPQA | 4 | 32.2545 | 27.0089 | 5.2455 |
| 44 | 86 | Qwen2.5-Math | GPQA | 4 | 30.1339 | 29.3527 | 0.7812 |
| 45 | 87 | Qwen2.5-Math | GPQA | 4 | 32.8125 | 27.567 | 5.2455 |
| 46 | 88 | Qwen2.5-Math | GPQA | 4 | 30.4688 | 28.125 | 2.3438 |
| 47 | 89 | Qwen2.5-Math | GPQA | 4 | 30.2455 | 28.3482 | 1.8973 |
| 48 | 90 | Qwen2.5-Math | GPQA | 4 | 29.2411 | 25.558 | 3.683 |
| 49 | 91 | Qwen2.5-Math | GPQA | 4 | 32.2545 | 27.1205 | 5.1339 |
| 0 | 42 | Qwen2.5-Math | GPQA | 8 | 31.8638 | 27.9018 | 3.9621 |
| 1 | 43 | Qwen2.5-Math | GPQA | 8 | 32.1429 | 27.9576 | 4.1853 |
| 2 | 44 | Qwen2.5-Math | GPQA | 8 | 31.529 | 28.5156 | 3.0134 |
| 3 | 45 | Qwen2.5-Math | GPQA | 8 | 31.1384 | 28.404 | 2.7344 |
| 4 | 46 | Qwen2.5-Math | GPQA | 8 | 31.1942 | 28.7946 | 2.3996 |
| 5 | 47 | Qwen2.5-Math | GPQA | 8 | 31.0826 | 28.9062 | 2.1763 |
| 6 | 48 | Qwen2.5-Math | GPQA | 8 | 30.9152 | 27.7902 | 3.125 |
| 7 | 49 | Qwen2.5-Math | GPQA | 8 | 29.9107 | 28.125 | 1.7857 |
| 8 | 50 | Qwen2.5-Math | GPQA | 8 | 31.8638 | 27.9576 | 3.9062 |
| 9 | 51 | Qwen2.5-Math | GPQA | 8 | 31.6964 | 28.7388 | 2.9576 |
| 10 | 52 | Qwen2.5-Math | GPQA | 8 | 30.7478 | 27.567 | 3.1808 |
| 11 | 53 | Qwen2.5-Math | GPQA | 8 | 32.4777 | 28.2924 | 4.1853 |
| 12 | 54 | Qwen2.5-Math | GPQA | 8 | 30.1897 | 28.1808 | 2.0089 |
| 13 | 55 | Qwen2.5-Math | GPQA | 8 | 31.529 | 28.0134 | 3.5156 |
| 14 | 56 | Qwen2.5-Math | GPQA | 8 | 30.4688 | 27.567 | 2.9018 |
| 15 | 57 | Qwen2.5-Math | GPQA | 8 | 31.8638 | 28.5156 | 3.3482 |
| 16 | 58 | Qwen2.5-Math | GPQA | 8 | 31.529 | 27.846 | 3.683 |
| 17 | 59 | Qwen2.5-Math | GPQA | 8 | 32.3103 | 28.5156 | 3.7946 |
| 18 | 60 | Qwen2.5-Math | GPQA | 8 | 32.5335 | 27.9576 | 4.5759 |
| 19 | 61 | Qwen2.5-Math | GPQA | 8 | 31.5848 | 27.6228 | 3.9621 |
| 20 | 62 | Qwen2.5-Math | GPQA | 8 | 31.9196 | 29.2969 | 2.6228 |
| 21 | 63 | Qwen2.5-Math | GPQA | 8 | 31.9196 | 29.2969 | 2.6228 |
| 22 | 64 | Qwen2.5-Math | GPQA | 8 | 31.3616 | 29.6875 | 1.6741 |
| 23 | 65 | Qwen2.5-Math | GPQA | 8 | 29.7433 | 28.5714 | 1.1719 |
| 24 | 66 | Qwen2.5-Math | GPQA | 8 | 31.7522 | 28.3482 | 3.404 |
| 25 | 67 | Qwen2.5-Math | GPQA | 8 | 30.9152 | 28.1808 | 2.7344 |
| 26 | 68 | Qwen2.5-Math | GPQA | 8 | 32.6451 | 27.7344 | 4.9107 |
| 27 | 69 | Qwen2.5-Math | GPQA | 8 | 31.529 | 28.125 | 3.404 |
| 28 | 70 | Qwen2.5-Math | GPQA | 8 | 31.3616 | 28.125 | 3.2366 |
| 29 | 71 | Qwen2.5-Math | GPQA | 8 | 31.1384 | 29.5759 | 1.5625 |
| 30 | 72 | Qwen2.5-Math | GPQA | 8 | 31.4732 | 28.9621 | 2.5112 |
| 31 | 73 | Qwen2.5-Math | GPQA | 8 | 31.1384 | 27.9576 | 3.1808 |
| 32 | 74 | Qwen2.5-Math | GPQA | 8 | 31.9754 | 29.2411 | 2.7344 |
| 33 | 75 | Qwen2.5-Math | GPQA | 8 | 31.3616 | 28.3482 | 3.0134 |
| 34 | 76 | Qwen2.5-Math | GPQA | 8 | 32.5335 | 27.567 | 4.9665 |
| 35 | 77 | Qwen2.5-Math | GPQA | 8 | 33.0357 | 28.9062 | 4.1295 |
| 36 | 78 | Qwen2.5-Math | GPQA | 8 | 31.0268 | 27.2321 | 3.7946 |
| 37 | 79 | Qwen2.5-Math | GPQA | 8 | 31.6964 | 29.1853 | 2.5112 |
| 38 | 80 | Qwen2.5-Math | GPQA | 8 | 29.9107 | 28.404 | 1.5067 |
| 39 | 81 | Qwen2.5-Math | GPQA | 8 | 31.808 | 26.8415 | 4.9665 |
| 40 | 82 | Qwen2.5-Math | GPQA | 8 | 33.0357 | 29.7991 | 3.2366 |
| 41 | 83 | Qwen2.5-Math | GPQA | 8 | 32.3661 | 27.2321 | 5.1339 |
| 42 | 84 | Qwen2.5-Math | GPQA | 8 | 31.529 | 28.0134 | 3.5156 |
| 43 | 85 | Qwen2.5-Math | GPQA | 8 | 31.25 | 28.8504 | 2.3996 |
| 44 | 86 | Qwen2.5-Math | GPQA | 8 | 30.4129 | 29.9665 | 0.4464 |
| 45 | 87 | Qwen2.5-Math | GPQA | 8 | 30.3013 | 27.846 | 2.4554 |
| 46 | 88 | Qwen2.5-Math | GPQA | 8 | 31.25 | 28.2924 | 2.9576 |
| 47 | 89 | Qwen2.5-Math | GPQA | 8 | 30.5246 | 28.404 | 2.1205 |
| 48 | 90 | Qwen2.5-Math | GPQA | 8 | 31.25 | 26.8973 | 4.3527 |
| 49 | 91 | Qwen2.5-Math | GPQA | 8 | 32.3661 | 27.2879 | 5.0781 |
| 0 | 42 | Qwen2.5-Math | GPQA | 12 | 31.6964 | 28.4226 | 3.2738 |
| 1 | 43 | Qwen2.5-Math | GPQA | 12 | 31.7708 | 28.3482 | 3.4226 |
| 2 | 44 | Qwen2.5-Math | GPQA | 12 | 30.692 | 28.2366 | 2.4554 |
| 3 | 45 | Qwen2.5-Math | GPQA | 12 | 31.3244 | 28.7946 | 2.5298 |
| 4 | 46 | Qwen2.5-Math | GPQA | 12 | 31.7708 | 28.5342 | 3.2366 |
| 5 | 47 | Qwen2.5-Math | GPQA | 12 | 31.7336 | 28.6458 | 3.0878 |
| 6 | 48 | Qwen2.5-Math | GPQA | 12 | 31.1012 | 28.4226 | 2.6786 |
| 7 | 49 | Qwen2.5-Math | GPQA | 12 | 31.25 | 28.7202 | 2.5298 |
| 8 | 50 | Qwen2.5-Math | GPQA | 12 | 30.9896 | 28.5714 | 2.4182 |
| 9 | 51 | Qwen2.5-Math | GPQA | 12 | 31.1756 | 28.5342 | 2.6414 |
| 10 | 52 | Qwen2.5-Math | GPQA | 12 | 31.064 | 28.5714 | 2.4926 |
| 11 | 53 | Qwen2.5-Math | GPQA | 12 | 31.5848 | 28.3854 | 3.1994 |
| 12 | 54 | Qwen2.5-Math | GPQA | 12 | 31.0268 | 28.7946 | 2.2321 |
| 13 | 55 | Qwen2.5-Math | GPQA | 12 | 31.5104 | 28.869 | 2.6414 |
| 14 | 56 | Qwen2.5-Math | GPQA | 12 | 31.1012 | 27.6786 | 3.4226 |
| 15 | 57 | Qwen2.5-Math | GPQA | 12 | 31.622 | 28.5714 | 3.0506 |
| 16 | 58 | Qwen2.5-Math | GPQA | 12 | 31.622 | 28.125 | 3.497 |
| 17 | 59 | Qwen2.5-Math | GPQA | 12 | 32.1429 | 28.4598 | 3.683 |
| 18 | 60 | Qwen2.5-Math | GPQA | 12 | 31.9568 | 28.1622 | 3.7946 |
| 19 | 61 | Qwen2.5-Math | GPQA | 12 | 31.5848 | 28.1622 | 3.4226 |
| 20 | 62 | Qwen2.5-Math | GPQA | 12 | 31.8824 | 28.5714 | 3.311 |
| 21 | 63 | Qwen2.5-Math | GPQA | 12 | 32.0685 | 28.8318 | 3.2366 |
| 22 | 64 | Qwen2.5-Math | GPQA | 12 | 31.808 | 28.7946 | 3.0134 |
| 23 | 65 | Qwen2.5-Math | GPQA | 12 | 30.8036 | 28.311 | 2.4926 |
| 24 | 66 | Qwen2.5-Math | GPQA | 12 | 31.1384 | 28.1994 | 2.939 |
| 25 | 67 | Qwen2.5-Math | GPQA | 12 | 31.3988 | 28.6458 | 2.753 |
| 26 | 68 | Qwen2.5-Math | GPQA | 12 | 31.994 | 28.7574 | 3.2366 |
| 27 | 69 | Qwen2.5-Math | GPQA | 12 | 31.25 | 28.2366 | 3.0134 |
| 28 | 70 | Qwen2.5-Math | GPQA | 12 | 31.1012 | 28.9062 | 2.1949 |
| 29 | 71 | Qwen2.5-Math | GPQA | 12 | 32.1429 | 28.0506 | 4.0923 |
| 30 | 72 | Qwen2.5-Math | GPQA | 12 | 31.9568 | 28.1994 | 3.7574 |
| 31 | 73 | Qwen2.5-Math | GPQA | 12 | 30.878 | 28.6086 | 2.2693 |
| 32 | 74 | Qwen2.5-Math | GPQA | 12 | 31.622 | 28.6086 | 3.0134 |
| 33 | 75 | Qwen2.5-Math | GPQA | 12 | 31.4732 | 28.2366 | 3.2366 |
| 34 | 76 | Qwen2.5-Math | GPQA | 12 | 31.5848 | 28.7202 | 2.8646 |
| 35 | 77 | Qwen2.5-Math | GPQA | 12 | 32.4033 | 28.7202 | 3.683 |
| 36 | 78 | Qwen2.5-Math | GPQA | 12 | 31.6964 | 28.0134 | 3.683 |
| 37 | 79 | Qwen2.5-Math | GPQA | 12 | 31.7336 | 29.5015 | 2.2321 |
| 38 | 80 | Qwen2.5-Math | GPQA | 12 | 30.9896 | 28.0878 | 2.9018 |
| 39 | 81 | Qwen2.5-Math | GPQA | 12 | 31.6592 | 28.0134 | 3.6458 |
| 40 | 82 | Qwen2.5-Math | GPQA | 12 | 32.4405 | 28.7202 | 3.7202 |
| 41 | 83 | Qwen2.5-Math | GPQA | 12 | 31.7336 | 28.1622 | 3.5714 |
| 42 | 84 | Qwen2.5-Math | GPQA | 12 | 31.8824 | 27.9018 | 3.9807 |
| 43 | 85 | Qwen2.5-Math | GPQA | 12 | 31.8824 | 28.4226 | 3.4598 |
| 44 | 86 | Qwen2.5-Math | GPQA | 12 | 31.2872 | 29.2411 | 2.0461 |
| 45 | 87 | Qwen2.5-Math | GPQA | 12 | 30.7292 | 27.7902 | 2.939 |
| 46 | 88 | Qwen2.5-Math | GPQA | 12 | 31.622 | 28.3482 | 3.2738 |
| 47 | 89 | Qwen2.5-Math | GPQA | 12 | 31.6964 | 28.7946 | 2.9018 |
| 48 | 90 | Qwen2.5-Math | GPQA | 12 | 31.808 | 27.8274 | 3.9807 |
| 49 | 91 | Qwen2.5-Math | GPQA | 12 | 32.2917 | 28.0506 | 4.2411 |
| 0 | 42 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 1 | 43 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 2 | 44 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 3 | 45 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 4 | 46 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 5 | 47 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 6 | 48 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 7 | 49 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 8 | 50 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 9 | 51 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 10 | 52 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 11 | 53 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 12 | 54 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 13 | 55 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 14 | 56 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 15 | 57 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 16 | 58 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 17 | 59 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 18 | 60 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 19 | 61 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 20 | 62 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 21 | 63 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 22 | 64 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 23 | 65 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 24 | 66 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 25 | 67 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 26 | 68 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 27 | 69 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 28 | 70 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 29 | 71 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 30 | 72 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 31 | 73 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 32 | 74 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 33 | 75 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 34 | 76 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 35 | 77 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 36 | 78 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 37 | 79 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 38 | 80 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 39 | 81 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 40 | 82 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 41 | 83 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 42 | 84 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 43 | 85 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 44 | 86 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 45 | 87 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 46 | 88 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 47 | 89 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 48 | 90 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 49 | 91 | Qwen2.5-Math | GPQA | 16 | 31.5569 | 28.4319 | 3.125 |
| 0 | 42 | Qwen2.5-Math | GSM8K | 4 | 67.7407 | 73.6922 | -5.9515 |
| 1 | 43 | Qwen2.5-Math | GSM8K | 4 | 68.3472 | 74.2608 | -5.9136 |
| 2 | 44 | Qwen2.5-Math | GSM8K | 4 | 67.627 | 75.0569 | -7.4299 |
| 3 | 45 | Qwen2.5-Math | GSM8K | 4 | 70.0152 | 73.6164 | -3.6012 |
| 4 | 46 | Qwen2.5-Math | GSM8K | 4 | 68.2714 | 74.2608 | -5.9894 |
| 5 | 47 | Qwen2.5-Math | GSM8K | 4 | 68.2714 | 73.9575 | -5.6861 |
| 6 | 48 | Qwen2.5-Math | GSM8K | 4 | 67.9682 | 73.9955 | -6.0273 |
| 7 | 49 | Qwen2.5-Math | GSM8K | 4 | 67.9303 | 74.2229 | -6.2926 |
| 8 | 50 | Qwen2.5-Math | GSM8K | 4 | 68.8021 | 75.5118 | -6.7096 |
| 9 | 51 | Qwen2.5-Math | GSM8K | 4 | 69.5603 | 73.6922 | -4.1319 |
| 10 | 52 | Qwen2.5-Math | GSM8K | 4 | 69.4466 | 73.5027 | -4.0561 |
| 11 | 53 | Qwen2.5-Math | GSM8K | 4 | 68.7642 | 74.4503 | -5.6861 |
| 12 | 54 | Qwen2.5-Math | GSM8K | 4 | 67.21 | 74.4124 | -7.2024 |
| 13 | 55 | Qwen2.5-Math | GSM8K | 4 | 69.2191 | 75.5876 | -6.3685 |
| 14 | 56 | Qwen2.5-Math | GSM8K | 4 | 68.044 | 74.7536 | -6.7096 |
| 15 | 57 | Qwen2.5-Math | GSM8K | 4 | 67.9303 | 74.2987 | -6.3685 |
| 16 | 58 | Qwen2.5-Math | GSM8K | 4 | 66.9447 | 75.0569 | -8.1122 |
| 17 | 59 | Qwen2.5-Math | GSM8K | 4 | 66.5277 | 75.2085 | -8.6808 |
| 18 | 60 | Qwen2.5-Math | GSM8K | 4 | 68.1577 | 73.9575 | -5.7998 |
| 19 | 61 | Qwen2.5-Math | GSM8K | 4 | 67.5512 | 73.768 | -6.2168 |
| 20 | 62 | Qwen2.5-Math | GSM8K | 4 | 68.6126 | 74.1092 | -5.4966 |
| 21 | 63 | Qwen2.5-Math | GSM8K | 4 | 68.0061 | 74.9431 | -6.9371 |
| 22 | 64 | Qwen2.5-Math | GSM8K | 4 | 68.3851 | 74.5641 | -6.1789 |
| 23 | 65 | Qwen2.5-Math | GSM8K | 4 | 67.0963 | 74.3366 | -7.2403 |
| 24 | 66 | Qwen2.5-Math | GSM8K | 4 | 69.4086 | 73.8438 | -4.4352 |
| 25 | 67 | Qwen2.5-Math | GSM8K | 4 | 68.0819 | 75.0569 | -6.975 |
| 26 | 68 | Qwen2.5-Math | GSM8K | 4 | 69.0296 | 73.768 | -4.7384 |
| 27 | 69 | Qwen2.5-Math | GSM8K | 4 | 67.5512 | 74.1092 | -6.558 |
| 28 | 70 | Qwen2.5-Math | GSM8K | 4 | 67.4754 | 75.2843 | -7.8089 |
| 29 | 71 | Qwen2.5-Math | GSM8K | 4 | 69.9014 | 75.2843 | -5.3829 |
| 30 | 72 | Qwen2.5-Math | GSM8K | 4 | 68.1577 | 74.0713 | -5.9136 |
| 31 | 73 | Qwen2.5-Math | GSM8K | 4 | 68.044 | 75.019 | -6.975 |
| 32 | 74 | Qwen2.5-Math | GSM8K | 4 | 67.21 | 74.2987 | -7.0887 |
| 33 | 75 | Qwen2.5-Math | GSM8K | 4 | 68.2714 | 74.602 | -6.3306 |
| 34 | 76 | Qwen2.5-Math | GSM8K | 4 | 68.423 | 74.5262 | -6.1031 |
| 35 | 77 | Qwen2.5-Math | GSM8K | 4 | 68.84 | 73.5785 | -4.7384 |
| 36 | 78 | Qwen2.5-Math | GSM8K | 4 | 69.0296 | 74.4503 | -5.4208 |
| 37 | 79 | Qwen2.5-Math | GSM8K | 4 | 68.3472 | 74.4882 | -6.141 |
| 38 | 80 | Qwen2.5-Math | GSM8K | 4 | 68.423 | 74.8673 | -6.4443 |
| 39 | 81 | Qwen2.5-Math | GSM8K | 4 | 69.1054 | 74.5262 | -5.4208 |
| 40 | 82 | Qwen2.5-Math | GSM8K | 4 | 67.0584 | 73.768 | -6.7096 |
| 41 | 83 | Qwen2.5-Math | GSM8K | 4 | 68.1198 | 74.3366 | -6.2168 |
| 42 | 84 | Qwen2.5-Math | GSM8K | 4 | 67.2858 | 73.6922 | -6.4064 |
| 43 | 85 | Qwen2.5-Math | GSM8K | 4 | 67.8544 | 73.4647 | -5.6103 |
| 44 | 86 | Qwen2.5-Math | GSM8K | 4 | 68.9538 | 73.6164 | -4.6626 |
| 45 | 87 | Qwen2.5-Math | GSM8K | 4 | 66.6414 | 74.7915 | -8.1501 |
| 46 | 88 | Qwen2.5-Math | GSM8K | 4 | 68.6126 | 74.7536 | -6.141 |
| 47 | 89 | Qwen2.5-Math | GSM8K | 4 | 68.1956 | 72.4412 | -4.2456 |
| 48 | 90 | Qwen2.5-Math | GSM8K | 4 | 68.2714 | 74.9052 | -6.6338 |
| 49 | 91 | Qwen2.5-Math | GSM8K | 4 | 68.0819 | 74.6778 | -6.5959 |
| 0 | 42 | Qwen2.5-Math | GSM8K | 8 | 68.2904 | 74.8294 | -6.539 |
| 1 | 43 | Qwen2.5-Math | GSM8K | 8 | 68.2525 | 74.7536 | -6.5011 |
| 2 | 44 | Qwen2.5-Math | GSM8K | 8 | 68.2525 | 74.2798 | -6.0273 |
| 3 | 45 | Qwen2.5-Math | GSM8K | 8 | 68.3662 | 74.185 | -5.8188 |
| 4 | 46 | Qwen2.5-Math | GSM8K | 8 | 67.0584 | 74.8673 | -7.8089 |
| 5 | 47 | Qwen2.5-Math | GSM8K | 8 | 67.4375 | 74.4882 | -7.0508 |
| 6 | 48 | Qwen2.5-Math | GSM8K | 8 | 67.5701 | 74.5451 | -6.975 |
| 7 | 49 | Qwen2.5-Math | GSM8K | 8 | 67.3237 | 74.7157 | -7.392 |
| 8 | 50 | Qwen2.5-Math | GSM8K | 8 | 68.0629 | 74.0902 | -6.0273 |
| 9 | 51 | Qwen2.5-Math | GSM8K | 8 | 68.4989 | 74.3177 | -5.8188 |
| 10 | 52 | Qwen2.5-Math | GSM8K | 8 | 68.1387 | 74.7915 | -6.6528 |
| 11 | 53 | Qwen2.5-Math | GSM8K | 8 | 67.9492 | 74.7726 | -6.8234 |
| 12 | 54 | Qwen2.5-Math | GSM8K | 8 | 68.1387 | 74.7726 | -6.6338 |
| 13 | 55 | Qwen2.5-Math | GSM8K | 8 | 68.4799 | 74.6967 | -6.2168 |
| 14 | 56 | Qwen2.5-Math | GSM8K | 8 | 67.608 | 74.3745 | -6.7665 |
| 15 | 57 | Qwen2.5-Math | GSM8K | 8 | 67.4943 | 74.7346 | -7.2403 |
| 16 | 58 | Qwen2.5-Math | GSM8K | 8 | 68.2525 | 74.166 | -5.9136 |
| 17 | 59 | Qwen2.5-Math | GSM8K | 8 | 67.6839 | 74.1281 | -6.4443 |
| 18 | 60 | Qwen2.5-Math | GSM8K | 8 | 67.608 | 74.185 | -6.577 |
| 19 | 61 | Qwen2.5-Math | GSM8K | 8 | 67.3995 | 74.6967 | -7.2972 |
| 20 | 62 | Qwen2.5-Math | GSM8K | 8 | 68.1956 | 74.3556 | -6.16 |
| 21 | 63 | Qwen2.5-Math | GSM8K | 8 | 68.044 | 74.7726 | -6.7286 |
| 22 | 64 | Qwen2.5-Math | GSM8K | 8 | 68.423 | 74.5262 | -6.1031 |
| 23 | 65 | Qwen2.5-Math | GSM8K | 8 | 68.0819 | 74.7536 | -6.6717 |
| 24 | 66 | Qwen2.5-Math | GSM8K | 8 | 69.3707 | 74.4124 | -5.0417 |
| 25 | 67 | Qwen2.5-Math | GSM8K | 8 | 68.8021 | 75.4549 | -6.6528 |
| 26 | 68 | Qwen2.5-Math | GSM8K | 8 | 68.025 | 74.7346 | -6.7096 |
| 27 | 69 | Qwen2.5-Math | GSM8K | 8 | 67.5512 | 73.9386 | -6.3874 |
| 28 | 70 | Qwen2.5-Math | GSM8K | 8 | 67.9303 | 74.9052 | -6.975 |
| 29 | 71 | Qwen2.5-Math | GSM8K | 8 | 68.442 | 74.6778 | -6.2358 |
| 30 | 72 | Qwen2.5-Math | GSM8K | 8 | 67.9303 | 74.6967 | -6.7665 |
| 31 | 73 | Qwen2.5-Math | GSM8K | 8 | 68.1008 | 74.8484 | -6.7475 |
| 32 | 74 | Qwen2.5-Math | GSM8K | 8 | 67.7407 | 74.166 | -6.4253 |
| 33 | 75 | Qwen2.5-Math | GSM8K | 8 | 67.8544 | 73.8628 | -6.0083 |
| 34 | 76 | Qwen2.5-Math | GSM8K | 8 | 67.3995 | 74.8863 | -7.4867 |
| 35 | 77 | Qwen2.5-Math | GSM8K | 8 | 67.4943 | 74.8294 | -7.3351 |
| 36 | 78 | Qwen2.5-Math | GSM8K | 8 | 68.3283 | 75.0948 | -6.7665 |
| 37 | 79 | Qwen2.5-Math | GSM8K | 8 | 68.2904 | 74.3935 | -6.1031 |
| 38 | 80 | Qwen2.5-Math | GSM8K | 8 | 67.6839 | 74.602 | -6.9181 |
| 39 | 81 | Qwen2.5-Math | GSM8K | 8 | 67.9682 | 74.0334 | -6.0652 |
| 40 | 82 | Qwen2.5-Math | GSM8K | 8 | 67.9871 | 74.0523 | -6.0652 |
| 41 | 83 | Qwen2.5-Math | GSM8K | 8 | 67.9113 | 74.9242 | -7.0129 |
| 42 | 84 | Qwen2.5-Math | GSM8K | 8 | 67.4185 | 74.4314 | -7.0129 |
| 43 | 85 | Qwen2.5-Math | GSM8K | 8 | 68.1956 | 73.9007 | -5.7051 |
| 44 | 86 | Qwen2.5-Math | GSM8K | 8 | 67.5701 | 74.3556 | -6.7854 |
| 45 | 87 | Qwen2.5-Math | GSM8K | 8 | 67.2858 | 75.3222 | -8.0364 |
| 46 | 88 | Qwen2.5-Math | GSM8K | 8 | 67.4375 | 75.1327 | -7.6952 |
| 47 | 89 | Qwen2.5-Math | GSM8K | 8 | 67.4943 | 74.2608 | -6.7665 |
| 48 | 90 | Qwen2.5-Math | GSM8K | 8 | 68.0629 | 75.4738 | -7.4109 |
| 49 | 91 | Qwen2.5-Math | GSM8K | 8 | 68.044 | 73.9196 | -5.8757 |
| 0 | 42 | Qwen2.5-Math | GSM8K | 12 | 67.627 | 74.4503 | -6.8234 |
| 1 | 43 | Qwen2.5-Math | GSM8K | 12 | 68.2588 | 74.4882 | -6.2295 |
| 2 | 44 | Qwen2.5-Math | GSM8K | 12 | 67.7154 | 74.6652 | -6.9497 |
| 3 | 45 | Qwen2.5-Math | GSM8K | 12 | 67.7154 | 74.2482 | -6.5327 |
| 4 | 46 | Qwen2.5-Math | GSM8K | 12 | 67.2732 | 74.4882 | -7.2151 |
| 5 | 47 | Qwen2.5-Math | GSM8K | 12 | 67.5891 | 74.4251 | -6.836 |
| 6 | 48 | Qwen2.5-Math | GSM8K | 12 | 67.905 | 74.7031 | -6.7981 |
| 7 | 49 | Qwen2.5-Math | GSM8K | 12 | 67.3616 | 74.5262 | -7.1645 |
| 8 | 50 | Qwen2.5-Math | GSM8K | 12 | 68.2082 | 74.2861 | -6.0778 |
| 9 | 51 | Qwen2.5-Math | GSM8K | 12 | 67.9808 | 74.8926 | -6.9118 |
| 10 | 52 | Qwen2.5-Math | GSM8K | 12 | 67.7028 | 74.9052 | -7.2024 |
| 11 | 53 | Qwen2.5-Math | GSM8K | 12 | 67.5764 | 74.6778 | -7.1013 |
| 12 | 54 | Qwen2.5-Math | GSM8K | 12 | 67.905 | 74.7536 | -6.8486 |
| 13 | 55 | Qwen2.5-Math | GSM8K | 12 | 67.3111 | 74.8926 | -7.5815 |
| 14 | 56 | Qwen2.5-Math | GSM8K | 12 | 67.6144 | 74.8421 | -7.2277 |
| 15 | 57 | Qwen2.5-Math | GSM8K | 12 | 67.7786 | 74.7031 | -6.9244 |
| 16 | 58 | Qwen2.5-Math | GSM8K | 12 | 67.8418 | 74.602 | -6.7602 |
| 17 | 59 | Qwen2.5-Math | GSM8K | 12 | 67.905 | 74.5009 | -6.5959 |
| 18 | 60 | Qwen2.5-Math | GSM8K | 12 | 67.5133 | 74.3493 | -6.836 |
| 19 | 61 | Qwen2.5-Math | GSM8K | 12 | 67.8292 | 74.5009 | -6.6717 |
| 20 | 62 | Qwen2.5-Math | GSM8K | 12 | 67.905 | 74.3493 | -6.4443 |
| 21 | 63 | Qwen2.5-Math | GSM8K | 12 | 67.7533 | 74.5767 | -6.8234 |
| 22 | 64 | Qwen2.5-Math | GSM8K | 12 | 68.0945 | 74.6778 | -6.5833 |
| 23 | 65 | Qwen2.5-Math | GSM8K | 12 | 67.7281 | 74.7915 | -7.0634 |
| 24 | 66 | Qwen2.5-Math | GSM8K | 12 | 68.461 | 74.6778 | -6.2168 |
| 25 | 67 | Qwen2.5-Math | GSM8K | 12 | 67.8292 | 75.019 | -7.1898 |
| 26 | 68 | Qwen2.5-Math | GSM8K | 12 | 67.3111 | 74.9431 | -7.632 |
| 27 | 69 | Qwen2.5-Math | GSM8K | 12 | 67.7407 | 74.1597 | -6.419 |
| 28 | 70 | Qwen2.5-Math | GSM8K | 12 | 67.7154 | 74.5388 | -6.8234 |
| 29 | 71 | Qwen2.5-Math | GSM8K | 12 | 67.6017 | 74.6525 | -7.0508 |
| 30 | 72 | Qwen2.5-Math | GSM8K | 12 | 67.8418 | 74.9179 | -7.0761 |
| 31 | 73 | Qwen2.5-Math | GSM8K | 12 | 68.0692 | 74.3745 | -6.3053 |
| 32 | 74 | Qwen2.5-Math | GSM8K | 12 | 67.5133 | 74.7031 | -7.1898 |
| 33 | 75 | Qwen2.5-Math | GSM8K | 12 | 67.8292 | 74.6399 | -6.8107 |
| 34 | 76 | Qwen2.5-Math | GSM8K | 12 | 67.6017 | 74.8421 | -7.2403 |
| 35 | 77 | Qwen2.5-Math | GSM8K | 12 | 68.044 | 74.602 | -6.558 |
| 36 | 78 | Qwen2.5-Math | GSM8K | 12 | 67.8671 | 74.9179 | -7.0508 |
| 37 | 79 | Qwen2.5-Math | GSM8K | 12 | 67.9176 | 74.741 | -6.8234 |
| 38 | 80 | Qwen2.5-Math | GSM8K | 12 | 67.6649 | 74.8041 | -7.1392 |
| 39 | 81 | Qwen2.5-Math | GSM8K | 12 | 67.8544 | 74.3998 | -6.5454 |
| 40 | 82 | Qwen2.5-Math | GSM8K | 12 | 67.7281 | 74.185 | -6.4569 |
| 41 | 83 | Qwen2.5-Math | GSM8K | 12 | 67.8923 | 74.7283 | -6.836 |
| 42 | 84 | Qwen2.5-Math | GSM8K | 12 | 67.488 | 74.7031 | -7.2151 |
| 43 | 85 | Qwen2.5-Math | GSM8K | 12 | 67.766 | 74.324 | -6.558 |
| 44 | 86 | Qwen2.5-Math | GSM8K | 12 | 67.905 | 74.2103 | -6.3053 |
| 45 | 87 | Qwen2.5-Math | GSM8K | 12 | 67.4501 | 74.4882 | -7.0382 |
| 46 | 88 | Qwen2.5-Math | GSM8K | 12 | 67.6523 | 74.8926 | -7.2403 |
| 47 | 89 | Qwen2.5-Math | GSM8K | 12 | 67.6017 | 74.5262 | -6.9244 |
| 48 | 90 | Qwen2.5-Math | GSM8K | 12 | 68.0945 | 74.5767 | -6.4822 |
| 49 | 91 | Qwen2.5-Math | GSM8K | 12 | 67.5133 | 74.5767 | -7.0634 |
| 0 | 42 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 1 | 43 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 2 | 44 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 3 | 45 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 4 | 46 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 5 | 47 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 6 | 48 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 7 | 49 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 8 | 50 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 9 | 51 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 10 | 52 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 11 | 53 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 12 | 54 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 13 | 55 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 14 | 56 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 15 | 57 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 16 | 58 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 17 | 59 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 18 | 60 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 19 | 61 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 20 | 62 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 21 | 63 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 22 | 64 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 23 | 65 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 24 | 66 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 25 | 67 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 26 | 68 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 27 | 69 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 28 | 70 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 29 | 71 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 30 | 72 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 31 | 73 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 32 | 74 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 33 | 75 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 34 | 76 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 35 | 77 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 36 | 78 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 37 | 79 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 38 | 80 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 39 | 81 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 40 | 82 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 41 | 83 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 42 | 84 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 43 | 85 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 44 | 86 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 45 | 87 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 46 | 88 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 47 | 89 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 48 | 90 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 49 | 91 | Qwen2.5-Math | GSM8K | 16 | 67.6933 | 74.6399 | -6.9466 |
| 0 | 42 | Qwen2.5-Math | MATH500 | 4 | 63.5 | 59.7 | 3.8 |
| 1 | 43 | Qwen2.5-Math | MATH500 | 4 | 63.3 | 58.1 | 5.2 |
| 2 | 44 | Qwen2.5-Math | MATH500 | 4 | 62.7 | 58.6 | 4.1 |
| 3 | 45 | Qwen2.5-Math | MATH500 | 4 | 63.2 | 58.8 | 4.4 |
| 4 | 46 | Qwen2.5-Math | MATH500 | 4 | 64.1 | 58.4 | 5.7 |
| 5 | 47 | Qwen2.5-Math | MATH500 | 4 | 62.9 | 58.6 | 4.3 |
| 6 | 48 | Qwen2.5-Math | MATH500 | 4 | 62.7 | 59.2 | 3.5 |
| 7 | 49 | Qwen2.5-Math | MATH500 | 4 | 62.9 | 58.3 | 4.6 |
| 8 | 50 | Qwen2.5-Math | MATH500 | 4 | 63.3 | 57.3 | 6 |
| 9 | 51 | Qwen2.5-Math | MATH500 | 4 | 64 | 58.3 | 5.7 |
| 10 | 52 | Qwen2.5-Math | MATH500 | 4 | 62.4 | 58.4 | 4 |
| 11 | 53 | Qwen2.5-Math | MATH500 | 4 | 62.3 | 60.1 | 2.2 |
| 12 | 54 | Qwen2.5-Math | MATH500 | 4 | 64.1 | 58.9 | 5.2 |
| 13 | 55 | Qwen2.5-Math | MATH500 | 4 | 63.6 | 58 | 5.6 |
| 14 | 56 | Qwen2.5-Math | MATH500 | 4 | 62.3 | 58.9 | 3.4 |
| 15 | 57 | Qwen2.5-Math | MATH500 | 4 | 63.2 | 59.3 | 3.9 |
| 16 | 58 | Qwen2.5-Math | MATH500 | 4 | 63.5 | 59.6 | 3.9 |
| 17 | 59 | Qwen2.5-Math | MATH500 | 4 | 62.6 | 59.6 | 3 |
| 18 | 60 | Qwen2.5-Math | MATH500 | 4 | 64.2 | 59.1 | 5.1 |
| 19 | 61 | Qwen2.5-Math | MATH500 | 4 | 63.8 | 59.1 | 4.7 |
| 20 | 62 | Qwen2.5-Math | MATH500 | 4 | 62.5 | 58.7 | 3.8 |
| 21 | 63 | Qwen2.5-Math | MATH500 | 4 | 62.4 | 58.6 | 3.8 |
| 22 | 64 | Qwen2.5-Math | MATH500 | 4 | 63.3 | 58.5 | 4.8 |
| 23 | 65 | Qwen2.5-Math | MATH500 | 4 | 63.2 | 59.4 | 3.8 |
| 24 | 66 | Qwen2.5-Math | MATH500 | 4 | 62.2 | 58 | 4.2 |
| 25 | 67 | Qwen2.5-Math | MATH500 | 4 | 64.1 | 57.8 | 6.3 |
| 26 | 68 | Qwen2.5-Math | MATH500 | 4 | 63.6 | 58.6 | 5 |
| 27 | 69 | Qwen2.5-Math | MATH500 | 4 | 63.1 | 59.2 | 3.9 |
| 28 | 70 | Qwen2.5-Math | MATH500 | 4 | 63.5 | 58.5 | 5 |
| 29 | 71 | Qwen2.5-Math | MATH500 | 4 | 64 | 59.6 | 4.4 |
| 30 | 72 | Qwen2.5-Math | MATH500 | 4 | 61.6 | 59 | 2.6 |
| 31 | 73 | Qwen2.5-Math | MATH500 | 4 | 63.9 | 58.9 | 5 |
| 32 | 74 | Qwen2.5-Math | MATH500 | 4 | 62.1 | 58.2 | 3.9 |
| 33 | 75 | Qwen2.5-Math | MATH500 | 4 | 63.4 | 56.7 | 6.7 |
| 34 | 76 | Qwen2.5-Math | MATH500 | 4 | 63 | 58.3 | 4.7 |
| 35 | 77 | Qwen2.5-Math | MATH500 | 4 | 63.4 | 59.6 | 3.8 |
| 36 | 78 | Qwen2.5-Math | MATH500 | 4 | 63.2 | 58 | 5.2 |
| 37 | 79 | Qwen2.5-Math | MATH500 | 4 | 63.4 | 59.7 | 3.7 |
| 38 | 80 | Qwen2.5-Math | MATH500 | 4 | 63.5 | 60.5 | 3 |
| 39 | 81 | Qwen2.5-Math | MATH500 | 4 | 64.1 | 57.8 | 6.3 |
| 40 | 82 | Qwen2.5-Math | MATH500 | 4 | 64.1 | 58.4 | 5.7 |
| 41 | 83 | Qwen2.5-Math | MATH500 | 4 | 62.7 | 58.6 | 4.1 |
| 42 | 84 | Qwen2.5-Math | MATH500 | 4 | 62.2 | 58.2 | 4 |
| 43 | 85 | Qwen2.5-Math | MATH500 | 4 | 63.1 | 57.3 | 5.8 |
| 44 | 86 | Qwen2.5-Math | MATH500 | 4 | 62.2 | 58.6 | 3.6 |
| 45 | 87 | Qwen2.5-Math | MATH500 | 4 | 62.4 | 57.6 | 4.8 |
| 46 | 88 | Qwen2.5-Math | MATH500 | 4 | 62.6 | 59.9 | 2.7 |
| 47 | 89 | Qwen2.5-Math | MATH500 | 4 | 63.1 | 59.7 | 3.4 |
| 48 | 90 | Qwen2.5-Math | MATH500 | 4 | 63.1 | 58.5 | 4.6 |
| 49 | 91 | Qwen2.5-Math | MATH500 | 4 | 64.4 | 57.5 | 6.9 |
| 0 | 42 | Qwen2.5-Math | MATH500 | 8 | 63.7 | 58.85 | 4.85 |
| 1 | 43 | Qwen2.5-Math | MATH500 | 8 | 62.5 | 58.2 | 4.3 |
| 2 | 44 | Qwen2.5-Math | MATH500 | 8 | 62.75 | 58.35 | 4.4 |
| 3 | 45 | Qwen2.5-Math | MATH500 | 8 | 63.15 | 58.65 | 4.5 |
| 4 | 46 | Qwen2.5-Math | MATH500 | 8 | 63.8 | 58.35 | 5.45 |
| 5 | 47 | Qwen2.5-Math | MATH500 | 8 | 63.5 | 58.5 | 5 |
| 6 | 48 | Qwen2.5-Math | MATH500 | 8 | 63.65 | 58.7 | 4.95 |
| 7 | 49 | Qwen2.5-Math | MATH500 | 8 | 63.35 | 58.65 | 4.7 |
| 8 | 50 | Qwen2.5-Math | MATH500 | 8 | 63.15 | 58.65 | 4.5 |
| 9 | 51 | Qwen2.5-Math | MATH500 | 8 | 63.2 | 58.95 | 4.25 |
| 10 | 52 | Qwen2.5-Math | MATH500 | 8 | 62.35 | 59.1 | 3.25 |
| 11 | 53 | Qwen2.5-Math | MATH500 | 8 | 63.75 | 58 | 5.75 |
| 12 | 54 | Qwen2.5-Math | MATH500 | 8 | 63.65 | 58.7 | 4.95 |
| 13 | 55 | Qwen2.5-Math | MATH500 | 8 | 63.25 | 59.05 | 4.2 |
| 14 | 56 | Qwen2.5-Math | MATH500 | 8 | 63.35 | 58.1 | 5.25 |
| 15 | 57 | Qwen2.5-Math | MATH500 | 8 | 63.4 | 58.75 | 4.65 |
| 16 | 58 | Qwen2.5-Math | MATH500 | 8 | 63.55 | 58.45 | 5.1 |
| 17 | 59 | Qwen2.5-Math | MATH500 | 8 | 63.25 | 58.95 | 4.3 |
| 18 | 60 | Qwen2.5-Math | MATH500 | 8 | 63.45 | 59.1 | 4.35 |
| 19 | 61 | Qwen2.5-Math | MATH500 | 8 | 63.2 | 59.2 | 4 |
| 20 | 62 | Qwen2.5-Math | MATH500 | 8 | 63.45 | 58.7 | 4.75 |
| 21 | 63 | Qwen2.5-Math | MATH500 | 8 | 63.6 | 58.55 | 5.05 |
| 22 | 64 | Qwen2.5-Math | MATH500 | 8 | 63.9 | 57.55 | 6.35 |
| 23 | 65 | Qwen2.5-Math | MATH500 | 8 | 63.45 | 59.35 | 4.1 |
| 24 | 66 | Qwen2.5-Math | MATH500 | 8 | 63.45 | 58.25 | 5.2 |
| 25 | 67 | Qwen2.5-Math | MATH500 | 8 | 63.85 | 58.5 | 5.35 |
| 26 | 68 | Qwen2.5-Math | MATH500 | 8 | 64.15 | 59 | 5.15 |
| 27 | 69 | Qwen2.5-Math | MATH500 | 8 | 63.6 | 58.35 | 5.25 |
| 28 | 70 | Qwen2.5-Math | MATH500 | 8 | 63.6 | 58.6 | 5 |
| 29 | 71 | Qwen2.5-Math | MATH500 | 8 | 63.3 | 59.2 | 4.1 |
| 30 | 72 | Qwen2.5-Math | MATH500 | 8 | 63.15 | 58.4 | 4.75 |
| 31 | 73 | Qwen2.5-Math | MATH500 | 8 | 64.1 | 58.6 | 5.5 |
| 32 | 74 | Qwen2.5-Math | MATH500 | 8 | 63.05 | 58.05 | 5 |
| 33 | 75 | Qwen2.5-Math | MATH500 | 8 | 64.2 | 58.2 | 6 |
| 34 | 76 | Qwen2.5-Math | MATH500 | 8 | 62.65 | 58.5 | 4.15 |
| 35 | 77 | Qwen2.5-Math | MATH500 | 8 | 63.4 | 59.05 | 4.35 |
| 36 | 78 | Qwen2.5-Math | MATH500 | 8 | 63.35 | 58.6 | 4.75 |
| 37 | 79 | Qwen2.5-Math | MATH500 | 8 | 63.65 | 59.45 | 4.2 |
| 38 | 80 | Qwen2.5-Math | MATH500 | 8 | 63.5 | 59.25 | 4.25 |
| 39 | 81 | Qwen2.5-Math | MATH500 | 8 | 63.4 | 57.5 | 5.9 |
| 40 | 82 | Qwen2.5-Math | MATH500 | 8 | 63.3 | 59.3 | 4 |
| 41 | 83 | Qwen2.5-Math | MATH500 | 8 | 63.6 | 58.95 | 4.65 |
| 42 | 84 | Qwen2.5-Math | MATH500 | 8 | 63.1 | 58.05 | 5.05 |
| 43 | 85 | Qwen2.5-Math | MATH500 | 8 | 63.6 | 58.45 | 5.15 |
| 44 | 86 | Qwen2.5-Math | MATH500 | 8 | 62.8 | 57.95 | 4.85 |
| 45 | 87 | Qwen2.5-Math | MATH500 | 8 | 63.2 | 58.7 | 4.5 |
| 46 | 88 | Qwen2.5-Math | MATH500 | 8 | 63.6 | 59.05 | 4.55 |
| 47 | 89 | Qwen2.5-Math | MATH500 | 8 | 63.65 | 59.7 | 3.95 |
| 48 | 90 | Qwen2.5-Math | MATH500 | 8 | 63.9 | 58.35 | 5.55 |
| 49 | 91 | Qwen2.5-Math | MATH500 | 8 | 63.75 | 57.85 | 5.9 |
| 0 | 42 | Qwen2.5-Math | MATH500 | 12 | 64.0667 | 58.6667 | 5.4 |
| 1 | 43 | Qwen2.5-Math | MATH500 | 12 | 63.2333 | 58.5667 | 4.6667 |
| 2 | 44 | Qwen2.5-Math | MATH500 | 12 | 63.0333 | 58.5 | 4.5333 |
| 3 | 45 | Qwen2.5-Math | MATH500 | 12 | 63.5 | 58.6667 | 4.8333 |
| 4 | 46 | Qwen2.5-Math | MATH500 | 12 | 63.7 | 58.4 | 5.3 |
| 5 | 47 | Qwen2.5-Math | MATH500 | 12 | 63.4667 | 58.3333 | 5.1333 |
| 6 | 48 | Qwen2.5-Math | MATH500 | 12 | 63.6 | 58.6667 | 4.9333 |
| 7 | 49 | Qwen2.5-Math | MATH500 | 12 | 63.5667 | 58.1 | 5.4667 |
| 8 | 50 | Qwen2.5-Math | MATH500 | 12 | 63.7667 | 58.4667 | 5.3 |
| 9 | 51 | Qwen2.5-Math | MATH500 | 12 | 63.2667 | 58.8333 | 4.4333 |
| 10 | 52 | Qwen2.5-Math | MATH500 | 12 | 62.8667 | 58.8667 | 4 |
| 11 | 53 | Qwen2.5-Math | MATH500 | 12 | 63.3333 | 58.7333 | 4.6 |
| 12 | 54 | Qwen2.5-Math | MATH500 | 12 | 64 | 58.6333 | 5.3667 |
| 13 | 55 | Qwen2.5-Math | MATH500 | 12 | 63.6 | 58.6 | 5 |
| 14 | 56 | Qwen2.5-Math | MATH500 | 12 | 63.5333 | 58.5 | 5.0333 |
| 15 | 57 | Qwen2.5-Math | MATH500 | 12 | 63.2 | 58.3667 | 4.8333 |
| 16 | 58 | Qwen2.5-Math | MATH500 | 12 | 63.8 | 58.4667 | 5.3333 |
| 17 | 59 | Qwen2.5-Math | MATH500 | 12 | 63.3333 | 58.7 | 4.6333 |
| 18 | 60 | Qwen2.5-Math | MATH500 | 12 | 63.3667 | 58.8333 | 4.5333 |
| 19 | 61 | Qwen2.5-Math | MATH500 | 12 | 64 | 58.5 | 5.5 |
| 20 | 62 | Qwen2.5-Math | MATH500 | 12 | 63.4 | 58.7667 | 4.6333 |
| 21 | 63 | Qwen2.5-Math | MATH500 | 12 | 63.8667 | 58.3667 | 5.5 |
| 22 | 64 | Qwen2.5-Math | MATH500 | 12 | 63.9333 | 58.3667 | 5.5667 |
| 23 | 65 | Qwen2.5-Math | MATH500 | 12 | 63.4 | 59.1 | 4.3 |
| 24 | 66 | Qwen2.5-Math | MATH500 | 12 | 63.5667 | 58.5 | 5.0667 |
| 25 | 67 | Qwen2.5-Math | MATH500 | 12 | 63.5667 | 58.5 | 5.0667 |
| 26 | 68 | Qwen2.5-Math | MATH500 | 12 | 63.4333 | 59.0333 | 4.4 |
| 27 | 69 | Qwen2.5-Math | MATH500 | 12 | 63.3 | 58 | 5.3 |
| 28 | 70 | Qwen2.5-Math | MATH500 | 12 | 63.3667 | 58.6 | 4.7667 |
| 29 | 71 | Qwen2.5-Math | MATH500 | 12 | 63.3667 | 58.7667 | 4.6 |
| 30 | 72 | Qwen2.5-Math | MATH500 | 12 | 63.4667 | 58.4667 | 5 |
| 31 | 73 | Qwen2.5-Math | MATH500 | 12 | 63.6667 | 58.6667 | 5 |
| 32 | 74 | Qwen2.5-Math | MATH500 | 12 | 63.2667 | 58.6667 | 4.6 |
| 33 | 75 | Qwen2.5-Math | MATH500 | 12 | 63.7 | 58.8667 | 4.8333 |
| 34 | 76 | Qwen2.5-Math | MATH500 | 12 | 63.3333 | 58.5 | 4.8333 |
| 35 | 77 | Qwen2.5-Math | MATH500 | 12 | 63.3333 | 58.6333 | 4.7 |
| 36 | 78 | Qwen2.5-Math | MATH500 | 12 | 63.3333 | 58.9 | 4.4333 |
| 37 | 79 | Qwen2.5-Math | MATH500 | 12 | 63.7667 | 58.7 | 5.0667 |
| 38 | 80 | Qwen2.5-Math | MATH500 | 12 | 63.6667 | 58.2 | 5.4667 |
| 39 | 81 | Qwen2.5-Math | MATH500 | 12 | 63.6 | 58.3333 | 5.2667 |
| 40 | 82 | Qwen2.5-Math | MATH500 | 12 | 63.5333 | 58.5333 | 5 |
| 41 | 83 | Qwen2.5-Math | MATH500 | 12 | 63.3333 | 58.5333 | 4.8 |
| 42 | 84 | Qwen2.5-Math | MATH500 | 12 | 63.5333 | 58.3 | 5.2333 |
| 43 | 85 | Qwen2.5-Math | MATH500 | 12 | 63.5333 | 58.6667 | 4.8667 |
| 44 | 86 | Qwen2.5-Math | MATH500 | 12 | 63.6333 | 58.1667 | 5.4667 |
| 45 | 87 | Qwen2.5-Math | MATH500 | 12 | 63.7333 | 58.8333 | 4.9 |
| 46 | 88 | Qwen2.5-Math | MATH500 | 12 | 63.4333 | 58.4 | 5.0333 |
| 47 | 89 | Qwen2.5-Math | MATH500 | 12 | 63.4 | 58.6667 | 4.7333 |
| 48 | 90 | Qwen2.5-Math | MATH500 | 12 | 63.5667 | 58.4667 | 5.1 |
| 49 | 91 | Qwen2.5-Math | MATH500 | 12 | 63.5 | 58.2667 | 5.2333 |
| 0 | 42 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 1 | 43 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 2 | 44 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 3 | 45 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 4 | 46 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 5 | 47 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 6 | 48 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 7 | 49 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 8 | 50 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 9 | 51 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 10 | 52 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 11 | 53 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 12 | 54 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 13 | 55 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 14 | 56 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 15 | 57 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 16 | 58 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 17 | 59 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 18 | 60 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 19 | 61 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 20 | 62 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 21 | 63 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 22 | 64 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 23 | 65 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 24 | 66 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 25 | 67 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 26 | 68 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 27 | 69 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 28 | 70 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 29 | 71 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 30 | 72 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 31 | 73 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 32 | 74 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 33 | 75 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 34 | 76 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 35 | 77 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 36 | 78 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 37 | 79 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 38 | 80 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 39 | 81 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 40 | 82 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 41 | 83 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 42 | 84 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 43 | 85 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 44 | 86 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 45 | 87 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 46 | 88 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 47 | 89 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 48 | 90 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 49 | 91 | Qwen2.5-Math | MATH500 | 16 | 63.5 | 58.65 | 4.85 |
| 0 | 42 | Qwen2.5-Math | SVAMP | 4 | 70.55 | 80.45 | -9.9 |
| 1 | 43 | Qwen2.5-Math | SVAMP | 4 | 70.35 | 80.9 | -10.55 |
| 2 | 44 | Qwen2.5-Math | SVAMP | 4 | 70.7 | 81.3 | -10.6 |
| 3 | 45 | Qwen2.5-Math | SVAMP | 4 | 70.75 | 80.5 | -9.75 |
| 4 | 46 | Qwen2.5-Math | SVAMP | 4 | 68.9 | 82 | -13.1 |
| 5 | 47 | Qwen2.5-Math | SVAMP | 4 | 70.35 | 81.35 | -11 |
| 6 | 48 | Qwen2.5-Math | SVAMP | 4 | 71.2 | 82.4 | -11.2 |
| 7 | 49 | Qwen2.5-Math | SVAMP | 4 | 70.2 | 81.6 | -11.4 |
| 8 | 50 | Qwen2.5-Math | SVAMP | 4 | 70.35 | 81.8 | -11.45 |
| 9 | 51 | Qwen2.5-Math | SVAMP | 4 | 70.75 | 81.85 | -11.1 |
| 10 | 52 | Qwen2.5-Math | SVAMP | 4 | 69.1 | 81.15 | -12.05 |
| 11 | 53 | Qwen2.5-Math | SVAMP | 4 | 70.35 | 80.9 | -10.55 |
| 12 | 54 | Qwen2.5-Math | SVAMP | 4 | 69.75 | 81.25 | -11.5 |
| 13 | 55 | Qwen2.5-Math | SVAMP | 4 | 69.15 | 80.8 | -11.65 |
| 14 | 56 | Qwen2.5-Math | SVAMP | 4 | 69.6 | 81.25 | -11.65 |
| 15 | 57 | Qwen2.5-Math | SVAMP | 4 | 70.2 | 81.75 | -11.55 |
| 16 | 58 | Qwen2.5-Math | SVAMP | 4 | 73 | 80.45 | -7.45 |
| 17 | 59 | Qwen2.5-Math | SVAMP | 4 | 71.6 | 81.85 | -10.25 |
| 18 | 60 | Qwen2.5-Math | SVAMP | 4 | 70.65 | 81.4 | -10.75 |
| 19 | 61 | Qwen2.5-Math | SVAMP | 4 | 69.35 | 81.35 | -12 |
| 20 | 62 | Qwen2.5-Math | SVAMP | 4 | 69.95 | 80.95 | -11 |
| 21 | 63 | Qwen2.5-Math | SVAMP | 4 | 70 | 81.45 | -11.45 |
| 22 | 64 | Qwen2.5-Math | SVAMP | 4 | 69.65 | 80.2 | -10.55 |
| 23 | 65 | Qwen2.5-Math | SVAMP | 4 | 70.9 | 80.95 | -10.05 |
| 24 | 66 | Qwen2.5-Math | SVAMP | 4 | 70.45 | 81.3 | -10.85 |
| 25 | 67 | Qwen2.5-Math | SVAMP | 4 | 70 | 81.7 | -11.7 |
| 26 | 68 | Qwen2.5-Math | SVAMP | 4 | 70 | 80 | -10 |
| 27 | 69 | Qwen2.5-Math | SVAMP | 4 | 70.05 | 80.25 | -10.2 |
| 28 | 70 | Qwen2.5-Math | SVAMP | 4 | 71.25 | 81.05 | -9.8 |
| 29 | 71 | Qwen2.5-Math | SVAMP | 4 | 69.9 | 81.1 | -11.2 |
| 30 | 72 | Qwen2.5-Math | SVAMP | 4 | 71.35 | 80.9 | -9.55 |
| 31 | 73 | Qwen2.5-Math | SVAMP | 4 | 70.35 | 80.7 | -10.35 |
| 32 | 74 | Qwen2.5-Math | SVAMP | 4 | 70.2 | 80.8 | -10.6 |
| 33 | 75 | Qwen2.5-Math | SVAMP | 4 | 70.2 | 81.7 | -11.5 |
| 34 | 76 | Qwen2.5-Math | SVAMP | 4 | 70.65 | 81 | -10.35 |
| 35 | 77 | Qwen2.5-Math | SVAMP | 4 | 71.5 | 81 | -9.5 |
| 36 | 78 | Qwen2.5-Math | SVAMP | 4 | 69.8 | 81.6 | -11.8 |
| 37 | 79 | Qwen2.5-Math | SVAMP | 4 | 68.75 | 81.2 | -12.45 |
| 38 | 80 | Qwen2.5-Math | SVAMP | 4 | 71.35 | 82.55 | -11.2 |
| 39 | 81 | Qwen2.5-Math | SVAMP | 4 | 70.45 | 80.45 | -10 |
| 40 | 82 | Qwen2.5-Math | SVAMP | 4 | 68.35 | 81.5 | -13.15 |
| 41 | 83 | Qwen2.5-Math | SVAMP | 4 | 70.9 | 81.15 | -10.25 |
| 42 | 84 | Qwen2.5-Math | SVAMP | 4 | 71.4 | 81.6 | -10.2 |
| 43 | 85 | Qwen2.5-Math | SVAMP | 4 | 68.85 | 81.4 | -12.55 |
| 44 | 86 | Qwen2.5-Math | SVAMP | 4 | 70.25 | 81.35 | -11.1 |
| 45 | 87 | Qwen2.5-Math | SVAMP | 4 | 71.2 | 81.5 | -10.3 |
| 46 | 88 | Qwen2.5-Math | SVAMP | 4 | 70.45 | 82.3 | -11.85 |
| 47 | 89 | Qwen2.5-Math | SVAMP | 4 | 71.05 | 81.5 | -10.45 |
| 48 | 90 | Qwen2.5-Math | SVAMP | 4 | 70.45 | 80.95 | -10.5 |
| 49 | 91 | Qwen2.5-Math | SVAMP | 4 | 69.1 | 82.35 | -13.25 |
| 0 | 42 | Qwen2.5-Math | SVAMP | 8 | 70.4 | 80.85 | -10.45 |
| 1 | 43 | Qwen2.5-Math | SVAMP | 8 | 69.725 | 81.05 | -11.325 |
| 2 | 44 | Qwen2.5-Math | SVAMP | 8 | 69.725 | 81.175 | -11.45 |
| 3 | 45 | Qwen2.5-Math | SVAMP | 8 | 70.5 | 81.15 | -10.65 |
| 4 | 46 | Qwen2.5-Math | SVAMP | 8 | 69.25 | 81.45 | -12.2 |
| 5 | 47 | Qwen2.5-Math | SVAMP | 8 | 70.35 | 81.5 | -11.15 |
| 6 | 48 | Qwen2.5-Math | SVAMP | 8 | 70.525 | 81.35 | -10.825 |
| 7 | 49 | Qwen2.5-Math | SVAMP | 8 | 69.975 | 81.275 | -11.3 |
| 8 | 50 | Qwen2.5-Math | SVAMP | 8 | 69.975 | 81.35 | -11.375 |
| 9 | 51 | Qwen2.5-Math | SVAMP | 8 | 70.775 | 81.725 | -10.95 |
| 10 | 52 | Qwen2.5-Math | SVAMP | 8 | 69.85 | 81.425 | -11.575 |
| 11 | 53 | Qwen2.5-Math | SVAMP | 8 | 70.45 | 82 | -11.55 |
| 12 | 54 | Qwen2.5-Math | SVAMP | 8 | 70.65 | 81.575 | -10.925 |
| 13 | 55 | Qwen2.5-Math | SVAMP | 8 | 70.7 | 81.125 | -10.425 |
| 14 | 56 | Qwen2.5-Math | SVAMP | 8 | 69.85 | 81.475 | -11.625 |
| 15 | 57 | Qwen2.5-Math | SVAMP | 8 | 70.625 | 80.975 | -10.35 |
| 16 | 58 | Qwen2.5-Math | SVAMP | 8 | 71.825 | 81.3 | -9.475 |
| 17 | 59 | Qwen2.5-Math | SVAMP | 8 | 70.3 | 81.575 | -11.275 |
| 18 | 60 | Qwen2.5-Math | SVAMP | 8 | 70.15 | 81.175 | -11.025 |
| 19 | 61 | Qwen2.5-Math | SVAMP | 8 | 70.075 | 81.15 | -11.075 |
| 20 | 62 | Qwen2.5-Math | SVAMP | 8 | 69.75 | 80.75 | -11 |
| 21 | 63 | Qwen2.5-Math | SVAMP | 8 | 69.725 | 81.15 | -11.425 |
| 22 | 64 | Qwen2.5-Math | SVAMP | 8 | 70.125 | 81.1 | -10.975 |
| 23 | 65 | Qwen2.5-Math | SVAMP | 8 | 70.5 | 81.425 | -10.925 |
| 24 | 66 | Qwen2.5-Math | SVAMP | 8 | 69.475 | 81.45 | -11.975 |
| 25 | 67 | Qwen2.5-Math | SVAMP | 8 | 70.2 | 81.4 | -11.2 |
| 26 | 68 | Qwen2.5-Math | SVAMP | 8 | 71.025 | 81.325 | -10.3 |
| 27 | 69 | Qwen2.5-Math | SVAMP | 8 | 70.45 | 81.75 | -11.3 |
| 28 | 70 | Qwen2.5-Math | SVAMP | 8 | 69.925 | 81.4 | -11.475 |
| 29 | 71 | Qwen2.5-Math | SVAMP | 8 | 70.575 | 81.525 | -10.95 |
| 30 | 72 | Qwen2.5-Math | SVAMP | 8 | 70.225 | 80.875 | -10.65 |
| 31 | 73 | Qwen2.5-Math | SVAMP | 8 | 70.425 | 81.45 | -11.025 |
| 32 | 74 | Qwen2.5-Math | SVAMP | 8 | 69.275 | 81.575 | -12.3 |
| 33 | 75 | Qwen2.5-Math | SVAMP | 8 | 70 | 81.625 | -11.625 |
| 34 | 76 | Qwen2.5-Math | SVAMP | 8 | 70.575 | 81.575 | -11 |
| 35 | 77 | Qwen2.5-Math | SVAMP | 8 | 70.05 | 81.1 | -11.05 |
| 36 | 78 | Qwen2.5-Math | SVAMP | 8 | 69.375 | 81.7 | -12.325 |
| 37 | 79 | Qwen2.5-Math | SVAMP | 8 | 69.175 | 81.375 | -12.2 |
| 38 | 80 | Qwen2.5-Math | SVAMP | 8 | 70.35 | 82.9 | -12.55 |
| 39 | 81 | Qwen2.5-Math | SVAMP | 8 | 70.25 | 80.75 | -10.5 |
| 40 | 82 | Qwen2.5-Math | SVAMP | 8 | 70.025 | 81.575 | -11.55 |
| 41 | 83 | Qwen2.5-Math | SVAMP | 8 | 70.075 | 81.925 | -11.85 |
| 42 | 84 | Qwen2.5-Math | SVAMP | 8 | 70.35 | 81.625 | -11.275 |
| 43 | 85 | Qwen2.5-Math | SVAMP | 8 | 70.275 | 81.1 | -10.825 |
| 44 | 86 | Qwen2.5-Math | SVAMP | 8 | 69.575 | 81.725 | -12.15 |
| 45 | 87 | Qwen2.5-Math | SVAMP | 8 | 70.875 | 81.975 | -11.1 |
| 46 | 88 | Qwen2.5-Math | SVAMP | 8 | 70.125 | 81.925 | -11.8 |
| 47 | 89 | Qwen2.5-Math | SVAMP | 8 | 69.55 | 81.825 | -12.275 |
| 48 | 90 | Qwen2.5-Math | SVAMP | 8 | 70.725 | 81.5 | -10.775 |
| 49 | 91 | Qwen2.5-Math | SVAMP | 8 | 69.85 | 81.775 | -11.925 |
| 0 | 42 | Qwen2.5-Math | SVAMP | 12 | 70.1167 | 81.3167 | -11.2 |
| 1 | 43 | Qwen2.5-Math | SVAMP | 12 | 70.3 | 81.1 | -10.8 |
| 2 | 44 | Qwen2.5-Math | SVAMP | 12 | 69.9333 | 81.2833 | -11.35 |
| 3 | 45 | Qwen2.5-Math | SVAMP | 12 | 70.3667 | 81.4667 | -11.1 |
| 4 | 46 | Qwen2.5-Math | SVAMP | 12 | 70.2833 | 81.4667 | -11.1833 |
| 5 | 47 | Qwen2.5-Math | SVAMP | 12 | 70.7167 | 81.4333 | -10.7167 |
| 6 | 48 | Qwen2.5-Math | SVAMP | 12 | 70.3167 | 81.5667 | -11.25 |
| 7 | 49 | Qwen2.5-Math | SVAMP | 12 | 69.9833 | 81.3333 | -11.35 |
| 8 | 50 | Qwen2.5-Math | SVAMP | 12 | 69.8167 | 81.3167 | -11.5 |
| 9 | 51 | Qwen2.5-Math | SVAMP | 12 | 70.55 | 81.6833 | -11.1333 |
| 10 | 52 | Qwen2.5-Math | SVAMP | 12 | 70.3833 | 81.6667 | -11.2833 |
| 11 | 53 | Qwen2.5-Math | SVAMP | 12 | 69.7167 | 81.85 | -12.1333 |
| 12 | 54 | Qwen2.5-Math | SVAMP | 12 | 69.85 | 81.9 | -12.05 |
| 13 | 55 | Qwen2.5-Math | SVAMP | 12 | 70.35 | 81.4667 | -11.1167 |
| 14 | 56 | Qwen2.5-Math | SVAMP | 12 | 70.0833 | 81.4667 | -11.3833 |
| 15 | 57 | Qwen2.5-Math | SVAMP | 12 | 70.2667 | 81.5333 | -11.2667 |
| 16 | 58 | Qwen2.5-Math | SVAMP | 12 | 70.7 | 81.5667 | -10.8667 |
| 17 | 59 | Qwen2.5-Math | SVAMP | 12 | 70.0333 | 81.65 | -11.6167 |
| 18 | 60 | Qwen2.5-Math | SVAMP | 12 | 70 | 81.3667 | -11.3667 |
| 19 | 61 | Qwen2.5-Math | SVAMP | 12 | 69.65 | 81.45 | -11.8 |
| 20 | 62 | Qwen2.5-Math | SVAMP | 12 | 69.7833 | 81.4833 | -11.7 |
| 21 | 63 | Qwen2.5-Math | SVAMP | 12 | 70.2 | 81.2 | -11 |
| 22 | 64 | Qwen2.5-Math | SVAMP | 12 | 69.9167 | 81.75 | -11.8333 |
| 23 | 65 | Qwen2.5-Math | SVAMP | 12 | 70.4667 | 81.1333 | -10.6667 |
| 24 | 66 | Qwen2.5-Math | SVAMP | 12 | 69.5 | 81.65 | -12.15 |
| 25 | 67 | Qwen2.5-Math | SVAMP | 12 | 70.1167 | 81.3667 | -11.25 |
| 26 | 68 | Qwen2.5-Math | SVAMP | 12 | 70.35 | 81.25 | -10.9 |
| 27 | 69 | Qwen2.5-Math | SVAMP | 12 | 70.2833 | 81.5167 | -11.2333 |
| 28 | 70 | Qwen2.5-Math | SVAMP | 12 | 70.0667 | 81.4833 | -11.4167 |
| 29 | 71 | Qwen2.5-Math | SVAMP | 12 | 70.4167 | 81.4167 | -11 |
| 30 | 72 | Qwen2.5-Math | SVAMP | 12 | 70.1667 | 81.7667 | -11.6 |
| 31 | 73 | Qwen2.5-Math | SVAMP | 12 | 69.9833 | 81.9 | -11.9167 |
| 32 | 74 | Qwen2.5-Math | SVAMP | 12 | 70.05 | 81.8167 | -11.7667 |
| 33 | 75 | Qwen2.5-Math | SVAMP | 12 | 70.0667 | 81.0667 | -11 |
| 34 | 76 | Qwen2.5-Math | SVAMP | 12 | 69.8833 | 81.5833 | -11.7 |
| 35 | 77 | Qwen2.5-Math | SVAMP | 12 | 69.85 | 81.2167 | -11.3667 |
| 36 | 78 | Qwen2.5-Math | SVAMP | 12 | 70.0667 | 81.5167 | -11.45 |
| 37 | 79 | Qwen2.5-Math | SVAMP | 12 | 69.5333 | 81.4333 | -11.9 |
| 38 | 80 | Qwen2.5-Math | SVAMP | 12 | 70.7333 | 81.8167 | -11.0833 |
| 39 | 81 | Qwen2.5-Math | SVAMP | 12 | 69.95 | 81.35 | -11.4 |
| 40 | 82 | Qwen2.5-Math | SVAMP | 12 | 69.8833 | 81.5167 | -11.6333 |
| 41 | 83 | Qwen2.5-Math | SVAMP | 12 | 70.3167 | 81.7333 | -11.4167 |
| 42 | 84 | Qwen2.5-Math | SVAMP | 12 | 69.6833 | 81.6833 | -12 |
| 43 | 85 | Qwen2.5-Math | SVAMP | 12 | 70.05 | 81.2167 | -11.1667 |
| 44 | 86 | Qwen2.5-Math | SVAMP | 12 | 70.0167 | 81.5833 | -11.5667 |
| 45 | 87 | Qwen2.5-Math | SVAMP | 12 | 70.6 | 81.4333 | -10.8333 |
| 46 | 88 | Qwen2.5-Math | SVAMP | 12 | 70.0833 | 81.9833 | -11.9 |
| 47 | 89 | Qwen2.5-Math | SVAMP | 12 | 70.2333 | 81.6167 | -11.3833 |
| 48 | 90 | Qwen2.5-Math | SVAMP | 12 | 70.0667 | 81.7667 | -11.7 |
| 49 | 91 | Qwen2.5-Math | SVAMP | 12 | 69.9667 | 81.6 | -11.6333 |
| 0 | 42 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 1 | 43 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 2 | 44 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 3 | 45 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 4 | 46 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 5 | 47 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 6 | 48 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 7 | 49 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 8 | 50 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 9 | 51 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 10 | 52 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 11 | 53 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 12 | 54 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 13 | 55 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 14 | 56 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 15 | 57 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 16 | 58 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 17 | 59 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 18 | 60 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 19 | 61 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 20 | 62 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 21 | 63 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 22 | 64 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 23 | 65 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 24 | 66 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 25 | 67 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 26 | 68 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 27 | 69 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 28 | 70 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 29 | 71 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 30 | 72 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 31 | 73 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 32 | 74 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 33 | 75 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 34 | 76 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 35 | 77 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 36 | 78 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 37 | 79 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 38 | 80 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 39 | 81 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 40 | 82 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 41 | 83 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 42 | 84 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 43 | 85 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 44 | 86 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 45 | 87 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 46 | 88 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 47 | 89 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 48 | 90 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 49 | 91 | Qwen2.5-Math | SVAMP | 16 | 70.1375 | 81.4375 | -11.3 |
| 0 | 42 | Qwen3-4B | AQuA | 4 | 53.5433 | 46.6535 | 6.8898 |
| 1 | 43 | Qwen3-4B | AQuA | 4 | 54.5276 | 47.2441 | 7.2835 |
| 2 | 44 | Qwen3-4B | AQuA | 4 | 51.378 | 46.8504 | 4.5276 |
| 3 | 45 | Qwen3-4B | AQuA | 4 | 54.7244 | 48.622 | 6.1024 |
| 4 | 46 | Qwen3-4B | AQuA | 4 | 50.5906 | 48.8189 | 1.7717 |
| 5 | 47 | Qwen3-4B | AQuA | 4 | 52.9528 | 49.8031 | 3.1496 |
| 6 | 48 | Qwen3-4B | AQuA | 4 | 51.9685 | 48.622 | 3.3465 |
| 7 | 49 | Qwen3-4B | AQuA | 4 | 50.5906 | 45.2756 | 5.315 |
| 8 | 50 | Qwen3-4B | AQuA | 4 | 49.0157 | 47.4409 | 1.5748 |
| 9 | 51 | Qwen3-4B | AQuA | 4 | 54.9213 | 47.2441 | 7.6772 |
| 10 | 52 | Qwen3-4B | AQuA | 4 | 54.1339 | 46.8504 | 7.2835 |
| 11 | 53 | Qwen3-4B | AQuA | 4 | 50.9843 | 45.8661 | 5.1181 |
| 12 | 54 | Qwen3-4B | AQuA | 4 | 53.937 | 47.4409 | 6.4961 |
| 13 | 55 | Qwen3-4B | AQuA | 4 | 50 | 46.8504 | 3.1496 |
| 14 | 56 | Qwen3-4B | AQuA | 4 | 50.3937 | 50.7874 | -0.3937 |
| 15 | 57 | Qwen3-4B | AQuA | 4 | 52.5591 | 47.6378 | 4.9213 |
| 16 | 58 | Qwen3-4B | AQuA | 4 | 51.9685 | 50.5906 | 1.378 |
| 17 | 59 | Qwen3-4B | AQuA | 4 | 47.0472 | 49.2126 | -2.1654 |
| 18 | 60 | Qwen3-4B | AQuA | 4 | 53.3465 | 46.4567 | 6.8898 |
| 19 | 61 | Qwen3-4B | AQuA | 4 | 50 | 49.2126 | 0.7874 |
| 20 | 62 | Qwen3-4B | AQuA | 4 | 51.5748 | 48.8189 | 2.7559 |
| 21 | 63 | Qwen3-4B | AQuA | 4 | 51.5748 | 48.622 | 2.9528 |
| 22 | 64 | Qwen3-4B | AQuA | 4 | 52.1654 | 47.2441 | 4.9213 |
| 23 | 65 | Qwen3-4B | AQuA | 4 | 49.6063 | 48.2283 | 1.378 |
| 24 | 66 | Qwen3-4B | AQuA | 4 | 50.1969 | 47.0472 | 3.1496 |
| 25 | 67 | Qwen3-4B | AQuA | 4 | 52.1654 | 46.8504 | 5.315 |
| 26 | 68 | Qwen3-4B | AQuA | 4 | 48.0315 | 50.3937 | -2.3622 |
| 27 | 69 | Qwen3-4B | AQuA | 4 | 49.4094 | 48.622 | 0.7874 |
| 28 | 70 | Qwen3-4B | AQuA | 4 | 45.2756 | 47.0472 | -1.7717 |
| 29 | 71 | Qwen3-4B | AQuA | 4 | 51.7717 | 47.2441 | 4.5276 |
| 30 | 72 | Qwen3-4B | AQuA | 4 | 52.7559 | 46.6535 | 6.1024 |
| 31 | 73 | Qwen3-4B | AQuA | 4 | 51.5748 | 46.8504 | 4.7244 |
| 32 | 74 | Qwen3-4B | AQuA | 4 | 50.9843 | 48.4252 | 2.5591 |
| 33 | 75 | Qwen3-4B | AQuA | 4 | 49.8031 | 49.0157 | 0.7874 |
| 34 | 76 | Qwen3-4B | AQuA | 4 | 48.0315 | 47.0472 | 0.9843 |
| 35 | 77 | Qwen3-4B | AQuA | 4 | 47.6378 | 48.8189 | -1.1811 |
| 36 | 78 | Qwen3-4B | AQuA | 4 | 51.7717 | 46.8504 | 4.9213 |
| 37 | 79 | Qwen3-4B | AQuA | 4 | 49.2126 | 48.622 | 0.5906 |
| 38 | 80 | Qwen3-4B | AQuA | 4 | 51.5748 | 47.6378 | 3.937 |
| 39 | 81 | Qwen3-4B | AQuA | 4 | 53.3465 | 48.8189 | 4.5276 |
| 40 | 82 | Qwen3-4B | AQuA | 4 | 48.8189 | 48.8189 | 0 |
| 41 | 83 | Qwen3-4B | AQuA | 4 | 49.8031 | 47.6378 | 2.1654 |
| 42 | 84 | Qwen3-4B | AQuA | 4 | 54.7244 | 48.8189 | 5.9055 |
| 43 | 85 | Qwen3-4B | AQuA | 4 | 52.7559 | 50.3937 | 2.3622 |
| 44 | 86 | Qwen3-4B | AQuA | 4 | 51.7717 | 48.2283 | 3.5433 |
| 45 | 87 | Qwen3-4B | AQuA | 4 | 50.3937 | 48.622 | 1.7717 |
| 46 | 88 | Qwen3-4B | AQuA | 4 | 49.6063 | 49.6063 | 0 |
| 47 | 89 | Qwen3-4B | AQuA | 4 | 52.7559 | 50 | 2.7559 |
| 48 | 90 | Qwen3-4B | AQuA | 4 | 51.378 | 46.2598 | 5.1181 |
| 49 | 91 | Qwen3-4B | AQuA | 4 | 51.5748 | 48.622 | 2.9528 |
| 0 | 42 | Qwen3-4B | AQuA | 8 | 51.4764 | 47.2441 | 4.2323 |
| 1 | 43 | Qwen3-4B | AQuA | 8 | 53.6417 | 48.3268 | 5.315 |
| 2 | 44 | Qwen3-4B | AQuA | 8 | 51.9685 | 46.3583 | 5.6102 |
| 3 | 45 | Qwen3-4B | AQuA | 8 | 51.9685 | 48.5236 | 3.4449 |
| 4 | 46 | Qwen3-4B | AQuA | 8 | 51.1811 | 48.1299 | 3.0512 |
| 5 | 47 | Qwen3-4B | AQuA | 8 | 54.1339 | 47.3425 | 6.7913 |
| 6 | 48 | Qwen3-4B | AQuA | 8 | 52.2638 | 48.2283 | 4.0354 |
| 7 | 49 | Qwen3-4B | AQuA | 8 | 51.6732 | 46.1614 | 5.5118 |
| 8 | 50 | Qwen3-4B | AQuA | 8 | 49.6063 | 48.5236 | 1.0827 |
| 9 | 51 | Qwen3-4B | AQuA | 8 | 52.7559 | 47.4409 | 5.315 |
| 10 | 52 | Qwen3-4B | AQuA | 8 | 51.6732 | 47.8346 | 3.8386 |
| 11 | 53 | Qwen3-4B | AQuA | 8 | 51.9685 | 47.9331 | 4.0354 |
| 12 | 54 | Qwen3-4B | AQuA | 8 | 51.2795 | 47.4409 | 3.8386 |
| 13 | 55 | Qwen3-4B | AQuA | 8 | 51.6732 | 48.1299 | 3.5433 |
| 14 | 56 | Qwen3-4B | AQuA | 8 | 51.1811 | 47.5394 | 3.6417 |
| 15 | 57 | Qwen3-4B | AQuA | 8 | 51.9685 | 46.8504 | 5.1181 |
| 16 | 58 | Qwen3-4B | AQuA | 8 | 51.5748 | 48.1299 | 3.4449 |
| 17 | 59 | Qwen3-4B | AQuA | 8 | 50.3937 | 48.4252 | 1.9685 |
| 18 | 60 | Qwen3-4B | AQuA | 8 | 52.0669 | 46.9488 | 5.1181 |
| 19 | 61 | Qwen3-4B | AQuA | 8 | 49.9016 | 48.4252 | 1.4764 |
| 20 | 62 | Qwen3-4B | AQuA | 8 | 51.9685 | 48.1299 | 3.8386 |
| 21 | 63 | Qwen3-4B | AQuA | 8 | 49.7047 | 47.2441 | 2.4606 |
| 22 | 64 | Qwen3-4B | AQuA | 8 | 52.7559 | 47.3425 | 5.4134 |
| 23 | 65 | Qwen3-4B | AQuA | 8 | 52.3622 | 46.8504 | 5.5118 |
| 24 | 66 | Qwen3-4B | AQuA | 8 | 51.2795 | 47.3425 | 3.937 |
| 25 | 67 | Qwen3-4B | AQuA | 8 | 51.9685 | 46.5551 | 5.4134 |
| 26 | 68 | Qwen3-4B | AQuA | 8 | 50.4921 | 48.5236 | 1.9685 |
| 27 | 69 | Qwen3-4B | AQuA | 8 | 49.7047 | 47.8346 | 1.8701 |
| 28 | 70 | Qwen3-4B | AQuA | 8 | 49.4094 | 48.5236 | 0.8858 |
| 29 | 71 | Qwen3-4B | AQuA | 8 | 51.9685 | 46.3583 | 5.6102 |
| 30 | 72 | Qwen3-4B | AQuA | 8 | 52.2638 | 48.1299 | 4.1339 |
| 31 | 73 | Qwen3-4B | AQuA | 8 | 51.5748 | 48.1299 | 3.4449 |
| 32 | 74 | Qwen3-4B | AQuA | 8 | 51.378 | 46.9488 | 4.4291 |
| 33 | 75 | Qwen3-4B | AQuA | 8 | 52.1654 | 46.3583 | 5.8071 |
| 34 | 76 | Qwen3-4B | AQuA | 8 | 50.1969 | 47.0472 | 3.1496 |
| 35 | 77 | Qwen3-4B | AQuA | 8 | 50.0984 | 46.4567 | 3.6417 |
| 36 | 78 | Qwen3-4B | AQuA | 8 | 52.0669 | 47.2441 | 4.8228 |
| 37 | 79 | Qwen3-4B | AQuA | 8 | 52.5591 | 47.0472 | 5.5118 |
| 38 | 80 | Qwen3-4B | AQuA | 8 | 50.4921 | 48.1299 | 2.3622 |
| 39 | 81 | Qwen3-4B | AQuA | 8 | 51.378 | 47.9331 | 3.4449 |
| 40 | 82 | Qwen3-4B | AQuA | 8 | 50.689 | 48.9173 | 1.7717 |
| 41 | 83 | Qwen3-4B | AQuA | 8 | 50.4921 | 46.1614 | 4.3307 |
| 42 | 84 | Qwen3-4B | AQuA | 8 | 51.2795 | 47.6378 | 3.6417 |
| 43 | 85 | Qwen3-4B | AQuA | 8 | 51.2795 | 48.8189 | 2.4606 |
| 44 | 86 | Qwen3-4B | AQuA | 8 | 51.6732 | 48.4252 | 3.248 |
| 45 | 87 | Qwen3-4B | AQuA | 8 | 51.1811 | 47.6378 | 3.5433 |
| 46 | 88 | Qwen3-4B | AQuA | 8 | 50.4921 | 46.9488 | 3.5433 |
| 47 | 89 | Qwen3-4B | AQuA | 8 | 52.6575 | 47.5394 | 5.1181 |
| 48 | 90 | Qwen3-4B | AQuA | 8 | 51.8701 | 46.4567 | 5.4134 |
| 49 | 91 | Qwen3-4B | AQuA | 8 | 51.5748 | 47.1457 | 4.4291 |
| 0 | 42 | Qwen3-4B | AQuA | 12 | 51.7717 | 47.5066 | 4.2651 |
| 1 | 43 | Qwen3-4B | AQuA | 12 | 51.8373 | 47.769 | 4.0682 |
| 2 | 44 | Qwen3-4B | AQuA | 12 | 52.4278 | 47.6378 | 4.79 |
| 3 | 45 | Qwen3-4B | AQuA | 12 | 51.4436 | 48.0971 | 3.3465 |
| 4 | 46 | Qwen3-4B | AQuA | 12 | 51.1811 | 47.3753 | 3.8058 |
| 5 | 47 | Qwen3-4B | AQuA | 12 | 53.0184 | 46.8504 | 6.168 |
| 6 | 48 | Qwen3-4B | AQuA | 12 | 52.7559 | 47.6378 | 5.1181 |
| 7 | 49 | Qwen3-4B | AQuA | 12 | 51.8373 | 46.916 | 4.9213 |
| 8 | 50 | Qwen3-4B | AQuA | 12 | 51.2467 | 47.9003 | 3.3465 |
| 9 | 51 | Qwen3-4B | AQuA | 12 | 51.9685 | 46.8504 | 5.1181 |
| 10 | 52 | Qwen3-4B | AQuA | 12 | 51.1811 | 47.0472 | 4.1339 |
| 11 | 53 | Qwen3-4B | AQuA | 12 | 52.8215 | 47.0472 | 5.7743 |
| 12 | 54 | Qwen3-4B | AQuA | 12 | 52.5591 | 46.2598 | 6.2992 |
| 13 | 55 | Qwen3-4B | AQuA | 12 | 51.5748 | 47.4409 | 4.1339 |
| 14 | 56 | Qwen3-4B | AQuA | 12 | 51.5092 | 47.4409 | 4.0682 |
| 15 | 57 | Qwen3-4B | AQuA | 12 | 52.0997 | 47.0472 | 5.0525 |
| 16 | 58 | Qwen3-4B | AQuA | 12 | 51.4436 | 47.8346 | 3.6089 |
| 17 | 59 | Qwen3-4B | AQuA | 12 | 52.0341 | 47.1785 | 4.8556 |
| 18 | 60 | Qwen3-4B | AQuA | 12 | 53.084 | 46.3255 | 6.7585 |
| 19 | 61 | Qwen3-4B | AQuA | 12 | 51.4436 | 47.2441 | 4.1995 |
| 20 | 62 | Qwen3-4B | AQuA | 12 | 50.6562 | 47.4409 | 3.2152 |
| 21 | 63 | Qwen3-4B | AQuA | 12 | 52.1654 | 46.063 | 6.1024 |
| 22 | 64 | Qwen3-4B | AQuA | 12 | 51.6404 | 47.3753 | 4.2651 |
| 23 | 65 | Qwen3-4B | AQuA | 12 | 52.4278 | 47.3753 | 5.0525 |
| 24 | 66 | Qwen3-4B | AQuA | 12 | 51.7717 | 46.4567 | 5.315 |
| 25 | 67 | Qwen3-4B | AQuA | 12 | 51.4436 | 47.1785 | 4.2651 |
| 26 | 68 | Qwen3-4B | AQuA | 12 | 51.8373 | 47.5722 | 4.2651 |
| 27 | 69 | Qwen3-4B | AQuA | 12 | 50.853 | 47.2441 | 3.6089 |
| 28 | 70 | Qwen3-4B | AQuA | 12 | 50.3937 | 48.5564 | 1.8373 |
| 29 | 71 | Qwen3-4B | AQuA | 12 | 51.6404 | 47.1129 | 4.5276 |
| 30 | 72 | Qwen3-4B | AQuA | 12 | 52.3622 | 46.9816 | 5.3806 |
| 31 | 73 | Qwen3-4B | AQuA | 12 | 52.231 | 46.9816 | 5.2493 |
| 32 | 74 | Qwen3-4B | AQuA | 12 | 51.0499 | 47.4409 | 3.6089 |
| 33 | 75 | Qwen3-4B | AQuA | 12 | 51.4436 | 47.5066 | 3.937 |
| 34 | 76 | Qwen3-4B | AQuA | 12 | 51.5748 | 47.4409 | 4.1339 |
| 35 | 77 | Qwen3-4B | AQuA | 12 | 50.7218 | 46.2598 | 4.4619 |
| 36 | 78 | Qwen3-4B | AQuA | 12 | 51.706 | 47.3097 | 4.3963 |
| 37 | 79 | Qwen3-4B | AQuA | 12 | 51.4436 | 47.6378 | 3.8058 |
| 38 | 80 | Qwen3-4B | AQuA | 12 | 49.8688 | 48.294 | 1.5748 |
| 39 | 81 | Qwen3-4B | AQuA | 12 | 51.5748 | 48.294 | 3.2808 |
| 40 | 82 | Qwen3-4B | AQuA | 12 | 50.7874 | 48.0971 | 2.6903 |
| 41 | 83 | Qwen3-4B | AQuA | 12 | 50.7874 | 46.5223 | 4.2651 |
| 42 | 84 | Qwen3-4B | AQuA | 12 | 51.706 | 47.5066 | 4.1995 |
| 43 | 85 | Qwen3-4B | AQuA | 12 | 51.706 | 47.8346 | 3.8714 |
| 44 | 86 | Qwen3-4B | AQuA | 12 | 51.5092 | 47.5722 | 3.937 |
| 45 | 87 | Qwen3-4B | AQuA | 12 | 51.2467 | 46.6535 | 4.5932 |
| 46 | 88 | Qwen3-4B | AQuA | 12 | 52.0341 | 48.0315 | 4.0026 |
| 47 | 89 | Qwen3-4B | AQuA | 12 | 51.1811 | 47.3753 | 3.8058 |
| 48 | 90 | Qwen3-4B | AQuA | 12 | 51.7717 | 46.7192 | 5.0525 |
| 49 | 91 | Qwen3-4B | AQuA | 12 | 52.0341 | 46.8504 | 5.1837 |
| 0 | 42 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 1 | 43 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 2 | 44 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 3 | 45 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 4 | 46 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 5 | 47 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 6 | 48 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 7 | 49 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 8 | 50 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 9 | 51 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 10 | 52 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 11 | 53 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 12 | 54 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 13 | 55 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 14 | 56 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 15 | 57 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 16 | 58 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 17 | 59 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 18 | 60 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 19 | 61 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 20 | 62 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 21 | 63 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 22 | 64 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 23 | 65 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 24 | 66 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 25 | 67 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 26 | 68 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 27 | 69 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 28 | 70 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 29 | 71 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 30 | 72 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 31 | 73 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 32 | 74 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 33 | 75 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 34 | 76 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 35 | 77 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 36 | 78 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 37 | 79 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 38 | 80 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 39 | 81 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 40 | 82 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 41 | 83 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 42 | 84 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 43 | 85 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 44 | 86 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 45 | 87 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 46 | 88 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 47 | 89 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 48 | 90 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 49 | 91 | Qwen3-4B | AQuA | 16 | 51.378 | 47.4409 | 3.937 |
| 0 | 42 | Qwen3-4B | CommonsenseQA | 4 | 67.4038 | 68.878 | -1.4742 |
| 1 | 43 | Qwen3-4B | CommonsenseQA | 4 | 66.6257 | 69.2056 | -2.5799 |
| 2 | 44 | Qwen3-4B | CommonsenseQA | 4 | 67.3219 | 69.1237 | -1.8018 |
| 3 | 45 | Qwen3-4B | CommonsenseQA | 4 | 66.38 | 68.9189 | -2.5389 |
| 4 | 46 | Qwen3-4B | CommonsenseQA | 4 | 66.9943 | 69.4513 | -2.457 |
| 5 | 47 | Qwen3-4B | CommonsenseQA | 4 | 65.8886 | 69.2056 | -3.317 |
| 6 | 48 | Qwen3-4B | CommonsenseQA | 4 | 67.24 | 68.018 | -0.7781 |
| 7 | 49 | Qwen3-4B | CommonsenseQA | 4 | 67.1581 | 68.6323 | -1.4742 |
| 8 | 50 | Qwen3-4B | CommonsenseQA | 4 | 66.38 | 69.9017 | -3.5217 |
| 9 | 51 | Qwen3-4B | CommonsenseQA | 4 | 67.7314 | 69.3694 | -1.638 |
| 10 | 52 | Qwen3-4B | CommonsenseQA | 4 | 66.9124 | 70.3112 | -3.3989 |
| 11 | 53 | Qwen3-4B | CommonsenseQA | 4 | 66.8714 | 69.2465 | -2.3751 |
| 12 | 54 | Qwen3-4B | CommonsenseQA | 4 | 65.8477 | 70.2703 | -4.4226 |
| 13 | 55 | Qwen3-4B | CommonsenseQA | 4 | 66.7895 | 69.5332 | -2.7437 |
| 14 | 56 | Qwen3-4B | CommonsenseQA | 4 | 67.0762 | 67.4038 | -0.3276 |
| 15 | 57 | Qwen3-4B | CommonsenseQA | 4 | 67.24 | 68.2637 | -1.0238 |
| 16 | 58 | Qwen3-4B | CommonsenseQA | 4 | 66.4619 | 70.1884 | -3.7265 |
| 17 | 59 | Qwen3-4B | CommonsenseQA | 4 | 67.6085 | 69.3694 | -1.7609 |
| 18 | 60 | Qwen3-4B | CommonsenseQA | 4 | 67.3219 | 69.6151 | -2.2932 |
| 19 | 61 | Qwen3-4B | CommonsenseQA | 4 | 66.8714 | 69.2465 | -2.3751 |
| 20 | 62 | Qwen3-4B | CommonsenseQA | 4 | 67.1581 | 70.0246 | -2.8665 |
| 21 | 63 | Qwen3-4B | CommonsenseQA | 4 | 67.9361 | 68.7961 | -0.86 |
| 22 | 64 | Qwen3-4B | CommonsenseQA | 4 | 65.7248 | 69.8198 | -4.095 |
| 23 | 65 | Qwen3-4B | CommonsenseQA | 4 | 66.1753 | 68.4275 | -2.2523 |
| 24 | 66 | Qwen3-4B | CommonsenseQA | 4 | 66.3391 | 69.1646 | -2.8256 |
| 25 | 67 | Qwen3-4B | CommonsenseQA | 4 | 66.8714 | 69.6151 | -2.7437 |
| 26 | 68 | Qwen3-4B | CommonsenseQA | 4 | 67.1171 | 68.5094 | -1.3923 |
| 27 | 69 | Qwen3-4B | CommonsenseQA | 4 | 66.4619 | 68.7961 | -2.3342 |
| 28 | 70 | Qwen3-4B | CommonsenseQA | 4 | 66.6667 | 68.7961 | -2.1294 |
| 29 | 71 | Qwen3-4B | CommonsenseQA | 4 | 66.9124 | 69.4103 | -2.498 |
| 30 | 72 | Qwen3-4B | CommonsenseQA | 4 | 66.2981 | 69.9836 | -3.6855 |
| 31 | 73 | Qwen3-4B | CommonsenseQA | 4 | 66.7895 | 68.3456 | -1.5561 |
| 32 | 74 | Qwen3-4B | CommonsenseQA | 4 | 66.7895 | 67.6085 | -0.819 |
| 33 | 75 | Qwen3-4B | CommonsenseQA | 4 | 66.8714 | 69.9836 | -3.1122 |
| 34 | 76 | Qwen3-4B | CommonsenseQA | 4 | 66.0524 | 68.3047 | -2.2523 |
| 35 | 77 | Qwen3-4B | CommonsenseQA | 4 | 66.421 | 69.4103 | -2.9894 |
| 36 | 78 | Qwen3-4B | CommonsenseQA | 4 | 66.9124 | 68.5094 | -1.5971 |
| 37 | 79 | Qwen3-4B | CommonsenseQA | 4 | 66.9943 | 69.1237 | -2.1294 |
| 38 | 80 | Qwen3-4B | CommonsenseQA | 4 | 67.1581 | 69.9427 | -2.7846 |
| 39 | 81 | Qwen3-4B | CommonsenseQA | 4 | 66.9943 | 68.7551 | -1.7609 |
| 40 | 82 | Qwen3-4B | CommonsenseQA | 4 | 66.7895 | 69.0418 | -2.2523 |
| 41 | 83 | Qwen3-4B | CommonsenseQA | 4 | 66.6667 | 69.9836 | -3.317 |
| 42 | 84 | Qwen3-4B | CommonsenseQA | 4 | 65.561 | 69.0418 | -3.4808 |
| 43 | 85 | Qwen3-4B | CommonsenseQA | 4 | 66.3391 | 68.9189 | -2.5799 |
| 44 | 86 | Qwen3-4B | CommonsenseQA | 4 | 66.2572 | 69.0827 | -2.8256 |
| 45 | 87 | Qwen3-4B | CommonsenseQA | 4 | 65.8067 | 68.1409 | -2.3342 |
| 46 | 88 | Qwen3-4B | CommonsenseQA | 4 | 66.1343 | 68.1818 | -2.0475 |
| 47 | 89 | Qwen3-4B | CommonsenseQA | 4 | 67.0762 | 68.9189 | -1.8428 |
| 48 | 90 | Qwen3-4B | CommonsenseQA | 4 | 66.9124 | 69.4922 | -2.5799 |
| 49 | 91 | Qwen3-4B | CommonsenseQA | 4 | 67.5266 | 69.5741 | -2.0475 |
| 0 | 42 | Qwen3-4B | CommonsenseQA | 8 | 66.6462 | 69.3079 | -2.6618 |
| 1 | 43 | Qwen3-4B | CommonsenseQA | 8 | 66.2367 | 69.1851 | -2.9484 |
| 2 | 44 | Qwen3-4B | CommonsenseQA | 8 | 66.6052 | 69.7379 | -3.1327 |
| 3 | 45 | Qwen3-4B | CommonsenseQA | 8 | 66.6871 | 69.6355 | -2.9484 |
| 4 | 46 | Qwen3-4B | CommonsenseQA | 8 | 66.769 | 69.2056 | -2.4365 |
| 5 | 47 | Qwen3-4B | CommonsenseQA | 8 | 66.1138 | 69.0827 | -2.9689 |
| 6 | 48 | Qwen3-4B | CommonsenseQA | 8 | 67.0352 | 68.3866 | -1.3514 |
| 7 | 49 | Qwen3-4B | CommonsenseQA | 8 | 66.5643 | 69.1032 | -2.5389 |
| 8 | 50 | Qwen3-4B | CommonsenseQA | 8 | 66.7076 | 69.697 | -2.9894 |
| 9 | 51 | Qwen3-4B | CommonsenseQA | 8 | 67.1171 | 69.0008 | -1.8837 |
| 10 | 52 | Qwen3-4B | CommonsenseQA | 8 | 66.6257 | 69.7993 | -3.1736 |
| 11 | 53 | Qwen3-4B | CommonsenseQA | 8 | 66.7895 | 69.4513 | -2.6618 |
| 12 | 54 | Qwen3-4B | CommonsenseQA | 8 | 66.5029 | 69.3284 | -2.8256 |
| 13 | 55 | Qwen3-4B | CommonsenseQA | 8 | 66.6257 | 69.6355 | -3.0098 |
| 14 | 56 | Qwen3-4B | CommonsenseQA | 8 | 66.8919 | 68.5094 | -1.6175 |
| 15 | 57 | Qwen3-4B | CommonsenseQA | 8 | 66.6667 | 69.5332 | -2.8665 |
| 16 | 58 | Qwen3-4B | CommonsenseQA | 8 | 66.6052 | 69.5536 | -2.9484 |
| 17 | 59 | Qwen3-4B | CommonsenseQA | 8 | 66.9943 | 70.1065 | -3.1122 |
| 18 | 60 | Qwen3-4B | CommonsenseQA | 8 | 66.5233 | 69.5127 | -2.9894 |
| 19 | 61 | Qwen3-4B | CommonsenseQA | 8 | 67.3014 | 69.0418 | -1.7404 |
| 20 | 62 | Qwen3-4B | CommonsenseQA | 8 | 67.1171 | 69.5332 | -2.4161 |
| 21 | 63 | Qwen3-4B | CommonsenseQA | 8 | 67.6699 | 69.1441 | -1.4742 |
| 22 | 64 | Qwen3-4B | CommonsenseQA | 8 | 66.9328 | 68.9803 | -2.0475 |
| 23 | 65 | Qwen3-4B | CommonsenseQA | 8 | 66.7076 | 69.0622 | -2.3546 |
| 24 | 66 | Qwen3-4B | CommonsenseQA | 8 | 66.3391 | 69.7174 | -3.3784 |
| 25 | 67 | Qwen3-4B | CommonsenseQA | 8 | 66.2981 | 69.3079 | -3.0098 |
| 26 | 68 | Qwen3-4B | CommonsenseQA | 8 | 66.9124 | 69.5741 | -2.6618 |
| 27 | 69 | Qwen3-4B | CommonsenseQA | 8 | 66.2981 | 68.837 | -2.5389 |
| 28 | 70 | Qwen3-4B | CommonsenseQA | 8 | 66.6052 | 69.4103 | -2.8051 |
| 29 | 71 | Qwen3-4B | CommonsenseQA | 8 | 66.9533 | 69.1237 | -2.1704 |
| 30 | 72 | Qwen3-4B | CommonsenseQA | 8 | 66.4824 | 69.7789 | -3.2965 |
| 31 | 73 | Qwen3-4B | CommonsenseQA | 8 | 66.9124 | 68.9394 | -2.027 |
| 32 | 74 | Qwen3-4B | CommonsenseQA | 8 | 66.7486 | 69.0418 | -2.2932 |
| 33 | 75 | Qwen3-4B | CommonsenseQA | 8 | 66.7486 | 69.4308 | -2.6822 |
| 34 | 76 | Qwen3-4B | CommonsenseQA | 8 | 66.5848 | 69.1851 | -2.6003 |
| 35 | 77 | Qwen3-4B | CommonsenseQA | 8 | 66.3186 | 69.0213 | -2.7027 |
| 36 | 78 | Qwen3-4B | CommonsenseQA | 8 | 66.4414 | 69.226 | -2.7846 |
| 37 | 79 | Qwen3-4B | CommonsenseQA | 8 | 67.0352 | 69.6355 | -2.6003 |
| 38 | 80 | Qwen3-4B | CommonsenseQA | 8 | 67.0966 | 69.1646 | -2.068 |
| 39 | 81 | Qwen3-4B | CommonsenseQA | 8 | 67.3219 | 69.0008 | -1.679 |
| 40 | 82 | Qwen3-4B | CommonsenseQA | 8 | 66.38 | 69.7174 | -3.3374 |
| 41 | 83 | Qwen3-4B | CommonsenseQA | 8 | 66.6257 | 69.6765 | -3.0508 |
| 42 | 84 | Qwen3-4B | CommonsenseQA | 8 | 65.8681 | 69.5127 | -3.6446 |
| 43 | 85 | Qwen3-4B | CommonsenseQA | 8 | 66.8714 | 69.3694 | -2.498 |
| 44 | 86 | Qwen3-4B | CommonsenseQA | 8 | 66.5438 | 68.7346 | -2.1908 |
| 45 | 87 | Qwen3-4B | CommonsenseQA | 8 | 66.5848 | 68.7961 | -2.2113 |
| 46 | 88 | Qwen3-4B | CommonsenseQA | 8 | 66.0524 | 69.4922 | -3.4398 |
| 47 | 89 | Qwen3-4B | CommonsenseQA | 8 | 66.5233 | 69.4717 | -2.9484 |
| 48 | 90 | Qwen3-4B | CommonsenseQA | 8 | 66.8509 | 69.8812 | -3.0303 |
| 49 | 91 | Qwen3-4B | CommonsenseQA | 8 | 67.0557 | 69.5332 | -2.4775 |
| 0 | 42 | Qwen3-4B | CommonsenseQA | 12 | 66.38 | 69.2329 | -2.8529 |
| 1 | 43 | Qwen3-4B | CommonsenseQA | 12 | 66.8032 | 68.837 | -2.0339 |
| 2 | 44 | Qwen3-4B | CommonsenseQA | 12 | 66.6257 | 69.5468 | -2.9211 |
| 3 | 45 | Qwen3-4B | CommonsenseQA | 12 | 66.3254 | 69.4103 | -3.0849 |
| 4 | 46 | Qwen3-4B | CommonsenseQA | 12 | 66.7486 | 69.2738 | -2.5253 |
| 5 | 47 | Qwen3-4B | CommonsenseQA | 12 | 66.1343 | 69.2329 | -3.0986 |
| 6 | 48 | Qwen3-4B | CommonsenseQA | 12 | 66.7076 | 69.151 | -2.4434 |
| 7 | 49 | Qwen3-4B | CommonsenseQA | 12 | 66.7486 | 69.0554 | -2.3069 |
| 8 | 50 | Qwen3-4B | CommonsenseQA | 12 | 66.5029 | 69.5468 | -3.044 |
| 9 | 51 | Qwen3-4B | CommonsenseQA | 12 | 66.421 | 69.1919 | -2.771 |
| 10 | 52 | Qwen3-4B | CommonsenseQA | 12 | 66.5438 | 69.5195 | -2.9757 |
| 11 | 53 | Qwen3-4B | CommonsenseQA | 12 | 66.4892 | 69.5878 | -3.0986 |
| 12 | 54 | Qwen3-4B | CommonsenseQA | 12 | 66.6121 | 69.3148 | -2.7027 |
| 13 | 55 | Qwen3-4B | CommonsenseQA | 12 | 66.3664 | 69.7516 | -3.3852 |
| 14 | 56 | Qwen3-4B | CommonsenseQA | 12 | 66.8714 | 69.151 | -2.2796 |
| 15 | 57 | Qwen3-4B | CommonsenseQA | 12 | 66.8168 | 68.878 | -2.0612 |
| 16 | 58 | Qwen3-4B | CommonsenseQA | 12 | 66.2845 | 69.5059 | -3.2214 |
| 17 | 59 | Qwen3-4B | CommonsenseQA | 12 | 66.5984 | 69.7379 | -3.1395 |
| 18 | 60 | Qwen3-4B | CommonsenseQA | 12 | 66.8168 | 69.4103 | -2.5935 |
| 19 | 61 | Qwen3-4B | CommonsenseQA | 12 | 66.5438 | 69.2192 | -2.6754 |
| 20 | 62 | Qwen3-4B | CommonsenseQA | 12 | 66.7076 | 69.11 | -2.4024 |
| 21 | 63 | Qwen3-4B | CommonsenseQA | 12 | 66.7213 | 69.3421 | -2.6208 |
| 22 | 64 | Qwen3-4B | CommonsenseQA | 12 | 66.6667 | 69.2738 | -2.6072 |
| 23 | 65 | Qwen3-4B | CommonsenseQA | 12 | 66.7622 | 69.7379 | -2.9757 |
| 24 | 66 | Qwen3-4B | CommonsenseQA | 12 | 66.694 | 69.1919 | -2.498 |
| 25 | 67 | Qwen3-4B | CommonsenseQA | 12 | 66.2845 | 69.7379 | -3.4535 |
| 26 | 68 | Qwen3-4B | CommonsenseQA | 12 | 66.4619 | 69.7106 | -3.2487 |
| 27 | 69 | Qwen3-4B | CommonsenseQA | 12 | 66.6667 | 69.2602 | -2.5935 |
| 28 | 70 | Qwen3-4B | CommonsenseQA | 12 | 66.926 | 69.383 | -2.457 |
| 29 | 71 | Qwen3-4B | CommonsenseQA | 12 | 66.8305 | 69.4649 | -2.6345 |
| 30 | 72 | Qwen3-4B | CommonsenseQA | 12 | 66.3937 | 69.6014 | -3.2078 |
| 31 | 73 | Qwen3-4B | CommonsenseQA | 12 | 66.6394 | 69.4649 | -2.8256 |
| 32 | 74 | Qwen3-4B | CommonsenseQA | 12 | 66.3391 | 69.2738 | -2.9348 |
| 33 | 75 | Qwen3-4B | CommonsenseQA | 12 | 66.5302 | 69.0145 | -2.4843 |
| 34 | 76 | Qwen3-4B | CommonsenseQA | 12 | 66.5711 | 69.2465 | -2.6754 |
| 35 | 77 | Qwen3-4B | CommonsenseQA | 12 | 66.5711 | 69.3694 | -2.7983 |
| 36 | 78 | Qwen3-4B | CommonsenseQA | 12 | 66.5029 | 69.2465 | -2.7437 |
| 37 | 79 | Qwen3-4B | CommonsenseQA | 12 | 66.4483 | 69.424 | -2.9757 |
| 38 | 80 | Qwen3-4B | CommonsenseQA | 12 | 66.7076 | 69.5059 | -2.7983 |
| 39 | 81 | Qwen3-4B | CommonsenseQA | 12 | 67.1717 | 69.3011 | -2.1294 |
| 40 | 82 | Qwen3-4B | CommonsenseQA | 12 | 66.6257 | 69.3148 | -2.6891 |
| 41 | 83 | Qwen3-4B | CommonsenseQA | 12 | 66.7076 | 69.6287 | -2.9211 |
| 42 | 84 | Qwen3-4B | CommonsenseQA | 12 | 66.3254 | 69.5468 | -3.2214 |
| 43 | 85 | Qwen3-4B | CommonsenseQA | 12 | 66.8032 | 69.383 | -2.5799 |
| 44 | 86 | Qwen3-4B | CommonsenseQA | 12 | 66.421 | 69.4786 | -3.0576 |
| 45 | 87 | Qwen3-4B | CommonsenseQA | 12 | 66.7213 | 69.1919 | -2.4707 |
| 46 | 88 | Qwen3-4B | CommonsenseQA | 12 | 66.3391 | 69.4922 | -3.1532 |
| 47 | 89 | Qwen3-4B | CommonsenseQA | 12 | 66.6121 | 69.3421 | -2.73 |
| 48 | 90 | Qwen3-4B | CommonsenseQA | 12 | 66.7622 | 69.2738 | -2.5116 |
| 49 | 91 | Qwen3-4B | CommonsenseQA | 12 | 66.7622 | 69.6014 | -2.8392 |
| 0 | 42 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 1 | 43 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 2 | 44 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 3 | 45 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 4 | 46 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 5 | 47 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 6 | 48 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 7 | 49 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 8 | 50 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 9 | 51 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 10 | 52 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 11 | 53 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 12 | 54 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 13 | 55 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 14 | 56 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 15 | 57 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 16 | 58 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 17 | 59 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 18 | 60 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 19 | 61 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 20 | 62 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 21 | 63 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 22 | 64 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 23 | 65 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 24 | 66 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 25 | 67 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 26 | 68 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 27 | 69 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 28 | 70 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 29 | 71 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 30 | 72 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 31 | 73 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 32 | 74 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 33 | 75 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 34 | 76 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 35 | 77 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 36 | 78 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 37 | 79 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 38 | 80 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 39 | 81 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 40 | 82 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 41 | 83 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 42 | 84 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 43 | 85 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 44 | 86 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 45 | 87 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 46 | 88 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 47 | 89 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 48 | 90 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 49 | 91 | Qwen3-4B | CommonsenseQA | 16 | 66.5438 | 69.4922 | -2.9484 |
| 0 | 42 | Qwen3-4B | GPQA | 4 | 56.3616 | 59.2634 | -2.9018 |
| 1 | 43 | Qwen3-4B | GPQA | 4 | 53.7946 | 56.25 | -2.4554 |
| 2 | 44 | Qwen3-4B | GPQA | 4 | 54.7991 | 60.0446 | -5.2455 |
| 3 | 45 | Qwen3-4B | GPQA | 4 | 56.25 | 59.7098 | -3.4598 |
| 4 | 46 | Qwen3-4B | GPQA | 4 | 57.0312 | 59.5982 | -2.567 |
| 5 | 47 | Qwen3-4B | GPQA | 4 | 55.2455 | 59.4866 | -4.2411 |
| 6 | 48 | Qwen3-4B | GPQA | 4 | 57.0312 | 59.1518 | -2.1205 |
| 7 | 49 | Qwen3-4B | GPQA | 4 | 56.5848 | 59.933 | -3.3482 |
| 8 | 50 | Qwen3-4B | GPQA | 4 | 56.6964 | 61.8304 | -5.1339 |
| 9 | 51 | Qwen3-4B | GPQA | 4 | 58.7054 | 58.9286 | -0.2232 |
| 10 | 52 | Qwen3-4B | GPQA | 4 | 56.6964 | 59.2634 | -2.567 |
| 11 | 53 | Qwen3-4B | GPQA | 4 | 57.7009 | 59.7098 | -2.0089 |
| 12 | 54 | Qwen3-4B | GPQA | 4 | 59.2634 | 59.7098 | -0.4464 |
| 13 | 55 | Qwen3-4B | GPQA | 4 | 53.0134 | 60.2679 | -7.2545 |
| 14 | 56 | Qwen3-4B | GPQA | 4 | 55.2455 | 61.0491 | -5.8036 |
| 15 | 57 | Qwen3-4B | GPQA | 4 | 54.2411 | 61.942 | -7.7009 |
| 16 | 58 | Qwen3-4B | GPQA | 4 | 55.1339 | 58.5938 | -3.4598 |
| 17 | 59 | Qwen3-4B | GPQA | 4 | 56.9196 | 62.5 | -5.5804 |
| 18 | 60 | Qwen3-4B | GPQA | 4 | 55.692 | 60.9375 | -5.2455 |
| 19 | 61 | Qwen3-4B | GPQA | 4 | 57.1429 | 60.4911 | -3.3482 |
| 20 | 62 | Qwen3-4B | GPQA | 4 | 56.6964 | 60.4911 | -3.7946 |
| 21 | 63 | Qwen3-4B | GPQA | 4 | 58.4821 | 60.2679 | -1.7857 |
| 22 | 64 | Qwen3-4B | GPQA | 4 | 56.4732 | 59.375 | -2.9018 |
| 23 | 65 | Qwen3-4B | GPQA | 4 | 55.9152 | 59.1518 | -3.2366 |
| 24 | 66 | Qwen3-4B | GPQA | 4 | 57.1429 | 58.1473 | -1.0045 |
| 25 | 67 | Qwen3-4B | GPQA | 4 | 53.125 | 61.4955 | -8.3705 |
| 26 | 68 | Qwen3-4B | GPQA | 4 | 56.25 | 60.9375 | -4.6875 |
| 27 | 69 | Qwen3-4B | GPQA | 4 | 55.9152 | 59.933 | -4.0179 |
| 28 | 70 | Qwen3-4B | GPQA | 4 | 54.6875 | 61.942 | -7.2545 |
| 29 | 71 | Qwen3-4B | GPQA | 4 | 57.3661 | 57.2545 | 0.1116 |
| 30 | 72 | Qwen3-4B | GPQA | 4 | 56.1384 | 58.3705 | -2.2321 |
| 31 | 73 | Qwen3-4B | GPQA | 4 | 57.5893 | 58.5938 | -1.0045 |
| 32 | 74 | Qwen3-4B | GPQA | 4 | 56.5848 | 57.4777 | -0.8929 |
| 33 | 75 | Qwen3-4B | GPQA | 4 | 56.25 | 59.375 | -3.125 |
| 34 | 76 | Qwen3-4B | GPQA | 4 | 56.25 | 59.933 | -3.683 |
| 35 | 77 | Qwen3-4B | GPQA | 4 | 56.0268 | 59.8214 | -3.7946 |
| 36 | 78 | Qwen3-4B | GPQA | 4 | 55.1339 | 60.6027 | -5.4688 |
| 37 | 79 | Qwen3-4B | GPQA | 4 | 56.1384 | 60.2679 | -4.1295 |
| 38 | 80 | Qwen3-4B | GPQA | 4 | 55.3571 | 60.0446 | -4.6875 |
| 39 | 81 | Qwen3-4B | GPQA | 4 | 53.4598 | 61.1607 | -7.7009 |
| 40 | 82 | Qwen3-4B | GPQA | 4 | 56.4732 | 63.2812 | -6.808 |
| 41 | 83 | Qwen3-4B | GPQA | 4 | 57.2545 | 58.7054 | -1.4509 |
| 42 | 84 | Qwen3-4B | GPQA | 4 | 56.1384 | 60.3795 | -4.2411 |
| 43 | 85 | Qwen3-4B | GPQA | 4 | 57.1429 | 62.5 | -5.3571 |
| 44 | 86 | Qwen3-4B | GPQA | 4 | 56.808 | 59.5982 | -2.7902 |
| 45 | 87 | Qwen3-4B | GPQA | 4 | 54.9107 | 60.2679 | -5.3571 |
| 46 | 88 | Qwen3-4B | GPQA | 4 | 57.5893 | 60.7143 | -3.125 |
| 47 | 89 | Qwen3-4B | GPQA | 4 | 55.1339 | 59.8214 | -4.6875 |
| 48 | 90 | Qwen3-4B | GPQA | 4 | 57.2545 | 62.1652 | -4.9107 |
| 49 | 91 | Qwen3-4B | GPQA | 4 | 58.0357 | 60.9375 | -2.9018 |
| 0 | 42 | Qwen3-4B | GPQA | 8 | 56.529 | 60.9375 | -4.4085 |
| 1 | 43 | Qwen3-4B | GPQA | 8 | 55.2455 | 60.2679 | -5.0223 |
| 2 | 44 | Qwen3-4B | GPQA | 8 | 55.0223 | 61.1607 | -6.1384 |
| 3 | 45 | Qwen3-4B | GPQA | 8 | 56.1384 | 59.375 | -3.2366 |
| 4 | 46 | Qwen3-4B | GPQA | 8 | 54.4643 | 61.3839 | -6.9196 |
| 5 | 47 | Qwen3-4B | GPQA | 8 | 56.1384 | 59.4866 | -3.3482 |
| 6 | 48 | Qwen3-4B | GPQA | 8 | 56.529 | 59.5424 | -3.0134 |
| 7 | 49 | Qwen3-4B | GPQA | 8 | 55.8594 | 60.6027 | -4.7433 |
| 8 | 50 | Qwen3-4B | GPQA | 8 | 56.808 | 59.654 | -2.846 |
| 9 | 51 | Qwen3-4B | GPQA | 8 | 57.4777 | 58.9844 | -1.5067 |
| 10 | 52 | Qwen3-4B | GPQA | 8 | 56.6964 | 60.2679 | -3.5714 |
| 11 | 53 | Qwen3-4B | GPQA | 8 | 56.1384 | 60.2121 | -4.0737 |
| 12 | 54 | Qwen3-4B | GPQA | 8 | 57.0312 | 61.7188 | -4.6875 |
| 13 | 55 | Qwen3-4B | GPQA | 8 | 55.692 | 60.2121 | -4.5201 |
| 14 | 56 | Qwen3-4B | GPQA | 8 | 56.3616 | 60.4911 | -4.1295 |
| 15 | 57 | Qwen3-4B | GPQA | 8 | 56.0268 | 61.1607 | -5.1339 |
| 16 | 58 | Qwen3-4B | GPQA | 8 | 55.692 | 60.2121 | -4.5201 |
| 17 | 59 | Qwen3-4B | GPQA | 8 | 55.3013 | 61.4955 | -6.1942 |
| 18 | 60 | Qwen3-4B | GPQA | 8 | 56.1384 | 60.8817 | -4.7433 |
| 19 | 61 | Qwen3-4B | GPQA | 8 | 56.529 | 60.0446 | -3.5156 |
| 20 | 62 | Qwen3-4B | GPQA | 8 | 56.529 | 60.3795 | -3.8504 |
| 21 | 63 | Qwen3-4B | GPQA | 8 | 57.7567 | 60.4353 | -2.6786 |
| 22 | 64 | Qwen3-4B | GPQA | 8 | 55.6362 | 60.4911 | -4.8549 |
| 23 | 65 | Qwen3-4B | GPQA | 8 | 55.8594 | 59.5982 | -3.7388 |
| 24 | 66 | Qwen3-4B | GPQA | 8 | 57.1429 | 59.8772 | -2.7344 |
| 25 | 67 | Qwen3-4B | GPQA | 8 | 55.4129 | 60.7143 | -5.3013 |
| 26 | 68 | Qwen3-4B | GPQA | 8 | 55.5246 | 61.9978 | -6.4732 |
| 27 | 69 | Qwen3-4B | GPQA | 8 | 56.5848 | 59.5424 | -2.9576 |
| 28 | 70 | Qwen3-4B | GPQA | 8 | 56.4732 | 60.4353 | -3.9621 |
| 29 | 71 | Qwen3-4B | GPQA | 8 | 56.1942 | 58.817 | -2.6228 |
| 30 | 72 | Qwen3-4B | GPQA | 8 | 55.8036 | 60.0446 | -4.2411 |
| 31 | 73 | Qwen3-4B | GPQA | 8 | 55.692 | 60.0446 | -4.3527 |
| 32 | 74 | Qwen3-4B | GPQA | 8 | 56.529 | 59.096 | -2.567 |
| 33 | 75 | Qwen3-4B | GPQA | 8 | 56.6406 | 60.5469 | -3.9062 |
| 34 | 76 | Qwen3-4B | GPQA | 8 | 55.3571 | 59.8214 | -4.4643 |
| 35 | 77 | Qwen3-4B | GPQA | 8 | 56.4732 | 60.2121 | -3.7388 |
| 36 | 78 | Qwen3-4B | GPQA | 8 | 56.1384 | 59.4308 | -3.2924 |
| 37 | 79 | Qwen3-4B | GPQA | 8 | 56.4174 | 60.8259 | -4.4085 |
| 38 | 80 | Qwen3-4B | GPQA | 8 | 54.7433 | 60.5469 | -5.8036 |
| 39 | 81 | Qwen3-4B | GPQA | 8 | 55.4688 | 61.2165 | -5.7478 |
| 40 | 82 | Qwen3-4B | GPQA | 8 | 56.0826 | 60.2679 | -4.1853 |
| 41 | 83 | Qwen3-4B | GPQA | 8 | 55.8594 | 60.3795 | -4.5201 |
| 42 | 84 | Qwen3-4B | GPQA | 8 | 54.8549 | 61.3281 | -6.4732 |
| 43 | 85 | Qwen3-4B | GPQA | 8 | 55.971 | 60.6585 | -4.6875 |
| 44 | 86 | Qwen3-4B | GPQA | 8 | 56.5848 | 58.7054 | -2.1205 |
| 45 | 87 | Qwen3-4B | GPQA | 8 | 54.1295 | 61.9978 | -7.8683 |
| 46 | 88 | Qwen3-4B | GPQA | 8 | 56.0826 | 60.8259 | -4.7433 |
| 47 | 89 | Qwen3-4B | GPQA | 8 | 56.1384 | 60.5469 | -4.4085 |
| 48 | 90 | Qwen3-4B | GPQA | 8 | 55.5804 | 61.3839 | -5.8036 |
| 49 | 91 | Qwen3-4B | GPQA | 8 | 57.0312 | 61.1607 | -4.1295 |
| 0 | 42 | Qwen3-4B | GPQA | 12 | 56.1012 | 61.3095 | -5.2083 |
| 1 | 43 | Qwen3-4B | GPQA | 12 | 56.4732 | 60.3423 | -3.869 |
| 2 | 44 | Qwen3-4B | GPQA | 12 | 55.1339 | 60.5283 | -5.3943 |
| 3 | 45 | Qwen3-4B | GPQA | 12 | 56.25 | 60.1935 | -3.9435 |
| 4 | 46 | Qwen3-4B | GPQA | 12 | 56.064 | 60.4539 | -4.3899 |
| 5 | 47 | Qwen3-4B | GPQA | 12 | 55.9152 | 59.6354 | -3.7202 |
| 6 | 48 | Qwen3-4B | GPQA | 12 | 56.1384 | 60.0818 | -3.9435 |
| 7 | 49 | Qwen3-4B | GPQA | 12 | 56.0268 | 60.2307 | -4.2039 |
| 8 | 50 | Qwen3-4B | GPQA | 12 | 56.7336 | 60.4911 | -3.7574 |
| 9 | 51 | Qwen3-4B | GPQA | 12 | 56.2872 | 60.0446 | -3.7574 |
| 10 | 52 | Qwen3-4B | GPQA | 12 | 56.7336 | 60.8631 | -4.1295 |
| 11 | 53 | Qwen3-4B | GPQA | 12 | 56.8452 | 60.5283 | -3.683 |
| 12 | 54 | Qwen3-4B | GPQA | 12 | 57.1801 | 61.0119 | -3.8318 |
| 13 | 55 | Qwen3-4B | GPQA | 12 | 56.064 | 60.7887 | -4.7247 |
| 14 | 56 | Qwen3-4B | GPQA | 12 | 56.6592 | 59.7842 | -3.125 |
| 15 | 57 | Qwen3-4B | GPQA | 12 | 56.436 | 60.5283 | -4.0923 |
| 16 | 58 | Qwen3-4B | GPQA | 12 | 55.7664 | 60.4911 | -4.7247 |
| 17 | 59 | Qwen3-4B | GPQA | 12 | 55.5432 | 60.8631 | -5.3199 |
| 18 | 60 | Qwen3-4B | GPQA | 12 | 55.6176 | 60.6771 | -5.0595 |
| 19 | 61 | Qwen3-4B | GPQA | 12 | 55.8408 | 60.2679 | -4.4271 |
| 20 | 62 | Qwen3-4B | GPQA | 12 | 56.4732 | 60.4911 | -4.0179 |
| 21 | 63 | Qwen3-4B | GPQA | 12 | 56.0268 | 60.0818 | -4.0551 |
| 22 | 64 | Qwen3-4B | GPQA | 12 | 55.6176 | 60.1935 | -4.5759 |
| 23 | 65 | Qwen3-4B | GPQA | 12 | 56.1012 | 60.0446 | -3.9435 |
| 24 | 66 | Qwen3-4B | GPQA | 12 | 56.2872 | 60.3795 | -4.0923 |
| 25 | 67 | Qwen3-4B | GPQA | 12 | 55.7664 | 60.2307 | -4.4643 |
| 26 | 68 | Qwen3-4B | GPQA | 12 | 55.9896 | 61.4211 | -5.4315 |
| 27 | 69 | Qwen3-4B | GPQA | 12 | 55.9896 | 60.119 | -4.1295 |
| 28 | 70 | Qwen3-4B | GPQA | 12 | 55.8036 | 60.4167 | -4.6131 |
| 29 | 71 | Qwen3-4B | GPQA | 12 | 56.3244 | 59.8586 | -3.5342 |
| 30 | 72 | Qwen3-4B | GPQA | 12 | 56.1012 | 60.1935 | -4.0923 |
| 31 | 73 | Qwen3-4B | GPQA | 12 | 56.064 | 59.9702 | -3.9063 |
| 32 | 74 | Qwen3-4B | GPQA | 12 | 56.5104 | 60.2679 | -3.7574 |
| 33 | 75 | Qwen3-4B | GPQA | 12 | 56.1384 | 60.0446 | -3.9062 |
| 34 | 76 | Qwen3-4B | GPQA | 12 | 55.8036 | 60.3423 | -4.5387 |
| 35 | 77 | Qwen3-4B | GPQA | 12 | 56.6592 | 59.9702 | -3.311 |
| 36 | 78 | Qwen3-4B | GPQA | 12 | 55.8408 | 60.5283 | -4.6875 |
| 37 | 79 | Qwen3-4B | GPQA | 12 | 55.9896 | 60.3795 | -4.3899 |
| 38 | 80 | Qwen3-4B | GPQA | 12 | 54.8735 | 61.0863 | -6.2128 |
| 39 | 81 | Qwen3-4B | GPQA | 12 | 55.2455 | 60.9375 | -5.692 |
| 40 | 82 | Qwen3-4B | GPQA | 12 | 56.0268 | 61.0491 | -5.0223 |
| 41 | 83 | Qwen3-4B | GPQA | 12 | 55.5804 | 60.119 | -4.5387 |
| 42 | 84 | Qwen3-4B | GPQA | 12 | 55.5432 | 61.0119 | -5.4688 |
| 43 | 85 | Qwen3-4B | GPQA | 12 | 56.3244 | 60.5655 | -4.2411 |
| 44 | 86 | Qwen3-4B | GPQA | 12 | 56.8824 | 59.7842 | -2.9018 |
| 45 | 87 | Qwen3-4B | GPQA | 12 | 56.064 | 60.6399 | -4.5759 |
| 46 | 88 | Qwen3-4B | GPQA | 12 | 55.6176 | 60.119 | -4.5015 |
| 47 | 89 | Qwen3-4B | GPQA | 12 | 55.0223 | 61.3095 | -6.2872 |
| 48 | 90 | Qwen3-4B | GPQA | 12 | 56.1384 | 60.7887 | -4.6503 |
| 49 | 91 | Qwen3-4B | GPQA | 12 | 56.5476 | 60.3051 | -3.7574 |
| 0 | 42 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 1 | 43 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 2 | 44 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 3 | 45 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 4 | 46 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 5 | 47 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 6 | 48 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 7 | 49 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 8 | 50 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 9 | 51 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 10 | 52 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 11 | 53 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 12 | 54 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 13 | 55 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 14 | 56 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 15 | 57 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 16 | 58 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 17 | 59 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 18 | 60 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 19 | 61 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 20 | 62 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 21 | 63 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 22 | 64 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 23 | 65 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 24 | 66 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 25 | 67 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 26 | 68 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 27 | 69 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 28 | 70 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 29 | 71 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 30 | 72 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 31 | 73 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 32 | 74 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 33 | 75 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 34 | 76 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 35 | 77 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 36 | 78 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 37 | 79 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 38 | 80 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 39 | 81 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 40 | 82 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 41 | 83 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 42 | 84 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 43 | 85 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 44 | 86 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 45 | 87 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 46 | 88 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 47 | 89 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 48 | 90 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 49 | 91 | Qwen3-4B | GPQA | 16 | 56.1384 | 60.4353 | -4.2969 |
| 0 | 42 | Qwen3-4B | GSM8K | 4 | 80.8567 | 67.3616 | 13.4951 |
| 1 | 43 | Qwen3-4B | GSM8K | 4 | 80.1365 | 67.2479 | 12.8886 |
| 2 | 44 | Qwen3-4B | GSM8K | 4 | 80.2123 | 66.7172 | 13.4951 |
| 3 | 45 | Qwen3-4B | GSM8K | 4 | 81.539 | 66.5656 | 14.9735 |
| 4 | 46 | Qwen3-4B | GSM8K | 4 | 80.5534 | 67.2479 | 13.3055 |
| 5 | 47 | Qwen3-4B | GSM8K | 4 | 79.6437 | 65.58 | 14.0637 |
| 6 | 48 | Qwen3-4B | GSM8K | 4 | 80.1365 | 66.8688 | 13.2676 |
| 7 | 49 | Qwen3-4B | GSM8K | 4 | 81.1221 | 67.6649 | 13.4572 |
| 8 | 50 | Qwen3-4B | GSM8K | 4 | 80.743 | 66.793 | 13.95 |
| 9 | 51 | Qwen3-4B | GSM8K | 4 | 81.6528 | 66.7551 | 14.8976 |
| 10 | 52 | Qwen3-4B | GSM8K | 4 | 80.5534 | 66.3381 | 14.2153 |
| 11 | 53 | Qwen3-4B | GSM8K | 4 | 80.6293 | 66.3002 | 14.329 |
| 12 | 54 | Qwen3-4B | GSM8K | 4 | 81.7665 | 67.2858 | 14.4807 |
| 13 | 55 | Qwen3-4B | GSM8K | 4 | 80.9325 | 66.1107 | 14.8218 |
| 14 | 56 | Qwen3-4B | GSM8K | 4 | 81.2737 | 67.627 | 13.6467 |
| 15 | 57 | Qwen3-4B | GSM8K | 4 | 80.5914 | 67.6649 | 12.9265 |
| 16 | 58 | Qwen3-4B | GSM8K | 4 | 79.113 | 66.4898 | 12.6232 |
| 17 | 59 | Qwen3-4B | GSM8K | 4 | 80.0227 | 65.8832 | 14.1395 |
| 18 | 60 | Qwen3-4B | GSM8K | 4 | 80.6672 | 67.21 | 13.4572 |
| 19 | 61 | Qwen3-4B | GSM8K | 4 | 80.326 | 67.1342 | 13.1918 |
| 20 | 62 | Qwen3-4B | GSM8K | 4 | 79.7953 | 67.3237 | 12.4716 |
| 21 | 63 | Qwen3-4B | GSM8K | 4 | 80.7809 | 66.2623 | 14.5186 |
| 22 | 64 | Qwen3-4B | GSM8K | 4 | 80.8946 | 66.1107 | 14.7839 |
| 23 | 65 | Qwen3-4B | GSM8K | 4 | 80.326 | 66.6414 | 13.6846 |
| 24 | 66 | Qwen3-4B | GSM8K | 4 | 79.492 | 67.0584 | 12.4337 |
| 25 | 67 | Qwen3-4B | GSM8K | 4 | 80.2502 | 66.8309 | 13.4193 |
| 26 | 68 | Qwen3-4B | GSM8K | 4 | 81.6907 | 66.1107 | 15.58 |
| 27 | 69 | Qwen3-4B | GSM8K | 4 | 80.7051 | 66.0728 | 14.6323 |
| 28 | 70 | Qwen3-4B | GSM8K | 4 | 80.0227 | 65.5421 | 14.4807 |
| 29 | 71 | Qwen3-4B | GSM8K | 4 | 80.9704 | 66.376 | 14.5944 |
| 30 | 72 | Qwen3-4B | GSM8K | 4 | 81.2737 | 66.9067 | 14.3669 |
| 31 | 73 | Qwen3-4B | GSM8K | 4 | 81.6907 | 67.2479 | 14.4428 |
| 32 | 74 | Qwen3-4B | GSM8K | 4 | 81.7286 | 65.4663 | 16.2623 |
| 33 | 75 | Qwen3-4B | GSM8K | 4 | 81.0083 | 66.6035 | 14.4049 |
| 34 | 76 | Qwen3-4B | GSM8K | 4 | 80.1365 | 66.5277 | 13.6088 |
| 35 | 77 | Qwen3-4B | GSM8K | 4 | 81.0842 | 66.6035 | 14.4807 |
| 36 | 78 | Qwen3-4B | GSM8K | 4 | 80.9704 | 66.1865 | 14.7839 |
| 37 | 79 | Qwen3-4B | GSM8K | 4 | 82.1077 | 66.9826 | 15.1251 |
| 38 | 80 | Qwen3-4B | GSM8K | 4 | 80.6672 | 66.6414 | 14.0258 |
| 39 | 81 | Qwen3-4B | GSM8K | 4 | 80.2881 | 67.1721 | 13.116 |
| 40 | 82 | Qwen3-4B | GSM8K | 4 | 79.8332 | 66.1865 | 13.6467 |
| 41 | 83 | Qwen3-4B | GSM8K | 4 | 80.5914 | 66.9067 | 13.6846 |
| 42 | 84 | Qwen3-4B | GSM8K | 4 | 80.2123 | 65.6558 | 14.5565 |
| 43 | 85 | Qwen3-4B | GSM8K | 4 | 80.326 | 67.2479 | 13.0781 |
| 44 | 86 | Qwen3-4B | GSM8K | 4 | 79.9848 | 67.3995 | 12.5853 |
| 45 | 87 | Qwen3-4B | GSM8K | 4 | 80.1365 | 66.4898 | 13.6467 |
| 46 | 88 | Qwen3-4B | GSM8K | 4 | 79.7195 | 67.8544 | 11.865 |
| 47 | 89 | Qwen3-4B | GSM8K | 4 | 80.7809 | 66.6793 | 14.1016 |
| 48 | 90 | Qwen3-4B | GSM8K | 4 | 81.16 | 66.4898 | 14.6702 |
| 49 | 91 | Qwen3-4B | GSM8K | 4 | 80.2123 | 67.2479 | 12.9644 |
| 0 | 42 | Qwen3-4B | GSM8K | 8 | 81.3874 | 65.5989 | 15.7885 |
| 1 | 43 | Qwen3-4B | GSM8K | 8 | 81.6907 | 65.4852 | 16.2055 |
| 2 | 44 | Qwen3-4B | GSM8K | 8 | 81.1031 | 66.0538 | 15.0493 |
| 3 | 45 | Qwen3-4B | GSM8K | 8 | 81.5959 | 66.2244 | 15.3715 |
| 4 | 46 | Qwen3-4B | GSM8K | 8 | 81.6528 | 65.5231 | 16.1296 |
| 5 | 47 | Qwen3-4B | GSM8K | 8 | 80.7051 | 65.978 | 14.7271 |
| 6 | 48 | Qwen3-4B | GSM8K | 8 | 80.6861 | 66.5087 | 14.1774 |
| 7 | 49 | Qwen3-4B | GSM8K | 8 | 81.8234 | 66.0349 | 15.7885 |
| 8 | 50 | Qwen3-4B | GSM8K | 8 | 81.0842 | 65.8453 | 15.2388 |
| 9 | 51 | Qwen3-4B | GSM8K | 8 | 82.0697 | 66.4329 | 15.6368 |
| 10 | 52 | Qwen3-4B | GSM8K | 8 | 81.539 | 65.5042 | 16.0349 |
| 11 | 53 | Qwen3-4B | GSM8K | 8 | 81.2547 | 65.6748 | 15.58 |
| 12 | 54 | Qwen3-4B | GSM8K | 8 | 81.1221 | 66.3192 | 14.8029 |
| 13 | 55 | Qwen3-4B | GSM8K | 8 | 81.4822 | 65.5042 | 15.978 |
| 14 | 56 | Qwen3-4B | GSM8K | 8 | 81.558 | 66.5277 | 15.0303 |
| 15 | 57 | Qwen3-4B | GSM8K | 8 | 81.2926 | 65.9401 | 15.3525 |
| 16 | 58 | Qwen3-4B | GSM8K | 8 | 81.3874 | 65.2388 | 16.1486 |
| 17 | 59 | Qwen3-4B | GSM8K | 8 | 81.3306 | 65.2199 | 16.1107 |
| 18 | 60 | Qwen3-4B | GSM8K | 8 | 81.3874 | 66.4519 | 14.9356 |
| 19 | 61 | Qwen3-4B | GSM8K | 8 | 81.2358 | 65.9401 | 15.2957 |
| 20 | 62 | Qwen3-4B | GSM8K | 8 | 80.5914 | 66.2055 | 14.3859 |
| 21 | 63 | Qwen3-4B | GSM8K | 8 | 81.3306 | 66.0538 | 15.2767 |
| 22 | 64 | Qwen3-4B | GSM8K | 8 | 81.7854 | 64.6892 | 17.0963 |
| 23 | 65 | Qwen3-4B | GSM8K | 8 | 81.16 | 65.5421 | 15.6179 |
| 24 | 66 | Qwen3-4B | GSM8K | 8 | 81.3685 | 66.0728 | 15.2957 |
| 25 | 67 | Qwen3-4B | GSM8K | 8 | 81.4632 | 66.0349 | 15.4284 |
| 26 | 68 | Qwen3-4B | GSM8K | 8 | 81.5201 | 65.8453 | 15.6748 |
| 27 | 69 | Qwen3-4B | GSM8K | 8 | 81.6528 | 65.2957 | 16.3571 |
| 28 | 70 | Qwen3-4B | GSM8K | 8 | 81.2168 | 65.5042 | 15.7127 |
| 29 | 71 | Qwen3-4B | GSM8K | 8 | 81.2737 | 65.6748 | 15.5989 |
| 30 | 72 | Qwen3-4B | GSM8K | 8 | 82.1456 | 65.5042 | 16.6414 |
| 31 | 73 | Qwen3-4B | GSM8K | 8 | 81.4822 | 66.3381 | 15.144 |
| 32 | 74 | Qwen3-4B | GSM8K | 8 | 81.9939 | 66.0159 | 15.978 |
| 33 | 75 | Qwen3-4B | GSM8K | 8 | 81.1221 | 66.5845 | 14.5375 |
| 34 | 76 | Qwen3-4B | GSM8K | 8 | 81.9181 | 65.3336 | 16.5845 |
| 35 | 77 | Qwen3-4B | GSM8K | 8 | 81.7475 | 66.1107 | 15.6368 |
| 36 | 78 | Qwen3-4B | GSM8K | 8 | 81.3874 | 65.58 | 15.8074 |
| 37 | 79 | Qwen3-4B | GSM8K | 8 | 82.2593 | 65.7695 | 16.4898 |
| 38 | 80 | Qwen3-4B | GSM8K | 8 | 81.3116 | 66.4329 | 14.8787 |
| 39 | 81 | Qwen3-4B | GSM8K | 8 | 81.8992 | 65.0872 | 16.812 |
| 40 | 82 | Qwen3-4B | GSM8K | 8 | 80.9515 | 65.5231 | 15.4284 |
| 41 | 83 | Qwen3-4B | GSM8K | 8 | 81.2168 | 66.0538 | 15.163 |
| 42 | 84 | Qwen3-4B | GSM8K | 8 | 80.8946 | 65.6937 | 15.2009 |
| 43 | 85 | Qwen3-4B | GSM8K | 8 | 81.1789 | 66.1486 | 15.0303 |
| 44 | 86 | Qwen3-4B | GSM8K | 8 | 81.3685 | 66.5845 | 14.7839 |
| 45 | 87 | Qwen3-4B | GSM8K | 8 | 81.5011 | 65.2199 | 16.2813 |
| 46 | 88 | Qwen3-4B | GSM8K | 8 | 81.4632 | 65.7695 | 15.6937 |
| 47 | 89 | Qwen3-4B | GSM8K | 8 | 81.6907 | 65.7316 | 15.9591 |
| 48 | 90 | Qwen3-4B | GSM8K | 8 | 81.6907 | 66.0349 | 15.6558 |
| 49 | 91 | Qwen3-4B | GSM8K | 8 | 81.6528 | 65.8074 | 15.8453 |
| 0 | 42 | Qwen3-4B | GSM8K | 12 | 81.9434 | 65.4663 | 16.4771 |
| 1 | 43 | Qwen3-4B | GSM8K | 12 | 81.9181 | 65.2262 | 16.6919 |
| 2 | 44 | Qwen3-4B | GSM8K | 12 | 81.4632 | 65.8959 | 15.5673 |
| 3 | 45 | Qwen3-4B | GSM8K | 12 | 81.8423 | 65.2388 | 16.6035 |
| 4 | 46 | Qwen3-4B | GSM8K | 12 | 81.7412 | 65.719 | 16.0222 |
| 5 | 47 | Qwen3-4B | GSM8K | 12 | 81.7286 | 65.6558 | 16.0728 |
| 6 | 48 | Qwen3-4B | GSM8K | 12 | 81.6654 | 65.6684 | 15.997 |
| 7 | 49 | Qwen3-4B | GSM8K | 12 | 81.9687 | 65.7316 | 16.237 |
| 8 | 50 | Qwen3-4B | GSM8K | 12 | 81.7412 | 65.6053 | 16.136 |
| 9 | 51 | Qwen3-4B | GSM8K | 12 | 82.0192 | 65.6305 | 16.3887 |
| 10 | 52 | Qwen3-4B | GSM8K | 12 | 81.5643 | 65.58 | 15.9843 |
| 11 | 53 | Qwen3-4B | GSM8K | 12 | 81.6528 | 65.302 | 16.3508 |
| 12 | 54 | Qwen3-4B | GSM8K | 12 | 81.7791 | 65.997 | 15.7822 |
| 13 | 55 | Qwen3-4B | GSM8K | 12 | 81.7159 | 65.6053 | 16.1107 |
| 14 | 56 | Qwen3-4B | GSM8K | 12 | 82.0192 | 65.5421 | 16.4771 |
| 15 | 57 | Qwen3-4B | GSM8K | 12 | 81.7791 | 65.7063 | 16.0728 |
| 16 | 58 | Qwen3-4B | GSM8K | 12 | 81.8549 | 65.1883 | 16.6667 |
| 17 | 59 | Qwen3-4B | GSM8K | 12 | 81.8802 | 65.3778 | 16.5024 |
| 18 | 60 | Qwen3-4B | GSM8K | 12 | 81.8676 | 65.7316 | 16.136 |
| 19 | 61 | Qwen3-4B | GSM8K | 12 | 81.6275 | 65.6053 | 16.0222 |
| 20 | 62 | Qwen3-4B | GSM8K | 12 | 81.6907 | 65.4536 | 16.237 |
| 21 | 63 | Qwen3-4B | GSM8K | 12 | 81.5517 | 65.6179 | 15.9338 |
| 22 | 64 | Qwen3-4B | GSM8K | 12 | 82.0066 | 65.1756 | 16.8309 |
| 23 | 65 | Qwen3-4B | GSM8K | 12 | 81.7539 | 65.1756 | 16.5782 |
| 24 | 66 | Qwen3-4B | GSM8K | 12 | 81.9308 | 65.6432 | 16.2876 |
| 25 | 67 | Qwen3-4B | GSM8K | 12 | 81.8297 | 65.8706 | 15.9591 |
| 26 | 68 | Qwen3-4B | GSM8K | 12 | 81.956 | 65.58 | 16.376 |
| 27 | 69 | Qwen3-4B | GSM8K | 12 | 81.8802 | 65.1251 | 16.7551 |
| 28 | 70 | Qwen3-4B | GSM8K | 12 | 81.6275 | 65.2894 | 16.3381 |
| 29 | 71 | Qwen3-4B | GSM8K | 12 | 82.0445 | 65.2894 | 16.7551 |
| 30 | 72 | Qwen3-4B | GSM8K | 12 | 82.2214 | 65.58 | 16.6414 |
| 31 | 73 | Qwen3-4B | GSM8K | 12 | 82.1708 | 65.4663 | 16.7046 |
| 32 | 74 | Qwen3-4B | GSM8K | 12 | 82.1203 | 65.7822 | 16.3381 |
| 33 | 75 | Qwen3-4B | GSM8K | 12 | 81.8044 | 65.6305 | 16.1739 |
| 34 | 76 | Qwen3-4B | GSM8K | 12 | 81.7159 | 64.9987 | 16.7172 |
| 35 | 77 | Qwen3-4B | GSM8K | 12 | 82.0192 | 65.2515 | 16.7678 |
| 36 | 78 | Qwen3-4B | GSM8K | 12 | 82.1835 | 65.4157 | 16.7678 |
| 37 | 79 | Qwen3-4B | GSM8K | 12 | 81.8802 | 64.7334 | 17.1468 |
| 38 | 80 | Qwen3-4B | GSM8K | 12 | 81.8044 | 65.6179 | 16.1865 |
| 39 | 81 | Qwen3-4B | GSM8K | 12 | 81.956 | 65.5926 | 16.3634 |
| 40 | 82 | Qwen3-4B | GSM8K | 12 | 81.438 | 65.7063 | 15.7316 |
| 41 | 83 | Qwen3-4B | GSM8K | 12 | 81.9308 | 65.7063 | 16.2244 |
| 42 | 84 | Qwen3-4B | GSM8K | 12 | 81.4885 | 65.5042 | 15.9843 |
| 43 | 85 | Qwen3-4B | GSM8K | 12 | 81.7665 | 65.9717 | 15.7948 |
| 44 | 86 | Qwen3-4B | GSM8K | 12 | 81.7791 | 65.2009 | 16.5782 |
| 45 | 87 | Qwen3-4B | GSM8K | 12 | 81.5011 | 65.302 | 16.1991 |
| 46 | 88 | Qwen3-4B | GSM8K | 12 | 81.9939 | 65.5042 | 16.4898 |
| 47 | 89 | Qwen3-4B | GSM8K | 12 | 81.9055 | 65.4284 | 16.4771 |
| 48 | 90 | Qwen3-4B | GSM8K | 12 | 81.956 | 65.6053 | 16.3508 |
| 49 | 91 | Qwen3-4B | GSM8K | 12 | 82.4362 | 65.5673 | 16.8688 |
| 0 | 42 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 1 | 43 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 2 | 44 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 3 | 45 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 4 | 46 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 5 | 47 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 6 | 48 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 7 | 49 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 8 | 50 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 9 | 51 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 10 | 52 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 11 | 53 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 12 | 54 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 13 | 55 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 14 | 56 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 15 | 57 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 16 | 58 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 17 | 59 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 18 | 60 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 19 | 61 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 20 | 62 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 21 | 63 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 22 | 64 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 23 | 65 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 24 | 66 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 25 | 67 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 26 | 68 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 27 | 69 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 28 | 70 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 29 | 71 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 30 | 72 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 31 | 73 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 32 | 74 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 33 | 75 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 34 | 76 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 35 | 77 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 36 | 78 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 37 | 79 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 38 | 80 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 39 | 81 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 40 | 82 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 41 | 83 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 42 | 84 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 43 | 85 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 44 | 86 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 45 | 87 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 46 | 88 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 47 | 89 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 48 | 90 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 49 | 91 | Qwen3-4B | GSM8K | 16 | 81.9845 | 65.3715 | 16.613 |
| 0 | 42 | Qwen3-4B | MATH500 | 4 | 47.1 | 41.6 | 5.5 |
| 1 | 43 | Qwen3-4B | MATH500 | 4 | 48.6 | 42.9 | 5.7 |
| 2 | 44 | Qwen3-4B | MATH500 | 4 | 49.3 | 41.3 | 8 |
| 3 | 45 | Qwen3-4B | MATH500 | 4 | 47.1 | 43.2 | 3.9 |
| 4 | 46 | Qwen3-4B | MATH500 | 4 | 48.2 | 41.4 | 6.8 |
| 5 | 47 | Qwen3-4B | MATH500 | 4 | 47.9 | 42.2 | 5.7 |
| 6 | 48 | Qwen3-4B | MATH500 | 4 | 50.3 | 39.8 | 10.5 |
| 7 | 49 | Qwen3-4B | MATH500 | 4 | 48.4 | 39.5 | 8.9 |
| 8 | 50 | Qwen3-4B | MATH500 | 4 | 47.4 | 40.6 | 6.8 |
| 9 | 51 | Qwen3-4B | MATH500 | 4 | 48.1 | 42.2 | 5.9 |
| 10 | 52 | Qwen3-4B | MATH500 | 4 | 47.5 | 40.3 | 7.2 |
| 11 | 53 | Qwen3-4B | MATH500 | 4 | 46.5 | 42.9 | 3.6 |
| 12 | 54 | Qwen3-4B | MATH500 | 4 | 47.2 | 39 | 8.2 |
| 13 | 55 | Qwen3-4B | MATH500 | 4 | 49.1 | 40.3 | 8.8 |
| 14 | 56 | Qwen3-4B | MATH500 | 4 | 46.6 | 40.8 | 5.8 |
| 15 | 57 | Qwen3-4B | MATH500 | 4 | 47.3 | 41 | 6.3 |
| 16 | 58 | Qwen3-4B | MATH500 | 4 | 47.6 | 39.4 | 8.2 |
| 17 | 59 | Qwen3-4B | MATH500 | 4 | 48.8 | 40.8 | 8 |
| 18 | 60 | Qwen3-4B | MATH500 | 4 | 47.8 | 41.9 | 5.9 |
| 19 | 61 | Qwen3-4B | MATH500 | 4 | 47.9 | 42.2 | 5.7 |
| 20 | 62 | Qwen3-4B | MATH500 | 4 | 48.1 | 42.2 | 5.9 |
| 21 | 63 | Qwen3-4B | MATH500 | 4 | 47.3 | 41.7 | 5.6 |
| 22 | 64 | Qwen3-4B | MATH500 | 4 | 48.5 | 42.3 | 6.2 |
| 23 | 65 | Qwen3-4B | MATH500 | 4 | 47.1 | 42.4 | 4.7 |
| 24 | 66 | Qwen3-4B | MATH500 | 4 | 48.4 | 41.2 | 7.2 |
| 25 | 67 | Qwen3-4B | MATH500 | 4 | 48.5 | 40.9 | 7.6 |
| 26 | 68 | Qwen3-4B | MATH500 | 4 | 47.6 | 40.6 | 7 |
| 27 | 69 | Qwen3-4B | MATH500 | 4 | 48.1 | 40.8 | 7.3 |
| 28 | 70 | Qwen3-4B | MATH500 | 4 | 47.9 | 40.9 | 7 |
| 29 | 71 | Qwen3-4B | MATH500 | 4 | 48 | 40.7 | 7.3 |
| 30 | 72 | Qwen3-4B | MATH500 | 4 | 48.5 | 41.2 | 7.3 |
| 31 | 73 | Qwen3-4B | MATH500 | 4 | 47.6 | 41.4 | 6.2 |
| 32 | 74 | Qwen3-4B | MATH500 | 4 | 49.3 | 41.7 | 7.6 |
| 33 | 75 | Qwen3-4B | MATH500 | 4 | 45.6 | 41 | 4.6 |
| 34 | 76 | Qwen3-4B | MATH500 | 4 | 48.1 | 40.1 | 8 |
| 35 | 77 | Qwen3-4B | MATH500 | 4 | 48.8 | 40.5 | 8.3 |
| 36 | 78 | Qwen3-4B | MATH500 | 4 | 49 | 40.7 | 8.3 |
| 37 | 79 | Qwen3-4B | MATH500 | 4 | 48.6 | 39.8 | 8.8 |
| 38 | 80 | Qwen3-4B | MATH500 | 4 | 49.1 | 43 | 6.1 |
| 39 | 81 | Qwen3-4B | MATH500 | 4 | 48.9 | 41.8 | 7.1 |
| 40 | 82 | Qwen3-4B | MATH500 | 4 | 47.9 | 40.2 | 7.7 |
| 41 | 83 | Qwen3-4B | MATH500 | 4 | 47.8 | 41.1 | 6.7 |
| 42 | 84 | Qwen3-4B | MATH500 | 4 | 47.7 | 42.6 | 5.1 |
| 43 | 85 | Qwen3-4B | MATH500 | 4 | 48.3 | 43.1 | 5.2 |
| 44 | 86 | Qwen3-4B | MATH500 | 4 | 48.1 | 44.9 | 3.2 |
| 45 | 87 | Qwen3-4B | MATH500 | 4 | 48.6 | 41.3 | 7.3 |
| 46 | 88 | Qwen3-4B | MATH500 | 4 | 47.5 | 42.7 | 4.8 |
| 47 | 89 | Qwen3-4B | MATH500 | 4 | 47.7 | 41.2 | 6.5 |
| 48 | 90 | Qwen3-4B | MATH500 | 4 | 48.3 | 40.2 | 8.1 |
| 49 | 91 | Qwen3-4B | MATH500 | 4 | 46.3 | 41.8 | 4.5 |
| 0 | 42 | Qwen3-4B | MATH500 | 8 | 48.25 | 40.35 | 7.9 |
| 1 | 43 | Qwen3-4B | MATH500 | 8 | 48.85 | 41.05 | 7.8 |
| 2 | 44 | Qwen3-4B | MATH500 | 8 | 48.3 | 41.25 | 7.05 |
| 3 | 45 | Qwen3-4B | MATH500 | 8 | 48.25 | 40.6 | 7.65 |
| 4 | 46 | Qwen3-4B | MATH500 | 8 | 48.85 | 40.85 | 8 |
| 5 | 47 | Qwen3-4B | MATH500 | 8 | 48.9 | 41.45 | 7.45 |
| 6 | 48 | Qwen3-4B | MATH500 | 8 | 49.05 | 41.1 | 7.95 |
| 7 | 49 | Qwen3-4B | MATH500 | 8 | 48.85 | 40.35 | 8.5 |
| 8 | 50 | Qwen3-4B | MATH500 | 8 | 48.3 | 40.6 | 7.7 |
| 9 | 51 | Qwen3-4B | MATH500 | 8 | 47.9 | 42.4 | 5.5 |
| 10 | 52 | Qwen3-4B | MATH500 | 8 | 47.6 | 40.75 | 6.85 |
| 11 | 53 | Qwen3-4B | MATH500 | 8 | 47.55 | 41.55 | 6 |
| 12 | 54 | Qwen3-4B | MATH500 | 8 | 47.8 | 40.85 | 6.95 |
| 13 | 55 | Qwen3-4B | MATH500 | 8 | 48.35 | 41.1 | 7.25 |
| 14 | 56 | Qwen3-4B | MATH500 | 8 | 47.55 | 41.2 | 6.35 |
| 15 | 57 | Qwen3-4B | MATH500 | 8 | 47.55 | 42.05 | 5.5 |
| 16 | 58 | Qwen3-4B | MATH500 | 8 | 47 | 41.2 | 5.8 |
| 17 | 59 | Qwen3-4B | MATH500 | 8 | 48.4 | 41.05 | 7.35 |
| 18 | 60 | Qwen3-4B | MATH500 | 8 | 48.75 | 40.05 | 8.7 |
| 19 | 61 | Qwen3-4B | MATH500 | 8 | 48.1 | 40.75 | 7.35 |
| 20 | 62 | Qwen3-4B | MATH500 | 8 | 48.45 | 41.25 | 7.2 |
| 21 | 63 | Qwen3-4B | MATH500 | 8 | 48.45 | 41.5 | 6.95 |
| 22 | 64 | Qwen3-4B | MATH500 | 8 | 48.2 | 40.95 | 7.25 |
| 23 | 65 | Qwen3-4B | MATH500 | 8 | 48.1 | 41.8 | 6.3 |
| 24 | 66 | Qwen3-4B | MATH500 | 8 | 47.9 | 41.05 | 6.85 |
| 25 | 67 | Qwen3-4B | MATH500 | 8 | 48.8 | 41.75 | 7.05 |
| 26 | 68 | Qwen3-4B | MATH500 | 8 | 48.45 | 41.15 | 7.3 |
| 27 | 69 | Qwen3-4B | MATH500 | 8 | 47.85 | 41.4 | 6.45 |
| 28 | 70 | Qwen3-4B | MATH500 | 8 | 48.95 | 41.05 | 7.9 |
| 29 | 71 | Qwen3-4B | MATH500 | 8 | 48.65 | 40.85 | 7.8 |
| 30 | 72 | Qwen3-4B | MATH500 | 8 | 48.6 | 41.15 | 7.45 |
| 31 | 73 | Qwen3-4B | MATH500 | 8 | 48.5 | 39.8 | 8.7 |
| 32 | 74 | Qwen3-4B | MATH500 | 8 | 49.1 | 40.8 | 8.3 |
| 33 | 75 | Qwen3-4B | MATH500 | 8 | 48.15 | 40.8 | 7.35 |
| 34 | 76 | Qwen3-4B | MATH500 | 8 | 48 | 42.2 | 5.8 |
| 35 | 77 | Qwen3-4B | MATH500 | 8 | 49 | 39.95 | 9.05 |
| 36 | 78 | Qwen3-4B | MATH500 | 8 | 48.4 | 40.95 | 7.45 |
| 37 | 79 | Qwen3-4B | MATH500 | 8 | 49.05 | 39.85 | 9.2 |
| 38 | 80 | Qwen3-4B | MATH500 | 8 | 48.55 | 41.25 | 7.3 |
| 39 | 81 | Qwen3-4B | MATH500 | 8 | 48.25 | 41.15 | 7.1 |
| 40 | 82 | Qwen3-4B | MATH500 | 8 | 48.25 | 41.05 | 7.2 |
| 41 | 83 | Qwen3-4B | MATH500 | 8 | 48.4 | 40.35 | 8.05 |
| 42 | 84 | Qwen3-4B | MATH500 | 8 | 47.65 | 41.85 | 5.8 |
| 43 | 85 | Qwen3-4B | MATH500 | 8 | 48.55 | 42.05 | 6.5 |
| 44 | 86 | Qwen3-4B | MATH500 | 8 | 49.05 | 41.95 | 7.1 |
| 45 | 87 | Qwen3-4B | MATH500 | 8 | 48.15 | 40.5 | 7.65 |
| 46 | 88 | Qwen3-4B | MATH500 | 8 | 48.95 | 40.15 | 8.8 |
| 47 | 89 | Qwen3-4B | MATH500 | 8 | 48.05 | 41.4 | 6.65 |
| 48 | 90 | Qwen3-4B | MATH500 | 8 | 48 | 40.05 | 7.95 |
| 49 | 91 | Qwen3-4B | MATH500 | 8 | 47.05 | 41.2 | 5.85 |
| 0 | 42 | Qwen3-4B | MATH500 | 12 | 48.6333 | 40.8667 | 7.7667 |
| 1 | 43 | Qwen3-4B | MATH500 | 12 | 49.1333 | 40.7 | 8.4333 |
| 2 | 44 | Qwen3-4B | MATH500 | 12 | 48.8333 | 40.6667 | 8.1667 |
| 3 | 45 | Qwen3-4B | MATH500 | 12 | 49 | 40.8333 | 8.1667 |
| 4 | 46 | Qwen3-4B | MATH500 | 12 | 48.6333 | 40.9333 | 7.7 |
| 5 | 47 | Qwen3-4B | MATH500 | 12 | 48.6333 | 40.9333 | 7.7 |
| 6 | 48 | Qwen3-4B | MATH500 | 12 | 48.6 | 41.0667 | 7.5333 |
| 7 | 49 | Qwen3-4B | MATH500 | 12 | 49.1 | 40.7 | 8.4 |
| 8 | 50 | Qwen3-4B | MATH500 | 12 | 48.8333 | 40.3333 | 8.5 |
| 9 | 51 | Qwen3-4B | MATH500 | 12 | 47.9333 | 40.8333 | 7.1 |
| 10 | 52 | Qwen3-4B | MATH500 | 12 | 48.5333 | 40.6333 | 7.9 |
| 11 | 53 | Qwen3-4B | MATH500 | 12 | 48.2 | 40.9667 | 7.2333 |
| 12 | 54 | Qwen3-4B | MATH500 | 12 | 48.4 | 40.7667 | 7.6333 |
| 13 | 55 | Qwen3-4B | MATH500 | 12 | 48.2333 | 40.9667 | 7.2667 |
| 14 | 56 | Qwen3-4B | MATH500 | 12 | 48.4667 | 40 | 8.4667 |
| 15 | 57 | Qwen3-4B | MATH500 | 12 | 48.4 | 41.4333 | 6.9667 |
| 16 | 58 | Qwen3-4B | MATH500 | 12 | 48.0667 | 40.4667 | 7.6 |
| 17 | 59 | Qwen3-4B | MATH500 | 12 | 48.7667 | 40.5 | 8.2667 |
| 18 | 60 | Qwen3-4B | MATH500 | 12 | 49 | 39.8333 | 9.1667 |
| 19 | 61 | Qwen3-4B | MATH500 | 12 | 48.1333 | 41.0667 | 7.0667 |
| 20 | 62 | Qwen3-4B | MATH500 | 12 | 48.5 | 40.6333 | 7.8667 |
| 21 | 63 | Qwen3-4B | MATH500 | 12 | 48.4667 | 40.7333 | 7.7333 |
| 22 | 64 | Qwen3-4B | MATH500 | 12 | 48.5 | 41.1667 | 7.3333 |
| 23 | 65 | Qwen3-4B | MATH500 | 12 | 48.0333 | 40.8667 | 7.1667 |
| 24 | 66 | Qwen3-4B | MATH500 | 12 | 48.3333 | 40.4333 | 7.9 |
| 25 | 67 | Qwen3-4B | MATH500 | 12 | 48.6333 | 41.1 | 7.5333 |
| 26 | 68 | Qwen3-4B | MATH500 | 12 | 48.4667 | 41.2667 | 7.2 |
| 27 | 69 | Qwen3-4B | MATH500 | 12 | 48.2667 | 41.4 | 6.8667 |
| 28 | 70 | Qwen3-4B | MATH500 | 12 | 48.5 | 41.0667 | 7.4333 |
| 29 | 71 | Qwen3-4B | MATH500 | 12 | 48.6667 | 41.4 | 7.2667 |
| 30 | 72 | Qwen3-4B | MATH500 | 12 | 48.3667 | 40.8 | 7.5667 |
| 31 | 73 | Qwen3-4B | MATH500 | 12 | 48.4 | 40.7 | 7.7 |
| 32 | 74 | Qwen3-4B | MATH500 | 12 | 48.5333 | 40.4333 | 8.1 |
| 33 | 75 | Qwen3-4B | MATH500 | 12 | 48.2 | 40.7667 | 7.4333 |
| 34 | 76 | Qwen3-4B | MATH500 | 12 | 47.9667 | 41.5 | 6.4667 |
| 35 | 77 | Qwen3-4B | MATH500 | 12 | 48.2333 | 40.6 | 7.6333 |
| 36 | 78 | Qwen3-4B | MATH500 | 12 | 48.7 | 40.4667 | 8.2333 |
| 37 | 79 | Qwen3-4B | MATH500 | 12 | 48.5 | 40.7333 | 7.7667 |
| 38 | 80 | Qwen3-4B | MATH500 | 12 | 48.9333 | 40.8 | 8.1333 |
| 39 | 81 | Qwen3-4B | MATH500 | 12 | 48.5333 | 40.7333 | 7.8 |
| 40 | 82 | Qwen3-4B | MATH500 | 12 | 48.7333 | 40.1 | 8.6333 |
| 41 | 83 | Qwen3-4B | MATH500 | 12 | 48.6 | 39.2667 | 9.3333 |
| 42 | 84 | Qwen3-4B | MATH500 | 12 | 48.6667 | 40.8333 | 7.8333 |
| 43 | 85 | Qwen3-4B | MATH500 | 12 | 48.7333 | 41.2667 | 7.4667 |
| 44 | 86 | Qwen3-4B | MATH500 | 12 | 48.6667 | 40.9 | 7.7667 |
| 45 | 87 | Qwen3-4B | MATH500 | 12 | 48.7333 | 40.7 | 8.0333 |
| 46 | 88 | Qwen3-4B | MATH500 | 12 | 48.3333 | 41.3 | 7.0333 |
| 47 | 89 | Qwen3-4B | MATH500 | 12 | 48.7 | 41.3 | 7.4 |
| 48 | 90 | Qwen3-4B | MATH500 | 12 | 48.0667 | 40.0333 | 8.0333 |
| 49 | 91 | Qwen3-4B | MATH500 | 12 | 47.7333 | 40.8 | 6.9333 |
| 0 | 42 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 1 | 43 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 2 | 44 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 3 | 45 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 4 | 46 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 5 | 47 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 6 | 48 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 7 | 49 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 8 | 50 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 9 | 51 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 10 | 52 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 11 | 53 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 12 | 54 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 13 | 55 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 14 | 56 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 15 | 57 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 16 | 58 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 17 | 59 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 18 | 60 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 19 | 61 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 20 | 62 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 21 | 63 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 22 | 64 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 23 | 65 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 24 | 66 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 25 | 67 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 26 | 68 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 27 | 69 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 28 | 70 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 29 | 71 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 30 | 72 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 31 | 73 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 32 | 74 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 33 | 75 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 34 | 76 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 35 | 77 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 36 | 78 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 37 | 79 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 38 | 80 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 39 | 81 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 40 | 82 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 41 | 83 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 42 | 84 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 43 | 85 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 44 | 86 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 45 | 87 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 46 | 88 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 47 | 89 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 48 | 90 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 49 | 91 | Qwen3-4B | MATH500 | 16 | 48.425 | 40.775 | 7.65 |
| 0 | 42 | Qwen3-4B | SVAMP | 4 | 80.5 | 69.9 | 10.6 |
| 1 | 43 | Qwen3-4B | SVAMP | 4 | 81.4 | 69.4 | 12 |
| 2 | 44 | Qwen3-4B | SVAMP | 4 | 81.95 | 69.45 | 12.5 |
| 3 | 45 | Qwen3-4B | SVAMP | 4 | 81.9 | 70.25 | 11.65 |
| 4 | 46 | Qwen3-4B | SVAMP | 4 | 81.6 | 69.75 | 11.85 |
| 5 | 47 | Qwen3-4B | SVAMP | 4 | 83 | 70.15 | 12.85 |
| 6 | 48 | Qwen3-4B | SVAMP | 4 | 81 | 70 | 11 |
| 7 | 49 | Qwen3-4B | SVAMP | 4 | 82.4 | 70.05 | 12.35 |
| 8 | 50 | Qwen3-4B | SVAMP | 4 | 81.95 | 71.35 | 10.6 |
| 9 | 51 | Qwen3-4B | SVAMP | 4 | 81.8 | 70.7 | 11.1 |
| 10 | 52 | Qwen3-4B | SVAMP | 4 | 82.15 | 70.4 | 11.75 |
| 11 | 53 | Qwen3-4B | SVAMP | 4 | 81.8 | 70.55 | 11.25 |
| 12 | 54 | Qwen3-4B | SVAMP | 4 | 81.95 | 69.2 | 12.75 |
| 13 | 55 | Qwen3-4B | SVAMP | 4 | 81.85 | 69.5 | 12.35 |
| 14 | 56 | Qwen3-4B | SVAMP | 4 | 81.8 | 71.15 | 10.65 |
| 15 | 57 | Qwen3-4B | SVAMP | 4 | 81 | 68.15 | 12.85 |
| 16 | 58 | Qwen3-4B | SVAMP | 4 | 82.55 | 70.3 | 12.25 |
| 17 | 59 | Qwen3-4B | SVAMP | 4 | 82.45 | 70.85 | 11.6 |
| 18 | 60 | Qwen3-4B | SVAMP | 4 | 82.25 | 70.7 | 11.55 |
| 19 | 61 | Qwen3-4B | SVAMP | 4 | 83.2 | 70.05 | 13.15 |
| 20 | 62 | Qwen3-4B | SVAMP | 4 | 82.6 | 70.6 | 12 |
| 21 | 63 | Qwen3-4B | SVAMP | 4 | 82.4 | 70.25 | 12.15 |
| 22 | 64 | Qwen3-4B | SVAMP | 4 | 81.75 | 69.15 | 12.6 |
| 23 | 65 | Qwen3-4B | SVAMP | 4 | 82.2 | 70.45 | 11.75 |
| 24 | 66 | Qwen3-4B | SVAMP | 4 | 82.25 | 70.7 | 11.55 |
| 25 | 67 | Qwen3-4B | SVAMP | 4 | 80.5 | 71.35 | 9.15 |
| 26 | 68 | Qwen3-4B | SVAMP | 4 | 82.1 | 68.8 | 13.3 |
| 27 | 69 | Qwen3-4B | SVAMP | 4 | 81.7 | 68.65 | 13.05 |
| 28 | 70 | Qwen3-4B | SVAMP | 4 | 83.2 | 69.6 | 13.6 |
| 29 | 71 | Qwen3-4B | SVAMP | 4 | 81.85 | 70.85 | 11 |
| 30 | 72 | Qwen3-4B | SVAMP | 4 | 82.4 | 68.45 | 13.95 |
| 31 | 73 | Qwen3-4B | SVAMP | 4 | 81.15 | 70.25 | 10.9 |
| 32 | 74 | Qwen3-4B | SVAMP | 4 | 82.05 | 70.3 | 11.75 |
| 33 | 75 | Qwen3-4B | SVAMP | 4 | 81.4 | 68.85 | 12.55 |
| 34 | 76 | Qwen3-4B | SVAMP | 4 | 82.1 | 71.55 | 10.55 |
| 35 | 77 | Qwen3-4B | SVAMP | 4 | 82.4 | 69.75 | 12.65 |
| 36 | 78 | Qwen3-4B | SVAMP | 4 | 81.5 | 71.4 | 10.1 |
| 37 | 79 | Qwen3-4B | SVAMP | 4 | 82.2 | 71.25 | 10.95 |
| 38 | 80 | Qwen3-4B | SVAMP | 4 | 82.4 | 70.7 | 11.7 |
| 39 | 81 | Qwen3-4B | SVAMP | 4 | 82.2 | 70.4 | 11.8 |
| 40 | 82 | Qwen3-4B | SVAMP | 4 | 79.9 | 71.2 | 8.7 |
| 41 | 83 | Qwen3-4B | SVAMP | 4 | 82.7 | 71.05 | 11.65 |
| 42 | 84 | Qwen3-4B | SVAMP | 4 | 84 | 70.7 | 13.3 |
| 43 | 85 | Qwen3-4B | SVAMP | 4 | 81.2 | 71 | 10.2 |
| 44 | 86 | Qwen3-4B | SVAMP | 4 | 82.75 | 69.55 | 13.2 |
| 45 | 87 | Qwen3-4B | SVAMP | 4 | 81.35 | 70.6 | 10.75 |
| 46 | 88 | Qwen3-4B | SVAMP | 4 | 82.75 | 70.6 | 12.15 |
| 47 | 89 | Qwen3-4B | SVAMP | 4 | 81.4 | 70.45 | 10.95 |
| 48 | 90 | Qwen3-4B | SVAMP | 4 | 81.6 | 68.8 | 12.8 |
| 49 | 91 | Qwen3-4B | SVAMP | 4 | 81.95 | 70.4 | 11.55 |
| 0 | 42 | Qwen3-4B | SVAMP | 8 | 82.1 | 69.35 | 12.75 |
| 1 | 43 | Qwen3-4B | SVAMP | 8 | 83.375 | 69.45 | 13.925 |
| 2 | 44 | Qwen3-4B | SVAMP | 8 | 82.9 | 69.4 | 13.5 |
| 3 | 45 | Qwen3-4B | SVAMP | 8 | 83.2 | 68.4 | 14.8 |
| 4 | 46 | Qwen3-4B | SVAMP | 8 | 82.7 | 69.525 | 13.175 |
| 5 | 47 | Qwen3-4B | SVAMP | 8 | 83.325 | 69.8 | 13.525 |
| 6 | 48 | Qwen3-4B | SVAMP | 8 | 82.6 | 69.8 | 12.8 |
| 7 | 49 | Qwen3-4B | SVAMP | 8 | 82.6 | 69.075 | 13.525 |
| 8 | 50 | Qwen3-4B | SVAMP | 8 | 83.175 | 70.525 | 12.65 |
| 9 | 51 | Qwen3-4B | SVAMP | 8 | 82.65 | 69.825 | 12.825 |
| 10 | 52 | Qwen3-4B | SVAMP | 8 | 83.2 | 69.6 | 13.6 |
| 11 | 53 | Qwen3-4B | SVAMP | 8 | 82.375 | 69.475 | 12.9 |
| 12 | 54 | Qwen3-4B | SVAMP | 8 | 83.55 | 69.2 | 14.35 |
| 13 | 55 | Qwen3-4B | SVAMP | 8 | 83.7 | 68.725 | 14.975 |
| 14 | 56 | Qwen3-4B | SVAMP | 8 | 82.7 | 69.875 | 12.825 |
| 15 | 57 | Qwen3-4B | SVAMP | 8 | 82.6 | 69.325 | 13.275 |
| 16 | 58 | Qwen3-4B | SVAMP | 8 | 83.1 | 69.275 | 13.825 |
| 17 | 59 | Qwen3-4B | SVAMP | 8 | 82.95 | 69.2 | 13.75 |
| 18 | 60 | Qwen3-4B | SVAMP | 8 | 82.85 | 69.725 | 13.125 |
| 19 | 61 | Qwen3-4B | SVAMP | 8 | 83.6 | 68.925 | 14.675 |
| 20 | 62 | Qwen3-4B | SVAMP | 8 | 82.675 | 69.95 | 12.725 |
| 21 | 63 | Qwen3-4B | SVAMP | 8 | 82.575 | 69.325 | 13.25 |
| 22 | 64 | Qwen3-4B | SVAMP | 8 | 82.6 | 69.175 | 13.425 |
| 23 | 65 | Qwen3-4B | SVAMP | 8 | 83.175 | 69.55 | 13.625 |
| 24 | 66 | Qwen3-4B | SVAMP | 8 | 83.5 | 69.725 | 13.775 |
| 25 | 67 | Qwen3-4B | SVAMP | 8 | 81.9 | 70.4 | 11.5 |
| 26 | 68 | Qwen3-4B | SVAMP | 8 | 82.3 | 69.15 | 13.15 |
| 27 | 69 | Qwen3-4B | SVAMP | 8 | 82.475 | 68.15 | 14.325 |
| 28 | 70 | Qwen3-4B | SVAMP | 8 | 83.475 | 69.075 | 14.4 |
| 29 | 71 | Qwen3-4B | SVAMP | 8 | 83.35 | 68.7 | 14.65 |
| 30 | 72 | Qwen3-4B | SVAMP | 8 | 82.825 | 68.975 | 13.85 |
| 31 | 73 | Qwen3-4B | SVAMP | 8 | 82.975 | 69.675 | 13.3 |
| 32 | 74 | Qwen3-4B | SVAMP | 8 | 82.9 | 69.575 | 13.325 |
| 33 | 75 | Qwen3-4B | SVAMP | 8 | 83.35 | 68.825 | 14.525 |
| 34 | 76 | Qwen3-4B | SVAMP | 8 | 82.75 | 70.225 | 12.525 |
| 35 | 77 | Qwen3-4B | SVAMP | 8 | 83.325 | 69.3 | 14.025 |
| 36 | 78 | Qwen3-4B | SVAMP | 8 | 82.55 | 69.9 | 12.65 |
| 37 | 79 | Qwen3-4B | SVAMP | 8 | 83.175 | 69.7 | 13.475 |
| 38 | 80 | Qwen3-4B | SVAMP | 8 | 82.95 | 69.525 | 13.425 |
| 39 | 81 | Qwen3-4B | SVAMP | 8 | 83.25 | 70 | 13.25 |
| 40 | 82 | Qwen3-4B | SVAMP | 8 | 82.675 | 69.375 | 13.3 |
| 41 | 83 | Qwen3-4B | SVAMP | 8 | 83.15 | 69.425 | 13.725 |
| 42 | 84 | Qwen3-4B | SVAMP | 8 | 83.325 | 69.925 | 13.4 |
| 43 | 85 | Qwen3-4B | SVAMP | 8 | 82.825 | 69.875 | 12.95 |
| 44 | 86 | Qwen3-4B | SVAMP | 8 | 83.875 | 68.5 | 15.375 |
| 45 | 87 | Qwen3-4B | SVAMP | 8 | 82.4 | 69.975 | 12.425 |
| 46 | 88 | Qwen3-4B | SVAMP | 8 | 82.95 | 69.675 | 13.275 |
| 47 | 89 | Qwen3-4B | SVAMP | 8 | 83.375 | 69.2 | 14.175 |
| 48 | 90 | Qwen3-4B | SVAMP | 8 | 83.1 | 68.875 | 14.225 |
| 49 | 91 | Qwen3-4B | SVAMP | 8 | 82.7 | 69.375 | 13.325 |
| 0 | 42 | Qwen3-4B | SVAMP | 12 | 83.0167 | 68.8833 | 14.1333 |
| 1 | 43 | Qwen3-4B | SVAMP | 12 | 83.6667 | 69.3167 | 14.35 |
| 2 | 44 | Qwen3-4B | SVAMP | 12 | 82.9167 | 69.3333 | 13.5833 |
| 3 | 45 | Qwen3-4B | SVAMP | 12 | 83.5667 | 69 | 14.5667 |
| 4 | 46 | Qwen3-4B | SVAMP | 12 | 83.0667 | 69.1833 | 13.8833 |
| 5 | 47 | Qwen3-4B | SVAMP | 12 | 83.6333 | 68.6333 | 15 |
| 6 | 48 | Qwen3-4B | SVAMP | 12 | 83.2667 | 69.2 | 14.0667 |
| 7 | 49 | Qwen3-4B | SVAMP | 12 | 83.0167 | 69.0833 | 13.9333 |
| 8 | 50 | Qwen3-4B | SVAMP | 12 | 83.5833 | 69.1667 | 14.4167 |
| 9 | 51 | Qwen3-4B | SVAMP | 12 | 83.0167 | 69.1333 | 13.8833 |
| 10 | 52 | Qwen3-4B | SVAMP | 12 | 83.4 | 69.2167 | 14.1833 |
| 11 | 53 | Qwen3-4B | SVAMP | 12 | 82.8833 | 69.2167 | 13.6667 |
| 12 | 54 | Qwen3-4B | SVAMP | 12 | 83.4167 | 69.1 | 14.3167 |
| 13 | 55 | Qwen3-4B | SVAMP | 12 | 83.2667 | 69.1167 | 14.15 |
| 14 | 56 | Qwen3-4B | SVAMP | 12 | 83.1667 | 69.6667 | 13.5 |
| 15 | 57 | Qwen3-4B | SVAMP | 12 | 83.15 | 69.5 | 13.65 |
| 16 | 58 | Qwen3-4B | SVAMP | 12 | 83.35 | 68.8333 | 14.5167 |
| 17 | 59 | Qwen3-4B | SVAMP | 12 | 82.9833 | 69.0333 | 13.95 |
| 18 | 60 | Qwen3-4B | SVAMP | 12 | 83.2 | 69.0667 | 14.1333 |
| 19 | 61 | Qwen3-4B | SVAMP | 12 | 83.6 | 68.7667 | 14.8333 |
| 20 | 62 | Qwen3-4B | SVAMP | 12 | 83.0167 | 69.0833 | 13.9333 |
| 21 | 63 | Qwen3-4B | SVAMP | 12 | 83.1333 | 69.4167 | 13.7167 |
| 22 | 64 | Qwen3-4B | SVAMP | 12 | 83.1833 | 69.1 | 14.0833 |
| 23 | 65 | Qwen3-4B | SVAMP | 12 | 82.9 | 69.05 | 13.85 |
| 24 | 66 | Qwen3-4B | SVAMP | 12 | 83.6667 | 69.4167 | 14.25 |
| 25 | 67 | Qwen3-4B | SVAMP | 12 | 82.95 | 69.1667 | 13.7833 |
| 26 | 68 | Qwen3-4B | SVAMP | 12 | 82.6 | 69.0333 | 13.5667 |
| 27 | 69 | Qwen3-4B | SVAMP | 12 | 83.4 | 68.8833 | 14.5167 |
| 28 | 70 | Qwen3-4B | SVAMP | 12 | 83.35 | 69.3 | 14.05 |
| 29 | 71 | Qwen3-4B | SVAMP | 12 | 83.4333 | 68.9333 | 14.5 |
| 30 | 72 | Qwen3-4B | SVAMP | 12 | 83.6167 | 68.95 | 14.6667 |
| 31 | 73 | Qwen3-4B | SVAMP | 12 | 83.2167 | 69.35 | 13.8667 |
| 32 | 74 | Qwen3-4B | SVAMP | 12 | 83.15 | 69.5833 | 13.5667 |
| 33 | 75 | Qwen3-4B | SVAMP | 12 | 83.3 | 69.2667 | 14.0333 |
| 34 | 76 | Qwen3-4B | SVAMP | 12 | 82.9167 | 69.4333 | 13.4833 |
| 35 | 77 | Qwen3-4B | SVAMP | 12 | 83.3167 | 69.1167 | 14.2 |
| 36 | 78 | Qwen3-4B | SVAMP | 12 | 83.1667 | 69.0667 | 14.1 |
| 37 | 79 | Qwen3-4B | SVAMP | 12 | 83.3333 | 69.0667 | 14.2667 |
| 38 | 80 | Qwen3-4B | SVAMP | 12 | 83.2 | 69.1667 | 14.0333 |
| 39 | 81 | Qwen3-4B | SVAMP | 12 | 83.3333 | 69.5333 | 13.8 |
| 40 | 82 | Qwen3-4B | SVAMP | 12 | 83 | 69.15 | 13.85 |
| 41 | 83 | Qwen3-4B | SVAMP | 12 | 83.5667 | 69.1167 | 14.45 |
| 42 | 84 | Qwen3-4B | SVAMP | 12 | 83.45 | 69.25 | 14.2 |
| 43 | 85 | Qwen3-4B | SVAMP | 12 | 83.0333 | 69.4833 | 13.55 |
| 44 | 86 | Qwen3-4B | SVAMP | 12 | 83.65 | 68.9333 | 14.7167 |
| 45 | 87 | Qwen3-4B | SVAMP | 12 | 83.35 | 69.2 | 14.15 |
| 46 | 88 | Qwen3-4B | SVAMP | 12 | 83.05 | 69.1833 | 13.8667 |
| 47 | 89 | Qwen3-4B | SVAMP | 12 | 82.95 | 69.3 | 13.65 |
| 48 | 90 | Qwen3-4B | SVAMP | 12 | 83.6 | 68.0667 | 15.5333 |
| 49 | 91 | Qwen3-4B | SVAMP | 12 | 83.0333 | 69.1667 | 13.8667 |
| 0 | 42 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 1 | 43 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 2 | 44 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 3 | 45 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 4 | 46 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 5 | 47 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 6 | 48 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 7 | 49 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 8 | 50 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 9 | 51 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 10 | 52 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 11 | 53 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 12 | 54 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 13 | 55 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 14 | 56 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 15 | 57 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 16 | 58 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 17 | 59 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 18 | 60 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 19 | 61 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 20 | 62 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 21 | 63 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 22 | 64 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 23 | 65 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 24 | 66 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 25 | 67 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 26 | 68 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 27 | 69 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 28 | 70 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 29 | 71 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 30 | 72 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 31 | 73 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 32 | 74 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 33 | 75 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 34 | 76 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 35 | 77 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 36 | 78 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 37 | 79 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 38 | 80 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 39 | 81 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 40 | 82 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 41 | 83 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 42 | 84 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 43 | 85 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 44 | 86 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 45 | 87 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 46 | 88 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 47 | 89 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 48 | 90 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
| 49 | 91 | Qwen3-4B | SVAMP | 16 | 83.4 | 69.0625 | 14.3375 |
