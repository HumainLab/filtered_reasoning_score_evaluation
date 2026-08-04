# FRS vs other metrics — predictor feasibility report

**Audit date:** 2026-03-31  
**Method:** Files under `/Users/manaspathak11/Desktop/Everything/research paper/threshold` were opened or schema-checked; no new computations were run. **Parquet glob returned 0 files**; `source_pass16_jsonl_by_model*` JSONL is referenced by existing CSVs but not re-verified line-by-line in this pass.

---

## Executive summary

| Question | Answer |
|----------|--------|
| Finest grain available in artifacts | **Trace-level** for: token confidence, correctness, and (where judged) GPT-4o-mini **reasoning_score** + subscores. **Not** all traces are judged—typically **~250/pair** for the main FRS binning run and **100/pair** for the unfiltered baseline. |
| Is a **non-circular** “FRS predicts deployment-relevant selection quality better than X” test available **from CSVs alone**? | **No fully clean test.** FRS is defined from **confidence-ranked** pools and **the same judge** as bin-level reasoning scores. Any outcome that is **mean judge score in high-confidence strata** shares components with FRS’s construction. **pass@1** and **top-k accuracy** are **not** judge-based and can serve as **orthogonal predictors** or benchmarks, but they measure **correctness**, not **reasoning quality**. |
| Strongest honest analysis **without new data** | **Model×benchmark panel regression / partial correlation:** treat **FRS** (or bin-top reasoning) as outcome or predictor vs **pass@1** (`pass1_pct` / `base_acc`) and **SNR** / **base_reasoning** from `merged_pass1_frs_per_benchmark.csv`, plus **unfiltered** mean reasoning from `per_pair_scores.csv`, plus **top-10% accuracy** from `topk_ablation_results.csv`. Unit = **54 rows** (9 models × 6 benchmarks). Claims are **associational**, not causal, and **not** strictly non-circular for judge-based pairs. |
| What’s missing for stronger tests | **Held-out questions** with separate judge labels; **full trace** judge coverage; **deployment outcome** independent of confidence binning (e.g. human eval); explicit **pass@16** column (must **recompute** from JSONL). |

---

## STEP 1 — Inventory

A structured CSV is committed as **`analysis/frs_predictor_artifact_inventory.csv`**. Below is a condensed map by theme.

### FRS and paper tables

| Path | Role |
|------|------|
| `global_pass1_frs_analysis/paper_frs_by_benchmark.csv` | Wide table: per-benchmark **FRS** columns + `FRS_Avg` + `Acc_Avg` (9 models). |
| `global_pass1_frs_analysis/paper_pass1_reasoning_by_benchmark.csv` | Long table: `base_acc` (= pass@1 in `global_pass1_frs_pairwise_analysis.py`), `base_reasoning`, `snr`. |
| `global_pass1_frs_analysis/merged_pass1_frs_per_benchmark.csv` | **Per (model, benchmark):** `frs_pct`, `pass1_pct`, `base_reasoning`, `snr`. |

### Confidence-based accuracy (no judge)

| Path | Role |
|------|------|
| `topk_ablation_results/topk_ablation_results.csv` | `top_k_pct` ∈ {10,20,30,50,70,100}: **accuracy** in the top‑K% of traces by **token confidence**; `n_traces`, `n_total`, `mean_confidence`. |

### FRS pipeline outputs (judge + confidence bins)

| Path | Role |
|------|------|
| `reasoning_confidence_bins_results/reasoning_by_confidence_bin.csv` | Per-bin **sampled** judge means + `mean_accuracy_population` in each bin. |
| `reasoning_confidence_bins_results/reasoning_cumulative_topk.csv` | Cumulative top‑k% **mean_reasoning_score** vs `cumulative_accuracy_population`. |
| `reasoning_confidence_bins_results/judging_checkpoints/judged_<Model>__<Dataset>.json` | **Verified:** `judged_samples` is a **list**; each item has `idx`, `trace_idx`, `bin_label`, `confidence`, `correct`, `reasoning_score` (0–1 in sample), `judge_scores` {faithfulness, utility, coherence, factuality}. Example: **250** traces for `DS-R1-7B` × `GSM8K`. |

### Unfiltered reasoning baseline (different judge sample)

| Path | Role |
|------|------|
| `analysis_outputs/unfiltered_reasoning/per_pair_scores.csv` | One row per model×dataset: **mean_reasoning_score** over **100** judged traces (one trace per sampled question). |
| `analysis_outputs/unfiltered_reasoning/judging_checkpoints/unfiltered_judged_*.json` | `judged_samples` keyed by `"idx:trace_idx"`; `reasoning_score` in 0–1 range in sample. |

### Code defining FRS-style metrics

| Path | Role |
|------|------|
| `reasoning_confidence_bins.py` | Ranks traces by **token confidence**; top `top_pool_frac` (default 0.5); **5 equal-count bins**; FRS@k10-style = mean judge score in top bin (see `BinConfig`). |
| `analysis/run_self_consistency_frs_proxy.py` | Same binning with **self-consistency** as confidence; joins **existing** judge checkpoints. |

### Downstream / parquet

| Path | Role |
|------|------|
| `build_downstream_parquets.py` | Builds `problem_id`, `trace_id`, `confidence`, `correct`, `reasoning_score` parquets from JSONL + judge. |
| **`*.parquet`** in repo | **None found** (glob 0). |

---

## STEP 2 — Exact schemas (verified)

### `paper_frs_by_benchmark.csv` (header + 1 row)

```text
model,GSM8K,MATH,SVAMP,AQuA,GPQA,CSQA,FRS_Avg,Acc_Avg
DS-R1-7B,98.9,94.0,99.0,96.8,62.9,79.2,88.5,69.0
```

- Per-column **numeric** cells are **FRS-style percentages** per benchmark (not pass@1). **`Acc_Avg`** is a separate aggregate (see paper; not re-derived here).

### `paper_pass1_reasoning_by_benchmark.csv`

```text
model,benchmark,base_reasoning,base_acc,snr
DeepSeek-R1-Distill-Qwen-1.5B,GSM8K,66.79,60.1,-1.699841
```

- **`base_acc`** is treated as **pass@1** in `global_pass1_frs_pairwise_analysis.py` (see file docstring).

### `merged_pass1_frs_per_benchmark.csv`

```text
model,benchmark,frs_pct,pass1_pct,base_reasoning,snr
DS-R1-7B,GSM8K,98.9,91.5,88.8,3.799395
```

- **`frs_pct`** aligns with wide FRS table; **`pass1_pct`** aligns with `base_acc` after name mapping.

### `topk_ablation_results.csv`

```text
model,dataset,top_k_pct,accuracy,n_traces,n_total,mean_confidence
DS-R1-1.5B,AQuA,10,80.34,407,4064,0.2964
```

- **High-confidence accuracy** at percentile **top_k_pct** of the **trace** distribution (by confidence).

### `reasoning_by_confidence_bin.csv` (columns)

`model,dataset,bin_label,bin_start_pct,bin_end_pct,total_traces_in_bin,sampled_traces,random_seed,mean_reasoning_score,std_reasoning_score,stderr_reasoning_score,ci_lower,ci_upper,bootstrap_ci_lower,bootstrap_ci_upper,mean_accuracy_population`

### FRS checkpoint JSON (`judged_DS-R1-7B__GSM8K.json`)

- **`metadata`:** `input_file`, `n_pooled_traces`, `n_top_pool_kept`, `top_pool_fraction` (0.5), `samples_per_bin_target` (50), `timestamp`.
- **`judged_samples`:** list of objects with keys:  
  `idx`, `trace_idx`, `bin_label`, `confidence`, `correct`, `judge_scores`, `reasoning_score`, `judge_ok`.

### `per_pair_scores.csv` (columns)

`model,dataset,n_total_traces,n_distinct_questions,n_sampled_questions,n_judged_traces,mean_reasoning_score,std_reasoning_score,stderr_reasoning_score,cache_hits,new_judge_calls,seed,jsonl_path,sampled_questions_path,sampled_traces_path,checkpoint_path`

### `unfiltered_judged_*.json` (structure)

- Top-level: `model`, `dataset`, `jsonl_path`, `bin_label`, `judged_samples` as **object** map `"idx:trace_idx"` → `{reasoning_score, judge_raw, ...}`.

### `ranking_comparison_vs_frs.csv`

```text
model,unfiltered_reasoning_avg,frs_avg,unfiltered_rank,frs_rank,delta_rank
DS-R1-1.5B,0.6310416666666666,79.9,7,2,5
```

- Note `unfiltered_reasoning_avg` is **0–1 scale**; `frs_avg` is **0–100 scale** from paper CSV — **scale mismatch** in the same row (compare ranks, not raw Pearson without rescaling).

---

## STEP 3 — Metric → data mapping

| Metric | Confirmed in repo? | Granularity | Primary file(s) | Code path |
|--------|-------------------|-------------|-----------------|-----------|
| **pass@1** | **Yes** | model × benchmark | `merged_pass1_frs_per_benchmark.csv` (`pass1_pct`), `paper_pass1_reasoning_by_benchmark.csv` (`base_acc`) | `global_pass1_frs_pairwise_analysis.load_paper_merged` |
| **pass@16** | **No dedicated column** | — | Recomputable from `source_pass16` JSONL: per-problem `any(score)` over 16 traces | Not found as precomputed CSV; **topk_ablation** / **sample_count** use **k** subsamples of traces, not pass@16 as a scalar column |
| **Unfiltered reasoning score** | **Yes** | model × dataset (macro over 6 benchmarks also in `ranking_comparison_vs_frs.csv`) | `per_pair_scores.csv` (`mean_reasoning_score`); checkpoints | `analysis/run_unfiltered_reasoning_baseline.py` |
| **High-confidence accuracy** | **Yes** | model × dataset × top_k_pct | `topk_ablation_results.csv` | `topk_ablation.py` |
| **FRS** (paper default) | **Yes** | model × benchmark; macro `FRS_Avg` | `paper_frs_by_benchmark.csv`, `merged_pass1_frs_per_benchmark.csv` (`frs_pct`) | `reasoning_confidence_bins.py` + paper tables |

**Recomputable:** pass@16, trace-level confidence, bin assignments from JSONL + `compute_trace_confidence` (`topk_ablation.py`). **Judge** scores are **not** recomputable without API.

**Coverage:** 9 models × 6 benchmarks for merged FRS/pass1; same for `topk_ablation_results.csv` (324 rows = 54×6 percentile rows). Unfiltered `per_pair_scores.csv` lists **54** pairs in the sample read.

---

## STEP 4 — Predictor test feasibility

| Analysis | Status | Why |
|----------|--------|-----|
| **A. Correlation across model×dataset pairs** | **Fully possible** (associational) | Join `merged_pass1_frs_per_benchmark.csv` + `per_pair_scores` + aggregate `topk` (e.g. filter `top_k_pct=10`) on `model`,`dataset`. **N** = 54. |
| **B. Held-out split by questions** | **Not possible** from summary CSVs alone | Need per-`idx` outcomes; **partial** data in judge JSONs (~250 idx per pair, not full dataset). Would require **script** on JSONL + checkpoints + explicit split. |
| **C. Selection-gain (metric predicts confidence-selection beats random)** | **Approximately possible** | Bin-level CSVs give **mean_accuracy_population** vs **mean_reasoning_score** by bin; **not** a clean “selection policy” A/B without defining random baseline on **same** judged set. `topk_judge_eval` results (if present) compare most vs least confident **per problem**—different sample. |
| **D. Pairwise discrimination (similar pass@1)** | **Already partially done** | `global_pass1_frs_analysis/pass1_vs_frs_summary.txt` documents pairwise \|Δpass@1\| vs \|ΔFRS\|. **Predictor** framing: use merged table + pairwise script. |
| **E. Regression / residual (FRS beyond pass@1, pass@16, unfiltered, top‑k acc)** | **Approximately possible** | Multicollinearity: **pass@1**, **FRS**, **SNR** share model/dataset structure; **unfiltered** is same judge family. **pass@16** must be added by recomputation. |

---

## STEP 5 — Strongest deployment-aligned target (today)

**Available:**

1. **Mean GPT-4o-mini reasoning score in top confidence decile (top 10% of traces)** — `reasoning_cumulative_topk.csv` where `topk_pct=10` (`mean_reasoning_score` + `cumulative_accuracy_population`. **This is adjacent to FRS** (FRS uses top **bin** inside top 50% pool, 5 bins—not identical to global top 10% but **highly correlated**).

2. **Unfiltered mean reasoning** — one random trace per sampled question (100 questions); **not** confidence-selected.

3. **Top‑10% accuracy** (`topk_ablation_results`) — **correctness**, not judge quality.

**Circularity:**

| Target | Risk |
|--------|------|
| Top-bin / top‑decile **mean reasoning** (judge) | **High** vs FRS (same judge, same confidence ordering family). |
| **Unfiltered** mean reasoning | **Moderate** — same judge rubric, **different** trace selection. |
| **pass@1** / **top‑k accuracy** | **Low** relative to FRS definition — different signal (accuracy vs judge). |

**Strongest “deployment” proxy that is **not** accuracy-only:** **Unfiltered mean reasoning** or **cumulative top‑10% mean** `mean_reasoning_score` in `reasoning_cumulative_topk.csv`, with explicit caveat that **cumulative top‑10%** is **not** independent of FRS.

---

## STEP 6 — Circularity risks (blunt)

| Analysis | Risk | Reason |
|----------|------|--------|
| “FRS predicts **top‑bin** / **cumulative top‑10%** reasoning” | **High** | Same judge; confidence-based strata. |
| “FRS predicts **unfiltered** reasoning mean” | **Moderate** | Same judge; **different** trace selection (random question, one trace). |
| “FRS predicts **pass@1** or **top‑10% accuracy**” | **Low** for circularity | Different target; **but** confounds (model capability) remain. |
| “FRS vs **SNR**” (from merged table) | **Moderate** | SNR is **derived** from `base_reasoning` and `base_acc` in paper table—not independent of pass@1. |

---

## STEP 7 — Recommended analysis (best under constraints)

**Outcome (Y):** `mean_reasoning_score` at **top10** from `reasoning_cumulative_topk.csv` **OR** (less circular) **`mean_reasoning_score` from `per_pair_scores.csv`** (unfiltered).

**Predictors (X):** `frs_pct`, `pass1_pct`, `base_reasoning`, `snr` from `merged_pass1_frs_per_benchmark.csv`; add **`accuracy` where `top_k_pct=10`** from `topk_ablation_results.csv` (pivot to wide).

**Unit:** One row per **(model, benchmark)** — **54 rows**.

**Files:**  
`global_pass1_frs_analysis/merged_pass1_frs_per_benchmark.csv`,  
`reasoning_confidence_bins_results/reasoning_cumulative_topk.csv` (or `per_pair_scores.csv`),  
`topk_ablation_results/topk_ablation_results.csv`.

**Script:** New **`analysis/run_frs_predictor_panel.py`** (or notebook): merge on `model`,`dataset`/`benchmark`; standardize z-scores; fit OLS or report partial Spearman with bootstrap; **no** claim of causality.

**Outputs:**  
- `analysis_exports/frs_predictor_panel_merged.csv`  
- `frs_predictor_partial_correlations.csv`  
- Scatter: FRS vs unfiltered reasoning; **partial** residual of Y ~ pass@1 vs FRS.

**Supports:** “**After controlling for pass@1 and top‑10% accuracy, FRS still associates with unfiltered mean judge reasoning**” (or does not)—**associational**.

**Does not support:** “FRS **causes** better deployment” or non-circular **prediction** of **independent** human evaluation.

---

## STEP 8 — Minimum extra work for stronger tests

| Need | Effort |
|------|--------|
| **pass@16** as explicit column | **Recompute** from JSONL only (no judge). |
| **Held-out question split** | **Join** `idx` across JSONL and judge lists; split `idx`; recompute bin metrics on train vs judge on test — **requires new code**; judge calls only if test questions lack labels. |
| **Full trace-level judge** | **N × (dataset size) × cost** — large. |
| **Non-circular deployment target** | **New** human ratings or **external** task success — not in repo. |

---

## Final decision block

### Best analysis we can run now

**Panel associational study** on **54 (model, benchmark)** points: predict **unfiltered mean reasoning** (or explicit top‑decile judge mean) from **FRS**, **pass@1**, **top‑10% accuracy**, and **SNR**, with **partial correlations** and **multicollinearity** checks. **Files:** merged CSV + `per_pair_scores` + `topk_ablation_results` + `reasoning_cumulative_topk.csv`.

### Best stronger analysis with minimal extra work

1. **Recompute pass@16** from JSONL and add to panel.  
2. **Per-question** join: for each `idx` in a judge file, attach **max confidence** and **whether trace in top bin**; fit **mixed** model or **question-level** test — **still same judge** for Y if Y is reasoning score.

---

## Appendix — Definitions (multiple FRS-like quantities)

1. **Paper `FRS_Avg` / `frs_pct`:** From **`paper_frs_by_benchmark.csv`** / merged table — **rubric** FRS per benchmark (paper pipeline).  
2. **Bin pipeline FRS@k10:** From **`reasoning_confidence_bins`** — mean judge score in **first bin** of **top 50%** pool, **5 bins**.  
3. **Self-consistency proxy FRS:** From **`analysis/run_self_consistency_frs_proxy.py`** — vote-share confidence; same binning.  

**Do not mix** these without **renormalization** or column labels (see `analysis_exports/self_consistency_proxy_robustness/` for SC vs default ranks).

---

*Machine-readable inventory: `analysis/frs_predictor_artifact_inventory.csv`*
