# Confidence proxy feasibility (no new generation)

This report documents **what token-level information exists in saved artifacts today** and which alternative confidence proxies can be computed **without rerunning model inference**. Findings are based on code inspection in `threshold/` and spot-checks of JSONL rows (2026-03).

---

## 1. Canonical raw data: pass@16 JSONL (`source_pass16_jsonl_by_model*/**/*.jsonl`)

These files are the **primary** source for FRS / top-k / binning experiments (`topk_ablation.py`, `reasoning_confidence_bins.py`, `k_sensitivity_analysis.py`, `sample_count_ablation.py`, `build_downstream_parquets.py`).

### Verified top-level keys (representative `*_processed.jsonl` row)

From a DeepSeek GSM8K processed line:

- `idx` — problem id  
- `score` — list of 16 booleans (per trace correctness)  
- `code` — list of 16 strings (full CoT text per trace)  
- `pred`, `question`, `gt`, …  
- **`chosen_token_probs_per_path`** → `{"epoch_0": [trace0_probs, trace1_probs, …]}`  
  - Each `trace_k_probs` is a **flat list of per-token probabilities for the chosen token at each generation step** (length = number of generated tokens in that trace).  
  - Values are in **probability space** (observed range includes values in `(0, 1]`).  
- **`chosen_token_ids_per_path`** — parallel token ids (same structure).  
- **`probability_log`** / **`probability_log_per_path`** — present in schema; in spot-checks on main pass@16 GSM8K files, **per-path log lists were empty** while `chosen_token_probs_per_path` was populated. **Do not assume** log arrays are filled; **derive** \(\log p\) from probabilities when needed (clip to avoid \(\log 0\)).  
- **`entropies`** / **`entropies_per_path`** — present in schema; in the same GSM8K spot-check, **`entropies_per_path["epoch_0"]` had 16 entries but per-trace entropy token lists were empty lists** `[]`. Treat as **often unused in practice** unless you verify per file.  
- **`answer_confidence`** — single float per **problem row** (not per trace); not used by `compute_trace_confidence`.  
- **`expert_cot_probability`** — appears on some benchmarks; not the main FRS pipeline.

### What is **not** stored (verified absence in sampled rows / key unions)

- **Full vocabulary logits** or **top-k alternative token** probabilities at each step → **not available** for margin / “1 − p_top + p_second” style metrics.  
- **Per-step entropy** from the full softmax → **not reconstructible** from chosen-token prob alone (unless `entropies_per_path` is non-empty for that trace—see below).  

---

## 2. How the codebase currently defines “confidence”

| Location | Definition |
|----------|------------|
| `topk_ablation.compute_trace_confidence` | Mean of the **lowest 10%** of **chosen-token probabilities** (partition-based, same as `LOW_PROB_CUTOFF = 0.10`). |
| `correctness_conditioned.compute_trace_confidence` | Same recipe. |
| `topk_judge_eval.compute_trace_confidence` | Same recipe. |
| `temp0_confidence_analysis.confidence_low10` | **Sorted** lowest 10% of token probs (slightly different tie-breaking vs `np.partition`; same spirit). |
| `temp0_confidence_analysis.confidence_full_mean` | **Mean of all** chosen-token probabilities in the trace. |
| `temp0_confidence_analysis._maybe_logprobs_to_probs` | If values look like **logprobs** (≤0), converts with `exp` to probabilities for downstream use. |

**Important:** The pipeline assumes **one scalar per trace** derived from the **chosen-token probability list** only.

---

## 3. Downstream / judge artifacts (not full token sequences)

| Artifact | What is stored |
|----------|----------------|
| `reasoning_confidence_bins_results/judging_checkpoints/judged_<Model>__<Dataset>.json` | Per judged trace: `idx`, `trace_idx`, `bin_label`, **`confidence`** (scalar, same recipe as above), `correct`, `judge_scores`, `reasoning_score`. **No** per-token arrays. |
| Parquet from `build_downstream_parquets.py` | Columns like `confidence`, `reasoning_score`, `correct`, ids — **aggregated / merged**, reproducible from JSONL + judge JSON. |

Recomputing a **different** proxy for **FRS** (which needs binning / ranking by confidence) requires going back to **JSONL** `chosen_token_probs_per_path`, not only the judge JSON (unless you only study the 250 judged traces).

---

## 4. T=0 / prob sidecars (optional second source)

Under `t0_temp0_extracted/temp0/...` and `*_prob.jsonl` files, rows can include top-level `probability_log`, `entropies`, etc. (structure may differ from pass@16). These are **additional** runs, not the main T=0.7 pass@16 pool used for the core FRS tables unless your experiment explicitly targets T=0 files.

---

## 5. Proxy feasibility matrix

Legend: **YES** = can compute from typical pass@16 JSONL with populated `chosen_token_probs_per_path`; **MAYBE** = depends on non-empty optional fields; **NO** = missing information.

| Candidate proxy | Feasible now? | Notes |
|-----------------|---------------|--------|
| Full-trace **mean probability** (arithmetic mean of chosen \(p_t\)) | **YES** | Same as `confidence_full_mean` in `temp0_confidence_analysis.py`; implement in one pass over the list. |
| Full-trace **mean log-prob** \(\frac{1}{T}\sum \log(p_t)\) | **YES** | Use \(p_t\) from JSONL; use \(\log(\max(p_t,\epsilon))\) to avoid \(-\infty\). |
| **Sequence NLL** \(-\sum_t \log(p_t)\) or **per-token average NLL** | **YES** | Same inputs; standard for “length-normalized” use divide by \(T\). |
| **Geometric mean** probability \(\exp(\text{mean}(\log p))\) | **YES** | Equivalent to mean log-prob in log space. |
| Bottom **5% / 10% / 20%** mean **probability** (tail / “worst tokens”) | **YES** | Generalize `compute_trace_confidence` with a `LOW_PROB_CUTOFF` parameter (5%, 10%, 20%). Code pattern exists in `topk_ablation` / `reasoning_confidence_bins` style. |
| **Min** token probability (or min log \(p\)) | **YES** | `np.min` on the array. |
| **q-th percentile** of token probs (or of \(\log p\)) | **YES** | `np.percentile` on the per-token list. |
| **True Shannon entropy** per trace \(-\sum_i p_i \log p_i\) over the vocabulary at each step | **NO** | Requires full distribution at each step; not stored. |
| **Entropy using stored `entropies_per_path`** | **MAYBE** | If a file has **non-empty** per-token entropy lists, you could aggregate (e.g. mean). Spot-checks on main pass@16 GSM8K showed **empty** lists—**verify per dataset file** before relying on this. |
| **Margin** (e.g. \(p^* - p^{(2)}\)) | **NO** | Second-best probability not in artifacts. |
| **ECE / calibration bins** | **NO** (not from logits alone without labels binning design) | You have correctness + scalar confidence; you could compute **empirical calibration** of a *scalar* confidence vs accuracy—**YES** as a **derived analysis**, not “entropy of logits”. |

### Classification (required wording)

| Proxy | Classification |
|-------|------------------|
| Mean / geom-mean / NLL / min / percentile / tail-means (5–20%) from **chosen_token_probs** | **Possible now from saved artifacts** (pass@16 JSONL) + small new code. |
| Same, using **precomputed `confidence` in judge JSON** | **Possible now** only for the **subset of judged traces**; **not** sufficient to replicate full-pool FRS without re-reading JSONL. |
| **True** vocab entropy, margin, top-k logprobs | **Possible only if we rerun inference** (or change generation script to log them), **or** MAYBE if you discover files with full distributions (none found in audited schema). |

---

## 6. Exact files / scripts to use next

### Read token probabilities

- **Data:** `source_pass16_jsonl_by_model*/**/*_processed*.jsonl` (prefer `pick_best_jsonl` logic in `build_downstream_parquets.py` to pick one file per `(model, dataset)`).  
- **Loader reference:** `topk_ablation.load_jsonl_raw` → `token_probs_all` from `chosen_token_probs_per_path["epoch_0"]`.  
- **Alternative loader:** `temp0_confidence_analysis.get_trace_token_prob_lists` (handles `probability_log` / `chosen_token_probs` fallbacks for T=0-style rows).

### Implement a new scalar from the same arrays

- Copy the loop structure from `topk_ablation.records_to_trace_list` or `reasoning_confidence_bins.traces_to_dataframe`: iterate traces, get `probs` list, apply your function instead of `compute_trace_confidence`.

### Reuse existing variants

- **Full mean vs low-10%:** `temp0_confidence_analysis.py` — `confidence_full_mean`, `confidence_low10`.  
- **Tail mean (10%):** `compute_trace_confidence` in `topk_ablation.py`.

### FRS-style re-ranking (same judge scores, new confidence)

- **Judge scores** (reasoning) stay in `judged_*.json`.  
- **Re-bin / re-rank** requires recomputing confidence for **every** trace from JSONL, then repeating the binning + optional judge subset logic—**no new generation**, but **non-trivial** re-execution of `reasoning_confidence_bins`-style logic with a new `compute_*` function.

---

## 7. Recommendation: highest-ROI “tonight” robustness test

**Recommended:** **Full-trace mean log-probability** (or equivalently **mean \(\log p\)** then report on log scale, or **geometric mean probability**).

**Why**

1. **Verified inputs:** Only needs `chosen_token_probs_per_path` (already present).  
2. **Interpretable contrast:** Your current metric emphasizes the **tail** (lowest 10%); the mean / mean log is a **global** signal—strong robustness check.  
3. **Minimal code:** One function mirroring `confidence_full_mean` but on `np.log(np.maximum(p, eps))`.  
4. **Same machinery:** Reuse existing JSONL discovery (`build_file_map` / `dedupe_files` patterns).

**Runner-up:** **Bottom-20% mean probability** — trivial parameter change from `LOW_PROB_CUTOFF` (0.05 / 0.10 / 0.20) in a fork of `compute_trace_confidence`.

**Not recommended tonight without file-specific validation:** Entropy from `entropies_per_path` (often empty on pass@16 samples).

---

## 8. Blockers / caveats

1. **`probability_log_per_path` often empty** on audited pass@16 files—treat **probabilities** as ground truth; derive logs yourself.  
2. **`entropies_per_path` often empty** — do not build the paper’s entropy analysis on this field without per-file validation.  
3. **No margin / full-distribution entropy** without new inference or richer logs.  
4. **Judge JSON** stores one scalar confidence per judged trace—**cannot** recover alternative proxies for *unjudged* traces without JSONL.  
5. **Parquet** outputs (if generated elsewhere) typically store **final** `confidence` only—recompute from JSONL for new definitions.

---

## 9. Quick reference: key scripts touching token probs

| Script | Role |
|--------|------|
| `topk_ablation.py` | Loads `chosen_token_probs_per_path`, defines `compute_trace_confidence` (10% tail mean). |
| `reasoning_confidence_bins.py` | Same; bins + judge. |
| `temp0_confidence_analysis.py` | `confidence_low10`, `confidence_full_mean`, `_maybe_logprobs_to_probs`, T=0 discovery. |
| `k_sensitivity_analysis.py` | Pools traces with precomputed confidence. |
| `correctness_conditioned.py` | Median split analysis using `compute_trace_confidence`. |
| `build_downstream_parquets.py` | Merges JSONL + judge → parquet columns. |
| `topk_judge_eval.py` | Judge + duplicate `compute_trace_confidence`. |

---

*Generated as part of a repository audit; values “empty / present” may vary by model run—spot-check a second `*_processed.jsonl` if your pair behaves differently.*
