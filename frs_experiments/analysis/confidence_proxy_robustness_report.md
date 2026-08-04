# Confidence proxy robustness for FRS (pass@16, no new inference)

## What was run

Script: `analysis/run_confidence_proxy_robustness.py`

For each of **54** `(model, benchmark)` pairs (9 models × 6 benchmarks):

1. **JSONL:** `chosen_token_probs_per_path.epoch_0` via `topk_ablation.load_jsonl_raw` (same traces as the paper pool). Path chosen with `build_downstream_parquets.jsonl_from_judge_metadata` when present so `idx` / `trace_idx` align with judge checkpoints.
2. **Judge scores:** `reasoning_confidence_bins_results/judging_checkpoints/judged_<Model>__<Dataset>.json` — `reasoning_score` on 0–1 converted to 0–100.
3. **Binning (paper settings):** `top_pool_frac = 0.5`, `n_bins = 5`, matching `reasoning_sampling_metadata.json`. Implementation mirrors `reasoning_confidence_bins.slice_top_fraction` + `assign_disjoint_bins` (inlined in the script as `BinConfig` + `assign_disjoint_bins`).
4. **FRS (first bin):** Mean `reasoning_score` (0–100) over **judged** `(idx, trace_idx)` with **`bin_id == 0`** after sorting by the proxy (highest-confidence equal-count bin in the top-50% slice).  
   - **Important:** Checkpoint JSON may label this bin `"0-10"` while `BinConfig(n_bins=5)` string labels look like `"0-20"`, `"20-40"`, … We **do not** match on label strings; we use **`bin_id == 0`** so the bin matches the first decile of the sliced pool in count.

### Proxies

| `proxy_name` | Formula |
|--------------|---------|
| `bottom10_mean_prob` | `topk_ablation.compute_trace_confidence`: mean of lowest 10% of chosen-token probabilities (partition). |
| `full_trace_mean_logp` | \(C = \frac{1}{T}\sum_j \log(\max(p_j, \varepsilon))\), \(\varepsilon = 10^{-12}\). |
| `bottom20_mean_prob` | Same as bottom-10% but lowest **20%** of token probabilities. |

### Outputs

Directory: `analysis_exports/confidence_proxy_robustness/`

| File | Description |
|------|-------------|
| `proxy_benchmark_results_k10.csv` | Per model × benchmark × proxy: `frs_k10_bin010`, `n_judged_in_bin010`, `n_traces_in_top_pool_slice`. |
| `proxy_rankings_k10.csv` | Mean FRS over 6 benchmarks, rank per proxy. |
| `proxy_rank_comparison.csv` | Wide ranks + rank deltas vs current proxy. |
| `proxy_spearman_summary.csv` | Spearman ρ between model-level ranking vectors. |
| `proxy_key_reversals_summary.csv` | DS-R1-7B, DS-R1-1.5B, Qwen2.5-7B, Phi-4, Phi-4-Reas. |
| `proxy_data_coverage.csv` | Per-pair trace counts and paths. |
| `proxy_metadata.json` | Formulas, settings, notes. |

**Cumulative K beyond bin 1:** Not run by default; use `--also-cumulative-k` if needed.

---

## Coverage / missingness

- **54 / 54** pairs: judge JSON + JSONL present (`missing_or_skipped_pairs` empty in `proxy_metadata.json`).
- **Malformed token lists:** Counted in `proxy_data_coverage.csv` as `bad_empty_prob_lists_count` (skipped trace rows).

---

## Results (summary)

### Model ranks by mean FRS (first bin), averaged over 6 benchmarks

| Rank | bottom10_mean_prob | full_trace_mean_logp | bottom20_mean_prob |
|------|-------------------|----------------------|--------------------|
| 1 | DS-R1-7B | DS-R1-7B | DS-R1-7B |
| 2 | DS-R1-1.5B | DS-R1-1.5B | DS-R1-1.5B |
| 3 | Qwen3-4B | **Phi-4** | Qwen3-4B |
| 4 | Phi-4 | **Qwen3-4B** | Phi-4 |
| 5 | Qwen2.5-Math | Qwen2.5-Math | Qwen2.5-Math |
| 6 | Phi-4-Reas. | Phi-4-Reas. | Phi-4-Reas. |
| 7 | Qwen2.5-7B | Qwen2.5-7B | Qwen2.5-7B |
| 8 | LLaMA-3.1-8B | LLaMA-3.1-8B | LLaMA-3.1-8B |
| 9 | Gemma-7B | Gemma-7B | Gemma-7B |

**Spearman ρ (model ranks, n=9):**

- bottom10 vs full_trace_mean_logp: **0.983**
- bottom10 vs bottom20: **1.0**
- full_trace_mean_logp vs bottom20: **0.983**

**Only rank change vs bottom-10%:** Under **full_trace_mean_logp**, **Phi-4** and **Qwen3-4B** swap places (ranks 3 ↔ 4). All other models keep the same rank across all three proxies.

---

## Qualitative checks (paper narrative)

- **DS-R1-7B** remains **#1** under all three proxies.
- **DS-R1-1.5B** remains **#2** (strong second tier vs weaker models).
- **Qwen2.5-7B** remains mid–lower pack (**#7**) — does not jump toward the top under alternative proxies.
- **Gemma-7B** remains **#9**.
- **Major reversals** (e.g. Phi-4 vs Qwen3-4B) are **almost** unchanged; one pairwise swap appears under mean log-prob.

**Conclusion:** Under this re-analysis, the **global ranking story is highly stable** across tail-mean (10% vs 20%) and full-trace mean log-probability. The main qualitative conclusions for these nine models appear **robust** for appendix use, with the explicit caveat below.

---

## Caveats vs the published Table 2 pipeline

1. **Aggregation:** Published `reasoning_by_confidence_bin.csv` uses the mean of **up to 50 API-judged traces** sampled per bin. Here, FRS is the mean of **all judged traces** that fall in `bin_id == 0` after re-ranking by the proxy (often ~50, but can differ — see `n_judged_in_bin010` in the CSV).
2. **Numeric match:** Do not expect **bit-identical** FRS numbers to `bootstrap_table2_std.py` or the paper table; the **ranking** comparison is the robustness target.
3. **Bin label strings:** Judge files use `"0-10"`; code-generated labels for `n_bins=5` may differ — we align on **`bin_id`**, not string.

---

## Suggested appendix sentence (if desired)

> We recomputed filter-and-bin FRS using the same pass@16 token-probability traces and judge scores, replacing the default bottom-10% mean token probability with (i) mean log-probability of chosen tokens and (ii) bottom-20% mean token probability. Model rankings were nearly unchanged (Spearman ρ ≥ 0.98 vs the default proxy across nine models); the only swap among top models was between Phi-4 and Qwen3-4B under mean log-probability.

---

## Best ROI follow-up

If one more check is needed: focus on **pairwise benchmark** tables in `proxy_benchmark_results_k10.csv` where `n_judged_in_bin010` differs a lot across proxies (confidence-sensitive pairs).
