# Self-consistency confidence proxy — robustness report

**Generated:** 2026-03-31  
**Script:** `analysis/run_self_consistency_frs_proxy.py`  
**Export directory:** `analysis_exports/self_consistency_proxy_robustness/`

This is an **appendix / robustness** analysis. It does **not** replace the paper’s primary logit-based FRS procedure.

---

## What was run

- **Inputs (no new generation, no judge API):**
  - Pass@16 JSONL under `source_pass16_jsonl_by_model*/**/*.jsonl` (full lines, including `pred`, `score`, `idx`).
  - Existing judge checkpoints: `reasoning_confidence_bins_results/judging_checkpoints/judged_<Model>__<Dataset>.json`.
- **Discovery / alignment:** Same as `run_confidence_proxy_robustness.py`: `discover_jsonl_groups`, `jsonl_from_judge_metadata` when present.
- **Binning:** Reused from `analysis/run_confidence_proxy_robustness.py` (imported via `importlib`): `top_pool_frac=0.5`, `n_bins=5`, equal-count bins; **FRS@k10** = mean `reasoning_score` (0–100) over **judged** traces with `bin_id == 0` after ranking by `conf_sc`.

---

## Self-consistency definition (vote share)

For each problem line in JSONL with `k = len(pred)` (typically 16) and trace index `t`:

\[
\text{conf\_sc}(t) = \frac{1}{k}\sum_{j=0}^{k-1} \mathbf{1}[\texttt{norm}(\texttt{pred}[j]) = \texttt{norm}(\texttt{pred}[t])]
\]

**Normalization (`norm`):** `str(pred).strip()`. Empty string after strip stays empty; empty answers agree only with other empty normalized answers.

**No case-folding** (conservative; avoids merging `A` vs `a` unless we add a benchmark-specific policy later).

---

## Data coverage (critical)

| Quantity | Typical magnitude |
|----------|-------------------|
| Traces per model×benchmark (JSONL) | thousands–tens of thousands |
| Judged traces in checkpoint | **250** per pair |
| Traces in **bin 0** after SC ranking | hundreds–**~2000** (benchmark-dependent) |
| **Judged** traces that fall in bin 0 | **~10–46** per pair (see `sc_data_coverage.csv`) |
| **judged_coverage_fraction** in bin 0 | often **~0.02–0.07** |

**Interpretation:** FRS@k10 under self-consistency is **not** a stable estimate of “true” bin-0 quality; it is the mean of **judged** scores that happen to land in the highest-confidence bin after re-ranking. **Absolute levels** are not comparable to Table 2’s published FRS numbers. The **relative** model ranking is the main object.

**Empty `pred`:** Several benchmarks show `has_empty_preds=True` in `sc_data_coverage.csv`; agreement is defined on stripped strings, including empty.

---

## Model-level results (k10)

**Mean `sc_frs_avg_k10` over 6 benchmarks** (equal weight per benchmark). **Ranks:** 1 = best (highest mean).

| Rank | Model | sc_frs_avg_k10 | Mean judged cov. in bin0 (across benches) |
|------|-------|----------------|-------------------------------------------|
| 1 | DS-R1-7B | ~94.07 | ~0.032 |
| 2 | Qwen2.5-Math | ~90.48 | ~0.030 |
| 3 | DS-R1-1.5B | ~86.16 | ~0.027 |
| 4 | Phi-4-Reas. | ~85.04 | ~0.037 |
| 5 | Phi-4 | ~85.01 | ~0.030 |
| 6 | Qwen3-4B | ~82.19 | ~0.026 |
| 7 | Qwen2.5-7B | ~79.55 | ~0.029 |
| 8 | LLaMA-3.1-8B | ~77.01 | ~0.026 |
| 9 | Gemma-7B | ~62.02 | ~0.030 |

---

## Agreement with default paper FRS (Table-style averages)

**Source for “default”:** `global_pass1_frs_analysis/paper_frs_by_benchmark.csv` column `FRS_Avg` (same as other appendix analyses).

**Correlation (n = 9 models):**

| Metric | Spearman ρ | p-value | Pearson r | p-value |
|--------|------------|---------|-----------|---------|
| Model-level **FRS avg** (default vs SC) | **0.80** | ~0.0096 | **0.95** | ~1.1×10⁻⁴ |
| Model-level **rank** | **0.80** | ~0.0096 | — | — |

**Conclusion:** Rankings are **broadly consistent** with the default FRS ordering (high Spearman on both scores and ranks). This supports a **robustness narrative**: switching from logit-based confidence to a self-consistency vote-share proxy does **not** collapse the model ranking, despite sparse judged coverage in the top bin.

**Notable rank moves (see `sc_vs_default_rank_comparison.csv`):**

- **Qwen2.5-Math:** default rank **5** → SC rank **2** (largest upward move).
- **Qwen3-4B:** default rank **3** → SC rank **6** (largest downward move).
- **Phi-4-Reas.:** default **6** → SC **4**; **Phi-4:** default **4** → SC **5** (small swap between related models).

---

## Headline models (subset)

See `sc_key_models_summary.csv`. DS-R1-7B remains **rank 1** under both; most headline moves are ≤2 rank positions except where noted above.

---

## Conceptual caveats

1. **Self-consistency is problem-coupled:** all 16 traces share the same multiset of answers; scores are not independent across traces.
2. **Exact-string matching** on `pred`; no LaTeX/math equivalence (MATH500 may differ cosmetically).
3. **Judge subset** was sampled for the **logit** pipeline; SC re-ranking assigns **different** traces to bin 0 — **not** a reproduction of Table 2.
4. **Sparse bin-0 judge coverage** inflates variance; do not over-interpret small differences in `sc_frs_k10` between adjacent models.

---

## Files produced

| File | Description |
|------|-------------|
| `sc_proxy_benchmark_results_k10.csv` | Per model × benchmark FRS@k10 under SC + coverage |
| `sc_proxy_rankings_k10.csv` | Model-level macro mean + rank |
| `sc_vs_default_rank_comparison.csv` | Default vs SC ranks and averages |
| `sc_vs_default_correlation_summary.csv` | Spearman/Pearson summaries |
| `sc_key_models_summary.csv` | Headline models subset |
| `sc_data_coverage.csv` | Coverage, empty-pred flags |
| `sc_proxy_metadata.json` | Definitions + caveats |
| `sc_proxy_cumulative_k.csv` | *(If run with `--also-cumulative-k`)* cumulative FRS over bins 1–k for k20–k50 |

---

## Optional: cumulative K (k20–k50)

Run:

```bash
python3 analysis/run_self_consistency_frs_proxy.py --repo-root . --also-cumulative-k
```

Produces `sc_proxy_cumulative_k.csv` with `frs_cumulative` over the first 2–5 bins of the same top-50% slice.

---

## Suggested appendix paragraph (if included)

> **Answer-agreement (self-consistency) confidence.** As a robustness check orthogonal to token-probability confidence, we recomputed trace-level confidence as the fraction of stochastic samples that agree with each trace’s extracted final answer (exact match after whitespace stripping on stored `pred` strings), re-applied the same top-50% pool and five equal-count bins, and averaged existing GPT-4o-mini reasoning scores for judged traces in the highest-confidence bin. **No additional model inference or judging was performed.** Because judge coverage in the re-ranked top bin is sparse (~2–7% of bin traces in our checkpoint), absolute FRS values are not comparable to the main table; however, model-level rankings remained strongly correlated with the published FRS averages (Spearman ρ ≈ 0.8 across nine models), indicating that conclusions about relative model ordering are not an artifact of the logit-based confidence family alone.

---

## Honest assessment

- **Usable for appendix robustness / correlation story:** **Yes**, with the caveats above.
- **Usable to replace main FRS numbers:** **No**.
- **Best single number to cite:** Spearman ρ between default and SC **model-level** averages (and rank correlation), plus the explicit sparsity caveat.
