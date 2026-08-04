# Self-consistency / answer-agreement confidence — feasibility audit

**Date:** 2026-03-31  
**Scope:** Use **only** existing saved pass@16 JSONL and existing judge checkpoints — **no new model generation.**  
**Method:** Repo search, direct inspection of JSONL rows, `reasoning_confidence_bins_results` checkpoints, and `analysis/run_confidence_proxy_robustness.py` (proven pattern for alternate confidence → FRS-style metrics without regeneration).

---

## 1. Relevant artifacts and code (verified paths)

### Pass@16 trace sources (primary)

| Pattern | Role |
|--------|------|
| `source_pass16_jsonl_by_model*/**/*.jsonl` | One JSON object **per problem**; **16** traces per problem (T≈0.7 / pass@16 as used in paper). |
| Discovery | `build_downstream_parquets.discover_jsonl_groups`, `topk_ablation.build_file_map`, `extract_model_dataset`. |

**Filename convention:** `*_t0.7_*` in paths (verified on sample files).

### FRS / reasoning pipeline (logit baseline)

| Path | Role |
|------|------|
| `reasoning_confidence_bins.py` | Pools traces, ranks by **logit-derived** `compute_trace_confidence`, bins, judges subset. |
| `reasoning_confidence_bins_results/judging_checkpoints/judged_<Model>__<Dataset>.json` | GPT-4o-mini `reasoning_score` for sampled `(idx, trace_idx)`; includes `metadata.input_file` pointing at JSONL. |
| `analysis/run_confidence_proxy_robustness.py` | **No new inference:** recomputes alternate trace-level confidences from JSONL, re-ranks, re-slices top 50%, re-bins, joins **existing** `reasoning_score` by `(idx, trace_idx)` → `frs_k10_bin010`. |

### Unfiltered baseline (not required for self-consistency math)

| Path | Role |
|------|------|
| `analysis_outputs/unfiltered_reasoning/` | Separate judge run; **not** needed to define self-consistency from JSONL. |

### Search notes

- **No** dedicated `self_consistency`, `majority_vote`, or `parse_answer` module was found in the repo for evaluation.
- **No** `sympy` / LaTeX canonicalization utilities were found in `.py` files for answer comparison.
- `topk_ablation.load_jsonl_raw()` **drops** `pred` — it only keeps `idx`, `scores`, `token_probs_all`. Any self-consistency pipeline must **read full JSONL lines** (or extend `load_jsonl_raw`).

---

## 2. What is stored per problem / per trace (JSONL schema)

**Verified by reading live `json.loads(line)` from repo files** (one line each: AQuA, GSM8K, GPQA, MATH500, SVAMP, CommonsenseQA).

### Top-level keys (typical)

Includes at least: `idx`, `question`, `gt` (or `answer` in some rows — FRS code uses `row.get("gt", row.get("answer", ""))`), `score`, `pred`, `code`, `chosen_token_probs_per_path` → `epoch_0`, plus optional fields like `answer_text`, `exact_match_steps`, etc.

### Per-trace alignment

- **`score`**: list of length **16** — boolean (or 0/1) **correctness per trace** (already computed offline; this is the paper’s label).
- **`pred`**: list of length **16** — **string** “final answer” as produced by the evaluation harness (not raw CoT).
- **`code`**: list of length **16** — CoT text (full generated reasoning string).
- **Trace index**: integer `0 … 15` indexes into `pred`, `score`, `code`, and `chosen_token_probs_per_path["epoch_0"]` in lockstep.

### Grouping

- **One problem per line**; **`idx`** is the problem id (integer in samples).
- **All 16 samples for the same problem are on the same JSONL line** — no extra merge step.

### Example formats (observed)

| Benchmark | Example `pred` | Example `gt` |
|-----------|------------------|--------------|
| GSM8K | `'18'` | `'18'` |
| AQuA | `'A'` | `'E'` |
| GPQA | `'(C)'` | `'(A)'` |
| MATH500 | `'(3,\\frac{\\pi}{2})'` (LaTeX-like) | same style |
| SVAMP | `'145'` | `'145'` |
| CommonsenseQA | `'A'` or **`''`** (empty) | `'A'` |

**Critical:** At least one CommonsenseQA trace had **`pred == ''`**. Self-consistency must treat empty strings explicitly (e.g. exclude from agreement or bucket as “null answer”).

### Judged artifacts (`judged_*.json`)

- Stored as JSON list `judged_samples` (or `metadata` + samples — format varies slightly; older files use list under `judged_samples`).
- Each entry includes: **`idx`**, **`trace_idx`**, **`reasoning_score`**, **`confidence`** (logit), **`bin_label`**, **`correct`**, `judge_scores`, etc.
- **Does not** store `pred` or `code` — join back to JSONL via `(idx, trace_idx)` for any analysis that needs answers.

**Verified count:** `judged_DS-R1-1.5B__GSM8K.json` has **250** `judged_samples` entries (not full pool).

---

## 3. Feasibility of self-consistency confidence definitions

Assume **k = 16** traces per problem unless a line has fewer `score`/`pred` entries (then use actual `n`).

| Variant | Definition | Verdict | Notes |
|--------|------------|---------|--------|
| **A** | Per trace *t*: fraction of **other** traces whose `pred` equals `pred[t]` (or (count−1)/(k−1) style) | **YES** | Uses only `pred[]` on the JSONL line. |
| **B** | Per trace *t*: **vote share** of answer `pred[t]` = count(`pred == pred[t]`) / k | **YES** | Same data; standard “frequency of this trace’s answer.” |
| **C** | **Problem-level** scalar: e.g. max vote share / majority strength among unique answers | **YES** | Not a per-trace confidence for ranking traces — use if you want a problem-level diagnostic only. |
| **D** | Agreement after **benchmark-specific normalization** | **MAYBE / mostly NO in-repo** | No dedicated normalizer found in repo. Strings appear **already normalized by the upstream evaluator** when written to `pred`. Optional improvement: light `strip()`, case-fold for A–E MC, **not** verified against a gold standard library tonight. |
| **E** | **Exact string** agreement on `pred` | **YES** | **Conservative default for tonight.** Risk: MATH LaTeX strings could differ cosmetically if evaluator ever emitted two equivalent forms (not verified in bulk). |

**Classification**

- **Possible tonight from saved JSONL only:** **A, B, C, E** (with explicit handling of empty `pred`).
- **Possible with light reuse:** strip / uppercase for letter answers; **not** formally verified across all benchmarks.
- **Requires new work (not tonight):** full LaTeX / sympy equivalence for MATH — **not** present in repo.

---

## 4. Answer extraction / normalization (audit)

| Benchmark | Stored `pred` | Dedicated parser in repo? | Recommendation |
|-----------|----------------|---------------------------|----------------|
| GSM8K | Numeric string | **No** | Exact string match between traces is consistent if harness is consistent. |
| MATH500 | LaTeX-like string | **No** | Prefer **exact string** for self-consistency; semantic equivalence needs extra deps. |
| SVAMP | Numeric string | **No** | Exact string. |
| AQuA | Single letter | **No** | strip + upper case for agreement optional. |
| GPQA | e.g. `(A)` | **No** | Exact string observed; strip whitespace. |
| CommonsenseQA | Letter or empty | **No** | **Empty preds** — document or filter. |

**Bottom line:** `pred` is **already an extracted final answer** in the saved files. The repo does **not** add a second layer of normalization for these JSONL exports. Agreement is **implementation-defined** (exact string unless you add small rules).

---

## 5. Plugging into the current FRS machinery

### Can we assign a self-consistency score to each trace?

**Yes.** For each `(idx, trace_idx)` on a JSONL line, compute **A** or **B** from the 16 `pred` strings on that line.

### Can we rank traces within a model–benchmark pair?

**Yes.** Sort descending by self-consistency (higher = more agreement). Same as logit ranking in `run_confidence_proxy_robustness.py`, but replace `sort_col`.

### Can we form top pool + bins like FRS?

**Yes.** Same pipeline: `slice_top_pool` (e.g. top 50%), `assign_disjoint_bins` (e.g. 5 bins), identify bin `"0-10"` (first bin).

### Conceptual mismatch?

- **Logit confidence** is **per trace**, independent of other traces.
- **Self-consistency** is **per problem** — all traces on the same line share the same multiset of preds; scores are **coupled** across traces (sum of vote shares = 1 when using B with unique answers, etc.). This is **expected** for self-consistency and is not a bug, but it is a **different interpretation** than token-probability confidence.

### Joining to `reasoning_score` (FRS-style number)

**Same pattern as `run_confidence_proxy_robustness.py`:**

1. Build a DataFrame with one row per trace: `idx`, `trace_idx`, `self_consistency_proxy`, `correct`.
2. Rank by proxy, slice top pool, assign bins.
3. For rows in bin 0, if `(idx, trace_idx)` ∈ `load_judge_reasoning_map(judge_path)`, average `reasoning_score`.

**Important limitation (verified):** Judge files contain **~250** samples per pair (example: GSM8K), **not** every trace. Under a **new** ranking, the set of judged traces in “bin 0” may be **small or empty** for some pairs → `nan` or high variance. This is **already** the same structural limitation when swapping logit proxies; self-consistency does not change judge sparsity.

**No new generation** — but **new judge calls** are only needed if you insist on judging **new** `(idx, trace_idx)` that were never judged; the conservative “tonight” path **reuses only existing** judge entries.

---

## 6. Models and benchmarks coverage

**Models:** `build_downstream_parquets.MODEL_SLUG` lists **9** paper models (DS-R1-7B, DS-R1-1.5B, Qwen3-4B, LLaMA-3.1-8B, Qwen2.5-7B, Qwen2.5-Math, Gemma-7B, Phi-4, Phi-4-Reas.).

**Benchmarks:** Six core benchmarks match `reasoning_confidence_bins` / `DATASET_PANEL_ORDER`: GSM8K, MATH500, SVAMP, AQuA, GPQA, CommonsenseQA.

**Per pair:** `discover_jsonl_groups` yields one JSONL per model×benchmark when files exist.

---

## 7. Recommended fastest path for tonight

1. **Implement** a self-consistency column **B** (vote share of trace’s answer):  
   `conf_sc = count(pred[j] == pred[t]) / k` for each trace `t`, with `k = len(pred)` and **exact string** equality after optional `.strip()`.
2. **Extend** `analysis/run_confidence_proxy_robustness.py` (or a sibling script under `analysis/`) to:
   - Load JSONL with **full rows** (not `load_jsonl_raw` only) to get `pred`.
   - Filter traces with empty `pred` (log count) or assign `conf_sc = 0`.
   - Append `conf_sc` to the trace table; use it as `sort_col`.
   - Reuse `frs_bin010`, `load_judge_reasoning_map`, `Judge` paths unchanged.
3. **Report** `n_judged_in_bin010` per pair — if too low, interpret cautiously or pool cumulative bins (same script supports `--also-cumulative-k`).

**Why this variant:** **B** is standard, interpretable, per-trace, comparable across traces on the same line; **exact match** avoids new parser dependencies; implementation is **O(k)** per problem.

**Rankings vs current FRS:** You can produce **analogous** tables (per model×benchmark `frs_k10_bin010` under self-consistency ranking). **Global ranking correlation** with logit-FRS is **methodologically valid** only if you accept the same judge subsample limitation — **not** identical point estimates to the paper’s FRS column unless judge coverage is dense (it is not).

---

## 8. Blockers / risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Empty `pred` (CommonsenseQA) | Medium | Count/log; exclude or special bucket. |
| MATH LaTeX cosmetic mismatch | **Low–unknown** | Start with exact string; spot-check a few problems. |
| Sparse judge coverage | **High** for bin-specific FRS | Report `n_judged_in_bin010`; use cumulative bins if needed. |
| `load_jsonl_raw` omits `pred` | **Implementation** | Read full JSONL or extend loader. |

---

## 9. Exact next step (if you proceed)

1. Add `analysis/run_self_consistency_frs_proxy.py` (name flexible) that mirrors `run_confidence_proxy_robustness.py` but builds `conf_sc` from `pred` on full JSONL lines.
2. Run against repo root; output CSV next to `analysis_exports/confidence_proxy_robustness/`.
3. Compare Spearman / rank correlation vs existing proxy CSV **if** columns aligned.

---

## 10. Summary answers (constraints: conservative)

| Question | Answer |
|----------|--------|
| What answer info do we have? | **`pred[]`** per trace (16 strings), **`gt`**, **`score[]`** correctness, **`idx`**, **`code[]`** full CoT. |
| Feasible without rerunning inference? | **Yes** for computing self-consistency scores; **FRS-style metrics** need **existing** judge JSON only (no new generation). |
| Best variant tonight? | **Per-trace vote share (B)** with **exact-string** `pred` after strip; handle empty preds. |
| New code needed? | **Yes** — small script (~same size as `run_confidence_proxy_robustness.py`); **not** in repo yet. |
| Credible for paper? | **As a robustness check** alongside logit proxies — **yes**, with explicit caveats on judge sparsity and exact-string agreement. |

---

## Appendix: Example JSONL keys (AQuA, one line)

```
keys: answer_confidence, answer_text, ..., code, exact_match_steps, ..., gt, idx, pred, ..., score, question, ...
len(score)==len(pred)==16
pred[0] type: str  (e.g. 'A')
```
