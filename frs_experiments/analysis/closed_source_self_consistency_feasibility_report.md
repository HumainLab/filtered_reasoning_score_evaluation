# Closed-source self-consistency FRS experiment — feasibility audit

**Scope:** Evidence-only audit of this repository and the Portkey usage **as implemented in code**. No API calls were made; no generation jobs were run.

**Date:** 2026-03-31

---

## Executive summary

| Question | Finding |
|----------|---------|
| Is Portkey already the abstraction layer for **everything**? | **No.** Portkey (`portkey_ai.Portkey`) appears only for the **GPT-4o-mini judge** (`topk_judge_eval.py`). There is **no** Portkey-based (or other) **generation** client in the searched Python sources. |
| Can you run the full paper pipeline “as-is” for two closed models tonight? | **Not without new generation code and/or external tooling.** The repo **consumes** pre-built pass@k JSONL under `source_pass16_jsonl_by_model*/**/*.jsonl`; it does not contain a vLLM/OpenAI/Anthropic generation driver for frontier APIs. |
| Is k=8 practical? | **Yes in principle** (8 aligned list entries per problem). The analysis path for self-consistency FRS only needs parallel `pred[]` / `score[]` / `code[]` (see `analysis/run_self_consistency_frs_proxy.py`). **Implementation** is either 8 sequential chat calls or provider `n` if your Portkey route exposes it—**not implemented here.** |
| Biggest blocker | **Generation is out-of-repo** + **default judge drivers assume token-level probabilities** for trace selection in `topk_judge_eval.select_traces` and `reasoning_confidence_bins` trace loading—closed APIs often omit per-token logprobs. Self-consistency **metrics** do not need logprobs; the **stock** “pick traces then judge” scripts do. |

---

## 1. Model-calling infrastructure (what exists in code)

### 1.1 Portkey usage (evidence)

- **File:** `topk_judge_eval.py`
  - Imports `from portkey_ai import Portkey` and uses `self.client.chat.completions.create(...)` (OpenAI-compatible chat completions).
  - Default judge model string: `DEFAULT_JUDGE_MODEL = f"{DEFAULT_PORTKEY_GATEWAY}/{JUDGE_LLM}"` with `JUDGE_LLM = "gpt-4o-mini"` and `DEFAULT_PORTKEY_GATEWAY = "@irom-ll37364-op-b37b3e"` (lines 89–92). This is a **project-specific gateway prefix**, not a universal Portkey constant.
- **File:** `reasoning_confidence_bins.py` — imports `Judge` / `DEFAULT_JUDGE_MODEL` from `topk_judge_eval.py`; same Portkey path for judging.
- **File:** `analysis/run_unfiltered_reasoning_baseline.py` — uses the same `Judge`; requires `PORTKEY_API_KEY` for judging (not dry-run).

### 1.2 Grep results (generation / providers)

- **No matches** in `*.py` for: `vllm`, `Anthropic`, `Gemini`, `vertex`, `google.genai`, or `OpenAI(` as a generation client (aside from Portkey’s OpenAI-compatible surface used by the judge).
- **No** `portkey` references outside judge-related paths (`topk_judge_eval.py`, `reasoning_confidence_bins.py`, `analysis/run_unfiltered_reasoning_baseline.py`).

### 1.3 Concurrency / retry / rate limits

- **Concurrency:** `ThreadPoolExecutor` in `topk_judge_eval.py` (`parallel` parameter) and `reasoning_confidence_bins.py` for judge calls.
- **Retry / rate limiting:** The judge’s `Judge.score()` wraps the API call in `try/except` and returns null scores on failure; there is **no** explicit exponential backoff or rate-limit queue in these files.

### 1.4 Conclusion (Section 1)

- **Where calls are made:** Primarily `topk_judge_eval.Judge.score()` → Portkey chat completions.
- **Portkey as abstraction:** **For judging only**, not for generation.
- **Anthropic / Gemini in code:** **Not wired** by name. Any routing to Claude/Gemini would be **Portkey dashboard / virtual key configuration**, not something this repo encodes.
- **Targeting closed-source APIs with minimal changes:** Reusing the **same pattern** as `Judge` (Portkey client + `chat.completions.create` + `model=<gateway/model>`) is plausible for **new** generation code, but that code **does not exist** in this audit’s file set.

---

## 2. Multi-sample generation (k = 8)

### 2.1 What the repo implements

- **No** first-class “`n` completions in one request” loop for arbitrary models appears in the audited Python sources.
- The **data layout** assumes **per-problem** parallel arrays (`pred`, `score`, `code`, optional `chosen_token_probs_per_path`) with length equal to the number of traces (e.g. 16 in existing paper assets). See `topk_judge_eval.select_traces` (`topk_judge_eval.py`, ~442–475) and `topk_ablation.load_jsonl_raw`.

### 2.2 How k samples would be obtained (outside this repo)

- **Repeated single-sample calls:** Always possible if the provider allows chat completions; you control `temperature` per call (not centralized in repo for generation).
- **True OpenAI-style `n` parameter:** Would require calling `chat.completions.create` with `n=8` **and** provider support through your Portkey route. **Not present** in codebase; would be a **small** addition to a **new** script.
- **Reasoning / full text:** Preserved if you store each assistant message in `code[]` (CoT) and extracted answers in `pred[]`, matching existing JSONL semantics.

### 2.3 Provider-specific statements (codebase)

- **Anthropic via Portkey / Gemini via Portkey:** **No code** instantiates provider-specific SDKs. Feasibility is **entirely** a Portkey configuration + model string question, not something this repository validates.

---

## 3. Saved artifact format vs self-consistency FRS

### 3.1 Self-consistency FRS (this repo)

- **Files:** `analysis/run_self_consistency_frs_proxy.py`, `analysis/run_self_consistency_k_sensitivity.py`
- **Required per problem (for vote-share):** `idx`, `pred` (list of strings), `score` (list, same length as `pred` for correctness flags), aligned lengths. Normalization: `str(pred[i]).strip()` for agreement (`normalize_pred_for_agreement`).
- **Not required** for self-consistency confidence: `chosen_token_probs_per_path` (token-level confidence).

### 3.2 “Full” FRS / confidence-bin pipeline (default paper path)

- **Files:** `reasoning_confidence_bins.py`, `topk_ablation.py`, `build_downstream_parquets.py`
- **Typically requires** `chosen_token_probs_per_path["epoch_0"]` per trace for `compute_trace_confidence` (see `reasoning_confidence_bins.py` ~190, `topk_judge_eval.select_traces` ~446).

### 3.3 Adapter needed?

- **Self-consistency–only appendix:** If you emit JSONL with `idx`, `question`, `gt`, `pred[k]`, `code[k]`, `score[k]` (k=8), you align with what `load_jsonl_self_consistency_rows` / `load_jsonl_bundle` expect. You may omit or stub token probs **for SC analysis**, but then **do not** expect stock `topk_judge_eval` batch or `reasoning_confidence_bins` **without code changes** (see Section 4).
- **Small adapter:** New **folder naming** under `source_pass16_jsonl_by_model*/<ModelFolder>/` and filename prefix `test_<dataset>__...jsonl` so `extract_model_dataset` in `topk_ablation.py` / `build_downstream_parquets.py` can resolve model + benchmark—or adjust `MODEL_MAP` / add slugs in `build_downstream_parquets.MODEL_SLUG` if you use parquet build.

---

## 4. Judging compatibility (GPT-4o-mini)

### 4.1 Judge implementation

- **Class:** `Judge` in `topk_judge_eval.py`
- **Inputs to scoring:** `problem`, `cot`, `gold` — **no assumption** that the evaluated model is open-weight. Closed-source outputs are fine **as strings**.

### 4.2 Stock batch path caveat

- **`evaluate_file` / `judge_one_problem`** call `select_traces(row)`, which **requires** valid token probability lists for every trace; otherwise it returns no traces (`topk_judge_eval.py`, ~442–460). Closed APIs often **do not** return per-token logprobs → **top-k judge batch may skip all problems** unless you change selection logic.

### 4.3 Portkey and judging

- Judging goes through Portkey using the **same** `Portkey` client as above. Generation would be a **separate** API usage (if you add it); only the **pattern** is similar.

### 4.4 Checkpoint format

- Judge checkpoints used downstream: `reasoning_confidence_bins_results/judging_checkpoints/judged_<Model>__<Dataset>.json` with `judged_samples` entries containing `idx`, `trace_idx`, `reasoning_score` (see `build_downstream_parquets.load_judge_map`). **Format is model-agnostic** if you produce the same JSON structure.

---

## 5. Benchmark compatibility (cost / parsing)

- **Code:** All six benchmarks (`GSM8K`, `MATH500`, `SVAMP`, `AQuA`, `CommonsenseQA`, `GPQA`) are treated uniformly for file discovery and plotting; **no** benchmark-specific generation code was found in the audited paths.
- **Operational cost:** Longer prompts + longer CoTs increase cost; **MATH500** and **GPQA** are typically heavier than **GSM8K** / **SVAMP** for token count. **AQuA** / **CommonsenseQA** are multiple-choice–style in the broader literature; your pipeline still needs consistent **string** `pred[]` extraction for self-consistency.
- **Answer extraction / exact-string SC:** Self-consistency uses **exact normalized string equality** on `pred` entries (`strip` only in `run_self_consistency_frs_proxy.py`). There is **no** shared “boxed answer” normalizer in those analysis scripts—**extraction quality is your responsibility** when building `pred[]`.

**Recommended minimal subset for a cheap appendix:** **GSM8K + SVAMP** (shorter, numeric-friendly) first; add **CommonsenseQA** if you want a non-math MCQ contrast. Defer **MATH500** / **GPQA** unless budget allows.

---

## 6. Target models: Claude Sonnet 4.6 & Gemini 3.1 Pro Preview

- **These exact model IDs do not appear** anywhere in the repository’s Python sources.
- **What the code expects:** Any string accepted by `Portkey(...).chat.completions.create(model=...)` for the judge—currently `DEFAULT_JUDGE_MODEL` style gateway prefix + OpenAI-style model name.
- **For your two frontier models:** You must set model strings **in Portkey** (virtual keys / targets) and pass the same strings from **new** generation code. The repo **cannot** confirm that “Claude Sonnet 4.6” or “Gemini 3.1 Pro Preview” are valid on your gateway without your Portkey project settings.

**Nearest supported names:** Not determinable from this repo; check Portkey dashboard / provider routing for your workspace.

---

## 7. Implementation effort (classification)

| Component | Assessment |
|-----------|--------------|
| Closed-source **generation** via Portkey (OpenAI-compatible) | **Requires new script** (small–moderate) — pattern exists in `Judge`, but no generation harness. |
| Saving k=8 in usable JSONL | **Small** — match `pred`/`score`/`code`/`idx` schema; place files under expected tree or extend maps. |
| Self-consistency confidence | **Works already** on compatible JSONL (`analysis/run_self_consistency_frs_proxy.py`, `run_self_consistency_k_sensitivity.py`). |
| Judging with GPT-4o-mini | **Judge class works**; **stock trace selection** in `topk_judge_eval` / default `reasoning_confidence_bins` **may not** without logprobs → **small code change** or **custom judge driver** that selects `(idx, trace_idx)` by self-consistency bins. |
| Ranking vs existing FRS pipeline | **Moderate** if you must match confidence-bin FRS exactly; **small** if appendix is SC-FRS only + manual comparison CSVs. |

---

## 8. Minimal runnable experiment (if you proceed)

1. **Confirm** Portkey routes for two closed models (model strings + billing).
2. **Implement** a thin generator (new script, ~100–200 LOC) using `Portkey` + `chat.completions.create`, loop 8× per problem (or `n=8` if supported), collect `code[]` / raw answers → `pred[]` / `score[]` via your existing grader or answer check.
3. **Write** JSONL lines with `idx`, `question`, `gt`, `pred`, `score`, `code` (length 8).
4. **Judge** traces selected by **self-consistency** (top pool / bin 0) using `Judge.score()` in a **new** small loop writing `judged_<Model>__<Dataset>.json` compatible with `build_downstream_parquets` / `run_self_consistency_frs_proxy` join keys `(idx, trace_idx)`.
5. **Run** `analysis/run_self_consistency_frs_proxy.py` (or k-sensitivity variant) on the new JSONL + checkpoints.

---

## 9. Concrete recommendation

### Is this possible tonight with the **current** setup?

- **Partially:** Judging and SC **analysis** code paths exist; **generation** and **logprob-dependent** tooling do not cover closed APIs out of the box.
- **Fastest path:** New generator + new judge-selection script (or fork `select_traces`), not a large refactor.

### Easiest pair of closed-source models you can run **today**

- **Not decidable from the repo alone.** Use whatever two model strings your Portkey project already routes successfully (verify in dashboard). The codebase does not encode Claude/Gemini IDs.

### Easiest benchmark subset

- **GSM8K + SVAMP** (optionally **CommonsenseQA**).

### Is k=8 practical?

- **Yes**, as 8 parallel list slots; cost = 8× single-sample (or less if `n` works).

### Exact next script if you proceed

1. **Create** `scripts/generate_closed_source_passk.py` (or similar) using Portkey + your model strings — **not present**; this is the main deliverable.
2. **Create** `scripts/judge_sc_selected_traces.py` that uses `topk_judge_eval.Judge` but selects traces from self-consistency ranking (copy binning logic from `analysis/run_self_consistency_frs_proxy.py`).
3. Run **`python analysis/run_self_consistency_frs_proxy.py --repo-root .`** after JSONL + checkpoints exist.

---

## 10. Most likely blocker

1. **No in-repo generation** — you must add it or use an external tool.
2. **Token logprobs** — default `topk_judge_eval` / `reasoning_confidence_bins` trace selection **breaks** without them; **self-consistency** analysis does not need them, but your **judge invocation path** must be adapted.

---

## File index (primary)

| Path | Role |
|------|------|
| `topk_judge_eval.py` | Portkey judge client; `select_traces`; batch eval |
| `reasoning_confidence_bins.py` | Pooled confidence bins + judge (needs token probs) |
| `build_downstream_parquets.py` | JSONL discovery, judge checkpoint paths |
| `analysis/run_self_consistency_frs_proxy.py` | SC confidence → FRS proxy (pred/score only) |
| `analysis/run_self_consistency_k_sensitivity.py` | k=8 vs k=16 robustness (saved JSONL only) |
| `topk_ablation.py` | JSONL schema expectations, `load_jsonl_raw` |

---

*This document is intentionally conservative: anything not shown in code is marked as configuration-dependent or out of scope.*
