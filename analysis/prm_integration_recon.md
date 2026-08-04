# PRM integration recon report (`reasoning-models-eval`)

Prepared for COLM rebuttal planning (comparison vs process reward models). **No secrets:** do not paste `backend/path_config.json` contents into version control—it may contain API keys and HF tokens.

---

## Existing infrastructure

### 1. Trace storage

**Where (raw runs):** Primary artifacts under `evaluation/outputs/<model_alias>/<dataset>/` as `*.jsonl`. Filename pattern from `prepare_data()` in `evaluation/math_eval.py` (e.g. `test_<prompt_type>_<num_test_sample>_seed<seed>_t<temp>_s<start>_e<end>[_<job_id>].jsonl`) — lines **1000–1022**.

**Schema (conceptual):**

- **`idx`:** dataset example id (deduped by `idx` in `evaluation/evaluate.py` **20–22**).
- **`question` / `prompt`:** problem and rendered prompt (see filtered samples for full shape).
- **Generated reasoning:** **`code`** — list of strings, one per draw when `n_sampling` > 1. Comments in `evaluate.py` **181–192** refer to parallel predictions across sampling.
- **After grading:** **`pred`** = extracted answers; **`score`** = bool correctness — parallel lists merged in **`evaluate.py` 211–216**.
- **`trace_idx`:** not a named field; draw index = position in **`code` / `pred` / `score`** (0-based). Filtered subsets typically have **one** trace per `idx`.
- **Per-token logprobs / confidence (optional):** `probability_log`, `chosen_token_probs`, `chosen_token_ids`, etc.; **`answer_confidence`** when computed — **`math_eval.py` 2399–2444**. Optional **`*_prob.jsonl`** sidecar — **2499–2534**.
- **Model / benchmark IDs:** **`model`** is usually implied by directory **`evaluation/outputs/<alias>/`**; benchmark = subdirectory name (`gsm8k`, `math500`, …). **`job_id`** may appear in the filename (**1018–1022**).

**Filtered FRS subsets (`*_filtered_p1_only.jsonl`):**

- Symlinks under repo root → `/work/10757/manasp123/ls6/home-offload-from-home1/reasoning-models-eval/filtered-cot-*` (e.g. `filtered-cot-gsm8k`, `-math500`, `-svamp`, `-aqua`, `-gpqa`, `-commonsense`).
- Discussed alongside judge outputs in **`analysis/generate_frs_coverage_analysis.py`** (header **5–7**): `answer_confidence` in JSONL, fused scores in `*_results.json`.
- Sample records include **`idx`, `question`, `prompt`, `code` (often length-1 list), `pred`, `score`, `answer_confidence`, token metadata, `gt`, …**

**Uncertainty:** The exact script that **builds** `*_filtered_p1_only.jsonl` from raw multi-sample runs is **not clearly named** in-repo (consumers exist; producer may be external/offline).

**Note:** A repo-wide scan for `len(pred) > 3` in `evaluation/outputs` was attempted but **did not complete**; a narrow gsm8k check found **no** such rows—many stored runs appear **k=1**-like.

---

### 2. Judge pipeline

- **`backend/app/cot_eval_v2/judge.py`:** class **`Judge`** — `build_prompt` **59–198**; **`score(problem, cot, gold, flags_summary, evidence)`** returns four integers 1–5 — **200–290** (`chat.completions.create`, temperature 0). **`MockJudge`** **417–457**.
- **`backend/app/cot_eval_v2/evaluator.py`:** **`PillarsEvaluator.analyze(problem, cot_text, gold)`** **212–225** — parses steps, runs flags, integrates judge.
- **Batch scorer:** **`run_filtered_cot_eval.py`** — **`evaluate_sample` 145–202**; **`ThreadPoolExecutor` 362–366**; checkpoints **`results/<stem>_results.json`** **290–301**.

**Caching:** per-`idx` resume from results JSON; no separate LLM response cache.

**PRM as substitute judge:** Tightly coupled to **four pillar keys** via **`fuse_with_judge`** in **`backend/app/cot_eval_v2/scoring.py` **170–214**. A PRM does **not** drop in without an adapter mapping to that dict**, or **a parallel `prm_score` path** bypassing pillar fusion.

---

### 3. Score → ranking

- Per-sample fused score: **`fused_scores["overall"]`** mean of four pillars — **`scoring.py` 211–213**.
- Per-file averages: **`run_filtered_cot_eval.py` **384–389**.
- Cross-judge rankings (figures): **`analysis/generate_judge_rankings_500.py` **5–52**, **77–83** (reads validation JSON blobs).
- **`experiments/ranking_prediction.py`:** consumes **`evaluation/exports/cot_analysis/**/*.json`** with `per_sample` / `overall` — different shape than filtered results — **16–124**.

Reusable pattern: scalar per trace → mean per model → Spearman/rank; **no single generic ingest** for arbitrary score arrays today.

---

### 4. Benchmarks covered

From **`analysis/generate_postfiltered_validation_manifest_confidence.py` `FILTERED_DIRS` **39–46**:** `gsm8k`, `math500`, `svamp`, `aqua`, `gpqa`, `commonsense_qa`.

**Not** in that FRS filtered set: **`mmlu`**, **`mathqa`** as first-class dirs. **`humaneval`** exists as a prompt template in **`evaluation/utils.py` **224–235**.

**PRM realism:**

- Math-trained PRMs plausible for: **`gsm8k`, `math500`, `svamp`, `aqua`**.
- Weak/misleading for typical math PRMs: **`gpqa`, `commonsense_qa`** (domain shift).

---

### 5. GPU / cluster infra

- **SLURM:** **`backend/app/runner.py` **486–521** — `sbatch`, partition/account/wall time from config; **`#SBATCH -n 1`** “single task”; **`CUDA_VISIBLE_DEVICES`** default **0** (**509**).
- **vLLM:** **`evaluation/math_eval.py`** — e.g. **612–614**, **954–957**; subprocess pass-1 option to free VRAM (**951–670** region). **No prominent `tensor_parallel_size` wiring** in the inspected grep slice.
- **Conda paths:** **`backend/app/path_manager.py` **63–69** reads `backend/path_config.json`. README **`README.md` **83–89** mentions `/work/$USER/ls6/` layout.
- **Lonestar / manasp123:** symlinks point at **`/work/10757/manasp123/ls6/...`**. Many **`backend/scripts/job_*.sbatch`** examples exist (historical submits).

---

## PRM integration

### 6. Generator loading vs PRM loader

Orchestration is **`math_eval.py` + CLI** from **`runner.py` `_build_cli_args` **215–236**. No separate model registry abstraction. A PRM fits best as:

- **Offline JSONL post-processor**, or  
- **New script** reusing conda + SLURM patterns from **`runner.py`**.

---

### 7. Trace format vs PRM steps

In-repo heuristic splitting: **`backend/app/cot_eval_v2/parsing.py` `split_steps` **10–69**.

**Observed filtered traces (samples):**

1. **`Phi_4_reasoning` / `math500`:** dense LaTeX prose, mixed tags/blocks—not uniform `Step N:` layout.
2. **`DeepSeek_R1_Distill_Qwen_1.5B` / `math500`:** conversational “So, first…” paragraphs.
3. **`Llama_3.1_8B_Instruct` / `gpqa`:** explicit **`Step 1:`, `Step 2:`** prefixes.
4. **`Llama_3.1_8B_Instruct` / `svamp`:** short prose + `\boxed{}` plus **noise tokens** (`0` repeats)—may confuse token-level PRMs.

Expect **segmentation/injection** to match PRM training (e.g. Qwen-style step separators documented on HF).

---

### 8. Candidate PRMs (external)

| Model | Notes |
|--------|--------|
| **Qwen2.5-Math-PRM-7B** | Strong HF README with **transformers** + step-reward pattern and step separator tokens. **Adaptation:** align steps in traces. [Model card](https://huggingface.co/Qwen/Qwen2.5-Math-PRM-7B) |
| **Skywork-o1-Open-PRM-Qwen-2.5-7B** | Points to **`skywork-o1-prm-inference`** repo + optional vLLM server—more plumbing. [Model card](https://huggingface.co/Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B) |
| **Math-Shepherd family** | Less uniform “single id + one snippet” workflow; tokenizer/delimiter alignment varies. |
| **RLHFlow variants** | Many checkpoints; per-checkpoint inference alignment cost. |

**Lowest-integration-risk default:** **Qwen2.5-Math-PRM-7B** + explicit step-chunking policy + spot checks.

---

### 9. Math-Shepherd pass16 (Lonestar ops + downstream stats)

**Run plan:** **`scripts/job_prm_math_shepherd_full.sbatch`** — single **gpu-a100**, **24h** wall clock; **`~2.3–3 tr/s`** (~4K-seq 7B) ⇒ **full ~36 × ~500 × ~16 traces ≈ ~30h** inference, so **one timeout + identical resubmit** is normal. Second job **appends / skips done `(question_idx, trace_idx)`** on existing **`by_pair/*.jsonl`** (same **`PRM_OUTPUT_DIR` / `--output-dir`**).

**Throughput caveat:** Sanity logs showed **`~26%` of traces with `n_steps > 50`** (line-dense CoT): **`last_step_score`** can be noisy when the tail step is shallow “verification”. For **`aggregate_prm_baseline`-style correlations vs FRS**, compare **multiple trace-level scalars** when possible: **`last_step_score`**, **`min_step_score`**, and (if exported) **`mean` over step scores** (`mean` requires persisting step-level arrays in shards — not written today).

---

## Wrap-up

**What’s already here:** Generation + JSONL schema (`math_eval.py`, `evaluate.py`), optional rich token probs, SLURM/conda launcher (`runner.py`), pillar judge + fusion (`Judge`, `PillarsEvaluator`, `run_filtered_cot_eval.py`), filtered datasets under **`/work/...`** symlinks.

**What’s missing for PRM:** No PRM loader, no step formatting matched to PRM training, no mapping from PRM rewards to rankings/FRS aggregates, judge API is pillar JSON not scalar reward. Confirm multi-draw storage if **`n_sampling > 1`** is required for comparable “pass@16” analysis.

**Time-to-working-baseline (engineering hours, rough):** **~16–32 h** for a minimal scorer + aggregates on **1–2 math** benchmarks **before** cluster queue time. Full **six benchmarks × all models** is much heavier unless scoped.

**Top 3 risks to a ~5-day budget**

1. **Domain mismatch** using math PRMs on **`gpqa` / `commonsense_qa`**.
2. **Step segmentation** vs PRM’s expected token/step boundaries (**artifact-sensitive scores**).
3. **Throughput / GPU / SLURM** (wall-time defaults, OOM tuning on vLLM: **`math_eval.py` **612–714****) eating wall-clock faster than debugging time.

---

*Generated from codebase survey / chat recon. Update paths if symlinks or `path_config.json` layouts change.*
