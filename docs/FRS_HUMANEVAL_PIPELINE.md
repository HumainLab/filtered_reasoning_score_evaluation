# HumanEval FRS pipeline — code map and runbook

This document lists **every in-repo component** of the Filtered Reasoning Score (FRS) pipeline and how to adapt it for **HumanEval**. Share it with collaborators together with:

- `humaneval_extract_answer.py` — extract executable code from a model response
- `frs/build_filtered_cot.py` — build `*_filtered_p1_only.jsonl` from pass@k generation outputs (missing for math today; provided here)

---

## What FRS measures

**FRS** = mean **GPT judge “overall” reasoning score** on a **confidence-filtered** subset of model traces.

It is **not** pass@1 accuracy. Accuracy uses `evaluation/grader.py` (`humaneval_check`); FRS uses the four-pillar LLM judge.

| Stage | Output | Role |
|-------|--------|------|
| 1. Generation | `math_eval.py` JSONL | pass@k samples + token probs |
| 2. Filtering | `*_filtered_p1_only.jsonl` | high-confidence traces only |
| 3. Judge | `run_filtered_cot_eval.py` → `results/*_results.json` | faith/utility/coherence/factuality + `overall` |
| 4. Aggregate | analysis scripts / CSV | mean `fused_scores.overall` × 100 |

Published table: `experiments/all-result-pdf-data/filtered_reasoning_table.csv`.

---

## Stage 1 — Generation (pass@k + probabilities)

**Entry point:** `evaluation/math_eval.py`

```bash
cd evaluation
python math_eval.py \
  --model_name_or_path <MODEL> \
  --data_names humaneval \
  --split test \
  --output_dir <OUT_DIR> \
  --save_outputs \
  --n_sampling 16 \
  --enable_prob_tracking \
  --use_vllm
```

**Supporting code:**

| File | Purpose |
|------|---------|
| `evaluation/data_loader.py` | Loads HumanEval (HF or `evaluation/data/humaneval/test.jsonl`) |
| `evaluation/utils.py` | Prompt template `"humaneval"` (code-only today) |
| `evaluation/parser.py` | `extract_answer`, `parse_ground_truth` for HumanEval |
| `evaluation/evaluate.py` | Routes to `humaneval_check` when `data_name == "humaneval"` |
| `evaluation/grader.py` | `humaneval_check`, `humaneval_check_process` — unit-test execution |

**HumanEval gap:** Current `humaneval` prompt asks for **code only**. For reasoning-quality FRS you need a **reason-then-code** prompt (new template in `utils.py`, e.g. `humaneval_cot`) so `code[0]` can hold reasoning and `pred[0]` the final code.

---

## Stage 2 — Confidence filtering → `*_filtered_p1_only.jsonl`

**Producer (was missing in-repo):** `frs/build_filtered_cot.py` (repo root)

```bash
python build_filtered_cot.py \
  --input-jsonl evaluation/outputs/<MODEL>_humaneval.jsonl \
  --output-dir results/filtered_cot/humaneval \
  --model-stem Llama_3.1_8B_Instruct \
  --top-frac 0.10 \
  --humaneval-split-reasoning
```

**Confidence estimator** (bottom-10% mean token probability — matches paper / Appendix S):

```python
def trace_bottom10_mean_prob(probs):
    k = max(1, int(floor(0.10 * len(probs))))
    return mean(sort(probs)[:k])
```

Reference implementation: `analysis_outputs/rebuttal_prm/add_frs_top10pct_pool_predictor.py:56–61`.

Input probs: `chosen_token_probs_per_path["epoch_0"][trace_idx]` from generation JSONL.

**Expected JSONL schema** (one row per kept trace; `len(code)==1`):

| Field | HumanEval notes |
|-------|-----------------|
| `idx` | Problem index |
| `question` | Problem text (shown to judge as “problem”) |
| `gt` | **String** for judge (use problem spec / `entry_point`, not the test dict) |
| `code` | `[reasoning_text]` — **only** CoT, no final code block |
| `pred` | `[extracted_python_body]` — use `humaneval_extract_answer.py` |
| `score` | `[bool]` from `humaneval_check` |
| `answer_confidence` | Scalar from bottom-10% mean prob |
| `chosen_token_probs_per_path` | Optional; kept if present in input |

**Existing filtered math dirs** (for reference): `results/filtered_cot/gsm8k`, `results/filtered_cot/math500`, etc. (under results/filtered_cot/).

---

## Stage 3 — GPT judge (FRS scores)

**Entry point:** `run_filtered_cot_eval.py`

```bash
# Set OPENAI_API_KEY or backend/path_config.json openai_api_key
python run_filtered_cot_eval.py \
  --input-dir results/filtered_cot/humaneval \
  --files-parallel 3 \
  --samples-parallel 5
```

**Core judge stack:**

| File | Purpose |
|------|---------|
| `run_filtered_cot_eval.py` | Batch runner; reads `code[0]` as CoT, calls judge |
| `frs/cot_eval_v2/judge.py` | GPT-4o-mini rubric (4 pillars, 1–5), `build_prompt` |
| `frs/cot_eval_v2/evaluator.py` | `PillarsEvaluator.analyze(problem, cot_text, gold)` |
| `frs/cot_eval_v2/scoring.py` | Rule scores + `fuse_with_judge` → `fused_scores` |
| `frs/cot_eval_v2/flag_implementations.py` | Deterministic flags (math-oriented heuristics) |

**Important:** `evaluate_sample` in `run_filtered_cot_eval.py:161–166` sets:

```python
cot_text = code_field[0]   # reasoning only
gold = record.get("gt", ...)  # must be a string for the judge prompt
```

For HumanEval, **do not** pass the raw `gt` dict (tests + `entry_point`). `build_filtered_cot.py --humaneval-split-reasoning` sets `gt` to the problem prompt string.

**Output:** `results/filtered_cot/humaneval/results/<stem>_filtered_p1_only_results.json`

Per-sample fields include `fused_scores.overall` (0–1), `judge_scores`, `original_correct`.

---

## Stage 4 — Aggregate FRS number

**Quick aggregate** (per model file):

```python
import json
from pathlib import Path

p = Path("results/filtered_cot/humaneval/results/MODEL_filtered_p1_only_results.json")
data = json.loads(p.read_text())
scores = [r["fused_scores"]["overall"] for r in data["results"] if r.get("fused_scores")]
frs_pct = 100 * sum(scores) / len(scores)
print(f"FRS = {frs_pct:.1f}%  (n={len(scores)})")
```

**Analysis scripts** (math benchmarks; add `humaneval` to `FILTERED_DIRS`):

| Script | Purpose |
|--------|---------|
| `experiments/paper_figures/generate_frs_coverage_analysis.py` | FRS vs confidence threshold sweeps |
| `experiments/paper_figures/generate_postfiltered_validation_manifest.py` | Stratified judge validation samples |
| `experiments/paper_figures/aggregate_prm_baseline.py` | Join filtered JSONL + judge results |

**Published FRS table:** `experiments/all-result-pdf-data/filtered_reasoning_table.csv` — add a HumanEval column after recomputing.

---

## End-to-end HumanEval checklist

1. **Generate** with `humaneval_cot` prompt + `--enable_prob_tracking` + `--n_sampling 16`.
2. **Grade correctness** — automatic in `math_eval.py` via `evaluate()` → `humaneval_check`.
3. **Extract code** — `humaneval_extract_answer.py` on each trace.
4. **Split reasoning vs code** — `build_filtered_cot.py --humaneval-split-reasoning`.
5. **Filter** — `frs/build_filtered_cot.py` → `results/filtered_cot/humaneval/<model>_filtered_p1_only.jsonl`.
6. **Judge** — `frs/run_filtered_cot_eval.py --input-dir results/filtered_cot/humaneval`.
7. **Aggregate** — mean `fused_scores.overall` × 100.

---

## Related (not required for basic FRS)

| Path | Purpose |
|------|---------|
| `results/selection_gain/appendix_s/` | Selection-gain experiment CSVs (50 questions × 2 policies) |
| `analysis_outputs/rebuttal_prm/` | PRM vs FRS correlation analyses |
| `run_postfiltered_judge_validation.py` | Human validation of judge on filtered samples |
| `REPO_INTERFACE_REPORT.md` | Full repo interface report |

---

## Known limitations for HumanEval

1. **Judge rubric is math-oriented** (`judge.py` — “mathematical and logical reasoning”). Factuality checks assume numeric grounding; expect weaker signal on code tasks.
2. **Rule-based flags** in `PillarsEvaluator` target math steps, not Python AST structure.
3. **No `humaneval` in `filtered_reasoning_table.csv`** yet — column must be added after Stage 4.
4. **Filter builder was offline** for math; use `frs/build_filtered_cot.py` and verify subset sizes against your pass@k pool.
