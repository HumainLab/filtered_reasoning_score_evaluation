# Filtered Reasoning Score (FRS)

Evaluation code for **"Filtered Reasoning Score: Evaluating Reasoning Quality on a Model's Most Confident Traces"** (COLM 2026).

Manas Pathak, Xingyao Chen, Shuozhe Li, Amy Zhang, Liu Leqi — University of Texas at Austin

---

## What FRS is

Accuracy tells you whether a model got the right answer, not whether the reasoning behind it was sound. FRS scores reasoning quality on the traces a model is **most confident** in — the ones a deployed system would actually surface.

The pipeline has four stages:

1. **Generate** `k=16` reasoning traces per problem at `T=0.7`, saving per-token probabilities.
2. **Score confidence** per trace: the mean probability of the tokens in the bottom 10% of the trace's probability distribution (Eq. 2, §3.2). Low-probability tokens concentrate the model's uncertainty.
3. **Filter**: pool every trace in a model–benchmark pair, rank by confidence, keep the top `K=10%`.
4. **Judge** the filtered traces with a rubric-based LLM judge (GPT-4o-mini) on four 1–5 dimensions — faithfulness, utility, coherence, factuality — and average, normalized to 0–100 (Eq. 1 and 3).

A high FRS requires *both* strong reasoning *and* confidence that is well-aligned with that strong reasoning. Models tied on accuracy can differ by 16+ FRS points.

## Quickstart

```bash
git clone https://github.com/HumainLab/filtered_reasoning_score_evaluation.git
cd filtered_reasoning_score_evaluation
pip install -r requirements-frs.txt

export OPENAI_API_KEY=sk-...           # judge model access

# Filter to the top 10% most-confident traces, judge them, and report FRS
python frs_pipeline.py \
  --input  path/to/generation_outputs/ \
  --output-dir runs/my_frs_run \
  --top-frac 0.10
```

`frs_pipeline.py` is the canonical end-to-end entry point: it takes generation outputs, computes per-trace confidence, applies the top-K% filter, runs the judge, and writes FRS.

**Outputs**

| File | Contents |
| --- | --- |
| `runs/my_frs_run/<model>_filtered_p1_only.jsonl` | The retained top-K% traces |
| `runs/my_frs_run/results/<model>_..._results.json` | Per-trace judge scores plus `frs_pct` |
| `runs/my_frs_run/frs_summary.json` | FRS per model |

Add `--dry-run` to see what would be judged without spending anything, and `--limit N` to cap traces while testing.

### Input format

One JSON object per line, one file per model. Required fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `idx` | int | Problem id |
| `question` | str | Problem text |
| `gt` | str | Ground-truth answer (**string** — stringify structured answers) |
| `code` | list[str] | The `k` sampled reasoning traces |
| `pred` | list[str] | Extracted answer per trace |
| `score` | list[bool] | Correctness per trace |
| `chosen_token_probs_per_path` | `{"epoch_0": [[float, ...], ...]}` | Per-token probabilities, one list per trace — **this is what confidence is computed from** |

This is exactly what `evaluation/math_eval.py` emits with `--save_outputs --enable_prob_tracking`. `frs_pipeline.py` also accepts `probability_log_per_path`, `chosen_token_probs`, and `probability_log`.

### Generating traces from scratch

```bash
cd evaluation
pip install -r requirements.txt      # torch, vllm, transformers

python math_eval.py \
  --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --data_names gsm8k --split test \
  --use_vllm --save_outputs --enable_prob_tracking \
  --temperature 0.7 --n_sampling 16 --seed 42 \
  --output_dir outputs/DS-R1-7B
```

Benchmark data for the paper's six benchmarks ships in `evaluation/data/` (GSM8K, MATH500, SVAMP, AQuA, GPQA, CommonsenseQA), plus HumanEval. AQuA, GPQA, and CommonsenseQA load **only** from these local files — `data_loader.py` has no HuggingFace branch for them.

## Configuration

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Judge model access |
| `PORTKEY_API_KEY` | Only if routing judge calls through a Portkey gateway |
| `PORTKEY_MODEL_PREFIX` | Gateway model prefix, e.g. `@your-org-gateway`. Empty by default; leave unset for the direct OpenAI API |
| `FRS_REPO_ROOT` | Overrides the repo root that the PRM scripts resolve data against |

The web UI backend additionally reads `backend/path_config.json` — copy `backend/path_config.template.json` and fill it in. That file holds API keys and is gitignored.

## Repository layout

```
frs_pipeline.py              Canonical end-to-end FRS (filter -> judge -> score)
build_filtered_cot.py        Top-K% confidence filter only (no judging)
run_filtered_cot_eval.py     Batch judge over a filtered-cot directory
humaneval_extract_answer.py  HumanEval function-body extraction

evaluation/                  Generation + grading harness (Qwen2.5-Math fork)
  math_eval.py               Sampling with per-token probability tracking
  parser.py, grader.py       Answer extraction and correctness
  data/                      The paper's 6 benchmarks + HumanEval

backend/app/cot_eval_v2/     The judge itself
  judge.py                   4-pillar rubric prompt (Appendix A) + GPT judge
  scoring.py                 fuse_with_judge -> 0-1 per pillar, plus overall
  evaluator.py               Deterministic pre-judge flag collection

frs_experiments/             Main-paper experiments and appendix analyses
  reasoning_confidence_bins.py  The 5-bin x 50-trace FRS estimator (Sec 3.3)
  topk_ablation.py              Confidence estimator + top-K% machinery
  topk_judge_eval.py            Standalone judge engine
  analysis/                     Robustness and rebuttal analyses
  reasoning_confidence_bins_results/judging_checkpoints/
                                54 judged model-benchmark files (see below)

analysis/                    Figure generation and judge validation
experiments/                 Convergence, ranking stability, correlations
selection-gain/appendix_s/   Frozen selection-gain judge outputs (Appendix U)
filtered-cot-*/results/      Judge outputs per benchmark
docs/                        Extended documentation
```

## Reproducing the paper

The 54 judge checkpoints in `frs_experiments/reasoning_confidence_bins_results/judging_checkpoints/` (one per model × benchmark pair) are the key artifact: **most tables and appendices can be recomputed from them with no API calls and no GPU.** For example, Table 1's bootstrap intervals:

```bash
cd frs_experiments
python bootstrap_table2_std.py     # prints the table plus pasteable LaTeX
```

### Where each result comes from

| Paper element | Script |
| --- | --- |
| **Table 1** — FRS at K=10%, bootstrap CIs | `frs_experiments/bootstrap_table2_std.py` |
| **Figure 2** — pass@1 vs FRS ranking reversals | `analysis/generate_ranking_shift.py` |
| **Figure 3** — reasoning score converges faster | `frs_experiments/export_reasoning_convergence_data.py`, `experiments/convergence_analysis.py` |
| **Figure 4** — quality vs filter threshold K | `frs_experiments/reasoning_confidence_bins.py plot` |
| **Figure 5** — close-accuracy amplification | `frs_experiments/global_pass1_frs_pairwise_analysis.py` |
| **§3.3** — the FRS estimator itself | `frs_experiments/reasoning_confidence_bins.py run` |
| **App. A** — the 4-pillar rubric | `backend/app/cot_eval_v2/judge.py` |
| **App. B** — dimension correlations, leave-one-out | `frs_experiments/dimension_correlation_analysis.py`, `frs_experiments/analysis/run_rebuttal_dim_ablation.py` |
| **App. C** — percentile cutoff / SNR sweep | `frs_experiments/topk_ablation.py` |
| **App. D** — judge validation (GPT-4o, Claude, human) | `run_gpt4o_judge_validation.py`, `run_postfiltered_judge_validation.py`, `analysis/generate_judge_agreement_final.py` |
| **App. F** — ranking stability across conditions | `experiments/ranking_stability_graph.py` |
| **App. G** — FRS across K ∈ {10..50} | `frs_experiments/reasoning_confidence_bins.py` |
| **App. H** — composition of the filtered set | `frs_experiments/analysis/verify_difficulty_distribution_claims.py`, `math500_level_original_vs_selected.py` |
| **App. R** — sampling budget k ∈ {4, 8, 16} | `frs_experiments/k_sensitivity_analysis.py`, `sample_count_ablation.py` |
| **App. T** — alternative confidence estimators | `frs_experiments/analysis/run_confidence_proxy_robustness.py`, `run_self_consistency_frs_proxy.py` |
| **App. U** — selection gain | `frs_experiments/analysis/run_selection_gain_judging.py`; frozen outputs in `selection-gain/appendix_s/` |
| **App. U.5** — held-out Claude Haiku judge | `frs_experiments/analysis/run_selection_gain_haiku_with_progress.py` |
| **App. U.6** — controlling for response style | `frs_experiments/analysis/run_exp_a_style_partial_corr.py` |
| **App. V** — PRM comparison | `analysis/run_prm_math_shepherd.py`, `analysis/aggregate_prm_baseline.py` |
| **App. W** — evaluation cost | `frs_experiments/analysis/run_exp_f_cost_accounting.py` |

See `docs/REPRODUCING.md` for a tiered guide (what runs with no API key, what needs one, what needs a GPU).

`frs_experiments/FRS_experiment_results.md` and `MASTER_EXPERIMENT_RESULTS.md` contain the recorded numeric results; `frs_experiments/reasoning_confidence_bins_methods.md` is the detailed methods spec for the binning estimator.

## Data availability

This repository ships **code, judge outputs, and derived tables**. The raw pass@16 generation traces — 9 models × 6 benchmarks × 16 samples, with per-token probabilities — are hundreds of GB and are **not** included. One sample trace file is provided under `frs_experiments/source_pass16_jsonl_by_model/` so you can inspect the expected format.

Scripts that read raw traces (`topk_ablation.py`, `correctness_conditioned.py`, `sample_count_ablation.py`, `k_sensitivity_*.py`, `build_downstream_parquets.py`) will produce empty results without them. Everything that reads the judge checkpoints works out of the box. To regenerate the raw traces, run the generation command above for each model–benchmark pair; `frs_experiments/analysis/export_pass16_canonical_zip.py` packages them into the canonical layout.

## Known limitations

- **`temp0_confidence_analysis.py` needs T=0 traces** that are not in this release; it is unrunnable until you generate them.
- **Published figure files are not reproduced byte-for-byte.** Several paper figures were assembled outside this repo from hardcoded arrays; `export_reasoning_convergence_data.py` marks its Figure 3 outputs `_RECONSTRUCTED` for this reason.
- **HumanEval FRS is exploratory** and not part of the paper. Phi-4-reasoning's score in `filtered-cot-humaneval/frs_summary.json` (5.53) is an artifact of reasoning/code split failure on its `<think>` format, not a real result. See `docs/FRS_HUMANEVAL_PIPELINE.md`.
- **The rubric is math/QA-oriented.** Applying it to code or agentic traces likely needs prompt changes.
- **Judge non-determinism.** Calls run at `temperature=0`, but exact score reproduction across API versions is not guaranteed.

## Citation

```bibtex
@inproceedings{pathak2026frs,
  title     = {Filtered Reasoning Score: Evaluating Reasoning Quality on a Model's Most Confident Traces},
  author    = {Pathak, Manas and Chen, Xingyao and Li, Shuozhe and Zhang, Amy and Liu, Leqi},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2026}
}
```

## License and acknowledgments

The `evaluation/` harness derives from [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math) (MIT, © 2024 Zhibin Gou); see `evaluation/LICENSE`. The rubric dimensions follow the taxonomy of Lee and Hockenmaier (2025).
