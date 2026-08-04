# Reproducing the paper

This guide separates results by what they cost to reproduce. Start with tier 1 — it covers most of the paper.

## Tier 1 — no API key, no GPU (minutes)

Everything here recomputes from the 54 judge checkpoints shipped in
`experiments/outputs/reasoning_confidence_bins_results/judging_checkpoints/`
(one JSON per model × benchmark pair, ~250 judged traces each).

```bash
cd experiments

# Table 1: FRS at K=10% with bootstrap SDs. Prints the table and LaTeX.
python bootstrap_table2_std.py

# Appendix B: correlations among the four rubric dimensions
python dimension_correlation_analysis.py

# Figure 4 / Appendix G: reasoning quality vs filter threshold K
python reasoning_confidence_bins.py plot --output-dir reasoning_confidence_bins_results

# Figure 5: close-accuracy amplification (FRS gap vs accuracy gap)
python global_pass1_frs_pairwise_analysis.py
```

`bootstrap_table2_std.py` reproduces Table 1 exactly, including the per-benchmark
±values. It is the fastest way to confirm your checkout is intact.

Selection-gain results (Appendix U) recompute from the frozen judge outputs in
`results/selection_gain/appendix_s/` — `selection_gain_predictor_results.csv` already
contains the Pearson r ≈ 0.491 reported in the paper.

## Tier 2 — API key, no GPU (hours, ~$12)

Re-judging traces requires `OPENAI_API_KEY` and raw traces (see tier 3).

```bash
# The full FRS estimator: 5 confidence bins x 50 traces per model-benchmark pair
cd experiments
python reasoning_confidence_bins.py run \
  --data-root /path/to/pass16_traces \
  --output-dir my_run \
  --samples-per-bin 50 --top-pool-frac 0.5 --n-bins 5 --seed 42

# Or the single-command pipeline on one model's outputs
python ../frs/frs_pipeline.py --input /path/to/model.jsonl --output-dir runs/demo
```

Judging is checkpointed by `(idx, trace_idx, bin_label)` and resumable — re-run
the same command after an interruption and it skips completed work.

Cost reference from the paper (Appendix W): the full main evaluation is 13,500
judged traces at roughly $12 with GPT-4o-mini; selection gain adds 5,400 calls
(~$4.75).

## Tier 3 — GPU generation (days)

The raw pass@16 traces are not distributed. Regenerate per model × benchmark:

```bash
cd evaluation
python math_eval.py \
  --model_name_or_path <HF_MODEL> \
  --data_names gsm8k --split test \
  --use_vllm --save_outputs --enable_prob_tracking \
  --temperature 0.7 --n_sampling 16 --seed 42 \
  --output_dir outputs/<model_alias>
```

`--enable_prob_tracking` is **required** — without per-token probabilities there
is no confidence signal and FRS cannot be computed.

The nine models: DeepSeek-R1-Distill-Qwen-1.5B and -7B, LLaMA-3.1-8B-Instruct,
Qwen2.5-7B-Instruct, Qwen2.5-Math-7B, Gemma-7B, Phi-4, Phi-4-Reasoning,
Qwen3-4B (thinking mode). The six benchmarks: GSM8K, MATH500, SVAMP, AQuA,
GPQA, CommonsenseQA.

Per Appendix R, `k=8` reproduces `k=16` rankings at ρ=0.97 — halving the
sampling budget is a reasonable cost saving.

Once generated, arrange traces as `<Model>__<Benchmark>.jsonl` and point the
tier-1/tier-2 scripts at that directory with `--data-root`.
`experiments/robustness/export_pass16_canonical_zip.py` builds this layout.

## Verifying a checkout

```bash
# Should print 54
ls experiments/outputs/reasoning_confidence_bins_results/judging_checkpoints/judged_*.json | wc -l

# Should reproduce Table 1 (DS-R1-7B 88.5, Gemma-7B 26.3)
cd experiments && python bootstrap_table2_std.py | head -15
```

## Notes on exactness

- Judge calls use `temperature=0`, but LLM APIs are not bit-reproducible across
  model versions. Tier-1 results are deterministic because they reuse stored
  scores; tier-2 re-judging may shift values slightly.
- Some published figures were assembled outside this repository from hardcoded
  arrays. `experiments/export_reasoning_convergence_data.py` labels its Figure 3 outputs
  `_RECONSTRUCTED` to make this explicit.
- `experiments/temp0_confidence_analysis.py` needs T=0 traces that are not in this release.
