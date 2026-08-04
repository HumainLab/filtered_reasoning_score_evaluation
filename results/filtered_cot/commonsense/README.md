# Filtered CoT – Commonsense QA

## Layout

- **`*.jsonl`** – Extracted data files (one per model), ready for CoT evaluation.
  - Each line is a JSON record: `question`, `code` (chain-of-thought), `gt`, `pred`, `score`, etc.
- **`results/`** – CoT evaluation outputs (per-model `*_results.json`, `global_summary.json`, logs).
- **`archives/`** – Original zips (outer `drive-download-*.zip` and inner `*_filtered_p1_only.jsonl.zip`).

## Models (9)

| File | Description |
|------|-------------|
| DeepSeek_R1_Distill_Qwen_1.5B_filtered_p1_only.jsonl | DeepSeek R1 Distill Qwen 1.5B |
| DeepSeek_R1_Distill_Qwen_7B_filtered_p1_only.jsonl | DeepSeek R1 Distill Qwen 7B |
| gemma_7b_filtered_p1_only.jsonl | Gemma 7B |
| Llama_3.1_8B_Instruct_filtered_p1_only.jsonl | Llama 3.1 8B Instruct |
| phi_4_filtered_p1_only.jsonl | Phi 4 |
| Phi_4_reasoning_filtered_p1_only.jsonl | Phi 4 Reasoning |
| Qwen2.5_7B_Instruct_filtered_p1_only.jsonl | Qwen2.5 7B Instruct |
| Qwen2.5_Math_7B_filtered_p1_only.jsonl | Qwen2.5 Math 7B |
| Qwen3_4B_Thinking_2507_filtered_p1_only.jsonl | Qwen3 4B Thinking 2507 |

## Run CoT analysis

From `reasoning-models-eval/`:

```bash
python run_cot_analysis_aqua_svamp.py --datasets commonsense --no-extract --verbose
```
