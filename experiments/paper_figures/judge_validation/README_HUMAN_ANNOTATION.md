# Human annotation packet (500 stratified samples)

## What’s included

1. **Rubric (one place)** — `llm_judge_rubric_static.md`  
   Mirrors the four-pillar criteria in `backend/app/cot_eval_v2/judge.py` (`Judge.build_prompt`), without per-sample flags/evidence (humans don’t need the automated flag dump).

2. **All samples** — from `validation_500_samples_full.csv`  
   For each row: model, dataset, idx, ground truth, question, **GPT-4o-mini / GPT-4o / Claude** pillar scores, full prompt, full model output.

## Human scores CSV (for correlation)

1. Generate an empty template aligned to the 500 rows:

```bash
python3 analysis/judge_validation/build_human_annotation_template.py
```

→ writes `analysis/judge_validation/human_annotations_template.csv`  
Fill `human_faith`, `human_utili`, `human_coher`, `human_factu` with integers **1–5** (same rubric as LLM judges). Optionally set `annotator_id` / `notes`.

2. Copy to e.g. `human_annotations_rater1.csv` when filled, then compute **human ↔ mini / GPT-4o / Claude** Pearson *r*:

```bash
python3 analysis/judge_validation/compute_human_judge_correlation.py --human analysis/judge_validation/human_annotations_rater1.csv
```

## Generate the document

```bash
cd reasoning-models-eval

# HTML (recommended: open in browser → Print → Save as PDF)
python3 analysis/export_human_annotation_packet.py \
  --out analysis/judge_validation/human_annotation_packet.html

# Smaller test run
python3 analysis/export_human_annotation_packet.py --max-samples 10 --out /tmp/test.html

# Markdown instead (very large file)
python3 analysis/export_human_annotation_packet.py --format md --out analysis/judge_validation/human_annotation_packet.md
```

## Score columns in the CSV / export

| Prefix   | Judge        | Columns (short names in CSV)      |
|----------|--------------|-------------------------------------|
| `mini_*` | GPT-4o-mini  | `mini_faith`, `mini_utili`, `mini_coher`, `mini_factu` |
| `gpt4o_*`| GPT-4o       | `gpt4o_faith`, …                    |
| `claude_*`| Claude Sonnet | `claude_faith`, …                |

## Editing the rubric page

Change **`analysis/judge_validation/llm_judge_rubric_static.md`**, then re-run the export script.  
If you change the **live** LLM prompt in code, sync this file from `judge.py` so annotators see what judges saw.
