#!/usr/bin/env python3
"""
Compile all 500 stratified validation samples into a single spreadsheet.
Columns: model, dataset, idx, question, full_prompt, model_output, ground_truth,
         mini_{faith,util,coher,fact}, gpt4o_{faith,util,coher,fact}, claude_{faith,util,coher,fact}
"""

import json
import csv
import glob
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = BASE / "analysis"
EXPORTS_DIR = BASE / "evaluation" / "exports" / "cot_analysis"
OUTPUTS_DIR = BASE / "evaluation" / "outputs"

MANIFEST = ANALYSIS_DIR / "judge_validation" / "validation_manifest.json"
GPT4O_RESULTS = ANALYSIS_DIR / "judge_validation" / "judge_validation_all_20260308_150844.json"
CLAUDE_RESULTS = ANALYSIS_DIR / "judge_validation" / "judge_validation_all_claude-sonnet-4-5_20260308_152120.json"
OUT_CSV = ANALYSIS_DIR / "judge_validation" / "validation_500_samples_full.csv"

PILLARS = ["faithfulness", "utility", "coherence", "factuality"]

MODELS_SHORT = {
    "DeepSeek-R1-Distill-Qwen-1.5B": "DS-R1-1.5B",
    "DeepSeek-R1-Distill-Qwen-7B": "DS-R1-7B",
    "Llama-3.1-8B-Instruct": "LLaMA-3.1-8B",
    "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "Qwen2.5-Math-7B": "Qwen2.5-Math-7B",
    "gemma-7b": "Gemma-7B",
    "phi-4": "Phi-4",
    "Phi-4-reasoning": "Phi-4-Reasoning",
    "Qwen3-4B-Thinking-2507": "Qwen3-4B",
}

with open(MANIFEST) as f:
    manifest = json.load(f)
with open(GPT4O_RESULTS) as f:
    gpt4o_data = json.load(f)
with open(CLAUDE_RESULTS) as f:
    claude_data = json.load(f)

# Build lookup: combo_key -> {idx -> gpt4o_scores}
gpt4o_lookup = {}
for combo_key, combo in gpt4o_data["per_combo"].items():
    gpt4o_lookup[combo_key] = {p["idx"]: p["gpt4o"] for p in combo["sample_pairs"]}

claude_lookup = {}
for combo_key, combo in claude_data["per_combo"].items():
    claude_lookup[combo_key] = {p["idx"]: p["gpt4o"] for p in combo["sample_pairs"]}


def find_jsonl_for_job(model_dir, dataset, job_id):
    """Find the output JSONL file matching a job_id."""
    pattern = str(OUTPUTS_DIR / model_dir / dataset / f"*{job_id}*.jsonl")
    matches = glob.glob(pattern)
    # Filter out _prob.jsonl, _processed.jsonl, _metrics.json
    matches = [m for m in matches if not any(m.endswith(s) for s in
               ["_prob.jsonl", "_processed.jsonl", "_metrics.json"])]
    return matches[0] if matches else None


def load_prompts_from_jsonl(jsonl_path, idx_set):
    """Load prompts for specific indices from a JSONL file."""
    prompts = {}
    with open(jsonl_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("idx") in idx_set:
                    prompts[rec["idx"]] = rec.get("prompt", "")
            except json.JSONDecodeError:
                continue
    return prompts


rows = []
combos_processed = 0

for combo_key, combo_info in manifest["combos"].items():
    model_dir = combo_info["model_dir"]
    dataset = combo_info["dataset"]
    indices = combo_info["indices"]
    result_path = combo_info["result_path"]
    short_name = MODELS_SHORT.get(model_dir, model_dir)

    if not os.path.exists(result_path):
        print(f"  SKIP {combo_key}: result file missing")
        continue

    # Load the cot_analysis file (has question, model_output, judge_scores)
    with open(result_path) as f:
        cot_data = json.load(f)

    job_id = cot_data.get("job_id", "")
    idx_set = set(indices)

    # Build lookup from cot_analysis per_sample
    cot_lookup = {}
    for s in cot_data.get("per_sample", []):
        if s["idx"] in idx_set:
            cot_lookup[s["idx"]] = s

    # Find the JSONL to get prompts
    jsonl_path = find_jsonl_for_job(model_dir, dataset, job_id)
    prompts = {}
    if jsonl_path:
        prompts = load_prompts_from_jsonl(jsonl_path, idx_set)

    # Build rows
    for idx in indices:
        cot_sample = cot_lookup.get(idx)
        if cot_sample is None:
            continue

        mini_scores = cot_sample.get("judge_scores", {})
        gpt4o_scores = gpt4o_lookup.get(combo_key, {}).get(idx, {})
        claude_scores = claude_lookup.get(combo_key, {}).get(idx, {})

        row = {
            "model": short_name,
            "model_dir": model_dir,
            "dataset": dataset,
            "idx": idx,
            "question": cot_sample.get("question", ""),
            "ground_truth": cot_sample.get("ground_truth", ""),
            "full_prompt": prompts.get(idx, ""),
            "model_output": cot_sample.get("model_output", ""),
        }

        for p in PILLARS:
            short_p = p[:5]
            row[f"mini_{short_p}"] = mini_scores.get(p, "")
            row[f"gpt4o_{short_p}"] = gpt4o_scores.get(p, "")
            row[f"claude_{short_p}"] = claude_scores.get(p, "")

        rows.append(row)

    combos_processed += 1
    print(f"  [{combos_processed:>2}/{len(manifest['combos'])}] {short_name:20s} | {dataset:16s} -> {len([i for i in indices if i in cot_lookup])} samples")

# Sort by model, dataset, idx
rows.sort(key=lambda r: (r["model"], r["dataset"], r["idx"]))

# Write CSV
fieldnames = [
    "model", "model_dir", "dataset", "idx", "question", "ground_truth",
    "full_prompt", "model_output",
    "mini_faith", "mini_utili", "mini_coher", "mini_factu",
    "gpt4o_faith", "gpt4o_utili", "gpt4o_coher", "gpt4o_factu",
    "claude_faith", "claude_utili", "claude_coher", "claude_factu",
]

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nDone! {len(rows)} rows written to {OUT_CSV}")

# Also count how many have all 3 judges
full = sum(1 for r in rows if r["gpt4o_faith"] != "" and r["claude_faith"] != "")
print(f"Rows with all 3 judges: {full}")
missing_prompt = sum(1 for r in rows if r["full_prompt"] == "")
print(f"Rows missing prompt: {missing_prompt}")
