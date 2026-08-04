"""
Correctness-conditioned confidence analysis (median split; no peeking at labels for threshold).

For each model-benchmark pair:
  - Median-split traces by confidence (data-driven, no correctness used for the split).
  - Report accuracy in high-confidence vs low-confidence halves.
  - Report mean confidence for correct vs incorrect traces.

Usage:
    python correctness_conditioned.py --data_root /path/to/threshold
"""

import argparse
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOW_PROB_CUTOFF = 0.10


def compute_trace_confidence(token_probs: list) -> float:
    if not token_probs or len(token_probs) == 0:
        return float("nan")
    arr = np.asarray(token_probs, dtype=np.float64)
    n_low = max(1, int(len(arr) * LOW_PROB_CUTOFF))
    kth = min(n_low - 1, len(arr) - 1)
    lowest = np.partition(arr, kth)[:n_low]
    return float(np.mean(lowest))


def extract_model_dataset(filepath: str) -> tuple:
    parts = filepath.replace("\\", "/").split("/")
    model = None
    for i, p in enumerate(parts):
        if p.startswith("source_pass16"):
            if i + 1 < len(parts):
                model = parts[i + 1]
            break
    fname = parts[-1]
    m = re.match(r"^([a-z_0-9]+?)__", fname)
    dataset = m.group(1) if m else "unknown"
    dataset_map = {
        "gsm8k": "GSM8K",
        "math500": "MATH500",
        "svamp": "SVAMP",
        "aqua": "AQuA",
        "gpqa": "GPQA",
        "commonsense_qa": "CommonsenseQA",
    }
    dataset = dataset_map.get(dataset, dataset)
    model_map = {
        "DeepSeek_R1_Distill_Qwen_1.5B": "DS-R1-1.5B",
        "DeepSeek_R1_Distill_Qwen_7B": "DS-R1-7B",
        "Llama_3.1_8B_Instruct": "LLaMA-3.1-8B",
        "Qwen2.5_7B_Instruct": "Qwen2.5-7B",
        "Qwen2.5_Math_7B": "Qwen2.5-Math",
        "gemma_7b": "Gemma-7B",
        "phi_4": "Phi-4",
        "Phi_4_reasoning": "Phi-4-Reas.",
        "Qwen3_4B_Thinking_2507": "Qwen3-4B",
    }
    model = model_map.get(model, model)
    return model, dataset


def load_traces(filepath: str) -> list:
    rows = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            scores = obj.get("score", [])
            probs_all = obj.get("chosen_token_probs_per_path", {}).get("epoch_0", [])
            if not isinstance(probs_all, list) or not scores:
                continue
            for i in range(len(scores)):
                probs = probs_all[i] if i < len(probs_all) else []
                c = compute_trace_confidence(probs)
                if np.isnan(c):
                    continue
                rows.append(
                    {
                        "correct": bool(scores[i]),
                        "confidence": c,
                    }
                )
    return rows


def dedupe_files(data_root: str) -> dict:
    pattern = os.path.join(data_root, "source_pass16_jsonl_by_model*", "**", "*.jsonl")
    all_files = glob.glob(pattern, recursive=True)
    file_map = {}
    for fp in all_files:
        model, dataset = extract_model_dataset(fp)
        key = (model, dataset)
        if key not in file_map or "_processed" in fp:
            file_map[key] = fp
    return file_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./correctness_conditioned_results")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    file_map = dedupe_files(args.data_root)
    print(f"Unique (model, dataset) pairs: {len(file_map)}")

    results = []
    for (model, dataset), fp in sorted(file_map.items()):
        traces = load_traces(fp)
        if len(traces) < 4:
            print(f"  SKIP {model} x {dataset}: too few traces ({len(traces)})")
            continue
        df = pd.DataFrame(traces)
        med = df["confidence"].median()
        high = df[df["confidence"] >= med]
        low = df[df["confidence"] < med]
        acc_hi = high["correct"].mean() * 100
        acc_lo = low["correct"].mean() * 100
        mean_c_corr = df.loc[df["correct"], "confidence"].mean()
        mean_c_inc = df.loc[~df["correct"], "confidence"].mean()
        results.append(
            {
                "model": model,
                "dataset": dataset,
                "n_traces": len(df),
                "median_conf": med,
                "acc_high_conf_half": round(acc_hi, 2),
                "acc_low_conf_half": round(acc_lo, 2),
                "gap_pp": round(acc_hi - acc_lo, 2),
                "mean_conf_correct": round(mean_c_corr, 4),
                "mean_conf_incorrect": round(mean_c_inc, 4),
                "mean_conf_gap": round(mean_c_corr - mean_c_inc, 4),
            }
        )

    out_df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, "correctness_conditioned.csv")
    out_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}\n")
    print(out_df.to_string(index=False))

    # Bar chart: gap_pp by dataset (mean across models)
    if not out_df.empty:
        by_ds = out_df.groupby("dataset")["gap_pp"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(9, 4))
        by_ds.plot(kind="barh", ax=ax, color="steelblue")
        ax.axvline(0, color="k", linewidth=0.8)
        ax.set_xlabel("Accuracy gap (high-conf half − low-conf half), pp")
        ax.set_title("Median split: accuracy gap by dataset (mean across models)")
        plt.tight_layout()
        p = os.path.join(args.output_dir, "median_split_gap_by_dataset.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved plot: {p}")


if __name__ == "__main__":
    main()
