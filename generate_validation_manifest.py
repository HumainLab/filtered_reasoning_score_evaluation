#!/usr/bin/env python3
"""
Generate a stratified validation manifest for GPT-4o judge validation
across all model × dataset combos.

Stratifies by:
  - Correctness: correct vs incorrect answers
  - Reasoning quality: low (1-2), mid (3), high (4-5) avg judge score

Usage:
  python generate_validation_manifest.py --total-samples 500
  python generate_validation_manifest.py --samples-per-combo 9
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
COT_ANALYSIS_DIR = SCRIPT_DIR / "evaluation" / "exports" / "cot_analysis"
OUTPUT_DIR = SCRIPT_DIR / "analysis" / "judge_validation"

MODEL_DIRS = [
    "DeepSeek-R1-Distill-Qwen-1.5B",
    "DeepSeek-R1-Distill-Qwen-7B",
    "Llama-3.1-8B-Instruct",
    "Qwen2.5-7B-Instruct",
    "Qwen2.5-Math-7B",
    "gemma-7b",
    "phi-4",
    "Phi-4-reasoning",
    "Qwen3-4B-Thinking-2507",
]

DATASETS = ["gsm8k", "math500", "svamp", "aqua", "gpqa", "commonsense_qa"]

PILLARS = ["faithfulness", "utility", "coherence", "factuality"]


def find_latest_result(model_dir: str, dataset: str):
    d = COT_ANALYSIS_DIR / model_dir / dataset
    if not d.exists():
        return None
    files = sorted(d.glob("cot_analysis_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def classify_reasoning(judge_scores: dict) -> str:
    avg = sum(judge_scores[p] for p in PILLARS) / len(PILLARS)
    if avg <= 2.5:
        return "low"
    elif avg <= 3.5:
        return "mid"
    else:
        return "high"


def load_and_stratify(result_path: Path):
    """Load samples and bucket them into 6 strata: (correct/incorrect) × (low/mid/high)."""
    with open(result_path) as f:
        data = json.load(f)

    per_sample = data.get("per_sample", [])

    strata = defaultdict(list)
    for s in per_sample:
        js = s.get("judge_scores")
        if not js or not all(js.get(p) is not None for p in PILLARS):
            continue

        correct = s.get("evidence", {}).get("final_correct", False)
        correctness = "correct" if correct else "incorrect"
        quality = classify_reasoning(js)
        key = f"{correctness}_{quality}"
        strata[key].append(s["idx"])

    return strata


def stratified_sample(strata: dict, n: int, rng: random.Random) -> list:
    """
    Sample n items, drawing proportionally from each stratum.
    Guarantees at least 1 from each non-empty stratum, then fills proportionally.
    """
    non_empty = {k: v for k, v in strata.items() if v}
    if not non_empty:
        return []

    chosen = []
    remaining_budget = n

    # Phase 1: guarantee at least 1 from each non-empty stratum
    for key, indices in non_empty.items():
        pick = rng.sample(indices, min(1, len(indices)))
        chosen.extend(pick)
        remaining_budget -= len(pick)

    if remaining_budget <= 0:
        return chosen[:n]

    # Phase 2: distribute remaining budget proportionally
    total_available = sum(len(v) for v in non_empty.values())
    already_chosen = set(chosen)

    for key, indices in non_empty.items():
        available = [i for i in indices if i not in already_chosen]
        if not available:
            continue
        share = max(0, round(remaining_budget * len(indices) / total_available))
        pick = rng.sample(available, min(share, len(available)))
        chosen.extend(pick)

    # Phase 3: if still under budget, fill from any remaining
    if len(chosen) < n:
        all_remaining = []
        already_chosen = set(chosen)
        for indices in non_empty.values():
            all_remaining.extend(i for i in indices if i not in already_chosen)
        deficit = n - len(chosen)
        if all_remaining:
            chosen.extend(rng.sample(all_remaining, min(deficit, len(all_remaining))))

    return chosen[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-combo", type=int, default=None)
    parser.add_argument("--total-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.samples_per_combo is None and args.total_samples is None:
        args.total_samples = 500
    if args.samples_per_combo is not None and args.total_samples is not None:
        raise SystemExit("Use exactly one of --samples-per-combo or --total-samples")

    rng = random.Random(args.seed)

    combos = []
    for model_dir in MODEL_DIRS:
        for dataset in DATASETS:
            result_path = find_latest_result(model_dir, dataset)
            if result_path is None:
                continue
            strata = load_and_stratify(result_path)
            total_valid = sum(len(v) for v in strata.values())
            if total_valid == 0:
                continue
            combos.append((model_dir, dataset, strata, str(result_path)))

    n_combos = len(combos)
    if args.total_samples is not None:
        per_combo = max(1, args.total_samples // n_combos)
        remainder = args.total_samples - per_combo * n_combos
    else:
        per_combo = args.samples_per_combo
        remainder = 0

    manifest = {
        "seed": args.seed,
        "stratified": True,
        "strata": ["correct_low", "correct_mid", "correct_high",
                    "incorrect_low", "incorrect_mid", "incorrect_high"],
        "samples_per_combo": per_combo,
        "total_samples": 0,
        "n_combos": n_combos,
        "combos": {},
    }

    strata_global_counts = defaultdict(int)

    for i, (model_dir, dataset, strata, result_path) in enumerate(combos):
        n = per_combo + (1 if i < remainder else 0)
        chosen = stratified_sample(strata, n, rng)

        chosen_set = set(chosen)
        strata_breakdown = {}
        for skey, indices in strata.items():
            count = sum(1 for idx in indices if idx in chosen_set)
            if count > 0:
                strata_breakdown[skey] = count
                strata_global_counts[skey] += count

        key = f"{model_dir}::{dataset}"
        manifest["combos"][key] = {
            "model_dir": model_dir,
            "dataset": dataset,
            "result_path": result_path,
            "indices": sorted(chosen),
            "n": len(chosen),
            "strata_breakdown": strata_breakdown,
        }
        manifest["total_samples"] += len(chosen)

    manifest["global_strata_counts"] = dict(strata_global_counts)

    out_path = Path(args.output) if args.output else OUTPUT_DIR / "validation_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest written to {out_path}")
    print(f"  Combos: {n_combos}")
    print(f"  Total samples: {manifest['total_samples']}")
    print(f"  Per combo: ~{per_combo}")
    print(f"\n  Global strata distribution:")
    for skey in sorted(strata_global_counts.keys()):
        cnt = strata_global_counts[skey]
        pct = cnt / manifest["total_samples"] * 100
        print(f"    {skey:20s}: {cnt:>4} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
