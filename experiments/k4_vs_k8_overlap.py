#!/usr/bin/env python3
"""
Direct overlap between k=4 and k=8 global top-10% trace sets (independent subsamples from k=16).

k=8 and k=4 draws use separate RNG streams: same base_seed + resample_id structure,
with +1000 offset on the seed for k=4 (per paper prompt).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd

from k_sensitivity_analysis import (
    build_problems_from_raw,
    load_jsonl_raw,
    pool_traces,
    top_frac_set,
    subsample_problems,
)
from topk_ablation import build_file_map


def rng_k8(model: str, benchmark: str, base_seed: int, res_id: int) -> np.random.Generator:
    pair_h = int(hashlib.md5(f"{model}::{benchmark}".encode()).hexdigest()[:8], 16)
    return np.random.default_rng(base_seed + res_id * 10007 + (pair_h % 100000))


def rng_k4(model: str, benchmark: str, base_seed: int, res_id: int) -> np.random.Generator:
    pair_h = int(hashlib.md5(f"{model}::{benchmark}".encode()).hexdigest()[:8], 16)
    return np.random.default_rng(base_seed + res_id * 10007 + 1000 + (pair_h % 100000))


def main() -> None:
    p = argparse.ArgumentParser(description="k=4 vs k=8 top-10% set overlap")
    p.add_argument("--data_dir", type=str, default=".")
    p.add_argument("--num_resamples", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_csv", type=str, default="outputs/diagnostics_k4/k4_vs_k8_overlap.csv")
    args = p.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    out_path = args.output_csv
    if not os.path.isabs(out_path):
        out_path = os.path.join(data_dir, out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    file_map = build_file_map(data_dir)
    top_frac = 0.10
    rows: list[dict] = []

    for (model, benchmark), jsonl_path in sorted(file_map.items()):
        raw = load_jsonl_raw(jsonl_path)
        problems, _ = build_problems_from_raw(raw, expected_k=16)
        if not problems:
            continue

        for res_id in range(args.num_resamples):
            r8 = rng_k8(model, benchmark, args.seed, res_id)
            r4 = rng_k4(model, benchmark, args.seed, res_id)
            sub8 = subsample_problems(problems, 8, r8)
            sub4 = subsample_problems(problems, 4, r4)
            if not sub8 or not sub4:
                continue
            pool8 = pool_traces(sub8)
            pool4 = pool_traces(sub4)
            top8 = top_frac_set(pool8, top_frac)
            top4 = top_frac_set(pool4, top_frac)
            inter = top4 & top8
            uni = top4 | top8
            n4, n8 = len(top4), len(top8)
            recall = len(inter) / n4 if n4 else float("nan")
            jacc = len(inter) / len(uni) if uni else float("nan")
            rows.append(
                {
                    "model": model,
                    "benchmark": benchmark,
                    "resample_id": res_id,
                    "num_k4_top10": n4,
                    "num_k8_top10": n8,
                    "num_intersection": len(inter),
                    "recall_k4_in_k8": recall,
                    "jaccard": jacc,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    gm = float(df["recall_k4_in_k8"].mean()) if len(df) else float("nan")
    print(f"Wrote {out_path} ({len(df)} rows)")
    print(f"Grand mean recall_k4_in_k8: {gm:.6f}")
    if gm > 0.95:
        interp = "k=4 and k=8 select essentially the same high-confidence traces"
    elif gm >= 0.80:
        interp = "moderate agreement, some divergence between k=4 and k=8 filtered sets"
    else:
        interp = "meaningfully different filtered sets despite both being subsets of k=16"
    print(f"Interpretation: {interp}")


if __name__ == "__main__":
    main()
