#!/usr/bin/env python3
"""
Export CSVs for the "reasoning converges faster" / Figure 3 style analysis.

IMPORTANT (provenance):
  No script in this repository generates the published ``images/converges.png`` referenced
  in ``COLM 2026 Conference Template (3).tex`` (Appendix convergence). That asset is not
  committed here.

  This script exports:
    (1) Raw-ish per-problem tables from the *nearest* sources in-repo.
    (2) A RECONSTRUCTED bootstrap SD vs N curve using a documented procedure that
        matches the *intent* of the paper text (subsampling evaluation problems, comparing
        variance of pass@1 vs mean judge reasoning), but it is **not** guaranteed to match
        the exact numbers in the paper (different temperature / coverage / N grid).

  Primary inputs:
    - ``topk_ablation.build_file_map`` → pass@16 JSONL (trace 0 = pass@1 proxy).
    - ``reasoning_confidence_bins_results/judging_checkpoints/judged_*.json`` → sparse
      judge ``reasoning_score`` (250 traces / pair, not full problem coverage).

Usage:
  python export_reasoning_convergence_data.py \\
      --data-root . \\
      --out-dir analysis_exports/reasoning_converges_faster_csv \\
      --n-boot 10000 \\
      --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from topk_ablation import build_file_map, extract_model_dataset, load_jsonl_raw

JUDGE_GLOB = "outputs/reasoning_confidence_bins_results/judging_checkpoints/judged_*.json"

# Paper benchmarks (order for appendix-style tables)
BENCHMARKS = ["GSM8K", "MATH500", "SVAMP", "AQuA", "CommonsenseQA", "GPQA"]


def parse_judge_name(path: str) -> Optional[Tuple[str, str]]:
    base = os.path.basename(path)
    m = re.match(r"^judged_(.+)__(.+)\.json$", base)
    if not m:
        return None
    return m.group(1), m.group(2)


def load_judge_reasoning_by_idx(path: str) -> Dict[int, List[float]]:
    """idx -> list of reasoning_score values (0-1) for judged traces."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[int, List[float]] = {}
    for s in data.get("judged_samples", []):
        if s.get("judge_ok") is False:
            continue
        rs = s.get("reasoning_score")
        if rs is None:
            continue
        v = float(rs)
        if not np.isfinite(v):
            continue
        idx = int(s["idx"])
        out.setdefault(idx, []).append(v)
    return out


def pass1_trace0_from_records(records: List[Dict[str, Any]]) -> List[Tuple[int, float]]:
    """Return (idx, pass1 as 0/1) using first trace only."""
    rows: List[Tuple[int, float]] = []
    for row in records:
        scores = row.get("scores", [])
        if not scores:
            continue
        idx = row.get("idx")
        if idx is None:
            continue
        rows.append((int(idx), 1.0 if bool(scores[0]) else 0.0))
    rows.sort(key=lambda t: t[0])
    return rows


def stable_seed(parts: Sequence[str], base: int) -> int:
    h = hashlib.sha256("|".join([str(base)] + [str(p) for p in parts]).encode()).digest()
    return int.from_bytes(h[:8], "big") % (2**31 - 1) or 1


def bootstrap_std_of_means(
    x: np.ndarray,
    n_sub: int,
    n_boot: int,
    rng: np.random.Generator,
) -> float:
    """SD across bootstrap sample means (with replacement), x length P."""
    p = len(x)
    if p == 0 or n_sub <= 0:
        return float("nan")
    # Chunk to avoid huge (n_boot × n_sub) index arrays (e.g. 1e7 ints per call).
    means = np.empty(n_boot, dtype=np.float64)
    chunk = 256
    for start in range(0, n_boot, chunk):
        end = min(start + chunk, n_boot)
        idx = rng.integers(0, p, size=(end - start, n_sub))
        means[start:end] = x[idx].mean(axis=1)
    return float(np.std(means, ddof=0))


def default_n_grid(max_p: int) -> List[int]:
    """Evaluation set sizes to try (paper highlights small N e.g. 25)."""
    cand = [10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000]
    out = [n for n in cand if n <= max_p]
    if max_p > 0 and max_p not in out:
        out.append(max_p)
    return sorted(set(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default=".")
    ap.add_argument(
        "--out-dir",
        type=str,
        default="outputs/analysis_exports/reasoning_converges_faster_csv",
    )
    ap.add_argument(
        "--n-boot",
        type=int,
        default=2000,
        help="Bootstrap replicates per (combo, N). Use 10000 for paper-grade precision (slower).",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = os.path.abspath(args.data_root)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    file_map = build_file_map(root)
    judge_dir = os.path.join(root, "outputs/reasoning_confidence_bins_results", "judging_checkpoints")
    judge_files = []
    if os.path.isdir(judge_dir):
        for fn in sorted(os.listdir(judge_dir)):
            if fn.startswith("judged_") and fn.endswith(".json"):
                judge_files.append(os.path.join(judge_dir, fn))

    # --- per_problem_accuracy full (all idx in JSONL) ---
    acc_rows: List[Dict[str, Any]] = []
    for (model, bench), fp in sorted(file_map.items()):
        if bench not in BENCHMARKS:
            continue
        rec = load_jsonl_raw(fp)
        for idx, a in pass1_trace0_from_records(rec):
            acc_rows.append(
                {
                    "model": model,
                    "benchmark": bench,
                    "problem_idx": idx,
                    "pass1_trace0": a,
                    "source_jsonl": os.path.relpath(fp, root),
                }
            )

    df_acc = pd.DataFrame(acc_rows)
    p_acc = os.path.join(out_dir, "per_problem_accuracy_pass1_trace0_from_pass16.csv")
    df_acc.to_csv(p_acc, index=False)

    # --- judge reasoning: mean per idx (sparse) ---
    reas_rows: List[Dict[str, Any]] = []
    n_judge_pairs = 0
    for jpath in judge_files:
        parsed = parse_judge_name(jpath)
        if not parsed:
            continue
        model, bench = parsed
        if bench not in BENCHMARKS:
            continue
        by_idx = load_judge_reasoning_by_idx(jpath)
        n_judge_pairs += 1
        for idx, vals in by_idx.items():
            reas_rows.append(
                {
                    "model": model,
                    "benchmark": bench,
                    "problem_idx": idx,
                    "n_judged_traces_this_problem": len(vals),
                    "mean_reasoning_score_0_1": float(np.mean(vals)),
                    "source_judge_json": os.path.relpath(jpath, root),
                }
            )

    df_reas = pd.DataFrame(reas_rows)
    p_reas = os.path.join(out_dir, "per_problem_reasoning_judge_sparse_aggregated.csv")
    df_reas.to_csv(p_reas, index=False)

    # --- intersection (fair comparison for reconstructed bootstrap) ---
    merged = pd.merge(
        df_acc,
        df_reas[["model", "benchmark", "problem_idx", "mean_reasoning_score_0_1", "n_judged_traces_this_problem"]],
        on=["model", "benchmark", "problem_idx"],
        how="inner",
    )
    p_merge = os.path.join(out_dir, "per_problem_intersection_accuracy_and_reasoning.csv")
    merged.to_csv(p_merge, index=False)

    # --- bootstrap SD per combo × N × metric ---
    std_rows: List[Dict[str, Any]] = []
    n_boot = args.n_boot
    for (model, bench), g in merged.groupby(["model", "benchmark"]):
        acc_v = g["pass1_trace0"].values.astype(np.float64)
        reas_v = g["mean_reasoning_score_0_1"].values.astype(np.float64)
        p_int = len(g)
        rng_acc = np.random.default_rng(stable_seed([model, bench, "acc"], args.seed))
        rng_reas = np.random.default_rng(stable_seed([model, bench, "reas"], args.seed))
        for n in default_n_grid(p_int):
            s_acc = bootstrap_std_of_means(acc_v, n, n_boot, rng_acc)
            s_reas = bootstrap_std_of_means(reas_v, n, n_boot, rng_reas)
            std_rows.append(
                {
                    "model": model,
                    "benchmark": bench,
                    "metric_type": "pass1_trace0_accuracy",
                    "sample_size_N": n,
                    "bootstrap_std_of_mean_on_0_100_scale": s_acc * 100.0,
                    "n_problems_in_intersection": p_int,
                    "n_bootstrap": n_boot,
                }
            )
            std_rows.append(
                {
                    "model": model,
                    "benchmark": bench,
                    "metric_type": "mean_reasoning_judge_sparse",
                    "sample_size_N": n,
                    "bootstrap_std_of_mean_on_0_100_scale": s_reas * 100.0,
                    "n_problems_in_intersection": p_int,
                    "n_bootstrap": n_boot,
                }
            )

    df_std = pd.DataFrame(std_rows)
    p_std = os.path.join(out_dir, "figure3_stddev_by_combo_RECONSTRUCTED.csv")
    df_std.to_csv(p_std, index=False)

    # --- median curves (54 combos when all six benchmarks × nine models present) ---
    med_global: List[Dict[str, Any]] = []
    for n in sorted(df_std["sample_size_N"].unique()):
        sub = df_std[df_std["sample_size_N"] == n]
        for met in sub["metric_type"].unique():
            v = sub.loc[sub["metric_type"] == met, "bootstrap_std_of_mean_on_0_100_scale"]
            med_global.append(
                {
                    "sample_size_N": int(n),
                    "metric_type": met,
                    "median_std_across_model_benchmark_combos": float(np.median(v)),
                    "n_combos": int(len(v)),
                }
            )

    df_med = pd.DataFrame(med_global).sort_values(["sample_size_N", "metric_type"])
    p_med = os.path.join(out_dir, "figure3_median_curve_RECONSTRUCTED.csv")
    df_med.to_csv(p_med, index=False)

    # --- Appendix D style: per benchmark, median across models ---
    appd: List[Dict[str, Any]] = []
    for bench in BENCHMARKS:
        for n in sorted(df_std["sample_size_N"].unique()):
            sub = df_std[(df_std["benchmark"] == bench) & (df_std["sample_size_N"] == n)]
            for met in sub["metric_type"].unique():
                vv = sub.loc[sub["metric_type"] == met, "bootstrap_std_of_mean_on_0_100_scale"]
                appd.append(
                    {
                        "benchmark": bench,
                        "sample_size_N": int(n),
                        "metric_type": met,
                        "median_std_across_models": float(np.median(vv)) if len(vv) else float("nan"),
                        "n_models": int(len(vv)),
                    }
                )

    df_app = pd.DataFrame(appd).sort_values(["benchmark", "sample_size_N", "metric_type"])
    p_app = os.path.join(out_dir, "appendixD_per_benchmark_median_RECONSTRUCTED.csv")
    df_app.to_csv(p_app, index=False)

    # --- metadata ---
    meta = {
        "repo_note": (
            "No Python entrypoint in this repository was found that generates the published "
            "Figure 3 / Appendix convergence plot (see search in export_reasoning_convergence_data.py "
            "header). The LaTeX file COLM 2026 Conference Template (3).tex references images/converges.png "
            "which is not present in the threshold/ tree."
        ),
        "reconstruction_method": {
            "accuracy": (
                "pass1_trace0: first trace correctness from pass@16 JSONL (T=0.7 runs), "
                "same file_map as topk_ablation.build_file_map."
            ),
            "reasoning": (
                "mean_reasoning_score_0_1: for each problem idx, mean of all judged reasoning_score "
                "values appearing in judged_*.json (sparse: ~250 judged traces per model-benchmark, "
                "not all problems covered)."
            ),
            "intersection": (
                "Only problem_idx present in BOTH JSONL and judge export are used so accuracy and "
                "reasoning are defined on the same finite set."
            ),
            "bootstrap": (
                f"For each (model, benchmark) with P intersection problems, for each N in grid, "
                f"draw {n_boot} bootstrap samples of size N with replacement from the P rows, "
                "compute sample mean each time, take SD of those means. Scale ×100."
            ),
        },
        "inputs": {
            "pass16_jsonl": "Discovered via topk_ablation.build_file_map(data_root)",
            "judge_checkpoints": os.path.join(JUDGE_GLOB),
        },
        "outputs": {
            "per_problem_accuracy_pass1_trace0_from_pass16.csv": "Raw per-problem pass@1 proxy (all problems in JSONL).",
            "per_problem_reasoning_judge_sparse_aggregated.csv": "Sparse judge coverage; mean reasoning per idx.",
            "per_problem_intersection_accuracy_and_reasoning.csv": "Inner join used for RECONSTRUCTED bootstrap.",
            "figure3_stddev_by_combo_RECONSTRUCTED.csv": "Per model-benchmark-metric-N bootstrap SD.",
            "figure3_median_curve_RECONSTRUCTED.csv": "Median across combos at each N (global).",
            "appendixD_per_benchmark_median_RECONSTRUCTED.csv": "Median across models within each benchmark.",
        },
        "row_counts": {
            "per_problem_accuracy": len(df_acc),
            "per_problem_reasoning_sparse": len(df_reas),
            "intersection": len(merged),
            "stddev_by_combo": len(df_std),
            "median_curve": len(df_med),
            "appendixD": len(df_app),
        },
        "counts_summary": {
            "n_model_benchmark_jsonl_pairs": len({(m, b) for m, b, _ in file_map.items() if b in BENCHMARKS}),
            "n_judge_json_files_benchmark": n_judge_pairs,
            "n_unique_model_benchmark_in_intersection": merged.groupby(["model", "benchmark"]).ngroups,
        },
        "validation_notes": [
            "Paper text (Section 4.2 / Appendix) describes CoT zero-shot convergence; this export uses pass@16 T=0.7 JSONL + sparse judge subsample — expect numerical mismatch vs published figure.",
            "Check figure3_median_curve_RECONSTRUCTED.csv: reasoning median std should be < accuracy median std at each N if reconstruction matches qualitative claim (not guaranteed).",
        ],
    }

    p_meta = os.path.join(out_dir, "figure3_metadata.json")
    with open(p_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Validation print
    print("Wrote:", out_dir)
    print(json.dumps(meta["row_counts"], indent=2))
    if not df_med.empty:
        pivot = df_med.pivot_table(
            index="sample_size_N",
            columns="metric_type",
            values="median_std_across_model_benchmark_combos",
        )
        print("\nMedian std (RECONSTRUCTED), selected N:")
        for n in [10, 25, 50, 100]:
            if n in pivot.index:
                print(n, pivot.loc[n].to_dict())


if __name__ == "__main__":
    main()
