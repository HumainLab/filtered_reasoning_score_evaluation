#!/usr/bin/env python3
"""
Bootstrap standard deviations for Table 2 (FRS at top-10% bin = bin_label "0-10").

Reads judged_*.json checkpoints, resamples reasoning_score values 10,000 times per
model×dataset, reports SD of bootstrap means (×100 for 0–100 scale). Model
averages use bootstrap over the mean of six benchmark means (not the average of
per-benchmark SDs).

Usage:
  python bootstrap_table2_std.py
  python bootstrap_table2_std.py --checkpoint-dir reasoning_confidence_bins_results/judging_checkpoints
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

# Column order aligned with topk_ablation / paper tables (math first)
BENCHMARK_ORDER = ["GSM8K", "MATH500", "SVAMP", "AQuA", "CommonsenseQA", "GPQA"]


def stable_rng_seed(base: int, *parts: str) -> int:
    h = hashlib.sha256("|".join([str(base)] + list(parts)).encode()).digest()
    return int.from_bytes(h[:8], "big") % (2**31 - 1) or 1


def parse_judge_filename(path: str) -> Optional[Tuple[str, str]]:
    base = os.path.basename(path)
    m = re.match(r"^judged_(.+)__(.+)\.json$", base)
    if not m:
        return None
    return m.group(1), m.group(2)


def load_scores_010(path: str) -> List[float]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out: List[float] = []
    for s in data.get("judged_samples", []):
        if s.get("bin_label") != "0-10":
            continue
        if s.get("judge_ok") is False:
            continue
        rs = s.get("reasoning_score")
        if rs is None:
            continue
        v = float(rs)
        if not np.isfinite(v):
            continue
        out.append(v)
    return out


def bootstrap_std_of_means(
    scores: List[float],
    n_boot: int,
    seed: int,
) -> Tuple[float, float]:
    """
    Return (mean of scores on 0–1 scale, SD of n_boot bootstrap sample means).
    """
    arr = np.asarray(scores, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        means[b] = float(np.mean(sample))
    return float(np.mean(arr)), float(np.std(means, ddof=0))


def bootstrap_std_of_mean_of_benchmark_means(
    score_lists: List[List[float]],
    n_boot: int,
    seed: int,
) -> Tuple[float, float]:
    """
    score_lists: one list per benchmark (same order as BENCHMARK_ORDER), may be empty.
    Point estimate: mean of per-benchmark sample means (nanmean skips empty).
    Bootstrap: each iteration resamples within each non-empty list, takes mean per bench, then mean across benches.
    """
    arrays = [np.asarray(s, dtype=float) for s in score_lists]
    arrays = [a[np.isfinite(a)] for a in arrays]

    def observed_avg() -> float:
        ms = []
        for a in arrays:
            if len(a) == 0:
                continue
            ms.append(float(np.mean(a)))
        if not ms:
            return float("nan")
        return float(np.mean(ms))

    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        bench_means: List[float] = []
        for a in arrays:
            n = len(a)
            if n == 0:
                continue
            sample = rng.choice(a, size=n, replace=True)
            bench_means.append(float(np.mean(sample)))
        if not bench_means:
            means[b] = float("nan")
        else:
            means[b] = float(np.mean(bench_means))

    obs = observed_avg()
    if not np.isfinite(obs):
        return float("nan"), float("nan")
    return obs, float(np.std(means, ddof=0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint-dir",
        type=str,
        default="reasoning_confidence_bins_results/judging_checkpoints",
        help="Directory containing judged_*.json",
    )
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pattern = os.path.join(args.checkpoint_dir, "judged_*.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"No files matched: {pattern}")

    # model -> {benchmark: scores}
    data: Dict[str, Dict[str, List[float]]] = {}
    for p in paths:
        parsed = parse_judge_filename(p)
        if not parsed:
            continue
        model, dataset = parsed
        scores = load_scores_010(p)
        data.setdefault(model, {})[dataset] = scores

    benchmarks = list(BENCHMARK_ORDER)
    models = sorted(data.keys())

    # Per-cell mean & bootstrap SD (0–1), then ×100
    cell: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for model in models:
        for bench in benchmarks:
            scores = data.get(model, {}).get(bench, [])
            seed = stable_rng_seed(args.seed, model, bench, "cell")
            m_01, sd_01 = bootstrap_std_of_means(scores, args.n_boot, seed)
            cell[(model, bench)] = (m_01 * 100.0, sd_01 * 100.0)

    # Per-model average (bootstrap across benchmarks)
    row_avg: Dict[str, Tuple[float, float]] = {}
    for model in models:
        lists = [data.get(model, {}).get(b, []) for b in benchmarks]
        seed = stable_rng_seed(args.seed, model, "AVG_ROW")
        obs_01, sd_01 = bootstrap_std_of_mean_of_benchmark_means(lists, args.n_boot, seed)
        row_avg[model] = (obs_01 * 100.0, sd_01 * 100.0)

    # Sort by FRS Avg descending
    models_sorted = sorted(models, key=lambda m: row_avg[m][0], reverse=True)

    def fmt_cell(mean: float, std: float) -> str:
        if not np.isfinite(mean):
            return "—"
        s = round(std, 1) if np.isfinite(std) else float("nan")
        m = round(mean, 1)
        if not np.isfinite(s):
            return f"{m:.1f}"
        return f"{m:.1f} ± {s:.1f}"

    # --- Text table ---
    col_w_model = max(14, max(len(m) for m in models_sorted) + 2)
    header = f"{'Model':<{col_w_model}}" + "".join(f"{b:>14}" for b in benchmarks) + f"{'Avg':>16}"
    print()
    print("Table 2 — FRS (top-10% bin) with bootstrap SD of the mean (×100 scale)")
    print(f"(n_boot={args.n_boot}, seed={args.seed})")
    print()
    print(header)
    print("-" * len(header))
    for model in models_sorted:
        line = f"{model:<{col_w_model}}"
        for bench in benchmarks:
            m, s = cell[(model, bench)]
            line += f"{fmt_cell(m, s):>14}"
        am, astd = row_avg[model]
        line += f"{fmt_cell(am, astd):>16}"
        print(line)
    print()

    # --- LaTeX ---
    bench_tex = [b.replace("_", r"\_") for b in benchmarks]
    print("LaTeX (booktabs-style row; adjust column spec as needed):")
    print()
    print(r"\begin{tabular}{l" + "r" * (len(benchmarks) + 1) + "}")
    print(r"\toprule")
    print(
        "Model & "
        + " & ".join(bench_tex)
        + r" & Avg \\"
    )
    print(r"\midrule")
    for model in models_sorted:
        esc = model.replace("_", r"\_")
        parts = [esc]
        for bench in benchmarks:
            m, s = cell[(model, bench)]
            if not np.isfinite(m):
                parts.append("---")
            elif np.isfinite(s):
                parts.append(f"{m:.1f} $\\pm$ {s:.1f}")
            else:
                parts.append(f"{m:.1f}")
        am, astd = row_avg[model]
        if np.isfinite(am) and np.isfinite(astd):
            parts.append(f"{am:.1f} $\\pm$ {astd:.1f}")
        elif np.isfinite(am):
            parts.append(f"{am:.1f}")
        else:
            parts.append("---")
        print(" & ".join(parts) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print()


if __name__ == "__main__":
    main()
