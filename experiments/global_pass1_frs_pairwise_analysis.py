#!/usr/bin/env python3
"""
Pairwise analysis: |Δ pass@1| vs |Δ FRS| across model pairs.

Default (paper tables):
  - FRS: paper_frs_by_benchmark.csv — per-benchmark rubric FRS + FRS_Avg / Acc_Avg
  - Pass@1: paper_pass1_reasoning_by_benchmark.csv — column base_acc (long model names)

Legacy (repo proxies): --legacy uses topk_ablation + ablation_rankings.

Outputs under global_pass1_frs_analysis/ (see --out-dir).
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Nimbus Roman", "serif"],
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 120,
        "savefig.dpi": 300,
    }
)

# HuggingFace / eval API names -> short names used in FRS leaderboard CSV
LONG_TO_SHORT: dict[str, str] = {
    "DeepSeek-R1-Distill-Qwen-1.5B": "DS-R1-1.5B",
    "DeepSeek-R1-Distill-Qwen-7B": "DS-R1-7B",
    "LLaMA-3.1-8B-Instruct": "LLaMA-3.1-8B",
    "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "Qwen2.5-Math-7B": "Qwen2.5-Math",
    "Phi-4": "Phi-4",
    "Phi-4-Reasoning": "Phi-4-Reas.",
    "Qwen3-4B-thinking": "Qwen3-4B",
    "Gemma-7B": "Gemma-7B",
}

# Wide FRS column -> benchmark key aligned with pass@1 table (MATH == MATH500)
FRS_WIDE_TO_BENCH = [
    ("GSM8K", "GSM8K"),
    ("MATH", "MATH500"),
    ("SVAMP", "SVAMP"),
    ("AQuA", "AQuA"),
    ("GPQA", "GPQA"),
    ("CSQA", "CSQA"),
]

TOPK_PATH = "outputs/topk_ablation_results/topk_ablation_results.csv"
RANK_PATH = "outputs/sample_count_ablation_results/ablation_rankings.csv"


def _print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def frs_wide_to_long(frs_wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in frs_wide.iterrows():
        m = str(r["model"])
        for col, bench in FRS_WIDE_TO_BENCH:
            rows.append({"model": m, "benchmark": bench, "frs": float(r[col])})
    return pd.DataFrame(rows)


def load_paper_merged(frs_path: Path, pass_path: Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Returns (per_benchmark_table, model_avg_table or None).
    per_benchmark: model, benchmark, pass1_pct, frs_pct, base_reasoning (optional)
    """
    frs_w = pd.read_csv(frs_path)
    pass_df = pd.read_csv(pass_path)
    pass_df = pass_df.copy()
    pass_df["model_short"] = pass_df["model"].map(LONG_TO_SHORT)
    miss = pass_df["model_short"].isna()
    if miss.any():
        bad = pass_df.loc[miss, "model"].unique()
        raise ValueError(f"Unmapped model name(s) in pass@1 table: {bad}")

    frs_long = frs_wide_to_long(frs_w)
    merged = pd.merge(
        frs_long,
        pass_df.rename(columns={"base_acc": "pass1_pct"})[
            ["model_short", "benchmark", "pass1_pct", "base_reasoning", "snr"]
        ],
        left_on=["model", "benchmark"],
        right_on=["model_short", "benchmark"],
        how="inner",
    )
    merged = merged.rename(columns={"frs": "frs_pct"}).drop(columns=["model_short"])

    avg = None
    if "FRS_Avg" in frs_w.columns and "Acc_Avg" in frs_w.columns:
        avg = frs_w[["model", "FRS_Avg", "Acc_Avg"]].copy()
        avg = avg.rename(columns={"FRS_Avg": "frs_avg", "Acc_Avg": "acc_avg"})

    return merged, avg


def load_legacy_merged(root: Path) -> tuple[pd.DataFrame, None]:
    topk = pd.read_csv(root / TOPK_PATH)
    rank = pd.read_csv(root / RANK_PATH)
    topk100 = topk[topk["top_k_pct"] == 100].copy()
    topk100 = topk100.rename(columns={"accuracy": "pass1_pct", "dataset": "benchmark"})
    rank16 = rank[rank["k_sub"] == 16].copy()
    rank16 = rank16.rename(columns={"dataset": "benchmark", "frs_acc_mean": "frs_pct"})
    merged = pd.merge(
        topk100[["model", "benchmark", "pass1_pct"]],
        rank16[["model", "benchmark", "frs_pct"]],
        on=["model", "benchmark"],
        how="inner",
    )
    merged["base_reasoning"] = np.nan
    merged["snr"] = np.nan
    return merged, None


def pairwise_per_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bench, sub in df.groupby("benchmark", sort=True):
        sub = sub.dropna(subset=["pass1_pct", "frs_pct"])
        models = sorted(sub["model"].unique())
        for a, b in itertools.combinations(models, 2):
            pa = float(sub.loc[sub["model"] == a, "pass1_pct"].iloc[0])
            pb = float(sub.loc[sub["model"] == b, "pass1_pct"].iloc[0])
            fa = float(sub.loc[sub["model"] == a, "frs_pct"].iloc[0])
            fb = float(sub.loc[sub["model"] == b, "frs_pct"].iloc[0])
            rows.append(
                {
                    "benchmark": str(bench),
                    "model_a": a,
                    "model_b": b,
                    "pass1_a": pa,
                    "pass1_b": pb,
                    "frs_a": fa,
                    "frs_b": fb,
                    "signed_pass1_gap": pa - pb,
                    "signed_frs_gap": fa - fb,
                    "abs_pass1_gap": abs(pa - pb),
                    "abs_frs_gap": abs(fa - fb),
                }
            )
    return pd.DataFrame(rows)


def pairwise_from_averages(avg: pd.DataFrame) -> pd.DataFrame:
    """Unordered pairs on FRS_Avg vs Acc_Avg (one row per model pair)."""
    rows = []
    models = sorted(avg["model"].unique())
    for a, b in itertools.combinations(models, 2):
        pa = float(avg.loc[avg["model"] == a, "acc_avg"].iloc[0])
        pb = float(avg.loc[avg["model"] == b, "acc_avg"].iloc[0])
        fa = float(avg.loc[avg["model"] == a, "frs_avg"].iloc[0])
        fb = float(avg.loc[avg["model"] == b, "frs_avg"].iloc[0])
        rows.append(
            {
                "benchmark": "AGGREGATED_FRS_Avg_vs_Acc_Avg",
                "model_a": a,
                "model_b": b,
                "pass1_a": pa,
                "pass1_b": pb,
                "frs_a": fa,
                "frs_b": fb,
                "signed_pass1_gap": pa - pb,
                "signed_frs_gap": fa - fb,
                "abs_pass1_gap": abs(pa - pb),
                "abs_frs_gap": abs(fa - fb),
            }
        )
    return pd.DataFrame(rows)


def core_conditional_stats(pairs: pd.DataFrame) -> dict:
    out = {}
    for thr in (1.0, 2.0):
        sub = pairs[pairs["abs_pass1_gap"] <= thr]
        key = f"abs_pass1_gap_le_{thr:g}"
        row = {
            "n": len(sub),
            "mean_abs_frs_gap": float(sub["abs_frs_gap"].mean()) if len(sub) else np.nan,
            "median_abs_frs_gap": float(sub["abs_frs_gap"].median()) if len(sub) else np.nan,
            "max_abs_frs_gap": float(sub["abs_frs_gap"].max()) if len(sub) else np.nan,
        }
        for g in (5, 10, 15, 20):
            row[f"pct_ge_{g}"] = float((sub["abs_frs_gap"] >= g).mean()) if len(sub) else np.nan
        out[key] = row
    return out


def bucket_stats(pairs: pd.DataFrame) -> pd.DataFrame:
    bins = [(0, 1), (1, 2), (2, 3), (3, 5), (5, np.inf)]
    labels = ["[0,1)", "[1,2)", "[2,3)", "[3,5)", "[5,inf)"]
    x = pairs["abs_pass1_gap"].values
    records = []
    for (lo, hi), lab in zip(bins, labels):
        mask = (x >= lo) & (x < hi) if np.isfinite(hi) else (x >= lo)
        sub = pairs.loc[mask, "abs_frs_gap"]
        records.append(
            {
                "abs_pass1_gap_bucket": lab,
                "n_pairs": int(len(sub)),
                "mean_abs_frs_gap": float(sub.mean()) if len(sub) else np.nan,
                "median_abs_frs_gap": float(sub.median()) if len(sub) else np.nan,
                "max_abs_frs_gap": float(sub.max()) if len(sub) else np.nan,
                "p25_abs_frs_gap": float(sub.quantile(0.25)) if len(sub) else np.nan,
                "p75_abs_frs_gap": float(sub.quantile(0.75)) if len(sub) else np.nan,
            }
        )
    return pd.DataFrame(records)


def run_analysis(
    merged: pd.DataFrame,
    pairs_main: pd.DataFrame,
    pairs_agg: pd.DataFrame | None,
    out_dir: Path,
    case_a: str,
    case_b: str,
    case_benchmark: str,
    data_label: str,
) -> None:
    main_pairs = pairs_main.copy()
    assert main_pairs.duplicated(subset=["benchmark", "model_a", "model_b"]).sum() == 0

    _print_header("STATISTICS (main pairwise table)")
    x, y = main_pairs["abs_pass1_gap"].values, main_pairs["abs_frs_gap"].values
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    print(f"Pearson(|Δpass@1|, |ΔFRS|):  r = {pr:.4f}, p = {pp:.4g}")
    print(f"Spearman(|Δpass@1|, |ΔFRS|): rho = {sr:.4f}, p = {sp:.4g}")

    bstats = bucket_stats(main_pairs)
    bstats.to_csv(out_dir / "pairwise_gap_summary_buckets.csv", index=False)
    core = core_conditional_stats(main_pairs)
    core_rows = [{**{"subset": k}, **v} for k, v in core.items()]
    core_df = pd.DataFrame(core_rows)
    core_df.to_csv(out_dir / "pairwise_gap_summary_core.csv", index=False)
    with open(out_dir / "pairwise_gap_summary.csv", "w", encoding="utf-8") as f:
        f.write(f"# Data: {data_label}\n")
        f.write("# Bucket stats\n")
        bstats.to_csv(f, index=False)
        f.write("\n# Core conditionals\n")
        core_df.to_csv(f, index=False)
    print("\nBucket table:\n", bstats.to_string(index=False))
    print("\nCore:\n", core_df.to_string(index=False))

    if pairs_agg is not None and len(pairs_agg):
        _print_header("SUPPLEMENTARY: FRS_Avg vs Acc_Avg pairs (36)")
        xa, ya = pairs_agg["abs_pass1_gap"].values, pairs_agg["abs_frs_gap"].values
        pr2, pp2 = pearsonr(xa, ya)
        sr2, sp2 = spearmanr(xa, ya)
        print(f"Pearson:  r = {pr2:.4f}, p = {pp2:.4g}")
        print(f"Spearman: rho = {sr2:.4f}, p = {sp2:.4g}")
        pairs_agg.to_csv(out_dir / "pairwise_pass1_frs_gaps_aggregated.csv", index=False)

    top10 = main_pairs.sort_values(
        by=["abs_pass1_gap", "abs_frs_gap"], ascending=[True, False]
    ).head(10)
    top10.to_csv(out_dir / "top10_small_pass_gap_large_frs_gap.csv", index=False)
    print("\nTop 10 (smallest |Δpass@1|, then largest |ΔFRS|):\n", top10.to_string(index=False))

    # Figure
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.scatter(x, y, alpha=0.65, s=22, edgecolors="none")
    ax.axvline(1.0, color="0.45", linestyle="--", linewidth=0.9, zorder=0)
    ax.axvline(2.0, color="0.65", linestyle=":", linewidth=0.9, zorder=0)
    ax.axhline(10.0, color="0.55", linestyle="--", linewidth=0.9, zorder=0)
    ax.set_xlabel(r"|$\Delta$ pass@1| (percentage points)")
    ax.set_ylabel(r"|$\Delta$ FRS| (percentage points)")
    ax.set_title("Similar pass@1 gaps can mask large FRS gaps")

    cs = main_pairs[
        (main_pairs["model_a"].isin([case_a, case_b]))
        & (main_pairs["model_b"].isin([case_a, case_b]))
        & (main_pairs["benchmark"] == case_benchmark)
    ]
    seen: set[tuple[str, str, str]] = set()

    def _key(r) -> tuple[str, str, str]:
        a, b = sorted([r["model_a"], r["model_b"]])
        return (a, b, str(r["benchmark"]))

    if len(cs) == 1:
        row = cs.iloc[0]
        ax.annotate(
            f"{case_a} vs {case_b}\n({case_benchmark})",
            xy=(row["abs_pass1_gap"], row["abs_frs_gap"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=7,
            arrowprops=dict(arrowstyle="-", color="0.3", lw=0.6),
        )
        seen.add(_key(row))

    candidates = main_pairs[main_pairs["abs_pass1_gap"] < 2.0].nlargest(8, "abs_frs_gap")
    extra = 0
    for _, row in candidates.iterrows():
        k = _key(row)
        if k in seen:
            continue
        if extra >= 2:
            break
        ax.annotate(
            f'{row["model_a"]} vs {row["model_b"]}\n({row["benchmark"]})',
            xy=(row["abs_pass1_gap"], row["abs_frs_gap"]),
            xytext=(10, -10),
            textcoords="offset points",
            fontsize=6,
            alpha=0.9,
            arrowprops=dict(arrowstyle="-", color="0.35", lw=0.5),
        )
        seen.add(k)
        extra += 1

    fig.tight_layout()
    png, pdf = out_dir / "pass1_vs_frs_pairwise_scatter.png", out_dir / "pass1_vs_frs_pairwise_scatter.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(6.2, 4.0))
    order = ["[0,1)", "[1,2)", "[2,3)", "[3,5)", "[5,inf)"]
    data_box = [
        main_pairs[(main_pairs["abs_pass1_gap"] >= 0) & (main_pairs["abs_pass1_gap"] < 1)]["abs_frs_gap"],
        main_pairs[(main_pairs["abs_pass1_gap"] >= 1) & (main_pairs["abs_pass1_gap"] < 2)]["abs_frs_gap"],
        main_pairs[(main_pairs["abs_pass1_gap"] >= 2) & (main_pairs["abs_pass1_gap"] < 3)]["abs_frs_gap"],
        main_pairs[(main_pairs["abs_pass1_gap"] >= 3) & (main_pairs["abs_pass1_gap"] < 5)]["abs_frs_gap"],
        main_pairs[main_pairs["abs_pass1_gap"] >= 5]["abs_frs_gap"],
    ]
    ax2.boxplot([d.values for d in data_box], tick_labels=order, showfliers=True)
    ax2.set_ylabel(r"|$\Delta$ FRS| (pp)")
    ax2.set_xlabel(r"|$\Delta$ pass@1| bucket (pp)")
    ax2.set_title("|ΔFRS| distribution by |Δpass@1| bucket")
    fig2.tight_layout()
    fig2.savefig(out_dir / "pass1_bucket_frs_boxplot.png", bbox_inches="tight")
    fig2.savefig(out_dir / "pass1_bucket_frs_boxplot.pdf", bbox_inches="tight")
    plt.close()

    top_pairs = main_pairs.sort_values(
        by=["abs_pass1_gap", "abs_frs_gap"], ascending=[True, False]
    ).head(10)[["model_a", "model_b", "benchmark", "abs_pass1_gap", "abs_frs_gap"]]

    summary = f"""GLOBAL PASS@1 vs FRS — PAIRWISE ANALYSIS
==============================================
Data: {data_label}

Per-benchmark table: pass@1 = base_acc (paper); FRS = per-benchmark FRS from paper_frs_by_benchmark.csv.
Model names in pass@1 CSV were mapped to short names via LONG_TO_SHORT (see script).

Correlations (main, N={len(main_pairs)} unordered pairs × benchmarks)
- Pearson |Δpass@1| vs |ΔFRS|:  r = {pr:.4f} (p = {pp:.4g})
- Spearman: ρ = {sr:.4f} (p = {sp:.4g})

Among |Δpass@1| ≤ 1 pp (N = {core['abs_pass1_gap_le_1']['n']}):
- Median |ΔFRS|: {core['abs_pass1_gap_le_1']['median_abs_frs_gap']:.2f} pp; max: {core['abs_pass1_gap_le_1']['max_abs_frs_gap']:.2f} pp

Among |Δpass@1| ≤ 2 pp (N = {core['abs_pass1_gap_le_2']['n']}):
- Median |ΔFRS|: {core['abs_pass1_gap_le_2']['median_abs_frs_gap']:.2f} pp
- Max |ΔFRS|: {core['abs_pass1_gap_le_2']['max_abs_frs_gap']:.2f} pp
- Share |ΔFRS| ≥ 10: {core['abs_pass1_gap_le_2']['pct_ge_10']:.1%}

Top examples (smallest |Δpass@1|, then largest |ΔFRS|):
{top_pairs.to_string(index=False)}

Technical summary
-----------------
Pass@1 (base_acc) and FRS are different measurements on the same models and benchmarks.
Spearman ρ ≈ {sr:.2f} between |Δpass@1| and |ΔFRS| indicates only moderate alignment of
**gap magnitudes** across pairs: similar pass@1 differences do not imply similar FRS
differences. In the near-tie regime (|Δpass@1|≤2 pp), |ΔFRS| remains often large
(median {core['abs_pass1_gap_le_2']['median_abs_frs_gap']:.1f} pp; up to {core['abs_pass1_gap_le_2']['max_abs_frs_gap']:.1f} pp).

Figure caption (candidate)
--------------------------
**Near-equal pass@1 can hide large FRS differences.**
Each point is one unordered model pair within one benchmark (N={len(main_pairs)}).
Axes: absolute pass@1 gap and absolute FRS gap (percentage points). Vertical lines at
|Δpass@1| = 1 and 2 pp; horizontal line at |ΔFRS| = 10 pp.

Files
-----
- merged_pass1_frs_per_benchmark.csv
- pairwise_pass1_frs_gaps.csv
- pairwise_pass1_frs_gaps_aggregated.csv (FRS_Avg vs Acc_Avg)
- pairwise_gap_summary*.csv
- top10_small_pass_gap_large_frs_gap.csv
- pass1_vs_frs_pairwise_scatter.png / .pdf
- pass1_bucket_frs_boxplot.png / .pdf
"""
    (out_dir / "pass1_vs_frs_summary.txt").write_text(summary, encoding="utf-8")
    print(f"Saved {png}, {pdf}, {out_dir / 'pass1_vs_frs_summary.txt'}")

    print("\nStrongest findings:")
    print(f"  1. Spearman ρ = {sr:.3f} (global association of gap magnitudes).")
    print(
        f"  2. Near-ties |Δpass@1|≤2: median |ΔFRS| = {core['abs_pass1_gap_le_2']['median_abs_frs_gap']:.2f} pp; max = {core['abs_pass1_gap_le_2']['max_abs_frs_gap']:.2f} pp."
    )
    print(
        f"  3. Share with |ΔFRS|≥10 among |Δpass@1|≤2: {core['abs_pass1_gap_le_2']['pct_ge_10']:.0%}."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: <root>/global_pass1_frs_analysis",
    )
    ap.add_argument(
        "--legacy",
        action="store_true",
        help="Use topk_ablation + ablation_rankings proxies instead of paper CSVs",
    )
    ap.add_argument("--frs-csv", type=Path, default=None)
    ap.add_argument("--pass-csv", type=Path, default=None)
    ap.add_argument("--case-a", default="DS-R1-7B")
    ap.add_argument("--case-b", default="Qwen2.5-Math")
    ap.add_argument("--case-benchmark", default="SVAMP")
    args = ap.parse_args()
    root = args.root.resolve()
    out_dir = args.out_dir or (root / "outputs/global_pass1_frs_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    _print_header("PART 1: LOAD & VALIDATE")
    if args.legacy:
        merged, avg = load_legacy_merged(root)
        data_label = f"legacy: {TOPK_PATH} + {RANK_PATH}"
        print(data_label)
    else:
        frs_p = args.frs_csv or (out_dir / "paper_frs_by_benchmark.csv")
        pass_p = args.pass_csv or (out_dir / "paper_pass1_reasoning_by_benchmark.csv")
        merged, avg = load_paper_merged(frs_p, pass_p)
        data_label = f"paper tables: {frs_p.name} + {pass_p.name}"
        print(f"FRS CSV: {frs_p}")
        print(f"Pass@1 CSV: {pass_p}")
        print("LONG_TO_SHORT mapping:", LONG_TO_SHORT)

    print("\nMerged per-benchmark table (first 12 rows):")
    print(merged.head(12).to_string(index=False))
    print("\nColumns:", list(merged.columns))
    print("Rows:", len(merged))
    print("Models:", sorted(merged["model"].unique()))
    print("Benchmarks:", sorted(merged["benchmark"].unique()))

    pairs_bench = pairwise_per_benchmark(merged)
    pairs_bench.to_csv(out_dir / "pairwise_pass1_frs_gaps.csv", index=False)
    print(f"\nSaved {out_dir / 'pairwise_pass1_frs_gaps.csv'} ({len(pairs_bench)} rows)")
    merged.to_csv(out_dir / "merged_pass1_frs_per_benchmark.csv", index=False)
    print(f"Saved {out_dir / 'merged_pass1_frs_per_benchmark.csv'}")

    pairs_agg = pairwise_from_averages(avg) if avg is not None else None

    run_analysis(
        merged,
        pairs_bench,
        pairs_agg,
        out_dir,
        args.case_a,
        args.case_b,
        args.case_benchmark,
        data_label,
    )


if __name__ == "__main__":
    main()
