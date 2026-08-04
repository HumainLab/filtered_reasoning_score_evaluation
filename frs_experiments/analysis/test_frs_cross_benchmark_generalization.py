#!/usr/bin/env python3
"""
Cross-Benchmark Generalization Test for FRS.

Hypothesis: FRS captures a transferable model property, not just a
benchmark-specific reshuffling of rankings.

Protocol: Leave-One-Benchmark-Out (LOBO) rank-correlation analysis.
  For each held-out benchmark (6 folds):
    - Aggregate FRS (and pass@1) across the 5 training benchmarks per model
    - Compute Spearman / Pearson / Kendall τ between train-side aggregate
      and held-out benchmark FRS (primary) and pass@1 (secondary)
    - Bootstrap 95% CIs (n=9 is too small for asymptotic guarantees)
    - Permutation test p-value

Outputs saved to:
  analysis/cross_benchmark_generalization/
    frs_cross_benchmark_results.csv
    frs_cross_benchmark_summary.md

Usage:
  python analysis/test_frs_cross_benchmark_generalization.py --repo-root .
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

N_BOOTSTRAP = 10_000
RNG_SEED = 42
ALPHA = 0.05

# Aggregation methods to try (mean is primary; median as sensitivity check)
AGG_METHODS = ["mean", "median"]

# Metrics to compare (train aggregate vs held-out)
TRAIN_METRICS = ["frs_pct", "pass1_pct"]
TEST_TARGETS = ["frs_pct", "pass1_pct"]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _safe_corr(x: np.ndarray, y: np.ndarray, kind: str) -> tuple[float, float]:
    """Return (r, p) for Spearman / Pearson / Kendall; NaN if insufficient data."""
    if len(x) < 3:
        return float("nan"), float("nan")
    if kind == "spearman":
        r, p = stats.spearmanr(x, y)
    elif kind == "pearson":
        r, p = stats.pearsonr(x, y)
    elif kind == "kendall":
        r, p = stats.kendalltau(x, y)
    else:
        raise ValueError(kind)
    return float(r), float(p)


def bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    kind: str,
    n_boot: int = N_BOOTSTRAP,
    rng: np.random.Generator | None = None,
    alpha: float = ALPHA,
) -> tuple[float, float]:
    """Bootstrap percentile CI for a correlation coefficient."""
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    n = len(x)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        r, _ = _safe_corr(xb, yb, kind)
        boot[i] = r
    lo = float(np.nanpercentile(boot, 100 * alpha / 2))
    hi = float(np.nanpercentile(boot, 100 * (1 - alpha / 2)))
    return lo, hi


def permutation_pvalue(
    x: np.ndarray,
    y: np.ndarray,
    kind: str,
    n_perm: int = N_BOOTSTRAP,
    rng: np.random.Generator | None = None,
) -> float:
    """One-sided (positive) permutation p-value: P(r_perm >= r_obs)."""
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    r_obs, _ = _safe_corr(x, y, kind)
    if np.isnan(r_obs):
        return float("nan")
    count = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        rp, _ = _safe_corr(x, yp, kind)
        if rp >= r_obs:
            count += 1
    return count / n_perm


# ──────────────────────────────────────────────────────────────────────────────
# Core analysis
# ──────────────────────────────────────────────────────────────────────────────

def run_lobo(
    panel: pd.DataFrame,
    agg_method: str = "mean",
    n_boot: int = N_BOOTSTRAP,
) -> pd.DataFrame:
    """
    Leave-One-Benchmark-Out (LOBO) cross-benchmark rank correlation.

    Returns a DataFrame of per-fold results with one row per
    (held_out_benchmark, train_metric, test_target, corr_kind).
    """
    rng = np.random.default_rng(RNG_SEED)
    benchmarks = sorted(panel["benchmark"].unique())
    records = []

    for held_out in benchmarks:
        train_df = panel[panel["benchmark"] != held_out].copy()
        test_df  = panel[panel["benchmark"] == held_out].copy()

        # Aggregate per model over the 5 training benchmarks
        if agg_method == "mean":
            train_agg = train_df.groupby("model")[["frs_pct", "pass1_pct"]].mean()
        else:  # median
            train_agg = train_df.groupby("model")[["frs_pct", "pass1_pct"]].median()

        # Align model order
        test_agg = test_df.set_index("model")[["frs_pct", "pass1_pct"]]
        models_common = sorted(set(train_agg.index) & set(test_agg.index))
        n_models = len(models_common)

        for train_metric in TRAIN_METRICS:
            x = train_agg.loc[models_common, train_metric].values.astype(float)

            for test_target in TEST_TARGETS:
                y = test_agg.loc[models_common, test_target].values.astype(float)

                for kind in ["spearman", "pearson", "kendall"]:
                    r, p_param = _safe_corr(x, y, kind)
                    ci_lo, ci_hi = bootstrap_ci(x, y, kind, n_boot=n_boot, rng=rng)
                    p_perm = permutation_pvalue(x, y, kind, n_perm=n_boot, rng=rng)

                    records.append({
                        "held_out_benchmark": held_out,
                        "n_train_benchmarks": len(benchmarks) - 1,
                        "n_models": n_models,
                        "agg_method": agg_method,
                        "train_metric": train_metric,
                        "test_target": test_target,
                        "corr_kind": kind,
                        "r": round(r, 4),
                        "p_parametric": round(p_param, 4),
                        "p_permutation": round(p_perm, 4),
                        "ci_lo_95": round(ci_lo, 4),
                        "ci_hi_95": round(ci_hi, 4),
                    })

    return pd.DataFrame(records)


def compute_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and std of r across LOBO folds for each (train_metric, test_target, kind, agg)."""
    return (
        results.groupby(["agg_method", "train_metric", "test_target", "corr_kind"])
        .agg(
            n_folds=("r", "count"),
            mean_r=("r", "mean"),
            std_r=("r", "std"),
            n_positive=("r", lambda s: (s > 0).sum()),
            mean_ci_lo=("ci_lo_95", "mean"),
            mean_ci_hi=("ci_hi_95", "mean"),
            mean_p_perm=("p_permutation", "mean"),
        )
        .reset_index()
        .round(4)
    )


# ──────────────────────────────────────────────────────────────────────────────
# Markdown report
# ──────────────────────────────────────────────────────────────────────────────

def write_markdown_report(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    panel: pd.DataFrame,
    out_path: Path,
) -> None:
    benchmarks = sorted(panel["benchmark"].unique())
    models = sorted(panel["model"].unique())
    n_pairs = len(panel)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: list[str] = []
    A = lines.append

    A("# FRS Cross-Benchmark Generalization Test")
    A(f"\nGenerated: {ts}")
    A(f"\n**Source file:** `analysis/frs_predictor_panel.csv` ({n_pairs} model×benchmark pairs)")
    A(f"\n**Models (n={len(models)}):** {', '.join(models)}")
    A(f"\n**Benchmarks (n={len(benchmarks)}):** {', '.join(benchmarks)}")
    A("")

    A("## What this tests")
    A(textwrap.dedent("""
    **Hypothesis:** FRS is a stable, transferable model property — a model's FRS rank
    on 5 benchmarks should predict its FRS (and accuracy) on an unseen 6th benchmark.

    **Protocol:** Leave-One-Benchmark-Out (LOBO).
    For each held-out benchmark:
    1. Aggregate each model's FRS (and pass@1) over the 5 training benchmarks (mean & median).
    2. Rank-correlate train-side aggregate with held-out benchmark FRS and pass@1.
    3. Report Spearman ρ (primary), Pearson r, Kendall τ with bootstrap 95% CIs and
       permutation p-values (n=9 is too small for asymptotic tests).

    **How this differs from the existing LODO analysis** (`generalization_results.csv`):
    The existing LODO tests whether FRS *predicts reasoning quality* (judge-rated `unfiltered_
    reasoning_mean`) in an OLS regression — FRS as a predictor, not as the outcome.
    This analysis tests whether FRS *as a metric is consistent across benchmarks* —
    i.e., whether it measures a stable model property rather than a benchmark-specific artifact.
    """))

    A("## Per-fold results (primary: Spearman, agg=mean, train FRS → test FRS)")
    A("")

    primary = results[
        (results["agg_method"] == "mean") &
        (results["train_metric"] == "frs_pct") &
        (results["test_target"] == "frs_pct") &
        (results["corr_kind"] == "spearman")
    ][["held_out_benchmark", "n_models", "r", "ci_lo_95", "ci_hi_95",
       "p_parametric", "p_permutation"]].copy()

    A(primary.to_markdown(index=False))
    A("")

    primary_r = primary["r"].dropna()
    n_pos = (primary_r > 0).sum()
    A(f"**Mean ρ across folds:** {primary_r.mean():.3f} (std: {primary_r.std():.3f})")
    A(f"**Positive folds:** {n_pos}/{len(primary_r)}")
    A("")

    A("## Train FRS → Test pass@1 (cross-metric, agg=mean, Spearman)")
    A("")
    cross = results[
        (results["agg_method"] == "mean") &
        (results["train_metric"] == "frs_pct") &
        (results["test_target"] == "pass1_pct") &
        (results["corr_kind"] == "spearman")
    ][["held_out_benchmark", "n_models", "r", "ci_lo_95", "ci_hi_95",
       "p_parametric", "p_permutation"]].copy()
    A(cross.to_markdown(index=False))
    A("")

    A("## Train pass@1 → Test FRS (baseline comparison, agg=mean, Spearman)")
    A("")
    baseline = results[
        (results["agg_method"] == "mean") &
        (results["train_metric"] == "pass1_pct") &
        (results["test_target"] == "frs_pct") &
        (results["corr_kind"] == "spearman")
    ][["held_out_benchmark", "n_models", "r", "ci_lo_95", "ci_hi_95",
       "p_parametric", "p_permutation"]].copy()
    A(baseline.to_markdown(index=False))
    A("")

    A("## Summary across folds")
    A("")
    sum_primary = summary[
        (summary["agg_method"] == "mean") &
        (summary["corr_kind"].isin(["spearman", "pearson", "kendall"]))
    ][["agg_method", "train_metric", "test_target", "corr_kind",
       "n_folds", "mean_r", "std_r", "n_positive",
       "mean_ci_lo", "mean_ci_hi", "mean_p_perm"]].copy()
    A(sum_primary.to_markdown(index=False))
    A("")

    A("## Model × benchmark FRS matrix (input data)")
    A("")
    pivot = panel.pivot_table(index="model", columns="benchmark",
                               values="frs_pct", aggfunc="first")
    A(pivot.round(1).to_markdown())
    A("")

    A("## Honest assessment")
    A(textwrap.dedent("""
    ### Sample size caveat
    With n=9 models per fold, every correlation estimate has wide uncertainty.
    Bootstrap 95% CIs should be interpreted as rough guides, not precision estimates.
    The permutation p-value is reported but should not be over-interpreted given
    the small n — it is included to show whether the observed correlation is
    distinguishable from chance under random permutation.

    ### Benchmark heterogeneity
    The 6 benchmarks span math (GSM8K, MATH500, SVAMP, AQuA), science (GPQA), and
    commonsense (CSQA). FRS cross-benchmark consistency may be lower for CSQA
    (structurally different from math benchmarks) than within math benchmarks.
    Per-fold ρ values should be read with this in mind.

    ### What different correlation levels would mean

    **ρ ≥ 0.7 (strong):**
    FRS is largely a model property, consistent across benchmarks. Strong cross-
    benchmark transferability. Addresses the reviewer concern that FRS merely
    reshuffles rankings within each benchmark.

    **ρ ∈ [0.4, 0.7) (moderate):**
    FRS has a meaningful cross-benchmark signal but also benchmark-specific
    components. Partial transferability. Useful to note, but the claim should be
    qualified: *FRS is partially transferable, not purely benchmark-specific.*

    **ρ < 0.4 (weak) or inconsistent across folds:**
    FRS rankings are not stable across benchmarks. This does not invalidate
    FRS as a useful within-benchmark metric, but it would undermine claims of
    FRS as a universal model property. The argument would need to reframe
    FRS as a benchmark-conditioned diagnostic rather than a general capability
    indicator.

    ### What this analysis would and would not justify
    - **Would justify:** "Model FRS rankings are consistent across benchmark contexts,
      suggesting FRS captures a stable model property beyond within-benchmark variance."
    - **Would NOT justify (alone):** "FRS is a better ranking method than pass@1."
      That requires the incremental validity analysis already done in
      `generalization_results.csv`.
    - **Complementary framing:** This analysis + the existing LODO incremental-validity
      analysis together form a two-sided argument: (1) FRS is consistent across
      benchmarks as a model property; (2) FRS adds predictive signal beyond pass@1.

    ### Relationship to the existing LODO analysis
    The existing LODO (`generalization_results.csv`) tests:
      "Does FRS have incremental validity when predicting *reasoning quality* on a held-out
       benchmark?" (FRS as predictor of judge outcome)
    This new analysis tests:
      "Is FRS *itself* consistent as a model-level metric across benchmarks?" (FRS as outcome)
    Both are needed for a complete picture. The existing analysis answers the reviewer
    concern about predictive utility. This analysis answers the concern about construct
    validity (is FRS measuring something stable).
    """))

    A("## Limitations")
    A(textwrap.dedent("""
    1. **n=9 models**: With only 9 data points per fold, even ρ=0.7 has a wide bootstrap
       CI. Results should be described with appropriate uncertainty language.
    2. **n=6 benchmarks**: Six folds is the maximum possible; there is no held-out
       meta-test set. Cross-fold variance in ρ reflects genuine benchmark heterogeneity.
    3. **Non-independence**: Benchmarks are not an i.i.d. sample; math benchmarks are
       structurally similar. The effective number of independent folds is closer to 3–4.
    4. **Aggregation choice**: Mean vs. median aggregation across 5 training benchmarks
       could affect results, especially for outlier models. Sensitivity results are
       included in the full CSV.
    5. **Benchmark heterogeneity**: CSQA is a commonsense benchmark while others are
       math/science. Cross-benchmark ρ for CSQA may be lower for structural reasons
       unrelated to FRS construct validity.
    6. **Low statistical power**: At n=9, permutation p < 0.05 requires ρ ≈ 0.65+.
       Moderate but real correlations may not reach formal significance.
    """))

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", type=Path,
                        help="Path to the repository root (default: .)")
    parser.add_argument("--n-bootstrap", default=N_BOOTSTRAP, type=int,
                        help=f"Bootstrap iterations (default: {N_BOOTSTRAP})")
    args = parser.parse_args(argv)

    repo = args.repo_root.resolve()
    panel_path = repo / "analysis" / "frs_predictor_panel.csv"

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading panel from: {panel_path}")
    if not panel_path.is_file():
        sys.exit(f"ERROR: Panel file not found: {panel_path}\n"
                 "       Run analysis/run_frs_predictor_analysis.py first.")

    panel = pd.read_csv(panel_path)
    required = {"model", "benchmark", "frs_pct", "pass1_pct"}
    missing = required - set(panel.columns)
    if missing:
        sys.exit(f"ERROR: Panel missing columns: {missing}")

    panel = panel[["model", "benchmark", "frs_pct", "pass1_pct",
                   "pass16_pct", "high_conf_accuracy_pct", "unfiltered_reasoning_mean"]].copy()

    print(f"       Rows: {len(panel)}  |  Models: {panel['model'].nunique()}  "
          f"|  Benchmarks: {panel['benchmark'].nunique()}")
    print(f"       Models: {sorted(panel['model'].unique())}")
    print(f"       Benchmarks: {sorted(panel['benchmark'].unique())}")

    # Completeness check
    n_expected = panel["model"].nunique() * panel["benchmark"].nunique()
    if len(panel) < n_expected:
        print(f"WARNING: Expected {n_expected} pairs, got {len(panel)}. "
              "Some model×benchmark pairs are missing.")
    missing_pairs = (
        panel.groupby(["model", "benchmark"])["frs_pct"].count()
        .reset_index()
        .query("frs_pct == 0")
    )
    if len(missing_pairs):
        print(f"WARNING: Missing FRS for {len(missing_pairs)} pairs: {missing_pairs.values}")
    else:
        print("       Data completeness: OK — all model×benchmark pairs present.")

    # ── Run LOBO analysis ──────────────────────────────────────────────────────
    out_dir = repo / "analysis" / "cross_benchmark_generalization"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[2/5] Output directory: {out_dir}")

    n_boot = args.n_bootstrap
    print(f"\n[3/5] Running LOBO analysis (n_bootstrap={n_boot}) ...")

    all_results = []
    for agg in AGG_METHODS:
        print(f"       agg_method={agg} ...", end=" ", flush=True)
        res = run_lobo(panel, agg_method=agg, n_boot=n_boot)
        all_results.append(res)
        print(f"done ({len(res)} rows)")

    results = pd.concat(all_results, ignore_index=True)
    summary = compute_summary(results)

    # ── Terminal report ────────────────────────────────────────────────────────
    print("\n" + "="*72)
    print("RESULTS: FRS Cross-Benchmark Generalization (LOBO)")
    print("="*72)
    print("\n--- Primary: Spearman ρ, train FRS → test FRS (agg=mean) ---")
    primary = results[
        (results["agg_method"] == "mean") &
        (results["train_metric"] == "frs_pct") &
        (results["test_target"] == "frs_pct") &
        (results["corr_kind"] == "spearman")
    ]
    for _, row in primary.iterrows():
        ci = f"[{row.ci_lo_95:.3f}, {row.ci_hi_95:.3f}]"
        print(f"  held-out={row.held_out_benchmark:<10}  ρ={row.r:+.3f}  95%CI={ci:<20}"
              f"  p_perm={row.p_permutation:.3f}")

    r_vals = primary["r"].dropna()
    print(f"\n  Mean ρ = {r_vals.mean():.3f}  Std = {r_vals.std():.3f}  "
          f"Positive: {(r_vals>0).sum()}/{len(r_vals)}")

    print("\n--- Train FRS → Test pass@1 (Spearman, agg=mean) ---")
    cross = results[
        (results["agg_method"] == "mean") &
        (results["train_metric"] == "frs_pct") &
        (results["test_target"] == "pass1_pct") &
        (results["corr_kind"] == "spearman")
    ]
    for _, row in cross.iterrows():
        ci = f"[{row.ci_lo_95:.3f}, {row.ci_hi_95:.3f}]"
        print(f"  held-out={row.held_out_benchmark:<10}  ρ={row.r:+.3f}  95%CI={ci:<20}"
              f"  p_perm={row.p_permutation:.3f}")

    print("\n--- Train pass@1 → Test FRS (baseline, Spearman, agg=mean) ---")
    base = results[
        (results["agg_method"] == "mean") &
        (results["train_metric"] == "pass1_pct") &
        (results["test_target"] == "frs_pct") &
        (results["corr_kind"] == "spearman")
    ]
    for _, row in base.iterrows():
        ci = f"[{row.ci_lo_95:.3f}, {row.ci_hi_95:.3f}]"
        print(f"  held-out={row.held_out_benchmark:<10}  ρ={row.r:+.3f}  95%CI={ci:<20}"
              f"  p_perm={row.p_permutation:.3f}")

    print("\n--- Summary table (all combos, mean agg) ---")
    sum_display = summary[summary["agg_method"] == "mean"][
        ["train_metric", "test_target", "corr_kind", "mean_r", "std_r",
         "n_positive", "mean_ci_lo", "mean_ci_hi", "mean_p_perm"]
    ]
    print(sum_display.to_string(index=False))
    print("="*72)

    # ── Save outputs ───────────────────────────────────────────────────────────
    print(f"\n[4/5] Saving outputs ...")

    results_path = out_dir / "frs_cross_benchmark_results.csv"
    results.to_csv(results_path, index=False)
    print(f"  Saved: {results_path}")

    summary_path = out_dir / "frs_cross_benchmark_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")

    report_path = out_dir / "frs_cross_benchmark_summary.md"
    write_markdown_report(results, summary, panel, report_path)

    # ── Done ───────────────────────────────────────────────────────────────────
    print(f"\n[5/5] Done. All outputs in: {out_dir}")
    print(f"\n  Files created:")
    for f in sorted(out_dir.iterdir()):
        print(f"    {f.name}")


if __name__ == "__main__":
    main()
