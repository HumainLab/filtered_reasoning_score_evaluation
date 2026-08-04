#!/usr/bin/env python3
"""
Experiment B: ECE (expected calibration error) vs FRS.

Computes per-(model, benchmark) ECE from all pass@16 traces using the paper's
low-prob-tail confidence estimator, joins published FRS, and reports pair-level
and model-level correlations.

Usage:
  python analysis/run_exp_b_ece_vs_frs.py --repo-root .
  python analysis/run_exp_b_ece_vs_frs.py --repo-root . --smoke-test
  python analysis/run_exp_b_ece_vs_frs.py --repo-root . --reuse-ece
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR_DEFAULT = REPO_ROOT / "analysis_outputs" / "exp_b_ece_vs_frs"
FRS_CSV = REPO_ROOT / "global_pass1_frs_analysis" / "merged_pass1_frs_per_benchmark.csv"

N_BINS = 10
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 42

# FRS CSV uses CSQA; JSONL discovery uses CommonsenseQA
BENCHMARK_TO_FRS: Dict[str, str] = {
    "CommonsenseQA": "CSQA",
    "GSM8K": "GSM8K",
    "MATH500": "MATH500",
    "SVAMP": "SVAMP",
    "AQuA": "AQuA",
    "GPQA": "GPQA",
}


def setup_logger(out_dir: Path) -> logging.Logger:
    log = logging.getLogger("exp_b_ece_vs_frs")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    out_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(out_dir / "run.log", encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    return log


def benchmark_for_frs(benchmark: str) -> str:
    return BENCHMARK_TO_FRS.get(benchmark, benchmark)


def collect_traces_from_jsonl(jsonl_path: str, log: logging.Logger) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Return (confidences, corrects, n_skipped_nan, n_clipped)."""
    conf_list: List[float] = []
    correct_list: List[bool] = []
    n_skipped_nan = 0
    n_clipped = 0

    sys.path.insert(0, str(REPO_ROOT))
    from topk_ablation import compute_trace_confidence

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            scores = row.get("score", [])
            token_probs_all = row.get("chosen_token_probs_per_path", {}).get("epoch_0", [])
            if not isinstance(token_probs_all, list):
                continue
            n_traces = min(len(scores), len(token_probs_all))
            for i in range(n_traces):
                probs = token_probs_all[i]
                conf = compute_trace_confidence(probs)
                if np.isnan(conf):
                    n_skipped_nan += 1
                    continue
                if conf < 0.0 or conf > 1.0:
                    log.warning(
                        "Confidence outside [0,1]: %.4f (clipped) | %s trace %d",
                        conf,
                        jsonl_path,
                        i,
                    )
                    n_clipped += 1
                    conf = float(np.clip(conf, 0.0, 1.0))
                conf_list.append(conf)
                correct_list.append(bool(scores[i]))

    return (
        np.asarray(conf_list, dtype=np.float64),
        np.asarray(correct_list, dtype=bool),
        n_skipped_nan,
        n_clipped,
    )


def compute_ece_equal_width(
    confidences: np.ndarray,
    corrects: np.ndarray,
    n_bins: int = N_BINS,
) -> Tuple[float, List[Dict[str, Any]]]:
    """ECE with equal-width bins on [0, 1]; skip empty bins."""
    confidences = np.clip(confidences, 0.0, 1.0)
    n = len(confidences)
    if n == 0:
        return float("nan"), []

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_rows: List[Dict[str, Any]] = []

    for b in range(n_bins):
        low, high = float(edges[b]), float(edges[b + 1])
        if b < n_bins - 1:
            mask = (confidences >= low) & (confidences < high)
        else:
            mask = (confidences >= low) & (confidences <= high)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        mean_conf = float(confidences[mask].mean())
        acc = float(corrects[mask].mean())
        weight = cnt / n
        gap = abs(mean_conf - acc)
        ece += weight * gap
        bin_rows.append(
            {
                "bin_idx": b,
                "bin_low": low,
                "bin_high": high,
                "n_traces": cnt,
                "mean_conf": mean_conf,
                "accuracy": acc,
                "weight": weight,
                "calibration_gap": gap,
            }
        )

    return float(ece), bin_rows


def bootstrap_pearson_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(idx)) < 3:
            continue
        r, _ = pearsonr(x[idx], y[idx])
        if not np.isnan(r):
            rs.append(r)
    if not rs:
        return float("nan"), float("nan")
    lo, hi = np.percentile(rs, [2.5, 97.5])
    return float(lo), float(hi)


def compute_all_ece(
    jsonl_map: Dict[Tuple[str, str], str],
    log: logging.Logger,
    pairs_filter: Optional[set[Tuple[str, str]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    rows: List[Dict[str, Any]] = []
    stats = {"n_skipped_nan": 0, "n_clipped": 0, "n_pairs": 0}

    items = sorted(jsonl_map.items())
    if pairs_filter is not None:
        items = [(k, v) for k, v in items if k in pairs_filter]

    for i, ((model, benchmark), path) in enumerate(items, 1):
        t0 = time.perf_counter()
        confs, corrects, n_nan, n_clip = collect_traces_from_jsonl(path, log)
        stats["n_skipped_nan"] += n_nan
        stats["n_clipped"] += n_clip
        ece, _ = compute_ece_equal_width(confs, corrects)
        rows.append(
            {
                "model": model,
                "benchmark": benchmark,
                "benchmark_frs": benchmark_for_frs(benchmark),
                "ece": ece,
                "n_traces": len(confs),
                "mean_confidence": float(confs.mean()) if len(confs) else float("nan"),
                "accuracy": float(corrects.mean()) if len(corrects) else float("nan"),
                "n_skipped_nan": n_nan,
                "jsonl_path": path,
            }
        )
        stats["n_pairs"] += 1
        log.info(
            "[%d/%d] %s × %s | ECE=%.4f acc=%.3f mean_conf=%.3f n=%d (%.1fs)",
            i,
            len(items),
            model,
            benchmark,
            ece,
            rows[-1]["accuracy"],
            rows[-1]["mean_confidence"],
            len(confs),
            time.perf_counter() - t0,
        )

    return pd.DataFrame(rows), stats


def run_smoke_test(jsonl_map: Dict[Tuple[str, str], str], log: logging.Logger, out_dir: Path) -> None:
    key = ("DS-R1-7B", "GSM8K")
    if key not in jsonl_map:
        raise KeyError(f"Smoke-test pair {key} not found in discover_jsonl_groups()")
    path = jsonl_map[key]
    log.info("Smoke test: %s × %s | %s", key[0], key[1], path)

    confs, corrects, n_nan, n_clip = collect_traces_from_jsonl(path, log)
    ece, bin_rows = compute_ece_equal_width(confs, corrects)

    print("\n=== SMOKE TEST: DS-R1-7B / GSM8K ===")
    print(f"mean confidence: {confs.mean():.4f}")
    print(f"accuracy:        {100.0 * corrects.mean():.2f}%")
    print(f"ECE:             {ece:.4f}")
    print(f"n_traces:        {len(confs)} | skipped NaN: {n_nan} | clipped: {n_clip}")
    print("\nBin breakdown (10 equal-width bins):")
    print(f"{'bin_low':>8} {'bin_high':>8} {'n':>8} {'mean_conf':>10} {'accuracy':>10}")
    for row in bin_rows:
        print(
            f"{row['bin_low']:8.2f} {row['bin_high']:8.2f} "
            f"{row['n_traces']:8d} {row['mean_conf']:10.4f} {row['accuracy']:10.4f}"
        )

    pd.DataFrame(bin_rows).to_csv(out_dir / "smoke_test_ds_r1_7b_gsm8k_bins.csv", index=False)
    log.info("Smoke test complete — wrote smoke_test_ds_r1_7b_gsm8k_bins.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment B: ECE vs FRS")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--reuse-ece",
        action="store_true",
        help="Load analysis_outputs/exp_b_ece_vs_frs/ece_per_pair.csv instead of scanning JSONL",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run DS-R1-7B/GSM8K smoke test only (no full 54-pair run)",
    )
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    out_dir = repo / "analysis_outputs" / "exp_b_ece_vs_frs"
    log = setup_logger(out_dir)
    t_wall = time.perf_counter()

    sys.path.insert(0, str(repo))
    from build_downstream_parquets import discover_jsonl_groups

    jsonl_map = discover_jsonl_groups(str(repo))
    log.info("Discovered %d (model, benchmark) JSONL pairs", len(jsonl_map))

    if args.smoke_test:
        run_smoke_test(jsonl_map, log, out_dir)
        return

    ece_cache = out_dir / "ece_per_pair.csv"
    skip_stats = {"n_skipped_nan": 0, "n_clipped": 0}

    if args.reuse_ece and ece_cache.exists():
        ece_df = pd.read_csv(ece_cache)
        log.info("Reused ECE from %s (%d pairs)", ece_cache.name, len(ece_df))
    else:
        run_smoke_test(jsonl_map, log, out_dir)
        log.info("Smoke test passed — starting full 54-pair ECE computation")
        ece_df, skip_stats = compute_all_ece(jsonl_map, log)
        assert len(ece_df) == 54, f"expected 54 pairs, got {len(ece_df)}"
        ece_df.to_csv(ece_cache, index=False)
        log.info("Wrote %s", ece_cache)

    frs_df = pd.read_csv(FRS_CSV)
    log.info("FRS columns: %s", list(frs_df.columns))

    frs_join = frs_df.rename(columns={"benchmark": "benchmark_frs"})
    joined = ece_df.merge(frs_join, on=["model", "benchmark_frs"], how="inner")
    if joined.shape[0] != 54:
        log.warning("Joined rows = %d (expected 54)", joined.shape[0])
    joined.to_csv(out_dir / "ece_frs_joined.csv", index=False)

    x = joined["ece"].to_numpy()
    y = joined["frs_pct"].to_numpy()

    pearson_r, pearson_p = pearsonr(x, y)
    ci_lo, ci_hi = bootstrap_pearson_ci(x, y)
    spearman_rho, spearman_p = spearmanr(x, y)

    model_macro = (
        joined.groupby("model", as_index=False)
        .agg(mean_ece=("ece", "mean"), mean_frs=("frs_pct", "mean"))
    )
    model_spearman_rho, model_spearman_p = spearmanr(model_macro["mean_ece"], model_macro["mean_frs"])

    high_ece = joined[joined["ece"] > 0.3][["model", "benchmark", "ece"]]
    low_ece = joined[joined["ece"] < 0.02][["model", "benchmark", "ece"]]

    corr_rows = [
        {
            "metric": "pair_pearson_ece_frs",
            "value": float(pearson_r),
            "p_value": float(pearson_p),
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "n": len(joined),
        },
        {
            "metric": "pair_spearman_ece_frs",
            "value": float(spearman_rho),
            "p_value": float(spearman_p),
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n": len(joined),
        },
        {
            "metric": "model_spearman_ece_frs_ranking",
            "value": float(model_spearman_rho),
            "p_value": float(model_spearman_p),
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n": len(model_macro),
        },
    ]
    pd.DataFrame(corr_rows).to_csv(out_dir / "correlation_summary.csv", index=False)

    lines = [
        "# Experiment B: ECE vs FRS",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Headline",
        "",
        "| Metric | Value | p-value | 95% CI | N |",
        "|---|---:|---:|---|---:|",
        f"| Pair-level Pearson r (ECE ↔ FRS) | {pearson_r:.3f} | {pearson_p:.4f} | [{ci_lo:.3f}, {ci_hi:.3f}] | {len(joined)} |",
        f"| Pair-level Spearman ρ (ECE ↔ FRS) | {spearman_rho:.3f} | {spearman_p:.4f} | — | {len(joined)} |",
        f"| Model-level Spearman ρ (macro-avg rankings) | {model_spearman_rho:.3f} | {model_spearman_p:.4f} | — | {len(model_macro)} |",
        "",
        "## Data quality",
        "",
        f"- Traces skipped (NaN confidence): **{skip_stats.get('n_skipped_nan', 0)}**",
        f"- Confidence values clipped to [0, 1]: **{skip_stats.get('n_clipped', 0)}**",
        f"- ECE bins: {N_BINS} equal-width on [0, 1]",
        "",
    ]

    if not high_ece.empty:
        lines.append("## Outliers: ECE > 0.3")
        lines.append("")
        for _, r in high_ece.iterrows():
            lines.append(f"- {r['model']} × {r['benchmark']}: ECE = {r['ece']:.4f}")
        lines.append("")

    if not low_ece.empty:
        lines.append("## Outliers: ECE < 0.02")
        lines.append("")
        for _, r in low_ece.iterrows():
            lines.append(f"- {r['model']} × {r['benchmark']}: ECE = {r['ece']:.4f}")
        lines.append("")

    sign_word = "positive" if pearson_r > 0 else "negative"
    sig_word = "statistically significant" if pearson_p < 0.05 else "not statistically significant"
    lines.extend(
        [
            "## Interpretation",
            "",
            f"Pair-level expected calibration error (ECE) and FRS show a {sign_word} Pearson "
            f"correlation of r = {pearson_r:.3f} (p = {pearson_p:.4f}, 95% bootstrap CI "
            f"[{ci_lo:.3f}, {ci_hi:.3f}], N = {len(joined)} model×benchmark pairs). "
            f"This association is {sig_word} at α = 0.05. Model-level macro-averaged rankings "
            f"yield Spearman ρ = {model_spearman_rho:.3f} (p = {model_spearman_p:.4f}, N = {len(model_macro)} models).",
            "",
        ]
    )

    (out_dir / "key_numbers.md").write_text("\n".join(lines), encoding="utf-8")

    elapsed = time.perf_counter() - t_wall
    log.info("Wall-clock: %.1f s", elapsed)

    print("\n" + "\n".join(lines[5:]))
    print(f"\nPair-level ECE↔FRS: r = {pearson_r:.3f} (p = {pearson_p:.4f}), N = {len(joined)} pairs")
    print(f"Model-level ranking Spearman ρ: {model_spearman_rho:.3f} (n = {len(model_macro)} models)")
    print(f"Wall-clock: {elapsed:.1f} s")
    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
