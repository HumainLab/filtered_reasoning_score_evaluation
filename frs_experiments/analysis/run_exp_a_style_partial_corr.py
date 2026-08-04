#!/usr/bin/env python3
"""
Experiment A: style/length/repetition partial-correlation for selection gain.

Joins 5,400 selection-gain judged traces to pass@16 CoT text, computes style
features, aggregates to (model, benchmark), and tests whether mean top-conf
reasoning score predicts pair-level selection gain after controlling for length,
type-token ratio, and max 4-gram repetition.

Usage:
  python analysis/run_exp_a_style_partial_corr.py --repo-root .
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR_DEFAULT = REPO_ROOT / "analysis_outputs" / "exp_a_style_partial_corr"


def load_cot_for_path(jsonl_path: str, needed_qids: set[int]) -> dict[int, list[str]]:
    qid_to_traces: dict[int, list[str]] = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = int(row["idx"])
            if qid in needed_qids:
                qid_to_traces[qid] = row["code"]
    return qid_to_traces


def style_features(text: str) -> tuple[int, float, int]:
    tokens = text.split()
    n = len(tokens)
    ttr = len(set(tokens)) / max(n, 1)
    fourgrams = [tuple(tokens[i : i + 4]) for i in range(n - 3)]
    rep = max(Counter(fourgrams).values()) if fourgrams else 1
    return n, ttr, rep


def partial_correlation(y: np.ndarray, x: np.ndarray, controls: np.ndarray) -> tuple[float, float]:
    """Pearson r between residuals of y and x after regressing both on controls."""
    y_resid = y - LinearRegression().fit(controls, y).predict(controls)
    x_resid = x - LinearRegression().fit(controls, x).predict(controls)
    r, p = pearsonr(x_resid, y_resid)
    return float(r), float(p)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--reuse-features",
        action="store_true",
        help="Skip JSONL join; load analysis_outputs/exp_a_style_partial_corr/trace_level_with_features.csv",
    )
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    out_dir = repo / "analysis_outputs" / "exp_a_style_partial_corr"
    out_dir.mkdir(parents=True, exist_ok=True)

    sg_path = repo / "analysis" / "selection_gain_judge_outputs.csv"
    pair_path = repo / "analysis" / "selection_gain_pair_level.csv"
    features_cache = out_dir / "trace_level_with_features.csv"

    pair_gain = pd.read_csv(pair_path)

    if args.reuse_features and features_cache.exists():
        sg = pd.read_csv(features_cache)
        print(f"Reused {len(sg)} traces from {features_cache.name}")
    else:
        sg = pd.read_csv(sg_path)
        print(f"Loaded {len(sg)} judged traces, {len(pair_gain)} pairs")

        cot_lookup: dict[str, dict[int, list[str]]] = {}
        for path, group in sg.groupby("jsonl_path"):
            needed_qids = set(group["question_id"].astype(int))
            cot_lookup[path] = load_cot_for_path(path, needed_qids)

        def get_cot(row: pd.Series) -> str:
            return cot_lookup[row["jsonl_path"]][int(row["question_id"])][int(row["trace_id"])]

        sg["cot"] = sg.apply(get_cot, axis=1)
        sg[["length", "ttr", "max_4gram_rep"]] = sg["cot"].apply(
            lambda t: pd.Series(style_features(t), index=["length", "ttr", "max_4gram_rep"])
        )
        sg.to_csv(features_cache, index=False)

    top = (
        sg[sg["selection_type"] == "top_conf"]
        .groupby(["model", "benchmark"], as_index=False)
        .agg(
            mean_rs=("reasoning_score", "mean"),
            mean_length=("length", "mean"),
            mean_ttr=("ttr", "mean"),
            mean_rep=("max_4gram_rep", "mean"),
            n_top=("reasoning_score", "count"),
        )
    )

    merged = top.merge(
        pair_gain[["model", "benchmark", "mean_selection_gain", "n_questions"]],
        on=["model", "benchmark"],
        how="inner",
    )
    assert len(merged) == 54, f"expected 54 pairs, got {len(merged)}"

    y = merged["mean_selection_gain"].to_numpy()
    x = merged["mean_rs"].to_numpy()
    style = merged[["mean_length", "mean_ttr", "mean_rep"]].to_numpy()

    raw_r, raw_p = pearsonr(x, y)
    partial_r, partial_p = partial_correlation(y, x, style)

    # Individual style-feature partials (control only that feature)
    feature_labels = {
        "mean_length": "length",
        "mean_ttr": "TTR (lexical diversity)",
        "mean_rep": "max 4-gram repetition",
    }
    partial_rows = []
    for col, label in feature_labels.items():
        ctrl = merged[[col]].to_numpy()
        r, p = partial_correlation(y, x, ctrl)
        partial_rows.append(
            {
                "control": label,
                "control_col": col,
                "partial_r": r,
                "p_value": p,
                "delta_from_raw": r - raw_r,
            }
        )

    lines = [
        "# Experiment A: style/length partial-correlation",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Headline",
        "",
        f"- Raw FRS-proxy (mean top-conf reasoning) ↔ selection gain: **r = {raw_r:.3f}, p = {raw_p:.4f}**",
        f"- Partial (control length + TTR + max 4-gram rep jointly): **r = {partial_r:.3f}, p = {partial_p:.4f}**",
        f"- N = {len(merged)} (model, benchmark) pairs",
        "",
        "## Single-feature partials (control one covariate at a time)",
        "",
        "| Control removed | Partial r | p | Δ from raw |",
        "|---|---:|---:|---:|",
    ]
    for row in partial_rows:
        lines.append(
            f"| {row['control']} | {row['partial_r']:.3f} | {row['p_value']:.4f} | {row['delta_from_raw']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "**Read:** large |Δ| ⇒ that feature explains more of the raw FRS↔gain correlation.",
            "",
        ]
    )

    print("\n".join(lines[6:]))

    merged.to_csv(out_dir / "pair_level_merged.csv", index=False)

    summary = pd.concat(
        [
            pd.DataFrame(
                [
                    {"control": "none (raw)", "partial_r": raw_r, "p_value": raw_p, "delta_from_raw": 0.0},
                    {
                        "control": "all three jointly",
                        "partial_r": partial_r,
                        "p_value": partial_p,
                        "delta_from_raw": partial_r - raw_r,
                    },
                ]
            ),
            pd.DataFrame(partial_rows),
        ],
        ignore_index=True,
    )
    summary.to_csv(out_dir / "correlation_summary.csv", index=False)

    # Sensitivity: exclude Phi-4-Reas. (degenerate-repetition regime)
    EXCLUDE_MODEL = "Phi-4-Reas."
    sens = merged[merged["model"] != EXCLUDE_MODEL]
    if len(sens) == 48:
        sens_raw_r, sens_raw_p = pearsonr(sens["mean_rs"], sens["mean_selection_gain"])
        sens_ttr_r, sens_ttr_p = partial_correlation(
            sens["mean_selection_gain"].to_numpy(),
            sens["mean_rs"].to_numpy(),
            sens[["mean_ttr"]].to_numpy(),
        )
        lines.extend(
            [
                f"## Sensitivity: exclude `{EXCLUDE_MODEL}` (N = 48 pairs)",
                "",
                f"- Raw r = **{sens_raw_r:.3f}** (p = {sens_raw_p:.4f})",
                f"- TTR-only partial r = **{sens_ttr_r:.3f}** (p = {sens_ttr_p:.4f}, Δ from raw = {sens_ttr_r - sens_raw_r:+.3f})",
                "",
            ]
        )
        pd.DataFrame(
            [
                {
                    "subset": "all_54",
                    "n_pairs": len(merged),
                    "raw_r": raw_r,
                    "ttr_partial_r": partial_rows[1]["partial_r"],
                },
                {
                    "subset": f"exclude_{EXCLUDE_MODEL}",
                    "n_pairs": len(sens),
                    "raw_r": sens_raw_r,
                    "ttr_partial_r": sens_ttr_r,
                },
            ]
        ).to_csv(out_dir / "ttr_partial_phi4_reas_sensitivity.csv", index=False)

    (out_dir / "key_numbers.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
