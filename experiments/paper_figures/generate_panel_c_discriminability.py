#!/usr/bin/env python3
"""
Standalone regeneration of Panel C:
  "Filtered Reasoning Is More Discriminative (higher variability among competitive models)"

Source of truth: `experiments/all-result-pdf-data/generate_figure.py` (Panel C).
This script regenerates Panel C directly from the same hard-coded result arrays.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def main() -> None:
    outdir = Path(__file__).parent / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    datasets = ["GSM8K", "MATH500", "SVAMP", "AQuA", "GPQA", "CommonSense"]

    # Pass@1 accuracy (percent)
    acc = {
        "GSM8K":       [60.1, 91.5, 81.3, 90.9, 84.55, 36.9, 93.0, 95.4, 72.2],
        "MATH500":     [43.2, 63.6, 35.6, 60.6, 63.6,  18.2, 60.8, 74.6, 49.8],
        "SVAMP":       [65.8, 91.1, 85.2, 93.6, 90.8,  37.8, 92.3, 94.6, 79.5],
        "AQuA":        [11.0, 69.3, 48.8, 78.3, 29.1,   3.5, 77.6, 68.1, 57.5],
        "GPQA":        [37.9, 46.2, 38.4, 35.9, 24.3,  24.8, 29.7, 44.0, 60.5],
        "CommonSense": [39.6, 47.3, 66.3, 81.8, 45.9,  27.3, 21.5, 36.9, 69.5],
    }

    # Filtered Reasoning Score (percent)
    filt = {
        "GSM8K":       [90.4, 92.9, 80.0, 77.8, 76.1, 46.4, 72.8, 72.4, 90.2],
        "MATH500":     [76.1, 85.3, 62.3, 78.5, 77.7, 35.5, 65.9, 58.2, 73.2],
        "SVAMP":       [88.0, 92.0, 78.0, 70.0, 67.0, 46.0, 70.0, 65.0, 86.0],
        "AQuA":        [79.0, 93.0, 65.0, 76.0, 63.0, 26.0, 62.0, 65.0, 86.0],
        "GPQA":        [53.7, 67.1, 59.7, 60.1, 59.4, 41.2, 53.6, 31.4, 65.0],
        "CommonSense": [45.0, 59.0, 58.0, 50.0, 50.0, 32.0, 46.0, 37.0, 64.0],
    }

    # Compute: std dev among top-5 models by pass@1 (per dataset)
    std_acc5, std_filt5 = [], []
    for ds in datasets:
        a_arr = np.array(acc[ds], dtype=float)
        f_arr = np.array(filt[ds], dtype=float)
        top5 = np.argsort(a_arr)[-5:]
        std_acc5.append(np.std(a_arr[top5]))
        std_filt5.append(np.std(f_arr[top5]))

    # ---- Plot (match Panel C styling) ----
    fig, ax = plt.subplots(figsize=(7.1, 5.2), facecolor="white")

    x = np.arange(len(datasets))
    w = 0.33

    b1 = ax.bar(
        x - w / 2,
        std_acc5,
        w,
        label="Pass@1 Accuracy",
        color="#4393c3",
        edgecolor="white",
        linewidth=0.8,
        alpha=0.88,
    )
    b2 = ax.bar(
        x + w / 2,
        std_filt5,
        w,
        label="Filtered Reasoning",
        color="#d6604d",
        edgecolor="white",
        linewidth=0.8,
        alpha=0.88,
    )

    for bar in b1:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.2,
            f"{h:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#3070a0",
            fontweight="bold",
        )
    for bar in b2:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.2,
            f"{h:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#b04030",
            fontweight="bold",
        )

    max_h = max(max(std_acc5), max(std_filt5))
    for i in range(len(datasets)):
        if std_acc5[i] > 0.01:
            ratio = std_filt5[i] / std_acc5[i]
            badge_color = "#006600" if ratio > 1.0 else "#888888"
            badge_bg = "#e8ffe8" if ratio > 1.0 else "#f0f0f0"
            ax.text(
                x[i],
                max(std_acc5[i], std_filt5[i]) + 1.8,
                f"{ratio:.1f}×",
                fontsize=11,
                ha="center",
                fontweight="bold",
                color=badge_color,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    fc=badge_bg,
                    ec=badge_color,
                    alpha=0.85,
                ),
            )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11, fontweight="bold")
    ax.set_ylabel("Std Dev Among Top-5 Models (pp)", fontsize=11.5, fontweight="bold")
    ax.set_title(
        "Filtered Reasoning Is More Discriminative\n"
        "(higher variability among competitive models)",
        fontsize=14,
        fontweight="bold",
        loc="left",
        pad=8,
    )
    ax.legend(fontsize=11, loc="upper left", framealpha=0.9, edgecolor="#cccccc")
    ax.grid(axis="y", alpha=0.15)
    ax.set_ylim(0, max_h + 4.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    n_higher = sum(1 for i in range(len(datasets)) if std_filt5[i] > std_acc5[i])
    ax.text(
        0.98,
        0.05,
        f"Filtered Reasoning more\ndiscriminative in {n_higher}/6 datasets",
        transform=ax.transAxes,
        fontsize=10.5,
        ha="right",
        va="bottom",
        style="italic",
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fffff0", ec="#999999", alpha=0.9),
    )

    # Save as standalone panel
    out_png = outdir / "panel_c_filtered_reasoning_discriminative_regenerated.png"
    out_pdf = outdir / "panel_c_filtered_reasoning_discriminative_regenerated.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()

