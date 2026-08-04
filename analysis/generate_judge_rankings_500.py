#!/usr/bin/env python3
"""
Figure: model ranking differences across judges on the same 500 stratified samples.

Ranking is computed by:
  - per-sample score = mean of 4 pillars (faithfulness, utility, coherence, factuality)
  - per-model score = mean across that model's sampled items (from the manifest)
  - rank models by per-model score (1 = best)
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTDIR = Path(__file__).parent / "figures"
OUTDIR.mkdir(exist_ok=True)

GPT4O_FILE = Path(__file__).parent / "judge_validation" / "judge_validation_all_20260308_150844.json"
CLAUDE_FILE = Path(__file__).parent / "judge_validation" / "judge_validation_all_claude-sonnet-4-5_20260308_152120.json"

PILLARS = ["faithfulness", "utility", "coherence", "factuality"]


def _load(path: Path):
    with open(path) as f:
        return json.load(f)


def _mean_pillars(d):
    return sum(d[p] for p in PILLARS) / len(PILLARS)


def _extract_model_means(data, judge_key: str):
    """
    Returns dict: model_short_name -> mean per-sample( mean pillars ).
    judge_key: "mini" or "gpt4o" (field name in stored JSON).
    """
    per_model_vals = {}
    for combo in data["per_combo"].values():
        m = combo["short_name"]
        vals = []
        for pair in combo["sample_pairs"]:
            s = pair[judge_key]
            if all(s.get(p) is not None for p in PILLARS):
                vals.append(_mean_pillars(s))
        if not vals:
            continue
        if m not in per_model_vals:
            per_model_vals[m] = []
        per_model_vals[m].extend(vals)

    return {m: float(sum(v) / len(v)) for m, v in per_model_vals.items() if v}


def _spearman_rho(rank_a, rank_b, items):
    pos_a = {m: rank_a.index(m) for m in items}
    pos_b = {m: rank_b.index(m) for m in items}
    n = len(items)
    if n <= 1:
        return 1.0
    d_sq = sum((pos_a[m] - pos_b[m]) ** 2 for m in items)
    return 1 - 6 * d_sq / (n * (n**2 - 1))


def main():
    gpt4o_data = _load(GPT4O_FILE)
    claude_data = _load(CLAUDE_FILE)

    mini_mean = _extract_model_means(gpt4o_data, "mini")
    gpt4o_mean = _extract_model_means(gpt4o_data, "gpt4o")
    claude_mean = _extract_model_means(claude_data, "gpt4o")  # field name, but is Claude

    models = sorted(set(mini_mean) & set(gpt4o_mean) & set(claude_mean))
    if not models:
        raise RuntimeError("No overlapping models found across the three judge files.")

    rank_mini = sorted(models, key=lambda m: mini_mean[m], reverse=True)
    rank_4o = sorted(models, key=lambda m: gpt4o_mean[m], reverse=True)
    rank_claude = sorted(models, key=lambda m: claude_mean[m], reverse=True)

    rho_m4o = _spearman_rho(rank_mini, rank_4o, models)
    rho_mcl = _spearman_rho(rank_mini, rank_claude, models)
    rho_4ocl = _spearman_rho(rank_4o, rank_claude, models)

    # ── Plot ──
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "figure.dpi": 200,
        }
    )

    fig, ax = plt.subplots(figsize=(10.2, 6.2))

    x = np.array([0.0, 1.0, 2.0])
    x_labels = ["GPT-4o-mini", "GPT-4o", "Claude Sonnet 4.5"]

    # y positions are ranks (0 = best), display inverted axis
    y_mini = {m: rank_mini.index(m) for m in models}
    y_4o = {m: rank_4o.index(m) for m in models}
    y_claude = {m: rank_claude.index(m) for m in models}

    # Color map stable by mini rank (so legend is consistent)
    cmap = plt.cm.tab10(np.linspace(0, 1, max(10, len(models))))  # at least 10 distincts
    colors = {m: cmap[i % len(cmap)] for i, m in enumerate(rank_mini)}

    # Light grid for readability
    n = len(models)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlim(-0.45, 2.45)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"#{i+1}" for i in range(n)], color="#666666")
    ax.grid(axis="y", alpha=0.15, linewidth=0.6)

    # Draw lines and points
    for m in models:
        ys = [y_mini[m], y_4o[m], y_claude[m]]
        ax.plot(
            x,
            ys,
            "-",
            color=colors[m],
            linewidth=2.2,
            alpha=0.9,
            zorder=2,
        )
        ax.scatter(
            x,
            ys,
            s=70,
            color=colors[m],
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )

        # Label each model once (left side) at its mini rank row; include 3-judge means compactly
        label = f"{m}"
        ax.text(
            -0.08,
            y_mini[m],
            label,
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=colors[m],
            clip_on=False,
        )

    # Axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # Per-judge mean score annotations (numbers) next to each point.
    # Keep this minimal to avoid clutter; show 2 decimals.
    def annotate_means(x_pos, means_map, y_map):
        for m in models:
            ax.text(
                x_pos + 0.08,
                y_map[m],
                f"{means_map[m]:.2f}",
                ha="left",
                va="center",
                fontsize=9,
                color="#333333",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85),
                zorder=4,
            )

    annotate_means(0.0, mini_mean, y_mini)
    annotate_means(1.0, gpt4o_mean, y_4o)
    annotate_means(2.0, claude_mean, y_claude)

    title = "Model rankings across judges (same 500 stratified samples)"
    subtitle = (
        f"Spearman ρ: mini↔4o {rho_m4o:.2f} · mini↔Claude {rho_mcl:.2f} · 4o↔Claude {rho_4ocl:.2f}   "
        f"(ranking by mean of 4 pillars)"
    )
    ax.set_title(title + "\n" + subtitle, pad=14, fontweight="bold")

    fig.tight_layout()
    out_path = OUTDIR / "fig_judge_rankings_500.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

