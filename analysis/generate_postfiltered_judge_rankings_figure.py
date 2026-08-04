#!/usr/bin/env python3
"""
Figure: model ranking differences across 3 judges on *post-filtered* samples.

Data source:
  - analysis/judge_validation/judge_validation_postfiltered_gpt-4o_*.json
  - analysis/judge_validation/judge_validation_postfiltered_claude-sonnet-4-5_*.json

Ranking definition:
  - per-sample score = mean of 4 pillar integer scores (1–5)
  - per-model score  = mean across all post-filtered validation samples for that model
  - rank models by per-model score (1 = best)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "figures"
OUTDIR.mkdir(exist_ok=True)

PILLARS = ["faithfulness", "utility", "coherence", "factuality"]

GPT4O_FILE = HERE / "judge_validation" / "judge_validation_postfiltered_gpt-4o_20260310_015415.json"
# Use the Claude run with non-null judge scores.
CLAUDE_FILE = HERE / "judge_validation" / "judge_validation_postfiltered_claude-sonnet-4-5_20260310_020728.json"


def load(path: Path) -> Dict:
    with path.open() as f:
        return json.load(f)


def mean_pillars(scores: Dict) -> float:
    return float(sum(scores[p] for p in PILLARS) / len(PILLARS))


def collect_means(data: Dict, judge_key: str) -> Dict[str, float]:
    """
    judge_key in {"mini","judge"}.
    Returns model_short_name -> mean per-sample avg(pillars).
    """
    per_model: Dict[str, List[float]] = {}
    for combo in data["per_combo"].values():
        m = combo["short_name"]
        for pair in combo["sample_pairs"]:
            s = pair.get(judge_key) if judge_key == "judge" else pair.get("mini")
            if not s:
                continue
            if all(s.get(p) is not None for p in PILLARS):
                per_model.setdefault(m, []).append(mean_pillars(s))
    return {m: float(sum(v) / len(v)) for m, v in per_model.items() if v}


def spearman_rho(rank_a: List[str], rank_b: List[str], items: List[str]) -> float:
    pos_a = {m: rank_a.index(m) for m in items}
    pos_b = {m: rank_b.index(m) for m in items}
    n = len(items)
    if n <= 1:
        return 1.0
    d2 = sum((pos_a[m] - pos_b[m]) ** 2 for m in items)
    return 1 - 6 * d2 / (n * (n**2 - 1))


def main():
    gpt4o = load(GPT4O_FILE)
    claude = load(CLAUDE_FILE)

    mini_mean = collect_means(gpt4o, "mini")
    gpt4o_mean = collect_means(gpt4o, "judge")
    claude_mean = collect_means(claude, "judge")

    models = sorted(set(mini_mean) & set(gpt4o_mean) & set(claude_mean))
    if not models:
        raise RuntimeError("No overlapping models across the three judges.")

    rank_mini = sorted(models, key=lambda m: mini_mean[m], reverse=True)
    rank_4o = sorted(models, key=lambda m: gpt4o_mean[m], reverse=True)
    rank_cl = sorted(models, key=lambda m: claude_mean[m], reverse=True)

    rho_m4o = spearman_rho(rank_mini, rank_4o, models)
    rho_mcl = spearman_rho(rank_mini, rank_cl, models)
    rho_4ocl = spearman_rho(rank_4o, rank_cl, models)

    # ── Plot ──
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "figure.dpi": 200,
        }
    )

    fig, ax = plt.subplots(figsize=(10.2, 6.0))
    x = np.array([0.0, 1.0, 2.0])
    x_labels = ["GPT-4o-mini", "GPT-4o", "Claude Sonnet 4.5"]

    n = len(models)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlim(-0.55, 2.45)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontweight="bold")
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"#{i+1}" for i in range(n)], color="#666666")
    ax.grid(axis="y", alpha=0.15, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    y_mini = {m: rank_mini.index(m) for m in models}
    y_4o = {m: rank_4o.index(m) for m in models}
    y_cl = {m: rank_cl.index(m) for m in models}

    # stable colors: by mini rank
    cmap = plt.cm.tab10(np.linspace(0, 1, max(10, n)))
    colors = {m: cmap[i % len(cmap)] for i, m in enumerate(rank_mini)}

    for m in models:
        ys = [y_mini[m], y_4o[m], y_cl[m]]
        ax.plot(x, ys, "-", color=colors[m], linewidth=2.3, alpha=0.9, zorder=2)
        ax.scatter(x, ys, s=70, color=colors[m], edgecolor="white", linewidth=1.2, zorder=3)

        # label once on the left of mini column
        ax.text(
            -0.10,
            y_mini[m],
            m,
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=colors[m],
            clip_on=False,
        )

    title = "Model rankings across judges (post-filtered samples)"
    subtitle = (
        f"Spearman ρ: mini↔4o {rho_m4o:.2f} · mini↔Claude {rho_mcl:.2f} · 4o↔Claude {rho_4ocl:.2f}   "
        f"(rank by mean of 4 pillars, n={gpt4o.get('total_samples','?')})"
    )
    ax.set_title(title + "\n" + subtitle, pad=14, fontweight="bold")

    fig.tight_layout()
    out_path = OUTDIR / "fig_judge_rankings_postfiltered.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

