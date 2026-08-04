#!/usr/bin/env python3
"""Single clean, convincing judge agreement figure for the paper."""

import json
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyArrowPatch
from pathlib import Path
from collections import defaultdict

OUTDIR = Path(__file__).parent / "figures"
GPT4O_FILE = Path(__file__).parent / "judge_validation" / "judge_validation_all_20260308_150844.json"
CLAUDE_FILE = Path(__file__).parent / "judge_validation" / "judge_validation_all_claude-sonnet-4-5_20260308_152120.json"

with open(GPT4O_FILE) as f:
    gpt4o_data = json.load(f)
with open(CLAUDE_FILE) as f:
    claude_data = json.load(f)

PILLARS = ["faithfulness", "utility", "coherence", "factuality"]


def extract_all(data):
    mini, judge = [], []
    for combo in data["per_combo"].values():
        for pair in combo["sample_pairs"]:
            for p in PILLARS:
                mv, gv = pair["mini"].get(p), pair["gpt4o"].get(p)
                if mv is not None and gv is not None:
                    mini.append(mv)
                    judge.append(gv)
    return mini, judge


mini_g, gpt4o_scores = extract_all(gpt4o_data)
mini_c, claude_scores = extract_all(claude_data)


def get_distance_pcts(x, y):
    n = len(x)
    counts = defaultdict(int)
    for a, b in zip(x, y):
        counts[abs(a - b)] += 1
    return {d: counts.get(d, 0) / n * 100 for d in range(5)}


dist_g = get_distance_pcts(mini_g, gpt4o_scores)
dist_c = get_distance_pcts(mini_c, claude_scores)

# ── Figure ──
fig, ax = plt.subplots(figsize=(9, 3.5))
plt.rcParams.update({"font.family": "serif", "font.size": 11, "figure.dpi": 200})

bar_height = 0.35
y_gpt4o = 0.6
y_claude = 0.0
colors = ["#1B5E20", "#4CAF50", "#BDBDBD"]

for y_pos, dist, accent in [
    (y_gpt4o, dist_g, "#1565C0"),
    (y_claude, dist_c, "#6A1B9A"),
]:
    exact = dist[0]
    off1 = dist[1]
    off2 = 100 - exact - off1
    w1 = exact + off1

    ax.barh(y_pos, exact, height=bar_height, color=colors[0], edgecolor="white", linewidth=0.8)
    ax.barh(y_pos, off1, height=bar_height, left=exact, color=colors[1], edgecolor="white", linewidth=0.8)
    ax.barh(y_pos, off2, height=bar_height, left=w1, color=colors[2], edgecolor="white", linewidth=0.8)

    # Labels inside bars — only exact and off-by-1 (both are wide enough)
    ax.text(exact / 2, y_pos, f"{exact:.0f}%", ha="center", va="center",
            fontsize=11, fontweight="bold", color="white")
    ax.text(exact + off1 / 2, y_pos, f"{off1:.0f}%", ha="center", va="center",
            fontsize=11, fontweight="bold", color="white")

    # "within ±1" annotation with arrow, placed below the bar to avoid clipping
    ax.annotate(f"{w1:.0f}% within ±1",
                xy=(w1, y_pos - bar_height/2 - 0.01),
                xytext=(w1 + 2, y_pos - bar_height/2 - 0.15),
                fontsize=11, fontweight="bold", color=accent,
                arrowprops=dict(arrowstyle="-", color=accent, lw=1.5),
                va="top", ha="left")

# Y-axis labels
ax.set_yticks([y_gpt4o, y_claude])
ax.set_yticklabels(["vs. GPT-4o\n(same family)", "vs. Claude Sonnet 4.5\n(cross-family)"],
                    fontsize=10, fontweight="bold")

ax.set_xlim(0, 115)
ax.set_ylim(-0.42, 1.05)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.set_xlabel("Percentage of individual scores (n = 2,000 per validator)", fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend
legend_elements = [
    Patch(facecolor=colors[0], edgecolor="white", label="Exact match (Δ = 0)"),
    Patch(facecolor=colors[1], edgecolor="white", label="Off by 1 (Δ = 1)"),
    Patch(facecolor=colors[2], edgecolor="white", label="Off by ≥ 2"),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=9,
          framealpha=0.95, edgecolor="#CCCCCC", ncol=3)

ax.set_title("GPT-4o-mini Judge Agreement with Independent Validators\n"
             "500 stratified samples  ·  9 models  ·  6 datasets  ·  4 reasoning dimensions",
             fontsize=12, fontweight="bold", pad=12)

fig.tight_layout()
fig.savefig(OUTDIR / "fig_judge_agreement_final.png", bbox_inches="tight", dpi=200)
plt.close(fig)
print("Saved fig_judge_agreement_final.png")
