#!/usr/bin/env python3
"""Pass@1 ranking vs Filtered Reasoning Score ranking — the core thesis figure."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTDIR = Path(__file__).parent / "figures"
OUTDIR.mkdir(exist_ok=True)

# Data from pdfpdf.pdf — averaged across all 6 datasets
models = {
    "DS-R1-1.5B":      {"acc": [60.1, 43.2, 65.8, 11.0, 37.9, 39.6],  "frs": [90.4, 76.1, 88.0, 79.0, 53.7, 45.0]},
    "DS-R1-7B":         {"acc": [91.5, 63.6, 91.1, 69.3, 46.2, 47.3],  "frs": [92.9, 85.3, 92.0, 93.0, 67.1, 59.0]},
    "LLaMA-3.1-8B":     {"acc": [81.3, 35.6, 85.2, 48.8, 38.4, 66.3],  "frs": [80.0, 62.3, 78.0, 65.0, 59.7, 58.0]},
    "Qwen2.5-7B":       {"acc": [90.9, 60.6, 93.6, 78.3, 35.9, 81.8],  "frs": [77.8, 78.5, 70.0, 76.0, 60.1, 50.0]},
    "Qwen2.5-Math-7B":  {"acc": [84.55,63.6, 90.8, 29.1, 24.3, 45.9],  "frs": [76.1, 77.7, 67.0, 63.0, 59.4, 50.0]},
    "Gemma-7B":         {"acc": [36.9, 18.2, 37.8, 3.5,  24.8, 27.3],  "frs": [46.4, 35.5, 46.0, 26.0, 41.2, 32.0]},
    "Phi-4":            {"acc": [93.0, 60.8, 92.3, 77.6, 29.7, 21.5],  "frs": [72.8, 65.9, 70.0, 62.0, 53.6, 46.0]},
    "Phi-4-Reasoning":  {"acc": [95.4, 74.6, 94.6, 68.1, 44.0, 36.9],  "frs": [72.4, 58.2, 65.0, 65.0, 31.4, 37.0]},
    "Qwen3-4B":         {"acc": [72.2, 49.8, 79.5, 57.5, 60.5, 69.5],  "frs": [90.2, 73.2, 86.0, 86.0, 65.0, 64.0]},
}

# Compute average across datasets
avg_acc = {m: np.mean(v["acc"]) for m, v in models.items()}
avg_frs = {m: np.mean(v["frs"]) for m, v in models.items()}

# Rank (1 = best)
rank_acc = sorted(avg_acc, key=lambda m: avg_acc[m], reverse=True)
rank_frs = sorted(avg_frs, key=lambda m: avg_frs[m], reverse=True)

n = len(rank_acc)

# ── Figure ──
# Wider canvas so rank / name / % columns do not overlap (esp. left side).
fig, ax = plt.subplots(figsize=(9.2, 5.9))
plt.rcParams.update({"font.family": "serif", "font.size": 11, "figure.dpi": 200})

# Color by magnitude of rank change
rank_changes = {}
for m in rank_acc:
    pos_a = rank_acc.index(m)
    pos_f = rank_frs.index(m)
    rank_changes[m] = pos_a - pos_f  # positive = improved in FRS

max_change = max(abs(v) for v in rank_changes.values())

# Assign colors: big movers get saturated colors, stable ones get gray
def get_color(change):
    if abs(change) <= 1:
        return "#999999"
    elif change > 0:
        return "#1B5E20"  # improved (moved up in FRS)
    else:
        return "#B71C1C"  # declined (moved down in FRS)

for m in rank_acc:
    pos_a = rank_acc.index(m)
    pos_f = rank_frs.index(m)
    change = rank_changes[m]
    color = get_color(change)
    lw = 3.0 if abs(change) >= 3 else 2.0 if abs(change) >= 2 else 1.3

    ax.plot([0, 1], [pos_a, pos_f], "o-", color=color, linewidth=lw,
            markersize=9, markeredgecolor="white", markeredgewidth=1.2, zorder=5)

    # One string per side: rank + name + % (rank fixed-width so columns stay aligned)
    left_txt = f"{pos_a + 1:>2}. {m}  ({avg_acc[m]:.1f}%)"
    ax.text(-0.06, pos_a, left_txt, ha="right", va="center",
            fontsize=8.8, fontweight="bold", color=color, zorder=6, clip_on=False)
    right_txt = f"#{pos_f + 1}  {m}  ({avg_frs[m]:.1f}%)"
    ax.text(1.06, pos_f, right_txt, ha="left", va="center",
            fontsize=8.8, fontweight="bold", color=color, zorder=6, clip_on=False)

# Axis setup — inner margin for labels; long text uses clip_on=False
ax.set_xlim(-0.48, 1.52)
ax.set_ylim(n - 0.5, -0.7)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Pass@1\n(Accuracy)", "Filtered Reasoning\nScore"],
                    fontsize=12, fontweight="bold")
ax.set_yticks(range(n))
ax.set_yticklabels([])
ax.grid(axis="y", alpha=0.15, linewidth=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.tick_params(left=False)

# Highlight biggest movers
biggest = sorted(rank_changes.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
callouts = []
for m, ch in biggest:
    direction = "▲" if ch > 0 else "▼"
    callouts.append(f"{m}: {direction}{abs(ch)} ranks")

ax.text(0.5, -0.58, "  |  ".join(callouts),
        ha="center", va="center", fontsize=8.5, fontstyle="italic", color="#444444",
        bbox=dict(boxstyle="round,pad=0.55", facecolor="#FFF9C4", edgecolor="#F9A825", alpha=0.9))

ax.set_title("Model Rankings: Pass@1 vs. Filtered Reasoning Score\nAveraged across 6 benchmarks",
             fontsize=13, fontweight="bold", pad=18)

fig.tight_layout()
fig.savefig(
    OUTDIR / "fig_ranking_shift_pass1_vs_frs.png",
    bbox_inches="tight",
    pad_inches=0.35,
    dpi=200,
)
plt.close(fig)
print("Saved fig_ranking_shift_pass1_vs_frs.png")
