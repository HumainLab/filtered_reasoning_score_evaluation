#!/usr/bin/env python3
"""
Figure: Models that look similar under Pass@1 separate clearly
when evaluated with Filtered Reasoning Score.
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

models_short = [
    "DeepSeek-1.5B", "DeepSeek-7B", "LLaMA-8B", "Qwen2.5-7B",
    "Qwen2.5-Math", "Gemma-7B", "Phi-4", "Phi-4-Reasoning", "Qwen3-4B",
]

datasets = ["GSM8K", "MATH500", "SVAMP", "AQuA", "GPQA", "CommonSense"]

acc = {
    "GSM8K":       [60.1, 91.5, 81.3, 90.9, 84.55, 36.9, 93.0, 95.4, 72.2],
    "MATH500":     [43.2, 63.6, 35.6, 60.6, 63.6,  18.2, 60.8, 74.6, 49.8],
    "SVAMP":       [65.8, 91.1, 85.2, 93.6, 90.8,  37.8, 92.3, 94.6, 79.5],
    "AQuA":        [11.0, 69.3, 48.8, 78.3, 29.1,   3.5, 77.6, 68.1, 57.5],
    "GPQA":        [37.9, 46.2, 38.4, 35.9, 24.3,  24.8, 29.7, 44.0, 60.5],
    "CommonSense": [39.6, 47.3, 66.3, 81.8, 45.9,  27.3, 21.5, 36.9, 69.5],
}

filt = {
    "GSM8K":       [90.4, 92.9, 80.0, 77.8, 76.1, 46.4, 72.8, 72.4, 90.2],
    "MATH500":     [76.1, 85.3, 62.3, 78.5, 77.7, 35.5, 65.9, 58.2, 73.2],
    "SVAMP":       [88.0, 92.0, 78.0, 70.0, 67.0, 46.0, 70.0, 65.0, 86.0],
    "AQuA":        [79.0, 93.0, 65.0, 76.0, 63.0, 26.0, 62.0, 65.0, 86.0],
    "GPQA":        [53.7, 67.1, 59.7, 60.1, 59.4, 41.2, 53.6, 31.4, 65.0],
    "CommonSense": [45.0, 59.0, 58.0, 50.0, 50.0, 32.0, 46.0, 37.0, 64.0],
}

colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#17becf",
]

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 18), facecolor="white")

fig.text(0.5, 0.975,
         "Models That Look Similar Under Pass@1  Separate Clearly Under Filtered Reasoning Score",
         fontsize=22, fontweight="bold", ha="center", va="top", color="#1a1a1a")

# Shared legend at top
leg_circle = plt.Line2D([0], [0], marker="o", color="gray", markerfacecolor="gray",
                         markersize=11, linestyle="None", label="Pass@1 Accuracy")
leg_diamond = plt.Line2D([0], [0], marker="D", color="gray", markerfacecolor="gray",
                          markersize=11, linestyle="None", label="Filtered Reasoning Score")
fig.legend(handles=[leg_circle, leg_diamond], loc="upper center",
           bbox_to_anchor=(0.5, 0.955), ncol=2, fontsize=13,
           frameon=True, edgecolor="#cccccc", fancybox=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Panel A: SVAMP dumbbell (most dramatic)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax_a = fig.add_axes([0.055, 0.55, 0.43, 0.36])

a_svamp = np.array(acc["SVAMP"])
f_svamp = np.array(filt["SVAMP"])
order_s = np.argsort(a_svamp)

for i, idx in enumerate(order_s):
    ax_a.plot([a_svamp[idx], f_svamp[idx]], [i, i],
              color=colors[idx], linewidth=2.5, alpha=0.45, zorder=1)
    ax_a.scatter(a_svamp[idx], i, color=colors[idx], s=160, zorder=3,
                 edgecolors="black", linewidth=0.7, marker="o")
    ax_a.scatter(f_svamp[idx], i, color=colors[idx], s=160, zorder=3,
                 edgecolors="black", linewidth=0.7, marker="D")
    label_x = max(a_svamp[idx], f_svamp[idx]) + 1.5
    ax_a.text(label_x, i, models_short[idx], fontsize=9, va="center",
              color=colors[idx], fontweight="bold")

ax_a.set_yticks([])
ax_a.set_xlabel("Score (%)", fontsize=12, fontweight="bold")
ax_a.set_title("A.  SVAMP", fontsize=15, fontweight="bold", loc="left", pad=8)
ax_a.set_xlim(28, 115)
ax_a.grid(axis="x", alpha=0.2)
ax_a.spines["top"].set_visible(False)
ax_a.spines["right"].set_visible(False)

# Cluster highlight: models with acc >= 89%
cluster_s = [i for i in range(9) if a_svamp[i] >= 89]
positions_s = sorted([list(order_s).index(i) for i in cluster_s])
cy_min_s, cy_max_s = positions_s[0] - 0.45, positions_s[-1] + 0.45

rect_s = FancyBboxPatch((88, cy_min_s), 9, cy_max_s - cy_min_s,
                         boxstyle="round,pad=0.2", facecolor="#ff000008",
                         edgecolor="#cc0000", linewidth=1.8, linestyle="--", zorder=0)
ax_a.add_patch(rect_s)

# Cluster annotation (right side)
ax_a.annotate(
    "5 models within\n4pp on Pass@1",
    xy=(94.6, cy_max_s), xytext=(102, cy_max_s + 0.2),
    fontsize=10, color="#cc0000", fontweight="bold", ha="center",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cc0000", alpha=0.95),
    arrowprops=dict(arrowstyle="-[, widthB=3.0, lengthB=0.5", color="#cc0000", lw=1.5),
)

# Spread arrow for Filtered Reasoning
filt_cluster_s = [f_svamp[i] for i in cluster_s]
fmin_s, fmax_s = min(filt_cluster_s), max(filt_cluster_s)
arr_y_s = cy_min_s - 0.8
ax_a.annotate("", xy=(fmin_s, arr_y_s), xytext=(fmax_s, arr_y_s),
              arrowprops=dict(arrowstyle="<->", color="#006600", lw=2.5))
ax_a.text((fmin_s + fmax_s) / 2, arr_y_s - 0.6,
          f"{fmax_s - fmin_s:.0f}pp spread in Filtered Reasoning",
          fontsize=10.5, ha="center", color="#006600", fontweight="bold",
          bbox=dict(boxstyle="round,pad=0.3", fc="#e8ffe8", ec="#006600", alpha=0.95))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Panel B: GSM8K dumbbell
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax_b = fig.add_axes([0.545, 0.55, 0.43, 0.36])

a_gsm = np.array(acc["GSM8K"])
f_gsm = np.array(filt["GSM8K"])
order_g = np.argsort(a_gsm)

for i, idx in enumerate(order_g):
    ax_b.plot([a_gsm[idx], f_gsm[idx]], [i, i],
              color=colors[idx], linewidth=2.5, alpha=0.45, zorder=1)
    ax_b.scatter(a_gsm[idx], i, color=colors[idx], s=160, zorder=3,
                 edgecolors="black", linewidth=0.7, marker="o")
    ax_b.scatter(f_gsm[idx], i, color=colors[idx], s=160, zorder=3,
                 edgecolors="black", linewidth=0.7, marker="D")
    label_x = max(a_gsm[idx], f_gsm[idx]) + 1.5
    ax_b.text(label_x, i, models_short[idx], fontsize=9, va="center",
              color=colors[idx], fontweight="bold")

ax_b.set_yticks([])
ax_b.set_xlabel("Score (%)", fontsize=12, fontweight="bold")
ax_b.set_title("B.  GSM8K", fontsize=15, fontweight="bold", loc="left", pad=8)
ax_b.set_xlim(28, 115)
ax_b.grid(axis="x", alpha=0.2)
ax_b.spines["top"].set_visible(False)
ax_b.spines["right"].set_visible(False)

cluster_g = [i for i in range(9) if a_gsm[i] >= 84]
positions_g = sorted([list(order_g).index(i) for i in cluster_g])
cy_min_g, cy_max_g = positions_g[0] - 0.45, positions_g[-1] + 0.45

rect_g = FancyBboxPatch((83, cy_min_g), 14, cy_max_g - cy_min_g,
                         boxstyle="round,pad=0.2", facecolor="#ff000008",
                         edgecolor="#cc0000", linewidth=1.8, linestyle="--", zorder=0)
ax_b.add_patch(rect_g)

ax_b.annotate(
    "5 models within\n11pp on Pass@1",
    xy=(95.4, cy_max_g), xytext=(103, cy_max_g + 0.2),
    fontsize=10, color="#cc0000", fontweight="bold", ha="center",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cc0000", alpha=0.95),
    arrowprops=dict(arrowstyle="-[, widthB=3.0, lengthB=0.5", color="#cc0000", lw=1.5),
)

filt_cluster_g = [f_gsm[i] for i in cluster_g]
fmin_g, fmax_g = min(filt_cluster_g), max(filt_cluster_g)
arr_y_g = cy_min_g - 0.8
ax_b.annotate("", xy=(fmin_g, arr_y_g), xytext=(fmax_g, arr_y_g),
              arrowprops=dict(arrowstyle="<->", color="#006600", lw=2.5))
ax_b.text((fmin_g + fmax_g) / 2, arr_y_g - 0.6,
          f"{fmax_g - fmin_g:.0f}pp spread in Filtered Reasoning",
          fontsize=10.5, ha="center", color="#006600", fontweight="bold",
          bbox=dict(boxstyle="round,pad=0.3", fc="#e8ffe8", ec="#006600", alpha=0.95))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Panel C: Variability (std dev among top-5 models)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax_c = fig.add_axes([0.055, 0.065, 0.40, 0.38])

std_acc5, std_filt5 = [], []
for ds_name in datasets:
    a_arr = np.array(acc[ds_name])
    f_arr = np.array(filt[ds_name])
    top5 = np.argsort(a_arr)[-5:]
    std_acc5.append(np.std(a_arr[top5]))
    std_filt5.append(np.std(f_arr[top5]))

x_bar = np.arange(len(datasets))
w = 0.33

b1 = ax_c.bar(x_bar - w/2, std_acc5, w, label="Pass@1 Accuracy",
              color="#4393c3", edgecolor="white", linewidth=0.8, alpha=0.88)
b2 = ax_c.bar(x_bar + w/2, std_filt5, w, label="Filtered Reasoning",
              color="#d6604d", edgecolor="white", linewidth=0.8, alpha=0.88)

for bar in b1:
    h = bar.get_height()
    ax_c.text(bar.get_x() + bar.get_width()/2., h + 0.2, f'{h:.1f}',
              ha='center', va='bottom', fontsize=9, color='#3070a0', fontweight='bold')
for bar in b2:
    h = bar.get_height()
    ax_c.text(bar.get_x() + bar.get_width()/2., h + 0.2, f'{h:.1f}',
              ha='center', va='bottom', fontsize=9, color='#b04030', fontweight='bold')

max_h = max(max(std_acc5), max(std_filt5))
for i in range(len(datasets)):
    if std_acc5[i] > 0.01:
        ratio = std_filt5[i] / std_acc5[i]
        badge_color = "#006600" if ratio > 1.0 else "#888888"
        badge_bg = "#e8ffe8" if ratio > 1.0 else "#f0f0f0"
        ax_c.text(x_bar[i], max(std_acc5[i], std_filt5[i]) + 1.8,
                  f"{ratio:.1f}×", fontsize=11, ha="center", fontweight="bold",
                  color=badge_color,
                  bbox=dict(boxstyle="round,pad=0.2", fc=badge_bg, ec=badge_color, alpha=0.85))

ax_c.set_xticks(x_bar)
ax_c.set_xticklabels(datasets, fontsize=11, fontweight="bold")
ax_c.set_ylabel("Std Dev Among Top-5 Models (pp)", fontsize=11.5, fontweight="bold")
ax_c.set_title("C.  Filtered Reasoning Is More Discriminative\n"
               "     (higher variability among competitive models)",
               fontsize=14, fontweight="bold", loc="left", pad=8)
ax_c.legend(fontsize=11, loc="upper left", framealpha=0.9, edgecolor="#cccccc")
ax_c.grid(axis="y", alpha=0.15)
ax_c.set_ylim(0, max_h + 4.5)
ax_c.spines["top"].set_visible(False)
ax_c.spines["right"].set_visible(False)

# Summary annotation
n_higher = sum(1 for i in range(len(datasets)) if std_filt5[i] > std_acc5[i])
ax_c.text(0.98, 0.05,
          f"Filtered Reasoning more\ndiscriminative in {n_higher}/6 datasets",
          transform=ax_c.transAxes, fontsize=10.5, ha="right", va="bottom",
          style="italic", color="#333333",
          bbox=dict(boxstyle="round,pad=0.4", fc="#fffff0", ec="#999999", alpha=0.9))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Panel D: Rank-change slope chart
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax_d = fig.add_axes([0.545, 0.065, 0.43, 0.38])

avg_acc = np.mean([acc[d] for d in datasets], axis=0)
avg_filt = np.mean([filt[d] for d in datasets], axis=0)

rank_acc = np.argsort(np.argsort(-avg_acc)) + 1
rank_filt = np.argsort(np.argsort(-avg_filt)) + 1

for idx in range(len(models_short)):
    change = abs(rank_acc[idx] - rank_filt[idx])
    lw = 3.0 if change >= 3 else 2.0 if change >= 1 else 1.2
    alpha = 0.8 if change >= 3 else 0.55

    ax_d.plot([0, 1], [rank_acc[idx], rank_filt[idx]], color=colors[idx],
              linewidth=lw, alpha=alpha, zorder=2)
    ax_d.scatter(0, rank_acc[idx], color=colors[idx], s=200, zorder=3,
                 edgecolors="black", linewidth=0.8, marker="o")
    ax_d.scatter(1, rank_filt[idx], color=colors[idx], s=200, zorder=3,
                 edgecolors="black", linewidth=0.8, marker="D")

    ax_d.text(-0.07, rank_acc[idx], models_short[idx], fontsize=9, ha="right",
              va="center", color=colors[idx], fontweight="bold")
    ax_d.text(1.07, rank_filt[idx], models_short[idx], fontsize=9, ha="left",
              va="center", color=colors[idx], fontweight="bold")

ax_d.set_xlim(-0.48, 1.48)
ax_d.set_ylim(10, 0)
ax_d.set_xticks([0, 1])
ax_d.set_xticklabels(["Pass@1\nRank", "Filtered Reasoning\nRank"],
                      fontsize=13, fontweight="bold")
ax_d.set_yticks(range(1, 10))
ax_d.set_yticklabels([f"#{i}" for i in range(1, 10)], fontsize=10, color="#555555")
ax_d.set_title("D.  Model Rankings Shift Dramatically\n"
               "     (averaged across all 6 datasets)",
               fontsize=14, fontweight="bold", loc="left", pad=8)
ax_d.grid(axis="y", alpha=0.1)
ax_d.axvline(x=0, color="#dddddd", linewidth=0.8, zorder=0)
ax_d.axvline(x=1, color="#dddddd", linewidth=0.8, zorder=0)
ax_d.spines["top"].set_visible(False)
ax_d.spines["right"].set_visible(False)
ax_d.spines["bottom"].set_visible(False)

n_cross = sum(1 for i in range(9) for j in range(i+1, 9)
              if (rank_acc[i] - rank_acc[j]) * (rank_filt[i] - rank_filt[j]) < 0)
ax_d.text(0.5, 9.7,
          f"{n_cross} rank inversions  →  same accuracy ≠ same reasoning quality",
          fontsize=10.5, ha="center", va="center", style="italic", color="#333333",
          bbox=dict(boxstyle="round,pad=0.4", fc="#f8f8f8", ec="#aaaaaa"))

# ── Save ──────────────────────────────────────────────────────────────────────
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "filtered_reasoning_figure.png")
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
plt.close()
