#!/usr/bin/env python3
"""Generate figures for the 500-sample stratified GPT-4o judge validation."""

import json
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from pathlib import Path

RESULT_FILE = Path(__file__).parent / "judge_validation" / "judge_validation_all_20260308_150844.json"
OUTDIR = Path(__file__).parent / "figures"
OUTDIR.mkdir(exist_ok=True)

PILLARS = ["faithfulness", "utility", "coherence", "factuality"]
PILLAR_LABELS = {"faithfulness": "Faithfulness", "utility": "Utility",
                 "coherence": "Coherence", "factuality": "Factuality"}
DATASETS = ["gsm8k", "math500", "svamp", "aqua", "gpqa", "commonsense_qa"]
DS_LABELS = {"gsm8k": "GSM8K", "math500": "MATH500", "svamp": "SVAMP",
             "aqua": "AQuA", "gpqa": "GPQA", "commonsense_qa": "CSQA"}

with open(RESULT_FILE) as f:
    data = json.load(f)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 200,
})

COLORS = {
    "faithfulness": "#2196F3",
    "utility": "#FF9800",
    "coherence": "#4CAF50",
    "factuality": "#E91E63",
    "overall": "#333333",
}


def pearson(x, y):
    n = len(x)
    if n < 2:
        return 0.0
    mx, my = sum(x)/n, sum(y)/n
    cov = sum((a - mx)*(b - my) for a, b in zip(x, y)) / (n - 1)
    sx = math.sqrt(sum((a - mx)**2 for a in x) / (n - 1))
    sy = math.sqrt(sum((b - my)**2 for b in y) / (n - 1))
    return cov / (sx * sy) if sx > 0 and sy > 0 else 0.0


# ── Collect all sample pairs globally ──
all_mini, all_4o = {p: [] for p in PILLARS}, {p: [] for p in PILLARS}
for combo in data["per_combo"].values():
    for pair in combo["sample_pairs"]:
        for p in PILLARS:
            mv, gv = pair["mini"].get(p), pair["gpt4o"].get(p)
            if mv is not None and gv is not None:
                all_mini[p].append(mv)
                all_4o[p].append(gv)

# ── Per-dataset aggregates ──
ds_agg = {}
for ds in DATASETS:
    mx, gx = [], []
    for combo in data["per_combo"].values():
        if combo["dataset"] != ds:
            continue
        for pair in combo["sample_pairs"]:
            for p in PILLARS:
                mv, gv = pair["mini"].get(p), pair["gpt4o"].get(p)
                if mv is not None and gv is not None:
                    mx.append(mv)
                    gx.append(gv)
    n = len(mx)
    if n < 2:
        continue
    abs_d = [abs(a - b) for a, b in zip(mx, gx)]
    ds_agg[ds] = {
        "n": n,
        "r": pearson(mx, gx),
        "exact": sum(1 for d in abs_d if d == 0) / n,
        "w1": sum(1 for d in abs_d if d <= 1) / n,
        "mad": sum(abs_d) / n,
    }


# ═══════════════════════════════════════════════════════
# FIGURE 1: Per-pillar agreement bars (Pearson, Exact, ±1)
# ═══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
pooled = data["pooled_agreement"]
labels = list(PILLAR_LABELS.values()) + ["Overall"]
keys = PILLARS + ["overall"]
colors = [COLORS[k] for k in keys]

for ax, metric, title in zip(axes,
    ["pearson_r", "exact_match_rate", "within_1_rate"],
    ["Pearson r", "Exact Match Rate", "Within ±1 Rate"]):
    vals = [pooled[k].get(metric, 0) for k in keys]
    bars = ax.barh(range(len(labels)), vals, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title(title, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.axvline(x=0.5, color="gray", ls="--", alpha=0.4)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", va="center", fontsize=9)
    ax.invert_yaxis()

fig.suptitle("GPT-4o-mini vs GPT-4o Judge Agreement (500 Stratified Samples)", fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUTDIR / "fig5_judge_validation_pillars.png", bbox_inches="tight")
plt.close(fig)
print("Saved fig5_judge_validation_pillars.png")


# ═══════════════════════════════════════════════════════
# FIGURE 2: Per-dataset agreement heatmap
# ═══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
metrics_to_plot = [("r", "Pearson r"), ("w1", "Within ±1"), ("mad", "Mean Abs Diff")]

for ax, (metric_key, metric_label) in zip(axes, metrics_to_plot):
    matrix = []
    for ds in DATASETS:
        row = []
        for combo in data["per_combo"].values():
            pass
        row_val = ds_agg[ds][metric_key] if ds in ds_agg else 0
        matrix.append(row_val)

    # Build model×dataset matrix
    models = sorted(set(c["short_name"] for c in data["per_combo"].values()))
    mat = np.zeros((len(models), len(DATASETS)))
    for combo in data["per_combo"].values():
        mi = models.index(combo["short_name"])
        di = DATASETS.index(combo["dataset"])
        ov = combo["agreement"].get("overall", {})
        val = ov.get({"r": "pearson_r", "w1": "within_1_rate", "mad": "mean_abs_diff"}[metric_key], 0)
        if val is None:
            val = 0
        mat[mi, di] = val

    vmin, vmax = (0, 1) if metric_key != "mad" else (0, 2)
    cmap = "RdYlGn" if metric_key != "mad" else "RdYlGn_r"
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels([DS_LABELS[d] for d in DATASETS], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)
    ax.set_title(metric_label, fontweight="bold")
    for i in range(len(models)):
        for j in range(len(DATASETS)):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if (mat[i,j] < 0.3 if metric_key != "mad" else mat[i,j] > 1.5) else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle("Judge Agreement by Model × Dataset", fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUTDIR / "fig6_judge_validation_heatmap.png", bbox_inches="tight")
plt.close(fig)
print("Saved fig6_judge_validation_heatmap.png")


# ═══════════════════════════════════════════════════════
# FIGURE 3: Confusion matrix (score distribution)
# ═══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
for ax, p in zip(axes, PILLARS):
    conf = np.zeros((5, 5))
    for mv, gv in zip(all_mini[p], all_4o[p]):
        conf[int(mv)-1, int(gv)-1] += 1
    total = conf.sum()
    conf_pct = conf / total * 100

    im = ax.imshow(conf_pct, cmap="Blues", vmin=0, vmax=conf_pct.max())
    ax.set_xticks(range(5))
    ax.set_xticklabels([1,2,3,4,5])
    ax.set_yticks(range(5))
    ax.set_yticklabels([1,2,3,4,5])
    ax.set_xlabel("GPT-4o score")
    if p == "faithfulness":
        ax.set_ylabel("GPT-4o-mini score")
    ax.set_title(PILLAR_LABELS[p], fontweight="bold")
    for i in range(5):
        for j in range(5):
            count = int(conf[i,j])
            if count > 0:
                ax.text(j, i, str(count), ha="center", va="center", fontsize=7,
                        color="white" if conf_pct[i,j] > conf_pct.max()*0.6 else "black")

    # Highlight diagonal
    for i in range(5):
        rect = plt.Rectangle((i-0.5, i-0.5), 1, 1, fill=False, edgecolor="red", linewidth=1.5, linestyle="--")
        ax.add_patch(rect)

fig.suptitle("Score Confusion Matrices: GPT-4o-mini vs GPT-4o", fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUTDIR / "fig7_judge_confusion_matrices.png", bbox_inches="tight")
plt.close(fig)
print("Saved fig7_judge_confusion_matrices.png")


# ═══════════════════════════════════════════════════════
# FIGURE 4: Model-level average score scatter + ranking
# ═══════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Left: scatter of model avg scores
mini_avgs = data["model_avgs_mini"]
gpt4o_avgs = data["model_avgs_4o"]
models_common = sorted(set(mini_avgs.keys()) & set(gpt4o_avgs.keys()))

model_colors = plt.cm.Set2(np.linspace(0, 1, len(models_common)))
for i, m in enumerate(models_common):
    ax1.scatter(mini_avgs[m], gpt4o_avgs[m], s=100, color=model_colors[i],
                edgecolor="black", linewidth=0.8, zorder=5, label=m)
    ax1.annotate(m, (mini_avgs[m], gpt4o_avgs[m]),
                 textcoords="offset points", xytext=(6, 6), fontsize=8)

lo = min(min(mini_avgs.values()), min(gpt4o_avgs.values())) - 0.2
hi = max(max(mini_avgs.values()), max(gpt4o_avgs.values())) + 0.2
ax1.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="y = x")
ax1.set_xlim(lo, hi)
ax1.set_ylim(lo, hi)
ax1.set_xlabel("GPT-4o-mini Avg Reasoning Score")
ax1.set_ylabel("GPT-4o Avg Reasoning Score")
ax1.set_title("Model-Level Score Agreement", fontweight="bold")
ax1.set_aspect("equal")

rank_info = data["rank_agreement"]
ax1.text(0.05, 0.95, f"Spearman ρ = {rank_info['spearman_rho']:.2f}",
         transform=ax1.transAxes, fontsize=11, fontweight="bold",
         va="top", bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

# Right: bump chart of rankings
rank_mini = rank_info["rank_mini"]
rank_4o = rank_info["rank_4o"]
n_models = len(rank_mini)

for m in rank_mini:
    pos_m = rank_mini.index(m)
    pos_g = rank_4o.index(m)
    color = model_colors[models_common.index(m)] if m in models_common else "gray"
    ax2.plot([0, 1], [pos_m, pos_g], "o-", color=color, markersize=8,
             linewidth=2, markeredgecolor="black", markeredgewidth=0.5)
    ax2.text(-0.08, pos_m, m, ha="right", va="center", fontsize=9, color=color, fontweight="bold")
    ax2.text(1.08, pos_g, m, ha="left", va="center", fontsize=9, color=color, fontweight="bold")

ax2.set_xlim(-0.5, 1.5)
ax2.set_ylim(n_models - 0.5, -0.5)
ax2.set_xticks([0, 1])
ax2.set_xticklabels(["GPT-4o-mini\nRanking", "GPT-4o\nRanking"], fontweight="bold")
ax2.set_yticks(range(n_models))
ax2.set_yticklabels([f"#{i+1}" for i in range(n_models)])
ax2.set_title("Model Ranking Comparison", fontweight="bold")
ax2.grid(axis="y", alpha=0.2)

fig.suptitle("Model-Level Judge Agreement (500 Stratified Samples, All Datasets)",
             fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUTDIR / "fig8_judge_model_ranking.png", bbox_inches="tight")
plt.close(fig)
print("Saved fig8_judge_model_ranking.png")


# ═══════════════════════════════════════════════════════
# FIGURE 5: Systematic bias analysis
# ═══════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Left: distribution of score differences (4o_mini - 4o)
all_diffs = []
for p in PILLARS:
    diffs = [m - g for m, g in zip(all_mini[p], all_4o[p])]
    all_diffs.extend(diffs)

diff_counts = defaultdict(int)
for d in all_diffs:
    diff_counts[d] += 1

x_vals = sorted(diff_counts.keys())
y_vals = [diff_counts[x] / len(all_diffs) * 100 for x in x_vals]
bar_colors = ["#4CAF50" if x == 0 else ("#FF9800" if abs(x) == 1 else "#F44336") for x in x_vals]
ax1.bar(x_vals, y_vals, color=bar_colors, edgecolor="white", width=0.7)
ax1.set_xlabel("Score Difference (GPT-4o-mini − GPT-4o)")
ax1.set_ylabel("Percentage of Scores (%)")
ax1.set_title("Score Difference Distribution", fontweight="bold")
mean_diff = sum(all_diffs) / len(all_diffs)
ax1.axvline(mean_diff, color="red", ls="--", linewidth=1.5, label=f"Mean = +{mean_diff:.2f}")
ax1.legend(fontsize=10)

exact_pct = sum(1 for d in all_diffs if d == 0) / len(all_diffs) * 100
w1_pct = sum(1 for d in all_diffs if abs(d) <= 1) / len(all_diffs) * 100
ax1.text(0.95, 0.95,
         f"Exact: {exact_pct:.1f}%\nWithin ±1: {w1_pct:.1f}%\nn = {len(all_diffs)}",
         transform=ax1.transAxes, fontsize=10, va="top", ha="right",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))

# Right: per-dataset Pearson r comparison
ds_names = [DS_LABELS[d] for d in DATASETS]
ds_r = [ds_agg[d]["r"] for d in DATASETS]
ds_w1 = [ds_agg[d]["w1"] for d in DATASETS]

x_pos = np.arange(len(DATASETS))
width = 0.35
bars1 = ax2.bar(x_pos - width/2, ds_r, width, label="Pearson r", color="#2196F3", edgecolor="white")
bars2 = ax2.bar(x_pos + width/2, ds_w1, width, label="Within ±1", color="#4CAF50", edgecolor="white")
ax2.set_xticks(x_pos)
ax2.set_xticklabels(ds_names, rotation=30, ha="right")
ax2.set_ylabel("Score")
ax2.set_ylim(0, 1.1)
ax2.set_title("Agreement by Dataset", fontweight="bold")
ax2.legend(loc="lower right")
ax2.axhline(y=0.5, color="gray", ls="--", alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

fig.suptitle("Bias Analysis & Dataset Breakdown", fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUTDIR / "fig9_judge_bias_analysis.png", bbox_inches="tight")
plt.close(fig)
print("Saved fig9_judge_bias_analysis.png")

print("\nAll figures generated successfully!")
