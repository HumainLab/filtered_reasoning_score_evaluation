#!/usr/bin/env python3
"""Generate figures comparing all 3 judges: GPT-4o-mini (baseline), GPT-4o, Claude Sonnet 4.5."""

import json
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

OUTDIR = Path(__file__).parent / "figures"
OUTDIR.mkdir(exist_ok=True)

GPT4O_FILE = Path(__file__).parent / "judge_validation" / "judge_validation_all_20260308_150844.json"
CLAUDE_FILE = Path(__file__).parent / "judge_validation" / "judge_validation_all_claude-sonnet-4-5_20260308_152120.json"

with open(GPT4O_FILE) as f:
    gpt4o_data = json.load(f)
with open(CLAUDE_FILE) as f:
    claude_data = json.load(f)

PILLARS = ["faithfulness", "utility", "coherence", "factuality"]
PILLAR_LABELS = {"faithfulness": "Faith.", "utility": "Util.", "coherence": "Coher.", "factuality": "Fact."}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 200,
})


def pearson(x, y):
    n = len(x)
    if n < 2:
        return 0.0
    mx, my = sum(x)/n, sum(y)/n
    cov = sum((a - mx)*(b - my) for a, b in zip(x, y)) / (n - 1)
    sx = math.sqrt(sum((a - mx)**2 for a in x) / (n - 1))
    sy = math.sqrt(sum((b - my)**2 for b in y) / (n - 1))
    return cov / (sx * sy) if sx > 0 and sy > 0 else 0.0


def extract_scores(data, pillar):
    """Extract (mini_score, judge_score) pairs for a given pillar."""
    mini, judge = [], []
    for combo in data["per_combo"].values():
        for pair in combo["sample_pairs"]:
            mv = pair["mini"].get(pillar)
            gv = pair["gpt4o"].get(pillar)
            if mv is not None and gv is not None:
                mini.append(mv)
                judge.append(gv)
    return mini, judge


def agreement_metrics(x, y):
    n = len(x)
    if n < 2:
        return {}
    abs_d = [abs(a - b) for a, b in zip(x, y)]
    return {
        "r": pearson(x, y),
        "exact": sum(1 for d in abs_d if d == 0) / n,
        "w1": sum(1 for d in abs_d if d <= 1) / n,
        "mad": sum(abs_d) / n,
        "mean_diff": sum(a - b for a, b in zip(x, y)) / n,
    }


# ── Collect all pairwise scores ──
# mini vs gpt4o
mini_g, gpt4o_g = [], []
for p in PILLARS:
    m, g = extract_scores(gpt4o_data, p)
    mini_g.extend(m)
    gpt4o_g.extend(g)

# mini vs claude
mini_c, claude_c = [], []
for p in PILLARS:
    m, c = extract_scores(claude_data, p)
    mini_c.extend(m)
    claude_c.extend(c)

# gpt4o vs claude (matched by combo_key + idx + pillar)
gpt4o_matched, claude_matched = [], []
for combo_key in gpt4o_data["per_combo"]:
    if combo_key not in claude_data["per_combo"]:
        continue
    g_pairs = {p["idx"]: p["gpt4o"] for p in gpt4o_data["per_combo"][combo_key]["sample_pairs"]}
    c_pairs = {p["idx"]: p["gpt4o"] for p in claude_data["per_combo"][combo_key]["sample_pairs"]}
    for idx in g_pairs:
        if idx not in c_pairs:
            continue
        for p in PILLARS:
            gv = g_pairs[idx].get(p)
            cv = c_pairs[idx].get(p)
            if gv is not None and cv is not None:
                gpt4o_matched.append(gv)
                claude_matched.append(cv)

# Compute all pairwise metrics
pairs = {
    "4o-mini ↔ GPT-4o": agreement_metrics(mini_g, gpt4o_g),
    "4o-mini ↔ Claude": agreement_metrics(mini_c, claude_c),
    "GPT-4o ↔ Claude": agreement_metrics(gpt4o_matched, claude_matched),
}


# ═══════════════════════════════════════════════════════
# FIGURE: 3-Judge Pairwise Agreement (comprehensive)
# ═══════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Panel A: Pairwise Pearson r, Exact Match, Within ±1
ax = axes[0, 0]
pair_names = list(pairs.keys())
metrics_keys = [("r", "Pearson r"), ("exact", "Exact Match"), ("w1", "Within ±1")]
x_pos = np.arange(len(pair_names))
width = 0.25
colors_bar = ["#2196F3", "#4CAF50", "#FF9800"]

for i, (mk, ml) in enumerate(metrics_keys):
    vals = [pairs[pn][mk] for pn in pair_names]
    bars = ax.bar(x_pos + i * width, vals, width, label=ml, color=colors_bar[i], edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xticks(x_pos + width)
ax.set_xticklabels(pair_names, fontsize=10)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score")
ax.set_title("(A) Pairwise Agreement Metrics", fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.axhline(0.5, color="gray", ls="--", alpha=0.3)

# Panel B: Mean score by judge
ax = axes[0, 1]
judge_means = {}
for p in PILLARS:
    m_g, g = extract_scores(gpt4o_data, p)
    m_c, c = extract_scores(claude_data, p)
    if "GPT-4o-mini" not in judge_means:
        judge_means["GPT-4o-mini"] = {}
    if "GPT-4o" not in judge_means:
        judge_means["GPT-4o"] = {}
    if "Claude Sonnet 4.5" not in judge_means:
        judge_means["Claude Sonnet 4.5"] = {}
    judge_means["GPT-4o-mini"][p] = sum(m_g) / len(m_g)
    judge_means["GPT-4o"][p] = sum(g) / len(g)
    judge_means["Claude Sonnet 4.5"][p] = sum(c) / len(c)

x_pos = np.arange(len(PILLARS))
width = 0.25
judge_colors = {"GPT-4o-mini": "#66BB6A", "GPT-4o": "#42A5F5", "Claude Sonnet 4.5": "#AB47BC"}

for i, (judge_name, scores) in enumerate(judge_means.items()):
    vals = [scores[p] for p in PILLARS]
    bars = ax.bar(x_pos + i * width, vals, width, label=judge_name,
                  color=judge_colors[judge_name], edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x_pos + width)
ax.set_xticklabels([PILLAR_LABELS[p] for p in PILLARS], fontsize=10)
ax.set_ylim(0, 5.5)
ax.set_ylabel("Mean Score (1-5)")
ax.set_title("(B) Mean Scores by Judge & Pillar", fontweight="bold")
ax.legend(fontsize=9)

# Panel C: Bias distributions (all 3 pairs)
ax = axes[1, 0]
diffs_data = {
    "4o-mini − GPT-4o": [a - b for a, b in zip(mini_g, gpt4o_g)],
    "4o-mini − Claude": [a - b for a, b in zip(mini_c, claude_c)],
    "GPT-4o − Claude": [a - b for a, b in zip(gpt4o_matched, claude_matched)],
}
diff_colors = ["#42A5F5", "#AB47BC", "#FF7043"]
positions = [-0.25, 0, 0.25]

bins = np.arange(-4.5, 5.5, 1)
for i, (label, diffs) in enumerate(diffs_data.items()):
    counts, _ = np.histogram(diffs, bins=bins)
    pcts = counts / len(diffs) * 100
    centers = (bins[:-1] + bins[1:]) / 2
    ax.plot(centers, pcts, "o-", color=diff_colors[i], label=f"{label} (μ={np.mean(diffs):+.2f})",
            linewidth=2, markersize=5)

ax.set_xlabel("Score Difference")
ax.set_ylabel("Percentage (%)")
ax.set_title("(C) Score Difference Distributions", fontweight="bold")
ax.legend(fontsize=9, loc="upper right")
ax.axvline(0, color="gray", ls="--", alpha=0.4)

# Panel D: Model-level ranking comparison (3 judges)
ax = axes[1, 1]
mini_avgs = gpt4o_data["model_avgs_mini"]
gpt4o_avgs = gpt4o_data["model_avgs_4o"]
claude_avgs = claude_data["model_avgs_4o"]

models = sorted(set(mini_avgs.keys()) & set(gpt4o_avgs.keys()) & set(claude_avgs.keys()))
rank_mini = sorted(models, key=lambda m: mini_avgs[m], reverse=True)
rank_gpt4o = sorted(models, key=lambda m: gpt4o_avgs[m], reverse=True)
rank_claude = sorted(models, key=lambda m: claude_avgs[m], reverse=True)

n_m = len(models)
model_colors_map = plt.cm.tab10(np.linspace(0, 1, n_m))

for m in models:
    ci = models.index(m)
    pos_m = rank_mini.index(m)
    pos_g = rank_gpt4o.index(m)
    pos_c = rank_claude.index(m)
    ax.plot([0, 0.5, 1], [pos_m, pos_g, pos_c], "o-",
            color=model_colors_map[ci], linewidth=1.8, markersize=7,
            markeredgecolor="black", markeredgewidth=0.4, label=m)

ax.set_xlim(-0.3, 1.3)
ax.set_ylim(n_m - 0.5, -0.5)
ax.set_xticks([0, 0.5, 1])
ax.set_xticklabels(["GPT-4o-mini", "GPT-4o", "Claude 4.5"], fontweight="bold", fontsize=10)
ax.set_yticks(range(n_m))
ax.set_yticklabels([f"#{i+1}" for i in range(n_m)])
ax.set_title("(D) Model Rankings Across 3 Judges", fontweight="bold")
ax.legend(fontsize=7, loc="center right", bbox_to_anchor=(1.35, 0.5))
ax.grid(axis="y", alpha=0.2)

# Compute 3-way Spearman rho (average pairwise)
def spearman_rho(rank_a, rank_b, items):
    pos_a = {m: rank_a.index(m) for m in items}
    pos_b = {m: rank_b.index(m) for m in items}
    n = len(items)
    d_sq = sum((pos_a[m] - pos_b[m])**2 for m in items)
    return 1 - 6 * d_sq / (n * (n**2 - 1)) if n > 1 else 1.0

rho_mg = spearman_rho(rank_mini, rank_gpt4o, models)
rho_mc = spearman_rho(rank_mini, rank_claude, models)
rho_gc = spearman_rho(rank_gpt4o, rank_claude, models)
avg_rho = (rho_mg + rho_mc + rho_gc) / 3

ax.text(0.5, n_m + 0.3,
        f"Avg pairwise Spearman ρ = {avg_rho:.2f}\n(mini↔4o: {rho_mg:.2f}, mini↔Claude: {rho_mc:.2f}, 4o↔Claude: {rho_gc:.2f})",
        ha="center", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9))

fig.suptitle("Three-Judge Validation: GPT-4o-mini vs GPT-4o vs Claude Sonnet 4.5\n(500 Stratified Samples Across 54 Model×Dataset Combos)",
             fontweight="bold", fontsize=14, y=1.03)
fig.tight_layout()
fig.savefig(OUTDIR / "fig10_three_judge_comparison.png", bbox_inches="tight")
plt.close(fig)
print("Saved fig10_three_judge_comparison.png")


# ═══════════════════════════════════════════════════════
# Print summary table for the paper
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("  THREE-JUDGE SUMMARY FOR PAPER")
print("="*70)
print(f"\n{'Pair':25s} {'Pearson r':>10} {'Exact':>8} {'±1':>8} {'MAD':>8} {'Bias':>8}")
print("-"*70)
for pn, pm in pairs.items():
    print(f"{pn:25s} {pm['r']:>10.3f} {pm['exact']:>8.3f} {pm['w1']:>8.3f} {pm['mad']:>8.3f} {pm['mean_diff']:>+8.3f}")

print(f"\nRanking Spearman ρ:")
print(f"  mini ↔ GPT-4o:  {rho_mg:.3f}")
print(f"  mini ↔ Claude:  {rho_mc:.3f}")
print(f"  GPT-4o ↔ Claude: {rho_gc:.3f}")
print(f"  Average:         {avg_rho:.3f}")

# ICC (two-way random, average measures, consistency)
print(f"\nMean scores per judge:")
for judge_name, scores in judge_means.items():
    avg = sum(scores[p] for p in PILLARS) / len(PILLARS)
    print(f"  {judge_name:20s}: {avg:.2f}")
