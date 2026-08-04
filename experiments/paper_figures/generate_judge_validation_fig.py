import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
VAL_DIR = BASE / "judge_validation"
OUT_DIR = BASE / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(VAL_DIR.glob("judge_validation_gsm8k_*.json"))
if not files:
    raise SystemExit("No judge_validation_gsm8k_*.json files found")
val_path = files[-1]
with val_path.open() as f:
    data = json.load(f)

pooled = data["pooled_agreement"]
rank = data.get("rank_agreement", {})

pillars = ["faithfulness", "utility", "coherence", "factuality"]
labels = ["Faithfulness", "Utility", "Coherence", "Factuality"]

pearsons = [pooled[p]["pearson_r"] for p in pillars]
mad = [pooled[p]["mean_abs_diff"] for p in pillars]
exact = [pooled[p]["exact_match_rate"] for p in pillars]
within1 = [pooled[p]["within_1_rate"] for p in pillars]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# Left: per-pillar agreement bars
ax = axes[0]
x = range(len(pillars))

ax.bar(x, pearsons, color="#4c72b0", alpha=0.85, label="Pearson r")
for i, (r, em, w1) in enumerate(zip(pearsons, exact, within1)):
    ax.text(i, r + 0.03, f"r={r:.2f}\nexact={em*100:.0f}%\n±1={w1*100:.0f}%",
            ha="center", va="bottom", fontsize=8)

ax.set_xticks(list(x))
ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_ylim(0, 1.05)
ax.set_ylabel("Agreement with GPT-4o", fontweight="bold")
ax.set_title("Pillar-wise agreement: GPT-4o-mini vs GPT-4o", fontweight="bold")
ax.grid(axis="y", alpha=0.2)

patch = matplotlib.patches.Patch(color="#4c72b0", label="Correlation (Pearson r)")
ax.legend(handles=[patch], fontsize=8, loc="lower left")

# Right: model-level scatter of average scores
ax2 = axes[1]
models_mini = data["model_avgs_mini"]
models_4o = data["model_avgs_4o"]
common = sorted(set(models_mini.keys()) & set(models_4o.keys()))

colors = {
    "DS-R1-1.5B": "#1b9e77",
    "DS-R1-7B": "#d95f02",
    "LLaMA-3.1-8B": "#7570b3",
    "Qwen2.5-7B": "#e7298a",
    "Qwen2.5-Math-7B": "#66a61e",
    "Gemma-7B": "#a6761d",
    "Phi-4": "#e41a1c",
    "Phi-4-Reasoning": "#984ea3",
    "Qwen3-4B": "#377eb8",
}

for m in common:
    xval = models_mini[m]
    yval = models_4o[m]
    ax2.scatter(xval, yval, color=colors.get(m, "#333333"), s=60, edgecolors="black", linewidths=0.5)
    ax2.text(xval + 0.01, yval + 0.01, m, fontsize=8, color=colors.get(m, "#333333"))

lo, hi = 3.0, 5.1
ax2.plot([lo, hi], [lo, hi], "k--", alpha=0.3, linewidth=1)
ax2.set_xlim(lo, hi)
ax2.set_ylim(lo, hi)
ax2.set_xlabel("Average overall score (GPT-4o-mini)", fontweight="bold")
ax2.set_ylabel("Average overall score (GPT-4o)", fontweight="bold")
ax2.set_title("Model ranking is stable (Spearman cc=%.2f)" % rank.get("spearman_rho", 0.0), fontweight="bold")
ax2.grid(True, alpha=0.2)
ax2.set_aspect("equal", adjustable="box")

fig.suptitle("GPT-4o-mini judge closely tracks GPT-4o on GSM8K", fontweight="bold")
fig.tight_layout(rect=[0, 0.02, 1, 0.96])

out_path = OUT_DIR / "fig5_judge_agreement_gpt4omini_vs_gpt4o.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print("Saved", out_path)
