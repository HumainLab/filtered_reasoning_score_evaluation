import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.pad_inches': 0.15,
})

models = [
    'DS-R1-1.5B',
    'DS-R1-7B',
    'LLaMA-3.1-8B',
    'Qwen2.5-7B',
    'Qwen2.5-Math-7B',
    'Gemma-7B',
    'Phi-4',
    'Phi-4-Reasoning',
    'Qwen3-4B',
]

datasets = ['GSM8K', 'MATH500', 'SVAMP', 'AQuA', 'GPQA', 'CommonSense']

acc = {
    'DS-R1-1.5B':       [60.1,  43.2,  65.8,  11.0,  37.9, 39.6],
    'DS-R1-7B':         [91.5,  63.6,  91.1,  69.3,  46.2, 47.3],
    'LLaMA-3.1-8B':     [81.3,  35.6,  85.2,  48.8,  38.4, 66.3],
    'Qwen2.5-7B':       [90.9,  60.6,  93.6,  78.3,  35.9, 81.8],
    'Qwen2.5-Math-7B':  [84.55, 63.6,  90.8,  29.1,  24.3, 45.9],
    'Gemma-7B':         [36.9,  18.2,  37.8,   3.5,  24.8, 27.3],
    'Phi-4':            [93.0,  60.8,  92.3,  77.6,  29.7, 21.5],
    'Phi-4-Reasoning':  [95.4,  74.6,  94.6,  68.1,  44.0, 36.9],
    'Qwen3-4B':         [72.2,  49.8,  79.5,  57.5,  60.5, 69.5],
}

reasoning = {
    'DS-R1-1.5B':       [66.79, 61.72, 59.74, 52.31, 53.29, 52.65],
    'DS-R1-7B':         [88.80, 75.88, 85.63, 78.37, 65.47, 66.78],
    'LLaMA-3.1-8B':     [78.01, 53.66, 77.39, 58.12, 61.24, 65.08],
    'Qwen2.5-7B':       [82.98, 71.43, 78.16, 77.85, 58.92, 67.53],
    'Qwen2.5-Math-7B':  [87.47, 80.53, 90.88, 79.94, 52.67, 63.24],
    'Gemma-7B':         [53.39, 41.45, 52.26, 41.87, 55.11, 30.78],
    'Phi-4':            [81.82, 73.08, 79.71, 78.12, 51.54, 50.59],
    'Phi-4-Reasoning':  [90.79, 78.43, 87.18, 81.12, 50.97, 38.70],
    'Qwen3-4B':         [78.87, 70.09, 75.94, 76.43, 55.86, 64.27],
}

filtered_reasoning = {
    'DS-R1-1.5B':       [90.4, 76.1, 88.0, 79.0, 53.7, 45.0],
    'DS-R1-7B':         [92.9, 85.3, 92.0, 93.0, 67.1, 59.0],
    'LLaMA-3.1-8B':     [80.0, 62.3, 78.0, 65.0, 59.7, 58.0],
    'Qwen2.5-7B':       [77.8, 78.5, 70.0, 76.0, 60.1, 50.0],
    'Qwen2.5-Math-7B':  [76.1, 77.7, 67.0, 63.0, 59.4, 50.0],
    'Gemma-7B':         [46.4, 35.5, 46.0, 26.0, 41.2, 32.0],
    'Phi-4':            [72.8, 65.9, 70.0, 62.0, 53.6, 46.0],
    'Phi-4-Reasoning':  [72.4, 58.2, 65.0, 65.0, 31.4, 37.0],
    'Qwen3-4B':         [90.2, 73.2, 86.0, 86.0, 65.0, 64.0],
}

filtered_pass1 = {
    'DS-R1-1.5B':       [90.5, 68.4, 88.6, 75.0, 33.0, 40.6],
    'DS-R1-7B':         [96.3, 77.0, 95.4, 83.8, 55.9, 63.2],
    'LLaMA-3.1-8B':     [82.9, 50.0, 86.9, 60.0, 39.3, 65.4],
    'Qwen2.5-7B':       [62.8, 48.0, 68.0, 74.0, 37.7, 74.1],
    'Qwen2.5-Math-7B':  [43.4, 59.7, 32.5, 48.0, 41.9, 50.4],
    'Gemma-7B':         [28.2, 11.4, 34.0, 15.0, 11.4, 17.7],
    'Phi-4':            [63.8, 56.0, 54.5, 56.3, 34.3, 72.0],
    'Phi-4-Reasoning':  [93.6, 77.6, 93.0, 81.1, 40.2, 70.0],
    'Qwen3-4B':         [90.8, 58.8, 87.4, 57.1, 87.0, 90.6],
}

snr = {
    'DS-R1-1.5B':       [-1.70, 0.58, -1.90, -1.23, -3.65, -1.23],
    'DS-R1-7B':         [3.80, 3.69, -0.79, 2.15, 0.76, -0.12],
    'LLaMA-3.1-8B':     [0.55, 0.69, 0.57, 0.57, 1.25, 0.22],
    'Qwen2.5-7B':       [0.72, 1.05, 0.84, 1.56, 1.12, 0.33],
    'Qwen2.5-Math-7B':  [0.28, -0.18, -0.76, -1.47, -3.79, -0.11],
    'Gemma-7B':         [-4.84, -1.60, -3.46, -4.02, -0.13, -0.17],
    'Phi-4':            [-0.79, -4.46, -2.42, -3.71, -0.05, 1.29],
    'Phi-4-Reasoning':  [0.74, -0.04, -0.04, -0.04, 0.86, 2.15],
    'Qwen3-4B':         [3.91, 1.01, 4.14, 0.03, -0.91, 4.31],
}

COLORS = {
    'DS-R1-1.5B':      '#1b9e77',
    'DS-R1-7B':        '#d95f02',
    'LLaMA-3.1-8B':    '#7570b3',
    'Qwen2.5-7B':      '#e7298a',
    'Qwen2.5-Math-7B': '#66a61e',
    'Gemma-7B':        '#a6761d',
    'Phi-4':           '#e41a1c',
    'Phi-4-Reasoning': '#984ea3',
    'Qwen3-4B':        '#377eb8',
}

MARKERS = {
    'DS-R1-1.5B':      'v',
    'DS-R1-7B':        'D',
    'LLaMA-3.1-8B':    's',
    'Qwen2.5-7B':      'P',
    'Qwen2.5-Math-7B': 'X',
    'Gemma-7B':        'd',
    'Phi-4':           'o',
    'Phi-4-Reasoning': '^',
    'Qwen3-4B':        '*',
}

import os
outdir = os.environ.get(
    "FRS_FIGURES_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"),
)
os.makedirs(outdir, exist_ok=True)


# ============================================================
# FIGURE 1: Scatter — Pass@1 vs Filtered Reasoning (per dataset, averaged)
# ============================================================
def fig1_scatter():
    fig, ax = plt.subplots(figsize=(8, 7))

    for m in models:
        avg_acc = np.mean(acc[m])
        avg_fr = np.mean(filtered_reasoning[m])
        ax.scatter(avg_acc, avg_fr, color=COLORS[m], marker=MARKERS[m],
                   s=140, zorder=5, edgecolors='black', linewidths=0.5)

    lo, hi = 15, 100
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.3, linewidth=1, label='y = x')

    offsets = {
        'DS-R1-1.5B':      (5, 6),
        'DS-R1-7B':        (5, -2),
        'LLaMA-3.1-8B':    (-2, 8),
        'Qwen2.5-7B':      (5, -8),
        'Qwen2.5-Math-7B': (5, 4),
        'Gemma-7B':        (5, -2),
        'Phi-4':           (-2, -12),
        'Phi-4-Reasoning':  (5, -2),
        'Qwen3-4B':        (5, 5),
    }
    for m in models:
        avg_acc = np.mean(acc[m])
        avg_fr = np.mean(filtered_reasoning[m])
        ox, oy = offsets.get(m, (5, 0))
        ax.annotate(m, (avg_acc, avg_fr), textcoords='offset points',
                    xytext=(ox, oy), fontsize=8.5, color=COLORS[m], fontweight='bold')

    ax.fill_between([lo, hi], [lo, hi], [hi, hi], alpha=0.04, color='green')
    ax.fill_between([lo, hi], [lo, lo], [lo, hi], alpha=0.04, color='red')

    ax.text(30, 82, 'Filtered Reasoning > Pass@1\n(Underrated by accuracy)',
            fontsize=8, color='#2d7d2d', alpha=0.7, style='italic')
    ax.text(62, 32, 'Filtered Reasoning < Pass@1\n(Overrated by accuracy)',
            fontsize=8, color='#aa2222', alpha=0.7, style='italic')

    ax.set_xlabel('Average Pass@1 Accuracy (%)', fontweight='bold')
    ax.set_ylabel('Average Filtered Reasoning Score (%)', fontweight='bold')
    ax.set_title('The Illusion of Pass@1:\nAccuracy vs. Reasoning Quality Under Confidence', fontweight='bold')
    ax.set_xlim(20, 80)
    ax.set_ylim(30, 85)
    ax.grid(True, alpha=0.15)
    ax.set_aspect('equal')

    fig.savefig(f'{outdir}/fig1_scatter_pass1_vs_filtered_reasoning.png', bbox_inches='tight')
    plt.close(fig)
    print('Figure 1 saved.')


# ============================================================
# FIGURE 2: Bump chart — Ranking reversal on GSM8K
# ============================================================
def fig2_bump():
    gsm_acc = {m: acc[m][0] for m in models}
    gsm_fr = {m: filtered_reasoning[m][0] for m in models}

    rank_acc = sorted(models, key=lambda m: gsm_acc[m], reverse=True)
    rank_fr = sorted(models, key=lambda m: gsm_fr[m], reverse=True)

    pos_acc = {m: i for i, m in enumerate(rank_acc)}
    pos_fr = {m: i for i, m in enumerate(rank_fr)}

    fig, ax = plt.subplots(figsize=(10, 7))

    n = len(models)
    x_left = 0.0
    x_right = 1.0

    for m in models:
        y_left = n - 1 - pos_acc[m]
        y_right = n - 1 - pos_fr[m]
        change = pos_acc[m] - pos_fr[m]

        if change > 0:
            color = '#2ca02c'
            alpha = 0.7
        elif change < 0:
            color = '#d62728'
            alpha = 0.7
        else:
            color = '#888888'
            alpha = 0.5

        ax.plot([x_left, x_right], [y_left, y_right],
                color=COLORS[m], linewidth=2.5, alpha=0.8, zorder=3)

        acc_val = gsm_acc[m]
        fr_val = gsm_fr[m]

        ax.scatter(x_left, y_left, color=COLORS[m], s=100, zorder=5,
                   edgecolors='black', linewidths=0.5, marker=MARKERS[m])
        ax.scatter(x_right, y_right, color=COLORS[m], s=100, zorder=5,
                   edgecolors='black', linewidths=0.5, marker=MARKERS[m])

        ax.text(x_left - 0.03, y_left, f'{m}  ({acc_val}%)',
                ha='right', va='center', fontsize=8.5, color=COLORS[m], fontweight='bold')
        ax.text(x_right + 0.03, y_right, f'({fr_val})  {m}',
                ha='left', va='center', fontsize=8.5, color=COLORS[m], fontweight='bold')

    for i in range(n):
        ax.text(x_left, n - 1 - i, f'#{i+1} ', ha='right', va='center',
                fontsize=7, color='grey', alpha=0.5)
        ax.text(x_right, n - 1 - i, f' #{i+1}', ha='left', va='center',
                fontsize=7, color='grey', alpha=0.5)

    ax.text(x_left, n - 0.3, 'Pass@1 Ranking', ha='center', fontsize=11,
            fontweight='bold', color='#333333')
    ax.text(x_right, n - 0.3, 'Filtered Reasoning Ranking', ha='center', fontsize=11,
            fontweight='bold', color='#333333')

    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(-1, n + 0.5)
    ax.axis('off')
    ax.set_title('GSM8K: Ranking Reversal — Pass@1 vs Filtered Reasoning',
                 fontsize=13, fontweight='bold', pad=20)

    fig.savefig(f'{outdir}/fig2_bump_ranking_reversal.png', bbox_inches='tight')
    plt.close(fig)
    print('Figure 2 saved.')


# ============================================================
# FIGURE 3: Heatmap — Filtered Uplift (Filtered Reasoning - Reasoning)
# ============================================================
def fig3_heatmap():
    uplift_matrix = []
    model_labels = []
    for m in models:
        row = [filtered_reasoning[m][i] - reasoning[m][i] for i in range(6)]
        uplift_matrix.append(row)
        model_labels.append(m)

    data = np.array(uplift_matrix)

    sorted_idx = np.argsort(-data.mean(axis=1))
    data = data[sorted_idx]
    model_labels = [model_labels[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 6))

    vmax = max(abs(data.min()), abs(data.max()))
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(6))
    ax.set_xticklabels(datasets, fontsize=10, fontweight='bold')
    ax.set_yticks(range(len(model_labels)))
    ax.set_yticklabels(model_labels, fontsize=10, fontweight='bold')

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            sign = '+' if val > 0 else ''
            color = 'white' if abs(val) > vmax * 0.6 else 'black'
            ax.text(j, i, f'{sign}{val:.1f}', ha='center', va='center',
                    fontsize=9, fontweight='bold', color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Filtered Reasoning − Reasoning Score', fontsize=10)

    ax.set_title('Filtered Uplift: Does Reasoning Improve Under Confidence?',
                 fontsize=13, fontweight='bold', pad=12)

    ax.set_xlabel('')
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    fig.savefig(f'{outdir}/fig3_heatmap_filtered_uplift.png', bbox_inches='tight')
    plt.close(fig)
    print('Figure 3 saved.')


# ============================================================
# FIGURE 4: Phi-4-Reasoning vs DS-R1-7B — The Paradox
# ============================================================
def fig4_paradox():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    x = np.arange(len(datasets))
    width = 0.35

    for ax_idx, (m, title) in enumerate([
        ('Phi-4-Reasoning', 'Phi-4-Reasoning (Highest Pass@1)'),
        ('DS-R1-7B', 'DeepSeek-R1-Distill-Qwen-7B')
    ]):
        ax = axes[ax_idx]
        r_vals = reasoning[m]
        fr_vals = filtered_reasoning[m]

        bars1 = ax.bar(x - width/2, r_vals, width, label='Reasoning (all)',
                       color='#4c72b0', alpha=0.85, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, fr_vals, width, label='Filtered Reasoning',
                       color='#dd8452', alpha=0.85, edgecolor='black', linewidth=0.5)

        for i in range(len(datasets)):
            delta = fr_vals[i] - r_vals[i]
            sign = '+' if delta > 0 else ''
            color = '#2ca02c' if delta > 0 else '#d62728'
            y_pos = max(r_vals[i], fr_vals[i]) + 1.5
            ax.text(x[i], y_pos, f'{sign}{delta:.1f}', ha='center', va='bottom',
                    fontsize=8.5, fontweight='bold', color=color)

        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=9, rotation=25, ha='right')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.15)
        ax.legend(fontsize=8.5, loc='lower right')

    axes[0].set_ylabel('Score', fontweight='bold')

    fig.suptitle('The Paradox: Confidence Degrades vs. Improves Reasoning Quality',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(f'{outdir}/fig4_paradox_phi4r_vs_dsr1.png', bbox_inches='tight')
    plt.close(fig)
    print('Figure 4 saved.')


if __name__ == '__main__':
    fig1_scatter()
    fig2_bump()
    fig3_heatmap()
    fig4_paradox()
    print('\nAll figures saved to:', outdir)
