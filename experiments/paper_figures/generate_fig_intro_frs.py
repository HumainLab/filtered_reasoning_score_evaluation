import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

OUTDIR = Path(__file__).parent / "figures"
OUTDIR.mkdir(exist_ok=True)

# ── palette ──────────────────────────────────────────────────────────────
C_DSR1       = "#1565C0"
C_QWEN       = "#E65100"
C_MUTED_DSR1 = "#90CAF9"
C_MUTED_QWEN = "#FFAB91"
C_GAP_SMALL  = "#757575"
C_GAP_LARGE  = "#C62828"
C_GRID       = "#E0E0E0"

# ── data ─────────────────────────────────────────────────────────────────
models     = ["DeepSeek-R1-7B", "Qwen2.5-Math-7B"]
pass1      = [91.1, 90.8]
frs        = [92.0, 67.0]
filt_pass1 = [95.4, 32.5]
quality    = [0.922, 0.673]

# ── figure ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14.0, 5.6), facecolor="white")

gs = gridspec.GridSpec(
    2, 3,
    height_ratios=[1, 0.20],
    width_ratios=[1, 1, 1],
    hspace=0.55, wspace=0.55,
    left=0.09, right=0.89, top=0.81, bottom=0.07,
)

ax_pass1 = fig.add_subplot(gs[0, 0])
ax_frs   = fig.add_subplot(gs[0, 1])
ax_filt  = fig.add_subplot(gs[0, 2])
ax_strip = fig.add_subplot(gs[1, :])

# ── titles ───────────────────────────────────────────────────────────────
fig.suptitle(
    "Same Accuracy, Different Reasoning",
    fontsize=19, fontweight="bold", y=0.97, fontfamily="serif",
)
fig.text(
    0.515, 0.905,
    "SVAMP  ·  DeepSeek-R1-7B  vs  Qwen2.5-Math-7B",
    ha="center", fontsize=11.5, color="#555555", fontfamily="serif",
)

# ── draw one bar panel ───────────────────────────────────────────────────
def draw_panel(ax, vals, title, gap_text, sub_text,
               colors, gap_color, bold_gap=False):
    y_pos = [1.0, 0.0]
    bars = ax.barh(y_pos, vals, height=0.52, color=colors,
                   edgecolor="white", linewidth=1.3, zorder=3)

    for bar, v in zip(bars, vals):
        ax.text(v - 1.2, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%", va="center", ha="right",
                fontsize=13, fontweight="bold", color="white", zorder=4)

    ax.set_xlim(0, 100)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=9.5, fontfamily="serif")
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=10,
                 fontfamily="serif")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    ax.grid(axis="x", color=C_GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticks([0, 25, 50, 75, 100])

    brace_x = min(max(vals) + 2, 97)
    lw = 2.2 if bold_gap else 1.3
    ax.annotate("", xy=(brace_x, -0.22), xytext=(brace_x, 1.22),
                arrowprops=dict(arrowstyle="]-[", color=gap_color,
                                lw=lw, mutation_scale=7))
    fs = 12.5 if bold_gap else 10.5
    wt = "bold" if bold_gap else "normal"
    ax.text(brace_x + 0.5, 0.5, gap_text, va="center", ha="left",
            fontsize=fs, fontweight=wt, color=gap_color, fontfamily="serif",
            clip_on=False)

    ax.text(0.5, -0.21, sub_text, transform=ax.transAxes,
            ha="center", va="top", fontsize=8.5, color="#888888",
            fontstyle="italic", fontfamily="serif")


# ── Panel A ──────────────────────────────────────────────────────────────
draw_panel(ax_pass1, pass1,
           title="pass@1 Accuracy",
           gap_text="Δ 0.3 pp",
           sub_text="Indistinguishable",
           colors=[C_MUTED_DSR1, C_MUTED_QWEN],
           gap_color=C_GAP_SMALL, bold_gap=False)

# ── Panel B ──────────────────────────────────────────────────────────────
draw_panel(ax_frs, frs,
           title="Filtered Reasoning\nScore (FRS)",
           gap_text="Δ 25 pp",
           sub_text="25 pp gap revealed",
           colors=[C_DSR1, C_QWEN],
           gap_color=C_GAP_LARGE, bold_gap=True)

# ── Panel C ──────────────────────────────────────────────────────────────
draw_panel(ax_filt, filt_pass1,
           title="Filtered pass@1\n(high-confidence)",
           gap_text="Δ 63 pp",
           sub_text="63 pp gap in reliable region",
           colors=[C_DSR1, C_QWEN],
           gap_color=C_GAP_LARGE, bold_gap=True)

# ── Bottom metric strip ─────────────────────────────────────────────────
ax_strip.axis("off")

headers  = ["", "pass@1", "FRS", "Filtered pass@1", "Reasoning Quality"]
row1     = ["DeepSeek-R1-7B",    "91.1 %", "92.0 %", "95.4 %", "0.922"]
row2     = ["Qwen2.5-Math-7B", "90.8 %", "67.0 %", "32.5 %", "0.673"]

tbl = ax_strip.table(
    cellText=[row1, row2], colLabels=headers,
    cellLoc="center", loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1.0, 1.50)

for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#C0C0C0")
    cell.set_linewidth(0.5)
    if r == 0:
        cell.set_facecolor("#E0E0E0")
        cell.set_text_props(fontweight="bold", fontfamily="serif", fontsize=9)
    else:
        cell.set_text_props(fontfamily="serif", fontsize=9.5)
        if c == 0:
            cell.set_text_props(fontweight="bold", fontfamily="serif")
        if c >= 2 and r == 1:
            cell.set_text_props(fontweight="bold", color=C_DSR1,
                                fontfamily="serif")
        if c >= 2 and r == 2:
            cell.set_text_props(color=C_QWEN, fontfamily="serif")
        cell.set_facecolor("white")

# ── save ─────────────────────────────────────────────────────────────────
for ext in ("png", "pdf"):
    path = OUTDIR / f"fig_intro_frs_comparison.{ext}"
    fig.savefig(path, dpi=300, facecolor="white",
                bbox_inches="tight", pad_inches=0.15)
    print(f"Saved  {path}")

plt.close(fig)
