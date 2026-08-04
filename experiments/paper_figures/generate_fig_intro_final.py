"""
Final camera-ready intro figure: 2-panel pass@1 vs FRS.

Variants:
  A – Balanced  (recommended)
  B – Most minimal
  C – More interpretive / annotated
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

OUTDIR = Path(__file__).parent / "figures"
OUTDIR.mkdir(exist_ok=True)

# ── palette ──────────────────────────────────────────────────────────
C_A      = "#1D4ED8"   # blue-700  (DeepSeek)
C_B      = "#B45309"   # amber-700 (Qwen)
C_A_MUT  = "#BFDBFE"   # blue-200
C_B_MUT  = "#FDE68A"   # amber-200
C_RED    = "#991B1B"
C_GRAY   = "#9CA3AF"
C_TXT    = "#111827"
C_TXT2   = "#6B7280"
C_SPINE  = "#D1D5DB"
C_FRS_BG = "#EFF6FF"

NAME_A = "DeepSeek-R1-7B"
NAME_B = "Qwen2.5-Math-7B"
PASS1 = [91.1, 90.8]
FRS   = [92.0, 67.0]


def _rc():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    })


def _tc(c, thr=0.52):
    r, g, b, _ = to_rgba(c)
    return "white" if (0.299*r + 0.587*g + 0.114*b) < thr else C_TXT


def _save(fig, name):
    for ext in ("png", "pdf"):
        p = OUTDIR / f"{name}.{ext}"
        fig.savefig(p, dpi=300, facecolor="white",
                    bbox_inches="tight", pad_inches=0.15)
        print(f"  -> {p}")
    plt.close(fig)


def _draw_panel(ax, vals, colors, title, title_fs, bg=None, names=True):
    """Draw two horizontal bars. Returns bar objects."""
    if bg:
        ax.set_facecolor(bg)

    y = [0.65, 0.0]
    bh = 0.40
    bars = ax.barh(y, vals, height=bh, color=colors,
                   edgecolor="white", linewidth=0.6, zorder=3)

    for bar, v, c in zip(bars, vals, colors):
        ax.text(v - 1.8, bar.get_y() + bh / 2,
                f"{v:.1f}%", ha="right", va="center",
                fontsize=10.5, fontweight="bold",
                color=_tc(c), zorder=4)

    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100"],
                       fontsize=6.5, color=C_TXT2)
    ax.set_yticks(y)
    if names:
        ax.set_yticklabels([NAME_A, NAME_B], fontsize=8, color=C_TXT)
    else:
        ax.set_yticklabels(["", ""])

    ax.set_title(title, fontsize=title_fs, fontweight="bold",
                 color=C_TXT, pad=10)

    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(C_SPINE)
    ax.tick_params(left=False, bottom=True, colors=C_TXT2)
    ax.set_axisbelow(True)


# ═════════════════════════════════════════════════════════════════════
#  VARIANT A – Balanced  (recommended)
# ═════════════════════════════════════════════════════════════════════

def variant_a():
    _rc()
    fig = plt.figure(figsize=(6.8, 3.2), facecolor="white")

    gs = gridspec.GridSpec(
        1, 2, width_ratios=[0.90, 1.10], wspace=0.18,
        left=0.155, right=0.96, top=0.68, bottom=0.22,
    )
    ax_p = fig.add_subplot(gs[0])
    ax_f = fig.add_subplot(gs[1])

    fig.text(0.555, 0.96,
             "Same Accuracy, Different Reasoning",
             ha="center", fontsize=15, fontweight="bold", color=C_TXT)
    fig.text(0.555, 0.88,
             "On SVAMP, DeepSeek-R1-7B and Qwen2.5-Math-7B differ by "
             "0.3 pp in pass@1 but 25 pp in FRS.",
             ha="center", fontsize=8.5, color=C_TXT2, fontstyle="italic")

    _draw_panel(ax_p, PASS1, [C_A_MUT, C_B_MUT],
                "pass@1 Accuracy", 11, names=True)
    _draw_panel(ax_f, FRS, [C_A, C_B],
                "Filtered Reasoning Score (FRS)", 11.5,
                bg=C_FRS_BG, names=False)

    ax_p.text(0.50, -0.17, "Near tie  \u00b7  \u0394 = 0.3 pp",
              transform=ax_p.transAxes, ha="center", va="top",
              fontsize=9.5, fontweight="bold", color=C_GRAY)

    ax_f.text(0.50, -0.17, "25 pp gap revealed",
              transform=ax_f.transAxes, ha="center", va="top",
              fontsize=11, fontweight="bold", color=C_RED)

    fig.text(0.555, 0.06,
             "Accuracy suggests parity \u2014 "
             "FRS reveals a large hidden reasoning gap.",
             ha="center", fontsize=8.5, color=C_TXT2, fontstyle="italic")

    _save(fig, "fig_intro_final_A")


# ═════════════════════════════════════════════════════════════════════
#  VARIANT B – Most minimal
# ═════════════════════════════════════════════════════════════════════

def variant_b():
    _rc()
    fig = plt.figure(figsize=(6.4, 2.7), facecolor="white")

    gs = gridspec.GridSpec(
        1, 2, width_ratios=[0.90, 1.10], wspace=0.18,
        left=0.155, right=0.96, top=0.72, bottom=0.16,
    )
    ax_p = fig.add_subplot(gs[0])
    ax_f = fig.add_subplot(gs[1])

    fig.text(0.555, 0.965,
             "Same Accuracy, Different Reasoning",
             ha="center", fontsize=14, fontweight="bold", color=C_TXT)
    fig.text(0.555, 0.88,
             "SVAMP  \u00b7  DeepSeek-R1-7B  vs  Qwen2.5-Math-7B",
             ha="center", fontsize=8.5, color=C_TXT2)

    _draw_panel(ax_p, PASS1, [C_A_MUT, C_B_MUT],
                "pass@1", 11, names=True)
    _draw_panel(ax_f, FRS, [C_A, C_B],
                "Filtered Reasoning Score (FRS)", 11,
                bg=C_FRS_BG, names=False)

    ax_p.text(0.50, -0.18, "\u0394 = 0.3 pp",
              transform=ax_p.transAxes, ha="center", va="top",
              fontsize=9, fontweight="bold", color=C_GRAY)
    ax_f.text(0.50, -0.18, "\u0394 = 25 pp",
              transform=ax_f.transAxes, ha="center", va="top",
              fontsize=10.5, fontweight="bold", color=C_RED)

    _save(fig, "fig_intro_final_B")


# ═════════════════════════════════════════════════════════════════════
#  VARIANT C – More interpretive / annotated
# ═════════════════════════════════════════════════════════════════════

def variant_c():
    _rc()
    fig = plt.figure(figsize=(6.8, 3.6), facecolor="white")

    gs = gridspec.GridSpec(
        1, 2, width_ratios=[0.88, 1.12], wspace=0.18,
        left=0.155, right=0.96, top=0.62, bottom=0.20,
    )
    ax_p = fig.add_subplot(gs[0])
    ax_f = fig.add_subplot(gs[1])

    fig.text(0.555, 0.965,
             "Same Accuracy, Different Reasoning",
             ha="center", fontsize=15, fontweight="bold", color=C_TXT)
    fig.text(0.555, 0.895,
             "On SVAMP, DeepSeek-R1-7B and Qwen2.5-Math-7B differ by "
             "0.3 pp in pass@1 but 25 pp in FRS.",
             ha="center", fontsize=8.5, color=C_TXT2, fontstyle="italic")

    # narrative labels
    fig.text(0.285, 0.80, "\u2248  Tied",
             ha="center", fontsize=10.5, fontweight="bold", color=C_GRAY)
    fig.text(0.72, 0.80, "Hidden gap",
             ha="center", fontsize=11, fontweight="bold", color=C_RED,
             bbox=dict(boxstyle="round,pad=0.25", facecolor="#FEF2F2",
                       edgecolor=C_RED, linewidth=0.8, alpha=0.9))
    fig.patches.append(FancyArrowPatch(
        (0.40, 0.805), (0.56, 0.805), transform=fig.transFigure,
        arrowstyle="->,head_width=0.15,head_length=0.08",
        color=C_SPINE, lw=1.2))

    _draw_panel(ax_p, PASS1, [C_A_MUT, C_B_MUT],
                "pass@1 Accuracy", 11, names=True)
    _draw_panel(ax_f, FRS, [C_A, C_B],
                "Filtered Reasoning Score (FRS)", 11.5,
                bg=C_FRS_BG, names=False)

    ax_p.text(0.50, -0.17, "Near tie  \u00b7  \u0394 = 0.3 pp",
              transform=ax_p.transAxes, ha="center", va="top",
              fontsize=9.5, fontweight="bold", color=C_GRAY)
    ax_f.text(0.50, -0.17, "25 pp gap revealed",
              transform=ax_f.transAxes, ha="center", va="top",
              fontsize=11, fontweight="bold", color=C_RED)

    fig.text(0.555, 0.055,
             "Accuracy suggests parity \u2014 "
             "FRS reveals a large hidden reasoning gap.",
             ha="center", fontsize=8.5, color=C_TXT2, fontstyle="italic")

    _save(fig, "fig_intro_final_C")


if __name__ == "__main__":
    for label, fn in [("A (balanced)", variant_a),
                      ("B (minimal)", variant_b),
                      ("C (annotated)", variant_c)]:
        print(f"Variant {label}:")
        fn()
    print("\nDone.")
