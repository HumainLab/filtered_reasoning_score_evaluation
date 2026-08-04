"""
Camera-ready introductory figure: Same Accuracy, Different Reasoning.

Generates three variants:
  A – Balanced 3-panel  (recommended for main paper)
  B – Minimal dumbbell chart
  C – Annotated 3-panel with narrative labels
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.colors import to_rgba
from pathlib import Path

OUTDIR = Path(__file__).parent / "figures"
OUTDIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
#  Palette & data
# ═══════════════════════════════════════════════════════════════════════

C_A       = "#1D4ED8"   # blue-700   DS-R1-7B  (vivid)
C_B       = "#C2410C"   # orange-700 Qwen-Math  (vivid)
C_A_MUT   = "#93C5FD"   # blue-300   (muted)
C_B_MUT   = "#FDBA74"   # orange-300 (muted)
C_A_MED   = "#3B82F6"   # blue-500   (medium)
C_B_MED   = "#EA580C"   # orange-600 (medium)

C_RED     = "#991B1B"   # gap (large)
C_GRAY    = "#9CA3AF"   # gap (small)
C_TXT     = "#1F2937"
C_TXT2    = "#6B7280"
C_SPINE   = "#D1D5DB"
C_FRS_BG  = "#EFF6FF"   # blue-50

NAME_A = "DeepSeek-R1-7B"
NAME_B = "Qwen2.5-Math-7B"

PASS1 = [91.1, 90.8]
FRS   = [92.0, 67.0]
FILT  = [95.4, 32.5]


def _rc():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    })


def _text_color(hex_color, threshold=0.55):
    r, g, b, _ = to_rgba(hex_color)
    return "white" if (0.299*r + 0.587*g + 0.114*b) < threshold else C_TXT


def _save(fig, name):
    for ext in ("png", "pdf"):
        p = OUTDIR / f"{name}.{ext}"
        fig.savefig(p, dpi=300, facecolor="white",
                    bbox_inches="tight", pad_inches=0.12)
        print(f"  -> {p}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
#  VARIANT A — Balanced 3-panel
# ═══════════════════════════════════════════════════════════════════════

def variant_a():
    _rc()
    fig = plt.figure(figsize=(7.5, 2.85), facecolor="white")

    gs = gridspec.GridSpec(
        1, 3,
        width_ratios=[1.05, 1.20, 0.85],
        wspace=0.12,
        left=0.15, right=0.97, top=0.74, bottom=0.12,
    )
    axs = [fig.add_subplot(gs[i]) for i in range(3)]

    # ── titles ──
    fig.text(
        0.56, 0.96,
        "Same Accuracy, Different Reasoning",
        ha="center", fontsize=14, fontweight="bold", color=C_TXT,
    )
    fig.text(
        0.56, 0.885,
        "SVAMP  \u00b7  DeepSeek-R1-7B  vs  Qwen2.5-Math-7B",
        ha="center", fontsize=9.5, color=C_TXT2,
    )

    # ── panel configs ──
    configs = [
        dict(vals=PASS1, colors=[C_A_MUT, C_B_MUT],
             title="pass@1", title_fs=10.5,
             gap="\u0394 = 0.3 pp", gap_c=C_GRAY, gap_fs=8.5, gap_bold=False,
             bg=None, names=True),
        dict(vals=FRS, colors=[C_A, C_B],
             title="Filtered Reasoning Score (FRS)", title_fs=10.5,
             gap="\u0394 = 25 pp", gap_c=C_RED, gap_fs=12, gap_bold=True,
             bg=C_FRS_BG, names=False),
        dict(vals=FILT, colors=[C_A_MED, C_B_MED],
             title="Filtered pass@1", title_fs=9.5,
             gap="\u0394 = 63 pp", gap_c=C_RED, gap_fs=9, gap_bold=False,
             bg=None, names=False),
    ]

    y = [0.6, 0.0]
    bh = 0.38

    for ax, cfg in zip(axs, configs):
        if cfg["bg"]:
            ax.set_facecolor(cfg["bg"])

        bars = ax.barh(y, cfg["vals"], height=bh, color=cfg["colors"],
                       edgecolor="white", linewidth=0.6, zorder=3)

        for bar, v, c in zip(bars, cfg["vals"], cfg["colors"]):
            tc = _text_color(c)
            ax.text(
                v - 1.5, bar.get_y() + bh / 2,
                f"{v:.1f}%", ha="right", va="center",
                fontsize=9.5, fontweight="bold", color=tc, zorder=4,
            )

        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
        ax.set_xticklabels(["0", "50", "100"], fontsize=7, color=C_TXT2)
        ax.set_yticks(y)

        if cfg["names"]:
            ax.set_yticklabels([NAME_A, NAME_B], fontsize=8, color=C_TXT)
        else:
            ax.set_yticklabels(["", ""])

        ax.set_title(cfg["title"], fontsize=cfg["title_fs"],
                     fontweight="bold", color=C_TXT, pad=8)

        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(C_SPINE)
        ax.tick_params(left=False, bottom=True)
        ax.tick_params(axis="x", colors=C_TXT2)
        ax.set_axisbelow(True)

        wt = "bold" if cfg["gap_bold"] else "normal"
        ax.text(
            0.5, -0.18, cfg["gap"],
            transform=ax.transAxes, ha="center", va="top",
            fontsize=cfg["gap_fs"], fontweight=wt, color=cfg["gap_c"],
        )

    _save(fig, "fig_intro_A")


# ═══════════════════════════════════════════════════════════════════════
#  VARIANT B — Minimal dumbbell chart
# ═══════════════════════════════════════════════════════════════════════

def variant_b():
    _rc()
    fig, ax = plt.subplots(figsize=(6.0, 2.6), facecolor="white")
    plt.subplots_adjust(left=0.20, right=0.92, top=0.74, bottom=0.13)

    fig.text(0.56, 0.955, "Same Accuracy, Different Reasoning",
             ha="center", fontsize=13, fontweight="bold", color=C_TXT)
    fig.text(0.56, 0.87,
             "SVAMP  \u00b7  DeepSeek-R1-7B  vs  Qwen2.5-Math-7B",
             ha="center", fontsize=8.5, color=C_TXT2)

    rows = [
        ("pass@1",          PASS1[0], PASS1[1], C_GRAY, 1.5,  50,
         "\u0394 = 0.3 pp", C_GRAY,  8,    "normal"),
        ("FRS",             FRS[0],   FRS[1],   C_RED,  3.5,  70,
         "\u0394 = 25 pp",  C_RED,   10.5, "bold"),
        ("Filtered pass@1", FILT[0],  FILT[1],  C_RED,  2.0,  55,
         "\u0394 = 63 pp",  C_RED,   8.5,  "normal"),
    ]

    yvals = [2, 1, 0]

    # FRS band
    ax.axhspan(0.55, 1.45, color=C_FRS_BG, zorder=0)

    gap_y_offsets = [0.32, 0.32, -0.32]

    for i, (yi, (metric, va, vb, lc, lw, ds,
             gtxt, gc, gfs, gfw)) in enumerate(zip(yvals, rows)):
        lo, hi = min(va, vb), max(va, vb)

        ax.plot([lo, hi], [yi, yi], color=lc, linewidth=lw,
                solid_capstyle="round", zorder=2, alpha=0.65)
        ax.scatter(va, yi, s=ds, color=C_A, zorder=4,
                   edgecolor="white", linewidths=0.7)
        ax.scatter(vb, yi, s=ds, color=C_B, zorder=4,
                   edgecolor="white", linewidths=0.7)

        if abs(va - vb) < 2:
            ax.text(max(va, vb) + 1.8, yi,
                    f"{va:.1f}% / {vb:.1f}%",
                    fontsize=7.5, color=C_TXT, va="center", ha="left")
        else:
            ax.text(hi + 1.8, yi + 0.08,
                    f"{va:.1f}%", fontsize=7.5, color=C_A,
                    fontweight="bold", ha="left", va="bottom")
            ax.text(lo - 1.8, yi - 0.08,
                    f"{vb:.1f}%", fontsize=7.5, color=C_B,
                    fontweight="bold", ha="right", va="top")

        mid = (va + vb) / 2
        g_y = yi + gap_y_offsets[i]
        g_va = "bottom" if gap_y_offsets[i] > 0 else "top"
        ax.text(mid, g_y, gtxt, ha="center", va=g_va,
                fontsize=gfs, fontweight=gfw, color=gc)

    ax.set_xlim(0, 112)
    ax.set_ylim(-0.65, 2.75)
    ax.set_yticks(yvals)
    ax.set_yticklabels([r[0] for r in rows],
                        fontsize=9.5, color=C_TXT, fontweight="bold")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"],
                        fontsize=7, color=C_TXT2)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(C_SPINE)
    ax.tick_params(left=False, bottom=True, colors=C_TXT2)
    ax.grid(axis="x", color="#EEEEEE", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)

    leg_a = mpatches.Patch(color=C_A, label=NAME_A)
    leg_b = mpatches.Patch(color=C_B, label=NAME_B)
    ax.legend(handles=[leg_a, leg_b], loc="upper left",
              fontsize=7.5, frameon=True, framealpha=0.95,
              edgecolor="#E5E7EB", fancybox=False,
              handlelength=1.0, handleheight=0.7,
              bbox_to_anchor=(0.22, 1.0))

    _save(fig, "fig_intro_B")


# ═══════════════════════════════════════════════════════════════════════
#  VARIANT C — Annotated 3-panel with narrative labels
# ═══════════════════════════════════════════════════════════════════════

def variant_c():
    _rc()
    fig = plt.figure(figsize=(7.5, 3.5), facecolor="white")

    gs = gridspec.GridSpec(
        1, 3,
        width_ratios=[1.05, 1.20, 0.85],
        wspace=0.12,
        left=0.15, right=0.97, top=0.62, bottom=0.16,
    )
    axs = [fig.add_subplot(gs[i]) for i in range(3)]

    # ── title ──
    fig.text(0.56, 0.97, "Same Accuracy, Different Reasoning",
             ha="center", fontsize=14, fontweight="bold", color=C_TXT)
    fig.text(0.56, 0.915,
             "SVAMP  \u00b7  DeepSeek-R1-7B  vs  Qwen2.5-Math-7B",
             ha="center", fontsize=9.5, color=C_TXT2)

    # ── narrative labels ──
    panel_cx = [0.30, 0.58, 0.87]

    fig.text(panel_cx[0], 0.835, "\u2248  Tied",
             ha="center", fontsize=10, fontweight="bold", color=C_GRAY)

    bbox_kw = dict(boxstyle="round,pad=0.3", facecolor="#FEF2F2",
                   edgecolor=C_RED, linewidth=0.9)
    fig.text(panel_cx[1], 0.835, " 25 pp Gap ",
             ha="center", fontsize=11, fontweight="bold",
             color=C_RED, bbox=bbox_kw)

    fig.text(panel_cx[2], 0.835, "Supporting",
             ha="center", fontsize=9, fontstyle="italic", color=C_TXT2)

    # ── arrows ──
    akw = dict(arrowstyle="->,head_width=0.12,head_length=0.07",
               color="#C0C0C0", lw=1.0)
    fig.patches.append(mpatches.FancyArrowPatch(
        (0.40, 0.838), (0.46, 0.838), transform=fig.transFigure, **akw))
    fig.patches.append(mpatches.FancyArrowPatch(
        (0.70, 0.838), (0.77, 0.838), transform=fig.transFigure, **akw))

    # ── panels ──
    configs = [
        dict(vals=PASS1, colors=[C_A_MUT, C_B_MUT],
             title="pass@1", title_fs=10.5,
             gap="\u0394 = 0.3 pp", gap_c=C_GRAY, gap_fs=8.5, gap_bold=False,
             bg=None, names=True),
        dict(vals=FRS, colors=[C_A, C_B],
             title="Filtered Reasoning Score (FRS)", title_fs=10.5,
             gap="\u0394 = 25 pp", gap_c=C_RED, gap_fs=12, gap_bold=True,
             bg=C_FRS_BG, names=False),
        dict(vals=FILT, colors=[C_A_MED, C_B_MED],
             title="Filtered pass@1", title_fs=9.5,
             gap="\u0394 = 63 pp", gap_c=C_RED, gap_fs=9, gap_bold=False,
             bg=None, names=False),
    ]

    y = [0.6, 0.0]
    bh = 0.38

    for ax, cfg in zip(axs, configs):
        if cfg["bg"]:
            ax.set_facecolor(cfg["bg"])

        bars = ax.barh(y, cfg["vals"], height=bh, color=cfg["colors"],
                       edgecolor="white", linewidth=0.6, zorder=3)

        for bar, v, c in zip(bars, cfg["vals"], cfg["colors"]):
            tc = _text_color(c)
            ax.text(v - 1.5, bar.get_y() + bh / 2,
                    f"{v:.1f}%", ha="right", va="center",
                    fontsize=9.5, fontweight="bold", color=tc, zorder=4)

        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
        ax.set_xticklabels(["0", "50", "100"], fontsize=7, color=C_TXT2)
        ax.set_yticks(y)

        if cfg["names"]:
            ax.set_yticklabels([NAME_A, NAME_B], fontsize=8, color=C_TXT)
        else:
            ax.set_yticklabels(["", ""])

        ax.set_title(cfg["title"], fontsize=cfg["title_fs"],
                     fontweight="bold", color=C_TXT, pad=8)

        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(C_SPINE)
        ax.tick_params(left=False, bottom=True, colors=C_TXT2)
        ax.set_axisbelow(True)

        wt = "bold" if cfg["gap_bold"] else "normal"
        ax.text(0.5, -0.18, cfg["gap"],
                transform=ax.transAxes, ha="center", va="top",
                fontsize=cfg["gap_fs"], fontweight=wt, color=cfg["gap_c"])

    _save(fig, "fig_intro_C")


# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Variant A (balanced 3-panel):")
    variant_a()
    print("\nVariant B (minimal dumbbell):")
    variant_b()
    print("\nVariant C (annotated 3-panel):")
    variant_c()
    print("\nDone.")
