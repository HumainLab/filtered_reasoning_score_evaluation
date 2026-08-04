import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

GREEN = '#1b5e20'
RED = '#b71c1c'
AMBER = '#e65100'
ERROR_BG = '#fce4ec'
CODE_BG = '#f5f5f5'
CODE_BORDER = '#d0d0d0'
TXT = '#212121'
SUB = '#757575'
BAR_TRACK = '#eeeeee'
MONO = 'monospace'
SERIF = 'serif'
SANS = 'sans-serif'

W = 10.5

def score_color(v):
    if v >= 3.5:
        return GREEN
    if v <= 2.5:
        return RED
    return AMBER

MID = W / 2
GAP = 0.16
PW = MID - GAP - 0.12
TITLE_H = 0.54


def draw_excerpt_box(ax, x, y_top, w, lines, is_err=False, tag=None):
    line_h = 0.155
    tag_h = 0.14 if tag else 0.0
    box_h = len(lines) * line_h + 0.08 + tag_h

    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y_top - box_h), w, box_h,
        boxstyle="round,pad=0.035",
        linewidth=0.8 if is_err else 0.45,
        edgecolor=RED if is_err else CODE_BORDER,
        facecolor=ERROR_BG if is_err else CODE_BG
    ))

    ty = y_top - 0.06
    for line in lines:
        ax.text(
            x + 0.08, ty, line,
            fontsize=5.5,
            color=RED if is_err else TXT,
            va='top',
            fontfamily=MONO,
            fontweight='bold' if is_err else 'normal'
        )
        ty -= line_h

    if tag:
        tag_color = RED if is_err else GREEN
        tag_w = min(0.12 + 0.045 * max(len(s) for s in tag.split('\n')), w * 0.58)
        tx = x + w - tag_w - 0.06
        ty2 = y_top - box_h + 0.04

        ax.add_patch(mpatches.FancyBboxPatch(
            (tx, ty2), tag_w, 0.12,
            boxstyle="round,pad=0.02",
            linewidth=0.6,
            edgecolor=tag_color,
            facecolor='white',
            alpha=0.95
        ))
        ax.text(
            tx + tag_w / 2, ty2 + 0.06, tag,
            fontsize=4.85,
            color=tag_color,
            ha='center', va='center',
            fontfamily=SANS,
            fontweight='bold'
        )

    return box_h


def draw_panel(ax, xl, top, panel_label, model_info, question_lines, excerpts,
               is_correct, pass1, pillars, overall):
    border_c = GREEN if is_correct else RED

    y = top - 0.12
    ax.text(
        xl + 0.18, y, panel_label,
        fontsize=8.8, fontweight='bold',
        color=TXT, va='center',
        fontfamily=SERIF
    )

    verdict = "CORRECT" if is_correct else "INCORRECT"
    bw = 0.72 if is_correct else 0.84
    bx = xl + PW - bw - 0.18
    ax.add_patch(mpatches.FancyBboxPatch(
        (bx, y - 0.085), bw, 0.17,
        boxstyle="round,pad=0.03",
        linewidth=0.9,
        edgecolor=border_c,
        facecolor=border_c,
        alpha=0.1
    ))
    ax.text(
        bx + bw / 2, y, verdict,
        fontsize=6.7, fontweight='bold',
        color=border_c,
        ha='center', va='center',
        fontfamily=SANS
    )

    y -= 0.18
    ax.text(
        xl + 0.18, y, model_info,
        fontsize=6.1, color=SUB,
        va='center', fontfamily=SANS
    )

    y -= 0.22
    ax.text(
        xl + 0.18, y, "Q:",
        fontsize=6.1, fontweight='bold',
        color=SUB, va='top',
        fontfamily=SANS
    )
    for i, line in enumerate(question_lines):
        ax.text(
            xl + 0.40, y - i * 0.15, line,
            fontsize=6.0, color=TXT,
            va='top',
            fontfamily=SERIF,
            style='italic'
        )

    y -= len(question_lines) * 0.15 + 0.14
    ax.text(
        xl + 0.18, y, "Key output excerpts:",
        fontsize=5.8, fontweight='bold',
        color=SUB, va='top',
        fontfamily=SANS
    )
    y -= 0.15

    box_x = xl + 0.18
    box_w = PW - 0.36

    for lines, is_err, tag in excerpts:
        bh = draw_excerpt_box(ax, box_x, y, box_w, lines, is_err=is_err, tag=tag)
        y -= bh + 0.045

    y -= 0.01
    ax.plot([xl + 0.18, xl + PW - 0.18], [y, y], color=CODE_BORDER, lw=0.4)

    y -= 0.12
    pass_c = GREEN if is_correct else RED
    ax.text(
        xl + 0.18, y, "Pass@1",
        fontsize=7.0, fontweight='bold',
        color=TXT, va='center',
        fontfamily=SANS
    )
    ax.add_patch(mpatches.FancyBboxPatch(
        (xl + 0.84, y - 0.085), 0.28, 0.17,
        boxstyle="round,pad=0.03",
        linewidth=0.9,
        edgecolor=pass_c,
        facecolor=pass_c,
        alpha=0.1
    ))
    ax.text(
        xl + 0.98, y, str(pass1),
        fontsize=8.4, fontweight='bold',
        color=pass_c,
        va='center', ha='center',
        fontfamily=SANS
    )

    overall_c = score_color(overall)
    ax.text(
        xl + 1.34, y, "Reasoning",
        fontsize=7.0, fontweight='bold',
        color=TXT, va='center',
        fontfamily=SANS
    )
    ax.add_patch(mpatches.FancyBboxPatch(
        (xl + 2.12, y - 0.085), 0.62, 0.17,
        boxstyle="round,pad=0.03",
        linewidth=0.9,
        edgecolor=overall_c,
        facecolor=overall_c,
        alpha=0.1
    ))
    ax.text(
        xl + 2.43, y, f"{overall:.1f} / 5",
        fontsize=8.4, fontweight='bold',
        color=overall_c,
        va='center', ha='center',
        fontfamily=SANS
    )

    if is_correct and overall < 3.0:
        vt, vc = "Misleading success", RED
    elif not is_correct and overall >= 3.5:
        vt, vc = "Hidden capability", GREEN
    else:
        vt, vc = "", SUB

    ax.text(
        xl + PW - 0.18, y, vt,
        fontsize=6.9, fontweight='bold',
        color=vc, style='italic',
        va='center', ha='right',
        fontfamily=SANS
    )

    y -= 0.25
    labels = ["Faith.", "Utility", "Coher.", "Factual."]
    bar_x = xl + 1.02
    bar_w = PW - 2.68
    bar_h = 0.09
    gap = 0.03

    for i, (lbl, val) in enumerate(zip(labels, pillars)):
        by = y - i * (bar_h + gap)

        ax.text(
            bar_x - 0.05, by + bar_h / 2, lbl,
            fontsize=5.2, color=SUB,
            va='center', ha='right',
            fontfamily=SANS
        )

        ax.add_patch(mpatches.FancyBboxPatch(
            (bar_x, by), bar_w, bar_h,
            boxstyle="round,pad=0.01",
            linewidth=0,
            facecolor=BAR_TRACK
        ))

        ax.add_patch(mpatches.FancyBboxPatch(
            (bar_x, by), max(bar_w * (val / 5.0), 0.04), bar_h,
            boxstyle="round,pad=0.01",
            linewidth=0,
            facecolor=score_color(val),
            alpha=0.65
        ))

        ax.text(
            bar_x + bar_w + 0.06, by + bar_h / 2, f"{val:.1f}",
            fontsize=5.6, fontweight='bold',
            color=score_color(val),
            va='center',
            fontfamily=SANS
        )

    bottom = by - 0.12

    # Draw border and header now that we know the extent
    ph = top - bottom
    ax.add_patch(mpatches.FancyBboxPatch(
        (xl, bottom), PW, ph,
        boxstyle="round,pad=0.05",
        linewidth=1.7,
        edgecolor=border_c,
        facecolor='white',
        zorder=0
    ))
    ax.add_patch(mpatches.FancyBboxPatch(
        (xl, top - 0.36), PW, 0.36,
        boxstyle="round,pad=0.04",
        linewidth=0,
        facecolor=border_c,
        alpha=0.06,
        zorder=0
    ))

    return bottom


# --- First pass: measure content height ---
# We draw to a temporary figure to find the bottom, then create the real one.

import io

def make_figure(save=False):
    H_est = 6.5
    fig = plt.figure(figsize=(W, H_est), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H_est)
    ax.axis('off')

    TOP = H_est - TITLE_H

    # Title
    ax.text(
        W / 2, H_est - 0.16,
        "Pass@1 cannot distinguish reasoning quality from answer correctness",
        fontsize=10.8, fontweight='bold', color=TXT,
        ha='center', va='center', fontfamily=SERIF
    )
    ax.text(
        W / 2, H_est - 0.37,
        "Two real examples where accuracy metrics give misleading signals"
        "  (full model outputs in Appendix B)",
        fontsize=6.6, color=SUB,
        ha='center', va='center', fontfamily=SANS
    )

    bot_a = draw_panel(
        ax, xl=0.12, top=TOP,
        panel_label="(a) Correct Answer, Poor Reasoning",
        model_info="Gemma-7B  |  MATH500 #452  |  GT: 143",
        question_lines=[
            "The sum of the digits of a two-digit number is 13.",
            "The difference between the number and the number",
            "with its digits reversed is 27. What is the sum of the",
            "original number and the number with its digits reversed?",
        ],
        excerpts=[
            (["x+y=13  and  (10x+y)-(10y+x)=27"], False, None),
            (["9x=14,  x=14/9=1.5",
              "\"x must be integer, so x=1\" -> y=12"], True, "invalid digit"),
            (["Number=10(1)+12=22; Reversed=10(12)+1=121",
              "Sum = 22+121 = 143  ->  \\boxed{143}"], True, "correct by coincidence"),
        ],
        is_correct=True,
        pass1=1,
        pillars=[2.0, 2.0, 2.3, 2.0],
        overall=2.1,
    )

    bot_b = draw_panel(
        ax, xl=MID + GAP, top=TOP,
        panel_label="(b) Wrong Answer, Strong Reasoning",
        model_info="Phi-4-Reasoning  |  GSM8K #242  |  GT: 3",
        question_lines=[
            "Mike was a pen pal with 5 people. He stopped being",
            "penpals with 2 of them. They each send 2 letters a",
            "week that are 5 pages long. He responds in kind.",
            "He can write a page every 6 minutes. How many",
            "hours does he spend writing a week?",
        ],
        excerpts=[
            (["3 pals x 2 letters = 6 letters, 6x5 = 30 pages",
              "He writes 30 pages. 30x6min = 180min = 3 hrs"], False, None),
            (["<think> \"Final answer: \\boxed{3} hours.\""], False, "correct CoT"),
            (["</think> \"Combined, he writes 30+30 = 60 pages\"",
              "60 x 6min = 360min = 6 hrs  ->  \\boxed{6}"], True, "double-count"),
        ],
        is_correct=False,
        pass1=0,
        pillars=[3.7, 3.7, 4.3, 4.7],
        overall=4.1,
    )

    lowest = min(bot_a, bot_b)

    if save:
        for fmt in ["pdf", "png"]:
            path = f"figures/fig1_pass1_vs_reasoning.{fmt}"
            fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved: {path}")

    plt.close(fig)
    return lowest

# First pass: measure
lowest = make_figure(save=False)
# We want lowest to map to ~0.10 in the figure. Adjust H.
H_needed = 6.5 - lowest + 0.15
print(f"Content bottom at {lowest:.2f}, setting H = {H_needed:.2f}")

# Now create properly sized figure
# Override H_est in make_figure via a global
import types

def make_final_figure():
    H = H_needed
    fig = plt.figure(figsize=(W, H), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis('off')

    TOP = H - TITLE_H

    ax.text(
        W / 2, H - 0.16,
        "Pass@1 cannot distinguish reasoning quality from answer correctness",
        fontsize=10.8, fontweight='bold', color=TXT,
        ha='center', va='center', fontfamily=SERIF
    )
    ax.text(
        W / 2, H - 0.37,
        "Two real examples where accuracy metrics give misleading signals"
        "  (full model outputs in Appendix B)",
        fontsize=6.6, color=SUB,
        ha='center', va='center', fontfamily=SANS
    )

    draw_panel(
        ax, xl=0.12, top=TOP,
        panel_label="(a) Correct Answer, Poor Reasoning",
        model_info="Gemma-7B  |  MATH500 #452  |  GT: 143",
        question_lines=[
            "The sum of the digits of a two-digit number is 13.",
            "The difference between the number and the number",
            "with its digits reversed is 27. What is the sum of the",
            "original number and the number with its digits reversed?",
        ],
        excerpts=[
            (["x+y=13  and  (10x+y)-(10y+x)=27"], False, None),
            (["9x=14,  x=14/9=1.5",
              "\"x must be integer, so x=1\" -> y=12"], True, "invalid digit"),
            (["Number=10(1)+12=22; Reversed=10(12)+1=121",
              "Sum = 22+121 = 143  ->  \\boxed{143}"], True, "correct by coincidence"),
        ],
        is_correct=True,
        pass1=1,
        pillars=[2.0, 2.0, 2.3, 2.0],
        overall=2.1,
    )

    draw_panel(
        ax, xl=MID + GAP, top=TOP,
        panel_label="(b) Wrong Answer, Strong Reasoning",
        model_info="Phi-4-Reasoning  |  GSM8K #242  |  GT: 3",
        question_lines=[
            "Mike was a pen pal with 5 people. He stopped being",
            "penpals with 2 of them. They each send 2 letters a",
            "week that are 5 pages long. He responds in kind.",
            "He can write a page every 6 minutes. How many",
            "hours does he spend writing a week?",
        ],
        excerpts=[
            (["3 pals x 2 letters = 6 letters, 6x5 = 30 pages",
              "He writes 30 pages. 30x6min = 180min = 3 hrs"], False, None),
            (["<think> \"Final answer: \\boxed{3} hours.\""], False, "correct CoT"),
            (["</think> \"Combined, he writes 30+30 = 60 pages\"",
              "60 x 6min = 360min = 6 hrs  ->  \\boxed{6}"], True, "double-count"),
        ],
        is_correct=False,
        pass1=0,
        pillars=[3.7, 3.7, 4.3, 4.7],
        overall=4.1,
    )

    for fmt in ["pdf", "png"]:
        path = f"figures/fig1_pass1_vs_reasoning.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {path}")
    plt.close(fig)

make_final_figure()