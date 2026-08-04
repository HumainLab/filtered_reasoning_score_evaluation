import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import textwrap

CORRECT_GREEN = '#1b5e20'
WRONG_RED = '#b71c1c'
ERROR_BG = '#fce4ec'
CODE_BG = '#f5f5f5'
CODE_BORDER = '#d0d0d0'
TEXT_COLOR = '#212121'
SUBTLE = '#757575'
MONO = 'monospace'
SERIF = 'serif'
SANS = 'sans-serif'
THINK_BLUE = '#1565c0'

LH = 0.019
FS = 6.2
ANNOT_FS = 5.8
CODE_WIDTH = 0.69

OUTPUT_A = [
    "Let the two-digit number be $10x+y$ where $x$ and $y$",
    "are the digits of the two-digit number.",
    "Let the number with its digits reversed be $10y+x.$",
    "We are given the following information:",
    "$x+y=13$ and $(10x+y)-(10y+x)=27.$",
    "",
    "Substituting the first equation into the second,",
    "$(10x+y)-(10y+x)=27$",
    "$(10x+13)-(10y+x)=27$",
    "$9x+13=27$",
    "$9x=14$",
    "$x=\\frac{14}{9}$",
    "$x=1.\\overline{5}$",
    "",
    "However, $x$ must be an integer, so we must have",
    "$x=1.$ Substituting into the first equation:",
    "$x+y=13$",
    "$1+y=13$",
    "$y=12$",
    "",
    "Then, the original two-digit number is",
    "$10x+y=10(1)+12=22.$",
    "The number with its digits reversed is",
    "$10y+x=10(12)+1=121.$",
    "",
    "Then, the sum of the original number and the number",
    "with its digits reversed is $22+121=143.$",
    "",
    "Therefore, the answer is  \\boxed{143}",
]

ERRORS_A = [
    (9, 13, "Wrong substitution\nleads to non-integer\nx = 14/9"),
    (14, 19, "Unjustified rounding:\nx=1, y=12 is not\na valid digit"),
    (20, 24, "22 is not a valid\n2-digit number (y=12);\n121 has 3 digits"),
]

OUTPUT_B_THINK = [
    "Mike is a pen pal with 5 people. However, he stops",
    "being pen pals with 2 of them. That leaves him with",
    "3 pen pals. They each send 2 letters a week that are",
    "5 pages long. He responds in kind.",
    "",
    "... each of the 3 pen pals sends 2 letters a week.",
    "Total letters received: 3 * 2 = 6 letters.",
    "Each letter is 5 pages, so pages received:",
    "6 letters * 5 pages = 30 pages.",
    "",
    "He responds in kind. That means he writes a letter",
    "for each letter he receives. So he writes 6 letters.",
    "Pages written by him: 6 * 5 = 30 pages.",
    "",
    "Total pages he writes per week = 30 pages.",
    "Time per week = 30 pages * 6 min/page = 180 min.",
    "Convert: 180 / 60 = 3 hours.",
    "",
    "Thus, he spends 3 hours writing a week.",
    "Final answer: \\boxed{3} hours.",
]

OUTPUT_B_RESP = [
    "Mike originally had 5 pen pals, but he stopped being",
    "pen pals with 2, leaving him with 3. Each of these 3",
    "sends 2 letters per week, each letter being 5 pages.",
    "He receives 3 x 2 = 6 letters, totaling 6 x 5 = 30",
    "pages. Since he responds in kind (writing a letter",
    "for each received letter, also 5 pages each), he",
    "writes another 6 x 5 = 30 pages. Combined, he",
    "writes 30 + 30 = 60 pages per week.",
    "",
    "At a rate of 1 page every 6 minutes, the total",
    "writing time is 60 pages x 6 min/page = 360 min.",
    "Converting: 360 / 60 = 6 hours.",
    "",
    "\\boxed{6}",
]

ERRORS_B_RESP = [
    (6, 8, "Double-counts: adds\nreceived pages to\nwritten pages"),
    (9, 12, "Propagates error\nto final answer"),
]


def draw_code_block(ax, x, y_top, lines, errors=None, label=None, label_color=None):
    """Draw a code block starting at y_top, return y_bottom."""
    n = len(lines)
    block_h = n * LH + 0.02
    bg = mpatches.FancyBboxPatch(
        (x, y_top - block_h), CODE_WIDTH, block_h,
        boxstyle="round,pad=0.008", linewidth=0.7,
        edgecolor=CODE_BORDER, facecolor=CODE_BG
    )
    ax.add_patch(bg)

    if label:
        lw = 0.10 + len(label) * 0.005
        lbl_bg = mpatches.FancyBboxPatch(
            (x + 0.005, y_top + 0.003), lw, 0.016,
            boxstyle="round,pad=0.003", linewidth=0.7,
            edgecolor=label_color or SUBTLE, facecolor='white'
        )
        ax.add_patch(lbl_bg)
        ax.text(x + 0.005 + lw / 2, y_top + 0.011, label,
                fontsize=5.8, fontweight='bold', color=label_color or SUBTLE,
                va='center', ha='center', fontfamily=SANS)

    if errors:
        for (start, end, annot) in errors:
            hl_h = (end - start) * LH
            hl_y = y_top - 0.012 - start * LH
            hl = mpatches.FancyBboxPatch(
                (x + 0.003, hl_y - hl_h + LH * 0.35), CODE_WIDTH - 0.006, hl_h,
                boxstyle="round,pad=0.003", linewidth=1.0,
                edgecolor=WRONG_RED, facecolor=ERROR_BG, alpha=0.55, zorder=2
            )
            ax.add_patch(hl)
            ann_x = x + CODE_WIDTH + 0.01
            ann_y = hl_y - hl_h / 2 + LH * 0.35
            ax.annotate(
                annot,
                xy=(x + CODE_WIDTH - 0.003, ann_y),
                xytext=(ann_x + 0.01, ann_y),
                fontsize=ANNOT_FS, color=WRONG_RED, fontfamily=SANS,
                fontweight='bold', va='center',
                arrowprops=dict(arrowstyle='->', color=WRONG_RED, lw=0.7,
                                shrinkA=0, shrinkB=2),
            )

    y = y_top - 0.013
    for line in lines:
        ax.text(x + 0.01, y, line,
                fontsize=FS, color=TEXT_COLOR, va='top',
                fontfamily=MONO, zorder=3)
        y -= LH

    return y_top - block_h


def draw_scores(ax, y, pass1, pass1_correct, rs, rs_good, gt_text, verdict, verdict_color):
    ax.plot([0.04, 0.96], [y + 0.01, y + 0.01], color=CODE_BORDER, linewidth=0.5)
    ax.text(0.04, y - 0.005, "SCORES", fontsize=6, fontweight='bold',
            color=SUBTLE, va='top', fontfamily=SANS)
    y -= 0.028

    p_color = CORRECT_GREEN if pass1_correct else WRONG_RED
    ax.text(0.13, y, "Pass@1", fontsize=9.5, fontweight='bold',
            color=TEXT_COLOR, va='center', ha='right', fontfamily=SANS)
    p_bg = mpatches.FancyBboxPatch(
        (0.15, y - 0.015), 0.05, 0.03,
        boxstyle="round,pad=0.005", linewidth=1.3,
        edgecolor=p_color, facecolor=p_color, alpha=0.12
    )
    ax.add_patch(p_bg)
    ax.text(0.175, y, str(pass1), fontsize=11, fontweight='bold',
            color=p_color, va='center', ha='center', fontfamily=SANS)

    rs_color = CORRECT_GREEN if rs_good else WRONG_RED
    ax.text(0.41, y, "Reasoning Score", fontsize=9.5, fontweight='bold',
            color=TEXT_COLOR, va='center', ha='right', fontfamily=SANS)
    rs_bg = mpatches.FancyBboxPatch(
        (0.43, y - 0.015), 0.11, 0.03,
        boxstyle="round,pad=0.005", linewidth=1.3,
        edgecolor=rs_color, facecolor=rs_color, alpha=0.12
    )
    ax.add_patch(rs_bg)
    ax.text(0.485, y, rs, fontsize=11, fontweight='bold',
            color=rs_color, va='center', ha='center', fontfamily=SANS)

    ax.text(0.585, y, gt_text, fontsize=7.5, color=SUBTLE,
            va='center', fontfamily=SANS, style='italic')

    ax.annotate('', xy=(0.90, y), xytext=(0.74, y),
                arrowprops=dict(arrowstyle='->', color='#bdbdbd', lw=1.0))
    ax.text(0.90, y, verdict, fontsize=8.5, fontweight='bold',
            color=verdict_color, va='center', ha='center',
            fontfamily=SANS, style='italic')


# ===== FIGURE =====
fig_h = 11.8
fig = plt.figure(figsize=(15, fig_h), facecolor='white')

ax1 = fig.add_axes([0.01, 0.02, 0.48, 0.88])
ax2 = fig.add_axes([0.51, 0.02, 0.48, 0.88])

for ax in [ax1, ax2]:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


# ===== PANEL A =====
border1 = mpatches.FancyBboxPatch(
    (0.01, 0.01), 0.98, 0.98,
    boxstyle="round,pad=0.012", linewidth=2.2,
    edgecolor=CORRECT_GREEN, facecolor='white'
)
ax1.add_patch(border1)

hdr1 = mpatches.FancyBboxPatch(
    (0.01, 0.945), 0.98, 0.045,
    boxstyle="round,pad=0.005", linewidth=0,
    facecolor=CORRECT_GREEN, alpha=0.07
)
ax1.add_patch(hdr1)
ax1.text(0.04, 0.968, "(a)  Correct Answer, Poor Reasoning",
         fontsize=12.5, fontweight='bold', color=TEXT_COLOR,
         va='center', fontfamily=SERIF)
ax1.text(0.04, 0.952, "Gemma-7B  |  MATH500 #452",
         fontsize=8, color=SUBTLE, va='center', fontfamily=SANS)
badge1 = mpatches.FancyBboxPatch(
    (0.83, 0.955), 0.12, 0.03,
    boxstyle="round,pad=0.005", linewidth=1.3,
    edgecolor=CORRECT_GREEN, facecolor=CORRECT_GREEN, alpha=0.12
)
ax1.add_patch(badge1)
ax1.text(0.89, 0.970, "CORRECT",
         fontsize=8, fontweight='bold', color=CORRECT_GREEN,
         va='center', ha='center', fontfamily=SANS)

y = 0.935
ax1.text(0.04, y, "QUESTION", fontsize=6, fontweight='bold',
         color=SUBTLE, va='top', fontfamily=SANS)
y -= 0.015
q1 = textwrap.fill(
    "The sum of the digits of a two-digit number is 13. The difference "
    "between the number and the reversed number is 27. Find their sum.", width=78)
q1_lines = q1.count('\n') + 1
q1_h = q1_lines * 0.017 + 0.012
q_bg1 = mpatches.FancyBboxPatch(
    (0.03, y - q1_h), CODE_WIDTH, q1_h,
    boxstyle="round,pad=0.007", linewidth=0.5,
    edgecolor='#e0e0e0', facecolor='#fafafa'
)
ax1.add_patch(q_bg1)
ax1.text(0.045, y - 0.005, q1, fontsize=7.2, color=TEXT_COLOR,
         va='top', fontfamily=SERIF, style='italic')
y -= q1_h + 0.015

ax1.text(0.04, y, "MODEL OUTPUT (verbatim, lightly reformatted for width)",
         fontsize=6, fontweight='bold', color=SUBTLE, va='top', fontfamily=SANS)
y -= 0.025

bot1 = draw_code_block(ax1, 0.03, y, OUTPUT_A, errors=ERRORS_A)

draw_scores(ax1, bot1 - 0.008, pass1=1, pass1_correct=True,
            rs="2.1 / 5.0", rs_good=False,
            gt_text="Ground truth: 143",
            verdict="Misleading\nsuccess", verdict_color=WRONG_RED)


# ===== PANEL B =====
border2 = mpatches.FancyBboxPatch(
    (0.01, 0.01), 0.98, 0.98,
    boxstyle="round,pad=0.012", linewidth=2.2,
    edgecolor=WRONG_RED, facecolor='white'
)
ax2.add_patch(border2)

hdr2 = mpatches.FancyBboxPatch(
    (0.01, 0.945), 0.98, 0.045,
    boxstyle="round,pad=0.005", linewidth=0,
    facecolor=WRONG_RED, alpha=0.05
)
ax2.add_patch(hdr2)
ax2.text(0.04, 0.968, "(b)  Wrong Answer, Strong Reasoning",
         fontsize=12.5, fontweight='bold', color=TEXT_COLOR,
         va='center', fontfamily=SERIF)
ax2.text(0.04, 0.952, "Phi-4-Reasoning  |  GSM8K #242",
         fontsize=8, color=SUBTLE, va='center', fontfamily=SANS)
badge2 = mpatches.FancyBboxPatch(
    (0.81, 0.955), 0.14, 0.03,
    boxstyle="round,pad=0.005", linewidth=1.3,
    edgecolor=WRONG_RED, facecolor=WRONG_RED, alpha=0.12
)
ax2.add_patch(badge2)
ax2.text(0.88, 0.970, "INCORRECT",
         fontsize=8, fontweight='bold', color=WRONG_RED,
         va='center', ha='center', fontfamily=SANS)

y = 0.935
ax2.text(0.04, y, "QUESTION", fontsize=6, fontweight='bold',
         color=SUBTLE, va='top', fontfamily=SANS)
y -= 0.015
q2 = textwrap.fill(
    "Mike has 5 pen pals, stops writing to 2. Each remaining pal sends "
    "2 letters/week (5 pages each). He responds in kind. He writes a page "
    "in 6 min. How many hours does he spend writing per week?", width=78)
q2_lines = q2.count('\n') + 1
q2_h = q2_lines * 0.017 + 0.012
q_bg2 = mpatches.FancyBboxPatch(
    (0.03, y - q2_h), CODE_WIDTH, q2_h,
    boxstyle="round,pad=0.007", linewidth=0.5,
    edgecolor='#e0e0e0', facecolor='#fafafa'
)
ax2.add_patch(q_bg2)
ax2.text(0.045, y - 0.005, q2, fontsize=7.2, color=TEXT_COLOR,
         va='top', fontfamily=SERIF, style='italic')
y -= q2_h + 0.015

ax2.text(0.04, y, "MODEL OUTPUT -- CHAIN-OF-THOUGHT",
         fontsize=6, fontweight='bold', color=THINK_BLUE, va='top', fontfamily=SANS)
y -= 0.013

bot2a = draw_code_block(ax2, 0.03, y, OUTPUT_B_THINK)

y = bot2a - 0.015
ax2.text(0.04, y, "MODEL OUTPUT -- FINAL RESPONSE",
         fontsize=6, fontweight='bold', color=WRONG_RED, va='top', fontfamily=SANS)
y -= 0.013

bot2b = draw_code_block(ax2, 0.03, y, OUTPUT_B_RESP, errors=ERRORS_B_RESP)

draw_scores(ax2, bot2b - 0.008, pass1=0, pass1_correct=False,
            rs="4.1 / 5.0", rs_good=True,
            gt_text="Ground truth: 3",
            verdict="Hidden\ncapability", verdict_color=CORRECT_GREEN)


# ===== TITLE =====
fig.text(0.5, 0.955,
         "Pass@1 cannot distinguish reasoning quality from answer correctness",
         fontsize=14, fontweight='bold', color=TEXT_COLOR,
         ha='center', va='center', fontfamily=SERIF)
fig.text(0.5, 0.935,
         "Two real examples from our evaluation where accuracy-based metrics give misleading signals",
         fontsize=9, color=SUBTLE, ha='center', va='center', fontfamily=SANS)

for fmt in ["pdf", "png"]:
    path = f"figures/fig1_pass1_vs_reasoning.{fmt}"
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {path}")
