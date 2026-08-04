#!/usr/bin/env python3
"""
Build a single HTML (or Markdown) document for human annotators:

  1. First section: full LLM judge rubric (from llm_judge_rubric_static.md by default).
  2. Then: every row of validation_500_samples_full.csv with question, ground truth,
     model output, full prompt, and all three judges' pillar scores.

Usage:
  python analysis/export_human_annotation_packet.py
  python analysis/export_human_annotation_packet.py --out experiments/paper_figures/judge_validation/human_annotation_packet.html
  python analysis/export_human_annotation_packet.py --format md --max-samples 5

Outputs are large (~tens of MB HTML); open in a browser and Print to PDF if needed.
"""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO / "experiments" / "paper_figures" / "judge_validation" / "validation_500_samples_full.csv"
DEFAULT_RUBRIC = REPO / "experiments" / "paper_figures" / "judge_validation" / "llm_judge_rubric_static.md"


def escape(s: str) -> str:
    return html.escape(s or "", quote=True)


def md_simple_to_html(md: str) -> str:
    """Very small subset: # headers, ## headers, **bold**, ---, blank lines -> paragraphs."""
    lines = md.split("\n")
    out = []
    in_ul = False
    for line in lines:
        s = line.rstrip()
        if s.startswith("### "):
            if in_ul:
                out.append("</ul>")
                in_ul = False
            out.append(f"<h3>{escape(s[4:])}</h3>")
        elif s.startswith("## "):
            if in_ul:
                out.append("</ul>")
                in_ul = False
            out.append(f"<h2>{escape(s[3:])}</h2>")
        elif s.startswith("# "):
            if in_ul:
                out.append("</ul>")
                in_ul = False
            out.append(f"<h1>{escape(s[2:])}</h1>")
        elif s.strip() == "---":
            if in_ul:
                out.append("</ul>")
                in_ul = False
            out.append("<hr/>")
        elif s.startswith("- "):
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            body = s[2:]
            body = body.replace("**", "")  # strip bold markers for simplicity
            out.append(f"<li>{escape(body)}</li>")
        elif not s.strip():
            if in_ul:
                out.append("</ul>")
                in_ul = False
            out.append("<p></p>")
        else:
            if in_ul:
                out.append("</ul>")
                in_ul = False
            t = s.replace("**", "")
            out.append(f"<p>{escape(t)}</p>")
    if in_ul:
        out.append("</ul>")
    return "\n".join(out)


def write_html(
    out_path: Path,
    rubric_md: str,
    rows: list[dict],
) -> None:
    css = """
    body { font-family: Georgia, "Times New Roman", serif; margin: 2rem; line-height: 1.45; max-width: 52rem; }
    .rubric { border: 1px solid #333; padding: 1.25rem; margin-bottom: 2rem; background: #fafafa; }
    .sample { border-top: 2px solid #222; padding: 1.5rem 0; page-break-inside: avoid; }
    .meta { font-size: 0.95rem; color: #333; margin-bottom: 0.75rem; }
    pre { white-space: pre-wrap; word-break: break-word; background: #f4f4f4; padding: 0.75rem;
          border: 1px solid #ccc; font-size: 0.88rem; }
    h1 { font-size: 1.5rem; }
    h2 { font-size: 1.2rem; margin-top: 1.25rem; }
    table.scores { border-collapse: collapse; margin: 0.75rem 0; font-size: 0.9rem; }
    table.scores th, table.scores td { border: 1px solid #999; padding: 0.35rem 0.6rem; text-align: center; }
    table.scores th { background: #eee; }
    .hint { font-size: 0.85rem; color: #555; }
    @media print {
      .sample { page-break-before: always; }
      .sample.first { page-break-before: auto; }
    }
    """
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Human annotation packet — 500 stratified samples</title>",
        f"<style>{css}</style></head><body>",
        "<h1>Human annotation packet</h1>",
        "<p class='hint'>Section 1: rubric aligned with <code>backend/app/cot_eval_v2/judge.py</code>. "
        "Section 2: stratified validation rows from <code>validation_500_samples_full.csv</code>.</p>",
        "<div class='rubric'>",
        "<h2>1. Scoring rubric (LLM judges)</h2>",
        md_simple_to_html(rubric_md),
        "</div>",
        "<h2>2. Samples (one block per row)</h2>",
    ]

    for i, row in enumerate(rows):
        m = row.get("model", "")
        ds = row.get("dataset", "")
        idx = row.get("idx", "")
        cls = "sample first" if i == 0 else "sample"
        parts.append(f"<div class='{cls}' id='s-{escape(str(ds))}-{escape(str(idx))}-{escape(str(m))}'>")
        parts.append(
            f"<div class='meta'><strong>#{i+1}</strong> &nbsp;|&nbsp; Model: <strong>{escape(m)}</strong> "
            f"&nbsp;|&nbsp; Dataset: <code>{escape(str(ds))}</code> &nbsp;|&nbsp; idx: <code>{escape(str(idx))}</code></div>"
        )

        parts.append("<h3>Ground truth</h3>")
        parts.append(f"<pre>{escape(row.get('ground_truth', ''))}</pre>")

        parts.append("<h3>Question</h3>")
        parts.append(f"<pre>{escape(row.get('question', ''))}</pre>")

        parts.append("<h3>LLM judge scores (1–5 each pillar)</h3>")
        parts.append("<table class='scores'><thead><tr>")
        parts.append("<th>Judge</th><th>Faithfulness</th><th>Utility</th><th>Coherence</th><th>Factuality</th><th>Mean</th>")
        parts.append("</tr></thead><tbody>")

        def mean4(a, b, c, d):
            try:
                vals = [float(x) for x in (a, b, c, d) if x not in ("", None)]
                if len(vals) != 4:
                    return ""
                return f"{sum(vals)/4:.2f}"
            except (ValueError, TypeError):
                return ""

        judges = [
            ("GPT-4o-mini", "mini_faith", "mini_utili", "mini_coher", "mini_factu"),
            ("GPT-4o", "gpt4o_faith", "gpt4o_utili", "gpt4o_coher", "gpt4o_factu"),
            ("Claude", "claude_faith", "claude_utili", "claude_coher", "claude_factu"),
        ]
        for label, kf, ku, kc, kx in judges:
            vf, vu, vc, vx = row.get(kf, ""), row.get(ku, ""), row.get(kc, ""), row.get(kx, "")
            mu = mean4(vf, vu, vc, vx)
            parts.append(
                "<tr>"
                f"<td>{escape(label)}</td>"
                f"<td>{escape(str(vf))}</td><td>{escape(str(vu))}</td>"
                f"<td>{escape(str(vc))}</td><td>{escape(str(vx))}</td>"
                f"<td>{escape(mu)}</td>"
                "</tr>"
            )
        parts.append("</tbody></table>")

        parts.append("<h3>Full prompt to model</h3>")
        parts.append(f"<pre>{escape(row.get('full_prompt', ''))}</pre>")

        parts.append("<h3>Model output (CoT / answer)</h3>")
        parts.append(f"<pre>{escape(row.get('model_output', ''))}</pre>")

        parts.append(
            "<p class='hint'><em>Human scores (optional):</em> "
            "Faithfulness ___ Utility ___ Coherence ___ Factuality ___ &nbsp; Notes: _______________</p>"
        )
        parts.append("</div>")

    parts.append("</body></html>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def write_md(out_path: Path, rubric_md: str, rows: list[dict]) -> None:
    chunks = ["# Human annotation packet\n\n", "## Part A — Rubric\n\n", rubric_md, "\n\n---\n\n## Part B — Samples\n\n"]
    for i, row in enumerate(rows):
        m, ds, idx = row.get("model", ""), row.get("dataset", ""), row.get("idx", "")
        chunks.append(f"### Sample {i+1}: {m} | {ds} | idx={idx}\n\n")
        chunks.append(f"**Ground truth:** `{row.get('ground_truth', '')}`\n\n")
        chunks.append("**Question:**\n\n```\n")
        chunks.append(row.get("question", "") or "")
        chunks.append("\n```\n\n")
        chunks.append("| Judge | Faith | Util | Coher | Fact | Mean |\n|---|---|---|---|---|---|\n")
        def mean4(a, b, c, d):
            try:
                vals = [float(x) for x in (a, b, c, d) if x not in ("", None)]
                if len(vals) != 4:
                    return ""
                return f"{sum(vals)/4:.2f}"
            except (ValueError, TypeError):
                return ""
        for label, kf, ku, kc, kx in [
            ("GPT-4o-mini", "mini_faith", "mini_utili", "mini_coher", "mini_factu"),
            ("GPT-4o", "gpt4o_faith", "gpt4o_utili", "gpt4o_coher", "gpt4o_factu"),
            ("Claude", "claude_faith", "claude_utili", "claude_coher", "claude_factu"),
        ]:
            vf, vu, vc, vx = row.get(kf, ""), row.get(ku, ""), row.get(kc, ""), row.get(kx, "")
            mu = mean4(vf, vu, vc, vx)
            chunks.append(f"| {label} | {vf} | {vu} | {vc} | {vx} | {mu} |\n")
        chunks.append("\n**Full prompt:**\n\n```\n")
        chunks.append(row.get("full_prompt", "") or "")
        chunks.append("\n```\n\n**Model output:**\n\n```\n")
        chunks.append(row.get("model_output", "") or "")
        chunks.append("\n```\n\n---\n\n")
    out_path.write_text("".join(chunks), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export human annotation HTML/MD packet")
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Stratified 500 CSV")
    ap.add_argument("--rubric", type=Path, default=DEFAULT_RUBRIC, help="Rubric markdown file")
    ap.add_argument("--out", type=Path, default=None, help="Output path (.html or .md)")
    ap.add_argument("--format", choices=("html", "md"), default="html")
    ap.add_argument("--max-samples", type=int, default=0, help="If >0, only first N rows (debug)")
    args = ap.parse_args()

    rubric_md = args.rubric.read_text(encoding="utf-8")
    rows = []
    with args.csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    if args.out is None:
        ext = "html" if args.format == "html" else "md"
        args.out = REPO / "experiments" / "paper_figures" / "judge_validation" / f"human_annotation_packet.{ext}"

    if args.format == "html":
        write_html(args.out, rubric_md, rows)
    else:
        write_md(args.out, rubric_md, rows)

    print(f"Wrote {args.out} ({len(rows)} samples)")


if __name__ == "__main__":
    main()
