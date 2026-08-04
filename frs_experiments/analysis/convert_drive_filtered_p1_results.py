#!/usr/bin/env python3
"""
Import pre-judged filtered_p1_only_results.json (Drive export) into FRS checkpoint + CSV format.

Does not call the judge API — source files already contain judge_scores from the
same $PORTKEY_MODEL_PREFIX/gpt-4o-mini pipeline.

Usage:
  python analysis/convert_drive_filtered_p1_results.py \\
    --input-dir results/7b/drive_download_20260529 \\
    --output-dir results/7b
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from topk_judge_eval import reasoning_score_from_judge  # noqa: E402

PAPER_MODEL = {
    "Qwen2.5-Math-7B": "Qwen2.5-Math",
    "Phi-4-reasoning": "Phi-4-Reas.",
    "Qwen3-4B-Thinking-2507": "Qwen3-4B",
    "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "gemma-7b": "Gemma-7B",
    "Llama-3.1-8B-Instruct": "LLaMA-3.1-8B",
    "phi-4": "Phi-4",
}


def reasoning_score_from_row(row: Dict[str, Any]) -> Optional[float]:
    js = row.get("judge_scores")
    if isinstance(js, dict) and js:
        rs = reasoning_score_from_judge(js)
        if rs is not None:
            return float(rs)
    fs = row.get("fused_scores") or {}
    ov = fs.get("overall")
    if ov is not None:
        return float(ov)
    return None


def convert_file(src: Path, out_dir: Path) -> Dict[str, Any]:
    data = json.loads(src.read_text(encoding="utf-8"))
    raw_model = data.get("model", src.stem.split("_filtered")[0])
    paper_model = PAPER_MODEL.get(raw_model, raw_model)
    stem = src.stem.replace("_results", "")

    judged_samples: List[Dict[str, Any]] = []
    for r in data.get("results", []):
        rs = reasoning_score_from_row(r)
        ok = r.get("status") == "ok" and rs is not None
        judged_samples.append(
            {
                "idx": int(r["idx"]),
                "trace_idx": int(r["trace_idx"]),
                "bin_label": "filtered_p1",
                "confidence": r.get("confidence"),
                "correct": bool(r.get("original_correct", False)),
                "judge_scores": r.get("judge_scores"),
                "reasoning_score": round(rs, 4) if rs is not None else None,
                "judge_ok": ok,
                "error": "" if ok else str(r.get("status", "unknown")),
                "pred": r.get("pred"),
            }
        )

    ok_scores = [s["reasoning_score"] for s in judged_samples if s.get("judge_ok")]
    mean_rs = sum(ok_scores) / len(ok_scores) if ok_scores else float("nan")
    mean_pct = 100.0 * mean_rs if ok_scores else float("nan")

    ck = {
        "metadata": {
            "model": paper_model,
            "raw_model": raw_model,
            "benchmark": "filtered_p1",
            "source_file": str(src.resolve()),
            "imported_from": "drive_filtered_p1_only_results.json",
            "bin_label": "filtered_p1",
            "top_frac": data.get("top_frac"),
            "judge_model": data.get("judge_model"),
            "n_traces_total": len(judged_samples),
            "n_judged_ok": len(ok_scores),
            "n_judged_fail": len(judged_samples) - len(ok_scores),
            "frs_pct_source": data.get("frs_pct"),
            "mean_reasoning_score": mean_rs,
            "mean_reasoning_score_pct": mean_pct,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "judged_samples": judged_samples,
    }

    out_json = out_dir / f"judged_{stem}.json"
    out_csv = out_dir / f"judged_{stem}.csv"
    out_json.write_text(json.dumps(ck, indent=2, ensure_ascii=False), encoding="utf-8")

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        fields = [
            "idx",
            "trace_idx",
            "confidence",
            "correct",
            "faithfulness",
            "utility",
            "coherence",
            "factuality",
            "reasoning_score",
            "judge_ok",
            "error",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in judged_samples:
            js = s.get("judge_scores") or {}
            w.writerow(
                {
                    "idx": s["idx"],
                    "trace_idx": s["trace_idx"],
                    "confidence": s.get("confidence"),
                    "correct": s.get("correct"),
                    "faithfulness": js.get("faithfulness"),
                    "utility": js.get("utility"),
                    "coherence": js.get("coherence"),
                    "factuality": js.get("factuality"),
                    "reasoning_score": s.get("reasoning_score"),
                    "judge_ok": s.get("judge_ok"),
                    "error": s.get("error", ""),
                }
            )

    return {
        "paper_model": paper_model,
        "raw_model": raw_model,
        "n_total": len(judged_samples),
        "n_ok": len(ok_scores),
        "mean_reasoning_score_pct": mean_pct,
        "frs_pct_source": data.get("frs_pct"),
        "out_json": str(out_json),
        "out_csv": str(out_csv),
    }


def load_local_judged(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    m = d.get("metadata", {})
    ok = [s for s in d.get("judged_samples", []) if s.get("judge_ok")]
    rs = [s["reasoning_score"] for s in ok if s.get("reasoning_score") is not None]
    mean_pct = 100.0 * sum(rs) / len(rs) if rs else float("nan")
    return {
        "paper_model": m.get("model"),
        "raw_model": path.stem,
        "n_total": len(d.get("judged_samples", [])),
        "n_ok": len(ok),
        "mean_reasoning_score_pct": mean_pct,
        "frs_pct_source": mean_pct,
        "out_json": str(path),
        "out_csv": str(path.with_suffix(".csv")),
        "source": "local_frs_judge_run",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "results/7b/drive_download_20260529",
    )
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/7b")
    args = ap.parse_args()

    in_dir = args.input_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for src in sorted(in_dir.glob("*_filtered_p1_only_results.json")):
        info = convert_file(src, out_dir)
        info["source"] = "drive_import"
        rows.append(info)
        print(f"Imported {info['paper_model']}: {info['n_ok']}/{info['n_total']} ok, mean={info['mean_reasoning_score_pct']:.2f}%")

    for stem, paper in [
        ("DeepSeek-R1-Distill-Qwen-7B_filtered_p1_only", "DS-R1-7B"),
        ("DeepSeek-R1-Distill-Qwen-1.5B_filtered_p1_only", "DS-R1-1.5B"),
    ]:
        p = out_dir / f"judged_{stem}.json"
        info = load_local_judged(p)
        if info:
            rows.append(info)
            print(f"Included {paper}: {info['n_ok']}/{info['n_total']} ok, mean={info['mean_reasoning_score_pct']:.2f}%")

    summary_csv = out_dir / "filtered_p1_all_models_summary.csv"
    fields = [
        "paper_model",
        "raw_model",
        "n_total",
        "n_ok",
        "n_fail",
        "mean_reasoning_score_pct",
        "frs_pct_source",
        "source",
        "out_json",
    ]
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(rows, key=lambda x: -x["mean_reasoning_score_pct"]):
            w.writerow(
                {
                    "paper_model": r["paper_model"],
                    "raw_model": r["raw_model"],
                    "n_total": r["n_total"],
                    "n_ok": r["n_ok"],
                    "n_fail": r["n_total"] - r["n_ok"],
                    "mean_reasoning_score_pct": r["mean_reasoning_score_pct"],
                    "frs_pct_source": r.get("frs_pct_source"),
                    "source": r["source"],
                    "out_json": r["out_json"],
                }
            )

    summary_md = out_dir / "filtered_p1_all_models_summary.md"
    lines = [
        "# Filtered top-10% (p1-only) — all models",
        "",
        f"Imported from `{in_dir}` + local DeepSeek judge runs.",
        "",
        "| Model | Judged OK | Total | Mean RS (%) | Source frs_pct | Source |",
        "|-------|-----------|-------|-------------|----------------|--------|",
    ]
    for r in sorted(rows, key=lambda x: -x["mean_reasoning_score_pct"]):
        lines.append(
            f"| {r['paper_model']} | {r['n_ok']} | {r['n_total']} | "
            f"{r['mean_reasoning_score_pct']:.2f} | {r.get('frs_pct_source', ''):.2f} | {r['source']} |"
        )
    lines += ["", f"CSV: `{summary_csv}`", ""]
    summary_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nSummary: {summary_csv}")
    print(f"Summary: {summary_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
