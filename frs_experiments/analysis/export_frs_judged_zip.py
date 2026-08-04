#!/usr/bin/env python3
"""
Export FRS judge checkpoints (54 pairs × 250 traces) for Drive sharing.

Bundle includes:
  - judged_<Model>__<Benchmark>.json (raw checkpoints)
  - all_judged_traces.csv (13,500 rows, 4 rubric dims + metadata)
  - per_pair_summary.csv (250-count verification + mean scores)
  - reasoning_by_confidence_bin.csv, reasoning_sampling_metadata.json
  - README.md, manifest.json

Usage:
  python analysis/export_frs_judged_zip.py --repo-root .
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
JUDGE_DIR = REPO_ROOT / "reasoning_confidence_bins_results" / "judging_checkpoints"
RESULTS_DIR = REPO_ROOT / "reasoning_confidence_bins_results"


def flatten_checkpoints(judge_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    pair_rows: list[dict] = []

    for path in sorted(judge_dir.glob("judged_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = data.get("metadata", {})
        model = meta.get("model", "")
        benchmark = meta.get("dataset") or meta.get("benchmark", "")
        samples = data.get("judged_samples", [])

        pair_rows.append(
            {
                "model": model,
                "benchmark": benchmark,
                "n_judged": len(samples),
                "source_file": path.name,
            }
        )

        for s in samples:
            js = s.get("judge_scores") or {}
            rows.append(
                {
                    "model": model,
                    "benchmark": benchmark,
                    "question_idx": s.get("idx"),
                    "trace_idx": s.get("trace_idx"),
                    "bin_label": s.get("bin_label"),
                    "confidence": s.get("confidence"),
                    "correct": s.get("correct"),
                    "faithfulness": js.get("faithfulness"),
                    "utility": js.get("utility"),
                    "coherence": js.get("coherence"),
                    "factuality": js.get("factuality"),
                    "reasoning_score": s.get("reasoning_score"),
                    "judge_ok": s.get("judge_ok"),
                    "source_file": path.name,
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(pair_rows)


def readme_text() -> str:
    return """# FRS judged trace export

## Contents

- `checkpoints/judged_<Model>__<Benchmark>.json` — 54 files, 250 judged traces each (13,500 total)
- `all_judged_traces.csv` — flat table with all traces and 4 rubric dimensions
- `per_pair_summary.csv` — row counts per model×benchmark
- `reasoning_by_confidence_bin.csv` — aggregated bin-level FRS statistics
- `reasoning_sampling_metadata.json` — bin definitions and sampling parameters

## Sampling protocol

For each model×benchmark pair:
1. Pool all pass@16 traces; rank by low-prob-tail confidence
2. Keep top 50% of traces by confidence
3. Split into 5 equal-count bins (0–10%, …, 40–50% within pool)
4. Judge 50 traces per bin → **250 judged traces per pair**

## Rubric dimensions (1–5 each)

- faithfulness, utility, coherence, factuality
- reasoning_score = normalized mean in [0, 1]

## Joining to raw CoT text

Use pass@16 JSONL (`source_pass16_jsonl_by_model*/**/*.jsonl`):
- question_idx = JSONL field `idx`
- trace_idx = index into `code[]`, `score[]`, `pred[]`

Judge checkpoints do **not** include raw chain-of-thought text.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    judge_dir = repo / "reasoning_confidence_bins_results" / "judging_checkpoints"
    results_dir = repo / "reasoning_confidence_bins_results"
    out_dir = (args.output_dir or repo / "analysis_outputs" / "frs_judged_export").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not judge_dir.is_dir():
        print(f"Missing {judge_dir}", file=sys.stderr)
        return 1

    flat, pairs = flatten_checkpoints(judge_dir)
    assert len(pairs) == 54, f"expected 54 pairs, got {len(pairs)}"
    assert len(flat) == 13500, f"expected 13500 traces, got {len(flat)}"

    staging = out_dir / "_staging"
    staging.mkdir(exist_ok=True)
    ckpt_staging = staging / "checkpoints"
    ckpt_staging.mkdir(exist_ok=True)

    flat.to_csv(staging / "all_judged_traces.csv", index=False)
    pairs.to_csv(staging / "per_pair_summary.csv", index=False)
    (staging / "README.md").write_text(readme_text(), encoding="utf-8")

    for src in tqdm(sorted(judge_dir.glob("judged_*.json")), desc="copy checkpoints"):
        dst = ckpt_staging / src.name
        dst.write_bytes(src.read_bytes())

    for extra in ("reasoning_by_confidence_bin.csv", "reasoning_sampling_metadata.json"):
        p = results_dir / extra
        if p.exists():
            (staging / extra).write_bytes(p.read_bytes())

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_pairs": len(pairs),
        "n_traces": len(flat),
        "judged_per_pair": int(pairs["n_judged"].iloc[0]),
        "columns_csv": list(flat.columns),
        "files": sorted(f.name for f in staging.rglob("*") if f.is_file()),
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    zip_path = out_dir / "frs_judged_all54.zip"
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in sorted(staging.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(staging).as_posix())

    zip_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {zip_path} ({zip_mb:.2f} MB)")
    print(f"  {len(pairs)} pairs, {len(flat)} judged traces")
    print(f"  Staging kept at {staging}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
