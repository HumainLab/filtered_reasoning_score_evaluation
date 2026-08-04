#!/usr/bin/env python3
"""
Export canonical pass@16 JSONL files (54 model×benchmark pairs) into a zip bundle.

Uses discover_jsonl_groups() — same paths as topk_ablation / PRM audit.

Usage:
  python analysis/export_pass16_canonical_zip.py --repo-root .
  python analysis/export_pass16_canonical_zip.py --math-only   # 9×4 math pairs only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
MATH_BENCHES = {"GSM8K", "MATH500", "SVAMP", "AQuA"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--math-only", action="store_true", help="Export 9×4 math pairs only")
    ap.add_argument("--compression", choices=["stored", "deflated"], default="deflated")
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    out_dir = (args.output_dir or repo / "outputs/analysis_outputs" / "pass16_prm_export").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(repo))
    from build_downstream_parquets import discover_jsonl_groups  # noqa: E402

    file_map = discover_jsonl_groups(str(repo))
    if args.math_only:
        file_map = {k: v for k, v in file_map.items() if k[1] in MATH_BENCHES}

    suffix = "math36" if args.math_only else "all54"
    zip_path = out_dir / f"pass16_canonical_{suffix}.zip"
    manifest_path = out_dir / f"manifest_{suffix}.json"

    entries = []
    total_bytes = 0
    for (model, benchmark), src in sorted(file_map.items()):
        src_path = Path(src).resolve()
        if not src_path.is_file():
            print(f"MISSING: {src_path}", file=sys.stderr)
            return 1
        n_q = 0
        n16 = 0
        with open(src_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                n_q += 1
                sc = row.get("score", [])
                if isinstance(sc, list) and len(sc) == 16:
                    n16 += 1
        size = src_path.stat().st_size
        total_bytes += size
        arcname = f"pass16_canonical/jsonl/{model}__{benchmark}.jsonl"
        entries.append(
            {
                "model": model,
                "benchmark": benchmark,
                "source_path": str(src_path),
                "zip_path": arcname,
                "bytes": size,
                "n_questions": n_q,
                "n_questions_with_16_traces": n16,
                "coverage_16": n16 / n_q if n_q else 0,
            }
        )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo),
        "math_only": args.math_only,
        "n_pairs": len(entries),
        "total_bytes": total_bytes,
        "schema": "A (one row/question; code[], pred[], score[], chosen_token_probs_per_path.epoch_0[] length 16)",
        "entries": entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    compression = zipfile.ZIP_STORED if args.compression == "stored" else zipfile.ZIP_DEFLATED
    readme = f"""# pass@16 canonical JSONL bundle ({suffix})

Created: {manifest['created_at']}
Pairs: {len(entries)}
Uncompressed total: {total_bytes / 1e9:.2f} GB

## Layout

- `jsonl/<Model>__<Benchmark>.jsonl` — one file per model×benchmark pair
- `manifest.json` — source paths and question counts

## Schema (per line)

One JSON object per question with list fields of length 16:
- `idx`, `prompt`, `gt`
- `code[i]` — full chain-of-thought text for trace i
- `pred[i]`, `score[i]` — prediction and correctness
- `chosen_token_probs_per_path['epoch_0'][i]` — token probabilities

## Usage

```python
from build_downstream_parquets import discover_jsonl_groups  # if repo available
# Or read directly:
import json
with open("jsonl/DS-R1-1.5B__GSM8K.jsonl") as f:
    row = json.loads(f.readline())
    cot_trace_3 = row["code"][3]
```
"""

    print(f"Writing {zip_path} ({len(entries)} files, {total_bytes/1e9:.2f} GB uncompressed)...")
    with zipfile.ZipFile(zip_path, "w", compression=compression, compresslevel=1) as zf:
        zf.writestr("pass16_canonical/README.md", readme)
        zf.writestr("pass16_canonical/manifest.json", json.dumps(manifest, indent=2))
        for e in tqdm(entries, desc="Zipping"):
            zf.write(e["source_path"], arcname=e["zip_path"])

    zip_size = zip_path.stat().st_size
    print(f"Done: {zip_path}")
    print(f"  Zip size: {zip_size/1e9:.2f} GB ({100*zip_size/max(1,total_bytes):.0f}% of raw)")
    print(f"  Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
