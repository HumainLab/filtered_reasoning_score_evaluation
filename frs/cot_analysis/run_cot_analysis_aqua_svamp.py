#!/usr/bin/env python3
"""
Run CoT analysis on results/filtered_cot/{aqua,svamp,commonsense}.

- If data is still in zips (drive-download-*.zip with nested *_filtered_p1_only.jsonl.zip),
  extracts them so each dir contains *_filtered_p1_only.jsonl files.
- Runs the four-pillar CoT evaluation (Faithfulness, Utility, Coherence, Factuality)
  via run_filtered_cot_eval.py on each directory in turn.
- Results go to filtered-cot-<name>/results/.

Usage:
  python run_cot_analysis_aqua_svamp.py
  python run_cot_analysis_aqua_svamp.py --datasets aqua          # only aqua
  python run_cot_analysis_aqua_svamp.py --datasets svamp         # only svamp
  python run_cot_analysis_aqua_svamp.py --datasets commonsense   # only commonsense
  python run_cot_analysis_aqua_svamp.py --dry-run                # extract only, no eval
  python run_cot_analysis_aqua_svamp.py --max-samples 10        # limit samples per file (testing)
  python run_cot_analysis_aqua_svamp.py --no-extract            # skip extraction, run eval only
"""

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
FILTERED_ROOT = REPO_ROOT / "results" / "filtered_cot"
DATASETS = ["aqua", "svamp", "commonsense"]  # dirs under results/filtered_cot/


def ensure_extracted(data_dir: Path) -> bool:
    """
    If data_dir only has drive-download-*.zip (and optionally .jsonl.zip files),
    extract so we have *_filtered_p1_only.jsonl. Return True if extraction was done or already present.
    """
    data_dir = data_dir.resolve()
    jsonl_files = list(data_dir.glob("*_filtered_p1_only.jsonl"))
    if jsonl_files:
        return True

    # Find outer zip
    outer_zips = list(data_dir.glob("drive-download-*.zip"))
    if not outer_zips:
        print(f"  No JSONL files and no drive-download-*.zip in {data_dir}", file=sys.stderr)
        return False

    outer_zip = outer_zips[0]
    print(f"  Extracting outer archive: {outer_zip.name}")
    with zipfile.ZipFile(outer_zip, "r") as z:
        z.extractall(data_dir)

    # Unzip each *_filtered_p1_only.jsonl.zip in place
    inner_zips = list(data_dir.glob("*_filtered_p1_only.jsonl.zip"))
    for inner in inner_zips:
        print(f"  Extracting {inner.name}")
        with zipfile.ZipFile(inner, "r") as z:
            z.extractall(data_dir)

    jsonl_files = list(data_dir.glob("*_filtered_p1_only.jsonl"))
    if not jsonl_files:
        print(f"  Still no JSONL files in {data_dir}", file=sys.stderr)
        return False
    print(f"  Found {len(jsonl_files)} JSONL files in {data_dir.name}")
    return True


def run_cot_eval(input_dir: Path, extra_args: list) -> int:
    """Run run_filtered_cot_eval.py for input_dir. Return exit code."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "frs" / "run_filtered_cot_eval.py"),
        "--input-dir", str(input_dir),
        "--files-parallel", "5",
        "--samples-parallel", "20",
    ] + extra_args
    print(f"\nRunning: {' '.join(cmd)}\n")
    return subprocess.run(cmd, cwd=str(SCRIPT_DIR)).returncode


def main():
    parser = argparse.ArgumentParser(
        description="Extract (if needed) and run CoT analysis on the aqua and svamp filtered sets",
    )
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS,
                        help=f"Which datasets to process (default: {' '.join(DATASETS)})")
    parser.add_argument("--no-extract", action="store_true",
                        help="Skip extraction; assume *_filtered_p1_only.jsonl already exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only extract (if needed); do not run CoT evaluation")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per file (for testing)")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging for eval")
    parser.add_argument("--files-parallel", type=int, default=5, help="Files in parallel (default: 5)")
    parser.add_argument("--samples-parallel", type=int, default=20, help="Samples per file in parallel (default: 20)")
    args = parser.parse_args()

    eval_extra = []
    if args.max_samples is not None:
        eval_extra.extend(["--max-samples", str(args.max_samples)])
    if args.verbose:
        eval_extra.append("--verbose")
    if args.files_parallel != 5:
        eval_extra.extend(["--files-parallel", str(args.files_parallel)])
    if args.samples_parallel != 20:
        eval_extra.extend(["--samples-parallel", str(args.samples_parallel)])

    for name in args.datasets:
        data_dir = FILTERED_ROOT / f"{name}"
        if not data_dir.is_dir():
            print(f"Skip {name}: directory not found: {data_dir}", file=sys.stderr)
            continue

        print(f"\n{'='*60}")
        print(f"  Dataset: {name} ({data_dir})")
        print(f"{'='*60}")

        if not args.no_extract and not ensure_extracted(data_dir):
            print(f"  Skipping eval for {name} (extraction failed).", file=sys.stderr)
            continue

        if args.dry_run:
            print(f"  Dry run: skipping CoT evaluation for {name}")
            continue

        code = run_cot_eval(data_dir, eval_extra)
        if code != 0:
            print(f"  CoT evaluation failed for {name} (exit code {code})", file=sys.stderr)
            sys.exit(code)

    print("\n  All requested datasets finished successfully.")


if __name__ == "__main__":
    main()
