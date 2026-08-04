#!/bin/bash
# Full CoT analysis: aqua + svamp
# - 5 files in parallel, 20 samples per file
# - Verbose logging, results saved per-file and global_summary per dataset

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python run_cot_analysis_aqua_svamp.py \
  --no-extract \
  --verbose \
  --files-parallel 5 \
  --samples-parallel 20 \
  "$@"
