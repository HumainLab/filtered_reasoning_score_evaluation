#!/bin/bash
# Full CoT analysis: aqua + svamp + commonsense
# - 5 files in parallel, 20 samples per file
# - Verbose logging; results saved per-file plus a global_summary per dataset

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

python "$REPO_ROOT/frs/cot_analysis/run_cot_analysis_aqua_svamp.py" \
  --no-extract \
  --verbose \
  --files-parallel 5 \
  --samples-parallel 20 \
  "$@"
