#!/bin/bash
# Run CoT reasoning evaluation on the GSM8K filtered set with:
# - 20 samples in parallel per file
# - 5 files in parallel
# Reads the judge API key from webui/backend/path_config.json (or $OPENAI_API_KEY).

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

python "$REPO_ROOT/frs/run_filtered_cot_eval.py" \
  --input-dir "$REPO_ROOT/results/filtered_cot/gsm8k" \
  --samples-parallel 20 \
  --files-parallel 5 \
  "$@"
