#!/bin/bash
# Run CoT reasoning evaluation on filtered-cot-gsm8k with:
# - 20 samples in parallel per file
# - 5 files in parallel
# Uses GPT-4o-mini via backend/path_config.json

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python run_filtered_cot_eval.py \
  --input-dir filtered-cot-gsm8k \
  --samples-parallel 20 \
  --files-parallel 5 \
  "$@"
