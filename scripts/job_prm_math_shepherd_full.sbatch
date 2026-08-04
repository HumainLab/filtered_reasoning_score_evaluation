#!/bin/bash
#SBATCH -J prm-shepherd-full
#SBATCH -o /home1/10757/manasp123/reasoning-models-eval/backend/logs/prm-shepherd-full-%j.out
#SBATCH -e /home1/10757/manasp123/reasoning-models-eval/backend/logs/prm-shepherd-full-%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH -A ECS26006

# Do not commit secrets: set HF_TOKEN in the submission env if the model hub requires it.

export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Lonestar work is /work/10757/$USER/, not /work/$USER/.
PYTHON="${PYTHON:-/work/10757/$USER/ls6/miniconda3/envs/math_eval/bin/python}"
REPO="/home1/10757/manasp123/reasoning-models-eval"
JSONL_DIR="/work/10757/manasp123/pass16_prm_export/pass16_canonical_math36/pass16_canonical/jsonl"
OUT="${PRM_OUTPUT_DIR:-$REPO/analysis_outputs/prm_math_shepherd_pass16}"

cd "$REPO"

exec "$PYTHON" -u "$REPO/analysis/run_prm_math_shepherd.py" \
  --prm-model "peiyi9979/math-shepherd-mistral-7b-prm" \
  --jsonl-dir "$JSONL_DIR" \
  --output-dir "$OUT"
