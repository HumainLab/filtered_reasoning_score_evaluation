#!/bin/bash
#SBATCH -J prm-qwen256-baseline
#SBATCH -o /home1/10757/manasp123/reasoning-models-eval/backend/logs/prm-baseline-%j.out
#SBATCH -e /home1/10757/manasp123/reasoning-models-eval/backend/logs/prm-baseline-%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 4:00:00
#SBATCH --mail-type=END,FAIL

# Allocation (Lonestar example used elsewhere in this repo; edit if yours differs).
#SBATCH -A ECS26006

# Do NOT commit secrets: set HF_TOKEN in the submission environment instead of here.
# Example: HF_TOKEN=$(cat ~/.hf_token) sbatch ...

export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Python from your conda env (match backend/path_config.json `python_path` locally; do not commit keys).
# Lonestar work is /work/10757/$USER/, not /work/$USER/.
PYTHON="${PYTHON:-/work/10757/$USER/ls6/miniconda3/envs/math_eval/bin/python}"
REPO="/home1/10757/manasp123/reasoning-models-eval"
OUT="${PRM_OUTPUT_DIR:-$REPO/analysis_outputs/prm_qwen225_baseline}"
FILTER="${FILTERED_ROOT:-/work/$USER/ls6/home-offload-from-home1/reasoning-models-eval}"

cd "$REPO"

exec "$PYTHON" -u "$REPO/analysis/run_prm_baseline.py" \
  --prm-model "Qwen/Qwen2.5-Math-PRM-7B" \
  --filtered-root "$FILTER" \
  --output-dir "$OUT" \
  --benchmarks gsm8k math500 svamp aqua \
  --models all \
  --questions-per-pair 50 \
  --traces-per-question 16 \
  --max-length 4096 \
  --batch-size 1
