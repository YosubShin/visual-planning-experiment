#!/bin/bash
#SBATCH --job-name=visual-planning-experiment
#SBATCH --partition=kill-shared
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h200_nvl:1

set -euo pipefail

module purge >/dev/null 2>&1 || true
module load lang/Python/3.11.5-GCCcore-13.2.0 >/dev/null 2>&1 || true
module load system/CUDA/12.2.0 >/dev/null 2>&1 || true


RESULTS_ROOT="${KOA_ML_RESULTS_ROOT:-$HOME/koa-results}"
JOB_DIR="${KOA_RUN_DIR:-${RESULTS_ROOT}/${SLURM_JOB_ID}}"
REPO_DIR="${JOB_DIR}/repo"
RESULTS_DIR="${JOB_DIR}/results"
mkdir -p "${REPO_DIR}"
mkdir -p "${RESULTS_DIR}"

if [[ -d "${REPO_DIR}" ]]; then
  cd "${REPO_DIR}"
fi

echo "Writing outputs to ${RESULTS_DIR}"

echo "==== Job Info ====="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "KOA_ML_CODE_ROOT=${KOA_ML_CODE_ROOT:-unset}"
echo "KOA_RUN_DIR=${KOA_RUN_DIR:-unset}"
echo "KOA_RUN_METADATA_DIR=${KOA_RUN_METADATA_DIR:-unset}"
echo "KOA_SHARED_ENV=${KOA_SHARED_ENV:-unset}"
echo

echo "==== GPU Info ====="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not available"
fi
echo

echo "==== Python Environment ====="
if command -v python >/dev/null 2>&1; then
  which python
  python --version
else
  echo "python not found"
fi
echo

source "scripts/setup_env.sh"

srun uv run python -m frozenlake_benchmark.src.run_vlm_eval \
  --backend transformers \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --dataset frozenlake_benchmark/data/test.jsonl \
  --variant ascii \
  --limit 1000 \
  --max-new-tokens 2048 \
  --batch-size 32 \
  --output ../results/qwen25vl_7b_local.jsonl
