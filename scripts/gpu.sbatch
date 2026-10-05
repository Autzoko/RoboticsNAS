#!/bin/bash
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ll5582/RoboticsNAS/logs/%x-%j.out
# Usage: sbatch [-J name] [--time ..] scripts/gpu.sbatch <python -m module> [args...]
# Set NPROC>1 (and --gres=gpu:a100:N) for torchrun data parallel.
source /etc/profile
set -eo pipefail
cd /scratch/ll5582/RoboticsNAS
export HF_HOME=/scratch/ll5582/RoboticsNAS/cache/huggingface
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH=$PWD
export LIBERO_CONFIG_PATH=$PWD/configs/libero
export PATH=$PWD/env/bin:$PATH
echo "host=$(hostname) gpus=$CUDA_VISIBLE_DEVICES commit=$(git rev-parse --short HEAD) start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ "${NPROC:-1}" -gt 1 ]; then
  torchrun --standalone --nproc_per_node=$NPROC -m "$@"
else
  if [ "$1" = "--sh" ]; then shift; bash "$@"; else python -m "$@"; fi
fi
echo "end=$(date -Is)"
