#!/bin/bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ll5582/RoboticsNAS/logs/%x-%j.out
source /etc/profile
set -eo pipefail
cd /scratch/ll5582/RoboticsNAS
export HF_HOME=/scratch/ll5582/RoboticsNAS/cache/huggingface PYTHONPATH=$PWD PATH=$PWD/env/bin:$PATH
echo "host=$(hostname) commit=$(git rev-parse --short HEAD)"
python -m "$@"
