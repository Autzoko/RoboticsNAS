#!/bin/bash
#SBATCH --job-name=rnas-env
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ll5582/RoboticsNAS/logs/%x-%j.out
# Builds the RoboticsNAS python env: pinned LeRobot (3f2c29e) + SmolVLA + LIBERO sim.
source /etc/profile
set -euo pipefail
module purge
module load miniconda-nobashrc/3-4.11.0 gcc/11.5.0
export CC=gcc CXX=g++

ROOT=/scratch/ll5582/RoboticsNAS
ENV=$ROOT/env
export HF_HOME=$ROOT/cache/huggingface
export PIP_NO_CACHE_DIR=1
export CONDA_PKGS_DIRS="${SLURM_TMPDIR:-/tmp}/conda-pkgs"
mkdir -p "$CONDA_PKGS_DIRS"

[ -x "$ENV/bin/python" ] || conda create -y -p "$ENV" python=3.12 pip
PY=$ENV/bin/python
$PY -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
$PY -m pip install -e "$ROOT/third_party/lerobot[smolvla,libero]"
$PY -m pip install scipy pandas pyarrow matplotlib
$PY -m pip check || true

# Pinned model snapshots. smolvla_base is pretrained on SO-100 community data (no LIBERO).
$PY - <<'PY'
from huggingface_hub import snapshot_download, HfApi
api = HfApi()
for rid in ("lerobot/smolvla_base", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"):
    sha = api.model_info(rid).sha
    print(rid, sha, snapshot_download(repo_id=rid, revision=sha))
PY
$PY -c "import torch,transformers,mujoco,robosuite,libero;print(torch.__version__,torch.version.cuda,transformers.__version__,mujoco.__version__,robosuite.__version__)"
conda clean -a -y >/dev/null 2>&1 || true
date -Is > $ROOT/.env_ready
# LIBERO assets (HF Hub) into project cache; package-local symlink (some LIBERO code ignores the config).
HF_HOME=$ROOT/cache/huggingface $PY -c "from libero.libero.utils.download_utils import download_assets_from_huggingface as d; d('$ROOT/cache/libero/assets')"
ln -sfn $ROOT/cache/libero/assets $ENV/lib/python3.12/site-packages/libero/libero/assets
