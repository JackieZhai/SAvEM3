#!/usr/bin/env bash
# Install the SAvEM³ reproduction environment on a GPU server (conda + CUDA).
# Use a new environment name to preserve historical runs. Native postprocessing is separate.
set -euo pipefail

ENV_NAME="${1:-savem3}"
CONDA="${CONDA:-conda}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo "[1/4] Create the conda environment (Python 3.12 + PyTorch)"
"$CONDA" create -y -n "$ENV_NAME" python=3.12 pip

# Override with the official CPU index when no CUDA device is available.
SAVEM3_TORCH_INDEX_URL="${SAVEM3_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
"$CONDA" run -n "$ENV_NAME" python -m pip install --upgrade pip setuptools wheel
"$CONDA" run -n "$ENV_NAME" python -m pip install torch==2.14.1 torchvision==0.29.1 \
    --index-url "$SAVEM3_TORCH_INDEX_URL"

echo "[2/4] Install core dependencies"
"$CONDA" run -n "$ENV_NAME" python -m pip install -r "$ROOT/requirements-train.txt"

echo "[3/4] Check dependency consistency"
"$CONDA" run -n "$ENV_NAME" python -m pip check

echo "[4/4] Validate the installation"
"$CONDA" run -n "$ENV_NAME" python -c \
    "import torch, torchvision, numpy, skimage, h5py, yaml; print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'cuda', torch.cuda.is_available())"

echo "Done. Example next commands:"
echo "  conda activate $ENV_NAME"
echo "  cd /path/to/SAvEM3/savem3"
echo "  python main_devoem_sparse_membrane_triplet_2.py -c mem3c2c_3ds_t3t -m train"
echo "Install ELF/WaterZ separately using repro/env/environment-post.yml and requirements-post.txt."
