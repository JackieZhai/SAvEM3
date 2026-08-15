#!/usr/bin/env bash
# SAvEM³ 复现环境安装（在 GPU 服务器上执行，conda + CUDA）
# 参考：Watershed/README.md 与论文 Implementation Details（Py3.9、V100/A100）
set -euo pipefail

ENV_NAME="${1:-savem3}"
CONDA="${CONDA:-conda}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo "[1/4] 创建 conda 环境（Python 3.9 + PyTorch CUDA）"
$CONDA create -y -n "$ENV_NAME" python=3.9
$CONDA activate "$ENV_NAME"

# 按服务器 CUDA 版本选择（示例为 cu118，可改）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo "[2/4] 基础依赖"
pip install -r "$ROOT/SAvEM3/requirements-train.txt"

echo "[3/4] 图分割与后处理（waterz 需现场 JIT 编译；elf 依赖 conda 渠道）"
pip install cython  # waterz 首次调用会自动编译到 ~/.cython/inline/
pip install waterz
$CONDA install -y -c conda-forge nifty vigra affogato
# elf master 需要 Python>=3.10；Python 3.9 环境固定 0.5.0 tag
pip install 'git+https://github.com/constantinpape/elf.git@0.5.0'

echo "[4/4] 校验"
python - <<'PY'
import torch, numpy, skimage, h5py, yaml, attr
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
import waterz, elf
print('waterz + elf OK')
PY

echo "完成。后续命令示例："
echo "  conda activate $ENV_NAME"
echo "  cd /path/to/SAvEM3/savem3"
echo "  python main_devoem_sparse_membrane_triplet_2.py -c mem3c2c_3ds_t3t -m train"
