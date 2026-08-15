#!/usr/bin/env bash
# 安装 SAvEM3 后处理环境（waterz + elf/多割 + 官方 ERL 依赖）到本机。
#
# 背景：
#   - waterz 来自 https://github.com/funkey/waterz（JIT 编译需要 boost 头文件与 C++ SDK）
#   - elf 来自 https://github.com/constantinpape/elf tag 0.5.0
#     （elf master 要求 Python>=3.10 与 bioimage-cpp）
#   - nifty/vigra/affogato 来自 conda-forge（elf 的运行时依赖）
#   - 官方 ERL 实现已 clone 到 ../evaluate/AxonEM-challenge
#
# 本机 Homebrew/用户缓存不可写，所以 conda 使用 HOME=/tmp/condahome 与独立 PKGS 目录。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
POST="$ROOT/.conda-envs/savem3-post"
CACHE_HOME=/tmp/condahome
PKGS="$ROOT/.cache/conda-pkgs"
mkdir -p "$CACHE_HOME" "$PKGS" "$ROOT/.cache/witty"

echo "==> [1/3] conda 环境（Python 3.9 + nifty/vigra/affogato/libboost 头文件）"
# 注意：不要装 `boost` 元包，它会把 nifty/vigra 拉低到旧版，导致
# accumulateAffinityStandartFeatures 在 mac arm64 上触发 vigra histogram 断言。
HOME="$CACHE_HOME" CONDA_PKGS_DIRS="$PKGS" conda create -y \
    -p "$POST" -c conda-forge \
    python=3.9 pip nifty vigra affogato libboost-headers libboost-devel

echo "==> [2/3] elf 0.5.0（constantinpape/elf GitHub tag 0.5.0；master 需 Python>=3.10）"
HOME="$CACHE_HOME" "$POST/bin/pip" install 'git+https://github.com/constantinpape/elf.git@0.5.0' 

echo "==> [3/3] waterz（funkey/waterz GitHub 版，Apple Silicon 需显式 SDK/boost include）"
SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk
HOME="$CACHE_HOME" SDKROOT="$SDKROOT" \
CXXFLAGS="-isysroot $SDKROOT -isystem $SDKROOT/usr/include/c++/v1 -I$POST/include" \
LDFLAGS="-L$POST/lib -lc++" \
"$POST/bin/pip" install 'git+https://github.com/funkey/waterz.git'

HOME="$CACHE_HOME" "$POST/bin/pip" install h5py imageio kimimaro

echo "==> [3.5/3] 官方 ERL 仓库（若尚未 clone）"
EVAL_DIR="$ROOT/repro/evaluate"
if [ ! -f "$EVAL_DIR/AxonEM-challenge/erl_wrapper/eval_erl.py" ]; then
    git clone --depth 1 https://github.com/PytorchConnectomics/AxonEM-challenge.git "$EVAL_DIR/AxonEM-challenge"
else
    echo "已包含：$EVAL_DIR/AxonEM-challenge"
fi

echo "==> 校验"
HOME="$CACHE_HOME" SDKROOT="$SDKROOT" \
CXXFLAGS="-isysroot $SDKROOT -isystem $SDKROOT/usr/include/c++/v1 -I$POST/include" \
LDFLAGS="-L$POST/lib -lc++" \
WITTY_CACHE_DIR="$ROOT/.cache/witty" \
"$POST/bin/python" - <<'PY'
import waterz, elf, numpy as np
from elf.segmentation import watershed as ws, features as feats, multicut as mc
affs = np.zeros((3, 10, 32, 32), np.float32) + 0.7
affs[:, :, 16, :] = 0.0
segs = list(waterz.agglomerate(affs, [0.3, 0.5], discretize_queue=256))
print('waterz OK', [s.shape for s in segs])
bmap = np.zeros((32, 64, 64), np.float32); bmap[:, :, 32] = 1.0
frag, _ = ws.distance_transform_watershed(bmap, threshold=0.25, sigma_seeds=1.6)
rag = feats.compute_rag(frag)
costs = mc.transform_probabilities_to_costs(feats.compute_boundary_features(rag, bmap)[:, 0], beta=0.5)
seg = feats.project_node_labels_to_pixels(rag, mc.multicut_kernighan_lin(rag, costs))
print('elf multicut OK', seg.shape, len(np.unique(seg)))
PY
echo "完成。运行后处理脚本请使用：$ROOT/SAvEM3/repro/env/run_postprocess.sh"
