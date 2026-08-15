#!/usr/bin/env bash
# MPS 运行器：以本机 MPS 后端运行 SAvEM³ 复现脚本（训练/推理/预处理通用）。
# 依赖：repro/mps_adapt/sitecustomize.py（.cuda() 透明重定向）+ patch_mps.py 定点补丁（已执行）。
# 用法：
#   ./run_mps.sh /path/to/script.py [args...]
#   cd savem3 && ../repro/mps_adapt/run_mps.sh main_devoem_sparse_membrane_triplet_2.py -c mem3c2c_3ds_t3t -m train
set -euo pipefail

MPS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$MPS_DIR/../.." && pwd)"
VENV="$REPO_ROOT/.venvs/savem3/bin"

export PYTHONPATH="$MPS_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ENABLE_MPS_FALLBACK=1   # MPS 未实现的算子回退 CPU

# waterz JIT（验证段会调用）需要 boost 头文件与可写 cache；
# 后处理环境已由 SAvEM3/repro/env/setup_postprocess.sh 创建。
POST="$REPO_ROOT/.conda-envs/savem3-post"
if [ -x "$POST/bin/python" ]; then
    SDKROOT="${SDKROOT:-/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk}"
    export SDKROOT
    export CXXFLAGS="-isysroot $SDKROOT -isystem $SDKROOT/usr/include/c++/v1 -I$POST/include"
    export LDFLAGS="-L$POST/lib -lc++"
    export WITTY_CACHE_DIR="$REPO_ROOT/.cache/witty"
    mkdir -p "$WITTY_CACHE_DIR"
fi

if [ -x "$VENV/python" ]; then
    exec "$VENV/python" "$@"
else
    exec python3 "$@"
fi
