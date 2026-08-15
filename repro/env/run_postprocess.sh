#!/usr/bin/env bash
# 用 savem3-post conda 环境运行 waterz/elf/官方 ERL 相关脚本。
# 用法：
#   ./run_postprocess.sh path/to/script.py [args...]
#   ./run_postprocess.sh -m python -c 'import waterz; print(waterz.__file__)'
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
POST="$ROOT/.conda-envs/savem3-post"
SDKROOT="${SDKROOT:-/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk}"

if [ ! -x "$POST/bin/python" ]; then
    echo "未找到 $POST；请先运行 ./setup_postprocess.sh" >&2
    exit 1
fi

export SDKROOT
export CXXFLAGS="-isysroot $SDKROOT -isystem $SDKROOT/usr/include/c++/v1 -I$POST/include"
export LDFLAGS="-L$POST/lib -lc++"
export WITTY_CACHE_DIR="$ROOT/.cache/witty"
export MPLCONFIGDIR="$ROOT/.cache/matplotlib"
mkdir -p "$WITTY_CACHE_DIR" "$MPLCONFIGDIR"

exec "$POST/bin/python" "$@"
