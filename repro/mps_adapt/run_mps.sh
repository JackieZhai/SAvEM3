#!/usr/bin/env bash
# Run SAvEM³ reproduction scripts with the local MPS backend (training/inference/preprocessing).
# Requires sitecustomize.py for .cuda() redirection and the patches from patch_mps.py.
# Usage:
#   ./run_mps.sh /path/to/script.py [args...]
#   cd savem3 && ../repro/mps_adapt/run_mps.sh main_devoem_sparse_membrane_triplet_2.py -c mem3c2c_3ds_t3t -m train
set -euo pipefail

MPS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$MPS_DIR/../.." && pwd)"
VENV="$REPO_ROOT/.venvs/savem3/bin"

export PYTHONPATH="$MPS_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ENABLE_MPS_FALLBACK=1   # Fall back to CPU for unsupported MPS operators.

# WaterZ JIT, used during validation, needs Boost headers and a writable cache.
# Create the postprocessing environment with repro/env/setup_postprocess.sh first.
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
