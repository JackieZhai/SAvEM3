#!/usr/bin/env bash
# Run WaterZ/ELF/official ERL scripts in the savem3-post conda environment.
# Usage:
#   ./run_postprocess.sh path/to/script.py [args...]
#   ./run_postprocess.sh -c 'import waterz; print(waterz.__file__)'
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
POST="${SAVEM3_POST_ENV:-$ROOT/.conda-envs/savem3-post}"
if [ ! -x "$POST/bin/python" ] && [ -x "$ROOT/../.conda-envs/savem3-post/bin/python" ]; then
    POST="$ROOT/../.conda-envs/savem3-post"
fi

if [ ! -x "$POST/bin/python" ]; then
    echo "Cannot find $POST; run ./repro/env/setup_postprocess.sh first" >&2
    exit 1
fi

if [[ "${OSTYPE:-}" == darwin* ]]; then
    export SDKROOT="${SDKROOT:-/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk}"
    export CXXFLAGS="${CXXFLAGS:-} -isysroot $SDKROOT -isystem $SDKROOT/usr/include/c++/v1 -I$POST/include"
    export LDFLAGS="${LDFLAGS:-} -L$POST/lib -lc++"
fi
export WITTY_CACHE_DIR="$ROOT/.cache/witty"
export MPLCONFIGDIR="$ROOT/.cache/matplotlib"
mkdir -p "$WITTY_CACHE_DIR" "$MPLCONFIGDIR"

exec "$POST/bin/python" "$@"
