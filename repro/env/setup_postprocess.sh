#!/usr/bin/env bash
# Install local SAvEM3 postprocessing dependencies: WaterZ, ELF/multicut, and official ERL.
#
# Background:
#   - WaterZ: https://github.com/funkey/waterz (JIT needs Boost headers and a C++ SDK).
#   - ELF: https://github.com/constantinpape/elf at tag 0.5.0.
#     ELF master requires Python>=3.10 and bioimage-cpp.
#   - nifty/vigra/affogato: conda-forge packages required by ELF at runtime.
#   - The official ERL implementation is vendored in ../evaluate/AxonEM-challenge.
#
# This legacy setup uses a temporary home and a separate package cache for conda
# because the original machine's Homebrew/user caches were not writable.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
POST="$ROOT/.conda-envs/savem3-post"
CACHE_HOME=/tmp/condahome
PKGS="$ROOT/.cache/conda-pkgs"
mkdir -p "$CACHE_HOME" "$PKGS" "$ROOT/.cache/witty"

echo "==> [1/3] Conda environment (Python 3.9 + nifty/vigra/affogato/libboost headers)"
# Avoid the `boost` metapackage: it can downgrade nifty/vigra and trigger a vigra
# histogram assertion in accumulateAffinityStandartFeatures on macOS ARM64.
HOME="$CACHE_HOME" CONDA_PKGS_DIRS="$PKGS" conda create -y \
    -p "$POST" -c conda-forge \
    python=3.9 pip nifty vigra affogato libboost-headers libboost-devel

echo "==> [2/3] ELF 0.5.0 (constantinpape/elf GitHub tag; master requires Python>=3.10)"
HOME="$CACHE_HOME" "$POST/bin/pip" install 'git+https://github.com/constantinpape/elf.git@0.5.0' 

echo "==> [3/3] WaterZ (funkey/waterz; Apple Silicon needs explicit SDK/Boost include paths)"
SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk
HOME="$CACHE_HOME" SDKROOT="$SDKROOT" \
CXXFLAGS="-isysroot $SDKROOT -isystem $SDKROOT/usr/include/c++/v1 -I$POST/include" \
LDFLAGS="-L$POST/lib -lc++" \
"$POST/bin/pip" install 'git+https://github.com/funkey/waterz.git'

HOME="$CACHE_HOME" "$POST/bin/pip" install h5py imageio kimimaro

echo "==> [3.5/3] Official ERL repository (clone if missing)"
EVAL_DIR="$ROOT/repro/evaluate"
if [ ! -f "$EVAL_DIR/AxonEM-challenge/erl_wrapper/eval_erl.py" ]; then
    git clone --depth 1 https://github.com/PytorchConnectomics/AxonEM-challenge.git "$EVAL_DIR/AxonEM-challenge"
else
    echo "Already available: $EVAL_DIR/AxonEM-challenge"
fi

echo "==> Validate the installation"
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
echo "Done. Run postprocessing scripts with: $ROOT/SAvEM3/repro/env/run_postprocess.sh"
