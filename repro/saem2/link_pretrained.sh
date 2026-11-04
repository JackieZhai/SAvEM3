#!/usr/bin/env bash
# Create SAM/HQ-SAM pretrained-weight symlinks for SAEM² training.
# Weights are not tracked in Git; place them in sam-hq/pretrained_checkpoint/ first.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$ROOT/sam-hq/pretrained_checkpoint"
DST="$ROOT/saem2/pretrained_checkpoint"
mkdir -p "$SRC" "$DST"
for name in sam_vit_h_4b8939.pth sam_vit_h_maskdecoder.pth \
            sam_vit_l_0b3195.pth sam_vit_l_maskdecoder.pth \
            sam_vit_b_01ec64.pth sam_vit_b_maskdecoder.pth; do
    if [ -f "$SRC/$name" ]; then
        ln -sfn "../../sam-hq/pretrained_checkpoint/$name" "$DST/$name"
        echo "link $name"
    else
        echo "skip $name (not found: $SRC/$name)" >&2
    fi
done
