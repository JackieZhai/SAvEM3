#!/usr/bin/env bash
# 为 SAEM² 训练准备 SAM/HQ-SAM 预训练权重符号链接。
# 权重不入 git；先把权重放到本仓库 sam-hq/pretrained_checkpoint/。
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
        echo "skip $name（未找到 $SRC/$name）" >&2
    fi
done
