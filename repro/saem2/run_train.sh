#!/usr/bin/env bash
# M2 · SAEM² 训练一键脚本（GPU 服务器；按论文配方：batch 8 / 12 epochs / lr 1e-3 / seed 42 / ~60h V100）
# 前置：link_pretrained.sh → make_record2d.py → json2d_create.py
#       → savem3/precompute_teacher.py --write-tif
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TRAIN="$ROOT/saem2"
cd "$TRAIN"

if [ ! -f pretrained_checkpoint/sam_vit_h_4b8939.pth ]; then
    "$ROOT/repro/saem2/link_pretrained.sh"
fi

MODEL_TYPE="${MODEL_TYPE:-vit_h}"
GPU="${GPU:-0}"
CKPT="${CKPT:-pretrained_checkpoint/sam_vit_h_4b8939.pth}"
OUT="${OUT:-work_dirs/hq_sam_h_memproisbi_2d_v4}"

echo "==> SAEM² 训练（$MODEL_TYPE）"
CUDA_VISIBLE_DEVICES="$GPU" python trainMemProISBI_2d_v4.py \
    --model-type "$MODEL_TYPE" \
    --checkpoint "$CKPT" \
    --output "$OUT" \
    --gpu 0 \
    --device "${DEVICE:-cuda}"
echo "完成：$OUT（每 epoch 存 epoch_N.pth，最后合并 sam_hq_epoch_N.pth）"
