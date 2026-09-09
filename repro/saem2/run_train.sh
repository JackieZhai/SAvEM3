#!/usr/bin/env bash
# M2: SAEM² launcher (paper: batch 8 / 12 epochs / lr 1e-3 / seed 42 / ~60h on V100).
# Prerequisites:link_pretrained.sh → make_record2d.py → json2d_create.py
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

echo "==> SAEM² training ($MODEL_TYPE)"
CUDA_VISIBLE_DEVICES="$GPU" python trainMemProISBI_2d_v4.py \
    --model-type "$MODEL_TYPE" \
    --checkpoint "$CKPT" \
    --output "$OUT" \
    --gpu 0 \
    --device "${DEVICE:-cuda}" \
    --max-epoch-num 12 \
    --batch-size-train 8 \
    --learning-rate 1e-3 "$@"
echo "Done: $OUT (epoch_N.pth each epoch; merged sam_hq_epoch_N.pth at the end)"
