#!/usr/bin/env bash
# M4 · SAvEM³ 全阶段蒸馏训练（GPU 服务器）
# 前置：restore_savem3_scripts.sh（scripts_savem3/）、link_teacher_data.sh、
#       gen_ablation_cfg.py（可选）、precompute_teacher.py 生成教师特征 h5
# 论文配方：200K iters / batch 4 / Adam lr 1e-4 / ~40h V100 / 编码器自监督预训练初始化
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPTS="$ROOT/savem3"
CFG="${1:-mem3c2c_3ds_t3t}"
GPU="${GPU:-0}"

cd "$SCRIPTS"
if [ ! -e "../data/AC3-AC4/AC4_features.h5" ] || [ ! -e "../data/AC3-AC4/AC4_embeddings.h5" ]; then
    "$ROOT/repro/savem3/link_teacher_data.sh"
fi
EXTRA=()
if [ "${FRESH:-1}" = "1" ]; then
    EXTRA+=(--fresh)
fi
echo "==> SAvEM³ 蒸馏训练：$CFG（GPU $GPU，fresh=${FRESH:-1}）"
echo "    数据：$ROOT/data/（AC3-AC4 原始 + AC3-AC4_new 教师特征）"
CUDA_VISIBLE_DEVICES="$GPU" python main_devoem_sparse_membrane_triplet_2.py -c "$CFG" -m train "${EXTRA[@]}"
echo "完成：models/<时间戳>_$CFG/model-*.ckpt（每 1000 iters 存一次，训练中每 1000 iters 在 AC4 后 20 层验证 waterz/LMC）"
