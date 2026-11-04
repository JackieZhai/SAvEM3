#!/usr/bin/env bash
# M4: full-stage SAvEM³ distillation training on a GPU server.
# Historical prerequisites: restore_savem3_scripts.sh (scripts_savem3/),
# link_teacher_data.sh, optional gen_ablation_cfg.py, and teacher H5 files from precompute_teacher.py.
# Paper recipe: 200K iterations, batch 4, Adam lr 1e-4, ~40 h on V100, self-supervised encoder initialization.
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
echo "==> SAvEM³ distillation training: $CFG (GPU $GPU, fresh=${FRESH:-1})"
echo "    Data: $ROOT/data/ (AC3-AC4 raw data + AC3-AC4_new teacher features)"
CUDA_VISIBLE_DEVICES="$GPU" python main_devoem_sparse_membrane_triplet_2.py -c "$CFG" -m train "${EXTRA[@]}"
echo "Done: models/<timestamp>_$CFG/model-*.ckpt (checkpoints every 1000 iterations; WaterZ/LMC validation on the final 20 AC4 sections every 1000 iterations)"
