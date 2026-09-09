# repro/

Executable reproduction scripts for SAEM²-SAEvM³.

See the [reproduction guide](../docs/REPRODUCTION.md) for the recommended workflow and validated scope.
Module guides: [data engine](data_engine/README.md), [SAEM² preparation](saem2/README.md),
[three-target distillation](savem3/README.md), [prompt graph cut](graph_cut/README.md),
[evaluation/ROI inference](evaluate/README.md), [environments](env/README.md), and [MPS compatibility](mps_adapt/README.md).
The typical order below preserves the historical workflow; new three-target distillation uses
`savem3/export_teacher.py` and `savem3/distill.py`.

## Layout

```
repro/
├── data_engine/          # Data cleansing / unification / selection scripts
├── saem2/                # SAEM² data export and launcher
├── savem3/               # Teacher-feature precomputation and SAvEM3 postprocessing
├── graph_cut/            # Prompt-based graph-cut (LMC / WaterZ)
├── evaluate/             # Dense metrics and official ERL
├── mps_adapt/            # Apple Silicon MPS compatibility layer
└── env/                  # GPU server and local postprocessing environments
```

## Environment

Training:

```bash
pip install -r ../requirements-train.txt
```

Postprocessing (waterz + elf + ERL dependencies):

```bash
./env/setup_postprocess.sh
./env/run_postprocess.sh /path/to/script.py [args...]
```

## Data links

```bash
./saem2/link_pretrained.sh        # SAM/HQ-SAM weights for SAEM²
./savem3/link_teacher_data.sh     # AC4 teacher features for SAvEM3
```

## Typical order

```bash
export SAVEM3_DATA_ROOT=/path/to/data-bank-root

# Phase I data
python data_engine/resample_to_8nm.py --datasets snemi,ac3
python data_engine/mask_myelin_glia.py --datasets snemi,ac3
python data_engine/select_training_set.py

# SAEM² data
python saem2/make_record2d.py --datasets snemi,ac3
python savem3/precompute_teacher.py --datasets snemi --write-tif --device mps
./saem2/run_train.sh

# SAvEM3 distillation
./savem3/link_teacher_data.sh
python savem3/gen_ablation_cfg.py
./savem3/train_ablation.sh mem3c2c_3ds_t3t

# Postprocessing / evaluation
./env/run_postprocess.sh savem3/distill_postprocess.py --affs out_affs.h5 --out-dir post/
./env/run_postprocess.sh graph_cut/prompt_graph_cut.py --fragments frag.h5 --feats out_feats.h5
./env/run_postprocess.sh graph_cut/lmc_prompt.py --fragments frag.h5 --affs out_affs.h5 --edge-feats edge_feats.json
./env/run_postprocess.sh graph_cut/waterz_iou.py --affs out_affs.h5 --fragments frag.h5 --edge-feats edge_feats.json
./env/run_postprocess.sh evaluate/dense_eval.py --gt gt.h5 --preds seg.hdf --names SAvEM3
./env/run_postprocess.sh evaluate/erl.py --seg pred.h5 --gt-stats gt_skel_stats.p
```

## Platform notes

- Apple Silicon: use `mps_adapt/run_mps.sh` for training scripts.
- `SAVEM3_NUM_WORKERS=0` avoids multiprocessing pickling issues on macOS.
- `savem3/precompute_teacher.py` supports `--max-slices N` for partial export.
- On mac arm64, nifty `accumulateAffinityStandartFeatures` can trigger a vigra histogram
  assertion; the LMC scripts fall back to boundary features automatically.
