# SAEM²

SAEM² auxiliary pretraining for EM neuron segmentation. This directory contains the
training, data-preparation, and evaluation scripts described in the SAvEM³ paper.

See the [reproduction contract](../docs/REPRODUCTION.md). `modeling.py` extracts the shared
`MaskDecoderHQ` (HQ + membrane token) from the trainer while retaining parameter names compatible
with historical `epoch_N.pth` checkpoints. Teacher export strictly loads the trained membrane head;
a standard HQ-SAM checkpoint cannot replace SAEM² membrane weights.
`repro/saem2/run_train.sh` selects the paper's 12 epochs; direct invocation still defaults to 16
and requires an explicit override. Membrane TIFFs and embeddings must use the same `--xy-nm`
and center-cropped 1024² field of view.

## Files

- `trainMemProISBI_2d_v4.py`: main EM training with HQ-SAM encoders and an extra
  membrane token (`mem_token`); losses are BCE + Dice for both mask and membrane heads.
- `trainNonMemProISBI_*.py`, `trainSAMfineISBI_*.py`: HQ-SAM / SAM baselines.
- `test_*.py`: 2D IoU and boundary-IoU evaluation.
- `json2d_create.py`: membrane TIFF -> connected-component labels -> `record2d` JSON.
- `utils/dataloader_isbi_2d_v4.py`: dataloader for precomputed embeddings.
- `segment_anything_training/`: SAM/HQ-SAM model code used by the training scripts.

## Setup

```bash
cd /path/to/SAvEM3

# 1. Install training dependencies
pip install -r requirements-train.txt

# 2. Put SAM/HQ-SAM checkpoints into sam-hq/pretrained_checkpoint/
#    sam_vit_h_4b8939.pth, sam_vit_h_maskdecoder.pth, ...
./repro/saem2/link_pretrained.sh

# 3. Prepare 2D data
export SAVEM3_DATA_ROOT=/path/to/data-bank-root
./repro/saem2/make_record2d.py --datasets snemi,ac3
cd saem2
python json2d_create.py

# 4. Precompute image-encoder features
cd ../repro/savem3
python precompute_teacher.py --datasets snemi --write-tif --device mps
```

Data-path environment variables:

| Variable | Default |
| --- | --- |
| `SAVEM3_DATA_ROOT` | repository root |
| `SAVEM3_EMBED_ROOT` | `$SAVEM3_DATA_ROOT/prepared_embedding/` |
| `SAVEM3_LABEL_ROOT` | `$SAVEM3_DATA_ROOT/prepared_segments_mul_2d/` |
| `SAVEM3_MEM_ROOT` | `$SAVEM3_DATA_ROOT/prepared_segments_mul_mem/` |
| `SAVEM3_NUM_WORKERS` | automatic; set `0` on macOS if multiprocessing pickling fails |

## Training

```bash
cd saem2
python trainMemProISBI_2d_v4.py \
    --model-type vit_h \
    --checkpoint pretrained_checkpoint/sam_vit_h_4b8939.pth \
    --output work_dirs/hq_sam_h_memproisbi_2d_v4 \
    --device mps \
    --gpu 0
```

Main arguments:

| Argument | Default | Description |
| --- | --- | --- |
| `--model-type` | `vit_h` | `vit_h` / `vit_l` / `vit_b` |
| `--checkpoint` | `pretrained_checkpoint/sam_vit_h_4b8939.pth` | SAM encoder weights |
| `--device` | `cuda` | `cuda` / `mps` / `cpu` |
| `--learning-rate` | `1e-3` | Adam learning rate |
| `--batch-size-train` | `8` | Training batch size |
| `--max-epoch-num` | `16` | Number of epochs; the paper uses 12 |
| `--record-train` / `--record-valid` | `./record2d_train.json` / same as train | record2d JSON paths |
| `--output` | `work_dirs/hq_sam_h_memproisbi_2d_v4` | Output directory |

## Evaluation

```bash
cd saem2
python test_SAEM_2d_v4.py \
    --model-type vit_h \
    --checkpoint pretrained_checkpoint/sam_vit_h_4b8939.pth \
    --restore-model work_dirs/hq_sam_h_memproisbi_2d_v4/epoch_10.pth \
    --device mps
```

## Notes

- The image encoder is kept frozen; the optimizer only updates the extended HQ-SAM decoder
  and membrane head.
- The dataloader accepts precomputed embedding TIFFs in either CHW or HWC layout.
