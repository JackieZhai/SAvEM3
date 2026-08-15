# SAEM²-SAEvM³

**SAEM²-SAEvM³: Pretrained and Distilled Models for General-purpose 3D Neuron
Reconstruction**

*Hao Zhai, Jinyue Guo, Yanchao Zhang, Jing Liu, Hua Han* <br>
*Key Laboratory of Brain Cognition and Brain-inspired Intelligence Technology,
Institute of Automation, Chinese Academy of Sciences*

Official code repository for the BIBM 2024 paper:

```
@inproceedings{Zhai-SAvEM3,
    author={Zhai, Hao and Guo, Jinyue and Zhang, Yanchao and Liu, Jing and Han, Hua},
    booktitle={2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
    title={SAvEM3: Pretrained and Distilled Models for General-purpose 3D Neuron Reconstruction},
    year={2024},
    pages={3972-3977},
    doi={10.1109/BIBM62325.2024.10822494}
}
```

## Repository layout

| Path | Description |
| --- | --- |
| `sam-hq/` | SAM / HQ-SAM model code (upstream fork) |
| `saem2/` | SAEM² auxiliary pretraining code (HQ-SAM + membrane token) |
| `savem3/` | SAvEM³ full-stage distillation code (3D residual U-Net) |
| `probe-em/` | Probe-EM targeted neuron tracing and SAM 2 semantic verification |
| `repro/` | Data engine, MPS/CUDA environment scripts, graph-cut, postprocessing, evaluation |
| `requirements-train.txt` | Python dependencies for SAEM²/SAvEM³ training |
| `requirements-post.txt` | Python dependencies for waterz / elf / ERL postprocessing |
| `data/` | Optional local data directory (not committed) |

## Installation

### Training environment

```bash
python -m venv .venvs/savem3
source .venvs/savem3/bin/activate
pip install torch torchvision   # or install the CUDA build for your server
pip install -r requirements-train.txt
```

On Apple Silicon, use the MPS wrapper:

```bash
cd savem3
../repro/mps_adapt/run_mps.sh main_devoem_sparse_membrane_triplet_2.py -c mem3c2c_3ds_t3t --fresh -m train
```

### Postprocessing environment

```bash
./repro/env/setup_postprocess.sh
./repro/env/run_postprocess.sh repro/savem3/distill_postprocess.py --help
```

### Probe-EM environment

Probe-EM uses Python 3.10 + SAM 2 and is installed separately:

```bash
cd probe-em
# see probe-em/INSTALL.md
pip install -r requirements.txt
export PYTORCH_ENABLE_MPS_FALLBACK=1
python scripts/run_probe_em.py --config configs/config.json
```

The postprocessing environment installs:
- [funkey/waterz](https://github.com/funkey/waterz)
- [constantinpape/elf](https://github.com/constantinpape/elf) 0.5.0
- conda-forge `nifty`, `vigra`, `affogato`, `libboost-headers`, `libboost-devel`
- `kimimaro` for AxonEM-challenge ERL skeletons

## Data and pretrained weights

Large files are intentionally not committed. Prepare them as follows.

### SAM / HQ-SAM weights

Put the following files under `sam-hq/pretrained_checkpoint/`:
`sam_vit_h_4b8939.pth`, `sam_vit_h_maskdecoder.pth` (and optional `vit_l`, `vit_b` variants),
then create links for SAEM²:

```bash
./repro/saem2/link_pretrained.sh
```

### SAvEM³ teacher features

Put `AC4_inputs.h5`, `AC4_labels.h5`, `AC4_features.h5`, `AC4_embeddings.h5`
under `data/AC3-AC4_new/`, then create the expected data links:

```bash
./repro/savem3/link_teacher_data.sh
```

### Data bank

The data-engine scripts read CloudVolume layers according to
`sam-hq/train/utils/location.py`. Set `SAVEM3_DATA_ROOT` if the data bank is outside
this repository:

```bash
export SAVEM3_DATA_ROOT=/path/to/data-bank-root
```

## Training

### SAEM² (Phase I)

```bash
cd saem2
python trainMemProISBI_2d_v4.py \
    --model-type vit_h \
    --checkpoint pretrained_checkpoint/sam_vit_h_4b8939.pth \
    --output work_dirs/hq_sam_h_memproisbi_2d_v4 \
    --device mps
```

See [`saem2/README.md`](saem2/README.md).

### SAvEM³ (Phase II)

```bash
cd savem3
python main_devoem_sparse_membrane_triplet_2.py \
    -c mem3c2c_3ds_t3t --fresh -m train
```

On a machine without waterz/elf, disable in-training validation:

```bash
python main_devoem_sparse_membrane_triplet_2.py \
    -c mem3c2c_3ds_t3t --fresh --no-valid --num-workers 0 -m train
```

### Probe-EM targeted tracing

```bash
cd probe-em
python scripts/run_probe_em.py --config configs/config.json
```

### Postprocessing and evaluation

```bash
./repro/env/run_postprocess.sh repro/savem3/distill_postprocess.py --affs out_affs.h5 --out-dir post/
./repro/env/run_postprocess.sh repro/evaluate/dense_eval.py --gt gt.h5 --preds seg.hdf --names SAvEM3
./repro/env/run_postprocess.sh repro/evaluate/erl.py --seg pred.h5 --gt-stats gt_skel_stats.p
```

## Acknowledgements

- [segment-anything](https://github.com/facebookresearch/segment-anything)
- [sam-hq](https://github.com/SysCV/sam-hq)
- [micro-sam](https://github.com/computational-cell-analytics/micro-sam)
- [SuperHuman](https://github.com/weih527/SuperHuman)
- [AxonEM-challenge](https://github.com/PytorchConnectomics/AxonEM-challenge)
