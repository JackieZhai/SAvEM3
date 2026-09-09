# Historical SAvEM³ experiment code

`model/model_superhuman.py::UNetST_PNI` is the three-target distillation student:
a residual 3D U-Net with 32/64/128/256 channels, XY downsampling, no Z downsampling,
and `(boundary, shallow_feature, embedding)` outputs. Thirty-two shallow channels
align to HQ features and the deepest 256 channels align to SAM embeddings.

Start with the [standalone distillation entry points](../repro/savem3/README.md).
They consume SAEM² teacher targets, preserve the original model, and add explicit
device selection, checkpoint resume, and training without manual GT.

## Historical entry points

`main_devoem_sparse_membrane_triplet_2.py` is the retained main version. Other
dated or `old` scripts are experiment archives; a later filename does not imply
that every experiment switch has been validated.

~~~bash
cd savem3
python main_devoem_sparse_membrane_triplet_2.py \
  -c mem3c2c_3ds_t3t --fresh --no-valid --num-workers 0 -m train
~~~

This path reads `AC4_inputs/labels/features/embeddings.h5` by default. Its historical
loader converts labels into targets: verify whether those labels are teacher
predictions or manual annotations before calling an experiment "0% labels".
The YAML defaults resume from an old checkpoint; use `--fresh` for a new run.
`--no-valid` skips in-training WaterZ/ELF; evaluate with separate postprocessing later.

| Directory | Role |
| --- | --- |
| `model/` | SuperHuman/MALA students and upsampling |
| `data/` | Historical AC4 triplet loader, augmentation, sliding-window validation |
| `loss/` | Output, feature, motion, smoothness, and experimental graph-relation losses |
| `config/` | Historical experiments/ablations, including unsupported placeholder settings |
| `augmentation/` | Anisotropic volume augmentation; images/features must share spatial transforms |
| `utils/` | Affinity, RAG, and segmentation postprocessing utilities |

The historical BDY+DST two-output configuration has no independent dual-head
loss implementation; the entry point now rejects `output_nc=2`.
`mem3c2c_3ds_t3t` enables only the three primary targets, not motion/smoothness losses.
