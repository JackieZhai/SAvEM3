# SAvEM³ teacher export, distillation, and postprocessing

Use these standalone entry points for new experiments. See the
[reproduction guide](../../docs/REPRODUCTION.md) for the full workflow and paper differences.

| Entry point | Input | Output |
| --- | --- | --- |
| `export_teacher.py` | 1024² raw ROI, HQ-SAM, trained SAEM² decoder | Three aligned teacher targets in H5 |
| `distill.py train` | Teacher H5 | `last.pt` with model, optimizer, RNG, configuration, and provenance |
| `distill.py infer` | Student-resolution raw and checkpoint | Boundary-probability H5, `main:ZYX` |
| `distill_postprocess.py` | Boundary probabilities or affinities | Fragments, affinities, WaterZ/MC instances, and report |
| `precompute_teacher.py` | Data bank / historical embedding caches | Separate embedding/feature H5 and optional TIFF; no boundary target |
| `gen_ablation_cfg.py` | Historical YAML | Supported ablations; does not generate the unsupported BDY+DST placeholder |

## Teacher contract

Teacher H5 contains `raw:ZYX uint8`, `boundary:ZYX float32`,
`features:32ZYX float32`, and `embeddings:256ZYX float32`. Their XY size ratio
is `8:8:4:1`. Root attributes record physical resolution, voxel offset, provenance,
and completion. A `.partial` file is an unfinished export and must not be used for training.

HQ features and membrane predictions come from separately loaded modules: frozen
HQ features retain their pretrained source, while the membrane head uses trained
SAEM² weights. `export_teacher.py --resolution` describes the input encoder raw,
not the output student raw.

## Training

~~~bash
python repro/savem3/distill.py train --teacher outputs/teacher.h5 \
  --output outputs/student --ablation full --steps 200000 --batch 4 --device auto
python repro/savem3/distill.py train --teacher outputs/teacher.h5 \
  --output outputs/student --ablation full --steps 200000 --batch 4 \
  --resume outputs/student/last.pt
~~~

`bdy` trains only boundary loss; `bdy_embed` adds deep embeddings; `full` also
aligns 32 HQ channels. Remaining shallow student channels learn through boundary loss.
The default patch is `18×160×160`; custom XY dimensions must be divisible by 8.
Targets are read as H5 patches rather than copying the whole feature volume to GPU memory.
Resume requires matching teacher path, loss selection, patch, batch, and learning rate.

Enable secondary losses with `--secondary-weight 0.05 --epsilon 0.1` and provide
`flow:(2,Z-1,Hraw/2,Wraw/2)` in HQ-feature pixel units. The convention aligns
`current(x-fx,y-fy,z)` with `previous(x,y,z-1)`. Out-of-bounds samples are excluded
from motion loss; epsilon is an explicit experiment parameter. An arbitrary optical-flow
estimator is not automatically equivalent to the paper's dual-scale cross-correlation method.

## Inference and postprocessing

~~~bash
python repro/savem3/distill.py infer --raw outputs/teacher.h5 --raw-key raw \
  --checkpoint outputs/student/last.pt --output outputs/boundary.h5
./repro/env/run_postprocess.sh repro/savem3/distill_postprocess.py \
  --affs outputs/boundary.h5 --out-dir outputs/post --method both
~~~

Sliding-window inference averages overlaps and pads edges. It accumulates the full
ROI in memory and is intended for demo/experiment blocks, not petabyte scheduling.
The student remains `savem3/model/model_superhuman.py::UNetST_PNI`.

Reusing teacher raw above checks connectivity of the software pipeline only.
Use independent student-resolution test ROIs for evaluation. H5/TIFF/NPY inputs
can be passed to `--raw`; the default H5 key is `main`.

Postprocessing always writes `fragments_ws.h5`, `fragments.h5`, and `affinities.h5`
for reuse and comparison. `--skip-2d-mc` is an explicit ablation; normally both
3D aggregation branches consume the same 2D MC result. The historical filename
`seg_lmc.hdf` refers here to Kernighan–Lin multicut on a local RAG, without extra lifted edges.
