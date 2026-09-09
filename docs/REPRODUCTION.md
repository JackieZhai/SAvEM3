# SAEM² → SAvEM³ → Probe-EM reproduction guide

This repository is the maintained workflow entry point. Historical experiments
and external baseline checkouts may coexist in a local workspace, but are not
required to be part of this Git repository. To generate a local source inventory:

~~~bash
python repro/audit_workspace.py --root . --output docs/workspace_inventory.json
~~~

The generated inventory is ignored by Git because it records machine-specific
paths. It covers source/configuration/documentation discovery and Python syntax
checks, not complete training and evaluation of every baseline.

## Paper and implementation boundaries

The reference is Sections 2.1–2.3 and the implementation details of
[SAvEM3, BIBM 2024](https://doi.org/10.1109/BIBM62325.2024.10822494).
HQ features and decoder components reference [HQ-SAM](https://github.com/SysCV/sam-hq);
prompt points and mask selection reference [SAM-Graph](https://github.com/zju3dv/SAM-Graph).
This implementation uses EM-section RAGs; it does not port SAM-Graph's full
multi-view scene point-cloud graph-training workflow.

| Paper function | Entry point | Status and validation scope |
| --- | --- | --- |
| Data cleansing, unification, sampling | `repro/data_engine/` | Corrected XY/YX, mask prompts, boundary replacement; manual masks, same-superset exclusion, and retraining rounds remain dataset-specific |
| SAEM² membrane-assisted pretraining | `saem2/trainMemProISBI_2d_v4.py` | Shared HQ/membrane decoder extracted; historical checkpoint parameter names retained |
| SAM embeddings + HQ features + SAEM² boundaries | `repro/savem3/export_teacher.py` | Three targets cover one physical field of view; trained membrane weights are required |
| Three-target 3D distillation | `repro/savem3/distill.py train` | Original UNetST_PNI; H5 patch reads, CPU/CUDA/MPS selection, resume, and actual backpropagation tested |
| Cross-section motion/smoothness | Same, `--secondary-weight 0.05` | Requires explicit flow; generation and calibration remain separate, and no-flow regularization is not the full secondary loss |
| BDY+DST ablation | Historical `ab2_bdy_dst.yaml` | No independent two-head loss; the generator skips this placeholder and the historical trainer rejects it |
| Whole-ROI inference | `repro/savem3/distill.py infer` | Overlapping windows cover edges and inputs smaller than a patch; output is boundary probability |
| Sparse ROI inference | `repro/evaluate/sparse_eval.py` | Actual tiled inference/stitching with coverage; unprocessed areas are not treated as neuron interiors |
| DT watershed + 2D MC + 3D aggregation | `repro/savem3/distill_postprocess.py` | Corrected boundary/affinity direction, section-unique IDs, and premerge data flow |
| Prompt graph cut | `repro/graph_cut/` | Same-section prompt IoU, 32-channel node means, correct cost sign; local multicut without extra lifted edges; WaterZ uses a voxel-interface approximation |
| Probe-EM | `probe-em/scripts/run_probe_em.py` | PEC/ASP retained; lazy predictor reuse, explicit failure/limited/complete states, resume |
| Neuroglancer | `probe-em/scripts/demo.py` | Local raw/instances/trace/review/skeletons; default-on optional registration, A comparison, `showSlices=false`; real zebrafish browser checks |

## Shared data contract

| Data | Memory/H5 shape | Values and coordinates |
| --- | --- | --- |
| Grayscale raw | `Z,Y,X`, uint8 | 0–255; uint16 brightness ranges are not guessed |
| Boundary probability | `Z,Y,X` or `1,Z,Y,X` | 1 = membrane; 0 = neuron interior |
| Affinity | `3,Z,Y,X` | 1 = same instance; channels connect negative z/y/x neighbors |
| Instances/fragments | `Z,Y,X`, integer | 0 = background; positive IDs must not repeat across independent sections |
| HQ features | `32,Z,H,W` | May have lower XY resolution than labels, but must cover the same field of view |
| Neuroglancer/CloudVolume | `X,Y,Z[,C]` | XYZ resolution in nm/voxel; XYZ voxel offset |

Boundary-to-affinity conversion is `a(p,p-e)=1-max(b(p),b(p-e))`.
ELF consumes cut probabilities: positive converted costs penalize cutting, so high
cosine/IoU must increase costs. Subtracting similarity would encourage the opposite.
WaterZ directly consumes affinities; do not replicate a boundary map into three
channels and feed it unchanged.

## From training to demonstration

Run commands from the repository root unless stated otherwise. Training, C++
postprocessing, and SAM 2 may use separate Python environments, exchanging files
with explicit axis conventions.

1. Prepare aligned membranes, instances, and embeddings using [SAEM²](../saem2/README.md),
   then train the membrane decoder. The paper's recipe is batch 8, 12 epochs,
   Adam learning rate 1e-3, and seed 42.
2. Provide calibrated `(Z,1024,1024)` uint8 encoder raw. The resolution below is
   an example, not a value to apply blindly to other datasets:

   ~~~bash
   python repro/savem3/export_teacher.py --raw roi_encoder.h5 \
     --checkpoint sam-hq/pretrained_checkpoint/sam_hq_vit_h.pth \
     --saem2-checkpoint saem2/work_dirs/run/epoch_11.pth \
     --resolution 4 4 30 --output outputs/teacher.h5 --device auto
   python repro/savem3/distill.py train --teacher outputs/teacher.h5 \
     --output outputs/student --steps 200000 --batch 4 --ablation full
   ~~~

   Exported student raw is `512²`; HQ `256²` and embedding `64²` cover the same
   physical field of view. In this example student resolution is `8×8×30 nm`.
   The standalone trainer does not read manual GT. Historical targets derived
   from `AC4_labels.h5` cannot automatically be called a "0% labels" experiment.
3. Infer at student resolution, not directly on unprocessed encoder raw:

   ~~~bash
   python repro/savem3/distill.py infer --raw outputs/teacher.h5 --raw-key raw \
     --checkpoint outputs/student/last.pt --output outputs/boundary.h5
   ./repro/env/run_postprocess.sh repro/savem3/distill_postprocess.py \
     --affs outputs/boundary.h5 --input-kind boundary --out-dir outputs/post
   ~~~

   Reusing teacher raw checks the pipeline only. Formal evaluation requires an
   independent test ROI; standalone H5 files default to key `main`.
4. Optionally build prompt features on `outputs/post/fragments.h5`. Subsequent
   graph-cut stages must reuse exactly those fragments; see [graph cut](../repro/graph_cut/README.md).
5. In the Probe-EM environment, export raw and instances to local precomputed
   volumes with skeletons:

   ~~~bash
   python probe-em/scripts/export_savem3.py --raw outputs/teacher.h5 --raw-key raw \
     --seg outputs/post/seg_lmc.hdf --resolution 8 8 30 --offset 0 0 0 \
     --output outputs/probe_roi --skeletonize --seed 1 \
     --checkpoint /absolute/path/to/neurosam2.pt \
     --model-config configs/sam2.1/sam2.1_hiera_l.yaml
   python probe-em/scripts/run_probe_em.py --config outputs/probe_roi/trace_config.json
   python probe-em/scripts/demo.py --local-dir outputs/probe_roi \
     --results-dir outputs/probe_roi/trace_results --seed 1
   ~~~

   Choose an existing instance ID for `--seed`. Generated configurations default
   to `target_mip=0` and `debug_limit=2` for workflow checks; adjust the budget for
   your experiment. Export rejects an existing output directory to avoid mixing chunks.

## Requirements for quantitative reproduction

For OOD evaluation, exclude every dataset in the test set's superset from training,
not just the test cutout. Record train/validation JSON, cleansing rounds, resolution,
ROIs, checkpoints, flow method, and postprocessing thresholds.

The standalone entry point provides a BCE+L1 primary loss and mean-normalized
feature/motion losses. Exact experiment weights, self-supervised encoder
initialization, and table scores still require calibration on the formal datasets.
`--init-encoder` requires a complete matching encoder.

Short training runs, synthetic regressions, and demos do not reproduce Tables 2/3/4.
Full SAEM² 12-epoch training, SAvEM³ 200K iterations, formal ERL/OOD benchmarks,
and Probe-EM manual-time evaluation were not completed in this work. Standard
SAM 2 verifies software compatibility; the quality claims in the
[Probe-EM paper](https://arxiv.org/html/2607.04696v1) depend on NeuroSAM 2 weights,
not the generic Tiny checkpoint used for these demos.

## Validation and external dependencies

See [VALIDATION.md](VALIDATION.md) for environment-specific results and their limits.

~~~bash
python -m pytest tests/test_reproduction.py tests/test_distillation.py -q
./repro/env/run_postprocess.sh -m pytest tests/test_reproduction.py -q
# Run in the separate Probe-EM environment:
python -m pytest tests/test_probe_em.py tests/test_registration.py \
  tests/test_demo_alignment.py tests/test_demo_regions.py -q
~~~

Coverage includes probability direction, same-section prompts, section label IDs,
student forward/backward/resume, window edges, native graph cut, uint64 export and
registration, and tracing failure recovery. Missing specialized dependencies cause
explicit skips.

The local validation used an editable historical SAM 2 checkout with additional
device-selection fixes in `sam2/build_sam.py`, `sam2/utils/misc.py`,
`sam2/sam2_video_predictor.py`, and `sam2/modeling/sam2_base.py`. These external
files are not part of this repository or commit. A fresh installation must use
a SAM 2 version whose weight loading, frames, and memory follow the selected
model device. Run the opt-in real-weight integration tests to verify it; local
success is not evidence that every unpatched historical fork works on CPU/MPS.
CUDA/MPS performance requires measurements on the corresponding hardware.

Skeleton export pins `kimimaro==5.8.4`. Version 5.8.0 failed with the local
osteoid/NumPy combination when adding ROI offsets to integer vertices; the newer
coordinate conversion resolves that failure. See the
[Kimimaro intake implementation](https://github.com/seung-lab/kimimaro/blob/master/kimimaro/intake.py).
