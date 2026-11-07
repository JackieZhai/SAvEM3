# Training and inference

SAEM² learns an image decoder with a membrane head. SAvEM³ distills image
features and boundaries into a 3D network. Postprocessing converts its boundary
predictions into instances for tracing and review.

## Data formats

| Data | Array shape | Values |
| --- | --- | --- |
| Grayscale EM | `Z,Y,X`, uint8 | 0-255 |
| Boundary probability | `Z,Y,X` or `1,Z,Y,X` | 1 = membrane |
| Affinity | `3,Z,Y,X` | Same-instance probabilities for negative Z/Y/X neighbors |
| Instances | `Z,Y,X`, integer | 0 = background |
| HQ features | `32,Z,H,W` | Same physical field of view as the raw data |
| Precomputed volumes | `X,Y,Z[,C]` | XYZ resolution in nm and absolute voxel offset |

Use distinct instance IDs for independent 2D sections. Preserve the physical
origin when cropping or changing resolution. Boundary probabilities become
nearest-neighbor affinities as `1 - max(boundary[p], boundary[p - e])`.

## Train SAEM²

Follow the [SAEM² guide](../saem2/README.md) to prepare membrane targets and
train the decoder. The [data engine](../repro/data_engine/README.md) supplies
sampling and prompt-feature tools. Keep training and test specimens separate;
[LC-NE test](LC_NE_TEST.md) is the supplied test data.

## Export teacher targets and train SAvEM³

The encoder input is calibrated uint8 EM with a 1024 x 1024 field of view.
Provide its actual XYZ resolution. For example:

```bash
python repro/savem3/export_teacher.py --raw encoder_raw.h5 \
  --checkpoint /models/sam_hq_vit_h.pth \
  --saem2-checkpoint /models/saem2_decoder.pth \
  --resolution 4 4 30 --output outputs/teacher.h5 --device auto
python repro/savem3/distill.py train --teacher outputs/teacher.h5 \
  --output outputs/student --steps 200000 --batch 4 --ablation full
```

The export contains 512 x 512 student raw, 256 x 256 HQ features, and 64 x 64
embeddings over the same physical extent. The example above therefore produces
student raw at 8 x 8 x 30 nm. A trained SAEM² decoder supplies the membrane target.
Motion regularization requires an explicit flow input. See the
[SAvEM³ guide](../repro/savem3/README.md) for initialization, losses, and resume.

## Infer and postprocess

Run inference on raw at the student's resolution:

```bash
python repro/savem3/distill.py infer --raw test_raw.h5 --raw-key main \
  --checkpoint outputs/student/last.pt --output outputs/boundary.h5
./repro/env/run_postprocess.sh repro/savem3/distill_postprocess.py \
  --affs outputs/boundary.h5 --input-kind boundary --out-dir outputs/post
```

The postprocessing environment provides watershed, multicut, and WaterZ.
[Prompt graph cut](../repro/graph_cut/README.md) takes the same fragments used
by the subsequent aggregation stage.

## Export for Neuroglancer

```bash
python probe-em/scripts/export_savem3.py --raw test_raw.h5 --raw-key main \
  --seg outputs/post/seg_lmc.hdf --resolution 8 8 30 --offset 0 0 0 \
  --output outputs/probe_roi --skeletonize --seed 1 \
  --checkpoint /models/sam2_checkpoint.pt --model-config sam2_hiera_t.yaml
python probe-em/scripts/run_probe_em.py --config outputs/probe_roi/trace_config.json
python probe-em/scripts/demo.py --local-dir outputs/probe_roi \
  --results-dir outputs/probe_roi/trace_results --seed 1
```

Supply the actual resolution, offset, and an existing seed ID. Use a new export
directory for each volume. Set the tracing budget and model paths in the generated
configuration before running a full trace. See the [script guide](../probe-em/scripts/README.md)
for interactive review and [evaluation](../repro/evaluate/README.md) for metrics.
