# Bounded evaluation on large local volumes

`scripts/evaluate_local_gt.py` selects small physical windows from local
Neuroglancer precomputed data, runs actual SAM2 video propagation, and scores the
native-coordinate masks against annotated neuron instances. It compares original,
ECC and optional SOFIMA inputs using identical windows and conditioning points.
It does not align, segment or load an entire large volume.

The [test-window browser](README_windows.md) displays these cached ROIs behind
Previous/Next buttons and a selector, retaining each window's points and predictions.
The **LC-NE test overview** provides an unregistered multiscale overview with five
process ROI bounding boxes, lc meshes and mito/syn/ves context. It caches viewed chunks locally
instead of reading the full 672 GB native raw array.

## Data and coordinate contract

The historical pilot used the original `zfishlc` collection with raw dimensions
20600 x 20600 x 1583 at **4 x 4 x 30 nm**. Its raw mip 1 is 8 x 8 x 30 nm;
the full native uint8 array would contain about 672 GB. The GT archive contains
three datasets, `zfishlc_lc`, `zfishlc_lcpre` and `zfishlc_lcsyn`. The neuron
evaluation uses `zfishlc_lc`, which contains IDs 30001 through 30005 at its
coarsest scale. The other two label datasets are retained without assigning
unverified semantics to them.

The prepared example dataset is named [LC-NE test](../docs/LC_NE_TEST.md). Read
each volume's `info` for its actual bounds and resolutions. Only selected
evaluation windows are loaded and cached; the full native array is not required.

Raw and GT must use the same physical coordinate origin. The selected raw grid
must be an integer coarsening of the selected GT grid. Instance IDs are sampled
at voxel centers using nearest neighbors, never averaged. In the historical pilot,
4 nm GT was sampled with XY phase 1 and stride 2 to match 8 nm raw. The cropped
release provides both raw and GT at 8 nm in mip 0; its new 8/16 nm label levels
use the majority-vote policy described in the dataset provenance. Missing chunks
are errors, not implicit background. Raw, locator and pre-resampling GT reads
have explicit voxel limits; their limits do not describe total model memory.

## Run a reproducible comparison

Run from the repository root in the installed Probe-EM/SAM2 environment:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python probe-em/scripts/evaluate_local_gt.py \
  --raw-path /data/LC-NE-test/zfishlc_raw \
  --gt-path /data/LC-NE-test/zfishlc_lc \
  --raw-mip 0 --gt-mip 0 --locator-mip 5 \
  --xy 1024 --slices 33 \
  --checkpoint /models/sam2_hiera_tiny.pt \
  --model-config sam2_hiera_t.yaml --device mps \
  --methods unregistered ecc sofima \
  --sofima-python /envs/sofima/bin/python \
  --output-dir /local-ssd/evaluation
```

Use an available device and matching installed SAM2 configuration. SOFIMA needs
the [isolated environment](README_sofima.md); omit it from `--methods` to run
without that dependency. Default methods are `unregistered ecc`. The output
directory must be new, so earlier results cannot be overwritten accidentally.
`--labels` selects explicit IDs; otherwise all nonzero coarse labels are used.
Very thin annotations can disappear at coarse mips and need a finer locator.

Selection is deterministic and precedes inference. For each label, the locator
chooses the occupied section nearest its median occupied z and the largest
connected component there. A central interior point locates the fine ROI. One
positive point is selected on the central fine section, maximizing distance from
both the target boundary and the crop boundary. Padding the distance transform
prevents points on truncated objects from falling on fragile crop edges.

GT assists window and central-point selection; this is a controlled interactive
benchmark, not automatic neuron discovery. Only that one point is supplied to
SAM2. Each method starts fresh video memory, propagates forward and backward,
and must return every requested section. All three methods use the same native
point. Masks are transformed back to the native raw grid before scoring.

Each neuron directory contains:

- `cache/raw`, `cache/seg` and `cache/target`: local raw, full instance GT and
  binary target volumes with physical metadata.
- `registration.json`: adjacent-section NCC on common valid pixels, field
  provenance and optional SOFIMA geometry checks.
- `<method>/frames`: the exact JPEG files supplied to the legacy video API;
  their hashes are recorded in the method metrics.
- `<method>/prediction.npz` and `<method>/mask`: native masks as ZYX arrays and
  an XYZ precomputed label volume, respectively.
- `<method>/metrics.json`: per-section scores, prompted/held-out summaries and
  overlap with other annotated neurons. `report.json` aggregates all methods.

The top-level `status.json` distinguishes running, complete, failed and cancelled
runs. A partial report must not be mistaken for a completed comparison.

## Interpretation and limits

Prompted-section and unprompted-section Dice/IoU are reported separately. The
target-instance metric assumes that target's annotation is complete inside the
ROI. Sparse annotations do not establish exhaustive background ground truth.
Overlap with another annotated neuron provides evidence of a merge; absence of
such overlap does not prove there are no merges. Cropped targets are flagged.

The pilot uses all five coarse neuron IDs and 1024 x 1024 x 33 windows at
8 x 8 x 30 nm: **8.192 x 8.192 x 0.99 micrometers per window**, with one prompted
and 32 unprompted sections. This predominantly samples large cell-body profiles,
not complete axonal arbors. Generic SAM2 can select a nucleus or only part of a
cell. More prompts or EM-adapted weights may be needed. No full-neuron topology,
long-distance tracing or paper-level reproduction is established by this pilot.

Registration NCC measures image consistency, not segmentation accuracy. SOFIMA
may retain ECC unchanged when its elastic refinement fails quality guards.
Compare GT scores as well as NCC; improved NCC does not guarantee improved Dice.

## Historical soma pilot

The completed 2026-11-04 run used the local generic SAM2 Hiera Tiny checkpoint on
MPS, one positive point per neuron and all 33 sections per method. No negative
points or propagated-section GT masks were supplied. The following Dice scores
include only the 32 unprompted sections of each window:

| Neuron | Original input | ECC | ECC + SOFIMA |
| --- | ---: | ---: | ---: |
| 30001 | 0.7893 | 0.7808 | 0.7811 |
| 30002 | 0.7220 | 0.9088 | 0.7303 |
| 30003 | 0.9018 | 0.8687 | 0.8774 |
| 30004 | 0.6811 | 0.8311 | 0.8311 |
| 30005 | 0.9056 | 0.9101 | 0.9050 |
| Mean across neurons | **0.8000** | **0.8599** | **0.8250** |

The corresponding adjacent-section NCC values, measured on common valid pixels:

| Neuron | Original input | ECC | ECC + SOFIMA |
| --- | ---: | ---: | ---: |
| 30001 | 0.39957 | 0.42939 | 0.43121 |
| 30002 | 0.33202 | 0.35577 | 0.36124 |
| 30003 | 0.41477 | 0.43815 | 0.43938 |
| 30004 | 0.41301 | 0.44215 | 0.44215 |
| 30005 | 0.32648 | 0.35003 | 0.35243 |

ECC helped three windows and reduced Dice in two. SOFIMA increased NCC in four
windows, but improved Dice over ECC in only two; its guards retained ECC unchanged
for 30004. ECC remains the configurable default and SOFIMA is optional.
These results do not establish a universal ranking of alignment methods or statistical
significance from five windows.

Important remaining errors are visible in the separate precision/recall scores.
For ECC, 30001 has precision 0.6413 and recall 0.9979, while 30004 has precision
0.7122 and recall 0.9977: improved coverage still includes substantial excess area.
Neuron 30003 overlaps another annotated neuron in **903,476 voxels across the
33-section prediction**, indicating a merge despite its relatively high Dice.
All five target windows touch an XY crop boundary. Whole-neuron reconstruction
quality therefore remains unresolved.

The native 8 nm ROI offsets are `(5680,4624,1196)`, `(5040,5648,940)`,
`(5040,4624,812)`, `(4880,5712,612)` and `(6288,5680,788)` for IDs 30001-30005.
Local reports retain per-section metrics, exact conditioning points, frame hashes,
model metadata and field fingerprints. The earlier nine-section trial is retained
as a diagnostic; its crop-edge point-selection issue was corrected before this
33-section comparison. Source data, intermediate trials and browser artifacts
are excluded from the repository.

## Review GT, predictions and organelle context

The demo accepts repeatable `--label-layer NAME=PATH` and
`--image-layer NAME=PATH` options. Every auxiliary volume must match the raw
shape, resolution and voxel offset exactly. Label layers retain native slices
and use the same near-isotropic mesh policy. All auxiliary layers start hidden;
enable individual comparisons in the layer list.

```bash
python probe-em/scripts/demo.py \
  --local-dir /local-ssd/evaluation/neuron_30002/cache --seed 30002 \
  --interactive --prompt-backend sam2 \
  --prompt-checkpoint /models/sam2_hiera_tiny.pt \
  --prompt-config sam2_hiera_t.yaml --prompt-device mps \
  --propagate-max-slices 64 --propagate-radius 16 \
  --label-layer ground_truth=/local-ssd/evaluation/neuron_30002/cache/target \
  --label-layer sam2_original=/local-ssd/evaluation/neuron_30002/unregistered/mask \
  --label-layer sam2_ecc=/local-ssd/evaluation/neuron_30002/ecc/mask \
  --label-layer sam2_sofima=/local-ssd/evaluation/neuron_30002/sofima/mask \
  --output-dir /local-ssd/demo
```

In the window browser, mito/syn/ves appear only in the full overview. Add `--align-field`
with the saved ECC field to reuse the exact evaluation transform. Registration
is on by default, `showSlices` is false, and meshes use 32 x 32 x 30 nm here.
`A` switches the display coordinates of every layer together. It does not rerun
the saved comparisons or change how those predictions were produced. Interactive
`V` uses the display space selected when that new task starts. See the
[interactive controls](README_interactive.md) for point correction and acceptance.

The LC-NE test mito/syn/ves volumes are grayscale context images with their own
precomputed metadata and display scales. Their physical extent matches raw.
They are not instance labels or calibrated probabilities, and they are not fed
into the SAM2 comparison.

## Process-window comparison

The 2026-11-05 comparison uses five new **512 x 512 x 63** cached windows at
8 x 8 x 30 nm. Entire boxes exclude all five estimated soma regions. The selected
processes lie approximately 11.0, 15.6, 23.2, 29.9 and 38.0 micrometers from
their respective soma centers. These distances are Euclidean, not neurite path
lengths. Fine-grid checks confirm target presence in every cached section.

Generic SAM2 Hiera Tiny on MPS propagated **33 central sections** from one
positive interior point. Each method used the same native point and fresh video
memory. Dice below covers only the 32 unprompted sections; no negative points or
GT masks were supplied to the model.

| Neuron | Original input Dice | ECC Dice |
| --- | ---: | ---: |
| 30001 | 0.0023 | 0.2557 |
| 30002 | 0.7940 | 0.1603 |
| 30003 | 0.2948 | 0.2833 |
| 30004 | 0.8952 | 0.9046 |
| 30005 | 0.1100 | 0.1155 |

The process benchmark is substantially harder than the historical soma pilot.
ECC's 30001 and 30002 predictions cover almost all target voxels but have
precision only 0.1466 and 0.0871, respectively, indicating extensive excess area.
Several other predictions have high precision and very low recall, indicating
missing process coverage. Improved image alignment does not guarantee better
segmentation. Keep both results available for inspection and use corrective
positive/negative points; these measurements do not establish whole-neuron
tracing accuracy. SOFIMA was not evaluated on these new process windows.

Saved prediction layers cover only the evaluated 33-section subrange; zeros
outside it mean **not evaluated**, not validated background. Interactive
propagation can select a different range within the 63-section cache. Evaluation
frames, native masks, field fingerprints, conditioning points and checkpoint
metadata remain in the local run directory, outside the source repository.

## Scaling beyond this pilot

Use a coarse overview to locate an annotated object or problematic section,
then read a bounded window with sufficient XY context. Estimate and cache the
local field there, retaining native coordinates for points and accepted masks.
The existing viewer and evaluator implement this bounded workflow.

For longer tracing, a subsequent extension should use overlapping windows and
check overlap masks and field consistency before continuing. A field from one
window must not be applied outside its estimated domain. Global stitching,
automatic window continuation and complete-neuron evaluation are not implemented
by this evaluator. Whole-volume SOFIMA is unnecessary for this local review
workflow and would need a separate distributed alignment pipeline.
