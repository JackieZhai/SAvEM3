# Interactive point-prompt review in Neuroglancer

`scripts/demo.py --interactive` adds an explicit image-prediction workflow:
place positive/negative points, explicitly predict the current section or propagate
with SAM2 through a selected z range, inspect the result, then accept it into a
local review artifact. Adding points does not start inference. A completed Probe-EM
trace is optional. Points are retained separately on each section of the same object.

## Start a local session

Run from the repository root in the environment described in [INSTALL.md](INSTALL.md):

HQ-SAM/SAEM2 also require `python -m pip install -r probe-em/requirements-saem2.txt`.

```bash
python probe-em/scripts/demo.py \
  --local-dir /absolute/path/to/cached_volumes \
  --seed 1459916967 \
  --interactive --prompt-backend sam2 \
  --prompt-checkpoint /absolute/path/to/sam2_hiera_tiny.pt \
  --prompt-config sam2_hiera_t.yaml \
  --prompt-device auto \
  --output-dir .cache/interactive-session
```

The local directory must contain matching `raw/` and `seg/` precomputed volumes.
The seed selects the initial navigation location in the existing labels; it is
not supplied to the model as a mask or a point. Add `--results-dir` to load a
completed trace for comparison. The trace starts hidden in interactive mode so
it does not obscure the prediction; enable `probe_trace` in the layer list.

Open the printed URL and keep the Python process running. Local registration is
on by default; use `--align-field /path/to/field.npz` to reuse a validated field,
`--align-motion translation|euclidean|affine` to choose an estimator when no field
is supplied, or `--no-align-z` to disable it. The current recommendation remains
translation with the robust graph solver; see [registration evidence](README_alignment.md).
`showSlices` defaults to false. Meshes use the existing near-isotropic policy.
For optional nonrigid registration, add `--align-backend sofima --sofima-python
/path/to/sofima-env/bin/python`, or load its saved field with `--align-field`.
See the [SOFIMA guide](README_sofima.md) for installation and quality controls.

## Annotate, run, refine and accept

| Action | Control |
| --- | --- |
| Add a positive point inside the target | Hover in the XY panel and press **P** |
| Add a negative point on an unwanted neighbor/background | Hover in the XY panel and press **N** |
| Predict the currently viewed XY section from its points | **I** |
| Move to the previous/next XY section | **J** / **K**, or normal Neuroglancer navigation |
| Set the propagation range's first/last section | **B** / **E** at the desired sections |
| Propagate all conditioning points through the selected range (SAM2) | **V** |
| Cancel the current task and discard its pending preview | **X** |
| Undo the last point added with P/N | **U** |
| Start another object: clear all sections' points and unaccepted preview | **C** |
| Accept and save the current image or complete volume preview | **Y** |
| Compare registered/original display | **A** |

Click once in the XY panel to give it keyboard focus, then hover and use the keys.
Keys are unmodified letters, not Shift+letter. Green `positive_points` and red
`negative_points` are native editable Neuroglancer annotation layers. Their
annotation tabs also support manual edits/deletion. U only undoes points added
through P/N. All points belong to the same target object. Negative points are
optional, and each prompted section needs at least one positive point. Use C when
starting a different target; changing sections does not require clearing points.
Use the XY panel for placement. P/N bindings also receive positions from other
slice panels, but points are interpreted as XY prompts on their snapped z section.

I reads only the points on the currently viewed XY section. If it has no positive
point, the status message asks for one; it never silently reruns a previous section.
For example: P/N -> I -> K -> P/N -> I works without C. Returning with J lets you
edit and rerun the previous section's retained points. A direct annotation-tab
edit may move a point to another section; its z coordinate determines its group.
P/N always use the current XY section's z, even when mouse picking snaps to an
annotation on a neighboring section. This prevents old points from trapping new
annotations on the previous slice.

`sam_preview` is the latest explicitly computed mask (yellow). Edit points and
press I (image) or V (volume) to refine it. A manual annotation edit does not
automatically recompute the preview; Y checks its conditioning points and rejects
stale masks. P/N/U/C and range changes invalidate the preview. Inference runs in
one background worker; repeated I/V presses cannot queue extra jobs. Edits during
inference discard the old result. X cancels cooperatively: the current model step
or frame-loading operation may finish first. Loading/inference errors appear in
the status line and can be retried.

Y marks the result accepted and shows it as `sam_accepted` (green). This layer
shows the most recent accepted image or volume; every earlier result stays on disk. C starts
a new point set while retaining accepted results. Existing instance labels and
Probe-EM merge decisions are not edited by acceptance.

## SAM2 propagation across sections

The initial range is the starting section plus/minus four sections, clipped to
the local ROI. The status line always shows its inclusive absolute z endpoints.
Use B on the lower section and E on the upper section to select another range.
`--propagate-radius` changes the initial radius; `--propagate-max-slices` sets the
per-task limit (default 32). At least two sections are required. Prompts outside
the range are rejected with a message to expand the range or remove those points.

V uses the actual [SAM2 video predictor and temporal memory](https://github.com/facebookresearch/sam2#video-prediction).
Each EM section is a frame, and all prompted sections condition object 1. It
tracks from the earliest prompted section toward increasing z, then backward to
the range start. I is optional before V. To correct a result, add positive/negative
points on the problematic section and press V again. Each run starts fresh video
memory with all current conditioning points; loaded model weights are reused.
SAM, HQ-SAM and SAEM2 image backends support I; V requires `--prompt-backend sam2`.

Video inference uses the display coordinate frame selected when V is pressed:
registered images/points when local registration is on, original images/points
when it is off. This includes the SOFIMA nonlinear field. Toggling A during a task
changes its display only; that task's input frame stays fixed. Masks are mapped
back to the original native grid before saving and are rewarped for display.
The original local data are preserved. The legacy SAM2 directory API consumes
JPEG frames; quality-100, 4:4:4 files and their SHA-256 hashes are retained for
exact replay. JPEG encoding is not bit-exact raw-image storage.

Frames are held in CPU memory. On MPS, video masks and temporal memory stay on
the device: CPU-offloaded states produced nonfinite logits in local validation
of the installed legacy SAM2 runtime. CPU inference and MPS inference were tested;
CUDA video propagation is not validated by these local runs.

The volume preview spans the full selected range. Scroll or J/K to inspect
sections without prompts; Y accepts the whole preview. Empty output sections
are reported, and X stops a running task. No automatic anatomical stopping or
complete-neuron criterion is claimed. This is bounded one-object propagation
with user review; existing reconstruction IDs and merge/split decisions are retained.

## Model options

| `--prompt-backend` | Required local weights | Behavior |
| --- | --- | --- |
| `sam` | Original SAM checkpoint matching `--prompt-model-type` | Original SAM mask decoder |
| `hq_sam` | Full HQ-SAM checkpoint matching the ViT type | SAM + HQ mask output |
| `sam2` (default) | SAM 2 checkpoint and matching `--prompt-config` | SAM 2 image predictor |
| `saem2` | Full HQ-SAM `--prompt-checkpoint` plus trained `--prompt-decoder` | Historical SAEM2 SAM + HQ logits; no calibrated score |

`--prompt-model-type` defaults to `vit_b` and also supports `vit_l`/`vit_h`.
`--prompt-device` accepts `auto`, `cpu`, `mps`, `cuda`, or `cuda:N`. Select the
backend at launch. Model weights load on the first I/V action; predictors and the
last image-section embedding are reused. No weights or volume data are downloaded
by this entry point. SAM 2 requires the separate local SAM 2 installation.

One image mask is requested for the current section's point set. Scores are model
estimates, not accuracy measurements. These image predictors do not establish
the performance of NeuroSAM 2 or reproduce the paper's quantitative results.

## Coordinates, local cache and scope

Image inference uses the original native-resolution raw section. Display-space
points are mapped back through the same field used by the viewer. Padding/out-of-ROI
points and contradictory labels at the same XYZ pixel are rejected. Z is snapped
to the containing voxel cell (`floor(z)`); points use absolute XYZ voxel units,
not nanometers. Images and masks passed to the model use local YX array order.
The XY view samples the section center (`z + 0.5`) so a one-section mask is visible.
Toggling alignment transforms the annotations and mask overlay together, including
when a prediction is running. The task's selected model input stays the same.

Each completed image prediction gets a unique `prediction_*` directory under
`<output-dir>/interactive/`:

- `prediction.npz`: native `mask_yx`, cached `raw_rgb_yxc`, original XYZ points,
  labels and that section's display transform. Read with `allow_pickle=False`.
- `metadata.json`: model/device/options, checkpoint SHA-256 hashes, source cache,
  raw image hash, grid/offset, point set, timestamp and predicted score if present.
- `accepted.json`: created only after Y; distinguishes accepted results from previews.

Affine predictions store `display_matrix`. Elastic predictions instead store
`display_field_kind`, `display_base_shift_xy`, `display_displacement_cyx`,
`display_grid_origin_xy` and `display_grid_spacing_xy`. These describe the actual
nonlinear display transform; original points and `mask_yx` stay in native space.

Video jobs use separate `propagation_*` directories with:

- `frames/`: the exact sequential JPEG frames used for SAM2, including registered padding when enabled.
- `prediction.npz`: native `mask_zyx`, absolute `frame_z`, `processed` flags, original XYZ points and labels.
- `alignment.npz`: the complete affine/elastic display field, when available.
- `metadata.json`: range, input frame/weights hashes, coordinate frame, empty sections and job status.
- `accepted.json`: written only after Y accepts the completed volume preview.

Cancelled or failed jobs retain their diagnostic metadata and any encoded inputs,
but never publish a valid preview or accepted output. Each completed job must
return every requested section. A result may still contain empty masks or tracking
errors; it does not assign a persistent neuron ID or modify existing instances.

`--check` validates viewer setup without loading model weights or running a
prediction. Tests in `tests/test_interactive.py` cover actual action dispatch,
coordinate transforms, stale-result rejection and local acceptance.
`tests/test_interactive_propagation.py`
checks section changes, range limits, cancellation/recovery, dense coordinates,
stale-volume rejection and real bidirectional SAM2 memory. Browser rendering and
multi-section interaction are checked separately.
