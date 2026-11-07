# Interactive segmentation

Place positive and negative points in Neuroglancer, predict an image or a short
volume, and save the reviewed mask. Use the [LC-NE window browser](README_windows.md)
for the supplied test data.

## Controls

| Action | Key |
| --- | --- |
| Add a positive point inside the target | **P** |
| Add a negative point on an unwanted region | **N** |
| Predict the current section | **I** |
| Previous / next Z section | **D / F** |
| Set first / last propagation section | **B / E** |
| Propagate through the selected range with SAM2 | **V** |
| Cancel a running task | **X** |
| Undo the last point added with P/N | **U** |
| Clear points and unaccepted preview for a new object | **C** |
| Accept and save the current preview | **Y** |
| Toggle local realignment | **R** |
| Restore reference result selections | **Shift+R** |

Click the XY panel for keyboard focus. Hover over the image when placing P/N
points. Press letter keys without Shift unless explicitly shown. D/F also works
while dragging, including Shift-drag, and stops at the cached volume bounds.
Text editing does not move sections.

Green `positive_points` and red `negative_points` are editable annotation layers.
Their annotation tabs support manual edits and deletion. Each prompted section
needs at least one positive point. Points across sections describe the same
object; press C before starting another object. Switching sections retains points.

## Image prediction

Add points and press I. Placing a point alone does not run inference. With SAM2,
the controller checks point agreement and evaluates refinement candidates when
a prediction includes a negative point or misses a positive point. The toolbar
reports unresolved point conflicts. Inspect the boundary as well as those counts.

Add or move points and press I again to refine the result. The controller reuses
image features and compatible mask history for that object and section. Editing
or removing an earlier point invalidates incompatible history.

## Volume propagation

Move to the first section and press B, then the last section and press E. Bounds
are inclusive and must contain all annotated sections. The window browser permits
up to 64 sections by default; its initial range is the current section plus four
on either side. Each process window contains 63 sections.

Press V to predict the selected volume. For each prompted section, V reuses its
matching I result or computes an image prediction automatically. These corrected
masks initialize SAM2 video memory. Prompted sections retain their native image
masks; unprompted sections use SAM2's propagated predictions. I is optional
before V.

Inspect several unprompted sections with D/F. Add P/N points on a problematic
section and run V again. Each run starts fresh temporal memory. X cancels a job;
Y accepts the complete volume preview. The model follows one object within the
selected range and may still merge, split, or lose thin processes.

Local realignment selects the video coordinate frame when V starts. Toggling R
while a job runs changes the display, not that job's inputs. Saved masks always
use the source cache's native grid.

## Standalone session

```bash
python probe-em/scripts/demo.py \
  --local-dir /data/LC-NE-test/demo/process_windows/neuron_30001/cache \
  --seed 30001 --interactive --prompt-backend sam2 \
  --prompt-checkpoint /models/sam2_hiera_tiny.pt --prompt-config sam2_hiera_t.yaml \
  --prompt-device auto \
  --align-field /data/LC-NE-test/demo/process_windows/neuron_30001/alignment/field.npz \
  --output-dir /data/interactive-review
```

The seed controls navigation through reference labels; it is not a model prompt.
The cache must contain matching `raw/` and `seg/` precomputed volumes. Use
`--results-dir` to add a completed trace. See [local realignment](README_alignment.md)
and [runtime options](README_acceleration.md) for optional settings.

| Backend | Weights | Actions |
| --- | --- | --- |
| `sam2` | Checkpoint and matching model config | I and V |
| `sam` | Original SAM checkpoint | I |
| `hq_sam` | Full HQ-SAM checkpoint | I |
| `saem2` | HQ-SAM checkpoint plus `--prompt-decoder` | I |

For SAM/HQ-SAM/SAEM², `--prompt-model-type` selects `vit_b`, `vit_l`, or `vit_h`.
`--prompt-device` accepts `auto`, `cpu`, `mps`, `cuda`, or `cuda:N`. Models load
on the first inference action. See [INSTALL.md](INSTALL.md) for dependencies.

## Saved results

Predictions are stored beneath `OUTPUT/interactive/` in standalone sessions,
or `OUTPUT/window_NNN/interactive/` in the window browser.

| Path | Contents |
| --- | --- |
| `prediction_*/prediction.npz` | Native `mask_yx`, input image, points, labels, display transform, and refinement data |
| `propagation_*/prediction.npz` | Native `mask_zyx`, absolute `frame_z`, processed flags, points, and labels |
| `propagation_*/frames/` | Encoded input frames used by the video predictor |
| `metadata.json` in each job | Model settings, hashes, geometry, point checks, and status |
| `accepted.json` in each accepted job | Acceptance record written by Y |

Read NPZ files with `allow_pickle=False`. Points use absolute XYZ voxel units;
model masks use YX or ZYX order. Image inference uses native raw pixels, with
annotation coordinates mapped back from the displayed field. Z selects the
containing voxel cell. Duplicate contradictory points and out-of-bounds points
are rejected.

Cancelled, failed, or stale predictions are not published as valid previews.
The original raw and label volumes remain separate from review outputs.
