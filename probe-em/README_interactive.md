# Interactive segmentation

Place positive and negative points in Neuroglancer, predict an image or a short
volume, and save the reviewed mask. Use the [Fish-X LC-NE Test window browser](README_windows.md)
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
| Save the current object and its points, then start another | **C** |
| Accept/save all objects as one combined mask | **Y**, then two confirmations |
| Confirm the next acceptance step / cancel acceptance | **Enter / Esc** |
| Toggle local realignment | **R** |
| Restore reference result selections | **Shift+R** |

Click the XY panel for keyboard focus. Hover over the image when placing P/N
points. Press letter keys without Shift unless explicitly shown. D/F also works
while dragging, including Shift-drag, and stops at the cached volume bounds.
Text editing does not move sections.

The active object's color identifies `positive_points`; `negative_points` are red.
Their annotation tabs support manual edits and deletion. Each prompted section
needs at least one positive point. Active points across sections describe the same
object. Switching sections retains points.

## Multiple objects and final acceptance

Predict an object with I or V, then press **C** or click **New object (C)**.
The finished mask remains in its own layer, such as `sam_object_001`. Its positive
and negative points are retained as `sam_object_001_positive_points` and
`sam_object_001_negative_points`. Only nonempty point layers are saved and
restored; an object without negative points has no saved negative-point layer.
A fresh pair of active point layers starts the
next object in a different color. Its predictions use only its own active points
and mask history. C requires a valid preview; unfinished or changed prompts are
retained until you run I or V again.

Repeat for additional objects. When ready, press **Y** or click **Accept/save (Y)**.
This includes all saved objects in the current window and the current valid
preview, if present. Press **Enter** for confirmation 1/2, then press **Enter**
again for confirmation 2/2; the toolbar also provides confirmation buttons.
**Esc** or **Cancel** aborts without discarding objects, points or the preview.
Editing active points cancels the pending acceptance; confirmations expire after
60 seconds. Holding a key does not supply both confirmations.

The final native-grid union is saved and shown as `sam_final`. Individual mask
layers are hidden to show the union, but remain available, and every object's
point layers are kept. The original per-object files remain available as well.
Overlapping voxels appear once in the union; gaps are not filled. Each test window
has an independent object collection. Starting another object uses a new color
and hides the previous final overlay.

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
on either side, or the saved range when loading a starter session. Each process
window contains 63 sections.

Press V to predict the selected volume. For each prompted section, V reuses its
matching I result or computes an image prediction automatically. These corrected
masks initialize SAM2 video memory. Prompted sections retain their native image
masks; unprompted sections use SAM2's propagated predictions. I is optional
before V.

Inspect several unprompted sections with D/F. Add P/N points on a problematic
section and run V again. Each run starts fresh temporal memory. X cancels a job;
C saves that object's complete volume preview and starts the next object.
Y performs final acceptance as described above. Each prediction follows one object within the
selected range and may still merge, split, or lose thin processes.

Local realignment selects the video coordinate frame when V starts. Toggling R
while a job runs changes the display, not that job's inputs. Saved masks always
use the source cache's native grid.

## Standalone session

Prepare the bundled Fish-X LC-NE Test boxes once, then open an individual box:

```bash
python probe-em/scripts/demo_windows.py --dataset examples --prepare-only --output-dir .cache/fish-x-review
python probe-em/scripts/demo.py \
  --local-dir .cache/fish-x-review/dataset/process_windows/neuron_30001/cache \
  --seed 30001 --interactive --prompt-backend sam2 \
  --prompt-checkpoint /models/sam2_hiera_tiny.pt --prompt-config sam2_hiera_t.yaml \
  --prompt-device auto \
  --align-field .cache/fish-x-review/dataset/process_windows/neuron_30001/alignment/field.npz \
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

The supplied five-window demo loads [bundled example reviews](examples/fish_x_lc_ne/README.md)
into fresh output directories. Their masks and point layers are already saved;
add points to trace the next object in a new color. Use `--no-starter-session`
with a new output directory to begin without the example objects. Existing
review directories are never replaced by the starter session.

Predictions are stored beneath `OUTPUT/interactive/` in standalone sessions,
`OUTPUT/neuron_<ID>/interactive/` for the supplied Fish-X LC-NE Test options,
or `OUTPUT/window_NNN/interactive/` for custom window manifests.

| Path | Contents |
| --- | --- |
| `prediction_*/prediction.npz` | Native `mask_yx`, input image, points, labels, display transform, and refinement data |
| `propagation_*/prediction.npz` | Native `mask_zyx`, absolute `frame_z`, processed flags, points, and labels |
| `propagation_*/frames/` | Encoded input frames used by the video predictor |
| `metadata.json` in each job | Model settings, hashes, geometry, point checks, and status |
| `accepted.json` in each saved object job | Object ID, color and retained native point annotations |
| `review_objects.json` | Saved object collection, native geometry and next object ID |
| `object_*/prediction.npz` | Bundled starter masks and point arrays in native coordinates |
| `session.json` | Starter view, layer visibility, and propagation range |
| `merge_*/prediction.npz` | Final combined native `mask_zyx` and absolute `frame_z` |
| `merge_*/metadata.json` and `accepted.json` | Final acceptance and source-object records |

Read NPZ files with `allow_pickle=False`. Points use absolute XYZ voxel units;
model masks use YX or ZYX order. Image inference uses native raw pixels, with
annotation coordinates mapped back from the displayed field. Z selects the
containing voxel cell. Duplicate contradictory points and out-of-bounds points
are rejected.

Cancelled, failed, or stale predictions are not published as valid previews.
Saved objects and their point layers reload when reopening the same source and
output directory. Unfinished active prompts are retained during window switching;
they are not automatically restored after restarting the Python server.
The original raw and label volumes remain separate from review outputs.
