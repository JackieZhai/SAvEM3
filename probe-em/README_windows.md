# Neuroglancer test windows

The **SAEM²-SAEvM³** toolbar provides a window selector, **Previous**, **Next**,
and **Open in tab**. Its right side shows the active window's prediction and
local realignment status. Click the status text to expand it.
**New object (C)** saves the current object and its point layers before starting
another in a new color. **Accept/save (Y)** combines the window's objects after
two confirmations. The matching keys work inside the viewer.

## Start the Fish-X LC-NE Test demo

Install [Probe-EM and SAM2](INSTALL.md), including matching SAM2 weights. There
are three dataset options. Run the commands below from the repository root.

### 1. Bundled lc overview and five examples (default)

The repository includes the full `lc` segmentation and meshes, the five
`test_windows` bounding boxes, and all five process examples with their labels
and local fields in [examples/fish_x_lc_ne/](examples/fish_x_lc_ne/). The bundle
totals about 114 MiB. No dataset download or Hugging Face login is needed.
The launcher extracts and caches the data automatically.

```bash
python probe-em/scripts/demo_windows.py \
  --dataset examples \
  --checkpoint /models/sam2_hiera_tiny.pt \
  --model-config sam2_hiera_t.yaml --device auto \
  --output-dir .cache/fish-x-review --open
```

Omitting `--dataset` also selects `examples`. It starts at
**Fish-X LC-NE Test overview**, showing `lc` and `test_windows`, followed by the
five locally realigned examples. Raw images are available within each example;
the full option supplies whole-volume raw and context images.

Fresh output directories load the [bundled starter session](examples/fish_x_lc_ne/README.md):
eight saved objects and 55 annotation points across the five process windows.
Masks, object colors, visibility, camera views, and propagation ranges are restored.
The next object receives a new color. Existing reviews take priority. To start
without saved objects, add `--no-starter-session` and use a new `--output-dir`.

### 2. Full dataset: overview and five examples

[Download and extract the full dataset](../docs/LC_NE_TEST.md#2-full-dataset-from-hugging-face)
from Hugging Face first, then point `--data-root` at the directory containing the
seven extracted `zfishlc_*` volumes:

```bash
python probe-em/scripts/demo_windows.py \
  --dataset full --data-root /data/Fish-X-LC-NE-Test \
  --checkpoint /models/sam2_hiera_tiny.pt \
  --model-config sam2_hiera_t.yaml --device auto \
  --output-dir .cache/fish-x-review --open
```

This starts at **Fish-X LC-NE Test overview**, followed by the same five bundled
examples. Full volumes must be extracted before launch; the viewer does not
download them in the background. Keep the Python server running while using
any option.

The full overview shows the whole cropped dataset, neuron meshes, and named boxes
for the five process windows. Mito, syn, ves, lcpre, and lcsyn are available only
in the overview and start hidden. Click a layer tab to change its visibility.
Choose a process window for interactive prediction.

Each process window loads its supplied local field. **R** toggles local
realignment. **D/F** moves through Z, including while dragging. **P/N** adds
positive/negative points, **I** predicts a section, and **V** propagates through
the range set with **B/E**. See [interactive controls](README_interactive.md).
`ground_truth` uses neutral gray (`#b0b0b0`), distinct from the first five object colors.

### 3. Nucleus segmentation on the overview

Use the extracted raw volume with `--dataset nuclei`:

```bash
python probe-em/scripts/demo_windows.py \
  --dataset nuclei --data-root /data/Fish-X-LC-NE-Test \
  --checkpoint /models/sam2_hiera_tiny.pt \
  --output-dir .cache/fish-x-review --open
```

The overview uses **32 x 32 x 30 nm** raw data. **P** places one positive nucleus
center, **V** propagates above and below it, and **C** confirms the nucleus and
starts the next in a new color. All centers stay in one `nucleus_centers` layer.
The five process windows remain available. See the [nucleus guide](README_nuclei.md)
for propagation limits, the two-confirmation union export, and saved results.

## Display and session options

Views use `4panel-alt`, visible axes, and `showSlices=false`. Process meshes use
near-isotropic sampling; the image sections retain their native resolution.
`--no-align-z` disables local realignment in process windows. The overview never applies an additional
local field to the already aligned dataset.

In process windows without a starter session, the initial propagation range extends four sections
to either side of the current section. Set `--propagate-radius` and
`--propagate-max-slices` to change it; the
window browser's maximum defaults to 64. Model settings are described in the
[runtime guide](README_acceleration.md).

Windows load on demand and retain separate points, sections, ranges, and objects
for the lifetime of the server process. Previous/Next wraps at either end.
`#window=N` selects a zero-based window in a saved URL. Restarting Python starts
new interactive sessions and reloads saved objects and their point layers from
the same output directory. Unfinished active process-window annotations are not
automatically restored. Nuclei mode also restores its active center; V recomputes
an unfinished preview.

Process predictions and acceptance records in all dataset modes are stored under
`OUTPUT/neuron_30001/interactive/` through `OUTPUT/neuron_30005/interactive/`.
Reusing the output directory preserves saved process objects when changing modes.
Nucleus reviews use `OUTPUT/nuclei/interactive/`. Bundled data are cached under
`OUTPUT/dataset/`. To prepare them
without starting a viewer or loading a model, use `--prepare-only` in place of
`--checkpoint ... --open`.

Requested overview chunks are cached under
`OUTPUT/overview_cache/`, or `--overview-cache /local/cache`. Use a new cache
when replacing source data. A partial chunk cache is not a complete offline copy.

## Custom manifests

Use `--windows PATH` instead of `--dataset` for a custom manifest. Existing
`demo/windows.json` and `demo/windows.roi-only.json` manifests from the Hugging
Face package remain supported. Custom manifests save reviews under
`OUTPUT/window_NNN/interactive/`; keep their output directory separate from
the supplied dataset options.

`demo_windows.py` accepts a JSON object with `schema_version: 1`, `initial_index`,
and a nonempty `windows` array. All local paths resolve relative to the manifest.

| Entry | Required fields | Optional fields |
| --- | --- | --- |
| Overview | `kind: "overview"`, `name`, `seg_path` | `raw_path`, `segments`, `image_layers`, `label_layers`, `visible_layers`, `test_windows_path` |
| Process ROI | `name`, `local_dir`, positive integer `seed` | `kind: "roi"`, `align_field`, `position_xyz`, `traced_ids`, `starter_session`, auxiliary layers |

An ROI's `local_dir` contains matching `raw/` and `seg/` precomputed volumes.
`position_xyz` uses absolute native voxel coordinates. `image_layers` and
`label_layers` map layer names to precomputed paths; `visible_layers` lists
initially visible auxiliary layers. ROI `colors` maps binary label-layer names
to colors for ID 1. Mito/syn/ves names are restricted to overview entries.

Overview volumes must share a physical extent. ROI auxiliary volumes must share
the raw grid. Bounding boxes are derived from each ROI's `raw/info` and transformed
into the overview's voxel coordinates. A label-only overview uses the segmentation
grid. `test_windows_path` can supply a saved Neuroglancer annotation layer; its
boxes and dimensions must match those derived from the ROI caches. A supplied alignment field is loaded as-is;
without one, the viewer estimates an ECC field from that ROI.
