# Neuroglancer test windows

The **SAEM²-SAEvM³** toolbar provides a window selector, **Previous**, **Next**,
and **Open in tab**. Its right side shows the active window's prediction and
local realignment status. Click the status text to expand it.

## Start the LC-NE test demo

Install [Probe-EM and SAM2](INSTALL.md) and extract the [LC-NE test release](../docs/LC_NE_TEST.md)
according to its `TUTORIAL.md`. From the repository root:

```bash
python probe-em/scripts/demo_windows.py \
  --windows /data/LC-NE-test/demo/windows.json \
  --checkpoint /models/sam2_hiera_tiny.pt \
  --model-config sam2_hiera_t.yaml --device auto \
  --output-dir /data/review-output --open
```

For the five process windows without the overview, use
`/data/LC-NE-test/demo/windows.roi-only.json`. That mode needs only the small
demo archive. Keep the Python server running while using the viewer.

The overview shows the whole cropped dataset, neuron meshes, and named boxes
for the five process windows. Mito, syn, ves, lcpre, and lcsyn are available only
in the overview and start hidden. Click a layer tab to change its visibility.
Choose a process window for interactive prediction.

Each process window loads its supplied local field. **R** toggles local
realignment. **D/F** moves through Z, including while dragging. **P/N** adds
positive/negative points, **I** predicts a section, and **V** propagates through
the range set with **B/E**. See [interactive controls](README_interactive.md).

## Display and session options

Views use `4panel-alt`, visible axes, and `showSlices=false`. Process meshes use
near-isotropic sampling; the image sections retain their native resolution.
`--no-align-z` disables local realignment. The overview never applies an additional
local field to the already aligned dataset.

The initial propagation range extends four sections to either side of the current
section. Set `--propagate-radius` and `--propagate-max-slices` to change it; the
window browser's maximum defaults to 64. Model settings are described in the
[runtime guide](README_acceleration.md).

Windows load on demand and retain separate points, sections, ranges, and results
for the lifetime of the server process. Previous/Next wraps at either end.
`#window=N` selects a zero-based window in a saved URL. Restarting Python starts
new interactive sessions; it does not automatically restore annotations.

Predictions and acceptance records are stored under
`OUTPUT/window_NNN/interactive/`. Requested overview chunks are cached under
`OUTPUT/overview_cache/`, or `--overview-cache /local/cache`. Use a new cache
when replacing source data. A partial chunk cache is not a complete offline copy.

## Custom manifests

`demo_windows.py` accepts a JSON object with `schema_version: 1`, `initial_index`,
and a nonempty `windows` array. All local paths resolve relative to the manifest.

| Entry | Required fields | Optional fields |
| --- | --- | --- |
| Overview | `kind: "overview"`, `name`, `raw_path`, `seg_path` | `segments`, `image_layers`, `label_layers`, `visible_layers` |
| Process ROI | `name`, `local_dir`, positive integer `seed` | `kind: "roi"`, `align_field`, `position_xyz`, `traced_ids`, auxiliary layers |

An ROI's `local_dir` contains matching `raw/` and `seg/` precomputed volumes.
`position_xyz` uses absolute native voxel coordinates. `image_layers` and
`label_layers` map layer names to precomputed paths; `visible_layers` lists
initially visible auxiliary layers. ROI `colors` maps binary label-layer names
to colors for ID 1. Mito/syn/ves names are restricted to overview entries.

Overview volumes must share a physical extent. ROI auxiliary volumes must share
the raw grid. Bounding boxes are derived from each ROI's `raw/info` and transformed
into the overview's voxel coordinates. A supplied alignment field is loaded as-is;
without one, the viewer estimates an ECC field from that ROI.
