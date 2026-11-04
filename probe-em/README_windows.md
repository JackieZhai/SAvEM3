# Switch between cached Neuroglancer test windows

`scripts/demo_windows.py` serves one browser page with **Previous window**,
**Next window**, a named window selector and **Open in tab**. Each window has its
own Neuroglancer viewer, points, current section, propagation range, preview and
accepted result. Switching returns to that state; it does not start inference.
Click the image for keyboard focus: **D/F** moves one section, **P/N** adds points,
**I** predicts the current section and **V** propagates through the **B/E** range.
See the [interactive guide](README_interactive.md) for editing and acceptance.

## Describe the cached windows

Create a local JSON manifest. Paths are relative to the manifest, or absolute.
Each `local_dir` contains matching `raw/` and `seg/` precomputed volumes. `seed`
must be an existing positive segment ID; it controls navigation, not model input.

```json
{
  "schema_version": 1,
  "initial_index": 0,
  "windows": [
    {
      "name": "Neuron A",
      "local_dir": "neuron_a/cache",
      "seed": 30001,
      "position_xyz": [6246, 5073, 1212],
      "align_field": "neuron_a/ecc/alignment/field.npz",
      "label_layers": {
        "ground_truth": "neuron_a/cache/target",
        "sam2_ecc": "neuron_a/ecc/mask"
      },
      "visible_layers": ["ground_truth", "sam2_ecc"],
      "colors": {"ground_truth": "#2a83ff", "sam2_ecc": "#49d468"}
    },
    {
      "name": "Neuron B",
      "local_dir": "neuron_b/cache",
      "seed": 30002
    }
  ]
}
```

Only `name`, `local_dir` and `seed` are required for each window.
`position_xyz` uses absolute native voxel coordinates at the cached resolution;
the viewer maps it into the registered display. `traced_ids` optionally selects
existing trace labels. `image_layers` adds named context images, such as mito,
syn and ves. All auxiliary volumes must match the native raw grid. `colors`
assigns the displayed ID **1** in binary auxiliary masks; it is intended for
target/prediction volumes, not arbitrary multiclass labels. `visible_layers`
sets initial visibility for auxiliary layers only.

## Start the browser

From the repository root in the installed Probe-EM environment:

```bash
python probe-em/scripts/demo_windows.py \
  --windows /path/to/windows.json \
  --checkpoint /path/to/sam2_hiera_tiny.pt \
  --model-config sam2_hiera_t.yaml --device auto \
  --output-dir .cache/window-review --open
```

Local registration is enabled by default. A supplied field is reused; otherwise
ECC estimates it from the bounded ROI. A saved SOFIMA field is also supported.
`--no-align-z` skips registration, and **A** toggles the registered/original view
when a field is available. `showSlices=false` and near-isotropic meshing remain
the defaults. The initial propagation radius is four sections; use
`--propagate-radius` and `--propagate-max-slices` to change the range and limit.

Windows load lazily, with a loading message and an explicit retry button after
failure. Next/Previous wrap at the ends. A loading window stays hidden to prevent
placing points in the previous window by mistake. The URL's `#window=N` selects
the initial zero-based index. Old standalone demo processes are unaffected.

One SAM2 image model and one video model serve the collection. Model calls are
serialized, while each window owns its annotations and starts fresh temporal
memory for each propagation. A 512 MiB CPU feature cache is shared across windows;
it evicts old features without discarding annotations. See the
[acceleration guide](README_acceleration.md) for runtime options and measurements.

Keep the Python server running. A switch preserves state for that process's
lifetime, including pending inference in other windows. It is not an automatic
restore mechanism after restarting Python. Prediction inputs, masks and acceptance
records are saved under `OUTPUT/window_NNN/interactive/`; local alignment and
mesh caches are separate subdirectories. Both the toolbar URL and each loaded
window's standalone URL are saved as `viewer_url.txt`.

The local five-neuron demonstration uses the bounded windows described in the
[local GT guide](README_local_gt.md). Raw, labels and comparison results are read
from local caches; this browser does not download or register the whole volume.
