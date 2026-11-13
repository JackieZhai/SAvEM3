# Local realignment

The Fish-X LC-NE Test process windows use supplied local residual fields. All five use the
same wide-context ECC translation method and parameters. A separate field is
fitted to each window, with its middle section fixed and its linear XY drift
removed to preserve the whole-volume SOFIMA coordinate trend.

Estimation uses a 512 x 512 x 63 context at 32 x 32 x 30 nm. The fields are
applied to the 512 x 512 x 63 demo windows at 8 x 8 x 30 nm. Settings are a
maximum pair shift of 800 nm, 128 nm coarse estimation, minimum NCC 0.25,
minimum overlap 0.7, and neighbor gaps 1 and 2. The release includes the fields
and metadata; opening the demo does not require re-estimating them.

## Viewer controls

Local realignment starts enabled. Press **R** to toggle it. Raw, labels, meshes,
points, and the navigation position transform together. **Shift+R** restores
reference result selections. The overview uses the dataset geometry without
additional local realignment.

## Other local data

`demo.py` supports these options:

| Option | Method |
| --- | --- |
| `--align-field field.npz` | Load an explicit affine or elastic field |
| `--align-motion translation` | ECC with a robust translation graph; default estimator |
| `--align-motion euclidean` | ECC rotations and translations |
| `--align-motion affine` | ECC affine transforms |
| `--align-backend sofima` | ECC initialization with SOFIMA elastic refinement |
| `--no-align-z` | Disable local realignment |

An explicit field bypasses estimation. SOFIMA requires the
[optional worker](README_sofima.md) to estimate fields, but not to load them.
The Fish-X LC-NE Test residual fields are dataset-specific; generic ECC estimation does not
automatically remove a linear trend from new data.

## Coordinates and caches

Fields map absolute native voxel coordinates into a registered display grid.
Images use linear interpolation and labels use nearest-neighbor sampling.
Output bounds expand to contain the transformed ROI. Fields, coordinates, and
mask mapping are shared by the viewer and interactive controller.

Image prediction uses native image pixels. Video propagation uses the selected
coordinate frame at task start and saves masks back in native coordinates.
Automatically estimated field caches include raw content, geometry, and estimator
settings. Use a field whose resolution, bounds, and source match the current ROI.

Inspect XZ/YZ geometry and boundary correspondence as well as correlation.
Correlation alone does not measure neuron segmentation accuracy. For a comparison
on a local cache:

```bash
python probe-em/scripts/demo_windows.py --dataset examples --prepare-only --output-dir .cache/fish-x-review
python probe-em/scripts/compare_alignment.py \
  --local-dir .cache/fish-x-review/dataset/process_windows/neuron_30003/cache \
  --sofima-python /envs/sofima/bin/python \
  --output-dir /data/alignment-comparison
```
