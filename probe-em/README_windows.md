# Switch between cached Neuroglancer test windows

`scripts/demo_windows.py` serves the compact **SAEM²-SAEvM³** browser with
**Previous**, **Next**, a named window selector and **Open in tab**. Each ROI has its
own Neuroglancer viewer, points, current section, propagation range, preview and
accepted result. Switching returns to that state; it does not start inference.
Click the image for keyboard focus: **D/F** moves one section, **P/N** adds points,
**I** predicts the current section and **V** propagates through the **B/E** range.
See the [interactive guide](README_interactive.md) for editing and acceptance.

The desktop toolbar uses 12 px controls and a 31 px bar, matching the scale of
Neuroglancer's layer tabs. Alignment/seed information and propagation/prediction
status appear on its right side instead of below the images. Click that status
area to expand the complete messages; Escape closes it. Long messages are
ellipsized in the compact bar, not discarded. Standalone `demo.py` sessions keep
their native status messages.

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
      "kind": "overview",
      "name": "LC-NE test overview",
      "raw_path": "working/zfishlc_raw",
      "seg_path": "working/zfishlc_lc",
      "image_layers": {
        "mito": "working/zfishlc_mito",
        "syn": "working/zfishlc_syn",
        "ves": "working/zfishlc_ves"
      },
      "segments": [30001, 30002, 30003, 30004, 30005]
    },
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

For an ROI, only `name`, `local_dir` and `seed` are required; `kind` defaults to `roi`.
`position_xyz` uses absolute native voxel coordinates at the cached resolution;
the viewer maps it into the registered display. `traced_ids` optionally selects
existing trace labels. **Mito, syn and ves belong only to the overview**; ROI
manifests reject those names in both image and label layers. Other ROI auxiliary
volumes must match the native raw grid. `colors`
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

Local registration is enabled by default for ROIs. A supplied field is reused; otherwise
ECC estimates it from the bounded ROI. A saved SOFIMA field is also supported.
`--no-align-z` skips registration, and **A** toggles the registered/original view
when a field is available. `showSlices=false` and near-isotropic meshing remain
the defaults. The initial propagation radius is four sections; use
`--propagate-radius` and `--propagate-max-slices` to change the range and limit.

## Full-volume overview

An optional `kind: "overview"` entry adds **LC-NE test overview** to the selector. Its
`raw_path` and `seg_path` are local multiscale precomputed directories describing
the same physical extent. This view always uses original coordinates, with local
registration disabled. It does not run SAM2 or construct a whole-volume array.
Choose an ROI in the selector to use point prediction and propagation.

The `test_windows` annotation layer contains a named bounding box for every ROI.
Bounds come from each cached `raw/info`: offset and size are converted through
physical nm into the overview's native coordinates. Upper bounds are exclusive.
This correctly places the 8 nm test windows inside the 4 nm full `zfishlc` volume.
All boxes are visible in 3D; slice views show their intersections. D/F navigates
native sections in the overview too. A does not enable overview registration.

Raw and lc labels stream from their published mip pyramids. Overview `image_layers`
adds mito, syn and ves as grayscale context, hidden by default. Each must match
raw's physical extent; `visible_layers` can enable selected context layers.
The optional `segments` list selects the displayed lc IDs; by default it uses
the ROI seeds. Precomputed lc meshes are served from the selected dataset.
The cropped working collection clips the original 32 x 32 x 30 nm meshes to
the crop's physical bounds. Newly computed ROI meshes retain their near-isotropic
policy, and every view keeps `showSlices=false`.

Only requested metadata/chunks/mesh fragments are copied to
`OUTPUT/overview_cache/`, or `--overview-cache /path/on/local-ssd`. Gzip chunks are
served with their HTTP content encoding; uncompressed files support range reads.
Requests are confined to the configured volume directories. A successful cached
chunk can be revisited without rereading the source. Unseen data still requires
the source volume, and this is not a complete offline copy of the dataset.
Cache identities include the source path and metadata hash. Sources are treated
as immutable; use a fresh cache after replacing chunks without changing metadata.

## Session state and saved links

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
mesh caches are separate subdirectories. Both the main URL and each loaded
window's URL are saved as `viewer_url.txt`; window links include `#window=N` and
retain the toolbar. **Open in tab** also retains switching and status controls.
The internal iframe URL is separately saved as `neuroglancer_url.txt`.

The current local demonstration uses five soma-free process windows at varying
distances from their soma centers, replacing the earlier soma pilot. See the
[working-volume guide](README_working_volume.md) for selection and crop geometry,
and the [local GT guide](README_local_gt.md) for comparison limits. Raw, labels and
comparison results use local caches. The overview fetches working-volume chunks
on demand and never registers the whole volume.
