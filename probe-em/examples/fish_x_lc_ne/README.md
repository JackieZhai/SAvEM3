# Fish-X LC-NE Test examples

The full lc segmentation and five neuronal process boxes are bundled here for
the default Neuroglancer demo.
They cover neurons 30001-30005, each with 512 x 512 x 63 voxels at 8 x 8 x 30 nm.
Each archive contains raw images, instance labels, a binary reference mask, and
the supplied local realignment field. These five archives total about 78 MiB.

`lc.zip` contains the complete lc labels at all six resolutions and their
supplied neuron meshes, compressed to about 36 MiB. `test_windows.json` is a
Neuroglancer annotation layer containing the five bounding boxes in native
dataset coordinates. The complete bundle is about 114 MiB.

From the repository root, after [installation](../../INSTALL.md):

```bash
python probe-em/scripts/demo_windows.py --dataset examples --checkpoint /models/sam2_hiera_tiny.pt --output-dir .cache/fish-x-review --open
```

The launcher verifies the archive hashes and extracts them under
`OUTPUT/dataset/`. Subsequent launches reuse matching files and repair missing
or damaged cached files. No dataset download or Hugging Face login is needed
for this option. It opens an overview with the `lc` and `test_windows` layers,
followed by the five locally realigned examples. SAM2 weights must be installed
separately.

`starter_session.zip` supplies eight saved tracing objects and 55 positive/negative
annotation points across the five windows (about 191 KiB). A fresh review directory
loads these automatically, with their object colors, visibility, camera views,
and propagation ranges. Each object retains its native-resolution binary mask and
point layers. These are editable demo results, separate from the reference labels.

| Neuron | Saved objects | Positive points | Negative points |
| --- | --- | --- | --- |
| 30001 | 1 | 7 | 7 |
| 30002 | 1 | 9 | 16 |
| 30003 | 2 | 2 | 0 |
| 30004 | 2 | 2 | 0 |
| 30005 | 2 | 7 | 5 |

Existing reviews always take priority and are never replaced by the starter session.
Use `--no-starter-session` with a new `--output-dir` to start without saved objects.
The full dataset option uses the same five starter reviews. See the
[interactive controls](../../README_interactive.md) to trace another object or
accept/save a combined result.

Voxel chunks, mesh payloads, and the five process examples are byte-identical
to the corresponding members of the published
[Fish-X LC-NE Test release](https://huggingface.co/datasets/JackieZhai/Fish-X_LC-NE_Test).
`manifest.json` records the release revision, source archive hash, and bundled
archive hashes. `windows.json` defines the five boxes and their initial views.
To support native Windows filesystems, lc mesh filenames use percent encoding
on disk; `info` records `probe_em_filename_encoding: "percent"`. The local server
maps standard Neuroglancer mesh URLs to these files, including its chunk cache.
The whole-volume raw and context images are downloaded separately; see the
[dataset options](../../../docs/LC_NE_TEST.md).

The examples are distributed under [Apache-2.0](../../../LICENSE). Reference
labels cover selected neurons and are not exhaustive annotations of the tissue.
