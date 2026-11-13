# Nucleus segmentation on the overview

Option 3 uses **Fish-X LC-NE Test** raw images at **32 x 32 x 30 nm**. Place a
positive point at a nucleus center, propagate with SAM2 in both section
directions, and confirm the result. Every nucleus has its own colored mask;
all center annotations remain in one `nucleus_centers` layer.

## Start

Install [Probe-EM and SAM2](INSTALL.md) and restore the raw volume from the
[Hugging Face dataset](../docs/LC_NE_TEST.md#2-full-dataset-from-hugging-face).
`--data-root` contains `zfishlc_raw/info` and its `32_32_30` chunks. Coarser scales
are used for overview navigation when available. This mode does not require the
whole-volume mito, syn, ves, lcpre, or lcsyn datasets.

From the repository root:

```bash
python probe-em/scripts/demo_windows.py \
  --dataset nuclei --data-root /data/Fish-X-LC-NE-Test \
  --checkpoint /models/sam2_hiera_tiny.pt --device auto \
  --output-dir .cache/fish-x-review --open
```

The first view is the nucleus overview. The five process examples are also
available in the selector. The overview's lc and test-window layers start hidden.
The raw layer starts at 32 nm in XY; inference always reads that exact scale,
regardless of the display zoom. It uses the released dataset's alignment without
an additional local realignment. Layout, axes, and slice display follow the demo
defaults: `4panel-alt`, axes on, and `showSlices=false`.

## P → V → C

1. Focus the XY panel and hover over a nucleus center. Press **P** to place one
   positive center point. Choose a section near the middle of the nucleus.
2. Press **V**. SAM2 predicts the center section, then propagates toward increasing
   and decreasing Z independently. Follow progress in the toolbar; **X** stops.
3. Inspect the preview with **D/F**, including while dragging. Press **C**, or
   **Confirm nucleus (C)**, to save it and begin another nucleus in a new color.

**U** removes the active center and preview while retaining saved nuclei.
Move an active center in the annotation tab and rerun V to revise it. Confirm
each nucleus before adding another center. Negative points are not used.
Saved center descriptions identify their nucleus IDs; no per-nucleus point
layers are created. Saved centers identify accepted masks and should remain at
their recorded positions.

**Y / Accept/save** uses two separate **Enter** confirmations to save a binary
union. **Esc** cancels. Each nucleus mask and every center remain available after
the union is shown. Overlap is included once; gaps are not filled.

## Propagation and memory

Only a bounded XY crop is read around the center. Each direction uses short
video blocks, initialized from the last foreground section of the preceding
block. Video memory resets between blocks. Raw chunks are cached locally by
the viewer server; temporary video frames stay inside the review directory and
are removed after each block.

The center-containing mask component initializes the trace. Later sections keep
the component overlapping the previous mask after a two-pixel dilation. Two
consecutive empty sections stop that direction. Abrupt area growth, crop edges,
section limits, and dataset boundaries produce a **partial preview** message.
Inspect these cases and enlarge the crop or section limit when appropriate.
Unavailable raw chunks fail the run instead of being treated as empty tissue.

| Option | Default | Meaning |
| --- | --- | --- |
| `--nuclei-box-size` | `384` | XY crop width in 32 nm voxels, about 12.3 micrometers |
| `--nuclei-max-radius` | `256` | Maximum distance in sections on each side of the center |
| `--nuclei-block-slices` | `16` | New sections per video block, plus its conditioning section |
| `--nuclei-empty-sections` | `2` | Consecutive empty sections needed to stop |
| `--nuclei-auto-run` | off | Start V automatically when P places the center |

## Saved results

Results are stored in `OUTPUT/nuclei/interactive/`:

| Path | Contents |
| --- | --- |
| `nuclei.json` | Nucleus IDs, colors, source geometry, saved masks, and active center |
| `nucleus_centers.json` | One Neuroglancer annotation layer containing all centers |
| `nucleus_*/prediction.npz` | Cropped binary `mask_zyx`, absolute `voxel_offset_xyz`, `frame_z`, center, and resolution |
| `nucleus_*/metadata.json` | Model settings, block hashes, propagation parameters, and stopping reasons |
| `nucleus_*/accepted.json` | Confirmation record and nucleus ID |
| `union_*/precomputed/` | Sparse, gzip-compressed binary union at 32 x 32 x 30 nm |

Coordinates use the 32 x 32 x 30 nm grid. Multiply XYZ voxel coordinates by
`[32, 32, 30]` for absolute nanometers. NPZ files load with `allow_pickle=False`.
The sparse union uses zero for omitted chunks; programmatic CloudVolume reads
should set `fill_missing=True`.

Restarting with the same output directory restores saved nuclei and the active
center. P saves a new center immediately; annotation-tab edits are saved when V
starts. An unfinished preview is recomputed with V. The process-window reviews
remain in their separate `neuron_<ID>/interactive/` directories.
