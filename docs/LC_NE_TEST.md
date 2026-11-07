# LC-NE test

LC-NE test is the test dataset supplied with SAEM²-SAEvM³. It contains cropped,
SOFIMA-aligned zebrafish EM with normalized raw intensities, corresponding context
volumes, and selected neuron annotations. The release includes five process
windows for the [Neuroglancer demo](../probe-em/README_windows.md).

## Volumes

| Directory | Contents |
| --- | --- |
| `zfishlc_raw` | CLAHE-normalized grayscale EM |
| `zfishlc_lc` | Selected neuron instance labels, including IDs 30001-30005 |
| `zfishlc_mito` | Mito grayscale context |
| `zfishlc_syn` | Syn grayscale context |
| `zfishlc_ves` | Ves grayscale context |
| `zfishlc_lcpre` | Additional integer labels |
| `zfishlc_lcsyn` | Additional integer labels |

Mito, syn, and ves are grayscale images, not instance labels or calibrated
probabilities. The annotations cover selected structures; unlabeled voxels are
not exhaustive negative ground truth. The additional label sets retain their
source names.

## Geometry

All seven precomputed volumes share these XYZ scales:

| Mip | Resolution (nm) |
| --- | --- |
| 0 | 8 x 8 x 30 |
| 1 | 16 x 16 x 30 |
| 2 | 32 x 32 x 30 |
| 3 | 64 x 64 x 60 |
| 4 | 128 x 128 x 120 |
| 5 | 256 x 256 x 240 |

At mip 0, the voxel offset is `[192, 288, 0]` and size is
`[8640, 8160, 1583]`. Half-open physical bounds are `[1536, 2304, 0]` to
`[70656, 67584, 47490]` nm. Read each volume's `info` for coarse-grid bounds.

The whole-volume SOFIMA field is anchored at section 791 and applied to all
corresponding volumes. CLAHE is applied to raw only. The shared crop excludes
no-data margins. The dataset card contains the sampling and provenance details.

## Download and open

The release contains seven sets of independent ZIP64 parts, `restore.py`,
checksums, and `demo/lc_ne_process_windows.zip`. Extract all parts of each volume
into the same destination; do not concatenate ZIP parts. The demo archive contains
five cached ROIs, their local alignment fields, and relative-path manifests.

The Hugging Face repository ID is pending. Use `TUTORIAL.md` in the prepared
release for verification, extraction, installation, and demo controls. Dataset
files are not stored in this Git repository. The dataset license is Apache-2.0.

## Demo views

**LC-NE test overview** shows the complete release with five ROI boxes and
optional context layers. It uses the dataset geometry without additional local
realignment. Each process window uses its supplied local residual field by
default. **R** toggles that local realignment.

See [test regions](DEMO_REGIONS.md), [interactive controls](../probe-em/README_interactive.md),
and [window setup](../probe-em/README_windows.md).
