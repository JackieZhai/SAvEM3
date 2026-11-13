# Fish-X LC-NE Test

Fish-X LC-NE Test is the test dataset supplied with SAEM²-SAEvM³. It contains cropped,
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

## Choose a dataset option

| Option | Included views | Data setup |
| --- | --- | --- |
| `examples` (default) | lc overview with five boxes, then five process examples | Bundled in this repository, about 114 MiB |
| `full` | Full overview with bounding boxes, then the same five examples | Download and extract the full Hugging Face release first, about 201 GB |
| `nuclei` | 32 x 32 x 30 nm overview for positive-center SAM2 segmentation, plus the five process examples | Restore at least the raw volume's 32 nm scale and its metadata |

All options use the same images, reference labels, and supplied local realignment fields
for the five boxes. Fresh review directories also load eight saved tracing objects
and their 55 annotation points from the bundled starter session. Existing reviews
are preserved. Add `--no-starter-session` with a new output directory to start blank.
SAM2 weights and [viewer dependencies](../probe-em/INSTALL.md)
must be installed separately. The examples need no Hugging Face account.

### 1. Bundled lc overview and five examples

From the repository root:

```bash
python probe-em/scripts/demo_windows.py --dataset examples --checkpoint /models/sam2_hiera_tiny.pt --output-dir .cache/fish-x-review --open
```

The launcher verifies and extracts the archives in
[`probe-em/examples/fish_x_lc_ne/`](../probe-em/examples/fish_x_lc_ne/README.md)
to `OUTPUT/dataset/`. It starts at **Fish-X LC-NE Test overview** with the full
`lc` segmentation, neuron meshes, and `test_windows` bounding-box annotations.
The selector also contains the five locally realigned process examples.
Omitting `--dataset` selects this option as well.

The five raw/label/field archives total about 78 MiB. `lc.zip` adds the complete
six-resolution label volume and supplied meshes in about 36 MiB, while
`test_windows.json` stores the five annotation boxes in under 2 KiB. Full-volume
raw and context images are provided by option 2. Raw images remain available
inside every bundled process example.

### 2. Full dataset from Hugging Face

The public release is hosted at
[JackieZhai/Fish-X_LC-NE_Test](https://huggingface.co/datasets/JackieZhai/Fish-X_LC-NE_Test),
with manually approved access requests. Request access on the dataset page, then
authenticate with the approved account. Download the release matching the bundled
boxes using the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli#hf-download):

```bash
python -m pip install --upgrade huggingface_hub
hf auth login
hf download JackieZhai/Fish-X_LC-NE_Test --repo-type dataset --revision b209e99872579da55d1855422a13bb1affcd2872 --local-dir /data/fish-x-package
```

Allow approximately 201 GB for the download and another 201 GB for extracted
volumes, plus review outputs and caches. Use paths appropriate to your machine.
On Windows, restore the full volumes into a WSL Linux filesystem and run the
full viewer there: their legacy mesh filenames contain colons. The bundled
option, including the lc overview, works on native Windows, macOS, and Linux.

Restore all seven volumes with the release's Python utility, which verifies
archive/member checksums and resumes matching existing files:

```bash
python /data/fish-x-package/restore.py restore --package /data/fish-x-package/zfishlc_raw --destination /data/Fish-X-LC-NE-Test
python /data/fish-x-package/restore.py restore --package /data/fish-x-package/zfishlc_lc --destination /data/Fish-X-LC-NE-Test
python /data/fish-x-package/restore.py restore --package /data/fish-x-package/zfishlc_mito --destination /data/Fish-X-LC-NE-Test
python /data/fish-x-package/restore.py restore --package /data/fish-x-package/zfishlc_syn --destination /data/Fish-X-LC-NE-Test
python /data/fish-x-package/restore.py restore --package /data/fish-x-package/zfishlc_ves --destination /data/Fish-X-LC-NE-Test
python /data/fish-x-package/restore.py restore --package /data/fish-x-package/zfishlc_lcpre --destination /data/Fish-X-LC-NE-Test
python /data/fish-x-package/restore.py restore --package /data/fish-x-package/zfishlc_lcsyn --destination /data/Fish-X-LC-NE-Test
```

Each ZIP64 part is an independent archive; do not concatenate parts. The
destination now contains all seven `zfishlc_*` directories, each with `info`
and six scale directories. No separate extraction of the Hugging Face demo ZIP
is needed: the launcher uses the same five boxes already bundled in this repo.

Start from the repository root:

```bash
python probe-em/scripts/demo_windows.py --dataset full --data-root /data/Fish-X-LC-NE-Test --checkpoint /models/sam2_hiera_tiny.pt --output-dir .cache/fish-x-review --open
```

The initial view is **Fish-X LC-NE Test overview**, with all five named boxes.
The window selector also offers each locally realigned example. The launcher
checks the seven volumes and their geometry before starting the viewer. It
does not initiate a full download. To check setup without loading a model,
replace `--checkpoint ... --open` with `--prepare-only`.

Saved process objects use `OUTPUT/neuron_<ID>/interactive/` in all options, so they stay
with their neuron when switching modes. The example archives and full dataset
are licensed under Apache-2.0.

### 3. Nucleus segmentation on the 32 nm overview

```bash
python probe-em/scripts/demo_windows.py --dataset nuclei --data-root /data/Fish-X-LC-NE-Test --checkpoint /models/sam2_hiera_tiny.pt --output-dir .cache/fish-x-review --open
```

Place a positive nucleus-center point with **P**, run bidirectional SAM2
propagation with **V**, and confirm with **C**. All saved centers share one
`nucleus_centers` layer; masks keep separate colors. This mode uses bounded
32 x 32 x 30 nm blocks without additional local realignment. Whole-volume raw
must be restored first; the other full context datasets are optional.
See [nucleus controls, limits, and saved files](../probe-em/README_nuclei.md).

## Demo views

**Fish-X LC-NE Test overview** shows the complete lc labels and meshes with five
ROI boxes in the examples and full options. The full option adds whole-volume raw and optional
context layers. The overview uses the dataset geometry without additional local
realignment. Each process window uses its supplied local residual field by
default. **R** toggles that local realignment.

See [test regions](DEMO_REGIONS.md), [interactive controls](../probe-em/README_interactive.md),
and [window setup](../probe-em/README_windows.md).
