# Cropped working volumes and portable archives

The working-volume pipeline removes shared XY padding from local Neuroglancer
precomputed datasets while retaining original physical coordinates and all Z
sections. It reads bounded chunks, uses CloudVolume codecs and metadata, and
publishes a new directory. It does not use igneous or modify source datasets.

## Copy whole chunks when the source grid is preserved

Determine one physical bbox from downsampled raw, then convert it to each
corresponding dataset's voxel coordinates. Raw, mito, syn, ves and labels share
the physical crop; different resolutions and chunk sizes still need their own
chunk-coordinate calculations.

The exporter copies any output chunk that exactly matches one source chunk and
uses the same encoding. Its stored bytes, including an optional gzip wrapper,
are retained and checked after writing. No pixel decoding, JPEG processing or
label re-encoding is performed for these chunks. The `copied` count in each
scale report distinguishes this path from rewritten boundary chunks. A crop
containing only whole JPEG chunks does not need the JPEG helper.

This shortcut depends on the output grid. The standard
[Neuroglancer precomputed format](https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/volume.md#chunked-representation-of-volume-data)
anchors that grid at `voxel_offset`; chunk starts are
`voxel_offset + chunk_index * chunk_size`. An off-grid lower crop edge therefore
shifts every output chunk, including interior chunks. Simply retaining the old
interior filenames and shortening the first chunk would produce an invalid
precomputed volume. Such crops require retiling to retain the exact bbox.

The current LC-NE bbox begins at native X=96 while the image source chunks begin
at X=0, 512, 1024, and so on. It therefore uses the retiling path. Moving its left
edge inward to X=512 would align native image chunks but remove another 1.664
micrometers of valid tissue. Label chunks and coarser scales have different grids;
aligning one grid does not automatically align the others. Crop bounds are never
silently rounded to enable copying.

Encoding is also specified per scale. If a JPEG boundary cannot be cropped
losslessly with the helper, the exact-pixel export uses raw/gzip for that entire
scale; JPEG interior chunks cannot be mixed with raw boundary chunks under a
single scale's encoding. These constraints are separate from finding the bbox.

## Determine the shared crop

`scripts/crop_working_volume.py --scan-raw ...` inspects every original Z plane
at a selected XY mip. The mip must have the native Z resolution. Pixels at or
below the threshold are treated as padding only when connected to the bottom-right
padding seed. This avoids treating unrelated dark membranes touching another
image edge as missing tissue. A dataset without that padding geometry requires
explicit bounds or a different coverage method.

The intersection of valid pixels across all sections is eroded inward from the
padding by the requested guard. A histogram/stack algorithm finds the largest
all-valid rectangle on that scan grid. This is a grid-resolution result, not a
claim of the exact maximum at native resolution.

The LC-NE crop uses all 1,583 sections at **32 x 32 x 30 nm**, threshold 5 and
a one-pixel (32 nm) XY guard. Its half-open physical bounds are
`[384, 0, 0]` to `[71040, 68256, 47490]` nm. At native **4 x 4 x 30 nm**, the
offset is `[96, 0, 0]` and size is **17664 x 17064 x 1583**. The same physical
bounds apply to raw, mito, syn, ves, lc, lcpre and lcsyn. All original Z sections
are retained. Coarse mip extents round outward to cover partial edge cells.

## Export all corresponding datasets

Create a JSON manifest with original physical bounds and independent source
directories. Relative paths resolve against the manifest, not the current shell.

```json
{
  "bounds_nm": [[384, 0, 0], [71040, 68256, 47490]],
  "datasets": [
    {"name": "zfishlc_raw", "source": "/data/zfishlc_raw"},
    {"name": "zfishlc_mito", "source": "/data/zfishlc_mito", "info_path": "context-info.json"},
    {"name": "zfishlc_lc", "source": "/local-ssd/gt/zfishlc_lc"}
  ]
}
```

Add syn, ves, lcpre and lcsyn in the same form. The original context directories
lacked `info`; their grayscale JPEG stack geometry was checked against native raw
before deriving separate metadata. They are intensity images, not instance IDs
or calibrated probabilities. The original GT ZIP can be verified and extracted
before use. Alternatively add `"archive_path": "/local-ssd/zfishlc_lc.zip"` to
each GT manifest entry to read chunks, metadata and meshes directly from the
archive. The `source` basename selects the dataset directory within the ZIP.
This avoids creating hundreds of thousands of extracted input files; CRC checks
remain active. The crop plan records the source ZIP SHA256, and resume requires
the same archive. For very large source archives, `"archive_identity":
"central-directory"` records a SHA256 of the ZIP directory plus archive size and
modification time, avoiding an extra full read before cropping. Each member still
undergoes ZIP CRC validation. This mode assumes immutable source data; it is not
a full-archive content hash. Use `--verify-resume` to compare resumed output pixels
against source data again.

```bash
python probe-em/scripts/crop_working_volume.py \
  --manifest /local-ssd/crop.json --output /data/zfishlc_working_cropped \
  --scratch-dir /local-ssd/scratch --workers 8
```

The output must be new. `--resume` requires identical source metadata and crop
bounds. It trusts final-named chunks previously published by this exporter;
`--verify-resume` additionally decodes and compares every resumed chunk against
the source. Missing or empty source chunks fail explicitly. `info` is published
only after every voxel chunk is complete. Per-dataset `crop_status.json` and the
collection status distinguish completion from interruption.

Integer labels remain exact, including uint64 IDs. Nonuniform output chunks are
decoded and compared pixel by pixel when rewritten; copied chunks retain and
verify their original bytes. All-background compressed segmentation
chunks are certified from the compressed label palettes and a decoded, verified
zero template. No label interpolation or averaging occurs during cropping.

JPEG crops aligned to complete 8 x 8 XY DCT blocks can use the optional helper:

```bash
cc -O2 probe-em/probe_em/jpeg_crop.c -o /local-ssd/bin/jpeg_crop -ljpeg
python probe-em/scripts/crop_working_volume.py \
  --manifest /local-ssd/crop.json --output /data/zfishlc_working_cropped \
  --jpeg-crop /local-ssd/bin/jpeg_crop --scratch-dir /local-ssd/scratch
```

Install libjpeg headers/libraries and add platform-specific include/library paths
if needed. The helper copies quantized DCT coefficients and tables without
requantizing; the Python caller still checks decoded pixel equality. Requests
and results travel through bounded memory pipes. It creates **no temporary JPEG
files**. `--scratch-dir` sets the helper's temporary-directory environment. Final
chunks use atomic temporary writes in the destination dataset before rename.

If the crop cuts a JPEG block or no helper is provided, that scale uses raw
uint8 plus gzip. This preserves decoded source pixels but can need more space.
The exporter supports one-channel, unsharded volumes with one chunk size per
scale, using raw, PNG, JPEG or compressed segmentation. JPEG coefficient copying
also requires unchanged Z chunking and compatible grayscale quantization tables.

Legacy mesh triangles are clipped to the physical crop; fully interior geometry
and winding are retained. Crossing triangles are clipped even if none of their
vertices lie inside. Cut surfaces remain open, and stale spatial indices are
removed. This preserves the existing LC meshes originally computed at
32 x 32 x 30 nm. It does not regenerate a mesh from 4 nm labels.

## Add display scales for native-only context images

`scripts/build_context_pyramid.py` creates a separate display pyramid for a
grayscale JPEG context dataset:

```bash
python probe-em/scripts/build_context_pyramid.py \
  --source /data/zfishlc_mito --info /local-ssd/context-info.json \
  --crop-manifest /local-ssd/crop.json --output /local-ssd/mito-pyramid
```

The first scale uses libjpeg's 1/8 XY DCT reduction and preserves every Z plane
(32 x 32 x 30 nm for LC-NE). Two further scales use 2 x 2 x 2 arithmetic means,
with partial edge cells averaged over available voxels only. DCT reduction is
not an exact native-pixel area average. These scales are for display; the native
working copy remains pixel-exact. Once both exports are complete, copy these
scale directories into the corresponding working dataset and append their scale
metadata to its `info`. Keep the pyramid provenance with the working dataset.

## Select process windows and use the overview

```bash
python probe-em/scripts/select_process_windows.py \
  --raw-path /data/zfishlc_working_cropped/zfishlc_raw \
  --gt-path /data/zfishlc_working_cropped/zfishlc_lc \
  --crop-manifest /local-ssd/crop.json --output /local-ssd/process-windows
```

Selection estimates thick soma cores from the coarse GT, excludes every soma
from the entire ROI with a locator-grid guard, and selects thinner processes at
different Euclidean distances from their soma centers. Fine-grid checks reject
large central profiles and require target occupancy in at least 80% of sections.
The geometry does not establish axon/dendrite identity or geodesic distance.

The new five windows are 512 x 512 x 63 at 8 x 8 x 30 nm, with soma-center
distances approximately 11.0, 15.6, 23.2, 29.9 and 38.0 micrometers. They replace
the earlier soma windows in the browser. Their source coordinates and selected
interior points are recorded in each `selection.json`.

Use **LC-NE test overview** as the browser's overview name, with cropped raw/lc
and the three cropped context images. Only this overview includes mito, syn and
ves; ROI manifests reject those layer names. It stays unregistered and shows all
five process boxes. The process windows retain default local registration, the
**A** toggle, **D/F** navigation and `showSlices=false`. See the
[window browser guide](README_windows.md).

## Prepare ZIP parts for Hugging Face

Run the standard-library packager after a dataset and its optional display scales
and meshes are complete:

```bash
python probe-em/probe_em/archive.py pack \
  --source /data/zfishlc_working_cropped/zfishlc_raw \
  --output /data/zfishlc_huggingface/zfishlc_raw --part-gib 5
```

Repeat for all seven datasets. JPEG and gzip data use `ZIP_STORED` to avoid
recompressing already compressed payloads. ZIP64 supports large parts and large
file counts. The cap includes conservative archive-header overhead and is checked
against the actual result. A source file larger than the cap fails explicitly.

Each part is an independent ZIP with disjoint paths under `zfishlc_NAME/`.
Extract all parts into one parent directory; do not concatenate them. Small
datasets use one ZIP. The 5 GiB default is an operational choice for convenient
downloads, not a Hugging Face hard limit. The package excludes local crop plans,
logs, scratch files and source paths. A portable `provenance.json`, if present,
is included. Packaging rejects incomplete crops and symbolic links.

Every member is reread and checked against its source SHA256 and ZIP CRC before
the part is published. Per-part manifests store archive SHA256, counts, sizes
and compressed inventories of member hashes. `--resume` requires an unchanged
source inventory and verifies completed archive hashes before reusing them.

```bash
python probe-em/probe_em/archive.py restore \
  --package /data/zfishlc_huggingface/zfishlc_raw --verify-only
python probe-em/probe_em/archive.py restore \
  --package /data/zfishlc_huggingface/zfishlc_raw --destination /restored
```

Restore verifies all parts before extraction, rejects unsafe/duplicate paths,
checks extracted hashes and refuses to overwrite different existing files.
Matching existing files permit a restart. Legacy Neuroglancer mesh filenames
contain colons; on Windows extract those datasets into a POSIX filesystem such
as WSL. Voxel archives themselves use ordinary portable coordinate filenames.

For the LC-NE publication bundle, include an Apache-2.0 `LICENSE`, a dataset card,
the standalone restore script, collection manifest and `SHA256SUMS`. Once the
owner creates the Hugging Face dataset repository, upload with the current
[Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/upload):

```bash
hf auth login
hf upload OWNER/REPOSITORY /data/zfishlc_huggingface . --repo-type dataset
```

Keep Hugging Face caches on an appropriate local disk. The ZIP format preserves
Neuroglancer layout for download and reconstruction; it does not provide direct
precomputed HTTP access or an automatic Hugging Face image-table viewer.
