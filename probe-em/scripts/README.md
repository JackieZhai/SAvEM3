# Probe-EM tracing and demonstration scripts

## Local Neuroglancer demo

Activate your Probe-EM/SAM 2 Python environment and run commands in `probe-em/`
unless noted otherwise. Cached volumes, checkpoints, and trace results are local
artifacts; they are not included in the GitHub clone. Use the export workflow below
for your own data, or the four-region workflow to download the online zebrafish ROIs.

If the original zebrafish cache and trace results are already present:

~~~bash
python scripts/demo.py \
  --local-dir volumes_local --results-dir trace_results --seed 1486287284 \
  --align-field trace_results_align_test/align_z_field_seed1486287284.npz \
  --output-dir demo_session_aligned
~~~

The script prints a local Neuroglancer URL and writes `viewer_url.txt` in the
output directory. Open it in a browser and keep Python running. The default port
is dynamically assigned on `127.0.0.1`; `--port` selects a fixed port.
The browser must be able to load Neuroglancer frontend assets; Python serves
the local volume data. `viewer_state.json` is a diagnostic snapshot, not a
standalone public viewer link.

Viewers default to `"showSlices": false`, hiding slice planes in the 3D panel
while retaining the three 2D panels and 3D segments/skeletons. The four-region demo
and both historical visualization entry points share this default. Enable slice
planes in Neuroglancer settings if needed; A toggles do not reset that choice.

| Layer | Purpose |
| --- | --- |
| `raw` | Raw EM imagery |
| `input_segments` | Original/SAvEM³ instances; initially hidden to emphasize the trace |
| `probe_trace` | Automatically traced original instance IDs, without relabeling |
| `manual_correction` | Editable instance set, initialized from the trace |
| `skeletons` | Available skeletons in physical coordinates, including 3D paths |

Enable and select `manual_correction` to edit its segment set with Neuroglancer's
selection tools. A synchronously switches raw, all segmentation layers, skeletons,
and the viewing position between original and registered coordinates; review IDs
remain unchanged. Q restores the automatic trace set. G saves a new JSON version
under `demo_reviews/`.

Review files record the seed, full uint64 IDs as strings, sources, resolution,
offset, timestamp, and current registration state. Saving changes neither input
segmentation nor earlier review versions.

Check data and layer state without keeping a server alive:

~~~bash
python scripts/demo.py --local-dir volumes_local --results-dir trace_results \
  --seed 1486287284 --check --output-dir demo_session_check
~~~

`--check` briefly initializes Neuroglancer and validates data/state, not browser
rendering. The default ROI limit is 128M voxels; larger inputs require an explicit
`--max-voxels` override.

## Local registration options

| Option | Behavior |
| --- | --- |
| `--align-z` | Enabled by default; may be omitted |
| `--no-align-z` | Skip estimation and registered-volume construction entirely |
| `--align-field FILE.npz` | Reuse an explicit field covering every ROI section |
| `--align-motion translation` | Default automatic motion model; euclidean/affine are also available experimentally |
| `--align-reference-z Z` | Absolute voxel reference z for estimation; defaults to the middle ROI section |
| `--align-max-shift-nm 800` | Automatic physical displacement limit in nm, independent of mip |
| `--align-coarse-resolution-nm 32` | Estimate at approximately 32 nm first, then refine through the pyramid |
| `--align-min-ncc 0.25` | Minimum coarse-scale NCC; at least 70% overlap is also required |

Without `--align-field`, translation estimates both adjacent and skip-section
constraints with physical-scale coarse-to-fine ECC, then combines them in a robust
graph anchored to the reference section. It uses the volume's real resolution,
not a fixed zebrafish pixel size. Euclidean/affine use an experimental pyramid-chain path.

The three automatic quality/scale parameters do not alter explicitly supplied
historical fields. Fields and quality metadata are cached under `OUTPUT/alignment/`.
Cache keys include raw content, ROI coordinates, resolution, algorithm version,
and all registration parameters. Historical NPZ files lacking a resolution
sidecar use the zebrafish `8×8×30 nm` convention.

~~~bash
# Default-on local estimation:
python scripts/demo.py --local-dir volumes_local --results-dir trace_results --seed 1486287284
# No registration computation:
python scripts/demo.py --local-dir volumes_local --results-dir trace_results --seed 1486287284 --no-align-z
~~~

Raw uses linear interpolation. Instances use nearest-neighbor interpolation with
compact indices to retain exact uint64 IDs. Skeletons use the same section transforms;
cross-section edges are densified to avoid moving only their endpoints. The output
grid expands to contain the transformed ROI. Added zeros are padding, not newly
acquired imagery.

`report.json` records input/output coordinates, reference section, field source,
and quality failures. `rejected_pair_slices` lists adjacent edges failing quality
gates; skip-section constraints may still locate those sections. Sections without
reliable observations are separately listed in
`quality_summary.interpolated_or_extrapolated_slices`. Their displacement is inferred
from nearby observations, not measured, and no missing image content is synthesized.
Estimation fails explicitly if no reliable component exists. Historical v2 fields
retain their original fallback interpretation: reuse of a neighboring transform.
See the [alignment investigation](../README_alignment.md).

This is display registration. It does not rerun PEC/ASP, alter raw data, or change
existing tracing decisions. The tracer's `align_z` configuration is separate and
retains its historical default.

## Four regions from the same online database

`demo_regions.py` extracts four disjoint ROIs from exactly these sources:

- Raw: `precomputed://https://ng.zebrafish.digital-brain.cn/srv/raw/`
- Existing instances: `precomputed://https://ng.zebrafish.digital-brain.cn/srv/merge_neuro_label/`

| Region | Inclusive mip-0 XYZ start | Exclusive mip-0 XYZ end |
| --- | --- | --- |
| region_1 | 32500, 19500, 13700 | 33000, 20000, 13800 |
| region_2 | 34000, 19500, 14100 | 34500, 20000, 14200 |
| region_3 | 32500, 21500, 14200 | 33000, 22000, 14300 |
| region_4 | 34000, 21500, 13600 | 34500, 22000, 13700 |

Each region is `500×500×100` voxels at `8×8×30 nm`, or `4×4×3 µm`. They do not
overlap the original seed `1486287284` demo ROI. Each matches one source chunk
per volume; the workflow does not download the entire database.

~~~bash
# Only prepare reads remote volume data; complete caches are hash-checked and reused.
python scripts/demo_regions.py prepare
# Run real PEC/ASP with local raw/seg/skeletons under a one-node budget.
python scripts/demo_regions.py trace --checkpoint /absolute/path/to/sam2_hiera_tiny.pt
# Check cache, trace state, retained IDs, and adjacent-section correlation.
python scripts/demo_regions.py audit
# Keep one service alive and open four independent viewer tabs.
python scripts/demo_regions.py serve --open
# Disable display registration without changing trace results.
python scripts/demo_regions.py serve --no-align-z --open
# Select the field stored by the original prepare step.
python scripts/demo_regions.py serve --legacy-alignment --open
~~~

Use `--region region_1` for a single region and `--root` for a different cache root.
After closing a demo, rerun only `serve --open`; new URLs are generated without
downloading or retracing.

If `alignment_optimized/report.json` is validated and matches the raw checksum,
`serve` selects its field by default and writes `viewer_optimized/`. Original
`alignment/`, `viewer/`, and `trace_results/` remain intact. The comparison holds
trace IDs fixed and records when display and trace fields differ. All services bind
to loopback and require their Python process to remain alive.

The default region directory is `demo_session_regions/region_N/`:

| Path | Contents |
| --- | --- |
| `cache/raw/`, `cache/seg/` | Local precomputed data; lossless after raw decoding, unchanged label values |
| `cache/seg/skeletons/`, `cache/traced_skeletons.json` | Kimimaro ROI skeletons with absolute-nm vertices |
| `region.json` | Online sources/info snapshots, coordinates, download time, checksums, seed-selection evidence |
| `alignment/` | Prepared field, cache metadata, and rejected adjacent pairs |
| `trace_config.json`, `trace_results/` | Configuration, status, accepted IDs, pending queue, PEC/ASP visual evidence |
| `validation.json` | Cache/ID audit and intensity NCC sanity check, not annotated accuracy |
| `viewer/`, `viewer_optimized/`, `reviews/` | Original/optimized viewer state, reports, versioned review |
| `alignment_investigation/`, `alignment_optimized/` | Candidate fields, validated new fields, common-support NCC, real XZ comparison figures |

Cache export is checked by exact voxel readback. Missing remote chunks raise
errors rather than being filled with zeros. The online raw encoding is JPEG:
lossless local caching retains server-decoded voxels but cannot restore information
lost in source JPEG compression. uint32 source label values are exactly preserved
in uint64 storage. These labels are existing site segmentation; SAvEM³ is not
retrained or reinferred on these four ROIs.

Seeds are chosen before inference by skeleton geometry: at least 1,000 voxels,
2–8 endpoints, and an endpoint at least 500 nm inside the ROI; uncut central
fragments are preferred. Selection does not depend on successful model merges.
Skeletons remain ROI-limited, not whole-neuron skeletons.

The original four-region experiment used tracing `align_z=true` with v2 fields.
The subsequent v3 comparison kept those decisions fixed. Fresh preparation now
uses v3.2; it does not reproduce the historical v2 experiment. To retrace with a
different field, create a new result directory and explicitly set `align_z_field`.
Do not resume an old experiment with changed registration.

Default `--node-limit 1` expands the seed once but runs actual PEC/ASP on its
candidates. Pending work yields `limited`, not failure or whole-neuron completion.
`trace --resume --checkpoint ...` expands another budget under a compatible
configuration; `--node-limit N` sets the per-invocation budget. Do not rerun an
incomplete experiment without `--resume`. Serving rejects failed tracing results.

The generic Tiny checkpoint demonstrates the data/software path, not paper
NeuroSAM 2 accuracy. Registration failures remain explicit in reports and viewer
status; an NCC increase is not proof of anatomical alignment. Press A to inspect.

## New SAvEM³ segmentation results

From the repository root, in the Probe-EM environment:

~~~bash
python probe-em/scripts/export_savem3.py --raw roi.h5 --seg seg.h5 \
  --resolution 8 8 30 --offset 0 0 0 --output outputs/probe_roi \
  --skeletonize --seed 1 --checkpoint /absolute/path/to/neurosam2.pt \
  --model-config configs/sam2.1/sam2.1_hiera_l.yaml
python probe-em/scripts/run_probe_em.py --config outputs/probe_roi/trace_config.json
python probe-em/scripts/demo.py --local-dir outputs/probe_roi \
  --results-dir outputs/probe_roi/trace_results --seed 1
~~~

`export_savem3.py` accepts ZYX H5/TIFF/NPY and exports XYZ precomputed with the
actual offset. `--skeletonize` generates Kimimaro skeletons and segmentation
skeleton metadata. `kimimaro==5.8.4` avoids the observed 5.8.0 vertex-dtype conflict.

For three-target teacher H5, pass `--raw outputs/teacher.h5 --raw-key raw`.
Resolution/offset must describe student raw, not the 1024² encoder input.
New volumes contain only mip 0, so the generated configuration uses `target_mip=0`.
Choose an existing seed from `manifest.json`'s `skeleton_ids`. Truncated ROI
endpoints are for local demonstration, not evidence of complete neuron ends.

## Script responsibilities

`run_probe_em.py` runs HSS → PEC/ASP → tracing. `demo.py` reviews existing results
and saves versioned corrections. `verify_traces_neuroglancer.py` retains the
historical cursor-based trace lookup and GT-saving workflow.

`open_trace_viewer.py`, `backup_local_volumes.py`, and `make_verification_figs.py`
are historical zebrafish/alignment tools with dataset-specific conventions; prefer
`demo.py` for generic data. `investigate_alignment.py` and
`optimize_demo_alignment.py` compare preserved original caches with candidate fields.
`evaluate_tracing.py` compares trace and manual ID sets; it is not an ERL substitute.
