# Probe-EM tracing and demonstration scripts

For the optional SAEM2 checkpoint experiment, see the
[paired image and bounded 3D trial](../README_saem2_trial.md). It uses separate
outputs and does not replace Probe-EM's SAM2 video backend.

For tracing errors in the five cached examples, start with
[guarded tracing and the quality audit](../README_trace_quality.md).
`audit_trace_quality.py` prepares separate experiments, reruns actual verification,
summarizes decision changes, and serves baseline/guarded/uncertain layers with
the same registration field. Existing display-only demos remain available below.

## Local Neuroglancer demo

For point-based model interaction, use `demo.py --interactive` and see the
[interactive guide](../README_interactive.md): P/N adds positive/negative points,
I predicts the current section, B/E selects an inclusive z range, V runs SAM2
propagation, X stops, and Y accepts a reviewed image or volume. J/K changes
sections while retaining each section's points. This mode can start without
completed trace results and shares the registration and mesh settings below.

For larger local volumes, `evaluate_local_gt.py` caches bounded ROIs and evaluates
SAM2 propagation against neuron GT. See the [local GT guide](../README_local_gt.md)
for the workflow and score interpretation. `demo.py` also accepts repeatable
`--image-layer NAME=PATH` and `--label-layer NAME=PATH` arguments for same-grid
precomputed context, GT and prediction volumes. They start hidden and follow the
same A-toggle geometry and mesh policy as the main layers.

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
remain unchanged. Q restores the manual layer's automatic starting set. G saves
a new JSON version under `demo_reviews/`. Reference trace/comparison layers
disable spatial picking so double-clicking the data does not alter their ID sets;
sidebar selections are still editable. R restores these reference sets without
changing manual review. Hover highlighting is off in trace/review/comparison layers.

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

## Near-isotropic meshes (default for every dataset)

Mesh generation has its own resolution; raw imagery, 2D segmentation slices,
tracing, registration estimation, and skeleton coordinates retain their original
resolution. The default policy takes the coarsest native axis as a target and
rounds each target/native ratio to the nearest power of two in log scale, with
a minimum factor of 1. It never upsamples an axis and does not hard-code XY=4.

| Native XYZ resolution (nm) | Mesh downsampling XYZ | Actual mesh resolution (nm) |
| --- | --- | --- |
| 8, 8, 30 | 4, 4, 1 | 32, 32, 30 |
| 4, 4, 40 | 8, 8, 1 | 32, 32, 40 |
| 40, 8, 8 | 1, 4, 4 | 40, 32, 32 |
| 8, 8, 8 | 1, 1, 1 | 8, 8, 8 |

`--mesh-isotropic` is on by default in `demo.py`, `demo_regions.py serve`, and
the local mesh paths of `open_trace_viewer.py`. Use `--no-mesh-isotropic` for native
meshes or `--mesh-target-nm 64` for an explicit physical target. The latter is an
approximate target because factors remain powers of two; native coarser axes are
not upsampled. Registration and mesh policy are independent switches.

~~~bash
python scripts/demo_regions.py serve --open
python scripts/demo_regions.py serve --no-mesh-isotropic --open
python scripts/demo.py --local-dir volumes_local --results-dir trace_results \
  --seed 1486287284 --mesh-target-nm 64
~~~

`probe_em/meshes.py::MeshVolume` serves original label chunks but delegates mesh
requests to a separate nearest-neighbor-strided label volume. It does not linearly
interpolate IDs. Coarse mesh vertices are in physical units relative to the same
ROI origin. The original source transform retains the exact voxel offset, even
when registered offsets are not divisible by the mesh factors. Original and
registered meshes use separate fingerprints and switch together on A.

Entire labels lost in the coarse grid fall back to native meshing on request, with
an explicit per-ID record. Degenerate grids or empty coarse meshes also fall back.
This prevents a missing object from silently disappearing; it does not preserve
every thin branch or topology of a surviving label. Coarse surfaces approximate
the segmentation and are not a new, more accurate segmentation result.
Uniform ROIs without an internal label interface produce an explicitly empty
mesh record; their 2D slices remain available rather than aborting viewer startup.

The generic demo caches meshes in `OUTPUT/meshes/`; four-region demos use
`region_N/mesh_cache/`. Each fingerprint records segmentation SHA-256, shape,
dtype, native resolution/offset, mesh factors, algorithm version, Neuroglancer
version, and simplification options. Per-ID `.bin`/`.json` files contain geometry,
checksums, effective resolution, and fallback status. Traced IDs are precomputed
for both display spaces; other selected IDs are generated lazily and cached.
After editing an in-memory label array, call `invalidate()` before reusing its source.

Neuroglancer's `LocalVolume.downsampling` alone does not change its mesh input;
see the [upstream implementation](https://github.com/google/neuroglancer/blob/master/python/neuroglancer/local_volume.py).
Existing remote precomputed meshes are already generated on their server and are
not remeshed by this policy. The legacy remote verification tool continues to use
them; export/cache an ROI and open the generic local demo to use the new policy.

## Local registration options

| Option | Behavior |
| --- | --- |
| `--align-z` | Enabled by default; may be omitted |
| `--no-align-z` | Skip estimation and registered-volume construction entirely |
| `--align-field FILE.npz` | Reuse an explicit field covering every ROI section |
| `--align-backend ecc\|sofima` | ECC by default; optional SOFIMA flow/elastic mesh with translation initialization |
| `--sofima-python PATH`, `--sofima-config FILE.json` | Isolated SOFIMA interpreter and optional physical/quality settings |
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
See the [SOFIMA guide](../README_sofima.md) for its isolated installation, dense field
format, interactive integration and reproducible `compare_alignment.py` benchmark.
`sofima_worker.py` is the internal numeric-NPZ/JSON subprocess entry point.

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
`serve` selects its field by default and writes `viewer_optimized_mesh_isotropic/`.
Native-mesh comparisons use a separate `_mesh_native/` directory. Original
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
| `viewer/`, `viewer_optimized/`, `viewer*_mesh_isotropic/`, `viewer*_mesh_native/`, `reviews/` | Preserved historical views, new mesh-policy views, versioned review |
| `mesh_cache/` | Content-addressed physical-space meshes, per-ID checksums, and fallback records |
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

Current viewer URLs are in `viewers.json`, with policy-specific copies in
`viewers_mesh_isotropic.json` or `viewers_mesh_native.json`. The index from before
the mesh update is preserved in `viewers_before_mesh_update.json`; old services
are not stopped by launching the new ones.

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
