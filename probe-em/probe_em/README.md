# Probe-EM core modules

`interactive.py` connects Neuroglancer point annotations to explicit, asynchronous
image/video prediction and local review artifacts. `prompt_models.py` provides lazy
SAM/HQ-SAM/SAM2/SAEM2 image adapters and bidirectional SAM2 video memory.
`propagation.py` retains encoded inputs and maps video prompts/masks between
registered and original grids, including SOFIMA. See the
[interactive guide](../README_interactive.md) for controls, coordinates and scope.

This implementation follows
[Probe-EM: Targeted Neuron Tracing via Training-Free Semantic Verification](https://arxiv.org/html/2607.04696v1).
Tracing starts at a seed segment. Models verify candidate connections without
modifying the original label volume.

| Module | Inputs, outputs, and responsibilities |
| --- | --- |
| `get_endpoints_vectors.py` | Physical-nm skeletons to endpoint coordinates/directions at the selected mip |
| `get_neighbors.py` | Candidates within a 500 nm endpoint sphere, including non-touching fragments; coordinates returned at mip 0 |
| `get_slices.py` | Same-section image pairs and cross-section pseudo-videos; read failures propagate to tracing status |
| `find_merge_candidates.py` | PEC: bidirectional prompts, five randomized trials by default, at least four passes |
| `find_merge_candidates_3d_region.py` | ASP: SAM 2 video propagation and candidate-coverage checks |
| `trace_inputs.py` | Anchored components, multislice gaps, dense frames and unsupported-case reporting |
| `trace_verification.py` | Auditable ASP coverage/precision/temporal gates and reverse-cycle overlap |
| `topology.py` | Pre-acceptance candidate skeleton QC; complex labels become review proposals before a budget stop |
| `runtime.py` | Lazy image/video predictor construction once per tracing task, reused across nodes |
| `device.py` | CUDA/MPS/CPU selection, two generations of SAM 2 Hydra resources, CUDA AMP |
| `z_align.py` | Optional historical zebrafish alignment extension; exact uint64 label warping |
| `demo_alignment.py` | Default-on demo alignment, coordinate/cache validation, expanded grids, synchronized skeleton transforms |
| `registration.py` | Physical-nm limits, coarse-to-fine ECC, NCC/overlap gates, adjacent/skip-section displacement graph, unobserved-section flags |
| `sofima_registration.py` | Optional isolated SOFIMA flow/elastic-mesh refinement, physical limits and adjacent-pair quality guards |
| `alignment_field.py` | Portable affine/elastic field loading, invertible dense coordinates, raw/uint64 resampling and interactive transform artifacts |
| `meshes.py` | Default near-isotropic mesh sampling from actual resolution; unchanged native slices/origin, uint64 IDs, per-ID fallback, content-addressed geometry cache |

## Coordinate and runtime contracts

Raw and segmentation CloudVolumes must share the XYZ grid, resolution, and physical ROI.
Skeleton vertices are absolute nm. Divide by the selected mip's resolution to obtain
voxel coordinates, then use the actual resolution ratio to convert back to mip 0.
Preserve nonzero volume offsets during neighborhood reads. Candidate search uses
physical distance, not an assumed voxel-axis ratio.

Empty image/video inputs do not load models. MPS/CPU use float32. CUDA AMP takes
a device type rather than an indexed string such as `cuda:0`. SAM 2 and SAM 2.1
configurations are not interchangeable by name alone; the resolver checks installed
resources and preserves newer `configs/sam2.1/...` paths.

Tracing's historical `align_z` estimator/cache convention targets zebrafish
`8×8×30 nm` data; disable that automatic path at other resolutions. Registration
is a local experiment extension, not a required default step of the published algorithms.

Demo alignment estimates from the loaded local ROI at its actual resolution.
It defaults on and can be disabled with `--no-align-z`. Both paths share per-section
transforms and exact-ID warping; rotations correctly account for nonzero cutout origins.
New demo translation uses `physical-pyramid-graph-v3.2` and does not identify v2
caches as new-algorithm results. The tracer's automatic estimator remains historical;
to trace with a new field, explicitly supply its NPZ and use a new result directory.
See the [alignment investigation](../README_alignment.md).
The [SOFIMA backend](../README_sofima.md) adds nonrigid residual grids through an
optional worker. The same saved field is used for images, labels, points, skeletons
and tracing cutouts; inference and saved masks retain original native coordinates.

Mesh resolution is independent of image/label slice resolution. Every new local
demo automatically selects near-isotropic factors from its own XYZ metadata;
8×8×30 nm becomes 32×32×30 nm for mesh generation only. Small labels missing after
sampling are meshed natively and flagged, while thin branches may still change.
Use `--no-mesh-isotropic` to opt out. See the
[mesh policy and cache contract](../scripts/README.md#near-isotropic-meshes-default-for-every-dataset).

## Tracing states and resume

Each seed directory contains `status.json`, `trace_state.json`, original ID sets,
and a merge tree. States are `running`, `complete`, `limited` (node budget), and
`failed`. A directory's existence does not imply completion. Exceptions record the
failing segment and cause the batch CLI to return a nonzero status.

`--resume` restores stack/visited/accepted/edges under a matching configuration
and retries failed nodes. Use a new output directory when changing data or checkpoints.
PEC samples from repeatable seed-plus-segment random sequences, so resume order
does not change a node's prompt samples.

`voxel_threshold` is a retained historical field: voxel counts from precomputed
skeletons are still incomplete, so this is not an effective voxel-volume filter.
Topology pruning uses endpoint count/density; guarded density pruning also
requires >15 endpoints. `complete` denotes queue exhaustion, not a verified
complete neuron. See [tracing quality](../README_trace_quality.md).

The paper uses NeuroSAM 2. Standard SAM 2 checkpoints support compatibility and
workflow demos, not reproduction of its quality claims. See the
[scripts guide](../scripts/README.md) and [configuration guide](../configs/README.md).
