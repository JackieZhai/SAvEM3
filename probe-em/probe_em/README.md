# Probe-EM modules

| Module | Responsibility |
| --- | --- |
| `interactive.py` | Per-object points, asynchronous image/video inference, previews, and acceptance |
| `point_refinement.py` | SAM2 candidate selection and mask refinement using positive/negative points |
| `prompt_models.py` | SAM/HQ-SAM/SAEM²/SAM2 adapters and corrected-mask video initialization |
| `propagation.py` | Video input grids, frame encoding, and native-coordinate output mapping |
| `window_browser.py` | Persistent per-window sessions and selector service |
| `overview.py` | Multiscale image/label layers and process-window boxes |
| `precomputed_files.py` | Scoped volume serving and disk chunk cache |
| `viewer_navigation.py`, `viewer_navigation.js` | D/F navigation during mouse dragging |
| `viewer_status.py` | Live toolbar messages |
| `sam2_acceleration.py` | Feature cache and optional ONNX encoder |

## Tracing and geometry

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
| `z_align.py` | Affine fields and aligned cutouts; exact uint64 label warping |
| `demo_alignment.py` | Demo alignment, coordinate/cache validation, expanded grids, synchronized skeleton transforms |
| `registration.py` | Physical-nm limits, coarse-to-fine ECC, NCC/overlap gates, adjacent/skip-section displacement graph, unobserved-section flags |
| `sofima_registration.py` | Optional isolated SOFIMA flow/elastic-mesh refinement, physical limits and adjacent-pair quality guards |
| `alignment_field.py` | Portable affine/elastic field loading, invertible dense coordinates, raw/uint64 resampling and interactive transform artifacts |
| `meshes.py` | Default near-isotropic mesh sampling from actual resolution; unchanged native slices/origin, uint64 IDs, per-ID fallback, content-addressed geometry cache |

## Coordinates

CloudVolume arrays use XYZ order, skeleton vertices use absolute nanometers,
and model masks use YX/ZYX order. Preserve resolution and nonzero offsets when
converting between them. Raw and labels must share a physical grid.

Saved alignment fields map native voxel coordinates into the registered display.
Images, labels, points, and meshes use the same field. Interactive outputs are
saved in native coordinates. See [local realignment](../README_alignment.md).

## State and runtime

One shared model serializes image/video calls across windows. Each controller
owns its points, correction history, and results; each propagation starts fresh
temporal memory. The bounded feature cache stores image features only.

Tracing records running, complete, limited, and failed states. Resume requires
matching data and configuration. See [verification](../README_trace_quality.md)
and [configuration](../configs/README.md).
