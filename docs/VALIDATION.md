# Validation record

Last checked: 2026-09-09. Existing zebrafish caches, registration fields, and tracing
experiments are retained locally. Publishing the code does not rerun or overwrite them.
Generated caches, screenshots, machine-specific configurations, and inventory JSON
are intentionally excluded from Git.

## Regression tests

The following three environments were checked before publication. Counts include
shared tests run in different environments; do not add them as unique test cases.
Real-weight tests used checkpoints already on disk and did not download models.

| Environment and suite | Result | Coverage |
| --- | --- | --- |
| Training: distillation, reproduction, real HQ integration | 12 passed, 7 skipped | Student forward/backward, 1→2 step resume, sliding-window edges, membrane-decoder interface, real HQ mask prompts, batched three-target H5 export |
| Postprocessing: reproduction | 9 passed, 2 skipped | Native ELF/nifty multicut, WaterZ, boundary/affinity direction, section IDs, large uint64 edge mappings |
| Probe-EM: tracing, registration, demos, real SAM 2 integration | 32 passed, 2 skipped | PEC voting/stopping, HSS gaps and offsets, ASP uint64 IDs, failure recovery, skeleton export, registration geometry/quality, image prediction and two-frame video memory |

Skips reflect environment roles: training lacks ELF/SAM 2; postprocessing lacks
OpenCV; the Probe-EM invocation does not configure HQ integration weights.
Third-party deprecation/registry warnings remain and are not claimed resolved.

Real-weight tests use HQ-SAM ViT-B and standard SAM 2 Hiera Tiny on CPU.
The teacher-export membrane head is randomly initialized for the integration test:
it validates tensors and files, not meaningful SAEM² predictions. Short student
training uses synthetic teacher targets and does not establish segmentation accuracy.

From the repository root, in the appropriate environments:

~~~bash
# Training environment:
SAVEM3_HQ_CHECKPOINT=/absolute/path/to/sam_hq_vit_b.pth \
  python -m pytest tests/test_distillation.py tests/test_reproduction.py \
  tests/test_model_integration.py -q

# Postprocessing environment:
./repro/env/run_postprocess.sh -m pytest tests/test_reproduction.py -q

# Probe-EM environment:
PROBE_EM_SAM2_CHECKPOINT=/absolute/path/to/sam2_hiera_tiny.pt \
  python -m pytest tests/test_registration.py tests/test_demo_alignment.py \
  tests/test_demo_regions.py tests/test_probe_em.py tests/test_model_integration.py -q
~~~

CloudVolume uses `~/.cloudfiles/locks/` while writing skeleton metadata. Allow the
library to use its lock directory in restricted environments; do not treat a
permission failure as a missing skeleton.

## Neuroglancer and registration

The four non-overlapping online zebrafish ROIs were downloaded, cached locally,
traced with actual models under a one-node budget, and checked in Chrome.
See [DEMO_REGIONS.md](DEMO_REGIONS.md) for the original experiment and
[the alignment investigation](../probe-em/README_alignment.md) for the optimized comparison.

The 2026-09-09 optimization uses physical-scale coarse-to-fine estimation and a
robust adjacent/skip-section graph. On identical valid pixels, it improves the
four-region mean adjacent-section NCC from approximately 0.285 to 0.388.
Rejected adjacent edges fall from 150/396 to 4/396. Region 2 at absolute z=14128
still lacks a reliable observation and is explicitly marked as displacement
interpolation/extrapolation, not a successfully measured alignment.

The 29-test tracing/registration/demo subset passes; the three opt-in SAM 2 tests
bring the Probe-EM environment total to 32. Geometry tests cover XY direction,
nonzero-origin rotation, expanded bounds, uint64 IDs, skeleton transforms, caching,
default-on alignment, toggling, and preservation of review sets. Quality tests
cover physical shifts at different resolutions, bad coarse initialization,
missing-section bridging/flags, unrelated-image rejection, robust graph outliers,
invalid parameters, and validated optimized-field selection.

All four optimized viewers passed HTTP 200, actual canvas rendering, A off/on
comparison, and persistence of `showSlices=false`, with no browser runtime
exceptions or failed requests during the checks. The 3D slice planes are hidden;
the three 2D panels remain. These are local browser checks, not public hosted demos.

### Original single-ROI reference

This earlier demo uses an existing real zebrafish cache and prior tracing results,
not a new SAvEM³ training output.

| Item | Value |
| --- | --- |
| Seed | `1486287284` |
| Trace IDs | `1486284686`, `1486287284`, `1504550065` |
| Original XYZ shape | `261×357×191` |
| XYZ resolution | `8×8×30 nm` |
| Original XYZ voxel offset | `33386,20334,13868` |
| Layers | raw, input_segments, probe_trace, manual_correction, skeletons |

The earlier unregistered view passed data/state and HTTP checks. The subsequent
registered view was also rendered and toggled in Chrome. Its historical field
uses reference z=13963, expanded XYZ shape `419×474×191`, and offset
`33308,20290,13868`; all three trace IDs are retained.

A separate historical automatic-estimation check covered 191 sections and
rejected 40 adjacent estimates, reusing neighboring transforms and recording those
fallbacks. Those sections were not claimed successfully registered. Historical
fields without per-section quality records do not establish universal alignment
accuracy. Display registration does not rerun PEC/ASP.

The [demo guide](../probe-em/scripts/README.md) describes launch commands and
versioned manual review. Generated viewer URLs require the corresponding Python
process. `--check` validates data and layer state, not browser rendering.

## Source audit and environment limits

A local workspace inventory parsed 1,736 Python files at an earlier checkpoint,
excluding environments, model weights, and volume caches. Semantic review and
changes concentrated on SAEM²/SAvEM³/Probe-EM, not every line of external baselines.
The main repository had no Python syntax errors. One unused historical
AxonEM-wrapper example in a separate checkout had syntax errors and was not rewritten.

The local editable SAM 2 device fixes are outside this Git repository; see the
[dependency caveat](REPRODUCTION.md#validation-and-external-dependencies).
The Probe-EM environment upgraded only Kimimaro 5.8.0 to 5.8.4 for the observed
skeleton coordinate-type conflict; other dependencies were not upgraded.

## Outstanding paper evaluation

Full SAEM² 12 epochs, SAvEM³ 200K iterations, formal ERL/OOD, and manual-time
evaluation remain outstanding. They require the corresponding data splits,
trained membrane/NeuroSAM 2 weights, self-supervised encoder initialization, and
compute resources. Graph cut currently uses local multicut; WaterZ prompting is
a voxel-interface approximation. Neither is presented as a complete lifted-edge
experiment or reproduction of the paper's quantitative tables.
