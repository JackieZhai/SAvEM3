# Validation and evaluation

Last reviewed: 2026-11-05. Generated datasets, local configurations, browser
automation, screenshots, model execution experiments and detailed machine logs
are retained outside the published source. Maintained unit and regression tests
cover implementation contracts and previously observed failures.

The latest regression suite passed **163 tests**, with 7 skips and 1 third-party
deprecation warning. It used the real SAM2 checkpoint on MPS. Skips cover unavailable training/native dependencies, the separate
SOFIMA worker and an opt-in runtime-version check. Earlier validation exercised
the isolated SOFIMA and upgraded-runtime regressions. English-source and
whitespace checks passed. Model execution experiments and local sampling records
remain outside the published tree.

## Run the regression suite

From the repository root in the installed Probe-EM/training environment:

```bash
python -m pytest tests/ -q
python repro/check_english.py
git diff --check
```

Native ELF/WaterZ tests require the separate postprocessing environment described
in [repro/env](../repro/env/README.md). Two optional integration regressions use
local dependencies without downloads:

```bash
PROBE_EM_SAM2_CHECKPOINT=/path/to/sam2_hiera_tiny.pt \
SAVEM3_SOFIMA_PYTHON=/path/to/sofima-env/bin/python \
  python -m pytest tests/test_interactive_propagation.py tests/test_sofima_alignment.py -q
```

The SAM2 regression checks bidirectional propagation, cancellation and fresh
memory on correction runs. The SOFIMA regression checks actual nonzero refinement,
geometric validity, cache reuse and loading a saved field without its worker.
For MPS, set `SAVEM3_TEST_DEVICE=mps PYTORCH_ENABLE_MPS_FALLBACK=1`. CUDA is not
validated by the local runs recorded here.

Set `SAVEM3_SECURITY_VALIDATE_RUNTIME=1` to additionally check that the executing
environment contains the exact Torch/Torchvision versions pinned in the manifests.

## Interactive Neuroglancer

The [interactive guide](../probe-em/README_interactive.md) documents section-specific
points, current-section image prediction, bounded SAM2 propagation and local acceptance.
Regression coverage includes nonzero volume offsets, exact uint64 IDs, affine and
nonrigid point/overlay coordinates, stale-client rejection, cancellation, error
recovery and accepting only results matching their conditioning points.

Browser testing exposed two additional navigation issues: picking could snap a
new point to a neighboring section's annotation, and slice-panel key bindings
could override viewer actions. P/N now use the current XY section's z, and the
interactive controls are bound in both relevant input scopes. Relative D/F
navigation also tolerates pending client updates.

A real Chrome/WebGL session on cached zebrafish region 1 used SAM2/MPS with the
SOFIMA field. It predicted sections 13754 and 13755 consecutively without clearing
points, propagated through 13753-13756 in both z directions, toggled registration,
inspected an unprompted section and accepted the native four-section mask. There
were no JavaScript exceptions. Raw imagery, masks and mesh rendering were inspected.
Defaults remained registration on, `showSlices=false` and near-isotropic meshes.

The installed legacy SAM2 runtime produced nonfinite video logits with CPU-offloaded
MPS states. Keeping masks and temporal memory on MPS resolved the observed failure;
video input frames remain in CPU memory. This is a compatibility result, not a
segmentation accuracy claim.

A second browser session used a local 1024 x 1024 x 33 GT window at 8 x 8 x 30 nm.
Raw, GT, three saved SAM2 comparisons and mito/syn/ves context layers rendered
successfully. Slice navigation and A toggles were exercised with all eight volume
sources synchronized. The completed check recorded no JavaScript errors or failed
local requests. The 3D view contains a cropped slab, not a complete neuron.

Dense affine mask mapping now batches points by section. Regression coverage
checks fractional sections, inverse transforms, nonzero origins and exact label
geometry. This removes a Python loop per pixel; the optimization changes mapping
cost, not the model, estimated field or scoring definition.

The [window browser](../probe-em/README_windows.md) was exercised in Chrome/WebGL
with all five cached GT windows. Previous/Next, the selector, D/F section movement
and per-window point retention passed with zero JavaScript errors or failed local
requests. Every window retained default-on registration, `showSlices=false` and
32 x 32 x 30 nm mesh sampling. Unit tests cover duplicate loads, explicit retry,
shutdown during loading, exact seed IDs and relative manifest paths.
Real SAM2/MPS image prediction ran in two separate windows using the shared model.
A nine-section registered propagation completed in one window; switching away and
back preserved that volume and the other window's image preview. Its native saved
mask covered every requested section, with zero browser errors or failed requests.

The compact SAEM²-SAEvM³ toolbar adds an unregistered overview, now named
**LC-NE test overview**, to those five ROIs. Chrome measured a 31 px desktop bar
with 12 px controls. Full multiscale
raw/lc data, the existing neuron meshes and all five native-coordinate boxes rendered
without loading a whole-volume array. Viewed chunks were copied into the persistent
local cache. The six-view browser check exercised D/F, status expansion, image
prediction, nine-section propagation and returning to a retained ROI preview.
Live messages appeared in the toolbar without duplicate footer messages; browser
exceptions and failed local requests were zero. Window links retain the toolbar.

Regression coverage verifies physical bounds across different voxel resolutions
and nonzero offsets, exact segment IDs, overview registration remaining disabled,
status isolation, concurrent chunk-cache reuse, gzip/CORS/range responses and path
containment. Cached chunks remain available if the source disappears; uncached
requests then return an unavailable-source error. These checks establish viewer
and data-access behavior, not additional segmentation accuracy.

## SAM2 encoder acceleration

The [runtime guide](../probe-em/README_acceleration.md) records actual image/video
latency and mask agreement for Torch CPU/MPS, ONNX CPU/CoreML and a bounded feature
cache. The cached MPS runs were about 4.9x faster for revisiting image sections and
1.5x for repeated five-section propagation, with identical masks to uncached MPS.
ONNX CPU/CoreML worked but did not improve latency in that local workload; their
image/video Dice against Torch exceeded 0.999997. Torch remains the default.

Feature-store regression tests cover memory limits, eviction, pixel-content keys,
caller mutation, encoder checkpoint/artifact mismatch and shared-model concurrency.
The real SAM2 propagation regression now runs with caching disabled and enabled;
correction after cancellation agrees with a fresh uncached model. Temporal memory
is never reused between runs or windows. Windows/Linux and CUDA/DirectML hardware
were not available; the provider interfaces are not hardware-validation claims.

## Local registration

The [registration investigation](../probe-em/README_alignment.md) preserves controlled
ECC comparisons. The [SOFIMA guide](../probe-em/README_sofima.md) records the pinned
upstream source, separate JAX CPU environment, dense-field schema and final quality
guards. All four 500x500x100 zebrafish ROIs were reused from the original local caches.

| Region | ECC mean NCC | ECC + SOFIMA mean NCC | Nonzero elastic sections |
| --- | ---: | ---: | ---: |
| 1 | 0.36799 | 0.37014 | 41/100 |
| 2 | 0.39280 | 0.40036 | 65/100 |
| 3 | 0.39874 | 0.40029 | 53/100 |
| 4 | 0.43851 | 0.44085 | 62/100 |

Each comparison uses common valid pixels. Gains are modest and not uniform;
the worst pair lost 0.00999 NCC. Strength selection itself uses NCC, so these are
internal consistency checks rather than independent anatomical measurements.
The final minimum Jacobian was 0.9847, sampled point round-trip error was below
2.7e-7 voxels, and no label IDs were introduced or lost in these four volumes.
An initial unguarded trial made region 1 worse, motivating zero-mean residuals,
explicitly reported inference and final adjacent-pair guards. ECC remains default.

The [large local GT evaluation](../probe-em/README_local_gt.md) adds a separate
five-neuron dataset and scores native masks on unprompted sections. Its results
must not be conflated with the four online ROI image-consistency measurements.

## Tracing and reconstruction limits

The [five-ROI investigation](../probe-em/README_trace_quality.md) and
[SAEM2 comparison](../probe-em/README_saem2_trial.md) describe measured behavior
and remaining merge/split errors. Coordinate correctness, finite model outputs,
NCC gains and successful propagation do not establish complete-neuron accuracy.
Generic SAM2 weights are not the paper's NeuroSAM2 weights. Historical SAEM2
checkpoint compatibility does not establish training-data provenance.

Teacher export, distillation/resume, sliding-window edges, graph-cut interfaces,
physical coordinate handling and tracing failure/resume states have regression
coverage. Quantitative reproduction still requires the specified weights, data
and ground truth. A bounded interactive review does not edit the source instance
labels or automatically merge/split the reconstruction.

## LC-NE test

[LC-NE test](LC_NE_TEST.md) is the separately distributed example dataset for
bounded evaluation and Neuroglancer review. Five 512 x 512 x 63 process windows
exclude estimated soma regions, with soma-center distances from 11.0 to 38.0
micrometers. Actual SAM2/MPS inference compared unregistered and ECC inputs across
33 central sections per window; the [local GT guide](../probe-em/README_local_gt.md#process-window-comparison)
records the scores and remaining errors.

Chrome rendered the five process views with a 31 px toolbar, `showSlices=false`,
default ECC and no mito/syn/ves layers. Image prediction, D/F movement,
nine-section propagation and retaining the preview after switching were exercised
against the real model. Browser JavaScript errors were zero. The full-volume
overview displays the five process boxes and provides mito/syn/ves context with
registration disabled. Physical extents and overview-only context layers have
regression coverage.

## Dependency and source checks

The [security report](SECURITY_AUDIT.md) records the Torch 2.14.1 / Torchvision 0.29.1
upgrade and the affected ranges of the eight reported advisories. Local CPU/MPS
compatibility was exercised with installed weights. The native postprocessing
environment passed 11 tests with 2 environment skips in its preceding validation.
Third-party deprecation warnings remain.

After publication, GitHub marked all eight historical alerts `fixed` on
2026-11-04; the authenticated API reported zero open alerts. None were dismissed.

The separate SOFIMA environment passed `pip check`; its advisory scan found zero
known vulnerabilities among 54 audited distributions. The pinned SOFIMA development
version was unavailable to that advisory database and was skipped. An advisory
scan does not constitute an upstream source-code audit.

The publishing audit checks tracked and non-ignored new text for Han characters,
CJK punctuation and encoded Unicode, including hidden files. Chinese originals
and historical local artifacts remain in external backups. Git history is not
rewritten by the English-source cleanup.
