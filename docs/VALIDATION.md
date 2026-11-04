# Validation and evaluation

Last reviewed: 2026-11-04. Generated datasets, local configurations, browser
automation, screenshots, model execution experiments and detailed machine logs
are retained outside the published source. Maintained unit and regression tests
cover implementation contracts and previously observed failures.

The publication regression suite passed **144 tests**, with 4 native-package
skips and 1 third-party deprecation warning. It included the optional real SAM2
propagation and isolated SOFIMA regressions. English-source and whitespace checks
passed. Model execution experiments and local sampling records were removed from
the published tree after being retained in external backups.

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

## Interactive Neuroglancer

The [interactive guide](../probe-em/README_interactive.md) documents section-specific
points, current-section image prediction, bounded SAM2 propagation and local acceptance.
Regression coverage includes nonzero volume offsets, exact uint64 IDs, affine and
nonrigid point/overlay coordinates, stale-client rejection, cancellation, error
recovery and accepting only results matching their conditioning points.

Browser testing exposed two additional navigation issues: picking could snap a
new point to a neighboring section's annotation, and slice-panel key bindings
could override viewer actions. P/N now use the current XY section's z, and the
interactive controls are bound in both relevant input scopes. Relative J/K
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

## Dependency and source checks

The [security report](SECURITY_AUDIT.md) records the Torch 2.14.1 / Torchvision 0.29.1
upgrade and the affected ranges of the eight reported advisories. Local CPU/MPS
compatibility was exercised with installed weights. The native postprocessing
environment passed 11 tests with 2 environment skips in its preceding validation.
Third-party deprecation warnings remain.

The separate SOFIMA environment passed `pip check`; its advisory scan found zero
known vulnerabilities among 54 audited distributions. The pinned SOFIMA development
version was unavailable to that advisory database and was skipped. An advisory
scan does not constitute an upstream source-code audit.

The publishing audit checks tracked and non-ignored new text for Han characters,
CJK punctuation and encoded Unicode, including hidden files. Chinese originals
and historical local artifacts remain in external backups. Git history is not
rewritten by the English-source cleanup.
