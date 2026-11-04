# Optional SOFIMA local registration

`--align-backend sofima` adds nonrigid registration to the local Neuroglancer
demo. The default remains `ecc`, with the existing robust translation graph.
Registration itself is still enabled by default. Both backends preserve the
`showSlices: false` and metadata-driven near-isotropic mesh defaults.

This adapter uses the real [SOFIMA flow estimator and elastic mesh solver](https://github.com/google-research/sofima),
following its [EM section-alignment example](https://github.com/google-research/sofima/blob/main/notebooks/em_alignment.ipynb).
It is a bounded ROI implementation with additional quality controls, not the
upstream distributed/block alignment pipeline or a reproduction of its accuracy.

## Install the optional worker

Run from the repository root. SOFIMA/JAX and the Torch inference environment use
different NumPy/OpenCV dependencies, so install SOFIMA in a separate environment:

```bash
python3.12 -m venv /path/to/sofima-env
/path/to/sofima-env/bin/python -m pip install --upgrade pip setuptools wheel
/path/to/sofima-env/bin/python -m pip install -r probe-em/requirements-sofima.txt
/path/to/sofima-env/bin/python -m pip check
/path/to/sofima-env/bin/python probe-em/scripts/sofima_worker.py --describe
```

The requirements pin SOFIMA commit `c8b5e254a9b031a9c00f3d399e47bafda3c96a22`
and the tested JAX CPU stack. On Apple Silicon this worker uses CPU; it does not
use PyTorch's MPS backend. The viewer and SAM predictors continue to run in their
own existing environment. CUDA/JAX GPU installation is not validated here.

## Select the backend

In the usual Probe-EM/Torch environment:

```bash
python probe-em/scripts/demo.py \
  --local-dir /path/to/cached_volumes --seed 1459916967 \
  --results-dir /path/to/trace_results \
  --align-backend sofima --sofima-python /path/to/sofima-env/bin/python \
  --output-dir .cache/demo-sofima
```

The interactive mode uses the same selector. Replace `--results-dir` with the
`--interactive --prompt-backend ... --prompt-checkpoint ...` arguments from the
[interactive guide](README_interactive.md) if no completed trace is available.
P/N prompts, prediction masks, skeletons and raw/segmentation layers all follow
the selected nonrigid field. A toggles all of them between registered and original
space. I image inference uses native original image pixels. V video propagation
uses the selected display space at task start and saves masks back in native
coordinates; its registered inputs include the SOFIMA nonlinear correction.

For the four existing cached regions, run from `probe-em/`:

```bash
python scripts/demo_regions.py serve --align-backend sofima \
  --sofima-python /path/to/sofima-env/bin/python --open
```

This estimates separate fields under each region's `alignment_sofima/` and writes
`viewers_sofima.json`. Existing ECC fields and viewer artifacts remain available.
`--legacy-alignment` is an ECC-only option. The region viewer holds trace IDs
fixed; changing display registration does not recompute tracing decisions.

`--align-motion` continues to control ECC's translation/euclidean/affine model.
SOFIMA uses translation initialization and therefore requires the default
`--align-motion translation`. Its residual motion is nonrigid. An explicit
`--align-field` bypasses estimation and loads exactly that field, regardless of
the selected backend; loading a saved SOFIMA field requires no SOFIMA/JAX install.

## Estimation and quality controls

1. Estimate the existing physical pyramid/robust translation graph. This handles
   large section shifts and identifies unobservable sections.
2. On the translated images, compute SOFIMA masked cross-correlation flows on a
   physical patch grid. Reject unreliable peaks, excessive displacement and
   locally inconsistent flow vectors.
3. Relax SOFIMA's spring mesh toward the flow. Keep zero mean residual displacement
   per section so the nonlinear correction does not accumulate global drift.
   Keep the reference section fixed. The solver uses physical nanometer units.
4. Check convergence, displacement, Jacobians, valid support and correlation.
   Missing residuals are interpolated/extrapolated from reliable sections and
   explicitly reported as inferred; they are not counted as measured flow.
5. Validate the resulting stack against the translation initializer on shared
   valid pixels at native image detail (sampled every two XY pixels, without area
   averaging). Choose each section's residual strength from 0, 0.125, 0.25, 0.5
   and 1 with dynamic programming. Bound every adjacent pair's regression,
   including transitions to translation-only sections, and apply a small strength
   smoothness penalty (0.002). Require a positive mean gain; otherwise retain the
   translation initializer for the whole stack and report that outcome.

This final check matters: an improvement against one already warped neighbor
does not guarantee improvement across the final stack, especially near missing
observations. NCC is a consistency measure, not anatomical ground truth.

Use `--sofima-config /path/to/options.json` to override the supported settings:

| Setting | Default | Meaning |
| --- | --- | --- |
| `resolution_nm` | 16 | Target estimation XY pixel size; no upsampling |
| `patch_nm` | 768 | Spatial context per correlation patch |
| `stride_nm` | 256 | Spacing of flow/mesh nodes |
| `max_residual_nm` | 160 | Maximum nonlinear correction beyond translation |
| `max_gradient` | 0.45 | Bound on the physical residual gradient norm |
| `min_jacobian`, `max_jacobian` | 0.3, 2 | Bounds on local area change |
| `min_peak_ratio`, `min_peak_sharpness` | 1.6, 1.6 | SOFIMA flow confidence filters |
| `min_valid_fraction` | 0.1 | Required fraction of reliable patches; also at least four |
| `min_ncc_gain` | 0.001 | Minimum mean gain to accept refinement |
| `max_pair_ncc_regression` | 0.01 | Largest allowed individual pair regression |
| `mesh_k`, `mesh_k0` | 0.1, 0.01 | In-plane and data spring constants |
| `mesh_max_iters` | 10000 | Iteration limit, in multiples of 250 |
| `batch_size` | 64 | Flow patches processed together |

For example, `{"resolution_nm": 8}` estimates flow at native resolution on the
8x8x30 nm zebrafish cache. Small ROIs need at least a 3x3 patch grid; reduce patch
size/stride or select a larger ROI if this condition is not met. Parameters must
be finite and positive. No unknown settings are silently ignored.

## Fields, coordinates and artifacts

`ZElasticField` stores a per-section translation and a sampled XY residual grid
in native voxel units. The grid has an explicit absolute origin, spacing, source
window, physical resolution and reference z. Its forward map is translation
followed by bilinear residual interpolation. The bounded gradient gives a
convergent inverse. Raw pixels use linear interpolation; labels use direct nearest
neighbor indexing, preserving full uint64 values. Resampling can remove very small
instances; it does not guarantee preservation of every voxel or discrete topology.

Output bounds include interior displacement extrema. Skeleton edges are subdivided
in both XY and Z before transformation. Saved interactive predictions contain the
local dense display grid and translation instead of pretending the field is a
single affine matrix. The native mask and original XYZ prompts remain unchanged.

Cache fingerprints include raw bytes, grid/physical settings, algorithm, source
hashes and worker dependency versions. Field NPZ files contain numeric/string
arrays only and load with `allow_pickle=False`. Sidecar JSON records all accepted,
rejected and inferred sections, solver results and final per-section strengths.

The generic `demo.py`, `demo_regions.py serve`, and `AlignedVolume` support the
new field. Tracing can consume an explicitly selected saved field through
`align_z_field`; the automatic tracing estimator remains the historical ECC path.
Dense cutouts are restricted to the fitted ROI. Historical affine-only diagnostic
scripts (`investigate_alignment.py`, `optimize_demo_alignment.py`, `try_saem2.py`,
and the old tracing-quality plot exporter) require their original affine fields.
They fail explicitly if given an elastic NPZ.

## Reproducible local comparison

```bash
python probe-em/scripts/compare_alignment.py \
  --local-dir /path/to/cached_volumes \
  --sofima-python /path/to/sofima-env/bin/python \
  --output-dir .cache/alignment-comparison
```

The script reads only the local raw/label cache. It evaluates unregistered, ECC
and SOFIMA results on identical support, reports label-ID losses and point inverse
error, and retains both field caches plus `comparison.json`. Optional absolute
`--z-start` / exclusive `--z-stop` select a shorter stack. These comparisons do
not rerun SAM, alter trace IDs, or establish merge/split accuracy.

Tests cover invertibility, fold rejection, uint64 labels, expanded cutouts,
interactive point/overlay consistency, field serialization and actual SOFIMA
execution through the isolated worker. Set `SAVEM3_SOFIMA_PYTHON` when running
`tests/test_sofima_alignment.py` to include the real flow/mesh/cache test.

## Local validation (2026-11-04)

All four previously selected online zebrafish ROIs were reused from local disk,
at their complete 500x500x100 size and 8x8x30 nm resolution. No data was downloaded
again. The final default settings produced these adjacent-section NCC results;
all three methods use identical valid pixels within each comparison:

| Cached ROI | Unregistered | ECC | ECC + SOFIMA | Sections with nonzero elastic residual |
| --- | ---: | ---: | ---: | ---: |
| region_1 | 0.13361 | 0.36799 | 0.37014 | 41/100 |
| region_2 | 0.18925 | 0.39280 | 0.40036 | 65/100 |
| region_3 | 0.26324 | 0.39874 | 0.40029 | 53/100 |
| region_4 | 0.21266 | 0.43851 | 0.44085 | 62/100 |

The extra gain is modest. Some individual pairs regress (worst -0.00999 NCC);
this is bounded, not a claim that every pair improves. Strength selection itself
uses NCC, so these are internal consistency checks, not an independent accuracy
evaluation. No landmark ground truth or downstream merge/split assessment was
performed. ECC therefore remains the default.

The initial unguarded region_1 trial decreased NCC from 0.3681 to 0.3519 despite
many locally accepted refinements. Abrupt translation-only fallback sections and
residual drift motivated zero-mean meshes, explicit residual inference and the
final adjacent-pair guards. Preserved initial/global-strength trials document
these unsuccessful alternatives; they are not mixed into the final table.

Final shared support was 84.7-92.4%; the minimum Jacobian across all four fields
was 0.9847. Round-trip error stayed below 2.7e-7 voxels on a 5x5 grid over every
section. No label IDs were introduced or lost in these four resampled volumes.
That observation does not guarantee preservation of smaller instances in other data.
Numeric fields, full QC and comparison JSON are retained locally under
`.cache/sofima-20261104/region_{1..4}_selected/`, separate from historical fields.

The complete regression suite passed 142 tests with 4 native-package skips and
9 third-party warnings. It included real SAM/HQ-SAM/SAEM2/SAM2 CPU weights and
the isolated SOFIMA worker. The 12 SOFIMA tests include real nonzero elastic
refinement, safe rejection, geometry, large labels and interactive coordinates.
`pip check` passed in the worker environment. Its dependency audit found zero
known vulnerabilities among 54 audited distributions; the local SOFIMA development
version was skipped by the advisory service. This is not an audit of upstream source.

A Chrome/WebGL session on region_1 used a section with a nonzero SOFIMA residual
and real SAM2/MPS inference: P/N -> I -> A -> A -> Y. It produced a nonempty mask,
kept the positive prompt inside and the negative prompt outside, and saved the
native mask plus actual dense display transform. Reprojected saved points matched
the displayed annotations; screenshots verified visible raw data and the mask.
There were no JavaScript exceptions. `showSlices=false`, registration enabled,
and 32x32x30 nm mesh defaults were retained. Browser evidence and accepted test
artifacts are under `.cache/sofima-20261104/interactive-demo/`. This checks the
interactive coordinate/rendering path, not segmentation accuracy.
