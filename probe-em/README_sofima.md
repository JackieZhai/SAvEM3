# Optional SOFIMA local registration

`demo.py --align-backend sofima` estimates a local elastic field using
[SOFIMA](https://github.com/google-research/sofima). Use this option for a bounded
raw/label cache. Opening the supplied LC-NE test fields does not require SOFIMA.

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
and a JAX CPU stack. On Apple Silicon this worker uses CPU; it does not
use PyTorch's MPS backend. The viewer and SAM predictors continue to run in their
own environment.

## Select the backend

In the usual Probe-EM/Torch environment:

```bash
python probe-em/scripts/demo.py \
  --local-dir /path/to/cached_volumes --seed 30001 \
  --results-dir /path/to/trace_results \
  --align-backend sofima --sofima-python /path/to/sofima-env/bin/python \
  --output-dir .cache/demo-sofima
```

The interactive mode uses the same selector. Replace `--results-dir` with the
`--interactive --prompt-backend ... --prompt-checkpoint ...` arguments from the
[interactive guide](README_interactive.md) if no completed trace is available.
P/N prompts, prediction masks, skeletons and raw/segmentation layers all follow
the selected nonrigid field. R toggles local realignment for all of them between registered and original
space. I image inference uses native original image pixels. V video propagation
uses the selected display space at task start and saves masks back in native
coordinates; its registered inputs include the SOFIMA nonlinear correction.

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

`demo.py`, `demo_windows.py`, and `AlignedVolume` load saved elastic fields.
Tracing accepts an explicit field through `align_z_field`. Dense cutouts must
remain inside the field's estimated domain.

## Compare local fields

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
