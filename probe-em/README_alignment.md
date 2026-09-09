# Local registration investigation and improvement

Validated on 2026-09-09 using the same four cached online zebrafish regions.
Original volumes, fields, trace IDs, and viewer artifacts were retained.
No volume data was redownloaded and no region was replaced to improve metrics.

## Cause

The new demo path applied an inconsistent physical displacement limit, rejecting
many valid section shifts.

| Setting | Original single-region example | Original four-region v2 |
| --- | --- | --- |
| Estimation grid | mip 1, 32×32×30 nm | mip 0, 8×8×30 nm |
| Estimation XY window | 676×740 mip-0 pixels | 500×500 mip-0 pixels |
| Effective single-step limit | 8 estimation pixels = 256 nm | 10 estimation pixels = 80 nm |
| ECC Gaussian kernel | 5 pixels, approximately 160 nm | 5 pixels, approximately 40 nm |
| Failure handling | Reuse neighboring cumulative transform | Reuse neighboring cumulative transform |

The original estimator internally allowed 10 pixels, but its caller imposed an
additional 8-pixel limit. Moving estimation to native resolution without converting
physical thresholds materially changed the behavior. The finer grid also lacked
coarse-to-fine initialization, causing some valid pairs not to converge.

The new regions contain larger jumps: some reliable adjacent estimates exceed
320 nm, reaching approximately 756 nm. Low-quality sections also exist; relaxing
the displacement limit cannot recover missing matching information.

Controlled ablation on the same cache, measuring adjacent edges from lower to higher z:

| Estimation setting | Rejected adjacent edges across regions 1–4 |
| --- | --- |
| Original cached v2 | 49, 28, 15, 58; total 150/396 |
| Native resolution, 320 nm limit | 4, 5, 10, 7; total 26/396 |
| 32 nm → 16 nm → native, 320 nm limit | 4, 2, 1, 7; total 14/396 |
| Final v3.2 with quality gates and cross-section graph | 0, 3, 1, 0; total 4/396 |

Detailed local ablations are written to `demo_session_regions/alignment_ablation.json`.
Fewer rejected edges alone do not establish quality; image correlation, orthogonal
sections, and ID preservation were also checked.

## Implementation

[registration.py](probe_em/registration.py) implements estimation;
[demo_alignment.py](probe_em/demo_alignment.py) handles caching, coordinates,
volume resampling, and skeleton transforms. The default algorithm is
`physical-pyramid-graph-v3.2`.

1. Convert coarse scale and shift limits using actual nm/voxel, including
   anisotropic XY resolution and odd image sizes.
2. Start near 32 nm and refine through the pyramid. Quality-check each candidate
   before propagation so an incorrect coarse peak does not lock all finer levels.
3. Apply an 800 nm displacement limit, at least 70% overlap, and coarse NCC ≥ 0.25.
   Require no meaningful NCC regression versus unregistered data on identical
   valid pixels (numerical tolerance 0.005). All parameters and the algorithm
   version contribute to the cache key.
4. For translation, estimate adjacent and skip-section constraints and solve a
   correlation-weighted robust displacement graph. A 16 nm Huber scale downweights
   inconsistent edges without globally smoothing Z displacements, preserving real jumps.
5. Infer displacements outside the largest reliable connected component and flag
   them explicitly. Do not repair or generate raw imagery. Fail if there is no
   reliable connected component.

`euclidean` and `affine` retain experimental coarse-to-fine chain estimation with
explicit neighboring-transform fallback records. They are not presented as the
translation graph solution. Translation remains the default model.

## Four-region results

The table uses mean adjacent-section grayscale NCC on the same valid-pixel
intersection for original/v2/v3. Newly padded borders are excluded; XY is sampled
every other pixel. Mean shared valid support ranges from 96.1% to 98.5%.

| Region | Original | Four-region v2 | Optimized v3.2 | Sections without reliable observations |
| --- | --- | --- | --- | --- |
| 1 | 0.134 | 0.209 | 0.358 | None |
| 2 | 0.190 | 0.311 | 0.384 | z=14128 |
| 3 | 0.285 | 0.360 | 0.380 | None |
| 4 | 0.218 | 0.259 | 0.429 | None |

The four-region mean improves from approximately 0.285 to 0.388. Regions 1 and 4
improve most; region 3's v2 alignment was already stronger, so equal visual
improvements should not be expected everywhere. The original single-region
reference improved from 0.213 to 0.361 on its own valid support; it is not a
shared-sample quantitative benchmark against these four regions.

Rejected adjacent edges and unobserved sections are different: region 3's rejected
edge is bridged by skip-section evidence. Region 2 at z=14128 cannot be matched
reliably to nearby sections; its displacement is inferred and the imagery remains
unreliable. All 4, 2, 1, and 8 traced IDs are retained after optimization.

Local generated evidence for each `region_N/` includes:

- `alignment_optimized/report.json`: quality, common-support NCC, retained IDs, and field provenance.
- `alignment_optimized/comparison_xz.png`: original/v2/v3 at the same reference y, X range, scale, and contrast.
- `viewer_optimized/registered.png` and `original.png`: browser comparison screenshots from local validation.
- `viewer_optimized/browser_check.json`: browser validation record.

These artifacts remain in the local cache and are not bundled with GitHub.
Black transformed borders are outside the cached support, not new acquisition.
NCC and visual continuity are not anatomical ground truth. Local translation does
not solve every deformation, damaged section, or segmentation error.

## Viewer behavior and rerunning

All four optimized viewers passed HTTP 200, actual canvas display, A OFF → ON,
and persistence of `showSlices=false`, with no runtime exceptions or failed requests
during testing. Local registration defaults on. 3D slice planes default off;
the three 2D panels remain.

Generated URLs are in `demo_session_regions/viewers.json` and
`viewers_optimized.json`; preserved old URLs are in `viewers_original.json`.
They depend on live local Python services and are not reusable public links.

The comparison holds original trace IDs fixed and optimizes display registration
only; PEC/ASP was not rerun. Reports distinguish `trace_alignment_field` from
the displayed field. New tracing with a changed field requires a new result
directory and an explicit `align_z_field`, not continuation of the old experiment.

For a locally preserved original four-region cache, run in `probe-em/` with
the Probe-EM environment active:

~~~bash
# Recheck the controlled ablation without replacing the original field.
python scripts/investigate_alignment.py
# Estimate/reuse v3 fields, validate them, and generate XZ comparison figures.
python scripts/optimize_demo_alignment.py
# Prefer a validated optimized field matching the raw checksum.
python scripts/demo_regions.py serve --open
# Display the field from the original preparation step for comparison.
python scripts/demo_regions.py serve --legacy-alignment --open
~~~

A fresh clone has no cached v2 experiment: create data with `demo_regions.py prepare`
and run tracing as documented in [the scripts guide](scripts/README.md).
Fresh preparation uses v3.2. The historical v2/v3 numbers above are not promised
to reappear when the "original" field was itself generated by v3.2.

Generic `demo.py` supports `--align-max-shift-nm`, `--align-coarse-resolution-nm`,
and `--align-min-ncc` for automatic estimation only. Explicit historical fields
are unchanged. `--no-align-z` disables registration.

## Regression tests

The tracing/registration/demo subset has 29 passing tests, with third-party
SciPy/NumPy and Tornado deprecation warnings. It covers large physical shifts at
multiple resolutions, recovery from bad coarse initialization, missing-section
bridging and flags, unrelated images, robust graph outliers, invalid parameters,
and default selection of validated optimized fields.

~~~bash
# Repository root, in the Probe-EM environment:
python -m pytest -q tests/test_registration.py tests/test_demo_alignment.py \
  tests/test_demo_regions.py tests/test_probe_em.py
~~~
