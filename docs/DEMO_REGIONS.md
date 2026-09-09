# Four online zebrafish demo regions

This document records the original v2 four-region experiment. The 2026-09-09
physical-scale registration optimization is documented in the
[alignment investigation](../probe-em/README_alignment.md). Old fields and tracing
results were preserved, not overwritten or relabeled as rerun results.

All raw ROIs came directly from
`precomputed://https://ng.zebrafish.digital-brain.cn/srv/raw/`.
Existing instance labels came from the same site's `srv/merge_neuro_label/` layer.
No synthetic volume or alternative dataset was substituted, and the original seed
`1486287284` demo was retained. The four-region labels are the site's existing
segmentation, not newly trained/inferred SAvEM³ outputs. Tracing used local
standard SAM 2 Hiera Tiny weights.

## Regions and viewer behavior

Coordinates are original-space XYZ at mip 0, with inclusive starts and exclusive
ends. Each ROI is `500×500×100` voxels at `8×8×30 nm/voxel`, or `4×4×3 µm`.
The regions are mutually disjoint and do not overlap the original demo cache.

| Region | Start XYZ | End XYZ | Seed | Retained fragments including seed |
| --- | --- | --- | --- | --- |
| 1 | 32500,19500,13700 | 33000,20000,13800 | 1459916967 | 4 |
| 2 | 34000,19500,14100 | 34500,20000,14200 | 1527977774 | 2 |
| 3 | 32500,21500,14200 | 33000,22000,14300 | 1541210292 | 1; no new merges |
| 4 | 34000,21500,13600 | 34500,22000,13700 | 1442531511 | 8 |

One persistent Python process serves four independent viewer URLs on `127.0.0.1`.
URLs are generated at launch and are intentionally not published as reusable links.
Each viewer defaults to local registration ON. Press A to switch raw, all instance
layers, skeletons, and the viewing position together; original data and trace IDs
are unchanged.

All visualization entry points default to `"showSlices": false`. This hides slice
planes in the 3D panel while retaining three 2D sections and 3D segments/skeletons.
The setting can be changed in Neuroglancer and is preserved across A toggles.

## Original tracing results and limits

Seeds were selected geometrically before inference. None of the selected seed
segments touched the original ROI boundary. Skeletons were generated with Kimimaro
from downloaded ROI labels, not copied from server-side whole-neuron skeletons.

Tracing used CPU, real SAM 2 Tiny weights, five ASP frames, `target_mip=0`,
and `align_z=true` with the same original v2 field as its viewer.
The budget expanded one seed node per region, not an entire neuron.

| Region | HSS candidate records | PEC result visualizations | ASP video tests | Final status | Pending fragments |
| --- | --- | --- | --- | --- | --- |
| 1 | 69 | 13 | 6 | limited | 3 |
| 2 | 195 | 79 | 20 | limited | 1 |
| 3 | 82 | 39 | 5 | complete | 0 |
| 4 | 121 | 57 | 12 | limited | 7 |

HSS counts endpoint-neighbor records and may include multiple endpoints for one
neighbor. Raw PEC pair counts were 15, 80, 42, and 58. Pairs failing area-ratio
or other prechecks do not produce result images; the table counts generated
evidence, not skipped pairs as completed inference.

All four runs recorded `errors=[]`. `limited` means a node-budget stop with the
pending queue retained. Region 3's `complete` means only that its current local
queue was exhausted, with no additional merges passing the threshold; it does not
mean a complete neuron was found. This region was not replaced based on the outcome.

| Region | Rejected/reused v2 adjacent estimates | Adjacent mean NCC, before → after | Registered XYZ grid |
| --- | --- | --- | --- |
| 1 | 49 / 99 | 0.135 → 0.209 | 534×529×100 |
| 2 | 28 / 99 | 0.190 → 0.311 | 567×532×100 |
| 3 | 15 / 99 | 0.285 → 0.360 | 554×585×100 |
| 4 | 58 / 99 | 0.219 → 0.260 | 525×548×100 |

The 150/396 rejected pairs were explicitly recorded as conservative fallback,
not successful estimates. NCC is a shared-valid-pixel intensity sanity check,
excluding added padding and sampling every other XY pixel. It is not anatomical
ground truth or a paper accuracy metric. The later three-way comparison uses a
common intersection across original/v2/v3, so its numbers differ slightly.

Chained translations can accumulate drift and do not solve nonrigid deformation.
The expanded display grid avoids clipping; all trace IDs remain present.

## Local cache and browser checks

The default local root is `probe-em/demo_session_regions/`, ignored by Git.
Each region contains raw, complete ROI labels, ROI skeletons, fields/quality metadata,
trace configurations, and results. Online JPEG is decoded and then cached losslessly;
uint32 labels are stored as uint64 without changing values. Exact voxel round trips
and SHA-256 checks passed. Repeating `prepare` verified and reused all four caches
without downloading volume data again.

`region.json` records source URLs, info snapshots, coordinates, download time,
checksums, and seed selection. Registration results, browser state, screenshots,
and check JSON remain local artifacts; the GitHub clone does not contain them.

Chrome checks covered all four original URLs: HTTP 200, actual canvas rendering,
and registration ON → A OFF → A ON. No runtime exceptions or failed requests were
observed. Later checks verified `showSlices=false`. Separate optimized-viewer
checks are described in the alignment guide. Test browser sessions were closed;
the user's browser and Python services were retained.

These short workflow tests, generic Tiny weights, and NCC changes do not reproduce
NeuroSAM 2 accuracy.

## Reopen or create the demos

With an existing cache, in `probe-em/` and an activated Probe-EM environment:

~~~bash
python scripts/demo_regions.py serve --open
~~~

Registration defaults on; add `--no-align-z` to disable display alignment.
Do not rerun tracing just to reopen viewers. Use `trace --resume` only when
intentionally continuing a compatible incomplete experiment.

A fresh clone must first run `prepare` and `trace`, using your own checkpoint;
see the [scripts guide](../probe-em/scripts/README.md#four-regions-from-the-same-online-database).
Fresh preparation now uses v3.2, so it does not recreate the historical v2 trace
experiment above. Original v2/v3 comparison scripts require the preserved v2 cache.
