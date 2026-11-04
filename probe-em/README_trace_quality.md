# Tracing quality and guarded proofreading

The investigation keeps the same five zebrafish ROIs, caches and seeds. It does
not modify online segmentation or equate more/fewer accepted IDs with accuracy.
New local results live in ignored `demo_session_trace_quality/`; old viewers and
experiments remain intact.

## Findings

| Problem | Evidence and fix |
| --- | --- |
| Incomplete traversal | Each additional ROI expanded only the seed. Regions 1/2/4 still had 3/1/7 pending nodes. New results expand multiple nodes and expose remaining queues; the published CPU runs are bounded checkpoints, not completed neurons. |
| Different tracing/display fields | New viewers used v3.2, while decisions came from v2. New runs use the same field for both. |
| Registered reads clipped labels | `AlignedVolume` used the original bounds and a clipped intermediate grid. It now expands bounds and warps directly into the requested grid with constant padding. |
| Centroids were not reliable contacts | 83/482 saved original-space queries missed their nominated label: original 6/15; regions 1–4: 13/69, 36/195, 9/82, 19/121. New contacts are actual nearest candidate voxels. |
| Wrong prompt components | Boundary-clipped crops are not query-centered. Components now follow endpoint/contact anchors, not crop center or globally largest area. |
| Gaps silently skipped | Source search was limited to Z±1, and failed videos were reported as Z-gaps. Guarded preparation searches 500 nm and records unsupported cases explicitly. |
| Incidental ASP joins | All 13 additional IDs in historical traces had ASP support. Maximum label coverage alone can reward large predictions swallowing tiny fragments. Both additions in the original case were absent from saved HSS candidates. |
| Small-ROI topology overinterpreted | High density is not sufficient to prune a fragment with ≤15 endpoints in guarded mode. Direction lookback now follows physical cable length, not ten skeleton edges. |
| Topology checked too late | Region 4's new candidate 1374840086 has 133 endpoints. It was accepted into a pending queue before pruning, so a budget stop exposed it as merged. Guarded candidates are now checked before acceptance; matching resumes also recheck previously accepted candidates. |
| Browser selection differed from model output | Region 4's live trace layer contained 1374840086 despite its exclusion from the saved result. The initiating interaction was not established; this was not merely hover highlighting. Preserve live state before restoring references. Reference layers now disable spatial picking; manual review remains separate and hover highlight is off. |

Native labels contained one 55-voxel satellite in region 3's seed (19,106 voxels
in its main component). Every other displayed ID had one 26-connected component.
Region 2's added segment reaches the ROI edge. Neither connectivity nor truncation
proves biological correctness. Meshing can simplify thin branches independently.

These runs use **generic SAM 2 Hiera Tiny**, not EM-finetuned NeuroSAM 2. The
[Probe-EM paper](https://arxiv.org/html/2607.04696v1) uses the latter and includes
human proofreading; its accuracy must not be attributed to this demo. Input
labels are existing online `merge_neuro_label` data, not new SAvEM3 predictions.

## Verification policies

`verification_policy: "guarded"` is the default engineering extension for safer
proofreading, not a paper claim. The paper's 500 nm HSS sphere is retained.

| Stage | Additional safeguards |
| --- | --- |
| PEC | Keep 0.5/0.1 bidirectional thresholds and 4-of-5 voting. Each prediction must retain ≥50% of its source and place ≥50% of its pixels in the pair. |
| ASP input | At least five consecutive sections; extend only to reach the candidate plus one confirmation section, capped at 24. Unsupported gaps are explicit. Frame names remain dense. |
| ASP evidence | ID belongs to current HSS candidates; exclude prompted frame; coverage >0.6 and prediction precision ≥0.25 on two consecutive frames. |
| ASP return path | Prompt the actual candidate and propagate backwards: source coverage ≥0.5 and IoU ≥0.3. |
| Uncertainty | Short-lived fragments, leakage and failed return paths remain review proposals, not automatic joins. |

`verification_policy: "paper"` retains historical thresholds, preparation and
topology pruning with common coordinate/runtime fixes. It is not a byte-for-byte
legacy replay. `paper_accepted` versus guarded decisions on the same saved forward
predictions isolates ASP gates. Historical/revised final traces also differ in
registration, inputs and queue depth, so that comparison cannot isolate one cause.

Single-frame fragments can be real and reverse propagation can agree on the wrong
neurite. These checks are neither ground truth nor calibrated probabilities. The
24-frame cap bounds memory/drift; longer videos are not inherently more accurate.

## Rerun and compare

From `probe-em/` with a compatible environment and checkpoint:

~~~bash
python scripts/audit_trace_quality.py prepare \
  --output demo_session_trace_quality \
  --checkpoint /absolute/path/to/sam2_hiera_tiny.pt
python scripts/audit_trace_quality.py run \
  --output demo_session_trace_quality --node-limit 8
python scripts/audit_trace_quality.py summarize --output demo_session_trace_quality
python scripts/audit_trace_quality.py serve \
  --output demo_session_trace_quality --open-browser
~~~

`--cases original region_1 ...` selects existing cases, not new regions. The four
caches and validated v3.2 fields are reused. The original example gets a separate
lossless bundle with ROI skeletons and a v3.2 field because its historical cache
has no uploaded ROI skeletons. Voxel values/coordinates/seeds remain fixed, but
this skeleton regeneration is another difference from its historical run.

Runs use CPU float32 and two Torch threads. `--resume` continues matching runs.
Use new directories for different models/policies/fields/data. Model and field
SHA-256 values enter the signature; provided fields are not overwritten.
`complete` means local queue exhaustion, not a correctly completed neuron.
An interrupted node stays pending for resume. Previous scratch video attempts
are moved into `input_attempts/` before rebuilding, not silently discarded.

For a stopped worker, `snapshot --output ... --cases ...` publishes a separate
completed-node checkpoint, excluding evidence from the unfinished node. Stop
the worker before invoking this command; it does not terminate processes itself.
The raw run and pending queue remain resumable; `case.json` identifies the active
snapshot. A subsequent successful `run --resume` restores the raw run as active
without deleting older snapshots. This investigation uses bounded prefixes,
not equal-length or ground-truth-matched neuron reconstructions.
`topology --output ... --cases ...` records candidate skeleton QC and publishes
a separate topology-filtered checkpoint without changing the raw inference
evidence. The five-case investigation applies this check before visualization;
the 133-endpoint region-4 candidate is retained for review, not automatic merging.
Region 2 also defers candidate 1514667251 (22 endpoints and high density). These
topology heuristics flag risk; they do not prove either label is biologically merged.

The published completed-node counts are 4/4/2/2/2 for original/regions 1–4;
pending counts are 2/1/3/0/5. Region 3 exhausts its local queue; the other four
are explicitly limited. Earlier workers were stopped at their last atomic node
checkpoint; partially processed node evidence stays on disk but is excluded.
The same eager topology rule was applied to the completed inference snapshots
because some workers had started before that code change.

## Evidence and controls

Each case has `baseline.json`, `trace_config.json`, `comparison.json`, and the old
result path. New traces retain queue/errors/topology in `status.json`,
`trace_state.json`, and `node_audit.json`; HSS plus accepted IDs in
`verification_<id>.json`; preparation outcomes and per-trial PEC scores in
`temp_vis_<id>/preparation.json` and `connection_results.json`; ASP frame metrics,
reasons and reverse-cycle scores in `temp_vis3d_<id>/*_evidence.json`; forward masks
and prompts in `*_masks.npz`. Exact duplicate videos have explicit report aliases.
`keep_inputs: true` additionally retains `temp_slices*` for this investigation.

Viewers show `probe_trace` with hidden `baseline_trace` and `uncertain_proposals`
comparison layers. `manual_correction` is editable; `g` saves a reversible local
ID selection, `q` restores its guarded starting set, and `a` switches all images,
segmentation/comparison layers and skeletons together. Registration defaults ON,
`showSlices` false and near-isotropic meshing ON; 2D data remain native.
Spatial picking is disabled in algorithm/comparison layers, not in manual review.
These are not immutable layers: sidebar edits remain possible. `r` restores the
algorithm/comparison selections without changing the manual review. Browser checks
must compare exact live trace IDs with the saved result, not just request every
expected mesh, since extra selected meshes can otherwise escape validation.

Probe-EM joins existing IDs. It cannot undo a false merge **inside one input ID**:
that needs voxel-level splitting/resegmentation and proofreading. This workflow
does not automatically discard satellites or invent split labels. Extend context
from the same source into a separate cache when a neurite reaches the ROI edge.

## Validation limits

`tests/test_trace_quality.py` covers translated boundaries, exact uint64 IDs,
anchored components, multislice gaps/dense numbering, leakage, prompted-frame
exclusion, candidate gating and valid temporal support. Run it with existing
tracing/registration/mesh/demo/real-model tests. Precision/recall and ERL still
require proofread ground truth, excluding the seed for incremental tracing scores.
