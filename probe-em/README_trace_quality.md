# Tracing verification

The tracer expands a seed through candidate segments near skeleton endpoints.
`verification_policy` selects the connection checks in the tracing configuration.

| Policy | Behavior |
| --- | --- |
| `guarded` | Bidirectional image checks, temporal agreement, return-path checks, and topology checks |
| `paper` | Original PEC/ASP thresholds and topology rules |

The guarded policy requires image predictions to retain at least half of their
source and place at least half of their pixels within the candidate pair. Video
checks exclude the prompted frame, require candidate coverage above 0.6 and
prediction precision of at least 0.25 on two consecutive frames, and check a
reverse prediction with source coverage at least 0.5 and IoU at least 0.3.
Unsupported gaps and uncertain connections remain review proposals.

## Results and resume

Each seed produces `status.json`, `trace_state.json`, selected IDs, and a merge
tree. Status is `running`, `complete`, `limited`, or `failed`. A `limited` run
has pending work after reaching its node budget. `complete` means the search
queue is exhausted; it does not certify a complete neuron.

`--resume` restores a run with the same configuration and retries failed nodes.
Use a new output directory when changing data, checkpoints, alignment, or policy.
Keep the same saved field for tracing and display when alignment is enabled.

Inspect accepted connections and crop boundaries in Neuroglancer. Existing
instance agreement is not independent ground truth. For the supplied LC-NE
annotations, see [bounded evaluation](README_local_gt.md). Interactive I/V mask
review and automatic fragment tracing are separate workflows.
