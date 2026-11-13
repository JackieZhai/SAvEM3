# Probe-EM scripts

Run commands from the repository root after [installation](../INSTALL.md).
[Fish-X LC-NE Test](../../docs/LC_NE_TEST.md) is the supplied demo dataset.

## Entry points

| Script | Purpose |
| --- | --- |
| `demo_windows.py` | Fish-X LC-NE Test lc overview and five examples; full mode adds raw/context volumes |
| `demo.py` | Interactive prediction or trace review on a local cache |
| `run_probe_em.py` | Seed-based fragment tracing from a JSON configuration |
| `export_savem3.py` | H5 raw and instances to precomputed volumes and optional skeletons |
| `evaluate_local_gt.py` | Bounded SAM2 evaluation against local neuron masks |
| `evaluate_tracing.py` | Traced segment-set evaluation |
| `verify_traces_neuroglancer.py` | Segment-set annotation and review |
| `compare_alignment.py` | Local raw/ECC/SOFIMA geometry comparison |
| `sofima_worker.py` | Isolated SOFIMA execution |
| `export_sam2_onnx.py` | Export a SAM2 image encoder for ONNX Runtime |
| `benchmark_sam2_runtime.py` | Measure encoder and propagation performance |

Use `python probe-em/scripts/SCRIPT.py --help` for complete arguments.

## Fish-X LC-NE Test browser

```bash
python probe-em/scripts/demo_windows.py \
  --dataset examples \
  --checkpoint /models/sam2_hiera_tiny.pt \
  --model-config sam2_hiera_t.yaml --device auto \
  --output-dir .cache/fish-x-review --open
```

This default option opens the bundled lc labels/meshes and `test_windows` boxes,
with the five local examples in the selector. After
[downloading and extracting the full dataset](../../docs/LC_NE_TEST.md#2-full-dataset-from-hugging-face),
replace `--dataset examples` with `--dataset full --data-root /data/Fish-X-LC-NE-Test`
to add whole-volume raw and context images to the overview. `--prepare-only` prepares and
checks data without a model or server. `--windows PATH` accepts a custom manifest.
Fresh review directories load the bundled saved objects and points. Use
`--no-starter-session` with a new output directory to start blank.
`--dataset nuclei --data-root PATH` opens positive-center SAM2 segmentation on
the 32 x 32 x 30 nm overview; see the [nucleus guide](../README_nuclei.md).
See the
[window guide](../README_windows.md) and [interactive controls](../README_interactive.md).

## Meshes

Meshes use a near-isotropic grid selected from the source resolution. For
8 x 8 x 30 nm data, the mesh grid is 32 x 32 x 30 nm. The viewer's raw and
label slices retain their source resolution and coordinate origin.

`demo.py --no-mesh-isotropic` uses native-grid meshing. `--mesh-target-nm 64`
sets a different target. Small labels lost during sampling fall back to native
meshing; fine branches may still be simplified. The geometry cache includes
label content, sampling, and alignment settings. Dataset overview meshes are
served from their precomputed volume.

## Local realignment

`demo.py --align-field field.npz` loads a saved field. Otherwise the viewer
estimates a bounded ECC field. `--align-motion` selects translation, euclidean,
or affine motion; `--align-backend sofima` uses the optional elastic worker.
`--no-align-z` disables local realignment. See the
[alignment guide](../README_alignment.md) for controls and coordinate behavior.

## Export model results

```bash
python probe-em/scripts/export_savem3.py \
  --raw raw.h5 --seg instances.h5 --resolution 8 8 30 --offset 0 0 0 \
  --output /data/exported-roi --skeletonize --seed 1 \
  --checkpoint /models/sam2_checkpoint.pt --model-config sam2_hiera_t.yaml
```

Inputs are ZYX arrays with matching shapes. Supply the actual resolution and
absolute voxel offset. The exporter writes raw and segmentation precomputed
volumes, optional skeletons, and a tracing configuration. Review model paths
and budgets before starting a trace. It requires a new output directory.
