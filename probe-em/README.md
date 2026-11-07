# Probe-EM

Probe-EM provides targeted neuron tracing and interactive review in Neuroglancer.
It supports point-guided image segmentation, SAM2 propagation across sections,
and saved review masks.

## LC-NE test demo

Install the dependencies in [INSTALL.md](INSTALL.md), then follow the
[window browser guide](README_windows.md). The supplied
[LC-NE test data](../docs/LC_NE_TEST.md) contains an overview and five process
windows. The release's `TUTORIAL.md` covers downloading, extraction, and controls.

| Guide | Contents |
| --- | --- |
| [Interactive review](README_interactive.md) | Positive/negative points, I/V prediction, acceptance |
| [Test windows](README_windows.md) | Window selector, manifests, overview layers |
| [Local realignment](README_alignment.md) | Saved fields and method selection |
| [SOFIMA](README_sofima.md) | Optional local elastic registration |
| [SAM2 runtime](README_acceleration.md) | Feature caching and ONNX Runtime |
| [SAEM² backend](README_saem2.md) | Image prediction with the membrane decoder |
| [Evaluation](README_local_gt.md) | Bounded mask evaluation against neuron labels |

## Tracing

Copy `configs/config.example.json` to `configs/config.json` and configure raw,
segmentation, model weights, seed IDs, and output paths:

```bash
python probe-em/scripts/run_probe_em.py --config probe-em/configs/config.json
```

The tracer searches neighboring segments around skeleton endpoints and verifies
candidate connections using image and video predictions. Results are written to
`<output_root>/<seed_id>_results_<suffix>/`. See the
[configuration guide](configs/README.md), [verification guide](README_trace_quality.md),
and [script reference](scripts/README.md).

## Modules

`probe_em/` contains the viewer, model adapters, registration, and tracing code.
See its [module guide](probe_em/README.md). `finetune_sam2/` contains
[training tools](finetune_sam2/README.md) for an EM-adapted SAM2 checkpoint.
