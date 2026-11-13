# Probe-EM

Probe-EM provides targeted neuron tracing and interactive review in Neuroglancer.
It supports point-guided image segmentation, SAM2 propagation across sections,
and saved review masks.

## Fish-X LC-NE Test demo

Install the dependencies in [INSTALL.md](INSTALL.md), then choose one of three
launch options in the [window browser guide](README_windows.md):

| Option | Views and data |
| --- | --- |
| **1. `--dataset examples`** (default) | Bundled lc overview, five window boxes, and five process examples; about 114 MiB, with no dataset download or login |
| **2. `--dataset full --data-root PATH`** | Full raw/label/context overview and the same five examples; download and extract the Hugging Face volumes first |
| **3. `--dataset nuclei --data-root PATH`** | SAM2 nucleus segmentation on the 32 x 32 x 30 nm raw overview, plus the same five examples; requires the extracted raw volume's 32 nm scale and metadata |

The official dataset is
[Fish-X LC-NE Test on Hugging Face](https://huggingface.co/datasets/JackieZhai/Fish-X_LC-NE_Test).
See the [dataset guide](../docs/LC_NE_TEST.md) for access and extraction.
Fresh process reviews load eight saved objects and 55 annotation points from
the bundled starter session. Existing reviews are preserved; use
`--no-starter-session` with a new output directory to start blank.

For nuclei, **P → V → C** places one positive center, propagates in both section
directions, and confirms the nucleus. All centers share one `nucleus_centers`
layer, while each accepted nucleus has its own colored mask. Negative points
are not used in this mode. In either workflow, **C** saves the current object
and starts the next in a new color; **Y** saves a combined mask after two
confirmations, retaining the individual masks and annotations.

| Guide | Contents |
| --- | --- |
| [Interactive review](README_interactive.md) | Positive/negative points, I/V prediction, acceptance |
| [Nucleus segmentation](README_nuclei.md) | Positive centers, bounded bidirectional propagation, one shared annotation layer |
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

## Citation

**Probe-EM: Targeted Neuron Tracing via Training-Free Semantic Verification**

*Liuyun Jiang, Yanchao Zhang, Jinyue Guo, Chuanyue Chen, Haiyang Yan, Ye Yuan,
Jing Liu, Hua Han* <br>
*State Key Laboratory of Brain Cognition and Brain-inspired Intelligence Technology,
Institute of Automation, Chinese Academy of Sciences*

MICCAI 2026 paper: [published proceedings](https://doi.org/10.1007/978-3-032-38224-5_59).

```bibtex
@inproceedings{Jiang-Probe-EM,
    author={Jiang, Liuyun and Zhang, Yanchao and Guo, Jinyue and Chen, Chuanyue and Yan, Haiyang and Yuan, Ye and Liu, Jing and Han, Hua},
    booktitle={Medical Image Computing and Computer Assisted Intervention -- MICCAI 2026},
    title={Probe-EM: Targeted Neuron Tracing via Training-Free Semantic Verification},
    year={2027},
    pages={624--633},
    doi={10.1007/978-3-032-38224-5_59}
}
```
