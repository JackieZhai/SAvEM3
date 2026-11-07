# SAM2 runtime options for interactive review

Interactive demos default to **PyTorch with a 512 MiB CPU image-feature cache**.
The cache accelerates returning to earlier sections and repeating propagation
after point edits. It does not store predictions or temporal memory. An optional
hybrid backend runs the SAM2 image encoder in ONNX Runtime while retaining the
upstream PyTorch prompt decoder, memory encoder, memory attention and propagation.
Both **I** and **V** use the selected encoder; V still performs real SAM2 tracking.

This integration applies to `demo.py --interactive` and `demo_windows.py`.
The existing tracing CLI and training paths retain their runtime defaults.
SAEM2/HQ-SAM use their existing image backends; the new encoder options require SAM2.

## Portable installation and provider selection

Install Probe-EM, PyTorch and the matching upstream SAM2 source first, following
[INSTALL.md](INSTALL.md). ONNX is optional and needs Python **3.11 or newer** for
the pinned runtime. The pinned exporter/runtime pair is ONNX 1.23.2 and ONNX
Runtime 1.31.1, with Torch 2.14.1 / Torchvision 0.29.1.

```bash
python -m pip install -r probe-em/requirements-onnx.txt
python -m pip check
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

| Host | `--onnx-provider` | ORT distribution | PyTorch decoder/memory device |
| --- | --- | --- | --- |
| macOS, Windows, Linux | `cpu` | `onnxruntime` | `cpu`, or another available Torch device |
| macOS | `coreml` | `onnxruntime` with CoreML provider | `mps` on Apple Silicon, or `cpu` |
| Windows/Linux with compatible NVIDIA GPU | `cuda` | `onnxruntime-gpu` | `cuda` recommended |
| Windows with supported DirectX 12 GPU | `directml` | `onnxruntime-directml` | `cpu`; this does not add DirectML to Torch |

Install **one** ORT distribution per environment. For CUDA, use the official
[installation and CUDA compatibility guidance](https://onnxruntime.ai/docs/install/),
installing `onnx==1.23.2` and a compatible `onnxruntime-gpu` instead of the CPU
requirements file. Select a provider package compatible with the installed accelerator runtime.
The [CoreML provider](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)
uses MLProgram with static input shapes. The
[DirectML provider](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)
uses sequential execution with memory patterns disabled.

CPU is the common portable option. An unavailable or failed requested provider raises an error. Accelerator sessions
allow CPU execution of unsupported graph nodes and report their provider list.
This is separate from silently replacing the requested provider, which is rejected.

## Export the encoder for your exact weights

```bash
python probe-em/scripts/export_sam2_onnx.py \
  --checkpoint /path/to/sam2_hiera_tiny.pt \
  --model-config sam2_hiera_t.yaml \
  --output-dir .cache/sam2-tiny-onnx
```

The exporter uses the installed SAM2 implementation and a fixed batch of one
normalized RGB image at the model's encoder size (1024 squared for the tested Tiny
model). It exports the three projected feature levels used by both image and video
predictors. Constant positional encodings are stored once in `positions.npz`.
An ONNX checker and CPU numerical comparison must pass before `manifest.json`
is marked validated. The export tolerance is 1e-3 absolute/relative.

The runtime verifies artifact hashes, the checkpoint SHA-256 and the config name
before loading. Keep the same SAM2 source/config implementation for export and
inference; re-export after changing that source, config or checkpoint. An incompatible pyramid or input geometry raises an error. Exported weights and benchmark data belong in local
caches and are excluded from Git.

## Run and tune

Append these options to the [interactive demo](README_interactive.md) or
[window browser](README_windows.md) command:

```text
--sam2-runtime onnx --sam2-onnx-dir .cache/sam2-tiny-onnx --onnx-provider cpu
```

For a CoreML trial on Apple Silicon, select `--onnx-provider coreml` and Torch
`--prompt-device mps` (`--device mps` in `demo_windows.py`). `--onnx-threads 4`
sets ORT's CPU thread count; zero uses its default. The Torch device controls the
decoder and video memory independently of the ONNX provider. This implementation
copies encoder inputs/outputs through CPU memory, including for GPU providers;
it does not yet implement device I/O binding or full-video ONNX export.

`--feature-cache-mb 512` is the interactive default for either encoder. Use zero
to disable it. Entries are detached CPU feature tensors, keyed by normalized pixel
content, shape, device and precision within one model instance. A 1024-squared
float32 Tiny pyramid occupies about 16 MiB. Callers receive independent copies,
so decoder mutations cannot corrupt another window's cached features. Position
encodings, active predictor state, model weights and loaded ROI arrays are additional
memory; the option is a feature-store budget, not a total process-memory limit.

Pixel hashing prevents reusing features for a different ROI or a differently
registered section. Revising points reuses valid image features but always creates
fresh temporal memory. Calls sharing a model are serialized to keep each window's
image and points together. Cancellation remains cooperative between model steps.
Eviction affects speed only; the original forward pass reconstructs the features.

## Measure your workload

`benchmark_sam2_runtime.py` measures image edits that alternate between three
sections and repeated bidirectional propagation through an odd-sized local window.
It records cold startup separately, synchronizes the selected Torch device, checks
complete frame coverage and writes masks, input fingerprints, timings, repeated-run
agreement and optional agreement with a reference. Failures leave a failed status.

```bash
python probe-em/scripts/benchmark_sam2_runtime.py \
  --local-dir /path/to/neuron/cache --point 5590 6119 956 \
  --checkpoint /path/to/sam2_hiera_tiny.pt --device cpu \
  --output-dir .cache/bench-torch
python probe-em/scripts/benchmark_sam2_runtime.py \
  --local-dir /path/to/neuron/cache --point 5590 6119 956 \
  --checkpoint /path/to/sam2_hiera_tiny.pt --device cpu \
  --runtime onnx --onnx-dir .cache/sam2-tiny-onnx --provider cpu \
  --reference .cache/bench-torch --output-dir .cache/bench-onnx
```

The absolute point and cached ROI must match. `--feature-cache-mb`, `--threads`,
`--slices` and `--repeats` control the experiment. A reference from different raw
pixels, frame encoding or prompts is rejected. These are runtime/mask-agreement
measurements, not GT accuracy or whole-neuron tracing scores.
