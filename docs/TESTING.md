# Tests

Run the maintained unit and regression suite from the repository root:

```bash
python -m pytest tests/ -q
python repro/check_english.py
git diff --check
```

Tests cover training losses and resume, probability conventions, graph cut,
coordinate conversion, uint64 labels, alignment, tracing state, interactive
point correction, SAM2 propagation, and viewer navigation.
The Fish-X LC-NE Test checks verify bundled archive extraction, both overview
modes, saved window annotations, portable mesh serving, and stable review paths.
Starter-session checks cover native mask integrity, aligned/native point mapping,
continued tracing, and protection of existing review outputs.
Full-volume setup tests use metadata fixtures; they do not download the full release.

## Optional model tests

Set a local SAM2 checkpoint to enable real image/video inference tests:

```bash
PROBE_EM_SAM2_CHECKPOINT=/models/sam2_hiera_tiny.pt \
  python -m pytest tests/test_interactive_propagation.py -q
```

The checkpoint must match `sam2_hiera_t.yaml`. `SAVEM3_TEST_DEVICE` selects
`cpu`, `mps`, or `cuda`; the default is CPU. On Apple Silicon, enable
`PYTORCH_ENABLE_MPS_FALLBACK=1` when required by the installed SAM2 version.

Set `SAVEM3_SOFIMA_PYTHON` to the isolated SOFIMA Python executable to include
the worker integration test. Set `SAVEM3_SECURITY_VALIDATE_RUNTIME=1` to check
the installed Torch/Torchvision pair against the dependency pins.

ELF/WaterZ tests require the [postprocessing environment](../repro/env/README.md).
Tests for unavailable optional dependencies report a skip. Model tests use local
weights and do not download datasets or checkpoints.

## Manual viewer checks

Use the supplied [Fish-X LC-NE Test windows](DEMO_REGIONS.md) to check rendering and
interaction: switch windows, move through Z with D/F during dragging, toggle R,
add positive and negative points, run I, set B/E, run V, and save an object with C.
Create another object and check its new color and retained per-object point layers.
Use Y and both confirmations to save the final union; verify cancellation as well.
Verify annotation persistence and overlays on prompted and unprompted sections.
Keep new review outputs, browser captures, and performance measurements outside
source control. The bundled starter reviews are versioned demo examples.
