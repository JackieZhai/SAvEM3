# Installation

Use a fresh environment so historical Demo environments remain reproducible.
Python 3.12 is recommended; the pinned Torch pair requires Python >=3.10.
Commands below run from the repository root unless stated otherwise.
See [the security audit](../docs/SECURITY_AUDIT.md) for the upgrade scope and limits.

## Conda Environment

```bash
conda create -n ntracing python=3.12 pip
conda activate ntracing
python -m pip install --upgrade pip setuptools wheel
```

## PyTorch

For Linux/Windows with a compatible NVIDIA driver, choose the CUDA 12.6 wheels:

```bash
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
```

For Linux/Windows CPU-only machines, use `--index-url https://download.pytorch.org/whl/cpu`.
For macOS Apple Silicon, omit the index option. The macOS wheels require macOS 14+.
Do not keep an old cu118/cu124 wheel index or install mismatched Torchvision wheels.
Other accelerator builds must follow the [official version matrix](https://pytorch.org/get-started/previous-versions/).
CUDA installation choices are not evidence that CUDA inference was tested locally.
Torchaudio is not required by this repository or SAM 2 and is intentionally omitted.

## SAM 2

Install SAM 2 from your local clone or from the official repository, then make
sure its config files and checkpoints match the paths in `configs/config.json`.

```bash
python -m pip install --no-build-isolation -e /path/to/sam2
```

## Probe-EM Dependencies

Install the repository requirements without downgrading the pinned Torch pair:

```bash
python -m pip install -r probe-em/requirements.txt
python -m pip check
```

## Configuration

For optional SOFIMA registration, follow the [separate worker installation](README_sofima.md#install-the-optional-worker).
Do not install `requirements-sofima.txt` into the Torch environment; it pins a
separate tested JAX/NumPy/OpenCV stack. Saved fields do not require that worker to view.

```bash
cp probe-em/configs/config.example.json probe-em/configs/config.json
```

Edit `probe-em/configs/config.json` and set local paths for `raw_path`, `seg_path`,
`checkpoint_sam`, and other output paths.


## Apple Silicon / MPS

Create a separate Python 3.12 environment and install the macOS arm64 wheels:

```bash
python3.12 -m venv probe-em-env
source probe-em-env/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r probe-em/requirements.txt
python -m pip install --no-build-isolation -e /path/to/sam2
python -m pip check
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

Set `"device": "auto"` in `probe-em/configs/config.json`. Probe-EM will use MPS when no
CUDA device is available.

For the SAEM2/HQ-SAM comparison, also install `probe-em/requirements-saem2.txt`.
Run `python -m pytest -q tests/` from the repository root after installing pytest
and configuring the optional real-checkpoint environment variables documented in
[the validation record](../docs/VALIDATION.md). Native ELF/WaterZ tests belong in
the separate postprocessing environment, not this Torch environment.
