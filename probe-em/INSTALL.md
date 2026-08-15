# Installation

These steps reproduce the environment used for Probe-EM. Adjust the CUDA and
PyTorch versions to match your machine.

## Conda Environment

```bash
conda create -n ntracing python=3.10.18
conda activate ntracing
```

## PyTorch

For CUDA 12.4:

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

For other CUDA versions, install the matching PyTorch build from the official
PyTorch instructions.

## SAM 2

Install SAM 2 from your local clone or from the official repository, then make
sure its config files and checkpoints match the paths in `configs/config.json`.

```bash
cd /path/to/sam2
pip install -e .
```

## Probe-EM Dependencies

From this repository:

```bash
pip install -r requirements.txt
```

## Configuration

```bash
cp configs/config.example.json configs/config.json
```

Edit `configs/config.json` and set local paths for `raw_path`, `seg_path`,
`checkpoint_sam`, and other output paths.


## Apple Silicon / MPS

Create a Python 3.10 environment and install the macOS arm64 wheels:

```bash
python -m venv probe-em-env
source probe-em-env/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
pip install -e /path/to/sam2
pip install -r requirements.txt
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

Set `"device": "auto"` in `configs/config.json`. Probe-EM will use MPS when no
CUDA device is available.
