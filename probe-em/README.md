# Probe-EM

[English](README.md) | [中文](README_zh.md)

[![arXiv](https://img.shields.io/badge/arXiv-2607.04696-b31b1b.svg)](https://arxiv.org/abs/2607.04696)

Official implementation of the MICCAI 2026 paper:

**Probe-EM: Targeted Neuron Tracing via Training-Free Semantic Verification**

Probe-EM is a targeted neuron tracing framework for large-scale electron
microscopy volumes. It starts from seed segment IDs, searches geometrically
plausible neighboring segments, and verifies candidate connections with SAM
2-based semantic verification.

Paper: [arXiv:2607.04696](https://arxiv.org/abs/2607.04696)

## Repository Structure

```text
Probe-EM/
  configs/              Example configuration files
  probe_em/             Core tracing and semantic verification modules
  scripts/              Entry-point scripts for tracing, evaluation, and review
  finetune_sam2/        SAM 2 adaptation code for NeuroSAM 2
  INSTALL.md            Environment setup notes
  requirements.txt      Python dependencies
```

## Installation

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Probe-EM depends on the official SAM 2 implementation. Install SAM 2 and prepare
the corresponding model config and checkpoint before running the tracing
pipeline.

See [INSTALL.md](INSTALL.md) for the environment setup used in our experiments.

## Configuration

Copy the example config and edit the local paths:

```bash
cp configs/config.example.json configs/config.json
```

Key fields:

- `raw_path`: path or CloudVolume URL for the raw EM volume.
- `seg_path`: path or CloudVolume URL for the segmentation volume.
- `checkpoint_sam`: path to the SAM 2 checkpoint.
- `model_cfg_sam`: SAM 2 model config path.
- `seed_ids`: seed segment IDs to trace.
- `seed_list_file`: optional text file with one seed ID per line.
- `output_root`: output directory for tracing results.
- `max_workers`: number of seed IDs traced in parallel.
- `slice_workers`: number of workers used for slice extraction inside each seed run.
- `debug_limit`: optional maximum number of graph nodes processed per seed.
- `sam2_num_frames`: number of axial frames used for 3D propagation.

Use local paths such as `/path/to/...` or `file:///path/to/...`.

## Run Tracing

```bash
python scripts/run_probe_em.py --config configs/config.json
```

The output for each seed is written to:

```text
<output_root>/<seed_id>_results_<suffix>/
```

## SAM 2 Fine-Tuning

The training script for adapting SAM 2 to neuron segmentation prompts is in
`finetune_sam2/`. See [finetune_sam2/README.md](finetune_sam2/README.md) for
the EMNeuron dataset reference, expected 2D image/mask format, and training
command.

The tracing pipeline uses the resulting checkpoint for semantic verification and
does not update model parameters during tracing.

## Evaluation

`scripts/evaluate_tracing.py` evaluates traced segment IDs against ground-truth
segment sets. The ground-truth folder should contain files named like:

```text
<seed_id>_gt_segments.txt
```

Example:

```bash
python scripts/evaluate_tracing.py \
  --seg-path /path/to/segmentation/precomputed \
  --gt-folder /path/to/gt_labels \
  --results-dir trace_results \
  --suffix sam \
  --output-dir quantification_results \
  --excel-name evaluation_results.xlsx
```

By default, the seed segment is excluded from metric calculation. Add
`--include-seed` if you want to include it.

## Interactive Verification

`scripts/verify_traces_neuroglancer.py` opens a Neuroglancer viewer for offline
inspection of Probe-EM tracing results and manual correction.

```bash
cp configs/verification_config.example.json configs/verification_config.json
python scripts/verify_traces_neuroglancer.py --config configs/verification_config.json
```

The tool expects tracing outputs under `results_folder` with folders named like
`<seed_id>_results...`. Saved annotations are written to `gt_folder` as
`<seed_id>_gt_segments.txt`.

Useful hotkeys in the Neuroglancer viewer:

- `q`: load tracing results for the segment under the cursor.
- `a`: highlight all seed IDs that have tracing result folders.
- `t`: load saved ground-truth labels for the selected ID.
- `w`: display all annotated ground-truth segment IDs.
- `g`: save the current segment set as ground truth.
- `b`: jump to the configured `home_position`, if provided.
- `c`: clear selected segments and log annotations.

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@article{jiang2026probeem,
  title={Probe-EM: Targeted Neuron Tracing via Training-Free Semantic Verification},
  author={Jiang, Liuyun and Zhang, Yanchao and Guo, Jinyue and Chen, Chuanyue and Yan, Haiyang and Yuan, Ye and Liu, Jing and Han, Hua},
  journal={arXiv preprint arXiv:2607.04696},
  year={2026}
}
```

The official proceedings citation will be added after the MICCAI 2026 version is
available.


## Apple Silicon / MPS

Probe-EM now auto-selects `cuda` -> `mps` -> `cpu` when `"device": "auto"` is set
in the config. A local MPS environment can be created as follows:

```bash
conda create -n probe-em python=3.10
conda activate probe-em
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
pip install -e /path/to/sam2        # use an MPS-compatible SAM 2 checkout
pip install -r requirements.txt
```

Then run:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
python scripts/run_probe_em.py --config configs/config.json
```

Notes:
- `probe_em/device.py` handles CUDA/MPS/CPU selection.
- `probe_em/mps_patch.py` patches SAM 2 rotary-position encoding for MPS.
- The 2D SAM 2 image-verification path and the 3D video-propagation path were
  verified on an M4 Pro with `sam2_hiera_tiny.pt`.
- Full tracing still needs a real precomputed `raw_path` and `seg_path`.
