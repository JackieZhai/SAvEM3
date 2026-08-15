# SAM 2 Fine-Tuning for NeuroSAM 2

This folder provides the fine-tuning script used to adapt SAM 2 to electron
microscopy neuron segmentation prompts. The training setup freezes the SAM 2
image encoder and updates the prompt encoder and mask decoder.

The script depends on the official SAM 2 implementation:

https://github.com/facebookresearch/sam2

Install SAM 2 first, then run the commands below from the root of this
repository.

## Dataset

We used the EMNeuron dataset for SAM 2 adaptation:

https://huggingface.co/datasets/yanchaoz/EMNeuron

The Hugging Face repository provides the dataset archives. After downloading and
extracting the data, arrange the training samples as paired 2D raw images and
instance-label masks before running this script.

## Expected Data Format

`train_sam2_neuron.py` expects a prepared 2D dataset with this layout:

```text
data_root/
  imgs/
    sample_001_raw.tif
    sample_002_raw.tif
  masks/
    sample_001_label.tif
    sample_002_label.tif
```

The default filename mapping is:

```text
imgs/<name>_raw.tif  ->  masks/<name>_label.tif
```

Mask files should store integer instance IDs, with `0` reserved for background.
The raw image files should be readable by OpenCV. If your filenames use a
different convention, override the suffixes:

```bash
--raw-suffix .tif --label-suffix .tif
```

or any other suffix pair matching your prepared files.

## Training

```bash
python finetune_sam2/train_sam2_neuron.py \
  --sam2-checkpoint /path/to/sam2.1_hiera_large.pt \
  --model-cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --data-root /path/to/data_root \
  --output-dir checkpoints_finetuned \
  --max-iterations 25000 \
  --batch-size 16 \
  --lr 1e-5 \
  --weight-decay 4e-5
```

Important options:

- `--area-num`: maximum number of neuron instances sampled from each image.
- `--min-pixel-area`: minimum connected-component area for a sampled instance.
- `--safety-margin`: erosion radius used to sample positive prompts away from
  object boundaries.
- `--save-every`: checkpoint interval in iterations.

Checkpoints are saved as:

```text
sam2_iter_<iteration>.pt
```

## Logging

Training logs are written to `training_log.txt` in the output directory.
Weights & Biases logging is optional and disabled by default:

```bash
python finetune_sam2/train_sam2_neuron.py ... --use-wandb
```

## Notes

This fine-tuning code is provided to document the NeuroSAM 2 adaptation step.
The Probe-EM tracing pipeline uses the resulting checkpoint for zero-shot
connectivity inference and does not update model parameters during tracing.
