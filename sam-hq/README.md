# HQ-SAM teacher and shared foundation model

Based on [SysCV/sam-hq](https://github.com/SysCV/sam-hq). This directory contains
SAM image/prompt encoders, the HQ-token decoder, predictors, and data-bank locations.

`segment_anything/predictor.py` returns `(masks, scores, low_res_logits)`.
`mask_input` must contain `1×256×256` logits aligned with image resize/padding.
The shared wrapper in `repro/sam.py` handles these details and local HQ-SAM imports.

## Weights

Place a complete HQ checkpoint such as `sam_hq_vit_h.pth` under
`sam-hq/pretrained_checkpoint/`. Original SAM and base mask-decoder checkpoints
remain necessary for SAEM² initialization. A complete HQ checkpoint, SAM encoder
checkpoint, and SAEM² `epoch_N.pth` are not interchangeable.

Export/prompt entry points check for HQ feature-encoder weights to avoid using
random HQ layers. Model size must match `--model-type vit_b/vit_l/vit_h`.

## Relationship to SAEM²

HQ features are `embedding_encoder(SAM embedding)+compress_vit_feat(early ViT)`
with shape `32×256×256`; SAM embeddings have shape `256×64×64`.
SAEM² adds a membrane token/MLP in [saem2/modeling.py](../saem2/modeling.py) and
uses a full-image box to produce membrane predictions. Probe-EM uses SAM 2 as
a separate downstream verification model.
