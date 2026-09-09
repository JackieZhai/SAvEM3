"""HQ-SAM loading and original-image → padded SAM mask-prompt conversion."""
from pathlib import Path
import sys

import numpy as np


def resolve_device(device="auto"):
    import torch
    if device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else
                        "mps" if torch.backends.mps.is_available() else "cpu")


def load_hq_sam(checkpoint, model_type="vit_h", device="auto"):
    import torch
    source = Path(__file__).resolve().parents[1] / "sam-hq"
    sys.path.insert(0, str(source))
    from segment_anything import sam_model_registry
    import segment_anything
    if source not in Path(segment_anything.__file__).resolve().parents:
        raise RuntimeError("A different segment_anything package is already imported; run this entry point in a fresh process")
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f"HQ-SAM checkpoint not found: {checkpoint}")
    keys = torch.load(checkpoint, map_location='meta', weights_only=True)
    if 'mask_decoder.embedding_encoder.0.weight' not in keys:
        raise ValueError('Expected full HQ-SAM weights, not a base SAM or decoder-only checkpoint')
    del keys
    model = sam_model_registry[model_type](checkpoint=str(checkpoint)).to(resolve_device(device)).eval()
    return model


def load_predictor(checkpoint, model_type="vit_h", device="auto"):
    model = load_hq_sam(checkpoint, model_type, device)
    from segment_anything import SamPredictor
    return SamPredictor(model)


def mask_prompt(mask, predictor, magnitude=6.0):
    """Resize and pad a binary mask like SAM images; return 1x256x256 logits."""
    import cv2
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim != 2 or mask.shape != tuple(predictor.original_size):
        raise ValueError("Mask must be in the original image's YX coordinates")
    input_h, input_w = predictor.input_size
    size = predictor.model.image_encoder.img_size
    padded = np.zeros((size, size), dtype=np.float32)
    padded[:input_h, :input_w] = cv2.resize(mask, (input_w, input_h), interpolation=cv2.INTER_NEAREST)
    low = cv2.resize(padded, (256, 256), interpolation=cv2.INTER_LINEAR)
    return ((2.0 * low - 1.0) * magnitude)[None].astype(np.float32)


def choose_mask(output):
    """SAM/HQ-SAM return (masks, IoU estimates, logits), not a dictionary."""
    masks, scores = output[:2]
    scores = np.asarray(scores).reshape(-1)
    masks = np.asarray(masks)
    if not len(masks) or len(scores) != len(masks) or not np.isfinite(scores).all():
        raise ValueError("Invalid SAM prediction")
    # SAM-Graph's 0.05 margin, also valid for HQ-SAM's single selected mask.
    index = int(np.argmax(scores))
    if len(scores) == 3:
        if scores[2] > scores.max() - 0.05:
            index = 2
        elif scores[1] > scores[0] - 0.05:
            index = 1
    return masks[index].astype(bool), float(np.clip(scores[index], 0, 1))
