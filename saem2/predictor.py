"""Image-only SAEM2 inference with a strictly loaded auxiliary decoder.

This is not a SAM2 video predictor. The SAEM2 evaluation convention adds the
single SAM and HQ mask logits; the membrane branch is returned separately.
"""
import numpy as np
import torch

from repro.sam import load_predictor
from repro.savem3.export_teacher import load_decoder


class SAEM2Predictor:
    def __init__(self, hq_checkpoint, decoder_checkpoint, model_type='vit_b', device='cpu'):
        # Reject missing/incompatible trained heads before loading the encoder.
        self.decoder = load_decoder(decoder_checkpoint, model_type, device)
        self.backbone = load_predictor(hq_checkpoint, model_type, device)

    def set_image(self, image):
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Expected RGB uint8 HWC image')
        self.backbone.set_image(image)

    @torch.inference_mode()
    def predict(self, point_coords=None, point_labels=None, box=None):
        """Return native-YX masks and membrane probabilities, not IoU scores.

        Input prompts use original-image XY coordinates. Inference uses SAM's
        aspect-preserving resize/padding; native output coordinates are restored.
        The historical decoder has no calibrated quality head for its new masks.
        """
        predictor = self.backbone
        if not predictor.is_image_set:
            raise RuntimeError('Call set_image before predict')
        points = None
        if point_coords is not None:
            coords = np.asarray(point_coords, dtype=np.float32)
            labels = np.asarray(point_labels)
            if (coords.ndim != 2 or coords.shape[1] != 2 or len(coords) == 0
                    or labels.shape != (len(coords),) or not np.isfinite(coords).all()
                    or not np.isin(labels, [0, 1]).all()):
                raise ValueError('Expected finite Nx2 XY points and N binary labels')
            height, width = predictor.original_size
            if np.any(coords < 0) or np.any(coords >= [width, height]):
                raise ValueError('Points must be inside the original image')
            coords = predictor.transform.apply_coords(coords, predictor.original_size)
            points = (torch.as_tensor(coords, dtype=torch.float32, device=predictor.device)[None],
                      torch.as_tensor(labels, dtype=torch.int64, device=predictor.device)[None])
        elif point_labels is not None:
            raise ValueError('Point labels require coordinates')
        boxes = None
        if box is not None:
            box = np.asarray(box, dtype=np.float32)
            if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
                raise ValueError('Expected a finite nonempty XYXY box')
            box = predictor.transform.apply_boxes(box, predictor.original_size)
            boxes = torch.as_tensor(box, dtype=torch.float32, device=predictor.device)[None]
        if points is None and boxes is None:
            raise ValueError('Provide point or box prompts explicitly')
        sparse, dense = predictor.model.prompt_encoder(points=points, boxes=boxes, masks=None)
        sam, hq, membrane = self.decoder(
            predictor.features, predictor.model.prompt_encoder.get_dense_pe(),
            sparse[:, None], dense[:, None], False, False, predictor.interm_features)

        def native(logits):
            return predictor.model.postprocess_masks(
                logits, predictor.input_size, predictor.original_size)[0, 0]

        combined = native(sam + hq)
        return {'mask': (combined > 0).cpu().numpy(),
                'logits': combined.cpu().numpy(),
                'hq_mask': (native(hq) > 0).cpu().numpy(),
                'membrane': native(membrane).sigmoid().cpu().numpy()}
