"""Process-local, lazy SAM 2 predictors; a seed reuses weights across all nodes."""
from probe_em.device import resolve_device, sam2_config_name


class PredictorCache:
    def __init__(self, checkpoint, model_config, device='auto'):
        self.checkpoint = checkpoint
        self.model_config = model_config
        self.device = resolve_device(device)
        self._image = None
        self._video = None

    def image(self):
        if self._image is None:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            model = build_sam2(sam2_config_name(self.model_config), self.checkpoint, device=self.device)
            self._image = SAM2ImagePredictor(model)
        return self._image

    def video(self):
        if self._video is None:
            from sam2.build_sam import build_sam2_video_predictor
            self._video = build_sam2_video_predictor(
                sam2_config_name(self.model_config), self.checkpoint, device=self.device)
        return self._video
