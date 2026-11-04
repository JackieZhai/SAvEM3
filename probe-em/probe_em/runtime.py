"""Process-local, lazy SAM 2 predictors; a seed reuses weights across all nodes."""
from probe_em.device import resolve_device, sam2_config_name


class PredictorCache:
    def __init__(self, checkpoint, model_config, device='auto', *, sam2_runtime='torch',
                 onnx_dir=None, onnx_provider='cpu', onnx_threads=0, feature_cache_mb=0):
        self.checkpoint = checkpoint
        self.model_config = model_config
        self.device = resolve_device(device)
        self._image = None
        self._video = None
        from probe_em.sam2_acceleration import EncoderRuntime
        self.acceleration = EncoderRuntime(checkpoint, model_config, runtime=sam2_runtime,
                                          onnx_dir=onnx_dir, provider=onnx_provider,
                                          threads=onnx_threads, feature_cache_mb=feature_cache_mb)

    def image(self):
        if self._image is None:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            model = build_sam2(sam2_config_name(self.model_config), self.checkpoint, device=self.device)
            self.acceleration.install(model)
            self._image = SAM2ImagePredictor(model)
        return self._image

    def video(self):
        if self._video is None:
            from sam2.build_sam import build_sam2_video_predictor
            self._video = build_sam2_video_predictor(
                sam2_config_name(self.model_config), self.checkpoint, device=self.device)
            self.acceleration.install(self._video)
        return self._video
