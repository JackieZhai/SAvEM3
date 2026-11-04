"""Lazy local image predictors for explicitly triggered point-prompt sessions."""
import hashlib
from pathlib import Path

import numpy as np


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class BaseSAMPredictor:
    """Original SAM decoder with the repository's tuple-returning image encoder."""

    def __init__(self, checkpoint, model_type, device):
        import torch
        from saem2.segment_anything_training import sam_model_registry
        from saem2.segment_anything_training.utils.transforms import ResizeLongestSide
        self.model = sam_model_registry[model_type]()
        self.model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        self.model = self.model.to(device).eval()
        self.device = device
        self.transform = ResizeLongestSide(self.model.image_encoder.img_size)

    def set_image(self, image):
        import torch
        self.original_size = image.shape[:2]
        resized = self.transform.apply_image(image)
        self.input_size = resized.shape[:2]
        tensor = torch.as_tensor(resized, device=self.device).permute(2, 0, 1)[None]
        self.features, _ = self.model.image_encoder(self.model.preprocess(tensor))

    def predict(self, point_coords, point_labels, multimask_output=False):
        import torch
        coords = self.transform.apply_coords(point_coords, self.original_size)
        points = (torch.as_tensor(coords, dtype=torch.float32, device=self.device)[None],
                  torch.as_tensor(point_labels, dtype=torch.int64, device=self.device)[None])
        sparse, dense = self.model.prompt_encoder(points=points, boxes=None, masks=None)
        masks, scores = self.model.mask_decoder(
            image_embeddings=self.features, image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            multimask_output=multimask_output)
        masks = self.model.postprocess_masks(masks, self.input_size, self.original_size)
        return (masks[0] > self.model.mask_threshold).cpu().numpy(), scores[0].cpu().numpy(), None


class PointModel:
    """One backend per session; reuse the last section embedding across edits.

    Inference always uses native raw pixels. Alignment affects display and prompt
    coordinates only, so toggling the viewer does not change model inputs.
    """

    def __init__(self, backend, checkpoint, *, model_type='vit_b', model_config=None,
                 decoder_checkpoint=None, device='auto'):
        if backend not in ('sam', 'hq_sam', 'sam2', 'saem2'):
            raise ValueError('Expected sam, hq_sam, sam2 or saem2 backend')
        if model_type not in ('vit_b', 'vit_l', 'vit_h'):
            raise ValueError('Expected a ViT-B/L/H model type')
        if backend == 'sam2' and not model_config:
            raise ValueError('SAM 2 requires a model config matching its checkpoint')
        if backend == 'saem2' and not decoder_checkpoint:
            raise ValueError('SAEM2 requires both HQ and trained decoder checkpoints')
        paths = {'checkpoint': checkpoint}
        if backend == 'saem2':
            paths['decoder_checkpoint'] = decoder_checkpoint
        self.paths = {}
        for key, path in paths.items():
            if path is None or not Path(path).is_file():
                raise FileNotFoundError(f'Local {key} missing: {path}')
            self.paths[key] = str(Path(path).resolve())
        from probe_em.device import resolve_device
        self.device = resolve_device(device)
        self.backend, self.model_type, self.model_config = backend, model_type, model_config
        self.predictor = None
        self.sam2_cache = None
        self.image_key = None
        self.metadata = {'backend': backend, 'device': self.device, 'model_type': model_type,
                         'model_config': model_config, 'weights': {},
                         'multimask_output': False, 'inference_space': 'original_native_xy'}

    def _load(self):
        for name, path in self.paths.items():
            self.metadata['weights'][name] = {'path': path, 'sha256': file_digest(path)}
        if self.backend == 'sam2':
            self.predictor = self._sam2().image()
        elif self.backend == 'saem2':
            from saem2.predictor import SAEM2Predictor
            self.predictor = SAEM2Predictor(self.paths['checkpoint'], self.paths['decoder_checkpoint'],
                                            self.model_type, self.device)
        elif self.backend == 'hq_sam':
            from repro.sam import load_predictor
            self.predictor = load_predictor(self.paths['checkpoint'], self.model_type, self.device)
        else:
            self.predictor = BaseSAMPredictor(self.paths['checkpoint'], self.model_type, self.device)

    def predict(self, image, points, labels):
        import torch
        from probe_em.device import inference_autocast
        key = (image.shape, hashlib.sha256(image.tobytes()).hexdigest())
        with torch.inference_mode(), inference_autocast(self.device):
            if self.predictor is None:
                self._load()
            if key != self.image_key:
                self.predictor.set_image(image)
                self.image_key = key
            prompts = dict(point_coords=np.asarray(points, np.float32), point_labels=np.asarray(labels, np.int32))
            if self.backend == 'saem2':
                result = self.predictor.predict(**prompts)
                return result['mask'], None
            masks, scores, _ = self.predictor.predict(**prompts, multimask_output=False)
            masks, scores = np.asarray(masks), np.asarray(scores).reshape(-1)
            if masks.shape != (1, *image.shape[:2]) or scores.shape != (1,) or not np.isfinite(scores).all():
                raise ValueError('Predictor returned an invalid single-mask result')
            return masks[0], float(scores[0])

    def _sam2(self):
        if self.backend != 'sam2':
            raise ValueError('Cross-section propagation requires --prompt-backend sam2')
        if self.sam2_cache is None:
            from probe_em.runtime import PredictorCache
            for name, path in self.paths.items():
                self.metadata['weights'][name] = {'path': path, 'sha256': file_digest(path)}
            self.sam2_cache = PredictorCache(self.paths['checkpoint'], self.model_config, self.device)
        return self.sam2_cache

    def propagate(self, video_path, prompts, *, cancel, progress=None):
        """Yield one native-video-grid mask per frame using real SAM2 memory.

        Prompts map frame indices to (XY points, labels). Each run has fresh video
        memory; model weights are reused. Track forward from the earliest prompt,
        then backward to the range start. Never duplicate the anchor output.
        """
        import torch
        from probe_em.device import inference_autocast
        from probe_em.propagation import check_cancelled
        state = None
        with torch.inference_mode(), inference_autocast(self.device):
            check_cancelled(cancel)
            predictor = self._sam2().video()
            try:
                check_cancelled(cancel)
                # Legacy SAM2 consolidates offloaded masks with CPU operations
                # after nonblocking device copies. Keep masks/memory on MPS to
                # avoid reading an unfinished MPS-to-CPU transfer.
                offload_state = torch.device(self.device).type != 'mps'
                state = predictor.init_state(video_path=str(video_path), offload_video_to_cpu=True,
                                             offload_state_to_cpu=offload_state, async_loading_frames=False)
                add_points = getattr(predictor, 'add_new_points_or_box', None)
                if add_points is None:
                    add_points = predictor.add_new_points
                for index, (points, labels) in sorted(prompts.items()):
                    check_cancelled(cancel)
                    if not 0 <= index < state['num_frames'] or not np.any(np.asarray(labels) == 1):
                        raise ValueError('Each prompted video section needs a positive point inside the range')
                    add_points(state, frame_idx=index, obj_id=1, points=np.asarray(points, np.float32),
                               labels=np.asarray(labels, np.int32))
                anchor = min(prompts)
                visited = set()
                for reverse, steps in ((False, state['num_frames']-1-anchor), (True, anchor)):
                    if reverse and not anchor:
                        continue
                    check_cancelled(cancel)
                    for index, ids, logits in predictor.propagate_in_video(
                            state, start_frame_idx=anchor, max_frame_num_to_track=steps, reverse=reverse):
                        check_cancelled(cancel)
                        if index in visited:
                            continue
                        if list(ids) != [1] or logits.shape != (1, 1, state['video_height'], state['video_width']):
                            raise ValueError('Unexpected SAM2 video object or mask geometry')
                        if not torch.isfinite(logits).all():
                            raise ValueError('SAM2 returned nonfinite video logits')
                        mask = (logits[0, 0] > 0).cpu().numpy()
                        visited.add(index)
                        if progress is not None:
                            progress(index, len(visited), state['num_frames'])
                        yield index, mask
                check_cancelled(cancel)
            finally:
                if state is not None:
                    predictor.reset_state(state)
