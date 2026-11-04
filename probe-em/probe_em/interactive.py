"""Per-section point prompts, explicit image/video inference and reviewed masks.

The source labels and Probe-EM graph are never changed by this controller.
All prompts and saved masks use absolute original voxel coordinates. The viewer
may display the original or registered frame, including during inference.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import uuid

import neuroglancer
import numpy as np

from probe_em.demo_alignment import map_points
from probe_em.alignment_field import field_slice_payload, warp_field_slice
from probe_em.meshes import MeshVolume
from probe_em.prompt_models import PointModel
from probe_em.propagation import PropagationCancelled, PropagationGrid, check_cancelled


POINT_LAYERS = {1: 'positive_points', 0: 'negative_points'}
LAYER_NAMES = (*POINT_LAYERS.values(), 'sam_preview', 'sam_accepted')
HELP = ('p/n: points; i: current slice; j/k: previous/next slice; b/e: range start/end; '
        'v: SAM2 propagate; x: stop; u: undo; c: new object; y: accept/save. ')


class InteractiveSession:
    def __init__(self, viewer, raw, offset, resolution, field, aligned_shape, aligned_offset,
                 mode, output_dir, model_options, *, mesh_isotropic=True, mesh_target_nm=None,
                 source=None, propagation_radius=4, propagation_max_slices=32):
        if raw.dtype != np.uint8 or raw.ndim != 3:
            raise ValueError('Interactive prediction requires a uint8 XYZ raw volume')
        self.model = PointModel(**model_options)
        self.viewer, self.raw, self.field, self.mode = viewer, raw, field, mode
        self.offset, self.resolution = np.asarray(offset), np.asarray(resolution)
        self.aligned_shape, self.aligned_offset = aligned_shape, aligned_offset
        self.output = Path(output_dir)
        self.source = str(Path(source).resolve()) if source else None
        self.mesh_options = dict(mesh_isotropic=mesh_isotropic, mesh_target_nm=mesh_target_nm)
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='probe-em-prompt')
        self.future = None
        self.preview = self.accepted = None
        self.edit_history = []
        self.generation = 0
        self.closed = False
        if propagation_radius < 1 or propagation_max_slices < 2:
            raise ValueError('Propagation radius must be >=1 and maximum slices >=2')
        self.propagation_max_slices = int(propagation_max_slices)
        initial_z = int(np.floor(viewer.state.position[2]))
        self.propagation_range = [max(int(self.offset[2]), initial_z-int(propagation_radius)),
                                  min(int(self.offset[2])+raw.shape[2]-1, initial_z+int(propagation_radius))]
        self.cancel_event = threading.Event()
        with viewer.txn() as state:
            for label, name in POINT_LAYERS.items():
                state.layers[name] = neuroglancer.LocalAnnotationLayer(
                    dimensions=state.dimensions, annotation_color='#00ff80' if label else '#ff4060')
            # Interactive review starts with raw visible through the new mask overlay.
            state.layers['probe_trace'].visible = False
            state.selected_layer.layer = POINT_LAYERS[1]
            self._center_section(state, int(np.floor(state.position[2])))
        callbacks = {'prompt-positive': lambda action: self.add_point(action, 1),
                     'prompt-negative': lambda action: self.add_point(action, 0),
                     'prompt-run': self.run, 'prompt-clear': self.clear,
                     'prompt-undo': self.undo, 'prompt-accept': self.accept,
                     'prompt-propagate': self.propagate, 'prompt-stop': self.stop,
                     'prompt-range-start': lambda action: self.set_range(action, 0),
                     'prompt-range-end': lambda action: self.set_range(action, 1),
                     'prompt-previous-section': lambda action: self.navigate(action, -1),
                     'prompt-next-section': lambda action: self.navigate(action, 1)}
        for name, callback in callbacks.items():
            needs_frame = name in ('prompt-positive', 'prompt-negative', 'prompt-run',
                                   'prompt-propagate', 'prompt-accept')
            viewer.actions.add(name, self._guard(callback, needs_frame))
        with viewer.config_state.txn() as state:
            for key, action in (('p', 'positive'), ('n', 'negative')):
                state.input_event_bindings.slice_view[f'key{key}'] = f'prompt-{action}'
            for key, action in (('i', 'run'), ('c', 'clear'), ('u', 'undo'), ('y', 'accept'),
                                ('v', 'propagate'), ('x', 'stop'), ('b', 'range-start'), ('e', 'range-end'),
                                ('j', 'previous-section'), ('k', 'next-section')):
                state.input_event_bindings.viewer[f'key{key}'] = f'prompt-{action}'
                # Slice-panel defaults take priority over viewer bindings.
                state.input_event_bindings.slice_view[f'key{key}'] = f'prompt-{action}'
        self.status('Add points on an XY section. I predicts that slice; V propagates with SAM2. ' + HELP)

    def _guard(self, callback, needs_frame=True):
        def invoke(action):
            try:
                with self.lock:
                    if not self.closed:
                        client = action.viewer_state
                        if (needs_frame and 'raw' in client.layers and client.layers['raw'].source[0].url
                                != self.viewer.state.layers['raw'].source[0].url):
                            raise ValueError('Display coordinates changed; wait for the viewer to refresh and retry')
                        return callback(action)
            except Exception as error:
                self.status(f'{type(error).__name__}: {error}. ' + HELP)
        return invoke

    def status(self, message):
        with self.viewer.config_state.txn() as state:
            state.status_messages['probe-em-interactive'] = (
                f'Propagation range: z={self.propagation_range[0]}..{self.propagation_range[1]} (inclusive). ' + message)

    def _current_z(self, action=None):
        state = self.viewer.state
        if action is not None and 'position' in action.viewer_state.to_json():
            state = action.viewer_state
        z = int(np.floor(state.position[2]))
        if not int(self.offset[2]) <= z < int(self.offset[2])+self.raw.shape[2]:
            raise ValueError('Navigate to an XY section inside the cached ROI')
        return z

    def navigate(self, action, step):
        # Relative navigation applies to the latest server position so rapid
        # key presses cannot all step from the same unacknowledged client frame.
        z = int(np.clip(self._current_z()+step, self.offset[2], self.offset[2]+self.raw.shape[2]-1))
        with self.viewer.txn() as state:
            position = np.asarray(state.position).copy()
            position[2] = z+.5
            state.position = position
        self.status(f'Current z={z}. Points on other slices are retained; add points here or use V. ' + HELP)

    def set_range(self, action, endpoint):
        self.propagation_range[endpoint] = self._current_z(action)
        self._invalidate()
        self.status('Range updated. B sets the lower z and E the upper z; then V. ' + HELP)

    def stop(self, _=None):
        self._invalidate()
        self.status('Stop requested; the current model step may finish before stopping. Saved results retained. ' + HELP)

    def _native(self, points, aligned=None):
        points = np.asarray(points, dtype=float).reshape(-1, 3).copy()
        if not np.isfinite(points).all():
            raise ValueError('Prompt coordinates must be finite')
        # Mouse positions select voxel cells; fields are defined at integer z.
        points[:, 2] = np.floor(points[:, 2])
        if (self.mode['aligned'] if aligned is None else aligned):
            points = map_points(self.field, points, inverse=True)
        local = points - self.offset
        if np.any(local < 0) or np.any(local >= self.raw.shape):
            raise ValueError('Prompt is outside valid raw data (possibly registration padding)')
        return points

    def _snapshot(self, state=None, *, z=None, all_sections=False):
        state = self.viewer.state if state is None else state
        z = self._current_z() if z is None and not all_sections else z
        records = []
        for label, name in POINT_LAYERS.items():
            for annotation in state.layers[name].annotations:
                if annotation.type != 'point':
                    raise ValueError('Only point annotations are supported')
                if all_sections or int(np.floor(annotation.point[2])) == z:
                    records.append((annotation.id, label, annotation.point))
        if not records or not any(label == 1 for _, label, _ in records):
            location = 'the propagation range' if all_sections else f'current section z={z}'
            raise ValueError(f'Add at least one positive point on {location}; other-section points are retained')
        if len(records) > 256:
            raise ValueError('Use at most 256 points per prediction')
        records.sort(key=lambda item: (item[1], item[0]))
        points = self._native([item[2] for item in records])
        labels = np.asarray([item[1] for item in records], dtype=np.int32)
        pixels = np.floor(points - self.offset).astype(int)
        positive = {tuple(pixel) for pixel in pixels[labels == 1]}
        if positive.intersection(tuple(pixel) for pixel in pixels[labels == 0]):
            raise ValueError('A pixel cannot have both positive and negative prompts')
        return {'ids': [item[0] for item in records], 'points': points, 'labels': labels,
                'z': z, 'scope': 'volume' if all_sections else 'image', 'generation': self.generation}

    @staticmethod
    def _same_prompts(current, snapshot):
        # Annotation positions are float32 in Neuroglancer. Allow a small
        # round-trip error when toggling at large absolute voxel offsets.
        return (current['generation'] == snapshot['generation'] and current['ids'] == snapshot['ids']
                and np.array_equal(current['labels'], snapshot['labels'])
                and np.allclose(current['points'], snapshot['points'], atol=.02, rtol=0))

    def _matches(self, snapshot):
        try:
            return self._same_prompts(self._snapshot(z=snapshot['z'],
                                                     all_sections=snapshot['scope'] == 'volume'), snapshot)
        except (ValueError, KeyError):
            return False

    def _check_client_prompts(self, action, snapshot):
        if action is not None and all(name in action.viewer_state.layers for name in POINT_LAYERS.values()):
            if not self._same_prompts(self._snapshot(action.viewer_state, z=snapshot['z'],
                                                     all_sections=snapshot['scope'] == 'volume'), snapshot):
                raise ValueError('Annotation edits are still synchronizing; wait and retry')

    def _invalidate(self):
        self.cancel_event.set()
        self.generation += 1
        self.preview = None
        with self.viewer.txn() as state:
            if 'sam_preview' in state.layers:
                del state.layers['sam_preview']

    def add_point(self, action, label):
        point = action.mouse_voxel_coordinates
        if point is None or len(point) != 3:
            raise ValueError('Hover over the target in an XY slice before adding a point')
        point = np.asarray(point, dtype=float).copy()
        # Neuroglancer picking may snap the mouse to a visible annotation from
        # another z. P/N target the currently viewed XY section, not that picked z.
        point[2] = self._current_z(action)
        native = self._native([point])[0]
        annotation_id = uuid.uuid4().hex
        with self.viewer.txn() as state:
            state.layers[POINT_LAYERS[label]].annotations.append(
                neuroglancer.PointAnnotation(id=annotation_id, point=point,
                                            description='positive' if label else 'negative'))
            self._center_section(state, int(native[2]))
        self.edit_history.append((POINT_LAYERS[label], annotation_id))
        self._invalidate()
        self.status(f'Added {"positive" if label else "negative"} point at z={int(native[2])}; other slices retained. ' + HELP)

    def clear(self, _=None):
        with self.viewer.txn() as state:
            for name in POINT_LAYERS.values():
                state.layers[name].annotations = []
        self.edit_history.clear()
        self._invalidate()
        self.status('Cleared prompts and preview; saved results retained. ' + HELP)

    def undo(self, _=None):
        if self.edit_history:
            name, annotation_id = self.edit_history.pop()
            with self.viewer.txn() as state:
                state.layers[name].annotations = [a for a in state.layers[name].annotations if a.id != annotation_id]
            self._invalidate()
        self.status('Undid last p/n point. Use the annotation tab for manual edits. ' + HELP)

    def run(self, action=None):
        if self.future is not None and not self.future.done():
            self.status('Prediction is running; edit points if needed, then rerun when it finishes.')
            return
        snapshot = self._snapshot(z=self._current_z(action))
        self._check_client_prompts(action, snapshot)
        self.cancel_event = threading.Event()
        self.preview = None
        with self.viewer.txn() as state:
            if 'sam_preview' in state.layers:
                del state.layers['sam_preview']
        self.status(f'Running {self.model.backend} on z={snapshot["z"]}; model loads on first run...')
        self.future = self.executor.submit(self._predict, snapshot)

    def _predict(self, snapshot):
        try:
            z = snapshot['z']
            image = np.repeat(self.raw[:, :, z-int(self.offset[2])].T[..., None], 3, axis=2)
            points = snapshot['points'][:, :2] - self.offset[:2]
            mask, score = self.model.predict(image, points, snapshot['labels'])
            mask = np.asarray(mask)
            if mask.shape != image.shape[:2] or not np.isin(mask, [0, 1]).all():
                raise ValueError('Predictor must return a binary mask in native YX coordinates')
            mask = mask.astype(bool)
            with self.lock:
                if self.closed:
                    return
                if not self._matches(snapshot):
                    self.status('Points changed during inference; discarded stale result. Press i to rerun.')
                    return
                folder = self.output / ('prediction_' + uuid.uuid4().hex)
                folder.mkdir(parents=True, exist_ok=False)
                metadata = dict(self.model.metadata, schema_version=1, scope='single_section_review',
                                created_at=datetime.now(timezone.utc).isoformat(), source=self.source,
                                resolution_xyz_nm=self.resolution.tolist(), absolute_z=z,
                                voxel_offset_xyz=[*map(int, self.offset[:2]), z], mask_axes='YX',
                                points_xyz_original=snapshot['points'].tolist(), point_labels=snapshot['labels'].tolist(),
                                raw_rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                                predicted_score=score, mask_pixels=int(mask.sum()),
                                display_alignment=self.field.stats() if self.field else None)
                np.savez_compressed(folder / 'prediction.npz', mask_yx=mask, raw_rgb_yxc=image,
                                    points_xyz=snapshot['points'], point_labels=snapshot['labels'],
                                    **field_slice_payload(self.field, z))
                (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
                self.preview = {'mask': mask, 'snapshot': snapshot, 'folder': folder}
                with self.viewer.txn() as state:
                    self._render_mask(state, 'sam_preview', self.preview, self.mode['aligned'])
                    self._center_section(state, z)
                self.status(f'Preview z={z}, {int(mask.sum())} pixels. y: accept/save; edit points then i: rerun. ' + HELP)
        except Exception as error:
            with self.lock:
                if not self.closed:
                    self.preview = None
                    self.status(f'Prediction failed: {type(error).__name__}: {error}. Press i to retry.')

    def propagate(self, action=None):
        if self.model.backend != 'sam2':
            raise ValueError('V propagation requires --prompt-backend sam2; I remains available for image models')
        if self.future is not None and not self.future.done():
            self.status('Prediction is running. X stops it; wait for the current step before rerunning.')
            return
        start, end = self.propagation_range
        if end-start+1 > self.propagation_max_slices:
            raise ValueError(f'Range exceeds {self.propagation_max_slices} slices; narrow B/E or set --propagate-max-slices')
        snapshot = self._snapshot(all_sections=True)
        self._check_client_prompts(action, snapshot)
        grid = PropagationGrid(self.raw, self.offset, start, end,
                               field=self.field if self.mode['aligned'] else None,
                               shape=self.aligned_shape, origin=self.aligned_offset)
        grid.prompts(snapshot['points'], snapshot['labels'])  # Validate before creating a job or loading SAM2.
        self._invalidate()
        snapshot['generation'] = self.generation
        self.cancel_event = threading.Event()
        self.status(f'Preparing SAM2 propagation on z={start}..{end}; X stops after the current step.')
        self.future = self.executor.submit(self._propagate, snapshot, grid, self.cancel_event)

    def _propagate(self, snapshot, grid, cancel):
        folder = self.output / ('propagation_' + uuid.uuid4().hex)
        metadata = dict(schema_version=2, scope='bounded_section_propagation', status='running',
                        created_at=datetime.now(timezone.utc).isoformat(), source=self.source,
                        resolution_xyz_nm=self.resolution.tolist(), z_range_inclusive=[grid.z_start, grid.z_end],
                        voxel_offset_xyz=[*map(int, self.offset[:2]), grid.z_start], mask_axes='ZYX',
                        inference_space='registered_xy' if grid.field is not None else 'original_native_xy',
                        inference_origin_xy=grid.origin.tolist(), inference_shape_xy=list(grid.shape),
                        frame_encoding='JPEG quality=100, subsampling=0; encoded frames retained',
                        points_xyz_original=snapshot['points'].tolist(), point_labels=snapshot['labels'].tolist(),
                        object_id=1, identity_scope='one user-selected object per C/new-object session',
                        display_alignment=self.field.stats() if self.field else None)

        def write_metadata():
            (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

        def progress(index, done, total):
            with self.lock:
                if not self.closed and not cancel.is_set():
                    self.status(f'SAM2 propagation {done}/{total}: z={grid.z_start+index}. X stops. '
                                'Display alignment may be toggled; inference space stays fixed for this run.')

        try:
            check_cancelled(cancel)
            folder.mkdir(parents=True, exist_ok=False)
            write_metadata()
            metadata['frames'] = grid.encode(folder/'frames', cancel)
            if self.field is not None:
                self.field.save(folder/'alignment.npz')
                metadata['alignment_field'] = 'alignment.npz'
            prompts = grid.prompts(snapshot['points'], snapshot['labels'])
            count = grid.z_end-grid.z_start+1
            masks = np.zeros((count, self.raw.shape[1], self.raw.shape[0]), bool)
            processed = np.zeros(count, bool)
            for index, mask in self.model.propagate(folder/'frames', prompts, cancel=cancel, progress=progress):
                check_cancelled(cancel)
                with self.lock:
                    if self.closed or not self._matches(snapshot):
                        raise PropagationCancelled('Points changed during propagation; discarded stale result')
                if not 0 <= index < count or processed[index]:
                    raise ValueError('Video predictor returned a duplicate or out-of-range frame')
                masks[index] = grid.native_mask(mask, grid.z_start+index)
                processed[index] = True
            check_cancelled(cancel)
            if not processed.all():
                raise ValueError('Video predictor did not return every requested section')
            with self.lock:
                if self.closed or not self._matches(snapshot):
                    raise PropagationCancelled('Points changed during propagation; discarded stale result')
                metadata.update(self.model.metadata)
                # Model-level metadata describes image prediction; override the video-specific entries.
                metadata.update(status='complete', inference_space='registered_xy' if grid.field else 'original_native_xy',
                                mask_pixels_per_z=[int(mask.sum()) for mask in masks],
                                empty_sections=[grid.z_start+i for i, mask in enumerate(masks) if not mask.any()],
                                directions=['increasing_z', 'decreasing_z'], memory_reset_per_run=True)
                np.savez_compressed(folder/'prediction.npz', mask_zyx=masks,
                                    frame_z=np.arange(grid.z_start, grid.z_end+1), processed=processed,
                                    points_xyz=snapshot['points'], point_labels=snapshot['labels'])
                write_metadata()
                self.preview = {'mask': masks, 'snapshot': snapshot, 'folder': folder, 'z_start': grid.z_start}
                with self.viewer.txn() as state:
                    self._render_mask(state, 'sam_preview', self.preview, self.mode['aligned'])
                empty = len(metadata['empty_sections'])
                self.status(f'Propagation preview: {count} slices, {empty} empty. Scroll or J/K to inspect; '
                            'Y accepts the full range. Add correction points and V to rerun. ' + HELP)
        except Exception as error:
            if folder.is_dir():
                metadata.update(status='cancelled' if isinstance(error, PropagationCancelled) else 'failed',
                                error=f'{type(error).__name__}: {error}')
                write_metadata()
            with self.lock:
                if not self.closed:
                    self.preview = None
                    self.status(f'Propagation {"stopped" if isinstance(error, PropagationCancelled) else "failed"}: '
                                f'{error}. V retries with the current points/range. ' + HELP)

    @staticmethod
    def _center_section(state, z):
        # A one-voxel-thick source may not intersect an integer boundary plane.
        # Keep navigation on the current section, but sample its voxel center.
        position = np.asarray(state.position, dtype=float).copy()
        if int(np.floor(position[2])) == z:
            position[2] = z + .5
            state.position = position

    def _render_mask(self, state, name, result, aligned):
        is_volume = result['snapshot']['scope'] == 'volume'
        z0 = result['z_start'] if is_volume else result['snapshot']['z']
        masks = result['mask'] if is_volume else result['mask'][None]
        output = []
        for index, mask in enumerate(masks):
            mask = mask.T.astype(np.uint8)
            if aligned:
                mask = warp_field_slice(self.field, mask, z0+index, self.offset[:2], self.aligned_offset[:2],
                                         self.aligned_shape[:2], seg=True)
            output.append(mask)
        offset = np.array([*self.offset[:2], z0])
        if aligned:
            offset[:2] = self.aligned_offset[:2]
        source = MeshVolume(np.stack(output, axis=2), dimensions=state.dimensions,
                            voxel_offset=offset, **self.mesh_options)
        state.layers[name] = neuroglancer.SegmentationLayer(
            source=source, segments=[1], segment_colors={'1': '#00ff80' if name == 'sam_accepted' else '#ffc040'},
            selected_alpha=.45, hover_highlight=False, pick=False)

    def transform_layers(self, state, aligned):
        """Called in the viewer's alignment transaction, with self.lock held."""
        for name in POINT_LAYERS.values():
            for annotation in state.layers[name].annotations:
                point = np.asarray(annotation.point).copy()
                point[2] = np.floor(point[2])
                annotation.point = map_points(self.field, [point], inverse=not aligned)[0]
        for name, result in (('sam_preview', self.preview), ('sam_accepted', self.accepted)):
            if result is not None:
                self._render_mask(state, name, result, aligned)

    def accept(self, action=None):
        if self.preview is None:
            raise ValueError('Run a prediction before accepting')
        if not self._matches(self.preview['snapshot']):
            self._invalidate()
            raise ValueError('Points changed after prediction; press i to rerun before accepting')
        self._check_client_prompts(action, self.preview['snapshot'])
        if not self.preview['mask'].any():
            raise ValueError('Empty masks cannot be accepted; adjust the prompts')
        path = self.preview['folder'] / 'accepted.json'
        if not path.exists():
            with path.open('x', encoding='utf-8') as stream:
                json.dump({'accepted_at': datetime.now(timezone.utc).isoformat(),
                           'prediction': 'prediction.npz', 'metadata': 'metadata.json'}, stream, indent=2)
        self.accepted = self.preview
        with self.viewer.txn() as state:
            self._render_mask(state, 'sam_accepted', self.accepted, self.mode['aligned'])
            del state.layers['sam_preview']
        self.preview = None
        self.status(f'Accepted mask saved to {path.parent}. Keep points to refine this object; C starts another. ' + HELP)

    def close(self):
        with self.lock:
            self.closed = True
            self.cancel_event.set()
        self.executor.shutdown(wait=False, cancel_futures=True)
