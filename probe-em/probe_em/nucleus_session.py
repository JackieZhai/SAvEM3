"""Positive-only nucleus review with one persistent center annotation layer."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time
import uuid

import neuroglancer
import numpy as np

from probe_em.meshes import MeshVolume
from probe_em.nuclei import NucleusOptions, trace_nucleus
from probe_em.nucleus_review import NucleusReview, export_union
from probe_em.propagation import PropagationCancelled, check_cancelled
from probe_em.review_objects import object_color, write_json
from probe_em.viewer_navigation import install_section_navigation
from probe_em.viewer_status import set_status


CENTERS = 'nucleus_centers'
HELP = ('P: nucleus center; V: propagate up/down; C: confirm nucleus and start next; '
        'U: remove active center; X: stop; D/F: slices; Y: accept/save all (two confirmations).')


class NucleusSession:
    object_label = 'nucleus'

    def __init__(self, viewer, raw, model, output, source, *, options=None, auto_run=False):
        self.viewer, self.raw, self.model = viewer, raw, model
        self.output = Path(output)
        self.options = options or NucleusOptions()
        self.auto_run = auto_run
        self.review_objects = NucleusReview(output, raw, source)
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='nucleus-review')
        self.future, self.preview, self.merge_pending = None, None, None
        self.confirmation_bindings = {}
        self.cancel_event = threading.Event()
        self.generation, self.closed = 0, False
        self.volumes = {}
        viewer.probe_em_interactive = self
        with viewer.txn() as state:
            for key, result in self.review_objects.objects.items():
                self._render_mask(state, f'nucleus_{key:03d}', result)
            self._render_centers(state)
            state.selected_layer.layer = CENTERS
            if self.review_objects.final is not None:
                self._render_union(state)
        self._save_centers_layer()
        with viewer.config_state.txn() as state:
            if 'overview-navigation' in state.status_messages:
                del state.status_messages['overview-navigation']
        callbacks = {'prompt-positive': self.add_center, 'prompt-negative': self.reject_negative,
                     'prompt-propagate': self.propagate, 'prompt-clear': self.new_object,
                     'prompt-undo': self.undo, 'prompt-stop': self.stop,
                     'prompt-accept': self.request_merge, 'prompt-merge-confirm': self.confirm_merge,
                     'prompt-merge-cancel': self.cancel_merge,
                     'prompt-previous-section': lambda a: self.navigate(a, -1),
                     'prompt-next-section': lambda a: self.navigate(a, 1)}
        for name, callback in callbacks.items():
            viewer.actions.add(name, self._guard(callback))
        with viewer.config_state.txn() as state:
            for scope in (state.input_event_bindings.viewer, state.input_event_bindings.slice_view):
                for key, action in [('p', 'positive'), ('n', 'negative'), ('v', 'propagate'), ('c', 'clear'),
                                    ('u', 'undo'), ('x', 'stop'), ('y', 'accept'),
                                    ('d', 'previous-section'), ('f', 'next-section')]:
                    scope[f'key{key}'] = 'prompt-'+action
        install_section_navigation(viewer, int(raw.offset[2]), int(raw.offset[2]+raw.shape[2]-1))
        self.status('Place a positive point at the center of one nucleus.')

    def status(self, message, controls=True):
        mode = 'Placing P starts propagation automatically. ' if self.auto_run else ''
        set_status(self.viewer, 'probe-em-interactive',
            f'Nucleus {self.review_objects.next_id}; {len(self.review_objects.objects)} saved. '
            f'32 x 32 x 30 nm; local realignment OFF. {mode}{message}' + (' '+HELP if controls else ''))

    def _guard(self, callback):
        def invoke(action):
            try:
                with self.lock:
                    if self.closed:
                        return
                    if self.merge_pending and callback not in (self.request_merge, self.confirm_merge, self.cancel_merge):
                        self._cancel_merge()
                    return callback(action)
            except Exception as error:
                self._cancel_merge()
                self.status(f'{type(error).__name__}: {error}')
        return invoke

    def _busy(self):
        return self.future is not None and not self.future.done()

    def _pending(self, state=None):
        state = self.viewer.state if state is None else state
        known = {r['center_id']: r for r in self.review_objects.objects.values()}
        pending = []
        for annotation in state.layers[CENTERS].annotations:
            if annotation.type != 'point':
                raise ValueError('Use only positive point annotations for nucleus centers')
            point = np.asarray(annotation.point, dtype=float).copy()
            point[2] = np.floor(point[2])
            if annotation.id in known:
                if not np.allclose(point, known[annotation.id]['center'], atol=.01, rtol=0):
                    raise ValueError('A saved center was moved; restore it before tracing another nucleus')
            else:
                self.review_objects.validate_center(point)
                pending.append(dict(id=annotation.id, point=point.tolist()))
        if len(pending) > 1:
            raise ValueError('Use one active nucleus center at a time; confirm each nucleus with C')
        return pending[0] if pending else None

    @staticmethod
    def _same(first, second):
        if first is None or second is None:
            return first is second
        return first['id'] == second['id'] and np.allclose(first['point'], second['point'], atol=.01, rtol=0)

    def _check_client(self, action):
        if action is not None and CENTERS in action.viewer_state.layers:
            if not self._same(self._pending(), self._pending(action.viewer_state)):
                raise ValueError('Center edits are still synchronizing; wait and retry')

    def _render_centers(self, state):
        annotations = [neuroglancer.PointAnnotation(id=r['center_id'], point=r['center'],
            description=f'Nucleus {key:03d} (saved)') for key, r in self.review_objects.objects.items()]
        if self.review_objects.active is not None:
            point = self.review_objects.active
            annotations.append(neuroglancer.PointAnnotation(id=point['id'], point=point['point'],
                description=f'Nucleus {self.review_objects.next_id:03d} (active)'))
        state.layers[CENTERS] = neuroglancer.LocalAnnotationLayer(
            dimensions=state.dimensions, annotation_color='#00ff80', annotations=annotations)

    def _save_centers_layer(self):
        write_json(self.output/'nucleus_centers.json', self.viewer.state.layers[CENTERS].layer.to_json())

    def _render_mask(self, state, name, result):
        source = MeshVolume(result['mask'].transpose(2, 1, 0).astype(np.uint8),
                            dimensions=state.dimensions, voxel_offset=result['offset'])
        self.volumes[name] = source
        state.layers[name] = neuroglancer.SegmentationLayer(source=source, segments=[1],
            segment_colors={'1': result.get('color', object_color(self.review_objects.next_id))},
            selected_alpha=.45, hover_highlight=False, pick=False)

    def _render_union(self, state):
        parents = self.review_objects.final['parents']
        state.layers['nuclei_final'] = neuroglancer.SegmentationLayer(
            source=[self.volumes[f'nucleus_{key:03d}'] for key in parents], segments=[1],
            segment_colors={'1': '#ffffff'}, selected_alpha=.45, hover_highlight=False, pick=False)
        for key in parents:
            state.layers[f'nucleus_{key:03d}'].visible = False

    def _invalidate(self):
        self.cancel_event.set()
        self.generation += 1
        self.preview = None
        self._cancel_merge()
        with self.viewer.txn() as state:
            if 'nucleus_preview' in state.layers:
                del state.layers['nucleus_preview']
        self.volumes.pop('nucleus_preview', None)

    def add_center(self, action):
        if self._busy():
            raise ValueError('Wait for propagation to finish, or press X to stop it')
        self._check_client(action)
        if self._pending() is not None:
            raise ValueError('Confirm the current nucleus with C, or remove its center with U')
        if action.mouse_voxel_coordinates is None:
            raise ValueError('Hover over an XY section before placing the nucleus center')
        center = np.asarray(action.mouse_voxel_coordinates, float).copy()
        center[2] = np.floor(self.viewer.state.position[2])
        self.review_objects.validate_center(center)
        self._invalidate()
        self.review_objects.save(active=dict(id=uuid.uuid4().hex, point=center.tolist()))
        with self.viewer.txn() as state:
            self._render_centers(state)
            if 'nuclei_final' in state.layers:
                state.layers['nuclei_final'].visible = False
            state.selected_layer.layer = CENTERS
        self._save_centers_layer()
        if self.auto_run:
            self.propagate()
        else:
            self.status('Center added. Press V to propagate above and below it.')

    def reject_negative(self, _=None):
        raise ValueError('Nuclei mode uses positive center points only')

    def undo(self, _=None):
        self._invalidate()
        self.review_objects.save(active=None)
        with self.viewer.txn() as state:
            self._render_centers(state)
        self._save_centers_layer()
        self.status('Active center removed. Saved nuclei and their centers are retained.')

    def stop(self, _=None):
        self._invalidate()
        self.status('Stop requested; the current model step may finish first. The center is retained.')

    def navigate(self, action, step):
        with self.viewer.txn() as state:
            position = np.asarray(state.position).copy()
            position[2] = np.clip(np.floor(position[2])+step, self.raw.offset[2],
                                   self.raw.offset[2]+self.raw.shape[2]-1)+.5
            state.position = position

    def propagate(self, action=None):
        if self._busy():
            raise ValueError('Propagation is running; X stops it')
        self._check_client(action)
        center = self._pending()
        if center is None:
            raise ValueError('Place a nucleus center with P before propagating')
        self._invalidate()
        self.review_objects.save(active=center)
        self._save_centers_layer()
        generation = self.generation
        self.cancel_event = threading.Event()
        self.future = self.executor.submit(self._predict, center, generation, self.cancel_event)

    def _predict(self, center, generation, cancel):
        folder = self.output/('nucleus_'+uuid.uuid4().hex)
        folder.mkdir()
        metadata = dict(schema_version=1, status='running', center=center,
                        created_at=datetime.now(timezone.utc).isoformat(),
                        context=self.review_objects.context)
        def progress(message):
            check_cancelled(cancel)
            with self.lock:
                if self.closed or self.generation != generation or not self._same(center, self._pending()):
                    raise PropagationCancelled('Center changed; stale nucleus result discarded')
                self.status(message+' X stops.')
        try:
            write_json(folder/'metadata.json', metadata)
            result = trace_nucleus(self.raw, self.model, center['point'], folder, self.options, cancel, progress)
            progress('Preparing nucleus preview.')
            with self.lock:
                check_cancelled(cancel)
                if self.closed or generation != self.generation or not self._same(center, self._pending()):
                    raise PropagationCancelled('Center changed before publishing the nucleus preview')
                result.update(folder=folder, center_id=center['id'], generation=generation)
                metadata.update(result['metadata'], model=self.model.metadata, status='complete')
                np.savez_compressed(folder/'prediction.npz', mask_zyx=result['mask'],
                    voxel_offset_xyz=result['offset'], resolution_xyz_nm=self.raw.resolution,
                    frame_z=np.arange(result['offset'][2], result['offset'][2]+len(result['mask'])),
                    points_xyz=result['center'][None], point_labels=np.array([1]))
                write_json(folder/'metadata.json', metadata)
                self.preview = result
                with self.viewer.txn() as state:
                    self._render_mask(state, 'nucleus_preview', result)
                warning = ('Preview is partial: '+', '.join(sorted({v['reason'] for v in result['metadata']['stops'].values()
                            if v['reason'] != 'empty'}))+'. Increase crop/radius or inspect the stopping section. '
                           if result['metadata']['partial'] else '')
                self.status(warning+f'Preview: {len(result["mask"])} sections. Inspect, then confirm with C.')
        except Exception as error:
            metadata.update(status='cancelled' if isinstance(error, PropagationCancelled) else 'failed',
                            error=f'{type(error).__name__}: {error}')
            write_json(folder/'metadata.json', metadata)
            with self.lock:
                if not self.closed and self.generation == generation:
                    self.preview = None
                    self.status(f'Nucleus propagation {metadata["status"]}: {error}')

    def _ready(self, action=None):
        if self._busy():
            raise ValueError('Wait for the current task before accepting')
        self._check_client(action)
        pending = self._pending()
        if pending is not None:
            if (self.preview is None or self.preview['generation'] != self.generation
                    or not self._same(pending, dict(id=self.preview['center_id'], point=self.preview['center']))):
                raise ValueError('Run V for the current center before confirming; the center is retained')
        elif self.preview is not None:
            raise ValueError('The preview center was removed; place a center and run V again')
        return pending

    def new_object(self, action=None):
        if self._ready(action) is None:
            raise ValueError('Place a nucleus center and run V before confirming')
        result = self.review_objects.accept(self.preview)
        self._invalidate()
        with self.viewer.txn() as state:
            self._render_mask(state, f'nucleus_{result["object_id"]:03d}', result)
            self._render_centers(state)
            state.selected_layer.layer = CENTERS
        self._save_centers_layer()
        self.status(f'Confirmed nucleus {result["object_id"]}. Place the next center; its mask uses a new color.')

    def _merge_signature(self, action=None):
        pending = self._ready(action)
        if not self.review_objects.objects and pending is None:
            raise ValueError('Trace at least one nucleus before accepting all')
        return (tuple(self.review_objects.objects), json.dumps(pending, sort_keys=True))

    def request_merge(self, action=None):
        signature = self._merge_signature(action)
        self._cancel_merge()
        self.merge_pending = dict(stage=1, signature=signature, expires=time.monotonic()+60)
        with self.viewer.config_state.txn() as state:
            for scope in ('viewer', 'slice_view'):
                bindings = getattr(state.input_event_bindings, scope)
                for key, value in [('enter', 'prompt-merge-confirm'), ('escape', 'prompt-merge-cancel')]:
                    self.confirmation_bindings[scope, key] = bindings.get(key)
                    bindings[key] = value
        self.status('Accept/save all nuclei as a binary union? Confirmation 1/2: Enter continues; Esc cancels.', False)

    def _cancel_merge(self):
        self.merge_pending = None
        if self.confirmation_bindings:
            with self.viewer.config_state.txn() as state:
                for (scope, key), previous in self.confirmation_bindings.items():
                    bindings = getattr(state.input_event_bindings, scope)
                    if previous is None:
                        if key in bindings:
                            del bindings[key]
                    else:
                        bindings[key] = previous
            self.confirmation_bindings.clear()

    def cancel_merge(self, _=None):
        self._cancel_merge()
        self.status('Acceptance cancelled. Nuclei and centers are retained.')

    def confirm_merge(self, action=None):
        pending = self.merge_pending
        if pending is None or time.monotonic() > pending['expires']:
            raise ValueError('Request acceptance again; no active confirmation remains')
        if pending['signature'] != self._merge_signature(action):
            raise ValueError('Nuclei or center changed; request acceptance again')
        if pending['stage'] == 1:
            pending['stage'] = 2
            self.status('Confirmation 2/2: Enter saves the union. All nucleus layers and centers are retained. Esc cancels.', False)
            return
        if self.preview is not None:
            self.new_object(action)
        self._cancel_merge()
        self.cancel_event = threading.Event()
        self.future = self.executor.submit(self._export, self.cancel_event)

    def _export(self, cancel):
        folder = self.output/('union_'+uuid.uuid4().hex)
        try:
            final = export_union(self.review_objects.objects, folder, cancel, self.status)
            with self.lock:
                check_cancelled(cancel)
                self.review_objects.save(final=final)
                with self.viewer.txn() as state:
                    self._render_union(state)
                self.status('Accepted/saved the binary union. Individual nuclei and the shared center layer are retained.')
        except Exception as error:
            with self.lock:
                if not self.closed:
                    self.status(f'Union export failed: {error}. Saved nuclei are retained.')

    def close(self):
        with self.lock:
            self.closed = True
            self.cancel_event.set()
            self._cancel_merge()
        self.executor.shutdown(wait=False, cancel_futures=True)
