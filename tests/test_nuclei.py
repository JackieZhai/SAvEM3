"""Center-outward tracking, bounded reads, positive-only review, and sparse unions."""
import json
from pathlib import Path
import sys
import threading

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'probe-em'))
from probe_em.nuclei import NucleusOptions, center_component, trace_nucleus
from probe_em.nucleus_review import NucleusReview, export_union
from probe_em.propagation import PropagationCancelled


class Raw:
    offset = np.array([48, 72, 10])
    shape = np.array([96, 80, 45])
    resolution = np.array([32, 32, 30])

    def __init__(self):
        self.calls = []
        self.data = np.broadcast_to(np.arange(10, 55, dtype=np.uint8), self.shape).copy()

    def read(self, low, high):
        low, high = np.array(low), np.array(high)
        assert np.all(low >= self.offset) and np.all(high <= self.offset+self.shape)
        self.calls.append((low.copy(), high.copy()))
        return self.data[tuple(slice(a, b) for a, b in zip(low-self.offset, high-self.offset))].copy()


class Model:
    backend = 'sam2'
    metadata = {'backend': 'test-sam2'}

    def __init__(self):
        self.calls, self.closed = [], 0
        self.started, self.release = threading.Event(), None
        self.seed_extent = 3

    def predict_refined(self, image, points, labels):
        assert list(labels) == [1] and points.shape == (1, 2)
        x, y = np.floor(points[0]).astype(int)
        mask = np.zeros(image.shape[:2], bool)
        r = self.seed_extent
        mask[max(0, y-r):y+r+1, max(0, x-r):x+r+1] = True
        self.started.set()
        if self.release is not None:
            assert self.release.wait(15)
        return dict(mask=mask, score=.9)

    def propagate(self, folder, prompts, *, cancel, mask_inputs):
        assert prompts == {} and len(mask_inputs) == 1
        frames = sorted(Path(folder).glob('*.jpg'))
        sections = [int(np.asarray(Image.open(path))[0, 0, 0]) for path in frames]
        anchor, previous = next(iter(mask_inputs.items()))
        y, x = np.argwhere(previous).mean(axis=0).astype(int)
        self.calls.append(dict(sections=sections, anchor_z=sections[anchor]))
        try:
            for index in [anchor, *range(anchor+1, len(frames)), *range(anchor-1, -1, -1)]:
                mask = np.zeros(previous.shape, bool)
                if 17 <= sections[index] <= 27:
                    mask[y-3:y+4, x-3:x+4] = True
                    mask[1:3, 1:3] = True  # Unrelated component must not become another nucleus.
                yield index, mask
        finally:
            self.closed += 1


def test_center_outward_blocks_stop_at_both_ends_and_keep_native_32nm_geometry(tmp_path):
    raw, model = Raw(), Model()
    result = trace_nucleus(raw, model, [90, 114, 22], tmp_path,
        NucleusOptions(box_size=48, max_radius=20, block_slices=4), threading.Event(), lambda _: None)
    np.testing.assert_array_equal(result['offset'], [87, 111, 17])
    assert result['mask'].shape == (11, 7, 7) and result['mask'].all()
    assert {stop['reason'] for stop in result['metadata']['stops'].values()} == {'empty'}
    assert not result['metadata']['partial']
    assert any(call['anchor_z'] < 22 for call in model.calls)
    assert any(call['anchor_z'] > 22 for call in model.calls)
    assert max(high[2]-low[2] for low, high in raw.calls) <= 5
    assert max((high-low).prod() for low, high in raw.calls) <= 48*48*5
    assert model.closed == len(model.calls)
    assert not list(tmp_path.glob('video-*'))


def test_radius_limits_are_reported_and_empty_block_boundary_terminates(tmp_path):
    # z=28 is the first empty section: reaching it must not restart forever from z=27.
    result = trace_nucleus(Raw(), Model(), [90, 114, 22], tmp_path,
        NucleusOptions(box_size=48, max_radius=6, block_slices=3), threading.Event(), lambda _: None)
    assert result['metadata']['partial']
    assert all(stop['reason'] == 'radius_limit' for stop in result['metadata']['stops'].values())
    assert result['mask'].shape[0] == 11


def test_center_component_excludes_other_objects_and_rejects_missing_center():
    mask = np.zeros((20, 20), bool)
    mask[3:6, 3:6] = True
    mask[10:15, 10:15] = True
    selected = center_component(mask, point=[4, 4])
    assert selected.sum() == 9
    assert center_component(mask, previous=selected).sum() == 9
    assert not center_component(mask, point=[0, 0]).any()


@pytest.mark.parametrize('bad', ['center', 'crop', 'missing-data', 'cancel'])
def test_invalid_or_cancelled_nuclei_never_publish_a_result(tmp_path, bad):
    raw, model, cancel = Raw(), Model(), threading.Event()
    center = [90, 114, 22]
    if bad == 'center':
        center[0] = 0
    elif bad == 'crop':
        model.seed_extent = 30
    elif bad == 'missing-data':
        raw.read = lambda *args: (_ for _ in ()).throw(FileNotFoundError('missing raw chunk'))
    else:
        cancel.set()
    expected = {'center': ValueError, 'crop': ValueError, 'missing-data': FileNotFoundError, 'cancel': PropagationCancelled}[bad]
    with pytest.raises(expected):
        trace_nucleus(raw, model, center, tmp_path, NucleusOptions(box_size=48), cancel, lambda _: None)
    assert not model.calls


def make_session(raw, model, output, **kwargs):
    import neuroglancer
    from probe_em.nucleus_session import NucleusSession
    viewer = neuroglancer.Viewer()
    with viewer.txn() as state:
        state.dimensions = neuroglancer.CoordinateSpace(names=['x', 'y', 'z'], units='nm', scales=raw.resolution)
        state.position = [90, 114, 22.5]
        state.layers['raw'] = neuroglancer.ImageLayer(source=neuroglancer.LocalVolume(
            raw.data, dimensions=state.dimensions, voxel_offset=raw.offset))
    return NucleusSession(viewer, raw, model, output, 'fixture-32nm',
        options=NucleusOptions(box_size=48, max_radius=20, block_slices=4), **kwargs)


@pytest.fixture
def session(tmp_path):
    value = make_session(Raw(), Model(), tmp_path/'review')
    yield value
    if value.model.release is not None:
        value.model.release.set()
    value.close()
    if value.future is not None:
        value.future.result(timeout=15)


def action(session, name, **data):
    session.viewer.actions.invoke('prompt-'+name, data)


def add_and_predict(session, point):
    action(session, 'positive', mousePosition=point)
    action(session, 'propagate')
    session.future.result(timeout=15)
    assert session.preview is not None


def test_two_nuclei_share_one_center_layer_and_reload_with_distinct_colors(session, tmp_path):
    from probe_em.nucleus_session import CENTERS
    add_and_predict(session, [90, 114, 22])
    first = session.preview['mask'].copy()
    action(session, 'clear')
    add_and_predict(session, [104, 114, 22])
    action(session, 'clear')
    state = session.viewer.state
    assert [layer.name for layer in state.layers if layer.type == 'annotation'] == [CENTERS]
    assert len(state.layers[CENTERS].annotations) == 2
    assert state.layers['nucleus_001'].segment_colors['1'] != state.layers['nucleus_002'].segment_colors['1']
    assert session.review_objects.next_id == 3 and session.review_objects.active is None
    centers = json.loads((session.output/'nucleus_centers.json').read_text())
    assert len(centers['annotations']) == 2
    restored = make_session(session.raw, Model(), session.output)
    try:
        assert restored.review_objects.next_id == 3
        np.testing.assert_array_equal(restored.review_objects.objects[1]['mask'], first)
        assert len(restored.viewer.state.layers[CENTERS].annotations) == 2
    finally:
        restored.close()
    action(session, 'negative', mousePosition=[95, 114, 22])
    assert 'negative_points' not in session.viewer.state.layers
    assert 'positive center points only' in session.viewer.config_state.state.status_messages['probe-em-interactive']


def test_two_confirmations_export_a_true_sparse_union_and_keep_centers(session):
    from cloudvolume import CloudVolume
    from probe_em.nucleus_session import CENTERS
    for point in ([90, 114, 22], [94, 114, 22]):
        add_and_predict(session, point)
        action(session, 'clear')
    action(session, 'accept')
    action(session, 'merge-confirm')
    assert session.review_objects.final is None and session.merge_pending['stage'] == 2
    action(session, 'merge-cancel')
    assert session.review_objects.final is None
    action(session, 'accept')
    action(session, 'merge-confirm')
    action(session, 'merge-confirm')
    session.future.result(timeout=15)
    final = session.review_objects.final
    assert final is not None
    volume = CloudVolume((session.output/final['path']/'precomputed').as_uri(), fill_missing=True, progress=False)
    mask = np.asarray(volume[volume.bounds])[..., 0]
    assert mask.shape == (11, 7, 11) and mask.sum() == 11*7*11
    assert session.viewer.state.layers['nuclei_final'].visible
    assert len(session.viewer.state.layers[CENTERS].annotations) == 2
    assert not session.viewer.state.layers['nucleus_001'].visible
    restored = NucleusReview(session.output, session.raw, 'fixture-32nm')
    assert restored.final['parents'] == [1, 2]


def test_unpredicted_center_survives_acceptance_and_restart(session):
    from probe_em.nucleus_session import CENTERS
    action(session, 'positive', mousePosition=[90, 114, 22])
    action(session, 'clear')
    action(session, 'accept')
    assert not session.review_objects.objects
    assert session.review_objects.active is not None
    restored = make_session(session.raw, Model(), session.output)
    try:
        assert len(restored.viewer.state.layers[CENTERS].annotations) == 1
        action(restored, 'undo')
        assert not restored.viewer.state.layers[CENTERS].annotations
    finally:
        restored.close()


def test_editing_center_during_prediction_discards_stale_result(session):
    from probe_em.nucleus_session import CENTERS
    session.model.release = threading.Event()
    action(session, 'positive', mousePosition=[90, 114, 22])
    action(session, 'propagate')
    assert session.model.started.wait(5)
    with session.viewer.txn() as state:
        state.layers[CENTERS].annotations[0].point = [92, 114, 22]
    session.model.release.set()
    session.future.result(timeout=15)
    assert session.preview is None and not session.review_objects.objects
    assert not list(session.output.glob('nucleus_*/accepted.json'))


def test_optional_automatic_propagation_starts_on_center_placement(tmp_path):
    value = make_session(Raw(), Model(), tmp_path/'auto', auto_run=True)
    try:
        action(value, 'positive', mousePosition=[90, 114, 22])
        assert value.future is not None
        value.future.result(timeout=15)
        assert value.preview is not None
    finally:
        value.close()


def test_union_keeps_empty_space_sparse_and_preserves_nonzero_offsets(tmp_path):
    from cloudvolume import CloudVolume
    objects = {1: dict(mask=np.ones((3, 3, 3), bool), offset=np.array([48, 72, 10])),
               2: dict(mask=np.ones((3, 3, 3), bool), offset=np.array([100048, 72, 10]))}
    folder = tmp_path/'union'
    export_union(objects, folder, threading.Event(), lambda _: None)
    assert len(list((folder/'precomputed/32_32_30').iterdir())) == 2
    metadata = json.loads((folder/'metadata.json').read_text())
    assert metadata['mask_voxels'] == 54
    volume = CloudVolume((folder/'precomputed').as_uri(), fill_missing=True, progress=False)
    assert np.asarray(volume[48:51, 72:75, 10:13]).sum() == 27
    assert np.asarray(volume[200:203, 72:75, 10:13]).sum() == 0
    assert np.asarray(volume[100048:100051, 72:75, 10:13]).sum() == 27


def test_corrupt_saved_mask_is_rejected_without_overwriting_review(session):
    add_and_predict(session, [90, 114, 22])
    action(session, 'clear')
    manifest = (session.output/'nuclei.json').read_bytes()
    path = session.review_objects.objects[1]['folder']/'prediction.npz'
    with np.load(path, allow_pickle=False) as value:
        arrays = {key: value[key].copy() for key in value.files}
    arrays['mask_zyx'][0, 0, 0] = False
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match='checksum'):
        NucleusReview(session.output, session.raw, 'fixture-32nm')
    assert (session.output/'nuclei.json').read_bytes() == manifest
