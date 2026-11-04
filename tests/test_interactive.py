"""Prompt geometry, explicit triggering, stale-job rejection and local acceptance."""
import importlib.util
import json
from pathlib import Path
import sys
import threading

import numpy as np
import pytest

pytest.importorskip('neuroglancer')
pytest.importorskip('torch')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'probe-em'))
from probe_em.demo_alignment import map_points, translation
from probe_em.z_align import ZAffineField


class FakeModel:
    backend = 'test'
    metadata = {'backend': 'test'}

    def __init__(self, **_):
        self.calls = []
        self.release = None
        self.started = threading.Event()
        self.fail = False

    def predict(self, image, points, labels):
        self.calls.append((image.copy(), points.copy(), labels.copy()))
        self.started.set()
        if self.release is not None:
            assert self.release.wait(10)
        if self.fail:
            raise RuntimeError('test inference failure')
        mask = np.zeros(image.shape[:2], bool)
        for (x, y), label in zip(points.astype(int), labels):
            mask[y, x] = label
        return mask, .75


@pytest.fixture
def session(tmp_path, monkeypatch):
    import probe_em.interactive as interactive
    monkeypatch.setattr(interactive, 'PointModel', FakeModel)
    spec = importlib.util.spec_from_file_location('interactive_demo', ROOT / 'probe-em/scripts/demo.py')
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    raw = np.arange(24 * 20 * 3, dtype=np.uint8).reshape(24, 20, 3)
    labels = np.zeros_like(raw, dtype=np.uint64)
    labels[4:10, 4:10, :] = 2**63+7
    offset, resolution = np.array([100, 200, 300]), np.array([8, 8, 30])
    field = ZAffineField(300, 302, {300: translation([0, 0]), 301: translation([3, -2]),
                                   302: translation([-1, 2])}, 300, [100, 200, 124, 220])
    path = tmp_path / 'field.npz'
    field.save(path)
    monkeypatch.setattr(demo, 'load_local', lambda p, *a: (raw if Path(p).name == 'raw' else labels,
                                                         resolution, offset))
    viewer, report = demo.build_viewer(tmp_path, [], 2**63+7, tmp_path / 'review',
                                      align_field=path,
                                      interactive={'backend': 'test', 'output_dir': tmp_path / 'interactive'})
    controller = viewer.probe_em_interactive
    assert report['alignment']['enabled'] and report['interactive']['enabled']
    assert viewer.state.show_slices is False
    bindings = viewer.config_state.state.input_event_bindings
    assert bindings.viewer['keyd'] == bindings.slice_view['keyd'] == 'prompt-previous-section'
    assert bindings.viewer['keyf'] == bindings.slice_view['keyf'] == 'prompt-next-section'
    assert 'keyj' not in bindings.viewer and 'keyk' not in bindings.viewer
    assert viewer.state.position[2] % 1 == .5
    assert not viewer.state.layers['probe_trace'].visible
    yield controller
    if controller.model.release is not None:
        controller.model.release.set()
    controller.close()
    if controller.future is not None:
        controller.future.result(timeout=15)


def point(session, native, label=1):
    display = map_points(session.field, [native])[0] if session.mode['aligned'] else native
    with session.viewer.txn() as state:
        state.position = [display[0], display[1], np.floor(display[2])+.5]
    session.viewer.actions.invoke('prompt-positive' if label else 'prompt-negative', {'mousePosition': list(display)})


def action(session, name):
    session.viewer.actions.invoke(name, {})


def test_explicit_multi_point_run_alignment_toggle_and_accept(session):
    with session.viewer.txn() as state:
        state.position = [106, 207, 301]  # A boundary plane must become the slice center.
    point(session, [106, 207, 301])
    point(session, [110, 211, 301])
    point(session, [114, 214, 301], 0)
    assert not session.model.calls
    action(session, 'prompt-run')
    session.future.result(timeout=15)
    assert session.preview is not None
    assert session.viewer.state.position[2] == 301.5
    image, points, labels = session.model.calls[0]
    np.testing.assert_array_equal(image[..., 0], session.raw[:, :, 1].T)
    assert sorted((tuple(p), int(l)) for p, l in zip(points, labels)) == [((6, 7), 1), ((10, 11), 1), ((14, 14), 0)]
    original_mask = session.preview['mask'].copy()
    assert original_mask[7, 6] and original_mask[11, 10] and not original_mask[14, 14]
    action(session, 'toggle-alignment')
    assert session._matches(session.preview['snapshot'])
    np.testing.assert_array_equal(session.preview['mask'], original_mask)
    action(session, 'toggle-alignment')
    assert session._matches(session.preview['snapshot'])
    action(session, 'prompt-accept')
    folder = session.accepted['folder']
    assert session.preview is None and (folder / 'accepted.json').is_file()
    metadata = json.loads((folder / 'metadata.json').read_text())
    assert metadata['absolute_z'] == 301 and metadata['mask_axes'] == 'YX'
    with np.load(folder / 'prediction.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['mask_yx'], original_mask)
        np.testing.assert_array_equal(saved['display_matrix'], session.field.transform(301))
    action(session, 'prompt-clear')
    assert session.accepted is not None and (folder / 'accepted.json').is_file()


def test_bad_prompts_and_undo_do_not_call_model(session):
    action(session, 'prompt-run')
    assert session.future is None
    point(session, [106, 207, 301], 0)
    action(session, 'prompt-run')
    assert session.future is None
    point(session, [106, 207, 301])  # Contradictory pixel labels.
    action(session, 'prompt-run')
    assert session.future is None
    action(session, 'prompt-undo')
    assert len(session.viewer.state.layers['positive_points'].annotations) == 0
    point(session, [106, 207, 302])  # Other-section points are retained independently.
    assert len(session.viewer.state.layers['positive_points'].annotations) == 1
    point(session, [99, 205, 301])  # Padding maps outside original data.
    assert len(session.viewer.state.layers['positive_points'].annotations) == 1
    assert not session.model.calls


def test_switching_sections_predicts_current_points_without_clearing_previous(session):
    point(session, [106, 207, 301])
    action(session, 'prompt-run')
    session.future.result(timeout=15)
    action(session, 'prompt-next-section')
    assert session.viewer.state.position[2] == 302.5
    action(session, 'prompt-run')
    assert len(session.model.calls) == 1  # Never rerun the old section implicitly.
    assert 'current section z=302' in session.viewer.config_state.state.status_messages['probe-em-interactive']
    point(session, [109, 208, 302])
    action(session, 'prompt-run')
    session.future.result(timeout=15)
    assert len(session.viewer.state.layers['positive_points'].annotations) == 2
    assert session.preview['snapshot']['z'] == 302
    np.testing.assert_array_equal(session.model.calls[-1][0][..., 0], session.raw[:, :, 2].T)
    np.testing.assert_allclose(session.model.calls[-1][1], [[9, 8]])
    action(session, 'prompt-previous-section')
    action(session, 'prompt-run')
    session.future.result(timeout=15)
    assert session.preview['snapshot']['z'] == 301
    np.testing.assert_allclose(session.model.calls[-1][1], [[6, 7]])


def test_hovering_an_old_annotation_cannot_put_new_points_on_the_old_section(session):
    point(session, [106, 207, 301])
    action(session, 'prompt-next-section')
    session.viewer.actions.invoke('prompt-positive', {'mousePosition': [109, 205, 301]})
    assert [a.point[2] for a in session.viewer.state.layers['positive_points'].annotations] == [301, 302]
    action(session, 'prompt-run')
    session.future.result(timeout=15)
    assert session.preview['snapshot']['z'] == 302
    np.testing.assert_allclose(session.model.calls[-1][1], [[10, 3]])


def test_relative_navigation_and_stop_tolerate_pending_client_frame_updates(session):
    with session.viewer.txn() as state:
        state.position = [106, 207, 300.5]
    old_client = session.viewer.state.to_json()
    for _ in range(2):
        session.viewer.actions.invoke('prompt-next-section', {'viewerState': old_client})
    assert session.viewer.state.position[2] == 302.5
    action(session, 'toggle-alignment')
    session.cancel_event.clear()
    session.viewer.actions.invoke('prompt-stop', {'viewerState': old_client})
    assert session.cancel_event.is_set()


def test_clear_and_edits_during_prediction_discard_stale_result(session):
    session.model.release = threading.Event()
    point(session, [106, 207, 301])
    action(session, 'prompt-run')
    assert session.model.started.wait(10)
    first = session.future
    action(session, 'prompt-run')
    assert session.future is first  # No unbounded job queue.
    action(session, 'prompt-clear')
    session.model.release.set()
    first.result(timeout=15)
    assert session.preview is None and not list(session.output.glob('prediction_*'))
    assert 'stale' in session.viewer.config_state.state.status_messages['probe-em-interactive']


def test_direct_annotation_edit_prevents_accepting_old_mask(session):
    point(session, [106, 207, 301])
    action(session, 'prompt-run')
    session.future.result(timeout=15)
    folder = session.preview['folder']
    with session.viewer.txn() as state:
        state.layers['positive_points'].annotations[0].point = [111, 206, 301]
    action(session, 'prompt-accept')
    assert session.preview is None and not (folder / 'accepted.json').exists()


def test_inference_failure_is_recoverable_and_toggle_during_run_is_safe(session):
    session.model.fail = True
    point(session, [106, 207, 301])
    action(session, 'prompt-run')
    session.future.result(timeout=15)
    assert session.preview is None
    assert 'test inference failure' in session.viewer.config_state.state.status_messages['probe-em-interactive']
    session.model.fail = False
    session.model.release = threading.Event()
    session.model.started.clear()
    action(session, 'prompt-run')
    assert session.model.started.wait(10)
    action(session, 'toggle-alignment')
    session.model.release.set()
    session.future.result(timeout=15)
    assert session.preview is not None and session._matches(session.preview['snapshot'])


def test_stale_client_frame_and_unsynchronized_annotations_are_rejected(session):
    old_frame = session.viewer.state.to_json()
    action(session, 'toggle-alignment')
    session.viewer.actions.invoke('prompt-positive', {'mousePosition': [106, 207, 301], 'viewerState': old_frame})
    assert len(session.viewer.state.layers['positive_points'].annotations) == 0
    assert 'coordinates changed' in session.viewer.config_state.state.status_messages['probe-em-interactive']
    point(session, [106, 207, 301])
    edited_client = json.loads(json.dumps(session.viewer.state.to_json()))
    next(l for l in edited_client['layers'] if l['name'] == 'positive_points')['annotations'][0]['point'][0] += 2
    session.viewer.actions.invoke('prompt-run', {'viewerState': edited_client})
    assert session.future is None
    assert 'synchronizing' in session.viewer.config_state.state.status_messages['probe-em-interactive']
