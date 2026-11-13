"""Object isolation, retained annotations and explicitly confirmed native-mask unions."""
import json

import numpy as np
from test_interactive import action, enable_refinement, point, session
from test_interactive_propagation import video_stub
from probe_em.demo_alignment import map_points
from probe_em.review_objects import ReviewObjects, layer_name


def predict(session, native, label=1):
    point(session, native, label)
    action(session, 'prompt-run')
    session.future.result(timeout=15)
    assert session.preview is not None
    return session.preview


def status(session):
    return session.viewer.config_state.state.status_messages['probe-em-interactive']


def test_new_object_keeps_masks_and_both_point_layers_with_independent_history(session):
    inputs = enable_refinement(session)
    point(session, [114, 214, 301], 0)
    first = predict(session, [106, 207, 301])
    original_ids = {a.id for name in ('positive_points', 'negative_points')
                    for a in session.viewer.state.layers[name].annotations}
    first_color = session.viewer.state.layers['sam_preview'].segment_colors['1']
    assert status(session).count('Y:') == 1
    action(session, 'prompt-clear')
    assert len(session.review_objects.objects) == 1
    state = session.viewer.state
    assert state.layers['sam_object_001'].segment_colors['1'] == first_color
    kept_ids = {a.id for name in ('sam_object_001_positive_points', 'sam_object_001_negative_points')
                for a in state.layers[name].annotations}
    assert kept_ids == original_ids
    assert not list(state.layers['positive_points'].annotations)
    assert not list(state.layers['negative_points'].annotations)
    assert not session.image_history
    second = predict(session, [112, 210, 301])
    assert inputs[-1] is None and session.model.calls[-1][1].shape == (1, 2)
    assert session.viewer.state.layers['sam_preview'].segment_colors['1'] != first_color
    np.testing.assert_array_equal(session.review_objects.objects[1]['mask'], first['mask'])
    action(session, 'prompt-clear')
    assert set(session.review_objects.objects) == {1, 2}
    assert session.review_objects.next_id == 3
    assert 'sam_object_002_negative_points' not in session.viewer.state.layers
    for _ in range(2):
        action(session, 'toggle-alignment')
        for object_id, expected in ((1, first), (2, second)):
            np.testing.assert_array_equal(session.review_objects.objects[object_id]['mask'], expected['mask'])
            annotation = session.viewer.state.layers[f'{layer_name(object_id)}_positive_points'].annotations[0]
            native = [106, 207, 301] if object_id == 1 else [112, 210, 301]
            shown = map_points(session.field, [native])[0] if session.mode['aligned'] else native
            np.testing.assert_allclose(annotation.point, shown, atol=.02)


def test_final_acceptance_requires_two_confirmations_and_keeps_every_object(session, monkeypatch):
    first = predict(session, [106, 207, 300])
    action(session, 'prompt-clear')
    video_stub(session, monkeypatch)
    point(session, [112, 210, 301])
    point(session, [114, 214, 301], 0)
    action(session, 'prompt-propagate')
    session.future.result(timeout=15)
    second = session.preview
    assert status(session).count('Y:') == 1
    action(session, 'prompt-accept')
    assert session.merge_pending['stage'] == 1 and not list(session.output.glob('merge_*'))
    action(session, 'prompt-merge-confirm')
    assert session.merge_pending['stage'] == 2 and len(session.review_objects.objects) == 1
    action(session, 'prompt-merge-cancel')
    assert session.preview is second and not list(session.output.glob('merge_*'))
    assert len(session.viewer.state.layers['positive_points'].annotations) == 1
    action(session, 'prompt-accept')
    action(session, 'prompt-merge-confirm')
    action(session, 'prompt-merge-confirm')
    assert session.merge_pending is None and session.preview is None
    assert set(session.review_objects.objects) == {1, 2}
    final = session.review_objects.final
    expected = second['mask'].copy()
    expected[0] |= first['mask']
    np.testing.assert_array_equal(final['mask'], expected)
    with np.load(final['folder']/'prediction.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['mask_zyx'], expected)
        np.testing.assert_array_equal(saved['frame_z'], [300, 301, 302])
    metadata = json.loads((final['folder']/'metadata.json').read_text())
    assert metadata['confirmations'] == 2 and metadata['operation'] == 'binary_union'
    assert [p['object_id'] for p in metadata['parents']] == [1, 2]
    for object_id in (1, 2):
        assert layer_name(object_id) in session.viewer.state.layers
        assert (session.review_objects.objects[object_id]['folder']/'accepted.json').is_file()
        assert session.viewer.state.layers[f'{layer_name(object_id)}_positive_points'].annotations
    assert len(session.viewer.state.layers['sam_object_002_negative_points'].annotations) == 1
    assert 'sam_object_001_negative_points' not in session.viewer.state.layers
    assert session.viewer.state.layers['sam_final'].visible
    assert 'enter' not in session.viewer.config_state.state.input_event_bindings.viewer
    action(session, 'prompt-next-section')
    assert session.viewer.state.layers['sam_final'].visible
    restored = ReviewObjects(session.output, session.raw.shape, session.offset, session.resolution, session.source)
    assert restored.next_id == 3 and set(restored.objects) == {1, 2}
    np.testing.assert_array_equal(restored.final['mask'], expected)
    assert restored.objects[2]['annotations'] == session.review_objects.objects[2]['annotations']
    import neuroglancer
    from probe_em.interactive import InteractiveSession
    viewer = neuroglancer.Viewer()
    with viewer.txn() as state:
        state.dimensions = session.viewer.state.dimensions
        state.position = session.viewer.state.position
        state.layers['probe_trace'] = neuroglancer.SegmentationLayer()
    reopened = InteractiveSession(viewer, session.raw, session.offset, session.resolution,
        session.field, session.aligned_shape, session.aligned_offset, session.mode.copy(),
        session.output, {'backend': 'test'}, source=session.source)
    try:
        assert 'sam_object_001_positive_points' in viewer.state.layers
        assert 'sam_object_001_negative_points' not in viewer.state.layers
        assert len(viewer.state.layers['sam_object_002_negative_points'].annotations) == 1
        assert 'negative_points' in viewer.state.layers
        np.testing.assert_array_equal(reopened.review_objects.final['mask'], expected)
    finally:
        reopened.close()


def test_single_object_can_be_finally_saved_and_changed_points_cancel_confirmation(session):
    preview = predict(session, [106, 207, 301])
    action(session, 'prompt-accept')
    action(session, 'prompt-merge-confirm')
    with session.viewer.txn() as state:
        state.layers['positive_points'].annotations[0].point = [111, 206, 301]
    action(session, 'prompt-merge-confirm')
    assert session.merge_pending is None and session.review_objects.final is None
    assert not (preview['folder']/'accepted.json').exists()
    assert session.viewer.state.layers['positive_points'].annotations
    action(session, 'prompt-run'); session.future.result(timeout=15)
    action(session, 'prompt-accept')
    session.merge_pending['expires'] = 0
    action(session, 'prompt-merge-confirm')
    assert session.merge_pending is None and 'expired' in status(session)
    action(session, 'prompt-accept')
    action(session, 'prompt-merge-confirm')
    action(session, 'prompt-merge-confirm')
    assert len(session.review_objects.objects) == 1 and session.review_objects.final is not None
    assert 'sam_object_001_negative_points' not in session.viewer.state.layers


def test_unpredicted_points_are_never_discarded_by_new_object_or_accept(session):
    point(session, [106, 207, 301])
    for operation in ('prompt-clear', 'prompt-accept'):
        action(session, operation)
        assert len(session.viewer.state.layers['positive_points'].annotations) == 1
        assert not session.review_objects.objects and session.review_objects.next_id == 1
        assert 'Run I or V' in status(session)


def test_failed_save_preserves_active_points_and_preview(session, monkeypatch):
    preview = predict(session, [106, 207, 301])
    def failed(*args, **kwargs):
        raise OSError('output unavailable')
    monkeypatch.setattr(session.review_objects, '_persist', failed)
    action(session, 'prompt-clear')
    assert session.preview is preview and len(session.viewer.state.layers['positive_points'].annotations) == 1
    assert not session.review_objects.objects and 'output unavailable' in status(session)
