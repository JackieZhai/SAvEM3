"""Portable review restoration, coordinate mapping, and protection of existing work."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from test_interactive import ROOT, action, session
from probe_em.demo_alignment import map_points
from probe_em.review_objects import ReviewObjects, object_color, write_json
from probe_em.starter_session import restore_starter_view, seed_review


@pytest.fixture
def starter(session, tmp_path):
    source = tmp_path/'portable'
    folder = source/'object_001'
    folder.mkdir(parents=True)
    mask = np.zeros((2, 20, 24), dtype=bool)
    mask[:, 4:10, 4:10] = True
    np.savez_compressed(folder/'prediction.npz', mask_zyx=mask, frame_z=[301, 302])
    write_json(folder/'metadata.json', {'scope': 'volume'})
    review = ReviewObjects(source, session.raw.shape, session.offset, session.resolution)
    review.accept(dict(mask=mask, folder=folder, snapshot=dict(scope='volume'), z_start=301), [
        dict(id='positive-1', point=[106, 207, 301], label=1, description='inside'),
        dict(id='negative-1', point=[114, 214, 301], label=0, description='outside')])
    state = dict(schema_version=1, raw_sha256=hashlib.sha256(session.raw.tobytes()).hexdigest(),
                 propagation_range=[301, 302], position_native_xyz=[109, 210, 301.5],
                 active_annotations=[dict(id='next-positive', point=[112, 210, 302], label=1)],
                 view=dict(crossSectionScale=1.25, projectionScale=75),
                 layers={'sam_object_001': dict(visible=False, selectedAlpha=.25,
                                                segments=['1'], segmentColors={'1': object_color(1)})})
    write_json(source/'session.json', state)
    return source, state, mask


@pytest.mark.parametrize('aligned', [True, False])
def test_fresh_viewer_restores_native_results_points_and_next_object(starter, session, tmp_path, monkeypatch, aligned):
    source, saved, mask = starter
    spec = importlib.util.spec_from_file_location('starter_demo', ROOT/'probe-em/scripts/demo.py')
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    labels = np.full(session.raw.shape, 7, dtype=np.uint64)
    monkeypatch.setattr(demo, 'load_local', lambda path, *a: (
        session.raw if Path(path).name == 'raw' else labels, session.resolution, session.offset))
    field = tmp_path/'starter-field.npz'
    session.field.save(field)
    destination = tmp_path/'relocated'/'interactive'
    raw_source = tmp_path/'relocated'/'data'
    viewer, _ = demo.build_viewer(raw_source, [], 7, tmp_path/'corrections',
        align_z=aligned, align_field=field if aligned else None,
        interactive=dict(backend='test', output_dir=destination, starter_session=source))
    restored = viewer.probe_em_interactive
    try:
        assert restore_starter_view(restored, viewer.probe_em_starter_state)
        np.testing.assert_array_equal(restored.review_objects.objects[1]['mask'], mask)
        assert restored.review_objects.context['source'] == str(raw_source.resolve())
        assert restored.review_objects.next_id == 2
        assert restored.review_objects.color == object_color(2)
        assert restored.propagation_range == [301, 302]
        state = viewer.state
        assert not state.layers['sam_object_001'].visible
        assert state.layers['sam_object_001'].selected_alpha == .25
        assert state.layers['sam_object_001'].segment_colors['1'] == object_color(1)
        assert state.layers['positive_points'].annotation_color == object_color(2)
        assert state.cross_section_scale == 1.25 and state.projection_scale == 75
        for name, native in [('sam_object_001_positive_points', [106, 207, 301]),
                             ('sam_object_001_negative_points', [114, 214, 301]),
                             ('positive_points', [112, 210, 302])]:
            shown = map_points(restored.field, [native])[0] if aligned else native
            np.testing.assert_allclose(state.layers[name].annotations[0].point, shown, atol=1e-4)
        position = saved['position_native_xyz']
        if aligned:
            position = map_points(restored.field, [position])[0]
        np.testing.assert_allclose(state.position, position, atol=1e-4)
        assert viewer.config_state.state.status_messages['probe-em-interactive'].count('Y:') == 1
        action(restored, 'prompt-next-section')
        action(restored, 'prompt-run')
        restored.future.result(timeout=15)
        action(restored, 'prompt-clear')
        assert set(restored.review_objects.objects) == {1, 2}
        assert restored.review_objects.next_id == 3
        assert 'sam_object_002_negative_points' not in viewer.state.layers
        np.testing.assert_array_equal(restored.review_objects.objects[1]['mask'], mask)
    finally:
        restored.close()


def test_installation_never_replaces_existing_review_files(starter, session, tmp_path):
    source, state, _ = starter
    output = tmp_path/'review'
    assert seed_review(source, output, session.raw, session.offset, session.resolution, tmp_path) == state
    (output/'review_objects.json').write_text('user-edited review')
    prediction = output/'object_001'/'prediction.npz'
    prediction.unlink()
    # A subsequent startup must neither reseed deleted objects nor restore stale files.
    assert seed_review(source, output, session.raw, session.offset, session.resolution, tmp_path) is None
    assert (output/'review_objects.json').read_text() == 'user-edited review'
    assert not prediction.exists()
    assert not restore_starter_view(session, None)


@pytest.mark.parametrize('damage', ['raw', 'grid', 'mask', 'annotation', 'range', 'outside-path'])
def test_invalid_starter_leaves_output_untouched(starter, session, tmp_path, damage):
    source, state, _ = starter
    raw = session.raw.copy()
    if damage == 'raw':
        raw[0, 0, 0] ^= 1
    elif damage in ('grid', 'mask', 'outside-path'):
        manifest_path = source/'review_objects.json'
        manifest = json.loads(manifest_path.read_text())
        if damage == 'grid':
            manifest['context']['voxel_offset_xyz'][0] += 1
        elif damage == 'mask':
            manifest['objects'][0]['mask_sha256'] = '0'*64
        else:
            manifest['objects'][0]['prediction_dir'] = '../outside'
        write_json(manifest_path, manifest)
    else:
        if damage == 'annotation':
            state['active_annotations'][0]['point'][0] = 9999
        else:
            state['propagation_range'] = [300, 303]
        write_json(source/'session.json', state)
    output = tmp_path/'uncreated'/'review'
    with pytest.raises(ValueError):
        seed_review(source, output, raw, session.offset, session.resolution, tmp_path)
    assert not output.exists()
