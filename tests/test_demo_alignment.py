"""Registration geometry, lossless IDs, and synchronized Neuroglancer controls."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'probe-em'))
from probe_em.z_align import ZAffineField, _estimate_pair_affine, warp_slice
from probe_em.demo_alignment import (map_points, prepare_field, register_arrays, skeleton_lines,
                                      translation, validate_field)


def test_pair_estimation_has_correct_xy_direction():
    rng = np.random.default_rng(4)
    image = cv2.GaussianBlur(rng.uniform(0, 255, (96, 80)).astype(np.float32), (5, 5), 0)
    moved = warp_slice(image, translation([4, -3]))
    matrix = _estimate_pair_affine(image, moved)
    assert matrix is not None
    np.testing.assert_allclose(matrix[:2, 2], [-4, 3], atol=0.2)
    corrected = warp_slice(moved, matrix)
    assert np.abs(corrected[8:-8, 8:-8]-image[8:-8, 8:-8]).mean() < 1.0


def shifted_bundle():
    offset = np.array([100, 200, 300])
    raw = np.zeros((16, 16, 3), np.uint8)
    seg = np.zeros(raw.shape, np.uint64)
    matrices, vertices = {}, []
    label = 2**63+7
    for z, xy in enumerate(([6, 6], [9, 4], [4, 8])):
        raw[xy[0], xy[1], z] = 255
        seg[xy[0], xy[1], z] = label
        matrices[300+z] = translation(np.array([6, 6])-xy)
        vertices.append(offset + [*xy, z])
    field = ZAffineField(300, 302, matrices, 300, [100, 200, 116, 216])
    return raw, seg, offset, field, np.asarray(vertices)


def test_expanded_grid_and_skeleton_use_same_global_transform():
    raw, seg, offset, field, vertices = shifted_bundle()
    aligned_raw, aligned_seg, aligned_offset = register_arrays(raw, seg, offset, field)
    assert aligned_seg.dtype == np.uint64
    assert set(np.unique(aligned_seg)) == {0, 2**63+7}
    assert np.count_nonzero(aligned_seg) == 3
    np.testing.assert_array_equal(aligned_raw[aligned_seg > 0], 255)
    locations = np.argwhere(aligned_seg > 0) + aligned_offset
    np.testing.assert_array_equal(locations[:, :2], [[106, 206]]*3)
    mapped = map_points(field, vertices)
    np.testing.assert_allclose(mapped[:, :2], [[106, 206]]*3)
    np.testing.assert_allclose(map_points(field, mapped, inverse=True), vertices)
    skeleton = {'vertices': (vertices * [8, 8, 30]).tolist(), 'edges': [[0, 1], [1, 2]]}
    for _, a, b in skeleton_lines(skeleton, [8, 8, 30], field):
        np.testing.assert_allclose([a[:2], b[:2]], [[106, 206]]*2)
    with pytest.raises(ValueError, match='max-voxels'):
        register_arrays(raw, seg, offset, field, max_voxels=1)


def test_rotated_crop_conjugates_nonzero_origin():
    image = np.zeros((9, 9), np.uint64)
    image[2, 4] = 2**63+7
    local = np.vstack([cv2.getRotationMatrix2D((4, 4), 90, 1), [0, 0, 1]])
    origin = np.array([1000, 2000])
    global_matrix = translation(origin) @ local @ translation(-origin)
    field = ZAffineField(30, 30, {30: global_matrix}, 30, [1000, 2000, 1009, 2009])
    expected = warp_slice(image, local, seg=True)
    result = field.align_cutout(image, 30, seg=True, origin_xy=origin)
    np.testing.assert_array_equal(result, expected)


def test_field_cache_matches_data_and_configuration(tmp_path, monkeypatch):
    import probe_em.demo_alignment as alignment
    rng = np.random.default_rng(5)
    base = cv2.GaussianBlur(rng.integers(0, 255, (64, 64), dtype=np.uint8), (5, 5), 0)
    raw = np.stack([base, warp_slice(base, translation([1, 0])), base], axis=2)
    field, report = prepare_field(raw, [8, 9, 30], [100, 200, 300], cache_dir=tmp_path)
    assert report['source'] == 'estimated'
    monkeypatch.setattr(alignment, 'estimate_local_field', lambda *a, **k: pytest.fail('Matching cache was ignored'))
    cached, report = prepare_field(raw, [8, 9, 30], [100, 200, 300], cache_dir=tmp_path)
    assert report['source'] == 'cache'
    np.testing.assert_allclose(cached.transform(300), field.transform(300))
    with pytest.raises(ValueError, match='cover'):
        validate_field(cached, [100, 200, 299], 3)


def test_viewer_alignment_defaults_on_and_toggle_preserves_review(tmp_path, monkeypatch):
    pytest.importorskip('neuroglancer')
    spec = importlib.util.spec_from_file_location('aligned_demo', ROOT / 'probe-em/scripts/demo.py')
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    raw, seg, offset, field, vertices = shifted_bundle()
    resolution = np.array([8, 8, 30])
    monkeypatch.setattr(demo, 'load_local', lambda p, *a: (raw if Path(p).name == 'raw' else seg, resolution, offset))
    field_path = tmp_path / 'field.npz'
    field.save(field_path)
    label = 2**63+7
    (tmp_path / 'traced_skeletons.json').write_text(json.dumps({str(label): {
        'vertices': (vertices * resolution).tolist(), 'edges': [[0, 1], [1, 2]]}}))
    viewer, report = demo.build_viewer(tmp_path, [label], label, tmp_path / 'reviews', align_field=field_path)
    assert report['alignment']['enabled'] is True
    original_state = viewer.state.to_json()
    assert original_state['showSlices'] is False
    assert report['show_slices'] is False
    position = np.array(viewer.state.position)
    viewer.actions.invoke('toggle-alignment', {})
    assert 'Local alignment: OFF' in viewer.config_state.state.status_messages['probe-em']
    assert viewer.state.to_json()['layers'][0]['source'] != original_state['layers'][0]['source']
    viewer.actions.invoke('save-review', {})
    saved = json.loads(next((tmp_path / 'reviews').glob('*.json')).read_text())
    assert saved['display_space'] == 'original' and saved['segments'] == [str(label)]
    viewer.actions.invoke('toggle-alignment', {})
    assert 'Local alignment: ON' in viewer.config_state.state.status_messages['probe-em']
    np.testing.assert_allclose(viewer.state.position, position)
    assert viewer.state.to_json()['layers'][0]['source'] == original_state['layers'][0]['source']
    assert list(viewer.state.layers['manual_correction'].segments) == [label]
    assert viewer.state.show_slices is False

    monkeypatch.setattr('probe_em.demo_alignment.prepare_field', lambda *a, **k: pytest.fail('Disabled alignment was executed'))
    disabled, report = demo.build_viewer(tmp_path, [label], label, tmp_path / 'reviews', align_z=False)
    assert report['alignment']['enabled'] is False
    assert disabled.state.to_json()['showSlices'] is False
    disabled.actions.invoke('toggle-alignment', {})
    assert 'disabled at startup' in disabled.config_state.state.status_messages['probe-em']


def test_aligned_cloudvolume_keeps_channel_and_nonzero_bounds(tmp_path):
    pytest.importorskip('cloudvolume')
    spec = importlib.util.spec_from_file_location('bridge', ROOT / 'probe-em/scripts/export_savem3.py')
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    from probe_em.z_align import AlignedVolume
    raw, seg, offset, field, _ = shifted_bundle()
    volume = bridge.write_precomputed(tmp_path / 'seg', seg, [8, 8, 30], offset, segmentation=True)
    wrapped = AlignedVolume(volume, field)
    result = wrapped[100:116, 200:216, 301:302]
    assert result.shape == (16, 16, 1, 1) and result.dtype == np.uint64
    assert result[6, 6, 0, 0] == 2**63+7
