"""Elastic geometry and opt-in real SOFIMA flow/mesh integration."""
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'probe-em'))
from probe_em.alignment_field import ZElasticField, deformation_limits, load_field, warp_field_slice
from probe_em.demo_alignment import map_points, prepare_field, register_arrays, skeleton_lines
from probe_em.z_align import AlignedVolume, ZAffineField


def elastic_field():
    y, x = np.indices((5, 5), dtype=float)
    residual = np.zeros((3, 2, 5, 5), np.float32)
    residual[1, 0] = .7*np.sin(y/1.7)+.3*np.cos(x)
    residual[1, 1] = .4*np.sin(x/1.5)
    residual[2] = -.5*residual[1]
    return ZElasticField(300, [[0, 0], [3, -2], [-1, 1]], residual,
                          [100, 200], [8, 6], 300, [100, 200, 132, 224], [8, 11, 30])


def test_elastic_point_roundtrip_with_anisotropy_fractional_z_and_nonzero_origin(tmp_path):
    field = elastic_field()
    rng = np.random.default_rng(21)
    points = rng.uniform([100, 200, 300], [132, 224, 302.5], (300, 3))
    mapped = map_points(field, points)
    np.testing.assert_allclose(map_points(field, mapped, inverse=True), points, atol=2e-5)
    path = tmp_path/'elastic.npz'
    field.save(path)
    loaded = load_field(path)
    np.testing.assert_allclose(loaded.forward_pts(points), mapped)
    assert loaded.stats()['min_jacobian'] > 0
    with pytest.raises(ValueError, match='Non-affine'):
        ZAffineField.load(path)
    with pytest.raises(ValueError, match='cover'):
        field.forward_pts([[100, 200, 350]])


def test_folding_or_excessive_gradients_are_rejected():
    field = elastic_field()
    residual = field.displacement.copy()
    residual[1, 0] = -3*np.arange(5)[None, :]*field.spacing[0]
    assert deformation_limits(residual[1], field.spacing, field.resolution)['min_jacobian'] < 0
    with pytest.raises(ValueError, match='gradients or folding'):
        ZElasticField(field.z0, field.shifts, residual, field.origin, field.spacing,
                       field.ref_z, field.window_bbox, field.resolution)


def test_dense_warp_preserves_large_labels_and_nonconstant_geometry():
    field = elastic_field()
    seg = np.zeros((32, 24, 3), np.uint64)
    seg[4:12, 5:13] = 2**63+7
    seg[20:28, 15:21] = 2**64-1
    raw = (seg > 0).astype(np.uint8)*200
    aligned, labels, offset = register_arrays(raw, seg, [100, 200, 300], field)
    assert labels.dtype == np.uint64 and set(np.unique(labels)) == {0, 2**63+7, 2**64-1}
    assert aligned[labels > 0].mean() > 170
    for z in range(300, 303):
        point = field.forward_pts([[108.2, 209.3, z]])[0]
        cell = np.floor(point-np.array(offset)+.5).astype(int)
        assert labels[tuple(cell)] == 2**63+7
    skeleton = {'vertices': (np.array([[100, 210, 301], [131, 210, 301]])*field.resolution).tolist(),
                'edges': [[0, 1]]}
    lines = skeleton_lines(skeleton, field.resolution, field)
    assert len(lines) > 1  # Same-z edges must also follow nonlinear XY deformation.


def test_aligned_volume_uses_dense_preimages_and_limits_to_fitted_roi():
    pytest.importorskip('cloudvolume')
    from cloudvolume import Bbox
    field = elastic_field()
    data = np.zeros((32, 24, 3, 1), np.uint64)
    data[8:22, 6:18, :, 0] = 2**63+7

    class Volume:
        bounds = Bbox([100, 200, 300], [132, 224, 303])
        resolution = np.array([8, 11, 30])
        layer_type = 'segmentation'
        mip, dtype = 0, data.dtype

        def __getitem__(self, key):
            return data[tuple(slice(s.start-o, s.stop-o) for s, o in zip(key, self.bounds.minpt))]

    aligned = AlignedVolume(Volume(), field)
    lo, hi = np.asarray(aligned.bounds.minpt), np.asarray(aligned.bounds.maxpt)
    result = aligned[tuple(slice(a, b) for a, b in zip(lo, hi))][..., 0]
    for i, z in enumerate(range(300, 303)):
        expected = warp_field_slice(field, data[..., i, 0], z, [100, 200], lo[:2], result.shape[:2], seg=True)
        np.testing.assert_array_equal(result[..., i], expected)


def test_interactive_dense_field_maps_prompts_and_saves_replayable_transform(tmp_path, monkeypatch):
    pytest.importorskip('neuroglancer')
    import probe_em.interactive as interactive

    class Model:
        backend, metadata = 'test', {'backend': 'test'}

        def __init__(self, **kwargs):
            pass

        def predict(self, image, points, labels):
            mask = np.zeros(image.shape[:2], bool)
            for (x, y), label in zip(points.astype(int), labels):
                mask[y, x] = label
            return mask, .7

    monkeypatch.setattr(interactive, 'PointModel', Model)
    spec = importlib.util.spec_from_file_location('sofima_demo', ROOT/'probe-em/scripts/demo.py')
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    field = elastic_field()
    path = tmp_path/'field.npz'
    field.save(path)
    # Explicit metadata-free elastic fields carry their own resolution.
    raw = np.full((32, 24, 3), 100, np.uint8)
    seg = np.ones(raw.shape, np.uint64)
    monkeypatch.setattr(demo, 'load_local', lambda p, *a: (raw if Path(p).name == 'raw' else seg,
                                                         field.resolution, np.array([100, 200, 300])))
    viewer, report = demo.build_viewer(tmp_path, [], 1, tmp_path/'reviews', align_field=path,
                                      interactive={'backend': 'test', 'output_dir': tmp_path/'interactive'})
    session = viewer.probe_em_interactive
    try:
        assert report['alignment']['backend'] == 'sofima' and viewer.state.show_slices is False
        point = field.forward_pts([[108.2, 209.3, 301]])[0]
        with viewer.txn() as state:
            state.position = [point[0], point[1], 301.5]
        viewer.actions.invoke('prompt-positive', {'mousePosition': point.tolist()})
        viewer.actions.invoke('prompt-run', {})
        session.future.result(timeout=15)
        np.testing.assert_allclose(session.preview['snapshot']['points'][0], [108.2, 209.3, 301], atol=1e-3)
        for _ in range(2):
            viewer.actions.invoke('toggle-alignment', {})
            assert session._matches(session.preview['snapshot'])
        viewer.actions.invoke('prompt-accept', {})
        with np.load(session.accepted['folder']/'prediction.npz', allow_pickle=False) as saved:
            assert str(saved['display_field_kind']) == field.kind
            assert 'display_matrix' not in saved
            np.testing.assert_array_equal(saved['display_displacement_cyx'], field.displacement[1])
    finally:
        session.close()


@pytest.mark.parametrize('options', [{'not_a_parameter': 1}, {'max_gradient': 1}, {'max_residual_nm': -1},
                                    {'mesh_max_iters': 251}, {'batch_size': 0}])
def test_invalid_sofima_configuration_fails_before_loading_jax(options):
    from probe_em.sofima_registration import parameters
    with pytest.raises(ValueError):
        parameters(options)


@pytest.mark.skipif(not os.environ.get('SAVEM3_SOFIMA_PYTHON'), reason='No separate SOFIMA worker configured')
def test_real_sofima_nonrigid_refinement_cache_and_portable_saved_field(tmp_path, monkeypatch):
    from probe_em import sofima_registration
    rng = np.random.default_rng(14)
    image = cv2.GaussianBlur(rng.uniform(0, 255, (256, 256)).astype(np.float32), (5, 5), 0)
    y, x = np.indices(image.shape, dtype=np.float32)
    moved = cv2.remap(image, x+4*np.sin(y/45), y+3*np.cos(x/55), cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REFLECT)
    raw = np.stack([image.T, moved.T, image.T], axis=2)
    kwargs = dict(backend='sofima', sofima_python=os.environ['SAVEM3_SOFIMA_PYTHON'],
                  sofima_options={'patch_nm': 512, 'stride_nm': 128}, ref_z=300, cache_dir=tmp_path)
    field, info = prepare_field(raw, [8, 8, 30], [100, 200, 300], **kwargs)
    quality = json.loads(Path(info['field_path']).with_suffix('.json').read_text())['quality']
    measured = [r for r in quality['refinement_sections'] if r['accepted']]
    assert measured and max(r['ncc_after']-r['ncc_before'] for r in measured) > .1
    assert quality['global_validation']['status'] == 'accepted'
    assert np.any(field.displacement)
    assert field.stats()['min_jacobian'] > .3
    monkeypatch.setattr(sofima_registration, 'estimate_with_worker', lambda *a, **kw: pytest.fail('Cache was ignored'))
    _, cached = prepare_field(raw, [8, 8, 30], [100, 200, 300], **kwargs)
    assert cached['source'] == 'cache'
    monkeypatch.setattr(sofima_registration, 'describe_runtime', lambda *a: pytest.fail('Saved field needs no worker'))
    loaded, _ = prepare_field(raw, [8, 8, 30], [100, 200, 300], field_path=info['field_path'])
    np.testing.assert_array_equal(loaded.displacement, field.displacement)


def test_native_validation_retains_translation_when_elastic_candidate_is_worse():
    from probe_em.sofima_registration import parameters, validate_refinement
    rng = np.random.default_rng(42)
    image = rng.uniform(0, 255, (32, 24)).astype(np.float32)
    raw = np.repeat(image[..., None], 3, axis=2)
    candidate = elastic_field()
    shifts = np.zeros((3, 2))
    base = SimpleNamespace(z0=300, ref_z=300, window_bbox=candidate.window_bbox)
    field, quality = validate_refinement(raw, np.array([100, 200, 300]), candidate.resolution,
                                         base, shifts, candidate.displacement, candidate.origin,
                                         candidate.spacing, parameters(), .7)
    assert quality['status'] == 'retained_translation_initializer'
    assert not np.any(field.displacement)
    np.testing.assert_array_equal(field.align_cutout(raw, 300, origin_xy=(100, 200)), raw)
