"""Independent mesh scale, physical coordinates, exact IDs, and reusable caches."""
from pathlib import Path
import sys

import numpy as np
import pytest

ng = pytest.importorskip('neuroglancer')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'probe-em'))
from probe_em.meshes import MeshVolume, mesh_factors, mesh_geometry


@pytest.mark.parametrize('resolution,factors', [
    ([8, 8, 30], [4, 4, 1]), ([8, 8, 8], [1, 1, 1]),
    ([4, 4, 40], [8, 8, 1]), ([40, 8, 8], [1, 4, 4]),
    ([6, 10, 40], [8, 4, 1]), ([8, 16, 60], [8, 4, 1]),
])
def test_default_policy_adapts_to_actual_axis_resolutions(resolution, factors):
    np.testing.assert_array_equal(mesh_factors(resolution), factors)
    actual = np.asarray(resolution) * factors
    assert actual.max() / actual.min() <= np.sqrt(2)


@pytest.mark.parametrize('resolution,target', [
    ([0, 8, 30], None), ([np.nan, 8, 30], None), ([8, 8], None),
    ([8, 8, np.inf], None), ([8, 8, 30], -1), ([8, 8, 30], np.nan),
    ([8, 8, 30], np.inf), ([1e-100, 8, 30], None),
])
def test_invalid_mesh_scales_fail_early(resolution, target):
    with pytest.raises(ValueError):
        mesh_factors(resolution, target_nm=target)


def test_explicit_target_and_native_override():
    np.testing.assert_array_equal(mesh_factors([8, 8, 30], target_nm=64), [8, 8, 2])
    np.testing.assert_array_equal(mesh_factors([8, 8, 30], isotropic=False), [1, 1, 1])
    with pytest.raises(ValueError, match='requires isotropic'):
        mesh_factors([8, 8, 30], isotropic=False, target_nm=64)


def labels():
    data = np.zeros((41, 37, 12), np.uint64)
    data[8:32, 8:28, 2:10] = 2**63 + 7
    data[5, 5, 4] = 2**63 + 9  # Deliberately missed by 4x4 striding.
    return data


def volume(data=None, resolution=(8, 8, 30), offset=(33387, 19503, 13701), **kwargs):
    return MeshVolume(labels() if data is None else data,
                      dimensions=ng.CoordinateSpace(names=['x', 'y', 'z'], units='nm', scales=resolution),
                      voxel_offset=offset, mesh_options={'max_quadrics_error': -1}, **kwargs)


def test_mesh_input_is_coarse_but_slices_and_nondivisible_origin_are_native():
    source = volume()
    report = source.mesh_report()
    np.testing.assert_array_equal(report['downsample_xyz'], [4, 4, 1])
    np.testing.assert_allclose(report['mesh_resolution_xyz_nm'], [32, 32, 30])
    assert report['mesh_shape_xyz'] == [11, 10, 12]
    assert source.info()['shape'] == labels().shape
    np.testing.assert_array_equal(source.info()['voxelOffset'], [33387, 19503, 13701])
    encoded, _ = source.get_encoded_subvolume('raw', np.zeros(3, dtype=int), np.array(labels().shape), '1,1,1')
    assert encoded == labels().tobytes(order='C')  # Neuroglancer's Python raw encoding.
    reference = ng.LocalVolume(labels()[::4, ::4, :],
        dimensions=ng.CoordinateSpace(names=['x', 'y', 'z'], units='nm', scales=[32, 32, 30]),
        mesh_options={'max_quadrics_error': -1})
    actual = mesh_geometry(source.get_object_mesh(2**63 + 7))
    expected = mesh_geometry(reference.get_object_mesh(2**63 + 7))
    assert actual['triangles'] == expected['triangles'] > 0
    origin_nm = np.array([33387, 19503, 13701]) * [8, 8, 30]
    np.testing.assert_allclose(np.array(actual['local_bounds_nm']) + origin_nm,
                               np.array(expected['local_bounds_nm']) + origin_nm, atol=1e-4)
    assert source.mesh_report()['objects'][str(2**63+7)]['method'] == 'near_isotropic'


def test_missing_tiny_id_falls_back_without_changing_label_values():
    source = volume()
    assert source.mesh_report()['native_fallback_ids'] == [str(2**63+9)]
    assert mesh_geometry(source.get_object_mesh(2**63+9))['triangles'] > 0
    report = source.mesh_report()['objects'][str(2**63+9)]
    assert report['method'] == 'native_fallback_missing_label'
    np.testing.assert_allclose(report['mesh_resolution_xyz_nm'], [8, 8, 30])
    np.testing.assert_array_equal(source.data[:, :, :], labels())
    with pytest.raises(ng.local_volume.InvalidObjectIdForMesh):
        source.get_object_mesh(100)


def test_native_override_produces_native_meshes():
    source = volume(mesh_isotropic=False)
    source.get_object_mesh(2**63+7)
    report = source.mesh_report()
    assert report['objects'][str(2**63+7)]['method'] == 'native'
    assert report['downsample_xyz'] == [1, 1, 1]


def test_uniform_roi_reports_empty_mesh_without_blocking_slice_viewer():
    source = volume(data=np.ones((8, 8, 8), np.uint64))
    report = source.precompute_meshes([1])
    assert report['objects']['1']['empty_mesh'] is True
    assert mesh_geometry(source.get_object_mesh(1))['triangles'] == 0
    np.testing.assert_array_equal(source.data[:, :, :], 1)


def test_signed_nonnegative_label_values_are_supported_without_remapping():
    data = np.zeros((16, 16, 5), dtype=np.int64)
    data[4:12, 4:12, 1:4] = 1234567890123
    source = volume(data=data)
    assert mesh_geometry(source.get_object_mesh(1234567890123))['triangles'] > 0
    np.testing.assert_array_equal(source.data[:, :, :], data)


def test_mesh_cache_reuses_geometry_and_fingerprints_all_inputs(tmp_path, monkeypatch):
    original = volume(mesh_cache=tmp_path)
    encoded = original.get_object_mesh(2**63+7)
    fingerprint = original.mesh_report()['fingerprint']
    changed = labels()
    changed[0, 0, 0] = 42
    alternatives = [volume(data=changed, mesh_cache=tmp_path),
                    volume(resolution=[9, 8, 30], mesh_cache=tmp_path),
                    volume(offset=[33388, 19503, 13701], mesh_cache=tmp_path),
                    volume(mesh_isotropic=False, mesh_cache=tmp_path),
                    volume(mesh_target_nm=64, mesh_cache=tmp_path)]
    assert all(v.mesh_report()['fingerprint'] != fingerprint for v in alternatives)
    cached = volume(mesh_cache=tmp_path)
    monkeypatch.setattr(ng.LocalVolume, 'get_object_mesh', lambda *a: pytest.fail('Cached mesh was regenerated'))
    assert cached.get_object_mesh(2**63+7) == encoded
    assert cached.mesh_report()['objects'][str(2**63+7)]['source'] == 'disk_cache'


def test_mesh_cache_corruption_is_not_silently_accepted(tmp_path):
    source = volume(mesh_cache=tmp_path)
    source.get_object_mesh(2**63+7)
    directory = Path(source.mesh_report()['cache_directory'])
    (directory / f'{2**63+7}.bin').write_bytes(b'broken')
    with pytest.raises(ValueError, match='checksum'):
        volume(mesh_cache=tmp_path).get_object_mesh(2**63+7)


def test_invalidation_rebuilds_mesh_data_and_cache_fingerprint(tmp_path):
    data = labels()
    source = volume(data=data, mesh_cache=tmp_path)
    previous = source.mesh_report()['fingerprint']
    source.get_object_mesh(2**63+7)
    data[data == 2**63+7] = 42
    source.invalidate()
    assert source.mesh_report()['fingerprint'] != previous
    assert mesh_geometry(source.get_object_mesh(42))['triangles'] > 0
    with pytest.raises(ng.local_volume.InvalidObjectIdForMesh):
        source.get_object_mesh(2**63+7)
