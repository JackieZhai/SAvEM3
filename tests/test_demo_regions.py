import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('demo_regions', ROOT / 'probe-em/scripts/demo_regions.py')
regions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(regions)


def test_four_online_regions_are_disjoint_and_new():
    assert regions.RAW_SOURCE == 'precomputed://https://ng.zebrafish.digital-brain.cn/srv/raw/'
    assert len(regions.REGIONS) == 4
    regions.validate_regions(regions.REGIONS + [
        {'name': 'previous_demo', 'offset': [33386, 20334, 13868], 'shape': [261, 357, 191]}])
    with pytest.raises(ValueError, match='overlap'):
        regions.validate_regions([dict(regions.REGIONS[0], name='a'), dict(regions.REGIONS[0], name='b')])
    with pytest.raises(ValueError):
        regions.validate_regions([dict(regions.REGIONS[0], name='../unsafe')])


def test_seed_selection_uses_geometry_and_physical_offsets():
    labels = np.zeros((120, 120, 60), np.uint64)
    labels[40:50, 45:55, 20:40] = 2**63 + 7
    labels[:35, 60:70, 20:40] = 9
    resolution, offset = np.array([20, 20, 50]), np.array([100, 200, 300])
    skeletons = {}
    for label, points in ((2**63+7, [[45, 50, 22], [45, 50, 37]]),
                          (9, [[26, 65, 25], [28, 65, 35]])):
        skeletons[str(label)] = {'vertices': ((np.array(points) + offset) * resolution).tolist(),
                                 'edges': [[0, 1]]}
    seed, report = regions.select_seed(labels, skeletons, resolution, offset)
    assert seed == 2**63+7
    assert report['candidate_count'] == 2 and not report['seed_touches_roi_boundary']
    assert report['interior_endpoint_count'] == 2
    with pytest.raises(ValueError, match='No suitable'):
        regions.select_seed(labels, {}, resolution, offset)


def test_cache_reuse_verifies_content_and_coordinates(tmp_path):
    pytest.importorskip('cloudvolume')
    from export_savem3 import export_bundle
    region = {'name': 'small', 'offset': [10, 20, 30], 'shape': [12, 8, 4]}
    raw = np.arange(384, dtype=np.uint16).reshape(4, 8, 12).astype(np.uint8)
    labels = np.full(raw.shape, 2**63+7, np.uint64)
    directory = tmp_path / 'small'
    export_bundle(raw, labels, directory / 'cache', [8, 8, 30], region['offset'])
    report = {'region': region, 'sources': {'raw': regions.RAW_SOURCE, 'seg': regions.SEG_SOURCE},
              'resolution_xyz_nm': [8, 8, 30], 'sha256_decoded_xyz': {
                  'raw': regions.array_hash(raw.transpose(2, 1, 0)),
                  'seg': regions.array_hash(labels.transpose(2, 1, 0))}}
    regions.write_json(directory / 'region.json', report)
    assert regions.prepare_region(tmp_path, region) == report
    report['sha256_decoded_xyz']['raw'] = 'invalid'
    regions.write_json(directory / 'region.json', report)
    with pytest.raises(ValueError, match='checksum/coordinates changed'):
        regions.prepare_region(tmp_path, region)


def test_alignment_sanity_check_uses_global_coordinates_and_shared_support():
    pytest.importorskip('cv2')
    from probe_em.z_align import ZAffineField
    texture = np.random.default_rng(7).integers(0, 256, (64, 64), dtype=np.uint8)
    raw = np.stack([np.roll(texture, 3*index, axis=0) for index in range(3)], axis=2)
    matrices = {30+index: np.array([[1., 0., -3*index], [0., 1., 0.], [0., 0., 1.]])
                for index in range(3)}
    field = ZAffineField(30, 32, matrices, 30, [100, 200, 164, 264])
    report = regions.alignment_quality(raw, [100, 200, 30], field)
    assert report['mean_after'] > 0.99 and report['mean_before'] < 0.1
    assert len(report['pairs']) == 2


def test_viewer_prefers_validated_optimization_without_changing_trace_configuration(tmp_path):
    metadata = {'alignment': {'field_path': 'original.npz'}, 'sha256_decoded_xyz': {'raw': 'abc'}}
    assert regions.display_alignment(tmp_path, metadata) == (metadata['alignment'], False)
    report = {'validated': True, 'raw_sha256': 'abc', 'alignment': {'field_path': 'optimized.npz'}}
    regions.write_json(tmp_path / 'alignment_optimized/report.json', report)
    assert regions.display_alignment(tmp_path, metadata) == (report['alignment'], True)
    assert regions.display_alignment(tmp_path, metadata, legacy=True) == (metadata['alignment'], False)
    report['raw_sha256'] = 'mismatched'
    regions.write_json(tmp_path / 'alignment_optimized/report.json', report)
    with pytest.raises(ValueError, match='different raw'):
        regions.display_alignment(tmp_path, metadata)
