import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip('sam2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'probe-em'))
from probe_em.device import inference_autocast, sam2_config_name
from probe_em.z_align import warp_slice


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'probe-em' / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_uint64_alignment_is_exact():
    ids = np.array([[0, 2**63+3], [2**32+5, 7]], np.uint64)
    np.testing.assert_array_equal(warp_slice(ids, np.eye(3), seg=True), ids)
    # uint8 labels must use nearest-neighbour, not the raw-image interpolation.
    small = np.array([[0, 9], [9, 0]], np.uint8)
    output = warp_slice(small, np.array([[1, 0, 0.4], [0, 1, 0.4]]), seg=True)
    assert set(np.unique(output)) <= {0, 9}


def test_sam2_config_matches_installed_resource_and_cpu_context():
    import sam2
    resources = list((Path(sam2.__file__).parent / 'configs').rglob('*hiera_t.yaml'))
    if not resources:
        import sam2_configs
        resources = list(Path(sam2_configs.__file__).parent.glob('*hiera_t.yaml'))
    resource = resources[0]
    assert sam2_config_name(str(resource)).endswith('hiera_t.yaml')
    with inference_autocast('cpu'):
        pass
    with pytest.raises(FileNotFoundError):
        sam2_config_name('nonexistent_hiera.yaml')


def test_precomputed_bridge_preserves_coordinates_and_large_ids(tmp_path):
    bridge = script('export_savem3')
    demo = script('demo')
    raw = np.arange(4*8*12, dtype=np.uint16).reshape(4, 8, 12).astype(np.uint8)
    labels = np.full(raw.shape, 2**63+7, np.uint64)
    manifest = bridge.export_bundle(raw, labels, tmp_path / 'bundle', [8, 9, 30], [10, 20, 30])
    data, resolution, offset = demo.load_local(tmp_path / 'bundle' / 'seg')
    np.testing.assert_array_equal(data.transpose(2, 1, 0), labels)
    np.testing.assert_array_equal(offset, [10, 20, 30])
    np.testing.assert_array_equal(resolution, [8, 9, 30])
    assert manifest['target_mip'] == 0


def test_trace_failure_is_not_success_and_can_resume(tmp_path, monkeypatch):
    runner = script('run_probe_em')
    config = dict(runner.DEFAULT_CONFIG, output_root=str(tmp_path), device='cpu',
                  max_workers=1, checkpoint_sam='unused-in-mocked-verification', debug_limit=0)
    def endpoints(label, *args):
        if label == 2:
            raise RuntimeError('simulated missing skeleton')
        return np.ones((1, 3)), np.ones((1, 3)), np.array([8, 8, 30]), 100
    monkeypatch.setattr(runner, 'get_endpoints_vectors_precomputed', endpoints)
    monkeypatch.setattr(runner, 'get_neighbors', lambda *args, **kwargs: [{'target_id': 1, 'neighbor_id': 2, 'x': 1, 'y': 1, 'z': 1}])
    monkeypatch.setattr(runner, 'get_slices', lambda *args, **kwargs: [])
    monkeypatch.setattr(runner, 'find_merge_candidates', lambda *args, **kwargs: [2])
    monkeypatch.setattr(runner, 'find_merge_candidates_3d_region', lambda *args, **kwargs: [])
    assert runner.run_one_seed(1, config) == (1, 'failed')
    folder = tmp_path / '1_results_sam'
    state = json.loads((folder / 'trace_state.json').read_text())
    assert state['errors'][0]['segment'] == '2'
    with pytest.raises(FileExistsError):
        runner.run_one_seed(1, config)
    monkeypatch.setattr(runner, 'get_endpoints_vectors_precomputed', lambda *args: (np.ones((1, 3)), np.ones((1, 3)), [8, 8, 30], 100))
    monkeypatch.setattr(runner, 'get_neighbors', lambda *args, **kwargs: [])
    config['resume'] = True
    assert runner.run_one_seed(1, config) == (1, 'complete')
    assert runner.run_one_seed(1, config) == (1, 'skipped')
    assert json.loads((folder / 'trace_1_ng_segments.txt').read_text()) == {'segments': ['1', '2']}


def test_skeleton_export_preserves_physical_offset(tmp_path):
    from cloudvolume import CloudVolume
    bridge = script('export_savem3')
    labels = np.zeros((8, 16, 32), np.uint64)
    labels[2:6, 6:10, 2:30] = 7
    resolution, offset = np.array([8, 8, 30]), np.array([100, 200, 300])
    manifest = bridge.export_bundle(np.full(labels.shape, 127, np.uint8), labels,
                                    tmp_path / 'skeleton_bundle', resolution, offset, True)
    assert manifest['skeleton_ids'] == ['7']
    volume = CloudVolume(manifest['seg_path'], mip=0, fill_missing=False)
    skeleton = volume.skeleton.get(7)
    assert len(skeleton.vertices) >= 2 and len(skeleton.edges) >= 1
    assert np.all(skeleton.vertices >= offset * resolution)
    assert np.all(skeleton.vertices < (offset + labels.shape[::-1]) * resolution)


def test_asp_collision_resize_keeps_uint64_labels():
    from probe_em.find_merge_candidates_3d_region import calculate_collision
    segment_id = 2**63+7
    labels = np.full((2, 2), segment_id, np.uint64)
    collisions = calculate_collision(np.ones((16, 16), bool), labels, 1)
    assert collisions == {segment_id: 1.0}


def test_hss_finds_gap_candidate_at_nonzero_offset(tmp_path):
    from cloudvolume import CloudVolume
    from probe_em.get_neighbors import process_single_endpoint_xyz
    bridge = script('export_savem3')
    labels = np.zeros((1, 8, 12), np.uint64)
    labels[0, 2:6, 1:4] = 2**63+1
    labels[0, 2:6, 7:10] = 2**63+2
    manifest = bridge.export_bundle(np.zeros(labels.shape, np.uint8), labels,
                                    tmp_path / 'gap_bundle', [8, 8, 30], [100, 200, 300])
    volume = CloudVolume(manifest['seg_path'], mip=0, fill_missing=False)
    neighbors = process_single_endpoint_xyz(volume, [103, 204, 300], np.array([1., 0., 0.]),
                                           2**63+1, 0, search_dist_nm=500)
    assert len(neighbors) == 1 and neighbors[0]['neighbor_id'] == 2**63+2
    assert neighbors[0]['z'] == 300 and 107 <= neighbors[0]['x'] <= 109


@pytest.mark.parametrize('connected, expected_calls', [(True, 8), (False, 4)])
def test_pec_requires_four_bidirectional_votes(tmp_path, connected, expected_calls):
    from PIL import Image
    from probe_em.find_merge_candidates import process_pair
    labels = np.zeros((64, 64), np.uint8)
    labels[16:48, 8:28] = 1
    labels[16:48, 36:56] = 2
    raw_path, mask_path = tmp_path / '1_2_raw.jpg', tmp_path / '1_2_mask.png'
    Image.fromarray(np.full(labels.shape, 127, np.uint8)).save(raw_path)
    Image.fromarray(labels).save(mask_path)
    class Predictor:
        calls = 0
        def set_image(self, image):
            pass
        def predict(self, **kwargs):
            self.calls += 1
            mask = labels > 0 if connected else np.zeros(labels.shape, bool)
            return mask[None], np.array([1.]), None
    predictor = Predictor()
    result = process_pair(predictor, str(raw_path), str(mask_path), '', str(tmp_path), need_negative=False)
    assert bool(result['is_connected']) == connected
    assert predictor.calls == expected_calls
