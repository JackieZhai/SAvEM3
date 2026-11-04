"""Regression tests for the five-ROI tracing investigation (no model weights)."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip('cloudvolume')
pytest.importorskip('cv2')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'probe-em'))
from probe_em.trace_inputs import nearest_component, prepare_video
from probe_em.trace_verification import occupancy_evidence, cycle_overlap


def test_asp_leakage_and_one_frame_collision_require_review():
    labels = np.zeros((20, 20), np.uint64)
    labels[2:10, 2:10] = 2**63+7
    predictions = {i: np.ones(labels.shape, bool) for i in range(3)}
    evidence = occupancy_evidence(predictions, {i: labels for i in predictions}, 1, [2**63+7])[0]
    assert evidence['paper_accepted']
    assert not evidence['is_connected']  # Full label coverage, only 16% precision.
    assert not occupancy_evidence({0: labels > 0, 1: labels > 0}, {0: labels, 1: labels},
                                   1, [2**63+7])[0]['is_connected']


def test_asp_candidate_gate_and_real_temporal_support():
    labels = np.full((10, 10), 2**63+7, np.uint64)
    predictions = {i: np.ones(labels.shape, bool) for i in range(3)}
    frames = {i: labels for i in predictions}
    assert not occupancy_evidence(predictions, frames, 1, [2])[0]['is_connected']
    assert occupancy_evidence(predictions, frames, 1, [2**63+7])[0]['is_connected']
    assert occupancy_evidence(predictions, frames, 1, [], policy='paper')[0]['is_connected']
    assert cycle_overlap(labels > 0, labels > 0) == {'coverage': 1., 'iou': 1.}


def test_component_follows_query_not_crop_center():
    mask = np.zeros((20, 20), bool)
    mask[8:12, 8:12] = True
    mask[1:3, 1:3] = True
    result = nearest_component(mask, [1, 1])
    assert result.sum() == 4 and result[1, 1]


def bridge_module():
    path = Path(__file__).resolve().parents[1] / 'probe-em/scripts/export_savem3.py'
    spec = importlib.util.spec_from_file_location('bridge_quality', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_aligned_volume_preserves_translated_roi_boundary(tmp_path):
    from cloudvolume import CloudVolume
    from probe_em.z_align import AlignedVolume, ZAffineField
    labels = np.zeros((2, 10, 12), np.uint64)
    labels[:, 2:8, 8:12] = 2**63+1
    manifest = bridge_module().export_bundle(np.full(labels.shape, 120, np.uint8), labels,
                                              tmp_path/'bundle', [8, 8, 30], [100, 200, 300])
    volume = CloudVolume(manifest['seg_path'], fill_missing=False)
    field = ZAffineField(300, 301, {z: np.array([[1., 0., 20.], [0., 1., -30.], [0., 0., 1.]])
                                   for z in (300, 301)}, 300, [100, 200, 112, 210])
    aligned = AlignedVolume(volume, field)
    assert list(aligned.bounds.minpt) == [120, 170, 300]
    result = aligned[aligned.bounds.to_slices()][..., 0]
    assert np.array_equal(result, labels.transpose(2, 1, 0))
    assert aligned[120:132, 160:190, 300:302].shape == (12, 30, 2, 1)


def test_asp_searches_multislice_gap_and_dense_frame_names(tmp_path):
    from cloudvolume import CloudVolume
    labels = np.zeros((10, 20, 20), np.uint64)
    labels[0:2, 5:15, 5:15] = 1
    labels[5:9, 5:15, 5:15] = 2
    manifest = bridge_module().export_bundle(np.full(labels.shape, 120, np.uint8), labels,
                                              tmp_path/'bundle', [8, 8, 30], [100, 200, 300])
    seg, raw = (CloudVolume(manifest[key], fill_missing=False) for key in ('seg_path', 'raw_path'))
    result = prepare_video(seg, raw, [110, 210, 305], 1, 2, np.array([20, 20, 20]),
                           tmp_path/'videos', '1_2_0')
    assert result['status'] == 'saved' and result['z_sequence'][:2] == [301, 302]
    assert 306 in result['z_sequence'] and result['extended_window']
    images = sorted(Path(result['path'], 'video_frames').glob('*.jpg'))
    assert [p.stem for p in images] == [f'{i:05d}' for i in range(len(images))]
    repeated = prepare_video(seg, raw, [110, 210, 305], 1, 2, np.array([20, 20, 20]),
                              tmp_path/'videos', '1_2_0')
    assert repeated['status'] == 'saved'
    assert len(list((tmp_path/'input_attempts/videos').glob('1_2_0_*'))) == 1


def test_comparison_layers_follow_alignment_toggle(tmp_path):
    pytest.importorskip('neuroglancer')
    path = Path(__file__).resolve().parents[1] / 'probe-em/scripts/demo.py'
    spec = importlib.util.spec_from_file_location('demo_quality', path)
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    from probe_em.z_align import ZAffineField
    labels = np.zeros((2, 10, 12), np.uint64)
    labels[:, 2:8, 2:8] = 1
    bundle = tmp_path/'bundle'
    bridge_module().export_bundle(np.full(labels.shape, 120, np.uint8), labels,
                                  bundle, [8, 8, 30], [100, 200, 300])
    field = ZAffineField(300, 301, {z: np.eye(3) for z in (300, 301)}, 300, [100, 200, 112, 210])
    field_path = tmp_path/'field.npz'
    field.save(field_path)
    viewer, report = demo.build_viewer(bundle, [1], 1, tmp_path/'reviews',
                                      align_field=field_path, comparison_ids={'baseline_trace': ['1']})
    assert not viewer.state.show_slices
    assert not viewer.state.layers['baseline_trace'].visible
    assert viewer.state.layers['baseline_trace'].hover_highlight is False
    assert viewer.state.layers['probe_trace'].hover_highlight is False
    assert viewer.state.layers['probe_trace'].pick is False
    assert viewer.state.layers['baseline_trace'].pick is False
    assert viewer.state.layers['manual_correction'].pick is True
    before = viewer.state.layers['baseline_trace'].to_json()['source']
    viewer.actions.invoke('toggle-alignment', {})
    after = viewer.state.layers['baseline_trace'].to_json()['source']
    assert before != after
    assert after == viewer.state.layers['probe_trace'].to_json()['source']
    with viewer.txn() as state:
        state.layers['probe_trace'].segments = [99]
        state.layers['baseline_trace'].segments = [99]
        state.layers['manual_correction'].segments = [1, 99]
    viewer.actions.invoke('restore-result', {})
    assert set(viewer.state.layers['probe_trace'].segments) == {1}
    assert set(viewer.state.layers['baseline_trace'].segments) == {1}
    assert set(viewer.state.layers['manual_correction'].segments) == {1, 99}


def test_explicit_alignment_field_is_read_only(tmp_path, monkeypatch):
    pytest.importorskip('sam2')
    path = Path(__file__).resolve().parents[1] / 'probe-em/scripts/run_probe_em.py'
    spec = importlib.util.spec_from_file_location('runner_quality', path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    from probe_em.z_align import ZAffineField
    field_path = tmp_path/'field.npz'
    ZAffineField(0, 0, {0: np.eye(3)}, 0, [0, 0, 8, 8]).save(field_path)
    def unexpected_write(*args):
        raise AssertionError('Provided fields must not be rewritten')
    monkeypatch.setattr(ZAffineField, 'save', unexpected_write)
    tracer = runner.NeuronTracer('unused', 'unused', device='cpu', align_z=True,
                                 align_z_field=str(field_path))
    assert tracer._ensure_z_field(1).ref_z == 0
    tracer = runner.NeuronTracer('unused', 'unused', device='cpu', align_z=True,
                                 align_z_field=str(tmp_path/'missing.npz'))
    with pytest.raises(FileNotFoundError):
        tracer._ensure_z_field(1)


def test_interrupted_node_is_pending_not_visited(tmp_path, monkeypatch):
    import json
    pytest.importorskip('sam2')
    path = Path(__file__).resolve().parents[1] / 'probe-em/scripts/run_probe_em.py'
    spec = importlib.util.spec_from_file_location('runner_interrupted', path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    def interrupted(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(runner, 'get_endpoints_vectors_precomputed', interrupted)
    config = dict(runner.DEFAULT_CONFIG, output_root=str(tmp_path), device='cpu', debug_limit=8)
    with pytest.raises(KeyboardInterrupt):
        runner.run_one_seed(1, config)
    state = json.loads((tmp_path/'1_results_sam/trace_state.json').read_text())
    assert state['visited'] == [] and state['pending'] == ['1']
    assert json.loads((tmp_path/'1_results_sam/status.json').read_text())['interrupted']


def test_pec_rejects_large_mask_despite_full_pair_coverage(tmp_path):
    pytest.importorskip('sam2')
    from PIL import Image
    from probe_em.find_merge_candidates import process_pair
    labels = np.zeros((64, 64), np.uint8)
    labels[16:48, 8:24] = 1
    labels[16:48, 40:56] = 2
    leaked = np.zeros_like(labels, bool)
    leaked[4:60, 4:60] = True  # Below the historical 90% full-image guard.
    raw, mask = tmp_path/'1_2_raw.jpg', tmp_path/'1_2_mask.png'
    Image.fromarray(np.full(labels.shape, 128, np.uint8)).save(raw)
    Image.fromarray(labels).save(mask)
    class Predictor:
        def set_image(self, image):
            pass
        def predict(self, **kwargs):
            return leaked[None], np.array([1.]), None
    for policy, expected in [('paper', True), ('guarded', False)]:
        result = process_pair(Predictor(), str(raw), str(mask), '', str(tmp_path),
                              need_negative=False, verification_policy=policy)
        assert bool(result['is_connected']) == expected


def test_short_fragment_density_is_warning_not_pruning(tmp_path, monkeypatch):
    pytest.importorskip('sam2')
    path = Path(__file__).resolve().parents[1] / 'probe-em/scripts/run_probe_em.py'
    spec = importlib.util.spec_from_file_location('runner_density', path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, 'get_endpoints_vectors_precomputed',
                        lambda *a: (np.ones((4, 3)), np.ones((4, 3)), [8, 8, 30], 0))
    monkeypatch.setattr(runner, 'is_messy_segment', lambda *a, **k: True)
    monkeypatch.setattr(runner, 'get_neighbors', lambda path, uid, *a, **k:
                        [{'target_id': 1, 'neighbor_id': 2, 'x': 1, 'y': 1, 'z': 1}] if uid == 1 else [])
    monkeypatch.setattr(runner, 'get_slices', lambda *a, **k: [])
    monkeypatch.setattr(runner, 'find_merge_candidates', lambda *a, **k: [2])
    monkeypatch.setattr(runner, 'find_merge_candidates_3d_region', lambda *a, **k: [])
    tracer = runner.NeuronTracer('unused', 'unused', device='cpu', verification_policy='guarded')
    result = tracer.trace(1, str(tmp_path))
    assert set(result) == {1, 2}
    assert tracer.node_audit[1]['density_warning'] and not tracer.node_audit[1]['pruned']


def test_complex_candidate_is_rejected_before_budget_stop_and_on_resume(tmp_path, monkeypatch):
    import json
    pytest.importorskip('sam2')
    path = Path(__file__).resolve().parents[1] / 'probe-em/scripts/run_probe_em.py'
    spec = importlib.util.spec_from_file_location('runner_eager', path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, 'get_endpoints_vectors_precomputed',
                        lambda uid, *a: (np.ones((133 if uid == 2 else 2, 3)),
                                         np.ones((133 if uid == 2 else 2, 3)), [8, 8, 30], 0))
    monkeypatch.setattr(runner, 'get_neighbors', lambda *a, **k:
                        [{'target_id': 1, 'neighbor_id': 2, 'x': 1, 'y': 1, 'z': 1}])
    monkeypatch.setattr(runner, 'get_slices', lambda *a, **k: [])
    monkeypatch.setattr(runner, 'find_merge_candidates', lambda *a, **k: [2])
    monkeypatch.setattr(runner, 'find_merge_candidates_3d_region', lambda *a, **k: [])
    tracer = runner.NeuronTracer('unused', 'unused', device='cpu', debug_limit=1)
    assert tracer.trace(1, str(tmp_path)) == [1]
    assert tracer.stack == [] and tracer.node_audit[0]['candidate_topology'][0]['rejected']
    runner.atomic_json(tmp_path/'trace_state.json', {'seed': '1', 'visited': ['1'],
        'accepted': ['1', '2', '3'], 'pending': ['2', '3'],
        'edges': [['1', '2'], ['2', '3']], 'errors': []})
    resumed = runner.NeuronTracer('unused', 'unused', device='cpu', debug_limit=1)
    assert resumed.trace(1, str(tmp_path), resume=True) == [1]
    state = json.loads((tmp_path/'trace_state.json').read_text())
    assert state['accepted'] == ['1'] and state['pending'] == []
    assert state['edges'] == [] and set(resumed.graph) == {1}


def test_snapshot_excludes_unfinished_evidence_and_preserves_raw_state(tmp_path, monkeypatch):
    import json
    pytest.importorskip('sam2')
    scripts = Path(__file__).resolve().parents[1] / 'probe-em/scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('audit_snapshot', scripts/'audit_trace_quality.py')
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    directory = tmp_path/'region'
    source = directory/'trace_results/1_results_sam'
    state = {'seed': '1', 'visited': ['1'], 'accepted': ['1', '2'],
             'pending': ['2'], 'edges': [['1', '2']], 'errors': []}
    for path, value in [(source/'trace_state.json', state),
                        (source/'status.json', {'status': 'running', 'config_hash': 'test'}),
                        (directory/'case.json', {'name': 'region', 'seed': '1'}),
                        (directory/'trace_config.json', {'seg_path': 'unused'}),
                        (directory/'baseline.json', {'ids': ['1']})]:
        audit.atomic_json(path, value)
    audit.atomic_json(source/'temp_vis3d_1/1_3_0_evidence.json',
                      [{'neighbor_id': 3, 'paper_accepted': True, 'is_connected': 0, 'reasons': ['review']}])
    partial = source/'temp_vis3d_2/unfinished_evidence.json'
    partial.parent.mkdir()
    partial.write_text('incomplete JSON from stopped inference', encoding='utf-8')
    audit.snapshot_checkpoint(directory)
    first = json.loads((directory/'comparison.json').read_text())
    assert first['review_ids'] == ['3'] and first['guarded_ids'] == ['1', '2']
    from probe_em import topology
    monkeypatch.setattr(topology, 'candidate_topology', lambda *a, **k:
                        {'segment': '2', 'endpoint_count': 133, 'rejected': True})
    audit.snapshot_checkpoint(directory, topology=True)
    second = json.loads((directory/'comparison.json').read_text())
    assert second['guarded_ids'] == ['1'] and second['review_ids'] == ['2', '3']
    assert json.loads((source/'trace_state.json').read_text()) == state
    assert Path(first['result_directory'], 'trace_1_ids.txt').read_text().split() == ['1', '2']
