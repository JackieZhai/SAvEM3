"""Cross-section review, real SAM2 memory, cancellation and nonrigid geometry."""
import json
import os
from pathlib import Path
import threading

import numpy as np
from PIL import Image
import pytest

from test_interactive import action, point, session
from test_sofima_alignment import elastic_field
from probe_em.alignment_field import warp_field_slice
from probe_em.demo_alignment import registered_bounds
from probe_em.propagation import PropagationCancelled, PropagationGrid


def video_stub(session, monkeypatch, *, started=None, release=None):
    session.model.backend = 'sam2'
    calls = []

    def propagate(folder, prompts, *, cancel, progress, mask_inputs=None):
        frames = sorted(Path(folder).glob('*.jpg'))
        shape = np.asarray(Image.open(frames[0])).shape[:2]
        calls.append((frames, prompts, mask_inputs))
        if started is not None:
            started.set()
            assert release.wait(10)
        points, labels = prompts[min(prompts)]
        x, y = np.rint(points[labels == 1][0]).astype(int)
        for index in range(len(frames)):
            mask = np.zeros(shape, bool)
            mask[max(0, y-1):y+2, max(0, x-1):x+2] = True
            progress(index, index+1, len(frames))
            yield index, mask

    monkeypatch.setattr(session.model, 'propagate', propagate, raising=False)
    return calls


def test_volume_actions_preserve_prompts_geometry_and_accept_complete_native_stack(session, monkeypatch):
    calls = video_stub(session, monkeypatch)
    point(session, [106, 207, 301])
    action(session, 'prompt-previous-section')
    action(session, 'prompt-range-start')
    action(session, 'prompt-next-section')
    action(session, 'prompt-next-section')
    action(session, 'prompt-range-end')
    action(session, 'prompt-propagate')
    session.future.result(timeout=15)
    result = session.preview
    assert result is not None and result['mask'].shape == (3, 20, 24)
    assert result['mask'][1, 7, 6]
    assert len(calls) == 1 and len(calls[0][0]) == 3
    metadata = json.loads((result['folder']/'metadata.json').read_text())
    assert metadata['inference_space'] == 'registered_xy'
    assert metadata['z_range_inclusive'] == [300, 302] and metadata['status'] == 'complete'
    assert metadata['initialization'] == 'corrected_native_image_masks'
    assert metadata['point_check']['positive_missed'] == 0
    assert (result['folder']/'alignment.npz').is_file()
    original_mask = result['mask'].copy()
    for _ in range(2):
        action(session, 'toggle-alignment')
        assert session._matches(result['snapshot'])
        np.testing.assert_array_equal(result['mask'], original_mask)
    action(session, 'prompt-accept')
    assert (result['folder']/'accepted.json').is_file()
    with np.load(result['folder']/'prediction.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['mask_zyx'], original_mask)
        np.testing.assert_array_equal(saved['frame_z'], [300, 301, 302])
        assert saved['processed'].all()
    point(session, [109, 208, 302])
    action(session, 'prompt-run')
    session.future.result(timeout=15)
    assert session.preview['snapshot']['scope'] == 'image' and session.accepted is result
    assert len(session.viewer.state.layers['positive_points'].annotations) == 2


@pytest.mark.parametrize('registered', [False, True])
def test_video_conditions_on_exact_image_result_and_retains_point_corrections(session, monkeypatch, registered):
    calls = video_stub(session, monkeypatch)
    point(session,[106,207,301])
    point(session,[114,214,301],0)
    action(session,'prompt-run');session.future.result(timeout=15)
    image = session.preview
    if not registered:
        action(session,'toggle-alignment')
    action(session,'prompt-propagate');session.future.result(timeout=15)
    video = session.preview
    np.testing.assert_array_equal(video['mask'][1],image['mask'])
    assert len(session.model.calls) == 1
    grid = PropagationGrid(session.raw,session.offset,300,302,
        field=session.field if registered else None,shape=session.aligned_shape,origin=session.aligned_offset)
    np.testing.assert_array_equal(calls[0][2][1],grid.video_mask(image['mask'],301))
    metadata = json.loads((video['folder']/'metadata.json').read_text())
    assert metadata['conditioning_masks'][0]['reused_image_prediction'] is True
    assert metadata['conditioning_masks'][0]['prediction_dir'] == str(image['folder'])
    assert metadata['point_check']['negative_included'] == 0
    assert metadata['point_check']['negative_points'] == 1
    point(session,[115,213,301],0)
    action(session,'prompt-propagate');session.future.result(timeout=15)
    assert len(session.model.calls) == 2
    metadata = json.loads((session.preview['folder']/'metadata.json').read_text())
    assert metadata['conditioning_masks'][0]['reused_image_prediction'] is False


def test_video_range_backend_and_positive_anchor_validation_precede_model_loading(session, monkeypatch):
    point(session, [106, 207, 301])
    action(session, 'prompt-propagate')
    assert session.future is None and 'requires' in session.viewer.config_state.state.status_messages['probe-em-interactive']
    calls = video_stub(session, monkeypatch)
    session.propagation_range = [302, 302]
    action(session, 'prompt-propagate')
    assert session.future is None
    session.propagation_range = [300, 301]
    point(session, [106, 207, 302])
    action(session, 'prompt-propagate')
    assert session.future is None and 'outside' in session.viewer.config_state.state.status_messages['probe-em-interactive']
    session.propagation_range = [300, 302]
    session.propagation_max_slices = 2
    action(session, 'prompt-propagate')
    assert session.future is None and 'exceeds' in session.viewer.config_state.state.status_messages['probe-em-interactive']
    session.propagation_max_slices = 3
    point(session, [110, 210, 300], 0)
    action(session, 'prompt-propagate')
    assert session.future is None and not calls


@pytest.mark.parametrize('operation', ['prompt-stop', 'prompt-clear', 'edit'])
def test_video_cancellation_or_edits_never_publish_stale_masks(session, monkeypatch, operation):
    started, release = threading.Event(), threading.Event()
    video_stub(session, monkeypatch, started=started, release=release)
    point(session, [106, 207, 301])
    action(session, 'prompt-propagate')
    assert started.wait(10)
    first = session.future
    action(session, 'prompt-propagate')
    assert session.future is first
    if operation == 'edit':
        point(session, [109, 208, 302])
    else:
        action(session, operation)
    release.set()
    first.result(timeout=15)
    assert session.preview is None and session.accepted is None
    assert not list(session.output.glob('*/accepted.json'))
    metadata = json.loads(next(session.output.glob('propagation_*/metadata.json')).read_text())
    assert metadata['status'] == 'cancelled'
    action(session, 'prompt-clear')
    point(session, [106, 207, 301])
    action(session, 'prompt-propagate')
    session.future.result(timeout=15)
    assert session.preview is not None


def test_editing_any_conditioning_section_blocks_accepting_stale_video(session, monkeypatch):
    video_stub(session, monkeypatch)
    point(session, [106, 207, 300])
    point(session, [107, 208, 302])
    action(session, 'prompt-propagate')
    session.future.result(timeout=15)
    folder = session.preview['folder']
    with session.viewer.txn() as state:
        state.layers['positive_points'].annotations[0].point = [110, 210, 300]
    action(session, 'prompt-accept')
    assert session.preview is None and not (folder/'accepted.json').exists()


@pytest.mark.parametrize('registered', [False, True])
def test_video_grid_maps_dense_prompts_and_masks_back_to_native(tmp_path, registered):
    field = elastic_field() if registered else None
    raw = np.zeros((32, 24, 3), np.uint8)
    raw[5:22, 4:18] = 180
    shape, offset = registered_bounds(raw.shape, [100, 200, 300], field, 100000) if field else (None, None)
    grid = PropagationGrid(raw, [100, 200, 300], 300, 302, field=field, shape=shape, origin=offset)
    manifest = grid.encode(tmp_path/'frames', threading.Event())
    assert [row['absolute_z'] for row in manifest] == [300, 301, 302]
    points = np.array([[108., 209., 301], [119., 215., 301]])
    prompts = grid.prompts(points, np.array([1, 0]))
    native = raw[:, :, 1] > 0
    displayed = warp_field_slice(field, native.astype(np.uint8), 301, [100, 200], grid.origin, grid.shape, seg=True)
    recovered = grid.native_mask(displayed.T, 301)
    np.testing.assert_array_equal(grid.video_mask(native.T,301),displayed.T.astype(bool))
    assert 2*np.count_nonzero(recovered & native.T)/(recovered.sum()+native.sum()) > .95
    assert recovered[9, 8]
    x, y = np.rint(prompts[1][0][0]).astype(int)
    assert displayed[x, y]


@pytest.mark.skipif(not os.environ.get('PROBE_EM_SAM2_CHECKPOINT'), reason='No local SAM2 checkpoint configured')
@pytest.mark.parametrize('feature_cache_mb', [0, 64])
def test_real_sam2_forward_reverse_video_cancellation_and_fresh_memory(tmp_path, feature_cache_mb):
    import torch
    from probe_em.prompt_models import PointModel
    torch.set_num_threads(2)
    for index in range(3):
        image = np.full((48, 64, 3), 127, np.uint8)
        image[15:30, 20+index:40+index] = 60
        Image.fromarray(image).save(tmp_path/f'{index:05d}.jpg', quality=100, subsampling=0)
    model = PointModel('sam2', os.environ['PROBE_EM_SAM2_CHECKPOINT'], model_config='sam2_hiera_t.yaml',
                        device=os.environ.get('SAVEM3_TEST_DEVICE', 'cpu'), feature_cache_mb=feature_cache_mb)
    prompts = {1: (np.array([[30, 22], [4, 4]]), np.array([1, 0]))}
    cancel = threading.Event()
    iterator = model.propagate(tmp_path, prompts, cancel=cancel)
    first = next(iterator)
    assert first[0] == 1 and first[1].shape == (48, 64)
    cancel.set()
    with pytest.raises(PropagationCancelled):
        next(iterator)
    frames = list(model.propagate(tmp_path, prompts, cancel=threading.Event()))
    assert [index for index, mask in frames] == [1, 2, 0]
    assert all(mask.dtype == bool and mask.shape == (48, 64) for _, mask in frames)
    video = model.sam2_cache.video()
    prompts[2] = (np.array([[32, 23]]), np.array([1]))
    corrected = list(model.propagate(tmp_path, prompts, cancel=threading.Event()))
    assert model.sam2_cache.video() is video and len(corrected) == 3
    if feature_cache_mb:
        assert model.metadata['acceleration']['feature_cache']['hits'] > 0
        fresh = PointModel('sam2', os.environ['PROBE_EM_SAM2_CHECKPOINT'], model_config='sam2_hiera_t.yaml',
                           device=os.environ.get('SAVEM3_TEST_DEVICE', 'cpu'))
        reference = dict(fresh.propagate(tmp_path, prompts, cancel=threading.Event()))
        for index, mask in corrected:
            np.testing.assert_array_equal(mask, reference[index])


@pytest.mark.skipif(not os.environ.get('PROBE_EM_SAM2_CHECKPOINT'), reason='No local SAM2 checkpoint configured')
def test_real_sam2_uses_corrected_masks_as_video_conditions(tmp_path, monkeypatch):
    import torch
    from probe_em.prompt_models import PointModel
    torch.set_num_threads(2)
    for index in range(3):
        image = np.full((48,64,3),127,np.uint8)
        image[15:30,20+index:40+index] = 60
        Image.fromarray(image).save(tmp_path/f'{index:05d}.jpg',quality=100,subsampling=0)
    model = PointModel('sam2',os.environ['PROBE_EM_SAM2_CHECKPOINT'],model_config='sam2_hiera_t.yaml',
                       device=os.environ.get('SAVEM3_TEST_DEVICE','cpu'))
    predictor = model._sam2().video()
    original = predictor.add_new_mask
    calls = []

    def add_mask(state, *, frame_idx, obj_id, mask):
        calls.append((frame_idx,mask.copy()))
        return original(state,frame_idx=frame_idx,obj_id=obj_id,mask=mask)

    monkeypatch.setattr(predictor,'add_new_mask',add_mask)
    point_method = 'add_new_points_or_box' if hasattr(predictor,'add_new_points_or_box') else 'add_new_points'
    monkeypatch.setattr(predictor,point_method,lambda *args,**kwargs: pytest.fail('Points replaced the corrected mask'))
    corrected = np.zeros((48,64),bool)
    corrected[17:29,22:38] = True
    frames = dict(model.propagate(tmp_path,{1:(np.array([[30,22],[4,4]]),np.array([1,0]))},
                                  mask_inputs={1:corrected},cancel=threading.Event()))
    assert set(frames) == {0,1,2} and len(calls) == 1 and calls[0][0] == 1
    np.testing.assert_array_equal(calls[0][1],corrected)
    np.testing.assert_array_equal(frames[1],corrected)
