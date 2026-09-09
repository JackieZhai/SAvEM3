"""Opt-in checks with real local weights; no downloads and no quality claims."""
import os
from pathlib import Path
import sys

import numpy as np
import pytest


@pytest.mark.skipif(not os.environ.get('SAVEM3_HQ_CHECKPOINT'), reason='No local HQ checkpoint configured')
def test_real_hq_predictor_mask_interface():
    import torch
    from repro.sam import load_predictor, mask_prompt, choose_mask
    torch.set_num_threads(2)
    predictor = load_predictor(os.environ['SAVEM3_HQ_CHECKPOINT'], 'vit_b', 'cpu')
    image = np.full((64, 128, 3), 127, np.uint8)
    image[20:45, 35:80] = 60
    predictor.set_image(image)
    mask = np.zeros(image.shape[:2], bool)
    mask[25:40, 40:75] = True
    result, confidence = choose_mask(predictor.predict(mask_input=mask_prompt(mask, predictor), multimask_output=True))
    assert result.shape == image.shape[:2] and np.isfinite(confidence)


@pytest.mark.skipif(not os.environ.get('SAVEM3_HQ_CHECKPOINT'), reason='No local HQ checkpoint configured')
def test_teacher_export_with_real_hq_and_untrained_membrane_interface(tmp_path):
    import h5py
    import torch
    from repro.sam import load_hq_sam
    from repro.savem3.export_teacher import export
    from saem2.modeling import MaskDecoderHQ
    torch.set_num_threads(2)
    sam = load_hq_sam(os.environ['SAVEM3_HQ_CHECKPOINT'], 'vit_b', 'cpu')
    # Random membrane weights validate the tensor/file interface, NOT predictions.
    decoder = MaskDecoderHQ('vit_b', initialize_from_sam=False).eval()
    path = tmp_path / 'teacher.h5'
    export(np.full((2, 1024, 1024), 127, np.uint8), sam, decoder, path,
           [4, 4, 30], [20, 40, 60], batch_size=2,
           provenance={'test_only': 'untrained membrane decoder'})
    with h5py.File(path, 'r') as handle:
        assert handle.attrs['complete']
        assert handle['raw'].shape == handle['boundary'].shape == (2, 512, 512)
        assert handle['features'].shape == (32, 2, 256, 256)
        assert handle['embeddings'].shape == (256, 2, 64, 64)
        np.testing.assert_array_equal(handle.attrs['resolution_xyz_nm'], [8, 8, 30])
        np.testing.assert_array_equal(handle.attrs['voxel_offset_xyz'], [10, 20, 60])
        boundary = handle['boundary'][:]
        assert np.isfinite(boundary).all() and 0 <= boundary.min() <= boundary.max() <= 1
    assert not path.with_suffix('.h5.partial').exists()


@pytest.mark.skipif(not os.environ.get('PROBE_EM_SAM2_CHECKPOINT'), reason='No local SAM 2 checkpoint configured')
def test_real_sam2_predictor_reuse():
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'probe-em'))
    from probe_em.runtime import PredictorCache
    from probe_em.device import inference_autocast
    torch.set_num_threads(2)
    cache = PredictorCache(os.environ['PROBE_EM_SAM2_CHECKPOINT'], 'sam2_hiera_t.yaml', 'cpu')
    with torch.inference_mode(), inference_autocast('cpu'):
        predictor = cache.image()
        assert cache.image() is predictor
        predictor.set_image(np.full((64, 64, 3), 127, np.uint8))
        masks, scores, _ = predictor.predict(point_coords=np.array([[32, 32]]), point_labels=np.array([1]))
    assert masks.shape[-2:] == (64, 64) and np.isfinite(scores).all()


@pytest.mark.skipif(not os.environ.get('PROBE_EM_SAM2_CHECKPOINT'), reason='No local SAM 2 checkpoint configured')
@pytest.mark.parametrize('async_loading', [False, True])
def test_real_sam2_video_memory_on_selected_device(tmp_path, async_loading):
    import torch
    from PIL import Image
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'probe-em'))
    from probe_em.runtime import PredictorCache
    from probe_em.device import inference_autocast
    torch.set_num_threads(2)
    for frame in range(2):
        image = np.full((48, 64, 3), 127, np.uint8)
        image[15:30, 20+frame:40+frame] = 60
        Image.fromarray(image).save(tmp_path / f'{frame:05d}.jpg')
    mask = np.zeros((48, 64), bool)
    mask[15:30, 20:40] = True
    cache = PredictorCache(os.environ['PROBE_EM_SAM2_CHECKPOINT'], 'sam2_hiera_t.yaml', 'cpu')
    with torch.inference_mode(), inference_autocast('cpu'):
        predictor = cache.video()
        assert cache.video() is predictor
        state = predictor.init_state(video_path=str(tmp_path), async_loading_frames=async_loading)
        assert state['device'].type == 'cpu'
        predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=mask)
        frames = list(predictor.propagate_in_video(state))
    assert [frame[0] for frame in frames] == [0, 1]
    for _, ids, logits in frames:
        assert ids == [1] and logits.shape[-2:] == (48, 64)
        assert torch.isfinite(logits).all()
