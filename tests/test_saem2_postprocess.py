"""The bounded membrane graph uses probabilities, not old label masks."""
import json

import numpy as np
import pytest

from repro.saem2 import postprocess_trial


def test_postprocess_zyx_seed_and_background_contract(tmp_path, monkeypatch):
    boundary = np.zeros((3, 5, 7), np.float32)
    valid = np.ones(boundary.shape, bool)
    valid[:, :, 0] = False
    boundary[~valid] = 1
    np.savez(tmp_path/'postprocess_inputs.npz', boundary=boundary, valid=valid, seed_zyx=[1, 2, 3])
    def watershed(data, **kwargs):
        np.testing.assert_array_equal(data, boundary)
        return np.ones(data.shape, np.uint64)
    def multicut(data, fragments, **kwargs):
        np.testing.assert_array_equal(data, boundary)
        assert not fragments[~valid].any()
        return fragments
    monkeypatch.setattr(postprocess_trial, 'watershed_oversegment', watershed)
    monkeypatch.setattr(postprocess_trial, 'lmc_agglomerate', multicut)
    postprocess_trial.process(tmp_path)
    selected = np.load(tmp_path/'saem2_multicut/selected_xyz.npy')
    np.testing.assert_array_equal(selected, valid.transpose(2, 1, 0))
    report = json.loads((tmp_path/'saem2_multicut/status.json').read_text())
    assert report['selected_sections'] == 3 and not report['uses_old_labels_for_graph']
    with pytest.raises(FileExistsError):
        postprocess_trial.process(tmp_path)


def test_invalid_seed_marks_failure_before_graph(tmp_path, monkeypatch):
    np.savez(tmp_path/'postprocess_inputs.npz', boundary=np.zeros((3, 5, 7), np.float32),
             valid=np.zeros((3, 5, 7), bool), seed_zyx=[1, 2, 3])
    with pytest.raises(ValueError, match='Invalid'):
        postprocess_trial.process(tmp_path)
    report = json.loads((tmp_path/'saem2_multicut/status.json').read_text())
    assert report['status'] == 'failed'
    assert not (tmp_path/'saem2_multicut/selected_xyz.npy').exists()
