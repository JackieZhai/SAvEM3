"""SAEM2 image-trial contracts, decoder interfaces and failure recovery."""
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('torch')
from saem2.predictor import SAEM2Predictor


@pytest.mark.parametrize('directory,module', [('.', 'saem2.modeling'), ('saem2', 'modeling')])
def test_decoder_import_in_a_fresh_process(directory, module):
    result = subprocess.run([sys.executable, '-c', f'from {module} import MaskDecoderHQ'],
                            cwd=Path(__file__).resolve().parents[1] / directory,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_resized_prompts_remain_float32():
    import torch

    class StopAtEncoder(Exception):
        pass

    def encode(points, boxes, masks):
        assert points[0].dtype == boxes.dtype == torch.float32
        assert points[1].dtype == torch.int64
        raise StopAtEncoder

    predictor = object.__new__(SAEM2Predictor)
    predictor.backbone = SimpleNamespace(
        is_image_set=True, original_size=(20, 30), device='cpu',
        transform=SimpleNamespace(apply_coords=lambda array, _: array.astype(np.float64),
                                  apply_boxes=lambda array, _: array.astype(np.float64)),
        model=SimpleNamespace(prompt_encoder=encode))
    with pytest.raises(StopAtEncoder):
        predictor.predict(point_coords=[[3, 4]], point_labels=[1], box=[0, 0, 10, 10])


def test_membrane_seed_fills_internal_structure_without_crossing_membrane():
    from saem2.membrane import seeded_component
    membrane = np.zeros((20, 30), np.float32)
    membrane[[2, 17], 2:18] = 1
    membrane[2:18, [2, 17]] = 1
    membrane[7:10, 7:10] = 1
    valid = np.ones_like(membrane, bool)
    mask, report = seeded_component(membrane, [5, 5], valid)
    assert mask[8, 8] and mask.sum() == 14*14 and not mask[:, 18:].any()
    assert not report['touches_valid_roi_boundary']
    outside, report = seeded_component(membrane, [25, 5], valid)
    assert report['touches_valid_roi_boundary'] and outside[5, 25]
    mask, report = seeded_component(membrane, [2, 2], valid)
    assert not mask.any() and report['status'] == 'prompt_on_predicted_boundary'
    valid[:, :2] = False
    with pytest.raises(ValueError, match='outside'):
        seeded_component(membrane, [1, 1], valid)


def test_predictor_requires_image_and_explicit_valid_prompts():
    predictor = object.__new__(SAEM2Predictor)
    predictor.backbone = SimpleNamespace(is_image_set=False)
    with pytest.raises(RuntimeError, match='set_image'):
        predictor.predict(point_coords=[[1, 1]], point_labels=[1])
    predictor.backbone = SimpleNamespace(is_image_set=True, original_size=(20, 30))
    with pytest.raises(ValueError, match='explicitly'):
        predictor.predict()
    for points, labels in [([[1, 1]], None), ([[1, np.nan]], [1]), ([[30, 1]], [1]),
                           ([[-1, 1]], [1]), ([[1, 1]], [2]), ([], [])]:
        with pytest.raises(ValueError):
            predictor.predict(point_coords=points, point_labels=labels)
    with pytest.raises(ValueError):
        predictor.set_image(np.zeros((20, 30), np.uint8))


def trial_module(monkeypatch):
    pytest.importorskip('sam2')
    scripts = Path(__file__).resolve().parents[1]/'probe-em/scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('saem2_trial', scripts/'try_saem2.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trial_prompt_is_inside_largest_component_even_at_boundary(monkeypatch):
    trial = trial_module(monkeypatch)
    mask = np.zeros((20, 30), bool)
    mask[:10, :10] = True
    mask[15:17, 25:27] = True
    point = trial.interior_point(mask).astype(int)
    assert mask[point[1], point[0]] and point[0] < 10 and point[1] < 10
    assert trial.interior_point(np.zeros_like(mask)) is None
    result = trial.agreement(mask, mask)
    assert result['input_seed_iou_not_accuracy'] == 1
    assert 'accuracy' not in result
    assert trial.agreement(mask.astype(np.float32), mask) == result
    with pytest.raises(ValueError, match='binary'):
        trial.agreement(mask.astype(np.float32)*.2, mask)


def test_trial_failure_is_explicit_and_existing_result_is_not_overwritten(tmp_path, monkeypatch):
    import json
    trial = trial_module(monkeypatch)
    output = tmp_path/'trial'
    args = SimpleNamespace(output=output)
    def fail(_):
        output.mkdir()
        (output/'manifest.json').write_text(json.dumps({'status': 'running'}))
        raise RuntimeError('inference failure')
    monkeypatch.setattr(trial, '_run', fail)
    with pytest.raises(RuntimeError, match='inference failure'):
        trial.run(args)
    assert json.loads((output/'manifest.json').read_text())['status'] == 'failed'
    before = (output/'manifest.json').read_bytes()
    with pytest.raises(FileExistsError):
        trial.run(args)
    assert (output/'manifest.json').read_bytes() == before
