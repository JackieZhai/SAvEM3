"""SAM2 point agreement, untouched candidates and safe correction history."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'probe-em'))
from probe_em.point_refinement import predict_refined


class Predictor:
    def __init__(self, callback):
        self.callback = callback
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        masks, scores = self.callback(kwargs)
        count = len(masks)
        return np.asarray(masks), np.asarray(scores), np.full((count,256,256), len(self.calls), np.float32)


def test_consistent_fresh_mask_does_not_inherit_an_older_masks_extent():
    mask = np.zeros((8,9),bool)
    mask[2:6,2:7] = True
    predictor = Predictor(lambda _: ([mask],[.8]))
    result = predict_refined(predictor, (8,9,3), [[3,3],[1,1]], [1,0], np.ones((1,256,256)))
    assert len(predictor.calls) == 1
    assert predictor.calls[0]['mask_input'] is None
    np.testing.assert_array_equal(result['mask'], mask)
    assert not result['diagnostics']['previous_mask_used']


def test_previous_logits_resolve_a_negative_point_without_pixel_overwrites():
    wrong = np.ones((8,9),bool)
    corrected = wrong.copy()
    corrected[:3,:3] = False
    prior = np.full((1,256,256), -4, np.float32)
    predictor = Predictor(lambda call: ([corrected if call['mask_input'] is not None else wrong],[.7]))
    result = predict_refined(predictor, (8,9,3), [[4,4],[1,1]], [1,0], prior)
    assert len(predictor.calls) == 2
    np.testing.assert_array_equal(predictor.calls[1]['mask_input'], prior)
    np.testing.assert_array_equal(result['mask'], corrected)
    assert result['diagnostics']['negative_included'] == 0
    assert result['diagnostics']['candidate_source'] == 'previous_mask'


def test_alternative_mask_must_respect_negative_points_even_with_a_lower_score():
    wrong = np.ones((8,9),bool)
    correct = wrong.copy()
    correct[:3,:3] = False
    empty = np.zeros_like(wrong)
    predictor = Predictor(lambda call: ([wrong,correct,empty],[.99,.6,.9])
                          if call['multimask_output'] else ([wrong],[.99]))
    result = predict_refined(predictor, (8,9,3), [[4,4],[1,1]], [1,0])
    np.testing.assert_array_equal(result['mask'], correct)
    assert result['score'] == .6
    assert result['diagnostics']['positive_missed'] == result['diagnostics']['negative_included'] == 0


def test_unresolved_negative_points_are_reported_and_not_erased():
    wrong = np.ones((8,9),bool)
    predictor = Predictor(lambda call: ([wrong]*3,[.8]*3) if call['multimask_output'] else ([wrong],[.8]))
    result = predict_refined(predictor, (8,9,3), [[4,4],[1,1]], [1,0])
    assert result['diagnostics']['negative_included'] == 1
    assert result['diagnostics']['positive_missed'] == 0
    np.testing.assert_array_equal(result['mask'], wrong)
    assert any(call['mask_input'] is not None for call in predictor.calls)


@pytest.mark.parametrize('points,labels', [([[9,1]],[1]), ([[1,1]],[0]), ([[1,float('nan')]],[1])])
def test_bad_points_fail_before_model_execution(points, labels):
    predictor = Predictor(lambda _: pytest.fail('Invalid input reached the model'))
    with pytest.raises(ValueError, match='native XY'):
        predict_refined(predictor,(8,9,3),points,labels)
