"""Crop-aware point selection, instance-label sampling and optional alignment."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'probe-em'/'scripts'))
from compare_alignment import compare
from evaluate_local_gt import interior_point, metrics, sample_gt


def test_interior_point_avoids_cropped_object_edge():
    mask = np.ones((31, 41), bool)
    mask[:5, :10] = False
    x, y = interior_point(mask)
    assert mask[x, y] and min(x, y, 30-x, 40-y) >= 10
    mask[:] = False
    mask[0, 0] = True
    assert interior_point(mask) == (0, 0)
    with pytest.raises(ValueError, match='nonempty'):
        interior_point(np.zeros((4, 5), bool))


def test_gt_center_sampling_preserves_large_integer_ids_and_phase():
    labels = np.zeros((8, 8, 3), np.uint64)
    labels[1::2, 1::2] = 2**63+13
    sampled = sample_gt(labels, [2, 2, 1])
    assert sampled.shape == (4, 4, 3) and sampled.dtype == np.uint64
    assert np.all(sampled == 2**63+13)
    with pytest.raises(ValueError, match='positive integers'):
        sample_gt(labels, [1.5, 2, 1])


def test_empty_prediction_and_false_merge_metrics():
    target = np.array([True, True, False, False])
    empty = metrics(np.zeros(4, bool), target)
    assert empty['dice'] == 0 and empty['recall'] == 0 and empty['precision'] is None
    merged = metrics(np.ones(4, bool), target)
    assert merged['precision'] == .5 and merged['recall'] == 1 and merged['iou'] == .5


def test_registration_comparison_does_not_require_sofima():
    raw = np.random.default_rng(8).integers(0, 255, (32, 32, 3), dtype=np.uint8)
    result = compare(raw, np.zeros(raw.shape, np.uint32), [8, 8, 30], [12, 15, 4],
                     {'unregistered': None, 'ecc': None})
    assert result['mean_ncc']['unregistered'] == result['mean_ncc']['ecc']
    assert result['mean_support_fraction'] == 1
    assert result['elastic_geometry'] is None and result['lost_label_ids'] is None
