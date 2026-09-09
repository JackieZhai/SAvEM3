import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')
pytest.importorskip('scipy')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'probe-em'))
from probe_em.registration import estimate_field, solve_translation_graph
from probe_em.z_align import warp_slice


def matrix(xy):
    return np.array([[1., 0., xy[0]], [0., 1., xy[1]], [0., 0., 1.]])


@pytest.mark.parametrize('resolution, scale', [([8, 8, 30], 1), ([16, 16, 30], .5)])
def test_large_shifts_are_limited_in_nm_not_fixed_pixels(resolution, scale):
    texture = cv2.GaussianBlur(np.random.default_rng(1).uniform(0, 255, (192, 160)).astype(np.float32), (5, 5), 0)
    shifts = np.array([[0, 0], [22, -12], [4, -20], [-8, 4], [10, 12]]) * scale
    raw = np.stack([warp_slice(texture, matrix(shift)) for shift in shifts], axis=2)
    field, quality = estimate_field(raw, resolution, [300, 400, 500], ref_z=500)
    assert quality['observed_slice_count'] == 5
    assert quality['interpolated_or_extrapolated_slices'] == []
    for index, shift in enumerate(shifts):
        np.testing.assert_allclose(field.transform(500+index)[:2, 2], -shift, atol=.4)


def test_weak_section_is_bridged_but_not_claimed_as_measured():
    rng = np.random.default_rng(8)
    texture = cv2.GaussianBlur(rng.uniform(0, 255, (160, 160)).astype(np.float32), (5, 5), 0)
    shifts = np.array([[0, 0], [14, -8], [7, -4], [0, 0], [-6, 8]])
    raw = np.stack([warp_slice(texture, matrix(shift)) for shift in shifts], axis=2)
    raw[:, :, 2] = rng.uniform(0, 255, texture.shape)
    field, quality = estimate_field(raw, [8, 8, 30], [100, 200, 300], ref_z=300)
    assert quality['observed_slice_count'] == 4
    assert quality['interpolated_or_extrapolated_slices'] == [302]
    for index in (0, 1, 3, 4):
        np.testing.assert_allclose(field.transform(300+index)[:2, 2], -shifts[index], atol=.4)


def test_unrelated_sections_are_not_reported_as_success():
    raw = np.random.default_rng(19).uniform(0, 255, (160, 160, 3)).astype(np.float32)
    with pytest.raises(ValueError, match='trustworthy'):
        estimate_field(raw, [8, 8, 30], [100, 200, 300])


def test_graph_preserves_real_jumps_and_downweights_conflicting_edges():
    positions = np.array([[0, 0], [10, -4], [-2, 6], [12, 0], [1, 7]], dtype=float)
    edges = []
    for gap in (1, 2):
        for moving in range(gap, len(positions)):
            fixed = moving-gap
            edges.append({'fixed': fixed, 'moving': moving, 'ncc_after': .9, 'overlap': .95,
                          'translation_xy_vox': (positions[moving]-positions[fixed]).tolist()})
    edges.append({'fixed': 0, 'moving': 4, 'ncc_after': .3, 'overlap': .8,
                  'translation_xy_vox': [100., 100.]})
    recovered, quality = solve_translation_graph(5, edges, 0, np.array([8, 8, 30]))
    np.testing.assert_allclose(recovered, positions, atol=.3)
    assert quality['graph_residual_p95_nm'] > 100  # Bad edge is reported, not silently hidden.


@pytest.mark.parametrize('parameters', [{'coarse_resolution_nm': np.inf}, {'max_shift_nm': np.nan},
                                       {'min_overlap': 0}, {'min_ncc': 2}])
def test_invalid_physical_parameters_do_not_start_pyramid_estimation(parameters):
    with pytest.raises(ValueError, match='limits or quality'):
        estimate_field(np.ones((16, 16, 3)), [8, 8, 30], [0, 0, 0], **parameters)
