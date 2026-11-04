"""Naive image-only seed extraction from SAEM2 membrane probabilities.

This fixed-threshold component baseline is not a learned head, a graph-cut
algorithm, or proof of cross-section identity. Boundary gaps can merge cells;
hole filling can also absorb enclosed cells when the selected component leaks.
Retained as a diagnostic failure baseline, not the recommended 3D reconstruction.
"""
import numpy as np
from scipy import ndimage as ndi


def seeded_component(probability, point_xy, valid, threshold=.5):
    probability, valid = np.asarray(probability), np.asarray(valid, dtype=bool)
    if (probability.ndim != 2 or valid.shape != probability.shape
            or not np.isfinite(probability).all() or probability.min() < 0
            or probability.max() > 1 or not 0 < threshold < 1):
        raise ValueError('Expected finite YX probabilities, matching valid mask and threshold in (0,1)')
    result = np.zeros(probability.shape, bool)
    if point_xy is None:
        return result, {'status': 'no_seed_prompt'}
    point = np.asarray(point_xy)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError('Expected a finite XY seed point')
    x, y = np.rint(point).astype(int)
    if not (0 <= x < valid.shape[1] and 0 <= y < valid.shape[0]) or not valid[y, x]:
        raise ValueError('Seed is outside valid image data')
    components, _ = ndi.label((probability < threshold) & valid)
    uid = components[y, x]
    if uid == 0:
        return result, {'status': 'prompt_on_predicted_boundary'}
    # Fill enclosed internal structures, but never fill outside the acquired ROI.
    result = ndi.binary_fill_holes(components == uid) & valid
    return result, {'status': 'component_selected', 'threshold': threshold,
                    'touches_valid_roi_boundary': bool(np.any(result & ~ndi.binary_erosion(valid)))}
