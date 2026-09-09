"""Physical-scale, coarse-to-fine slice registration with a robust translation graph.

Arrays use XYZ order. Pair transforms map moving -> fixed local voxel indices;
the returned field maps absolute original coordinates -> a shared voxel frame.
Weak/unobservable sections are reported, never counted as successfully measured.
"""
import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsqr

from probe_em.z_align import ZAffineField, _estimate_pair_affine, warp_slice

ALGORITHM = 'physical-pyramid-graph-v3.2'


def ncc(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    a, b = a-a.mean(), b-b.mean()
    denominator = np.sqrt(np.sum(a*a) * np.sum(b*b))
    return float(np.sum(a*b)/denominator) if denominator > 1e-10 else -1.0


def image_pyramid(raw, resolution, coarse_resolution_nm):
    """Explicit XYZ voxel scales, including odd shapes and anisotropic XY pixels."""
    target = np.maximum(1., coarse_resolution_nm / np.asarray(resolution[:2]))
    factors = [target]
    while np.max(factors[-1]) > 1:
        factors.append(np.maximum(1., factors[-1]/2))
    levels = []
    seen = set()
    for factor in factors:
        shape = np.maximum(np.minimum(raw.shape[:2], [32, 32]),
                           np.rint(np.asarray(raw.shape[:2])/factor).astype(int))
        if tuple(shape) in seen:
            continue
        seen.add(tuple(shape))
        scale = np.asarray(raw.shape[:2], dtype=float) / shape
        images = [cv2.resize(raw[:, :, z].T, tuple(map(int, shape)), interpolation=cv2.INTER_AREA).T
                  for z in range(raw.shape[2])]
        levels.append((images, scale))
    return levels


def estimate_pair(levels, fixed, moving, resolution, max_shift_nm=800., min_ncc=.25,
                  min_overlap=.70, motion='translation'):
    estimate = None
    converged_levels = 0
    # Do not propagate an erroneous coarse peak to every finer level. Assess
    # each candidate in the same coarse frame; rejected levels restart phase
    # correlation if no trustworthy initialization has been obtained yet.
    coarse_images, coarse_scale = levels[0]
    coarse_scaling = np.diag([*coarse_scale, 1.])
    coarse_fixed, coarse_moving = coarse_images[fixed], coarse_images[moving]
    coarse_ones = np.ones(coarse_fixed.shape, np.uint8)
    for images, scale in levels:
        scaling = np.diag([*scale, 1.])
        init = None if estimate is None else np.linalg.inv(scaling) @ estimate @ scaling
        pair = _estimate_pair_affine(images[fixed], images[moving], init=init,
                                    max_trans=1e6, motion=motion)
        if pair is None:
            continue
        proposed = scaling @ pair @ np.linalg.inv(scaling)
        # The limit is in physical nm, not pixels at a particular pyramid level.
        if np.linalg.norm(proposed[:2, 2] * resolution[:2]) <= max_shift_nm:
            coarse_matrix = np.linalg.inv(coarse_scaling) @ proposed @ coarse_scaling
            support = warp_slice(coarse_ones, coarse_matrix, seg=True) > 0
            if support.mean() < min_overlap:
                continue
            coarse_warped = warp_slice(coarse_moving, coarse_matrix, border_mode='constant')
            candidate_score = ncc(coarse_fixed[support], coarse_warped[support])
            coarse_baseline = ncc(coarse_fixed[support], coarse_moving[support])
            if candidate_score >= min_ncc and candidate_score >= coarse_baseline-.005:
                estimate = proposed
                converged_levels += 1
    info = {'fixed': int(fixed), 'moving': int(moving), 'accepted': False,
            'converged_levels': converged_levels}
    if estimate is None:
        return None, dict(info, reason='no_reliable_pyramid_candidate')
    images, scale = levels[0]
    scaling = np.diag([*scale, 1.])
    matrix = np.linalg.inv(scaling) @ estimate @ scaling
    fixed_image, moving_image = images[fixed], images[moving]
    valid = warp_slice(np.ones(fixed_image.shape, np.uint8), matrix, seg=True) > 0
    overlap = float(valid.mean())
    if overlap < min_overlap:
        return None, dict(info, reason='insufficient_overlap', overlap=overlap)
    warped = warp_slice(moving_image, matrix, border_mode='constant')
    before, after = ncc(fixed_image[valid], moving_image[valid]), ncc(fixed_image[valid], warped[valid])
    info.update(overlap=overlap, ncc_before=before, ncc_after=after,
                translation_xy_vox=estimate[:2, 2].tolist())
    if after < min_ncc or after < before-.005:
        return None, dict(info, reason='low_correlation_or_regression')
    return estimate, dict(info, accepted=True, reason='measured')


def solve_translation_graph(depth, edges, reference, resolution, huber_nm=16.):
    """IRLS solves pair offsets without z smoothing; real jumps remain possible.

    Skip-section edges bridge a weak section. Only sections outside the largest
    observed component are interpolated/extrapolated, explicitly listed in QC.
    """
    rows = [edge['fixed'] for edge in edges] + [edge['moving'] for edge in edges]
    cols = [edge['moving'] for edge in edges] + [edge['fixed'] for edge in edges]
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(depth, depth)).tocsr()
    _, labels = connected_components(graph, directed=False)
    counts = np.bincount(labels)
    component = int(np.argmax(counts))
    supported = np.flatnonzero(labels == component)
    if len(supported) < 2 and depth > 1:
        raise ValueError('No trustworthy registration component; use --no-align-z or inspect the images')
    selected = [edge for edge in edges if labels[edge['fixed']] == component]
    anchor = int(supported[np.argmin(np.abs(supported-reference))])
    columns = {int(node): index for index, node in enumerate(supported[supported != anchor])}
    row_indices, col_indices, values = [], [], []
    for index, edge in enumerate(selected):
        for node, sign in ((edge['fixed'], -1.), (edge['moving'], 1.)):
            if node != anchor:
                row_indices.append(index)
                col_indices.append(columns[node])
                values.append(sign)
    matrix = coo_matrix((values, (row_indices, col_indices)),
                        shape=(len(selected), len(columns))).tocsr()
    target = np.asarray([edge['translation_xy_vox'] for edge in selected]) * resolution[:2]
    confidence = np.asarray([edge['ncc_after']**2 * edge['overlap'] /
                             (edge['moving']-edge['fixed']) for edge in selected])
    weights = confidence.copy()
    for _ in range(6):
        weighted = matrix.multiply(np.sqrt(weights)[:, None]).tocsr()
        solution = np.column_stack([lsqr(weighted, target[:, axis]*np.sqrt(weights),
                                         atol=1e-10, btol=1e-10)[0] for axis in range(2)])
        residual = np.linalg.norm(matrix @ solution-target, axis=1)
        weights = confidence * np.minimum(1., huber_nm/np.maximum(residual, 1e-8))
    shifts = np.zeros((depth, 2))
    for node, column in columns.items():
        shifts[node] = solution[column] / resolution[:2]
    missing = np.flatnonzero(labels != component)
    for axis in range(2):
        shifts[missing, axis] = np.interp(missing, supported, shifts[supported, axis])
    shifts -= shifts[reference]
    return shifts, {'interpolated_or_extrapolated_slices': missing.tolist(),
                    'observed_slice_count': int(len(supported)), 'graph_edge_count': len(selected),
                    'graph_residual_median_nm': float(np.median(residual)),
                    'graph_residual_p95_nm': float(np.percentile(residual, 95))}


def estimate_field(raw, resolution, offset, motion='translation', ref_z=None,
                   max_shift_nm=800., coarse_resolution_nm=32., min_ncc=.25,
                   min_overlap=.70, neighbor_gaps=(1, 2)):
    resolution, offset = np.asarray(resolution, dtype=float), np.asarray(offset)
    if (offset.shape != (3,) or not np.isfinite(offset).all()
            or np.any(offset != np.round(offset))):
        raise ValueError('Registration requires three integer XYZ voxel offsets')
    offset = offset.astype(np.int64)
    if raw.ndim != 3 or min(raw.shape[:2]) < 8 or raw.shape[2] < 1:
        raise ValueError('Registration needs nonempty XYZ images at least 8 pixels wide/high')
    if resolution.shape != (3,) or not np.isfinite(resolution).all() or np.any(resolution <= 0):
        raise ValueError('Registration requires positive finite XYZ nm/voxel resolution')
    if (not np.isfinite([max_shift_nm, coarse_resolution_nm, min_ncc, min_overlap]).all()
            or max_shift_nm <= 0 or coarse_resolution_nm <= 0
            or not 0 <= min_ncc <= 1 or not 0 < min_overlap <= 1):
        raise ValueError('Invalid physical registration limits or quality thresholds')
    if motion not in ('translation', 'euclidean', 'affine'):
        raise ValueError('Unsupported local motion model')
    depth = raw.shape[2]
    z0, z1 = int(offset[2]), int(offset[2])+depth-1
    reference = (depth-1)//2 if ref_z is None else int(ref_z)-z0
    if not 0 <= reference < depth:
        raise ValueError('Alignment reference slice must lie inside the ROI')
    if raw[:, :, reference].std() < 1e-6:
        raise ValueError('Reference image is constant; cannot estimate registration')
    levels = image_pyramid(raw, resolution, coarse_resolution_nm)
    edges, attempts = [], []
    pair_matrices = {}
    for gap in (neighbor_gaps if motion == 'translation' else (1,)):
        if not isinstance(gap, int) or gap < 1:
            raise ValueError('Neighbor gaps must be positive integers')
        for moving in range(gap, depth):
            matrix, info = estimate_pair(levels, moving-gap, moving, resolution,
                                         max_shift_nm, min_ncc, min_overlap, motion)
            attempts.append(info)
            if matrix is not None:
                edges.append(info)
                pair_matrices[(moving-gap, moving)] = matrix
            if moving % 25 == 0 or moving == depth-1:
                print(f'[registration v3] gap={gap}, section {moving+1}/{depth}, '
                      f'{len(edges)}/{len(attempts)} accepted edges', flush=True)
    if depth == 1:
        matrices = {z0: np.eye(3)}
        quality = {'interpolated_or_extrapolated_slices': [], 'observed_slice_count': 1}
    elif motion == 'translation':
        shifts, quality = solve_translation_graph(depth, edges, reference, resolution)
        quality['solver'] = 'robust-translation-graph'
        matrices = {z0+index: np.array([[1., 0., dx], [0., 1., dy], [0., 0., 1.]])
                    for index, (dx, dy) in enumerate(shifts)}
    else:
        if not edges:
            raise ValueError('All registration pairs failed')
        chained = [np.eye(3)]
        for moving in range(1, depth):
            chained.append(chained[-1] @ pair_matrices.get((moving-1, moving), np.eye(3)))
        origin = np.eye(3)
        origin[:2, 2] = offset[:2]
        anchor = np.linalg.inv(chained[reference])
        matrices = {z0+index: origin @ anchor @ matrix @ np.linalg.inv(origin)
                    for index, matrix in enumerate(chained)}
        quality = {'interpolated_or_extrapolated_slices': [], 'solver': 'affine-chain',
                   'reused_neighbor_slices': [z0+index for index in range(1, depth)
                                             if (index-1, index) not in pair_matrices]}
    rejected = [z0+item['moving'] for item in attempts
                if item['moving']-item['fixed'] == 1 and not item['accepted']]
    quality['interpolated_or_extrapolated_slices'] = [z0+index for index in quality['interpolated_or_extrapolated_slices']]
    quality.update(algorithm=ALGORITHM, pair_measurements=attempts,
                   rejected_pair_slices=rejected, reference_z=z0+reference)
    window = [*offset[:2], *(offset[:2]+raw.shape[:2])]
    return ZAffineField(z0, z1, matrices, z0+reference, window), quality
