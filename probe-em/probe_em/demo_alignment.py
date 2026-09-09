"""Local, non-destructive registration for the Neuroglancer ROI demo.

All arrays are XYZ and field matrices map absolute original XY voxel indices
to a common registered voxel frame. The source data and instance IDs stay intact.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from probe_em.z_align import ZAffineField, warp_slice


def translation(xy):
    matrix = np.eye(3)
    matrix[:2, 2] = xy
    return matrix


def validate_field(field, offset, depth):
    if field.ref_z < field.z0 or field.ref_z > field.z1:
        raise ValueError('Alignment reference slice is outside its field')
    for z in range(int(offset[2]), int(offset[2]) + depth):
        matrix = field.transform(z)
        if matrix is None:
            raise ValueError(f'Alignment field does not cover ROI slice {z}')
        if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
                or not np.allclose(matrix[2], [0, 0, 1])
                or abs(np.linalg.det(matrix[:2, :2])) < 1e-8):
            raise ValueError(f'Invalid affine transform for slice {z}')


def estimate_local_field(raw, offset, motion='translation', ref_z=None, *, resolution, **parameters):
    """Physical limits and a multiscale translation graph; resolution is required."""
    from probe_em.registration import estimate_field
    return estimate_field(raw, resolution, offset, motion, ref_z, **parameters)


def prepare_field(raw, resolution, offset, field_path=None, cache_dir=None,
                  motion='translation', ref_z=None, *, max_shift_nm=800.,
                  coarse_resolution_nm=32., min_ncc=.25, min_overlap=.70,
                  neighbor_gaps=(1, 2)):
    """Reuse only a content/config-matched cache, or an explicitly selected field."""
    from probe_em.registration import ALGORITHM
    parameters = dict(max_shift_nm=max_shift_nm, coarse_resolution_nm=coarse_resolution_nm,
                      min_ncc=min_ncc, min_overlap=min_overlap, neighbor_gaps=list(neighbor_gaps))
    metadata = {'schema_version': 2, 'algorithm': ALGORITHM,
                'shape_xyz': list(raw.shape), 'resolution_xyz_nm': np.asarray(resolution).tolist(),
                'voxel_offset_xyz': np.asarray(offset).tolist(), 'motion': motion,
                'reference_z': ref_z, 'registration_parameters': parameters}
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    digest.update(np.ascontiguousarray(raw).view(np.uint8))
    metadata['fingerprint'] = digest.hexdigest()
    field = None
    rejected = []
    source = 'estimated'
    quality_available = True
    quality = {}
    stored = {}
    if field_path:
        path = Path(field_path).resolve()
        field = ZAffineField.load(path)
        source = 'provided'
        sidecar = path.with_suffix('.json')
        quality_available = sidecar.is_file()
        if sidecar.is_file():
            stored = json.loads(sidecar.read_text())
            if not np.allclose(stored.get('resolution_xyz_nm', resolution), resolution):
                raise ValueError('Alignment field resolution does not match the loaded ROI')
            rejected = stored.get('rejected_pair_slices', [])
            quality = stored.get('quality', {})
        elif not np.allclose(resolution, [8, 8, 30]):
            raise ValueError('Legacy field has no resolution metadata and is only valid for 8x8x30 nm')
    else:
        path = (Path(cache_dir) / f"field_{metadata['fingerprint'][:20]}.npz").resolve() if cache_dir else None
        if path is not None and path.is_file() and path.with_suffix('.json').is_file():
            stored = json.loads(path.with_suffix('.json').read_text())
            if stored.get('fingerprint') == metadata['fingerprint']:
                field = ZAffineField.load(path)
                rejected = stored.get('rejected_pair_slices', [])
                quality = stored.get('quality', {})
                source = 'cache'
        if field is None:
            field, quality = estimate_local_field(raw, offset, motion, ref_z,
                                                  resolution=resolution, **parameters)
            rejected = quality['rejected_pair_slices']
            stored = dict(metadata, rejected_pair_slices=rejected, quality=quality)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                field.save(path)
                path.with_suffix('.json').write_text(json.dumps(
                    stored, indent=2), encoding='utf-8')
    validate_field(field, offset, raw.shape[2])
    return field, {'source': source, 'field_path': str(path) if path else None,
                   'requested_motion': motion if source != 'provided' else None,
                   'algorithm': stored.get('algorithm', 'legacy-unspecified'),
                   'registration_parameters': stored.get('registration_parameters'),
                   'quality_summary': {key: value for key, value in quality.items()
                                       if key not in ('pair_measurements', 'rejected_pair_slices')},
                   'quality_report_available': quality_available,
                   'rejected_pair_slices': rejected if quality_available else None, 'stats': field.stats()}


def registered_bounds(shape, offset, field, max_voxels):
    """Expand the output grid to retain all transformed source voxel centers."""
    offset = np.asarray(offset, dtype=np.int64)
    corners = np.array([[0, 0], [shape[0]-1, 0], [0, shape[1]-1],
                        [shape[0]-1, shape[1]-1]], dtype=float) + offset[:2]
    transformed = []
    for z in range(int(offset[2]), int(offset[2]) + shape[2]):
        matrix = field.transform(z)
        transformed.append(corners @ matrix[:2, :2].T + matrix[:2, 2])
    points = np.concatenate(transformed)
    lo, hi = np.floor(points.min(0)).astype(np.int64), np.ceil(points.max(0)).astype(np.int64) + 1
    result_shape = (int(hi[0]-lo[0]), int(hi[1]-lo[1]), int(shape[2]))
    if np.prod(result_shape, dtype=object) > max_voxels:
        raise ValueError('Registered ROI exceeds --max-voxels; use a smaller ROI or --no-align-z')
    return result_shape, np.array([lo[0], lo[1], offset[2]])


def register_arrays(raw, seg, offset, field, max_voxels=128_000_000):
    if raw.shape != seg.shape or raw.ndim != 3:
        raise ValueError('Registration requires matching XYZ raw and segmentation arrays')
    validate_field(field, offset, raw.shape[2])
    shape, aligned_offset = registered_bounds(raw.shape, offset, field, max_voxels)
    raw_out = np.empty(shape, dtype=raw.dtype)
    seg_out = np.empty(shape, dtype=seg.dtype)
    # Coordinates: local input -> absolute original -> aligned -> local output.
    input_shift, output_shift = translation(offset[:2]), translation(-aligned_offset[:2])
    for index in range(shape[2]):
        matrix = output_shift @ field.transform(int(offset[2]) + index) @ input_shift
        raw_out[:, :, index] = warp_slice(raw[:, :, index], matrix, output_shape=shape[:2], border_mode='constant')
        seg_out[:, :, index] = warp_slice(seg[:, :, index], matrix, seg=True, output_shape=shape[:2])
    return raw_out, seg_out, aligned_offset


def map_points(field, points, inverse=False):
    """Map absolute voxel positions; interpolate between section transforms."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    result = points.copy()
    for i, point in enumerate(points):
        lower, upper = int(np.floor(point[2])), int(np.ceil(point[2]))
        low, high = field.transform(lower), field.transform(upper)
        if low is None and high is None:
            continue
        low = high if low is None else low
        high = low if high is None else high
        weight = point[2]-lower
        matrix = (1-weight)*low + weight*high
        if inverse:
            matrix = np.linalg.inv(matrix)
        result[i, :2] = matrix[:2, :2] @ point[:2] + matrix[:2, 2]
    return result


def skeleton_lines(skeleton, resolution, field=None):
    """Subdivide cross-section edges so nonlinear-in-z warps follow the volume."""
    vertices = np.asarray(skeleton['vertices'], dtype=float) / np.asarray(resolution)
    result = []
    for edge_index, (u, v) in enumerate(skeleton['edges']):
        start, end = vertices[u], vertices[v]
        count = max(1, int(np.ceil(abs(end[2]-start[2])))) if field is not None else 1
        points = np.linspace(start, end, count+1)
        if field is not None:
            points = map_points(field, points)
        result.extend((f'{edge_index}-{i}', a, b) for i, (a, b) in enumerate(zip(points[:-1], points[1:])))
    return result
