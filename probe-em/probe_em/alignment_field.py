"""Affine and bounded elastic fields shared by display, prompts and cutouts.

Elastic maps apply a translation first, then a bilinearly interpolated residual
in the translated frame. Components are native XY voxels; grids are Z,C,Y,X.
The physical residual gradient is bounded below 1, giving an invertible map
under constant boundary extension and a convergent fixed-point inverse.
No SOFIMA/JAX dependency is needed to load or display a saved field.
"""
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

from probe_em.z_align import ZAffineField, warp_slice

ELASTIC_KIND = 'translation-elastic-v1'


def deformation_limits(displacement, spacing, resolution):
    """Bound every bilinear cell's physical Jacobian, including its corners."""
    physical = np.asarray(displacement, dtype=float) * np.asarray(resolution[:2])[:, None, None]
    dx = np.diff(physical, axis=2) / (spacing[0] * resolution[0])
    dy = np.diff(physical, axis=1) / (spacing[1] * resolution[1])
    norms, determinants = [], []
    for ix in (0, 1):
        for iy in (0, 1):
            a = dx[:, iy:iy+physical.shape[1]-1, :]
            b = dy[:, :, ix:ix+physical.shape[2]-1]
            norms.append(np.sqrt(np.sum(a*a+b*b, axis=0)))
            determinants.append((1+a[0])*(1+b[1])-a[1]*b[0])
    return {'max_gradient_norm': float(np.max(norms)),
            'min_jacobian': float(np.min(determinants)), 'max_jacobian': float(np.max(determinants)),
            'max_residual_nm': float(np.linalg.norm(physical, axis=0).max())}


class ZElasticField:
    kind = ELASTIC_KIND

    def __init__(self, z0, shifts, displacement, grid_origin_xy, grid_spacing_xy,
                 ref_z, window_bbox, resolution):
        self.z0, self.ref_z = int(z0), int(ref_z)
        self.shifts = np.array(shifts, dtype=np.float64, copy=True)
        self.displacement = np.array(displacement, dtype=np.float32, copy=True)
        self.origin = np.array(grid_origin_xy, dtype=float, copy=True)
        self.spacing = np.array(grid_spacing_xy, dtype=float, copy=True)
        self.window_bbox = np.array(window_bbox, dtype=np.int64, copy=True)
        self.resolution = np.array(resolution, dtype=float, copy=True)
        if (self.displacement.ndim != 4 or self.displacement.shape[1] != 2
                or min(self.displacement.shape[2:]) < 2 or self.displacement.shape[0] < 1
                or self.shifts.shape != (len(self.displacement), 2)
                or self.origin.shape != (2,) or self.spacing.shape != (2,)
                or self.resolution.shape != (3,) or self.window_bbox.shape != (4,)
                or np.any(self.spacing <= 0) or np.any(self.resolution <= 0)
                or np.any(self.window_bbox[2:] <= self.window_bbox[:2])
                or not all(np.isfinite(a).all() for a in (self.displacement, self.shifts, self.origin,
                                                        self.spacing, self.resolution))):
            raise ValueError('Invalid elastic field arrays or physical grid')
        self.z1 = self.z0 + len(self.displacement) - 1
        if not self.z0 <= self.ref_z <= self.z1:
            raise ValueError('Elastic reference is outside the field')
        self.limits = [deformation_limits(d, self.spacing, self.resolution) for d in self.displacement]
        if any(item['max_gradient_norm'] >= .8 or item['min_jacobian'] <= 0 for item in self.limits):
            raise ValueError('Elastic field has excessive gradients or folding; cannot invert safely')
        if (np.max(np.abs(self.shifts[self.ref_z-self.z0])) > 1e-4
                or np.max(np.abs(self.displacement[self.ref_z-self.z0])) > 1e-4):
            raise ValueError('Elastic reference section must remain fixed')
        for array in (self.shifts, self.displacement, self.origin, self.spacing,
                      self.window_bbox, self.resolution):
            array.flags.writeable = False
        self._grid_cache = None

    def _section(self, z):
        if not self.z0 <= z <= self.z1:
            raise ValueError('Elastic field does not cover the requested section')
        return int(z)-self.z0

    def _residual(self, xy, z):
        # Interpolate between transforms for navigation and skeleton vertices.
        z = float(np.clip(z, self.z0, self.z1))
        lower, upper = int(np.floor(z))-self.z0, int(np.ceil(z))-self.z0
        weight = z-np.floor(z)
        data = self.displacement[lower] if upper == lower else (
            (1-weight)*self.displacement[lower] + weight*self.displacement[upper])
        coordinates = ((np.asarray(xy)-self.origin)/self.spacing).T[::-1]
        return np.column_stack([map_coordinates(component, coordinates, order=1, mode='nearest',
                                                 prefilter=False) for component in data])

    def _shift(self, z):
        z = float(np.clip(z, self.z0, self.z1))
        lower, upper = int(np.floor(z))-self.z0, int(np.ceil(z))-self.z0
        return self.shifts[lower]*(1-(z % 1)) + self.shifts[upper]*(z % 1)

    def map_points(self, points, inverse=False):
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        if not np.isfinite(points).all():
            raise ValueError('Field coordinates must be finite')
        result = points.copy()
        for z in np.unique(points[:, 2]):
            # Match the affine viewer's interpolation at the outer half voxel.
            if z < self.z0-1 or z > self.z1+1:
                raise ValueError('Elastic field does not cover these points')
            selected = points[:, 2] == z
            xy = points[selected, :2]
            if inverse:
                q = xy.copy()
                for _ in range(80):
                    next_q = xy-self._residual(q, z)
                    error = np.max(np.abs(next_q-q), initial=0.)
                    q = next_q
                    if error < 1e-5:
                        break
                if np.max(np.abs(q+self._residual(q, z)-xy), initial=0.) > 1e-3:
                    raise ValueError('Elastic inverse did not converge')
                result[selected, :2] = q-self._shift(z)
            else:
                q = xy+self._shift(z)
                result[selected, :2] = q+self._residual(q, z)
        return result

    def forward_pts(self, points):
        return self.map_points(points)

    def inverse_pts(self, points):
        return self.map_points(points, inverse=True)

    inverse_connections = ZAffineField.inverse_connections

    def bounds_for_roi(self, shape, offset):
        offset = np.asarray(offset, dtype=int)
        lo, hi = [], []
        for z in range(offset[2], offset[2]+shape[2]):
            index = self._section(z)
            d = self.displacement[index].reshape(2, -1)
            lo.append(offset[:2]+self.shifts[index]+d.min(1))
            hi.append(offset[:2]+np.asarray(shape[:2])-1+self.shifts[index]+d.max(1))
        return np.floor(np.min(lo, axis=0)).astype(int), np.ceil(np.max(hi, axis=0)).astype(int)+1

    def warp_image(self, data, z, source_offset, output_offset, output_shape, seg=False):
        index = self._section(z)
        if not np.any(self.displacement[index]):
            # Retaining the initializer should also retain its exact rasterization.
            matrix = np.eye(3)
            matrix[:2, 2] = self.shifts[index]+np.asarray(source_offset)-np.asarray(output_offset)
            return warp_slice(data, matrix, seg=seg, output_shape=output_shape, border_mode='constant')
        key = (int(z), tuple(source_offset), tuple(output_offset), tuple(output_shape))
        cached = self._grid_cache
        if cached is None or cached[0] != key:
            x, y = np.indices(output_shape)
            points = np.column_stack((x.ravel()+output_offset[0], y.ravel()+output_offset[1],
                                      np.full(x.size, z)))
            inverse = (self.inverse_pts(points)[:, :2]-source_offset).reshape((*output_shape, 2))
            self._grid_cache = (key, inverse)
        else:
            inverse = cached[1]
        if seg:
            # Index labels directly. Never interpolate uint64 through float.
            nearest = np.floor(inverse+.5).astype(np.int64)
            valid = ((nearest >= 0) & (nearest < data.shape[:2])).all(axis=2)
            result = np.zeros(output_shape, data.dtype)
            result[valid] = data[nearest[..., 0][valid], nearest[..., 1][valid]]
            return result
        return cv2.remap(np.ascontiguousarray(data.T), inverse[..., 0].T.astype(np.float32),
                         inverse[..., 1].T.astype(np.float32), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT).T.copy()

    def align_cutout(self, data, z0, seg=False, origin_xy=(0, 0)):
        if data.ndim == 2:
            return self.warp_image(data, z0, origin_xy, origin_xy, data.shape, seg)
        return np.stack([self.warp_image(data[..., i], z0+i, origin_xy, origin_xy,
                                        data.shape[:2], seg) for i in range(data.shape[2])], axis=2)

    def stats(self):
        return {'kind': self.kind, 'z_range': [self.z0, self.z1], 'ref_z': self.ref_z,
                'n_slices': len(self.shifts), 'grid_shape_yx': list(self.displacement.shape[2:]),
                'grid_spacing_xy_vox': self.spacing.tolist(),
                'max_abs_xy_shift_vox': float(np.abs(self.shifts).max()),
                'max_residual_nm': max(item['max_residual_nm'] for item in self.limits),
                'max_gradient_norm': max(item['max_gradient_norm'] for item in self.limits),
                'min_jacobian': min(item['min_jacobian'] for item in self.limits),
                'max_jacobian': max(item['max_jacobian'] for item in self.limits)}

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, kind=self.kind, z0=self.z0, ref_z=self.ref_z, shifts=self.shifts,
                            displacement=self.displacement, grid_origin_xy=self.origin,
                            grid_spacing_xy=self.spacing, window_bbox=self.window_bbox,
                            resolution=self.resolution)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if str(data['kind']) != ELASTIC_KIND:
                raise ValueError('Unsupported elastic field schema')
            return cls(**{key: data[key] for key in ('z0', 'shifts', 'displacement', 'grid_origin_xy',
                                                   'grid_spacing_xy', 'ref_z', 'window_bbox', 'resolution')})


def load_field(path):
    with np.load(path, allow_pickle=False) as data:
        kind = str(data['kind']) if 'kind' in data else 'affine'
    if kind == ELASTIC_KIND:
        return ZElasticField.load(path)
    if kind != 'affine':
        raise ValueError(f'Unsupported alignment field: {kind}')
    return ZAffineField.load(path)


def warp_field_slice(field, data, z, source_offset, output_offset, output_shape, seg=False):
    if isinstance(field, ZElasticField):
        return field.warp_image(data, z, source_offset, output_offset, output_shape, seg)
    source, target = np.eye(3), np.eye(3)
    source[:2, 2], target[:2, 2] = source_offset, -np.asarray(output_offset)
    matrix = field.transform(z) if field is not None else np.eye(3)
    if matrix is None:
        matrix = np.eye(3)
    return warp_slice(data, target @ matrix @ source, seg=seg, output_shape=output_shape, border_mode='constant')


def field_slice_payload(field, z):
    if isinstance(field, ZElasticField):
        return {'display_field_kind': field.kind, 'display_base_shift_xy': field.shifts[field._section(z)],
                'display_displacement_cyx': field.displacement[field._section(z)],
                'display_grid_origin_xy': field.origin, 'display_grid_spacing_xy': field.spacing}
    return {'display_matrix': field.transform(z) if field else np.eye(3)}
