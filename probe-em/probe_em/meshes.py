"""Near-isotropic display meshes without downsampling segmentation slices.

LocalVolume.downsampling controls image/label chunks, not the meshing input.
Delegate mesh requests to a separate, origin-anchored coarse LocalVolume. Its
vertices are already in physical units relative to the original ROI origin;
the native source's frontend transform supplies the original voxel offset.
"""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import threading
import uuid

import neuroglancer
from neuroglancer.local_volume import InvalidObjectIdForMesh
import numpy as np


ALGORITHM = 'near-isotropic-stride-mesh-v1'


def mesh_factors(resolution_nm, isotropic=True, target_nm=None):
    """Round physical scale ratios to powers of two; never upsample an axis."""
    resolution = np.asarray(resolution_nm, dtype=float)
    if resolution.shape != (3,) or not np.isfinite(resolution).all() or np.any(resolution <= 0):
        raise ValueError('Mesh resolution must contain three positive finite XYZ values')
    if target_nm is not None and (not np.isfinite(target_nm) or target_nm <= 0):
        raise ValueError('Mesh target must be positive and finite')
    if not isotropic:
        if target_nm is not None:
            raise ValueError('A mesh target requires isotropic meshing to be enabled')
        return np.ones(3, dtype=np.int64)
    target = float(resolution.max() if target_nm is None else target_nm)
    exponents = np.maximum(0, np.floor(np.log2(target) - np.log2(resolution) + .5))
    if np.any(exponents > 30):
        raise ValueError('Requested mesh scale ratio is too large')
    return np.left_shift(1, exponents.astype(np.int64))


def mesh_geometry(data):
    """Validate the native Neuroglancer mesh packet and describe its geometry."""
    if len(data) < 4:
        raise ValueError('Truncated mesh packet')
    vertices = int(np.frombuffer(data, dtype='<u4', count=1)[0])
    triangle_start = 4 + 12 * vertices
    if triangle_start > len(data) or (len(data) - triangle_start) % 12:
        raise ValueError('Invalid mesh packet layout')
    points = np.frombuffer(data, dtype='<f4', count=vertices * 3, offset=4).reshape(-1, 3)
    triangles = np.frombuffer(data, dtype='<u4', offset=triangle_start)
    if not np.isfinite(points).all() or (triangles.size and triangles.max() >= vertices):
        raise ValueError('Invalid mesh vertices or triangle indices')
    return {'vertices': vertices, 'triangles': int(triangles.size // 3), 'bytes': len(data),
            'local_bounds_nm': ([(points.min(0).astype(float) / 1e-9).tolist(),
                                 (points.max(0).astype(float) / 1e-9).tolist()]
                                if vertices else None)}


def _write_atomic(path, data):
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    try:
        temporary.write_bytes(data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class MeshVolume(neuroglancer.LocalVolume):
    """Native-resolution labels with independently sampled, cached 3D meshes.

    Data is treated as immutable, as with LocalVolume. Call invalidate() after
    an in-place edit to rebuild coarse labels and select a new cache fingerprint.
    Thin branches can change topology even if their segment ID survives sampling.
    Entire missing IDs fall back to native meshing, rather than being relabeled.
    """

    def __init__(self, data, dimensions, voxel_offset=None, *, mesh_isotropic=True,
                 mesh_target_nm=None, mesh_cache=None, **kwargs):
        array = np.asarray(data)
        if (array.ndim != 3 or any(size == 0 for size in array.shape)
                or array.dtype.kind not in 'ui' or np.any(array < 0)):
            raise ValueError('Meshes require a nonempty XYZ array of nonnegative integer labels')
        if array.dtype.kind == 'i':
            array = array.astype(np.uint64)  # Native meshing requires unsigned labels; values are unchanged.
        if dimensions.rank != 3 or list(dimensions.units) != ['m', 'm', 'm']:
            raise ValueError('Mesh dimensions must be three spatial axes with length units')
        kwargs.setdefault('volume_type', 'segmentation')
        if kwargs['volume_type'] != 'segmentation':
            raise ValueError('MeshVolume only supports segmentation sources')
        super().__init__(array, dimensions=dimensions, voxel_offset=voxel_offset, **kwargs)
        self._resolution_nm = np.asarray(dimensions.scales) / 1e-9
        self._isotropic = bool(mesh_isotropic)
        self._target_nm = mesh_target_nm
        self._factors = mesh_factors(self._resolution_nm, mesh_isotropic, mesh_target_nm)
        self._cache_root = Path(mesh_cache).resolve() if mesh_cache is not None else None
        self._display_mesh_lock = threading.RLock()
        self._prepare_mesh_data()

    def _prepare_mesh_data(self):
        array = np.asarray(self.data[:, :, :])
        coarse = array[tuple(slice(None, None, int(factor)) for factor in self._factors)]
        # Native and coarse vertex coordinates share a physical origin. Do not
        # integer-divide voxel_offset: registered ROI offsets may not divide evenly.
        coarse_dimensions = neuroglancer.CoordinateSpace(
            names=list(self.dimensions.names), units='m',
            scales=np.asarray(self.dimensions.scales) * self._factors)
        self._coarse_volume = neuroglancer.LocalVolume(
            coarse, dimensions=coarse_dimensions, voxel_offset=[0, 0, 0],
            volume_type='segmentation', mesh_options=self._mesh_options)
        self._native_ids = set(map(int, np.unique(array))) - {0}
        coarse_ids = set(map(int, np.unique(coarse))) - {0}
        missing = self._native_ids - coarse_ids
        self._mesh_bytes, self._object_reports = {}, {}
        metadata = {'algorithm': ALGORITHM, 'neuroglancer_version': version('neuroglancer'),
                    'source_sha256': hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest(),
                    'source_dtype': array.dtype.str, 'source_shape_xyz': list(array.shape),
                    'native_resolution_xyz_nm': self._resolution_nm.tolist(),
                    'native_voxel_offset_xyz': self.voxel_offset.tolist(),
                    'isotropic': self._isotropic,
                    'target_nm': float(self._target_nm) if self._target_nm is not None else None,
                    'downsample_xyz': self._factors.tolist(),
                    'mesh_resolution_xyz_nm': (self._resolution_nm * self._factors).tolist(),
                    'mesh_shape_xyz': list(coarse.shape), 'mesh_options': self._mesh_options,
                    'sampling': 'origin-anchored nearest-neighbor striding; mesh vertices in local physical units',
                    'native_fallback_ids': [str(label) for label in sorted(missing)],
                    'native_fallback_reason': 'label absent after downsampling',
                    'slices_unchanged': True}
        self._missing_ids = missing
        self._coarse_degenerate = min(coarse.shape) < 2
        fingerprint = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        self._cache_directory = self._cache_root / fingerprint if self._cache_root is not None else None
        if self._cache_directory is not None:
            self._cache_directory.mkdir(parents=True, exist_ok=True)
            _write_atomic(self._cache_directory / 'manifest.json',
                          (json.dumps(metadata, indent=2) + '\n').encode())
        self._metadata = dict(metadata, fingerprint=fingerprint,
                              cache_directory=str(self._cache_directory) if self._cache_directory else None)

    def get_object_mesh(self, object_id):
        object_id = int(object_id)
        with self._display_mesh_lock:
            if object_id not in self._native_ids:
                raise InvalidObjectIdForMesh()
            if object_id in self._mesh_bytes:
                return self._mesh_bytes[object_id]
            cache = self._cache_directory
            if cache is not None and (cache / f'{object_id}.json').is_file() and (cache / f'{object_id}.bin').is_file():
                report = json.loads((cache / f'{object_id}.json').read_text())
                data = (cache / f'{object_id}.bin').read_bytes()
                if (report.get('sha256') == hashlib.sha256(data).hexdigest()
                        and report.get('segment_id') == str(object_id)):
                    mesh_geometry(data)
                    self._mesh_bytes[object_id] = data
                    self._object_reports[object_id] = dict(report, source='disk_cache')
                    return data
                raise ValueError(f'Mesh cache checksum or ID mismatch: {cache / str(object_id)}')

            method = 'native' if np.all(self._factors == 1) else 'near_isotropic'
            if method == 'near_isotropic' and (object_id in self._missing_ids or self._coarse_degenerate):
                method = 'native_fallback_missing_label' if object_id in self._missing_ids else 'native_fallback_degenerate_grid'
            if method == 'near_isotropic':
                try:
                    data = self._coarse_volume.get_object_mesh(object_id)
                    if not mesh_geometry(data)['triangles']:
                        method = 'native_fallback_empty_mesh'
                except InvalidObjectIdForMesh:
                    method = 'native_fallback_empty_mesh'
            if method != 'near_isotropic':
                try:
                    data = super().get_object_mesh(object_id)
                except InvalidObjectIdForMesh:
                    # A uniform ROI has a valid label but no internal interface.
                    # Keep its 2D data usable and return an explicitly empty mesh.
                    data = np.array([0], dtype='<u4').tobytes()
            geometry = mesh_geometry(data)
            actual_resolution = self._resolution_nm * self._factors if method == 'near_isotropic' else self._resolution_nm
            report = dict(geometry, segment_id=str(object_id), method=method, source='computed',
                          mesh_resolution_xyz_nm=actual_resolution.tolist(),
                          empty_mesh=geometry['triangles'] == 0,
                          sha256=hashlib.sha256(data).hexdigest())
            if cache is not None:
                _write_atomic(cache / f'{object_id}.bin', data)
                _write_atomic(cache / f'{object_id}.json', (json.dumps(report, indent=2) + '\n').encode())
            self._mesh_bytes[object_id], self._object_reports[object_id] = data, report
            return data

    def mesh_report(self):
        with self._display_mesh_lock:
            return dict(self._metadata, objects={str(label): dict(report)
                                                 for label, report in self._object_reports.items()})

    def precompute_meshes(self, ids):
        for label in ids:
            self.get_object_mesh(label)
        return self.mesh_report()

    def invalidate(self):
        with self._display_mesh_lock:
            self._prepare_mesh_data()
            super().invalidate()
