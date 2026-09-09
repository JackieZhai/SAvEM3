"""Shared data-engine tools: dataset locations and CloudVolume I/O.

Dataset settings are loaded from sam-hq/train/utils/location.py:
    data_location_list  : {dataset: relative paths for [img, seg, mem, dst, msk]}
    data_range_list     : {dataset: [[xs,ys,zs],[xe,ye,ze]] volume bounds}
    data_label_range_list: {dataset: label bounds, which may differ from volume bounds}
    data_resolution_list: {dataset: [x,y,z] nm/voxel}
"""
import os
import sys
import importlib.util

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
DATA_ROOT = os.environ.get('SAVEM3_DATA_ROOT', REPO_ROOT)

_HQ_TRAIN = None
for _candidate in (os.path.join(REPO_ROOT, 'sam-hq', 'train'), os.path.join(DATA_ROOT, 'sam-hq', 'train'), os.path.join(DATA_ROOT, 'HQ-SAM', 'sam-hq', 'train')):
    if os.path.isdir(_candidate):
        _HQ_TRAIN = _candidate
        break
if _HQ_TRAIN is None:
    raise RuntimeError('Cannot find sam-hq/train; set SAVEM3_DATA_ROOT.')
_spec = importlib.util.spec_from_file_location('savem3_data_locations', os.path.join(_HQ_TRAIN, 'utils', 'location.py'))
_locations = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_locations)
data_location_list = _locations.data_location_list
data_range_list = _locations.data_range_list
data_label_range_list = _locations.data_label_range_list
data_resolution_list = _locations.data_resolution_list

# Imaging-method groups (Table 1 in the paper).
IMAGING_METHODS = {
    'ATUM-SEM': ['snemi', 'ac3'] + [k for k in data_location_list if k.startswith('axonem-h')],
    'FIB-SEM':  ['fib25'] + [k for k in data_location_list if k.startswith('hemibrain')],
    'ssTEM':    [k for k in data_location_list if k.startswith(('axonem-m', 'cremi'))],
    'SBEM':     [k for k in data_location_list if k.startswith(('j0126', 'segem'))],
}
METHOD_OF = {name: m for m, names in IMAGING_METHODS.items() for name in names}

# Known excluded labels (myelin/glia/defects), from HQ-SAM/Ours/seg2mem_*.py.
BLACKLIST = {
    'snemi':  [401],                # SNEMI3D myelin
    'ac3':    [401],                # AC3 myelin (same-source annotation)
    'axonem-h': [],                 # AxonEM-H: myelin annotated separately; extend as needed
    'axonem-m': [],
    'fib25':  [],
    'hemibrain': [317, 326, 370, 340],   # Glia (see seg2mem_hemibrain.py for eb/fb/pb blocks)
    'cremi':  [],                   # Defect masks use msk (CREMI msk includes defect weights)
    'j0126':  [],                   # Add myelin labels as needed (see seg2mem_j0126.py)
    'segem':  [],
}


def cv_path(dataset, layer, root=None):
    """Return a dataset layer's absolute local path for file:// access."""
    root = root or DATA_ROOT
    layers = ['img', 'seg', 'mem', 'dst', 'msk']
    if layer not in layers:
        if layer not in ('seg_refine', 'seg_unified', 'segb'):
            raise ValueError(f'Unknown layer: {layer}')
        base = cv_path(dataset, 'seg', root)
        if base is None:
            return None
        return base if layer == 'segb' else base + '_' + layer.removeprefix('seg_')
    rel = data_location_list[dataset][layers.index(layer)]
    if rel is None:
        return None
    for base in ('data', 'HQ-SAM', 'Ours'):
        p = os.path.join(root, base, rel)
        if os.path.exists(p):
            return p
    return os.path.join(root, 'data', rel)

def open_cv(dataset, layer, mip=0, root=None):
    from cloudvolume import CloudVolume
    p = cv_path(dataset, layer, root=root)
    if p is None:
        return None
    return CloudVolume('file://' + p, mip=mip, fill_missing=False)


def ranges_of(dataset):
    """Return (volume bounds, label bounds, resolution)."""
    return (data_range_list[dataset], data_label_range_list.get(dataset, data_range_list[dataset]),
            data_resolution_list[dataset])


def encoder_plane(plane_xy, resolution_xyz, target_nm=4.0, labels=False):
    """Native CloudVolume XY → calibrated, center-cropped 1024 YX encoder ROI."""
    import numpy as np
    import cv2
    plane = np.asarray(plane_xy).squeeze().T
    if plane.ndim != 2 or target_nm <= 0:
        raise ValueError('Expected one XY section and a positive target resolution')
    nx = int(round(plane.shape[1] * resolution_xyz[0] / target_nm))
    ny = int(round(plane.shape[0] * resolution_xyz[1] / target_nm))
    if min(nx, ny) < 1024:
        raise ValueError('Physical ROI is smaller than 1024 encoder pixels; select a larger ROI or smaller --xy-nm')
    if labels:
        yi = np.minimum(np.arange(ny)*plane.shape[0]//ny, plane.shape[0]-1)
        xi = np.minimum(np.arange(nx)*plane.shape[1]//nx, plane.shape[1]-1)
        resized = plane[yi][:, xi]
    else:
        resized = cv2.resize(plane.astype(np.float32), (nx, ny), interpolation=cv2.INTER_LINEAR)
    top, left = ny//2-512, nx//2-512
    return np.ascontiguousarray(resized[top:top+1024, left:left+1024])
