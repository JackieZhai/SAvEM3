"""Evaluate bounded SAM2 propagation on local precomputed neuron ground truth.

Locate annotated targets at a coarse mip, read bounded fine-resolution windows,
and compare original/ECC/SOFIMA inputs with identical native prompts. Ground truth
is used only for window selection, one central positive point and scoring; no
propagated-section labels are supplied to the predictor.
"""
import argparse
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np
from cloudvolume import CloudVolume
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_alignment import compare
from export_savem3 import export_bundle, write_precomputed
from probe_em.demo_alignment import prepare_field, registered_bounds
from probe_em.prompt_models import PointModel
from probe_em.propagation import PropagationGrid


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def volume(path, mip=0):
    return CloudVolume('file://'+str(Path(path).resolve()), mip=mip, parallel=False,
                       fill_missing=False, progress=False)


def metrics(prediction, truth):
    tp = int(np.count_nonzero(prediction & truth))
    predicted, target = int(prediction.sum()), int(truth.sum())
    union = predicted+target-tp
    return {'intersection': tp, 'predicted': predicted, 'target': target,
            'iou': tp/union if union else 1., 'dice': 2*tp/(predicted+target) if predicted+target else 1.,
            'precision': tp/predicted if predicted else None, 'recall': tp/target if target else None}


def interior_point(mask):
    """Maximize distance from both annotation boundaries and the crop boundary."""
    mask = np.asarray(mask, bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError('Interior-point selection requires a nonempty 2D target')
    # Without padding, EDT can select a crop-edge pixel when an object extends
    # outside the ROI. Such points become fragile after registration padding.
    distance = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    return np.unravel_index(distance.argmax(), mask.shape)


def sample_gt(labels, factors):
    """Sample voxel centers from a finer, integer-aligned instance-label grid."""
    factors = np.asarray(factors)
    if factors.shape != (3,) or np.any(factors < 1) or np.any(factors != factors.astype(int)):
        raise ValueError('GT sampling factors must be three positive integers')
    return labels[tuple(slice(int(f//2), None, int(f)) for f in factors)].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-path', required=True, type=Path)
    parser.add_argument('--gt-path', required=True, type=Path)
    parser.add_argument('--raw-mip', type=int, default=1)
    parser.add_argument('--gt-mip', type=int, default=0, help='GT scale sampled for scoring (default: finest)')
    parser.add_argument('--locator-mip', type=int, help='Default: coarsest GT scale')
    parser.add_argument('--labels', type=int, nargs='+', help='Default: all nonzero coarse GT labels')
    parser.add_argument('--xy', type=int, default=1024)
    parser.add_argument('--slices', type=int, default=9)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--model-config', default='sam2_hiera_t.yaml')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--methods', nargs='+', choices=['unregistered', 'ecc', 'sofima'],
                        default=['unregistered', 'ecc'])
    parser.add_argument('--sofima-python', help='Isolated SOFIMA Python; required with --methods sofima')
    parser.add_argument('--max-voxels', type=int, default=64_000_000,
                        help='Maximum raw voxels per ROI; also limits the coarse GT locator')
    parser.add_argument('--max-gt-voxels', type=int, default=256_000_000,
                        help='Maximum GT voxels before resampling; prevents oversized fine-grid reads')
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    if args.xy < 64 or args.slices < 3 or args.slices % 2 == 0:
        parser.error('Use XY >=64 and an odd section count >=3')
    if args.xy**2*args.slices > args.max_voxels:
        parser.error('Requested ROI exceeds --max-voxels')
    if 'sofima' in args.methods and not args.sofima_python:
        parser.error('--methods sofima requires --sofima-python')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'status.json', {'status': 'running'})
    started = time.monotonic()
    try:
        raw_volume, gt_volume = volume(args.raw_path, args.raw_mip), volume(args.gt_path, args.gt_mip)
        if raw_volume.info['type'] != 'image' or gt_volume.info['type'] != 'segmentation':
            raise ValueError('Raw must be an image volume and GT must be an instance-label segmentation volume')
        if raw_volume.info['num_channels'] != 1 or gt_volume.info['num_channels'] != 1:
            raise ValueError('Evaluation requires single-channel raw and GT volumes')
        locator_mip = len(gt_volume.info['scales'])-1 if args.locator_mip is None else args.locator_mip
        locator = volume(args.gt_path, locator_mip)
        if np.prod(locator.bounds.size(), dtype=object) > args.max_voxels:
            raise ValueError('Choose a coarser --locator-mip; locator exceeds --max-voxels')
        coarse = np.asarray(locator[:])[..., 0]
        available = sorted(int(i) for i in np.unique(coarse) if i)
        labels = available if args.labels is None else list(dict.fromkeys(args.labels))
        if not labels or set(labels)-set(available):
            raise ValueError('Requested labels must be present in coarse ground truth')
        resolution = np.asarray(raw_volume.resolution, float)
        ratio = resolution/np.asarray(gt_volume.resolution)
        if not np.allclose(ratio, np.rint(ratio)) or np.any(ratio < 1):
            raise ValueError('Raw grid must be an integer coarsening of the selected GT grid')
        ratio = np.rint(ratio).astype(int)
        if args.xy**2*args.slices*int(np.prod(ratio)) > args.max_gt_voxels:
            raise ValueError('Fine GT read exceeds --max-gt-voxels; reduce the ROI or select an appropriate --gt-mip')
        model = PointModel('sam2', args.checkpoint, model_config=args.model_config, device=args.device)
        reports = []
        for label in labels:
            coordinates = np.argwhere(coarse == label)
            occupied_z = np.unique(coordinates[:, 2])
            coarse_z = int(occupied_z[np.abs(occupied_z-np.median(coordinates[:, 2])).argmin()])
            components, _ = ndimage.label(coarse[:, :, coarse_z] == label)
            sizes = np.bincount(components.ravel()); sizes[0] = 0
            component = components == sizes.argmax()
            cx, cy = interior_point(component)
            center_nm = (np.array([cx, cy, coarse_z])+np.asarray(locator.voxel_offset)+.5)*locator.resolution
            center = np.floor(center_nm/resolution).astype(int)
            shape = np.array([args.xy, args.xy, args.slices])
            if np.any(shape > raw_volume.bounds.size()):
                raise ValueError('Requested ROI exceeds the raw volume extent')
            lower = np.clip(center-shape//2, raw_volume.bounds.minpt, raw_volume.bounds.maxpt-shape).astype(int)
            upper = lower+shape
            raw = np.asarray(raw_volume[tuple(slice(int(a), int(b)) for a,b in zip(lower,upper))])[..., 0]
            gt_lower, gt_upper = lower*ratio, upper*ratio
            if np.any(gt_lower < gt_volume.bounds.minpt) or np.any(gt_upper > gt_volume.bounds.maxpt):
                raise ValueError('GT does not cover the selected raw ROI')
            fine_gt = np.asarray(gt_volume[tuple(slice(int(a),int(b)) for a,b in zip(gt_lower,gt_upper))])[..., 0]
            # Center-aligned nearest-neighbor labels, never average instance IDs.
            gt = sample_gt(fine_gt, ratio)
            del fine_gt
            if gt.shape != raw.shape:
                raise ValueError('Resampled GT and raw grids disagree')
            anchor = args.slices//2
            target = gt == label
            if not target[:, :, anchor].any():
                raise ValueError(f'Label {label} absent from the central fine-resolution section')
            px, py = interior_point(target[:, :, anchor])
            points = np.array([[px+lower[0], py+lower[1], anchor+lower[2]]], float)
            point_labels = np.array([1], np.int32)
            directory = output/f'neuron_{label}'
            export_bundle(raw.transpose(2,1,0), gt.transpose(2,1,0), directory/'cache', resolution, lower)
            write_precomputed(directory/'cache'/'target', target.astype(np.uint8), resolution, lower, True)
            fields, infos = {}, {}
            for name in dict.fromkeys(args.methods):
                if name == 'unregistered':
                    fields[name] = None
                    continue
                field, info = prepare_field(raw, resolution, lower, cache_dir=directory/name/'alignment',
                                             ref_z=int(points[0,2]), backend=name,
                                             sofima_python=args.sofima_python if name == 'sofima' else None)
                fields[name], infos[name] = field, info
            registration = compare(raw, gt, resolution, lower, dict(unregistered=None, **{
                name: field for name, field in fields.items() if name != 'unregistered'}))
            write_json(directory/'registration.json', dict(registration, fields=infos))
            methods = {}
            for name, field in fields.items():
                folder = directory/name
                folder.mkdir(parents=True, exist_ok=True)
                aligned_shape, aligned_offset = registered_bounds(raw.shape, lower, field, 128_000_000) if field else (None,None)
                grid = PropagationGrid(raw, lower, int(lower[2]), int(upper[2]-1), field=field,
                                       shape=aligned_shape, origin=aligned_offset)
                frames = grid.encode(folder/'frames', threading.Event())
                prompts = grid.prompts(points, point_labels)
                masks = np.zeros((args.slices, raw.shape[1], raw.shape[0]), bool)
                seen = set()
                for index, mask in model.propagate(folder/'frames', prompts, cancel=threading.Event()):
                    if index in seen or not 0 <= index < args.slices:
                        raise ValueError('Unexpected video frame index')
                    masks[index] = grid.native_mask(mask, int(lower[2])+index)
                    seen.add(index)
                if seen != set(range(args.slices)):
                    raise ValueError('Propagation omitted requested sections')
                truth = target.transpose(2,1,0)
                heldout = np.arange(args.slices) != anchor
                per_section = [dict(z=int(lower[2])+i, prompted=i==anchor, **metrics(mask, truth[i]))
                               for i, mask in enumerate(masks)]
                score = dict(heldout=metrics(masks[heldout], truth[heldout]),
                             prompted=metrics(masks[anchor],truth[anchor]), per_section=per_section,
                             other_annotated_neuron_voxels=int(np.count_nonzero(masks &
                                 ((gt != 0) & (gt != label)).transpose(2,1,0))),
                             gt_touches_xy_boundary=bool(target[[0,-1]].any() or target[:,[0,-1]].any()),
                             model=model.metadata, encoded_frames=frames)
                np.savez_compressed(folder/'prediction.npz', mask_zyx=masks, frame_z=np.arange(lower[2],upper[2]),
                                    points_xyz=points, point_labels=point_labels,
                                    voxel_offset_xyz=lower, resolution_xyz_nm=resolution)
                write_precomputed(folder/'mask', masks.transpose(2,1,0).astype(np.uint8), resolution, lower, True)
                write_json(folder/'metrics.json', score)
                methods[name] = score
                print(json.dumps({'label':label,'method':name,'heldout':score['heldout']}),flush=True)
            report = dict(label=label, offset_xyz=lower.tolist(), shape_xyz=raw.shape,
                          resolution_xyz_nm=resolution.tolist(), points_xyz=points.tolist(),
                          methods=methods, registration_mean_ncc=registration['mean_ncc'])
            write_json(directory/'report.json', report)
            reports.append(report)
            write_json(output/'report.json', dict(raw_source=str(args.raw_path.resolve()), gt_source=str(args.gt_path.resolve()),
                       roi_selection='All selected coarse labels; nearest occupied median z; largest component; maximum crop-aware interior distance',
                       point_selection='One central-section positive; maximum distance from target and crop boundaries',
                       raw_mip=args.raw_mip, locator_mip=locator_mip, gt_mip=args.gt_mip,
                       gt_resampling='Center-aligned nearest-neighbor sampling from the selected GT scale',
                       scope='Bounded ROI target-instance agreement; assumes each target annotation is complete within the ROI',
                       prompted_and_heldout_reported_separately=True, neurons=reports))
        write_json(output/'status.json', {'status':'complete','elapsed_seconds':time.monotonic()-started})
    except KeyboardInterrupt:
        write_json(output/'status.json', {'status':'cancelled','elapsed_seconds':time.monotonic()-started})
        raise
    except Exception as error:
        write_json(output/'status.json', {'status':'failed','error':f'{type(error).__name__}: {error}'})
        raise


if __name__ == '__main__':
    main()
