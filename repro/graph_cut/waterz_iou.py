"""WaterZ with SAM prompt IoU evidence on matching voxel interfaces.

A one-channel input is boundary probability and is converted to affinity first.
Only the negative-axis edge joining the specified u/v pair is boosted. This is
an explicit voxel-edge approximation, not a custom WaterZ graph-score kernel.
See README.md for paper and implementation differences.
"""
import argparse
import json
import os

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repro.volume import prediction_maps, neighbour_slices, validate_labels


def load_h5(path):
    import h5py
    with h5py.File(path, 'r') as f:
        return f['main'][:]


def boost_interfaces(affs, frag, edge_feats, weight=1.0):
    """Boost only the directed voxel edge joining the specified fragment pair.

    No dilation: unrelated interfaces and other affinity directions are untouched.
    This is a voxel-level approximation of the paper's edge-score modification.
    """
    validate_labels(frag)
    _, affinity = prediction_maps(affs)
    if affinity.shape[1:] != frag.shape or not 0 <= weight <= 1:
        raise ValueError('Mismatched shapes or IoU weight outside [0,1]')
    out = affinity.copy()
    scores = {}
    for key, evidence in edge_feats.items():
        u, v = sorted(map(int, key.split(',')))
        score = float(evidence.get('iou', 0))
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('IoU evidence must be a finite probability')
        if u and v:
            scores[(u, v)] = weight * score
    for axis in range(3):
        current, previous = neighbour_slices(axis)
        a, b = frag[current], frag[previous]
        mask = (a != b) & (a != 0) & (b != 0)
        if not mask.any():
            continue
        pairs, inverse = np.unique(np.sort(np.stack((a[mask], b[mask]), axis=1), axis=1),
                                   axis=0, return_inverse=True)
        values = np.array([scores.get((int(u), int(v)), 0.0) for u, v in pairs])[inverse]
        plane = out[axis][current]
        plane[mask] = np.maximum(plane[mask], values)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--affs', required=True)
    ap.add_argument('--fragments', required=True)
    ap.add_argument('--edge-feats', required=True, help='edge_feats.json containing iou fields')
    ap.add_argument('--gt', default=None)
    ap.add_argument('--out-dir', default='./seg')
    ap.add_argument('--thresholds', type=str, default='0.3,0.4,0.5')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    affs = load_h5(args.affs).astype(np.float32)
    frag = load_h5(args.fragments).astype('uint64')
    with open(args.edge_feats) as f:
        ef = json.load(f)

    print('1) Boost interface affinities using prompt IoU ...')
    affs = boost_interfaces(affs, frag, ef)

    print('2) WaterZ agglomeration (OneMinus<HistogramQuantileAffinity>)...')
    from repro.savem3.distill_postprocess import waterz_agglomerate
    thresholds = [float(t) for t in args.thresholds.split(',')]
    segs = list(waterz_agglomerate(affs, frag, thresholds).values())
    import h5py
    if args.gt:
        gt = load_h5(args.gt).astype('uint64')
        from skimage.metrics import adapted_rand_error, variation_of_information
        print('\n== WaterZ+IoU evaluation (VoI / ARand; lower is better)==')
        for t, s in zip(thresholds, segs):
            s = s.astype('uint64')
            with h5py.File(os.path.join(args.out_dir, f'seg_waterz_iou_t{t}.hdf'), 'w') as f:
                f.create_dataset('main', data=s, compression='gzip')
            arand = adapted_rand_error(gt, s, ignore_labels=(0,))[0]
            vs, vm = variation_of_information(gt, s, ignore_labels=(0,))
            print(f't={t}: VoI={vs + vm:.4f} (split {vs:.4f} / merge {vm:.4f}), ARand={arand:.4f}')
    else:
        for t, s in zip(thresholds, segs):
            with h5py.File(os.path.join(args.out_dir, f'seg_waterz_iou_t{t}.hdf'), 'w') as f:
                f.create_dataset('main', data=s.astype('uint64'), compression='gzip')
    print(f'Done -> {args.out_dir}/')


if __name__ == '__main__':
    main()
