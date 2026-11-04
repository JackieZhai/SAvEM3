"""M6: a z-path approximation related to Expected Run Length (ERL).

Paper section 3: "We adopt an established metric for long-range connectivity analysis:
the expected running length (ERL)."

Implementation (sample a path through a ground-truth neuron):
1. Read a binary GT mask H5, or extract --neuron-id from a label H5, together with
   a predicted instance segmentation H5.
2. Sample predicted segment IDs at the GT mask's median coordinates per z section.
3. Measure consecutive nonzero, unchanged IDs along this path and report the mean
   run length in sections and physical units.

This is not the official skeleton-based ERL statistic (januszewski2018high).
It is a voxel-level z-path approximation; compare units and site definitions with
the official J0126 tools before using it in a reproduction.

Usage:
    python erl_approx.py --gt-path neuron_gt.h5 --pred-seg seg_waterz_t0.5.hdf \
        [--neuron-id 2252715458] [--resolution 8,8,33]
"""
import argparse

import numpy as np


def load_h5(path):
    import h5py
    with h5py.File(path, 'r') as f:
        return f['main'][:]


def erl_voxel_path(gt_bin, seg, resolution):
    """Return the mean consistent z-path run length in sections and nanometers."""
    # Use the per-axis median of each GT section as the path point.
    zs = np.nonzero(gt_bin.any(axis=(1, 2)))[0]
    path = []
    for z in zs:
        ys, xs = np.nonzero(gt_bin[z])
        path.append((z, int(np.median(ys)), int(np.median(xs))))
    if not path:
        return 0.0, 0

    # Predicted segment IDs along the sampled path.
    seg_ids = np.array([seg[z, y, x] for z, y, x in path], dtype=np.int64)
    runs, run_len = [], 0
    prev_id = seg_ids[0]
    prev_z = path[0][0]
    for (z, _, _), sid in zip(path, seg_ids):
        gap = z - prev_z
        if sid != 0 and sid == prev_id and gap <= 1:
            run_len += gap
        else:
            if run_len > 0:
                runs.append(run_len)
            run_len = 0 if sid == 0 else 1
        prev_id, prev_z = sid, z
    if run_len > 0:
        runs.append(run_len)
    if not runs:
        return 0.0, 0
    zr = resolution[2]  # nm per section
    return float(np.mean(runs)), float(np.mean(runs)) * zr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt-path', required=True, help='Ground-truth mask or label H5')
    ap.add_argument('--pred-seg', required=True)
    ap.add_argument('--neuron-id', type=int, default=None,
                    help='Extract this instance from GT labels; otherwise treat GT as a binary mask')
    ap.add_argument('--resolution', type=str, default='8,8,33', help='nm per voxel (the third value is z)')
    args = ap.parse_args()

    gt = load_h5(args.gt_path)
    if args.neuron_id is not None:
        gt_bin = (gt == args.neuron_id)
    else:
        gt_bin = gt > 0
    seg = load_h5(args.pred_seg)
    res = [float(x) for x in args.resolution.split(',')]

    mean_vox, mean_nm = erl_voxel_path(gt_bin, seg, res)
    print(f'ERL = {mean_vox:.1f} sections, approximately {mean_nm / 1000:.2f} µm (z-path approximation)')
    print('Note: the paper reports H01 #2252715458 ERL=12267 (SAvEM³) / 10184 (Superhuman). '
          'Calibrate units and site definitions against the official tools.')


if __name__ == '__main__':
    main()
