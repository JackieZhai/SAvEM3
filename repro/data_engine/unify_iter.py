"""M1 data engine: model-in-the-loop label unification.
The paper describes iterative model refinement and boundary-only updates:

"We iteratively train our model and save the high-confidential masks. We also
 used the mask prompting of finetuned HQ-SAM and SAEM² and the alternative NMS
 method to filter duplicated masks. Here, we modified masks with IoU between
 before and after higher than 0.7, which means we optimized the original
 boundary only. This cycle was repeated several times."

This script implements mask updates; model training remains an external loop:
train SAEM²/HQ-SAM, refine with sam_refine.py or SAEM², then update the data bank.

1. Read --pred-layer (seg_refine or connected-component SAEM² predictions).
2. Match predictions to --seg-layer section by section using per-label IoU.
3. Replace boundaries only for IoU > 0.7 (--iou-threshold);
   retain original labels otherwise.
4. Retain larger masks when prediction IoU exceeds --nms-iou. Note that
   distinct masks in a discrete label map do not overlap; this is not proposal NMS.
5. Write --out-layer (seg_unified) for one round; log the round with --round.

Usage:
    python unify_iter.py --datasets snemi --seg-layer segb --pred-layer seg_refine \
        --out-layer seg_unified --round 1
"""
import argparse

import numpy as np
from tqdm import tqdm

from common import ranges_of, cv_path, open_cv
from cloudvolume import CloudVolume


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / max(union, 1)


def unify_slice(seg_orig, seg_pred, iou_threshold=0.7, nms_iou=0.9):
    """Update one section with predictions matching original labels above the IoU threshold."""
    out = seg_orig.copy()
    pred_labels = np.unique(seg_pred)
    pred_labels = pred_labels[pred_labels != 0]

    # Sort predictions by area and discard masks overlapping retained predictions.
    kept_preds = []
    areas = [(l, int((seg_pred == l).sum())) for l in pred_labels]
    areas.sort(key=lambda t: -t[1])
    for l, _ in areas:
        m = seg_pred == l
        if any(iou(m, km) > nms_iou for km in kept_preds):
            continue
        kept_preds.append(m)

    # Match each prediction to an original label.
    orig_labels = np.unique(seg_orig)
    orig_labels = orig_labels[orig_labels != 0]
    assigned = np.zeros_like(seg_orig, dtype=bool)
    for m in kept_preds:
        if assigned[m].sum() / max(m.sum(), 1) > 0.5:
            continue  # This region has already been assigned.
        best_iou, best_lab = 0.0, 0
        for ol in orig_labels:
            v = iou(m, seg_orig == ol)
            if v > best_iou:
                best_iou, best_lab = v, ol
        if best_iou > iou_threshold and best_lab != 0:
            # Replacing a mask must remove its old boundary as well as grow it.
            out[(seg_orig == best_lab) & ~assigned] = 0
            m = m & (~assigned) & ((seg_orig == best_lab) | (seg_orig == 0))
            out[m] = best_lab      # Replace the original boundary with the refined mask.
            assigned |= m
    return out


def process(dataset, seg_layer, pred_layer, out_layer, iou_threshold, nms_iou, round_no):
    seg_vol = open_cv(dataset, seg_layer)
    pred_vol = open_cv(dataset, pred_layer)
    if seg_vol is None or pred_vol is None:
        print(f'[{dataset}] Missing layer ({seg_layer}/{pred_layer}); skipping')
        return
    (xs, ys, zs), (xe, ye, ze) = ranges_of(dataset)[0]
    dst = CloudVolume('file://' + cv_path(dataset, out_layer),
                      info=seg_vol.info, non_aligned_writes=True, fill_missing=True)
    dst.commit_info()

    for z in tqdm(range(zs, ze), desc=dataset):
        seg = seg_vol[xs:xe, ys:ye, z][..., 0]
        pred = pred_vol[xs:xe, ys:ye, z][..., 0]
        out = unify_slice(seg, pred, iou_threshold, nms_iou)
        dst[xs:xe, ys:ye, z] = out.astype(seg.dtype)[..., None]
    print(f'[{dataset}] round {round_no} complete -> {out_layer}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', type=str, default='snemi')
    ap.add_argument('--seg-layer', type=str, default='segb', help='Current/original label layer')
    ap.add_argument('--pred-layer', type=str, default='seg_refine', help='Model-refined mask layer')
    ap.add_argument('--out-layer', type=str, default='seg_unified')
    ap.add_argument('--iou-threshold', type=float, default=0.7, help='Update boundaries only above IoU 0.7 in the paper')
    ap.add_argument('--nms-iou', type=float, default=0.9)
    ap.add_argument('--round', type=int, default=1)
    args = ap.parse_args()

    for name in [d for d in args.datasets.split(',') if d]:
        print(f'== {name} ==')
        process(name, args.seg_layer, args.pred_layer, args.out_layer,
                args.iou_threshold, args.nms_iou, args.round)


if __name__ == '__main__':
    main()
