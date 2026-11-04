"""M1 data engine: select a 188K-mask training set.

The paper's Data selection section states:

"We picked around 188K representative masks from the total of 649K masks ...
 Specifically, we chose 9 of 20 ATUM-SEM volumes, 2 of 3 FIB-SEM, 2 of 3 ssTEM,
 and 94 of 312 SBEM with nine volumes stitched into a volume. For each epoch
 during training, we randomly sample 30K instances from each imaging method to
 further balance them. Note that approximately half of the SegEM region is
 masked, so we need to double its number of instances."

Implementation:
1. Group volumes by common.IMAGING_METHODS and take the first N volumes in each
   group according to --n-volumes.
2. Scan the selected seg layers section by section, collecting instance labels
   and areas until --total is reached (default: 188000 masks). The optional
   SegEM adjustment doubles reported areas; it does not duplicate record entries.
3. Write two outputs:
   - record2d_train.json: {dataset: {z: [label_ids...]}}, matching the format of
     SALEM2 json2d_create.py for the trainMemProISBI_2d_v4.py dataloader.
   - sampling_manifest.json: selection statistics and the intended 30K instances
     per imaging method per epoch. This script does not perform epoch sampling.

Usage:
    python select_training_set.py --total 188000 --n-volumes 9,2,2,94 \
        --out-json record2d_train.json --out-manifest sampling_manifest.json
"""
import argparse
import json

import numpy as np

from common import IMAGING_METHODS, METHOD_OF, ranges_of, open_cv


def collect_instances(dataset, zs, ze, seg_vol, segem_double=False):
    """Scan a volume and yield (dataset, z, label, area)."""
    from tqdm import tqdm
    (xs, ys, _), (xe, ye, _) = ranges_of(dataset)[0]
    for z in tqdm(range(zs, ze), desc=dataset, leave=False):
        seg = seg_vol[xs:xe, ys:ye, z][..., 0]
        labels, counts = np.unique(seg, return_counts=True)
        for lab, cnt in zip(labels, counts):
            if lab == 0:
                continue
            yield dataset, int(z), int(lab), int(cnt) * (2 if segem_double else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--total', type=int, default=188000)
    ap.add_argument('--n-volumes', type=str, default='9,2,2,94',
                    help='Volume counts for ATUM-SEM,FIB-SEM,ssTEM,SBEM (paper: 9/2/2/94)')
    ap.add_argument('--methods', type=str, default='ATUM-SEM,FIB-SEM,ssTEM,SBEM')
    ap.add_argument('--out-json', type=str, default='record2d_train.json')
    ap.add_argument('--out-manifest', type=str, default='sampling_manifest.json')
    ap.add_argument('--segem-double', action='store_true',
                    help='Double reported SegEM instance areas (paper motivates oversampling masked SegEM regions)')
    args = ap.parse_args()

    quotas = [int(x) for x in args.n_volumes.split(',')]
    methods = [m for m in args.methods.split(',') if m]

    selected = {}
    for method, n in zip(methods, quotas):
        vols = [v for v in IMAGING_METHODS.get(method, [])][:n]
        selected[method] = vols
        print(f'{method}: selected {len(vols)}/{len(IMAGING_METHODS.get(method, []))} volumes -> {vols}')

    record = {}
    stats = {'total_masks': 0, 'per_method': {}, 'per_dataset': {}}
    for method, vols in selected.items():
        cnt_method = 0
        for v in vols:
            seg_vol = open_cv(v, 'seg')
            if seg_vol is None:
                print(f'[skip] {v}: no seg layer')
                continue
            (_, _, zs), (_, _, ze) = ranges_of(v)[0]
            cnt_ds = 0
            for ds, z, lab, area in collect_instances(
                    v, zs, ze, seg_vol, segem_double=args.segem_double and v.startswith('segem')):
                if stats['total_masks'] >= args.total:
                    break
                record.setdefault(ds, {}).setdefault(str(z), []).append(lab)
                stats['total_masks'] += 1
                cnt_method += 1
                cnt_ds += 1
            stats['per_dataset'][v] = cnt_ds
            if stats['total_masks'] >= args.total:
                break
        stats['per_method'][method] = cnt_method
        if stats['total_masks'] >= args.total:
            break

    with open(args.out_json, 'w') as f:
        json.dump(record, f)
    manifest = {
        'selected_volumes': selected,
        'stats': stats,
        'sampling_rule': 'Intended training policy: sample 30000 instances per imaging method per epoch; this selection script does not implement epoch sampling or SegEM instance duplication',
    }
    with open(args.out_manifest, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'Done: {stats["total_masks"]} masks -> {args.out_json} / {args.out_manifest}')


if __name__ == '__main__':
    main()
