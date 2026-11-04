"""M1 data engine, cleansing step 2: background myelin and mask glia/defects.

The paper's Data cleansing section states:

"Second, we set the myelin of AC3/AC4 and AxonEM as backgrounds and masked the
 glia and defects of CREMI, FIB25, HemiBrain, and J0126."

Procedure (following HQ-SAM/Ours/seg2mem_*.py):
1. Remove excluded labels from seg using common.BLACKLIST. The --blacklist-json
   option overrides or extends the per-dataset exclusions.
2. Generate cleaned labels and a validity mask: 0 for excluded/defective regions,
   1 elsewhere. Output layers use the configured suffix.
3. Combine excluded labels with existing defect masks, such as CREMI's msk layer.

Usage:
    python mask_myelin_glia.py --datasets snemi,ac3 --blacklist-json /path/to/bl.json
"""
import argparse
import os
import json
import numpy as np
from tqdm import tqdm

from common import BLACKLIST, ranges_of, cv_path, open_cv
from cloudvolume import CloudVolume


def clean_seg(seg, blacklist):
    """Set excluded labels to zero and return (seg_clean, mask_valid)."""
    seg = seg.copy()
    msk = np.ones_like(seg, dtype=np.uint8)
    for b in blacklist:
        hit = (seg == b)
        msk[hit] = 0
        seg[hit] = 0
    return seg, msk


def process(dataset, blacklist, write_msk=True, suffix='_clean'):
    from cloudvolume import CloudVolume
    seg_vol = open_cv(dataset, 'seg')
    if seg_vol is None:
        print(f'[{dataset}] No seg layer; skipping')
        return
    (xs, ys, zs), (xe, ye, ze) = ranges_of(dataset)[0]

    # Write seg/msk plus the suffix, preserving historical layers by default.
    dst_seg = CloudVolume('file://' + cv_path(dataset, 'seg') + suffix, info=seg_vol.info, non_aligned_writes=True, fill_missing=True)
    dst_seg.commit_info()
    dst_msk = None
    if write_msk:
        msk_path = cv_path(dataset, 'msk')
        if msk_path is None:   # No msk entry in location.py; place it beside seg.
            msk_path = os.path.join(os.path.dirname(cv_path(dataset, 'seg')), 'msk')
        info = CloudVolume.create_new_info(num_channels=1, layer_type='image',
                                           data_type='uint8', encoding='raw',
                                           resolution=seg_vol.resolution, voxel_offset=[0, 0, 0],
                                           chunk_size=list(seg_vol.chunk_size),
                                           volume_size=list(seg_vol.volume_size))
        dst_msk = CloudVolume('file://' + msk_path + suffix, info=info, non_aligned_writes=True, fill_missing=True)
        dst_msk.commit_info()

    src_msk = open_cv(dataset, 'msk')  # Existing defect mask, possibly None.
    mye_vol = None
    mye_dir = os.path.join(os.path.dirname(cv_path(dataset, 'seg')), 'mye')
    if os.path.isdir(mye_dir):        # Use a dedicated myelin layer if available.
        mye_vol = CloudVolume('file://' + mye_dir, mip=0, fill_missing=True)
        print(f'[{dataset}] Using myelin layer {mye_dir}')
    for z in tqdm(range(zs, ze), desc=dataset):
        seg = seg_vol[xs:xe, ys:ye, z][..., 0]
        seg_c, msk = clean_seg(seg, blacklist)
        if mye_vol is not None:
            mye = mye_vol[xs:xe, ys:ye, z][..., 0] > 0
            seg_c[mye] = 0
            msk[mye] = 0
        if src_msk is not None:
            msk *= (src_msk[xs:xe, ys:ye, z][..., 0] > 0).astype(np.uint8)
        dst_seg[xs:xe, ys:ye, z] = seg_c[..., None]
        if dst_msk is not None:
            dst_msk[xs:xe, ys:ye, z] = msk[..., None]
    print(f'[{dataset}] Done -> {cv_path(dataset, "seg")}b, msk')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', type=str, default='',
                    help='Comma-separated datasets; defaults to all datasets with exclusion lists')
    ap.add_argument('--blacklist-json', type=str, default=None,
                    help='Optional JSON: {dataset_name: [labels...]}, overriding common.BLACKLIST')
    ap.add_argument('--no-msk', action='store_true', help='Do not write the msk weight layer')
    args = ap.parse_args()

    bl = dict(BLACKLIST)
    if args.blacklist_json:
        with open(args.blacklist_json) as f:
            bl.update(json.load(f))

    names = [d for d in args.datasets.split(',') if d] if args.datasets else sorted(bl.keys())
    for name in names:
        print(f'== {name} ==')
        process(name, bl.get(name, []), write_msk=not args.no_msk)


if __name__ == '__main__':
    main()
