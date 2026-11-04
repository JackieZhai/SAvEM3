"""M1 data engine, cleansing step 3: realign CREMI sections.
The paper's Data cleansing section states: "Third, we realigned the CREMI dataset."

Estimate (dy, dx) translations between adjacent z sections using
skimage.phase_cross_correlation. Accumulate these shifts and write translated
img/seg layers in the first section's coordinate frame.
Outputs: new sibling layers named `img_al` and `seg_al`.

Usage:
    python realign_cremi.py --datasets cremi_a,cremi_b,cremi_c --max-shift 20
"""
import argparse
import numpy as np
from tqdm import tqdm

from common import ranges_of, cv_path, open_cv
from cloudvolume import CloudVolume


def estimate_shift(ref, mov, max_shift=20, upsample=20):
    """Estimate the integer (dy, dx) alignment shift, clipped to max_shift."""
    from skimage.registration import phase_cross_correlation
    shift, _, _ = phase_cross_correlation(ref, mov, upsample_factor=upsample)
    shift = np.clip(np.round(shift).astype(int), -max_shift, max_shift)
    return int(shift[0]), int(shift[1])   # (dy, dx)


def apply_shift(img, dy, dx, cval):
    return np.roll(np.roll(img, dy, axis=0), dx, axis=1)


def process(dataset, max_shift):
    img_vol = open_cv(dataset, 'img')
    seg_vol = open_cv(dataset, 'seg')
    if img_vol is None:
        print(f'[{dataset}] No img layer; skipping')
        return
    (xs, ys, zs), (xe, ye, ze) = ranges_of(dataset)[0]

    dst_img = CloudVolume('file://' + cv_path(dataset, 'img') + '_al', info=img_vol.info, non_aligned_writes=True, fill_missing=True)
    dst_img.commit_info()
    dst_seg = None
    if seg_vol is not None:
        dst_seg = CloudVolume('file://' + cv_path(dataset, 'seg') + '_al', info=seg_vol.info, non_aligned_writes=True, fill_missing=True)
        dst_seg.commit_info()

    # Estimate shifts against the previous raw section and accumulate them.
    cum_dy = cum_dx = 0
    shifts = [(0, 0)]
    prev = None
    print(f'[{dataset}] Estimating shifts ...')
    for z in tqdm(range(zs, ze)):
        cur = img_vol[xs:xe, ys:ye, z][..., 0]
        if prev is None:
            prev = cur
            continue
        dy, dx = estimate_shift(prev, cur, max_shift)
        cum_dy += dy
        cum_dx += dx
        shifts.append((cum_dy, cum_dx))
        prev = cur
    shifts = np.array(shifts, dtype=int)

    print(f'[{dataset}] Writing aligned layers (maximum accumulated shift {np.abs(shifts).max(axis=0)}) ...')
    for i, z in enumerate(tqdm(range(zs, ze))):
        dy, dx = shifts[i]
        im = img_vol[xs:xe, ys:ye, z][..., 0]
        dst_img[xs:xe, ys:ye, z] = apply_shift(im, dy, dx, cval=im.mean())[..., None]
        if dst_seg is not None:
            sg = seg_vol[xs:xe, ys:ye, z][..., 0]
            dst_seg[xs:xe, ys:ye, z] = apply_shift(sg, dy, dx, cval=0)[..., None]
    np.save(cv_path(dataset, 'img') + '_al_shifts.npy', shifts)
    print(f'[{dataset}] Done -> img_al / seg_al (shifts saved to *_al_shifts.npy)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', type=str, default='cremi_a,cremi_b,cremi_c')
    ap.add_argument('--max-shift', type=int, default=20)
    args = ap.parse_args()
    for name in [d for d in args.datasets.split(',') if d]:
        print(f'== {name} ==')
        process(name, args.max_shift)


if __name__ == '__main__':
    main()
