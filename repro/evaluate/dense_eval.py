"""M6: dense 3D evaluation (Table 3 format: VoI/ARand, lower is better; WaterZ/MC).

Read multiple segmentation H5 files and GT, then print a metric table.
Paper reference values are printed separately, not as measured results.

Usage:
    python dense_eval.py --gt AC3_labels.h5 \
        --preds superhuman/seg_waterz.hdf savem3/seg_waterz_t0.5.hdf savem3/seg_lmc_prompt.hdf \
        --names Superhuman SAvEM3-waterz SAvEM3-lmc-prompt
"""
import argparse

import numpy as np


def load_h5(path):
    import h5py
    with h5py.File(path, 'r') as f:
        return f['main'][:]


def metrics(gt, seg):
    if gt.shape != seg.shape or not np.any(gt):
        raise ValueError('Evaluation requires matching shapes and nonempty foreground GT')
    from skimage.metrics import adapted_rand_error, variation_of_information
    arand = adapted_rand_error(gt, seg, ignore_labels=(0,))[0]
    vs, vm = variation_of_information(gt, seg, ignore_labels=(0,))
    return vs, vm, vs + vm, arand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', required=True)
    ap.add_argument('--preds', required=True, nargs='+', help='Space- or comma-separated segmentation h5/hdf paths')
    ap.add_argument('--names', default=None, nargs='+', help='Space- or comma-separated display names (default: filenames)')
    args = ap.parse_args()

    gt = load_h5(args.gt).astype('uint64')
    preds = [p for group in args.preds for p in group.split(',') if p]
    names = ([n for group in args.names for n in group.split(',') if n] if args.names else
             [p.rsplit('/', 1)[-1] for p in preds])
    assert len(names) == len(preds), '--names and --preds must have equal lengths'

    print(f'{"Method":<22}{"VoI-split":>10}{"VoI-merge":>10}{"VoI":>10}{"ARand":>10}')
    print('-' * 62)
    for name, path in zip(names, preds):
        seg = load_h5(path).astype('uint64')
        vs, vm, voi, arand = metrics(gt, seg)
        print(f'{name:<22}{vs:>10.4f}{vm:>10.4f}{voi:>10.4f}{arand:>10.4f}')
    print('\nPaper reference values (final SAvEM³ row, 0% labels; not measured here):')
    print('  AC3 WaterZ: VoI=1.289 ARand=0.123 | LMC: VoI=1.362 ARand=0.115')
    print('  CREMI-C WaterZ: VoI=1.599 ARand=0.154 | LMC: VoI=1.610 ARand=0.169')


if __name__ == '__main__':
    main()
