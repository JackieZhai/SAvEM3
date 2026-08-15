"""M6 · 3D 稠密评测汇总（论文 Table 3 格式：VoI↓ / ARand↓，WaterZ 与 LMC 两路）。

输入多个分割 h5 与 GT，输出对齐的指标表。支持把"论文复现目标行"一并打印对照。

用法：
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
    from skimage.metrics import adapted_rand_error, variation_of_information
    arand = adapted_rand_error(gt, seg, ignore_labels=(0,))[0]
    vs, vm = variation_of_information(gt, seg, ignore_labels=(0,))
    return vs, vm, vs + vm, arand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', required=True)
    ap.add_argument('--preds', required=True, help='逗号分隔的分割 h5/hdf 路径')
    ap.add_argument('--names', default=None, help='逗号分隔的显示名（默认用文件名）')
    args = ap.parse_args()

    gt = load_h5(args.gt).astype('uint64')
    preds = [p for p in args.preds.split(',') if p]
    names = ([n for n in args.names.split(',') if n] if args.names else
             [p.rsplit('/', 1)[-1] for p in preds])
    assert len(names) == len(preds), '--names 数量需与 --preds 一致'

    print(f'{"Method":<22}{"VoI-split":>10}{"VoI-merge":>10}{"VoI":>10}{"ARand":>10}')
    print('-' * 62)
    for name, path in zip(names, preds):
        seg = load_h5(path).astype('uint64')
        vs, vm, voi, arand = metrics(gt, seg)
        print(f'{name:<22}{vs:>10.4f}{vm:>10.4f}{voi:>10.4f}{arand:>10.4f}')
    print('\n论文目标（SAvEM³ 最终行，0% 标注）：')
    print('  AC3 WaterZ: VoI=1.289 ARand=0.123 | LMC: VoI=1.362 ARand=0.115')
    print('  CREMI-C WaterZ: VoI=1.599 ARand=0.154 | LMC: VoI=1.610 ARand=0.169')


if __name__ == '__main__':
    main()
