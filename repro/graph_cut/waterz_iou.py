"""M5 · 提示图割 · WaterZ 加权 IoU 版（论文 §2.3 WaterZ 段原话：

"To simplify the code, we only use a weighted IoU score (our edge features) here
 to increase the maximum affinity between the adjacent oversegments. Following
 funke2018large, we initialize the edge scores by using 1 minus the maximum
 affinity between the oversegments and adopting the quantile merge function to
 update them."

实现选择（对论文"简化版"的体素级实现）：
    对每对相邻 fragment (u,v)，取两者交界处体素，把该处膜概率提升为
        aff_new = max(aff, IoU(u,v))
    （IoU 越高 → 交界亲和度越接近 1 → waterz 按 OneMinus<HistogramQuantileAffinity>
    打分时更易合并）。之后按标准 waterz 流程聚合。

用法：
    python waterz_iou.py --affs out_affs.h5 --fragments frag.h5 \
        --edge-feats graph_feats/edge_feats.json [--gt AC3_labels.h5] --out-dir ./seg
"""
import argparse
import json
import os

import numpy as np


def load_h5(path):
    import h5py
    with h5py.File(path, 'r') as f:
        return f['main'][:]


def boost_interfaces(affs, frag, edge_feats, dilate=1):
    """按边 IoU 提升相邻 fragment 交界处的最大亲和度。"""
    from scipy import ndimage
    out = affs.copy()
    for key, d in edge_feats.items():
        if 'iou' not in d or d['iou'] <= 0:
            continue
        u, v = map(int, key.split(','))
        if u == 0 or v == 0:
            continue
        mu = frag == u
        mv = frag == v
        # 交界 = 双方各自膨胀后的交叠带（体素级近似）
        iface = np.logical_and(ndimage.binary_dilation(mu, iterations=dilate),
                               ndimage.binary_dilation(mv, iterations=dilate))
        if iface.sum() == 0:
            continue
        # 提升 z 向与面内最大亲和（affs 为 3ch affinity 时取 max；1ch 膜概率时直接提升概率）
        if out.shape[0] >= 3:
            mx = out[:, iface].max(axis=0, keepdims=True)
            boost = np.maximum(mx, d['iou'])
            for c in range(min(3, out.shape[0])):
                out[c][iface] = np.maximum(out[c][iface], boost[0])
        else:
            out[0][iface] = np.maximum(out[0][iface], d['iou'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--affs', required=True)
    ap.add_argument('--fragments', required=True)
    ap.add_argument('--edge-feats', required=True, help='edge_feats.json（含 iou 字段）')
    ap.add_argument('--gt', default=None)
    ap.add_argument('--out-dir', default='./seg')
    ap.add_argument('--thresholds', type=str, default='0.3,0.4,0.5')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    affs = load_h5(args.affs).astype(np.float32)
    frag = load_h5(args.fragments).astype('uint64')
    with open(args.edge_feats) as f:
        ef = json.load(f)

    print('1) 按提示 IoU 提升交界亲和度 ...')
    affs = boost_interfaces(affs, frag, ef)

    print('2) waterz 聚合（OneMinus<HistogramQuantileAffinity>）...')
    import waterz
    sf = 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>'
    thresholds = [float(t) for t in args.thresholds.split(',')]
    segs = list(waterz.agglomerate(affs.astype(np.float32), thresholds,
                                   fragments=frag, scoring_function=sf,
                                   discretize_queue=256))
    import h5py
    if args.gt:
        gt = load_h5(args.gt).astype('uint64')
        from skimage.metrics import adapted_rand_error, variation_of_information
        print('\n== WaterZ+IoU 评测（VoI↓ / ARand↓）==')
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
    print(f'完成 -> {args.out_dir}/')


if __name__ == '__main__':
    main()
