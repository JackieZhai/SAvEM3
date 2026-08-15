"""M5 · 提示图割 · LMC 增强聚合（论文 §2.3 Lifted multi-cut 段原话：

"Following beier2017multicut, we first extract edge features used as costs from
 the predicted affinity maps (i.e., 1 minus boundary maps). Then, we overlay two
 extra edge features: one is the cosine similarity of hq-feature vectors of two
 adjacent oversegments (our node features), and the other one is the IoU between
 the oversegment-seeded auto-prompting masks (our edge features). Finally, we
 solve this multi-cut problem with the Kernighan-Lin solver."

实现：基础代价 = transform_probabilities_to_costs(affinity 特征)；叠加提示特征——
代价 = 基础代价 − w_cos·cos(u,v) − w_iou·IoU(u,v) − w_ioa·IoA − w_iob·IoB
（多割中代价越低越倾向合并；IoU/IoA/IoB 三元组借鉴 SAM-Graph
[arXiv:2312.08372] edge_weights.py，默认权重只开 IoU 与余弦）；
然后用 multicut_kernighan_lin 求解、投影回体素。与 distill_postprocess.py 的 LMC
（无增强）形成对照。

用法：
    python lmc_prompt.py --fragments frag.h5 --affs out_affs.h5 \
        --edge-feats graph_feats/edge_feats.json [--w-cos 1.0 --w-iou 1.0] \
        [--w-ioa 0.0 --w-iob 0.0] [--gt AC3_labels.h5] --out-dir ./seg
"""
import argparse
import json
import os

import numpy as np


def load_h5(path):
    import h5py
    with h5py.File(path, 'r') as f:
        return f['main'][:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fragments', required=True)
    ap.add_argument('--affs', required=True, help='3ch affinity 或 1ch 膜概率 h5')
    ap.add_argument('--edge-feats', required=True, help='prompt_graph_cut.py 输出的 edge_feats.json')
    ap.add_argument('--w-cos', type=float, default=1.0)
    ap.add_argument('--w-iou', type=float, default=1.0)
    ap.add_argument('--w-ioa', type=float, default=0.0, help='SAM-Graph IoA 边特征权重')
    ap.add_argument('--w-iob', type=float, default=0.0, help='SAM-Graph IoB 边特征权重')
    ap.add_argument('--gt', default=None)
    ap.add_argument('--out-dir', default='./seg')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import elf.segmentation.features as feats
    import elf.segmentation.multicut as mc

    frag = load_h5(args.fragments).astype('uint64')
    affs = load_h5(args.affs).astype(np.float32)
    if affs.shape[0] == 1:
        affs3 = np.repeat(affs, 3, axis=0)          # 膜概率 → 近似 3ch affinity
    else:
        affs3 = affs

    rag = feats.compute_rag(frag)
    uv = rag.uvIds()
    try:
        costs = feats.compute_affinity_features(rag, affs3, [[-1, 0, 0], [0, -1, 0], [0, 0, -1]])[:, 0]
    except RuntimeError:
        costs = feats.compute_boundary_features(rag, 1.0 - np.max(affs3, axis=0))[:, 0]
    costs = mc.transform_probabilities_to_costs(costs)

    # 叠加提示特征（节点余弦 + 提示 IoU/IoA/IoB）
    with open(args.edge_feats) as f:
        ef = json.load(f)
    n_extra = 0
    for e in range(rag.numberOfEdges):
        u, v = int(uv[e, 0]), int(uv[e, 1])
        d = ef.get(f'{u},{v}') or ef.get(f'{v},{u}')
        if not d:
            continue
        extra = 0.0
        if 'cos' in d:
            extra += args.w_cos * d['cos']
        if 'iou' in d:
            extra += args.w_iou * d['iou']
        if 'ioa' in d:
            extra += args.w_ioa * d['ioa']
        if 'iob' in d:
            extra += args.w_iob * d['iob']
        costs[e] -= extra                            # 相似/重叠越高 → 代价越低 → 倾向合并
        n_extra += 1
    print(f'叠加提示特征：{n_extra}/{rag.numberOfEdges} 条边')

    seg = mc.multicut_kernighan_lin(rag, costs)
    seg = feats.project_node_labels_to_pixels(rag, seg).astype('uint64')

    import h5py
    with h5py.File(os.path.join(args.out_dir, 'seg_lmc_prompt.hdf'), 'w') as f:
        f.create_dataset('main', data=seg, compression='gzip')

    if args.gt:
        gt = load_h5(args.gt).astype('uint64')
        from skimage.metrics import adapted_rand_error, variation_of_information
        arand = adapted_rand_error(gt, seg, ignore_labels=(0,))[0]
        vs, vm = variation_of_information(gt, seg, ignore_labels=(0,))
        print(f'LMC+prompt: VoI={vs + vm:.4f} (split {vs:.4f} / merge {vm:.4f}), ARand={arand:.4f}')
    print(f'完成 -> {args.out_dir}/seg_lmc_prompt.hdf')


if __name__ == '__main__':
    main()
