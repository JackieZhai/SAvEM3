"""M4/M6 · 蒸馏模型的后处理管线：膜图 → 过分割 → 2D multicut 预整合 → WaterZ/LMC 聚合。

论文 Implementation Details 原话：
    "After the step of distance transformation-based watersheds for boundary maps,
     we first preprocess oversegmentation using the 2D multi-cut as an initial
     integration of fragments."

流程（与 scripts_savem3/main_devoem_sparse_membrane_triplet_2.py 的验证段一致，并加上
论文要求的 2D multicut 预整合）：
    1. 输入膜概率图（模型滑窗推理输出 out_affs，1ch，sigmoid 后 0~1）；
    2. elf distance_transform_watershed（阈值/σ 可调）生成 fragments；
    3. 2D multicut 逐片预整合（elf.compute_rag + boundary features + kernighan-lin）；
    4. WaterZ：waterz.agglomerate(OneMinus<HistogramQuantileAffinity<...>>)；
    5. LMC：elf 3D multicut（lmc.mc_baseline 同款）；
    6. 保存 seg_waterz.hdf / seg_lmc.hdf / seg_mc2d.hdf，并对 GT 计算 VoI/ARand。

用法：
    python distill_postprocess.py --affs out_affs.h5 --gt AC3_labels.h5 \
        --out-dir ./postprocess --ws-threshold 0.25 --sigma-seeds 2.0
"""
import argparse
import os

import numpy as np


def load_h5_main(path):
    import h5py
    with h5py.File(path, 'r') as f:
        return f['main'][:]


def watershed_oversegment(affs, threshold=0.25, sigma_seeds=2.0):
    """距离变换分水岭过分割。affs: (1,Z,H,W) 边界概率。"""
    import elf.segmentation.watershed as ws
    bmap = affs[0]
    fragments, _ = ws.distance_transform_watershed(bmap, threshold=threshold, sigma_seeds=sigma_seeds)
    return fragments.astype('uint64')


def premerge_2d_multicut(fragments, bmap, beta=0.5):
    """2D multicut 逐片预整合（论文 Implementation Details）。"""
    import elf.segmentation.features as feats
    import elf.segmentation.multicut as mc
    out = np.zeros_like(fragments)
    for z in range(fragments.shape[0]):
        frag = fragments[z]
        rag = feats.compute_rag(frag)
        costs = feats.compute_boundary_features(rag, bmap[z])[:, 0]
        costs = mc.transform_probabilities_to_costs(costs, edge_sizes=costs, beta=beta)
        seg = mc.multicut_kernighan_lin(rag, costs)
        out[z] = feats.project_node_labels_to_pixels(rag, seg)
    return out


def waterz_agglomerate(affs, fragments, thresholds=[0.3, 0.4, 0.5]):
    import waterz
    sf = 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>'
    segs = list(waterz.agglomerate(affs.astype(np.float32), thresholds,
                                   fragments=fragments, scoring_function=sf,
                                   discretize_queue=256))
    return {t: s.astype('uint64') for t, s in zip(thresholds, segs)}


def lmc_agglomerate(affs):
    """3D multicut（scripts_savem3/utils/lmc.py::mc_baseline 同款）。"""
    import elf.segmentation.features as feats
    import elf.segmentation.multicut as mc
    import elf.segmentation.watershed as ws
    bmap = np.maximum(affs[1], affs[2]) if affs.shape[0] >= 3 else affs[0]
    fragments = np.stack([ws.distance_transform_watershed(bmap[z], threshold=0.25, sigma_seeds=2.0)[0]
                          for z in range(bmap.shape[0])])
    rag = feats.compute_rag(fragments)
    try:
        costs = feats.compute_affinity_features(rag, affs, [[-1, 0, 0], [0, -1, 0], [0, 0, -1]])[:, 0]
    except RuntimeError:
        costs = feats.compute_boundary_features(rag, bmap)[:, 0]
    costs = mc.transform_probabilities_to_costs(costs)
    seg = mc.multicut_kernighan_lin(rag, costs)
    return feats.project_node_labels_to_pixels(rag, seg).astype('uint64')


def evaluate(seg, gt):
    from skimage.metrics import adapted_rand_error, variation_of_information
    arand = adapted_rand_error(gt, seg, ignore_labels=(0,))[0]
    vs, vm = variation_of_information(gt, seg, ignore_labels=(0,))
    return arand, vs + vm, vs, vm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--affs', required=True, help='模型输出的膜概率图 h5（main: (1,Z,H,W) 或 (3,Z,H,W)）')
    ap.add_argument('--gt', default=None, help='GT 标签 h5（可选，给则算 VoI/ARand）')
    ap.add_argument('--out-dir', default='./postprocess')
    ap.add_argument('--ws-threshold', type=float, default=0.25)
    ap.add_argument('--sigma-seeds', type=float, default=2.0)
    ap.add_argument('--skip-2d-mc', action='store_true', help='跳过 2D multicut 预整合（对照）')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    affs = load_h5_main(args.affs).astype(np.float32)
    if affs.shape[0] == 3:                      # 3ch affinity（旧模型）→ 边界 = 1-max
        bmap = 1.0 - np.max(affs, axis=0)
        affs1 = affs[None, ...]
    else:
        bmap = affs[0]
        affs1 = affs

    print('1) 距离变换分水岭过分割 ...')
    frag = watershed_oversegment(affs1[:1] if affs1.shape[0] > 1 else affs1, args.ws_threshold, args.sigma_seeds)
    print(f'   fragments: {len(np.unique(frag))}')

    if not args.skip_2d_mc:
        print('2) 2D multicut 预整合 ...')
        frag = premerge_2d_multicut(frag, bmap)
        print(f'   after 2D-multicut: {len(np.unique(frag))}')
        import h5py
        with h5py.File(os.path.join(args.out_dir, 'seg_mc2d.hdf'), 'w') as f:
            f.create_dataset('main', data=frag, compression='gzip')

    print('3) WaterZ 聚合 ...')
    waterz_input = np.repeat(affs1[0:1], 3, axis=0) if affs1.shape[0] == 1 else affs1
    segs_wz = waterz_agglomerate(waterz_input, frag)
    for t, s in segs_wz.items():
        import h5py
        with h5py.File(os.path.join(args.out_dir, f'seg_waterz_t{t}.hdf'), 'w') as f:
            f.create_dataset('main', data=s, compression='gzip')

    print('4) LMC 聚合 ...')
    lmc_input = np.repeat(affs1[0][None, ...], 3, axis=0) if affs1.shape[0] == 1 else affs1
    seg_lmc = lmc_agglomerate(lmc_input)
    import h5py
    with h5py.File(os.path.join(args.out_dir, 'seg_lmc.hdf'), 'w') as f:
        f.create_dataset('main', data=seg_lmc, compression='gzip')

    if args.gt:
        gt = load_h5_main(args.gt).astype('uint64')
        print('\n== 评测（VoI↓ / ARand↓）==')
        for t, s in segs_wz.items():
            arand, voi, vs, vm = evaluate(s, gt)
            print(f'WaterZ t={t}: VoI={voi:.4f} (split {vs:.4f} / merge {vm:.4f}), ARand={arand:.4f}')
        arand, voi, vs, vm = evaluate(seg_lmc, gt)
        print(f'LMC      : VoI={voi:.4f} (split {vs:.4f} / merge {vm:.4f}), ARand={arand:.4f}')


if __name__ == '__main__':
    main()
