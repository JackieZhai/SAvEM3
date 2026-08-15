"""M1 数据引擎 · 挑选 188K 训练集（论文 Data selection 原话：

"We picked around 188K representative masks from the total of 649K masks ...
 Specifically, we chose 9 of 20 ATUM-SEM volumes, 2 of 3 FIB-SEM, 2 of 3 ssTEM,
 and 94 of 312 SBEM with nine volumes stitched into a volume. For each epoch
 during training, we randomly sample 30K instances from each imaging method to
 further balance them. Note that approximately half of the SegEM region is
 masked, so we need to double its number of instances."

实现：
1. 按 common.IMAGING_METHODS 分组，每组按 --n-volumes 配额取前 N 个体数据；
2. 遍历所选体数据的 seg 图层，逐片收集实例（label, 面积），直到总掩码数达到配额
   （--total 188000；SBEM 中 SegEM 实例数按 ×2 计入，模拟"一半区域被掩膜"）；
3. 输出两份产物：
   - record2d_train.json  : {dataset: {z: [label_ids...]}}，与 SALEM2 json2d_create.py
     输出同构，可直接喂给 trainMemProISBI_2d_v4.py 的 dataloader；
   - sampling_manifest.json: 每成像法 30K/epoch 的采样规则与统计（649K→188K 的构成）。

用法：
    python select_training_set.py --total 188000 --n-volumes 9,2,2,94 \
        --out-json record2d_train.json --out-manifest sampling_manifest.json
"""
import argparse
import json

import numpy as np

from common import IMAGING_METHODS, METHOD_OF, ranges_of, open_cv


def collect_instances(dataset, zs, ze, seg_vol, segem_double=False):
    """遍历一个体数据，yield (dataset, z, label, area)。"""
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
                    help='各成像法取卷数：ATUM-SEM,FIB-SEM,ssTEM,SBEM（论文 9/2/2/94）')
    ap.add_argument('--methods', type=str, default='ATUM-SEM,FIB-SEM,ssTEM,SBEM')
    ap.add_argument('--out-json', type=str, default='record2d_train.json')
    ap.add_argument('--out-manifest', type=str, default='sampling_manifest.json')
    ap.add_argument('--segem-double', action='store_true',
                    help='SegEM 实例数 ×2（论文：近半区域被掩膜）')
    args = ap.parse_args()

    quotas = [int(x) for x in args.n_volumes.split(',')]
    methods = [m for m in args.methods.split(',') if m]

    selected = {}
    for method, n in zip(methods, quotas):
        vols = [v for v in IMAGING_METHODS.get(method, [])][:n]
        selected[method] = vols
        print(f'{method}: 选 {len(vols)}/{len(IMAGING_METHODS.get(method, []))} 卷 -> {vols}')

    record = {}
    stats = {'total_masks': 0, 'per_method': {}, 'per_dataset': {}}
    for method, vols in selected.items():
        cnt_method = 0
        for v in vols:
            seg_vol = open_cv(v, 'seg')
            if seg_vol is None:
                print(f'[跳过] {v} 无 seg 图层')
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
        'sampling_rule': '每 epoch 每成像法随机采样 30000 实例（SBEM 中 SegEM 已按 ×2 计入）',
    }
    with open(args.out_manifest, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'完成：{stats["total_masks"]} 掩码 -> {args.out_json} / {args.out_manifest}')


if __name__ == '__main__':
    main()
