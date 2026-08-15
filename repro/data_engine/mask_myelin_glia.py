"""M1 数据引擎 · 清洗第 2 步：髓鞘置背景 + 胶质/缺陷掩膜（论文 Data cleansing 原话：

"Second, we set the myelin of AC3/AC4 and AxonEM as backgrounds and masked the
 glia and defects of CREMI, FIB25, HemiBrain, and J0126."

做法（与 HQ-SAM/Ours/seg2mem_*.py 同款逻辑）：
1. 把黑名单标签（common.BLACKLIST，可用 --blacklist-json 覆盖/补充）从 seg 中剔除；
2. 剔除后生成 segb（清洗后标签）与 msk（权重图层：黑名单/缺陷处为 0，其余为 1）；
3. 缺陷区域优先使用数据自带 msk 图层（cremi 等），与黑名单取并集。

用法：
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
    """把黑名单标签置 0，返回 (seg_clean, mask_valid)。"""
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
        print(f'[{dataset}] 无 seg 图层，跳过')
        return
    (xs, ys, zs), (xe, ye, ze) = ranges_of(dataset)[0]

    # 目标层：seg+后缀 / msk+后缀（默认不覆盖原有 segm1b/segm2b 等历史层）
    dst_seg = CloudVolume('file://' + cv_path(dataset, 'seg') + suffix, info=seg_vol.info, non_aligned_writes=True, fill_missing=True)
    dst_seg.commit_info()
    dst_msk = None
    if write_msk:
        msk_path = cv_path(dataset, 'msk')
        if msk_path is None:   # location.py 未登记 msk 图层 → 放在 seg 同级
            msk_path = os.path.join(os.path.dirname(cv_path(dataset, 'seg')), 'msk')
        info = CloudVolume.create_new_info(num_channels=1, layer_type='image',
                                           data_type='uint8', encoding='raw',
                                           resolution=seg_vol.resolution, voxel_offset=[0, 0, 0],
                                           chunk_size=list(seg_vol.chunk_size),
                                           volume_size=list(seg_vol.volume_size))
        dst_msk = CloudVolume('file://' + msk_path + suffix, info=info, non_aligned_writes=True, fill_missing=True)
        dst_msk.commit_info()

    src_msk = open_cv(dataset, 'msk')  # 数据自带缺陷掩膜（可能为 None）
    mye_vol = None
    mye_dir = os.path.join(os.path.dirname(cv_path(dataset, 'seg')), 'mye')
    if os.path.isdir(mye_dir):        # 存在专用髓鞘图层（如 snemi）则直接用它掩膜
        mye_vol = CloudVolume('file://' + mye_dir, mip=0, fill_missing=True)
        print(f'[{dataset}] 使用髓鞘图层 {mye_dir}')
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
    print(f'[{dataset}] 完成 -> {cv_path(dataset, "seg")}b, msk')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', type=str, default='',
                    help='逗号分隔；默认全部有黑名单配置的数据集')
    ap.add_argument('--blacklist-json', type=str, default=None,
                    help='可选 JSON：{数据集名: [标签...]}，覆盖 common.BLACKLIST')
    ap.add_argument('--no-msk', action='store_true', help='不写 msk 权重图层')
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
