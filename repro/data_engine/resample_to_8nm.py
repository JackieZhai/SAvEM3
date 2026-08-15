"""M1 数据引擎 · 清洗第 1 步：xy 分辨率统一到 8nm（论文 Data cleansing 原话：

"we adjusted the xy-plane resolution of all datasets (including 4, 6, 8, 9 and
 11.24nm) to 4nm for SAM and 8nm for HQ-SAM, SAEM² and U-Nets"）

本脚本生成 8nm 版本（SAEM²/SAvEM³ 使用；SAM 用的 4nm 版本把 --target 改成 4 即可）。
输出：在源图层同级新建 `<layer>8nm` CloudVolume 层。
z 轴分辨率不变；仅对 x/y 做重采样（img/msk/mem/dst 双线性，seg 最近邻）。

用法：
    python resample_to_8nm.py --datasets snemi,ac3,cremi_b --target 8 --layers img,seg,mem,dst,msk
    python resample_to_8nm.py --all --target 8
"""
import argparse
import os
import numpy as np
import cv2
from tqdm import tqdm

from common import (data_location_list, ranges_of, cv_path, open_cv)
from cloudvolume import CloudVolume


def new_info(src_vol, scale_xy, layer, new_size, target_nm):
    """按 8nm 重采样后的目标 CloudVolume info（volume 尺寸 = 重采样后尺寸）。"""
    dt = src_vol.dtype.name
    enc = 'compressed_segmentation' if (layer == 'seg' and dt in ('uint32', 'uint64')) else 'raw'
    info = CloudVolume.create_new_info(
        num_channels=1,
        layer_type='segmentation' if layer == 'seg' else 'image',
        data_type=dt,
        encoding=enc,
        resolution=[target_nm, target_nm, int(src_vol.resolution[2])],
        voxel_offset=[0, 0, 0],
        chunk_size=[256, 256, 64],
        volume_size=[int(new_size[0]), int(new_size[1]), int(new_size[2])],
    )
    return info


def resample_layer(dataset, layer, target_nm=8.0):
    src = open_cv(dataset, layer)
    if src is None:
        print(f'  [{dataset}] 无 {layer} 图层，跳过')
        return
    res = src.resolution  # [x,y,z] nm
    scale_xy = res[0] / target_nm
    if abs(scale_xy - 1.0) < 1e-3:
        print(f'  [{dataset}] {layer} 已是 {target_nm}nm，跳过')
        return
    (xs, ys, zs), (xe, ye, ze) = ranges_of(dataset)[0]
    new_size = [int(round((xe - xs) * scale_xy)), int(round((ye - ys) * scale_xy)), ze - zs]

    dst_path = cv_path(dataset, layer) + f'{target_nm:.0f}nm'
    if os.path.isdir(dst_path):
        print(f'  [{dataset}] {layer}8nm 已存在，跳过')
        return
    dst = CloudVolume('file://' + dst_path, info=new_info(src, scale_xy, layer, new_size, target_nm), non_aligned_writes=True, fill_missing=True)
    dst.commit_info()

    interp = cv2.INTER_NEAREST if layer == 'seg' else cv2.INTER_LINEAR
    for z in tqdm(range(zs, ze), desc=f'{dataset}/{layer}'):
        patch = src[xs:xe, ys:ye, z][..., 0, 0]
        if patch.dtype in (np.uint32, np.uint64, np.int32, np.int64, np.uint16, np.int16):
            # cv2 不支持 CV_32U 等 dtype 的 resize → numpy 最近邻索引（标签安全）
            h, w = patch.shape
            yy = np.minimum((np.arange(new_size[0]) * h) // new_size[0], h - 1)
            xx = np.minimum((np.arange(new_size[1]) * w) // new_size[1], w - 1)
            out = patch[yy][:, xx]
        else:
            out = cv2.resize(patch, (new_size[1], new_size[0]), interpolation=interp)
        dst[0:new_size[0], 0:new_size[1], z - zs] = out.astype(src.dtype)[..., None]
    print(f'  [{dataset}] {layer} -> {dst_path} ({new_size})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', type=str, default='',
                    help='逗号分隔的数据集名（snemi/ac3/axonem-h_0-0-0/...）')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--target', type=float, default=8.0, help='目标 xy 分辨率 nm（论文：SAM=4，其余=8）')
    ap.add_argument('--layers', type=str, default='img,seg,mem,dst,msk')
    args = ap.parse_args()

    names = list(data_location_list.keys()) if args.all else [d for d in args.datasets.split(',') if d]
    layers = [l for l in args.layers.split(',') if l]
    for name in names:
        print(f'== {name} ==')
        for layer in layers:
            resample_layer(name, layer, args.target)


if __name__ == '__main__':
    main()
