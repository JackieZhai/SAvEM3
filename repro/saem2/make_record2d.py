"""M2 准备 · 把数据银行的 CloudVolume 图层导出为 SAEM² 所需的 2D 逐层 tif。

dataloader 约定（saem2/utils/dataloader_isbi_2d_v4.py）：
    prepared_segments_mul_mem/{ds}/%05d.tif    膜图（img>0 为膜）
    prepared_segments_mul_2d/{ds}/%05d.tif    实例标签（json2d_create.py 由膜图 mem2label 生成）
    prepared_embedding/{ds}/embed/%05d.tif    预计算 SAM 嵌入（savem3/precompute_teacher.py 生成）
    record2d_train.json                        {dataset: {layer: [label_id...]}}

本脚本负责第一步：从 location.py 的 mem 图层（若无则用 seg 图层按 seg2mem 逻辑
fastmorph.erode 后取 ==0 现场生成膜图）导出逐层 mem tif。
之后依次运行：python json2d_create.py → savem3/precompute_teacher.py --write-tif。

数据集名映射（location.py 名 → SALEM2 名）：
    snemi→SNEMI, ac3→AC3, cremi_a→cremiA, cremi_b→cremiB, cremi_c→cremiC,
    fib25→FIB25, hemibrain_*→HB-<区名>, axonem-h_<x>-<y>-<z>→AxonEM-H/seg_<x>-<y>-<z>,
    axonem-m_*→AxonEM-M/seg_*, j0126_*→J0126/<块名>, segem_*→SegEM/<块名>

用法：
    python make_record2d.py --datasets snemi,ac3 --out-root $SAVEM3_DATA_ROOT --mode mem
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_engine'))
from common import DATA_ROOT, ranges_of, cv_path, open_cv  # noqa: E402

from skimage import io  # noqa: E402
import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402


def salem2_name(dataset):
    """location.py 数据集名 → SALEM2 记录名。"""
    if dataset == 'snemi':
        return 'SNEMI'
    if dataset == 'ac3':
        return 'AC3'
    if dataset.startswith('cremi'):
        return {'cremi_a': 'cremiA', 'cremi_b': 'cremiB', 'cremi_c': 'cremiC'}.get(dataset, dataset)
    if dataset == 'fib25':
        return 'FIB25'
    if dataset.startswith('hemibrain'):
        return 'HB-' + dataset.split('_', 1)[1]          # HB-fb-inner ...
    if dataset.startswith('axonem-h'):
        return 'AxonEM-H/seg_' + dataset.split('_', 1)[1]  # seg_950-0-0
    if dataset.startswith('axonem-m'):
        return 'AxonEM-M/seg_' + dataset.split('_', 1)[1]
    if dataset.startswith('j0126'):
        return 'J0126/' + dataset.split('_', 1)[1]
    if dataset.startswith('segem'):
        return 'SegEM/' + dataset.split('_', 1)[1]
    return dataset


def export_mem_tifs(dataset, out_root):
    """导出 mem 层逐片 tif；无 mem 层时按 seg2mem 逻辑现场生成膜图。"""
    ds_name = salem2_name(dataset)
    out_dir = os.path.join(out_root, 'prepared_segments_mul_mem', ds_name)
    os.makedirs(out_dir, exist_ok=True)

    mem_vol = open_cv(dataset, 'mem')
    seg_vol = open_cv(dataset, 'seg')
    if mem_vol is None and seg_vol is None:
        print(f'[{dataset}] 无 mem/seg 图层，跳过')
        return
    (xs, ys, zs), (xe, ye, ze) = ranges_of(dataset)[0]

    need_erode = mem_vol is None
    for z in tqdm(range(zs, ze), desc=ds_name):
        if need_erode:
            import fastmorph
            seg = seg_vol[xs:xe, ys:ye, z][..., 0].copy()
            seg = fastmorph.erode(seg, parallel=4)
            mem = ((seg == 0).astype(np.uint8)) * 255
        else:
            mem = (mem_vol[xs:xe, ys:ye, z][..., 0] > 0).astype(np.uint8) * 255
        io.imsave(os.path.join(out_dir, '%05d.tif' % (z - zs)), mem)
    print(f'[{dataset}] -> {out_dir}（{ze - zs} 片）')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', type=str, default='snemi,ac3')
    ap.add_argument('--out-root', type=str, default=None,
                    help='默认 $SAVEM3_DATA_ROOT（即 prepared_segments_mul_mem 的父目录）')
    args = ap.parse_args()
    out_root = args.out_root or DATA_ROOT
    for name in [d for d in args.datasets.split(',') if d]:
        print(f'== {name} ==')
        export_mem_tifs(name, out_root)


if __name__ == '__main__':
    main()
