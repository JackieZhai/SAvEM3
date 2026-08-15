"""M6 · 大体积稀疏评测指标：Expected Run Length（ERL，januszewski2018high）。

论文 §3："We adopt an established metric for long-range connectivity analysis:
the expected running length (ERL)."

实现（沿 GT 神经元的中心线路径评测）：
1. 输入：gt_path 为 GT 神经元的二值掩码 h5（或标签 h5 + --neuron-id 提取）；
   pred_seg 为预测实例分割 h5；
2. 沿 GT 掩码的 z 方向主路径，逐体素检查预测段是否与 GT 相同实例（追踪一致性）；
3. ERL = 沿路径"预测正确连续段"长度的期望（以体素/物理长度计），
   即每次预测在相邻正确体素间断裂时重置游程，输出平均游程长度。

注：官方 ERL 定义基于 skeleton 站点游程统计；本实现采用体素级 z-主路径近似，
复现时请与 januszewski2018high 的 J0126 工具核对换算（site 级）。

用法：
    python erl.py --gt-path neuron_gt.h5 --pred-seg seg_waterz_t0.5.hdf \
        [--neuron-id 2252715458] [--resolution 8,8,33]
"""
import argparse

import numpy as np


def load_h5(path):
    import h5py
    with h5py.File(path, 'r') as f:
        return f['main'][:]


def erl_voxel_path(gt_bin, seg, resolution):
    """沿 GT 掩码 z 向主路径计算体素级 ERL（返回平均正确游程，物理长度 nm）。"""
    # 每片取 GT 掩码质心作为路径点
    zs = np.nonzero(gt_bin.any(axis=(1, 2)))[0]
    path = []
    for z in zs:
        ys, xs = np.nonzero(gt_bin[z])
        path.append((z, int(np.median(ys)), int(np.median(xs))))
    if not path:
        return 0.0, 0

    # 预测段 ID 序列
    seg_ids = np.array([seg[z, y, x] for z, y, x in path], dtype=np.int64)
    runs, run_len = [], 0
    prev_id = seg_ids[0]
    prev_z = path[0][0]
    for (z, _, _), sid in zip(path, seg_ids):
        gap = z - prev_z
        if sid != 0 and sid == prev_id and gap <= 1:
            run_len += gap
        else:
            if run_len > 0:
                runs.append(run_len)
            run_len = 0 if sid == 0 else 1
        prev_id, prev_z = sid, z
    if run_len > 0:
        runs.append(run_len)
    if not runs:
        return 0.0, 0
    zr = resolution[2]  # nm/片
    return float(np.mean(runs)), float(np.mean(runs)) * zr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt-path', required=True, help='GT 掩码或标签 h5')
    ap.add_argument('--pred-seg', required=True)
    ap.add_argument('--neuron-id', type=int, default=None,
                    help='gt-path 是标签图时提取该实例；否则按二值掩码处理')
    ap.add_argument('--resolution', type=str, default='8,8,33', help='nm/体素（z 用第 3 项）')
    args = ap.parse_args()

    gt = load_h5(args.gt_path)
    if args.neuron_id is not None:
        gt_bin = (gt == args.neuron_id)
    else:
        gt_bin = gt > 0
    seg = load_h5(args.pred_seg)
    res = [float(x) for x in args.resolution.split(',')]

    mean_vox, mean_nm = erl_voxel_path(gt_bin, seg, res)
    print(f'ERL = {mean_vox:.1f} 片 ≈ {mean_nm / 1000:.2f} µm（z 向路径近似）')
    print('注：论文 H01 #2252715458 ERL=12267（SAvEM³）/ 10184（Superhuman），'
          '单位与 site 定义请对照官方工具校准。')


if __name__ == '__main__':
    main()
