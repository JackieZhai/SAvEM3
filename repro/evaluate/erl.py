"""M6 · 大体积稀疏评测：Expected Run Length（ERL）。

本实现严格采用 AxonEM-challenge 官方评测逻辑：
    https://github.com/PytorchConnectomics/AxonEM-challenge
    erl_wrapper/eval_erl.py::compute_erl / expected_run_length
    erl_wrapper/test_axonEM.py::test_AxonEM

官方输入约定：
    - 预测分割：h5（任选 dataset，默认取第一个）或 tif，体素顺序 (z, y, x)
    - GT：AxonEM challenge 提供的 `gt_*_skel_stats.p`（pickle 依次存
      [gt_graph, gt_res]；gt_graph 为 networkx_lite 图，节点属性
      skeleton_id/x/y/z；gt_res 为 [z,y,x] 物理分辨率）
    - 可选 `gt_*_mask.h5`：用于把非神经元区域计入 false merge 的 mask

用法：
    python erl.py --seg path/to/pred.h5 --gt-stats path/to/gt_human_32nm_skel_stats.p
                  [--gt-mask path/to/gt_human_32nm_mask.h5]
                  [--num-chunk 1] [--merge-threshold 50]
                  [--erl-intervals 0-20000-40000-150000]

旧的“沿 z 路径近似”实现保留在 erl_approx.py，仅用于快速 sanity check，
不应作为论文 ERL 数值依据。
"""
import argparse
import os
import sys

import numpy as np

AXONEM_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "AxonEM-challenge", "erl_wrapper")
sys.path.insert(0, AXONEM_ROOT)

from data_io import read_vol, read_pkl  # noqa: E402
from eval_erl import compute_segment_lut, compute_erl  # noqa: E402


def parse_resolution(text):
    text = text.lower().replace(",", "x")
    res = [float(x) for x in text.split("x")]
    if len(res) != 3:
        raise ValueError(f"无法解析分辨率 {text!r}，应为 zxy，例如 32x32x30")
    return np.asarray(res)


def node_positions_voxel(gt_graph, gt_res):
    """从官方 GT 图得到节点体素坐标 (N, 3)，顺序为 (z, y, x)。

    - networkx_lite 图：nodes._nodes 列顺序为 skeleton_id/x/y/z
    - 普通 networkx 图：节点属性 z/y/x
    """
    # lite graph（AxonEM challenge pickle 的默认格式）
    nodes_obj = getattr(gt_graph, "nodes", None)
    is_lite = (nodes_obj is not None and hasattr(nodes_obj, "_nodes")
               and isinstance(getattr(nodes_obj, "_nodes"), np.ndarray))
    if is_lite:
        # 与 erl_wrapper/test_axonEM.py 保持一致：
        # _nodes 排序后为 [skeleton_id, x, y, z]，取 z,y,x 三列再除分辨率
        xyz = nodes_obj._nodes[:, -1:0:-1]  # z,y,x
        if xyz.shape[1] != 3:
            raise ValueError("networkx_lite 节点属性列数不是 4")
        pos = (xyz // gt_res).astype(np.uint16)
        return pos

    # 普通 networkx 图
    pos = []
    for _, data in gt_graph.nodes(data=True):
        if not all(k in data for k in ("z", "y", "x")):
            raise ValueError("networkx 节点缺少 z/y/x 属性")
        pos.append((data["z"], data["y"], data["x"]))
    pos = (np.asarray(pos, dtype=np.float64) // gt_res).astype(np.uint16)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("无法从 gt_graph 提取节点坐标")
    return pos


def main():
    ap = argparse.ArgumentParser(description="Official AxonEM-challenge ERL evaluation")
    ap.add_argument("--seg", required=True, help="预测实例分割 h5/tif")
    ap.add_argument("--gt-stats", required=True,
                    help="AxonEM GT 统计 pickle：依次存 [gt_graph, gt_res]")
    ap.add_argument("--gt-mask", default=None, help="可选 GT 非神经元 mask h5")
    ap.add_argument("--num-chunk", type=int, default=1,
                    help="预测分割 h5 分块读取数（内存不足时增大）")
    ap.add_argument("--merge-threshold", type=int, default=50,
                    help="seg 与 skeleton 重合体素少于该值不计 false merge")
    ap.add_argument("--erl-intervals", type=str, default=None,
                    help="如 0-20000-40000-150000；不给则输出总体 ERL")
    args = ap.parse_args()

    print("加载 GT graph/resolution ...")
    objs = read_pkl(args.gt_stats)
    if len(objs) < 2:
        raise ValueError(
            f"{args.gt_stats} 至少应包含两个 pickle 对象 [gt_graph, gt_res]")
    gt_graph, gt_res = objs[0], np.asarray(objs[1])
    if gt_res.shape != (3,):
        raise ValueError(f"gt_res 应为 3 个分辨率值，实际 {gt_res.shape}")

    node_position = node_positions_voxel(gt_graph, gt_res)
    print(f"节点数 {len(node_position)}，分辨率 {tuple(gt_res)}")

    print("计算 skeleton 节点 -> segment LUT ...")
    node_segment_lut, mask_segment_id = compute_segment_lut(
        args.seg, node_position, args.gt_mask, args.num_chunk)

    intervals = None
    if args.erl_intervals:
        intervals = [int(x) for x in args.erl_intervals.split("-") if x != ""]
        if len(intervals) < 2:
            intervals = None

    print("计算 ERL ...")
    scores = compute_erl(gt_graph, node_segment_lut, mask_segment_id,
                         args.merge_threshold, intervals)
    if isinstance(scores, np.ndarray):
        print("ERL 分段结果：")
        for i, row in enumerate(scores):
            print(f"  interval {i}: ERL={row[0]:.1f}, length^2/length={row[1]:.1f}")
    else:
        print(f"ERL = {scores[0]:.1f}  (sum l^2/sum l = {scores[1]:.1f})")


if __name__ == "__main__":
    main()
