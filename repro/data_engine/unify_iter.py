"""M1 数据引擎 · 模型在环统一（论文 Data unification 原话：

"We iteratively train our model and save the high-confidential masks. We also
 used the mask prompting of finetuned HQ-SAM and SAEM² and the alternative NMS
 method to filter duplicated masks. Here, we modified masks with IoU between
 before and after higher than 0.7, which means we optimized the original
 boundary only. This cycle was repeated several times."

本脚本实现"统一"的掩码更新半边（模型训练半边在外部循环：训练 SAEM²/HQ-SAM →
调用 sam_refine.py 或 SAEM² 推理产出精修掩码 → 用本脚本合并回数据银行）：

1. --pred-layer 读模型精修掩码层（seg_refine 或 SAEM² 输出的连通域实例层）；
2. 与 --seg-layer 原标签逐片对齐：逐标签求 IoU；
3. IoU > 0.7（--iou-threshold）：用精修掩码替换原标签边界（只优化边界）；
   IoU <= 0.7：保留原标签；
4. NMS 去重：精修掩码相互 IoU 过高的保留大者（--nms-iou）；
5. 写出 --out-layer（seg_unified），完成一轮；论文做两轮（--round 记入日志）。

用法：
    python unify_iter.py --datasets snemi --seg-layer segb --pred-layer seg_refine \
        --out-layer seg_unified --round 1
"""
import argparse

import numpy as np
from tqdm import tqdm

from common import ranges_of, cv_path, open_cv
from cloudvolume import CloudVolume


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / max(union, 1)


def unify_slice(seg_orig, seg_pred, iou_threshold=0.7, nms_iou=0.9):
    """一片的统一：pred 中与 orig 各标签 IoU>thr 的替换之；pred 内部 NMS 去重。"""
    out = seg_orig.copy()
    pred_labels = np.unique(seg_pred)
    pred_labels = pred_labels[pred_labels != 0]

    # pred 内部 NMS：按面积降序，与已保留掩码 IoU 过高则丢弃
    kept_preds = []
    areas = [(l, int((seg_pred == l).sum())) for l in pred_labels]
    areas.sort(key=lambda t: -t[1])
    for l, _ in areas:
        m = seg_pred == l
        if any(iou(m, km) > nms_iou for km in kept_preds):
            continue
        kept_preds.append(m)

    # 与 orig 逐标签对齐
    orig_labels = np.unique(seg_orig)
    orig_labels = orig_labels[orig_labels != 0]
    assigned = np.zeros_like(seg_orig, dtype=bool)
    for m in kept_preds:
        if assigned[m].sum() / max(m.sum(), 1) > 0.5:
            continue  # 该区域已被处理
        best_iou, best_lab = 0.0, 0
        for ol in orig_labels:
            v = iou(m, seg_orig == ol)
            if v > best_iou:
                best_iou, best_lab = v, ol
        if best_iou > iou_threshold and best_lab != 0:
            out[m] = best_lab      # 用精修边界替换原标签
            assigned |= m
    return out


def process(dataset, seg_layer, pred_layer, out_layer, iou_threshold, nms_iou, round_no):
    seg_vol = open_cv(dataset, seg_layer)
    pred_vol = open_cv(dataset, pred_layer)
    if seg_vol is None or pred_vol is None:
        print(f'[{dataset}] 缺图层（{seg_layer}/{pred_layer}），跳过')
        return
    (xs, ys, zs), (xe, ye, ze) = ranges_of(dataset)[0]
    dst = CloudVolume('file://' + cv_path(dataset, seg_layer).replace(seg_layer, out_layer),
                      info=seg_vol.info, non_aligned_writes=True, fill_missing=True)
    dst.commit_info()

    for z in tqdm(range(zs, ze), desc=dataset):
        seg = seg_vol[xs:xe, ys:ye, z][..., 0]
        pred = pred_vol[xs:xe, ys:ye, z][..., 0]
        out = unify_slice(seg, pred, iou_threshold, nms_iou)
        dst[xs:xe, ys:ye, z] = out.astype(seg.dtype)[..., None]
    print(f'[{dataset}] round {round_no} 完成 -> {out_layer}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', type=str, default='snemi')
    ap.add_argument('--seg-layer', type=str, default='segb', help='当前标签层（原标签）')
    ap.add_argument('--pred-layer', type=str, default='seg_refine', help='模型精修掩码层')
    ap.add_argument('--out-layer', type=str, default='seg_unified')
    ap.add_argument('--iou-threshold', type=float, default=0.7, help='论文 IoU>0.7 才改边界')
    ap.add_argument('--nms-iou', type=float, default=0.9)
    ap.add_argument('--round', type=int, default=1)
    args = ap.parse_args()

    for name in [d for d in args.datasets.split(',') if d]:
        print(f'== {name} ==')
        process(name, args.seg_layer, args.pred_layer, args.out_layer,
                args.iou_threshold, args.nms_iou, args.round)


if __name__ == '__main__':
    main()
