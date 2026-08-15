"""M5 · 提示图割 · 特征提取（论文 §2.3 Prompt-based Graph-cut）。

思路启发并部分借鉴 **SAM-Graph**（[arXiv:2312.08372](https://arxiv.org/abs/2312.08372)，
ECCV 2024，zju3dv/SAM-Graph，代码见 ../SAM-Graph/scripts/）：
  - 节点特征：SAM 编码特征（本实现沿用 SAvEM³ 论文的 32ch hq-feature 区域均值；
    SAM-Graph 的"提示位置双线性采样"见其 node_feature.py，可作替代）；
  - 边特征（两路，SAM-Graph 的 edge_weights.py 同款三元组 + SAvEM³ 的余弦）：
      (a) 相邻 oversegment 节点特征的余弦相似度（SAvEM³）；
      (b) SAM 自动提示掩码的 IoU / IoA / IoB（SAM-Graph 三元组；SAvEM³ 用 IoU），
          在 z-1/z/z+1 三片上提示并按预测 IoU 置信度加权聚合（SAM-Graph 的
          weighted_average，可选再加提示点距离权重）。
  - 提示方式两种：--prompt-mode mask（SAvEM³：腐蚀 1-2px 的掩码提示）
                 --prompt-mode points（SAM-Graph：距离变换迭代取点，--num-points 个）。
  - 多掩码选择：SAM 三掩码 + 预测 IoU 的边际启发式（SAM-Graph select_mask）。

输出：node_feats.h5 {label: (32,)}、edge_feats.json {(u,v): {cos, iou, ioa, iob}}。

用法（--prompt-iou 需 torch + SAM/HQ-SAM 权重）：
    python prompt_graph_cut.py --fragments frag.h5 --feats out_feats.h5 \
        --img AC3.tif --checkpoint sam_hq_vit_h.pth \
        [--prompt-mode points --num-points 5] [--no-prompt-iou]
"""
import argparse
import os

import numpy as np
from tqdm import tqdm


def load_h5(path, name='main'):
    import h5py
    with h5py.File(path, 'r') as f:
        return f[name][:]


def save_h5(path, data_dict):
    import h5py
    with h5py.File(path, 'w') as f:
        for k, v in data_dict.items():
            f.create_dataset(k, data=v)


def node_features(fragments, feats):
    """feats: (32,Z,H,W) hq-feature 图。返回 {label: mean_vec(32,)}。"""
    labels = np.unique(fragments)
    labels = labels[labels != 0]
    out = {}
    for lab in tqdm(labels, desc='node feats'):
        m = fragments == lab
        if m.sum() == 0:
            continue
        vec = feats[:, m].mean(axis=1)
        out[int(lab)] = vec.astype(np.float32)
    return out


def rag_edges(fragments):
    """相邻 fragment 对（含 z 邻接）。返回 set[(u,v)]，u<v。"""
    from elf.segmentation.features import compute_rag
    rag = compute_rag(fragments)
    edges = set()
    n_nodes = rag.numberOfNodes
    for e in range(rag.numberOfEdges):
        u, v = rag.uvIds()[e]
        if u >= n_nodes or v >= n_nodes:
            continue
        u, v = int(u), int(v)
        edges.add((min(u, v), max(u, v)))
    return rag, edges


def cosine(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 0 else 0.0


def edge_cos(fragments, node_feats):
    _, edges = rag_edges(fragments)
    out = {}
    for u, v in edges:
        if u in node_feats and v in node_feats:
            out[(u, v)] = {'cos': cosine(node_feats[u], node_feats[v])}
    return out


# ---------------------------------------------------------------------------
# 以下为借鉴 SAM-Graph 的部分（脚本源码 ../SAM-Graph/scripts/）
# ---------------------------------------------------------------------------

def sample_points_from_mask(mask, num_points=5):
    """SAM-Graph predict_masks.py 同款：在掩码距离变换图上迭代取极大值点。"""
    import cv2
    h, w = mask.shape
    ext = np.zeros((h + 2, w + 2), dtype=np.uint8)
    ext[1:-1, 1:-1] = (mask > 0).astype(np.uint8)
    dt = cv2.distanceTransform(ext, cv2.DIST_L2, 5)[1:-1, 1:-1]
    pts = []
    for _ in range(num_points):
        _, max_val, _, max_loc = cv2.minMaxLoc(dt)
        if max_val <= 0:
            break
        pts.append(max_loc)
        cv2.circle(dt, max_loc, int(max_val), 0, -1)
    return pts


def select_mask(masks, iou_preds):
    """SAM-Graph edge_weights.py 同款多掩码选择：按预测 IoU 与 0.05 边际选掩码。"""
    if iou_preds[2] > iou_preds.max() - 0.05:
        return masks[2], iou_preds[2]
    if iou_preds[1] > iou_preds[0] - 0.05:
        return masks[1], iou_preds[1]
    return masks[0], iou_preds[0]


def predict_one(predictor, prompt):
    """prompt: {'mask': (H,W)bool | 'points': [(x,y)...]}。返回 (mask, iou_pred)。"""
    if 'mask' in prompt:
        out = predictor.predict(mask_input=prompt['mask'], multimask_output=True)
    else:
        pts = np.array(prompt['points'])
        out = predictor.predict(point_coords=pts, point_labels=np.ones(len(pts)),
                                multimask_output=True)
    return select_mask(out['masks'], out['scores'])


def prompt_iou_edges(fragments, img_vol, predictor, erode=1, prompt_mode='mask',
                     num_points=5, use_distance=False):
    """SAM 自动提示掩码 IoU/IoA/IoB 边特征（SAvEM³ z-1/z/z+1 三片提示 +
    SAM-Graph 置信度加权聚合）。"""
    from scipy import ndimage
    labels = np.unique(fragments)
    labels = labels[labels != 0]
    _, edges = rag_edges(fragments)

    # 每个 fragment：中心 z + 提示（mask 腐蚀版 / 距离变换点）
    z_of, prompts, dist_maps = {}, {}, {}
    for lab in tqdm(labels, desc='build prompts'):
        m = fragments == lab
        zs = np.nonzero(m.any(axis=(1, 2)))[0]
        if len(zs) == 0:
            continue
        z_of[int(lab)] = int(zs[len(zs) // 2])
        if prompt_mode == 'points':
            prompts[int(lab)] = {'points': sample_points_from_mask(m[z_of[int(lab)]],
                                                                   num_points)}
        else:
            ero = ndimage.binary_erosion(m[z_of[int(lab)]], iterations=erode).astype(np.uint8)
            if ero.sum() == 0:
                ero = (m[z_of[int(lab)]] > 0).astype(np.uint8)
            prompts[int(lab)] = {'mask': ero}

    # 三片提示：z-1 / z / z+1，逐 fragment 取 SAM 掩码与预测 IoU
    prompted, scores = {}, {}
    for lab in tqdm(labels, desc='SAM prompting'):
        if lab not in z_of:
            continue
        z = z_of[lab]
        cur_m, cur_s = [], []
        for zz in (max(z - 1, 0), z, min(z + 1, img_vol.shape[0] - 1)):
            im = img_vol[zz]
            if im.ndim == 2:
                im = np.repeat(im[..., None], 3, axis=2)
            predictor.set_image(im)
            m_, s_ = predict_one(predictor, prompts[lab])
            cur_m.append(m_)
            cur_s.append(float(s_))
        prompted[lab] = cur_m
        scores[lab] = cur_s

    out = {}
    for u, v in tqdm(edges, desc='edge feats'):
        if u not in prompted or v not in prompted:
            continue
        # SAM-Graph weighted_average：按 iou_pred1×iou_pred2（可选×距离）跨片加权
        w = [scores[u][k] * scores[v][k] for k in range(3)]
        if use_distance:
            # 提示位置（点模式：均值坐标）2D 距离，SAM-Graph 同款
            d = _prompt_distance(prompts[u], prompts[v], img_vol.shape[-2:])
            w = [wk * d for wk in w]
        wsum = sum(w)
        if wsum <= 0:
            continue
        iou = ioa = iob = 0.0
        for k in range(3):
            a, b = prompted[u][k], prompted[v][k]
            inter = np.logical_and(a, b).sum()
            iou += w[k] / wsum * inter / max(np.logical_or(a, b).sum(), 1)
            ioa += w[k] / wsum * inter / max(a.sum(), 1)
            iob += w[k] / wsum * inter / max(b.sum(), 1)
        out[(u, v)] = {'iou': float(iou), 'ioa': float(ioa), 'iob': float(iob)}
    return out


def _prompt_distance(p1, p2, shape):
    def loc(p):
        if 'points' in p:
            pts = np.array(p['points'])
            return pts.mean(0)
        ys, xs = np.nonzero(p['mask'])
        return np.array([xs.mean(), ys.mean()])
    return float(np.linalg.norm(loc(p1) - loc(p2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fragments', required=True, help='过分割标签 h5')
    ap.add_argument('--feats', required=True, help='32ch hq-feature h5（(32,Z,H,W)）')
    ap.add_argument('--img', default=None, help='图像堆栈 h5/tif（--prompt-iou 需要）')
    ap.add_argument('--checkpoint', default=None, help='SAM/HQ-SAM 权重（--prompt-iou 需要）')
    ap.add_argument('--model-type', default='vit_h')
    ap.add_argument('--device', default='mps')
    ap.add_argument('--out-dir', default='./graph_feats')
    ap.add_argument('--no-prompt-iou', action='store_true')
    ap.add_argument('--prompt-mode', default='mask', choices=['mask', 'points'],
                    help='mask=SAvEM³ 腐蚀掩码提示；points=SAM-Graph 距离变换点提示')
    ap.add_argument('--num-points', type=int, default=5, help='points 模式的提示点数（SAM-Graph 默认 5）')
    ap.add_argument('--erode', type=int, default=1, help='mask 模式的腐蚀像素（SAvEM³ 论文 1-2）')
    ap.add_argument('--use-distance', action='store_true',
                    help='聚合权重再加提示点 2D 距离（SAM-Graph weighted_average 同款）')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    frag = load_h5(args.fragments).astype('uint64')
    feats = load_h5(args.feats)
    assert feats.ndim == 4 and feats.shape[0] == 32, 'feats 应为 (32,Z,H,W)'

    print('1) 节点特征（hq-feature 均值）...')
    nf = node_features(frag, feats)
    save_h5(os.path.join(args.out_dir, 'node_feats.h5'),
            {str(k): v for k, v in nf.items()})

    print('2) 边特征（余弦相似度）...')
    ef = edge_cos(frag, nf)

    if not args.no_prompt_iou and args.img and args.checkpoint:
        print(f'3) 边特征（SAM 提示 IoU/IoA/IoB，z-1/z/z+1，mode={args.prompt_mode}）...')
        import torch
        from segment_anything import sam_model_registry, SamPredictor
        sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(args.device)
        predictor = SamPredictor(sam)
        img_vol = load_h5(args.img)
        iou_edges = prompt_iou_edges(frag, img_vol, predictor, erode=args.erode,
                                     prompt_mode=args.prompt_mode,
                                     num_points=args.num_points,
                                     use_distance=args.use_distance)
        for k, v in iou_edges.items():
            ef.setdefault(k, {}).update(v)
    else:
        print('3) 跳过 SAM 提示 IoU（未给 --img/--checkpoint）')

    import json
    with open(os.path.join(args.out_dir, 'edge_feats.json'), 'w') as f:
        json.dump({f'{u},{v}': d for (u, v), d in ef.items()}, f)
    print(f'完成 -> {args.out_dir}/（node_feats.h5, edge_feats.json）')


if __name__ == '__main__':
    main()
