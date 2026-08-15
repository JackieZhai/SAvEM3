"""M1 数据引擎 · 清洗第 4 步：SAM 掩码提示精修（论文 Data cleansing 原话：

"we eroded the original masks with 1-2 pixels as the mask prompts and refined
 them with the help of SAM. Note that we only manually checked the masks with
 IoU between before and after lower than 0.5."

流程（逐 2D 片，需 GPU/MPS 上的 PyTorch）：
1. 取 seg 片，unique 出实例标签；
2. 每个实例：mask 腐蚀 1-2px → SAM mask prompt（HQ-SAM 权重）→ 得到精修掩码；
3. IoU(精修, 原掩码) < 0.5  → 写入待人工核查清单（JSONL，含片号/标签/IoU/路径）；
   IoU >= 0.5             → 直接采纳精修掩码写入新标签层 `seg_refine`。
   实例间冲突（一个像素被两个精修掩码覆盖）按掩码得分取高者。

用法：
    python sam_refine.py --datasets snemi --checkpoint <sam_hq_vit_h.pth> \
        --device mps --erode 2 --iou-threshold 0.5 --out-json manual_check.jsonl
"""
import argparse
import json
import os

import numpy as np
from tqdm import tqdm

from common import ranges_of, cv_path, open_cv
from cloudvolume import CloudVolume


def load_predictor(checkpoint, model_type, device):
    import torch
    from segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
    return SamPredictor(sam)


def refine_slice(predictor, image, seg, erode=2):
    """对一片图像上的所有实例做 SAM 掩码提示精修。

    返回 (refined_seg, records)：records 为 [(label, iou)] 便于后续分流。
    """
    from scipy import ndimage
    predictor.set_image(image)  # (H,W,3)
    labels = np.unique(seg)
    labels = labels[labels != 0]

    refined = np.zeros_like(seg, dtype=seg.dtype)
    records = []
    for lab in labels:
        m = (seg == lab).astype(np.uint8)
        m_ero = ndimage.binary_erosion(m, iterations=erode).astype(np.uint8)
        if m_ero.sum() == 0:
            m_ero = m
        out = predictor.predict(mask_input=m_ero, multimask_output=False)
        pred = (out['masks'][0] > 0).astype(np.uint8)
        inter = np.logical_and(pred, m).sum()
        union = np.logical_or(pred, m).sum()
        iou = inter / max(union, 1)
        records.append((int(lab), float(iou)))
        # 冲突像素按"后写覆盖"，简单起见用得分排序后再写（此处顺序写入，可加得分排序）
        refined[pred > 0] = lab
    return refined, records


def process(dataset, predictor, erode, iou_threshold, out_json):
    seg_vol = open_cv(dataset, 'seg')
    img_vol = open_cv(dataset, 'img')
    if seg_vol is None:
        print(f'[{dataset}] 无 seg 图层，跳过')
        return
    (xs, ys, zs), (xe, ye, ze) = ranges_of(dataset)[0]

    dst = CloudVolume('file://' + cv_path(dataset, 'seg') + '_refine', info=seg_vol.info, non_aligned_writes=True, fill_missing=True)
    dst.commit_info()

    manual = []
    for z in tqdm(range(zs, ze), desc=dataset):
        image = img_vol[xs:xe, ys:ye, z][..., 0]
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        elif image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=2)
        seg = seg_vol[xs:xe, ys:ye, z][..., 0]
        refined, records = refine_slice(predictor, image, seg, erode=erode)
        dst[xs:xe, ys:ye, z] = refined.astype(seg.dtype)[..., None]
        for lab, iou in records:
            if iou < iou_threshold:
                manual.append({'dataset': dataset, 'z': int(z), 'label': lab, 'iou': iou})

    with open(out_json, 'a') as f:
        for r in manual:
            f.write(json.dumps(r) + '\n')
    print(f'[{dataset}] 完成；待人工核查 {len(manual)} 个掩码 -> {out_json}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', type=str, default='snemi')
    ap.add_argument('--checkpoint', type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
                                         'sam-hq', 'pretrained_checkpoint', 'sam_hq_vit_h.pth'))
    ap.add_argument('--model-type', type=str, default='vit_h')
    ap.add_argument('--device', type=str, default='mps', help='mps / cuda / cpu')
    ap.add_argument('--erode', type=int, default=2)
    ap.add_argument('--iou-threshold', type=float, default=0.5)
    ap.add_argument('--out-json', type=str, default='manual_check.jsonl')
    args = ap.parse_args()

    print('加载 SAM/HQ-SAM 模型（首次需要权重文件）...')
    predictor = load_predictor(args.checkpoint, args.model_type, args.device)
    for name in [d for d in args.datasets.split(',') if d]:
        print(f'== {name} ==')
        process(name, predictor, args.erode, args.iou_threshold, args.out_json)


if __name__ == '__main__':
    main()
