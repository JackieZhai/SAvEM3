"""M1 data engine: SAM mask-prompt refinement (data-cleansing step 4).
The paper describes eroded-mask prompts and manual review below IoU 0.5:

"we eroded the original masks with 1-2 pixels as the mask prompts and refined
 them with the help of SAM. Note that we only manually checked the masks with
 IoU between before and after lower than 0.5."

Workflow (one 2D section at a time, using PyTorch on the selected device):
1. Read a segmentation section and enumerate its instance labels.
2. Erode each mask by 1-2 pixels and refine it with HQ-SAM mask prompting.
3. Write refined masks to seg_refine; flag IoU(refined, original) < 0.5 for
   manual review in JSONL (dataset, section, label, and IoU).
   Resolve overlapping predictions by confidence; ties retain the earlier label.

Usage:
    python sam_refine.py --datasets snemi --checkpoint <sam_hq_vit_h.pth> \
        --device mps --erode 2 --iou-threshold 0.5 --out-json manual_check.jsonl
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common import ranges_of, cv_path, open_cv
from cloudvolume import CloudVolume
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repro.sam import mask_prompt, choose_mask


def load_predictor(checkpoint, model_type, device):
    from repro.sam import load_predictor as load_hq_predictor
    return load_hq_predictor(checkpoint, model_type, device)


def refine_slice(predictor, image, seg, erode=2):
    """Refine all instances in one image using SAM mask prompts.

    Return (refined_seg, records), with [(label, iou)] records for review routing.
    """
    from scipy import ndimage
    predictor.set_image(image)  # (H,W,3)
    labels = np.unique(seg)
    labels = labels[labels != 0]

    refined = np.zeros_like(seg, dtype=seg.dtype)
    best_score = np.full(seg.shape, -np.inf)
    records = []
    for lab in labels:
        m = (seg == lab).astype(np.uint8)
        m_ero = ndimage.binary_erosion(m, iterations=erode).astype(np.uint8)
        if m_ero.sum() == 0:
            m_ero = m
        out = predictor.predict(mask_input=mask_prompt(m_ero, predictor), multimask_output=False)
        pred, score = choose_mask(out)
        inter = np.logical_and(pred, m).sum()
        union = np.logical_or(pred, m).sum()
        iou = inter / max(union, 1)
        records.append((int(lab), float(iou)))
        # Assign overlapping pixels to the highest-confidence mask; preserve ties.
        selected = pred & (score > best_score)
        refined[selected] = lab
        best_score[selected] = score
    return refined, records


def process(dataset, predictor, erode, iou_threshold, out_json):
    seg_vol = open_cv(dataset, 'seg')
    img_vol = open_cv(dataset, 'img')
    if seg_vol is None:
        print(f'[{dataset}] No seg layer; skipping')
        return
    (xs, ys, zs), (xe, ye, ze) = ranges_of(dataset)[0]

    dst = CloudVolume('file://' + cv_path(dataset, 'seg') + '_refine', info=seg_vol.info, non_aligned_writes=True, fill_missing=True)
    dst.commit_info()

    manual = []
    for z in tqdm(range(zs, ze), desc=dataset):
        image = img_vol[xs:xe, ys:ye, z][..., 0, 0].T
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        elif image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=2)
        seg = seg_vol[xs:xe, ys:ye, z][..., 0, 0].T
        refined, records = refine_slice(predictor, image, seg, erode=erode)
        dst[xs:xe, ys:ye, z] = refined.T.astype(seg.dtype)[..., None]
        for lab, iou in records:
            if iou < iou_threshold:
                manual.append({'dataset': dataset, 'z': int(z), 'label': lab, 'iou': iou})

    with open(out_json, 'a') as f:
        for r in manual:
            f.write(json.dumps(r) + '\n')
    print(f'[{dataset}] Done; {len(manual)} masks require manual review -> {out_json}')


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

    print('Loading SAM/HQ-SAM (a local checkpoint is required)...')
    predictor = load_predictor(args.checkpoint, args.model_type, args.device)
    for name in [d for d in args.datasets.split(',') if d]:
        print(f'== {name} ==')
        process(name, predictor, args.erode, args.iou_threshold, args.out_json)


if __name__ == '__main__':
    main()
