"""M3 · 教师特征预计算：SAM 嵌入 + HQ-SAM hq 特征 → 训练用 h5 / 可选逐层 tif。

完全对齐 HQ-SAM/sam-hq/train/train_prep.py 的计算逻辑：
    输入：img 图层逐片 → 重采样到 4nm（xr/yr→4）→ 中心裁 1024×1024 → GRAY2RGB
    编码：image_embeddings, interm_embeddings = sam.image_encoder(x)
          vit_features = interm_embeddings[0].permute(0,3,1,2)          # (B,1280,64,64)
          hq_features  = sam.mask_decoder.embedding_encoder(image_embeddings)
                       + sam.mask_decoder.compress_vit_feat(vit_features)  # (B,32,256,256)

输出 h5（与 SuperHuman/data/AC3-AC4_new 同构，data_provider_labeled.py 'ac4-triplet' 直接读取）：
    <name>_embeddings.h5 : {'main': (256, Z, 64, 64)  float32}   # SAM embed @1/16
    <name>_features.h5   : {'main': (32,  Z, 256, 256) float32}  # hq_feat  @1/4
--write-tif 时额外写 prepared_embedding/{ds}/embed/%05d.tif 与 embed_interm_0/%05d.tif
（SAEM² dataloader 用；tif 的通道维在最后一维）。

用法（需 torch + GPU/MPS，HQ-SAM 权重）：
    python precompute_teacher.py --datasets snemi,ac3 \
        --checkpoint ./sam-hq/pretrained_checkpoint/sam_hq_vit_h.pth \
        --out-dir ./data/AC3-AC4_new --device mps
"""
import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_engine'))
from common import DATA_ROOT, REPO_ROOT, ranges_of, open_cv  # noqa: E402


def load_sam(checkpoint, model_type, device):
    import torch
    from segment_anything import sam_model_registry
    sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
    sam.eval()
    return sam


def process_dataset(sam, dataset, out_dir, write_tif, device, batch=1, max_slices=0):
    import cv2
    import h5py
    import torch
    from skimage import io

    img_vol = open_cv(dataset, 'img')
    if img_vol is None:
        print(f'[{dataset}] 无 img 图层，跳过')
        return
    (xs, ys, zs), (xe, ye, ze) = ranges_of(dataset)[0]
    xr, yr, zr = img_vol.resolution[:3]          # nm
    z_total = ze - zs
    if max_slices and max_slices < z_total:
        z_total = max_slices
        ze = zs + max_slices

    emb_h5 = os.path.join(out_dir, f'{dataset}_embeddings.h5')
    feat_h5 = os.path.join(out_dir, f'{dataset}_features.h5')
    f_emb = h5py.File(emb_h5, 'w')
    f_feat = h5py.File(feat_h5, 'w')
    d_emb = f_emb.create_dataset('main', (256, z_total, 64, 64), dtype='f4')
    d_feat = f_feat.create_dataset('main', (32, z_total, 256, 256), dtype='f4')

    tif_dirs = {}
    if write_tif:
        base = os.path.join(DATA_ROOT, 'prepared_embedding', dataset)
        tif_dirs['embed'] = os.path.join(base, 'embed')
        tif_dirs['interm0'] = os.path.join(base, 'embed_interm_0')
        for d in tif_dirs.values():
            os.makedirs(d, exist_ok=True)

    z_done = 0
    pbar = tqdm(total=z_total, desc=dataset)
    zi = zs
    while zi < ze:
        zj = min(zi + batch, ze)
        imgs = []
        for z in range(zi, zj):
            im = img_vol[xs:xe, ys:ye, z][..., 0].astype(np.float32)
            # 与 train_prep.py 一致：重采样到 4nm → 中心裁 1024 → RGB
            ratio = yr / 4.0
            im = cv2.resize(im, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
            assert im.shape[0] >= 1024 and im.shape[1] >= 1024, f'{dataset} z={z} 尺寸不足 1024'
            ym, xm = im.shape[0] // 2, im.shape[1] // 2
            im = im[ym - 512:ym + 512, xm - 512:xm + 512]
            imgs.append(cv2.cvtColor(im, cv2.COLOR_GRAY2RGB))
        arr = np.stack(imgs, 0)                                   # (B,1024,1024,3)
        x = torch.from_numpy(arr.transpose(0, 3, 1, 2)).to(device)
        with torch.no_grad():
            x = torch.stack([sam.preprocess(x[i]) for i in range(len(x))], 0)
            image_embeddings, interm_embeddings = sam.image_encoder(x)
            vit_features = interm_embeddings[0].permute(0, 3, 1, 2)
            hq_features = (sam.mask_decoder.embedding_encoder(image_embeddings)
                           + sam.mask_decoder.compress_vit_feat(vit_features))
        for i in range(len(imgs)):
            z_out = z_done + i
            emb = image_embeddings[i].cpu().numpy()               # (256,64,64)
            feat = hq_features[i].cpu().numpy()                   # (32,256,256)
            d_emb[:, z_out] = emb
            d_feat[:, z_out] = feat
            if write_tif:
                io.imsave(os.path.join(tif_dirs['embed'], '%05d.tif' % z_out),
                          emb.transpose(1, 2, 0).astype(np.float32))
                interm0 = interm_embeddings[0][i].cpu().numpy()   # (64,64,1280)
                io.imsave(os.path.join(tif_dirs['interm0'], '%05d.tif' % z_out),
                          interm0.astype(np.float32))
        z_done += len(imgs)
        pbar.update(len(imgs))
        zi = zj
    pbar.close()
    f_emb.close()
    f_feat.close()
    print(f'[{dataset}] -> {emb_h5}\n            {feat_h5}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', type=str, default='snemi,ac3')
    ap.add_argument('--checkpoint', type=str,
                    default=os.path.join(REPO_ROOT, 'sam-hq', 'pretrained_checkpoint', 'sam_hq_vit_h.pth'))
    ap.add_argument('--model-type', type=str, default='vit_h')
    ap.add_argument('--device', type=str, default='mps')
    ap.add_argument('--out-dir', type=str, default=None,
                    help='默认 $SAVEM3_DATA_ROOT/data/AC3-AC4_new')
    ap.add_argument('--write-tif', action='store_true', help='同时写 prepared_embedding 逐层 tif（SAEM² 用）')
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--max-slices', type=int, default=0,
                    help='只处理前 N 片（0=全部；大卷可分片生成/验证）')
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(DATA_ROOT, 'data', 'AC3-AC4_new')
    os.makedirs(out_dir, exist_ok=True)

    print('加载 HQ-SAM ...')
    sam = load_sam(args.checkpoint, args.model_type, args.device)
    for name in [d for d in args.datasets.split(',') if d]:
        print(f'== {name} ==')
        process_dataset(sam, name, out_dir, args.write_tif, args.device,
                         args.batch, args.max_slices)


if __name__ == '__main__':
    main()
