# Copyright by Team of MiRA, Institute of Automation, CAS.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import numpy as np
import imageio
import torch
import cv2
from tqdm import tqdm
from cloudvolume.lib import mkdir
from cloudvolume import CloudVolume
from segment_anything_hq import sam_model_registry
from utils.location import *


def get_args_parser():
    parser = argparse.ArgumentParser('HQ-SAM', add_help=False)

    parser.add_argument("--input", "-i", type=str, default="/mnt/WGCJ/SAM/Ours")
    parser.add_argument("--output", "-o", type=str, default="/mnt/WGCJ/SAM/OursEmbed")

    parser.add_argument("--batch", "-b", type=int, default=1)
    parser.add_argument("--gpu", "-g", type=str, default='0')

    parser.add_argument("--model-type", type=str, default="vit_h", 
                        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--checkpoint", type=str, 
                        default="/mnt/WGCJ/SAM/sam-hq/train/pretrained_checkpoint/sam_hq_vit_h.pth",
                        help="The path to the SAM checkpoint to use for mask generation.")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="The device to run generation on.")

    return parser.parse_args()


args = get_args_parser()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
_ = sam.to(device=args.device)


print(len(data_location_list.keys()))
for data_name in list(data_location_list.keys())[30:40]:
    print(data_name)

    vol_img = CloudVolume('file://' + os.path.join(args.input, data_location_list[data_name][0]))
    xs, ys, zs = data_range_list[data_name][0]
    xe, ye, ze = data_range_list[data_name][1]
    images = vol_img[xs:xe, ys:ye, zs:ze][..., 0].transpose()

    xr, yr, zr = data_resolution_list[data_name]
    assert xr == yr

    batched_images = []
    for b_i in tqdm(range(0, images.shape[0], args.batch)):
        b_j = min(b_i + args.batch, images.shape[0])
        single_input = []
        for i in range(b_i, b_j):
            image = images[i]
            
            # resize to 4x4 nm
            ratio_y = yr / 4; ratio_x = xr / 4
            image = cv2.resize(image, (0, 0), fx=ratio_x, fy=ratio_y, interpolation=cv2.INTER_LINEAR)

            # center-crop to 1024x1024 vx
            assert image.shape[0] >= 1024 and image.shape[1] >= 1024
            ym, xm = image.shape[0]//2, image.shape[1]//2
            image = image[ym-512:ym+512, xm-512:xm+512]

            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            single_input.append(image)
        batched_images.append(np.stack(single_input, axis=0))

    info = CloudVolume.create_new_info(
        num_channels    = 256,
        layer_type      = 'image',
        data_type       = 'float32',
        # raw, png, jpeg, compressed_segmentation, fpzip, kempressed, zfpc, compresso, crackle
        encoding        = 'raw', 
        resolution      = [64, 64, zr], # Voxel scaling, units are in nanometers
        voxel_offset    = [0, 0, zs], # x,y,z offset in voxels from the origin
        # Pick a convenient size for your underlying chunk representation
        # Powers of two are recommended, doesn't need to cover image exactly
        chunk_size      = [64, 64, 1], # units are voxels
        volume_size     = [64, 64, ze-zs],
    )
    vol_emb = CloudVolume('file://' + os.path.join(args.output, data_location_list[data_name][0].replace('img', 'embed')), info=info, compress='gzip')
    vol_emb.commit_info()

    info = CloudVolume.create_new_info(
        num_channels    = 1280,
        layer_type      = 'image',
        data_type       = 'float32',
        # raw, png, jpeg, compressed_segmentation, fpzip, kempressed, zfpc, compresso, crackle
        encoding        = 'raw', 
        resolution      = [64, 64, zr], # Voxel scaling, units are in nanometers
        voxel_offset    = [0, 0, zs], # x,y,z offset in voxels from the origin
        # Pick a convenient size for your underlying chunk representation
        # Powers of two are recommended, doesn't need to cover image exactly
        chunk_size      = [64, 64, 1], # units are voxels
        volume_size     = [64, 64, ze-zs],
    )
    vol_int = CloudVolume('file://' + os.path.join(args.output, data_location_list[data_name][0].replace('img', 'embed_interm_0')), info=info, compress='gzip')
    vol_int.commit_info()

    for b in tqdm(range(len(batched_images))):
        imgs = batched_images[b]

        batched_input = []
        for i in range(len(imgs)):
            input_image = torch.as_tensor(imgs[i].astype(dtype=np.uint8), device=sam.device).permute(2, 0, 1).contiguous()
            batched_input.append(input_image)

        with torch.no_grad():
            input_images = torch.stack([sam.preprocess(x) for x in batched_input], dim=0)
            image_embeddings, interm_embeddings = sam.image_encoder(input_images)
            # print(image_embeddings.shape, len(interm_embeddings))

            # vit_features = interm_embeddings[0].permute(0, 3, 1, 2)
            # hq_features = sam.mask_decoder.embedding_encoder(image_embeddings) + sam.mask_decoder.compress_vit_feat(vit_features)
            # print(hq_features.shape)
            
        for i in range(len(image_embeddings)):
            z = b * args.batch + i

            image_embedding = image_embeddings[i].cpu().numpy()
            vol_emb[:, :, z+zs] = image_embedding.transpose()[:, :, np.newaxis, :]

            interm_embedding = interm_embeddings[0][i].cpu().numpy().transpose()
            vol_int[:, :, z+zs] = interm_embedding.transpose()[:, :, np.newaxis, :]

            # for l in range(len(interm_embeddings)):
            #     interm_embedding = interm_embeddings[l][i].cpu().numpy().transpose()
            #     print(interm_embedding.shape, interm_embedding.dtype)
            
            # hq_feature = hq_features[i].cpu().numpy()
            # print(hq_feature.shape, hq_feature.dtype)

'''
python train_prep.py -g 0 -b 1 -i ./prepared_images/FIB25.tif -o ./prepared_embedding/FIB25 \
    --checkpoint  ./pretrained_checkpoint/sam_hq_vit_h.pth --model-type vit_h

CUDA_VISIBLE_DEVICES=0 /mnt/WGCJ/miniconda4/envs/micro-sam/bin/python train_prep.py
'''