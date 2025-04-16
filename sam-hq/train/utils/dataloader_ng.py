# Copyright by HQ-SAM team
# All rights reserved.

## data loader
from __future__ import print_function, division

import numpy as np
import random
from copy import deepcopy
import os
import cv2

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, utils
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler

from cloudvolume import CloudVolume
import fastremap


def create_dataloaders(root, root_cache,
                       data_name_list, data_location_list, data_range_list, data_resolution_list,
                       my_transforms=[], batch_size=1, eval=False,
                       fast=True, aux_dim=1):
    dataloaders = []
    datasets = []

    num_workers_ = 1
    if(batch_size>1):
        num_workers_ = 2
    if(batch_size>4):
        num_workers_ = 4
    if(batch_size>8):
        num_workers_ = 8

    for data_name in range(len(data_name_list)):
        data_location = []; data_range = []; data_resolution = []
        for name in data_name:
            data_location.append(data_location_list[name])
            data_range.append(data_range_list[name])
            data_resolution.append(data_resolution_list[name])
        dataset = NeuroglancerDataset(root, root_cache, 
                                      data_location, data_range, data_resolution, 
                                      transform=transforms.Compose(my_transforms), 
                                      fast=fast, aux_dim=aux_dim)
        sampler = DistributedSampler(dataset, shuffle=(not eval))
        dataloader = DataLoader(dataset, batch_size, sampler=sampler, 
                                drop_last=(not eval), num_workers=num_workers_)

        datasets.append(dataset)
        dataloaders.append(dataloader)

    return dataloaders, datasets


class RandomHFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        # random horizontal flip
        if random.random() >= self.prob:
            if 'image' in sample:
                sample['image'] = torch.flip(sample['image'], dims=[2])
            if 'label' in sample:
                sample['label'] = torch.flip(sample['label'], dims=[2])
            if 'label_weight' in sample:
                sample['label_weight'] = torch.flip(sample['label_weight'], dims=[2])
            if 'auxiliary_label' in sample:
                sample['auxiliary_label'] = torch.flip(sample['auxiliary_label'], dims=[2])
            if 'embedding' in sample:
                sample['embedding'] = torch.flip(sample['embedding'], dims=[2])
            if 'interm_embedding' in sample:
                sample['interm_embedding'] = torch.flip(sample['interm_embedding'], dims=[2])

        return sample
    
class RandomVFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        # random vertical flip
        if random.random() >= self.prob:
            if 'image' in sample:
                sample['image'] = torch.flip(sample['image'], dims=[1])
            if 'label' in sample:
                sample['label'] = torch.flip(sample['label'], dims=[1])
            if 'label_weight' in sample:
                sample['label_weight'] = torch.flip(sample['label_weight'], dims=[1])
            if 'auxiliary_label' in sample:
                sample['auxiliary_label'] = torch.flip(sample['auxiliary_label'], dims=[1])
            if 'embedding' in sample:
                sample['embedding'] = torch.flip(sample['embedding'], dims=[1])
            if 'interm_embedding' in sample:
                sample['interm_embedding'] = torch.flip(sample['interm_embedding'], dims=[1])

        return sample


class NeuroglancerDataset(Dataset):
    def __init__(self, root, root_cache, 
                 data_location_list, data_range_list, data_resolution_list, 
                 transform=None, fast=True, aux_dim=1):
        self.transform = transform
        self.fast = fast
        self.aux_dim = aux_dim

        self.root = root
        self.root_cache = root_cache
        self.data_list = []
        for loc, ran, res in zip(data_location_list, data_range_list, data_resolution_list):
            img, seg, mem, dst, msk = loc
            xs, xe, ys, ye, zs, ze = ran
            xres, yres, zres = res
            assert xres == yres
            ratio = xres / 4
            if zres == 8:  # for FIB-SEM, ::3
                for z in range(zs, ze, 3):
                    self.data_list.append((img, seg, mem, dst, msk, xs, xe, ys, ye, z, ratio))
            else:
                for z in range(zs, ze):
                    self.data_list.append((img, seg, mem, dst, msk, xs, xe, ys, ye, z, ratio))

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        img, seg, mem, dst, msk, xs, xe, ys, ye, z, ratio = self.data_list[idx]

        def _center_crop(img, height=1024, width=1024):
            c, h, w = img.shape
            y = (h - height) // 2
            x = (w - width) // 2
            return img[:, y:y+height, x:x+width]
        
        sample = {}
        # transpose to (C, H, W)

        if self.fast:
            vol_embed = CloudVolume('file://' + os.path.join(self.root_cache, img.replace('img', 'embed')))
            embed = vol_embed[:, :, z][:, :, 0, :].transpose()
            sample['embedding'] = torch.tensor(embed, dtype=torch.float32)
            vol_interm_embed = CloudVolume('file://' + os.path.join(self.root_cache, img.replace('img', 'embed_interm_0')))
            interm_embed = vol_interm_embed[:, :, z][:, :, 0, :].transpose()
            sample['interm_embedding'] = torch.tensor(interm_embed, dtype=torch.float32)
        else:
            vol_img = CloudVolume('file://' + os.path.join(self.root, img))
            img = vol_img[xs:xe, ys:ye, z][:, :, 0, :].transpose()
            img = cv2.resize(img, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
            img = _center_crop(img)
            sample['image'] = torch.tensor(img, dtype=torch.float32)
        
        vol_seg = CloudVolume('file://' + os.path.join(self.root, seg), bounded=False)
        seg = vol_seg[xs:xe, ys:ye, z][:, :, 0, :].transpose()
        uniq, cts = fastremap.unique(seg, return_counts=True)
        seg_list = {u: c for u, c in zip(uniq, cts)}
        sample['label_list'] = seg_list
        seg = cv2.resize(seg, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
        seg = _center_crop(seg)
        sample['label'] = torch.tensor(seg, dtype=torch.int32)

        if msk is not None:
            vol_msk = CloudVolume('file://' + os.path.join(self.root, msk), bounded=False)
            msk = vol_msk[xs:xe, ys:ye, z][:, :, 0, :].transpose()
            msk = cv2.resize(msk, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
            msk = _center_crop(msk)
            sample['label_weight'] = torch.tensor(msk, dtype=torch.uint8)
        
        if self.aux_dim > 0:
            vol_mem = CloudVolume('file://' + os.path.join(self.root, mem), bounded=False)
            mem = vol_mem[xs:xe, ys:ye, z][:, :, 0, :].transpose()
            mem = cv2.resize(mem, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
            mem = _center_crop(mem)
            sample['auxiliary_label'] = torch.tensor(mem, dtype=torch.float32)
        if self.aux_dim > 1:
            vol_dst = CloudVolume('file://' + os.path.join(self.root, dst), bounded=False)
            dst = vol_dst[xs:xe, ys:ye, z][:, :, 0, :].transpose()
            dst = cv2.resize(dst, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
            dst = _center_crop(dst)
            sample['auxiliary_label'] = torch.tensor(np.stack([mem, dst], axis=0), dtype=torch.float32)
        if self.aux_dim > 2:
            raise NotImplementedError
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
