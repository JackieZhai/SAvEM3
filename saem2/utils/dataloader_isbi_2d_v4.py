# Copyright by HQ-SAM team
# All rights reserved.

## data loader
from __future__ import print_function, division

import numpy as np
import random
from copy import deepcopy
from skimage import io
import os
from glob import glob
import cv2
import json
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, RandomSampler, SequentialSampler
from torchvision import transforms, utils
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler

#### --------------------- dataloader online ---------------------####

def get_im_gt_name_dict(datasets, flag='valid', max_train=3000000, max_val=1000000, val_r = 10):
    print("------------------------------", flag, "--------------------------------")
    name_im_gt_list = []
    
    for i in range(len(datasets)):
        print("--->>>", flag, " dataset ",i,"/",len(datasets)," ",datasets[i]["name"],"<<<---")
        json_file_path = datasets[i]['path']
        # read json file
        with open(json_file_path, 'r') as file:
            json_data = json.load(file)
        # dataset_name：{neuron label：[[layer1, layer2, layer3],[layer7, layer8, layer9], ...]}
        for json_data_key, json_data_value in json_data.items():
            if 'AxonEM' not in json_data_key:
                dataset_name = json_data_key[:]
            else:
                # AxonEM-H_labels/
                dataset_name = json_data_key[0:9] + json_data_key[13:]
            single_dataset_part_list = []
            single_dataset_layer_list = []
            single_dataset_label_list = []
            layer_num = len(json_data_value)
            val_num = layer_num // val_r
            assert val_num > 0
            # print('Dataset: ', dataset_name, 'Number of neurons', len(json_data_value))
            value_list = list(json_data_value.items())
            if flag=='valid':
                select_value_list = value_list[-val_num:]
            elif flag=='train':
                select_value_list = value_list[:-val_num]
            else:
                raise ValueError

            for neu_layer, useful_label_list in select_value_list:

                for useful_label in useful_label_list:

                    single_dataset_part_list.append(None)
                    single_dataset_layer_list.append(int(neu_layer))
                    single_dataset_label_list.append(int(useful_label))
            assert len(single_dataset_part_list) == len(single_dataset_layer_list)
            assert len(single_dataset_part_list) == len(single_dataset_label_list)
            if flag=='valid':
                random_num = min(len(single_dataset_part_list), max_val)
            elif flag=='train':
                random_num = min(len(single_dataset_part_list), max_train)
            else:
                raise ValueError
            random.seed(42)

            # 生成相同的随机索引列表
            random_indices = random.sample(range(len(single_dataset_part_list)), len(single_dataset_part_list))

            # 使用随机索引重新排列两个列表
            single_dataset_part_list = [single_dataset_part_list[i] for i in random_indices]
            single_dataset_layer_list = [single_dataset_layer_list[i] for i in random_indices]
            single_dataset_label_list = [single_dataset_label_list[i] for i in random_indices]

            if len(single_dataset_part_list[:random_num]) > 0:

                name_im_gt_list.append({"dataset_name":dataset_name,
                                        "part":single_dataset_part_list[:random_num],
                                        "layer":single_dataset_layer_list[:random_num],
                                        "label":single_dataset_label_list[:random_num]})
            
            print(flag, 'dataset_name:', dataset_name, 'sample nums', len(single_dataset_part_list[:random_num]))

    return name_im_gt_list

# def create_dataloaders(name_im_gt_list, my_transforms=[], batch_size=1, training=False):
#     gos_dataloaders = []
#     gos_datasets = []

#     if(len(name_im_gt_list)==0):
#         return gos_dataloaders, gos_datasets

#     num_workers_ = 1
#     if(batch_size>1):
#         num_workers_ = 2
#     if(batch_size>4):
#         num_workers_ = 4
#     if(batch_size>8):
#         num_workers_ = 8


#     if training:
#         for i in range(len(name_im_gt_list)):   
#             gos_dataset = OnlineDataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms))
#             gos_datasets.append(gos_dataset)

#         gos_dataset = ConcatDataset(gos_datasets)
#         sampler = DistributedSampler(gos_dataset)
#         batch_sampler_train = torch.utils.data.BatchSampler(
#             sampler, batch_size, drop_last=True)
#         dataloader = DataLoader(gos_dataset, batch_sampler=batch_sampler_train, num_workers=num_workers_)

#         gos_dataloaders = dataloader
#         gos_datasets = gos_dataset

#     else:
#         for i in range(len(name_im_gt_list)):   
#             gos_dataset = OnlineDataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms), eval_ori_resolution = True)
#             sampler = DistributedSampler(gos_dataset, shuffle=False)
#             dataloader = DataLoader(gos_dataset, batch_size, sampler=sampler, drop_last=False, num_workers=num_workers_)

#             gos_dataloaders.append(dataloader)
#             gos_datasets.append(gos_dataset)

#     return gos_dataloaders, gos_datasets

def create_dataloaders(name_im_gt_list, my_transforms=[], batch_size=1, training=False):
    gos_dataloaders = []
    gos_datasets = []

    if(len(name_im_gt_list)==0):
        return gos_dataloaders, gos_datasets

    num_workers_ = 0
    if 'SAVEM3_NUM_WORKERS' in os.environ:
        num_workers_ = int(os.environ['SAVEM3_NUM_WORKERS'])
    elif batch_size > 1:
        num_workers_ = 2
    if batch_size > 4:
        num_workers_ = 4
    if batch_size > 8:
        num_workers_ = 8

    if training:
        for i in range(len(name_im_gt_list)):   
            gos_dataset = OnlineDataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms))
            gos_datasets.append(gos_dataset)

        gos_dataset = ConcatDataset(gos_datasets)
        # sampler = DistributedSampler(gos_dataset)
        #####guojy
        try:
            # 尝试使用DistributedSampler
            sampler = DistributedSampler(gos_dataset)
            # print("Using DistributedSampler")
        except:
            # 如果出错，切换到RandomSampler
            # print("DistributedSampler not available, using RandomSampler")
            sampler = RandomSampler(gos_dataset)
        #####
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler, batch_size, drop_last=True)
        dataloader = DataLoader(gos_dataset, batch_sampler=batch_sampler_train, num_workers=num_workers_)

        gos_dataloaders = dataloader
        gos_datasets = gos_dataset

    else:
        # guojy
        num_workers_ = 0
        for i in range(len(name_im_gt_list)):   
            gos_dataset = OnlineDataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms), eval_ori_resolution = True)
            # sampler = DistributedSampler(gos_dataset, shuffle=False)
            #####guojy
            try:
                # 尝试使用DistributedSampler
                sampler = DistributedSampler(gos_dataset, shuffle=False)
                # print("Using DistributedSampler")
            except:
                # 如果出错，切换到SequentialSampler
                # print("DistributedSampler not available, using SequentialSampler")
                sampler = SequentialSampler(gos_dataset)
            #####
            dataloader = DataLoader(gos_dataset, batch_size, sampler=sampler, drop_last=False, num_workers=num_workers_)

            gos_dataloaders.append(dataloader)
            gos_datasets.append(gos_dataset)

    return gos_dataloaders, gos_datasets

class OnlineDataset(Dataset):
    def __init__(self, name_im_gt_list, transform=None, eval_ori_resolution=False):

        self.transform = transform
        self.datasets = {}
        self.embed_root_path = os.environ.get('SAVEM3_EMBED_ROOT',
    os.path.join(os.environ.get('SAVEM3_DATA_ROOT', '.'), 'prepared_embedding/'))
        self.label_root_path = os.environ.get('SAVEM3_LABEL_ROOT',
    os.path.join(os.environ.get('SAVEM3_DATA_ROOT', '.'), 'prepared_segments_mul_2d/'))
        self.mem_root_path = os.environ.get('SAVEM3_MEM_ROOT',
    os.path.join(os.environ.get('SAVEM3_DATA_ROOT', '.'), 'prepared_segments_mul_mem/'))
        dt_name_list = []
        da_part_list = []
        da_layer_list = []
        da_label_list = []
        for i in range(0,len(name_im_gt_list)):
            dt_name_list.extend([name_im_gt_list[i]["dataset_name"] for x in name_im_gt_list[i]["part"]])
            da_part_list.extend(name_im_gt_list[i]["part"])
            da_layer_list.extend(name_im_gt_list[i]["layer"])
            da_label_list.extend(name_im_gt_list[i]["label"])

        self.datasets["dataset_name"] = dt_name_list
        self.datasets["part"] = da_part_list
        self.datasets["layer"] = da_layer_list
        self.datasets["label"] = da_label_list

        self.eval_ori_resolution = eval_ori_resolution
        # self.data_value={}

    def __len__(self):
        return len(self.datasets["dataset_name"])
    
    def read_tensor_from_folder(self, folder_path, layer, label, flag='gt'):
        img_c = io.imread(os.path.join(folder_path, '%05d.tif'%(layer)))
        # image_name = os.path.join(folder_path, '%05d.tif'%(layer))
        # if image_name in self.data_value:
        #     img_c = self.data_value[image_name].copy()
        # else:
        #     img_c = io.imread(image_name)
        #     self.data_value[image_name] = img_c.copy()

        if flag=='gt':
            img_c = (img_c == label).astype(np.uint8) * 255
            img_c =  cv2.resize(img_c, dsize=(1024, 1024), interpolation=cv2.INTER_NEAREST)
            img_c = torch.unsqueeze(torch.tensor(img_c, dtype=torch.float32),0)
        elif flag=='mem':
            img_c = (img_c > 0).astype(np.uint8) * 255
            img_c =  cv2.resize(img_c, dsize=(1024, 1024), interpolation=cv2.INTER_NEAREST)
            img_c = torch.unsqueeze(torch.tensor(img_c, dtype=torch.float32),0)
        elif flag=='embed':
            # 兼容两种缓存方向：
            #   precompute_teacher.py --write-tif 写 (H, W, C)
            #   旧数据/上游可能写 (C, H, W)
            if img_c.ndim == 3 and img_c.shape[-1] in (256, 768, 1280):
                img_c = np.ascontiguousarray(img_c.transpose(2, 0, 1))
            img_c = torch.tensor(img_c, dtype=torch.float32)
        else:
            raise ValueError
        return img_c

    def __getitem__(self, idx):
        sample_info_dataset = self.datasets["dataset_name"][idx]
        sample_info_part = self.datasets["part"][idx]
        sample_info_layer = self.datasets["layer"][idx]
        sample_info_label = self.datasets["label"][idx]

        # AxonEM 特殊处理
        if '/' in sample_info_dataset:
            gt_path = sample_info_dataset.split('/')[0] + '/seg_'+ sample_info_dataset.split('/')[1]
            embed_path = sample_info_dataset.split('/')[0] + '/im_'+ sample_info_dataset.split('/')[1]
            mem_path = sample_info_dataset.split('/')[0] + '/seg_'+ sample_info_dataset.split('/')[1]
        else:
            gt_path = sample_info_dataset[:]
            embed_path = sample_info_dataset[:]
            mem_path = sample_info_dataset[:]
        gt_c = \
            self.read_tensor_from_folder(os.path.join(self.label_root_path, gt_path), 
                                         layer=sample_info_layer, 
                                         label=sample_info_label, flag='gt')
        mem_c = \
            self.read_tensor_from_folder(os.path.join(self.mem_root_path, mem_path), 
                                         layer=sample_info_layer, 
                                         label=sample_info_label, flag='mem')

        embed_c = \
            self.read_tensor_from_folder(os.path.join(self.embed_root_path, embed_path+'/embed'), 
                                         layer=sample_info_layer, 
                                         label=sample_info_label, flag='embed')
        
        embed0_c = \
            self.read_tensor_from_folder(os.path.join(self.embed_root_path, embed_path+'/embed_interm_0'), 
                                         layer=sample_info_layer, 
                                         label=sample_info_label, flag='embed')


        sample = {
        "imidx": torch.from_numpy(np.array(idx)),
        "gt_c": gt_c,
        "mem_c": mem_c,
        "embed_c": embed_c,
        "embed0_c": embed0_c,
        "shape": torch.tensor(gt_c.shape[-2:]),
        }
        
        if self.transform:
            sample = self.transform(sample)

        if self.eval_ori_resolution:
            sample["ori_label"] = gt_c.type(torch.uint8)  # NOTE for evaluation only. And no flip here

        return sample