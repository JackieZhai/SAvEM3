from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import cv2
import h5py
import math
import time
import torch
import random
import numpy as np
import cc3d
from PIL import Image
import multiprocessing
from joblib import delayed
from joblib import Parallel
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from augmentation import Flip
from augmentation import Elastic
from augmentation import Grayscale
from augmentation import Rotate
from augmentation import Rescale
from utils.seg_util import mknhood3d, genSegMalis
from utils.aff_util import seg_to_affgraph
from utils.utils import center_crop
from data.data_segmentation import seg_widen_border, weight_binary_ratio
from data.data_affinity import seg_to_aff
# from utils.affinity_official import seg2affs
from utils.affinity_ours import gen_affs_mutex_3d

from utils.augmentation import IntensityAugment
from utils.consistency_aug import gen_mask, Filp_EMA


class Train2(Dataset):
    def __init__(self, cfg):
        super(Train2, self).__init__()
        # multiprocess settings
        num_cores = multiprocessing.cpu_count()
        self.parallel = Parallel(n_jobs=num_cores, backend='threading')
        self.cfg = cfg
        if cfg.MODEL.model_type.startswith('superhuman'):
            self.model_type = 'superhuman'
        else:
            self.model_type = cfg.MODEL.model_type
        if cfg.MODEL.model_type.endswith('shallow'):
            self.model_shallow = True
        else:
            self.model_shallow = False
        self.if_dilate = cfg.DATA.if_dilate

        self.if_aff = cfg.TRAIN.loss_aff_func is not None
        self.if_aff_ema = cfg.TRAIN.loss_aff_ema_func is not None
        self.if_discriminative = cfg.TRAIN.loss_discriminative_func is not None
        self.if_motion = cfg.TRAIN.loss_motion_func is not None
        
        self.shift_channels = cfg.shift
        self.output_nc = cfg.MODEL.output_nc
        self.separate_weight = cfg.TRAIN.separate_weight

        # basic settings
        # the input size of network
        if cfg.MODEL.model_type == 'superhuman':
            self.crop_size = [18, 160, 160]
            self.net_padding = [0, 0, 0]
        elif cfg.MODEL.model_type in ['superhuman_devo', 'superhuman_devo_shallow']:
            self.crop_size = [20, 160, 160]
            self.net_padding = [0, 0, 0]
        elif cfg.MODEL.model_type in ['superhuman_devo-t1_shallow', 'superhuman_devo-t1_triplet-MSE_shallow', 
                                      'superhuman_devo-t3_shallow', 'superhuman_devo-t3_triplet-MSE_shallow']:
            self.crop_size = [20, 240, 240]
            self.net_padding = [0, 0, 0]
        elif cfg.MODEL.model_type in ['superhuman_devo-t2_shallow', 'superhuman_devo-t2_triplet-MSE_shallow']:
            self.crop_size = [30, 240, 240]
            self.net_padding = [0, 0, 0]
        elif cfg.MODEL.model_type == 'mala':
            self.crop_size = [53, 268, 268]
            self.net_padding = [14, 106, 106]  # the edge size of patch reduced by network
        else:
            raise AttributeError('No this model type!')

        # the output size of network
        # for mala: [25, 56, 56]
        # for superhuman: [18, 160, 160]
        self.out_size = [self.crop_size[k] - 2 * self.net_padding[k] for k in range(len(self.crop_size))]

        # training dataset files (h5), may contain many datasets
        if cfg.DATA.dataset_name == 'cremi-A' or cfg.DATA.dataset_name == 'cremi':
            self.sub_path = 'cremi'
            self.train_datasets = ['cremiA_inputs_interp.h5']
            self.train_labels = ['cremiA_labels.h5']
        elif cfg.DATA.dataset_name == 'cremi-B':
            self.sub_path = 'cremi'
            self.train_datasets = ['cremiB_inputs_interp.h5']
            self.train_labels = ['cremiB_labels.h5']
        elif cfg.DATA.dataset_name == 'cremi-C':
            self.sub_path = 'cremi'
            self.train_datasets = ['cremiC_inputs_interp.h5']
            self.train_labels = ['cremiC_labels.h5']
        elif cfg.DATA.dataset_name == 'cremi-all':
            self.sub_path = 'cremi'
            self.train_datasets = ['cremiA_inputs_interp.h5', 'cremiB_inputs_interp.h5', 'cremiC_inputs_interp.h5']
            self.train_labels = ['cremiA_labels.h5', 'cremiB_labels.h5', 'cremiC_labels.h5']
        elif cfg.DATA.dataset_name == 'isbi':
            self.sub_path = 'snemi3d'
            self.train_datasets = ['isbi_inputs.h5']
            self.train_labels = ['isbi_labels.h5']
        elif cfg.DATA.dataset_name == 'ac3':
            self.sub_path = 'AC3-AC4'
            self.train_datasets = ['AC3_inputs.h5']
            self.train_labels = ['AC3_labels.h5']
        elif cfg.DATA.dataset_name == 'ac4':
            self.sub_path = 'AC3-AC4'
            self.train_datasets = ['AC4_inputs.h5']
            self.train_labels = ['AC4_labels.h5']
        elif cfg.DATA.dataset_name == 'ac4-triplet':
            self.sub_path = 'AC3-AC4'
            self.train_datasets = ['AC4_inputs.h5']
            self.train_labels = ['AC4_labels.h5']
            self.train_features = ['AC4_features.h5']
            self.train_embeddings = ['AC4_embeddings.h5']
        elif cfg.DATA.dataset_name == 'ac4-triplet-flow':
            self.sub_path = 'AC3-AC4'
            self.train_datasets = ['AC4_inputs.h5']
            self.train_labels = ['AC4_labels.h5']
            self.train_features = ['AC4_features.h5']
            self.train_embeddings = ['AC4_embeddings.h5']
            self.train_flows = ['AC4_flows.h5']
        elif cfg.DATA.dataset_name == 'fib':
            self.sub_path = 'fib'
            self.train_datasets = ['fib1_inputs.h5']
            self.train_labels = ['fib1_labels.h5']
        else:
            raise AttributeError('No this dataset type!')

        # the path of datasets, need first-level and second-level directory, such as: os.path.join('../data', 'cremi')
        self.folder_name = os.path.join(cfg.DATA.data_folder, self.sub_path)
        assert len(self.train_datasets) == len(self.train_labels)

        # split training data
        self.train_split = cfg.DATA.train_split

        # augmentation
        self.if_scale_aug = cfg.DATA.if_scale_aug
        self.if_filp_aug = cfg.DATA.if_filp_aug
        self.if_elastic_aug = cfg.DATA.if_elastic_aug
        self.if_intensity_aug = cfg.DATA.if_intensity_aug
        self.if_rotation_aug = cfg.DATA.if_rotation_aug

        self.if_ema_flip = cfg.DATA.if_ema_flip
        self.if_ema_intensity = cfg.DATA.if_ema_intensity
        self.if_ema_mask = cfg.DATA.if_ema_mask

        self.ema_inte = IntensityAugment()
        self.ema_flip = Filp_EMA()

        # load dataset
        self.dataset = []
        self.labels = []
        for k in range(len(self.train_datasets)):
            print('load ' + self.train_datasets[k] + ' ...')
            
            # load raw data
            f_raw = h5py.File(os.path.join(self.folder_name, self.train_datasets[k]), 'r')
            data = f_raw['main'][:]
            f_raw.close()
            data = data[:self.train_split]
            self.dataset.append(data)

            # load labels
            f_label = h5py.File(os.path.join(self.folder_name, self.train_labels[k]), 'r')
            label = f_label['main'][:]
            f_label.close()
            label = label[:self.train_split]
            if self.if_dilate:
                if cfg.DATA.widen_way:
                    label = seg_widen_border(label, tsz_h=1)
                else:
                    label = genSegMalis(label, 1)
            self.labels.append(label)

            # load features
            if hasattr(self, 'train_features'):
                self.features = []
                f_feature = h5py.File(os.path.join(self.folder_name, self.train_features[k]), 'r')
                feature = f_feature['main'][:]
                f_feature.close()
                feature = feature[:, :self.train_split, :, :]
                self.features.append(feature)
            
            # load embeddings
            if hasattr(self, 'train_embeddings'):
                self.embeddings = []
                f_embedding = h5py.File(os.path.join(self.folder_name, self.train_embeddings[k]), 'r')
                embedding = f_embedding['main'][:]
                f_embedding.close()
                embedding = embedding[:, :self.train_split, :, :]
                self.embeddings.append(embedding)

            # load flows
            if hasattr(self, 'train_flows'):
                self.flows = []
                f_flow = h5py.File(os.path.join(self.folder_name, self.train_flows[k]), 'r')
                flow = f_flow['main'][:]
                f_flow.close()
                flow = flow[:, :self.train_split, :-1, :-1]  # for flow: z=z-1, y=y+1, x=x+1
                self.flows.append(flow)

        # padding when the shape(z) of raw data is smaller than the input of network
        numz_dataset = self.dataset[0].shape[0]
        if numz_dataset < self.crop_size[0]:
            padding_size_z_left = (self.crop_size[0] - numz_dataset) // 2
            if numz_dataset % 2 == 0:
                padding_size_z_right = padding_size_z_left
            else:
                padding_size_z_right = padding_size_z_left + 1
            for k in range(len(self.dataset)):
                self.dataset[k] = np.pad(self.dataset[k], ((padding_size_z_left, padding_size_z_right), \
                                                            (0, 0), \
                                                            (0, 0)), mode='reflect')
                self.labels[k] = np.pad(self.labels[k], ((padding_size_z_left, padding_size_z_right), \
                                                            (0, 0), \
                                                            (0, 0)), mode='reflect')
                self.features[k] = np.pad(self.features[k], ((0, 0), \
                                                            (padding_size_z_left, padding_size_z_right), \
                                                            (0, 0), \
                                                            (0, 0)), mode='reflect')
                self.embeddings[k] = np.pad(self.embeddings[k], ((0, 0), \
                                                            (padding_size_z_left, padding_size_z_right), \
                                                            (0, 0), \
                                                            (0, 0)), mode='reflect')
                self.flows[k] = np.pad(self.flows[k], ((0, 0), \
                                                            (padding_size_z_left, padding_size_z_right), \
                                                            (0, 0), \
                                                            (0, 0)), mode='reflect')
                self.flows[k][:, :padding_size_z_left, :, :] *= -1; self.flows[k][:, -padding_size_z_right:, :, :] *= -1

        # padding by 'reflect' mode for mala network
        if cfg.MODEL.model_type == 'mala':
            for k in range(len(self.dataset)):
                self.dataset[k] = np.pad(self.dataset[k], ((self.net_padding[0], self.net_padding[0]), \
                                                           (self.net_padding[1], self.net_padding[1]), \
                                                           (self.net_padding[2], self.net_padding[2])), mode='reflect')
                self.labels[k] = np.pad(self.labels[k], ((self.net_padding[0], self.net_padding[0]), \
                                                           (self.net_padding[1], self.net_padding[1]), \
                                                           (self.net_padding[2], self.net_padding[2])), mode='reflect')

        # the training dataset size
        self.raw_data_shape = list(self.dataset[0].shape)
        print('raw data shape: ', self.raw_data_shape)

        # padding for random rotation
        self.crop_from_origin = [0, 0, 0]
        self.padding = cfg.DATA.padding
        self.crop_from_origin[0] = self.crop_size[0]
        self.crop_from_origin[1] = self.crop_size[1] + 2 * self.padding
        self.crop_from_origin[2] = self.crop_size[2] + 2 * self.padding

        # mask size
        if cfg.MODEL.model_type == 'mala':
            self.min_mask_size = [5, 5, 5]
            self.max_mask_size = [8, 12, 12]
            self.min_mask_counts = 0
            self.max_mask_counts = 60
        else:
            self.min_mask_size = [5, 10, 10]
            self.max_mask_size = [10, 20, 20]
            self.min_mask_counts = 0
            self.max_mask_counts = 60

        # augmentation initoalization
        self.augs_init()

    def __getitem__(self, index):
        # random select one dataset if contain many datasets
        k = random.randint(0, len(self.train_datasets)-1)
        used_data = self.dataset[k]
        used_label = self.labels[k]
        if hasattr(self, 'features') and hasattr(self, 'embeddings'):
            used_feature = self.features[k]
            used_embedding = self.embeddings[k]
        if hasattr(self, 'flows'):
            used_flow = self.flows[k]

        random_z = random.randint(0, self.raw_data_shape[0]-self.crop_from_origin[0])
        random_y = random.randint(0, self.raw_data_shape[1]-self.crop_from_origin[1])
        random_x = random.randint(0, self.raw_data_shape[2]-self.crop_from_origin[2])
        imgs = used_data[random_z:random_z+self.crop_from_origin[0], \
                        random_y:random_y+self.crop_from_origin[1], \
                        random_x:random_x+self.crop_from_origin[2]].copy()
        lb = used_label[random_z:random_z+self.crop_from_origin[0], \
                        random_y:random_y+self.crop_from_origin[1], \
                        random_x:random_x+self.crop_from_origin[2]].copy()
        if hasattr(self, 'features') and hasattr(self, 'embeddings'):
            feat = used_feature[:, random_z:random_z+self.crop_from_origin[0], \
                        random_y//4:random_y//4+self.crop_from_origin[1]//4, \
                        random_x//4:random_x//4+self.crop_from_origin[2]//4].copy()
            embed = used_embedding[:, random_z:random_z+self.crop_from_origin[0], \
                        random_y//16:random_y//16+self.crop_from_origin[1]//16, \
                        random_x//16:random_x//16+self.crop_from_origin[2]//16].copy()
        if hasattr(self, 'flows'):
            flow = used_flow[:, random_z:random_z+self.crop_from_origin[0], \
                        random_y//4:random_y//4+self.crop_from_origin[1]//4, \
                        random_x//4:random_x//4+self.crop_from_origin[2]//4].copy()

        imgs = imgs.astype(np.float32) / 255.0
        data = {'image': imgs, 'label': lb}
        if hasattr(self, 'features') and hasattr(self, 'embeddings'):
            data['feature'] = feat
            data['embedding'] = embed
        if hasattr(self, 'flows'):
            data['flow'] = flow
            
        # p=0.5 for augmentation
        if np.random.rand() < 0.5:
            data = self.augs_mix(data)
        imgs = data['image']
        lb = data['label']
        imgs = center_crop(imgs, det_shape=self.crop_size)
        lb = center_crop(lb, det_shape=self.crop_size)
        if hasattr(self, 'features') and hasattr(self, 'embeddings'):
            feat = data['feature']
            embed = data['embedding']
            feat_tmp = np.zeros((feat.shape[0], self.crop_size[0], 
                                 self.crop_size[1]//4, self.crop_size[2]//4), dtype=feat.dtype)
            for c in range(feat.shape[0]):
                feat_tmp[c] = center_crop(feat[c], 
                                          det_shape=[self.crop_size[0], self.crop_size[1]//4, self.crop_size[2]//4])
            feat = feat_tmp
            embed_tmp = np.zeros((embed.shape[0], self.crop_size[0], 
                                  self.crop_size[1]//16, self.crop_size[2]//16), dtype=feat.dtype)
            for c in range(embed.shape[0]):
                embed_tmp[c] = center_crop(embed[c], 
                                           det_shape=[self.crop_size[0], self.crop_size[1]//16, self.crop_size[2]//16])
            embed = embed_tmp
        if hasattr(self, 'flows'):
            flow = data['flow']
            flow_tmp = np.zeros((flow.shape[0], self.crop_size[0], 
                                 self.crop_size[1]//4, self.crop_size[2]//4), dtype=flow.dtype)
            for c in range(flow.shape[0]):
                flow_tmp[c] = center_crop(flow[c], 
                                          det_shape=[self.crop_size[0], self.crop_size[1]//4, self.crop_size[2]//4])
            flow = flow_tmp
        
        # relabel connected components
        lb = cc3d.connected_components(lb, connectivity=26)

        # downsample image and label
        if self.if_aff_ema or self.if_aff or self.if_discriminative:
            lb = np.transpose(lb, (1,2,0))
            label_x4 = cv2.resize(lb, (0,0), fx=1/4, fy=1/4, interpolation=cv2.INTER_NEAREST)
            label_x4 = np.transpose(label_x4, (2,0,1))
            label_x16 = cv2.resize(lb, (0,0), fx=1/16, fy=1/16, interpolation=cv2.INTER_NEAREST)
            label_x16 = np.transpose(label_x16, (2,0,1))
            lb = np.transpose(lb, (2,0,1))

        if self.model_shallow:
            imgs_tmp = np.zeros((imgs.shape[0], imgs.shape[1]//2, imgs.shape[2]//2), dtype=imgs.dtype)
            lb_tmp = np.zeros((lb.shape[0], lb.shape[1]//2, lb.shape[2]//2), dtype=lb.dtype)
            xy_ratio = 0.5
            for z in range(imgs.shape[0]):
                imgs_tmp[z] = cv2.resize(imgs[z], (0, 0), fx=xy_ratio, fy=xy_ratio, interpolation=cv2.INTER_LINEAR)
            for z in range(lb.shape[0]):
                lb_tmp[z] = cv2.resize(lb[z], (0, 0), fx=xy_ratio, fy=xy_ratio, interpolation=cv2.INTER_NEAREST)
            imgs = imgs_tmp
            lb = lb_tmp

        # convert label to affinity
        if self.model_type == 'mala':
            lb = lb[self.net_padding[0]:-self.net_padding[0], \
                    self.net_padding[1]:-self.net_padding[1], \
                    self.net_padding[2]:-self.net_padding[2]]
        # lb = genSegMalis(lb, 1)
        # lb_affs = seg_to_affgraph(lb, mknhood3d(1), pad='replicate').astype(np.float32)
        if self.shift_channels is None:
            if self.output_nc == 3:
                lb_affs = seg_to_aff(lb).astype(np.float32)
            elif self.output_nc == 1:
                lb_affs = seg_to_aff(lb).astype(np.float32)
                lb_affs = np.mean(lb_affs, axis=0, keepdims=True)
            elif self.output_nc == 12:
                nhood233 = np.asarray([-2, 0, 0, 0, -3, 0, 0, 0, -3]).reshape((3, 3))
                nhood399 = np.asarray([-3, 0, 0, 0, -9, 0, 0, 0, -9]).reshape((3, 3))
                nhood427 = np.asarray([-4, 0, 0, 0, -27, 0, 0, 0, -27]).reshape((3, 3))
                label111 = seg_to_aff(lb, pad='').astype(np.float32)
                label233 = seg_to_aff(lb, nhood233, pad='')
                label399 = seg_to_aff(lb, nhood399, pad='')
                label427 = seg_to_aff(lb, nhood427, pad='')
                lb_affs = np.concatenate((label111, label233, label399, label427), axis=0)
            else:
                raise NotImplementedError
        else:
            lb_affs = gen_affs_mutex_3d(lb, shift=self.shift_channels,
                                        padding=True, background=True)
        # lb = lb.astype(np.uint64)
        # lb_affs = seg2affs(lb, offsets=[[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
        # 				retain_mask=False,
        # 				ignore_label=0,
        # 				retain_segmentation=False,
        # 				segmentation_to_binary=False,
        # 				map_to_foreground=True,
        # 				learn_ignore_transitions=False)
        if self.if_aff_ema or self.if_aff:
            lb_affs_x4 = seg_to_aff(label_x4).astype(np.float32)
            lb_affs_x16 = seg_to_aff(label_x16).astype(np.float32)
            assert feat.shape[2] == lb_affs_x4.shape[2]
            assert feat.shape[3] == lb_affs_x4.shape[3]
            assert embed.shape[2] == lb_affs_x16.shape[2]
            assert embed.shape[3] == lb_affs_x16.shape[3]

        # generate weights map for affinity
        # weight_factor = np.sum(lb_affs) / np.size(lb_affs)
        # weight_factor = np.clip(weight_factor, 1e-3, 1)
        # weightmap = lb_affs * (1 - weight_factor) / weight_factor + (1 - lb_affs)
        # TODO:
        def _weightmap_direct_downsample(weightmap):
            weightmap_x4 = np.zeros((weightmap.shape[0], weightmap.shape[1], weightmap.shape[2]//2, weightmap.shape[3]//2), dtype=weightmap.dtype)
            for c in range(weightmap.shape[0]):
                for z in range(weightmap.shape[1]):
                    weightmap_x4[c, z] = cv2.resize(weightmap[c, z], (weightmap.shape[3]//2, weightmap.shape[2]//2), interpolation=cv2.INTER_LINEAR)
            weightmap_x16 = np.zeros((weightmap.shape[0], weightmap.shape[1], weightmap.shape[2]//8, weightmap.shape[3]//8), dtype=weightmap.dtype)
            for c in range(weightmap.shape[0]):
                for z in range(weightmap.shape[1]):
                    weightmap_x16[c, z] = cv2.resize(weightmap[c, z], (weightmap.shape[3]//8, weightmap.shape[2]//8), interpolation=cv2.INTER_LINEAR)
            return weightmap_x4, weightmap_x16

        if self.separate_weight:
            weightmap = np.zeros_like(lb_affs)
            for i in range(lb_affs.shape[0]):
                weightmap[i] = weight_binary_ratio(lb_affs[i])
            if hasattr(self, 'features') and hasattr(self, 'embeddings'):
                if self.if_aff_ema or self.if_aff:
                    weightmap_x4 = np.zeros_like(lb_affs_x4)
                    weightmap_x16 = np.zeros_like(lb_affs_x16)
                    for i in range(lb_affs_x4.shape[0]):
                        weightmap_x4[i] = weight_binary_ratio(lb_affs_x4[i])
                    for i in range(lb_affs_x16.shape[0]):
                        weightmap_x16[i] = weight_binary_ratio(lb_affs_x16[i])
                else:
                    weightmap_x4, weightmap_x16 = _weightmap_direct_downsample(weightmap)
        else:
            weightmap = weight_binary_ratio(lb_affs)
            if hasattr(self, 'features') and hasattr(self, 'embeddings'):
                if self.if_aff_ema or self.if_aff:
                    weightmap_x4 = weight_binary_ratio(lb_affs_x4)
                    weightmap_x16 = weight_binary_ratio(lb_affs_x16)
                else:
                    weightmap_x4, weightmap_x16 = _weightmap_direct_downsample(weightmap)

        if self.if_aff_ema:
            ema_imgs = imgs.copy()
            if self.if_ema_intensity:
                ema_imgs = self.ema_inte(ema_imgs)
            if self.if_ema_mask:
                mask = gen_mask(ema_imgs, model_type=self.model_type, \
                                min_mask_counts=self.min_mask_counts, \
                                max_mask_counts=self.max_mask_counts, \
                                min_mask_size=self.min_mask_size, \
                                max_mask_size=self.max_mask_size)
                ema_imgs = ema_imgs * mask
            if self.if_ema_flip:
                ema_imgs, rule = self.ema_flip(ema_imgs)
            else:
                rule = np.asarray([0,0,0], dtype=np.float32)
        
        # Norm images
        # if self.if_norm_images:
        # 	imgs = (imgs - 0.5) / 0.5
        # extend dimension
        imgs = imgs[np.newaxis, ...]
        imgs = np.ascontiguousarray(imgs, dtype=np.float32)
        lb_affs = np.ascontiguousarray(lb_affs, dtype=np.float32)
        weightmap = np.ascontiguousarray(weightmap, dtype=np.float32)
        if hasattr(self, 'features') and hasattr(self, 'embeddings'):
            feat = np.ascontiguousarray(feat, dtype=np.float32)
            embed = np.ascontiguousarray(embed, dtype=np.float32)
            weightmap_x4 = np.ascontiguousarray(weightmap_x4, dtype=np.float32)
            weightmap_x16 = np.ascontiguousarray(weightmap_x16, dtype=np.float32)
            if self.if_aff_ema:
                ema_imgs = ema_imgs[np.newaxis, ...]
                ema_imgs = np.ascontiguousarray(ema_imgs, dtype=np.float32)
                lb_affs_x4 = np.ascontiguousarray(lb_affs_x4, dtype=np.float32)
                lb_affs_x16 = np.ascontiguousarray(lb_affs_x16, dtype=np.float32)
                return imgs, ema_imgs, rule, lb_affs, feat, embed, lb_affs_x4, lb_affs_x16, weightmap, weightmap_x4, weightmap_x16
            elif self.if_aff:
                lb_affs_x4 = np.ascontiguousarray(lb_affs_x4, dtype=np.float32)
                lb_affs_x16 = np.ascontiguousarray(lb_affs_x16, dtype=np.float32)
                return imgs, lb_affs, feat, embed, lb_affs_x4, lb_affs_x16, weightmap, weightmap_x4, weightmap_x16
            elif self.if_discriminative:
                # lb = lb.astype(np.int32)
                lb_x4 = label_x4.astype(np.int32)
                lb_x16 = label_x16.astype(np.int32)
                return imgs, lb_x4, lb_x16, lb_affs, feat, embed, weightmap, weightmap_x4, weightmap_x16
            elif self.if_motion:
                flow = np.ascontiguousarray(flow, dtype=np.float32)
                return imgs, lb_affs, feat, embed, flow, weightmap, weightmap_x4, weightmap_x16
            else:
                return imgs, lb_affs, feat, embed, weightmap, weightmap_x4, weightmap_x16
        else:
            return imgs, lb_affs, weightmap

    def __len__(self):
        return int(sys.maxsize)

    def augs_init(self):
        # https://zudi-lin.github.io/pytorch_connectomics/build/html/notes/dataloading.html#data-augmentation
        self.aug_rotation = Rotate(p=0.5)
        self.aug_rescale = Rescale(p=0.5)
        self.aug_flip = Flip(p=1.0, do_ztrans=0)
        self.aug_elastic = Elastic(p=0.75, alpha=16, sigma=4.0)
        self.aug_grayscale = Grayscale(p=0.75)

    def augs_mix(self, data):
        if self.if_filp_aug and random.random() > 0.5:
            data = self.aug_flip(data)
        if self.if_rotation_aug and random.random() > 0.5:
            data = self.aug_rotation(data)
        if self.if_scale_aug and random.random() > 0.5:
            data = self.aug_rescale(data)
        if self.if_elastic_aug and random.random() > 0.5:
            data = self.aug_elastic(data)
        if self.if_intensity_aug and random.random() > 0.5:
            data = self.aug_grayscale(data)
        return data


class Train(Dataset):
    def __init__(self, cfg):
        super(Train, self).__init__()
        # multiprocess settings
        num_cores = multiprocessing.cpu_count()
        self.parallel = Parallel(n_jobs=num_cores, backend='threading')
        self.cfg = cfg
        if cfg.MODEL.model_type.startswith('superhuman'):
            self.model_type = 'superhuman'
        else:
            self.model_type = cfg.MODEL.model_type
        if cfg.MODEL.model_type.endswith('shallow'):
            self.model_shallow = True
        else:
            self.model_shallow = False
        self.if_dilate = cfg.DATA.if_dilate
        self.shift_channels = cfg.shift
        self.output_nc = cfg.MODEL.output_nc
        self.separate_weight = cfg.TRAIN.separate_weight

        # basic settings
        # the input size of network
        if cfg.MODEL.model_type == 'superhuman':
            self.crop_size = [18, 160, 160]
            self.net_padding = [0, 0, 0]
        elif cfg.MODEL.model_type in ['superhuman_devo', 'superhuman_devo_shallow']:
            self.crop_size = [20, 160, 160]
            self.net_padding = [0, 0, 0]
        elif cfg.MODEL.model_type in ['superhuman_devo-t1_shallow', 'superhuman_devo-t1_triplet-MSE_shallow', 
                                      'superhuman_devo-t3_shallow', 'superhuman_devo-t3_triplet-MSE_shallow']:
            self.crop_size = [20, 240, 240]
            self.net_padding = [0, 0, 0]
        elif cfg.MODEL.model_type in ['superhuman_devo-t2_shallow', 'superhuman_devo-t2_triplet-MSE_shallow']:
            self.crop_size = [30, 240, 240]
            self.net_padding = [0, 0, 0]
        elif cfg.MODEL.model_type == 'mala':
            self.crop_size = [53, 268, 268]
            self.net_padding = [14, 106, 106]  # the edge size of patch reduced by network
        else:
            raise AttributeError('No this model type!')

        # the output size of network
        # for mala: [25, 56, 56]
        # for superhuman: [18, 160, 160]
        self.out_size = [self.crop_size[k] - 2 * self.net_padding[k] for k in range(len(self.crop_size))]

        # training dataset files (h5), may contain many datasets
        if cfg.DATA.dataset_name == 'cremi-A' or cfg.DATA.dataset_name == 'cremi':
            self.sub_path = 'cremi'
            self.train_datasets = ['cremiA_inputs_interp.h5']
            self.train_labels = ['cremiA_labels.h5']
        elif cfg.DATA.dataset_name == 'cremi-B':
            self.sub_path = 'cremi'
            self.train_datasets = ['cremiB_inputs_interp.h5']
            self.train_labels = ['cremiB_labels.h5']
        elif cfg.DATA.dataset_name == 'cremi-C':
            self.sub_path = 'cremi'
            self.train_datasets = ['cremiC_inputs_interp.h5']
            self.train_labels = ['cremiC_labels.h5']
        elif cfg.DATA.dataset_name == 'cremi-all':
            self.sub_path = 'cremi'
            self.train_datasets = ['cremiA_inputs_interp.h5', 'cremiB_inputs_interp.h5', 'cremiC_inputs_interp.h5']
            self.train_labels = ['cremiA_labels.h5', 'cremiB_labels.h5', 'cremiC_labels.h5']
        elif cfg.DATA.dataset_name == 'isbi':
            self.sub_path = 'snemi3d'
            self.train_datasets = ['isbi_inputs.h5']
            self.train_labels = ['isbi_labels.h5']
        elif cfg.DATA.dataset_name == 'ac3':
            self.sub_path = 'AC3-AC4'
            self.train_datasets = ['AC3_inputs.h5']
            self.train_labels = ['AC3_labels.h5']
        elif cfg.DATA.dataset_name == 'ac4':
            self.sub_path = 'AC3-AC4'
            self.train_datasets = ['AC4_inputs.h5']
            self.train_labels = ['AC4_labels.h5']
        elif cfg.DATA.dataset_name == 'ac4-triplet':
            self.sub_path = 'AC3-AC4'
            self.train_datasets = ['AC4_inputs.h5']
            self.train_labels = ['AC4_labels.h5']
            self.train_features = ['AC4_features.h5']
            self.train_embeddings = ['AC4_embeddings.h5']
        elif cfg.DATA.dataset_name == 'fib':
            self.sub_path = 'fib'
            self.train_datasets = ['fib1_inputs.h5']
            self.train_labels = ['fib1_labels.h5']
        else:
            raise AttributeError('No this dataset type!')

        # the path of datasets, need first-level and second-level directory, such as: os.path.join('../data', 'cremi')
        self.folder_name = os.path.join(cfg.DATA.data_folder, self.sub_path)
        assert len(self.train_datasets) == len(self.train_labels)

        # split training data
        self.train_split = cfg.DATA.train_split

        # augmentation
        self.if_scale_aug = cfg.DATA.if_scale_aug
        self.if_filp_aug = cfg.DATA.if_filp_aug
        self.if_elastic_aug = cfg.DATA.if_elastic_aug
        self.if_intensity_aug = cfg.DATA.if_intensity_aug
        self.if_rotation_aug = cfg.DATA.if_rotation_aug

        # load dataset
        self.dataset = []
        self.labels = []
        for k in range(len(self.train_datasets)):
            print('load ' + self.train_datasets[k] + ' ...')
            
            # load raw data
            f_raw = h5py.File(os.path.join(self.folder_name, self.train_datasets[k]), 'r')
            data = f_raw['main'][:]
            f_raw.close()
            data = data[:self.train_split]
            self.dataset.append(data)

            # load labels
            f_label = h5py.File(os.path.join(self.folder_name, self.train_labels[k]), 'r')
            label = f_label['main'][:]
            f_label.close()
            label = label[:self.train_split]
            if self.if_dilate:
                if cfg.DATA.widen_way:
                    label = seg_widen_border(label, tsz_h=1)
                else:
                    label = genSegMalis(label, 1)
            self.labels.append(label)

            # load features
            if hasattr(self, 'train_features'):
                self.features = []

                f_feature = h5py.File(os.path.join(self.folder_name, self.train_features[k]), 'r')
                feature = f_feature['main'][:]
                f_feature.close()
                feature = feature[:, :self.train_split, :, :]
                self.features.append(feature)
            
            # load embeddings
            if hasattr(self, 'train_embeddings'):
                self.embeddings = []

                f_embedding = h5py.File(os.path.join(self.folder_name, self.train_embeddings[k]), 'r')
                embedding = f_embedding['main'][:]
                f_embedding.close()
                embedding = embedding[:, :self.train_split, :, :]
                self.embeddings.append(embedding)

        # padding when the shape(z) of raw data is smaller than the input of network
        numz_dataset = self.dataset[0].shape[0]
        if numz_dataset < self.crop_size[0]:
            padding_size_z_left = (self.crop_size[0] - numz_dataset) // 2
            if numz_dataset % 2 == 0:
                padding_size_z_right = padding_size_z_left
            else:
                padding_size_z_right = padding_size_z_left + 1
            for k in range(len(self.dataset)):
                self.dataset[k] = np.pad(self.dataset[k], ((padding_size_z_left, padding_size_z_right), \
                                                            (0, 0), \
                                                            (0, 0)), mode='reflect')
                self.labels[k] = np.pad(self.labels[k], ((padding_size_z_left, padding_size_z_right), \
                                                            (0, 0), \
                                                            (0, 0)), mode='reflect')
                self.features[k] = np.pad(self.features[k], ((0, 0), \
                                                            (padding_size_z_left, padding_size_z_right), \
                                                            (0, 0), \
                                                            (0, 0)), mode='reflect')
                self.embeddings[k] = np.pad(self.embeddings[k], ((0, 0), \
                                                            (padding_size_z_left, padding_size_z_right), \
                                                            (0, 0), \
                                                            (0, 0)), mode='reflect')

        # padding by 'reflect' mode for mala network
        if cfg.MODEL.model_type == 'mala':
            for k in range(len(self.dataset)):
                self.dataset[k] = np.pad(self.dataset[k], ((self.net_padding[0], self.net_padding[0]), \
                                                           (self.net_padding[1], self.net_padding[1]), \
                                                           (self.net_padding[2], self.net_padding[2])), mode='reflect')
                self.labels[k] = np.pad(self.labels[k], ((self.net_padding[0], self.net_padding[0]), \
                                                           (self.net_padding[1], self.net_padding[1]), \
                                                           (self.net_padding[2], self.net_padding[2])), mode='reflect')

        # the training dataset size
        self.raw_data_shape = list(self.dataset[0].shape)
        print('raw data shape: ', self.raw_data_shape)

        # padding for random rotation
        self.crop_from_origin = [0, 0, 0]
        self.padding = cfg.DATA.padding
        self.crop_from_origin[0] = self.crop_size[0]
        self.crop_from_origin[1] = self.crop_size[1] + 2 * self.padding
        self.crop_from_origin[2] = self.crop_size[2] + 2 * self.padding

        # augmentation initoalization
        self.augs_init()

    def __getitem__(self, index):
        # random select one dataset if contain many datasets
        k = random.randint(0, len(self.train_datasets)-1)
        used_data = self.dataset[k]
        used_label = self.labels[k]
        if hasattr(self, 'features') and hasattr(self, 'embeddings'):
            used_feature = self.features[k]
            used_embedding = self.embeddings[k]

        random_z = random.randint(0, self.raw_data_shape[0]-self.crop_from_origin[0])
        random_y = random.randint(0, self.raw_data_shape[1]-self.crop_from_origin[1])
        random_x = random.randint(0, self.raw_data_shape[2]-self.crop_from_origin[2])
        imgs = used_data[random_z:random_z+self.crop_from_origin[0], \
                        random_y:random_y+self.crop_from_origin[1], \
                        random_x:random_x+self.crop_from_origin[2]].copy()
        lb = used_label[random_z:random_z+self.crop_from_origin[0], \
                        random_y:random_y+self.crop_from_origin[1], \
                        random_x:random_x+self.crop_from_origin[2]].copy()
        if hasattr(self, 'features') and hasattr(self, 'embeddings'):
            feat = used_feature[:, random_z:random_z+self.crop_from_origin[0], \
                        random_y//4:random_y//4+self.crop_from_origin[1]//4, \
                        random_x//4:random_x//4+self.crop_from_origin[2]//4].copy()
            embed = used_embedding[:, random_z:random_z+self.crop_from_origin[0], \
                        random_y//16:random_y//16+self.crop_from_origin[1]//16, \
                        random_x//16:random_x//16+self.crop_from_origin[2]//16].copy()

        imgs = imgs.astype(np.float32) / 255.0
        data = {'image': imgs, 'label': lb}
        if hasattr(self, 'features') and hasattr(self, 'embeddings'):
            data['feature'] = feat
            data['embedding'] = embed
        # p=0.5 for augmentation
        if np.random.rand() < 0.5:
            data = self.augs_mix(data)
        imgs = data['image']
        lb = data['label']
        imgs = center_crop(imgs, det_shape=self.crop_size)
        lb = center_crop(lb, det_shape=self.crop_size)
        if hasattr(self, 'features') and hasattr(self, 'embeddings'):
            feat = data['feature']
            embed = data['embedding']
            feat_tmp = np.zeros((feat.shape[0], self.crop_size[0], 
                                 self.crop_size[1]//4, self.crop_size[2]//4), dtype=feat.dtype)
            for c in range(feat.shape[0]):
                feat_tmp[c] = center_crop(feat[c], 
                                          det_shape=[self.crop_size[0], self.crop_size[1]//4, self.crop_size[2]//4])
            feat = feat_tmp
            embed_tmp = np.zeros((embed.shape[0], self.crop_size[0], 
                                  self.crop_size[1]//16, self.crop_size[2]//16), dtype=feat.dtype)
            for c in range(embed.shape[0]):
                embed_tmp[c] = center_crop(embed[c], 
                                           det_shape=[self.crop_size[0], self.crop_size[1]//16, self.crop_size[2]//16])
            embed = embed_tmp

        # downsample image and label
        if self.model_shallow:
            imgs_tmp = np.zeros((imgs.shape[0], imgs.shape[1]//2, imgs.shape[2]//2), dtype=imgs.dtype)
            lb_tmp = np.zeros((lb.shape[0], lb.shape[1]//2, lb.shape[2]//2), dtype=lb.dtype)
            xy_ratio = 0.5
            for z in range(imgs.shape[0]):
                imgs_tmp[z] = cv2.resize(imgs[z], (0, 0), fx=xy_ratio, fy=xy_ratio, interpolation=cv2.INTER_LINEAR)
            for z in range(lb.shape[0]):
                lb_tmp[z] = cv2.resize(lb[z], (0, 0), fx=xy_ratio, fy=xy_ratio, interpolation=cv2.INTER_NEAREST)
            imgs = imgs_tmp
            lb = lb_tmp

        # convert label to affinity
        if self.model_type == 'mala':
            lb = lb[self.net_padding[0]:-self.net_padding[0], \
                    self.net_padding[1]:-self.net_padding[1], \
                    self.net_padding[2]:-self.net_padding[2]]
        # lb = genSegMalis(lb, 1)
        # lb_affs = seg_to_affgraph(lb, mknhood3d(1), pad='replicate').astype(np.float32)
        if self.shift_channels is None:
            if self.output_nc == 3:
                lb_affs = seg_to_aff(lb).astype(np.float32)
            elif self.output_nc == 1:
                lb_affs = seg_to_aff(lb).astype(np.float32)
                lb_affs = np.mean(lb_affs, axis=0, keepdims=True)
            elif self.output_nc == 12:
                nhood233 = np.asarray([-2, 0, 0, 0, -3, 0, 0, 0, -3]).reshape((3, 3))
                nhood399 = np.asarray([-3, 0, 0, 0, -9, 0, 0, 0, -9]).reshape((3, 3))
                nhood427 = np.asarray([-4, 0, 0, 0, -27, 0, 0, 0, -27]).reshape((3, 3))
                label111 = seg_to_aff(lb, pad='').astype(np.float32)
                label233 = seg_to_aff(lb, nhood233, pad='')
                label399 = seg_to_aff(lb, nhood399, pad='')
                label427 = seg_to_aff(lb, nhood427, pad='')
                lb_affs = np.concatenate((label111, label233, label399, label427), axis=0)
            else:
                raise NotImplementedError
        else:
            lb_affs = gen_affs_mutex_3d(lb, shift=self.shift_channels,
                                        padding=True, background=True)
        # lb = lb.astype(np.uint64)
        # lb_affs = seg2affs(lb, offsets=[[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
        # 				retain_mask=False,
        # 				ignore_label=0,
        # 				retain_segmentation=False,
        # 				segmentation_to_binary=False,
        # 				map_to_foreground=True,
        # 				learn_ignore_transitions=False)

        # generate weights map for affinity
        # weight_factor = np.sum(lb_affs) / np.size(lb_affs)
        # weight_factor = np.clip(weight_factor, 1e-3, 1)
        # weightmap = lb_affs * (1 - weight_factor) / weight_factor + (1 - lb_affs)
        if self.separate_weight:
            weightmap = np.zeros_like(lb_affs)
            for i in range(lb_affs.shape[0]):
                weightmap[i] = weight_binary_ratio(lb_affs[i])
        else:
            weightmap = weight_binary_ratio(lb_affs)

        # Norm images
        # if self.if_norm_images:
        # 	imgs = (imgs - 0.5) / 0.5
        # extend dimension
        imgs = imgs[np.newaxis, ...]
        imgs = np.ascontiguousarray(imgs, dtype=np.float32)
        lb_affs = np.ascontiguousarray(lb_affs, dtype=np.float32)
        weightmap = np.ascontiguousarray(weightmap, dtype=np.float32)
        if hasattr(self, 'features') and hasattr(self, 'embeddings'):
            feat = np.ascontiguousarray(feat, dtype=np.float32)
            embed = np.ascontiguousarray(embed, dtype=np.float32)
            
            # downsample weightmap
            weightmap_4x = np.zeros((weightmap.shape[0], weightmap.shape[1], weightmap.shape[2]//2, weightmap.shape[3]//2), dtype=weightmap.dtype)
            for c in range(weightmap.shape[0]):
                for z in range(weightmap.shape[1]):
                    weightmap_4x[c, z] = cv2.resize(weightmap[c, z], (weightmap.shape[3]//2, weightmap.shape[2]//2), interpolation=cv2.INTER_LINEAR)
            weightmap_16x = np.zeros((weightmap.shape[0], weightmap.shape[1], weightmap.shape[2]//8, weightmap.shape[3]//8), dtype=weightmap.dtype)
            for c in range(weightmap.shape[0]):
                for z in range(weightmap.shape[1]):
                    weightmap_16x[c, z] = cv2.resize(weightmap[c, z], (weightmap.shape[3]//8, weightmap.shape[2]//8), interpolation=cv2.INTER_LINEAR)
            
            weightmap_4x = np.ascontiguousarray(weightmap_4x, dtype=np.float32)
            weightmap_16x = np.ascontiguousarray(weightmap_16x, dtype=np.float32)
            return imgs, lb_affs, feat, embed, weightmap, weightmap_4x, weightmap_16x
        else:
            return imgs, lb_affs, weightmap

    def __len__(self):
        return int(sys.maxsize)

    def augs_init(self):
        # https://zudi-lin.github.io/pytorch_connectomics/build/html/notes/dataloading.html#data-augmentation
        self.aug_rotation = Rotate(p=0.5)
        self.aug_rescale = Rescale(p=0.5)
        self.aug_flip = Flip(p=1.0, do_ztrans=0)
        self.aug_elastic = Elastic(p=0.75, alpha=16, sigma=4.0)
        self.aug_grayscale = Grayscale(p=0.75)

    # TO DO
    def augs_single(self, data):
        random_id = np.random.randint(1, 5+1)
        if random_id == 1:
            data = self.aug_rotation(data)
        elif random_id == 2:
            data = self.aug_rescale(data)
        elif random_id == 3:
            data = self.aug_flip(data)
        elif random_id == 4:
            data = self.aug_elastic(data)
        elif random_id == 5:
            data = self.aug_grayscale(data)
        else:
            raise NotImplementedError
        return data

    def augs_mix(self, data):
        if self.if_filp_aug and random.random() > 0.5:
            data = self.aug_flip(data)
        if self.if_rotation_aug and random.random() > 0.5:
            data = self.aug_rotation(data)
        if self.if_scale_aug and random.random() > 0.5:
            data = self.aug_rescale(data)
        if self.if_elastic_aug and random.random() > 0.5:
            data = self.aug_elastic(data)
        if self.if_intensity_aug and random.random() > 0.5:
            data = self.aug_grayscale(data)
        return data


def collate_fn(batchs):
    out_input = []
    for batch in batchs:
        out_input.append(torch.from_numpy(batch['image']))
    
    out_input = torch.stack(out_input, 0)
    return {'image':out_input}

class Provider(object):
    def __init__(self, stage, cfg):
            #patch_size, batch_size, num_workers, is_cuda=True):
        self.stage = stage
        if self.stage == 'train':
            self.data = Train(cfg)
            self.batch_size = cfg.TRAIN.batch_size
            self.num_workers = cfg.TRAIN.num_workers
        elif self.stage == 'train2':
            self.data = Train2(cfg)
            self.batch_size = cfg.TRAIN.batch_size
            self.num_workers = cfg.TRAIN.num_workers
        elif self.stage == 'valid':
            # return valid(folder_name, kwargs['data_list'])
            pass
        else:
            raise AttributeError('Stage must be train/valid')
        self.is_cuda = cfg.TRAIN.if_cuda
        self.data_iter = None
        self.iteration = 0
        self.epoch = 1
    
    def __len__(self):
        return self.data.num_per_epoch
    
    def build(self):
        if self.stage == 'train':
            self.data_iter = iter(DataLoader(dataset=self.data, batch_size=self.batch_size, num_workers=self.num_workers,
                                             shuffle=False, drop_last=False, pin_memory=True))
        elif self.stage == 'train2':
            self.data_iter = iter(DataLoader(dataset=self.data, batch_size=self.batch_size, num_workers=self.num_workers,
                                             shuffle=False, drop_last=False, pin_memory=True))
        else:
            self.data_iter = iter(DataLoader(dataset=self.data, batch_size=1, num_workers=0,
                                             shuffle=False, drop_last=False, pin_memory=True))
    
    def next(self):
        if self.data_iter is None:
            self.build()
        try:
            batch = self.data_iter.next()
            self.iteration += 1
            if self.is_cuda:
                for b in range(len(batch)):
                    batch[b] = batch[b].cuda()
            return batch
        except StopIteration:
            self.epoch += 1
            self.build()
            self.iteration += 1
            batch = self.data_iter.next()
            if self.is_cuda:
                for b in range(len(batch)):
                    batch[b] = batch[b].cuda()
            return batch

def show(img3d):
    # only used for image with shape [18, 160, 160]
    num = img3d.shape[0]
    column = 5
    row = math.ceil(num / float(column))
    size = img3d.shape[1]
    img_all = np.zeros((size*row, size*column), dtype=np.uint8)
    for i in range(row):
        for j in range(column):
            index = i*column + j
            if index >= num:
                img = np.zeros_like(img3d[0], dtype=np.uint8)
            else:
                img = (img3d[index] * 255).astype(np.uint8)
            img_all[i*size:(i+1)*size, j*size:(j+1)*size] = img
    return img_all


if __name__ == '__main__':
    import yaml
    from attrdict import AttrDict
    from utils.show import show_one
    from utils.shift_channels import shift_func
    """"""
    seed = 555
    np.random.seed(seed)
    random.seed(seed)
    cfg_file = 'seg_c12_3d_ac4_data80_aug2_nodt.yaml'
    with open('./config/' + cfg_file, 'r') as f:
        cfg = AttrDict( yaml.load(f) )
    
    if cfg.DATA.shift_channels is not None:
        cfg.shift = shift_func(cfg.DATA.shift_channels)
    else:
        cfg.shift = None
    out_path = os.path.join('./', 'data_temp')
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    data = Train(cfg)
    t = time.time()
    for i in range(0, 50):
        t1 = time.time()
        tmp_data, affs, weightmap = iter(data).__next__()
        print('single cost time: ', time.time()-t1)
        tmp_data = np.squeeze(tmp_data)
        if cfg.MODEL.model_type == 'mala':
            tmp_data = tmp_data[14:-14,106:-106,106:-106]
        affs_xy = affs[-1]
        weightmap_xy = weightmap[-1]

        img_data = show_one(tmp_data)
        img_affs = show_one(affs_xy)
        img_weight = show_one(weightmap_xy)
        im_cat = np.concatenate([img_data, img_affs, img_weight], axis=1)
        Image.fromarray(im_cat).save(os.path.join(out_path, str(i).zfill(4)+'.png'))
    print(time.time() - t)