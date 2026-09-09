from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import yaml
import time
import logging
import argparse
import numpy as np
from attrdict import AttrDict
from tensorboardX import SummaryWriter
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.data_provider_labeled import Provider
from data.data_provider_valid import Provider_valid
from loss.loss import WeightedMSE, WeightedBCE, MSELoss, BCELoss, L1Loss
from loss.loss_embedding_mse import embedding_loss_norm1, ema_embedding_loss_norm1
from loss.loss_embedding_mse import embedding_loss_norm5, ema_embedding_loss_norm5
from loss.loss_embedding_mse import inf_embedding_loss_norm1_graph
from loss.loss_embedding import discriminative_loss
from loss.loss_embedding_motion import motion_loss_fast, motion_loss_fast_2
from loss.loss_embedding_motion import smooth_loss, smooth_weighted_loss
from utils.show import show_affs, show_affs_feats_embeds
from model.unet3d_mala import UNet3D_MALA
from model.model_superhuman import UNet_PNI, UNetS_PNI, UNetST_PNI
from utils.utils import setup_seed
from utils.shift_channels import shift_func
from utils.utils_rag_fc import construct_graph, calculate_mutual_node_similarity

from utils.fragment import watershed
from utils.consistency_aug import convert_consistency_flip
from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref
import warnings
warnings.filterwarnings("ignore")

def init_project(cfg):
    def init_logging(path):
        logging.basicConfig(
                level    = logging.INFO,
                format   = '%(message)s',
                datefmt  = '%m-%d %H:%M',
                filename = path,
                filemode = 'w')

        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)

        # set a format which is simpler for console use
        formatter = logging.Formatter('%(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    # seeds
    setup_seed(cfg.TRAIN.random_seed)
    if cfg.TRAIN.if_cuda:
        if (torch.cuda.is_available() is False) and (torch.backends.mps.is_available() is False):
            raise AttributeError('No GPU/MPS available')

    prefix = cfg.time
    model_name = prefix + '_' + cfg.NAME
    cfg.save_path = os.path.join(cfg.TRAIN.save_path, model_name)
    cfg.record_path = os.path.join(cfg.save_path, 'record')
    cfg.valid_path = os.path.join(cfg.save_path, 'valid')
    cfg.cache_path = os.path.join(cfg.save_path, 'cache')
    if not os.path.exists(cfg.cache_path):
        os.makedirs(cfg.cache_path)
    if not os.path.exists(cfg.save_path):
        os.makedirs(cfg.save_path)
    if not os.path.exists(cfg.record_path):
        os.makedirs(cfg.record_path)
    if not os.path.exists(cfg.valid_path):
        os.makedirs(cfg.valid_path)
    if not os.path.exists(os.path.join(cfg.save_path, 'model')):
        os.makedirs(os.path.join(cfg.save_path, 'model'))
    init_logging(os.path.join(cfg.record_path, prefix + '.log'))
    logging.info(cfg)
    writer = SummaryWriter(cfg.record_path)
    writer.add_text('cfg', str(cfg))
    return writer

def load_dataset(cfg):
    print('Caching datasets ... ', end='', flush=True)
    t1 = time.time()
    train_provider = Provider('train2', cfg)
    if cfg.TRAIN.if_valid:
        valid_provider = Provider_valid(cfg)
    else:
        valid_provider = None
    print('Done (time: %.2fs)' % (time.time() - t1))
    return train_provider, valid_provider

def build_model(cfg, writer):
    print('Building model on ', end='', flush=True)
    t1 = time.time()
    device = (torch.device('mps') if torch.backends.mps.is_available()
          else (torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')))
    if cfg.MODEL.model_type == 'mala':
        model = UNet3D_MALA(output_nc=cfg.MODEL.output_nc, 
                            if_sigmoid=cfg.MODEL.if_sigmoid, 
                            init_mode=cfg.MODEL.init_mode_mala).to(device)
    elif cfg.MODEL.model_type.startswith('superhuman'):
        if cfg.MODEL.model_type.endswith('shallow'):
            if 'triplet' in cfg.MODEL.model_type:
                model = UNetST_PNI(in_planes=cfg.MODEL.input_nc,
                                   out_planes=cfg.MODEL.output_nc,
                                   filters=cfg.MODEL.filters,
                                   upsample_mode=cfg.MODEL.upsample_mode,
                                   decode_ratio=cfg.MODEL.decode_ratio,
                                   merge_mode=cfg.MODEL.merge_mode,
                                   pad_mode=cfg.MODEL.pad_mode,
                                   bn_mode=cfg.MODEL.bn_mode,
                                   relu_mode=cfg.MODEL.relu_mode,
                                   init_mode=cfg.MODEL.init_mode).to(device)
            else:
                model = UNetS_PNI(in_planes=cfg.MODEL.input_nc,
                                  out_planes=cfg.MODEL.output_nc,
                                  filters=cfg.MODEL.filters,
                                  upsample_mode=cfg.MODEL.upsample_mode,
                                  decode_ratio=cfg.MODEL.decode_ratio,
                                  merge_mode=cfg.MODEL.merge_mode,
                                  pad_mode=cfg.MODEL.pad_mode,
                                  bn_mode=cfg.MODEL.bn_mode,
                                  relu_mode=cfg.MODEL.relu_mode,
                                  init_mode=cfg.MODEL.init_mode).to(device)
        else:
            model = UNet_PNI(in_planes=cfg.MODEL.input_nc,
                             out_planes=cfg.MODEL.output_nc,
                             filters=cfg.MODEL.filters,
                             upsample_mode=cfg.MODEL.upsample_mode,
                             decode_ratio=cfg.MODEL.decode_ratio,
                             merge_mode=cfg.MODEL.merge_mode,
                             pad_mode=cfg.MODEL.pad_mode,
                             bn_mode=cfg.MODEL.bn_mode,
                             relu_mode=cfg.MODEL.relu_mode,
                             init_mode=cfg.MODEL.init_mode).to(device)
    else:
        raise AttributeError('No this model type!')
        
    print('Load {:s} model!'.format(cfg.MODEL.model_type))

    cuda_count = torch.cuda.device_count()
    if cuda_count > 1:
        if cfg.TRAIN.batch_size % cuda_count == 0:
            print('%d GPUs ... ' % cuda_count, end='', flush=True)
            model = nn.DataParallel(model)
        else:
            raise AttributeError('Batch size (%d) cannot be equally divided by GPU number (%d)' % (cfg.TRAIN.batch_size, cuda_count))
    else:
        print('a single GPU ... ', end='', flush=True)
    print('Done (time: %.2fs)' % (time.time() - t1))
    return model

def resume_params(cfg, model, optimizer, resume):
    if resume:
        t1 = time.time()
        model_path = os.path.join(cfg.TRAIN.load_path, 'model', 'model-%06d.ckpt' % cfg.TRAIN.model_id)

        print('Resuming weights from %s ... ' % model_path, end='', flush=True)
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_weights'])
            # optimizer.load_state_dict(checkpoint['optimizer_weights'])
        else:
            raise AttributeError('No checkpoint found at %s' % model_path)
        print('Done (time: %.2fs)' % (time.time() - t1))
        print('valid %d' % checkpoint['current_iter'])
        return model, optimizer, checkpoint['current_iter']
    else:
        return model, optimizer, 0

def calculate_lr(iters):
    if iters < cfg.TRAIN.warmup_iters:
        current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(float(iters) / cfg.TRAIN.warmup_iters, cfg.TRAIN.power) + cfg.TRAIN.end_lr
    else:
        if iters < cfg.TRAIN.decay_iters:
            current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(1 - float(iters - cfg.TRAIN.warmup_iters) / cfg.TRAIN.decay_iters, cfg.TRAIN.power) + cfg.TRAIN.end_lr
        else:
            current_lr = cfg.TRAIN.end_lr
    return current_lr


def loop(cfg, train_provider, valid_provider, model, criterion, optimizer, iters, writer):
    f_loss_txt = open(os.path.join(cfg.record_path, 'loss.txt'), 'a')
    f_valid_txt = open(os.path.join(cfg.record_path, 'valid.txt'), 'a')
    rcd_time = []
    sum_time = 0
    sum_loss = 0; sum_loss_feat = 0; sum_loss_embed = 0
    sum_loss_aff_feat = 0; sum_loss_aff_embed = 0; sum_loss_aff_ema_feat = 0; sum_loss_aff_ema_embed = 0
    sum_loss_discriminative_feat = 0; sum_loss_discriminative_embed = 0
    sum_loss_motion_feat = 0; sum_loss_motion_embed = 0; sum_loss_smooth_feat = 0; sum_loss_smooth_embed = 0
    sum_loss_graph_feat = 0; sum_loss_graph_embed = 0
    sum_loss_graph_node = 0; sum_loss_graph_edge = 0; sum_loss_graph_aff = 0
    
    if cfg.TRAIN.loss_func == 'MSELoss':
        criterion = MSELoss()
    elif cfg.TRAIN.loss_func == 'BCELoss':
        criterion = BCELoss()
    elif cfg.TRAIN.loss_func == 'WeightedBCELoss':
        criterion = WeightedBCE()
    elif cfg.TRAIN.loss_func == 'WeightedMSELoss':
        criterion = WeightedMSE()
    else:
        raise AttributeError("NO this criterion")
    if cfg.TRAIN.loss_lambda_func is not None:
        if cfg.TRAIN.loss_lambda_func == 'MSELoss':
            criterion_feat = MSELoss()
            criterion_embed = MSELoss()
        elif cfg.TRAIN.loss_lambda_func == 'WeightedMSELoss':
            criterion_feat = WeightedMSE()
            criterion_embed = WeightedMSE()
        elif cfg.TRAIN.loss_lambda_func == 'L1Loss':
            criterion_feat = L1Loss()
            criterion_embed = L1Loss()
        else:
            raise AttributeError("NO this criterion")
    if cfg.TRAIN.loss_aff_func is not None:
        if cfg.TRAIN.loss_aff_func == 'WeightedMSELoss':
            criterion_aff = embedding_loss_norm1
        elif cfg.TRAIN.loss_aff_func == 'LRWeightedMSELoss':
            criterion_aff = embedding_loss_norm5
        else:
            raise AttributeError("NO this criterion")
    if cfg.TRAIN.loss_aff_ema_func is not None:
        if cfg.TRAIN.loss_aff_ema_func == 'WeightedMSELoss':
            criterion_aff_ema = ema_embedding_loss_norm1
        elif cfg.TRAIN.loss_aff_ema_func == 'LRWeightedMSELoss':
            criterion_aff_ema = ema_embedding_loss_norm5
        else:
            raise AttributeError("NO this criterion")
    if cfg.TRAIN.loss_discriminative_func is not None:
        if cfg.TRAIN.loss_discriminative_func == 'DiscriminativeLoss':
            criterion_discriminative = discriminative_loss
        else:
            raise AttributeError("NO this criterion")
    if cfg.TRAIN.loss_motion_func is not None:
        if cfg.TRAIN.loss_motion_func == 'MotionLoss':
            criterion_motion = motion_loss_fast
        elif cfg.TRAIN.loss_motion_func == 'MotionLoss2':
            criterion_motion = motion_loss_fast_2
        else:
            raise AttributeError("NO this criterion")
    if cfg.TRAIN.loss_smooth_func is not None:
        if cfg.TRAIN.loss_smooth_func == 'SmoothLoss':
            criterion_smooth = smooth_loss
        elif cfg.TRAIN.loss_smooth_func == 'WeightedSmoothLoss':
            criterion_smooth = smooth_weighted_loss
        else:
            raise AttributeError("NO this criterion")
    if cfg.TRAIN.loss_graph_func is not None:
        if cfg.TRAIN.loss_graph_func == 'WeightedMSELoss':
            criterion_graph = WeightedMSE()
        else:
            raise AttributeError("NO this criterion")

    if cfg.TRAIN.resume:
        iters -= 1  # resume init valid
    while iters <= cfg.TRAIN.total_iters:
        # train
        model.train()
        iters += 1
        t1 = time.time()
        
        # dataloader
        if 'triplet' in cfg.MODEL.model_type:
            if cfg.TRAIN.loss_aff_ema_func is not None:
                inputs, ema_inputs, ema_rules, target, target_feat, target_embed, aff_feat, aff_embed, weightmap, weightmap_x4, weightmap_x16 = train_provider.next()
            elif cfg.TRAIN.loss_aff_func is not None:
                inputs, target, target_feat, target_embed, aff_feat, aff_embed, weightmap, weightmap_x4, weightmap_x16 = train_provider.next()
            elif cfg.TRAIN.loss_discriminative_func is not None:
                inputs, target_gt_x4, target_gt_x16, target, target_feat, target_embed, weightmap, weightmap_x4, weightmap_x16 = train_provider.next()
            elif cfg.TRAIN.loss_motion_func is not None:
                inputs, target, target_feat, target_embed, target_motion, weightmap, weightmap_x4, weightmap_x16 = train_provider.next()
            elif cfg.TRAIN.loss_graph_func is not None:
                inputs, target_ins, target, target_feat, target_embed, weightmap, weightmap_x4, weightmap_x16 = train_provider.next()
            elif cfg.DATA.dataset_name.endswith('dst'):
                inputs, target, target_dst, target_feat, target_embed, weightmap, weightmap_x4, weightmap_x16 = train_provider.next()
                target = target_dst  # TODO: for dst
            else:
                inputs, target, target_feat, target_embed, weightmap, weightmap_x4, weightmap_x16 = train_provider.next()
        else:
            inputs, target, weightmap = train_provider.next()
        # print('time cost now [provider]: {:.5f}s'.format(time.time() - t1))
        
        # decay learning rate
        if cfg.TRAIN.end_lr == cfg.TRAIN.base_lr:
            current_lr = cfg.TRAIN.base_lr
        else:
            current_lr = calculate_lr(iters)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        
        ##############
        # loss
        optimizer.zero_grad()
        if 'triplet' in cfg.MODEL.model_type:
            pred, pred_feat, pred_embed = model(inputs)
            # print('pred:', pred.shape, 'target:', target.shape, 'weightmap:', weightmap.shape)
            # print('pred_feat:', pred_feat.shape, 'target_feat:', target_feat.shape, 'weightmap_x4:', weightmap_x4.shape)
            # print('pred_embed:', pred_embed.shape, 'target_embed:', target_embed.shape, 'weightmap_x16:', weightmap_x16.shape)
            
            def _loss_aff_pred_postprocess(pred):
                    shift = 1
                    pred[:, 1, :, :shift, :] = pred[:, 1, :, shift:shift*2, :]
                    pred[:, 2, :, :, :shift] = pred[:, 2, :, :, shift:shift*2]
                    pred[:, 0, :shift, :, :] = pred[:, 0, shift:shift*2, :, :]
                    pred = F.relu(pred)
                    return pred

            loss = criterion(pred, target, weightmap)
            
            if cfg.TRAIN.loss_lambda_func is not None:
                loss_feat = criterion_feat(pred_feat[:, :32, :, :, :], target_feat, 
                                           torch.mean(weightmap_x4, dim=1, keepdim=True) if weightmap_x4 is not None else None)
                loss_embed = criterion_embed(pred_embed, target_embed, 
                                             torch.mean(weightmap_x16, dim=1, keepdim=True) if weightmap_x16 is not None else None)
                loss += cfg.TRAIN.loss_lambda_feat * loss_feat + cfg.TRAIN.loss_lambda_embed * loss_embed
            
            if cfg.TRAIN.loss_aff_func is not None:
                loss_aff_feat, loss_aff_feat_pred = criterion_aff(pred_feat, aff_feat, weightmap_x4, WeightedMSE())
                loss_aff_embed, loss_aff_embed_pred = criterion_aff(pred_embed, aff_embed, weightmap_x16, WeightedMSE())
                loss_aff_feat_pred = _loss_aff_pred_postprocess(loss_aff_feat_pred)
                loss_aff_embed_pred = _loss_aff_pred_postprocess(loss_aff_embed_pred)
                loss += cfg.TRAIN.loss_aff_feat * loss_aff_feat + cfg.TRAIN.loss_aff_embed * loss_aff_embed

            if cfg.TRAIN.loss_aff_ema_func is not None:
                _, ema_pred_feat, ema_pred_embed = model(ema_inputs)
                if cfg.DATA.if_ema_flip:
                    ema_pred_feat = convert_consistency_flip(ema_pred_feat, ema_rules)
                loss_aff_ema_feat, _ = criterion_aff_ema(pred_feat, ema_pred_feat, aff_feat, weightmap_x4, WeightedMSE())
                loss_aff_ema_embed, _ = criterion_aff_ema(pred_embed, ema_pred_embed, aff_embed, weightmap_x16, WeightedMSE())
                loss += cfg.TRAIN.loss_aff_ema_feat * loss_aff_ema_feat + cfg.TRAIN.loss_aff_ema_embed * loss_aff_ema_embed
            
            if cfg.TRAIN.loss_discriminative_func is not None:
                loss_discriminative_feat = criterion_discriminative(pred_feat, target_gt_x4, background=True)
                # loss_discriminative_embed = criterion_discriminative(pred_embed, target_gt_x16, background=True)
                loss_discriminative_embed = criterion_discriminative(F.interpolate(pred_embed, scale_factor=(1, 4, 4), mode='trilinear', align_corners=True), 
                                                                     target_gt_x4, background=True)
                loss += cfg.TRAIN.loss_discriminative_feat * loss_discriminative_feat + cfg.TRAIN.loss_discriminative_embed * loss_discriminative_embed

            if cfg.TRAIN.loss_motion_func is not None:
                loss_motion_feat = torch.tensor(0)
                loss_motion_embed = torch.tensor(0)
                if iters >= cfg.TRAIN.loss_motion_iter:
                    loss_motion_feat = criterion_motion(pred_feat, target_motion, interp=cfg.TRAIN.loss_motion_interp)
                    if cfg.TRAIN.loss_motion_embed is not None:
                        loss_motion_embed = criterion_motion(pred_embed, target_motion[:, :, :, ::4, ::4], interp=1)
                loss += cfg.TRAIN.loss_motion_feat * loss_motion_feat + cfg.TRAIN.loss_motion_embed * loss_motion_embed

            if cfg.TRAIN.loss_smooth_func is not None:
                loss_smooth_feat = torch.tensor(0)
                loss_smooth_embed = torch.tensor(0)
                if iters >= cfg.TRAIN.loss_smooth_iter:
                    loss_smooth_feat = criterion_smooth(pred_feat, interp=cfg.TRAIN.loss_smooth_interp, weightmap=weightmap_x4)
                    if cfg.TRAIN.loss_smooth_embed is not None:
                        loss_smooth_embed = criterion_smooth(pred_embed, interp=1, weightmap=weightmap_x16)
                loss += cfg.TRAIN.loss_smooth_feat * loss_smooth_feat + cfg.TRAIN.loss_smooth_embed * loss_smooth_embed

            if cfg.TRAIN.loss_graph_func is not None:
                target_ins = target_ins.unsqueeze(1)
                assert len(target_ins.shape)==5

                h_list, edge_list = construct_graph(target_ins, [pred_feat[:, :32, :, :, :]], if_adjacent=cfg.TRAIN.if_neighbor)
                h_list_T, _ = construct_graph(target_ins, [target_feat])
                loss_graph_feat, loss_node_feat, loss_edge_feat = calculate_mutual_node_similarity(
                    h_list_T, h_list, edge_list, if_node=cfg.TRAIN.if_node,
                    if_edge_discrepancy=cfg.TRAIN.if_edge_discrepancy, if_edge_relation=cfg.TRAIN.if_edge_relation,
                    node_weight=cfg.TRAIN.loss_graph_node, edge_weight=cfg.TRAIN.loss_graph_edge)
                pred_feat_aff = inf_embedding_loss_norm1_graph(pred_feat)
                target_feat_aff = inf_embedding_loss_norm1_graph(target_feat)
                loss_graph_feat_aff = criterion_graph(pred_feat_aff, target_feat_aff) * cfg.TRAIN.loss_graph_aff
                loss += cfg.TRAIN.loss_graph_feat * (loss_graph_feat + loss_graph_feat_aff)
                
                if cfg.TRAIN.loss_graph_embed is not None:
                    h_list, edge_list = construct_graph(target_ins, [pred_embed], if_adjacent=cfg.TRAIN.if_neighbor)
                    h_list_T, _ = construct_graph(target_ins, [target_embed])
                    loss_graph_embed, loss_node_embed, loss_edge_embed = calculate_mutual_node_similarity(
                        h_list_T, h_list, edge_list, if_node=cfg.TRAIN.if_node,
                        if_edge_discrepancy=cfg.TRAIN.if_edge_discrepancy, if_edge_relation=cfg.TRAIN.if_edge_relation,
                        node_weight=cfg.TRAIN.loss_graph_node, edge_weight=cfg.TRAIN.loss_graph_edge)
                    pred_embed_aff = inf_embedding_loss_norm1_graph(pred_embed)
                    target_embed_aff = inf_embedding_loss_norm1_graph(target_embed)
                    loss_graph_embed_aff = criterion_graph(pred_embed_aff, target_embed_aff) * cfg.TRAIN.loss_graph_aff
                    loss += cfg.TRAIN.loss_graph_embed * (loss_graph_embed + loss_graph_embed_aff)

            loss.backward()
        else:
            pred = model(inputs)
            loss = criterion(pred, target, weightmap)
            loss.backward()
        ##############

        if cfg.TRAIN.weight_decay is not None:
            for group in optimizer.param_groups:
                for param in group['params']:
                    param.data.mul_(1.0 - cfg.TRAIN.weight_decay * group['lr'])
        optimizer.step()
        # print('time cost now [provider+optimizer]: {:.5f}s'.format(time.time() - t1))
        
        # log
        sum_loss += loss.item()
        if 'triplet' in cfg.MODEL.model_type:
            if cfg.TRAIN.loss_lambda_func is not None:
                sum_loss_feat += loss_feat.item()
                sum_loss_embed += loss_embed.item()
            if cfg.TRAIN.loss_aff_ema_func is not None:
                sum_loss_aff_ema_feat += loss_aff_ema_feat.item()
                sum_loss_aff_ema_embed += loss_aff_ema_embed.item()
            if cfg.TRAIN.loss_aff_func is not None:
                sum_loss_aff_feat += loss_aff_feat.item()
                sum_loss_aff_embed += loss_aff_embed.item()
            if cfg.TRAIN.loss_discriminative_func is not None:
                sum_loss_discriminative_feat += loss_discriminative_feat.item()
                sum_loss_discriminative_embed += loss_discriminative_embed.item()
            if cfg.TRAIN.loss_motion_func is not None:
                sum_loss_motion_feat += loss_motion_feat.item()
                sum_loss_motion_embed += loss_motion_embed.item()
            if cfg.TRAIN.loss_smooth_func is not None:
                sum_loss_smooth_feat += loss_smooth_feat.item()
                sum_loss_smooth_embed += loss_smooth_embed.item()
            if cfg.TRAIN.loss_graph_func is not None:
                if cfg.TRAIN.loss_graph_embed is not None:
                    sum_loss_graph_feat += loss_graph_feat.item() + loss_graph_feat_aff.item()
                    sum_loss_graph_embed += loss_graph_embed.item() + loss_graph_embed_aff.item()
                    sum_loss_graph_node += loss_node_feat.item() + loss_node_embed.item()
                    sum_loss_graph_edge += loss_edge_feat.item() + loss_edge_embed.item()
                    sum_loss_graph_aff += loss_graph_feat_aff.item() + loss_graph_embed_aff.item()
                else:
                    sum_loss_graph_feat += loss_graph_feat.item() + loss_graph_feat_aff.item()
                    sum_loss_graph_node += loss_node_feat.item()
                    sum_loss_graph_edge += loss_edge_feat.item()
                    sum_loss_graph_aff += loss_graph_feat_aff.item()
        sum_time += time.time() - t1
        
        if iters % cfg.TRAIN.display_freq == 0 or iters == 1:
            rcd_time.append(sum_time)
            sum_loss_name = ['loss', 'loss_feat', 'loss_embed', 
                             'loss_aff_feat', 'loss_aff_embed', 'loss_aff_ema_feat', 'loss_aff_ema_embed', 
                             'loss_dis_feat', 'loss_dis_embed', 
                             'loss_mot_feat', 'loss_mot_embed', 'loss_smo_feat', 'loss_smo_embed',
                             'loss_gra_feat', 'loss_gra_embed', 'loss_gra_node', 'loss_gra_edge', 'loss_gra_aff']
            sum_loss_item = [sum_loss, sum_loss_feat, sum_loss_embed, 
                             sum_loss_aff_feat, sum_loss_aff_embed, sum_loss_aff_ema_feat, sum_loss_aff_ema_embed, 
                             sum_loss_discriminative_feat, sum_loss_discriminative_embed, 
                             sum_loss_motion_feat, sum_loss_motion_embed, sum_loss_smooth_feat, sum_loss_smooth_embed, 
                             sum_loss_graph_feat, sum_loss_graph_embed, 
                             sum_loss_graph_node, sum_loss_graph_edge, sum_loss_graph_aff]
            if iters == 1:
                info_str = 'step %d' % (iters)
                for loss_name, loss_item in zip(sum_loss_name, sum_loss_item):
                    if loss_item > 0:
                        info_str += ', %s = %.6f' % (loss_name, loss_item * 1)
                        writer.add_scalar(loss_name, loss_item * 1, iters)
                info_str += ' (wt: *1, lr: %.8f, et: %.2f sec, rd: %.2f min)' % (current_lr, 
                    sum_time, (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(np.asarray(rcd_time)) / 60)
                logging.info(info_str)
            else:
                info_str = 'step %d' % (iters)
                for loss_name, loss_item in zip(sum_loss_name, sum_loss_item):
                    if loss_item > 0:
                        info_str += ', %s = %.6f' % (loss_name, loss_item / cfg.TRAIN.display_freq * 1)
                        writer.add_scalar(loss_name, loss_item / cfg.TRAIN.display_freq * 1, iters)
                info_str += ' (wt: *1, lr: %.8f, et: %.2f sec, rd: %.2f min)' % (current_lr, 
                    sum_time, (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(np.asarray(rcd_time)) / 60)
                logging.info(info_str)

            txt_str = 'step = ' + str(iters)
            for loss_name, loss_item in zip(sum_loss_name, sum_loss_item):
                if loss_item > 0:
                    txt_str += ', ' + loss_name + ' = ' + str(loss_item / cfg.TRAIN.display_freq * 1)
            f_loss_txt.write(txt_str)
            f_loss_txt.write('\n')
            f_loss_txt.flush()
            sys.stdout.flush()

            sum_time = 0
            sum_loss = 0; sum_loss_feat = 0; sum_loss_embed = 0
            sum_loss_aff_feat = 0; sum_loss_aff_embed = 0; sum_loss_aff_ema_feat = 0; sum_loss_aff_ema_embed = 0
            sum_loss_discriminative_feat = 0; sum_loss_discriminative_embed = 0
            sum_loss_motion_feat = 0; sum_loss_motion_embed = 0; sum_loss_smooth_feat = 0; sum_loss_smooth_embed = 0
            sum_loss_graph_feat = 0; sum_loss_graph_embed = 0
            sum_loss_graph_node = 0; sum_loss_graph_edge = 0; sum_loss_graph_aff = 0

        def _show_gt(labels):
            max_label = np.max(labels)
            lut = np.random.randint(
                    low=0,
                    high=255,
                    size=(int(max_label + 1), 3),
                    dtype=np.uint8)
            lut[0] = 0
            colored_labels = lut[labels]
            colored_labels = colored_labels.transpose((0, 4, 1, 2, 3))
            return colored_labels

        # display
        if iters % cfg.TRAIN.valid_freq == 0 or iters == 1:
            show_affs(iters, inputs, pred[:,:3], target[:,:3], cfg.cache_path, model_type=cfg.MODEL.model_type)
            feat_channel = np.random.randint(32, size=3)
            show_affs(iters, F.interpolate(inputs, scale_factor=(1, 1/2, 1/2), mode='trilinear', align_corners=True),
                      F.relu(pred_feat[:,feat_channel]), target_feat[:,feat_channel], 
                      cfg.cache_path, model_type=cfg.MODEL.model_type, suffix='feat')
            embed_channel = np.random.randint(256, size=3)
            show_affs(iters, F.interpolate(inputs, scale_factor=(1, 1/8, 1/8), mode='trilinear', align_corners=True),
                      F.relu(pred_embed[:,embed_channel]), target_embed[:,embed_channel], 
                      cfg.cache_path, model_type=cfg.MODEL.model_type, suffix='embed')

            if cfg.TRAIN.loss_aff_func is not None:
                show_affs(iters, F.interpolate(inputs, scale_factor=(1, 1/2, 1/2), mode='trilinear', align_corners=True),
                          loss_aff_feat_pred[:,:3], aff_feat[:,:3], 
                          cfg.cache_path, model_type=cfg.MODEL.model_type, suffix='aff_feat')
                show_affs(iters, F.interpolate(inputs, scale_factor=(1, 1/8, 1/8), mode='trilinear', align_corners=True),
                          loss_aff_embed_pred[:,:3], aff_embed[:,:3], 
                          cfg.cache_path, model_type=cfg.MODEL.model_type, suffix='aff_embed')
            elif cfg.TRAIN.loss_discriminative_func is not None:
                show_affs(iters, F.interpolate(inputs, scale_factor=(1, 1/2, 1/2), mode='trilinear', align_corners=True),
                          F.relu(pred_feat[:,:3]), _show_gt(target_gt_x4.data.cpu().numpy())[:,:3], 
                          cfg.cache_path, model_type=cfg.MODEL.model_type, suffix='dis_feat')
                show_affs(iters, F.interpolate(inputs, scale_factor=(1, 1/8, 1/8), mode='trilinear', align_corners=True),
                          F.relu(pred_embed[:,:3]), _show_gt(target_gt_x16.data.cpu().numpy())[:,:3], 
                          cfg.cache_path, model_type=cfg.MODEL.model_type, suffix='dis_embed')
            elif cfg.TRAIN.loss_graph_func is not None:
                show_affs(iters, F.interpolate(inputs, scale_factor=(1, 1/2, 1/2), mode='trilinear', align_corners=True),
                          _loss_aff_pred_postprocess(pred_feat_aff[:, :32, :, :, :])[:,:3], _loss_aff_pred_postprocess(target_feat_aff)[:,:3], 
                          cfg.cache_path, model_type=cfg.MODEL.model_type, suffix='gra_feat')
                if cfg.TRAIN.loss_graph_embed is not None:
                    show_affs(iters, F.interpolate(inputs, scale_factor=(1, 1/8, 1/8), mode='trilinear', align_corners=True),
                              _loss_aff_pred_postprocess(pred_embed_aff)[:,:3], _loss_aff_pred_postprocess(target_embed_aff)[:,:3],
                              cfg.cache_path, model_type=cfg.MODEL.model_type, suffix='gra_embed')

        # valid
        if cfg.TRAIN.if_valid and (iters % cfg.TRAIN.save_freq == 0):  # or iters == 1):
            model.eval()
            dataloader = torch.utils.data.DataLoader(valid_provider, batch_size=1, num_workers=0,
                                            shuffle=False, drop_last=False, pin_memory=True)
            losses_valid = []
            for k, batch in enumerate(dataloader, 0):
                inputs, target, weightmap = batch
                inputs = inputs.cuda()
                target = target.cuda()
                weightmap = weightmap.cuda()
                with torch.no_grad():
                    pred, feat, embed = model(inputs)
                tmp_loss = criterion(pred, target, weightmap)
                losses_valid.append(tmp_loss.item())
                valid_provider.add_vol_all(pred.data.cpu().numpy()[0], 
                                            feat.data.cpu().numpy()[0], 
                                            embed.data.cpu().numpy()[0])
            epoch_loss = sum(losses_valid) / len(losses_valid)
            out_affs, out_feats, out_embeds = valid_provider.get_results_all()
            gt_affs = valid_provider.get_gt_affs().copy()
            gt_seg = valid_provider.get_gt_lb()
            valid_provider.reset_output()
            
            out_affs = np.repeat(out_affs, 3, axis=0)  # for membrane
            show_affs_feats_embeds(iters, out_affs, out_feats, out_embeds, gt_affs, cfg.valid_path)

            ##############
            # segmentation
            if cfg.TRAIN.if_seg:
                import waterz
                from utils.lmc import mc_baseline
                if iters > 1:
                    fragments = watershed(out_affs, 'maxima_distance')
                    sf = 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>'
                    seg_waterz = list(waterz.agglomerate(out_affs, [0.50],
                                                        fragments=fragments,
                                                        scoring_function=sf,
                                                        discretize_queue=256))[0]
                    arand_waterz = adapted_rand_ref(gt_seg, seg_waterz, ignore_labels=(0))[0]
                    voi_split, voi_merge = voi_ref(gt_seg, seg_waterz, ignore_labels=(0))
                    voi_sum_waterz = voi_split + voi_merge

                    seg_lmc = mc_baseline(out_affs)
                    arand_lmc = adapted_rand_ref(gt_seg, seg_lmc, ignore_labels=(0))[0]
                    voi_split, voi_merge = voi_ref(gt_seg, seg_lmc, ignore_labels=(0))
                    voi_sum_lmc = voi_split + voi_merge
                else:
                    voi_sum_waterz = 0.0
                    arand_waterz = 0.0
                    voi_sum_lmc = 0.0
                    arand_lmc = 0.0
                    print('model-%d, segmentation failed!' % iters)
            else:
                voi_sum_waterz = 0.0
                arand_waterz = 0.0
                voi_sum_lmc = 0.0
                arand_lmc = 0.0
            ##############

            # MSE
            whole_mse = np.sum(np.square(out_affs - gt_affs)) / np.size(gt_affs)
            out_affs = np.clip(out_affs, 0.000001, 0.999999)
            bce = -(gt_affs * np.log(out_affs) + (1 - gt_affs) * np.log(1 - out_affs))
            whole_bce = np.sum(bce) / np.size(gt_affs)
            out_affs[out_affs <= 0.5] = 0
            out_affs[out_affs > 0.5] = 1
            # whole_f1 = 1 - f1_score(gt_affs.astype(np.uint8).flatten(), out_affs.astype(np.uint8).flatten())
            whole_f1 = f1_score(1 - gt_affs.astype(np.uint8).flatten(), 1 - out_affs.astype(np.uint8).flatten())
            print('model-%d, valid-loss=%.6f, MSE-loss=%.6f, BCE-loss=%.6f, F1-score=%.6f, VOI-waterz=%.6f, ARAND-waterz=%.6f, VOI-lmc=%.6f, ARAND-lmc=%.6f' % \
                (iters, epoch_loss, whole_mse, whole_bce, whole_f1, voi_sum_waterz, arand_waterz, voi_sum_lmc, arand_lmc), flush=True)
            writer.add_scalar('valid/epoch_loss', epoch_loss, iters)
            writer.add_scalar('valid/mse_loss', whole_mse, iters)
            writer.add_scalar('valid/bce_loss', whole_bce, iters)
            writer.add_scalar('valid/f1_score', whole_f1, iters)
            writer.add_scalar('valid/voi_waterz', voi_sum_waterz, iters)
            writer.add_scalar('valid/arand_waterz', arand_waterz, iters)
            writer.add_scalar('valid/voi_lmc', voi_sum_lmc, iters)
            writer.add_scalar('valid/arand_lmc', arand_lmc, iters)
            f_valid_txt.write('model-%d, valid-loss=%.6f, MSE-loss=%.6f, BCE-loss=%.6f, F1-score=%.6f, VOI-waterz=%.6f, ARAND-waterz=%.6f, VOI-lmc=%.6f, ARAND-lmc=%.6f' % \
                            (iters, epoch_loss, whole_mse, whole_bce, whole_f1, voi_sum_waterz, arand_waterz, voi_sum_lmc, arand_lmc))
            f_valid_txt.write('\n')
            f_valid_txt.flush()
            torch.cuda.empty_cache()

        # save
        if iters % cfg.TRAIN.save_freq == 0:
            states = {'current_iter': iters, 'valid_result': None,
                    'model_weights': model.state_dict()}
            torch.save(states, os.path.join(cfg.save_path, 'model', 'model-%06d.ckpt' % iters))
            print('##############')
            print('save modol, iters = %d' % (iters), flush=True)
            print('##############')
    f_loss_txt.close()
    f_valid_txt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='mem3c2_3ds_t3', help='path to config file')
    parser.add_argument('-m', '--mode', type=str, default='train', help='path to config file')
    parser.add_argument('--no-valid', action='store_true',
                        help='Disable in-training validation/WaterZ/MC when postprocessing dependencies are absent')
    parser.add_argument('--fresh', action='store_true',
                        help='Ignore config resume/load_path and train from scratch')
    parser.add_argument('--num-workers', type=int, default=-1,
                        help='Override TRAIN.num_workers (-1 keeps the config value)')
    args = parser.parse_args()

    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)
    print('mode: ' + args.mode)

    with open('./config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.safe_load(f))
    if cfg.MODEL.output_nc == 2:
        raise ValueError('The historical BDY+DST placeholder has no implemented two-head loss; do not report it as a DST ablation')

    timeArray = time.localtime()
    time_stamp = time.strftime('%Y-%m-%d--%H-%M-%S', timeArray)
    print('time stamp:', time_stamp)

    cfg.path = cfg_file
    cfg.time = time_stamp
    if cfg.DATA.shift_channels is None:
        cfg.shift = None
    else:
        assert cfg.MODEL.output_nc == cfg.DATA.shift_channels, "output_nc must be equal to shift_channels"
        cfg.shift = shift_func(cfg.DATA.shift_channels)

    if args.no_valid:
        cfg['TRAIN']['if_valid'] = False
        cfg['TRAIN']['if_seg'] = False
    if args.fresh:
        cfg['TRAIN']['resume'] = False
    if args.num_workers >= 0:
        cfg['TRAIN']['num_workers'] = args.num_workers

    if args.mode == 'train':
        writer = init_project(cfg)
        train_provider, valid_provider = load_dataset(cfg)
        model = build_model(cfg, writer)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999),
                                 eps=0.01, weight_decay=1e-6, amsgrad=True)
        # optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999), eps=1e-8, amsgrad=False)
        # optimizer = optim.Adamax(model.parameters(), lr=cfg.TRAIN.base_l, eps=1e-8)
        model, optimizer, init_iters = resume_params(cfg, model, optimizer, cfg.TRAIN.resume)
        loop(cfg, train_provider, valid_provider, model, nn.L1Loss(), optimizer, init_iters, writer)
        writer.close()
    else:
        pass
    print('***Done***')



'''
cd /mnt/WGCJ/SAM/superhuman/scripts

CUDA_VISIBLE_DEVICES=0 /mnt/WGCJ/miniconda4/envs/micro-sam/bin/python main_devoem_sparse_membrane_triplet_2.py -c mem3c2_3ds_t3
'''
