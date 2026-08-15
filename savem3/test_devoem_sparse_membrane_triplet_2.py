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
import cv2
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
from cloudvolume import CloudVolume
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

def resume_params(cfg, model, resume):
    if resume:
        t1 = time.time()
        model_path = os.path.join(cfg.TRAIN.load_path, 'model', 'model-%06d.ckpt' % cfg.TRAIN.model_id)

        print('Resuming weights from %s ... ' % model_path, end='', flush=True)
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_weights'])
        else:
            raise AttributeError('No checkpoint found at %s' % model_path)
        print('Done (time: %.2fs)' % (time.time() - t1))
        print('valid %d' % checkpoint['current_iter'])
        return model, checkpoint['current_iter']
    else:
        return model, 0


def loop(cfg, train_provider, valid_provider, model, iters, writer):
    f_valid_txt = open(os.path.join(cfg.record_path, 'valid.txt'), 'a')
    rcd_time = []
    sum_time = 0

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
    
    model.eval()
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
    
    if 'triplet' in cfg.MODEL.model_type:
        pred, pred_feat, pred_embed = model(inputs)
        # print('pred:', pred.shape, 'target:', target.shape, 'weightmap:', weightmap.shape)
        # print('pred_feat:', pred_feat.shape, 'target_feat:', target_feat.shape, 'weightmap_x4:', weightmap_x4.shape)
        # print('pred_embed:', pred_embed.shape, 'target_embed:', target_embed.shape, 'weightmap_x16:', weightmap_x16.shape)
        
        if cfg.TRAIN.loss_aff_ema_func is not None:
            _, ema_pred_feat, ema_pred_embed = model(ema_inputs)
            
    else:
        pred = model(inputs)

    # log
    sum_time += time.time() - t1
    
    if iters % cfg.TRAIN.display_freq == 0 or iters == 1:
        rcd_time.append(sum_time)

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
        for z in range(out_affs.shape[1]):
            show_affs_feats_embeds(iters, out_affs, out_feats, out_embeds, gt_affs, cfg.valid_path, z_mid=z)

        ##############
        # segmentation
        if cfg.TRAIN.if_seg:
            import waterz
            from utils.lmc import mc_baseline

            info = CloudVolume.create_new_info(
                num_channels = 1,
                layer_type = 'segmentation',  # 'image' or 'segmentation'
                data_type = 'uint32',  # can pick any popular uint
                encoding = 'compressed_segmentation',  # other options: 'jpeg', 'compressed_segmentation' (req. uint32 or uint64)
                resolution      = [6, 6, 30], # Voxel scaling, units are in nanometers
                voxel_offset    = [0, 0, 0], # x,y,z offset in voxels from the origin
                chunk_size      = [gt_seg.shape[2], gt_seg.shape[1], 1], # units are voxels
                volume_size     = [gt_seg.shape[2], gt_seg.shape[1], gt_seg.shape[0]],
            )

            if iters > 1:
                fragments = watershed(out_affs, 'maxima_distance')
                sf = 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>'
                seg_waterz = list(waterz.agglomerate(out_affs, [0.50],
                                                    fragments=fragments,
                                                    scoring_function=sf,
                                                    discretize_queue=256))[0]

                vol = CloudVolume('file://'+os.path.abspath(os.path.join(cfg.save_path, 'model', 'fragments')), info=info)
                vol.commit_info()
                vol[:] = fragments.astype(np.uint32).transpose()[..., np.newaxis]
                vol = CloudVolume('file://'+os.path.abspath(os.path.join(cfg.save_path, 'model', 'seg_waterz')), info=info)
                vol.commit_info()
                vol[:] = seg_waterz.astype(np.uint32).transpose()[..., np.newaxis]

                # upsample for shallow
                gt_seg_tmp = np.zeros((gt_seg.shape[0], gt_seg.shape[1]*2, gt_seg.shape[2]*2), dtype=np.uint32)
                for z in range(gt_seg.shape[0]):
                    gt_seg_tmp[z] = cv2.resize(gt_seg[z], (gt_seg.shape[2]*2, gt_seg.shape[1]*2), interpolation=cv2.INTER_NEAREST)
                gt_seg = gt_seg_tmp
                seg_waterz_tmp = np.zeros((seg_waterz.shape[0], seg_waterz.shape[1]*2, seg_waterz.shape[2]*2), dtype=np.uint32)
                for z in range(seg_waterz.shape[0]):
                    seg_waterz_tmp[z] = cv2.resize(seg_waterz[z], (seg_waterz.shape[2]*2, seg_waterz.shape[1]*2), interpolation=cv2.INTER_NEAREST)
                seg_waterz = seg_waterz_tmp
                arand_waterz = adapted_rand_ref(gt_seg, seg_waterz, ignore_labels=(0))[0]
                voi_split, voi_merge = voi_ref(gt_seg, seg_waterz, ignore_labels=(0))
                voi_sum_waterz = voi_split + voi_merge

                seg_lmc = mc_baseline(out_affs)

                vol = CloudVolume('file://'+os.path.abspath(os.path.join(cfg.save_path, 'model', 'seg_lmc')), info=info)
                vol.commit_info()
                vol[:] = seg_lmc.astype(np.uint32).transpose()[..., np.newaxis]

                # upsample for shallow
                seg_lmc_tmp = np.zeros((seg_lmc.shape[0], seg_lmc.shape[1]*2, seg_lmc.shape[2]*2), dtype=np.uint32)
                for z in range(seg_lmc.shape[0]):
                    seg_lmc_tmp[z] = cv2.resize(seg_lmc[z], (seg_lmc.shape[2]*2, seg_lmc.shape[1]*2), interpolation=cv2.INTER_NEAREST)
                seg_lmc = seg_lmc_tmp
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

    f_valid_txt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='mem3c2c_3ds_t3t', help='path to config file')
    parser.add_argument('-m', '--mode', type=str, default='train', help='path to config file')
    parser.add_argument('--no-valid', action='store_true',
                        help='关闭训练中验证/waterz/LMC 后处理（无后处理依赖时使用）')
    parser.add_argument('--fresh', action='store_true',
                        help='忽略 config 的 resume/load_path，从头训练')
    parser.add_argument('--num-workers', type=int, default=-1,
                        help='覆盖 TRAIN.num_workers（-1 表示使用 config）')
    args = parser.parse_args()

    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)
    print('mode: ' + args.mode)

    with open('./config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.safe_load(f))

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
        model, init_iters = resume_params(cfg, model, cfg.TRAIN.resume)
        loop(cfg, train_provider, valid_provider, model, init_iters, writer)
        writer.close()
    else:
        pass
    print('***Done***')



'''
cd /mnt/WGCJ/SAM/superhuman/scripts

CUDA_VISIBLE_DEVICES=0 /mnt/WGCJ/miniconda4/envs/micro-sam/bin/python test_devoem_sparse_membrane_triplet_2.py -c mem3c2c_3ds_t3t
'''