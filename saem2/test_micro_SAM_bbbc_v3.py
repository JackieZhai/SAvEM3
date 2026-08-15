# Copyright by HQ-SAM team
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import copy
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import random
from typing import Dict, List, Tuple

from segment_anything_training import sam_model_registry_fast_fine
from segment_anything_training.modeling import TwoWayTransformer, MaskDecoder

from utils.dataloader_isbi_lm_v3 import get_im_gt_name_dict, create_dataloaders
from utils.loss_mask import loss_masks
import utils.misc as misc

os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')


def get_args_parser():
    parser = argparse.ArgumentParser('Micro_SAM', add_help=False)
    # guojy
    # parser.add_argument("--output", type=str, required=True, 
    #                     help="Path to the directory where masks and checkpoints will be output")
    # parser.add_argument("--model-type", type=str, default="vit_l", 
    #                     help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    # parser.add_argument("--checkpoint", type=str, required=True, 
    #                     help="The path to the SAM checkpoint to use for mask generation.")
    parser.add_argument("--output", type=str, 
                        default='work_dirs/test_micro_SAM_bbbc', 
                        help="Path to the directory where masks and checkpoints will be output")
    parser.add_argument("--model-type", type=str, default="vit_h", 
                        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    # parser.add_argument("--checkpoint", type=str, 
    #                     default='pretrained_checkpoint/sam_vit_b_01ec64.pth', 
    #                     help="The path to the SAM checkpoint to use for mask generation.")
    parser.add_argument("--checkpoint", type=str, 
                        default="/home/zhaih/share/SAM/sam-hq-main/train/pretrained_checkpoint_micro_sam/vit_h_lm.pth", 
                        help="The path to the SAM checkpoint to use for mask generation.")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="The device to run generation on.")

    ##### guojy
    parser.add_argument('--gpu', type=int, default=0, help='GPU device index')
    #####

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--lr_drop_epoch', default=10, type=int)
    parser.add_argument('--max_epoch_num', default=12, type=int)
    parser.add_argument('--input_size', default=[1024,1024], type=list)
    parser.add_argument('--batch_size_train', default=8, type=int)
    parser.add_argument('--batch_size_valid', default=1, type=int)
    parser.add_argument('--model_save_fre', default=1, type=int)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', type=int, help='local rank for dist')
    parser.add_argument('--find_unused_params', action='store_true')

    # parser.add_argument('--eval', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument("--restore-model", type=str,
                        help="The path to the hq_decoder training checkpoint for evaluation")
    # guojy eval
    parser.add_argument('--eval', default=True)
    # # parser.add_argument('--visualize', default=True)
    # parser.add_argument('--visualize', default=False)
    # parser.add_argument("--restore-model", type=str, 
    #                     default=None,
    #                     help="The path to the hq_decoder training checkpoint for evaluation")

    return parser.parse_args()

class ListContainer:
    def __init__(self, input_list):
        self.list = input_list
        self.sample_select = {'cremiA': 18000,
                              'cremiC': 12000,
                              'FIB25': 20000,
                              'HB-fb-inner': 10000,
                              'SNEMI': 18000,
                              'AxonEM-H/950-0-0': 1500,
                              'AxonEM-H/950-0-3584': 1500,
                              'AxonEM-H/950-3584-0': 1500,
                              'AxonEM-H/950-3584-3584': 1500,
                              'AxonEM-M/700-0-3584': 1500,
                              'AxonEM-M/700-3584-0': 1500,
                              'AxonEM-M/700-3584-3584': 1500,
                              'AxonEM-M/350-1792-1792': 1500,}
        for datasets_dict in self.list:
            datase_name = datasets_dict['dataset_name']
            assert len(datasets_dict['part']) == len(datasets_dict['layer'])
            assert len(datasets_dict['part']) == len(datasets_dict['label'])
            assert len(datasets_dict['part']) >= self.sample_select[datase_name]

    def get_list(self):
        # random.seed(42)
        out_list = []
        for datasets_dict in self.list:
            datasets_dict_saved = copy.deepcopy(datasets_dict)
            datase_name = datasets_dict_saved['dataset_name']

            # 生成相同的随机索引列表
            random_indices = random.sample(range(len(datasets_dict_saved['part'])), self.sample_select[datase_name])

            # 使用随机索引重新排列两个列表
            datasets_dict_saved['part'] = [datasets_dict_saved['part'][i] for i in random_indices]
            datasets_dict_saved['layer'] = [datasets_dict_saved['layer'][i] for i in random_indices]
            datasets_dict_saved['label'] = [datasets_dict_saved['label'][i] for i in random_indices]

            out_list.append(datasets_dict_saved)

        return out_list

def main(train_datasets, valid_datasets, args):

    misc.init_distributed_mode(args)
    print('world size: {}'.format(args.world_size))
    print('rank: {}'.format(args.rank))
    print('local_rank: {}'.format(args.local_rank))
    print("args: " + str(args) + '\n')

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    ### --- Step 1: Train or Valid dataset ---
    if not args.eval:
        print("--- create training dataloader ---")
        train_im_gt_list = get_im_gt_name_dict(train_datasets, flag="train")
        train_dataloaders, train_datasets = create_dataloaders(train_im_gt_list,
                                                        my_transforms = [],
                                                        batch_size = args.batch_size_train,
                                                        training = True)
        print(len(train_dataloaders), " train dataloaders created")


    print("--- create valid dataloader ---")
    valid_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid")
    valid_dataloaders, valid_datasets = create_dataloaders(valid_im_gt_list,
                                                          my_transforms = [],
                                                          batch_size=args.batch_size_valid,
                                                          training=False)
    print(len(valid_dataloaders), " valid dataloaders created")

    # len(valid_dataloaders) * args.batch_size_valid ~~ len(valid_im_gt_list)
    sam = sam_model_registry_fast_fine[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)
    try:
        sam = torch.nn.parallel.DistributedDataParallel(sam, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
    except:
        sam = nn.DataParallel(sam, device_ids=[args.gpu])
    ### --- Step 3: Train or Evaluate ---
    if not args.eval:
        print("--- define optimizer ---")
        optimizer = optim.Adam(sam.module.mask_decoder.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop_epoch)
        lr_scheduler.last_epoch = args.start_epoch

        train(args, sam, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler)
    else:
        # sam = torch.nn.parallel.DistributedDataParallel(sam, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        ##### guojy

        #####
        if args.restore_model:
            print("restore model from:", args.restore_model)
            if torch.cuda.is_available():
                sam.module.mask_decoder.load_state_dict(torch.load(args.restore_model))
            else:
                sam.module.mask_decoder.load_state_dict(torch.load(args.restore_model,map_location="cpu"))
    
        evaluate(args, sam, valid_dataloaders, args.visualize)


def train(args, sam, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler):
    if misc.is_main_process():
        os.makedirs(args.output, exist_ok=True)

    epoch_start = args.start_epoch
    epoch_num = args.max_epoch_num

    for epoch in range(epoch_start,epoch_num): 
        print("epoch:   ",epoch, "  learning rate:  ", optimizer.param_groups[0]["lr"])
        metric_logger = misc.MetricLogger(delimiter="  ")
        # train_dataloaders.batch_sampler.sampler.set_epoch(epoch)
        ##### guojy
        try:
            train_dataloaders.batch_sampler.sampler.set_epoch(epoch)
        except:
            pass
            # shuffled_indices = torch.randperm(len(gos_dataset))
            # new_sampler = RandomSampler(shuffled_indices)
            # dataloader = DataLoader(gos_dataset, batch_size=batch_size, sampler=new_sampler)
        #####
        for data in metric_logger.log_every(train_dataloaders,1000):
            # zhaih input: torch.float32, 0~255 [batch_size, channels, height, width]
            gt_c, mem_c, embed_c, embed0_c = data['gt_c'], data['mem_c'], data['embed_c'], data['embed0_c']

            # ##### debug gt_c.shape: (4,1,1024,1024); mem_c.shape: (4,1,1024,1024); embed_c.shape: (4, 256, 64, 64); embed0_c.shape: (4, 1280, 64, 64)
            # from skimage import io
            # sample_id = 1
            # io.imsave('/home/guojy2/share/guojy/nnUNetFrame/CHECK_DATA_TEMP/gt_c.tif', data['gt_c'].cpu().numpy()[sample_id].astype(np.uint8))
            # io.imsave('/home/guojy2/share/guojy/nnUNetFrame/CHECK_DATA_TEMP/mem_c.tif', data['mem_c'].cpu().numpy()[sample_id].astype(np.uint8))
            # io.imsave('/home/guojy2/share/guojy/nnUNetFrame/CHECK_DATA_TEMP/embed_c.tif', data['embed_c'].cpu().numpy()[sample_id])
            # io.imsave('/home/guojy2/share/guojy/nnUNetFrame/CHECK_DATA_TEMP/embed0_c.tif', data['embed0_c'].cpu().numpy()[sample_id])
            # #####

            if torch.cuda.is_available():
                gt_c = gt_c.cuda()
                mem_c = mem_c.cuda()
                embed_c = embed_c.cuda()
                embed0_c = embed0_c.cuda()
            
            # input prompt
            input_keys = ['box','point','noise_mask']
            labels_box = misc.masks_to_boxes(gt_c[:,0,:,:])
            try:
                labels_points = misc.masks_sample_points(gt_c[:,0,:,:])
            except:
                # less than 10 points
                input_keys = ['box','noise_mask']
            labels_256 = F.interpolate(gt_c, size=(256, 256), mode='bilinear')
            labels_noisemask = misc.masks_noise(labels_256)

            batched_input = []
            for b_i in range(len(gt_c)):
                dict_input = dict()
                dict_input['image'] = [embed_c, embed0_c] 
                input_type = random.choice(input_keys)
                if input_type == 'box':
                    dict_input['boxes'] = labels_box[b_i:b_i+1]
                elif input_type == 'point':
                    point_coords = labels_points[b_i:b_i+1]
                    dict_input['point_coords'] = point_coords
                    dict_input['point_labels'] = torch.ones(point_coords.shape[1], device=point_coords.device)[None,:]
                elif input_type == 'noise_mask':
                    dict_input['mask_inputs'] = labels_noisemask[b_i:b_i+1]
                else:
                    raise NotImplementedError
                dict_input['original_size'] = gt_c[b_i].shape[-2:]
                batched_input.append(dict_input)

            image_embeddings, interm_embeddings = batched_input[0]["image"]
            interm_embeddings = [interm_embeddings.permute(0, 2, 3, 1)]

            mask_outputs = []
            for image_record, curr_embedding in zip(batched_input, image_embeddings):
                if "point_coords" in image_record:
                    points = (image_record["point_coords"], image_record["point_labels"])
                else:
                    points = None
                with torch.no_grad():
                    sparse_embeddings, dense_embeddings = sam.module.prompt_encoder(
                        points=points,
                        boxes=image_record.get("boxes", None),
                        masks=image_record.get("mask_inputs", None),
                    )
                low_res_masks, iou_predictions = sam.module.mask_decoder(
                    image_embeddings=curr_embedding.unsqueeze(0),
                    image_pe=sam.module.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False
                )
                
                masks = sam.module.postprocess_masks(
                    low_res_masks,
                    input_size=image_record["original_size"],
                    original_size=image_record["original_size"],
                )
                # masks = masks > sam.module.mask_threshold
                mask_outputs.append(masks)
            masks_sam = torch.cat(mask_outputs, 0)
            
            loss_mask, loss_dice = loss_masks(masks_sam, gt_c/255.0, len(masks_sam))
            loss = loss_mask + loss_dice
            
            loss_dict = {"loss_mask": loss_mask, "loss_dice":loss_dice}

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            losses_reduced_scaled = sum(loss_dict_reduced.values())
            loss_value = losses_reduced_scaled.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metric_logger.update(training_loss=loss_value, **loss_dict_reduced)


        print("Finished epoch:      ", epoch)
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}

        lr_scheduler.step()
        test_stats = evaluate(args, sam, valid_dataloaders)
        train_stats.update(test_stats)
        
        sam.train()  

        if epoch % args.model_save_fre == 0:
            model_name = "/epoch_"+str(epoch)+".pth"
            print('come here save at', args.output + model_name)
            misc.save_on_master(sam.module.mask_decoder.state_dict(), args.output + model_name)
    
    # Finish training
    print("Training Reaches The Maximum Epoch Number")
    
    # merge sam and fine_mask_decoder
    if misc.is_main_process():
        sam_ckpt = torch.load(args.checkpoint)
        fine_mask_decoder = torch.load(args.output + model_name)
        for key in fine_mask_decoder.keys():
            sam_ckpt['mask_decoder.'+key] = fine_mask_decoder[key]
        model_name = "/sam_hq_epoch_"+str(epoch)+".pth"
        torch.save(sam_ckpt, args.output + model_name)



def compute_iou(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0,len(preds)):
        iou = iou + misc.mask_iou(postprocess_preds[i],target[i])
    return iou / len(preds)

def compute_boundary_iou(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0,len(preds)):
        iou = iou + misc.boundary_iou(target[i],postprocess_preds[i])
    return iou / len(preds)

def evaluate(args, sam, valid_dataloaders, visualize=False):
    sam.eval()
    print("Validating...")
    test_stats = {}

    for k in range(len(valid_dataloaders)):
        metric_logger = misc.MetricLogger(delimiter="  ")
        valid_dataloader = valid_dataloaders[k]
        print('valid_dataloader len:', len(valid_dataloader))

        for data_val in metric_logger.log_every(valid_dataloader,1000):
            gt_c_val, mem_c_val, embed_c_val, embed0_c_val = data_val['gt_c'], data_val['mem_c'], data_val['embed_c'], data_val['embed0_c']

            if torch.cuda.is_available():
                gt_c_val = gt_c_val.cuda()
                mem_c_val = mem_c_val.cuda()
                embed_c_val = embed_c_val.cuda()
                embed0_c_val = embed0_c_val.cuda()

            
            labels_box = misc.masks_to_boxes(gt_c_val[:,0,:,:])
            input_keys = ['box']
            batched_input = []
            for b_i in range(len(gt_c_val)):
                dict_input = dict()

                dict_input['image'] = [embed_c_val, embed0_c_val]  
                input_type = random.choice(input_keys)
                if input_type == 'box':
                    dict_input['boxes'] = labels_box[b_i:b_i+1]
                elif input_type == 'point':
                    point_coords = labels_points[b_i:b_i+1]
                    dict_input['point_coords'] = point_coords
                    dict_input['point_labels'] = torch.ones(point_coords.shape[1], device=point_coords.device)[None,:]
                elif input_type == 'noise_mask':
                    dict_input['mask_inputs'] = labels_noisemask[b_i:b_i+1]
                else:
                    raise NotImplementedError
                dict_input['original_size'] = gt_c_val[b_i].shape[-2:]
                batched_input.append(dict_input)

            image_embeddings, interm_embeddings = batched_input[0]["image"]
            interm_embeddings = [interm_embeddings.permute(0, 2, 3, 1)]

            mask_outputs = []
            for image_record, curr_embedding in zip(batched_input, image_embeddings):
                if "point_coords" in image_record:
                    points = (image_record["point_coords"], image_record["point_labels"])
                else:
                    points = None
                with torch.no_grad():
                    sparse_embeddings, dense_embeddings = sam.module.prompt_encoder(
                        points=points,
                        boxes=image_record.get("boxes", None),
                        masks=image_record.get("mask_inputs", None),
                    )
                    low_res_masks, iou_predictions = sam.module.mask_decoder(
                        image_embeddings=curr_embedding.unsqueeze(0),
                        image_pe=sam.module.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False
                    )
                    
                    masks = sam.module.postprocess_masks(
                        low_res_masks,
                        input_size=image_record["original_size"],
                        original_size=image_record["original_size"],
                    )
                    # masks = masks > sam.module.mask_threshold
                    mask_outputs.append(masks)
            masks_sam = torch.cat(mask_outputs, 0)

            iou = compute_iou(masks_sam, gt_c_val.to(torch.uint8))
            boundary_iou = compute_boundary_iou(masks_sam, gt_c_val.to(torch.uint8))

            loss_dict = {"val_iou_"+str(k): iou, "val_boundary_iou_"+str(k): boundary_iou}
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            metric_logger.update(**loss_dict_reduced)


        print('============================')
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
        test_stats.update(resstat)


    return test_stats


if __name__ == "__main__":

    record_train = {"name": "record_train",
                    "path": "/home/guojy2/share/guojy/ProjectCodes/BigModel/sam-hq/BBBC039v1_train.json"}
    record_valid = {"name": "record_valid",
                    "path": "/home/guojy2/share/guojy/ProjectCodes/BigModel/sam-hq/BBBC039v1_val.json"}

    train_datasets = [record_train,]
    valid_datasets = [record_valid,] 

    args = get_args_parser()

    main(train_datasets, valid_datasets, args)
