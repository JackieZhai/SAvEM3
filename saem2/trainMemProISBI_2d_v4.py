# Copyright by HQ-SAM team
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import copy
import time
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

from segment_anything_training import sam_model_registry_fast
from segment_anything_training.modeling import TwoWayTransformer, MaskDecoder

from utils.dataloader_isbi_2d_v4 import get_im_gt_name_dict, create_dataloaders
from utils.loss_mask import loss_masks
import utils.misc as misc

os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')

from modeling import MaskDecoderHQ



def show_anns(masks, input_point, input_box, input_label, filename, image, ious, boundary_ious):
    if len(masks) == 0:
        return

    for i, (mask, iou, biou) in enumerate(zip(masks, ious, boundary_ious)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if input_box is not None:
            show_box(input_box, plt.gca())
        if (input_point is not None) and (input_label is not None): 
            show_points(input_point, input_label, plt.gca())

        plt.axis('off')
        plt.savefig(filename+'_'+str(i)+'.png',bbox_inches='tight',pad_inches=-0.1)
        plt.close()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


def get_args_parser():
    parser = argparse.ArgumentParser('HQ-SAM', add_help=False)
    # guojy
    # parser.add_argument("--output", type=str, required=True, 
    #                     help="Path to the directory where masks and checkpoints will be output")
    # parser.add_argument("--model-type", type=str, default="vit_l", 
    #                     help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    # parser.add_argument("--checkpoint", type=str, required=True, 
    #                     help="The path to the SAM checkpoint to use for mask generation.")
    parser.add_argument("--output", type=str, 
                        default='work_dirs/hq_sam_h_memproisbi_2d_v4', 
                        help="Path to the directory where masks and checkpoints will be output")
    parser.add_argument("--model-type", type=str, default="vit_h", 
                        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    # parser.add_argument("--checkpoint", type=str, 
    #                     default='pretrained_checkpoint/sam_vit_b_01ec64.pth', 
    #                     help="The path to the SAM checkpoint to use for mask generation.")
    parser.add_argument("--checkpoint", type=str, 
                        default='pretrained_checkpoint/sam_vit_h_4b8939.pth', 
                        help="The path to the SAM checkpoint to use for mask generation.")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="The device to run generation on.")

    ##### guojy
    parser.add_argument('--gpu', type=int, default=0, help='GPU device index')
    #####

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--start_epoch', default=11, type=int)
    parser.add_argument('--lr_drop_epoch', default=10, type=int)
    parser.add_argument('--max_epoch_num', default=16, type=int)
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
    parser.add_argument('--record-train', type=str, default='./record2d_train.json')
    parser.add_argument('--record-valid', type=str, default='',
                        help='Validation record2d JSON; defaults to --record-train')

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    # parser.add_argument("--restore-model", type=str,
    #                     help="The path to the hq_decoder training checkpoint for evaluation")
    # # guojy eval
    # parser.add_argument('--eval', default=True)
    # # parser.add_argument('--visualize', default=True)
    # parser.add_argument('--visualize', default=False)
    parser.add_argument("--restore-model", type=str, 
                        default='work_dirs/hq_sam_h_memproisbi_2d_v4/epoch_10.pth',
                        help="The path to the hq_decoder training checkpoint for evaluation")

    return parser.parse_args()


def training_device(args):
    """Resolve the training device; supports --device mps/cuda/cpu."""
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    if device.type == 'mps' and not torch.backends.mps.is_available():
        device = torch.device('cpu')
    return device


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

            # Generate a shared random index sequence.
            random_indices = random.sample(range(len(datasets_dict_saved['part'])), self.sample_select[datase_name])

            # Reorder both lists using the same random indices.
            datasets_dict_saved['part'] = [datasets_dict_saved['part'][i] for i in random_indices]
            datasets_dict_saved['layer'] = [datasets_dict_saved['layer'][i] for i in random_indices]
            datasets_dict_saved['label'] = [datasets_dict_saved['label'][i] for i in random_indices]

            out_list.append(datasets_dict_saved)

        return out_list

class ValListContainer:
    def __init__(self, input_list):
        self.list = input_list
        self.sample_select = {'cremiA': 180,
                              'cremiC': 120,
                              'FIB25': 200,
                              'HB-fb-inner': 100,
                              'SNEMI': 180,
                              'AxonEM-H/950-0-0': 15,
                              'AxonEM-H/950-0-3584': 15,
                              'AxonEM-H/950-3584-0': 15,
                              'AxonEM-H/950-3584-3584': 15,
                              'AxonEM-M/700-0-3584': 15,
                              'AxonEM-M/700-3584-0': 15,
                              'AxonEM-M/700-3584-3584': 15,
                              'AxonEM-M/350-1792-1792': 15,}
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

            # Generate a shared random index sequence.
            random_indices = random.sample(range(len(datasets_dict_saved['part'])), self.sample_select[datase_name])

            # Reorder both lists using the same random indices.
            datasets_dict_saved['part'] = [datasets_dict_saved['part'][i] for i in random_indices]
            datasets_dict_saved['layer'] = [datasets_dict_saved['layer'][i] for i in random_indices]
            datasets_dict_saved['label'] = [datasets_dict_saved['label'][i] for i in random_indices]

            out_list.append(datasets_dict_saved)

        return out_list

def main(net, train_datasets, valid_datasets, args):

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
        train_list_generator = ListContainer(train_im_gt_list)


    print("--- create valid dataloader ---")
    valid_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid")
    if not args.eval:
        valid_list_generator = ValListContainer(valid_im_gt_list)
    valid_dataloaders, valid_datasets = create_dataloaders(valid_im_gt_list,
                                                          my_transforms = [],
                                                          batch_size=args.batch_size_valid,
                                                          training=False)
    print(len(valid_dataloaders), " valid dataloaders created")

    # len(valid_dataloaders) = valid datasets_name
    # len(train_dataloaders)* args.batch_size_valid ~~ len(train_im_gt_list)
    ### --- Step 2: device / DistributedDataParallel---
    net.to(device=training_device(args))
    if torch.cuda.is_available():
        try:
            net = torch.nn.parallel.DistributedDataParallel(
                net, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        except Exception:
            net = nn.DataParallel(net, device_ids=[args.gpu])
    net_without_ddp = net.module if hasattr(net, 'module') else net

 
    ### --- Step 3: Train or Evaluate ---
    if not args.eval:
        print("--- define optimizer ---")
        optimizer = optim.Adam(net_without_ddp.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop_epoch)
        lr_scheduler.last_epoch = args.start_epoch

        train(args, net, optimizer, train_list_generator, valid_list_generator, lr_scheduler)
    else:
        sam = sam_model_registry_fast[args.model_type](checkpoint=args.checkpoint)
        _ = sam.to(device=training_device(args))
        if torch.cuda.is_available():
            try:
                sam = torch.nn.parallel.DistributedDataParallel(
                    sam, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
            except Exception:
                sam = nn.DataParallel(sam, device_ids=[args.gpu])
        if args.restore_model:
            print("restore model from:", args.restore_model)
            net_without_ddp.load_state_dict(torch.load(args.restore_model, map_location='cpu'))
    
        evaluate(args, net, sam, valid_dataloaders, args.visualize)


def train(args, net, optimizer, train_list_generator, valid_list_generator, lr_scheduler):
    if misc.is_main_process():
        os.makedirs(args.output, exist_ok=True)

    epoch_start = args.start_epoch
    epoch_num = args.max_epoch_num

    net.train()
    _ = net.to(device=args.device)
    
    sam = sam_model_registry_fast[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=training_device(args))
    if torch.cuda.is_available():
        try:
            sam = torch.nn.parallel.DistributedDataParallel(
                sam, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        except Exception:
            sam = nn.DataParallel(sam, device_ids=[args.gpu])

    for epoch in range(epoch_start,epoch_num): 
        print("epoch:   ",epoch, "  learning rate:  ", optimizer.param_groups[0]["lr"])
        train_im_gt_list_epoch = train_list_generator.get_list()
        train_dataloaders, train_datasets = create_dataloaders(train_im_gt_list_epoch,
                                                my_transforms = [],
                                                batch_size = args.batch_size_train,
                                                training = True)
        print(len(train_dataloaders), " train dataloaders created")
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
            # io.imsave('/path/to/debug-output/gt_c.tif', data['gt_c'].cpu().numpy()[sample_id].astype(np.uint8))
            # io.imsave('/path/to/debug-output/mem_c.tif', data['mem_c'].cpu().numpy()[sample_id].astype(np.uint8))
            # io.imsave('/path/to/debug-output/embed_c.tif', data['embed_c'].cpu().numpy()[sample_id])
            # io.imsave('/path/to/debug-output/embed0_c.tif', data['embed0_c'].cpu().numpy()[sample_id])
            # #####

            device = training_device(args)
            gt_c = gt_c.to(device)
            mem_c = mem_c.to(device)
            embed_c = embed_c.to(device)
            embed0_c = embed0_c.to(device)
            
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

            with torch.no_grad():
                batched_output, interm_embeddings = sam(batched_input, multimask_output=False)
            
            batch_len = len(batched_output)
            encoder_embedding = torch.cat([batched_output[i_l]['encoder_embedding'] for i_l in range(batch_len)], dim=0)
            image_pe = [batched_output[i_l]['image_pe'] for i_l in range(batch_len)]
            sparse_embeddings = [batched_output[i_l]['sparse_embeddings'] for i_l in range(batch_len)]
            dense_embeddings = [batched_output[i_l]['dense_embeddings'] for i_l in range(batch_len)]

            masks_hq, mask_mem = net(
                image_embeddings=encoder_embedding,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                hq_token_only=True,
                interm_embeddings=interm_embeddings,
            )
            ##### zhaih
            loss_mask, loss_dice = loss_masks(masks_hq, gt_c/255.0, len(masks_hq))
            loss_mask_mem, loss_dice_mem = loss_masks(mask_mem, mem_c/255.0, len(masks_hq))
            loss = (loss_mask + loss_dice) + (loss_mask_mem + loss_dice_mem)
            
            loss_dict = {"loss_mask": loss_mask, "loss_dice":loss_dice, 
                         "loss_mask_mem": loss_mask_mem, "loss_dice_mem":loss_dice_mem}

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
        valid_im_gt_list_epoch = valid_list_generator.get_list()
        valid_dataloaders, valid_datasets = create_dataloaders(valid_im_gt_list_epoch,
                                                my_transforms = [],
                                                batch_size = args.batch_size_valid,
                                                training = False)
        test_stats = evaluate(args, net, sam, valid_dataloaders)
        train_stats.update(test_stats)
        
        net.train()  

        if epoch % args.model_save_fre == 0:
            model_name = "/epoch_"+str(epoch)+".pth"
            print('come here save at', args.output + model_name)
            state_dict = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
            misc.save_on_master(state_dict, args.output + model_name)
    
    # Finish training
    print("Training Reaches The Maximum Epoch Number")
    
    # merge sam and hq_decoder
    if 'model_name' not in locals() or not os.path.isfile(args.output + model_name):
        print("No checkpoint saved; skip SAM/HQ-SAM weight merge.")
        return
    if misc.is_main_process():
        sam_ckpt = torch.load(args.checkpoint)
        hq_decoder = torch.load(args.output + model_name)
        for key in hq_decoder.keys():
            sam_key = 'mask_decoder.'+key
            if sam_key not in sam_ckpt.keys():
                sam_ckpt[sam_key] = hq_decoder[key]
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

def evaluate(args, net, sam, valid_dataloaders, visualize=False):
    net.eval()
    print("Validating...")
    test_stats = {}

    for k in range(len(valid_dataloaders)):
        metric_logger = misc.MetricLogger(delimiter="  ")
        valid_dataloader = valid_dataloaders[k]
        print('valid_dataloader len:', len(valid_dataloader))

        for data_val in metric_logger.log_every(valid_dataloader,1000):
            gt_c_val, mem_c_val, embed_c_val, embed0_c_val = data_val['gt_c'], data_val['mem_c'], data_val['embed_c'], data_val['embed0_c']

            device = training_device(args)
            gt_c_val = gt_c_val.to(device)
            mem_c_val = mem_c_val.to(device)
            embed_c_val = embed_c_val.to(device)
            embed0_c_val = embed0_c_val.to(device)

            
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

            with torch.no_grad():
                batched_output, interm_embeddings = sam(batched_input, multimask_output=False)
            
            batch_len = len(batched_output)
            encoder_embedding = torch.cat([batched_output[i_l]['encoder_embedding'] for i_l in range(batch_len)], dim=0)
            image_pe = [batched_output[i_l]['image_pe'] for i_l in range(batch_len)]
            sparse_embeddings = [batched_output[i_l]['sparse_embeddings'] for i_l in range(batch_len)]
            dense_embeddings = [batched_output[i_l]['dense_embeddings'] for i_l in range(batch_len)]
            ##### zhaih
            masks_sam, masks_hq, masks_mem = net(
                image_embeddings=encoder_embedding,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                hq_token_only=False,
                interm_embeddings=interm_embeddings,
            )

            iou = compute_iou(masks_hq, gt_c_val.to(torch.uint8))
            boundary_iou = compute_boundary_iou(masks_hq, gt_c_val.to(torch.uint8))

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

    args = get_args_parser()
    valid_record = args.record_valid or args.record_train
    record_train = {"name": "record_train", "path": args.record_train}
    record_valid = {"name": "record_valid", "path": valid_record}

    train_datasets = [record_train,]
    valid_datasets = [record_valid,]
    net = MaskDecoderHQ(args.model_type) 

    main(net, train_datasets, valid_datasets, args)
