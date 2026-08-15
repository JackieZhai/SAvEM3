import os
import cv2
import json
import copy
import random
from skimage import io
import numpy as np

from tqdm import tqdm
from skimage import io, measure, morphology, transform, data
from scipy.ndimage.morphology import binary_erosion,binary_dilation
import warnings
# 忽略 "is a low contrast image" 警告
warnings.filterwarnings('ignore', '.*low contrast image.*')

def mem2label(mem_img):
    binary_img = mem_img == 0
    thre_min = int((binary_img.shape[0]*binary_img.shape[1]*0.0001))
    unique_img_label = measure.label(binary_img, connectivity=2)
    unique_img_label_regions = measure.regionprops(unique_img_label)
    save_img = np.zeros(shape=binary_img.shape, dtype=np.uint16)
    id_list = []
    region_index = 0
    for unique_img_label_region in unique_img_label_regions:
        region_index = region_index + 1
        if unique_img_label_region.area > thre_min:

            region_mask = np.where(unique_img_label == region_index)
            save_img[region_mask] = region_index
            id_list.append(region_index)
    return save_img, id_list

def label2mem(label):
    assert len(label.shape) == 2
    label_temp = label.copy()
    label_temp = label_temp.astype(np.int64)
    label_pad = np.pad(label_temp, pad_width=1, mode='edge')
    mem_img = (label_temp != label_pad[2:, 1:-1]) + (label_temp != label_pad[:-2, 1:-1]) + \
              (label_temp != label_pad[1:-1, 2:]) + (label_temp != label_pad[1:-1, :-2]) + \
              (label_temp != label_pad[2:, 2:]) + (label_temp != label_pad[:-2, :-2]) + \
              (label_temp != label_pad[2:, :-2]) + (label_temp != label_pad[:-2, 2:])
    return mem_img.astype(np.uint8) * 255


if __name__== '__main__':
    # {'dataset_name':{layer1:[label1, label2, ...], layer2:[label6, label8, ...],...}...}
    # 生成mem，重新映射label
    source_root = '/home/zhaih/share/SAM/sam-hq-main/train/prepared_segments/'
    label_save_root = '/home/zhaih/share/SAM/sam-hq-main/train/prepared_segments_mul_2d/'
    mem_save_root = '/home/zhaih/share/SAM/sam-hq-main/train/prepared_segments_mul_mem/'


    datasets_list = ['BBBC039v1_labels', 'cell_part_elong_labels', 'cell_part_weak_labels']
    # datasets_list = ['cell_part_weak_labels']


    for datasets_name in datasets_list:

        json_data_train = {}
        json_data_val = {}

        datasets_root = os.path.join(source_root, datasets_name)
        file_name_list = os.listdir(datasets_root)
        if datasets_name == 'BBBC039v1_labels':
            file_name_list.sort()
            assert len(file_name_list) == 200
            file_name_list_train = copy.deepcopy(file_name_list[:100]) 
            file_name_list_val = copy.deepcopy(file_name_list[100:]) 
            save_folder_name = 'BBBC039v1'
        elif datasets_name == 'cell_part_elong_labels':
            file_name_list_train = ['cell_00143_label.tiff', 'cell_00144_label.tiff', 
                                    'cell_00529_label.tiff', 'cell_00548_label.tiff', ]
            file_name_list_val = ['cell_00142_label.tiff', 'cell_00528_label.tiff', 'cell_00547_label.tiff', ]
            save_folder_name = 'cell_part_elong'
        elif datasets_name == 'cell_part_weak_labels':
            file_name_list_train = copy.deepcopy(file_name_list) 
            file_name_list_train.sort(key=lambda x:int(x[5:10]))
            assert len(file_name_list) == 282
            file_name_list_train = copy.deepcopy(file_name_list[:141]) 
            file_name_list_val = copy.deepcopy(file_name_list[141:]) 
            save_folder_name = 'cell_part_weak'
        else:
            raise ValueError
        print(save_folder_name)
        dataset_mem_save_path = os.path.join(mem_save_root, save_folder_name)
        dataset_label_save_path = os.path.join(label_save_root, save_folder_name)
        os.makedirs(dataset_mem_save_path, exist_ok=True)
        os.makedirs(dataset_label_save_path, exist_ok=True)


        label_dict = {}
        for file_name in tqdm(file_name_list_train):
            file_name_clean = file_name.split('.')[0]
            if file_name_clean[-6:] == '_label':
                file_name_clean = file_name_clean[:-6]
            file_name_abs = os.path.join(datasets_root, file_name)
            image = io.imread(file_name_abs)
            # assert image.max() <= 255
            # image = image.astype(np.uint8)
            # image_resize = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_NEAREST)
            image_resize = transform.resize(image, (1024, 1024), order=0, mode='constant', anti_aliasing=False)
            mem_img = label2mem(image_resize)
            label_dict[file_name_clean] = np.unique(image_resize[image_resize != 0]).tolist()
            io.imsave(os.path.join(dataset_mem_save_path, file_name_clean + '.tif'), mem_img)
            io.imsave(os.path.join(dataset_label_save_path, file_name_clean + '.tif'), image_resize)

        json_data_train[save_folder_name] = label_dict

        label_dict = {}
        for file_name in tqdm(file_name_list_val):
            file_name_clean = file_name.split('.')[0]
            if file_name_clean[-6:] == '_label':
                file_name_clean = file_name_clean[:-6]
            file_name_abs = os.path.join(datasets_root, file_name)
            image = io.imread(file_name_abs)
            # assert image.max() <= 255
            # image = image.astype(np.uint8)
            # image_resize = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_NEAREST)
            image_resize = transform.resize(image, (1024, 1024), order=0, mode='constant', anti_aliasing=False)
            mem_img = label2mem(image_resize)
            label_dict[file_name_clean] = np.unique(image_resize[image_resize != 0]).tolist()
            io.imsave(os.path.join(dataset_mem_save_path, file_name_clean + '.tif'), mem_img)
            io.imsave(os.path.join(dataset_label_save_path, file_name_clean + '.tif'), image_resize)

        json_data_val[save_folder_name] = label_dict

        json_train_save = "/home/guojy2/share/guojy/ProjectCodes/BigModel/sam-hq/"+save_folder_name+"_train.json"
        json_val_save = "/home/guojy2/share/guojy/ProjectCodes/BigModel/sam-hq/"+save_folder_name+"_val.json"


        # 将字典保存为JSON文件，指定缩进为4个空格
        with open(json_train_save, 'w') as json_file:
            json.dump(json_data_train, json_file, indent=4)

        # 将字典保存为JSON文件，指定缩进为4个空格
        with open(json_val_save, 'w') as json_file:
            json.dump(json_data_val, json_file, indent=4)

        print(save_folder_name, 'ok')

    # train_dataset_name_list = ['cremiA', 'cremiC', 'FIB25', 'HB-fb-inner', 
    #                       'SNEMI', 
    #                       'AxonEM-H/seg_950-0-0', 'AxonEM-H/seg_950-0-3584', 
    #                       'AxonEM-H/seg_950-3584-0', 'AxonEM-H/seg_950-3584-3584', 
    #                       'AxonEM-M/seg_700-0-3584', 'AxonEM-M/seg_700-3584-0', 
    #                       'AxonEM-M/seg_700-3584-3584', 'AxonEM-M/seg_350-1792-1792', ]
    # test_dataset_name_list = ['AC3', 'cremiB', 'HB-pb', 'AxonEM-H/seg_0-0-0', 
    #                      'AxonEM-H/seg_0-0-3584', 'AxonEM-H/seg_0-3584-0', 
    #                      'AxonEM-H/seg_0-3584-3584', 'AxonEM-M/seg_0-0-3584', 
    #                      'AxonEM-M/seg_0-3584-0', 'AxonEM-M/seg_0-3584-3584', 
    #                      'AxonEM-M/seg_700-0-0', ]
    # label_root = os.environ.get('SAVEM3_MEM_ROOT',
    #     os.path.join(os.environ.get('SAVEM3_DATA_ROOT', '.'), 'prepared_segments_mul_mem/'))
    # label_root_2d = os.environ.get('SAVEM3_LABEL_ROOT',
    #     os.path.join(os.environ.get('SAVEM3_DATA_ROOT', '.'), 'prepared_segments_mul_2d/'))

    # json_data_train = {}
    # json_data_test = {}

    # for dataset_name in train_dataset_name_list:
    #     label_dict = {}
    #     dataset_name_path = os.path.join(label_root, dataset_name)
    #     dataset_save_path = os.path.join(label_root_2d, dataset_name)
    #     os.makedirs(dataset_save_path, exist_ok=True)
    #     print(dataset_save_path, 'created')
    #     layer_name_list = os.listdir(dataset_name_path)
    #     for layer_name in tqdm(layer_name_list):
    #         mem_img = io.imread(os.path.join(dataset_name_path, layer_name))
    #         layer_id = int(layer_name.split('.')[0])
    #         unique_label, unique_label_id_list = mem2label(mem_img=mem_img)
    #         label_dict[layer_id] = unique_label_id_list
    #         io.imsave(os.path.join(dataset_save_path, layer_name), unique_label)

    #     json_data_train[dataset_name] = label_dict

    # for dataset_name in test_dataset_name_list:
    #     label_dict = {}
    #     dataset_name_path = os.path.join(label_root, dataset_name)
    #     dataset_save_path = os.path.join(label_root_2d, dataset_name)
    #     os.makedirs(dataset_save_path, exist_ok=True)
    #     print(dataset_save_path, 'created')
    #     layer_name_list = os.listdir(dataset_name_path)
    #     for layer_name in tqdm(layer_name_list):
    #         mem_img = io.imread(os.path.join(dataset_name_path, layer_name))
    #         layer_id = int(layer_name.split('.')[0])
    #         unique_label, unique_label_id_list = mem2label(mem_img=mem_img)
    #         label_dict[layer_id] = unique_label_id_list
    #         io.imsave(os.path.join(dataset_save_path, layer_name), unique_label)

    #     json_data_test[dataset_name] = label_dict

    # json_train_save = "/home/guojy2/share/guojy/ProjectCodes/BigModel/sam-hq/record2d_lm_train.json"
    # json_test_save = "/home/guojy2/share/guojy/ProjectCodes/BigModel/sam-hq/record2d_lm_test.json"


    # # 将字典保存为JSON文件，指定缩进为4个空格
    # with open(json_train_save, 'w') as json_file:
    #     json.dump(json_data_train, json_file, indent=4)

    # # 将字典保存为JSON文件，指定缩进为4个空格
    # with open(json_test_save, 'w') as json_file:
    #     json.dump(json_data_test, json_file, indent=4)

    print('ok')

