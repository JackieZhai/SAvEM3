import os
import cv2
import json

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



if __name__== '__main__':
    # {'dataset_name':{layer1:[label1, label2, ...], layer2:[label6, label8, ...],...}...}
    train_dataset_name_list = ['cremiA', 'cremiC', 'FIB25', 'HB-fb-inner', 
                          'SNEMI', 
                          'AxonEM-H/seg_950-0-0', 'AxonEM-H/seg_950-0-3584', 
                          'AxonEM-H/seg_950-3584-0', 'AxonEM-H/seg_950-3584-3584', 
                          'AxonEM-M/seg_700-0-3584', 'AxonEM-M/seg_700-3584-0', 
                          'AxonEM-M/seg_700-3584-3584', 'AxonEM-M/seg_350-1792-1792', ]
    test_dataset_name_list = ['AC3', 'cremiB', 'HB-pb', 'AxonEM-H/seg_0-0-0', 
                         'AxonEM-H/seg_0-0-3584', 'AxonEM-H/seg_0-3584-0', 
                         'AxonEM-H/seg_0-3584-3584', 'AxonEM-M/seg_0-0-3584', 
                         'AxonEM-M/seg_0-3584-0', 'AxonEM-M/seg_0-3584-3584', 
                         'AxonEM-M/seg_700-0-0', ]
    label_root = os.environ.get('SAVEM3_MEM_ROOT',
    os.path.join(os.environ.get('SAVEM3_DATA_ROOT', '.'), 'prepared_segments_mul_mem/'))
    label_root_2d = os.environ.get('SAVEM3_LABEL_ROOT',
    os.path.join(os.environ.get('SAVEM3_DATA_ROOT', '.'), 'prepared_segments_mul_2d/'))

    json_data_train = {}
    json_data_test = {}

    for dataset_name in train_dataset_name_list:
        label_dict = {}
        dataset_name_path = os.path.join(label_root, dataset_name)
        dataset_save_path = os.path.join(label_root_2d, dataset_name)
        os.makedirs(dataset_save_path, exist_ok=True)
        print(dataset_save_path, 'created')
        layer_name_list = os.listdir(dataset_name_path)
        for layer_name in tqdm(layer_name_list):
            mem_img = io.imread(os.path.join(dataset_name_path, layer_name))
            layer_id = int(layer_name.split('.')[0])
            unique_label, unique_label_id_list = mem2label(mem_img=mem_img)
            label_dict[layer_id] = unique_label_id_list
            io.imsave(os.path.join(dataset_save_path, layer_name), unique_label)

        json_data_train[dataset_name] = label_dict

    for dataset_name in test_dataset_name_list:
        label_dict = {}
        dataset_name_path = os.path.join(label_root, dataset_name)
        dataset_save_path = os.path.join(label_root_2d, dataset_name)
        os.makedirs(dataset_save_path, exist_ok=True)
        print(dataset_save_path, 'created')
        layer_name_list = os.listdir(dataset_name_path)
        for layer_name in tqdm(layer_name_list):
            mem_img = io.imread(os.path.join(dataset_name_path, layer_name))
            layer_id = int(layer_name.split('.')[0])
            unique_label, unique_label_id_list = mem2label(mem_img=mem_img)
            label_dict[layer_id] = unique_label_id_list
            io.imsave(os.path.join(dataset_save_path, layer_name), unique_label)

        json_data_test[dataset_name] = label_dict

    json_train_save = './record2d_train.json'
    json_test_save = './record2d_test.json'


    # 将字典保存为JSON文件，指定缩进为4个空格
    with open(json_train_save, 'w') as json_file:
        json.dump(json_data_train, json_file, indent=4)

    # 将字典保存为JSON文件，指定缩进为4个空格
    with open(json_test_save, 'w') as json_file:
        json.dump(json_data_test, json_file, indent=4)

    print('ok')

