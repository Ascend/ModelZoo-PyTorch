# Copyright 2021-2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-
import sys
import os

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import numpy as np
import argparse
import json

import torch
import torch.utils.data as data

from torchvision import transforms
from cityscapes import CitySegmentation
from score import SegmentationMetric
from distributed import *

def read_info_from_json(json_path):
    '''
    此函数用于读取inference_tools生成的json文件
    input: json文件地址
    output: dict结构: 为原始的json转换出来的结构
    '''
    if os.path.exists(json_path) is False:
        print(json_path, 'is not exist')
    with open(json_path, 'r') as f:
        load_data = json.load(f)
        file_info = load_data['filesinfo']
        return file_info

def get_mask_name_maping(json_info):

    mask_name_mapping = {}

    for i in json_info.items():
        res_path = i[1]['outfiles'][0]
        # 获取对应的标签
        label_name = os.path.splitext(os.path.basename(i[1]['infiles'][0]))[0]

        mask_name_mapping[label_name] = res_path
    
    return mask_name_mapping

def postprocess(args):
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])

    result_dir_path = os.path.dirname(args.result_dir)
    file_info = read_info_from_json(args.result_dir)

    mask_name_mapping = get_mask_name_maping(file_info)

    # dataset and dataloader
    data_kwargs = {'transform': input_transform, 'base_size': 520, 'crop_size': 480}
    val_dataset = CitySegmentation(root = args.src_path, split='val', mode='val', **data_kwargs)

    val_sampler = make_data_sampler(val_dataset, False, False)
    val_batch_sampler = make_batch_data_sampler(val_sampler, args.batch_size)

    val_loader = data.DataLoader(dataset=val_dataset,
                                      batch_sampler=val_batch_sampler,
                                      num_workers=args.workers,
                                      pin_memory=True)

    metric = SegmentationMetric(19)
    for i, (image, target, filename) in enumerate(val_loader):
        mask_name = os.path.splitext(os.path.basename(filename[0]))[0]
        
        if mask_name in mask_name_mapping:
            # 获取预测文件名
            result_file_name = os.path.basename(mask_name_mapping[mask_name])
            # 使用result_dir的路径作为结果文件的路径，可以使得运行该脚本的路径更通用
            res_path = os.path.join(result_dir_path, result_file_name)

        else:
            print("{} does not exist in eval_dir".format(res_name))
            continue

        res = np.fromfile(res_path, np.float32)
        res = np.reshape(res, (1, 19, data_kwargs['crop_size'], data_kwargs['crop_size']))
        res = torch.from_numpy(res)

        metric.update(res, target)
        pixAcc, mIoU = metric.get()
        print("Sample: {:d}, validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
            i + 1, pixAcc * 100, mIoU * 100))
    print('############################################################')
    pixAcc, mIoU = metric.get()
    print("\tvalidation pixAcc: {:.3f}, mIoU: {:.3f}".format(pixAcc * 100, mIoU * 100))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, required=True)
    parser.add_argument('--result_dir', type=str, default='result/dumpOutput_device0')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--workers', '-j', type=int, default=4,
                        metavar='N', help='dataloader threads')
    args = parser.parse_args()

    postprocess(args)

