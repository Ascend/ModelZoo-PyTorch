# Copyright 2021 Huawei Technologies Co., Ltd
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

import torch
import torch.utils.data as data

from torchvision import transforms
from cityscapes import CitySegmentation
from score import SegmentationMetric
from distributed import *

def get_res(res_dir):

    output = []
    with open(res_dir) as res_f:
        for line in res_f:
            num_list = line.split()
            for num in num_list:
                output.append(float(num))
        output = torch.from_numpy(np.array(output).reshape((1, 19, 480, 480)))
    '''
    with open(res_dir, 'rb') as res_f:
        output = np.frombuffer(res_f.read(), np.float16)
        output = torch.from_numpy(output.reshape((1, 19, 480, 480)))
    '''
    return output


def postprocess(args):
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
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
        res_name = os.path.splitext(os.path.basename(filename[0]))[0]
        res_dir = os.path.join(args.result_dir, res_name + '_1.txt')
        #res_dir = os.path.join(args.result_dir, res_name + '_1.bin')
        res = get_res(res_dir)
        metric.update(res, target)
        pixAcc, mIoU = metric.get()
        print("Sample: {:d}, validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
            i + 1, pixAcc * 100, mIoU * 100))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-path', type=str, required=True)
    parser.add_argument('--result-dir', type=str, default='result/dumpOutput_device0')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--workers', '-j', type=int, default=4,
                        metavar='N', help='dataloader threads')
    args = parser.parse_args()

    postprocess(args)

# python ENet_postprocess.py --src-path=/root/.torch/datasets/citys --result-dir result/dumpOutput_device0