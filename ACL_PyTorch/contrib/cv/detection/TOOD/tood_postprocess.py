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
import shutil
import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
from mmdet.core import bbox2result
from mmdet.datasets import CocoDataset

def postprecess(args):
    dataset = CocoDataset(ann_file=args.ann_file_path, pipeline=[])
    bin_path = args.bin_file_path
    latest_result = os.listdir(bin_path)
    latest_result.sort()

    results = []
    for data_info in tqdm(dataset.data_infos):
        file_name = data_info['file_name']
        ori_h = data_info['height']
        ori_w = data_info['width']

        bin_h, bin_w = args.height, args.width
        scale_ratio = min(bin_h / ori_h, bin_w / ori_w)
        new_w = int(np.floor(ori_w * scale_ratio))
        new_h = int(np.floor(ori_h * scale_ratio))
        ratio_w, ratio_h = new_w/ori_w, new_h/ori_h
        scale_ratio = np.array([ratio_w, ratio_h, ratio_w, ratio_h])

        pad_w = bin_h - ori_w * ratio_w
        pad_h = bin_w - ori_h * ratio_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2

        path_base = os.path.join(bin_path, file_name.split('.')[0])
        
        bboxes = np.fromfile(path_base + '_0.bin', dtype=np.float32)
        bboxes = np.reshape(bboxes, [100, 5])
        labels = np.fromfile(path_base + '_1.bin', dtype=np.int64)
        bboxes[:, 0] = (bboxes[:, 0] - pad_left) 
        bboxes[:, 1] = (bboxes[:, 1] - pad_top)  
        bboxes[:, 2] = (bboxes[:, 2] - pad_left)
        bboxes[:, 3] = (bboxes[:, 3] - pad_top) 

        bboxes[..., 0:4] = bboxes[..., 0:4] / scale_ratio
        cls_bboxes = bbox2result(bboxes, labels, 80)
        results.append(cls_bboxes)

    dataset.evaluate(results, metric=['bbox'], classwise=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_file_path", default="data/coco/annotations/instances_val2017.json")
    parser.add_argument("--bin_file_path", default="output/2022_10_15-02_19_10")
    parser.add_argument("--height", type=int, default=1216)
    parser.add_argument("--width", type=int, default=1216)
    args = parser.parse_args()
    postprecess(args)