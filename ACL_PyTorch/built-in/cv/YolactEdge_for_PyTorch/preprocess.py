# Copyright 2023 Huawei Technologies Co., Ltd
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

import numpy as np
import json
import os
import argparse
from tqdm import tqdm
from yolact_edge.data import COCODetection
from yolact_edge.utils.augmentations import BaseTransform

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', default='./data/coco/val2017', type=str, 
                        help='The folder path of coco images')
    parser.add_argument('--json_file', default='./data/coco/annotations/instances_val2017.json', type=str, 
                        help='The path of coco json file')
    parser.add_argument('--save_path', default='./inputs', type=str, help='The folder path to save input files')
    args = parser.parse_args()

    image_path = args.image_path
    json_file = args.json_file
    save_path = args.save_path

    bin_path = os.path.join(save_path, 'bin_file')
    json_path = os.path.join(save_path, 'ids.json')
    dataset = COCODetection(image_path, json_file, transform=BaseTransform(), has_gt=True)
    dataset_size = len(dataset)
    ids = {'ids': dataset.ids}
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        os.mkdir(bin_path)
    with open(json_path, 'w') as f:
        json.dump(ids, f)
    for it, image_idx in tqdm(enumerate(range(dataset_size))):
        img, gt, gt_masks, h, w, num_crowd = dataset.pull_item(image_idx)
        batch = img.unsqueeze(0)
        data = batch.numpy()
        file_name = str(image_idx) + '.bin'
        file_path = os.path.join(bin_path, file_name)
        data.tofile(file_path)