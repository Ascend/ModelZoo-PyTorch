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

import cv2
import os
import argparse
import torch
from tqdm import tqdm
import yaml
import numpy as np
import sys
sys.path.append('./yolor')
from utils.datasets import create_dataloader
from utils.general import check_file, check_dataset


def preprocess(data, save_path, imgsz, batch_size):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    check_dataset(data)  # check
    path = data['val']
    dataloader = create_dataloader(path, imgsz, batch_size, 64, opt, pad=0.5, rect=False)[0]
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader)):
        # print(paths)
        file_name = paths[0].split('/')[-1]
        # print(file_name)
        img = img.float()
        img /= 255.0
        img = img.squeeze(0)
        img_np = img.numpy()
        img_np.tofile(os.path.join(save_path, file_name.split('.')[0] + ".bin"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--batch_size', type=int, default=1, help='size of each image batch')
    parser.add_argument('--img_size', type=int, default=1280, help='inference size (pixels)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--task', default='val', help="'val', 'test', 'study'")
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.cfg path')
    parser.add_argument('--save_path', type=str, required=True)
    opt = parser.parse_args()
    opt.data = check_file(opt.data)  # check file
    print(opt)

    preprocess(opt.data, opt.save_path, opt.img_size, opt.batch_size)
