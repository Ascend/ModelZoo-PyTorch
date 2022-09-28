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

from __future__ import print_function

import argparse
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retinaface')
    parser.add_argument('--dataset-folder', default='data/widerface/val/images/')
    parser.add_argument('--preinfo-folder', default='widerface/prep_info')
    parser.add_argument('--save-folder', default='widerface/prep')
    args = parser.parse_args()

    testset_folder = args.dataset_folder
    testset_list = args.dataset_folder[:-7] + "wider_val.txt"
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    with open(testset_list, 'r') as fr:
        test_dataset = fr.read().split()
    num_images = len(test_dataset)

    # testing begin
    for img_name in tqdm(test_dataset):
        image_path = testset_folder + img_name
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = np.float32(img_raw)

        # testing scale
        target_size = 1000
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = target_size / im_size_max
        img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_NEAREST)
        width_pad = target_size - img.shape[1]
        left = 0
        right = width_pad
        height_pad = target_size - img.shape[0]
        top = 0
        bottom = height_pad
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img = torch.from_numpy(img).unsqueeze(0).byte()

        # save bin image
        save_path = img_name[:-4].split('/')
        save_info_path = os.path.join(args.preinfo_folder, save_path[0])
        if not os.path.exists(save_info_path):
            os.makedirs(save_info_path)
        info = np.array(resize, dtype=np.float32)
        info.tofile(os.path.join(save_info_path, str(save_path[1]) + '.bin'))
        img.numpy().tofile(os.path.join(args.save_folder, str(save_path[1]) + '.bin'))
