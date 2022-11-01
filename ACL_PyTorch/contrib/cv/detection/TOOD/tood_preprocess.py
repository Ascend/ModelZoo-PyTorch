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
import os
import argparse
import numpy as np
import cv2
import mmcv
import torch
import multiprocessing
from tqdm import tqdm

def gen_input_bin(file_batches, batch, height, width):
    for file in file_batches[batch]:
        image = mmcv.imread(os.path.join(flags.image_src_path, file), backend='cv2')
        image = mmcv.imrescale(image, (1216, 1216), backend='cv2')
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        image = mmcv.imnormalize(image, mean, std)

        h, w = image.shape[0], image.shape[1]
        pad_left = (width - w) // 2
        pad_top = (height - h) // 2
        pad_right = width - pad_left - w
        pad_bottom = height - pad_top - h
        image = mmcv.impad(image, padding=(pad_left, pad_top, pad_right, pad_bottom), pad_val=0)

        h, w = image.shape[0], image.shape[1]
        image = image.transpose(2, 0, 1)
        image.tofile(os.path.join(flags.bin_file_path, file.split('.')[0] + ".bin"))

def preprocess(src_path, save_path, height, width):
    files = os.listdir(src_path)
    file_batches = [files[i:i + 100] for i in range(0, 5000, 100) if files[i:i + 100] != []]
    thread_pool = multiprocessing.Pool(len(file_batches))
    pbar = tqdm(range(len(file_batches)))
    for batch in range(len(file_batches)):
        thread_pool.apply_async(gen_input_bin,
                                args=(file_batches, batch, height, width),
                                callback=lambda _: pbar.update(1),
                                error_callback=lambda _: pbar.update(1))
    thread_pool.close()
    thread_pool.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess of MaskRCNN PyTorch model')
    parser.add_argument("--image_src_path", default="./data/coco/val2017/", help='image of dataset')
    parser.add_argument("--bin_file_path", default="./data/coco/val2017_bin_1216_1216")
    parser.add_argument("--height", type=int, default=1216)
    parser.add_argument("--width", type=int, default=1216)
    flags = parser.parse_args()    
    os.mkdir(flags.bin_file_path)
    preprocess(flags.image_src_path, flags.bin_file_path, flags.height, flags.width)