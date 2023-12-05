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

import os
import argparse
import numpy as np
import cv2
import mmcv
import torch
import pickle as pk
import multiprocessing
from tqdm import tqdm

flags = None

def resize(img, size):
    old_h = img.shape[0]
    old_w = img.shape[1]
    scale_ratio = min(size[0] / old_w, size[1] / old_h)
    new_w = int(np.floor(old_w * scale_ratio))
    new_h = int(np.floor(old_h * scale_ratio))
    resized_img = mmcv.imresize(img, (new_w, new_h))
    return resized_img, scale_ratio


def gen_input_bin(file_batches, batch):
    for file in file_batches[batch]:

        image = mmcv.imread(os.path.join(flags.image_src_path, file))
        ori_shape = image.shape
        image, scale_factor = resize(image, (flags.model_input_width, flags.model_input_height))
        img_shape = image.shape
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        image = mmcv.imnormalize(image, mean, std)
        h = image.shape[0]
        w = image.shape[1]
        pad_left = (flags.model_input_width - w) // 2
        pad_top = (flags.model_input_height - h) // 2
        pad_right = flags.model_input_width - pad_left - w
        pad_bottom = flags.model_input_height - pad_top - h
        image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        image = image.transpose(2, 0, 1)
        image.tofile(os.path.join(flags.bin_file_path, file.split('.')[0] + ".bin"))
        image_meta = {'img_shape': img_shape, 'scale_factor': scale_factor, 'ori_shape': ori_shape}
        with open(os.path.join(flags.meta_file_path, file.split('.')[0] + ".pk"), "wb") as fp:
            pk.dump(image_meta, fp)


def preprocess():
    files = os.listdir(flags.image_src_path)
    file_batches = [files[i:i + 100] for i in range(0, 5000, 100) if files[i:i + 100] != []]
    thread_pool = multiprocessing.Pool(len(file_batches))
    for batch in tqdm(range(len(file_batches))):
        thread_pool.apply_async(gen_input_bin, args=(file_batches, batch))
    thread_pool.close()
    thread_pool.join()
    print("in thread, except will not report! please ensure bin files generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess of SOLOV2 model')
    parser.add_argument("--image_src_path", default="/root/datasets/coco/val2017", help='image of dataset')
    parser.add_argument("--bin_file_path", default="val2017_bin", help='Preprocessed image buffer')
    parser.add_argument("--meta_file_path", default="val2017_bin_meta", help='Get image meta')
    parser.add_argument("--model_input_height", default=800, type=int, help='input tensor height')
    parser.add_argument("--model_input_width", default=1216, type=int, help='input tensor width')
    flags = parser.parse_args()
    if not os.path.exists(flags.bin_file_path):
        os.makedirs(flags.bin_file_path)
    if not os.path.exists(flags.meta_file_path):
        os.makedirs(flags.meta_file_path)
    preprocess()
