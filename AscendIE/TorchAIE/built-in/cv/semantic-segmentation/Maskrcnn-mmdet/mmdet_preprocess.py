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
import multiprocessing
import tqdm
from glob import glob


def resize(img, size):
    old_h = img.shape[0]
    old_w = img.shape[1]
    scale_ratio = min(size[0] / old_w, size[1] / old_h)
    new_w = int(np.floor(old_w * scale_ratio))
    new_h = int(np.floor(old_h * scale_ratio))
    resized_img = mmcv.imresize(img, (new_w, new_h), backend='cv2')
    return resized_img


def gen_input_bin(args, file_batches, batch):
    i = 0
    for im_file in tqdm.tqdm(file_batches[batch]):
        i = i + 1

        image = mmcv.imread(os.path.join(args.image_src_path, im_file), backend='cv2')
        image = resize(image, (args.model_input_width, args.model_input_height))
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        image = mmcv.imnormalize(image, mean, std)
        h = image.shape[0]
        w = image.shape[1]
        pad_left = (args.model_input_width - w) // 2
        pad_top = (args.model_input_height - h) // 2
        pad_right = args.model_input_width - pad_left - w
        pad_bottom = args.model_input_height - pad_top - h
        image = mmcv.impad(image, padding=(pad_left, pad_top, pad_right, pad_bottom), pad_val=0)
        image = image.transpose(2, 0, 1)
        image.tofile(os.path.join(args.bin_file_path, im_file.split('.')[0] + ".bin"))


def preprocess(args):
    files = os.listdir(args.image_src_path)
    file_batches = [files[i:i + 100] for i in range(0, 5000, 100) if files[i:i + 100] != []]
    thread_pool = multiprocessing.Pool(len(file_batches))
    for batch in range(len(file_batches)):
        thread_pool.apply_async(gen_input_bin, args=(args, file_batches, batch))
    thread_pool.close()
    thread_pool.join()
    print("in thread, except will not report! please ensure bin files generated.")


def get_jpg_info(file_path, info_name):
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    image_names = []
    for extension in extensions:
        image_names.append(glob(os.path.join(file_path, '*.' + extension)))  
    with open(info_name, 'w') as file:
        for image_name in image_names:
            if len(image_name) == 0:
                continue
            else:
                for index, img in tqdm.tqdm(enumerate(image_name), total = len(image_name), desc="get_jpg_info"):
                    img_cv = cv2.imread(img)
                    shape = img_cv.shape
                    width, height = shape[1], shape[0]
                    content = ' '.join([str(index), img, str(width), str(height)])
                    file.write(content)
                    file.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess of MaskRCNN PyTorch model')
    parser.add_argument("--image_src_path", default="./coco/val2017", help='image of dataset')
    parser.add_argument("--bin_file_path", default="./val2017_bin/", help='Preprocessed image buffer')
    parser.add_argument("--info_file_path", default="./val2017.info", help='basic image infomation')
    parser.add_argument("--model_input_height", default=1216, type=int, help='input tensor height')
    parser.add_argument("--model_input_width", default=1216, type=int, help='input tensor width')
    args = parser.parse_args()    
    if not os.path.exists(args.bin_file_path):
        os.makedirs(args.bin_file_path)
    preprocess(args)
    get_jpg_info(args.image_src_path, args.info_file_path)
