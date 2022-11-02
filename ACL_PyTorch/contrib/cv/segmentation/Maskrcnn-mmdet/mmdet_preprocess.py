# Copyright 2020 Huawei Technologies Co., Ltd
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

def resize(img, size):
    old_h = img.shape[0]
    old_w = img.shape[1]
    scale_ratio = min(size[0] / old_w, size[1] / old_h)
    new_w = int(np.floor(old_w * scale_ratio))
    new_h = int(np.floor(old_h * scale_ratio))
    resized_img = mmcv.imresize(img, (new_w, new_h), backend='cv2')
    return resized_img

def gen_input_bin(file_batches, batch):
    i = 0
    for file in file_batches[batch]:
        i = i + 1
        print("batch", batch, file, "===", i)

        image = mmcv.imread(os.path.join(flags.image_src_path, file), backend='cv2')
        #image = mmcv.imrescale(image, (flags.model_input_width, flags.model_input_height))
        image = resize(image, (flags.model_input_width, flags.model_input_height))
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        image = mmcv.imnormalize(image, mean, std)
        h = image.shape[0]
        w = image.shape[1]
        pad_left = (flags.model_input_width - w) // 2
        pad_top = (flags.model_input_height - h) // 2
        pad_right = flags.model_input_width - pad_left - w
        pad_bottom = flags.model_input_height - pad_top - h
        image = mmcv.impad(image, padding=(pad_left, pad_top, pad_right, pad_bottom), pad_val=0)
        #mmcv.imwrite(image, './paded_jpg/' + file.split('.')[0] + '.jpg')
        image = image.transpose(2, 0, 1)
        image.tofile(os.path.join(flags.bin_file_path, file.split('.')[0] + ".bin"))

def preprocess(src_path, save_path):
    files = os.listdir(src_path)
    file_batches = [files[i:i + 100] for i in range(0, 5000, 100) if files[i:i + 100] != []]
    thread_pool = multiprocessing.Pool(len(file_batches))
    for batch in range(len(file_batches)):
        thread_pool.apply_async(gen_input_bin, args=(file_batches, batch))
    thread_pool.close()
    thread_pool.join()
    print("in thread, except will not report! please ensure bin files generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess of MaskRCNN PyTorch model')
    parser.add_argument("--image_src_path", default="./coco2017/", help='image of dataset')
    parser.add_argument("--bin_file_path", default="./coco2017_bin/", help='Preprocessed image buffer')
    parser.add_argument("--model_input_height", default=800, type=int, help='input tensor height')
    parser.add_argument("--model_input_width", default=1216, type=int, help='input tensor width')
    flags = parser.parse_args()    
    if not os.path.exists(flags.bin_file_path):
        os.makedirs(flags.bin_file_path)
    preprocess(flags.image_src_path, flags.bin_file_path)