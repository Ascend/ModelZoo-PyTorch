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
import torch
import multiprocessing

def resize(img, size):
    old_h = img.shape[0]
    old_w = img.shape[1]
    scale_ratio = 800 / min(old_w, old_h)
    new_w = int(np.floor(old_w * scale_ratio))
    new_h = int(np.floor(old_h * scale_ratio))
    if max(new_h, new_w) > 1333:
        scale = 1333 / max(new_h, new_w)
        new_h = new_h * scale
        new_w = new_w * scale
    new_w = int(new_w + 0.5)
    new_h = int(new_h + 0.5)
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized_img

def gen_input_bin(file_batches, batch):
    i = 0
    for file in file_batches[batch]:
        i = i + 1
        print("batch", batch, file, "===", i)

        image = cv2.imread(os.path.join(flags.image_src_path, file), cv2.IMREAD_COLOR)
        image = resize(image, (800, 1333))
        mean = np.array([103.53, 116.28, 123.675], dtype=np.float32)
        std = np.array([1., 1., 1.], dtype=np.float32)
        img = image.copy().astype(np.float32)
        mean = np.float64(mean.reshape(1, -1))
        std = 1 / np.float64(std.reshape(1, -1))
        cv2.subtract(img, mean, img)
        cv2.multiply(img, std, img)
        img = cv2.copyMakeBorder(img, 0, flags.model_input_height - img.shape[0], 0, flags.model_input_width - img.shape[1], cv2.BORDER_CONSTANT, value=0)
        #os.makedirs('./paded_jpg/', exist_ok=True)
        #cv2.imwrite('./paded_jpg/' + file.split('.')[0] + '.jpg', img)
        img = img.transpose(2, 0, 1)
        img.tofile(os.path.join(flags.bin_file_path, file.split('.')[0] + ".bin"))

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
    parser.add_argument("--model_input_height", default=1344, type=int, help='input tensor height')
    parser.add_argument("--model_input_width", default=1344, type=int, help='input tensor width')
    flags = parser.parse_args()    
    if not os.path.exists(flags.bin_file_path):
        os.makedirs(flags.bin_file_path)
    preprocess(flags.image_src_path, flags.bin_file_path)

