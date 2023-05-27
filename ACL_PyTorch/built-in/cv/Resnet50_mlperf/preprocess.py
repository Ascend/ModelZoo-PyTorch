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
import sys
import multiprocessing
import cv2
from tqdm import tqdm
import numpy as np

img_resize = 224


def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img


def resize_with_aspectratio(img, size, scale=87.5, inter_pol=cv2.INTER_LINEAR):
    if isinstance(size, int):
        height, width, _ = img.shape
        new_height = int(100. * size / scale)
        new_width = int(100. * size / scale)
        if height > width:
            w = new_width
            h = int(new_height * height / width)
        else:
            h = new_height
            w = int(new_width * width / height)
        img = cv2.resize(img, (w, h), interpolation=inter_pol)
        return img
    else:
        img = img.resize(size[::-1], interpolation)
        return img


def gen_input_bin(file_batches, batch, src_path, save_path):
    for file in tqdm(file_batches[batch]):
        image = cv2.imread(os.path.join(src_path, file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        size = img_resize
        cv2_interpol = cv2.INTER_AREA
        img = resize_with_aspectratio(image, size, inter_pol=cv2_interpol)
        img = center_crop(img, size, size)
        img = np.asarray(img, dtype='float32')

        # normalize image
        means = np.array([123.68, 116.78, 103.94], dtype=np.float32)
        img -= means

        img = img.transpose([2, 0, 1])

        np.save(os.path.join(save_path, file.split('.')[0] + ".npy"), img)


def preprocess(source_path, dest_path):
    files = os.listdir(source_path)
    files.sort()
    if len(files) < 500:
        file_batches = [files[0: len(files)]]
    else:
        file_batches = [files[i:i + 500] for i in range(0, len(files), 500) if files[i:i + 500] != []]
    thread_pool = multiprocessing.Pool(len(file_batches))
    for batch in range(len(file_batches)):
        thread_pool.apply_async(gen_input_bin, args=(file_batches, batch, source_path, dest_path))
    thread_pool.close()
    thread_pool.join()
    print("in thread, except will not report! please ensure bin files generated.")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise Exception("usage: python3 xxx.py [input_path] [output_path]")
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    input_path = os.path.realpath(input_path)
    output_path = os.path.realpath(output_path)
    if not os.path.isdir(output_path):
        os.makedirs(os.path.realpath(output_path))
    preprocess(input_path, output_path)
