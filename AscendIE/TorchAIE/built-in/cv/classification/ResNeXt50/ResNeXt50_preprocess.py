# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import multiprocessing
from PIL import Image
from tqdm import tqdm
import numpy as np


def center_crop(img, output_size):
    if isinstance(output_size, int):
        output_size = (int(output_size), int(output_size))
    image_width, image_height = img.size
    crop_height, crop_width = output_size
    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))
    return img.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))


def resize(img, size, interpolation):
    if isinstance(size, int):
        w, h = img.size

        if (w <= h and w == size) or (h <= w and h == size):
            return img
            
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return img.resize((ow, oh), interpolation)

    else:
        return img.resize(size[::-1], interpolation)


def gen_input_bin(_file_batch, _batches, _save_paths):
    for files in tqdm(_file_batch[_batches]):
        # RGBA to RGB
        image = Image.open(files[1]).convert('RGB')
        image = resize(image, 256, Image.BILINEAR) # Resize
        image = center_crop(image, 224) # CenterCrop
        r, g, b = image.split()
        # Normalize the color channels
        b = (b - 103.53) * 0.0174291938997821
        g = (g - 116.28) * 0.0175070028011204
        r = (r - 123.675) * 0.0171247538316637
        image = image.merge('RGB', (r, g, b))
        img = np.array(image).astype(np.int8)
        img.tofile(os.path.join(_save_paths, files[0].split('.')[0] + ".bin"))


def preprocess(src_path, save_path):
    files = os.listdir(src_path)
    image_infos = []
    for file_name in files:
        file_path = os.path.join(src_path, file_name)
        if os.path.isdir(file_path):
            image_infos += [(image_name, os.path.join(file_path, image_name)) \
            for image_name in os.listdir(file_path)]
        else:
            image_infos.append((file_name, file_path))
    image_infos.sort()
    if len(image_infos) < 500:
        file_batches = [image_infos[0: len(image_infos)]]
    else:
        file_batches = [image_infos[i:i + 500] for i in range(0, len(image_infos), 500) if image_infos[i:i + 500] != []]
    thread_pool = multiprocessing.Pool(len(file_batches))
    for batch in range(len(file_batches)):
        thread_pool.apply_async(gen_input_bin, args=(file_batches, batch, save_path))
    thread_pool.close()
    thread_pool.join()
    

if __name__ == '__main__':
    src_paths = sys.argv[1]
    save_paths = sys.argv[2]
    src_paths = os.path.realpath(src_paths)
    save_paths = os.path.realpath(save_paths)
    if not os.path.isdir(save_paths):
        os.makedirs(os.path.realpath(save_paths))
    preprocess(src_paths, save_paths)
    