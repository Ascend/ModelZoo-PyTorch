# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import argparse
from PIL import Image
import numpy as np
import multiprocessing
import torchvision.transforms as transforms

from tqdm import tqdm

normalization = {
    'resize': 342,
    'centercrop': 299,
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
}


def center_crop(img, output_size):
    if isinstance(output_size, int):
        output_size = (int(output_size), int(output_size))
    image_width, image_height = img.size
    crop_height, crop_width = output_size
    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))
    img = img.crop((crop_left, crop_top, crop_left + \
                    crop_width, crop_top + crop_height))
    return img


def resize(img, size, interpolation=Image.BILINEAR):
    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)

def aipp(img):
    mean = np.array([123.675, 116.28, 103.53]).reshape(1, 1, 3)
    std = np.array([0.0171247538316637,
                    0.0175070028011204,
                    0.0174291938997821]).reshape(1, 1, 3)
    img = (img - mean) * std
    return img

def gen_input_bin(src_path, file, save_path):
    # RGBA to RGB
    image = Image.open(os.path.join(src_path, file)).convert('RGB')
    image = resize(image, normalization['resize'])  # Resize
    image = center_crop(image, normalization['centercrop'])  # CenterCrop
    img = np.array(aipp(image), dtype=np.float32).transpose(2, 0, 1)  # aipp
    img.tofile(os.path.join(save_path, file.split('.')[0] + '.bin'))

def preprocess(src_path, save_path):
    files = os.listdir(src_path)
    for file in tqdm(files):
        gen_input_bin(src_path, file, save_path)
    print('Done! please ensure bin files generated.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('data preprocess.')
    parser.add_argument('--src_path', type=str, required=True,
                        help='path to original dataset.')
    parser.add_argument('--save_path', type=str, required=True,
                        help='a directory to save bin files.')
    args = parser.parse_args()

    src_path = os.path.realpath(args.src_path)
    save_path = os.path.realpath(args.save_path)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    preprocess(src_path, save_path)