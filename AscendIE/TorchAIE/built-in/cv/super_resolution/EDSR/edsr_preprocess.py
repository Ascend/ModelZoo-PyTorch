#!/usr/bin/python3.7
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

import torchvision.transforms as transforms
import torch
import imageio.v2 as imageio
import numpy as np
import os
import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='EDSR preprocess script')
parser.add_argument('-s', default='', type=str, metavar='PATH',
                    help='path of source image files (default: none)')
parser.add_argument('-d', default='', type=str, metavar='PATH',
                    help='path of output (default: none)')
parser.add_argument('--save_img', action='store_true',
                    help='save image')
args = parser.parse_args()


def preprocess(src_path, save_path):
    # create dir
    if not os.path.isdir(src_path):
        os.makedirs(src_path)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if not os.path.isdir(os.path.join(save_path, "png")):
        os.makedirs(os.path.join(save_path, "png"))
    if not os.path.isdir(os.path.join(save_path, "bin")):
        os.makedirs(os.path.join(save_path, "bin"))
    count = 0
    pad_info = []
    for image_file in tqdm(os.listdir(src_path)):
        image = imageio.imread(os.path.join(
            src_path, image_file))
        image = np2Tensor(image)
        lr_image, pad_xy = pad(image, 1020)
        pad_info.append(
            {"name": image_file, "pad_x": pad_xy[0], "pad_y": pad_xy[1]})

        if args.save_img:
            imageio.imsave(os.path.join(save_path, "png", image_file), np.array(
                lr_image).astype(np.uint8).transpose(1, 2, 0))

        lr_image = np.array(lr_image).astype(np.uint8)
        lr_image = lr_image.astype(np.float32)

        lr_image.tofile(os.path.join(
            save_path, "bin", image_file.split('.')[0] + ".bin"))
    with open("pad_info.json", "w") as f:
        f.write(json.dumps(pad_info, indent=4, separators=(',', ': ')))


def pad(image, size):
    z, y, x = image.shape
    pad_x = size - x
    pad_y = size - y
    process = transforms.Compose([
        transforms.Pad([0, 0, pad_x, pad_y])
    ])
    return process(image), [pad_x, pad_y]


def np2Tensor(img, rgb_range=255):
    np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
    tensor = torch.from_numpy(np_transpose).float()
    tensor.mul_(rgb_range / 255)

    return tensor


if __name__ == '__main__':
    preprocess(args.s, args.d)