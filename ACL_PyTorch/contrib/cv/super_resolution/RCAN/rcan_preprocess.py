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
import argparse
import torch
import PIL.Image as pil_image
import imageio
import numpy as np
import os
import json


def preprocess(src_path, save_path, size):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    pad_info = []
    for image_file in os.listdir(src_path):
        print("file: ", image_file)
        image = imageio.imread(os.path.join(src_path, image_file))
        image = np2Tensor(image)
        lr_image, pad_xy = pad(image, size)
        pad_info.append(
            {"name": image_file, "pad_x": pad_xy[0], "pad_y": pad_xy[1]})

        lr_image = np.array(lr_image).astype(np.uint8)
        lr_image = lr_image.astype(np.float32)
        print(lr_image.shape)

        lr_image.tofile(os.path.join(
            save_path, image_file.split('.')[0] + ".bin"))

    with open("pad_info.json", "w") as f:
        f.write(json.dumps(pad_info, indent=4, separators=(',', ': ')))


def pad(image, size):
    z, y, x = image.shape
    pad_x = size - x
    pad_y = size - y
    pad_x1 = pad_x
    pad_y1 = pad_y
    pad_x2 = 0
    pad_y2 = 0
    if pad_x1 >= x:
        pad_x2 = pad_x - x + 1
        pad_x1 = x - 1
    if pad_y1 >= y:
        pad_y2 = pad_y - y + 1
        pad_y1 = y - 1
    print('reflect_pad:{}, normal_pad:{}'.format((pad_x1, pad_y1), (pad_x2,pad_y2)))
    process = transforms.Compose([
        transforms.Pad([0, 0, pad_x1, pad_y1], padding_mode='reflect'),
        transforms.Pad([0, 0, pad_x2, pad_y2])
    ])
    return process(image), [pad_x, pad_y]


def np2Tensor(img, rgb_range=255):
    np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
    tensor = torch.from_numpy(np_transpose).float()
    tensor.mul_(rgb_range / 255)

    return tensor


if __name__ == '__main__':
    paser = argparse.ArgumentParser(description="Script to preprocess data.")
    paser.add_argument('-s', type=str, help='dirctory of raw data')
    paser.add_argument('-d', type=str, help='dirctory of preprocessed data')
    paser.add_argument('--size', type=int, help='target size of preprocessing')
    args = paser.parse_args()
    preprocess(args.s, args.d, args.size)
