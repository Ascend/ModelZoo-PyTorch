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

import numpy as np
import PIL.Image as pil_image
import torchvision.transforms as transforms
import os
import argparse

parser = argparse.ArgumentParser(description='SRCNN preprocess script')
parser.add_argument('-s', default='', type=str, metavar='PATH',
                    help='path of source image files (default: none)')
parser.add_argument('-d', default='', type=str, metavar='PATH',
                    help='path of output (default: none)')
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
    scale = 2
    preprocess = transforms.Compose([
        transforms.Resize(256, pil_image.BICUBIC),
        transforms.CenterCrop(256),
    ])
    for image_file in os.listdir(src_path):
        if not "_256.png" in image_file:
            image = pil_image.open(os.path.join(
                src_path, image_file)).convert('RGB')
            image = preprocess(image)
            image.save(os.path.join(
                save_path, "png", image_file).replace('.png', '_256.png'))
            image_width = (image.width // scale) * scale
            image_height = (image.height // scale) * scale

            image = image.resize((image_width, image_height),
                                 resample=pil_image.BICUBIC)

            image = image.resize(
                (image.width // scale, image.height // scale), resample=pil_image.BICUBIC)
            image = image.resize(
                (image.width * scale, image.height * scale), resample=pil_image.BICUBIC)
            image = np.array(image).astype(np.float32).transpose()

            y = convert_rgb_to_y(image)
            y /= 255.

            y.tofile(os.path.join(
                save_path, "bin", image_file.split('.')[0] + ".bin"))

            print("OK")


def convert_rgb_to_y(img):
    y = 16. + (64.738 * img[0, :, :] + 129.057 *
               img[1, :, :] + 25.064 * img[2, :, :]) / 256.
    return np.array([y])


if __name__ == '__main__':
    preprocess(args.s, args.d)
