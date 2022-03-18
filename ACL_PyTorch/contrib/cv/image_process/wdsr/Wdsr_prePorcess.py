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

import argparse
import os

import numpy as np
from PIL import Image
from PIL import ImageOps

def load_item(lr_path, hr_path, scale, width, height):
    hr_image = Image.open(hr_path)
    lr_image = Image.open(lr_path)

    lr_image = ImageOps.pad(lr_image, (width, height), method=Image.ANTIALIAS, centering=(0.5, 0.5), color=0)

    lr_image = np.asarray(lr_image)
    hr_image = np.asarray(hr_image)
    return lr_image, hr_image


def sample_patch(lr_image, hr_image, scale):
    hr_image = hr_image[:lr_image.shape[0] * scale, :lr_image.shape[1] * scale]
    return lr_image, hr_image


def preProcess(lr_path, hr_path, scale, width, height):
    lr_image, hr_image = load_item(lr_path, hr_path, scale, width, height)
    lr_image, hr_image = sample_patch(lr_image, hr_image, scale)

    lr_image = np.ascontiguousarray(lr_image)

    lr_image = lr_image.transpose((2, 0, 1))

    lr_image = lr_image.astype(np.float32) / 255.

    return lr_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--lr_path',
        default=None,
        type=str,
        required=True
    )
    parser.add_argument(
        '--hr_path',
        default=None,
        type=str,
        required=True
    )
    parser.add_argument(
        '--save_lr_path',
        default=None,
        type=str,
        required=True
    )
    parser.add_argument(
        '--width',
        default=None,
        type=int,
        required=True
    )
    parser.add_argument(
        '--height',
        default=None,
        type=int,
        required=True
    )
    parser.add_argument(
        '--scale',
        help='Scale factor for image super-resolution.',
        default=2,
        type=int,
        required=True
    )
    # 解析参数
    args, _ = parser.parse_known_args()

    lrFiles = os.listdir(args.lr_path)
    hrFiles = os.listdir(args.hr_path)

    lrFiles.sort()
    hrFiles.sort()

    os.makedirs(args.save_lr_path)


    for i in range(0, len(lrFiles)):
        print(lrFiles[i])
        lr_image = preProcess(os.path.join(args.lr_path, lrFiles[i]), os.path.join(args.hr_path, hrFiles[i]),
                                        args.scale, args.width, args.height)
        lr_image.tofile(os.path.join(args.save_lr_path, lrFiles[i].split('.')[0] + '.bin'))
