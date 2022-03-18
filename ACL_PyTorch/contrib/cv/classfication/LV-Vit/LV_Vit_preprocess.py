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

import torch
from timm.data.transforms_factory import transforms_imagenet_eval
from torchvision import transforms
from PIL import Image


def preprocess(args, src_path, save_path):
    if isinstance(args.input_size, tuple):
        img_size = args.input_size[-2:]
    else:
        img_size = args.input_size

    preprocesser = transforms_imagenet_eval(
        img_size,
        interpolation=args.interpolation,
        use_prefetcher=args.use_prefetcher,
        mean=args.mean,
        std=args.std,
        crop_pct=args.crop_pct)

    i = 0
    in_files = os.listdir(src_path)
    for file in in_files:
        i = i + 1
        print(file, "===", i)
        input_image = Image.open(src_path + file).convert('RGB')
        input_tensor = preprocesser(input_image)
        img = np.array(input_tensor).astype(np.float32)
        img = (img - np.array([x * 255 for x in args.mean]).reshape(3, 1, 1)) / np.array(
            [x * 255 for x in args.std]).reshape(3, 1, 1)
        img = img.astype(np.float32)
        img.tofile(os.path.join(save_path, file.split('.')[0] + ".bin"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', default='', type=str)
    parser.add_argument('--save_path', default='', type=str)
    parser.add_argument('--interpolation', default='bicubic', type=str, metavar='NAME',
                        help='Image resize interpolation type (overrides model)')
    parser.add_argument('use_prefetcher', action='store_true', default=True,
                        help='enable fast prefetcher')
    parser.add_argument('--crop-pct', default=0.9, type=float,
                        metavar='N', help='Input image center crop percent (for validation only)')
    args = parser.parse_args()
    args.mean = (0.485, 0.456, 0.406)
    args.std = (0.229, 0.224, 0.225)
    args.input_size = (3, 224, 224)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    preprocess(args, args.src_path, args.save_path)


if __name__ == '__main__':
    main()
