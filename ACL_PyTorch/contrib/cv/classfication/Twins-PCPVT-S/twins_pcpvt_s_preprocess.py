# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from torchvision import datasets, transforms
import argparse
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
import os
from PIL import Image
import numpy as np


def get_args_parser():
    parser = argparse.ArgumentParser('Twins_PCPVT_S data2bin', add_help=False)

    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--data_path', default='/opt/npu/imagenet', type=str,
                        help='dataset path')
    parser.add_argument('--prep_dataset', default='prep_dataset')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    return parser

def build_transform():
    resize_im = args.input_size > 32
    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

def main(args):
    Transform = build_transform()
    val_path = os.path.join(args.data_path, 'val')
    save_path = os.path.realpath(args.prep_dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    val_files = os.listdir(val_path)

    i = 0
    for val_set in val_files:
        valset_p = os.path.join(val_path, val_set)

        if not os.path.isdir(valset_p):
            i = i + 1
            file = val_set
            input_image = Image.open(valset_p).convert('RGB')
            input_tensor = Transform(input_image)
            img = np.array(input_tensor).astype(np.float32)
            img.tofile(os.path.join(save_path, file.split('.')[0] + ".bin"))
            continue

        files = os.listdir(valset_p)
        save_path_ = os.path.join(save_path, val_set)
        os.makedirs(save_path_)
        for file in files:
            i = i + 1
            input_image = Image.open(valset_p + '/' + file).convert('RGB')
            input_tensor = Transform(input_image)
            img = np.array(input_tensor).astype(np.float32)
            img.tofile(os.path.join(save_path_, file.split('.')[0] + ".bin"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Twins_PCPVT_S data2bin', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
