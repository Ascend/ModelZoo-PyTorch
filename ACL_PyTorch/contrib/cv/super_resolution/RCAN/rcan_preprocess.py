# Copyright 2022 Huawei Technologies Co., Ltd
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
import json
import argparse
import os.path as osp

import numpy as np
import imageio.v2 as imageio
import PIL.Image as pil_image
import torch
import torchvision.transforms as transforms


def preprocess(src_path, save_path, size):
    bin_dir = osp.join(save_path, 'bin')
    os.makedirs(bin_dir, exist_ok=True)
    pad_info = []
    for image_file in os.listdir(src_path):
        print('file: ', image_file)
        image = imageio.imread(osp.join(src_path, image_file))
        image = np2Tensor(image)
        lr_image, pad_xy = pad(image, size)
        pad_info.append(
            {'name': image_file, 'pad_x': pad_xy[0], 'pad_y': pad_xy[1]})

        lr_image = np.array(lr_image).astype(np.uint8)
        lr_image = lr_image.astype(np.float32)
        print(lr_image.shape)

        des_path = osp.join(bin_dir, image_file.split('.')[0] + '.bin')
        lr_image.tofile(des_path)

    with open(osp.join(save_path, 'pad_info.json'), 'w') as f:
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
    print(f'reflect_pad:{(pad_x1, pad_y1)}, normal_pad:{(pad_x2, pad_y2)}')
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
    parser = argparse.ArgumentParser(description='preprocess data.')
    parser.add_argument('-s', '--source', type=str, 
                        help='the dirctory of saving LR images.')
    parser.add_argument('-o', '--output', type=str, 
                        help='the dirctory to save preprocessed data.')
    parser.add_argument('-sz', '--size', type=int, default=256,
                        help='target size of preprocessing.')
    args = parser.parse_args()
    preprocess(args.source, args.output, args.size)
