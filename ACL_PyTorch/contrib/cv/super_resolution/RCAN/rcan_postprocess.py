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
import math
import json
import argparse

import torch
import numpy as np
import PIL.Image as pil_image
import imageio

from evaluate import evaluate


def postprocess(infer_path, pad_info_path, save_path, scale, shape):
    with open(pad_info_path) as f:
        pad_info = json.load(f)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for file in os.listdir(infer_path):
        res_path = os.path.join(infer_path, file)
        res = np.fromfile(res_path, dtype=np.float32)
        h, w = shape
        res = res.reshape(3, h * scale, w * scale).transpose(1, 2, 0)
        img = torch.from_numpy(res)
        img = quantize(img, 255)
        img = crop(file, img, pad_info, scale)
        img = img.byte().cpu()
        des_path = os.path.join(save_path, file.split('_')[0]+'.png')
        imageio.imwrite(des_path, img.numpy())


def crop(file, img, pad_info, scale):
    for pad_meta in pad_info:
        if file[0:4] in pad_meta['name']:
            pad_x = pad_meta['pad_x'] * scale
            pad_y = pad_meta['pad_y'] * scale
    if pad_x == 0 and pad_y == 0:
        img = img
    elif pad_x == 0:
        img = img[0:-pad_y, :, :]
    elif pad_y == 0:
        img = img[:, 0:-pad_x, :]
    else:
        img = img[0:-pad_y, 0:-pad_x, :]

    return img


def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
                        description='postprocess and evaluate.')
    parser.add_argument('--infer_results', type=str, 
                        help='dirctory of inference results.')
    parser.add_argument('--pad_info', type=str, 
                        help='pad info path when preprocess.')
    parser.add_argument('--hr_images', type=str, 
                        help='dirctory of HR images.')
    parser.add_argument('--save_dir', type=str, default='./gen_images',
                        help='dirctory to save the generated images.')
    parser.add_argument('--shape', type=int, nargs=2, default=[256, 256], 
                        help='the height and width of model input.')
    parser.add_argument('--scale', type=int, default=2, 
                        help='the magnifying rates of output images.')
    args = parser.parse_args()

    postprocess(args.infer_results, args.pad_info, 
                args.save_dir, args.scale, args.shape)
    print('Images generated! path: {}'.format(args.save_dir))

    evaluate(args.save_dir, args.hr_images)
