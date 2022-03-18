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

import os
import torch
import numpy as np
import PIL.Image as pil_image
import imageio
import math
import json
import argparse

with open("pad_info.json") as f:
    pad_info = json.load(f)
scale = 2
size = 256
def postprocess(src_path, save_path):
    count = 0
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for file in os.listdir(src_path):
        array = np.fromfile(os.path.join(src_path, file), dtype=np.float32).reshape(
            3, size*scale, size*scale).transpose((1, 2, 0))
        img = torch.from_numpy(array)
        img = quantize(img, 255)
        img = crop(file, img, pad_info)
        img = img.byte().cpu()
        imageio.imwrite(os.path.join(save_path, file.split('_')[0]+".png"), img.numpy())
        count += 1
        print("OK, count = ", count)

def crop(file, img, pad_info):
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
    
    paser = argparse.ArgumentParser(description="Script to post-process data.")
    paser.add_argument('-s', type=str, help='dirctory of raw data')
    paser.add_argument('-d', type=str, help='dirctory of post-processed data')
    args = paser.parse_args()


    postprocess(args.s, args.d)
