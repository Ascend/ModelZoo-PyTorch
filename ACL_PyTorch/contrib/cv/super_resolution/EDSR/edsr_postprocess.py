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

import torch
import numpy as np
import os
import argparse
import json
import math
import imageio


parser = argparse.ArgumentParser(description='EDSR post process script')
parser.add_argument('--res', default='', type=str, metavar='PATH',
                    help='om result path')
parser.add_argument('--HR', default='', type=str, metavar='PATH',
                    help='high res path')

parser.add_argument('--save', action='store_true',
                    help='save image')
parser.add_argument('--save_path', default='', type=str, metavar='PATH',
                    help='result image save path')
args = parser.parse_args()

with open("pad_info.json") as f:
    pad_info = json.load(f)
scale = 2
size = 1020

def postprocess(img_src_path, src_path, save_path):
    data = []
    count = 0
    # create dir
    if not os.path.isdir(save_path) and args.save:
        os.makedirs(save_path)
    for file in os.listdir(src_path):
        array = np.fromfile(os.path.join(src_path, file), dtype=np.float32).reshape(
            3, size*scale, size*scale).transpose((1, 2, 0))
        img = torch.from_numpy(array)
        img = quantize(img, 255)
        img = crop(file, img, pad_info)
        for img_file in os.listdir(img_src_path):
            if img_file[0:4] in file:
                hr = imageio.imread(os.path.join(img_src_path, img_file))
                hr = torch.from_numpy(hr)
                psnr = calc_psnr(img, hr, scale, 255)
                data.append({"file": file, "psnr": psnr})
                break

        img = img.byte().cpu()
        if args.save:
            imageio.imwrite(os.path.join(save_path+file)+".png", img.numpy())
        count += 1
        print("OK, count = ", count)

    data = eval_acc(data)
    json_data = json.dumps(
        data, indent=4, separators=(',', ': '))
    with open("result.json", 'w') as f:
        f.write(json_data)

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

def eval_acc(data):
    acc = 0
    for item in data:
        acc += item["psnr"]
    acc /= len(data)
    print("accuracy: ",acc)
    return {
        "accuracy": acc,
        "data": data
    }


def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def calc_psnr(sr, hr, scale, rgb_range):
    sr = sr.permute(2, 0, 1).unsqueeze(0)
    hr = hr.permute(2, 0, 1).unsqueeze(0)
    if hr.nelement() == 1:
        return 0

    diff = (sr - hr) / rgb_range
    shave = scale + 6

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)

if __name__ == '__main__':
    postprocess(args.HR, args.res, args.save_path)
