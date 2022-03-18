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

parser = argparse.ArgumentParser(description='CSNLN post process script')
parser.add_argument('--res', default='', type=str, metavar='PATH',
                    help='om result path')
parser.add_argument('--hr', default='', type=str, metavar='PATH',
                    help='high res path')
parser.add_argument('--save_path', default='', type=str, metavar='PATH',
                    help='result image save path')
args = parser.parse_args()


with open("pad_info_56.json") as f:
    pad_info = json.load(f)
scale = 4

def postprocess(hr_src_path, bin_path, save_path):
    data = []
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    sr_list = merge(bin_path)
    files = os.listdir(hr_src_path)
    files.sort()
    for i, img_file in enumerate(files):
        img = sr_list[i]
        img = quantize(img, 1)
        hr = imageio.imread(os.path.join(hr_src_path, img_file))
        hr = torch.from_numpy(hr)
        hr = hr / 255
        psnr = calc_psnr(img, hr, scale, 1)
        data.append({"file": img_file, "psnr": psnr})
        
        img = (img * 255).byte().cpu()
        imageio.imwrite(os.path.join(save_path, img_file+".png"), img.numpy().astype(np.uint8).transpose(1, 2, 0))

    data = eval_acc(data)
    json_data = json.dumps(
        data, indent=4, separators=(',', ': '))
    with open("result.json", 'w') as f:
        f.write(json_data)

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
    sr = sr.unsqueeze(0)
    hr = hr.permute(2, 0, 1).unsqueeze(0)
    if hr.nelement() == 1:
        return 0

    diff = (sr - hr) / rgb_range
    shave = 4
    if diff.size(1) > 1:
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        diff = diff.mul(convert).sum(dim=1)

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)

def merge(src_path):
    min_list = []
    max_list = []
    for i, pad_meta in enumerate(pad_info):
        if i % 5 == 0 and i < 16:
            max_list.append(pad_meta)
        else:
            min_list.append(pad_meta)
    h_half, w_half = -1, -1
    h_size, w_size = -1, -1
    h, w = -1, -1
    temp_img = None
    sr_list = []
    sr = []
    files = os.listdir(src_path)
    files.sort()
    for i, file in enumerate(files):
        array = np.fromfile(os.path.join(src_path, file), dtype=np.float32)
        array = array.reshape(
            3, 56*4, 56*4)
        img = torch.from_numpy(array)
        pad_h, pad_w = min_list[i]['pad_h'], min_list[i]['pad_w']
        if pad_h == 0 and pad_w == 0:
          img = img
        elif pad_h == 0:
          img = img[:, :, 0:-pad_w]
        elif pad_w == 0:
          img = img[:, 0:-pad_h, :]
        else:
          img = img[:, 0:-pad_h, 0:-pad_w]
        if i % 4 == 0:
            h_half, w_half = int(min_list[i]['h_half'] * scale), int(min_list[i]['w_half'] * scale)
            h_size, w_size = min_list[i]['h_size'] * scale, min_list[i]['w_size'] * scale
            h, w = h_half * 2, w_half * 2
            temp_img = torch.zeros(3, h, w)
            temp_img[:, 0:h_half, 0:w_half] = img[:, 0:h_half, 0:w_half]
        elif i % 4 == 1:
            temp_img[:, 0:h_half, w_half:w] = img[:, 0:h_half, (w_size - w + w_half):w_size]
        elif i % 4 == 2:
            temp_img[:, h_half:h, 0:w_half] = img[:, (h_size - h + h_half):h_size, 0:w_half]
        elif i % 4 == 3:
            temp_img[:, h_half:h, w_half:w] = img[:, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]
            sr_list.append(temp_img)

    h_half, w_half = max_list[0]['h_half'] * scale, max_list[0]['w_half'] * scale
    h_size, w_size = max_list[0]['h_size'] * scale, max_list[0]['w_size'] * scale
    h, w = h_half * 2, w_half * 2
    output = torch.zeros(3, h, w)
    output[:, 0:h_half, 0:w_half] \
        = sr_list[0][:, 0:h_half, 0:w_half]
    output[:, 0:h_half, w_half:w] \
        = sr_list[1][:, 0:h_half, (w_size - w + w_half):w_size]
    output[:, h_half:h, 0:w_half] \
        = sr_list[2][:, (h_size - h + h_half):h_size, 0:w_half]
    output[:, h_half:h, w_half:w] \
        = sr_list[3][:, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]
    sr.append(output)
    sr.append(sr_list[4])
    sr.append(sr_list[5])
    sr.append(sr_list[6])
    sr.append(sr_list[7])
    return sr
    

if __name__ == '__main__':
    res = args.res
    hr = args.hr
    save_path = args.save_path
    postprocess(hr, res, save_path)