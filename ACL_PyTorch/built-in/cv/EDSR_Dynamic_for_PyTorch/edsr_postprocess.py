# Copyright 2022 Huawei Technologies Co., Ltd
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
import math
import json
import argparse
import torch
import imageio
import numpy as np
from tqdm import tqdm


parser = argparse.ArgumentParser(description='EDSR post process script')
parser.add_argument('--res', default='', type=str, metavar='PATH',
                    help='om result path')
parser.add_argument('--HR', default='', type=str, metavar='PATH',
                    help='high res path')
parser.add_argument('--save_path', default='', type=str, metavar='PATH',
                    help='result image save path')
args = parser.parse_args()


def postprocess(img_src_path, src_path, save_path):
    data = []
    for idx, file_name in tqdm(enumerate(os.listdir(src_path))):
        array = np.load(
            os.path.join(src_path, file_name), allow_pickle=True).squeeze(0).transpose(1, 2, 0)
        img = torch.from_numpy(array.astype("float32"))
        img = quantize(img, 255)

        img_path = os.path.join(
            img_src_path, "{}.png".format(file_name.split('x')[0])
        )
        hr = imageio.imread(img_path)
        hr = torch.from_numpy(hr)
        hr = hr[0:img.shape[0], 0:img.shape[1]]
        psnr = calc_psnr(img, hr, scale, 255)
        data.append({"file": file_name, "psnr": psnr})

    data = eval_acc(data)
    json_data = json.dumps(data, indent=4, separators=(',', ': '))
    with open(args.save_path, 'w') as f:
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
    sr = sr.permute(2, 0, 1).unsqueeze(0)
    hr = hr.permute(2, 0, 1).unsqueeze(0)
    if hr.nelement() == 1:
        return 0

    diff = (sr - hr) / rgb_range
    shave = scale
    if diff.size(1) > 1:
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        diff = diff.mul(convert).sum(dim=1)
    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()
    return -10 * math.log10(mse)


if __name__ == '__main__':
    scale = 2
    postprocess(args.HR, args.res, args.save_path)
