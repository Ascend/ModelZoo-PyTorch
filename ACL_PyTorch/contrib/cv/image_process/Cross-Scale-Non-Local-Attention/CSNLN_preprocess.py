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
from torch.serialization import save
import torchvision.transforms as transforms
import os
import argparse
import torch
import json
import math
import imageio

parser = argparse.ArgumentParser(description='CSNLN preprocess script')
parser.add_argument('--s', default='', type=str, metavar='PATH',
                    help='path of source image files (default: none)')
parser.add_argument('--d', default='', type=str, metavar='PATH',
                    help='path of output (default: none)')
args = parser.parse_args()



pad_info = []
def chop(x, file_name="", save_path="", shave=10, min_size=3600):
    scale = 4
    c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    if h % 2 != 0:
        temp_h_half = h_half + 0.5
    else:
        temp_h_half = h_half
    if w % 2 != 0:
        temp_w_half = w_half + 0.5
    else:
        temp_w_half = w_half
    h_size, w_size = h_half + shave, w_half + shave
    h_size += scale-h_size%scale
    w_size += scale-w_size%scale
    lr_list = [
        x[:, 0:h_size, 0:w_size],
        x[:, 0:h_size, (w - w_size):w],
        x[:, (h - h_size):h, 0:w_size],
        x[:, (h - h_size):h, (w - w_size):w]]
    if w_size * h_size < min_size:
        for i in range(0, 4, 1):
            final_fileName = file_name.split('.')[0] + "_" + str(i)
            lr_batch = torch.cat(lr_list[i:(i + 1)], dim=0)
            pad_h = 56-h_size
            pad_w = 56-w_size
            lr_batch = transforms.Compose([
               transforms.Pad(padding=(0, 0, 56-w_size, 56-h_size), padding_mode='edge')
            ])(lr_batch)
             
            imageio.imsave(os.path.join(save_path, "png", final_fileName + ".png"), np.array(
                lr_batch).astype(np.uint8).transpose(1, 2, 0))
            lr_batch = np.array(lr_batch).astype(np.float32)/255
            lr_batch.tofile(os.path.join(
                  save_path, "bin_56", final_fileName + ".bin"))
            pad_info.append(
                {"name":final_fileName, "h_half": temp_h_half, "w_half": temp_w_half, "h_size":h_size, "w_size":w_size, "pad_h":pad_h, "pad_w":pad_w})
            with open("pad_info_56.json", "w") as f:
                f.write(json.dumps(pad_info, indent=4, separators=(',', ': ')))
            
    else:
        count = 0
        for patch in lr_list:
            temp_fileName = file_name.split('.')[0] + "_" + str(count) + ".png"
            pad_info.append(
                {"name":temp_fileName.split('.')[0], "h_half": h_half, "w_half": w_half, "h_size":h_size, "w_size":w_size})
            count = count + 1
            chop(patch, file_name=temp_fileName, save_path=save_path, shave=shave, min_size=min_size)

def preprocess(src_path, save_path):
    if not os.path.isdir(src_path):
        os.makedirs(src_path)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if not os.path.isdir(os.path.join(save_path, "bin_56")):
        os.makedirs(os.path.join(save_path, "bin_56"))
    if not os.path.isdir(os.path.join(save_path, "png")):
        os.makedirs(os.path.join(save_path, "png"))
    files = os.listdir(src_path)
    files.sort()
    for image_file in files:
        image = imageio.imread(os.path.join(
            src_path, image_file))
        np_transpose = np.ascontiguousarray(image.transpose((2, 0, 1)))
        image = torch.from_numpy(np_transpose).float()
        image.mul_(255 / 255)
        chop(image, file_name=image_file, save_path=save_path)
    

if __name__ == '__main__':
    preprocess(args.s, args.d)