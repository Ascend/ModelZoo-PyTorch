#!/usr/bin/python3.7
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
from PIL import Image
import os
import argparse
import sys
from torchvision.transforms import transforms, ToTensor

parser = argparse.ArgumentParser(description='SRCNN preprocess script')
parser.add_argument('--src_path', default='../data/test', type=str,
                    help='path of source image files (default: none)')
parser.add_argument('--save_path', default='./preprocess_data', type=str,
                    help='path of output (default: none)')
parser.add_argument('--set5_only', default=True, type=bool,
                    help='only preprocess set5 images')
parser.add_argument('--upscale_factor', default=2, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
args = parser.parse_args()

# 通过后缀检查是否为图片
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def preprocess(args):
    #print(args)
    src_path = args.src_path
    save_path = args.save_path
    # create dir
    if not os.path.exists(src_path):
        print(f'输入的路径{src_path}不存在！')
        sys.exit()
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    upscale_factor = args.upscale_factor
    lr_path = src_path + '/data/'
    if args.set5_only:
        lr_filenames = [os.path.join(lr_path, x) for x in os.listdir(lr_path) if
                        is_image_file(x) and x.split('_')[0] == 'Set5']
    else:
        lr_filenames = [os.path.join(lr_path, x) for x in os.listdir(lr_path) if is_image_file(x)]


    for index, image_file in enumerate(lr_filenames):
        image_name = image_file.split('/')[-1]
        lr_image = Image.open(image_file).convert('RGB')
        width, height = lr_image.size
        # 创建文件保存的路径
        img_path = os.path.join(save_path, f'img_{width}_{height}','png')
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        bin_path = os.path.join(save_path, f'img_{width}_{height}','bin')
        if not os.path.exists(bin_path):
            os.makedirs(bin_path)

        # 保存处理后的图片
        lr_image.save(os.path.join(img_path,image_name))

        # 数据转换
        lr_image = transforms.ToTensor()(lr_image).unsqueeze(0)
        img = to_numpy(lr_image)
        # img = np.array(lr_image).astype(np.float32)
        img.tofile(os.path.join(bin_path, image_name.split('.')[0] + ".bin"))
    #print("OK")

if __name__ == '__main__':
    preprocess(args)