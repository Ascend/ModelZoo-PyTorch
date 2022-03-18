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

import torchvision.transforms as transforms
import argparse
import torch
from PIL import Image
import torch.utils.data as data
import numpy as np
import os
import sys
sys.path.append('./TextSnake.pytorch')
from util.augmentation import BaseTransform



def preprocess(args):
    # preprocess = transforms.Compose([
    #     transforms.Resize([args.input_size, args.input_size]),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=args.means, std=args.stds),
    # ])
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    transform = BaseTransform(size=args.input_size, mean=args.means, std=args.stds) 
    i = 0
    in_files = os.listdir(args.src_path)
    for file in in_files:
        i = i + 1
        print(file, "===", i)
        input_image = Image.open(args.src_path + '/' + file)
        input_image = np.array(input_image)
        img, _ = transform(input_image)
        img = img.transpose(2, 0, 1)
        # print(img.shape)
        # print(type(img))
        # input_tensor = preprocess(input_image)
        # print(input_tensor.shape)
        # img = np.array(input_tensor).astype(np.float32)
        img.tofile(os.path.join(args.save_path, file.split('.')[0] + ".bin"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--input_size', default=512, type=int, help='model input size')
    parser.add_argument('--means', type=int, default=(0.485, 0.456, 0.406), nargs='+', help='mean')
    parser.add_argument('--stds', type=int, default=(0.229, 0.224, 0.225), nargs='+', help='std')
    args = parser.parse_args()
    
    preprocess(args)
