# Copyright (C) Huawei Technologies Co., Ltd 2022-2022. All rights reserved.
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
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import numpy as np
import argparse

def preprocess(src_path, save_path, batch_size):

    in_files = sorted(os.listdir(src_path))

    preprocesser = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    num = batch_size
    imgs = np.array([]).astype(np.float32)
    for idx, file in enumerate(in_files):
        num = num - 1
        idx = idx + 1
        input_image = Image.open(src_path + '/' + file).convert('RGB')
        input_tensor = preprocesser(input_image)
        
        img = np.array(input_tensor).astype(np.float32)
        imgs = np.append(imgs, img)
        if num==0:
            num = batch_size
            imgs.tofile(os.path.join(save_path, file.split('.')[0] + ".bin"))
            print(imgs.shape)
            imgs = np.array([]).astype(np.float32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--src-path', type=str, default = None)
    parser.add_argument('--save-path', type=str, default = None)
    parser.add_argument('--batch-size', type=int, default = None)
    args = parser.parse_args()
    preprocess(args.src_path, args.save_path, args.batch_size)