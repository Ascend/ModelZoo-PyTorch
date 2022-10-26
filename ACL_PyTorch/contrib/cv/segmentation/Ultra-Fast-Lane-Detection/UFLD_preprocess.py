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
import argparse

import tqdm
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms


def jpg_to_bin(dataset_path, bin_path):

    test_txt_path = os.path.join(dataset_path, 'test.txt')
    image_names = []
    with open(test_txt_path, 'r', encoding='utf-8') as f:
        image_names.append(f.readline().strip())
   

    preprocess = transforms.Compose(
        [transforms.Resize((288, 800)),])

    if not os.path.isdir(bin_path):
        os.makedirs(bin_path)

    for img_name in tqdm.tqdm(image_names):
        src_path = os.path.join(dataset_path, img_name)
        if not os.path.isfile(src_path):
            continue
        dst_path = os.path.join(bin_path, 
            img_name.replace('/', '-').replace('.jpg', '.bin'))

        image = Image.open(src_path).convert('RGB')
        input_tensor = preprocess(image)
        img = np.array(input_tensor).astype(np.uint8)
        img.tofile(dst_path)


if __name__=='__main__':

    parser = argparse.ArgumentParser(
                        'convert original image to bin file.')
    parser.add_argument('--dataset-path', type=str,
                        help='the root path of the dataset.')
    parser.add_argument('--bin-path', type=str,
                        help='a directory to save bin file.')
    args = parser.parse_args()

    jpg_to_bin(args.dataset_path, args.bin_path)
    print('Preprocess Done.')
