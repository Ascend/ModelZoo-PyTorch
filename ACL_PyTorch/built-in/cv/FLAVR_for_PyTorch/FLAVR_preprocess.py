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

import argparse
import sys
import os

import torch
import numpy as np
from tqdm import tqdm

sys.path.append(r"./FLAVR")
from FLAVR.dataset.ucf101_test import get_loader

def parse_args():
    parser = argparse.ArgumentParser(description='process images and save to binary files')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='the data root of test dataset')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='the directory to save output binary files')
    parser.add_argument('--num_workers', type=int, default=16)

    args = parser.parse_args()
    return args

def preprocess(data_dir, save_dir):
    # build the dataloader
    test_loader = get_loader(data_dir, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # create directory
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for idx in range(4):
        sub_dir = os.path.join(save_dir, 'input_{}'.format(idx))
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)

    # save images to binary files
    for i, (images, _) in enumerate(tqdm(test_loader)):
        for idx in range(4):
            bin_path = os.path.join(save_dir, 'input_{}'.format(idx), '{}.bin'.format(i))
            img = np.array(images[idx]).astype(np.float32)
            img.tofile(bin_path)
    
    print('Preprocess finished, {} binary files generated'.format(i+1))

if __name__ == "__main__":
    args = parse_args()
    preprocess(args.data_dir, args.save_dir)