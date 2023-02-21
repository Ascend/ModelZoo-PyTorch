# Copyright 2023 Huawei Technologies Co., Ltd
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
import numpy as np
import cv2
from tqdm import tqdm


# rcf preprocess
def preprocess(args):
    """[rcf preprocess]

    Args:
        args ([argparse]): [rcf preprocess parameters]
    """
    src_dir = args.src_dir
    in_files = []

    image_names = os.listdir(src_dir)
    for image_name in image_names:
        if image_name.endswith(('jpg', 'png', 'jpeg', 'bmp')):
            comp_name = os.path.join(src_dir, image_name)
            in_files.append(comp_name)
    
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    i = 0
    for file in tqdm(in_files):
        if not os.path.isdir(file):
            i = i + 1
            image = cv2.imread(file).astype(np.float32)
            h, w, c = image.shape
            image -= np.array((104.00698793, 116.66876762, 122.67891434))
            image = np.transpose(image, (2, 0, 1))
            image = image[np.newaxis, :]
            temp_name = file[file.rfind('/') + 1:]
            np.save(os.path.join(save_path, temp_name.split('.')[0] + ".npy"), image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing of RCF model')
    parser.add_argument('--src_dir', default='BSR/BSDS500/data/images/test', type=str, 
    help='The file records the pictures that need to be preprocessed')
    parser.add_argument('--save_path', default='images_npy', type=str, help='Output path, If not exist, create it')
    args = parser.parse_args()
    preprocess(args)
