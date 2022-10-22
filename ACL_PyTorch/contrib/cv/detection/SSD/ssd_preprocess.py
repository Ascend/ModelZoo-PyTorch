# Copyright 2020 Huawei Technologies Co., Ltd
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
"""coco preprocess"""

import numpy as np
import os
import cv2
import argparse
import mmcv
import torch
from tqdm import tqdm

dataset_config = {
    'resize': (300, 300),
    'mean': [123.675, 116.28, 103.53],
    'std': [1, 1, 1],
}

tensor_height = 300
tensor_width = 300


def coco_preprocess(input_image, output_bin_path):
    """coco_preprocess"""
    # define the output file name
    img_name = input_image.split('/')[-1]
    bin_name = img_name.split('.')[0] + ".bin"
    bin_fl = os.path.join(output_bin_path, bin_name)

    one_img = mmcv.imread(input_image, backend='cv2')
    one_img = mmcv.imresize(one_img, (tensor_height, tensor_width))
    # calculate padding
    mean = np.array(dataset_config['mean'], dtype=np.float32)
    std = np.array(dataset_config['std'], dtype=np.float32)
    one_img = mmcv.imnormalize(one_img, mean, std)
    one_img = one_img.transpose(2, 0, 1)
    one_img.tofile(bin_fl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='preprocess of FasterRCNN pytorch model')
    parser.add_argument("--image_folder_path",
                        default="./coco2014/", help='image of dataset')
    parser.add_argument(
        "--bin_folder_path", default="./coco2014_bin/", help='Preprocessed image buffer')
    flags = parser.parse_args()

    if not os.path.exists(flags.bin_folder_path):
        os.makedirs(flags.bin_folder_path)
    images = os.listdir(flags.image_folder_path)
    for image_name in tqdm(images, desc="Starting to process image..."):
        if not (image_name.endswith(".jpeg") or image_name.endswith(".JPEG") or image_name.endswith(".jpg")):
            continue
        path_image = os.path.join(flags.image_folder_path, image_name)
        coco_preprocess(path_image, flags.bin_folder_path)
