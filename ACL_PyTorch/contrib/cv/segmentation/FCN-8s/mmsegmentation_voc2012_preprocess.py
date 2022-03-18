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
import os
import cv2
import argparse
import mmcv
import torch


dataset_config = {
    'mean': (123.675, 116.28, 103.53),
    'std': (58.395, 57.12, 57.375)
}


tensor_height = 500
tensor_width = 500


def resize(img, size):
    old_h = img.shape[0]
    old_w = img.shape[1]
    scale_ratio = min(size[0] / old_w, size[1] / old_h)
    new_w = int(np.floor(old_w * scale_ratio))
    new_h = int(np.floor(old_h * scale_ratio))
    resized_img = mmcv.imresize(img, (new_w, new_h), backend='cv2')
    return resized_img


def voc2012_preprocess(input_image, output_bin_path):
    img_name = input_image.split('/')[-1]
    bin_name = img_name.split('.')[0] + ".bin"
    bin_fl = os.path.join(output_bin_path, bin_name)

    one_img = mmcv.imread(os.path.join(input_image), backend='cv2')
    one_img = resize(one_img, (tensor_width, tensor_height))

    mean = np.array(dataset_config['mean'], dtype=np.float32)
    std = np.array(dataset_config['std'], dtype=np.float32)
    one_img = mmcv.imnormalize(one_img, mean, std)

    h = one_img.shape[0]
    w = one_img.shape[1]
    pad_left = (tensor_width - w) // 2
    pad_top = (tensor_height - h) // 2
    pad_right = tensor_width - pad_left - w
    pad_bottom = tensor_height - pad_top - h
    one_img = mmcv.impad(one_img, padding=(pad_left, pad_top, pad_right, pad_bottom), pad_val=0)

    one_img = one_img.transpose(2, 0, 1)
    one_img.tofile(bin_fl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess of FCN-8s pytorch model')
    parser.add_argument("--image_folder_path", default="/opt/npu/VOCdevkit/VOC2012/JPEGImages/", 
    help='image of dataset')
    parser.add_argument("--split", default="/opt/npu/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt")
    parser.add_argument("--bin_folder_path", default="./voc12_bin/", help='Preprocessed image buffer')
    flags = parser.parse_args()

    if not os.path.exists(flags.bin_folder_path):
        os.makedirs(flags.bin_folder_path)

    split = flags.split
    img_suffix = '.jpg'
    img_infos = []
    if split is not None:
        with open(split) as f:
            for line in f:
                img_name = line.strip()
                img_info = img_name + img_suffix
                img_infos.append(img_info)

    images = os.listdir(flags.image_folder_path)
    for image_name in images:

        if not (image_name.endswith(".jpeg") or image_name.endswith(".JPEG") or image_name.endswith(".jpg") and image_name in img_infos):
            continue
        print("start to process image {}....".format(image_name))
        path_image = os.path.join(flags.image_folder_path, image_name)
        voc2012_preprocess(path_image, flags.bin_folder_path)
