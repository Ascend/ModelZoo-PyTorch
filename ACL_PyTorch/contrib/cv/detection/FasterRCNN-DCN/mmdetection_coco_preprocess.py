# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================

import numpy as np
import os
import cv2
import argparse
import mmcv
import torch

dataset_config = {
        'resize': (1216, 1216),
        'mean': [123.675, 116.28, 103.53],
        'std': [58.395, 57.12, 57.375],
}

tensor_height = 1216
tensor_width = 1216


def coco_preprocess(input_image, output_bin_path):
    #define the output file name 
    img_name = input_image.split('/')[-1]
    bin_name = img_name.split('.')[0] + ".bin"
    bin_fl = os.path.join(output_bin_path, bin_name)

    one_img = mmcv.imread(os.path.join(input_image), backend='cv2')
    one_img = mmcv.imrescale(one_img, (tensor_height, tensor_width))
    # calculate padding
    h = one_img.shape[0]
    w = one_img.shape[1]
    pad_left = (tensor_width - w) // 2
    pad_top = (tensor_height - h) // 2
    pad_right = tensor_width - pad_left - w
    pad_bottom = tensor_height - pad_top - h

    mean = np.array(dataset_config['mean'], dtype=np.float32)
    std = np.array(dataset_config['std'], dtype=np.float32)
    one_img = mmcv.imnormalize(one_img, mean, std)
    one_img = mmcv.impad(one_img, padding=(pad_left, pad_top, pad_right, pad_bottom), pad_val=0)
    one_img = one_img.transpose(2, 0, 1)
    one_img.tofile(bin_fl)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess of FasterRCNN pytorch model')
    parser.add_argument("--image_folder_path", default="./coco2014/", help='image of dataset')
    parser.add_argument("--bin_folder_path", default="./coco2014_bin/", help='Preprocessed image buffer')
    flags = parser.parse_args()    

    if not os.path.exists(flags.bin_folder_path):
        os.makedirs(flags.bin_folder_path)
    images = os.listdir(flags.image_folder_path)
    for image_name in images:
        if not (image_name.endswith(".jpeg") or image_name.endswith(".JPEG") or image_name.endswith(".jpg")):
            continue
        print("start to process image {}....".format(image_name))
        path_image = os.path.join(flags.image_folder_path, image_name)
        coco_preprocess(path_image, flags.bin_folder_path)
