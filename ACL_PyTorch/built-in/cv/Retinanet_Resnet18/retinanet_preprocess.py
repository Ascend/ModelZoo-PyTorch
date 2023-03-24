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
import mmcv
import torch
from tqdm import tqdm


dataset_config = {
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
    parser = argparse.ArgumentParser(description='preprocess of Retinanet onnx model')
    parser.add_argument("--image_folder_path", default="./data/coco/", help='image of dataset')
    parser.add_argument("--bin_folder_path", default="./coco2017_bin/", help='Preprocessed image buffer')
    flags = parser.parse_args()    

    if not os.path.exists(flags.bin_folder_path):
        os.makedirs(flags.bin_folder_path)
    images = os.listdir(flags.image_folder_path)
    for image_name in tqdm(images):
        if not (image_name.endswith(".jpeg") or image_name.endswith(".JPEG") or image_name.endswith(".jpg")):
            continue
        path_image = os.path.join(flags.image_folder_path, image_name)
        coco_preprocess(path_image, flags.bin_folder_path)
