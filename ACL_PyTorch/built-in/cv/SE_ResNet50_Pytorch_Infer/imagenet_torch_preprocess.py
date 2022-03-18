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

# coding=UTF-8

import os
import sys
import cv2
import numpy as np


def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img

def preprocess(file_path, bin_path):
    in_files = os.listdir(file_path)
    if not os.path.exists(bin_path):
        os.makedirs(bin_path)
    i = 0

    resize_size = 256
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    for file in in_files:
        i = i + 1
        print(file, "===", i)
 
        img = cv2.imread(os.path.join(file_path, file))     
        b, g, r = cv2.split(img)
        
        img = cv2.merge([r, g, b])
        img = cv2.resize(img, (resize_size, resize_size), interpolation=cv2.INTER_CUBIC)      
        img = np.array(img, dtype=np.int8)
        
        img.tofile(os.path.join(bin_path, file.split('.')[0] + '.bin'))

if __name__ == "__main__":
    file_path = os.path.abspath(sys.argv[1])
    bin_path = os.path.abspath(sys.argv[2])
    preprocess(file_path, bin_path)
