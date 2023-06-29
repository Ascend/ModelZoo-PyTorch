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

# -*- coding:GB2312 -*- 

from __future__ import absolute_import
from __future__ import division

import sys
import os
import numpy as np
import cv2
import torch
import tqdm

from utils.image import get_affine_transform

def pre_process(img):
    height, width = img.shape[0:2]
    inp_height = inp_width = 800
    c = np.array([width / 2., height / 2.], dtype=np.float32)
    s = max(height, width) * 1.0
    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    resized_image = cv2.resize(img, (width, height))
    inp_image = cv2.warpAffine(
        resized_image, trans_input, (inp_width, inp_height),
        flags=cv2.INTER_LINEAR)
    images = torch.from_numpy(inp_image)
    return images

def preprocess(file_path, bin_path):
    in_files = os.listdir(file_path)
    
    if not os.path.exists(bin_path):
        os.makedirs(bin_path)
    for file in tqdm.tqdm(sorted(in_files)):
       os.chdir(os.path.join(file_path, file))
       cur_path = os.getcwd()
       doc = os.listdir(cur_path)   
       for document in doc:
          if document=='output':
              break
          image = cv2.imread(os.path.join(cur_path, document))
          images = pre_process(image)
          images.numpy().tofile(os.path.join(bin_path,document.split('.')[0] +'.bin'))
        
if __name__ == "__main__":
    fp = os.path.abspath(sys.argv[1])
    bp = os.path.abspath(sys.argv[2])
    preprocess(fp, bp)
