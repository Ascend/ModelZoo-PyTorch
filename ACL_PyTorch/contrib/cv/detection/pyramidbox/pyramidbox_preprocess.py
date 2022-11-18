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

from __future__ import absolute_import
from __future__ import division

import sys
import os
import torch
import numpy as np
import cv2
import tqdm

from PIL import Image
from data.config import cfg
from torch.autograd import Variable
from utils.augmentations import to_chw_bgr

use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

def process_image(img, min_side):
    h, w = img.shape[0], img.shape[1]
    #长边缩放为min_side
    scale = max(w, h) / float(min_side)
    new_w, new_h = int(w/scale), int(h/scale)
    resize_img = cv2.resize(img, (new_w, new_h))
    # 填充至min_side * min_side
    bottom = min_side-new_h
    right = min_side-new_w
    img = cv2.copyMakeBorder(resize_img, 0, int(bottom), 0, int(right), cv2.BORDER_CONSTANT, value=[0,0,0]) #从图像边界向上,下,左,右扩的像素数目
    return img
    
def preprocess(file_path, bin_path):
    in_files = os.listdir(file_path)
    if not os.path.exists(bin_path):
        os.makedirs(bin_path)
    for file in tqdm.tqdm(sorted(in_files)):
       os.chdir(os.path.join(file_path, file))
       cur_path = os.getcwd()
       doc = os.listdir(cur_path)   
       for document in sorted(doc):
          in_file = os.path.join(cur_path, document)
          img = Image.open(in_file)
          if img.mode == 'L':
              img = img.convert('RGB')
          img = np.array(img)
          img = process_image(img,1000) #对图片进行放缩加padding
          x = to_chw_bgr(img)
          
          x = x.astype('float32')
          x -= cfg.img_mean
          x = x[[2, 1, 0], :, :]
          x = Variable(torch.from_numpy(x).unsqueeze(0))
          if use_cuda:
              x = x.cuda()
          if not os.path.exists(os.path.join(bin_path,file)):
              os.makedirs(os.path.join(bin_path,file))
          des_path = os.path.join(bin_path,file)
          x.numpy().tofile(os.path.join(des_path,document.split('.')[0] +'.bin'))

def preprocess1(file_path, bin_path):
    in_files = os.listdir(file_path)
    if not os.path.exists(bin_path):
        os.makedirs(bin_path)
    for file in tqdm.tqdm(sorted(in_files)):
       os.chdir(os.path.join(file_path, file))
       cur_path = os.getcwd()
       doc = os.listdir(cur_path)   
       for document in sorted(doc):
          in_file = os.path.join(cur_path, document)
          img = Image.open(in_file)
          if img.mode == 'L':
              img = img.convert('RGB')  
          img = np.array(img)
          img = cv2.flip(img, 1)  
          img = process_image(img,1000)
          x = to_chw_bgr(img)
          x = x.astype('float32')
          x -= cfg.img_mean
          x = x[[2, 1, 0], :, :]
          x = Variable(torch.from_numpy(x).unsqueeze(0))
          if use_cuda:
              x = x.cuda()
          if not os.path.exists(os.path.join(bin_path,file)):
              os.makedirs(os.path.join(bin_path,file))
          des_path = os.path.join(bin_path,file)
          x.numpy().tofile(os.path.join(des_path,document.split('.')[0] +'.bin'))
          
if __name__ == "__main__":
    file_path = os.path.abspath(sys.argv[1])
    bin_path1 = os.path.abspath(sys.argv[2])
    bin_path2 = os.path.abspath(sys.argv[3])
    preprocess(file_path, bin_path1)
    preprocess1(file_path, bin_path2)
