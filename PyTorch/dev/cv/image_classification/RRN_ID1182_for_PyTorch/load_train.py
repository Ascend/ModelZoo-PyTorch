#
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
#
import os
import torch.utils.data as data
import torch
import numpy as np
from PIL import Image, ImageOps
import random
from bicubic import imresize
from Gaussian_downsample import gaussian_downsample
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

def load_img(image_path, scale):
    HR = []
    HR = []
    for img_num in range(7):
        GT_temp = modcrop(Image.open(os.path.join(image_path,'im{}.png'.format(img_num+1))).convert('RGB'), scale)
        HR.append(GT_temp)
    return HR

def modcrop(img,scale):
    (ih, iw) = img.size
    ih = ih - ( ih % scale)
    iw = iw - ( iw % scale)
    img = img.crop((0,0,ih,iw))
    return img

def train_process(GH, flip_h=True, rot=True, converse=True): 
    if random.random() < 0.5 and flip_h: 
        GH = [ImageOps.flip(LR) for LR in GH]
    if rot:
        if random.random() < 0.5:
            GH = [ImageOps.mirror(LR) for LR in GH]
    return GH

class DataloadFromFolder(data.Dataset): # load train dataset
    def __init__(self, image_dir, scale, data_augmentation, file_list, transform):
        super(DataloadFromFolder, self).__init__()
        alist = [line.rstrip() for line in open(os.path.join(image_dir,file_list))] 
        self.image_filenames = [os.path.join(image_dir,x) for x in alist] 
        self.scale = scale
        self.transform = transform # To_tensor
        self.data_augmentation = data_augmentation # flip and rotate
    def __getitem__(self, index):
        GT = load_img(self.image_filenames[index], self.scale) 
        GT = train_process(GT) # input: list (contain PIL), target: PIL
        GT = [np.asarray(HR) for HR in GT]  # PIL -> numpy # input: list (contatin numpy: [H,W,C])
        GT = np.asarray(GT) # numpy, [T,H,W,C]
        T,H,W,C = GT.shape
        if self.scale == 4:
            GT = np.lib.pad(GT, pad_width=((0,0),(2*self.scale,2*self.scale),(2*self.scale,2*self.scale),(0,0)), mode='reflect')
        t, h, w, c = GT.shape
        GT = GT.transpose(1,2,3,0).reshape(h, w, -1) # numpy, [H',W',CT]
        if self.transform:
            GT = self.transform(GT) # Tensor, [CT',H',W']
        GT = GT.view(c,t,h,w) # Tensor, [C,T,H,W]
        LR = gaussian_downsample(GT, self.scale)
        LR = torch.cat((LR[:,1:2,:,:], LR), dim=1)
        return LR, GT

    def __len__(self):
        return len(self.image_filenames) 

