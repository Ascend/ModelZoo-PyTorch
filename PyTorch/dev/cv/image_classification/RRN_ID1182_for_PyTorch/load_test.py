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
from Gaussian_downsample import gaussian_downsample
from bicubic import imresize
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

def modcrop(img,scale):
    (iw, ih) = img.size
    ih = ih - (ih % scale)
    iw = iw - (iw % scale)
    img = img.crop((0,0,iw,ih))
    return img

class DataloadFromFolderTest(data.Dataset): # load test dataset
    def __init__(self, image_dir, scale, scene_name, transform):
        super(DataloadFromFolderTest, self).__init__()
        alist = os.listdir(os.path.join(image_dir, scene_name))
        alist.sort()
        self.image_filenames = [os.path.join(image_dir, scene_name, x) for x in alist] 
        self.L = len(alist)
        self.scale = scale
        self.transform = transform # To_tensor
    def __getitem__(self, index):
        target = []
        for i in range(self.L):
            GT_temp = modcrop(Image.open(self.image_filenames[i]).convert('RGB'), self.scale)
            target.append(GT_temp)
        target = [np.asarray(HR) for HR in target] 
        target = np.asarray(target)
        if self.scale == 4:
            target = np.lib.pad(target, pad_width=((0,0), (2*self.scale,2*self.scale), (2*self.scale,2*self.scale), (0,0)), mode='reflect')
        t, h, w, c = target.shape
        target = target.transpose(1,2,3,0).reshape(h,w,-1) # numpy, [H',W',CT']
        if self.transform:
            target = self.transform(target) # Tensor, [CT',H',W']
        target = target.view(c,t,h,w)
        LR = gaussian_downsample(target, self.scale) # [c,t,h,w]
        LR = torch.cat((LR[:,1:2,:,:], LR), dim=1)
        return LR, target
        
    def __len__(self):
        return 1 

