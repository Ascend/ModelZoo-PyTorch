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
'''
Created on May 5, 2018
@author: vermavik
'''
import torch
from torch.autograd import Variable
import os, errno
import numpy as np
from scipy import linalg
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


def ZCA(data, reg=1e-6):
    mean = np.mean(data, axis=0)
    mdata = data - mean
    sigma = np.dot(mdata.T, mdata) / mdata.shape[0]
    U, S, V = linalg.svd(sigma)
    components = np.dot(np.dot(U, np.diag(1 / np.sqrt(S) + reg)), U.T)
    whiten = np.dot(data - mean, components.T)
    return components, mean, whiten


def compute_zca(data_aug, data_target_dir):
    import numpy as np
    from functools import reduce
    from operator import __or__
    from torch.utils.data.sampler import SubsetRandomSampler
      
    if data_aug==1:
            train_transform = transforms.Compose(
                                                 [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=2), transforms.ToTensor()])
    else:
        train_transform = transforms.Compose(
                                             [transforms.ToTensor()])
    
    train_data = datasets.CIFAR10(data_target_dir, train=True, transform=train_transform, download=True)
    import pdb; pdb.set_trace()
    num_classes = 10
    temp_data = train_data.train_data.astype(float)
    temp_data = temp_data.astype(float)
    temp_data[:,:,:,0] = ((temp_data[:,:,:,0] - 125.3))/(63.0)
    temp_data[:,:,:,1] = ((temp_data[:,:,:,1] - 123.0))/(62.1)
    temp_data[:,:,:,2] = ((temp_data[:,:,:,2] - 113.9))/(66.7)
    temp_data = np.transpose(temp_data, (0,3,1,2))
    temp_data = temp_data.reshape(temp_data.shape[0],temp_data.shape[1]*temp_data.shape[2]*temp_data.shape[3])
    components, mean, whiten = ZCA(temp_data)
    np.save('data/cifar10/zca_components', components)
    np.save('data/cifar10/zca_mean', mean)
    
if __name__ == '__main__':
    compute_zca(data_aug=0, data_target_dir="data/cifar10/")
