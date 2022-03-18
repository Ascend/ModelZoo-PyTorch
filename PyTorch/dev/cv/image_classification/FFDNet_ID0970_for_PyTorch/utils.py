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
import numpy as np
import cv2
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
#from skimage.measure.simple_metrics import compare_psnr
from skimage.metrics.simple_metrics import peak_signal_noise_ratio as compare_psnr
from skimage.util import random_noise

def is_image_gray(image):
    """
    :param image: cv2
    """
    # a[..., 0] == a.T[0].T
    return not(len(image.shape) == 3 and not(np.allclose(image[...,0], image[...,1]) and np.allclose(image[...,2], image[...,1])))

def downsample(x):
    """
    :param x: (C, H, W)
    :param noise_sigma: (C, H/2, W/2)
    :return: (4, C, H/2, W/2)
    """
    # x = x[:, :, :x.shape[2] // 2 * 2, :x.shape[3] // 2 * 2]
    N, C, W, H = x.size()
    idxL = [[0, 0], [0, 1], [1, 0], [1, 1]]

    Cout = 4 * C
    Wout = W // 2
    Hout = H // 2

    #if 'cuda' in x.type():
    if 'npu' in x.type():
        #down_features = torch.cuda.FloatTensor(N, Cout, Wout, Hout).fill_(0)
        down_features = torch.npu.FloatTensor(N, Cout, Wout, Hout).fill_(0)
    else:
        down_features = torch.FloatTensor(N, Cout, Wout, Hout).fill_(0)
    
    for idx in range(4):
        down_features[:, idx:Cout:4, :, :] = x[:, :, idxL[idx][0]::2, idxL[idx][1]::2]

    return down_features

def upsample(x):
    """
    :param x: (n, C, W, H)
    :return: (n, C/4, W*2, H*2)
    """
    N, Cin, Win, Hin = x.size()
    idxL = [[0, 0], [0, 1], [1, 0], [1, 1]]
    
    Cout = Cin // 4
    Wout = Win * 2
    Hout = Hin * 2

    up_feature = torch.zeros((N, Cout, Wout, Hout)).type(x.type())
    for idx in range(4):
        up_feature[:, :, idxL[idx][0]::2, idxL[idx][1]::2] = x[:, idx:Cin:4, :, :]

    return up_feature

def normalize(data):
    """
    // variable_to_cv2_image will reshape to *255
    """
    return np.float32(data / 255)

def image_to_patches(image, patch_size):
    """
    :param image: Image (C * W * H) Numpy
    :param patch_size: int
    :return: (patch_num, C, win, win)
    """
    W = image.shape[1]
    H = image.shape[2]
    if W < patch_size or H < patch_size:
        return []

    ret = []
    for ws in range(0, W // patch_size):
        for hs in range(0, H // patch_size):
            patch = image[:, ws * patch_size : (ws + 1) * patch_size, hs * patch_size : (hs + 1) * patch_size]
            ret.append(patch)
    return np.array(ret, dtype=np.float32)

def add_batch_noise(images, noise_sigma):
    """
    :param images: Image (n, C, W, H) Tensor
    :return: Image (n, C, W, H)
    """
    images = random_noise(images.numpy(), mode='gaussian', var=noise_sigma)
    return torch.FloatTensor(images)

def batch_psnr(img, imclean, data_range):
    """
    add the whole batch's PSNR
    """
    img_cpu = img.data.cpu().numpy().astype(np.float32)
    imgclean = imclean.data.cpu().numpy().astype(np.float32)
    psnr = 0
    for i in range(img_cpu.shape[0]):
        psnr += compare_psnr(imgclean[i, :, :, :], img_cpu[i, :, :, :], data_range=data_range)
    return psnr / img_cpu.shape[0]

def variable_to_cv2_image(varim):
    """
    Norm Variable -> Cv2
    """
    nchannels = varim.size()[1]
    if nchannels == 1:
        res = (varim.data.cpu().numpy()[0, 0, :] * 255.).clip(0, 255).astype(np.uint8)
    elif nchannels == 3:
        res = varim.data.cpu().numpy()[0]
        res = cv2.cvtColor(res.transpose(2, 1, 0), cv2.COLOR_RGB2BGR)
        res = (res*255.).clip(0, 255).astype(np.uint8)
    else:
        raise Exception('Number of color channels not supported')
    return res

def weights_init_kaiming(lyr):
    """
    Initializes weights of the model according to the "He" initialization
    method described in "Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification" - He, K. et al. (2015), using a
    normal distribution.
    This function is to be called by the torch.nn.Module.apply() method,
    which applies weights_init_kaiming() to every layer of the model.
    """
    classname = lyr.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(lyr.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(lyr.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        lyr.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).\
            clamp_(-0.025, 0.025)
        nn.init.constant_(lyr.bias.data, 0.0)