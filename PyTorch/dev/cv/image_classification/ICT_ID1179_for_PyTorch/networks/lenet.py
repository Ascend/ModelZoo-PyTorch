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
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys, os
import random
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import mixup_data


## code for CNN13 from https://github.com/benathi/fastswa-semi-sup/blob/master/mean_teacher/architectures.py
from torch.nn.utils import weight_norm
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

class CNN13(nn.Module):
       
    def __init__(self, num_classes=10, dropout=0.5):
        super(CNN13, self).__init__()

        #self.gn = GaussianNoise(0.15)
        self.activation = nn.LeakyReLU(0.1)
        self.conv1a = weight_norm(nn.Conv2d(3, 128, 3, padding=1))
        self.bn1a = nn.BatchNorm2d(128)
        self.conv1b = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.bn1b = nn.BatchNorm2d(128)
        self.conv1c = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.bn1c = nn.BatchNorm2d(128)
        self.mp1 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop1  = nn.Dropout(dropout)

        self.conv2a = weight_norm(nn.Conv2d(128, 256, 3, padding=1))
        self.bn2a = nn.BatchNorm2d(256)
        self.conv2b = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2b = nn.BatchNorm2d(256)
        self.conv2c = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2c = nn.BatchNorm2d(256)
        self.mp2 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop2  = nn.Dropout(dropout)
        
        self.conv3a = weight_norm(nn.Conv2d(256, 512, 3, padding=0))
        self.bn3a = nn.BatchNorm2d(512)
        self.conv3b = weight_norm(nn.Conv2d(512, 256, 1, padding=0))
        self.bn3b = nn.BatchNorm2d(256)
        self.conv3c = weight_norm(nn.Conv2d(256, 128, 1, padding=0))
        self.bn3c = nn.BatchNorm2d(128)
        self.ap3 = nn.AvgPool2d(6, stride=2, padding=0)
        
        self.fc1 =  weight_norm(nn.Linear(128, num_classes))
        
    def forward(self, x, target=None, mixup_hidden = False,  mixup_alpha = 0.1, layers_mix=None):
        if mixup_hidden == True:
            layer_mix = random.randint(0,layers_mix)
        
            out = x
            if layer_mix == 0:
                    out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)
                
            
            out = self.conv1a(out)
            out = self.bn1a(out)
            out = self.activation(out)
            out = self.conv1b(out)
            out = self.bn1b(out)
            out = self.activation(out)
            out = self.conv1c(out)
            out = self.bn1c(out)
            if layer_mix == 1:
                    out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)
            out = self.activation(out)
            out = self.mp1(out)
            out = self.drop1(out)
            out = self.conv2a(out)
            out = self.bn2a(out)
            out = self.activation(out)
            out = self.conv2b(out)
            out = self.bn2b(out)
            out = self.activation(out)
            out = self.conv2c(out)
            out = self.bn2c(out)
            if layer_mix == 2:
                    out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)
            out = self.activation(out)
            out = self.mp2(out)
            out = self.drop2(out)
            out = self.conv3a(out)
            out = self.bn3a(out)
            out = self.activation(out)
            out = self.conv3b(out)
            out = self.bn3b(out)
            out = self.activation(out)
            out = self.conv3c(out)
            out = self.bn3c(out)
            if layer_mix == 3:
                    out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)
            out = self.activation(out)
            out = self.ap3(out)
            out = out.view(-1, 128)
            out = self.fc1(out)
            
            lam = torch.tensor(lam).npu()
            lam = lam.repeat(y_a.size())
            return out, y_a, y_b, lam
        
        else:
            out = x
            ## layer 1-a###
            out = self.conv1a(out)
            out = self.bn1a(out)
            out = self.activation(out)
            
            ## layer 1-b###
            out = self.conv1b(out)
            out = self.bn1b(out)
            out = self.activation(out)
            
            ## layer 1-c###
            out = self.conv1c(out)
            out = self.bn1c(out)
            out = self.activation(out)
            
            out = self.mp1(out)
            out = self.drop1(out)
            
            
            ## layer 2-a###
            out = self.conv2a(out)
            out = self.bn2a(out)
            out = self.activation(out)
            
            ## layer 2-b###
            out = self.conv2b(out)
            out = self.bn2b(out)
            out = self.activation(out)
            
            ## layer 2-c###
            out = self.conv2c(out)
            out = self.bn2c(out)
            out = self.activation(out)
            
            
            out = self.mp2(out)
            out = self.drop2(out)
            
            
            ## layer 3-a###
            out = self.conv3a(out)
            out = self.bn3a(out)
            out = self.activation(out)
            
            ## layer 3-b###
            out = self.conv3b(out)
            out = self.bn3b(out)
            out = self.activation(out)
            
            ## layer 3-c###
            out = self.conv3c(out)
            out = self.bn3c(out)
            out = self.activation(out)
            
            out = self.ap3(out)
     
            out = out.view(-1, 128)
            out = self.fc1(out)
            return out
            
        

def cnn13(num_classes=10, dropout = 0.0):
    model = CNN13(num_classes = num_classes, dropout=dropout)
    return model

