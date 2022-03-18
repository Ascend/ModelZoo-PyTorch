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
"""
Created on Sun Mar 7 2021

@author: Kuan-Lin Chen
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

class BasicBlock(nn.Module):
    def __init__(self, in_planes, mid_planes, out_planes, stride=1, bias=False, bn=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(mid_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=bias)
        if bn is True:
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.bn2 = nn.BatchNorm2d(mid_planes)
        if stride != 1 or in_planes != out_planes:
            self.projection = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)
            )

    def forward(self, x):
        y = self.bn1(x) if hasattr(self,'bn1') else x
        y = F.relu(y)
        shortcut = self.projection(y) if hasattr(self, 'projection') else x
        y = self.conv1(y)
        y = self.bn2(y) if hasattr(self,'bn2') else y
        v = F.relu(y)
        out = self.conv2(v) + shortcut if hasattr(self,'conv2') else 0
        return out,v

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_out_channels, num_mid_channels, num_classes, bias, bn):
        super(ResNet, self).__init__()
        assert(len(num_blocks)==len(num_out_channels)), "size does not match between num_blocks and num_out_channels"
        assert(len(num_blocks)==len(num_mid_channels)), "size does not match between num_blocks and num_mid_channels"
        self.bias = bias
        self.bn = bn
        self.in_planes = num_out_channels[0]
        self.num_blocks = num_blocks
        self.expansion = nn.Conv2d(3, num_out_channels[0], kernel_size=3, stride=1, padding=1, bias=bias)
        self.stage = nn.ModuleList()
        self.stage.append(self._creat_block_seq(block, num_mid_channels[0], num_out_channels[0], num_blocks[0], stride=1))
        for j in range(1,len(num_blocks)):
            self.stage.append(self._creat_block_seq(block, num_mid_channels[j], num_out_channels[j], num_blocks[j], stride=2))
        self.final_bn = nn.BatchNorm2d(num_out_channels[-1])
        self.linear = nn.Linear(num_out_channels[-1], num_classes)

    def _creat_block_seq(self, block, mid_planes, out_planes, num_blocks, stride):
        stride_seq = [stride] + [1]*(num_blocks-1)
        block_seq = nn.ModuleList()
        for stride in stride_seq:
            block_seq.append(block(self.in_planes, mid_planes, out_planes, stride, self.bias, self.bn))
            self.in_planes = out_planes
        return block_seq

    def forward(self, x):
        out = self.expansion(x)
        for j in range(len(self.num_blocks)):
            for i in range(self.num_blocks[j]):
                out,_ = self.stage[j][i](out)
        out = self.final_bn(out)
        out = F.relu(out)
        out = F.avg_pool2d(out, out.size(2))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class BNResNEst(ResNet):
    def __init__(self, block, num_blocks, num_out_channels, num_mid_channels, num_classes, bias, bn):
        super().__init__(block, num_blocks, num_out_channels, num_mid_channels, num_classes, bias, bn)

    def forward(self, x):
        out = self.expansion(x)
        for j in range(len(self.num_blocks)):
            for i in range(self.num_blocks[j]):
                out,_ = self.stage[j][i](out)
        out = self.final_bn(out)
        out = F.avg_pool2d(out, out.size(2))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResNEst(ResNet):
    def __init__(self, block, num_blocks, num_out_channels, num_mid_channels, num_classes, bias, bn):
        super().__init__(block, num_blocks, num_out_channels, num_mid_channels, num_classes, bias, bn)
        delattr(self,'final_bn')

    def forward(self, x):
        out = self.expansion(x)
        for j in range(len(self.num_blocks)):
            for i in range(self.num_blocks[j]):
                out,_ = self.stage[j][i](out)
        out = F.avg_pool2d(out, out.size(2))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class AResNEst(ResNet):
    def __init__(self, block, num_blocks, num_out_channels, num_mid_channels, num_classes, bias, bn):
        super().__init__(block, num_blocks, num_out_channels, num_mid_channels, num_classes, bias, bn)
        delattr(self.stage[-1][-1],'conv2')
        delattr(self,'final_bn')
        delattr(self,'linear')
        self.H_0 = nn.Linear(3, num_classes)
        self.H_k = nn.ModuleList() # k = 1,2,...,L
        for j in range(len(num_blocks)):
            self.H_k.append(nn.ModuleList())
            for i in range(num_blocks[j]):
                self.H_k[j].append(nn.Linear(num_mid_channels[j], num_classes))

    def forward(self, x):
        r = self.expansion(x)
        z = F.avg_pool2d(x,x.size(2))
        z = z.view(z.size(0),-1)
        out = self.H_0(z)
        for j in range(len(self.num_blocks)):
            for i in range(self.num_blocks[j]):
                r,v = self.stage[j][i](r)
                z = F.avg_pool2d(v,v.size(2))
                z = z.view(z.size(0),-1)
                out += self.H_k[j][i](z)
        return out

# dataset_model_architecture
def CIFAR10_Standard_WRN_16_8(): return ResNet(BasicBlock,[2,2,2],[128,256,512],[128,256,512],10,False,True)
def CIFAR10_Standard_WRN_40_4(): return ResNet(BasicBlock,[6,6,6],[64,128,256],[64,128,256],10,False,True)
def CIFAR10_Standard_ResNet_110(): return ResNet(BasicBlock,[18,18,18],[16,32,64],[16,32,64],10,False,True)
def CIFAR10_Standard_ResNet_20(): return ResNet(BasicBlock,[3,3,3],[16,32,64],[16,32,64],10,False,True)

def CIFAR10_ResNEst_WRN_16_8(): return ResNEst(BasicBlock,[2,2,2],[128,256,512],[128,256,512],10,False,True)
def CIFAR10_ResNEst_WRN_40_4(): return ResNEst(BasicBlock,[6,6,6],[64,128,256],[64,128,256],10,False,True)
def CIFAR10_ResNEst_ResNet_110(): return ResNEst(BasicBlock,[18,18,18],[16,32,64],[16,32,64],10,False,True)
def CIFAR10_ResNEst_ResNet_20(): return ResNEst(BasicBlock,[3,3,3],[16,32,64],[16,32,64],10,False,True)

def CIFAR10_BNResNEst_WRN_16_8(): return BNResNEst(BasicBlock,[2,2,2],[128,256,512],[128,256,512],10,False,True)
def CIFAR10_BNResNEst_WRN_40_4(): return BNResNEst(BasicBlock,[6,6,6],[64,128,256],[64,128,256],10,False,True)
def CIFAR10_BNResNEst_ResNet_110(): return BNResNEst(BasicBlock,[18,18,18],[16,32,64],[16,32,64],10,False,True)
def CIFAR10_BNResNEst_ResNet_20(): return BNResNEst(BasicBlock,[3,3,3],[16,32,64],[16,32,64],10,False,True)

def CIFAR10_AResNEst_WRN_16_8(): return AResNEst(BasicBlock,[2,2,2],[128,256,512],[128,256,512],10,False,True)
def CIFAR10_AResNEst_WRN_40_4(): return AResNEst(BasicBlock,[6,6,6],[64,128,256],[64,128,256],10,False,True)
def CIFAR10_AResNEst_ResNet_110(): return AResNEst(BasicBlock,[18,18,18],[16,32,64],[16,32,64],10,False,True)
def CIFAR10_AResNEst_ResNet_20(): return AResNEst(BasicBlock,[3,3,3],[16,32,64],[16,32,64],10,False,True)

def CIFAR100_Standard_WRN_16_8(): return ResNet(BasicBlock,[2,2,2],[128,256,512],[128,256,512],100,False,True)
def CIFAR100_Standard_WRN_40_4(): return ResNet(BasicBlock,[6,6,6],[64,128,256],[64,128,256],100,False,True)
def CIFAR100_Standard_ResNet_110(): return ResNet(BasicBlock,[18,18,18],[16,32,64],[16,32,64],100,False,True)
def CIFAR100_Standard_ResNet_20(): return ResNet(BasicBlock,[3,3,3],[16,32,64],[16,32,64],100,False,True)

def CIFAR100_ResNEst_WRN_16_8(): return ResNEst(BasicBlock,[2,2,2],[128,256,512],[128,256,512],100,False,True)
def CIFAR100_ResNEst_WRN_40_4(): return ResNEst(BasicBlock,[6,6,6],[64,128,256],[64,128,256],100,False,True)
def CIFAR100_ResNEst_ResNet_110(): return ResNEst(BasicBlock,[18,18,18],[16,32,64],[16,32,64],100,False,True)
def CIFAR100_ResNEst_ResNet_20(): return ResNEst(BasicBlock,[3,3,3],[16,32,64],[16,32,64],100,False,True)

def CIFAR100_BNResNEst_WRN_16_8(): return BNResNEst(BasicBlock,[2,2,2],[128,256,512],[128,256,512],100,False,True)
def CIFAR100_BNResNEst_WRN_40_4(): return BNResNEst(BasicBlock,[6,6,6],[64,128,256],[64,128,256],100,False,True)
def CIFAR100_BNResNEst_ResNet_110(): return BNResNEst(BasicBlock,[18,18,18],[16,32,64],[16,32,64],100,False,True)
def CIFAR100_BNResNEst_ResNet_20(): return BNResNEst(BasicBlock,[3,3,3],[16,32,64],[16,32,64],100,False,True)

def CIFAR100_AResNEst_WRN_16_8(): return AResNEst(BasicBlock,[2,2,2],[128,256,512],[128,256,512],100,False,True)
def CIFAR100_AResNEst_WRN_40_4(): return AResNEst(BasicBlock,[6,6,6],[64,128,256],[64,128,256],100,False,True)
def CIFAR100_AResNEst_ResNet_110(): return AResNEst(BasicBlock,[18,18,18],[16,32,64],[16,32,64],100,False,True)
def CIFAR100_AResNEst_ResNet_20(): return AResNEst(BasicBlock,[3,3,3],[16,32,64],[16,32,64],100,False,True)

model_dict = {'CIFAR10_Standard_WRN_16_8':CIFAR10_Standard_WRN_16_8,'CIFAR10_Standard_WRN_40_4':CIFAR10_Standard_WRN_40_4,
                'CIFAR10_Standard_ResNet_110':CIFAR10_Standard_ResNet_110,'CIFAR10_Standard_ResNet_20':CIFAR10_Standard_ResNet_20,
                'CIFAR10_ResNEst_WRN_16_8':CIFAR10_ResNEst_WRN_16_8,'CIFAR10_ResNEst_WRN_40_4':CIFAR10_ResNEst_WRN_40_4,
                'CIFAR10_ResNEst_ResNet_110':CIFAR10_ResNEst_ResNet_110,'CIFAR10_ResNEst_ResNet_20':CIFAR10_ResNEst_ResNet_20,
                'CIFAR10_BNResNEst_WRN_16_8':CIFAR10_BNResNEst_WRN_16_8,'CIFAR10_BNResNEst_WRN_40_4':CIFAR10_BNResNEst_WRN_40_4,
                'CIFAR10_BNResNEst_ResNet_110':CIFAR10_BNResNEst_ResNet_110,'CIFAR10_BNResNEst_ResNet_20':CIFAR10_BNResNEst_ResNet_20,
                'CIFAR10_AResNEst_WRN_16_8':CIFAR10_AResNEst_WRN_16_8,'CIFAR10_AResNEst_WRN_40_4':CIFAR10_AResNEst_WRN_40_4,
                'CIFAR10_AResNEst_ResNet_110':CIFAR10_AResNEst_ResNet_110,'CIFAR10_AResNEst_ResNet_20':CIFAR10_AResNEst_ResNet_20,
                'CIFAR100_Standard_WRN_16_8':CIFAR100_Standard_WRN_16_8,'CIFAR100_Standard_WRN_40_4':CIFAR100_Standard_WRN_40_4,
                'CIFAR100_Standard_ResNet_110':CIFAR100_Standard_ResNet_110,'CIFAR100_Standard_ResNet_20':CIFAR100_Standard_ResNet_20,
                'CIFAR100_ResNEst_WRN_16_8':CIFAR100_ResNEst_WRN_16_8,'CIFAR100_ResNEst_WRN_40_4':CIFAR100_ResNEst_WRN_40_4,
                'CIFAR100_ResNEst_ResNet_110':CIFAR100_ResNEst_ResNet_110,'CIFAR100_ResNEst_ResNet_20':CIFAR100_ResNEst_ResNet_20,
                'CIFAR100_BNResNEst_WRN_16_8':CIFAR100_BNResNEst_WRN_16_8,'CIFAR100_BNResNEst_WRN_40_4':CIFAR100_BNResNEst_WRN_40_4,
                'CIFAR100_BNResNEst_ResNet_110':CIFAR100_BNResNEst_ResNet_110,'CIFAR100_BNResNEst_ResNet_20':CIFAR100_BNResNEst_ResNet_20,
                'CIFAR100_AResNEst_WRN_16_8':CIFAR100_AResNEst_WRN_16_8,'CIFAR100_AResNEst_WRN_40_4':CIFAR100_AResNEst_WRN_40_4,
                'CIFAR100_AResNEst_ResNet_110':CIFAR100_AResNEst_ResNet_110,'CIFAR100_AResNEst_ResNet_20':CIFAR100_AResNEst_ResNet_20}

"""
model_choices = ['CIFAR10_Standard_WRN_16_8','CIFAR10_Standard_WRN_40_4','CIFAR10_Standard_ResNet_110','CIFAR10_Standard_ResNet_20',
                'CIFAR10_ResNEst_WRN_16_8','CIFAR10_ResNEst_WRN_40_4','CIFAR10_ResNEst_ResNet_110','CIFAR10_ResNEst_ResNet_20',
                'CIFAR10_BNResNEst_WRN_16_8','CIFAR10_BNResNEst_WRN_40_4','CIFAR10_BNResNEst_ResNet_110','CIFAR10_BNResNEst_ResNet_20',
                'CIFAR10_AResNEst_WRN_16_8','CIFAR10_AResNEst_WRN_40_4','CIFAR10_AResNEst_ResNet_110','CIFAR10_AResNEst_ResNet_20',
                'CIFAR100_Standard_WRN_16_8','CIFAR100_Standard_WRN_40_4','CIFAR100_Standard_ResNet_110','CIFAR100_Standard_ResNet_20',
                'CIFAR100_ResNEst_WRN_16_8','CIFAR100_ResNEst_WRN_40_4','CIFAR100_ResNEst_ResNet_110','CIFAR100_ResNEst_ResNet_20',
                'CIFAR100_BNResNEst_WRN_16_8','CIFAR100_BNResNEst_WRN_40_4','CIFAR100_BNResNEst_ResNet_110','CIFAR100_BNResNEst_ResNet_20',
                'CIFAR100_AResNEst_WRN_16_8','CIFAR100_AResNEst_WRN_40_4','CIFAR100_AResNEst_ResNet_110','CIFAR100_AResNEst_ResNet_20']
"""
