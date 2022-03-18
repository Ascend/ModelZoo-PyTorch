#!/usr/bin/env python
# encoding: utf-8
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
@author: wujiyang
@contact: wujiyang@hust.edu.cn
@file: agentcenterloss.py
@time: 2019/1/7 10:53
@desc: the variety of center loss, which use the class weight as the class center and normalize both the weight and feature,
       in this way, the cos distance of weight and feature can be used as the supervised signal.
       It's similar with torch.nn.CosineEmbeddingLoss, x_1 means weight_i, x_2 means feature_i.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class AgentCenterLoss(nn.Module):

    def __init__(self, num_classes, feat_dim, scale):
        super(AgentCenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.scale = scale

        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        '''
        Parameters:
            x: input tensor with shape (batch_size, feat_dim)
            labels: ground truth label with shape (batch_size)
        Return:
            loss of centers
        '''
        cos_dis = F.linear(F.normalize(x), F.normalize(self.centers)) * self.scale

        one_hot = torch.zeros_like(cos_dis)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # loss = 1 - cosine(i)
        loss = one_hot * self.scale - (one_hot * cos_dis)

        return loss.mean()