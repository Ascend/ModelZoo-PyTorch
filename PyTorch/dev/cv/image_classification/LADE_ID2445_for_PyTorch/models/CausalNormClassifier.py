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
from utils import *
from os import path
import math
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

class Causal_Norm_Classifier(nn.Module):

    def __init__(self, num_classes=1000, feat_dim=2048, use_effect=True, num_head=2, tau=16.0, alpha=3.0, gamma=0.03125, *args):
        super(Causal_Norm_Classifier, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_classes, feat_dim).npu(), requires_grad=True)
        self.scale = tau / num_head   # 16.0 / num_head
        self.norm_scale = gamma       # 1.0 / 32.0
        self.alpha = alpha            # 3.0
        self.num_head = num_head
        self.head_dim = feat_dim // num_head
        self.use_effect = use_effect
        self.reset_parameters(self.weight)
        # self.relu = nn.ReLU(inplace=True)

    def reset_parameters(self, weight):
        stdv = 1. / math.sqrt(weight.size(1))
        weight.data.uniform_(-stdv, stdv)

    def forward(self, x, label, embed):
        # calculate capsule normalized feature vector and predict
        normed_w = self.multi_head_call(self.causal_norm, self.weight, weight=self.norm_scale)
        normed_x = self.multi_head_call(self.l2_norm, x)
        y = torch.mm(normed_x * self.scale, normed_w.t())

        # remove the effect of confounder c during test
        if (not self.training) and self.use_effect:
            self.embed = torch.from_numpy(embed).view(1, -1).to(f'npu:{NPU_CALCULATE_DEVICE}')
            normed_c = self.multi_head_call(self.l2_norm, self.embed)
            head_dim = x.shape[1] // self.num_head
            x_list = torch.split(normed_x, head_dim, dim=1)
            c_list = torch.split(normed_c, head_dim, dim=1)
            w_list = torch.split(normed_w, head_dim, dim=1)
            output = []

            for nx, nc, nw in zip(x_list, c_list, w_list):
                cos_val, sin_val = self.get_cos_sin(nx, nc)
                y0 = torch.mm((nx -  cos_val * self.alpha * nc) * self.scale, nw.t())
                output.append(y0)
            y = sum(output)

        return y, None

    def get_cos_sin(self, x, y):
        cos_val = (x * y).sum(-1, keepdim=True) / torch.norm(x, 2, 1, keepdim=True) / torch.norm(y, 2, 1, keepdim=True)
        sin_val = (1 - cos_val * cos_val).sqrt()
        return cos_val, sin_val

    def multi_head_call(self, func, x, weight=None):
        assert len(x.shape) == 2
        x_list = torch.split(x, self.head_dim, dim=1)
        if weight:
            y_list = [func(item, weight) for item in x_list]
        else:
            y_list = [func(item) for item in x_list]
        assert len(x_list) == self.num_head
        assert len(y_list) == self.num_head
        return torch.cat(y_list, dim=1)

    def l2_norm(self, x):
        normed_x = x / torch.norm(x, 2, 1, keepdim=True)
        return normed_x

    def capsule_norm(self, x):
        norm= torch.norm(x.clone(), 2, 1, keepdim=True)
        normed_x = (norm / (1 + norm)) * (x / norm)
        return normed_x

    def causal_norm(self, x, weight):
        norm= torch.norm(x, 2, 1, keepdim=True)
        normed_x = x / (norm + weight)
        return normed_x

def create_model(feat_dim, num_classes=1000, stage1_weights=False, dataset=None, log_dir=None, test=False, use_effect=True, num_head=None, tau=None, alpha=None, gamma=None, *args):
    print('Loading Causal Norm Classifier with use_effect: {}, num_head: {}, tau: {}, alpha: {}, gamma: {}.'.format(str(use_effect), num_head, tau, alpha, gamma))
    clf = Causal_Norm_Classifier(num_classes, feat_dim, use_effect=use_effect, num_head=num_head, tau=tau, alpha=alpha, gamma=gamma)

    return clf
