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
import torch
import torch.nn as nn
from option import args
# 3D EDSR model structure
def default_conv(in_channelss, out_channels, kernel_size, bias=True):
    return nn.Conv3d(
        in_channelss, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)

class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)

        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class EDSR(nn.Module):
    def __init__(self, conv=default_conv):
        super(EDSR, self).__init__()
        n_feats = args.n_feats # 64
        kernel_size = args.kernel_size
        n_resblock = args.n_resblocks  # 16
        act = nn.ReLU(True)
        res_scale = args.res_scale
        scale = args.scale

        self.head = nn.Sequential(conv(1, n_feats, kernel_size))

        modules_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale) for _ in range(n_resblock)]
        self.body = nn.Sequential(*modules_body)

        modules_tail = [
            nn.Upsample(scale_factor=scale, mode='trilinear', align_corners=False),
            conv(n_feats, 1, kernel_size)]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = x.contiguous()
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        #x = torch.squeeze(x, dim=1)
        return x
