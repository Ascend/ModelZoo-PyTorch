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
import numpy as np
import torch
from torch import nn

from blocks import ConvLayer, ResidualBlock


def define_P(in_size=512, out_size=512, min_feat_size=32, relu_type='LeakyReLU', isTrain=False, weight_path=None):
    net = ParseNet(in_size, out_size, min_feat_size, 64, 19, norm_type='bn', relu_type=relu_type, ch_range=[32, 256])
    if not isTrain:
        net.eval()  
    if weight_path is not None:
        net.load_state_dict(torch.load(weight_path))
    return net


class ParseNet(nn.Module):
    def __init__(self,
                in_size=128,
                out_size=128,
                min_feat_size=32,
                base_ch=64,
                parsing_ch=19,
                res_depth=10,
                relu_type='prelu',
                norm_type='bn',
                ch_range=None,
                ):
        super(ParseNet, self).__init__()
        if ch_range is None:
            ch_range = [32, 512]
        self.res_depth = res_depth
        act_args = {'norm_type': norm_type, 'relu_type': relu_type}
        min_ch, max_ch = ch_range

        ch_clip = lambda x: max(min_ch, min(x, max_ch))
        min_feat_size = min(in_size, min_feat_size)

        down_steps = int(np.log2(in_size//min_feat_size))
        up_steps = int(np.log2(out_size//min_feat_size))

        # =============== define encoder-body-decoder ==================== 
        self.encoder = []
        self.encoder.append(ConvLayer(3, base_ch, 3, 1))
        head_ch = base_ch
        for _ in range(down_steps):
            cin, cout = ch_clip(head_ch), ch_clip(head_ch * 2)
            self.encoder.append(ResidualBlock(cin, cout, scale='down', **act_args))
            head_ch = head_ch * 2

        self.body = []
        for _ in range(res_depth):
            self.body.append(ResidualBlock(ch_clip(head_ch), ch_clip(head_ch), **act_args))

        self.decoder = []
        for _ in range(up_steps):
            cin, cout = ch_clip(head_ch), ch_clip(head_ch // 2)
            self.decoder.append(ResidualBlock(cin, cout, scale='up', **act_args))
            head_ch = head_ch // 2

        self.encoder = nn.Sequential(*self.encoder)
        self.body = nn.Sequential(*self.body)
        self.decoder = nn.Sequential(*self.decoder)
        self.out_img_conv = ConvLayer(ch_clip(head_ch), 3)
        self.out_mask_conv = ConvLayer(ch_clip(head_ch), parsing_ch)

    def forward(self, x):
        feat = self.encoder(x)
        x = feat + self.body(feat)
        x = self.decoder(x)
        out_img = self.out_img_conv(x) 
        out_mask = self.out_mask_conv(x)
        return out_mask, out_img
