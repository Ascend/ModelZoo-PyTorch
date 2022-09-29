# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from models.blocks import *
import torch
from torch import nn
import numpy as np
import os


class SPARNet(nn.Module):
    """Deep residual network with spatial attention for face SR.
    # Arguments:
        - n_ch: base convolution channels
        - down_steps: how many times to downsample in the encoder
        - res_depth: depth of residual layers in the main body 
        - up_res_depth: depth of residual layers in each upsample block

    """
    def __init__(
        self,
        min_ch=32,
        max_ch=128,
        in_size=128,
        out_size=128,
        min_feat_size=16,
        res_depth=10,
        relu_type='leakyrelu',
        norm_type='bn',
        att_name='spar',
        bottleneck_size=4,
    ):
        super(SPARNet, self).__init__()
        nrargs = {'norm_type': norm_type, 'relu_type': relu_type}

        ch_clip = lambda x: max(min_ch, min(x, max_ch))

        down_steps = int(np.log2(in_size // min_feat_size))
        up_steps = int(np.log2(out_size // min_feat_size))
        n_ch = ch_clip(max_ch // int(np.log2(in_size // min_feat_size) + 1))

        # ------------ define encoder --------------------
        self.encoder = []
        self.encoder.append(ConvLayer(3, n_ch, 3, 1))
        hg_depth = int(np.log2(64 / bottleneck_size))
        for i in range(down_steps):
            cin, cout = ch_clip(n_ch), ch_clip(n_ch * 2)
            self.encoder.append(ResidualBlock(cin, cout, scale='down', hg_depth=hg_depth, att_name=att_name, **nrargs))

            n_ch = n_ch * 2
            hg_depth = hg_depth - 1
        hg_depth = hg_depth + 1
        self.encoder = nn.Sequential(*self.encoder)
        device_id=int(os.environ['ASCEND_DEVICE_ID'])
        CALCULATE_DEVICE = "npu:{}".format(device_id)
        self.encoder = self.encoder.to(CALCULATE_DEVICE)

        # ------------ define residual layers --------------------
        self.res_layers = []
        for i in range(res_depth + 3 - down_steps):
            channels = ch_clip(n_ch)
            self.res_layers.append(ResidualBlock(channels, channels, hg_depth=hg_depth, att_name=att_name, **nrargs))
        self.res_layers = nn.Sequential(*self.res_layers)
        device_id=int(os.environ['ASCEND_DEVICE_ID'])
        CALCULATE_DEVICE = "npu:{}".format(device_id)
        self.res_layers = self.res_layers.to(CALCULATE_DEVICE)

        # ------------ define decoder --------------------
        self.decoder = []
        for i in range(up_steps):
            hg_depth = hg_depth + 1
            cin, cout = ch_clip(n_ch), ch_clip(n_ch // 2)
            self.decoder.append(ResidualBlock(cin, cout, scale='up', hg_depth=hg_depth, att_name=att_name, **nrargs))
            n_ch = n_ch // 2

        self.decoder = nn.Sequential(*self.decoder)
        device_id=int(os.environ['ASCEND_DEVICE_ID'])
        CALCULATE_DEVICE = "npu:{}".format(device_id)
        self.decoder = self.decoder.to(CALCULATE_DEVICE)
        self.out_conv = ConvLayer(ch_clip(n_ch), 3, 3, 1)
        self.out_conv = self.out_conv.to(CALCULATE_DEVICE)
    
    def forward(self, input_img):
        device_id=int(os.environ['ASCEND_DEVICE_ID'])
        CALCULATE_DEVICE = "npu:{}".format(device_id)
        out = self.encoder(input_img).to(CALCULATE_DEVICE)
        out = self.res_layers(out).to(CALCULATE_DEVICE)
        out = self.decoder(out).to(CALCULATE_DEVICE)
        out_img = self.out_conv(out).to(CALCULATE_DEVICE)
        return out_img

