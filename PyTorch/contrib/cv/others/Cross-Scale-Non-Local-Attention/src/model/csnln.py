# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from model import common
import torch.nn as nn
import torch
if torch.__version__ >= '1.8':
    import torch_npu
from model.attention import CrossScaleAttention,NonLocalAttention
def make_model(args, parent=False):
    return CSNLN(args)

#projection between attention branches
class MultisourceProjection(nn.Module):
    def __init__(self, in_channel,kernel_size = 3,scale=2, conv=common.default_conv):
        super(MultisourceProjection, self).__init__()
        deconv_ksize, stride, padding, up_factor = {
            2: (6,2,2,2),
            3: (9,3,3,3),
            4: (6,2,2,2)
        }[scale]
        self.up_attention = CrossScaleAttention(scale = up_factor)
        self.down_attention = NonLocalAttention()
        self.upsample = nn.Sequential(*[nn.ConvTranspose2d(in_channel,in_channel,deconv_ksize,stride=stride,padding=padding),nn.PReLU()])
        self.encoder = common.ResBlock(conv, in_channel, kernel_size, act=nn.PReLU(), res_scale=1)
    
    def forward(self,x):
        #down_map = self.upsample(torch_npu.npu_format_cast(self.down_attention(x), 0))
        down_map = torch_npu.npu_format_cast(self.down_attention(x), 0)
        for i in range(len(self.upsample)):
            down_map = self.upsample[i](down_map)
            if i==0:
                down_map = torch_npu.npu_format_cast(down_map, 0)
        up_map = self.up_attention(x)

        err = self.encoder(up_map-down_map)
                
        final_map = down_map + err
        
        return final_map

#projection with local branch
class RecurrentProjection(nn.Module):
    def __init__(self, in_channel,kernel_size = 3, scale = 2, conv=common.default_conv):
        super(RecurrentProjection, self).__init__()
        self.scale = scale
        stride_conv_ksize, stride, padding = {
            2: (6,2,2),
            3: (9,3,3),
            4: (6,2,2)
        }[scale]

        self.multi_source_projection = MultisourceProjection(in_channel,kernel_size=kernel_size,scale = scale, conv=conv)
        self.down_sample_1 = nn.Sequential(*[nn.Conv2d(in_channel,in_channel,stride_conv_ksize,stride=stride,padding=padding),nn.PReLU()])
        if scale != 4:
            self.down_sample_2 = nn.Sequential(*[nn.Conv2d(in_channel,in_channel,stride_conv_ksize,stride=stride,padding=padding),nn.PReLU()])
        self.error_encode = nn.Sequential(*[nn.ConvTranspose2d(in_channel,in_channel,stride_conv_ksize,stride=stride,padding=padding),nn.PReLU()])
        self.post_conv = common.BasicBlock(conv,in_channel,in_channel,kernel_size,stride=1,bias=True,act=nn.PReLU())
        if scale == 4:
            self.multi_source_projection_2 = MultisourceProjection(in_channel,kernel_size=kernel_size,scale = scale, conv=conv)
            self.down_sample_3 = nn.Sequential(*[nn.Conv2d(in_channel,in_channel,8,stride=4,padding=2),nn.PReLU()])
            self.down_sample_4 = nn.Sequential(*[nn.Conv2d(in_channel,in_channel,8,stride=4,padding=2),nn.PReLU()])
            self.error_encode_2 = nn.Sequential(*[nn.ConvTranspose2d(in_channel,in_channel,8,stride=4,padding=2),nn.PReLU()])


    def forward(self, x):
        x_up = self.multi_source_projection(x)
        #x_down = self.down_sample_1(x_up)
        x_down = x_up
        for i in range(len(self.down_sample_1)):
            x_down = self.down_sample_1[i](x_down)
            if i==0:
                x_down = torch_npu.npu_format_cast(x_down, 0)
        
        #error_up = self.error_encode((x-x_down).npu_format_cast(0))
        error_up = torch_npu.npu_format_cast((x-x_down), 0)
        for i in range(len(self.error_encode)):
            error_up = self.error_encode[i](error_up)
            if i==0:
                error_up = torch_npu.npu_format_cast(error_up, 0)
        h_estimate = x_up + error_up
        if self.scale == 4:
            x_up_2 = self.multi_source_projection_2(h_estimate)
            x_down_2 = self.down_sample_3(x_up_2)
            error_up_2 = self.error_encode_2(x-x_down_2)
            h_estimate = x_up_2 + error_up_2
            x_final = self.post_conv(self.down_sample_4(h_estimate))
        else:
            #x_final = self.post_conv(self.down_sample_2(h_estimate))
            temp_x = h_estimate
            for i in range(len(self.down_sample_2)):
                temp_x = self.down_sample_2[i](temp_x)
                if i==0:
                    temp_x = torch_npu.npu_format_cast(temp_x, 0)
            for i in range(len(self.post_conv)):
                temp_x = self.post_conv[i](temp_x)
                if i==0:
                    temp_x = torch_npu.npu_format_cast(temp_x, 0)
            x_final = temp_x

        return x_final, h_estimate
        

        


class CSNLN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(CSNLN, self).__init__()

        #n_convblock = args.n_convblocks
        n_feats = args.n_feats
        self.depth = args.depth
        kernel_size = 3 
        scale = args.scale[0]       

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        
        # define head module
        m_head = [common.BasicBlock(conv, args.n_colors, n_feats, kernel_size,stride=1,bias=True,bn=False,act=nn.PReLU()),
        common.BasicBlock(conv,n_feats, n_feats, kernel_size,stride=1,bias=True,bn=False,act=nn.PReLU())]

        # define Self-Exemplar Mining Cell
        self.SEM = RecurrentProjection(n_feats,scale = scale)

        # define tail module
        m_tail = [
            nn.Conv2d(
                n_feats*self.depth, args.n_colors, kernel_size,
                padding=(kernel_size//2)
            )
        ]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.tail = nn.Sequential(*m_tail)

    def forward(self,input):
        x = self.sub_mean(input)
        #x = self.head(x)
        for i in range(len(self.head)):
            x = self.head[i](x)
            if i%2 ==0:
                x = torch_npu.npu_format_cast(x, 0)
        bag = []
        for i in range(self.depth):
            x, h_estimate = self.SEM(x)
            bag.append(h_estimate)
        h_feature = torch.cat(bag,dim=1)
        h_final = self.tail(h_feature)
        return self.add_mean(h_final)
