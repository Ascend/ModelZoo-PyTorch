# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

import torch
import torch.nn as nn
# from base_model import L2Norm, ResNet
from resnet import ResNet, resnet34

# from nhwc.conv import Conv2d_NHWC

class SSD300(nn.Module):
    """
        Build a SSD module to take 300x300 image input,
        and output 8732 per class bounding boxes

        label_num: number of classes (including background 0)
    """
    def __init__(self, args, label_num, use_nhwc=False, pad_input=False, bn_group=1, pretrained=True):

        super(SSD300, self).__init__()

        self.label_num = label_num
        self.use_nhwc = use_nhwc
        self.pad_input = pad_input
        self.bn_group = bn_group

        # Explicitly RN34 all the time
        out_channels = 256
        out_size = 38
        self.out_chan = [out_channels, 512, 512, 256, 256, 256]

        # self.model = ResNet(self.use_nhwc, self.pad_input, self.bn_group)

        rn_args = {
            'bn_group' : bn_group,
            'pad_input' : pad_input,
            'nhwc' : use_nhwc,
            'pretrained' : pretrained,
            'ssd_mods' : True,
        }

        self.model = resnet34(**rn_args)

        self._build_additional_features()

        padding_channels_to = 8
        self._build_multibox_heads(use_nhwc, padding_channels_to)

        # after l2norm, conv7, conv8_2, conv9_2, conv10_2, conv11_2
        # classifer 1, 2, 3, 4, 5 ,6

        # intitalize all weights
        with torch.no_grad():
            self._init_weights()

    def _build_multibox_heads(self, use_nhwc, padding_channels_to=8):
        self.num_defaults = [4, 6, 6, 6, 4, 4]
        self.mbox = []
        self.padding_amounts = []

        # if self.use_nhwc:
        #     conv_fn = Conv2d_NHWC
        # else:
        #     conv_fn = nn.Conv2d
        conv_fn = nn.Conv2d                                    
        # Multiple to pad channels to
        for nd, oc in zip(self.num_defaults, self.out_chan):
            # Horizontally fuse loc and conf convolutions
            my_num_channels = nd*(4+self.label_num)
            # if self.use_nhwc:
            #     # Want to manually pad to get HMMA kernels in NHWC case
            #     padding_amount = padding_channels_to - (my_num_channels % padding_channels_to)
            # else:
            #
            padding_amount = 0                                
            self.padding_amounts.append(padding_amount)
            self.mbox.append(conv_fn(oc, my_num_channels + padding_amount, kernel_size=3, padding=1))

        self.mbox = nn.ModuleList(self.mbox)


    """
    Output size from RN34 is always 38x38
    """
    def _build_additional_features(self):
        self.additional_blocks = []

        # if self.use_nhwc:
        #     conv_fn = Conv2d_NHWC
        # else:
        #     conv_fn = nn.Conv2d
        conv_fn = nn.Conv2d                                    
        def build_block(input_channels, inter_channels, out_channels, stride=1, pad=0):
            return nn.Sequential(
                conv_fn(input_channels, inter_channels, kernel_size=1),
                nn.ReLU(inplace=True),
                conv_fn(inter_channels, out_channels, kernel_size=3, stride=stride, padding=pad),
                nn.ReLU(inplace=True)
            )

        strides = [2, 2, 2, 1, 1]
        intermediates = [256, 256, 128, 128, 128]
        paddings = [1, 1, 1, 0, 0]

        for i, im, o, stride, pad in zip(self.out_chan[:-1], intermediates, self.out_chan[1:], strides, paddings):
            self.additional_blocks.append(build_block(i, im, o, stride=stride, pad=pad))

        self.additional_blocks = nn.ModuleList(self.additional_blocks)

    def _init_additional_weights(self):
        addn_blocks = [*self.additional_blocks]
        # Need to handle additional blocks differently in NHWC case due to xavier initialization
        for layer in addn_blocks:
            for param in layer.parameters():
                if param.dim() > 1:
                    # if self.use_nhwc:
                    #     # xavier_uniform relies on fan-in/-out, so need to use NCHW here to get
                    #     # correct values (K, R) instead of the correct (K, C)
                    #     nchw_param_data = param.data.permute(0, 3, 1, 2).contiguous()
                    #     nn.init.xavier_uniform_(nchw_param_data)
                    #     # Now permute correctly-initialized param back to NHWC
                    #     param.data.copy_(nchw_param_data.permute(0, 2, 3, 1).contiguous())
                    # else:
                    #     nn.init.xavier_uniform_(param)
                    nn.init.xavier_uniform_(param)                       
    def _init_multibox_weights(self):
        layers = [ *self.mbox ]

        for layer, default, padding in zip(layers, self.num_defaults, self.padding_amounts):
            for param in layer.parameters():
                if param.dim() > 1 and self.use_nhwc:
                    # Need to be careful - we're initialising [loc, conf, pad] with
                    # all 3 needing to be treated separately
                    conf_channels = default * self.label_num
                    loc_channels  = default * 4
                    pad_channels  = padding
                    # Split the parameter into separate parts along K dimension
                    conf, loc, pad = param.data.split([conf_channels, loc_channels, pad_channels], dim=0)

                    # Padding should be zero
                    pad_data = torch.zeros_like(pad.data)

                    def init_loc_conf(p):
                        p_data = p.data.permute(0, 3, 1, 2).contiguous()
                        nn.init.xavier_uniform_(p_data)
                        p_data = p_data.permute(0, 2, 3, 1).contiguous()
                        return p_data

                    # Location and confidence data
                    loc_data = init_loc_conf(loc)
                    conf_data = init_loc_conf(conf)

                    # Put the full weight together again along K and copy
                    param.data.copy_(torch.cat([conf_data, loc_data, pad_data], dim=0))
                elif param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    def _init_weights(self):
        self._init_additional_weights()
        self._init_multibox_weights()

    # Shape the classifier to the view of bboxes
    def bbox_view(self, src, mbox):
        locs = []
        confs = []
        for s, m, num_defaults, pad in zip(src, mbox, self.num_defaults, self.padding_amounts):
            mm = m(s)
            conf_channels = num_defaults * self.label_num
            loc_channels  = num_defaults * 4

            # if self.use_nhwc:
            #     conf, loc, _ = mm.split([conf_channels, loc_channels, pad], dim=3)
            #     conf, loc = conf.contiguous(), loc.contiguous()
            #     # We now have unfused [N, H, W, C]
            #     # Layout is a little awkward here.
            #     # Take C = c * d, then we actually have:
            #     # [N, H, W, c*d]
            #     # flatten HW first:
            #     #   [N, H, W, c*d] -> [N, HW, c*d]
            #     locs.append(
            #         loc.view(s.size(0), -1, 4 * num_defaults).permute(0, 2, 1).contiguous().view(loc.size(0), 4, -1))
            #     confs.append(
            #         conf.view(s.size(0), -1, self.label_num * num_defaults).permute(0, 2, 1).contiguous().view(conf.size(0), self.label_num, -1))
            # else:
            #     conf, loc = mm.split([conf_channels, loc_channels], dim=1)
            #     conf, loc = conf.contiguous(), loc.contiguous()
            #     # flatten the anchors for this layer
            #     locs.append(loc.view(s.size(0), 4, -1))
            #     confs.append(conf.view(s.size(0), self.label_num, -1))        
            conf, loc = mm.split([conf_channels, loc_channels], dim=1)
            conf, loc = conf.contiguous(), loc.contiguous()
            # flatten the anchors for this layer
            locs.append(loc.view(s.size(0), 4, -1))
            confs.append(conf.view(s.size(0), self.label_num, -1))
        cat_dim = 2
        locs, confs = torch.cat(locs, cat_dim), torch.cat(confs, cat_dim)

        return locs, confs

    def forward(self, data):

        layers = self.model(data)

        # last result from network goes into additional blocks
        x = layers
        # If necessary, transpose back to NCHW
        additional_results = []
        for i, l in enumerate(self.additional_blocks):
            x = l(x)
            additional_results.append(x)

        # do we need the l2norm on the first result?
        src = [layers, *additional_results]
        # Feature Map 38x38x4, 19x19x6, 10x10x6, 5x5x6, 3x3x4, 1x1x4

        locs, confs = self.bbox_view(src, self.mbox)

        # For SSD 300, shall return nbatch x 8732 x {nlabels, nlocs} results
        return locs, confs

