# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import torch
import torch.npu
import torch.nn as nn
from torch.nn import functional as F
from nets.resnet import ResNetBackbone
from config import cfg

class RootNet(nn.Module):

    def __init__(self):
        self.inplanes = 2048
        self.outplanes = 256

        super(RootNet, self).__init__()
       	self.deconv_layers = self._make_deconv_layer(3)
        self.xy_layer = nn.Conv2d(
            in_channels=self.outplanes,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.depth_layer = nn.Conv2d(
            in_channels=self.inplanes,
            out_channels=1, 
            kernel_size=1,
            stride=1,
            padding=0
        )

    def _make_deconv_layer(self, num_layers):
        layers = []
        inplanes = self.inplanes
        outplanes = self.outplanes
        for i in range(num_layers):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=inplanes,
                    out_channels=outplanes,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=False))
            layers.append(nn.BatchNorm2d(outplanes))
            layers.append(nn.ReLU(inplace=True))
            inplanes = outplanes

        return nn.Sequential(*layers)

    def forward(self, x, k_value):
        # x,y
        xy = self.deconv_layers(x)
        xy = self.xy_layer(xy)
        xy = xy.view(-1,1,cfg.output_shape[0]*cfg.output_shape[1])
        xy = F.softmax(xy,2)
        xy = xy.view(-1,1,cfg.output_shape[0],cfg.output_shape[1])

        hm_x = xy.sum(dim=(2))
        hm_y = xy.sum(dim=(3))

        coord_x = hm_x * torch.arange(cfg.output_shape[1]).float().npu()
        coord_y = hm_y * torch.arange(cfg.output_shape[0]).float().npu()
        
        coord_x = coord_x.sum(dim=2)
        coord_y = coord_y.sum(dim=2)

        # z
        img_feat = torch.mean(x.view(x.size(0), x.size(1), x.size(2)*x.size(3)), dim=2) # global average pooling
        img_feat = torch.unsqueeze(img_feat,2); img_feat = torch.unsqueeze(img_feat,3);
        gamma = self.depth_layer(img_feat)
        gamma = gamma.view(-1,1)
        depth = gamma * k_value.view(-1,1)
        coord = torch.cat((coord_x, coord_y, depth), dim=1)
        return coord

    def init_weights(self):
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.xy_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        for m in self.depth_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

class ResPoseNet(nn.Module):
    def __init__(self, backbone, root):
        super(ResPoseNet, self).__init__()
        self.backbone = backbone
        self.root = root

    def forward(self, input_img, k_value, target=None):
        fm = self.backbone(input_img)
        coord = self.root(fm, k_value)

        if target is None:
            return coord
        else:
            target_coord = target['coord']
            target_vis = target['vis']
            target_have_depth = target['have_depth']
            
            ## coordrinate loss
            loss_coord = torch.abs(coord - target_coord) * target_vis
            loss_coord = (loss_coord[:,0] + loss_coord[:,1] + loss_coord[:,2] * target_have_depth.view(-1))/3.
            return loss_coord

def get_pose_net(cfg, is_train):
    
    backbone = ResNetBackbone(cfg.resnet_type)
    root_net = RootNet()
    if is_train:
        backbone.init_weights()
        root_net.init_weights()

    model = ResPoseNet(backbone, root_net)
    return model


