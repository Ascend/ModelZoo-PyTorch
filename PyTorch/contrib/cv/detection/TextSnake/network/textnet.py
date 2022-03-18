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
import torch.nn as nn
import torch
import torch.nn.functional as F
from network.vgg import VGG16

class Upsample(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.deconv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, upsampled, shortcut):
        x = torch.cat([upsampled, shortcut], dim=1)
        x = self.conv1x1(x)
        x = F.relu(x)
        x = self.conv3x3(x)
        x = F.relu(x)
        x = self.deconv(x)
        return x

class TextNet(nn.Module):

    def __init__(self, backbone='vgg', output_channel=7, is_training=True):
        super().__init__()

        self.is_training = is_training
        self.backbone_name = backbone
        self.output_channel = output_channel

        if backbone == 'vgg':
            self.backbone = VGG16(pretrain=self.is_training)
            self.deconv5 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
            self.merge4 = Upsample(512 + 256, 128)
            self.merge3 = Upsample(256 + 128, 64)
            self.merge2 = Upsample(128 + 64, 32)
            self.merge1 = Upsample(64 + 32, 16)
            self.predict = nn.Sequential(
                nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(16, self.output_channel, kernel_size=1, stride=1, padding=0)
            )
        elif backbone == 'resnet':
            pass

    def forward(self, x):
        C1, C2, C3, C4, C5 = self.backbone(x)
        up5 = self.deconv5(C5)
        up5 = F.relu(up5)

        up4 = self.merge4(C4, up5)
        up4 = F.relu(up4)

        up3 = self.merge3(C3, up4)
        up3 = F.relu(up3)

        up2 = self.merge2(C2, up3)
        up2 = F.relu(up2)

        up1 = self.merge1(C1, up2)
        output = self.predict(up1)

        return output

    def load_model(self, model_path):
        print('Loading from {}'.format(model_path))
        #state_dict = torch.load(model_path)
        #self.load_state_dict(state_dict['model'])
        #多卡模型改单卡加载
        #state_dict = torch.load(model_path, map_location='cuda:0')
        state_dict = torch.load(model_path, lambda storage, loc: storage)
        self.load_state_dict({k.replace('module.',''):v for k,v in state_dict['model'].items()})

if __name__ == '__main__':
    import torch

    input = torch.randn((4, 3, 512, 512))
    net = TextNet().cuda()
    output = net(input.cuda())
    print(output.size())

