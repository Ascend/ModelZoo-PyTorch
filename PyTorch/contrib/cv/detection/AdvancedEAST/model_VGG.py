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

import torch
import torch.nn as nn
import torch.nn.functional as F

import cfg


class advancedEAST(nn.Module):
    def __init__(self):
        super(advancedEAST, self).__init__()

        # Bottom-up layers
        self.layer2 = self.make_layers([64, 64, 'M', 128, 128, 'M'], in_channels=3)
        self.layer3 = self.make_layers([256, 256, 256, 'M'], in_channels=128)
        self.layer4 = self.make_layers([512, 512, 512, 'M'], in_channels=256)
        self.layer5 = self.make_layers([512, 512, 512, 'M'], in_channels=512)
        # Top-down
        self.merging1 = self.merging(i=2)
        self.merging2 = self.merging(i=3)
        self.merging3 = self.merging(i=4)
        # before output layers
        self.last_bn = nn.BatchNorm2d(32)
        self.conv_last = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.inside_score_conv = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
        self.side_v_code_conv = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0)
        self.side_v_coord_conv = nn.Conv2d(32, 4, kernel_size=1, stride=1, padding=0)
        # locked first two conv layers
        if cfg.locked_layers:
            i = 1
            for m in self.layer2.children():
                if isinstance(m, nn.Conv2d) and i <= 2:
                    print('冻结第{}层参数，层属性：{}'.format(i, m))
                    for param in m.parameters():
                        param.requires_grad = False
                    i += 1

    def make_layers(self, cfg_list, in_channels=3, batch_norm=True):  # VGG part
        layers = []
        for v in cfg_list:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def merging(self, i=2):
        in_size = {'2': 1024, '3': 384, '4': 192}
        layers = [
            nn.BatchNorm2d(in_size[str(i)]),
            nn.Conv2d(in_size[str(i)], 128 // 2 ** (i - 2), kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(128 // 2 ** (i - 2)),
            nn.Conv2d(128 // 2 ** (i - 2), 128 // 2 ** (i - 2), kernel_size=3, stride=1, padding=1),
            nn.ReLU()]
        return nn.Sequential(*layers)

    def forward(self, x):
        # Bottom-up
        f4 = self.layer2(x)  # 128
        f3 = self.layer3(f4)  # 256
        f2 = self.layer4(f3)  # 512
        f1 = self.layer5(f2)  # 512
        # Top-down
        h1 = f1
        H1 = nn.UpsamplingNearest2d(scale_factor=2)(h1)
        concat1 = torch.cat((H1, f2), axis=1)  # 1024
        h2 = self.merging1(concat1)  # 128
        H2 = nn.UpsamplingNearest2d(scale_factor=2)(h2)
        concat2 = torch.cat((H2, f3), axis=1)  # 128+256
        h3 = self.merging2(concat2)  # 64
        H3 = nn.UpsamplingNearest2d(scale_factor=2)(h3)
        concat3 = torch.cat((H3, f4), axis=1)  # 64+128
        h4 = self.merging3(concat3)  # 32
        # before output layers
        bn = self.last_bn(h4)
        before_output = F.relu(self.conv_last(bn))
        inside_score = self.inside_score_conv(before_output)
        side_v_code = self.side_v_code_conv(before_output)
        side_v_coord = self.side_v_coord_conv(before_output)
        east_detect = torch.cat((inside_score, side_v_code, side_v_coord), axis=1)
        return east_detect


if __name__ == '__main__':
    net = advancedEAST()
    if cfg.model_summary:
        try:
            from torchsummary import summary
            summary(net, input_size=(3, 128, 128))
        except ImportError:
            print("\"torchsummary\" not found, please install to visualize the model architecture.")
            cfg.model_summary = False
