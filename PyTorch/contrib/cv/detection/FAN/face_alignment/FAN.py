# Copyright 2022 Huawei Technologies Co., Ltd
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


'''
https://github.com/1adrianb/face-alignment/blob/master/face_alignment/models.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['FAN', 'ResNetDepth']


def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding, bias=bias)


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HourGlass(nn.Module):
    def __init__(self, nModules, depth, num_features):
        super(HourGlass, self).__init__()
        self.nModules = nModules
        self.depth = depth
        self.features = num_features

        self._generate_network(self.nModules, self.depth)

    def _generate_network(self, nModules, level):
        # Upper branch
        for conv_module in range(nModules):
            self.add_module('b1_' + str(level), ConvBlock(self.features, self.features))

        # Lower branch
        for conv_module in range(nModules):
            self.add_module('b2_' + str(level), ConvBlock(self.features, self.features))

        # HG body recursion
        if level > 1:
            self._generate_network(nModules, level - 1)
        else:
            for conv_module in range(nModules):
                self.add_module('b2_plus_' + str(level),
                                ConvBlock(self.features, self.features))

        for conv_module in range(nModules):
            self.add_module('b3_' + str(level), ConvBlock(self.features, self.features))

    def _forward(self, nModules, level, inp):
        # Upper branch
        up1 = inp
        for conv_module in range(nModules):
            up1 = self._modules['b1_' + str(level)](up1)

        # Lower branch
        low1 = F.avg_pool2d(inp, 2, stride=2)
        for conv_module in range(nModules):
            low1 = self._modules['b2_' + str(level)](low1)

        # HG body recursion
        if level > 1:
            low2 = self._forward(nModules, level - 1, low1)
        else:
            low2 = low1
            for conv_module in range(nModules):
                low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        for conv_module in range(nModules):
            low3 = self._modules['b3_' + str(level)](low3)

        up2 = F.interpolate(low3, scale_factor=2, mode='nearest')

        # Bring two branches together
        return up1 + up2

    def forward(self, x):
        return self._forward(self.nModules, self.depth, x)


#######################################################################

# class HourGlass(nn.Module):
#     def __init__(self, nModules, depth, num_features):
#         super(HourGlass, self).__init__()
#         self.nModules = nModules
#         self.depth = depth
#         self.features = num_features
#
#         self._generate_network(self.nModules, self.depth)
#
#     def _generate_network(self, nModules, level):
#         # Upper branch
#         for conv_module in range(nModules):
#             self.add_module('b1_' + str(level), ConvBlock(self.features, self.features))
#
#         # Lower branch
#         for conv_module in range(nModules):
#             self.add_module('b2_' + str(level), ConvBlock(self.features, self.features))
#
#         # HG body recursion
#         if level > 1:
#             self._generate_network(nModules, level - 1)
#         else:
#             for conv_module in range(nModules):
#                 self.add_module('b2_plus_' + str(level),
#                                 ConvBlock(self.features, self.features))
#
#         for conv_module in range(nModules):
#             self.add_module('b3_' + str(level), ConvBlock(self.features, self.features))
#
#     def _forward(self, nModules, level, inp):
#         # Upper branch
#         up1 = inp
#         for conv_module in range(nModules):
#             up1 = self._modules['b1_' + str(level)](up1)
#
#         # Lower branch
#         low1 = F.avg_pool2d(inp, 2, stride=2)
#         for conv_module in range(nModules):
#             low1 = self._modules['b2_' + str(level)](low1)
#
#         # HG body recursion
#         if level > 1:
#             low2 = self._forward(nModules, level - 1, low1)
#         else:
#             low2 = low1
#             for conv_module in range(nModules):
#                 low2 = self._modules['b2_plus_' + str(level)](low2)
#
#         low3 = low2
#         for conv_module in range(nModules):
#             low3 = self._modules['b3_' + str(level)](low3)
#
#         up2 = F.interpolate(low3, scale_factor=2, mode='nearest2d')
#
#         # Bring two branches together
#         return up1 + up2
#
#     def forward(self, x):
#         return self._forward(self.nModules, self.depth, x)

#######################################################################


class FAN(nn.Module):

    def __init__(self, nStack=4, nModules=1, nHgDepth=4, num_feats=256, num_classes=68):
        super(FAN, self).__init__()

        self.inplanes = 64
        self.expansion = 2
        self.nStack = nStack  # num of how many HourGlassNet stack to the whole FAN Net
        self.nModules = nModules  # num of ConvBlock==Residual before HG block
        self.nHgDepth = nHgDepth  # num of recursion in HourGlassNet
        self.num_feats = num_feats
        self.num_classes = num_classes

        # Base part
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.conv2 = ConvBlock(self.inplanes, self.inplanes * self.expansion)
        self.conv3 = ConvBlock(self.inplanes * self.expansion, self.inplanes * self.expansion)
        self.conv4 = ConvBlock(self.inplanes * self.expansion, self.num_feats)

        # Stacking part
        for hg_stack in range(self.nStack):
            self.add_module('m' + str(hg_stack), HourGlass(self.nModules, self.nHgDepth, self.num_feats))

            for conv_module in range(self.nModules):
                self.add_module('top_m_' + str(hg_stack),
                                ConvBlock(self.num_feats, self.num_feats))

            self.add_module('conv_last' + str(hg_stack),
                            nn.Conv2d(self.num_feats, self.num_feats, kernel_size=1, stride=1, padding=0))
            self.add_module('bn_end' + str(hg_stack), nn.BatchNorm2d(self.num_feats))
            self.add_module('l' + str(hg_stack), nn.Conv2d(self.num_feats,
                                                           self.num_classes, kernel_size=1, stride=1, padding=0))

            if hg_stack < self.nStack - 1:
                self.add_module(
                    'bl' + str(hg_stack), nn.Conv2d(self.num_feats, self.num_feats, kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(hg_stack), nn.Conv2d(self.num_classes,
                                                                self.num_feats, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), True)
        x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        x = self.conv3(x)
        x = self.conv4(x)

        previous = x

        outputs = []
        for i in range(self.nStack):
            hg = self._modules['m' + str(i)](previous)

            # Residual layers at output resolution
            ll = hg
            for conv_module in range(self.nModules):
                ll = self._modules['top_m_' + str(i)](ll)

            # Linear layer to produce first set of predictions
            ll = F.relu(self._modules['bn_end' + str(i)]
                        (self._modules['conv_last' + str(i)](ll)), True)

            # Predict heatmaps
            tmp_out = self._modules['l' + str(i)](ll)
            outputs.append(tmp_out)

            if i < self.nStack - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_

        return outputs


class ResNetDepth(nn.Module):

    def __init__(self, block=Bottleneck, layers=[3, 8, 36, 3], num_classes=68):
        self.inplanes = 64
        super(ResNetDepth, self).__init__()
        self.conv1 = nn.Conv2d(3 + 68, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
