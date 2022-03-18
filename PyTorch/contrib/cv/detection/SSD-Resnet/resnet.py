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

import torch                    # for torch.cat and torch.zeros
import torch.nn as nn
import torch.utils.model_zoo as model_zoo



__all__ = ['resnet']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class Layers_NCHW:
    Conv2d = nn.Conv2d
    MaxPool = nn.MaxPool2d
    BnAddRelu = None # will be assigned at construction

    def __init__(self, bn_group, **kwargs):
        super(Layers_NCHW, self).__init__()
        self.nhwc = False
        self.bn_group = bn_group
        bn_base = nn.BatchNorm2d
        class BnAddRelu_(bn_base):
            def __init__(self, planes, fuse_relu=False, bn_group=1):
                super(BnAddRelu_, self).__init__(planes)

                self.fuse_relu_flag = fuse_relu

            def forward(self, x, z=None):
                out = super().forward(x)
                if z is not None:
                    out = out.add_(z)
                if self.fuse_relu_flag:
                    out = out.relu_()
                return out

        # this is still Layers_NCHW::__init__
        self.BnAddRelu = BnAddRelu_

    def build_bn(self, planes, fuse_relu=False):
        return self.BnAddRelu(planes, fuse_relu, self.bn_group)



def conv1x1(layer_types, in_planes, out_planes, stride=1):
     """1x1 convolution"""
     return layer_types.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                            bias=False)

def conv3x3(layer_types, in_planes, out_planes, stride=1):
     """3x3 convolution with padding"""
     return layer_types.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                            padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, layerImpls, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(layerImpls, inplanes, planes, stride=stride)
        self.bn1 = layerImpls.build_bn(planes, fuse_relu=True)
        self.conv2 = conv3x3(layerImpls, planes, planes)
        self.bn2 = layerImpls.build_bn(planes, fuse_relu=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out, residual)

        return out

class ResNet(nn.Module):

    def __init__(self, layerImpls, block, layers, num_classes=1000,
                 pad_input=False, ssd_mods=False, use_nhwc=False,
                 bn_group=1):
        self.inplanes = 64
        super(ResNet, self).__init__()
        if pad_input:
            input_channels = 4
        else:
            input_channels = 3
        self.conv1 = layerImpls.Conv2d(input_channels, 64, kernel_size=7, stride=2,
                                       padding=3, bias=False)
        self.bn1 = layerImpls.build_bn(64, fuse_relu=True)
        self.maxpool = layerImpls.MaxPool(kernel_size=3, stride=2, padding=1)

        # Add conv{2,3,4}
        self.layer1 = self._make_layer(layerImpls, block, 64, layers[0])
        self.layer2 = self._make_layer(layerImpls, block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(layerImpls, block, 256, layers[2], stride=1)

        # FIXME! This (a) fails for nhwc, and (b) is irrelevant if the user is
        # also loading pretrained data (which we don't know about here, but
        # know about in the caller (the "resnet()" function below).
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, layerImpls, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                layerImpls.Conv2d(self.inplanes, planes * block.expansion,
                                  kernel_size=1, stride=stride, bias=False),
                layerImpls.build_bn(planes * block.expansion, fuse_relu=False),
            )

        layers = []
        layers.append(block(layerImpls, self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(layerImpls, self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.classifier(x)

        return x

def _transpose_state(state, pad_input=False):
    for k in state.keys():
        if len(state[k].shape) == 4:
            if pad_input and "conv1.weight" in k and not 'layer' in k:
                s = state[k].shape
                state[k] = torch.cat([state[k], torch.zeros([s[0], 1, s[2], s[3]])], dim=1)
            state[k] = state[k].permute(0, 2, 3, 1).contiguous()
    return state

def resnet34(pretrained=False, nhwc=False, ssd_mods=False, **kwargs):
    """Constructs a ResNet model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    layerImpls = Layers_NCHW(**kwargs)
    block = BasicBlock
    layer_list = [3, 4, 6, 3]
    model = ResNet(layerImpls, block, layer_list, ssd_mods=ssd_mods, use_nhwc=nhwc, **kwargs)
    if pretrained:
        # orig_state_dict = model_zoo.load_url(model_urls['resnet34'])
        orig_state_dict = torch.load('./resnet34-333f7ec4.pth')
        
        
        # Modify the state dict to remove conv5 / layer4
        state_dict = {k:orig_state_dict[k] for k in orig_state_dict if (not k.startswith('layer4') and not k.startswith('fc'))}

        pad_input = kwargs.get('pad_input', False)
        if nhwc:
            state_dict = _transpose_state(state_dict, pad_input)

        model.load_state_dict(state_dict)
    return nn.Sequential(model.conv1, model.bn1, model.maxpool, model.layer1, model.layer2, model.layer3)
