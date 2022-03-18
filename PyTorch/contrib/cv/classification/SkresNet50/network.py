# Copyright (c) Soumith Chintala 2016,
# All rights reserved
#
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://spdx.org/licenses/BSD-3-Clause.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn
import torch
import torch.nn.functional as F
from apex import amp

#TODO sknet for basicblocks

__all__ = ['sk_resnet18', 'sk_resnet34', 'sk_resnet50', 'sk_resnet101',
           'sk_resnet152']


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, groups=groups)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2g = conv3x3(planes, planes, stride, groups = 32)
        self.bn2g   = nn.BatchNorm2d(planes)

        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_fc1 = nn.Conv2d(planes, planes//16, 1, bias=False)
        self.bn_fc1   = nn.BatchNorm2d(planes//16)
        self.conv_fc2 = nn.Conv2d(planes//16, 2 * planes, 1, bias=False)

        self.D = planes

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        d1 = self.conv2(out)
        d1 = self.bn2(d1)
        d1 = self.relu(d1)

        d2 = self.conv2g(out)
        d2 = self.bn2g(d2)
        d2 = self.relu(d2)

        d  = self.avg_pool(d1) + self.avg_pool(d2)
        d = F.relu(self.bn_fc1(self.conv_fc1(d)))
        d = self.conv_fc2(d)
        d = torch.unsqueeze(d, 1).view(-1, 2, self.D, 1, 1)
        d = F.softmax(d, 1)
        d1 = d1 * d[:, 0, :, :, :].squeeze(1)
        d2 = d2 * d[:, 1, :, :, :].squeeze(1)
        d  = d1 + d2

        out = self.conv3(d)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
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


def sk_resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def sk_resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def sk_resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def sk_resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

def sk_resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model

if __name__ == "__main__":

    class SoftCrossEntropyLoss(nn.NLLLoss):
        def __init__(self, label_smoothing=0, num_classes=1000, **kwargs):
            assert label_smoothing >= 0 and label_smoothing <= 1
            super(SoftCrossEntropyLoss, self).__init__(**kwargs)
            self.confidence = 1 - label_smoothing
            self.other      = label_smoothing * 1.0 / (num_classes - 1)
            self.criterion  = nn.KLDivLoss(reduction='batchmean')
            print('using soft celoss!!!, label_smoothing = ', label_smoothing)

        def forward(self, input, target):
            one_hot = torch.zeros_like(input)
            one_hot.fill_(self.other)
            one_hot.scatter_(1, target.unsqueeze(1).long(), self.confidence)
            input   = F.log_softmax(input, 1)
            return self.criterion(input, one_hot)


    torch.manual_seed(12345)
    usenpu = True

    if usenpu:
        print('**1')
        model = sk_resnet50()
        
        '''
        # add hook for debuging
        def hook_func(name, module):
            def hook_function(module, inputs, outputs):
                print(name+' inputs')
                print(name+' outputs')
            return hook_function

        for name, module in model.named_modules():
            module.register_forward_hook(hook_func('[forward]: '+name, module))
            module.register_backward_hook(hook_func('[backward]: '+name, module))
        '''

        print('**2')
        device = "npu:0"
        torch.npu.set_device(device)
        criterion_smooth = SoftCrossEntropyLoss()
        loss_function = criterion_smooth.to(device)
        print('**2.5')
        model = model.npu()
        print("model to npu ok ")
        optimizer = torch.optim.SGD(model.parameters(),
                                lr=0.5,
                                momentum=0.9,
                                weight_decay=4e-5)
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2")

        print('**3')
        bs = 5
        test_data = torch.rand(bs, 3, 224, 224, requires_grad=True)
        target = torch.randint(1,10,(bs,))
        target = target.type(torch.LongTensor)
        target = target.npu()
        test_data = test_data.npu()
        print('**4')
        test_outputs = model(test_data)
        print('**5')
        print(test_outputs.size())

        loss = test_outputs.sum()
        #loss = loss_function(test_outputs, target)
        optimizer.zero_grad()
        print(loss.size())
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        #loss.backward()
        print('**OK')
    else:
        model = sk_resnet50()

        bs = 5
        test_data = torch.rand(bs, 3, 224, 224, requires_grad=True)
        target = torch.randint(1,10,(bs,))
        print(test_data.sum())
        criterion_smooth = SoftCrossEntropyLoss()
        loss_function = criterion_smooth

        test_outputs = model(test_data)
        print(test_outputs.size())
        print(test_outputs.sum())
        loss = loss_function(test_outputs, target)
        loss.backward()

        print('**OK')