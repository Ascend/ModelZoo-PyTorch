# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017, 
# All rights reserved.
#
# Copyright 2021 Huawei Technologies Co., Ltd
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
# limitations under the License


import torch
import torch.nn as nn
import torch.npu
from apex import amp
from blocks import ShuffleV1Block
from utils import get_parameters, CrossEntropyLabelSmooth

class ShuffleNetV1(nn.Module):
    def __init__(self, input_size=224, n_class=1000, model_size='2.0x', group=None):
        super(ShuffleNetV1, self).__init__()
        print('model size is ', model_size)

        assert group is not None

        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        if group == 3:
            if model_size == '0.5x':
                self.stage_out_channels = [-1, 12, 120, 240, 480]
            elif model_size == '1.0x':
                self.stage_out_channels = [-1, 24, 240, 480, 960]
            elif model_size == '1.5x':
                self.stage_out_channels = [-1, 24, 360, 720, 1440]
            elif model_size == '2.0x':
                self.stage_out_channels = [-1, 48, 480, 960, 1920]
            else:
                raise NotImplementedError
        elif group == 8:
            if model_size == '0.5x':
                self.stage_out_channels = [-1, 16, 192, 384, 768]
            elif model_size == '1.0x':
                self.stage_out_channels = [-1, 24, 384, 768, 1536]
            elif model_size == '1.5x':
                self.stage_out_channels = [-1, 24, 576, 1152, 2304]
            elif model_size == '2.0x':
                self.stage_out_channels = [-1, 48, 768, 1536, 3072]
            else:
                raise NotImplementedError

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = []
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage+2]

            for i in range(numrepeat):
                stride = 2 if i == 0 else 1
                first_group = idxstage == 0 and i == 0
                self.features.append(ShuffleV1Block(input_channel, output_channel,
                                            group=group, first_group=first_group,
                                            mid_channels=output_channel // 4, ksize=3, stride=stride))
                input_channel = output_channel

        self.features = nn.Sequential(*self.features)

        self.globalpool = nn.AvgPool2d(7)

        self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], n_class, bias=False))
        self._initialize_weights()

    def forward(self, x):
        #print('****1')
        x = self.first_conv(x)
        x = self.maxpool(x)
        x = self.features(x)

        #x = self.globalpool(x.float().cpu()).npu().half()
        x = self.globalpool(x)
        x = x.contiguous().view(-1, self.stage_out_channels[-1])
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

if __name__ == "__main__":
    print('**1')
    model = ShuffleNetV1(group=3, model_size='1.0x')

    # add hook
    def hook_func(name, module):
        def hook_function(module, inputs, outputs):
            print(name+' inputs')
            print(name+' outputs')
        return hook_function

    for name, module in model.named_modules():
        module.register_forward_hook(hook_func('[forward]: '+name, module))
        module.register_backward_hook(hook_func('[backward]: '+name, module))

    print('**2')
    device = "npu:0"
    torch.npu.set_device(device)
    criterion_smooth = CrossEntropyLabelSmooth(1000, 0.1)
    loss_function = criterion_smooth.to(device)
    print('**2.5')
    model = model.npu()
    print("model to npu ok ")
    optimizer = torch.optim.SGD(get_parameters(model),
                            lr=0.5,
                            momentum=0.9,
                            weight_decay=4e-5)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O2")

    print('**3')
    bs = 5
    test_data = torch.rand(bs, 3, 224, 224, requires_grad=True)
    target = torch.randint(1,10,(bs,))
    target = target.type(torch.LongTensor)
    target = target.to(torch.int32)
    target = target.npu()
    test_data = test_data.npu()
    print('**4')
    test_outputs = model(test_data)
    print('**5')
    print(test_outputs.size())

    #loss = test_outputs.sum()
    loss = loss_function(test_outputs, target)
    optimizer.zero_grad()
    print(loss.size())
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
    optimizer.step()
    print('**OK')
