# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017
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


import torch
import torch.nn as nn
from apex import amp
from blocks import Shufflenet, Shuffle_Xception, HS, SELayer
from utils import get_parameters, CrossEntropyLabelSmooth

class ShuffleNetV2_Plus(nn.Module):
    def __init__(self, input_size=224, n_class=1000, architecture=None, model_size='Large'):
        super(ShuffleNetV2_Plus, self).__init__()

        print('model size is ', model_size)

        assert input_size % 32 == 0
        assert architecture is not None

        self.stage_repeats = [4, 4, 8, 4]
        if model_size == 'Large':
            self.stage_out_channels = [-1, 16, 68, 168, 336, 672, 1280]
        elif model_size == 'Medium':
            self.stage_out_channels = [-1, 16, 48, 128, 256, 512, 1280]
        elif model_size == 'Small':
            self.stage_out_channels = [-1, 16, 36, 104, 208, 416, 1280]
        else:
            raise NotImplementedError


        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            HS(),
        )

        self.features = []
        archIndex = 0
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage+2]

            activation = 'HS' if idxstage >= 1 else 'ReLU'
            useSE = 'True' if idxstage >= 2 else False

            for i in range(numrepeat):
                if i == 0:
                    inp, outp, stride = input_channel, output_channel, 2
                else:
                    inp, outp, stride = input_channel // 2, output_channel, 1

                blockIndex = architecture[archIndex]
                archIndex += 1
                if blockIndex == 0:
                    print('Shuffle3x3',inp,outp,outp // 2,3,stride,activation,useSE)
                    self.features.append(Shufflenet(inp, outp, base_mid_channels=outp // 2, ksize=3, stride=stride,
                                    activation=activation, useSE=useSE))
                elif blockIndex == 1:
                    print('Shuffle5x5',inp,outp,outp // 2,5,stride,activation,useSE)
                    self.features.append(Shufflenet(inp, outp, base_mid_channels=outp // 2, ksize=5, stride=stride,
                                    activation=activation, useSE=useSE))
                elif blockIndex == 2:
                    print('Shuffle7x7',inp,outp,outp // 2,7,stride,activation,useSE)
                    self.features.append(Shufflenet(inp, outp, base_mid_channels=outp // 2, ksize=7, stride=stride,
                                    activation=activation, useSE=useSE))
                elif blockIndex == 3:
                    print('Xception',inp,outp,outp // 2,stride,activation,useSE)
                    self.features.append(Shuffle_Xception(inp, outp, base_mid_channels=outp // 2, stride=stride,
                                    activation=activation, useSE=useSE))
                else:
                    raise NotImplementedError
                input_channel = output_channel
        assert archIndex == len(architecture)
        self.features = nn.Sequential(*self.features)

        self.conv_last = nn.Sequential(
            nn.Conv2d(input_channel, 1280, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1280),
            HS()
        )
        self.globalpool = nn.AvgPool2d(7)
        self.LastSE = SELayer(1280)
        self.fc = nn.Sequential(
            nn.Linear(1280, 1280, bias=False),
            HS(),
        )
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(nn.Linear(1280, n_class, bias=False))
        self._initialize_weights()

    def forward(self, x):
        x = self.first_conv(x)
        x = self.features(x)
        x = self.conv_last(x)

        x = self.globalpool(x)
        x = self.LastSE(x)

        x = x.contiguous().view(-1, 1280)

        x = self.fc(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name or 'SE' in name:
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
    torch.manual_seed(12345)
    usenpu = True

    if usenpu:
        print('**1')
        architecture = [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]
        model = ShuffleNetV2_Plus(architecture=architecture, model_size='Small')
        
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
        architecture = [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]
        model = ShuffleNetV2_Plus(architecture=architecture, model_size='Small')

        bs = 5
        test_data = torch.rand(bs, 3, 224, 224, requires_grad=True)
        target = torch.randint(1,10,(bs,))
        print(test_data.sum())
        criterion_smooth = CrossEntropyLabelSmooth(1000, 0.1)
        loss_function = criterion_smooth

        test_outputs = model(test_data)
        print(test_outputs.size())
        print(test_outputs.sum())
        loss = loss_function(test_outputs, target)
        loss.backward()

        print('**OK')