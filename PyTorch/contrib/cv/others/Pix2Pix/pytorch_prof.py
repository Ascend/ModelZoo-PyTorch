# Copyright 2021 Huawei Technologies Co., Ltd
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

# -*- coding: utf-8 -*-
"""pytorch_prof.py
"""

import torch
import torch.optim as optim
import torch.nn as nn
import time
import argparse
from models import networks

def build_model():
    # 请自定义模型并加载预训练模型
    # import torchvision
    # model = torchvision.models.resnet50(pretrained=True)
    model = networks.define_G(3, 3, 64, 'unet_256', 'instance',
                              True, 'normal', 0.02, '[0]')
    return model


def get_raw_data():
    # input_tensor = torch.randn(2, 3, 224, 224)
    input_tensor = torch.randn(1, 3, 256, 256)
    return input_tensor


def criterion(x):
    base_func = nn.CrossEntropyLoss()
    shape_list = x.shape
    N = shape_list[0]
    R = 1
    if len(shape_list) > 1:
        for r in shape_list[1:]:
            R *= r
    T = torch.randint(0,R, size=(N,)).to(x.device)
    if str(T.device).startswith('npu'):
        T = T.int()
    return base_func(x.reshape(N, -1), T)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Prof')
    parser.add_argument('--device', type=str, default='cpu',
                        help='set which type of device used. Support cuda:0(device_id), npu:0(device_id).')
    parser.add_argument('--amp', default=False, action='store_true',
                        help='use amp during prof')
    parser.add_argument('--loss-scale', default=64.0, type=float,
                        help='loss scale using in amp, default 64.0, -1 means dynamic')
    parser.add_argument('--opt-level', default='O2', type=str,
                        help='opt-level using in amp, default O2')
    parser.add_argument('--FusedSGD', default=False, action='store_true',
                        help='use FusedSGD during prof')

    args = parser.parse_args()

    # 1.准备工作
    if args.device.startswith('cuda'):
        torch.cuda.set_device(args.device)
        prof_kwargs = {'use_cuda': True}
    elif args.device.startswith('npu'):
        torch.npu.set_device(args.device)
        prof_kwargs = {'use_npu': True}
    else:
        prof_kwargs = {}

    # 2.构建模型
    model = build_model()
    if args.FusedSGD:
        from apex.optimizers import NpuFusedSGD
        optimizer = NpuFusedSGD(model.parameters(), lr=0.01)
        model = model.to(args.device)
        if args.amp:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level,
                                              loss_scale=None if args.loss_scale == -1 else args.loss_scale,
                                              combine_grad=True)
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        model = model.to(args.device)
        if args.amp:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level,
                                              loss_scale=None if args.loss_scale == -1 else args.loss_scale)

    # 3.生成input
    input_tensor = get_raw_data()
    input_tensor = input_tensor.to(args.device)

    # 先运行一次，保证prof得到的性能是正确的
    def run():
        output_tensor = model(input_tensor)
        loss = criterion(output_tensor)
        optimizer.zero_grad()
        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        return loss
    for i in range(5):
        start_time = time.time()
        loss = run()
        print('iter: %d, loss: %.2f, time: %.2f'%(i, loss, (time.time() - start_time)*1000))

    # 4. 执行forward+profiling
    with torch.autograd.profiler.profile(**prof_kwargs) as prof:
        run()
    print(prof.key_averages().table())
    prof.export_chrome_trace("pytorch_prof_%s.prof" % args.device)