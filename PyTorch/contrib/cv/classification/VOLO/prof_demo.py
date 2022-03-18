"""
BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Copyright 2020 Huawei Technologies Co., Ltd

Licensed under the BSD 3-Clause License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://spdx.org/licenses/BSD-3-Clause.html

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""prof_demo.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import argparse
from timm.models import create_model
from timm.loss import SoftTargetCrossEntropy
from timm.optim import create_optimizer
import models
from apex import amp

def build_model():
    model = create_model(
        'volo_d1',
        pretrained=False,
        num_classes=None,
        drop_rate=0.0,
        drop_connect_rate=None,  # DEPRECATED, use drop_path
        drop_path_rate=0.1,
        drop_block_rate=None,
        global_pool=None,
        bn_tf=False,
        bn_momentum=None,
        bn_eps=None,
        scriptable=False,
        checkpoint_path='',
        img_size=224)
    return model


def get_raw_data():
    input_tensor = torch.randn(128, 3, 224, 224)
    target = torch.randn(128, 1000)
    #target = torch.linspace(5, 624, steps = 128)
    #target = target.to(torch.int64)
    return input_tensor, target

def criterion(x, y=None):
    return x.sum()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Prof')
    parser.add_argument('--device', type=str, default='cpu',
                        help='set which type of device used. Support cpu, cuda:0(device_id), npu:0(device_id).')
    parser.add_argument('--amp', default=False, action='store_true',
                        help='use amp during prof')
    parser.add_argument('--loss-scale', default=64.0, type=float,
                        help='loss scale using in amp, default 64.0, -1 means dynamic')
    parser.add_argument('--opt-level', default='O2', type=str,
                        help='opt-level using in amp, default O2')
    parser.add_argument('--FusedAdam', default=False, action='store_true',
                        help='use FusedAdam during prof')
    parser.add_argument('--lr', type=float, default=1.6e-3, metavar='LR',
                        help='learning rate (default: 1.6e-3)')
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: None, use opt default)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Optimizer momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,    
                        help='weight decay (default: 0.05)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--clip-mode', type=str, default='norm',
                        help='Gradient clipping mode. One of ("norm", "value", "agc")')
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.device.startswith('cuda'):
        torch.cuda.set_device(args.device)
        prof_kwargs = {'use_cuda': True}
    elif args.device.startswith('npu'):
        torch.npu.set_device(args.device)
        prof_kwargs = {'use_npu': True}
    else:
        prof_kwargs = {}

    if args.amp:
        from apex import amp

    model = build_model()

    optimizer = create_optimizer(args, model)
    model = model.to(args.device)
    # if args.amp:
    model, optimizer = amp.initialize(model, optimizer, opt_level='O2', loss_scale=None if args.loss_scale == -1 else args.loss_scale, combine_grad=True)

    input_tensor, target = get_raw_data()
    input_tensor = input_tensor.to(args.device)
    target = target.to(args.device)
    train_loss_fn = SoftTargetCrossEntropy().npu()

    def run(train_loss_fn):
        output,_,_ = model(input_tensor) 
        loss = train_loss_fn(output, target) 
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
        loss = run(train_loss_fn)
        print('iter: %d, loss: %.2f, time: %.2f'%(i, loss, (time.time() - start_time)*1000))
    torch.npu.synchronize()

    # 4. profiling
    with torch.autograd.profiler.profile(**prof_kwargs) as prof:
        run(train_loss_fn)
    print(prof.key_averages().table())
    prof.export_chrome_trace("pytorch_prof_%s.prof" % args.device + ('_amp' if args.amp else ''))
