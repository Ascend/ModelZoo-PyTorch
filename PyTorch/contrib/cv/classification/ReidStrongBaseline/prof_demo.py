# -*- coding: utf-8 -*-
# Copyright 2020 Huawei Technologies Co., Ltd
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

"""prof_demo.py
"""

import torch
if torch.__version__ >= '1.8.1':
    import torch_npu
import torch.nn as nn
import torch.optim as optim
import time
import argparse
from config import cfg
from layers import make_loss_with_center

def build_model():
    from modeling.baseline import Baseline
    model = Baseline(751, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE)
    return model


def get_raw_data():
    input_tensor = torch.randn(64, 3, 256, 128)
    print(input_tensor.dtype)
    target = torch.linspace(5, 624, steps = 64)
    target = target.to(torch.int64)
    return input_tensor, target

def make_optimizer_with_center(cfg, model, center_criterion):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    params_dict = {}
    for param in params:
        p, l , w = param["params"], param["lr"], param["weight_decay"] 
        k = "{}_{}".format(l, w)
        
        if k not in params_dict:
            params_dict[k] = []
        params_dict[k].append(p[0])

    params = []
    for k in params_dict:
        lr, weight_decay = map(float, k.split("_"))
        params += [{"params": params_dict[k], "lr": lr, "weight_decay": weight_decay}]

    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        if  args.device.startswith('npu'):
            from apex.optimizers import NpuFusedSGD
            optimizer = NpuFusedSGD(params, momentum=cfg.SOLVER.MOMENTUM)
        else:
            optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'Adam':
        if  args.device.startswith('npu'):
            from apex.optimizers import NpuFusedAdam
            optimizer = NpuFusedAdam(params)
        else:
            optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    if  args.device.startswith('npu'):
        from apex.optimizers import NpuFusedSGD
        optimizer_center = NpuFusedSGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)
    else:
        optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)
    return optimizer, optimizer_center

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Prof')
    parser.add_argument('--device', type=str, default='cpu',
                        help='set which type of device used. Support cpu, cuda:0(device_id), npu:0(device_id).')
    parser.add_argument('--amp', default=False, action='store_true',
                        help='use amp during prof')
    parser.add_argument('--loss-scale', default='dynamic',
                        help='loss scale using in amp, default 64.0, -1 means dynamic')
    parser.add_argument('--opt-level', default='O2', type=str,
                        help='opt-level using in amp, default O2')
    parser.add_argument('--FusedAdam', default=False, action='store_true',
                        help='use FusedAdam during prof')
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

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

    loss_func, center_criterion = make_loss_with_center(cfg, 751)  # modified by gu
    optimizer, optimizer_center = make_optimizer_with_center(cfg, model, center_criterion)#waiting for additional
    model = model.to(args.device)
    if args.amp:
        model, [optimizer, optimizer_center] = amp.initialize(model, [optimizer, optimizer_center], opt_level=args.opt_level,
                                          loss_scale=None if args.loss_scale == -1 else args.loss_scale)

    input_tensor, target = get_raw_data()
    input_tensor = input_tensor.to(args.device)
    target = target.to(args.device)

    def run(loss_func, center_criterion):
        score, feat = model(input_tensor) 
        loss = loss_func(score, feat, target)
        center_loss_weight = cfg.SOLVER.CENTER_LOSS_WEIGHT
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        if args.amp:
            with amp.scale_loss(loss, [optimizer, optimizer_center]) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        for param in center_criterion.parameters():
            param.grad.data *= (1. / center_loss_weight)
        optimizer_center.step()
        return loss

    for i in range(5):
        start_time = time.time()
        loss = run(loss_func, center_criterion)
        print('iter: %d, loss: %.2f, time: %.2f'%(i, loss, (time.time() - start_time)*1000))

    # 4. profiling
    with torch.autograd.profiler.profile(**prof_kwargs) as prof:
        run(loss_func, center_criterion)
    # print(prof.key_averages().table())
    prof.export_chrome_trace("pytorch_prof_%s.prof" % args.device + ('_amp' if args.amp else ''))
