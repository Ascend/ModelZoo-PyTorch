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
"""
Train and eval functions used in main.py
"""
# -*- coding: utf-8 -*-

import math
import sys
from typing import Iterable, Optional

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd
import torch
# import torch_npu

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    max_steps=None,
                    set_training_mode=True):
    model.train(set_training_mode)
    # 设置自定义的指标记录对象
    if 'npu' in str(device):
        use_npu = True
    else:
        use_npu = False
    metric_logger = utils.MetricLogger(delimiter="  ", use_npu=use_npu)
    # 　在该对象中加入学习率
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}', use_npu=use_npu))
    header = 'Epoch: [{}]'.format(epoch)
    # 打印频率
    print_freq = 10
    steps = 0
    import pdb
    #pdb.set_trace()
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        steps += 1
        # print(type(max_steps))
        if max_steps and steps > int(max_steps):
            sys.exit(0)
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # print(f'tensor data shape:{samples.size()}')
        # print(f'tensor target shape:{targets.size()}')
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        # 在 autocast enable 区域运行 forward
        # with torch.cuda.amp.autocast():
        # print(f'tensor target shape:{targets.size()}')
        outputs = model(samples)
        loss = criterion(samples, outputs, targets)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        if 'npu' in str(device):
            torch.npu.synchronize()
        else:
            torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        # with torch.cuda.amp.autocast():
        output = model(images)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
