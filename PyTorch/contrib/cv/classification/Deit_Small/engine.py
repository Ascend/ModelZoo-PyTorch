# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
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
"""
Train and eval functions used in main.py
"""
import time
from typing import Iterable, Optional

import torch
import numpy as np
from apex import amp

from mixup import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils

TIME_ACC_SKIP = 5


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    end = time.time()
    cnt = 0

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        dt = time.time() - end

        samples = samples.to(device, non_blocking=True)
        if 'npu' in str(device):
            targets = targets.to(torch.int32)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        outputs = model(samples)
        loss = criterion(samples, outputs, targets)
        loss_value = loss.item()
        optimizer.zero_grad()

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        optimizer.step()

        torch.npu.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if cnt < TIME_ACC_SKIP:
            cnt += 1
        else:
            if "data_time" not in metric_logger.meters:
                metric_logger.add_meter('data_time', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
                metric_logger.add_meter('batch_time', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            metric_logger.update(data_time=dt)
            batch_time = time.time() - end
            metric_logger.update(batch_time=batch_time)
            
            batch_size = samples.shape[0]
            metric_logger.update(fps=batch_size * utils.get_world_size() / batch_time)

        end = time.time()
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
        if 'npu' in str(device):
            target = target.to(torch.int32)
        target = target.to(device, non_blocking=True)

        # compute output
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
