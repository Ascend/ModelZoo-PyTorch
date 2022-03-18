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

# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
Modified from: https://github.com/facebookresearch/deit
"""
import math
import sys
import time
import torch
from typing import Iterable, Optional
from apex import amp
from mixup import Mixup
from timm.utils import accuracy, ModelEma, AverageMeter

import utils
from losses import DistillationLoss

def train_one_epoch_npu(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, surgery=None, batch_frames=1024):

    model.train(set_training_mode)
    if surgery:
        model.module.patch_embed.eval()

    metric_logger = utils.MetricLogger_npu(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue_npu(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    optimizer.zero_grad()
    idx = 0
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    FPS = AverageMeter()
    start_time = 0.0
    end_time = time.time()
    for batch in metric_logger.log_every(data_loader, print_freq, header):
        idx = idx + 1
        if idx < 5:
            start_time = time.time()
        samples, targets = batch[0], batch[1]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        outputs = model(samples)
        loss = criterion(samples, outputs, targets)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        clip_grad = max_norm
        parameters = amp.master_params(optimizer)
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        if clip_grad is not None:
            assert parameters is not None
            torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
        optimizer.step()
        optimizer.zero_grad()
        torch.npu.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        if idx % print_freq == 0:
            memory_used = torch.npu.max_memory_allocated() / (1024.0 * 1024.0)
            print(
                f'Training: [{idx}/{num_steps}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'FPS {FPS.val:.3f} ({FPS.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if idx >= 5:
            time_step = time.time() - end_time
            batch_time.update(time_step)
            FPS.update(batch_frames / float(time_step))
        end_time = time.time()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    #torch.npu.synchronize()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, FPS.avg
def train_without_ddp(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, surgery=None,batch_frames=1024):
    model.train(set_training_mode)
    if surgery:
        model.module.patch_embed.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    optimizer.zero_grad()
    idx = 0
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    FPS = AverageMeter()
    end_time = time.time()
    for batch in metric_logger.log_every(data_loader, print_freq, header):
        idx = idx + 1
        if idx < 5:
            end_time = time.time()
        samples, targets = batch[0], batch[1]
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        outputs = model(samples)
        loss = criterion(samples, outputs, targets)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        loss.backward()
        clip_grad = max_norm
        parameters = amp.master_params(optimizer)
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        if clip_grad is not None:
            assert parameters is not None
            torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        if idx >= 5:
            time_step = time.time() - end_time
            batch_time.update(time_step)
            FPS.update(batch_frames / float(time_step))
        end_time = time.time()
        if idx % print_freq == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            print(
                f'Training: [{idx}/{num_steps}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'FPS {FPS.val:.3f} ({FPS.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    torch.npu.synchronize()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, FPS.avg

@torch.no_grad()
def evaluate_npu(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger_npu(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images, target = batch[0], batch[1]

        images = images.to(device, non_blocking=True)
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
    torch.npu.synchronize()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

