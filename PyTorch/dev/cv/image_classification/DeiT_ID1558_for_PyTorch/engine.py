#
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
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
#
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils

import random
import pdb
import torch.nn.functional as F
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    cnt = 0

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        cnt = cnt + 1
        if cnt > 100:
            pass
        resolution = [224, 196, 160, 128]
        heads = [12, 10, 8, 6, 4, 3]
        depth = [12, 10, 8, 6]
        max_reso, max_heads, max_depth = resolution[0], heads[0], depth[0]
        random_reso, random_heads, random_depth = random.choice(resolution), random.choice(heads), random.choice(depth)
        min_reso, min_heads, min_depth = resolution[-1], heads[-1], depth[-1]
        samples = samples.to(f'npu:{NPU_CALCULATE_DEVICE}', non_blocking=True)
        targets = targets.to(f'npu:{NPU_CALCULATE_DEVICE}', non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        
        #with torch.npu.amp.autocast():
#            pdb.set_trace()
        T = 1
        max_outputs = model(samples, max_reso, max_heads, max_depth)
        random_outputs = model(samples, random_reso, random_heads, random_depth)
        min_outputs = model(samples, min_reso, min_heads, min_depth)
        max_loss = criterion(samples, max_outputs, targets)
        #diss_random = F.kl_div(F.log_softmax(random_outputs / T, dim=1), F.log_softmax(max_outputs.detach() / T, dim=1), reduction='sum', log_target=True ) * (T * T) / random_outputs.shape[0]
        #diss_min = F.kl_div(F.log_softmax(min_outputs / T, dim=1), F.log_softmax(max_outputs.detach() / T, dim=1), reduction='sum', log_target=True ) * (T * T) / random_outputs.shape[0]
        diss_random = F.kl_div(F.log_softmax(random_outputs / T, dim=1), F.log_softmax(max_outputs.detach() / T, dim=1), reduction='sum') * (T * T) / random_outputs.shape[0]
        diss_min = F.kl_div(F.log_softmax(min_outputs / T, dim=1), F.log_softmax(max_outputs.detach() / T, dim=1), reduction='sum') * (T * T) / random_outputs.shape[0]
        loss = max_loss + diss_random + diss_min
#            print(max_loss, diss_random , diss_min) 
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.npu.synchronize()
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
    n = 0

    for images, target in metric_logger.log_every(data_loader, 10, header):
        n = n + 1
        if n > 2:
            pass
        images = images.to(f'npu:{NPU_CALCULATE_DEVICE}', non_blocking=True)
        target = target.to(f'npu:{NPU_CALCULATE_DEVICE}', non_blocking=True)

        # compute output
        #with torch.npu.amp.autocast():
#             output = model(images, 3, 6)
        output = model(images, 224, 12, 12)
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
