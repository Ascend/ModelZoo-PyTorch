# encoding=utf-8
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

import math
import sys
import time
from typing import Iterable, Optional
import os

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from levit.losses_levit import DistillationLoss
import utils
try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

import torch_npu

#获取Iterable的长度：
def count(iterable):
    c=0
    for el in iterable: c+=1
    return c


def train_one_epoch(args,
                    model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    clip_grad: float = 0,
                    clip_mode: str = 'norm',
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    mode1 = False
    mode2 = False  # True
    mode4 = False
    CTPEP = False
    num_steps = len(data_loader)
    start = time.time()
    step_n = 0
    if epoch < 200:
        model.module.stage_wise_prune = False
        model.module.set_learn_tradeoff(False)
    else:
        model.module.stage_wise_prune = True
        model.module.set_learn_tradeoff(True)

    for samples, targets in metric_logger.log_every(
            data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # 不计算前5个epoch时间内，若需要生成prof的时候，把下面这一段注释掉
        step_n += 1
        if step_n < 5:
            start = time.time()

        #节约时间，计算1000个step的fps，然后输出
        if step_n % 1000 == 999 and args.train_type == 'fps':
            timm_999=time.time()-start
            Fps_step = 995 * args.batch_size * utils.get_world_size() / float(timm_999)
            print("fps:", Fps_step)
            sys.exit()



        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)


        if True:  # with torch.cuda.amp.autocast():
            # outputs = model(samples)
            if (mode1 or mode4):
                outputs = model(samples, epoch)
                loss = criterion(samples, outputs, targets)  # net1distill
            elif (mode2 or CTPEP):
                outputs = model(samples, epoch)
                loss = criterion(samples, outputs, targets)
            else:
                outputs = model(samples)
                loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order

        loss_scaler(loss, optimizer, clip_grad=clip_grad, clip_mode=clip_mode,
                    parameters=model.parameters(), create_graph=is_second_order)
        torch_npu.npu.synchronize()

        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # 不用prof生成时的代码结束位置
    epoch_time = time.time() - start
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # 计算每个epoch的fps
    Fps_epoch = (num_steps - 5) * args.batch_size * utils.get_world_size() / float(epoch_time)
    print("Averaged stats:", metric_logger,"fps:",Fps_epoch)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()},Fps_epoch


@torch.no_grad()
def evaluate(data_loader, model, device, epoch=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    mode1 = False
    mode2 = False
    mutual = False  # True
    cls = True
    mode4 = False
    CTPEP = False  # False

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if True: #with torch.cuda.amp.autocast():这里添加了if TRUE，考虑到NPU不能用CUDA的混合精度训练
            if mode1:
                output = model(images, epoch)
                acc1, acc5 = accuracy(output[1], target, topk=(1, 5))
                acc12, acc52 = accuracy(output[2], target, topk=(1, 5))
                acc13, acc53 = accuracy(output[3], target, topk=(1, 5))
                print("net 2 accuracy: {}, {}, net 3 accuracy: {}, {}, net 4 accuracy: {}, {}".format(acc1.item(),
                                                                                                      acc5.item(),
                                                                                                      acc12.item(),
                                                                                                      acc52.item(),
                                                                                                      acc13.item(),
                                                                                                      acc53.item()))
                output = output[0]
            elif mode2:
                output = model(images, epoch)
                acc1, acc5 = accuracy(output[1], target, topk=(1, 5))
                acc12, acc52 = accuracy(output[2], target, topk=(1, 5))
                acc13, acc53 = accuracy(output[3], target, topk=(1, 5))
                acc14, acc54 = accuracy(output[4], target, topk=(1, 5))
                print(
                    "net 2 accuracy: {}, {}, net 3 accuracy: {}, {}, net 4 accuracy: {}, {}, net merge accuracy: {}, {}".format(
                        acc1.item(), acc5.item(), acc12.item(), acc52.item(), acc13.item(), acc53.item(), acc14.item(),
                        acc54.item()))
                output = output[0]
            elif mode4:
                output = model(images, epoch)
                acc1, acc5 = accuracy(output[1], target, topk=(1, 5))
                acc12, acc52 = accuracy(output[2], target, topk=(1, 5))
                print("net 2 accuracy: {}, {}, net 3 accuracy: {}, {}".format(acc1.item(), acc5.item(), acc12.item(),
                                                                              acc52.item()))
                output = output[0]
            elif mutual:
                output = model(images)
                acc1, acc5 = accuracy(output[1], target, topk=(1, 5))
                print("net depth accuracy: {}, {}".format(acc1, acc5))
                output = output[0]
            elif cls:
                if CTPEP:
                    output = model(images, epoch)
                else:
                    output = model(images)
            else:
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
    print(output.mean().item(), output.std().item())

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
