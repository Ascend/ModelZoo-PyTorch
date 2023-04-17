# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
import os
import time
from typing import Iterable
from contextlib import suppress

import torch

import util.misc as misc
import util.lr_sched as lr_sched
from torch_npu.utils.profiler import Profile

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    amp_autocast=suppress,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    start_FPS = time.time()
    profile = Profile(start_step=int(os.getenv('PROFILE_START_STEP', 10)),
                      profile_type=os.getenv('PROFILE_TYPE'))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if os.environ.get("CONTROL_STEPS"):
            if data_iter_step > int(os.environ.get("CONTROL_STEPS")):
                break

        if data_iter_step == 40:
            start_FPS = time.time()

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        profile.start()
        with amp_autocast():
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.npu.synchronize()
        profile.end()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    epoch_time = time.time() - start_FPS
    epoch_FPS = (len(data_loader) - 40) * args.batch_size * args.world_size / float(epoch_time)
    print(f"train_one_epoch FPS: {epoch_FPS}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, epoch_FPS


def profiling(model: torch.nn.Module,
              data_loader: Iterable, optimizer: torch.optim.Optimizer,
              loss_scaler,
              args=None):

    # switch to train mode
    model.train(True)
    accum_iter = args.accum_iter
    optimizer.zero_grad()

    def update(model, images, optimizer, step):
        loss, _, _ = model(images, mask_ratio=args.mask_ratio)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(step + 1) % accum_iter == 0)
        if (step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.npu.synchronize()

    for step, (images, _) in enumerate(data_loader):
        if args.device == 'npu':
            loc = 'npu:{}'.format(args.local_rank)
            images = images.to(loc, non_blocking=True).to(torch.float)
        else:
            images = images.cuda(args.locak_rank, non_blocking=True)

        if step < 100:
            update(model, images, optimizer, step)
        else:
            if args.device == 'npu':
                with torch.autograd.profiler.profile(use_npu=True) as prof:
                    update(model, images, optimizer, step)
            else:
                with torch.autograd.profiler.profile(use_cuda=True) as prof:
                    update(model, images, optimizer, step)
            break

    print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    prof.export_chrome_trace("./prof/910A_1p_pretrain.prof")

def cann_profiling(model: torch.nn.Module,
              data_loader: Iterable, optimizer: torch.optim.Optimizer,
              loss_scaler,
              args=None):

    cann_profiling_path = './cann_profiling_v2'
    if not os.path.exists(cann_profiling_path):
        os.makedirs(cann_profiling_path)
    # switch to train mode
    model.train(True)
    accum_iter = args.accum_iter
    optimizer.zero_grad()

    def update(model, images, optimizer, step):
        loss, _, _ = model(images, mask_ratio=args.mask_ratio)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(step + 1) % accum_iter == 0)
        if (step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.npu.synchronize()

    for step, (images, _) in enumerate(data_loader):
        if args.device == 'npu':
            loc = 'npu:{}'.format(args.local_rank)
            images = images.to(loc, non_blocking=True).to(torch.float)
        else:
            images = images.cuda(args.locak_rank, non_blocking=True)

        if step < 100:
            update(model, images, optimizer, step)
        else:
            with torch.npu.profile(cann_profiling_path):
                update(model, images, optimizer, step)
            break
