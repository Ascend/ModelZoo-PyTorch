# Copyright 2020 Huawei Technologies Co., Ltd
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

import torch
import time
import os
import sys

import torch
import torch.distributed as dist

from utils import AverageMeter, calculate_accuracy
from apex import amp


def train_epoch(epoch,
                data_loader,
                model,
                criterion,
                optimizer,
                opt,
                current_lr,
                epoch_logger,
                batch_logger,
                tb_writer=None):
    if opt.is_master_node:
        print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    fps = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)
        #if i == 20:
        #    with torch.autograd.profiler.profile(record_shapes=True, use_npu=True) as prof:
        #        targets = targets.to(opt.device, non_blocking=True)
        #        outputs = model(inputs.npu())
        #        loss = criterion(outputs, targets.int())
        #        acc = calculate_accuracy(outputs, targets)
        #        losses.update(loss.item(), inputs.size(0))
        #        accuracies.update(acc, inputs.size(0))
        #        optimizer.zero_grad()
        #        if opt.amp_cfg:
        #            with amp.scale_loss(loss, optimizer) as scaled_loss:
        #                scaled_loss.backward()
        #        else:
        #            loss.backward()
        #        optimizer.step()
        #    prof.export_chrome("Resnet3d_o2.prof")
        #    import sys
        #    sys.exit()
        targets = targets.to(opt.device, non_blocking=True)
        outputs = model(inputs.npu())
        loss = criterion(outputs, targets.int())
        acc = calculate_accuracy(outputs, targets)

        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        optimizer.zero_grad()
        if opt.amp_cfg:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        fps.update(opt.world_size * opt.batch_size / batch_time.val)
        end_time = time.time()
        if i < 2:
            print("step_time = {:.4f}".format(batch_time.val), flush=True)
            
        if epoch == 1 and i == 0:
            batch_time.reset()
            data_time.reset()

        if batch_logger is not None:
            batch_logger.log({
                'epoch': epoch,
                'batch': i + 1,
                'iter': (epoch - 1) * len(data_loader) + (i + 1),
                'loss': losses.val,
                'acc': accuracies.val,
                'lr': current_lr
            })

        if opt.is_master_node:
            print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Fps {fps.val:.3f} ({fps.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(epoch,
                                                         i + 1,
                                                         len(data_loader),
                                                         batch_time=batch_time,
                                                         fps=fps,
                                                         data_time=data_time,
                                                         loss=losses,
                                                         acc=accuracies))

    if opt.distributed:
        loss_sum = torch.tensor([losses.sum],
                                dtype=torch.float32,
                                device=opt.device)
        loss_count = torch.tensor([losses.count],
                                  dtype=torch.float32,
                                  device=opt.device)
        acc_sum = torch.tensor([accuracies.sum],
                               dtype=torch.float32,
                               device=opt.device)
        acc_count = torch.tensor([accuracies.count],
                                 dtype=torch.float32,
                                 device=opt.device)

        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_count, op=dist.ReduceOp.SUM)

        losses.avg = loss_sum.item() / loss_count.item()
        accuracies.avg = acc_sum.item() / acc_count.item()

    if epoch_logger is not None:
        epoch_logger.log({
            'epoch': epoch,
            'loss': losses.avg,
            'acc': accuracies.avg,
            'lr': current_lr
        })

    if tb_writer is not None:
        tb_writer.add_scalar('train/loss', losses.avg, epoch)
        tb_writer.add_scalar('train/acc', accuracies.avg, epoch)
        tb_writer.add_scalar('train/lr', accuracies.avg, epoch)
