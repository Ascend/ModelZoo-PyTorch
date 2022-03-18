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
import sys

import torch
import torch.distributed as dist

from utils import AverageMeter, calculate_accuracy


def val_epoch(epoch,
              data_loader,
              model,
              criterion,
              opt,
              logger,
              tb_writer=None):
    if opt.is_master_node:
        print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            targets = targets.to(opt.device, non_blocking=True)
            outputs = model(inputs.npu())
            loss = criterion(outputs, targets.int())
            acc = calculate_accuracy(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if epoch == 1 and i == 0:
                batch_time.reset()
                data_time.reset()

            if opt.is_master_node:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                        epoch,
                        i + 1,
                        len(data_loader),
                        batch_time=batch_time,
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

    if logger is not None:
        logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})

    if tb_writer is not None:
        tb_writer.add_scalar('val/loss', losses.avg, epoch)
        tb_writer.add_scalar('val/acc', accuracies.avg, epoch)

    return losses.avg
