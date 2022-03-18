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
import torch
from torch.autograd import Variable
import time
import os
import sys

from run.utils import AverageMeter, calculate_accuracy


def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger, device_ids=0):
    if device_ids == 0:  # distributed master or 1p
        print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    tot_time = AverageMeter()

    end_time = time.time()

    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if opt.gpu_or_npu == 'npu' and not opt.no_drive:
            targets = targets.to(opt.device, non_blocking=True)
            inputs = inputs.to(opt.device, non_blocking=True)

        elif opt.gpu_or_npu == 'gpu' and not opt.no_drive:
            targets = targets.to(opt.device)
            inputs = inputs.to(opt.device)

        inputs = Variable(inputs)
        targets = Variable(targets)

        if opt.use_prof and i == 5:
            if opt.gpu_or_npu == 'gpu':
                with torch.autograd.profiler.profile(use_cuda=True) as prof:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    losses.update(loss.data, inputs.size(0))
                    prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1,5))
                    top1.update(prec1, inputs.size(0))
                    top5.update(prec5, inputs.size(0))

                    optimizer.zero_grad()

                    if opt.use_apex == 1:
                        from apex import amp
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()

                    optimizer.step()

                # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                prof.export_chrome_trace(os.path.join(opt.result_path, "output.prof"))
            elif opt.gpu_or_npu == 'npu':
                with torch.autograd.profiler.profile(use_npu=True) as prof:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    losses.update(loss.data, inputs.size(0))
                    prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1, 5))
                    top1.update(prec1, inputs.size(0))
                    top5.update(prec5, inputs.size(0))

                    optimizer.zero_grad()

                    if opt.use_apex == 1:
                        from apex import amp
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()

                    optimizer.step()

                # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                prof.export_chrome_trace(os.path.join(opt.result_path, "output.prof"))
        else:
            outputs = model(inputs)

            loss = criterion(outputs, targets)

            losses.update(loss.data, inputs.size(0))
            prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1, 5))
            top1.update(prec1, inputs.size(0))
            top5.update(prec5, inputs.size(0))

            optimizer.zero_grad()

            if opt.use_apex == 1:
                from apex import amp
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()

        current_batch_time = time.time() - end_time
        batch_time.update(current_batch_time)
        end_time = time.time()

        # measure elapsed time
        fps = opt.batch_size * opt.device_num / current_batch_time
        if i >= 2:
            tot_time.update(current_batch_time)

        if device_ids == 0:  # distributed master or 1p
            batch_logger.log({
                'date': time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())),
                'epoch': epoch,
                'batch': i + 1,
                'iter': (epoch - 1) * len(data_loader) + (i + 1),
                'fps': fps,
                'loss': losses.val.item(),
                'prec1': top1.val.item(),
                'prec5': top5.val.item(),
                'lr': optimizer.param_groups[0]['lr']
            })
            if (i % 50 == 0) or (i+1 == len(data_loader)):
                print('train: Epoch [{0}][{1}/{2}]\t lr: {lr:.5f}\t'
                      'Batch FPS = {fps:.4f}\t'
                      'Batch Time {batch_time.avg:.3f}\t'
                      'Data Time {data_time.avg:.3f}\t'
                      'Loss {loss.avg:.4f}\t'
                      'Prec@1 {top1.avg:.5f}\t'
                      'Prec@5 {top5.avg:.5f}'.format(
                          epoch,
                          i,
                          len(data_loader),
                          fps=fps,
                          batch_time=batch_time,
                          data_time=data_time,
                          loss=losses,
                          top1=top1,
                          top5=top5,
                          lr=optimizer.param_groups[0]['lr']))

    epoch_fps = opt.batch_size * opt.device_num / tot_time.avg

    if device_ids == 0:  # distributed master or 1p
        epochlog = {
            'date': time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())),
            'epoch': epoch,
            'fps': epoch_fps,
            'loss': losses.avg.item(),
            'prec1': top1.avg.item(),
            'prec5': top5.avg.item(),
            'lr': optimizer.param_groups[0]['lr']
        }
        epoch_logger.log(epochlog)
        print(epochlog)
