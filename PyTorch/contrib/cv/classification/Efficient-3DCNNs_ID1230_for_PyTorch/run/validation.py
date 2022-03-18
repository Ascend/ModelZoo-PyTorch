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
import sys

from run.utils import calculate_accuracy, AverageMeter
from utils import *


def val_epoch(epoch, data_loader, model, criterion, opt, logger, device_ids=0):
    if device_ids == 0:  # distributed master or 1p
        print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if opt.gpu_or_npu == 'npu' and not opt.no_drive:
            targets = targets.to(opt.device, non_blocking=True)
            inputs = inputs.to(opt.device, non_blocking=True)

        elif opt.gpu_or_npu == 'gpu' and not opt.no_drive:
            targets = targets.to(opt.device)
            inputs = inputs.to(opt.device)

        with torch.no_grad():
            inputs = Variable(inputs)
            targets = Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1, 5))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        losses.update(loss.data, inputs.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if (i % 50 == 0) or (i+1 == len(data_loader)):
            if device_ids == 0:  # distributed master or 1p
                print('val: Epoch [{0}][{1}/{2}]\t'
                      'Batch Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
                      'Data Time {data_time.val:.5f} ({data_time.avg:.5f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
                      'Prec@5 {top5.val:.5f} ({top5.avg:.5f})'.format(
                          epoch,
                          i + 1,
                          len(data_loader),
                          batch_time=batch_time,
                          data_time=data_time,
                          loss=losses,
                          top1=top1,
                          top5=top5))

    # if (opt.distributed and dist.get_rank() == 0) or (opt.device_num==1): # distributed master or 1p

    if device_ids == 0:  # distributed master or 1p
        logger.log({
                    'date': time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())),
                    'epoch': epoch,
                    'loss': losses.avg.item(),
                    'prec1': top1.avg.item(),
                    'prec5': top5.avg.item()
                    })

    return losses.avg, top1.avg