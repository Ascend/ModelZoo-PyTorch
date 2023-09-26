# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import logging
import sys
import torch
from apex import amp
from core.evaluate import accuracy
try:
    from torch_npu.utils.profiler import Profile
except ImportError:
    print("Profile not in torch_npu.utils.profiler now.. Auto Profile disabled.", flush=True)
    class Profile:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def end(self):
            pass


logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict, device_num, bs, stop_step):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()


    # switch to train mode
    model.train()
    list1 = []
    list2 = []
    list3 = []
    end = time.time()
    profile = Profile(start_step=int(os.getenv('PROFILE_START_STEP', 10)),
                      profile_type=os.getenv('PROFILE_TYPE'))

    for i, (input, target) in enumerate(train_loader):
        if stop_step:
            # reduce time for 1p perf test
            if i > 200:
                break
        # measure data loading time
        data_time.update(time.time() - end)
        start_time = time.time()
        profile.start()
        # compute output
        input = input.npu()
        output = model(input)
        target = target.npu(non_blocking=True)
       
        loss = criterion(output, target)
        loss = loss.npu()

        # compute gradient and do update step
        optimizer.zero_grad()
        if os.getenv('ALLOW_FP32') or os.getenv('ALLOW_HF32'):
            loss.backward()
        else:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        optimizer.step()
        profile.end()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        prec1, prec5 = accuracy(output, target, (1, 5))

        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i < 2:
            print("step_time = %.4f" % (time.time() - start_time), flush=True)
        
        list1.append(batch_time.val)
        list2.append(losses.val)
        list3.append(top1.val)
        
        if i % 100 == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t\t' \
                  'Time {batch_time.val:.3f}s\t\t' \
                  'FPS {fps:.1f}\t\t' \
                  'Data {data_time.val:.3f}s\t\t' \
                  'Loss {loss.val:.5f}\t\t' \
                  'Accuracy@1 {top1.val:.3f}\t\t' \
                  'Accuracy@5 {top5.val:.3f}\t\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      fps=device_num*bs/batch_time.val,
                      data_time=data_time, loss=losses, top1=top1, top5=top5)
            logger.info(msg)
            
            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer.add_scalar('train_top1', top1.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1
    
    # data avg 
    batch_time_avg = sum(list1[5:-5]) / (len(list1)-10)
    fps_avg = device_num*bs/batch_time_avg
    loss_avg = sum(list2[5:]) / (len(list2)-5)
    acc1_avg = sum(list3[5:]) / (len(list3)-5)
    msg_avg = '\nData Average :\t\t' \
              'Time_Avg {batch_time_avg:.3f}s\t\t' \
              'Fps_Avg {fps_avg:.1f}\t\t' \
              'Loss {loss_avg:.5f}\t\t' \
              'Accuracy@1_Avg {acc1_avg:.3f}\t\t'.format(
               batch_time_avg=batch_time_avg, fps_avg=fps_avg, loss_avg=loss_avg, acc1_avg=acc1_avg)  
    logger.info(msg_avg)


def validate(config, val_loader, model, criterion, output_dir, tb_log_dir,
             writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            # compute output
            
            input = input.npu()
            output = model(input)
            target = target.npu(non_blocking=True)
          
            loss = criterion(output, target)
            loss = loss.npu()
            
            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            prec1, prec5 = accuracy(output, target, (1, 5))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        msg = 'Test: Time {batch_time.avg:.3f}\t' \
              'Loss {loss.avg:.4f}\t' \
              'Error@1 {error1:.3f}\t' \
              'Error@5 {error5:.3f}\t' \
              'Accuracy@1 {top1.avg:.3f}\t' \
              'Accuracy@5 {top5.avg:.3f}\t'.format(
                  batch_time=batch_time, loss=losses, top1=top1, top5=top5,
                  error1=100-top1.avg, error5=100-top5.avg)
        logger.info(msg)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', losses.avg, global_steps)
            writer.add_scalar('valid_top1', top1.avg, global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
