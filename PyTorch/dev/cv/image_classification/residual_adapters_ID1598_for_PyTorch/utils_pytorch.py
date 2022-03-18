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
# imdbfolder_coco.py
# created by Sylvestre-Alvise Rebuffi [srebuffi@robots.ox.ac.uk]
# Copyright 漏 The University of Oxford, 2017-2020
# This code is made available under the Apache v2.0 licence, see LICENSE.txt for details

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import os
import time
import numpy as np
from torch.autograd import Variable
import config_task
try:
    from apex import amp
except ImportError:
    apex = None

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


def adjust_learning_rate_and_learning_taks(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every X epochs"""
    if epoch >= args.step2: 
        lr = args.lr * 0.01
    elif epoch >= args.step1:
        lr = args.lr * 0.1
    else:
        lr = args.lr
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Return training classes
    return range(len(args.dataset))


# Training
def train(epoch, tloaders, tasks, net, args, optimizer,list_criterion=None):
    print('\nEpoch: %d' % epoch)
    net.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = [AverageMeter() for i in tasks]
    top1 = [AverageMeter() for i in tasks]
    end = time.time()
    
    loaders = [tloaders[i] for i in tasks]
    min_len_loader = np.min([len(i) for i in loaders])
    train_iter = [iter(i) for i in loaders]
        
    for batch_idx in range(min_len_loader*len(tasks)):
        config_task.first_batch = (batch_idx == 0)
        # Round robin process of the tasks
        current_task_index = batch_idx % len(tasks)
        inputs, targets = (train_iter[current_task_index]).next()
        config_task.task = tasks[current_task_index]
        # measure data loading time
        data_time.update(time.time() - end)
        if args.use_npu:
            inputs, targets = inputs.npu(), targets.npu()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = args.criterion(outputs, targets)
        # measure accuracy and record loss
        (losses[current_task_index]).update(loss.data, targets.size(0))
        _, predicted = torch.max(outputs.data, 1)
        correct = predicted.eq(targets.data).cpu().sum()
        (top1[current_task_index]).update(correct*100./targets.size(0), targets.size(0))     
        # apply gradients
        if args.apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        #loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        batch_size = targets.size(0)
        fps = (batch_size / batch_time.val)
        
        if batch_idx % 50 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'FPS: {fps:.3f}\t'
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                   epoch, batch_idx, min_len_loader*len(tasks), batch_time=batch_time, fps=fps,
                   data_time=data_time))
            for i in range(len(tasks)):
                print('Task {0} : Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc: {top1.val:.3f} ({top1.avg:.3f})'.format(tasks[i], loss=losses[i], top1=top1[i]))

    return [top1[i].avg for i in range(len(tasks))], [losses[i].avg for i in range(len(tasks))]



def test(epoch, loaders, all_tasks, net, best_acc, args, optimizer):
    net.eval()
    losses = [AverageMeter() for i in all_tasks]
    top1 = [AverageMeter() for i in all_tasks]
    print('Epoch: [{0}]'.format(epoch))
    for itera in range(len(all_tasks)):
        i = all_tasks[itera]
        config_task.task = i
        for batch_idx, (inputs, targets) in enumerate(loaders[i]):
            if args.use_npu:
                inputs, targets = inputs.npu(), targets.npu()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = net(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = args.criterion(outputs, targets)
            
            losses[itera].update(loss.data, targets.size(0))
            _, predicted = torch.max(outputs.data, 1)
            correct = predicted.eq(targets.data).cpu().sum()
            top1[itera].update(correct*100./targets.size(0), targets.size(0))
        
        print('Task {0} : Test Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Test Acc {top1.val:.3f} ({top1.avg:.3f})'.format(i, loss=losses[itera], top1=top1[itera]))
    
    # Save checkpoint.
    acc = np.sum([top1[i].avg for i in range(len(all_tasks))])
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, args.ckpdir+'/ckpt'+config_task.mode+args.archi+args.proj+''.join(args.dataset)+'.t7')
        best_acc = acc
    
    return [top1[i].avg for i in range(len(all_tasks))], [losses[i].avg for i in range(len(all_tasks))], best_acc

