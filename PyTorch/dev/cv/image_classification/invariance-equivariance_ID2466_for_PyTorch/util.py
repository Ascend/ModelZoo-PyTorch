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
from __future__ import absolute_import
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from dataloader import get_dataloaders
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
    

class BCEWithLogitsLoss(nn.Module):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None, num_classes=64):
        super(BCEWithLogitsLoss, self).__init__()
        self.num_classes = num_classes
        self.criterion = nn.BCEWithLogitsLoss(weight=weight, 
                                              size_average=size_average, 
                                              reduce=reduce, 
                                              reduction=reduction,
                                              pos_weight=pos_weight)
    def forward(self, input, target):
        target_onehot = F.one_hot(target, num_classes=self.num_classes)
        return self.criterion(input, target_onehot)
    

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


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def rotrate_concat(inputs):
    out = None
    for x in inputs:
        x_90 = x.transpose(2,3).flip(2)
        x_180 = x.flip(2).flip(3)
        x_270 = x.flip(2).transpose(2,3)
        if out is None:
            out = torch.cat((x, x_90, x_180, x_270),0)
        else:
            out = torch.cat((out, x, x_90, x_180, x_270),0)
    return out

    
class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, title=None, resume=False): 
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume: 
                self.file = open(fpath, 'r') 
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')  
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume: 
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()


    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def plot(self, names=None):   
        names = self.names if names == None else names
        numbers = self.numbers
        for _, name in enumerate(names):
            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))
        plt.legend([self.title + '(' + name + ')' for name in names])
        plt.grid(True)
        

    def close(self):
        if self.file is not None:
            self.file.close()
            
                   
def generate_final_report(model, opt, wandb):
    from eval.meta_eval import meta_test
    
    opt.n_shots = 1
    train_loader, val_loader, meta_testloader, meta_valloader, _, _ = get_dataloaders(opt)
    
    #validate
    meta_val_acc, meta_val_std = meta_test(model, meta_valloader)
    
    meta_val_acc_feat, meta_val_std_feat = meta_test(model, meta_valloader, use_logit=False)

    #evaluate
    meta_test_acc, meta_test_std = meta_test(model, meta_testloader)
    
    meta_test_acc_feat, meta_test_std_feat = meta_test(model, meta_testloader, use_logit=False)
        
    print('Meta Val Acc : {:.4f}, Meta Val std: {:.4f}'.format(meta_val_acc, meta_val_std))
    print('Meta Val Acc (feat): {:.4f}, Meta Val std (feat): {:.4f}'.format(meta_val_acc_feat, meta_val_std_feat))
    print('Meta Test Acc: {:.4f}, Meta Test std: {:.4f}'.format(meta_test_acc, meta_test_std))
    print('Meta Test Acc (feat): {:.4f}, Meta Test std (feat): {:.4f}'.format(meta_test_acc_feat, meta_test_std_feat))
    
    
    wandb.log({'Final Meta Test Acc @1': meta_test_acc,
               'Final Meta Test std @1': meta_test_std,
               'Final Meta Test Acc  (feat) @1': meta_test_acc_feat,
               'Final Meta Test std  (feat) @1': meta_test_std_feat,
               'Final Meta Val Acc @1': meta_val_acc,
               'Final Meta Val std @1': meta_val_std,
               'Final Meta Val Acc   (feat) @1': meta_val_acc_feat,
               'Final Meta Val std   (feat) @1': meta_val_std_feat
              })

    
    opt.n_shots = 5
    train_loader, val_loader, meta_testloader, meta_valloader, _, _ = get_dataloaders(opt)
    
    #validate
    meta_val_acc, meta_val_std = meta_test(model, meta_valloader)
    
    meta_val_acc_feat, meta_val_std_feat = meta_test(model, meta_valloader, use_logit=False)

    #evaluate
    meta_test_acc, meta_test_std = meta_test(model, meta_testloader)
    
    meta_test_acc_feat, meta_test_std_feat = meta_test(model, meta_testloader, use_logit=False)
        
    print('Meta Val Acc : {:.4f}, Meta Val std: {:.4f}'.format(meta_val_acc, meta_val_std))
    print('Meta Val Acc (feat): {:.4f}, Meta Val std (feat): {:.4f}'.format(meta_val_acc_feat, meta_val_std_feat))
    print('Meta Test Acc: {:.4f}, Meta Test std: {:.4f}'.format(meta_test_acc, meta_test_std))
    print('Meta Test Acc (feat): {:.4f}, Meta Test std (feat): {:.4f}'.format(meta_test_acc_feat, meta_test_std_feat))

    wandb.log({'Final Meta Test Acc @5': meta_test_acc,
               'Final Meta Test std @5': meta_test_std,
               'Final Meta Test Acc  (feat) @5': meta_test_acc_feat,
               'Final Meta Test std  (feat) @5': meta_test_std_feat,
               'Final Meta Val Acc @5': meta_val_acc,
               'Final Meta Val std @5': meta_val_std,
               'Final Meta Val Acc   (feat) @5': meta_val_acc_feat,
               'Final Meta Val std   (feat) @5': meta_val_std_feat
              })