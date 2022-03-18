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
import errno
import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import config as cfg
import apex
from apex import amp
from utils.cluster import kmeans
from utils.lr_scheduler import get_scheduler
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

__all__ = ['get_mean_and_std', 'init_params', 'mkdir_p', 'AverageMeter', 'cluster_weights', 'get_optimizer', 'resume_ckpt']


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=128)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

@torch.no_grad()
def cluster_weights(weights, n_clusters):
    """ Initialization of GMM with k-means algorithm, note this procedure may bring
    different initialization results, and the results may be slightly different.
    Args:
        weights:[weight_size]
        n_clusters: 1 + 2^i = K Gaussian Mixture Component number
    Returns:
        [n_clusters-1] initial region saliency obtained by k-means algorthm
    """
    flat_weight = weights.view(-1, 1).npu()
    _tol = 1e-15
    if cfg.IS_NORMAL is True:
        print("skip k-means")
        tmp = torch.rand(n_clusters-1).npu()
        return tmp, tmp , 0.5, tmp, 0.01
    _cluster_idx, region_saliency = kmeans(X=flat_weight, num_clusters=n_clusters, tol=_tol, \
                        distance='euclidean', device=torch.device(f'npu:{NPU_CALCULATE_DEVICE}'), tqdm_flag=False)
    pi_initialization = torch.tensor([torch.true_divide(_cluster_idx.eq(i).sum(), _cluster_idx.numel()) \
                            for i in range(n_clusters)], device=f'npu:{NPU_CALCULATE_DEVICE}')
    zero_center_idx = torch.argmin(torch.abs(region_saliency))
    region_saliency_tmp = region_saliency.clone()
    region_saliency_zero = region_saliency[zero_center_idx]
    region_saliency_tmp[zero_center_idx] = 0.0
    pi_zero = pi_initialization[zero_center_idx]

    sigma_tmp = torch.zeros(n_clusters,1).npu()
    for i in range(flat_weight.size(0)):
        _idx = _cluster_idx[i]
        sigma_tmp[_idx] += (flat_weight[i,0]-region_saliency_tmp[_idx])**2
    sigma_initialization = torch.tensor([torch.true_divide(sigma_tmp[i], _cluster_idx.eq(i).sum()-1) \
                                    for i in range(n_clusters)], device=f'npu:{NPU_CALCULATE_DEVICE}').sqrt()
    sigma_zero = sigma_initialization[zero_center_idx]
    sigma_initialization = sigma_initialization[torch.arange(region_saliency.size(0)).npu() != zero_center_idx]    

    pi_initialization = pi_initialization[torch.arange(region_saliency.size(0)).npu() != zero_center_idx]
    region_saliency = region_saliency[torch.arange(region_saliency.size(0)).npu() != zero_center_idx] # remove zero component center
    return region_saliency, pi_initialization, pi_zero, sigma_initialization, sigma_zero

def get_optimizer(model, args):
    train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr}]
    # optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
    #                             weight_decay=args.weight_decay, nesterov=args.nesterov)
    optimizer = apex.optimizers.NpuFusedSGD(train_params, momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=args.nesterov)
                                
    return optimizer

def resume_ckpt(args, model, train_loader, optimizer, lr_scheduler):
    if not os.path.isfile(args.resume):
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
    if args.only_inference:
        checkpoint = torch.load(args.resume)
        if args.cuda:
            model.module.load_state_dict(checkpoint)
            model = model.npu()
        else:
            model.load_state_dict(checkpoint['state_dict'])
            model.init_mask_params()
            optimizer = get_optimizer(model, args)
            lr_scheduler = get_scheduler(args, optimizer, \
                    args.lr, len(train_loader))
        best_pred = 0.0
        print("=> loaded checkpoint '{}'".format(args.resume))
    else:
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        if args.cuda:
            model.module.load_state_dict(checkpoint['state_dict'])
            model.module.init_mask_params()
            optimizer = get_optimizer(model, args)
            lr_scheduler = get_scheduler(args, optimizer, \
                    args.lr, len(train_loader))
            model = model.npu()
        else:
            model.load_state_dict(checkpoint['state_dict'])
            model.init_mask_params()
        optimizer.load_state_dict(checkpoint['optimizer'])
        if args.rt:
            best_pred = 0.0
        else:
            best_pred = checkpoint['best_pred']
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        print("Checkpoint Top-1 Acc: ", checkpoint['best_pred'])
    return model, optimizer, lr_scheduler, best_pred

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L363-L384
    """
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
