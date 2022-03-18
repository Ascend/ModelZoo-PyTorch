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
import os
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import joblib
from collections import OrderedDict
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from utils import *
from wide_resnet import WideResNet
from auto_augment import AutoAugment, Cutout
try:
    from apex import amp
except ImportError:
    amp = None
import apex

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--dataset', default='cifar10',
                        choices=['cifar10', 'cifar100'],
                        help='dataset name')
    parser.add_argument('--depth', default=28, type=int)
    parser.add_argument('--width', default=10, type=int)
    parser.add_argument('--cutout', default=False, type=str2bool)
    parser.add_argument('--auto-augment', default=False, type=str2bool)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float)
    parser.add_argument('--milestones', default='60,120,160', type=str)
    parser.add_argument('--gamma', default=0.2, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--nesterov', default=False, type=str2bool)
    parser.add_argument('--device_id', default=5, type=int, help='device_id')
    parser.add_argument('--apex', action='store_true',
                        help='User apex for mixed precision training')
    parser.add_argument('--apex-opt-level', default='O1', type=str,
                        help='For apex mixed precision training'
                             'O0 for FP32 training, O1 for mixed precison training.')
    parser.add_argument('--loss-scale-value', default=1024., type=float,
                        help='loss scale using in amp, default -1 means dynamic')
    args = parser.parse_args()

    return args


def train(args, train_loader, model, criterion, optimizer, epoch, scheduler=None):
    losses = AverageMeter()
    scores = AverageMeter()

    model.train()
    batch_size=128

    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        # from original paper's appendix
        start_time = time.time()
        input = input.npu()
        target = target.npu()

        output = model(input)
        loss = criterion(output, target)

        acc = accuracy(output, target)[0]

        losses.update(loss.item(), input.size(0))
        scores.update(acc.item(), input.size(0))
        fps = batch_size / (time.time() - start_time)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        if args.apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss', losses.avg),
        ('acc', scores.avg),
        ('fps', fps),
    ])

    return log


def validate(args, val_loader, model, criterion):
    losses = AverageMeter()
    scores = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.npu()
            target = target.npu()

            output = model(input)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            scores.update(acc1.item(), input.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
        ('acc', scores.avg),
    ])

    return log


def main():
    args = parse_args()

    if args.name is None:
        args.name = '%s_WideResNet%s-%s' %(args.dataset, args.depth, args.width)
        if args.cutout:
            args.name += '_wCutout'
        if args.auto_augment:
            args.name += '_wAutoAugment'

    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)

    print('Config -----\n')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------\n')

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' %args.name)

    criterion = nn.CrossEntropyLoss().npu()

    cudnn.benchmark = True

    # data loading code
    if args.dataset == 'cifar10':
        transform_train = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        if args.auto_augment:
            transform_train.append(AutoAugment())
        if args.cutout:
            transform_train.append(Cutout())
        transform_train.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        transform_train = transforms.Compose(transform_train)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761)),
        ])

        train_set = datasets.CIFAR10(
            root='~/data',
            train=True,
            download=True,
            transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=128,
            shuffle=True,
            num_workers=8)

        test_set = datasets.CIFAR10(
            root='~/data',
            train=False,
            download=True,
            transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=128,
            shuffle=False,
            num_workers=8)

        num_classes = 10

    elif args.dataset == 'cifar100':
        transform_train = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        if args.auto_augment:
            transform_train.append(AutoAugment())
        if args.cutout:
            transform_train.append(Cutout())
        transform_train.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        transform_train = transforms.Compose(transform_train)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        train_set = datasets.CIFAR100(
            root='~/data',
            train=True,
            download=True,
            transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=128,
            shuffle=True,
            num_workers=8)

        test_set = datasets.CIFAR100(
            root='~/data',
            train=False,
            download=True,
            transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=128,
            shuffle=False,
            num_workers=8)

        num_classes = 100

    device = torch.device(f'npu:{args.device_id}')
    torch.npu.set_device(device)
    # create model
    model = WideResNet(args.depth, args.width, num_classes=num_classes)
    model = model.npu()

    optimizer = apex.optimizers.NpuFusedSGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay)

    if args.apex:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.apex_opt_level,
                                          loss_scale=args.loss_scale_value,
                                          combine_grad=True)

    scheduler = lr_scheduler.MultiStepLR(optimizer,
            milestones=[int(e) for e in args.milestones.split(',')], gamma=args.gamma)

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'acc', 'fps', 'val_loss', 'val_acc'
    ])

    best_acc = 0
    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' %(epoch+1, args.epochs))

        scheduler.step()

        # train for one epoch
        train_log = train(args, train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        val_log = validate(args, test_loader, model, criterion)

        print('\n')
        print('\nloss %.4f - acc %.4f - fps %.4f - val_loss %.4f - val_acc %.4f'
            %(train_log['loss'], train_log['acc'], train_log['fps'], val_log['loss'], val_log['acc']))

        tmp = pd.Series([
            epoch,
            scheduler.get_lr()[0],
            train_log['loss'],
            train_log['acc'],
            train_log['fps'],
            val_log['loss'],
            val_log['acc'],
        ], index=['epoch', 'lr', 'loss', 'acc', 'fps', 'val_loss', 'val_acc'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('models/%s/log.csv' %args.name, index=False)

        if val_log['acc'] > best_acc:
            torch.save(model.state_dict(), 'models/%s/model.pth' %args.name)
            best_acc = val_log['acc']
            print("\n=> saved best model")


if __name__ == '__main__':
    main()
