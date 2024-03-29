# -*- coding: utf-8 -*-
# Copyright 2020 Huawei Technologies Co., Ltd
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


from __future__ import print_function

import os
import argparse
import csv
import time

import numpy as np
import pandas as pd

import torch
import torchvision

# add NPU adapter
import torch_npu
from torch_npu.contrib import transfer_to_npu

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import bugfix
from randomaug import RandAugment

import apex
from apex import amp


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
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

def train(net, device, optimizer, epoch, criterion_ls, trainloader):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        # Train with amp
        outputs = net(inputs)
        loss = criterion_ls(outputs, targets)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Avoid confusion caused by printing multiple cards at the same time
        if torch.distributed.get_rank() == 0:
            avg_step_time = bugfix.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Train_Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        else:
            avg_step_time = None
    return train_loss/(batch_idx+1), avg_step_time

##### Validation
def test(args, net, device, optimizer,
         epoch, best_acc, criterion, testloader):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Avoid confusion caused by printing multiple cards at the same time
            if torch.distributed.get_rank() == 0:
                avg_step_time = bugfix.progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Val_Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            else:
                avg_step_time = None
    
    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {"model": net.state_dict(),
              "optimizer": optimizer.state_dict()}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')

        # Avoid multiple cards writing at the same time and causing save errors
        if torch.distributed.get_rank() == 0:
            torch.save(state, './checkpoint/'+args.net+'-{}-ckpt.t7'.format(args.patch))
        best_acc = acc

    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f},\
              val loss: {test_loss/(batch_idx+1):.5f}, acc: {(acc):.5f}'
    if torch.distributed.get_rank() == 0:
        print(content + "\n", flush=True)
    return test_loss/(batch_idx+1), acc, best_acc, avg_step_time

def get_args():
    # parsers
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
    parser.add_argument('--opt', default="adam")
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--noaug', action='store_true', help='disable use randomaug')
    parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
    parser.add_argument('--net', default='vit')
    parser.add_argument('--bs', default='512')
    parser.add_argument('--size', default="32")
    parser.add_argument('--n_epochs', type=int, default='200')
    parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
    parser.add_argument('--dimhead', default="512", type=int)
    parser.add_argument('--convkernel', default='8', type=int, help="parameter for convmixer")

    # add extra parameter for ddp
    parser.add_argument('--world_size', default=1, type=int, help="world_size in ddp")
    parser.add_argument('--local_rank', default=0, type=int, help="local_rank in ddp")

    # add extra parameter for ddp
    parser.add_argument('--eval_interval', default=50, type=int, help="how many epoch eval at once")

    args = parser.parse_args()
    return args

def get_net(args):
    # Model factory..
    print('==> Building model..')
    if args.net=='res18':
        from models import ResNet18
        net = ResNet18()
    elif args.net=='vgg':
        from models import VGG
        net = VGG('VGG19')
    elif args.net=='res34':
        from models import ResNet34
        net = ResNet34()
    elif args.net=='res50':
        from models import ResNet50
        net = ResNet50()
    elif args.net=='res101':
        from models import ResNet101
        net = ResNet101()
    elif args.net=="convmixer":
        from models.convmixer import ConvMixer
        # from paper, accuracy >96%. you can tune the depth and dim to scale accuracy and speed.
        net = ConvMixer(256, 16, kernel_size=args.convkernel, patch_size=1, n_classes=10)
    elif args.net=="mlpmixer":
        from models.mlpmixer import MLPMixer
        net = MLPMixer(
        image_size = 32,
        channels = 3,
        patch_size = args.patch,
        dim = 512,
        depth = 6,
        num_classes = 10
    )
    elif args.net=="vit_small":
        from models.vit_small import ViT
        net = ViT(
        image_size = size,
        patch_size = args.patch,
        num_classes = 10,
        dim = int(args.dimhead),
        depth = 6,
        heads = 8,
        mlp_dim = 512,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    elif args.net=="vit_tiny":
        from models.vit_small import ViT
        net = ViT(
        image_size = size,
        patch_size = args.patch,
        num_classes = 10,
        dim = int(args.dimhead),
        depth = 4,
        heads = 6,
        mlp_dim = 256,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    elif args.net=="simplevit":
        from models.simplevit import SimpleViT
        net = SimpleViT(
        image_size = size,
        patch_size = args.patch,
        num_classes = 10,
        dim = int(args.dimhead),
        depth = 6,
        heads = 8,
        mlp_dim = 512
    )
    elif args.net=="vit":
        # ViT for cifar10
        net = ViT(
        image_size = size,
        patch_size = args.patch,
        num_classes = 10,
        dim = int(args.dimhead),
        depth = 6,
        heads = 8,
        mlp_dim = 512,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    elif args.net=="vit_timm":
        import timm
        net = timm.create_model("vit_base_patch16_384", pretrained=True)
        net.head = nn.Linear(net.head.in_features, 10)
    elif args.net=="cait":
        from models.cait import CaiT
        net = CaiT(
        image_size = size,
        patch_size = args.patch,
        num_classes = 10,
        dim = int(args.dimhead),
        depth = 6,   # depth of transformer for patch to patch attention only
        cls_depth=2, # depth of cross attention of CLS tokens to patch
        heads = 8,
        mlp_dim = 512,
        dropout = 0.1,
        emb_dropout = 0.1,
        layer_dropout = 0.05
    )
    elif args.net=="cait_small":
        from models.cait import CaiT
        net = CaiT(
        image_size = size,
        patch_size = args.patch,
        num_classes = 10,
        dim = int(args.dimhead),
        depth = 6,   # depth of transformer for patch to patch attention only
        cls_depth=2, # depth of cross attention of CLS tokens to patch
        heads = 6,
        mlp_dim = 256,
        dropout = 0.1,
        emb_dropout = 0.1,
        layer_dropout = 0.05
    )
    elif args.net=="swin":
        from models.swin import swin_t
        net = swin_t(window_size=args.patch,
                    num_classes=10,
                    downscaling_factors=(2,2,2,1))
    return net

def main():
    args = get_args()

    bs = int(args.bs)
    imsize = int(args.size)

    # Fixed the error judgment about bool type in the source code
    aug = not args.noaug

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    if args.net=="vit_timm":
        size = 384
    else:
        size = imsize

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Add RandAugment with N, M(hyperparameter)
    if aug:
        # Remove out-of-specification syntax
        N = 2
        M = 14
        transform_train.transforms.insert(0, RandAugment(N, M))

    # ddp init
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl',  init_method='tcp://127.0.0.1:23333',
                            world_size=args.world_size, rank=args.local_rank)

    # Prepare dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=False, num_workers=8,
                                              sampler=train_sampler, persistent_workers=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = get_net(args)
    net = net.cuda()

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/{}-ckpt.t7'.format(args.net))
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    # Loss is CE
    criterion = nn.CrossEntropyLoss()
    criterion_ls = LabelSmoothing()

    if args.opt == "adam":
        optimizer = apex.optimizers.NpuFusedAdam(net.parameters(), lr=args.lr)
    elif args.opt == "sgd":
        optimizer = apex.optimizers.NpuFusedSGD(net.parameters(), lr=args.lr)

    # use cosine scheduling
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

    # use apex O2 level
    net, optimizer = amp.initialize(net, optimizer, opt_level='O2', combine_grad=True)

    # For Multi-GPU
    if 'cuda' in device:
        print(device)
        print("using distributed data parallel")

        # set jit_compile=True avoiding unexpected ERROR
        torch.npu.set_compile_mode(jit_compile=True)

        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank])
        cudnn.benchmark = True

    ##### Training

    for epoch in range(start_epoch, args.n_epochs):
        start = time.time()
        trainloss, train_step_time = train(net, device,  optimizer,
                                           epoch, criterion_ls, trainloader)

        if (epoch + 1) % args.eval_interval == 0 or epoch > int(args.n_epochs * 0.9):
            val_loss, acc, best_acc, val_step_time = test(args, net, device, optimizer,
                                                          epoch, best_acc, criterion, testloader)
        else:
            val_loss, acc, best_acc, val_step_time = -1, -1, -1, None

        scheduler.step(epoch-1) # step cosine scheduling

        print(f"Epoch[{epoch}] epoch_time: {time.time() - start}", flush=True)
        print(f"Train: average_step_time: {train_step_time} train_loss: {trainloss}", flush=True)
        print(f"Val: average_step_time: {val_step_time} val_loss: {val_loss} ",
              f"acc: {acc} best_acc: {best_acc}", flush=True)

if __name__ == '__main__':
    main()
