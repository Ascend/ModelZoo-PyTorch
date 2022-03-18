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
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


def get_dataloaders(args):
    train_loader, val_loader, test_loader = None, None, None
    if args.data == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                         std=[0.2471, 0.2435, 0.2616])
        train_set = datasets.CIFAR10(args.data_root, train=True,
                                     transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize
                                     ]))
        val_set = datasets.CIFAR10(args.data_root, train=False,
                                   transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    normalize
                                   ]))
    elif args.data == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
        train_set = datasets.CIFAR100(args.data_root, train=True,
                                      transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize
                                      ]))
        val_set = datasets.CIFAR100(args.data_root, train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        normalize
                                    ]))
    else:
        # ImageNet
        traindir = os.path.join(args.data_root, 'train')
        valdir = os.path.join(args.data_root, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_set = datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]))
        val_set = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]))
    if args.use_valid:
        train_set_index = torch.randperm(len(train_set))
        if os.path.exists(os.path.join(args.save, 'index.pth')):
            print('!!!!!! Load train_set_index !!!!!!')
            train_set_index = torch.load(os.path.join(args.save, 'index.pth'))
        else:
            print('!!!!!! Save train_set_index !!!!!!')
            torch.save(train_set_index, os.path.join(args.save, 'index.pth'))
        if args.data.startswith('cifar'):
            num_sample_valid = 5000
        else:
            num_sample_valid = 50000

        if 'train' in args.splits:
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=args.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                    train_set_index[:-num_sample_valid]),
                num_workers=args.workers, pin_memory=False)
        if 'val' in args.splits:
            val_loader = torch.utils.data.DataLoader(
                train_set, batch_size=args.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                    train_set_index[-num_sample_valid:]),
                num_workers=args.workers, pin_memory=False)
        if 'test' in args.splits:
            test_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=False)
    else:
        if 'train' in args.splits:
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=False)
        if 'val' or 'test' in args.splits:
            val_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=False)
            test_loader = val_loader

    return train_loader, val_loader, test_loader
