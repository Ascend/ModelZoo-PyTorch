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
import logging
import os
import numpy as np
import torch
import torchvision
from PIL import Image

from torch.utils.data import SubsetRandomSampler, Subset, Dataset
from torchvision.transforms import transforms
from sklearn.model_selection import StratifiedShuffleSplit
from theconf import Config as C

from archive import autoaug_policy, autoaug_paper_cifar10, fa_reduced_cifar10
from augmentations import *
from common import get_logger
from samplers.stratified_sampler import StratifiedSampler
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

logger = get_logger('Unsupervised Data Augmentation')
logger.setLevel(logging.INFO)


def get_dataloaders(dataset, batch, batch_unsup, dataroot):
    if 'cifar' in dataset:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_valid = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        raise ValueError('dataset=%s' % dataset)

    autoaug = transforms.Compose([])
    if isinstance(C.get()['aug'], list):
        logger.debug('augmentation provided.')
        autoaug.transforms.insert(0, Augmentation(C.get()['aug']))
    else:
        logger.debug('augmentation: %s' % C.get()['aug'])
        if C.get()['aug'] == 'fa_reduced_cifar10':
            autoaug.transforms.insert(0, Augmentation(fa_reduced_cifar10()))
        elif C.get()['aug'] == 'autoaug_cifar10':
            autoaug.transforms.insert(0, Augmentation(autoaug_paper_cifar10()))
        elif C.get()['aug'] == 'autoaug_extend':
            autoaug.transforms.insert(0, Augmentation(autoaug_policy()))
        elif C.get()['aug'] == 'default':
            pass
        else:
            raise ValueError('not found augmentations. %s' % C.get()['aug'])
    transform_train.transforms.insert(0, autoaug)

    if C.get()['cutout'] > 0:
        transform_train.transforms.append(CutoutDefault(C.get()['cutout']))

    if dataset in ['cifar10', 'cifar100']:
        if dataset == 'cifar10':
            total_trainset = torchvision.datasets.CIFAR10(root=dataroot, train=True, download=True, transform=transform_train)
            unsup_trainset = torchvision.datasets.CIFAR10(root=dataroot, train=True, download=True, transform=None)
            testset = torchvision.datasets.CIFAR10(root=dataroot, train=False, download=True, transform=transform_test)
        elif dataset == 'cifar100':
            total_trainset = torchvision.datasets.CIFAR100(root=dataroot, train=True, download=True, transform=transform_train)
            unsup_trainset = torchvision.datasets.CIFAR100(root=dataroot, train=True, download=True, transform=None)
            testset = torchvision.datasets.CIFAR100(root=dataroot, train=False, download=True, transform=transform_test)
        else:
            raise ValueError

        sss = StratifiedShuffleSplit(n_splits=1, test_size=46000, random_state=0)   # 4000 trainset
        sss = sss.split(list(range(len(total_trainset))), total_trainset.targets)
        train_idx, valid_idx = next(sss)
        train_labels = [total_trainset.targets[idx] for idx in train_idx]

        trainset = Subset(total_trainset, train_idx)        # for supervised
        trainset.train_labels = train_labels

        otherset = Subset(unsup_trainset, valid_idx)        # for unsupervised
        # otherset = unsup_trainset
        otherset = UnsupervisedDataset(otherset, transform_valid, autoaug, cutout=C.get()['cutout'])
    else:
        raise ValueError('invalid dataset name=%s' % dataset)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch, shuffle=False, num_workers=8, pin_memory=True,
        sampler=StratifiedSampler(trainset.train_labels), drop_last=True)

    unsuploader = torch.utils.data.DataLoader(
        otherset, batch_size=batch_unsup, shuffle=True, num_workers=8, pin_memory=True,
        sampler=None, drop_last=True)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch, shuffle=False, num_workers=32, pin_memory=True,
        drop_last=False
    )
    return trainloader, unsuploader, testloader


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        if self.length <= 0:
            return img
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class Augmentation(object):
    def __init__(self, policies):
        self.policies = policies

    def __call__(self, img):
        for _ in range(1):
            policy = random.choice(self.policies)
            for name, pr, level in policy:
                if random.random() > pr:
                    continue
                img = apply_augment(img, name, level)
        return img


class UnsupervisedDataset(Dataset):
    def __init__(self, dataset, transform_default, transform_aug, cutout=0):
        self.dataset = dataset
        self.transform_default = transform_default
        self.transform_aug = transform_aug
        self.transform_cutout = CutoutDefault(cutout)   # issue 4 : https://github.com/ildoonet/unsupervised-data-augmentation/issues/4

    def __getitem__(self, index):
        img, _ = self.dataset[index]

        img1 = self.transform_default(img)
        img2 = self.transform_default(self.transform_aug(img))
        img2 = self.transform_cutout(img2)

        return img1, img2

    def __len__(self):
        return len(self.dataset)
