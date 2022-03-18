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
import torch
import torchvision
import config as cfg

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

class Aircraft(Dataset):
    """`FGVC Aircraft <https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/>`_ Dataset.
    """
    def __init__(self, args, **kwargs):
        super(Aircraft, self).__init__()
        self.args = args
        self.num_class = cfg.NUM_CLASSES[args.dataset.lower()]

    @property
    def mean(self):
        return cfg.MEANS[self.args.dataset.lower()]

    @property
    def std(self):
        return cfg.STDS[self.args.dataset.lower()]

    def train_transform(self):
        return transforms.Compose([
            transforms.Resize(self.args.base_size),
            transforms.RandomCrop(self.args.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

    def val_transform(self):
        return transforms.Compose([
            transforms.Resize(self.args.base_size),
            transforms.CenterCrop(self.args.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def train_dataloader(self):
        dataset = ImageFolder(
            root=self.args.train_dir,
            transform=self.train_transform(),
        )
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size,
                                num_workers=4, shuffle=True, drop_last=True, pin_memory=True)
        return dataloader
    
    def val_dataloader(self):
        dataset = ImageFolder(
            root=self.args.val_dir,
            transform=self.val_transform(),
        )
        dataloader = DataLoader(dataset, batch_size=self.args.test_batch_size,
                                num_workers=4, shuffle=True, drop_last=True, pin_memory=True)
        return dataloader
 