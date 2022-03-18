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
from .datasets import cifar10, imagenet, cub200, cars, aircraft
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))

def make_data_loader(args, **kwargs):

    if args.dataset == 'cifar10':
        _cifar10 = cifar10.CIFAR10_Module(args, **kwargs)
        train_loader = _cifar10.train_dataloader()
        val_loader = _cifar10.val_dataloader()
        test_loader = None
        num_class = _cifar10.num_class

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'cub200':
        _cub200 = cub200.CUB200(args, **kwargs)
        train_loader = _cub200.train_dataloader()
        val_loader = _cub200.val_dataloader()
        test_loader = None
        num_class = _cub200.num_class
        
        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'cars':
        _cars = cars.Cars(args, **kwargs)
        train_loader = _cars.train_dataloader()
        val_loader = _cars.val_dataloader()
        test_loader = None
        num_class = _cars.num_class
        
        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'aircraft':
        _aircfraft = aircraft.Aircraft(args, **kwargs)
        train_loader = _aircfraft.train_dataloader()
        val_loader = _aircfraft.val_dataloader()
        test_loader = None
        num_class = _aircfraft.num_class
        
        return train_loader, val_loader, test_loader, num_class


    elif args.dataset == 'imagenet':
        _imagenet = imagenet.ImageNet(args, **kwargs)
        train_loader = _imagenet.train_dataloader()
        val_loader = _imagenet.val_dataloader()
        test_loader = None
        num_class = _imagenet.num_class

        return train_loader, val_loader, test_loader, num_class
