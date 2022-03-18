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
import numpy as np
import timm
import os
import torch
import numpy as np
import torchvision.transforms as transforms
import torch.distributed as dist
from .samplers import SubsetRandomSampler
from torchvision.datasets import ImageFolder
from torch.utils.data import DistributedSampler, DataLoader
from timm.data import create_transform
from timm.data.transforms import _pil_interp
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    config.freeze()
    print("local rank '{}' / global rank '{}' successfully build train dataset".format(config.LOCAL_RANK, dist.get_rank()))

    dataset_valid, _ = build_dataset(is_train=False, config=config)
    print("local rank '{}' / global rank '{}' successfully build valid dataset".format(config.LOCAL_RANK, dist.get_rank()))

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler_train = DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    indices = np.arange(dist.get_rank(), len(dataset_valid), dist.get_world_size())
    sampler_valid = SubsetRandomSampler(indices)

    data_loader_train = DataLoader(dataset=dataset_train, sampler=sampler_train,
                                   batch_size=config.DATA.BATCH_SIZE,
                                   num_workers=config.DATA.NUM_WORKERS,
                                   pin_memory=config.DATA.PIN_MEMORY,
                                   drop_last=True)
    data_loader_valid = DataLoader(dataset=dataset_valid, sampler=sampler_valid,
                                   batch_size=config.DATA.BATCH_SIZE,
                                   num_workers=config.DATA.NUM_WORKERS,
                                   pin_memory=config.DATA.PIN_MEMORY,
                                   drop_last=False, shuffle=False)

    return dataset_train, dataset_valid, data_loader_train, data_loader_valid

def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    if config.DATA.DATASET == 'imagenet-tiny':
        prefix = 'train' if is_train else 'val'
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = ImageFolder(root, transform=transform)
        nb_classes = 200
    else:
        return NotImplementedError('Only support ImageNet-tiny Now')

    return dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)))
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                       interpolation=_pil_interp(config.DATA.INTERPOLATION)))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))

    return transforms.Compose(t)
