# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2021 Huawei Technologies Co., Ltd
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
# ============================================================================

import os
import logging

import torch

from . import video_sampler as sampler
from . import video_transforms as transforms
from .video_iterator import VideoIter

global_is_distributed = False

def get_kinetics(data_root='./dataset/UCF101',

                 clip_length=8,
                 train_interval=2,
                 val_interval=2,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 seed=torch.distributed.get_rank() if global_is_distributed and torch.distributed.is_initialized else 0,
                 **kwargs):
    """ data iter for kinetics
    """
    logging.debug("VideoIter:: clip_length = {}, interval = [train: {}, val: {}], seed = {}".format( \
                clip_length, train_interval, val_interval, seed))
    logging.info(f"=================useing dataset: {data_root}=================")

    normalize = transforms.Normalize(mean=mean, std=std)

    train_sampler = sampler.RandomSampling(num=clip_length,
                                           interval=train_interval,
                                           speed=[1.0, 1.0],
                                           seed=(seed+0))
    train = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data'),

                      txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'trainlist01.txt'),

                      sampler=train_sampler,
                      force_color=True,
                      video_transform=transforms.Compose([
                                         transforms.RandomScale(make_square=False,
                                                                aspect_ratio=[.8, 1./.8],
                                                                slen=[224, 340]),
                                         transforms.RandomCrop((224, 224)), # insert a resize if needed
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomHLS(vars=[15, 35, 25]), # too slow
                                         transforms.PixelJitter(vars=[-20, 20]), 
                                         transforms.ToTensor(),
                                         normalize,
                                      ],
                                      aug_seed=(seed+1)),
                      name='train',
                      shuffle_list_seed=(seed+2),
                      )

    val_sampler   = sampler.EvenlySampling(num=clip_length,
                                           interval=val_interval,
                                           num_times=1)
    val   = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data'),

                      txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'testlist01.txt'),


                      sampler=val_sampler,
                      force_color=True,
                      video_transform=transforms.Compose([
                                         transforms.Resize(256),
                                         transforms.CenterCrop((224, 224)),
                                         transforms.ToTensor(),
                                         normalize,
                                      ]),
                      name='test',
                      shuffle_list_seed=(seed+3),
                      )
    return (train, val)



def creat(name, batch_size, num_workers=50, distributed=False, **kwargs):
    global_is_distributed = distributed

    if name.upper() == 'KINETICS' or name.upper() == 'UCF101':
        train, val = get_kinetics(**kwargs)
    else:
        assert NotImplementedError("iter {} not found".format(name))

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train)
        val_sampler = None
    else:
        train_sampler = None
        val_sampler = None


    train_loader = torch.utils.data.DataLoader(train,
        batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True,
        drop_last=True,
        sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(val,
        batch_size=1, shuffle=(val_sampler is None),
        num_workers=num_workers, pin_memory=True,
        sampler=val_sampler)

    return (train_loader, val_loader, train_sampler)
