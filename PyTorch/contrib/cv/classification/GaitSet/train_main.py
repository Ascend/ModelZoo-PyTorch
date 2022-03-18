# -*- coding: utf-8 -*-
# Copyright (c) Soumith Chintala 2016,
# All rights reserved
#
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://spdx.org/licenses/BSD-3-Clause.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.npu
import torch.distributed as dist
import numpy as np

from model.initialization import initialization


def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--cache', default=True, type=boolean_string,
                    help='cache: if set as TRUE all the training data will be loaded at once'
                         ' before the training start. Default: TRUE')
parser.add_argument('--dist_backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--world_size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--device_num',default=-1,type=int,help='device_num')

parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--addr',default='192.168.88.168',type=str,help='masterip')
parser.add_argument('--port',default='46888',type=str,help='masterport')
# parser.add_argument('--device_num',default=-1,type=int,help='device_num')


def main():
    warnings.filterwarnings("ignore")
    args = parser.parse_args()
    
    os.environ['MASTER_ADDR'] = '127.0.0.3'
    os.environ['MASTER_PORT'] = '46888'
    
    if args.device_num > 1:
        from config import conf_8p as conf
        dist.init_process_group(backend=args.dist_backend, # init_method=args.dist_url,
    							world_size=args.world_size, rank=args.local_rank)
    else:
        from config import conf_1p as conf
    
    local_device = f'npu:{args.local_rank}'
    if args.device_num > 1:
        torch.npu.set_device(local_device)
    else:
        torch.npu.set_device(local_device)
    
    model = initialization(conf, train=True)[0]
    
    if args.local_rank == 0:
        print('Training...')
    model.fit()
    if args.local_rank == 0:
        print('Training finished!')

if __name__ == '__main__':
    main()
