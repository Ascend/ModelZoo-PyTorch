# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import random
import shutil
import time
import warnings

import torch
if torch.__version__>="1.8":
    import torch_npu
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
parser.add_argument('--port',default='46888', type=str,help='masterport')
parser.add_argument('--data_path',default='', type=str,help='data_path')
parser.add_argument('--iters',default=1000, type=int,help='iters number')

# runtime2.0
parser.add_argument('--rt2', action='store_true', default=False, help='enable runitme2.0 mode')
# parser.add_argument('--device_num',default=-1,type=int,help='device_num')


def main():
    warnings.filterwarnings("ignore")
    args = parser.parse_args()
    
    os.environ['MASTER_ADDR'] = '127.0.0.3'
    os.environ['MASTER_PORT'] = '46888'
    if args.rt2:
        torch.npu.set_compile_mode(jit_compile=False)
    if args.device_num > 1:
        from config import conf_8p as conf
        dist.init_process_group(backend=args.dist_backend, # init_method=args.dist_url,
    							world_size=args.world_size, rank=args.local_rank)
        conf['data'].update({'dataset_path': args.data_path})
        conf['model'].update({'total_iter': args.iters})
        print('The training config is:',conf)
    else:
        from config import conf_1p as conf
        conf['data'].update({'dataset_path': args.data_path})
        conf['model'].update({'total_iter': args.iters})
        print('The training config is:',conf)
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
