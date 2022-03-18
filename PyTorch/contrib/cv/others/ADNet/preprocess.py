# Copyright 2021 Huawei Technologies Co., Ltd
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
import os
import argparse

import torch
import sys
import torch.nn as nn
import torch.optim as optim
import apex.amp as amp
import apex
import logging
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import time
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torch.nn.modules.loss import _Loss 
from models import ADNet
from dataset import prepare_data, Dataset
from utils import *


parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=70, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=15, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=15, help='noise level used on validation set')
parser.add_argument("--is_distributed", type=int, default=0, help='choose ddp or not')
parser.add_argument('--world_size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--DeviceID', type=str, default="0")
parser.add_argument("--num_gpus", default=1, type=int)
'''
parser.add_argument("--clip",type=float,default=0.005,help='Clipping Gradients. Default=0.4') #tcw201809131446tcw
parser.add_argument("--momentum",default=0.9,type='float',help = 'Momentum, Default:0.9') #tcw201809131447tcw
parser.add_argument("--weight-decay","-wd",default=1e-3,type=float,help='Weight decay, Default:1e-4') #tcw20180913347tcw
'''
opt = parser.parse_args()
if __name__ == "__main__":
    if opt.preprocess:
        if opt.mode == 'S':
            prepare_data(data_path='data', patch_size=50, stride=40, aug_times=1) 
        if opt.mode == 'B':
            prepare_data(data_path='data', patch_size=50, stride=10, aug_times=2)
