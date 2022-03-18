# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------
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
# ===========================================================================
import argparse
import logging
import os
from types import MethodType

import torch

from .utils.config import update_config

parser = argparse.ArgumentParser(description='AlphaPose Training')

"----------------------------- Experiment options -----------------------------"
parser.add_argument('--cfg',
                    help='experiment configure file name',
                    required=True,
                    type=str)
parser.add_argument('--exp-id', default='default', type=str,
                    help='Experiment ID')

parser.add_argument('--device', default='gpu', type=str, help='npu or gpu')
parser.add_argument('--device-list', default='0,1,2,3,4,5,6,7', type=str, help='device id list')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
"----------------------------- General options -----------------------------"
parser.add_argument('--nThreads', default=20, type=int,
                    help='Number of data loading threads')
parser.add_argument('--snapshot', default=10, type=int,
                    help='How often to take a snapshot of the model (0 = never)')

parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://192.168.1.214:23345', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none',
                    help='job launcher')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--addr', default='127.0.0.1', type=str, help='master addr')

"----------------------------- Training options -----------------------------"
parser.add_argument('--sync', default=False, dest='sync',
                    help='Use Sync Batchnorm', action='store_true')
parser.add_argument('--detector', dest='detector',
                    help='detector name', default="yolo")
parser.add_argument('--amp', default=False, action='store_true',
                    help='use amp to train the model')
parser.add_argument('--loss-scale', default=-1, type=float,
                    help='loss scale using in amp, default -1 means dynamic')
parser.add_argument('--opt-level', default='O2', type=str,
                    help='loss scale using in amp, default -1 means dynamic')
"----------------------------- Log options -----------------------------"
parser.add_argument('--board', default=True, dest='board',
                    help='Logging with tensorboard', action='store_true')
parser.add_argument('--debug', default=False, dest='debug',
                    help='Visualization debug', action='store_true')
parser.add_argument('--map', default=True, dest='map',
                    help='Evaluate mAP per epoch', action='store_true')


opt = parser.parse_args()
cfg_file_name = os.path.basename(opt.cfg)
cfg = update_config(opt.cfg)

cfg['FILE_NAME'] = cfg_file_name
cfg.TRAIN.DPG_STEP = [i - cfg.TRAIN.DPG_MILESTONE for i in cfg.TRAIN.DPG_STEP]
# opt.world_size = cfg.TRAIN.WORLD_SIZE
opt.work_dir = './exp/{}-{}/'.format(opt.exp_id, cfg_file_name)

# opt.gpus = [i for i in range(torch.cuda.device_count())]
# opt.device = torch.device("cuda:" + str(opt.gpus[0]) if opt.gpus[0] >= 0 else "cpu")
opt.npus = [i for i in range(torch.npu.device_count())]
opt.device = torch.device("npu:" + str(opt.npus[0]) if opt.npus[0] >= 0 else "cpu")


if not os.path.exists("./exp/{}-{}".format(opt.exp_id, cfg_file_name)):
    os.makedirs("./exp/{}-{}".format(opt.exp_id, cfg_file_name))
if opt.world_size>1:
    filehandler = logging.FileHandler(
        './exp/{}-{}/training_{}p.log'.format(opt.exp_id, cfg_file_name,opt.world_size))
else:
    filehandler = logging.FileHandler(
        './exp/{}-{}/training_1p.log'.format(opt.exp_id, cfg_file_name))
streamhandler = logging.StreamHandler()

logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)


def epochInfo(self, set, idx, loss, acc):
    self.info('{set}-{idx:d} epoch | loss:{loss:.8f} | acc:{acc:.4f}'.format(
        set=set,
        idx=idx,
        loss=loss,
        acc=acc
    ))


logger.epochInfo = MethodType(epochInfo, logger)
