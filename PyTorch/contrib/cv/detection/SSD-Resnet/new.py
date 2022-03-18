# Copyright 2021 Huawei Technologies Co., Ltd

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
#sys.path.append('/share/home/litaotao/yzc/training_results_v0.7-master/NVIDIA/benchmarks/ssd/implementations/pytorch/')#
import os
#from base_model import Loss
from opt_loss import OptLoss
from mlperf_logger import configure_logger, log_start, log_end, log_event, set_seeds, get_rank, barrier
from mlperf_logging.mllog import constants
import torch
from torch.autograd import Variable
import time
import numpy as np
import io
from bisect import bisect       # for lr_scheduler

from ssd300 import SSD300
from master_params import create_flat_master
from parse_config import parse_args, validate_arguments, validate_group_bn

from async_evaluator import AsyncEvaluator
from eval import coco_eval

#import sys
import gc
from data.native_pipeline import build_train_pipe
# necessary pytorch imports
import torch.utils.data.distributed
import torch.distributed as dist
configure_logger(constants.SSD)
log_start(key=constants.INIT_START, log_all_ranks=True)
args = parse_args()
# make sure the epoch lists are in sorted order
args.evaluation.sort()
args.lr_decay_epochs.sort()

validate_arguments(args)

torch.set_num_threads(1)
torch.backends.cudnn.benchmark = not args.profile_cudnn_get
build_train_pipe(args)