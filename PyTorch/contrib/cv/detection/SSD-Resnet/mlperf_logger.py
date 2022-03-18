# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

import collections
import os
import subprocess
import torch
import numpy as np
from mlperf_logging import mllog

mllogger = mllog.get_mllogger()

def log_start(*args, **kwargs):
    _log_print(mllogger.start, *args, **kwargs)
def log_end(*args, **kwargs):
    _log_print(mllogger.end, *args, **kwargs)
def log_event(*args, **kwargs):
    _log_print(mllogger.event, *args, **kwargs)
def _log_print(logger, *args, **kwargs):
    """
    Wrapper for MLPerf compliance logging calls.
    All arguments but 'log_all_ranks' are passed to
    mlperf_logging.mllog.
    If 'log_all_ranks' is set to True then all distributed workers will print
    logging message, if set to False then only worker with rank=0 will print
    the message.
    """
    if 'stack_offset' not in kwargs:
        kwargs['stack_offset'] = 3
    if 'value' not in kwargs:
        kwargs['value'] = None

    if kwargs.pop('log_all_ranks', False):
        log = True
    else:
        log = (get_rank() == 0)

    if log:
        logger(*args, **kwargs)


def configure_logger(benchmark):
    mllog.config(filename=os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{benchmark}.log'))
    mllogger = mllog.get_mllogger()
    mllogger.logger.propagate = False


def barrier():
    """
    Works as a temporary distributed barrier, currently pytorch
    doesn't implement barrier for NCCL backend.
    Calls all_reduce on dummy tensor and synchronizes with GPU.
    """
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(torch.npu.FloatTensor(1))
        torch.npu.synchronize()


def get_rank():
    """
    Gets distributed rank or returns zero if distributed is not initialized.
    """
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    return rank

def broadcast_seeds(seed, device):
    if torch.distributed.is_initialized():
        seeds_tensor = torch.IntTensor([seed]).to(device)
        torch.distributed.broadcast(seeds_tensor, 0)
        seed = seeds_tensor.item()
    return seed

def set_seeds(args):
    torch.npu.set_device(args.local_rank)
    device = torch.device('npu')

    # make sure that all workers has the same master seed
    log_event(key=mllog.constants.SEED, value=args.seed)
    args.seed = 1

    local_seed = (args.seed + get_rank()) % 2**32
    print(get_rank(), "Using seed = {}".format(local_seed))
    torch.manual_seed(local_seed)
    np.random.seed(seed=local_seed)
    return local_seed
