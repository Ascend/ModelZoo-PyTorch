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

# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import socket
import logging

import torch
import torch.nn as nn
import torch.distributed as dist

from pysot.utils.log_helper import log_once

logger = logging.getLogger('global')


def average_reduce(v):
    if get_world_size() == 1:
        return v
    #tensor = torch.cuda.FloatTensor(1)
    tensor = torch.npu.FloatTensor(1)
    tensor[0] = v
    dist.all_reduce(tensor)
    v = tensor[0] / get_world_size()
    return v


class DistModule(nn.Module):
    def __init__(self, module, bn_method=0):
        super(DistModule, self).__init__()
        self.module = module
        self.bn_method = bn_method
        if get_world_size() > 1:
            broadcast_params(self.module)
        else:
            self.bn_method = 0  # single proccess

    def forward(self, *args, **kwargs):
        broadcast_buffers(self.module, self.bn_method)
        return self.module(*args, **kwargs)

    def train(self, mode=True):
        super(DistModule, self).train(mode)
        self.module.train(mode)
        return self


def broadcast_params(model):
    """ broadcast model parameters """
    for p in model.state_dict().values():
        dist.broadcast(p, 0)


def broadcast_buffers(model, method=0):
    """ broadcast model buffers """
    if method == 0:
        return

    world_size = get_world_size()

    for b in model._all_buffers():
        if method == 1:  # broadcast from main proccess
            dist.broadcast(b, 0)
        elif method == 2:  # average
            dist.all_reduce(b)
            b /= world_size
        else:
            raise Exception('Invalid buffer broadcast code {}'.format(method))


inited = False


def _dist_init():
    '''
    if guess right:
        ntasks: world_size (process num)
        proc_id: rank
    '''
    rank = int(os.environ['RANK'])
    #num_gpus = torch.cuda.device_count()
    num_gpus = torch.npu.device_count()
    #torch.cuda.set_device(rank % num_gpus)
    torch.npu.set_device(rank % num_gpus)
    #dist.init_process_group(backend='nccl')
    dist.init_process_group(backend='hccl')
    world_size = dist.get_world_size()
    return rank, world_size


def _get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip


def dist_init():
    global rank, world_size, inited
    try:
        rank, world_size = _dist_init()
    except RuntimeError as e:
        if 'public' in e.args[0]:
            logger.info(e)
            logger.info('Warning: use single process')
            rank, world_size = 0, 1
        else:
            raise RuntimeError(*e.args)
    inited = True
    return rank, world_size


def get_rank():
    if not inited:
        raise(Exception('dist not inited'))
    return rank


def get_world_size():
    if not inited:
        raise(Exception('dist not inited'))
    return world_size


def reduce_gradients(model, _type='sum'):
    types = ['sum', 'avg']
    assert _type in types, 'gradients method must be in "{}"'.format(types)
    log_once("gradients method is {}".format(_type))
    if get_world_size() > 1:
        for param in model.parameters():
            if param.requires_grad:
                dist.all_reduce(param.grad.data)
                if _type == 'avg':
                    param.grad.data /= get_world_size()
    else:
        return None
