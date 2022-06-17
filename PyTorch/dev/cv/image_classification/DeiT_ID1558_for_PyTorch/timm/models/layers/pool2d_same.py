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
""" AvgPool2d w/ Same Padding

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from .helpers import to_2tuple
from .padding import pad_same, get_padding_value
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


def avg_pool2d_same(x, kernel_size: List[int], stride: List[int], padding: List[int] = (0, 0),
                    ceil_mode: bool = False, count_include_pad: bool = True):
    # FIXME how to deal with count_include_pad vs not for external padding?
    x = pad_same(x, kernel_size, stride)
    return F.avg_pool2d(x, kernel_size, stride, (0, 0), ceil_mode, count_include_pad)


class AvgPool2dSame(nn.AvgPool2d):
    """ Tensorflow like 'SAME' wrapper for 2D average pooling
    """
    def __init__(self, kernel_size: int, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        super(AvgPool2dSame, self).__init__(kernel_size, stride, (0, 0), ceil_mode, count_include_pad)

    def forward(self, x):
        return avg_pool2d_same(
            x, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)


def max_pool2d_same(
        x, kernel_size: List[int], stride: List[int], padding: List[int] = (0, 0),
        dilation: List[int] = (1, 1), ceil_mode: bool = False):
    x = pad_same(x, kernel_size, stride, value=-float('inf'))
    return F.max_pool2d(x, kernel_size, stride, (0, 0), dilation, ceil_mode)


class MaxPool2dSame(nn.MaxPool2d):
    """ Tensorflow like 'SAME' wrapper for 2D max pooling
    """
    def __init__(self, kernel_size: int, stride=None, padding=0, dilation=1, ceil_mode=False, count_include_pad=True):
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)
        super(MaxPool2dSame, self).__init__(kernel_size, stride, (0, 0), dilation, ceil_mode, count_include_pad)

    def forward(self, x):
        return max_pool2d_same(x, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode)


def create_pool2d(pool_type, kernel_size, stride=None, **kwargs):
    stride = stride or kernel_size
    padding = kwargs.pop('padding', '')
    padding, is_dynamic = get_padding_value(padding, kernel_size, stride=stride, **kwargs)
    if is_dynamic:
        if pool_type == 'avg':
            return AvgPool2dSame(kernel_size, stride=stride, **kwargs)
        elif pool_type == 'max':
            return MaxPool2dSame(kernel_size, stride=stride, **kwargs)
        else:
            assert False, f'Unsupported pool type {pool_type}'
    else:
        if pool_type == 'avg':
            return nn.AvgPool2d(kernel_size, stride=stride, padding=padding, **kwargs)
        elif pool_type == 'max':
            return nn.MaxPool2d(kernel_size, stride=stride, padding=padding, **kwargs)
        else:
            assert False, f'Unsupported pool type {pool_type}'
