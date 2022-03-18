#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
"""
pytorch-dl
Created by raj at 11:05
Date: January 18, 2020
"""

import numpy
import torch
from torch import float32
from torch.nn import functional

from models.utils.model_utils import get_masks
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

torch.manual_seed(200)

x = torch.rand(2, 3)
y = torch.zeros(2, 3)

z = x + x
# print(z)
torch.add(x, x, out=y)
# print(y)
# print(torch.is_tensor(numpy.random.rand(2, 3)))
# print(x.matmul(y.transpose(0, 1)))

x = torch.tensor([1, 2, 3])
y = (x != 1).unsqueeze(-2)
r = torch.unsqueeze(x, 0)       # Size: 1x3
# print(x.size(), r.size())
r = torch.unsqueeze(x, 1)
# print(x.size(), r.size())

x = torch.tensor([[1, 2], [3, 4]])
# print(functional.softmax(x, dim=0))
attn_weights = torch.rand(4, 4, 4, 4)


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


def buffered_mask(tensor):
    _mask = None
    dim = tensor.size(-1)
    if _mask is None:
        _mask = torch.triu(fill_with_neg_inf(tensor.new(dim, dim)), 1)
    if _mask.size(0) < dim:
        _mask = torch.triu(fill_with_neg_inf(_mask.resize_(dim, dim)), 1)
    return _mask[:dim, :dim]


attn_weights += buffered_mask(attn_weights).unsqueeze(0)
# print(attn_weights)

attn_weights = torch.rand(4, 4, 4)
b, h, w = attn_weights.size()
# print(attn_weights)

mask_diagonal = False
maskval=float('-inf')
# indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
# x = attn_weights[:, indices[0], indices[1]] = maskval
# print(attn_weights)

x = torch.tensor([[1, 2, 4, 0], [3, 4, 0, 0], [1, 2, 23, 45]])
lenghts = torch.tensor([3, 2, 4])
bs, slen = x.size()
print(get_masks(slen, lenghts, causal=False))


