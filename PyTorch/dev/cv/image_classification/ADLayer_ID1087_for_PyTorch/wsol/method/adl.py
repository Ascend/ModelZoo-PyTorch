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
Original repository: https://github.com/junsukchoe/ADL
"""

import torch
import torch.nn as nn
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

__all__ = ['ADL']


class ADL(nn.Module):
    def __init__(self, adl_drop_rate=0.75, adl_drop_threshold=0.8):
        super(ADL, self).__init__()
        if not (0 <= adl_drop_rate <= 1):
            raise ValueError("Drop rate must be in range [0, 1].")
        if not (0 <= adl_drop_threshold <= 1):
            raise ValueError("Drop threshold must be in range [0, 1].")
        self.adl_drop_rate = adl_drop_rate
        self.adl_drop_threshold = adl_drop_threshold
        self.attention = None
        self.drop_mask = None

    def forward(self, input_):
        if not self.training:
            return input_
        else:
            attention = torch.mean(input_, dim=1, keepdim=True)
            importance_map = torch.sigmoid(attention)
            drop_mask = self._drop_mask(attention)
            selected_map = self._select_map(importance_map, drop_mask)
            return input_.mul(selected_map)

    def _select_map(self, importance_map, drop_mask):
        random_tensor = torch.rand([], dtype=torch.float32) + self.adl_drop_rate
        binary_tensor = random_tensor.floor()
        return (1. - binary_tensor) * importance_map + binary_tensor * drop_mask

    def _drop_mask(self, attention):
        b_size = attention.size(0)
        max_val, _ = torch.max(attention.view(b_size, -1), dim=1, keepdim=True)
        thr_val = max_val * self.adl_drop_threshold
        thr_val = thr_val.view(b_size, 1, 1, 1)
        return (attention < thr_val).float()

    def extra_repr(self):
        return 'adl_drop_rate={}, adl_drop_threshold={}'.format(
            self.adl_drop_rate, self.adl_drop_threshold)
