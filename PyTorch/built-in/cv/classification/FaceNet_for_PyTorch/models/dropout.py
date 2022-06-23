#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/19 10：20
# Author   :
# @Site    :
# @File    :dropout.py

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
import torch
if torch.__version__ >= "1.8.1":
    import torch_npu
import torch.nn as nn
import numpy as np


class DroupoutV2(nn.Module):
    def __init__(self, p=0.5, inplace=False, max_seed=2 ** 10 - 1):
        super(DroupoutV2, self).__init__()
        self.p = p
        self.seed = torch.from_numpy(np.random.uniform(1, max_seed, size=(32 * 1024 * 12,)).astype(np.float32))
        self.checked = False

    def check_self(self, x):
        """Check device equipment between tensors.
        """
        if self.seed.device == x.device:
            self.checked = True
            return

        self.seed = self.seed.to(x.device)

    def forward(self, x):
        if not self.training:
            return x

        if not self.checked:
            self.check_self(x)
        if torch.__version__ >= "1.8.1":
            x = nn.functional.dropout(x, p=self.p)
        else:
            x, mask, _ = torch.npu_dropoutV2(x, self.seed, p=self.p)
        return x
