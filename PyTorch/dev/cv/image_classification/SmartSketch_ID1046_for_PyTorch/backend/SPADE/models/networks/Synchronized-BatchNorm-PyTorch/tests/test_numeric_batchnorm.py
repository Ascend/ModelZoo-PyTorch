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
# File   : test_numeric_batchnorm.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 27/01/2018
#
# This file is part of Synchronized-BatchNorm-PyTorch.

import unittest

import torch
import torch.nn as nn
from torch.autograd import Variable

from sync_batchnorm.unittest import TorchTestCase


def handy_var(a, unbias=True):
    n = a.size(0)
    asum = a.sum(dim=0)
    as_sum = (a ** 2).sum(dim=0)  # a square sum
    sumvar = as_sum - asum * asum / n
    if unbias:
        return sumvar / (n - 1)
    else:
        return sumvar / n


class NumericTestCase(TorchTestCase):
    def testNumericBatchNorm(self):
        a = torch.rand(16, 10)
        bn = nn.BatchNorm1d(10, momentum=1, eps=1e-5, affine=False)
        bn.train()

        a_var1 = Variable(a, requires_grad=True)
        b_var1 = bn(a_var1)
        loss1 = b_var1.sum()
        loss1.backward()

        a_var2 = Variable(a, requires_grad=True)
        a_mean2 = a_var2.mean(dim=0, keepdim=True)
        a_std2 = torch.sqrt(handy_var(a_var2, unbias=False).clamp(min=1e-5))
        # a_std2 = torch.sqrt(a_var2.var(dim=0, keepdim=True, unbiased=False) + 1e-5)
        b_var2 = (a_var2 - a_mean2) / a_std2
        loss2 = b_var2.sum()
        loss2.backward()

        self.assertTensorClose(bn.running_mean, a.mean(dim=0))
        self.assertTensorClose(bn.running_var, handy_var(a))
        self.assertTensorClose(a_var1.data, a_var2.data)
        self.assertTensorClose(b_var1.data, b_var2.data)
        self.assertTensorClose(a_var1.grad, a_var2.grad)


if __name__ == '__main__':
    unittest.main()
