#! /usr/bin/env python3
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
# File   : test_numeric_batchnorm_v2.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 11/01/2018
#
# Distributed under terms of the MIT license.

"""
Test the numerical implementation of batch normalization.

Author: acgtyrant.
See also: https://github.com/vacancy/Synchronized-BatchNorm-PyTorch/issues/14
"""

import unittest

import torch
import torch.nn as nn
import torch.optim as optim

from sync_batchnorm.unittest import TorchTestCase
from sync_batchnorm.batchnorm_reimpl import BatchNorm2dReimpl


class NumericTestCasev2(TorchTestCase):
    def testNumericBatchNorm(self):
        CHANNELS = 16
        batchnorm1 = nn.BatchNorm2d(CHANNELS, momentum=1)
        optimizer1 = optim.SGD(batchnorm1.parameters(), lr=0.01)

        batchnorm2 = BatchNorm2dReimpl(CHANNELS, momentum=1)
        batchnorm2.weight.data.copy_(batchnorm1.weight.data)
        batchnorm2.bias.data.copy_(batchnorm1.bias.data)
        optimizer2 = optim.SGD(batchnorm2.parameters(), lr=0.01)

        for _ in range(100):
            input_ = torch.rand(16, CHANNELS, 16, 16)

            input1 = input_.clone().requires_grad_(True)
            output1 = batchnorm1(input1)
            output1.sum().backward()
            optimizer1.step()

            input2 = input_.clone().requires_grad_(True)
            output2 = batchnorm2(input2)
            output2.sum().backward()
            optimizer2.step()

        self.assertTensorClose(input1, input2)
        self.assertTensorClose(output1, output2)
        self.assertTensorClose(input1.grad, input2.grad)
        self.assertTensorClose(batchnorm1.weight.grad, batchnorm2.weight.grad)
        self.assertTensorClose(batchnorm1.bias.grad, batchnorm2.bias.grad)
        self.assertTensorClose(batchnorm1.running_mean, batchnorm2.running_mean)
        self.assertTensorClose(batchnorm2.running_mean, batchnorm2.running_mean)


if __name__ == '__main__':
    unittest.main()

