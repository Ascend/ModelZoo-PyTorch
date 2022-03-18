# Copyright 2021 Huawei Technologies Co., Ltd
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
# ============================================================================
# -*- coding: utf-8 -*-
# File   : unittest.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 27/01/2018
#
# This file is part of Synchronized-BatchNorm-PyTorch.
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
# Distributed under MIT License.

import unittest
import torch


class TorchTestCase(unittest.TestCase):
    def assertTensorClose(self, x, y):
        adiff = float((x - y).abs().max())
        if (y == 0).all():
            rdiff = 'NaN'
        else:
            rdiff = float((adiff / y).abs().max())

        message = (
            'Tensor close check failed\n'
            'adiff={}\n'
            'rdiff={}\n'
        ).format(adiff, rdiff)
        self.assertTrue(torch.allclose(x, y), message)

