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

import torch
import torch.nn as nn
import numpy as np

device = torch.device('npu' if torch.npu.is_available() else 'cpu')


class MockNoChange(nn.Module):
    def __init__(self):
        """Initialize the model"""
        super(MockNoChange, self).__init__()

    def forward(self, input):
        """A single forward pass of the model. Returns the input.

        Args:
            input: The input to the model as a tensor of
                batch_size X input_size
        """
        return input.clone().detach().to(device)


class MockAllZeros(nn.Module):
    def __init__(self):
        """Initialize the model"""
        super(MockAllZeros, self).__init__()

    def forward(self, input):
        """A single forward pass of the model. Returns all zeros.

        Args:
            input: The input to the model as a tensor of
                batch_size X input_size
        """
        return torch.zeros(input.size(), dtype=torch.double).to(device)


class MockOpposite(nn.Module):
    def __init__(self):
        """Initialize the model"""
        super(MockOpposite, self).__init__()

    def forward(self, input):
        """A single forward pass of the model. Returns input - 1.

        Args:
            input: The input to the model as a tensor of
                batch_size X input_size
        """
        output = input.clone().detach()
        output[input == 1] = 0
        output[input == 0] = 1
        return output.to(device)


class MockShiftRightByOne(nn.Module):
    def __init__(self):
        """Initialize the model"""
        super(MockShiftRightByOne, self).__init__()

    def forward(self, input):
        """A single forward pass of the model. Returns input shifted to the
        right by one along axis 1.

        Args:
            input: The input to the model as a tensor of
                batch_size X input_size
        """
        tmp = input.cpu().numpy()
        tmp = np.roll(tmp, shift=1, axis=1)
        return torch.from_numpy(tmp).to(device)
