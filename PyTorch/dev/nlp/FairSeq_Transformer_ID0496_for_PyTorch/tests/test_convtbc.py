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
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import unittest
from fairseq.modules import ConvTBC
import torch.nn as nn


class TestConvTBC(unittest.TestCase):

    def test_convtbc(self):
        # ksz, in_channels, out_channels
        conv_tbc = ConvTBC(4, 5, kernel_size=3, padding=1)
        # out_channels, in_channels, ksz
        conv1d = nn.Conv1d(4, 5, kernel_size=3, padding=1)

        conv_tbc.weight.data.copy_(conv1d.weight.data.transpose(0, 2))
        conv_tbc.bias.data.copy_(conv1d.bias.data)

        input_tbc = torch.randn(7, 2, 4, requires_grad=True)
        input1d = input_tbc.data.transpose(0, 1).transpose(1, 2)
        input1d.requires_grad = True

        output_tbc = conv_tbc(input_tbc)
        output1d = conv1d(input1d)

        self.assertAlmostEqual(output_tbc.data.transpose(0, 1).transpose(1, 2), output1d.data)

        grad_tbc = torch.randn(output_tbc.size())
        grad1d = grad_tbc.transpose(0, 1).transpose(1, 2).contiguous()

        output_tbc.backward(grad_tbc)
        output1d.backward(grad1d)

        self.assertAlmostEqual(conv_tbc.weight.grad.data.transpose(0, 2), conv1d.weight.grad.data)
        self.assertAlmostEqual(conv_tbc.bias.grad.data, conv1d.bias.grad.data)
        self.assertAlmostEqual(input_tbc.grad.data.transpose(0, 1).transpose(1, 2), input1d.grad.data)

    def assertAlmostEqual(self, t1, t2):
        self.assertEqual(t1.size(), t2.size(), "size mismatch")
        self.assertLess((t1 - t2).abs().max(), 1e-4)


if __name__ == '__main__':
    unittest.main()
