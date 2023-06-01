# coding:utf-8
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

import unittest

import torch
from fairseq.modules import RelPositionalEncoding
import numpy as np


class TestRelPositionalEncoding(unittest.TestCase):
    def setUp(self) -> None:
        self.T = 3
        self.B = 1
        self.C = 2
        torch.manual_seed(0)
        self.sample = torch.randn(self.T, self.B, self.C)  # TBC
        self.rel_pos_enc = RelPositionalEncoding(max_len=4, d_model=self.C)

    def test_extend_pe(self):
        inp = self.sample.transpose(0, 1)
        self.rel_pos_enc.extend_pe(inp)
        expected_pe = torch.tensor(
            [
                [
                    [0.1411, -0.9900],
                    [0.9093, -0.4161],
                    [0.8415, 0.5403],
                    [0.0000, 1.0000],
                    [-0.8415, 0.5403],
                    [-0.9093, -0.4161],
                    [-0.1411, -0.9900],
                ]
            ]
        )

        self.assertTrue(
            np.allclose(
                expected_pe.cpu().detach().numpy(),
                self.rel_pos_enc.pe.cpu().detach().numpy(),
                atol=1e-4,
            )
        )

    def test_forward(self):
        pos_enc = self.rel_pos_enc(self.sample)
        expected_pos_enc = torch.tensor(
            [
                [[0.9093, -0.4161]],
                [[0.8415, 0.5403]],
                [[0.0000, 1.0000]],
                [[-0.8415, 0.5403]],
                [[-0.9093, -0.4161]],
            ]
        )
        self.assertTrue(
            np.allclose(
                pos_enc.cpu().detach().numpy(),
                expected_pos_enc.cpu().detach().numpy(),
                atol=1e-4,
            )
        )


if __name__ == "__main__":
    unittest.main()
