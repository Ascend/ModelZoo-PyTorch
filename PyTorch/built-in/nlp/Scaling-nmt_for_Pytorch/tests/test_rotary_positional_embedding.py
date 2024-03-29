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


import torch
import numpy as np
import unittest
from fairseq.modules.rotary_positional_embedding import apply_rotary_pos_emb
from fairseq.modules import RotaryPositionalEmbedding


class TestRotaryPositionalEmbedding(unittest.TestCase):
    def setUp(self) -> None:
        self.T = 3
        self.B = 1
        self.C = 2
        torch.manual_seed(0)
        self.sample = torch.randn(self.T, self.B, self.C)  # TBC
        self.rope_pos_emd = RotaryPositionalEmbedding(dim=self.C)

    def test_forward(self):
        expected_cos = torch.tensor(
            [[[[1.0000, 1.0000]]], [[[0.5403, 0.5403]]], [[[-0.4161, -0.4161]]]]
        )
        expected_sin = torch.tensor(
            [[[[0.0000, 0.0000]]], [[[0.8415, 0.8415]]], [[[0.9093, 0.9093]]]]
        )
        cos, sin = self.rope_pos_emd(self.sample, self.T)
        self.assertTrue(
            np.allclose(
                expected_cos.cpu().detach().numpy(),
                cos.cpu().detach().numpy(),
                atol=1e-4,
            )
        )
        self.assertTrue(
            np.allclose(
                expected_sin.cpu().detach().numpy(),
                sin.cpu().detach().numpy(),
                atol=1e-4,
            )
        )

    def test_apply_rotary_pos_emb(self):
        cos, sin = self.rope_pos_emd(self.sample, self.T)
        query = self.sample.view(self.T, self.B, 1, self.C)
        expected_query = torch.tensor(
            [[[[1.5410, -0.2934]]], [[[-1.6555, -1.5263]]], [[[1.7231, -0.4041]]]]
        )
        new_query, new_key = apply_rotary_pos_emb(query, query, cos, sin)
        self.assertTrue(
            np.allclose(
                expected_query.cpu().detach().numpy(),
                new_query.cpu().detach().numpy(),
                atol=1e-4,
            )
        )
        self.assertTrue(
            np.allclose(
                expected_query.cpu().detach().numpy(),
                new_key.cpu().detach().numpy(),
                atol=1e-4,
            )
        )


if __name__ == "__main__":
    unittest.main()
