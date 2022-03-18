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

import unittest

import torch

from fairseq.data import TokenBlockDataset

import tests.utils as test_utils


class TestTokenBlockDataset(unittest.TestCase):

    def _build_dataset(self, data, **kwargs):
        sizes = [len(x) for x in data]
        underlying_ds = test_utils.TestDataset(data)
        return TokenBlockDataset(underlying_ds, sizes, **kwargs)

    def test_eos_break_mode(self):
        data = [
            torch.tensor([5, 4, 3, 2, 1], dtype=torch.long),
            torch.tensor([1], dtype=torch.long),
            torch.tensor([8, 7, 6, 1], dtype=torch.long),
        ]
        ds = self._build_dataset(data, block_size=None, pad=0, eos=1, break_mode='eos')
        self.assertEqual(ds[0].tolist(), [5, 4, 3, 2, 1])
        self.assertEqual(ds[1].tolist(), [1])
        self.assertEqual(ds[2].tolist(), [8, 7, 6, 1])

        data = [
            torch.tensor([5, 4, 3, 2, 1], dtype=torch.long),
            torch.tensor([8, 7, 6, 1], dtype=torch.long),
            torch.tensor([1], dtype=torch.long),
        ]
        ds = self._build_dataset(data, block_size=None, pad=0, eos=1, break_mode='eos')
        self.assertEqual(ds[0].tolist(), [5, 4, 3, 2, 1])
        self.assertEqual(ds[1].tolist(), [8, 7, 6, 1])
        self.assertEqual(ds[2].tolist(), [1])

    def test_block_break_mode(self):
        data = [
            torch.tensor([5, 4, 3, 2, 1], dtype=torch.long),
            torch.tensor([8, 7, 6, 1], dtype=torch.long),
            torch.tensor([9, 1], dtype=torch.long),
        ]
        ds = self._build_dataset(data, block_size=3, pad=0, eos=1, break_mode='none')
        self.assertEqual(ds[0].tolist(), [5, 4, 3])
        self.assertEqual(ds[1].tolist(), [2, 1, 8])
        self.assertEqual(ds[2].tolist(), [7, 6, 1])
        self.assertEqual(ds[3].tolist(), [9, 1])

    def test_complete_break_mode(self):
        data = [
            torch.tensor([5, 4, 3, 2, 1], dtype=torch.long),
            torch.tensor([8, 7, 6, 1], dtype=torch.long),
            torch.tensor([9, 1], dtype=torch.long),
        ]
        ds = self._build_dataset(data, block_size=6, pad=0, eos=1, break_mode='complete')
        self.assertEqual(ds[0].tolist(), [5, 4, 3, 2, 1])
        self.assertEqual(ds[1].tolist(), [8, 7, 6, 1, 9, 1])

        data = [
            torch.tensor([4, 3, 2, 1], dtype=torch.long),
            torch.tensor([5, 1], dtype=torch.long),
            torch.tensor([1], dtype=torch.long),
            torch.tensor([6, 1], dtype=torch.long),
        ]
        ds = self._build_dataset(data, block_size=3, pad=0, eos=1, break_mode='complete')
        self.assertEqual(ds[0].tolist(), [4, 3, 2, 1])
        self.assertEqual(ds[1].tolist(), [5, 1, 1])
        self.assertEqual(ds[2].tolist(), [6, 1])


if __name__ == "__main__":
    unittest.main()
