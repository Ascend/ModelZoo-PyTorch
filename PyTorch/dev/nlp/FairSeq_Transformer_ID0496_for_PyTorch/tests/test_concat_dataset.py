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
from fairseq.data import LanguagePairDataset, TokenBlockDataset
from fairseq.data.concat_dataset import ConcatDataset
from tests.test_train import mock_dict


class TestConcatDataset(unittest.TestCase):
    def setUp(self):
        d = mock_dict()
        tokens_1 = torch.LongTensor([1]).view(1, -1)
        tokens_ds1 = TokenBlockDataset(
            tokens_1,
            sizes=[tokens_1.size(-1)],
            block_size=1,
            pad=0,
            eos=1,
            include_targets=False,
        )
        self.dataset_1 = LanguagePairDataset(
            tokens_ds1, tokens_ds1.sizes, d, shuffle=False
        )
        tokens_2 = torch.LongTensor([2]).view(1, -1)
        tokens_ds2 = TokenBlockDataset(
            tokens_2,
            sizes=[tokens_2.size(-1)],
            block_size=1,
            pad=0,
            eos=1,
            include_targets=False,
        )
        self.dataset_2 = LanguagePairDataset(
            tokens_ds2, tokens_ds2.sizes, d, shuffle=False
        )

    def test_concat_dataset_basics(self):
        d = ConcatDataset(
            [self.dataset_1, self.dataset_2]
        )
        assert(len(d) == 2)
        assert(d[0]['source'][0] == 1)
        assert(d[1]['source'][0] == 2)

        d = ConcatDataset(
            [self.dataset_1, self.dataset_2], sample_ratios=[1, 2]
        )
        assert(len(d) == 3)
        assert(d[0]['source'][0] == 1)
        assert(d[1]['source'][0] == 2)
        assert(d[2]['source'][0] == 2)

        d = ConcatDataset(
            [self.dataset_1, self.dataset_2], sample_ratios=[2, 1]
        )
        assert(len(d) == 3)
        assert(d[0]['source'][0] == 1)
        assert(d[1]['source'][0] == 1)
        assert(d[2]['source'][0] == 2)
