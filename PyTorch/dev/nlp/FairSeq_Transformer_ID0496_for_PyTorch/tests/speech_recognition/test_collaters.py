#!/usr/bin/env python3
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

import numpy as np
import torch
from examples.speech_recognition.data.collaters import Seq2SeqCollater


class TestSeq2SeqCollator(unittest.TestCase):
    def test_collate(self):

        eos_idx = 1
        pad_idx = 0
        collater = Seq2SeqCollater(
            feature_index=0, label_index=1, pad_index=pad_idx, eos_index=eos_idx
        )

        # 2 frames in the first sample and 3 frames in the second one
        frames1 = np.array([[7, 8], [9, 10]])
        frames2 = np.array([[1, 2], [3, 4], [5, 6]])
        target1 = np.array([4, 2, 3, eos_idx])
        target2 = np.array([3, 2, eos_idx])
        sample1 = {"id": 0, "data": [frames1, target1]}
        sample2 = {"id": 1, "data": [frames2, target2]}
        batch = collater.collate([sample1, sample2])

        # collate sort inputs by frame's length before creating the batch
        self.assertTensorEqual(batch["id"], torch.tensor([1, 0]))
        self.assertEqual(batch["ntokens"], 7)
        self.assertTensorEqual(
            batch["net_input"]["src_tokens"],
            torch.tensor(
                [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [pad_idx, pad_idx]]]
            ),
        )
        self.assertTensorEqual(
            batch["net_input"]["prev_output_tokens"],
            torch.tensor([[eos_idx, 3, 2, pad_idx], [eos_idx, 4, 2, 3]]),
        )
        self.assertTensorEqual(batch["net_input"]["src_lengths"], torch.tensor([3, 2]))
        self.assertTensorEqual(
            batch["target"],
            torch.tensor([[3, 2, eos_idx, pad_idx], [4, 2, 3, eos_idx]]),
        )
        self.assertEqual(batch["nsentences"], 2)

    def assertTensorEqual(self, t1, t2):
        self.assertEqual(t1.size(), t2.size(), "size mismatch")
        self.assertEqual(t1.ne(t2).long().sum(), 0)


if __name__ == "__main__":
    unittest.main()
