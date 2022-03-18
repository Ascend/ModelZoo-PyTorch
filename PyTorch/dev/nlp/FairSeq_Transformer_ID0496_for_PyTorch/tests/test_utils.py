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

from fairseq import utils


class TestUtils(unittest.TestCase):

    def test_convert_padding_direction(self):
        pad = 1
        left_pad = torch.LongTensor([
            [2, 3, 4, 5, 6],
            [1, 7, 8, 9, 10],
            [1, 1, 1, 11, 12],
        ])
        right_pad = torch.LongTensor([
            [2, 3, 4, 5, 6],
            [7, 8, 9, 10, 1],
            [11, 12, 1, 1, 1],
        ])

        self.assertAlmostEqual(
            right_pad,
            utils.convert_padding_direction(
                left_pad,
                pad,
                left_to_right=True,
            ),
        )
        self.assertAlmostEqual(
            left_pad,
            utils.convert_padding_direction(
                right_pad,
                pad,
                right_to_left=True,
            ),
        )

    def test_make_positions(self):
        pad = 1
        left_pad_input = torch.LongTensor([
            [9, 9, 9, 9, 9],
            [1, 9, 9, 9, 9],
            [1, 1, 1, 9, 9],
        ])
        left_pad_output = torch.LongTensor([
            [2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5],
            [1, 1, 1, 2, 3],
        ])
        right_pad_input = torch.LongTensor([
            [9, 9, 9, 9, 9],
            [9, 9, 9, 9, 1],
            [9, 9, 1, 1, 1],
        ])
        right_pad_output = torch.LongTensor([
            [2, 3, 4, 5, 6],
            [2, 3, 4, 5, 1],
            [2, 3, 1, 1, 1],
        ])

        self.assertAlmostEqual(
            left_pad_output,
            utils.make_positions(left_pad_input, pad),
        )
        self.assertAlmostEqual(
            right_pad_output,
            utils.make_positions(right_pad_input, pad),
        )

    def assertAlmostEqual(self, t1, t2):
        self.assertEqual(t1.size(), t2.size(), "size mismatch")
        self.assertLess(utils.item((t1 - t2).abs().max()), 1e-4)


if __name__ == '__main__':
    unittest.main()
