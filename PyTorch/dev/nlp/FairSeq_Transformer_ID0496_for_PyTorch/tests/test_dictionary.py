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

import tempfile
import unittest

import torch

from fairseq.data import Dictionary


class TestDictionary(unittest.TestCase):

    def test_finalize(self):
        txt = [
            'A B C D',
            'B C D',
            'C D',
            'D',
        ]
        ref_ids1 = list(map(torch.IntTensor, [
            [4, 5, 6, 7, 2],
            [5, 6, 7, 2],
            [6, 7, 2],
            [7, 2],
        ]))
        ref_ids2 = list(map(torch.IntTensor, [
            [7, 6, 5, 4, 2],
            [6, 5, 4, 2],
            [5, 4, 2],
            [4, 2],
        ]))

        # build dictionary
        d = Dictionary()
        for line in txt:
            d.encode_line(line, add_if_not_exist=True)

        def get_ids(dictionary):
            ids = []
            for line in txt:
                ids.append(dictionary.encode_line(line, add_if_not_exist=False))
            return ids

        def assertMatch(ids, ref_ids):
            for toks, ref_toks in zip(ids, ref_ids):
                self.assertEqual(toks.size(), ref_toks.size())
                self.assertEqual(0, (toks != ref_toks).sum().item())

        ids = get_ids(d)
        assertMatch(ids, ref_ids1)

        # check finalized dictionary
        d.finalize()
        finalized_ids = get_ids(d)
        assertMatch(finalized_ids, ref_ids2)

        # write to disk and reload
        with tempfile.NamedTemporaryFile(mode='w') as tmp_dict:
            d.save(tmp_dict.name)
            d = Dictionary.load(tmp_dict.name)
            reload_ids = get_ids(d)
            assertMatch(reload_ids, ref_ids2)
            assertMatch(finalized_ids, reload_ids)


if __name__ == '__main__':
    unittest.main()
