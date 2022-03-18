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

import random
import unittest
from typing import Any, Iterable, Iterator, Tuple

from densepose.data import CombinedDataLoader


def _grouper(iterable: Iterable[Any], n: int, fillvalue=None) -> Iterator[Tuple[Any]]:
    """
    Group elements of an iterable by chunks of size `n`, e.g.
    grouper(range(9), 4) ->
        (0, 1, 2, 3), (4, 5, 6, 7), (8, None, None, None)
    """
    it = iter(iterable)
    while True:
        values = []
        for _ in range(n):
            try:
                value = next(it)
            except StopIteration:
                values.extend([fillvalue] * (n - len(values)))
                yield tuple(values)
                return
            values.append(value)
        yield tuple(values)


class TestCombinedDataLoader(unittest.TestCase):
    def test_combine_loaders_1(self):
        loader1 = _grouper([f"1_{i}" for i in range(10)], 2)
        loader2 = _grouper([f"2_{i}" for i in range(11)], 3)
        batch_size = 4
        ratios = (0.1, 0.9)
        random.seed(43)
        combined = CombinedDataLoader((loader1, loader2), batch_size, ratios)
        BATCHES_GT = [
            ["1_0", "1_1", "2_0", "2_1"],
            ["2_2", "2_3", "2_4", "2_5"],
            ["1_2", "1_3", "2_6", "2_7"],
            ["2_8", "2_9", "2_10", None],
        ]
        for i, batch in enumerate(combined):
            self.assertEqual(len(batch), batch_size)
            self.assertEqual(batch, BATCHES_GT[i])
