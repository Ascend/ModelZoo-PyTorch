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

from densepose.data.video import FirstKFramesSelector, LastKFramesSelector, RandomKFramesSelector


class TestFrameSelector(unittest.TestCase):
    def test_frame_selector_random_k_1(self):
        _SEED = 43
        _K = 4
        random.seed(_SEED)
        selector = RandomKFramesSelector(_K)
        frame_tss = list(range(0, 20, 2))
        _SELECTED_GT = [0, 8, 4, 6]
        selected = selector(frame_tss)
        self.assertEqual(_SELECTED_GT, selected)

    def test_frame_selector_random_k_2(self):
        _SEED = 43
        _K = 10
        random.seed(_SEED)
        selector = RandomKFramesSelector(_K)
        frame_tss = list(range(0, 6, 2))
        _SELECTED_GT = [0, 2, 4]
        selected = selector(frame_tss)
        self.assertEqual(_SELECTED_GT, selected)

    def test_frame_selector_first_k_1(self):
        _K = 4
        selector = FirstKFramesSelector(_K)
        frame_tss = list(range(0, 20, 2))
        _SELECTED_GT = frame_tss[:_K]
        selected = selector(frame_tss)
        self.assertEqual(_SELECTED_GT, selected)

    def test_frame_selector_first_k_2(self):
        _K = 10
        selector = FirstKFramesSelector(_K)
        frame_tss = list(range(0, 6, 2))
        _SELECTED_GT = frame_tss[:_K]
        selected = selector(frame_tss)
        self.assertEqual(_SELECTED_GT, selected)

    def test_frame_selector_last_k_1(self):
        _K = 4
        selector = LastKFramesSelector(_K)
        frame_tss = list(range(0, 20, 2))
        _SELECTED_GT = frame_tss[-_K:]
        selected = selector(frame_tss)
        self.assertEqual(_SELECTED_GT, selected)

    def test_frame_selector_last_k_2(self):
        _K = 10
        selector = LastKFramesSelector(_K)
        frame_tss = list(range(0, 6, 2))
        _SELECTED_GT = frame_tss[-_K:]
        selected = selector(frame_tss)
        self.assertEqual(_SELECTED_GT, selected)
