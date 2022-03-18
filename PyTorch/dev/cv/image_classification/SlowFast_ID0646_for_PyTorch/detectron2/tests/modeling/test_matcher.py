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
import unittest
from typing import List
import torch

from detectron2.config import get_cfg
from detectron2.modeling.matcher import Matcher


class TestMatcher(unittest.TestCase):
    def test_scriptability(self):
        cfg = get_cfg()
        anchor_matcher = Matcher(
            cfg.MODEL.RPN.IOU_THRESHOLDS, cfg.MODEL.RPN.IOU_LABELS, allow_low_quality_matches=True
        )
        match_quality_matrix = torch.tensor(
            [[0.15, 0.45, 0.2, 0.6], [0.3, 0.65, 0.05, 0.1], [0.05, 0.4, 0.25, 0.4]]
        )
        expected_matches = torch.tensor([1, 1, 2, 0])
        expected_match_labels = torch.tensor([-1, 1, 0, 1], dtype=torch.int8)

        matches, match_labels = anchor_matcher(match_quality_matrix)
        self.assertTrue(torch.allclose(matches, expected_matches))
        self.assertTrue(torch.allclose(match_labels, expected_match_labels))

        # nonzero_tuple must be import explicitly to let jit know what it is.
        # https://github.com/pytorch/pytorch/issues/38964
        from detectron2.layers import nonzero_tuple  # noqa F401

        def f(thresholds: List[float], labels: List[int]):
            return Matcher(thresholds, labels, allow_low_quality_matches=True)

        scripted_anchor_matcher = torch.jit.script(f)(
            cfg.MODEL.RPN.IOU_THRESHOLDS, cfg.MODEL.RPN.IOU_LABELS
        )
        matches, match_labels = scripted_anchor_matcher(match_quality_matrix)
        self.assertTrue(torch.allclose(matches, expected_matches))
        self.assertTrue(torch.allclose(match_labels, expected_match_labels))


if __name__ == "__main__":
    unittest.main()
