# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
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
