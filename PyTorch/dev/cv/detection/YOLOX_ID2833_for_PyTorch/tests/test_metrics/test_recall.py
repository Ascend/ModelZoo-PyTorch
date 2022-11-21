
# Copyright 2022 Huawei Technologies Co., Ltd
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

# Copyright (c) Open-MMLab. All rights reserved.    
import numpy as np

from mmdet.core.evaluation.recall import eval_recalls

det_bboxes = np.array([
    [0, 0, 10, 10],
    [10, 10, 20, 20],
    [32, 32, 38, 42],
])
gt_bboxes = np.array([[0, 0, 10, 20], [0, 10, 10, 19], [10, 10, 20, 20]])
gt_ignore = np.array([[5, 5, 10, 20], [6, 10, 10, 19]])


def test_eval_recalls():
    gts = [gt_bboxes, gt_bboxes, gt_bboxes]
    proposals = [det_bboxes, det_bboxes, det_bboxes]

    recall = eval_recalls(
        gts, proposals, proposal_nums=2, use_legacy_coordinate=True)
    assert recall.shape == (1, 1)
    assert 0.66 < recall[0][0] < 0.667
    recall = eval_recalls(
        gts, proposals, proposal_nums=2, use_legacy_coordinate=False)
    assert recall.shape == (1, 1)
    assert 0.66 < recall[0][0] < 0.667

    recall = eval_recalls(
        gts, proposals, proposal_nums=2, use_legacy_coordinate=True)
    assert recall.shape == (1, 1)
    assert 0.66 < recall[0][0] < 0.667
    recall = eval_recalls(
        gts,
        proposals,
        iou_thrs=[0.1, 0.9],
        proposal_nums=2,
        use_legacy_coordinate=False)
    assert recall.shape == (1, 2)
    assert recall[0][1] <= recall[0][0]
    recall = eval_recalls(
        gts,
        proposals,
        iou_thrs=[0.1, 0.9],
        proposal_nums=2,
        use_legacy_coordinate=True)
    assert recall.shape == (1, 2)
    assert recall[0][1] <= recall[0][0]
