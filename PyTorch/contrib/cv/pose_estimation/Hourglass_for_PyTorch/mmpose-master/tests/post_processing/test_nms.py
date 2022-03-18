# -*- coding: utf-8 -*-
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import numpy as np

from mmpose.core.post_processing.nms import nms, oks_iou, oks_nms, soft_oks_nms


def test_soft_oks_nms():
    oks_thr = 0.9
    kpts = []
    kpts.append({
        'keypoints': np.tile(np.array([10, 10, 0.9]), [17, 1]),
        'area': 100,
        'score': 0.9
    })
    kpts.append({
        'keypoints': np.tile(np.array([10, 10, 0.9]), [17, 1]),
        'area': 100,
        'score': 0.4
    })
    kpts.append({
        'keypoints': np.tile(np.array([100, 100, 0.9]), [17, 1]),
        'area': 100,
        'score': 0.7
    })

    keep = soft_oks_nms([kpts[i] for i in range(len(kpts))], oks_thr)
    assert (keep == np.array([0, 2, 1])).all()

    keep = oks_nms([kpts[i] for i in range(len(kpts))], oks_thr)
    assert (keep == np.array([0, 2])).all()


def test_func_nms():
    result = nms(np.array([[0, 0, 10, 10, 0.9], [0, 0, 10, 8, 0.8]]), 0.5)
    assert result == [0]


def test_oks_iou():
    result = oks_iou(np.ones([17 * 3]), np.ones([1, 17 * 3]), 1, [1])
    assert result[0] == 1.
    result = oks_iou(np.zeros([17 * 3]), np.ones([1, 17 * 3]), 1, [1])
    assert result[0] < 0.01
