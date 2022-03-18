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
from numpy.testing import assert_array_almost_equal

from mmpose.core import (affine_transform, flip_back, fliplr_joints,
                         get_affine_transform, rotate_point, transform_preds)


def test_affine_transform():
    pt = np.array([0, 1])
    trans = np.array([[1, 0, 1], [0, 1, 0]])
    ans = affine_transform(pt, trans)
    assert_array_almost_equal(ans, np.array([1, 1]), decimal=4)
    assert isinstance(ans, np.ndarray)


def test_rotate_point():
    src_point = np.array([0, 1])
    rot_rad = np.pi / 2.
    ans = rotate_point(src_point, rot_rad)
    assert_array_almost_equal(ans, np.array([-1, 0]), decimal=4)
    assert isinstance(ans, list)


def test_fliplr_joints():
    joints = np.array([[0, 0, 0], [1, 1, 0]])
    joints_vis = np.array([[1], [1]])
    joints_flip, _ = fliplr_joints(joints, joints_vis, 5, [[0, 1]])
    res = np.array([[3, 1, 0], [4, 0, 0]])
    assert_array_almost_equal(joints_flip, res)


def test_flip_back():
    heatmaps = np.random.random([1, 2, 32, 32])
    flipped_heatmaps = flip_back(heatmaps, [[0, 1]])
    heatmaps_new = flip_back(flipped_heatmaps, [[0, 1]])
    assert_array_almost_equal(heatmaps, heatmaps_new)

    heatmaps = np.random.random([1, 2, 32, 32])
    flipped_heatmaps = flip_back(heatmaps, [[0, 1]])
    heatmaps_new = flipped_heatmaps[..., ::-1]
    assert_array_almost_equal(heatmaps[:, 0], heatmaps_new[:, 1])
    assert_array_almost_equal(heatmaps[:, 1], heatmaps_new[:, 0])

    ori_heatmaps = heatmaps.copy()
    # test in-place flip
    heatmaps = heatmaps[:, :, :, ::-1]
    assert_array_almost_equal(ori_heatmaps[:, :, :, ::-1], heatmaps)


def test_transform_preds():
    coords = np.random.random([2, 2])
    center = np.array([50, 50])
    scale = np.array([100 / 200.0, 100 / 200.0])
    size = np.array([100, 100])
    ans = transform_preds(coords, center, scale, size)
    assert_array_almost_equal(coords, ans)


def test_get_affine_transform():
    center = np.array([50, 50])
    scale = np.array([100 / 200.0, 100 / 200.0])
    size = np.array([100, 100])
    ans = get_affine_transform(center, scale, 0, size)
    trans = np.array([[1, 0, 0], [0, 1, 0]])
    assert_array_almost_equal(trans, ans)
