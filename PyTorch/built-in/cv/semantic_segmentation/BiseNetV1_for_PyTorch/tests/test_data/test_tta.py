# Copyright (c) Facebook, Inc. and its affiliates.
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
# --------------------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import pytest
from mmcv.utils import build_from_cfg

from mmseg.datasets.builder import PIPELINES


def test_multi_scale_flip_aug():
    # test assertion if img_scale=None, img_ratios=1 (not float).
    with pytest.raises(AssertionError):
        tta_transform = dict(
            type='MultiScaleFlipAug',
            img_scale=None,
            img_ratios=1,
            transforms=[dict(type='Resize', keep_ratio=False)],
        )
        build_from_cfg(tta_transform, PIPELINES)

    # test assertion if img_scale=None, img_ratios=None.
    with pytest.raises(AssertionError):
        tta_transform = dict(
            type='MultiScaleFlipAug',
            img_scale=None,
            img_ratios=None,
            transforms=[dict(type='Resize', keep_ratio=False)],
        )
        build_from_cfg(tta_transform, PIPELINES)

    # test assertion if img_scale=(512, 512), img_ratios=1 (not float).
    with pytest.raises(AssertionError):
        tta_transform = dict(
            type='MultiScaleFlipAug',
            img_scale=(512, 512),
            img_ratios=1,
            transforms=[dict(type='Resize', keep_ratio=False)],
        )
        build_from_cfg(tta_transform, PIPELINES)

    tta_transform = dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        img_ratios=[0.5, 1.0, 2.0],
        flip=False,
        transforms=[dict(type='Resize', keep_ratio=False)],
    )
    tta_module = build_from_cfg(tta_transform, PIPELINES)

    results = dict()
    # (288, 512, 3)
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../data/color.jpg'), 'color')
    results['img'] = img
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0

    tta_results = tta_module(results.copy())
    assert tta_results['scale'] == [(256, 256), (512, 512), (1024, 1024)]
    assert tta_results['flip'] == [False, False, False]

    tta_transform = dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        img_ratios=[0.5, 1.0, 2.0],
        flip=True,
        transforms=[dict(type='Resize', keep_ratio=False)],
    )
    tta_module = build_from_cfg(tta_transform, PIPELINES)
    tta_results = tta_module(results.copy())
    assert tta_results['scale'] == [(256, 256), (256, 256), (512, 512),
                                    (512, 512), (1024, 1024), (1024, 1024)]
    assert tta_results['flip'] == [False, True, False, True, False, True]

    tta_transform = dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        img_ratios=1.0,
        flip=False,
        transforms=[dict(type='Resize', keep_ratio=False)],
    )
    tta_module = build_from_cfg(tta_transform, PIPELINES)
    tta_results = tta_module(results.copy())
    assert tta_results['scale'] == [(512, 512)]
    assert tta_results['flip'] == [False]

    tta_transform = dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        img_ratios=1.0,
        flip=True,
        transforms=[dict(type='Resize', keep_ratio=False)],
    )
    tta_module = build_from_cfg(tta_transform, PIPELINES)
    tta_results = tta_module(results.copy())
    assert tta_results['scale'] == [(512, 512), (512, 512)]
    assert tta_results['flip'] == [False, True]

    tta_transform = dict(
        type='MultiScaleFlipAug',
        img_scale=None,
        img_ratios=[0.5, 1.0, 2.0],
        flip=False,
        transforms=[dict(type='Resize', keep_ratio=False)],
    )
    tta_module = build_from_cfg(tta_transform, PIPELINES)
    tta_results = tta_module(results.copy())
    assert tta_results['scale'] == [(256, 144), (512, 288), (1024, 576)]
    assert tta_results['flip'] == [False, False, False]

    tta_transform = dict(
        type='MultiScaleFlipAug',
        img_scale=None,
        img_ratios=[0.5, 1.0, 2.0],
        flip=True,
        transforms=[dict(type='Resize', keep_ratio=False)],
    )
    tta_module = build_from_cfg(tta_transform, PIPELINES)
    tta_results = tta_module(results.copy())
    assert tta_results['scale'] == [(256, 144), (256, 144), (512, 288),
                                    (512, 288), (1024, 576), (1024, 576)]
    assert tta_results['flip'] == [False, True, False, True, False, True]

    tta_transform = dict(
        type='MultiScaleFlipAug',
        img_scale=[(256, 256), (512, 512), (1024, 1024)],
        img_ratios=None,
        flip=False,
        transforms=[dict(type='Resize', keep_ratio=False)],
    )
    tta_module = build_from_cfg(tta_transform, PIPELINES)
    tta_results = tta_module(results.copy())
    assert tta_results['scale'] == [(256, 256), (512, 512), (1024, 1024)]
    assert tta_results['flip'] == [False, False, False]

    tta_transform = dict(
        type='MultiScaleFlipAug',
        img_scale=[(256, 256), (512, 512), (1024, 1024)],
        img_ratios=None,
        flip=True,
        transforms=[dict(type='Resize', keep_ratio=False)],
    )
    tta_module = build_from_cfg(tta_transform, PIPELINES)
    tta_results = tta_module(results.copy())
    assert tta_results['scale'] == [(256, 256), (256, 256), (512, 512),
                                    (512, 512), (1024, 1024), (1024, 1024)]
    assert tta_results['flip'] == [False, True, False, True, False, True]

    #  test assertion if flip is True and Pad executed before RandomFlip
    with pytest.raises(AssertionError):
        tta_transform = dict(
            type='MultiScaleFlipAug',
            img_scale=[(256, 256), (512, 512), (1024, 1024)],
            img_ratios=None,
            flip=True,
            transforms=[
                dict(type='Resize', keep_ratio=False),
                dict(type='Pad', size_divisor=32),
                dict(type='RandomFlip'),
            ])
        tta_module = build_from_cfg(tta_transform, PIPELINES)

    tta_transform = dict(
        type='MultiScaleFlipAug',
        img_scale=[(256, 256), (512, 512), (1024, 1024)],
        img_ratios=None,
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Pad', size_divisor=32),
        ])
    tta_module = build_from_cfg(tta_transform, PIPELINES)
    tta_results = tta_module(results.copy())
    assert tta_results['scale'] == [(256, 256), (256, 256), (512, 512),
                                    (512, 512), (1024, 1024), (1024, 1024)]
    assert tta_results['flip'] == [False, True, False, True, False, True]
    assert tta_results['img_shape'] == [(144, 256, 3), (144, 256, 3),
                                        (288, 512, 3), (288, 512, 3),
                                        (576, 1024, 3), (576, 1024, 3)]
    assert tta_results['pad_shape'] == [(160, 256, 3), (160, 256, 3),
                                        (288, 512, 3), (288, 512, 3),
                                        (576, 1024, 3), (576, 1024, 3)]
    for i in range(len(tta_results['img'])):
        assert tta_results['img'][i].shape == tta_results['pad_shape'][i]
