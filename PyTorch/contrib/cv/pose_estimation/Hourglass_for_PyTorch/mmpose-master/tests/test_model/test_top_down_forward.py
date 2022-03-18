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
import torch

from mmpose.models.detectors import TopDown


def test_topdown_forward():
    model_cfg = dict(
        type='TopDown',
        pretrained=None,
        backbone=dict(type='ResNet', depth=18),
        keypoint_head=dict(
            type='TopDownSimpleHead',
            in_channels=512,
            out_channels=17,
        ),
        train_cfg=dict(),
        test_cfg=dict(
            flip_test=False,
            post_process=True,
            shift_heatmap=True,
            unbiased_decoding=False,
            modulate_kernel=11),
        loss_pose=dict(type='JointsMSELoss', use_target_weight=True))

    detector = TopDown(model_cfg['backbone'], model_cfg['keypoint_head'],
                       model_cfg['train_cfg'], model_cfg['test_cfg'],
                       model_cfg['pretrained'], model_cfg['loss_pose'])

    detector.init_weights()

    input_shape = (1, 3, 256, 256)
    mm_inputs = _demo_mm_inputs(input_shape)

    imgs = mm_inputs.pop('imgs')
    target = mm_inputs.pop('target')
    target_weight = mm_inputs.pop('target_weight')
    img_metas = mm_inputs.pop('img_metas')

    # Test forward train
    losses = detector.forward(
        imgs, target, target_weight, img_metas, return_loss=True)
    assert isinstance(losses, dict)

    # Test forward test
    with torch.no_grad():
        _ = detector.forward(imgs, img_metas=img_metas, return_loss=False)

    # flip test
    model_cfg = dict(
        type='TopDown',
        pretrained=None,
        backbone=dict(
            type='HourglassNet',
            num_stacks=1,
        ),
        keypoint_head=dict(
            type='TopDownMultiStageHead',
            in_channels=256,
            out_channels=17,
            num_stages=1,
            num_deconv_layers=0,
            extra=dict(final_conv_kernel=1, ),
        ),
        train_cfg=dict(),
        test_cfg=dict(
            flip_test=True,
            post_process=True,
            shift_heatmap=True,
            unbiased_decoding=False,
            modulate_kernel=11),
        loss_pose=dict(type='JointsMSELoss', use_target_weight=False))

    detector = TopDown(model_cfg['backbone'], model_cfg['keypoint_head'],
                       model_cfg['train_cfg'], model_cfg['test_cfg'],
                       model_cfg['pretrained'], model_cfg['loss_pose'])

    # Test forward train
    losses = detector.forward(
        imgs, target, target_weight, img_metas, return_loss=True)
    assert isinstance(losses, dict)

    # Test forward test
    with torch.no_grad():
        _ = detector.forward(imgs, img_metas=img_metas, return_loss=False)

    model_cfg = dict(
        type='TopDown',
        pretrained=None,
        backbone=dict(
            type='HourglassNet',
            num_stacks=1,
        ),
        keypoint_head=dict(
            type='TopDownMultiStageHead',
            in_channels=256,
            out_channels=17,
            num_stages=1,
            num_deconv_layers=0,
            extra=dict(final_conv_kernel=1, ),
        ),
        train_cfg=dict(loss_weights=([1])),
        test_cfg=dict(
            flip_test=False,
            post_process=True,
            shift_heatmap=True,
            unbiased_decoding=False,
            modulate_kernel=11),
        loss_pose=[dict(type='JointsMSELoss', use_target_weight=True)])

    detector = TopDown(model_cfg['backbone'], model_cfg['keypoint_head'],
                       model_cfg['train_cfg'], model_cfg['test_cfg'],
                       model_cfg['pretrained'], model_cfg['loss_pose'])

    detector.init_weights()

    input_shape = (1, 3, 256, 256)
    mm_inputs = _demo_mm_inputs(input_shape, num_outputs=1)

    imgs = mm_inputs.pop('imgs')
    target = mm_inputs.pop('target')
    target_weight = mm_inputs.pop('target_weight')
    img_metas = mm_inputs.pop('img_metas')

    # Test forward train
    losses = detector.forward(
        imgs, target, target_weight, img_metas, return_loss=True)
    assert isinstance(losses, dict)
    # Test forward test
    with torch.no_grad():
        _ = detector.forward(imgs, img_metas=img_metas, return_loss=False)

    model_cfg = dict(
        type='TopDown',
        pretrained=None,
        backbone=dict(
            type='RSN',
            unit_channels=256,
            num_stages=1,
            num_units=4,
            num_blocks=[2, 2, 2, 2],
            num_steps=4,
            norm_cfg=dict(type='BN')),
        keypoint_head=dict(
            type='TopDownMSMUHead',
            out_shape=(64, 48),
            unit_channels=256,
            out_channels=17,
            num_stages=1,
            num_units=4,
            use_prm=False,
            norm_cfg=dict(type='BN')),
        train_cfg=dict(num_units=4),
        test_cfg=dict(
            flip_test=True,
            post_process=True,
            shift_heatmap=True,
            unbiased_decoding=False,
            modulate_kernel=11),
        loss_pose=[dict(type='JointsMSELoss', use_target_weight=True)] * 3 +
        [dict(type='JointsOHKMMSELoss', use_target_weight=True)])
    detector = TopDown(model_cfg['backbone'], model_cfg['keypoint_head'],
                       model_cfg['train_cfg'], model_cfg['test_cfg'],
                       model_cfg['pretrained'], model_cfg['loss_pose'])

    detector.init_weights()

    input_shape = (1, 3, 256, 192)
    mm_inputs = _demo_mm_inputs(input_shape, num_outputs=4)

    imgs = mm_inputs.pop('imgs')
    target = mm_inputs.pop('target')
    target_weight = mm_inputs.pop('target_weight')
    img_metas = mm_inputs.pop('img_metas')

    # Test forward train
    losses = detector.forward(
        imgs, target, target_weight, img_metas, return_loss=True)
    assert isinstance(losses, dict)
    # Test forward test
    with torch.no_grad():
        _ = detector.forward(imgs, img_metas=img_metas, return_loss=False)
        _ = detector.forward_dummy(imgs)


def _demo_mm_inputs(input_shape=(1, 3, 256, 256), num_outputs=None):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions
    """
    (N, C, H, W) = input_shape

    rng = np.random.RandomState(0)

    imgs = rng.rand(*input_shape)
    if num_outputs is not None:
        target = np.zeros([N, num_outputs, 17, H // 4, W // 4],
                          dtype=np.float32)
        target_weight = np.ones([N, num_outputs, 17, 1], dtype=np.float32)
    else:
        target = np.zeros([N, 17, H // 4, W // 4], dtype=np.float32)
        target_weight = np.ones([N, 17, 1], dtype=np.float32)

    img_metas = [{
        'img_shape': (H, W, C),
        'center': np.array([W / 2, H / 2]),
        'scale': np.array([0.5, 0.5]),
        'bbox_score': 1.0,
        'flip_pairs': [],
        'inference_channel': np.arange(17),
        'image_file': '<demo>.png',
    } for _ in range(N)]

    mm_inputs = {
        'imgs': torch.FloatTensor(imgs).requires_grad_(True),
        'target': torch.FloatTensor(target),
        'target_weight': torch.FloatTensor(target_weight),
        'img_metas': img_metas
    }
    return mm_inputs
