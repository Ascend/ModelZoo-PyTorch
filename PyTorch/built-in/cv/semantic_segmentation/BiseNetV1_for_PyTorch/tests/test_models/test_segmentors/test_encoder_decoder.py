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
from mmcv import ConfigDict

from mmseg.models import build_segmentor
from .utils import _segmentor_forward_train_test


def test_encoder_decoder():

    # test 1 decode head, w.o. aux head

    cfg = ConfigDict(
        type='EncoderDecoder',
        backbone=dict(type='ExampleBackbone'),
        decode_head=dict(type='ExampleDecodeHead'),
        train_cfg=None,
        test_cfg=dict(mode='whole'))
    segmentor = build_segmentor(cfg)
    _segmentor_forward_train_test(segmentor)

    # test out_channels == 1
    segmentor.out_channels = 1
    segmentor.decode_head.out_channels = 1
    segmentor.decode_head.threshold = 0.3
    _segmentor_forward_train_test(segmentor)

    # test slide mode
    cfg.test_cfg = ConfigDict(mode='slide', crop_size=(3, 3), stride=(2, 2))
    segmentor = build_segmentor(cfg)
    _segmentor_forward_train_test(segmentor)

    # test 1 decode head, 1 aux head
    cfg = ConfigDict(
        type='EncoderDecoder',
        backbone=dict(type='ExampleBackbone'),
        decode_head=dict(type='ExampleDecodeHead'),
        auxiliary_head=dict(type='ExampleDecodeHead'))
    cfg.test_cfg = ConfigDict(mode='whole')
    segmentor = build_segmentor(cfg)
    _segmentor_forward_train_test(segmentor)

    # test 1 decode head, 2 aux head
    cfg = ConfigDict(
        type='EncoderDecoder',
        backbone=dict(type='ExampleBackbone'),
        decode_head=dict(type='ExampleDecodeHead'),
        auxiliary_head=[
            dict(type='ExampleDecodeHead'),
            dict(type='ExampleDecodeHead')
        ])
    cfg.test_cfg = ConfigDict(mode='whole')
    segmentor = build_segmentor(cfg)
    _segmentor_forward_train_test(segmentor)
