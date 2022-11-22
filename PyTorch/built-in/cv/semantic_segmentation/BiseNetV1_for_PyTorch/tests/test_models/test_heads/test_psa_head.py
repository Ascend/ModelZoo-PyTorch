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
import pytest
import torch

from mmseg.models.decode_heads import PSAHead
from .utils import _conv_has_norm, to_cuda


def test_psa_head():

    with pytest.raises(AssertionError):
        # psa_type must be in 'bi-direction', 'collect', 'distribute'
        PSAHead(
            in_channels=4,
            channels=2,
            num_classes=19,
            mask_size=(13, 13),
            psa_type='gather')

    # test no norm_cfg
    head = PSAHead(
        in_channels=4, channels=2, num_classes=19, mask_size=(13, 13))
    assert not _conv_has_norm(head, sync_bn=False)

    # test with norm_cfg
    head = PSAHead(
        in_channels=4,
        channels=2,
        num_classes=19,
        mask_size=(13, 13),
        norm_cfg=dict(type='SyncBN'))
    assert _conv_has_norm(head, sync_bn=True)

    # test 'bi-direction' psa_type
    inputs = [torch.randn(1, 4, 13, 13)]
    head = PSAHead(
        in_channels=4, channels=2, num_classes=19, mask_size=(13, 13))
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 13, 13)

    # test 'bi-direction' psa_type, shrink_factor=1
    inputs = [torch.randn(1, 4, 13, 13)]
    head = PSAHead(
        in_channels=4,
        channels=2,
        num_classes=19,
        mask_size=(13, 13),
        shrink_factor=1)
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 13, 13)

    # test 'bi-direction' psa_type with soft_max
    inputs = [torch.randn(1, 4, 13, 13)]
    head = PSAHead(
        in_channels=4,
        channels=2,
        num_classes=19,
        mask_size=(13, 13),
        psa_softmax=True)
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 13, 13)

    # test 'collect' psa_type
    inputs = [torch.randn(1, 4, 13, 13)]
    head = PSAHead(
        in_channels=4,
        channels=2,
        num_classes=19,
        mask_size=(13, 13),
        psa_type='collect')
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 13, 13)

    # test 'collect' psa_type, shrink_factor=1
    inputs = [torch.randn(1, 4, 13, 13)]
    head = PSAHead(
        in_channels=4,
        channels=2,
        num_classes=19,
        mask_size=(13, 13),
        shrink_factor=1,
        psa_type='collect')
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 13, 13)

    # test 'collect' psa_type, shrink_factor=1, compact=True
    inputs = [torch.randn(1, 4, 13, 13)]
    head = PSAHead(
        in_channels=4,
        channels=2,
        num_classes=19,
        mask_size=(13, 13),
        psa_type='collect',
        shrink_factor=1,
        compact=True)
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 13, 13)

    # test 'distribute' psa_type
    inputs = [torch.randn(1, 4, 13, 13)]
    head = PSAHead(
        in_channels=4,
        channels=2,
        num_classes=19,
        mask_size=(13, 13),
        psa_type='distribute')
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 13, 13)
