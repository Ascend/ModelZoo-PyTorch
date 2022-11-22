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

from mmseg.models.decode_heads import DMHead
from .utils import _conv_has_norm, to_cuda


def test_dm_head():

    with pytest.raises(AssertionError):
        # filter_sizes must be list|tuple
        DMHead(in_channels=8, channels=4, num_classes=19, filter_sizes=1)

    # test no norm_cfg
    head = DMHead(in_channels=8, channels=4, num_classes=19)
    assert not _conv_has_norm(head, sync_bn=False)

    # test with norm_cfg
    head = DMHead(
        in_channels=8,
        channels=4,
        num_classes=19,
        norm_cfg=dict(type='SyncBN'))
    assert _conv_has_norm(head, sync_bn=True)

    # fusion=True
    inputs = [torch.randn(1, 8, 23, 23)]
    head = DMHead(
        in_channels=8,
        channels=4,
        num_classes=19,
        filter_sizes=(1, 3, 5),
        fusion=True)
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    assert head.fusion is True
    assert head.dcm_modules[0].filter_size == 1
    assert head.dcm_modules[1].filter_size == 3
    assert head.dcm_modules[2].filter_size == 5
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 23, 23)

    # fusion=False
    inputs = [torch.randn(1, 8, 23, 23)]
    head = DMHead(
        in_channels=8,
        channels=4,
        num_classes=19,
        filter_sizes=(1, 3, 5),
        fusion=False)
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    assert head.fusion is False
    assert head.dcm_modules[0].filter_size == 1
    assert head.dcm_modules[1].filter_size == 3
    assert head.dcm_modules[2].filter_size == 5
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 23, 23)
