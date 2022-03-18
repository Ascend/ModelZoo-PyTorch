#     Copyright 2021 Huawei
#     Copyright 2021 Huawei Technologies Co., Ltd
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
#

import pytest
import torch

from mmseg.models.decode_heads import ASPPHead, DepthwiseSeparableASPPHead
from .utils import _conv_has_norm, to_cuda


def test_aspp_head():

    with pytest.raises(AssertionError):
        # pool_scales must be list|tuple
        ASPPHead(in_channels=32, channels=16, num_classes=19, dilations=1)

    # test no norm_cfg
    head = ASPPHead(in_channels=32, channels=16, num_classes=19)
    assert not _conv_has_norm(head, sync_bn=False)

    # test with norm_cfg
    head = ASPPHead(
        in_channels=32,
        channels=16,
        num_classes=19,
        norm_cfg=dict(type='SyncBN'))
    assert _conv_has_norm(head, sync_bn=True)

    inputs = [torch.randn(1, 32, 45, 45)]
    head = ASPPHead(
        in_channels=32, channels=16, num_classes=19, dilations=(1, 12, 24))
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    assert head.aspp_modules[0].conv.dilation == (1, 1)
    assert head.aspp_modules[1].conv.dilation == (12, 12)
    assert head.aspp_modules[2].conv.dilation == (24, 24)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)


def test_dw_aspp_head():

    # test w.o. c1
    inputs = [torch.randn(1, 32, 45, 45)]
    head = DepthwiseSeparableASPPHead(
        c1_in_channels=0,
        c1_channels=0,
        in_channels=32,
        channels=16,
        num_classes=19,
        dilations=(1, 12, 24))
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    assert head.c1_bottleneck is None
    assert head.aspp_modules[0].conv.dilation == (1, 1)
    assert head.aspp_modules[1].depthwise_conv.dilation == (12, 12)
    assert head.aspp_modules[2].depthwise_conv.dilation == (24, 24)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)

    # test with c1
    inputs = [torch.randn(1, 8, 45, 45), torch.randn(1, 32, 21, 21)]
    head = DepthwiseSeparableASPPHead(
        c1_in_channels=8,
        c1_channels=4,
        in_channels=32,
        channels=16,
        num_classes=19,
        dilations=(1, 12, 24))
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    assert head.c1_bottleneck.in_channels == 8
    assert head.c1_bottleneck.out_channels == 4
    assert head.aspp_modules[0].conv.dilation == (1, 1)
    assert head.aspp_modules[1].depthwise_conv.dilation == (12, 12)
    assert head.aspp_modules[2].depthwise_conv.dilation == (24, 24)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)
