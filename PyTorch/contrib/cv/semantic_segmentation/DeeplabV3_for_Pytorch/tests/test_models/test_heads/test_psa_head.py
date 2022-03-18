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

from mmseg.models.decode_heads import PSAHead
from .utils import _conv_has_norm, to_cuda


def test_psa_head():

    with pytest.raises(AssertionError):
        # psa_type must be in 'bi-direction', 'collect', 'distribute'
        PSAHead(
            in_channels=32,
            channels=16,
            num_classes=19,
            mask_size=(39, 39),
            psa_type='gather')

    # test no norm_cfg
    head = PSAHead(
        in_channels=32, channels=16, num_classes=19, mask_size=(39, 39))
    assert not _conv_has_norm(head, sync_bn=False)

    # test with norm_cfg
    head = PSAHead(
        in_channels=32,
        channels=16,
        num_classes=19,
        mask_size=(39, 39),
        norm_cfg=dict(type='SyncBN'))
    assert _conv_has_norm(head, sync_bn=True)

    # test 'bi-direction' psa_type
    inputs = [torch.randn(1, 32, 39, 39)]
    head = PSAHead(
        in_channels=32, channels=16, num_classes=19, mask_size=(39, 39))
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 39, 39)

    # test 'bi-direction' psa_type, shrink_factor=1
    inputs = [torch.randn(1, 32, 39, 39)]
    head = PSAHead(
        in_channels=32,
        channels=16,
        num_classes=19,
        mask_size=(39, 39),
        shrink_factor=1)
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 39, 39)

    # test 'bi-direction' psa_type with soft_max
    inputs = [torch.randn(1, 32, 39, 39)]
    head = PSAHead(
        in_channels=32,
        channels=16,
        num_classes=19,
        mask_size=(39, 39),
        psa_softmax=True)
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 39, 39)

    # test 'collect' psa_type
    inputs = [torch.randn(1, 32, 39, 39)]
    head = PSAHead(
        in_channels=32,
        channels=16,
        num_classes=19,
        mask_size=(39, 39),
        psa_type='collect')
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 39, 39)

    # test 'collect' psa_type, shrink_factor=1
    inputs = [torch.randn(1, 32, 39, 39)]
    head = PSAHead(
        in_channels=32,
        channels=16,
        num_classes=19,
        mask_size=(39, 39),
        shrink_factor=1,
        psa_type='collect')
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 39, 39)

    # test 'collect' psa_type, shrink_factor=1, compact=True
    inputs = [torch.randn(1, 32, 39, 39)]
    head = PSAHead(
        in_channels=32,
        channels=16,
        num_classes=19,
        mask_size=(39, 39),
        psa_type='collect',
        shrink_factor=1,
        compact=True)
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 39, 39)

    # test 'distribute' psa_type
    inputs = [torch.randn(1, 32, 39, 39)]
    head = PSAHead(
        in_channels=32,
        channels=16,
        num_classes=19,
        mask_size=(39, 39),
        psa_type='distribute')
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 39, 39)
