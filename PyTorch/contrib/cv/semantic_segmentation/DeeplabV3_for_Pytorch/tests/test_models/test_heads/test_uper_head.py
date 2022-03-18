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

from mmseg.models.decode_heads import UPerHead
from .utils import _conv_has_norm, to_cuda


def test_uper_head():

    with pytest.raises(AssertionError):
        # fpn_in_channels must be list|tuple
        UPerHead(in_channels=32, channels=16, num_classes=19)

    # test no norm_cfg
    head = UPerHead(
        in_channels=[32, 16], channels=16, num_classes=19, in_index=[-2, -1])
    assert not _conv_has_norm(head, sync_bn=False)

    # test with norm_cfg
    head = UPerHead(
        in_channels=[32, 16],
        channels=16,
        num_classes=19,
        norm_cfg=dict(type='SyncBN'),
        in_index=[-2, -1])
    assert _conv_has_norm(head, sync_bn=True)

    inputs = [torch.randn(1, 32, 45, 45), torch.randn(1, 16, 21, 21)]
    head = UPerHead(
        in_channels=[32, 16], channels=16, num_classes=19, in_index=[-2, -1])
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 45, 45)
