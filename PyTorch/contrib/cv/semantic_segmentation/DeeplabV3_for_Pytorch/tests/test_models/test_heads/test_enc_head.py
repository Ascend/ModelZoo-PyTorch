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

import torch

from mmseg.models.decode_heads import EncHead
from .utils import to_cuda


def test_enc_head():
    # with se_loss, w.o. lateral
    inputs = [torch.randn(1, 32, 21, 21)]
    head = EncHead(
        in_channels=[32], channels=16, num_classes=19, in_index=[-1])
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert isinstance(outputs, tuple) and len(outputs) == 2
    assert outputs[0].shape == (1, head.num_classes, 21, 21)
    assert outputs[1].shape == (1, head.num_classes)

    # w.o se_loss, w.o. lateral
    inputs = [torch.randn(1, 32, 21, 21)]
    head = EncHead(
        in_channels=[32],
        channels=16,
        use_se_loss=False,
        num_classes=19,
        in_index=[-1])
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 21, 21)

    # with se_loss, with lateral
    inputs = [torch.randn(1, 16, 45, 45), torch.randn(1, 32, 21, 21)]
    head = EncHead(
        in_channels=[16, 32],
        channels=16,
        add_lateral=True,
        num_classes=19,
        in_index=[-2, -1])
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert isinstance(outputs, tuple) and len(outputs) == 2
    assert outputs[0].shape == (1, head.num_classes, 21, 21)
    assert outputs[1].shape == (1, head.num_classes)
    test_output = head.forward_test(inputs, None, None)
    assert test_output.shape == (1, head.num_classes, 21, 21)
