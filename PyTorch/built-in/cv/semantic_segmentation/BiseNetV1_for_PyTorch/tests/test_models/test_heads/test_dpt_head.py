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

from mmseg.models.decode_heads import DPTHead


def test_dpt_head():

    with pytest.raises(AssertionError):
        # input_transform must be 'multiple_select'
        head = DPTHead(
            in_channels=[768, 768, 768, 768],
            channels=4,
            num_classes=19,
            in_index=[0, 1, 2, 3])

    head = DPTHead(
        in_channels=[768, 768, 768, 768],
        channels=4,
        num_classes=19,
        in_index=[0, 1, 2, 3],
        input_transform='multiple_select')

    inputs = [[torch.randn(4, 768, 2, 2),
               torch.randn(4, 768)] for _ in range(4)]
    output = head(inputs)
    assert output.shape == torch.Size((4, 19, 16, 16))

    # test readout operation
    head = DPTHead(
        in_channels=[768, 768, 768, 768],
        channels=4,
        num_classes=19,
        in_index=[0, 1, 2, 3],
        input_transform='multiple_select',
        readout_type='add')
    output = head(inputs)
    assert output.shape == torch.Size((4, 19, 16, 16))

    head = DPTHead(
        in_channels=[768, 768, 768, 768],
        channels=4,
        num_classes=19,
        in_index=[0, 1, 2, 3],
        input_transform='multiple_select',
        readout_type='project')
    output = head(inputs)
    assert output.shape == torch.Size((4, 19, 16, 16))
