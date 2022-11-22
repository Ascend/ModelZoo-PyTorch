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

from mmseg.models.decode_heads import SegformerHead


def test_segformer_head():
    with pytest.raises(AssertionError):
        # `in_channels` must have same length as `in_index`
        SegformerHead(
            in_channels=(1, 2, 3), in_index=(0, 1), channels=5, num_classes=2)

    H, W = (64, 64)
    in_channels = (32, 64, 160, 256)
    shapes = [(H // 2**(i + 2), W // 2**(i + 2))
              for i in range(len(in_channels))]
    model = SegformerHead(
        in_channels=in_channels,
        in_index=[0, 1, 2, 3],
        channels=256,
        num_classes=19)

    with pytest.raises(IndexError):
        # in_index must match the input feature maps.
        inputs = [
            torch.randn((1, in_channel, *shape))
            for in_channel, shape in zip(in_channels, shapes)
        ][:3]
        temp = model(inputs)

    # Normal Input
    # ((1, 32, 16, 16), (1, 64, 8, 8), (1, 160, 4, 4), (1, 256, 2, 2)
    inputs = [
        torch.randn((1, in_channel, *shape))
        for in_channel, shape in zip(in_channels, shapes)
    ]
    temp = model(inputs)

    assert temp.shape == (1, 19, H // 4, W // 4)
