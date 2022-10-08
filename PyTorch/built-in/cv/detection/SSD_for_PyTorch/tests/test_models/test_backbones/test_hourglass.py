# Copyright 2022 Huawei Technologies Co., Ltd.
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
# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmdet.models.backbones.hourglass import HourglassNet


def test_hourglass_backbone():
    with pytest.raises(AssertionError):
        # HourglassNet's num_stacks should larger than 0
        HourglassNet(num_stacks=0)

    with pytest.raises(AssertionError):
        # len(stage_channels) should equal len(stage_blocks)
        HourglassNet(
            stage_channels=[256, 256, 384, 384, 384],
            stage_blocks=[2, 2, 2, 2, 2, 4])

    with pytest.raises(AssertionError):
        # len(stage_channels) should lagrer than downsample_times
        HourglassNet(
            downsample_times=5,
            stage_channels=[256, 256, 384, 384, 384],
            stage_blocks=[2, 2, 2, 2, 2])

    # Test HourglassNet-52
    model = HourglassNet(
        num_stacks=1,
        stage_channels=(64, 64, 96, 96, 96, 128),
        feat_channel=64)
    model.train()

    imgs = torch.randn(1, 3, 256, 256)
    feat = model(imgs)
    assert len(feat) == 1
    assert feat[0].shape == torch.Size([1, 64, 64, 64])

    # Test HourglassNet-104
    model = HourglassNet(
        num_stacks=2,
        stage_channels=(64, 64, 96, 96, 96, 128),
        feat_channel=64)
    model.train()

    imgs = torch.randn(1, 3, 256, 256)
    feat = model(imgs)
    assert len(feat) == 2
    assert feat[0].shape == torch.Size([1, 64, 64, 64])
    assert feat[1].shape == torch.Size([1, 64, 64, 64])
