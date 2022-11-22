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

from mmseg.models.necks import JPU


def test_fastfcn_neck():
    # Test FastFCN Standard Forward
    model = JPU(
        in_channels=(64, 128, 256),
        mid_channels=64,
        start_level=0,
        end_level=-1,
        dilations=(1, 2, 4, 8),
    )
    model.init_weights()
    model.train()
    batch_size = 1
    input = [
        torch.randn(batch_size, 64, 64, 128),
        torch.randn(batch_size, 128, 32, 64),
        torch.randn(batch_size, 256, 16, 32)
    ]
    feat = model(input)

    assert len(feat) == 3
    assert feat[0].shape == torch.Size([batch_size, 64, 64, 128])
    assert feat[1].shape == torch.Size([batch_size, 128, 32, 64])
    assert feat[2].shape == torch.Size([batch_size, 256, 64, 128])

    with pytest.raises(AssertionError):
        # FastFCN input and in_channels constraints.
        JPU(in_channels=(256, 64, 128), start_level=0, end_level=5)

    # Test not default start_level
    model = JPU(in_channels=(64, 128, 256), start_level=1, end_level=-1)
    input = [
        torch.randn(batch_size, 64, 64, 128),
        torch.randn(batch_size, 128, 32, 64),
        torch.randn(batch_size, 256, 16, 32)
    ]
    feat = model(input)
    assert len(feat) == 2
    assert feat[0].shape == torch.Size([batch_size, 128, 32, 64])
    assert feat[1].shape == torch.Size([batch_size, 2048, 32, 64])
