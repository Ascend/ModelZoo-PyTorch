# -*- coding: utf-8 -*-
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import pytest
import torch

from mmpose.models import CPM


def test_cpm_backbone():
    with pytest.raises(AssertionError):
        # CPM's num_stacks should larger than 0
        CPM(in_channels=3, out_channels=17, num_stages=-1)

    with pytest.raises(AssertionError):
        # CPM's in_channels should be 3
        CPM(in_channels=2, out_channels=17)

    # Test CPM
    model = CPM(in_channels=3, out_channels=17, num_stages=1)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 256, 192)
    feat = model(imgs)
    assert len(feat) == 1
    assert feat[0].shape == torch.Size([1, 17, 32, 24])

    imgs = torch.randn(1, 3, 384, 288)
    feat = model(imgs)
    assert len(feat) == 1
    assert feat[0].shape == torch.Size([1, 17, 48, 36])

    imgs = torch.randn(1, 3, 368, 368)
    feat = model(imgs)
    assert len(feat) == 1
    assert feat[0].shape == torch.Size([1, 17, 46, 46])

    # Test CPM multi-stages
    model = CPM(in_channels=3, out_channels=17, num_stages=2)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 368, 368)
    feat = model(imgs)
    assert len(feat) == 2
    assert feat[0].shape == torch.Size([1, 17, 46, 46])
    assert feat[1].shape == torch.Size([1, 17, 46, 46])
