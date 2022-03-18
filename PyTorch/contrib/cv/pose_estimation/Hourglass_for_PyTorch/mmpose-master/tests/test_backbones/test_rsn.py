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

from mmpose.models import RSN


def test_rsn_backbone():
    with pytest.raises(AssertionError):
        # RSN's num_stages should larger than 0
        RSN(num_stages=0)
    with pytest.raises(AssertionError):
        # RSN's num_steps should larger than 1
        RSN(num_steps=1)
    with pytest.raises(AssertionError):
        # RSN's num_units should larger than 1
        RSN(num_units=1)
    with pytest.raises(AssertionError):
        # len(num_blocks) should equal num_units
        RSN(num_units=2, num_blocks=[2, 2, 2])

    # Test RSN's outputs
    model = RSN(num_stages=2, num_units=2, num_blocks=[2, 2])
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 511, 511)
    feat = model(imgs)
    assert len(feat) == 2
    assert len(feat[0]) == 2
    assert len(feat[1]) == 2
    assert feat[0][0].shape == torch.Size([1, 256, 64, 64])
    assert feat[0][1].shape == torch.Size([1, 256, 128, 128])
    assert feat[1][0].shape == torch.Size([1, 256, 64, 64])
    assert feat[1][1].shape == torch.Size([1, 256, 128, 128])
