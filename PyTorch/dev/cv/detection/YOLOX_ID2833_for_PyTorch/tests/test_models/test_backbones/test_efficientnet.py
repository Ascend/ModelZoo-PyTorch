
# Copyright 2022 Huawei Technologies Co., Ltd
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

# Copyright (c) Open-MMLab. All rights reserved.    
import pytest
import torch

from mmdet.models.backbones import EfficientNet


def test_efficientnet_backbone():
    """Test EfficientNet backbone."""
    with pytest.raises(AssertionError):
        # EfficientNet arch should be a key in EfficientNet.arch_settings
        EfficientNet(arch='c3')

    model = EfficientNet(arch='b0', out_indices=(0, 1, 2, 3, 4, 5, 6))
    model.train()

    imgs = torch.randn(2, 3, 32, 32)
    feat = model(imgs)
    assert len(feat) == 7
    assert feat[0].shape == torch.Size([2, 32, 16, 16])
    assert feat[1].shape == torch.Size([2, 16, 16, 16])
    assert feat[2].shape == torch.Size([2, 24, 8, 8])
    assert feat[3].shape == torch.Size([2, 40, 4, 4])
    assert feat[4].shape == torch.Size([2, 112, 2, 2])
    assert feat[5].shape == torch.Size([2, 320, 1, 1])
    assert feat[6].shape == torch.Size([2, 1280, 1, 1])
