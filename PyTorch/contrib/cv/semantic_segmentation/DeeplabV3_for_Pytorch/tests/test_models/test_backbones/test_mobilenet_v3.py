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

from mmseg.models.backbones import MobileNetV3


def test_mobilenet_v3():
    with pytest.raises(AssertionError):
        # check invalid arch
        MobileNetV3('big')

    with pytest.raises(AssertionError):
        # check invalid reduction_factor
        MobileNetV3(reduction_factor=0)

    with pytest.raises(ValueError):
        # check invalid out_indices
        MobileNetV3(out_indices=(0, 1, 15))

    with pytest.raises(ValueError):
        # check invalid frozen_stages
        MobileNetV3(frozen_stages=15)

    with pytest.raises(TypeError):
        # check invalid pretrained
        model = MobileNetV3()
        model.init_weights(pretrained=8)

    # Test MobileNetV3 with default settings
    model = MobileNetV3()
    model.init_weights()
    model.train()

    imgs = torch.randn(2, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 3
    assert feat[0].shape == (2, 16, 112, 112)
    assert feat[1].shape == (2, 16, 56, 56)
    assert feat[2].shape == (2, 576, 28, 28)

    # Test MobileNetV3 with arch = 'large'
    model = MobileNetV3(arch='large', out_indices=(1, 3, 16))
    model.init_weights()
    model.train()

    imgs = torch.randn(2, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 3
    assert feat[0].shape == (2, 16, 112, 112)
    assert feat[1].shape == (2, 24, 56, 56)
    assert feat[2].shape == (2, 960, 28, 28)

    # Test MobileNetV3 with norm_eval True, with_cp True and frozen_stages=5
    model = MobileNetV3(norm_eval=True, with_cp=True, frozen_stages=5)
    with pytest.raises(TypeError):
        # check invalid pretrained
        model.init_weights(pretrained=8)
    model.init_weights()
    model.train()

    imgs = torch.randn(2, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 3
    assert feat[0].shape == (2, 16, 112, 112)
    assert feat[1].shape == (2, 16, 56, 56)
    assert feat[2].shape == (2, 576, 28, 28)
