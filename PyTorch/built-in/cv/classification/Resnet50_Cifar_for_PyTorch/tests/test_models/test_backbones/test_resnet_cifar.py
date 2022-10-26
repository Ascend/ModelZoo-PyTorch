# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest
import torch
from mmcv.utils.parrots_wrapper import _BatchNorm

from mmcls.models.backbones import ResNet_CIFAR


def check_norm_state(modules, train_state):
    """Check if norm layer is in correct train state."""
    for mod in modules:
        if isinstance(mod, _BatchNorm):
            if mod.training != train_state:
                return False
    return True


def test_resnet_cifar():
    # deep_stem must be False
    with pytest.raises(AssertionError):
        ResNet_CIFAR(depth=18, deep_stem=True)

    # test the feature map size when depth is 18
    model = ResNet_CIFAR(depth=18, out_indices=(0, 1, 2, 3))
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 32, 32)
    feat = model.conv1(imgs)
    assert feat.shape == (1, 64, 32, 32)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == (1, 64, 32, 32)
    assert feat[1].shape == (1, 128, 16, 16)
    assert feat[2].shape == (1, 256, 8, 8)
    assert feat[3].shape == (1, 512, 4, 4)

    # test the feature map size when depth is 50
    model = ResNet_CIFAR(depth=50, out_indices=(0, 1, 2, 3))
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 32, 32)
    feat = model.conv1(imgs)
    assert feat.shape == (1, 64, 32, 32)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == (1, 256, 32, 32)
    assert feat[1].shape == (1, 512, 16, 16)
    assert feat[2].shape == (1, 1024, 8, 8)
    assert feat[3].shape == (1, 2048, 4, 4)

    # Test ResNet_CIFAR with first stage frozen
    frozen_stages = 1
    model = ResNet_CIFAR(depth=50, frozen_stages=frozen_stages)
    model.init_weights()
    model.train()
    check_norm_state([model.norm1], False)
    for param in model.conv1.parameters():
        assert param.requires_grad is False
    for i in range(1, frozen_stages + 1):
        layer = getattr(model, f'layer{i}')
        for mod in layer.modules():
            if isinstance(mod, _BatchNorm):
                assert mod.training is False
        for param in layer.parameters():
            assert param.requires_grad is False
