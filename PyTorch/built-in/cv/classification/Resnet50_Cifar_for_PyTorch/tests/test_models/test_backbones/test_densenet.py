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

from mmcls.models.backbones import DenseNet


def test_assertion():
    with pytest.raises(AssertionError):
        DenseNet(arch='unknown')

    with pytest.raises(AssertionError):
        # DenseNet arch dict should include essential_keys,
        DenseNet(arch=dict(channels=[2, 3, 4, 5]))

    with pytest.raises(AssertionError):
        # DenseNet out_indices should be valid depth.
        DenseNet(out_indices=-100)


def test_DenseNet():

    # Test forward
    model = DenseNet(arch='121')
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 1
    assert feat[0].shape == torch.Size([1, 1024, 7, 7])

    # Test memory efficient option
    model = DenseNet(arch='121', memory_efficient=True)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 1
    assert feat[0].shape == torch.Size([1, 1024, 7, 7])

    # Test drop rate
    model = DenseNet(arch='121', drop_rate=0.05)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 1
    assert feat[0].shape == torch.Size([1, 1024, 7, 7])

    # Test forward with multiple outputs
    model = DenseNet(arch='121', out_indices=(0, 1, 2, 3))

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 128, 28, 28])
    assert feat[1].shape == torch.Size([1, 256, 14, 14])
    assert feat[2].shape == torch.Size([1, 512, 7, 7])
    assert feat[3].shape == torch.Size([1, 1024, 7, 7])

    # Test with custom arch
    model = DenseNet(
        arch={
            'growth_rate': 20,
            'depths': [4, 8, 12, 16, 20],
            'init_channels': 40,
        },
        out_indices=(0, 1, 2, 3, 4))
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 5
    assert feat[0].shape == torch.Size([1, 60, 28, 28])
    assert feat[1].shape == torch.Size([1, 110, 14, 14])
    assert feat[2].shape == torch.Size([1, 175, 7, 7])
    assert feat[3].shape == torch.Size([1, 247, 3, 3])
    assert feat[4].shape == torch.Size([1, 647, 3, 3])

    # Test frozen_stages
    model = DenseNet(arch='121', out_indices=(0, 1, 2, 3), frozen_stages=2)
    model.init_weights()
    model.train()

    for i in range(2):
        assert not model.stages[i].training
        assert not model.transitions[i].training

    for i in range(2, 4):
        assert model.stages[i].training
        assert model.transitions[i].training
