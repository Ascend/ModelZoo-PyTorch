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

from mmseg.models import Feature2Pyramid


def test_feature2pyramid():
    # test
    rescales = [4, 2, 1, 0.5]
    embed_dim = 64
    inputs = [torch.randn(1, embed_dim, 32, 32) for i in range(len(rescales))]

    fpn = Feature2Pyramid(
        embed_dim, rescales, norm_cfg=dict(type='BN', requires_grad=True))
    outputs = fpn(inputs)
    assert outputs[0].shape == torch.Size([1, 64, 128, 128])
    assert outputs[1].shape == torch.Size([1, 64, 64, 64])
    assert outputs[2].shape == torch.Size([1, 64, 32, 32])
    assert outputs[3].shape == torch.Size([1, 64, 16, 16])

    # test rescales = [2, 1, 0.5, 0.25]
    rescales = [2, 1, 0.5, 0.25]
    inputs = [torch.randn(1, embed_dim, 32, 32) for i in range(len(rescales))]

    fpn = Feature2Pyramid(
        embed_dim, rescales, norm_cfg=dict(type='BN', requires_grad=True))
    outputs = fpn(inputs)
    assert outputs[0].shape == torch.Size([1, 64, 64, 64])
    assert outputs[1].shape == torch.Size([1, 64, 32, 32])
    assert outputs[2].shape == torch.Size([1, 64, 16, 16])
    assert outputs[3].shape == torch.Size([1, 64, 8, 8])

    # test rescales = [4, 2, 0.25, 0]
    rescales = [4, 2, 0.25, 0]
    with pytest.raises(KeyError):
        fpn = Feature2Pyramid(
            embed_dim, rescales, norm_cfg=dict(type='BN', requires_grad=True))
