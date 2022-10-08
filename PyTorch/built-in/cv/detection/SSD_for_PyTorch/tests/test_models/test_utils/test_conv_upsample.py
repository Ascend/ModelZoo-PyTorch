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

from mmdet.models.utils import ConvUpsample


@pytest.mark.parametrize('num_layers', [0, 1, 2])
def test_conv_upsample(num_layers):
    num_upsample = num_layers if num_layers > 0 else 0
    num_layers = num_layers if num_layers > 0 else 1
    layer = ConvUpsample(
        10,
        5,
        num_layers=num_layers,
        num_upsample=num_upsample,
        conv_cfg=None,
        norm_cfg=None)

    size = 5
    x = torch.randn((1, 10, size, size))
    size = size * pow(2, num_upsample)
    x = layer(x)
    assert x.shape[-2:] == (size, size)
