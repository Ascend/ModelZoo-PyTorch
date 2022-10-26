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
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.utils import AdaptiveAvgPool2d, adaptive_avg_pool2d

if torch.__version__ != 'parrots':
    torch_version = '1.7'
else:
    torch_version = 'parrots'


@patch('torch.__version__', torch_version)
def test_adaptive_avg_pool2d():
    # Test the empty batch dimension
    # Test the two input conditions
    x_empty = torch.randn(0, 3, 4, 5)
    # 1. tuple[int, int]
    wrapper_out = adaptive_avg_pool2d(x_empty, (2, 2))
    assert wrapper_out.shape == (0, 3, 2, 2)
    # 2. int
    wrapper_out = adaptive_avg_pool2d(x_empty, 2)
    assert wrapper_out.shape == (0, 3, 2, 2)

    # wrapper op with 3-dim input
    x_normal = torch.randn(3, 3, 4, 5)
    wrapper_out = adaptive_avg_pool2d(x_normal, (2, 2))
    ref_out = F.adaptive_avg_pool2d(x_normal, (2, 2))
    assert wrapper_out.shape == (3, 3, 2, 2)
    assert torch.equal(wrapper_out, ref_out)

    wrapper_out = adaptive_avg_pool2d(x_normal, 2)
    ref_out = F.adaptive_avg_pool2d(x_normal, 2)
    assert wrapper_out.shape == (3, 3, 2, 2)
    assert torch.equal(wrapper_out, ref_out)


@patch('torch.__version__', torch_version)
def test_AdaptiveAvgPool2d():
    # Test the empty batch dimension
    x_empty = torch.randn(0, 3, 4, 5)
    # Test the four input conditions
    # 1. tuple[int, int]
    wrapper = AdaptiveAvgPool2d((2, 2))
    wrapper_out = wrapper(x_empty)
    assert wrapper_out.shape == (0, 3, 2, 2)

    # 2. int
    wrapper = AdaptiveAvgPool2d(2)
    wrapper_out = wrapper(x_empty)
    assert wrapper_out.shape == (0, 3, 2, 2)

    # 3. tuple[None, int]
    wrapper = AdaptiveAvgPool2d((None, 2))
    wrapper_out = wrapper(x_empty)
    assert wrapper_out.shape == (0, 3, 4, 2)

    # 3. tuple[int, None]
    wrapper = AdaptiveAvgPool2d((2, None))
    wrapper_out = wrapper(x_empty)
    assert wrapper_out.shape == (0, 3, 2, 5)

    # Test the normal batch dimension
    x_normal = torch.randn(3, 3, 4, 5)
    wrapper = AdaptiveAvgPool2d((2, 2))
    ref = nn.AdaptiveAvgPool2d((2, 2))
    wrapper_out = wrapper(x_normal)
    ref_out = ref(x_normal)
    assert wrapper_out.shape == (3, 3, 2, 2)
    assert torch.equal(wrapper_out, ref_out)

    wrapper = AdaptiveAvgPool2d(2)
    ref = nn.AdaptiveAvgPool2d(2)
    wrapper_out = wrapper(x_normal)
    ref_out = ref(x_normal)
    assert wrapper_out.shape == (3, 3, 2, 2)
    assert torch.equal(wrapper_out, ref_out)

    wrapper = AdaptiveAvgPool2d((None, 2))
    ref = nn.AdaptiveAvgPool2d((None, 2))
    wrapper_out = wrapper(x_normal)
    ref_out = ref(x_normal)
    assert wrapper_out.shape == (3, 3, 4, 2)
    assert torch.equal(wrapper_out, ref_out)

    wrapper = AdaptiveAvgPool2d((2, None))
    ref = nn.AdaptiveAvgPool2d((2, None))
    wrapper_out = wrapper(x_normal)
    ref_out = ref(x_normal)
    assert wrapper_out.shape == (3, 3, 2, 5)
    assert torch.equal(wrapper_out, ref_out)
