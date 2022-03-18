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

import torch

from mmseg.models import MultiLevelNeck


def test_multilevel_neck():

    # Test init_weights
    MultiLevelNeck([266], 256).init_weights()

    # Test multi feature maps
    in_channels = [256, 512, 1024, 2048]
    inputs = [torch.randn(1, c, 14, 14) for i, c in enumerate(in_channels)]

    neck = MultiLevelNeck(in_channels, 256)
    outputs = neck(inputs)
    assert outputs[0].shape == torch.Size([1, 256, 7, 7])
    assert outputs[1].shape == torch.Size([1, 256, 14, 14])
    assert outputs[2].shape == torch.Size([1, 256, 28, 28])
    assert outputs[3].shape == torch.Size([1, 256, 56, 56])

    # Test one feature map
    in_channels = [768]
    inputs = [torch.randn(1, 768, 14, 14)]

    neck = MultiLevelNeck(in_channels, 256)
    outputs = neck(inputs)
    assert outputs[0].shape == torch.Size([1, 256, 7, 7])
    assert outputs[1].shape == torch.Size([1, 256, 14, 14])
    assert outputs[2].shape == torch.Size([1, 256, 28, 28])
    assert outputs[3].shape == torch.Size([1, 256, 56, 56])
