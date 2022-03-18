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

from mmseg.models import MLANeck


def test_mla():
    in_channels = [1024, 1024, 1024, 1024]
    mla = MLANeck(in_channels, 256)

    inputs = [torch.randn(1, c, 24, 24) for i, c in enumerate(in_channels)]
    outputs = mla(inputs)
    assert outputs[0].shape == torch.Size([1, 256, 24, 24])
    assert outputs[1].shape == torch.Size([1, 256, 24, 24])
    assert outputs[2].shape == torch.Size([1, 256, 24, 24])
    assert outputs[3].shape == torch.Size([1, 256, 24, 24])
