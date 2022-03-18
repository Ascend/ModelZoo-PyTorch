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

from mmseg.models.backbones import FastSCNN


def test_fastscnn_backbone():
    with pytest.raises(AssertionError):
        # Fast-SCNN channel constraints.
        FastSCNN(
            3, (32, 48),
            64, (64, 96, 128), (2, 2, 1),
            global_out_channels=127,
            higher_in_channels=64,
            lower_in_channels=128)

    # Test FastSCNN Standard Forward
    model = FastSCNN()
    model.init_weights()
    model.train()
    batch_size = 4
    imgs = torch.randn(batch_size, 3, 512, 1024)
    feat = model(imgs)

    assert len(feat) == 3
    # higher-res
    assert feat[0].shape == torch.Size([batch_size, 64, 64, 128])
    # lower-res
    assert feat[1].shape == torch.Size([batch_size, 128, 16, 32])
    # FFM output
    assert feat[2].shape == torch.Size([batch_size, 128, 64, 128])
