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
import torch

from mmseg.models.decode_heads import EMAHead
from .utils import to_cuda


def test_emanet_head():
    head = EMAHead(
        in_channels=4,
        ema_channels=3,
        channels=2,
        num_stages=3,
        num_bases=2,
        num_classes=19)
    for param in head.ema_mid_conv.parameters():
        assert not param.requires_grad
    assert hasattr(head, 'ema_module')
    inputs = [torch.randn(1, 4, 23, 23)]
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 23, 23)
