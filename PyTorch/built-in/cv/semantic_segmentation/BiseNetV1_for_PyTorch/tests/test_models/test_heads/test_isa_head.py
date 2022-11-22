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

from mmseg.models.decode_heads import ISAHead
from .utils import to_cuda


def test_isa_head():

    inputs = [torch.randn(1, 8, 23, 23)]
    isa_head = ISAHead(
        in_channels=8,
        channels=4,
        num_classes=19,
        isa_channels=4,
        down_factor=(8, 8))
    if torch.cuda.is_available():
        isa_head, inputs = to_cuda(isa_head, inputs)
    output = isa_head(inputs)
    assert output.shape == (1, isa_head.num_classes, 23, 23)
