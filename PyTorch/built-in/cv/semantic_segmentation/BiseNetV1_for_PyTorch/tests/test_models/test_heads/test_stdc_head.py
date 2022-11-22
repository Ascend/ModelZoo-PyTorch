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

from mmseg.models.decode_heads import STDCHead
from .utils import to_cuda


def test_stdc_head():
    inputs = [torch.randn(1, 32, 21, 21)]
    head = STDCHead(
        in_channels=32,
        channels=8,
        num_convs=1,
        num_classes=2,
        in_index=-1,
        loss_decode=[
            dict(
                type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0)
        ])
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert isinstance(outputs, torch.Tensor) and len(outputs) == 1
    assert outputs.shape == torch.Size([1, head.num_classes, 21, 21])

    fake_label = torch.ones_like(
        outputs[:, 0:1, :, :], dtype=torch.int16).long()
    loss = head.losses(seg_logit=outputs, seg_label=fake_label)
    assert loss['loss_ce'] != torch.zeros_like(loss['loss_ce'])
    assert loss['loss_dice'] != torch.zeros_like(loss['loss_dice'])
