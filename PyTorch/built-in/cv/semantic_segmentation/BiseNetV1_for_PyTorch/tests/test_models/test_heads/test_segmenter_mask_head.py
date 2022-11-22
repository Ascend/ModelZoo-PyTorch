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

from mmseg.models.decode_heads import SegmenterMaskTransformerHead
from .utils import _conv_has_norm, to_cuda


def test_segmenter_mask_transformer_head():
    head = SegmenterMaskTransformerHead(
        in_channels=2,
        channels=2,
        num_classes=150,
        num_layers=2,
        num_heads=3,
        embed_dims=192,
        dropout_ratio=0.0)
    assert _conv_has_norm(head, sync_bn=True)
    head.init_weights()

    inputs = [torch.randn(1, 2, 32, 32)]
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 32, 32)
