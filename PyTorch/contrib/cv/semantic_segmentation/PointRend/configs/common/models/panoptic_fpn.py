# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
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

from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling import PanopticFPN
from detectron2.modeling.meta_arch.semantic_seg import SemSegFPNHead

from .mask_rcnn_fpn import model

model._target_ = PanopticFPN
model.sem_seg_head = L(SemSegFPNHead)(
    input_shape={
        f: L(ShapeSpec)(stride=s, channels="${....backbone.out_channels}")
        for f, s in zip(["p2", "p3", "p4", "p5"], [4, 8, 16, 32])
    },
    ignore_value=255,
    num_classes=54,  # COCO stuff + 1
    conv_dims=128,
    common_stride=4,
    loss_weight=0.5,
    norm="GN",
)
