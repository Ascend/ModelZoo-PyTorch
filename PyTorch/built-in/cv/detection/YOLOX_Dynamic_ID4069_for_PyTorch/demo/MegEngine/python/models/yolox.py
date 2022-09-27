
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import megengine.module as M

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN


class YOLOX(M.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)
        assert not self.training
        outputs = self.head(fpn_outs)

        return outputs
