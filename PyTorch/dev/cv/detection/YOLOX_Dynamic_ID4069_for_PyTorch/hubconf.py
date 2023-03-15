
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
# -*- coding:utf-8 -*-

"""
Usage example:
    import torch
    model = torch.hub.load("Megvii-BaseDetection/YOLOX", "yolox_s")
"""
dependencies = ["torch"]

from yolox.models import (  # isort:skip  # noqa: F401, E402
    yolox_tiny,
    yolox_nano,
    yolox_s,
    yolox_m,
    yolox_l,
    yolox_x,
    yolov3,
)
