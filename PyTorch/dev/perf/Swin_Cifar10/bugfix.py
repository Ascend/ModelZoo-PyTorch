# -*- coding: utf-8 -*-
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

import torch
from torch import nn

class PatchMergingFixed(models.swin.PatchMerging):
    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor

        # TODO Im2col OP ERROR currently.. Calculate it on CPU temporarily.
        raw_device = x.device
        x = self.patch_merge(x.cpu()).view(b, -1, new_h, new_w).permute(0, 2, 3, 1).to(raw_device)

        x = self.linear(x)

        return x

models.swin.PatchMerging = models.swin.PatchMergingFixed
