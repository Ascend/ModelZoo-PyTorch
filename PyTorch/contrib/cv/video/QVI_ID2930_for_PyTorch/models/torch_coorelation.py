# Copyright 2017 Phil Ferriere. All Rights Reserved.
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
# ============================================================================
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

"""
core_costvol.py

Computes cross correlation between two feature maps.

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)

Based on:
    - https://github.com/tensorpack/tensorpack/blob/master/examples/OpticalFlow/flownet_models.py
        Written by Patrick Wieschollek, Copyright Yuxin Wu
        Apache License 2.0
"""

from __future__ import absolute_import, division, print_function
import torch


def cost_volume(c1, warp, search_range=4):
    """Build cost volume for associating a pixel from Image1 with its corresponding pixels in Image2.
    Args:
        c1: Level of the feature pyramid of Image1
        warp: Warped level of the feature pyramid of image22
        search_range: Search range (maximum displacement)
    """
    padded_lvl = torch.nn.functional.pad(warp, [search_range, search_range, search_range, search_range])
    _, _, h, w = c1.shape
    max_offset = search_range * 2 + 1

    cost_vol = []
    for y in range(0, max_offset):
        for x in range(0, max_offset):
            slice = padded_lvl[:, :, y:y+h, x:x+w]
            cost = torch.mean(c1 * slice, dim=1, keepdim=True)
            cost_vol.append(cost)
    cost_vol = torch.cat(cost_vol, dim=1)
    cost_vol = torch.nn.functional.leaky_relu(cost_vol, 0.1)

    return cost_vol
