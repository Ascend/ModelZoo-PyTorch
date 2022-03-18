# Copyright 2021 Huawei Technologies Co., Ltd
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

from __future__ import print_function

import numpy as np
import torch
from models.anchors import generate_anchors, shift


class StaticAnchors:
    def __init__(self,
                 pyramid_levels=None,
                 strides=None,
                 sizes=None,
                 ratios=None,
                 scales=None,
                 rotations=None):
        self.pyramid_levels = pyramid_levels
        self.strides =  strides
        self.sizes = sizes
        self.ratios = ratios
        self.scales = scales
        self.rotations = rotations
        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]
        if sizes is None:
            self.sizes = [2 ** (x + 1) for x in self.pyramid_levels]
        if ratios is None:
            self.ratios = np.array([1])
        if scales is None:
            self.scales = np.array([2 ** 0])
        if rotations is None:
            self.rotations = np.array([0])
        self.num_anchors = len(self.scales) * len(self.ratios) * len(self.rotations)

    def forward(self, ims_shape):
        image_shapes = [(ims_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]
        # compute anchors over all pyramid levels
        all_anchors = np.zeros((0, 5)).astype(np.float32)
        for idx, p in enumerate(self.pyramid_levels):
            anchors = generate_anchors(
                base_size=self.sizes[idx],
                ratios=self.ratios,
                scales=self.scales,
                rotations=self.rotations
            )
            shifted_anchors = shift(image_shapes[idx], self.strides[idx], anchors)
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)
        all_anchors = np.expand_dims(all_anchors, axis=0)
        all_anchors = torch.from_numpy(all_anchors.astype(np.float32))
        return all_anchors
