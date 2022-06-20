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
#
#
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Xingyi Zhou
"""
Implement many useful :class:`Augmentation`.
"""
import numpy as np
import sys
from fvcore.transforms.transform import (
    BlendTransform,
    CropTransform,
    HFlipTransform,
    NoOpTransform,
    Transform,
    VFlipTransform,
)
from PIL import Image

from detectron2.data.transforms.augmentation import Augmentation
from .custom_transform import EfficientDetResizeCropTransform

__all__ = [
    "EfficientDetResizeCrop",
]


class EfficientDetResizeCrop(Augmentation):
    """
    Scale the shorter edge to the given size, with a limit of `max_size` on the longer edge.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """

    def __init__(
        self, size, scale, interp=Image.BILINEAR
    ):
        """
        Args:
        """
        super().__init__()
        self.target_size = (size, size)
        self.scale = scale
        self.interp = interp

    def get_transform(self, img):
        # Select a random scale factor.
        scale_factor = np.random.uniform(*self.scale)
        scaled_target_height = scale_factor * self.target_size[0]
        scaled_target_width = scale_factor * self.target_size[1]
        # Recompute the accurate scale_factor using rounded scaled image size.
        width, height = img.shape[1], img.shape[0]
        img_scale_y = scaled_target_height / height
        img_scale_x = scaled_target_width / width
        img_scale = min(img_scale_y, img_scale_x)

        # Select non-zero random offset (x, y) if scaled image is larger than target size
        scaled_h = int(height * img_scale)
        scaled_w = int(width * img_scale)
        offset_y = scaled_h - self.target_size[0]
        offset_x = scaled_w - self.target_size[1]
        offset_y = int(max(0.0, float(offset_y)) * np.random.uniform(0, 1))
        offset_x = int(max(0.0, float(offset_x)) * np.random.uniform(0, 1))
        return EfficientDetResizeCropTransform(
            scaled_h, scaled_w, offset_y, offset_x, img_scale, self.target_size, self.interp)
