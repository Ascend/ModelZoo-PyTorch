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
import logging
import numpy as np
import pycocotools.mask as mask_util
import torch
from fvcore.common.file_io import PathManager
from PIL import Image

from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    RotatedBoxes,
    polygons_to_bitmask,
)

from detectron2.data import transforms as T
from .transforms.custom_augmentation_impl import EfficientDetResizeCrop

def build_custom_augmentation(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    Returns:
        list[Augmentation]
    """
    if cfg.INPUT.CUSTOM_AUG == 'ResizeShortestEdge':
        if is_train:
            min_size = cfg.INPUT.MIN_SIZE_TRAIN
            max_size = cfg.INPUT.MAX_SIZE_TRAIN
            sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        else:
            min_size = cfg.INPUT.MIN_SIZE_TEST
            max_size = cfg.INPUT.MAX_SIZE_TEST
            sample_style = "choice"
        augmentation = [T.ResizeShortestEdge(min_size, max_size, sample_style)]
    elif cfg.INPUT.CUSTOM_AUG == 'EfficientDetResizeCrop':
        if is_train:
            scale = cfg.INPUT.SCALE_RANGE
            size = cfg.INPUT.TRAIN_SIZE
        else:
            scale = (1, 1)
            size = cfg.INPUT.TEST_SIZE
        augmentation = [EfficientDetResizeCrop(size, scale)]
    else:
        assert 0, cfg.INPUT.CUSTOM_AUG

    if is_train:
        augmentation.append(T.RandomFlip())
    return augmentation


build_custom_transform_gen = build_custom_augmentation
"""
Alias for backward-compatibility.
"""