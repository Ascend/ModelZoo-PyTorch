# Copyright 2023 Huawei Technologies Co., Ltd
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
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from mmcv.ops import roi_align
from mmcv.ops import RoIAlign as ROIAlign
from mmcv.ops import ModulatedDeformConv2d as ModulatedDeformConv

from .batch_norm import FrozenBatchNorm2d, NaiveSyncBatchNorm2d
from .misc import Conv2d, _NewEmptyTensorOp
from .misc import ConvTranspose2d
from .misc import DFConv2d
from .misc import interpolate
from .misc import Scale
from .nms import nms
from .nms import ml_nms
from .nms import soft_nms
from .smooth_l1_loss import smooth_l1_loss
from .sigmoid_focal_loss import SigmoidFocalLoss, TokenSigmoidFocalLoss
from .iou_loss import IOULoss, IOUWHLoss
from .dropblock import DropBlock2D, DropBlock3D
from .evonorm import EvoNorm2d
from .dyrelu import DYReLU, swish
from .se import SELayer, SEBlock
from .dyhead import DyHead
from .set_loss import HungarianMatcher, SetCriterion

__all__ = ["nms", "ml_nms", "soft_nms", "roi_align", "ROIAlign",
           "smooth_l1_loss", "Conv2d", "ConvTranspose2d", "interpolate", "swish",
           "FrozenBatchNorm2d", "NaiveSyncBatchNorm2d", "SigmoidFocalLoss", "TokenSigmoidFocalLoss", "IOULoss",
           "IOUWHLoss", "Scale", "ModulatedDeformConv", "DyHead",
           "DropBlock2D", "DropBlock3D", "EvoNorm2d", "DYReLU", "SELayer", "SEBlock",
           "HungarianMatcher", "SetCriterion", "_NewEmptyTensorOp"]
