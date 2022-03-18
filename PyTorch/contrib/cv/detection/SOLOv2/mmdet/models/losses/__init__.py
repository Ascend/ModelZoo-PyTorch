# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .accuracy import Accuracy, accuracy
from .balanced_l1_loss import BalancedL1Loss, balanced_l1_loss
from .cross_entropy_loss import (CrossEntropyLoss, binary_cross_entropy,
                                 cross_entropy, mask_cross_entropy)
from .focal_loss import FocalLoss, sigmoid_focal_loss
from .ghm_loss import GHMC, GHMR
from .iou_loss import (BoundedIoULoss, GIoULoss, IoULoss, bounded_iou_loss,
                       iou_loss)
from .mse_loss import MSELoss, mse_loss
from .smooth_l1_loss import SmoothL1Loss, smooth_l1_loss
from .utils import reduce_loss, weight_reduce_loss, weighted_loss

__all__ = [
    'accuracy', 'Accuracy', 'cross_entropy', 'binary_cross_entropy',
    'mask_cross_entropy', 'CrossEntropyLoss', 'sigmoid_focal_loss',
    'FocalLoss', 'smooth_l1_loss', 'SmoothL1Loss', 'balanced_l1_loss',
    'BalancedL1Loss', 'mse_loss', 'MSELoss', 'iou_loss', 'bounded_iou_loss',
    'IoULoss', 'BoundedIoULoss', 'GIoULoss', 'GHMC', 'GHMR', 'reduce_loss',
    'weight_reduce_loss', 'weighted_loss'
]
