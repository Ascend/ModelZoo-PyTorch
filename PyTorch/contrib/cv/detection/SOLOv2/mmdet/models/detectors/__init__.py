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

# from .atss import ATSS
# from .base import BaseDetector
# from .cascade_rcnn import CascadeRCNN
# from .double_head_rcnn import DoubleHeadRCNN
# from .fast_rcnn import FastRCNN
# from .faster_rcnn import FasterRCNN
# from .fcos import FCOS
# from .fovea import FOVEA
# from .grid_rcnn import GridRCNN
# from .htc import HybridTaskCascade
# from .mask_rcnn import MaskRCNN
# from .mask_scoring_rcnn import MaskScoringRCNN
# from .reppoints_detector import RepPointsDetector
# from .retinanet import RetinaNet
# from .rpn import RPN
# from .single_stage import SingleStageDetector
# from .single_stage_ins import SingleStageInsDetector
# from .two_stage import TwoStageDetector
from .solo import SOLO
from .solov2 import SOLOv2

__all__ = [
    # 'ATSS', 'BaseDetector', 'SingleStageDetector', 'TwoStageDetector', 'RPN',
    # 'FastRCNN', 'FasterRCNN', 'MaskRCNN', 'CascadeRCNN', 'HybridTaskCascade',
    # 'DoubleHeadRCNN', 'RetinaNet', 'FCOS', 'GridRCNN', 'MaskScoringRCNN',
    # 'RepPointsDetector', 'FOVEA', 'SingleStageInsDetector',
    'SOLO', 'SOLOv2'
]
