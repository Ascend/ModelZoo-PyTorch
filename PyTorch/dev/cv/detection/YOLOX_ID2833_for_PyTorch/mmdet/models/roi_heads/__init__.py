
# Copyright 2022 Huawei Technologies Co., Ltd
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

# Copyright (c) Open-MMLab. All rights reserved.    
# Copyright (c) OpenMMLab. All rights reserved.
from .base_roi_head import BaseRoIHead
from .bbox_heads import (BBoxHead, ConvFCBBoxHead, DIIHead,
                         DoubleConvFCBBoxHead, SABLHead, SCNetBBoxHead,
                         Shared2FCBBoxHead, Shared4Conv1FCBBoxHead)
from .cascade_roi_head import CascadeRoIHead
from .double_roi_head import DoubleHeadRoIHead
from .dynamic_roi_head import DynamicRoIHead
from .grid_roi_head import GridRoIHead
from .htc_roi_head import HybridTaskCascadeRoIHead
from .mask_heads import (CoarseMaskHead, FCNMaskHead, FeatureRelayHead,
                         FusedSemanticHead, GlobalContextHead, GridHead,
                         HTCMaskHead, MaskIoUHead, MaskPointHead,
                         SCNetMaskHead, SCNetSemanticHead)
from .mask_scoring_roi_head import MaskScoringRoIHead
from .pisa_roi_head import PISARoIHead
from .point_rend_roi_head import PointRendRoIHead
from .roi_extractors import (BaseRoIExtractor, GenericRoIExtractor,
                             SingleRoIExtractor)
from .scnet_roi_head import SCNetRoIHead
from .shared_heads import ResLayer
from .sparse_roi_head import SparseRoIHead
from .standard_roi_head import StandardRoIHead
from .trident_roi_head import TridentRoIHead

__all__ = [
    'BaseRoIHead', 'CascadeRoIHead', 'DoubleHeadRoIHead', 'MaskScoringRoIHead',
    'HybridTaskCascadeRoIHead', 'GridRoIHead', 'ResLayer', 'BBoxHead',
    'ConvFCBBoxHead', 'DIIHead', 'SABLHead', 'Shared2FCBBoxHead',
    'StandardRoIHead', 'Shared4Conv1FCBBoxHead', 'DoubleConvFCBBoxHead',
    'FCNMaskHead', 'HTCMaskHead', 'FusedSemanticHead', 'GridHead',
    'MaskIoUHead', 'BaseRoIExtractor', 'GenericRoIExtractor',
    'SingleRoIExtractor', 'PISARoIHead', 'PointRendRoIHead', 'MaskPointHead',
    'CoarseMaskHead', 'DynamicRoIHead', 'SparseRoIHead', 'TridentRoIHead',
    'SCNetRoIHead', 'SCNetMaskHead', 'SCNetSemanticHead', 'SCNetBBoxHead',
    'FeatureRelayHead', 'GlobalContextHead'
]
