
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
from .coarse_mask_head import CoarseMaskHead
from .dynamic_mask_head import DynamicMaskHead
from .fcn_mask_head import FCNMaskHead
from .feature_relay_head import FeatureRelayHead
from .fused_semantic_head import FusedSemanticHead
from .global_context_head import GlobalContextHead
from .grid_head import GridHead
from .htc_mask_head import HTCMaskHead
from .mask_point_head import MaskPointHead
from .maskiou_head import MaskIoUHead
from .scnet_mask_head import SCNetMaskHead
from .scnet_semantic_head import SCNetSemanticHead

__all__ = [
    'FCNMaskHead', 'HTCMaskHead', 'FusedSemanticHead', 'GridHead',
    'MaskIoUHead', 'CoarseMaskHead', 'MaskPointHead', 'SCNetMaskHead',
    'SCNetSemanticHead', 'GlobalContextHead', 'FeatureRelayHead',
    'DynamicMaskHead'
]
