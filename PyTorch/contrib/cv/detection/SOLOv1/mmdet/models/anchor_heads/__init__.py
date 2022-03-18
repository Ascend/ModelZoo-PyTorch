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

# from .anchor_head import AnchorHead
# from .atss_head import ATSSHead
# from .fcos_head import FCOSHead
# from .fovea_head import FoveaHead
# from .free_anchor_retina_head import FreeAnchorRetinaHead
# from .ga_retina_head import GARetinaHead
# from .ga_rpn_head import GARPNHead
# from .guided_anchor_head import FeatureAdaption, GuidedAnchorHead
# from .reppoints_head import RepPointsHead
# from .retina_head import RetinaHead
# from .retina_sepbn_head import RetinaSepBNHead
# from .rpn_head import RPNHead
# from .ssd_head import SSDHead
from .solo_head import SOLOHead
from .solov2_head import SOLOv2Head
# from .solov2_light_head import SOLOv2LightHead
# from .decoupled_solo_head import DecoupledSOLOHead
# from .decoupled_solo_light_head import DecoupledSOLOLightHead

__all__ = [
    # 'AnchorHead', 'GuidedAnchorHead', 'FeatureAdaption', 'RPNHead',
    # 'GARPNHead', 'RetinaHead', 'RetinaSepBNHead', 'GARetinaHead', 'SSDHead',
    # 'FCOSHead', 'RepPointsHead', 'FoveaHead', 'FreeAnchorRetinaHead',
    # 'ATSSHead',
    'SOLOHead', 'SOLOv2Head',
    # 'SOLOv2LightHead', 'DecoupledSOLOHead', 'DecoupledSOLOLightHead'
]
