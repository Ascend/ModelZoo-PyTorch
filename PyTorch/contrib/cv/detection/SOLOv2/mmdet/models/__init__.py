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

from .anchor_heads import *  # noqa: F401,F403
from .backbones import *  # noqa: F401,F403
# from .bbox_heads import *  # noqa: F401,F403
from .builder import (build_backbone, build_detector, build_head, build_loss,
                      build_neck, build_roi_extractor, build_shared_head)
from .detectors import *  # noqa: F401,F403
from .losses import *  # noqa: F401,F403
from .mask_heads import *  # noqa: F401,F403
from .necks import *  # noqa: F401,F403
from .registry import (BACKBONES, DETECTORS, HEADS, LOSSES, NECKS,
                       ROI_EXTRACTORS, SHARED_HEADS)

# from .roi_extractors import *  # noqa: F401,F403
# from .shared_heads import *  # noqa: F401,F403

__all__ = [
    'BACKBONES', 'NECKS', 'ROI_EXTRACTORS', 'SHARED_HEADS', 'HEADS', 'LOSSES',
    'DETECTORS', 'build_backbone', 'build_neck', 'build_roi_extractor',
    'build_shared_head', 'build_head', 'build_loss', 'build_detector'
]
