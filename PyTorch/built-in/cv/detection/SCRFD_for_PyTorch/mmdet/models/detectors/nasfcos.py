# Copyright (c) 2020 Huawei Technologies Co., Ltd
# All rights reserved.
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

from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class NASFCOS(SingleStageDetector):
    """NAS-FCOS: Fast Neural Architecture Search for Object Detection.

    https://arxiv.org/abs/1906.0442
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(NASFCOS, self).__init__(backbone, neck, bbox_head, train_cfg,
                                      test_cfg, pretrained)
