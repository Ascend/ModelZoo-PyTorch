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

from .single_stage_ins import SingleStageInsDetector
from ..registry import DETECTORS


@DETECTORS.register_module
class SOLOv2(SingleStageInsDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 mask_feat_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SOLOv2, self).__init__(backbone, neck, bbox_head, mask_feat_head, train_cfg,
                                   test_cfg, pretrained)
