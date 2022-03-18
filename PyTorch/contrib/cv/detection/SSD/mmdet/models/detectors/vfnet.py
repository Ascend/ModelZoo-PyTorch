# Copyright 2021 Huawei Technologies Co., Ltd
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


from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class VFNet(SingleStageDetector):
    """Implementation of `VarifocalNet
    (VFNet).<https://arxiv.org/abs/2008.13367>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(VFNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                    test_cfg, pretrained)
