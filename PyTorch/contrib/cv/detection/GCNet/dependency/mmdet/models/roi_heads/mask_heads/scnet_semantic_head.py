# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
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
#

from mmdet.models.builder import HEADS
from mmdet.models.utils import ResLayer, SimplifiedBasicBlock
from .fused_semantic_head import FusedSemanticHead


@HEADS.register_module()
class SCNetSemanticHead(FusedSemanticHead):
    """Mask head for `SCNet <https://arxiv.org/abs/2012.10150>`_.

    Args:
        conv_to_res (bool, optional): if True, change the conv layers to
            ``SimplifiedBasicBlock``.
    """

    def __init__(self, conv_to_res=True, **kwargs):
        super(SCNetSemanticHead, self).__init__(**kwargs)
        self.conv_to_res = conv_to_res
        if self.conv_to_res:
            num_res_blocks = self.num_convs // 2
            self.convs = ResLayer(
                SimplifiedBasicBlock,
                self.in_channels,
                self.conv_out_channels,
                num_res_blocks,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
            self.num_convs = num_res_blocks
