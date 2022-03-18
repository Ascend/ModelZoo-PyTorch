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

from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS
from .fcn_mask_head import FCNMaskHead


@HEADS.register_module()
class HTCMaskHead(FCNMaskHead):

    def __init__(self, with_conv_res=True, *args, **kwargs):
        super(HTCMaskHead, self).__init__(*args, **kwargs)
        self.with_conv_res = with_conv_res
        if self.with_conv_res:
            self.conv_res = ConvModule(
                self.conv_out_channels,
                self.conv_out_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)

    def init_weights(self):
        super(HTCMaskHead, self).init_weights()
        if self.with_conv_res:
            self.conv_res.init_weights()

    def forward(self, x, res_feat=None, return_logits=True, return_feat=True):
        if res_feat is not None:
            assert self.with_conv_res
            res_feat = self.conv_res(res_feat)
            x = x + res_feat
        for conv in self.convs:
            x = conv(x)
        res_feat = x
        outs = []
        if return_logits:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
            mask_pred = self.conv_logits(x)
            outs.append(mask_pred)
        if return_feat:
            outs.append(res_feat)
        return outs if len(outs) > 1 else outs[0]
