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
import torch.nn as nn
from mmcv.cnn.bricks import ConvModule
from mmcv.runner import BaseModule

from ..backbones.resnet import Bottleneck, ResLayer
from ..builder import NECKS


@NECKS.register_module()
class HRFuseScales(BaseModule):
    """Fuse feature map of multiple scales in HRNet.

    Args:
        in_channels (list[int]): The input channels of all scales.
        out_channels (int): The channels of fused feature map.
            Defaults to 2048.
        norm_cfg (dict): dictionary to construct norm layers.
            Defaults to ``dict(type='BN', momentum=0.1)``.
        init_cfg (dict | list[dict], optional): Initialization config dict.
            Defaults to ``dict(type='Normal', layer='Linear', std=0.01))``.
    """

    def __init__(self,
                 in_channels,
                 out_channels=2048,
                 norm_cfg=dict(type='BN', momentum=0.1),
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01)):
        super(HRFuseScales, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_cfg = norm_cfg

        block_type = Bottleneck
        out_channels = [128, 256, 512, 1024]

        # Increase the channels on each resolution
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        increase_layers = []
        for i in range(len(in_channels)):
            increase_layers.append(
                ResLayer(
                    block_type,
                    in_channels=in_channels[i],
                    out_channels=out_channels[i],
                    num_blocks=1,
                    stride=1,
                ))
        self.increase_layers = nn.ModuleList(increase_layers)

        # Downsample feature maps in each scale.
        downsample_layers = []
        for i in range(len(in_channels) - 1):
            downsample_layers.append(
                ConvModule(
                    in_channels=out_channels[i],
                    out_channels=out_channels[i + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    bias=False,
                ))
        self.downsample_layers = nn.ModuleList(downsample_layers)

        # The final conv block before final classifier linear layer.
        self.final_layer = ConvModule(
            in_channels=out_channels[3],
            out_channels=self.out_channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            bias=False,
        )

    def forward(self, x):
        assert isinstance(x, tuple) and len(x) == len(self.in_channels)

        feat = self.increase_layers[0](x[0])
        for i in range(len(self.downsample_layers)):
            feat = self.downsample_layers[i](feat) + \
                self.increase_layers[i + 1](x[i + 1])

        return (self.final_layer(feat), )
