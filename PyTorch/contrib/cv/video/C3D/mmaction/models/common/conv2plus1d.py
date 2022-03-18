
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch.nn as nn
from mmcv.cnn import CONV_LAYERS, build_norm_layer, constant_init, kaiming_init
from torch.nn.modules.utils import _triple


@CONV_LAYERS.register_module()
class Conv2plus1d(nn.Module):
    """(2+1)d Conv module for R(2+1)d backbone.

    https://arxiv.org/pdf/1711.11248.pdf.

    Args:
        in_channels (int): Same as nn.Conv3d.
        out_channels (int): Same as nn.Conv3d.
        kernel_size (int | tuple[int]): Same as nn.Conv3d.
        stride (int | tuple[int]): Same as nn.Conv3d.
        padding (int | tuple[int]): Same as nn.Conv3d.
        dilation (int | tuple[int]): Same as nn.Conv3d.
        groups (int): Same as nn.Conv3d.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 norm_cfg=dict(type='BN3d')):
        super().__init__()

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        assert len(kernel_size) == len(stride) == len(padding) == 3

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.norm_cfg = norm_cfg
        self.output_padding = (0, 0, 0)
        self.transposed = False
        self.flag = 0

        # The middle-plane is calculated according to:
        # M_i = \floor{\frac{t * d^2 N_i-1 * N_i}
        #   {d^2 * N_i-1 + t * N_i}}
        # where d, t are spatial and temporal kernel, and
        # N_i, N_i-1 are planes
        # and inplanes. https://arxiv.org/pdf/1711.11248.pdf
        mid_channels = 3 * (
            in_channels * out_channels * kernel_size[1] * kernel_size[2])
        mid_channels /= (
            in_channels * kernel_size[1] * kernel_size[2] + 3 * out_channels)
        mid_channels = int(mid_channels)

        self.conv_s = nn.Conv3d(
            in_channels,
            mid_channels,
            kernel_size=(1, kernel_size[1], kernel_size[2]),
            stride=(1, stride[1], stride[2]),
            padding=(0, padding[1], padding[2]),
            bias=bias)
        _, self.bn_s = build_norm_layer(self.norm_cfg, mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv_t = nn.Conv3d(
            mid_channels,
            out_channels,
            kernel_size=(kernel_size[0], 1, 1),
            stride=(stride[0], 1, 1),
            padding=(padding[0], 0, 0),
            bias=bias)

        self.init_weights()

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        if x.shape[2]==1 and x.shape[3]==7:# and self.flag:
            #if self.flag%5<2:
                #x = x.npu()
            #    x = x.cpu().float()
            #    self.conv_s = self.conv_s.cpu().float()
            #    x = self.conv_s(x)
            #    x = x.npu()
            #    self.conv_s = self.conv_s.npu()

            #    x = self.bn_s(x)
            #    x = self.relu(x)
            #    x = self.conv_t(x)
            self.flag = self.flag + 1
            #x = x.cpu().float()
            #self.conv_s = self.conv_s.cpu().float()
            #x = self.conv_s(x)
            #x = x.npu()
            #self.conv_s = self.conv_s.npu()
            #self.flag = False
        else :
            x = x.npu()
            x = self.conv_s(x)

            x = self.bn_s(x)
            x = self.relu(x)
            x = self.conv_t(x)
        return x

    def init_weights(self):
        """Initiate the parameters from scratch."""
        kaiming_init(self.conv_s)
        kaiming_init(self.conv_t)
        constant_init(self.bn_s, 1, bias=0)
