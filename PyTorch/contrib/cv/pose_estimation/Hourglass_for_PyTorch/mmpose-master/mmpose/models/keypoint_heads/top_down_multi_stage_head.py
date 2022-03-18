# -*- coding: utf-8 -*-
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
# ============================================================================

import copy as cp

import torch.nn as nn
from mmcv.cnn import (ConvModule, DepthwiseSeparableConvModule, Linear,
                      build_activation_layer, build_conv_layer,
                      build_norm_layer, build_upsample_layer, constant_init,
                      kaiming_init, normal_init)

from ..registry import HEADS


@HEADS.register_module()
class TopDownMultiStageHead(nn.Module):
    """Heads for multi-stage pose models.

    TopDownMultiStageHead is consisted of multiple branches, each of
    which has num_deconv_layers(>=0) number of deconv layers
    and a simple conv2d layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_stages (int): Number of stages.
        num_deconv_layers (int): Number of deconv layers.
            num_deconv_layers should >= 0. Note that 0 means
            no deconv layers.
        num_deconv_filters (list|tuple): Number of filters.
            If num_deconv_layers > 0, the length of
        num_deconv_kernels (list|tuple): Kernel sizes.
    """

    def __init__(self,
                 in_channels=512,
                 out_channels=17,
                 num_stages=1,
                 num_deconv_layers=3,
                 num_deconv_filters=(256, 256, 256),
                 num_deconv_kernels=(4, 4, 4),
                 extra=None):
        super().__init__()

        self.in_channels = in_channels
        self.num_stages = num_stages

        if extra is not None and not isinstance(extra, dict):
            raise TypeError('extra should be dict or None.')

        # build multi-stage deconv layers
        self.multi_deconv_layers = nn.ModuleList([])
        for _ in range(self.num_stages):
            if num_deconv_layers > 0:
                deconv_layers = self._make_deconv_layer(
                    num_deconv_layers,
                    num_deconv_filters,
                    num_deconv_kernels,
                )
            elif num_deconv_layers == 0:
                deconv_layers = nn.Identity()
            else:
                raise ValueError(
                    f'num_deconv_layers ({num_deconv_layers}) should >= 0.')
            self.multi_deconv_layers.append(deconv_layers)

        identity_final_layer = False
        if extra is not None and 'final_conv_kernel' in extra:
            assert extra['final_conv_kernel'] in [0, 1, 3]
            if extra['final_conv_kernel'] == 3:
                padding = 1
            elif extra['final_conv_kernel'] == 1:
                padding = 0
            else:
                # 0 for Identity mapping.
                identity_final_layer = True
            kernel_size = extra['final_conv_kernel']
        else:
            kernel_size = 1
            padding = 0

        # build multi-stage final layers
        self.multi_final_layers = nn.ModuleList([])
        for i in range(self.num_stages):
            if identity_final_layer:
                final_layer = nn.Identity()
            else:
                final_layer = build_conv_layer(
                    cfg=dict(type='Conv2d'),
                    in_channels=num_deconv_filters[-1]
                    if num_deconv_layers > 0 else in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding)
            self.multi_final_layers.append(final_layer)

    def forward(self, x):
        """Forward function.

        Returns:
            out (list[Tensor]): a list of heatmaps from multiple stages.
        """
        out = []

        assert isinstance(x, list)
        for i in range(self.num_stages):
            y = self.multi_deconv_layers[i](x[i])
            y = self.multi_final_layers[i](y)
            out.append(y)

        return out

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        if num_layers != len(num_filters):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_filters({len(num_filters)})'
            raise ValueError(error_msg)
        if num_layers != len(num_kernels):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_kernels({len(num_kernels)})'
            raise ValueError(error_msg)

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=self.in_channels,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes

        return nn.Sequential(*layers)

    @staticmethod
    def _get_deconv_cfg(deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding

    def init_weights(self):
        """Initialize model weights."""
        for _, m in self.multi_deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for m in self.multi_final_layers.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)


class PredictHeatmap(nn.Module):
    """Predict the heat map for an input feature.

    Args:
        unit_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        out_shape (tuple): Shape of the output heatmap.
        use_prm (bool): Whether to use pose refine machine. Default: False.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self,
                 unit_channels,
                 out_channels,
                 out_shape,
                 use_prm=False,
                 norm_cfg=dict(type='BN')):
        # Protect mutable default arguments
        norm_cfg = cp.deepcopy(norm_cfg)
        super().__init__()
        self.unit_channels = unit_channels
        self.out_channels = out_channels
        self.out_shape = out_shape
        self.use_prm = use_prm
        if use_prm:
            self.prm = PRM(out_channels, norm_cfg=norm_cfg)
        self.conv_layers = nn.Sequential(
            ConvModule(
                unit_channels,
                unit_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_cfg=norm_cfg,
                inplace=False),
            ConvModule(
                unit_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=None,
                inplace=False))

    def forward(self, feature):
        feature = self.conv_layers(feature)
        output = nn.functional.interpolate(
            feature, size=self.out_shape, mode='bilinear', align_corners=True)
        if self.use_prm:
            output = self.prm(output)
        return output


class PRM(nn.Module):
    """Pose Refine Machine.

    For more details about PRM, refer to Learning Delicate
    Local Representations for Multi-Person Pose Estimation (ECCV 2020).
    Args:
        out_channels (int): Channel number of the output. Equals to
            the number of key points.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self, out_channels, norm_cfg=dict(type='BN')):
        # Protect mutable default arguments
        norm_cfg = cp.deepcopy(norm_cfg)
        super().__init__()
        self.out_channels = out_channels
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.middle_path = nn.Sequential(
            Linear(self.out_channels, self.out_channels),
            build_norm_layer(dict(type='BN1d'), out_channels)[1],
            build_activation_layer(dict(type='ReLU')),
            Linear(self.out_channels, self.out_channels),
            build_norm_layer(dict(type='BN1d'), out_channels)[1],
            build_activation_layer(dict(type='ReLU')),
            build_activation_layer(dict(type='Sigmoid')))

        self.bottom_path = nn.Sequential(
            ConvModule(
                self.out_channels,
                self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_cfg=norm_cfg,
                inplace=False),
            DepthwiseSeparableConvModule(
                self.out_channels,
                1,
                kernel_size=9,
                stride=1,
                padding=4,
                norm_cfg=norm_cfg,
                inplace=False), build_activation_layer(dict(type='Sigmoid')))
        self.conv_bn_relu_prm_1 = ConvModule(
            self.out_channels,
            self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_cfg=norm_cfg,
            inplace=False)

    def forward(self, x):
        out = self.conv_bn_relu_prm_1(x)
        out_1 = out

        out_2 = self.global_pooling(out_1)
        out_2 = out_2.view(out_2.size(0), -1)
        out_2 = self.middle_path(out_2)
        out_2 = out_2.unsqueeze(2)
        out_2 = out_2.unsqueeze(3)

        out_3 = self.bottom_path(out_1)
        out = out_1 * (1 + out_2 * out_3)

        return out


@HEADS.register_module()
class TopDownMSMUHead(nn.Module):
    """Heads for multi-stage multi-unit heads used in Multi-Stage Pose
    estimation Network (MSPN), and Residual Steps Networks (RSN).

    Args:
        unit_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        out_shape (tuple): Shape of the output heatmap.
        num_stages (int): Number of stages.
        num_units (int): Number of units in each stage.
        use_prm (bool): Whether to use pose refine machine (PRM).
            Default: False.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self,
                 out_shape,
                 unit_channels=256,
                 out_channels=17,
                 num_stages=4,
                 num_units=4,
                 use_prm=False,
                 norm_cfg=dict(type='BN')):
        # Protect mutable default arguments
        norm_cfg = cp.deepcopy(norm_cfg)
        super().__init__()

        self.out_shape = out_shape
        self.unit_channels = unit_channels
        self.out_channels = out_channels
        self.num_stages = num_stages
        self.num_units = num_units
        self.predict_layers = nn.ModuleList([])
        for i in range(self.num_stages):
            for j in range(self.num_units):
                self.predict_layers.append(
                    PredictHeatmap(
                        unit_channels,
                        out_channels,
                        out_shape,
                        use_prm,
                        norm_cfg=norm_cfg))

    def forward(self, x):
        """Forward function.

        Returns:
            out (list[Tensor]): a list of heatmaps from multiple stages
                                and units.
        """
        out = []
        assert isinstance(x, list)
        assert len(x) == self.num_stages
        assert isinstance(x[0], list)
        assert len(x[0]) == self.num_units
        assert x[0][0].shape[1] == self.unit_channels
        for i in range(self.num_stages):
            for j in range(self.num_units):
                y = self.predict_layers[i * self.num_units + j](x[i][j])
                out.append(y)

        return out

    def init_weights(self):
        """Initialize model weights."""

        for m in self.predict_layers.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.01)
