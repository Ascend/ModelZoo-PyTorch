# Copyright 2023 Huawei Technologies Co., Ltd
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
 
from typing import Union, OrderedDict
from dataclasses import dataclass
 
import numpy as np
import torch
from torch import nn
from torch.nn.common_types import _size_3_t
 
from det3d.models import BACKBONES
 
 
@dataclass
class MaskedConv3dConfig:
    in_channels: int
    out_channels: int
    kernel_size: _size_3_t
    stride: _size_3_t = 1
    padding: Union[str, _size_3_t] = 0
    dilation: _size_3_t = 1
    groups: int = 1
    bias: bool = True
    padding_mode: str = 'zeros'
    device: str = None
    dtype: str = None
    subm: bool = True
 
 
@dataclass
class NormLayerConfig:
    num_features: int 
    eps: float = 1e-05
    momentum: float = 0.1
    affine: bool = True
    track_running_stats: bool = True
    device: str = None
    dtype: str = None
 
 
class MaskedConv3d(nn.Module):
    def __init__(
        self, 
        config: MaskedConv3dConfig,
    ):
        super().__init__()
        self.subm = config.subm
        if self.subm:
            config.padding += 1
 
        self.conv3d = nn.Conv3d(
            config.in_channels, 
            config.out_channels, 
            config.kernel_size, 
            config.stride,
            config.padding,
            config.dilation,
            config.groups,
            config.bias,
            config.padding_mode,
            config.device,
            config.dtype,
        )
        self.out_channels = config.out_channels
 
    def forward(self, x):
        t, mask = x
        if not self.subm:
            mask = None
 
        elif mask is None:
            mask = t.sum(axis=1) == 0
            N, C, D, H, W = t.shape
            mask = mask.view(N, 1, D, H, W).repeat(1, self.out_channels, 1, 1, 1)
 
        ret = self.conv3d(t)
        if self.subm:
            ret[mask] = 0
        return ret, mask
 
 
class MaskedBatchNorm3d(nn.Module):
    def __init__(
        self, 
        config: NormLayerConfig,
    ):
        super().__init__()
        self.bn3d = nn.BatchNorm3d(
            config.num_features,
            config.eps, 
            config.momentum, 
            config.affine, 
            config.track_running_stats, 
            config.device, 
            config.dtype,
        )
 
    def forward(self, x):
        t, mask = x
        if mask is None:
            mask = t.sum(axis=1) == 0
            N, C, D, H, W = t.shape
            mask = mask.view(N, 1, D, H, W).repeat(1, C, 1, 1, 1)
 
        ret = self.bn3d(t)
        ret[mask] = 0
        return ret, mask
 
 
class ReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
 
    def forward(self, x):
        t, mask = x
        t = self.relu(t)
        return t, mask
 
 
def build_conv_block(
    conv_config: MaskedConv3dConfig,
    norm_config: NormLayerConfig,
    with_relu=True
) -> nn.Sequential:
    ret = nn.Sequential(
        MaskedConv3d(conv_config),
        MaskedBatchNorm3d(norm_config),
    )
    if with_relu:
        ret.append(ReLU())
 
    return ret
 
 
@BACKBONES.register_module
class MiddleFHD(nn.Module):
    def __init__(
        self, num_input_features=128, norm_cfg=None, name="MiddleFHD", **kwargs
    ):
        super(MiddleFHD, self).__init__()
        self.name = name
 
        self.dcn = None
        self.zero_init_residual = False
 
        self.relu = nn.ReLU()
 
        conv_configs = [
            MaskedConv3dConfig(num_input_features, 16, 3, bias=False),
            MaskedConv3dConfig(16, 16, 3, bias=False),
            MaskedConv3dConfig(16, 32, 3, 2, padding=1, bias=False, subm=False),
            MaskedConv3dConfig(32, 32, 3, bias=False),
            MaskedConv3dConfig(32, 32, 3, bias=False),
            MaskedConv3dConfig(32, 64, 3, 2, padding=1, bias=False, subm=False),
            MaskedConv3dConfig(64, 64, 3, bias=False),
            MaskedConv3dConfig(64, 64, 3, bias=False),
            MaskedConv3dConfig(64, 64, 3, bias=False),
            MaskedConv3dConfig(64, 64, 3, 2, padding=[0, 1, 1], bias=False, subm=False),
            MaskedConv3dConfig(64, 64, 3, bias=False),
            MaskedConv3dConfig(64, 64, 3, bias=False),
            MaskedConv3dConfig(64, 64, 3, bias=False),
            MaskedConv3dConfig(64, 64, (3, 1, 1), (2, 1, 1), bias=False, subm=False),
        ]
 
        norm_features = [16] * 2 + [32] * 3 + [64] * 9
        norm_configs = [NormLayerConfig(n) for n in norm_features]
 
        blocks = [
            build_conv_block(conv, norm) for conv, norm in zip(conv_configs, norm_configs)
        ]
 
        self.middle_conv = nn.Sequential(OrderedDict([
            (f"middle_conv_{i}", block) for i, block in enumerate(blocks) 
        ]))
 
    @staticmethod
    def sparse_to_dense(indices, values, out_shape):
        # scatter_nd from spconv
        ret = torch.zeros(*out_shape, dtype=values.dtype, device=values.device)
        indices = indices.long()
        ndim = indices.shape[-1]
        output_shape = list(indices.shape[:-1]) + out_shape[indices.shape[-1]:]
        flatted_indices = indices.view(-1, ndim)
        slices = [flatted_indices[:, i] for i in range(ndim)]
        slices += [Ellipsis]
        ret[slices] = values.view(*output_shape)
        return ret
 
    def forward(self, voxel_features, coors, batch_size, input_shape):
 
        # input: # [41, 1600, 1408]
        sparse_shape = np.array(input_shape[::-1]) + [1, 0, 0]
        coors = coors.int()
 
        output_shape = [batch_size] + list(sparse_shape) + [voxel_features.shape[1]]
        t = self.sparse_to_dense(coors, voxel_features, output_shape)
        t = t.permute(0, 4, 1, 2, 3).contiguous()   # NDHWC -> NCDHW
 
        ret, _ = self.middle_conv((t, None))
 
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
 
        return ret