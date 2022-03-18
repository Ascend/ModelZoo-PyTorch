#     Copyright [yyyy] [name of copyright owner]
#     Copyright 2021 Huawei Technologies Co., Ltd
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
#

""" Normalization layers and wrappers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels, num_groups, eps=1e-5, affine=True):
        # NOTE num_channels is swapped to first arg for consistency in swapping norm layers with BN
        super().__init__(num_groups, num_channels, eps=eps, affine=affine)

    def forward(self, x):
        return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)


class LayerNorm2d(nn.LayerNorm):
    """ Layernorm for channels of '2d' spatial BCHW tensors """
    def __init__(self, num_channels):
        super().__init__([num_channels, 1, 1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
