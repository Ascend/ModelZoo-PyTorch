#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================
#

import torch
import torch.nn as nn
import torch.nn.functional as F

from sg2im.layers import get_normalization_2d
from sg2im.layers import get_activation
from sg2im.utils import timeit, lineno, get_gpu_memory


"""
Cascaded refinement network architecture, as described in:

Qifeng Chen and Vladlen Koltun,
"Photographic Image Synthesis with Cascaded Refinement Networks",
ICCV 2017
"""


class RefinementModule(nn.Module):
  def __init__(self, layout_dim, input_dim, output_dim,
               normalization='instance', activation='leakyrelu'):
    super(RefinementModule, self).__init__()
    
    layers = []
    layers.append(nn.Conv2d(layout_dim + input_dim, output_dim,
                            kernel_size=3, padding=1))
    layers.append(get_normalization_2d(output_dim, normalization))
    layers.append(get_activation(activation))
    layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1))
    layers.append(get_normalization_2d(output_dim, normalization))
    layers.append(get_activation(activation))
    layers = [layer for layer in layers if layer is not None]
    for layer in layers:
      if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight)
    self.net = nn.Sequential(*layers)

  def forward(self, layout, feats):
    _, _, HH, WW = layout.size()
    _, _, H, W = feats.size()
    assert HH >= H
    if HH > H:
      factor = round(HH // H)
      assert HH % factor == 0
      assert WW % factor == 0 and WW // factor == W
      layout = F.avg_pool2d(layout, kernel_size=factor, stride=factor)
    net_input = torch.cat([layout, feats], dim=1)
    out = self.net(net_input)
    return out


class RefinementNetwork(nn.Module):
  def __init__(self, dims, normalization='instance', activation='leakyrelu'):
    super(RefinementNetwork, self).__init__()
    layout_dim = dims[0]
    self.refinement_modules = nn.ModuleList()
    for i in range(1, len(dims)):
      input_dim = 1 if i == 1 else dims[i - 1]
      output_dim = dims[i]
      mod = RefinementModule(layout_dim, input_dim, output_dim,
                             normalization=normalization, activation=activation)
      self.refinement_modules.append(mod)
    output_conv_layers = [
      nn.Conv2d(dims[-1], dims[-1], kernel_size=3, padding=1),
      get_activation(activation),
      nn.Conv2d(dims[-1], 3, kernel_size=1, padding=0)
    ]
    nn.init.kaiming_normal_(output_conv_layers[0].weight)
    nn.init.kaiming_normal_(output_conv_layers[2].weight)
    self.output_conv = nn.Sequential(*output_conv_layers)

  def forward(self, layout):
    """
    Output will have same size as layout
    """
    # H, W = self.output_size
    N, _, H, W = layout.size()
    self.layout = layout

    # Figure out size of input
    input_H, input_W = H, W
    for _ in range(len(self.refinement_modules)):
      input_H //= 2
      input_W //= 2

    assert input_H != 0
    assert input_W != 0

    feats = torch.zeros(N, 1, input_H, input_W).to(layout)
    for mod in self.refinement_modules:
      feats = F.upsample(feats, scale_factor=2, mode='nearest')
      feats = mod(layout, feats)

    out = self.output_conv(feats)
    return out

