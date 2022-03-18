"""
BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Copyright 2020 Huawei Technologies Co., Ltd

Licensed under the BSD 3-Clause License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://spdx.org/licenses/BSD-3-Clause.html

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import torch
import torch.nn as nn


class EvoNormBatch2d(nn.Module):
    def __init__(self, num_features, apply_act=True, momentum=0.1, eps=1e-5, drop_block=None):
        super(EvoNormBatch2d, self).__init__()
        self.apply_act = apply_act  # apply activation (non-linearity)
        self.momentum = momentum
        self.eps = eps
        param_shape = (1, num_features, 1, 1)
        self.weight = nn.Parameter(torch.ones(param_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(param_shape), requires_grad=True)
        if apply_act:
            self.v = nn.Parameter(torch.ones(param_shape), requires_grad=True)
        self.register_buffer('running_var', torch.ones(1, num_features, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.apply_act:
            nn.init.ones_(self.v)

    def forward(self, x):
        assert x.dim() == 4, 'expected 4D input'
        x_type = x.dtype
        if self.training:
            var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
            n = x.numel() / x.shape[1]
            self.running_var.copy_(
                var.detach() * self.momentum * (n / (n - 1)) + self.running_var * (1 - self.momentum))
        else:
            var = self.running_var

        if self.apply_act:
            v = self.v.to(dtype=x_type)
            d = x * v + (x.var(dim=(2, 3), unbiased=False, keepdim=True) + self.eps).sqrt().to(dtype=x_type)
            d = d.max((var + self.eps).sqrt().to(dtype=x_type))
            x = x / d
        return x * self.weight + self.bias


class EvoNormSample2d(nn.Module):
    def __init__(self, num_features, apply_act=True, groups=8, eps=1e-5, drop_block=None):
        super(EvoNormSample2d, self).__init__()
        self.apply_act = apply_act  # apply activation (non-linearity)
        self.groups = groups
        self.eps = eps
        param_shape = (1, num_features, 1, 1)
        self.weight = nn.Parameter(torch.ones(param_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(param_shape), requires_grad=True)
        if apply_act:
            self.v = nn.Parameter(torch.ones(param_shape), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.apply_act:
            nn.init.ones_(self.v)

    def forward(self, x):
        assert x.dim() == 4, 'expected 4D input'
        B, C, H, W = x.shape
        assert C % self.groups == 0
        if self.apply_act:
            n = x * (x * self.v).sigmoid()
            x = x.reshape(B, self.groups, -1)
            x = n.reshape(B, self.groups, -1) / (x.var(dim=-1, unbiased=False, keepdim=True) + self.eps).sqrt()
            x = x.reshape(B, C, H, W)
        return x * self.weight + self.bias
