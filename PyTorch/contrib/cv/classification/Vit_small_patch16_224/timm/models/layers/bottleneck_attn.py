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
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .helpers import to_2tuple
from .weight_init import trunc_normal_


def rel_logits_1d(q, rel_k, permute_mask: List[int]):
    """ Compute relative logits along one dimension

    As per: https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2
    Originally from: `Attention Augmented Convolutional Networks` - https://arxiv.org/abs/1904.09925

    Args:
        q: (batch, heads, height, width, dim)
        rel_k: (2 * width - 1, dim)
        permute_mask: permute output dim according to this
    """
    B, H, W, dim = q.shape
    x = (q @ rel_k.transpose(-1, -2))
    x = x.reshape(-1, W, 2 * W -1)

    # pad to shift from relative to absolute indexing
    x_pad = F.pad(x, [0, 1]).flatten(1)
    x_pad = F.pad(x_pad, [0, W - 1])

    # reshape and slice out the padded elements
    x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    x = x_pad[:, :W, W - 1:]

    # reshape and tile
    x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    return x.permute(permute_mask)


class PosEmbedRel(nn.Module):
    """ Relative Position Embedding
    As per: https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2
    Originally from: `Attention Augmented Convolutional Networks` - https://arxiv.org/abs/1904.09925
    """
    def __init__(self, feat_size, dim_head, scale):
        super().__init__()
        self.height, self.width = to_2tuple(feat_size)
        self.dim_head = dim_head
        self.scale = scale
        self.height_rel = nn.Parameter(torch.randn(self.height * 2 - 1, dim_head) * self.scale)
        self.width_rel = nn.Parameter(torch.randn(self.width * 2 - 1, dim_head) * self.scale)

    def forward(self, q):
        B, num_heads, HW, _ = q.shape

        # relative logits in width dimension.
        q = q.reshape(B * num_heads, self.height, self.width, -1)
        rel_logits_w = rel_logits_1d(q, self.width_rel, permute_mask=(0, 1, 3, 2, 4))

        # relative logits in height dimension.
        q = q.transpose(1, 2)
        rel_logits_h = rel_logits_1d(q, self.height_rel, permute_mask=(0, 3, 1, 4, 2))

        rel_logits = rel_logits_h + rel_logits_w
        rel_logits = rel_logits.reshape(B, num_heads, HW, HW)
        return rel_logits


class BottleneckAttn(nn.Module):
    """ Bottleneck Attention
    Paper: `Bottleneck Transformers for Visual Recognition` - https://arxiv.org/abs/2101.11605
    """
    def __init__(self, dim, dim_out=None, feat_size=None, stride=1, num_heads=4, qkv_bias=False):
        super().__init__()
        assert feat_size is not None, 'A concrete feature size matching expected input (H, W) is required'
        dim_out = dim_out or dim
        assert dim_out % num_heads == 0
        self.num_heads = num_heads
        self.dim_out = dim_out
        self.dim_head = dim_out // num_heads
        self.scale = self.dim_head ** -0.5

        self.qkv = nn.Conv2d(dim, self.dim_out * 3, 1, bias=qkv_bias)

        # NOTE I'm only supporting relative pos embedding for now
        self.pos_embed = PosEmbedRel(feat_size, dim_head=self.dim_head, scale=self.scale)

        self.pool = nn.AvgPool2d(2, 2) if stride == 2 else nn.Identity()

    def reset_parameters(self):
        trunc_normal_(self.qkv.weight, std=self.qkv.weight.shape[1] ** -0.5)
        trunc_normal_(self.pos_embed.height_rel, std=self.scale)
        trunc_normal_(self.pos_embed.width_rel, std=self.scale)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.pos_embed.height and W == self.pos_embed.width

        x = self.qkv(x)  # B, 3 * num_heads * dim_head, H, W
        x = x.reshape(B, -1, self.dim_head, H * W).transpose(-1, -2)
        q, k, v = torch.split(x, self.num_heads, dim=1)

        attn_logits = (q @ k.transpose(-1, -2)) * self.scale
        attn_logits = attn_logits + self.pos_embed(q)  # B, num_heads, H * W, H * W

        attn_out = attn_logits.softmax(dim = -1)
        attn_out = (attn_out @ v).transpose(1, 2).reshape(B, self.dim_out, H, W) # B, dim_out, H, W
        attn_out = self.pool(attn_out)
        return attn_out


