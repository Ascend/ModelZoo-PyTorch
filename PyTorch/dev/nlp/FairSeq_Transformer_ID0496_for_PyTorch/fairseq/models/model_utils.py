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
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch
from torch import Tensor


@torch.jit.script
def script_skip_tensor_list(x: List[Tensor], mask):
    res = [xi[mask] if xi.size(0) == mask.size(0) else xi[:, mask] for xi in x]
    outputs = []
    for i, t in enumerate(res):
        if t.numel() != 0:
            outputs.append(t)
        else:
            outputs.append(x[i])
    return outputs


@torch.jit.script
def script_skip_tensor(x: Tensor, mask):
    # None case
    if x.size(0) == 0:
        return x
    res = x[mask] if x.size(0) == mask.size(0) else x[:, mask]
    if res.numel() == 0:
        return x
    else:
        return res


@torch.jit.script
def expand_2d_or_3d_tensor(x, trg_dim: int, padding_idx: int):
    """
    Expand 2D/3D tensor on dim=1
    """
    if x is None:
        return None

    assert x.dim() == 2 or x.dim() == 3
    assert trg_dim >= x.size(1), (trg_dim, x.size())
    if trg_dim == x.size(1):
        return x

    dims = [x.size(0), trg_dim - x.size(1)]
    if x.dim() == 3:
        dims.append(x.size(2))
    x = torch.cat([x, torch.zeros(dims).to(x).fill_(padding_idx)], 1)

    return x


@torch.jit.script
def coalesce(x: Optional[Tensor], y: Tensor) -> Tensor:
    return x if x is not None else y


@torch.jit.script
def fill_tensors(x: Optional[Tensor], mask, y: Optional[Tensor], padding_idx: int) -> Optional[Tensor]:
    """
    Filling tensor x with y at masked positions (dim=0).
    """
    if x is None or x.size()[0] == 0 or y is None:
        return x
    assert x.dim() == y.dim() and mask.size(0) == x.size(0)
    assert x.dim() == 2 or (x.dim() == 3 and x.size(2) == y.size(2))

    n_selected = mask.sum()
    if n_selected == 0:
        return x
    assert n_selected == y.size(0)
    if n_selected == x.size(0):
        return y

    if x.size(1) < y.size(1):
        x = expand_2d_or_3d_tensor(x, y.size(1), padding_idx)
        x[mask] = y
    elif x.size(1) > y.size(1):
        x[mask] = torch.tensor(padding_idx).type_as(x)
        if x.dim() == 2:
            x[mask, :y.size(1)] = y
        else:
            x[mask, :y.size(1), :] = y
    else:
        x[mask] = y
    return x
