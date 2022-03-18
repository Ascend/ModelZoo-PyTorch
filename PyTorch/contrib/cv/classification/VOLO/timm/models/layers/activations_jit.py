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
""" Activations

A collection of jit-scripted activations fn and modules with a common interface so that they can
easily be swapped. All have an `inplace` arg even if not used.

All jit scripted activations are lacking in-place variations on purpose, scripted kernel fusion does not
currently work across in-place op boundaries, thus performance is equal to or less than the non-scripted
versions if they contain in-place ops.

Hacked together by / Copyright 2020 Ross Wightman
"""

import torch
from torch import nn as nn
from torch.nn import functional as F


@torch.jit.script
def swish_jit(x, inplace: bool = False):
    """Swish - Described in: https://arxiv.org/abs/1710.05941
    """
    return x.mul(x.sigmoid())


@torch.jit.script
def mish_jit(x, _inplace: bool = False):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    """
    return x.mul(F.softplus(x).tanh())


class SwishJit(nn.Module):
    def __init__(self, inplace: bool = False):
        super(SwishJit, self).__init__()

    def forward(self, x):
        return swish_jit(x)


class MishJit(nn.Module):
    def __init__(self, inplace: bool = False):
        super(MishJit, self).__init__()

    def forward(self, x):
        return mish_jit(x)


@torch.jit.script
def hard_sigmoid_jit(x, inplace: bool = False):
    # return F.relu6(x + 3.) / 6.
    return (x + 3).clamp(min=0, max=6).div(6.)  # clamp seems ever so slightly faster?


class HardSigmoidJit(nn.Module):
    def __init__(self, inplace: bool = False):
        super(HardSigmoidJit, self).__init__()

    def forward(self, x):
        return hard_sigmoid_jit(x)


@torch.jit.script
def hard_swish_jit(x, inplace: bool = False):
    # return x * (F.relu6(x + 3.) / 6)
    return x * (x + 3).clamp(min=0, max=6).div(6.)  # clamp seems ever so slightly faster?


class HardSwishJit(nn.Module):
    def __init__(self, inplace: bool = False):
        super(HardSwishJit, self).__init__()

    def forward(self, x):
        return hard_swish_jit(x)


@torch.jit.script
def hard_mish_jit(x, inplace: bool = False):
    """ Hard Mish
    Experimental, based on notes by Mish author Diganta Misra at
      https://github.com/digantamisra98/H-Mish/blob/0da20d4bc58e696b6803f2523c58d3c8a82782d0/README.md
    """
    return 0.5 * x * (x + 2).clamp(min=0, max=2)


class HardMishJit(nn.Module):
    def __init__(self, inplace: bool = False):
        super(HardMishJit, self).__init__()

    def forward(self, x):
        return hard_mish_jit(x)
