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

import torch

@torch.jit.script
def slice_helper(x, offset):
  return x[:, -offset: , : ]

@torch.jit.script
def slice_helper2(x: torch.Tensor, start: torch.Tensor, end: torch.Tensor):
  start = start.long()
  end = end.long()
  return x[:, start:end]

@torch.jit.script
def slice_helper3(x, start):
  return x[:, start:]

@torch.jit.script
def get_item(x):
  item = x.detach().item()
  output = torch.tensor(item)
  return output

@torch.jit.script
def get_next_cache_start(required_cache_size: torch.Tensor, xs: torch.Tensor):
  # required_cache_size = required_cache_size_tensor.detach().item()
  next_cache_start = 0
  if required_cache_size < 0:
    next_cache_start = 0
  elif required_cache_size == 0:
    next_cache_start = xs.size(1)
  else:
    if xs.size(1) - required_cache_size < 0:
      next_cache_start = 0
    else:
      next_cache_start = xs.size(1) - required_cache_size
  return torch.tensor(next_cache_start, dtype=torch.int64)
