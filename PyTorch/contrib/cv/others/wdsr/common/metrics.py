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
"""Metrics."""

import torch
import torch.nn.functional as F


def psnr(sr, hr, shave=4):
  sr = sr.to(hr.dtype)
  sr = (sr * 255).round().clamp(0, 255) / 255
  diff = sr - hr
  if shave:
    diff = diff[..., shave:-shave, shave:-shave]
  mse = diff.pow(2).mean([-3, -2, -1])
  psnr = -10 * mse.log10()
  return psnr.mean()


def psnr_y(sr, hr, shave=4):
  sr = sr.to(hr.dtype)
  sr = (sr * 255).round().clamp(0, 255) / 255
  diff = sr - hr
  if diff.shape[1] == 3:
    filters = torch.tensor([0.257, 0.504, 0.098],
                           dtype=diff.dtype,
                           device=diff.device)
    diff = F.conv2d(diff, filters.view([1, -1, 1, 1]))
  if shave:
    diff = diff[..., shave:-shave, shave:-shave]
  mse = diff.pow(2).mean([-3, -2, -1])
  psnr = -10 * mse.log10()
  return psnr.mean()
