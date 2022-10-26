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
"""Initialising Optimisers"""

import torch
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


def create_optimisers(
    optim_enc,
    optim_dec,
    lr_enc,
    lr_dec,
    mom_enc,
    mom_dec,
    wd_enc,
    wd_dec,
    param_enc,
    param_dec,
):
    """Create optimisers for encoder, decoder

    Args:
      optim_enc (str) : type of optimiser for encoder
      optim_dec (str) : type of optimiser for decoder
      lr_enc (float) : learning rate for encoder
      lr_dec (float) : learning rate for decoder
      mom_enc (float) : momentum for encoder
      mom_dec (float) : momentum for decoder
      wd_enc (float) : weight decay for encoder
      wd_dec (float) : weight decay for decoder
      param_enc (torch.parameters()) : encoder parameters
      param_dec (torch.parameters()) : decoder parameters

    Returns optim_enc, optim_dec (torch.optim)

    """
    if optim_enc == "sgd":
        optim_enc = torch.optim.SGD(
            param_enc, lr=lr_enc, momentum=mom_enc, weight_decay=wd_enc
        )
    elif optim_enc == "adam":
        optim_enc = torch.optim.Adam(param_enc, lr=lr_enc, weight_decay=wd_enc)
    else:
        raise ValueError("Unknown Encoder Optimiser: {}".format(optim_enc))

    if optim_dec == "sgd":
        optim_dec = torch.optim.SGD(
            param_dec, lr=lr_dec, momentum=mom_dec, weight_decay=wd_dec
        )
    elif optim_dec == "adam":
        optim_dec = torch.optim.Adam(param_dec, lr=lr_dec, weight_decay=wd_dec)
    else:
        raise ValueError("Unknown Decoder Optimiser: {}".format(optim_dec))
    return optim_enc, optim_dec
