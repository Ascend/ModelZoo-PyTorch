# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017
# All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd
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
# ==========================================================================

# Copyright (c) OpenMMLab. All rights reserved.
import torch


def compute_f1_score(preds, gts, ignores=[]):
    """Compute the F1-score of prediction.

    Args:
        preds (Tensor): The predicted probability NxC map
            with N and C being the sample number and class
            number respectively.
        gts (Tensor): The ground truth vector of size N.
        ignores (list): The index set of classes that are ignored when
            reporting results.
            Note: all samples are participated in computing.

     Returns:
        The numpy list of f1-scores of valid classes.
    """
    C = preds.size(1)
    classes = torch.LongTensor(sorted(set(range(C)) - set(ignores)))
    hist = torch.bincount(
        gts * C + preds.argmax(1), minlength=C**2).view(C, C).float()
    diag = torch.diag(hist)
    recalls = diag / hist.sum(1).clamp(min=1)
    precisions = diag / hist.sum(0).clamp(min=1)
    f1 = 2 * recalls * precisions / (recalls + precisions).clamp(min=1e-8)
    return f1[classes].cpu().numpy()
