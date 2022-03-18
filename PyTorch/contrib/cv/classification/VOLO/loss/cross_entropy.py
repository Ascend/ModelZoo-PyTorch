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
# Copyright 2021 Sea Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Loss functions for VOLO
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTargetCrossEntropy(nn.Module):
    """
    The native CE loss with soft target
    input: x is output of model, target is ground truth
    return: loss
    """
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        N_rep = x.shape[0]
        N = target.shape[0]
        if not N == N_rep:
            target = target.repeat(N_rep // N, 1)
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


class TokenLabelGTCrossEntropy(nn.Module):
    """
    Token labeling dense loss with ground gruth, see more from token labeling
    input: x is output of model, target is ground truth
    return: loss
    """
    def __init__(self,
                 dense_weight=1.0,
                 cls_weight=1.0,
                 mixup_active=True,
                 smoothing=0.1,
                 classes=1000):
        super(TokenLabelGTCrossEntropy, self).__init__()

        self.CE = SoftTargetCrossEntropy()

        self.dense_weight = dense_weight
        self.smoothing = smoothing
        self.mixup_active = mixup_active
        self.classes = classes
        self.cls_weight = cls_weight
        assert dense_weight + cls_weight > 0

    def forward(self, x, target):

        output, aux_output, bb = x
        bbx1, bby1, bbx2, bby2 = bb

        B, N, C = aux_output.shape
        if len(target.shape) == 2:
            target_cls = target
            target_aux = target.repeat(1, N).reshape(B * N, C)
        else:
            ground_truth = target[:, :, 0]
            target_cls = target[:, :, 1]
            ratio = (0.9 - 0.4 *
                     (ground_truth.max(-1)[1] == target_cls.max(-1)[1])
                     ).unsqueeze(-1)
            target_cls = target_cls * ratio + ground_truth * (1 - ratio)
            target_aux = target[:, :, 2:]
            target_aux = target_aux.transpose(1, 2).reshape(-1, C)
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / N)
        if lam < 1:
            target_cls = lam * target_cls + (1 - lam) * target_cls.flip(0)

        aux_output = aux_output.reshape(-1, C)

        loss_cls = self.CE(output, target_cls)
        loss_aux = self.CE(aux_output, target_aux)

        return self.cls_weight * loss_cls + self.dense_weight * loss_aux


class TokenLabelSoftTargetCrossEntropy(nn.Module):
    """
    Token labeling dense loss with soft target, see more from token labeling
    input: x is output of model, target is ground truth
    return: loss
    """
    def __init__(self):
        super(TokenLabelSoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        N_rep = x.shape[0]
        N = target.shape[0]
        if not N == N_rep:
            target = target.repeat(N_rep // N, 1)
        if len(target.shape) == 3 and target.shape[-1] == 2:
            target = target[:, :, 1]
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


class TokenLabelCrossEntropy(nn.Module):
    """
    Token labeling loss without ground truth
    input: x is output of model, target is ground truth
    return: loss
    """
    def __init__(self,
                 dense_weight=1.0,
                 cls_weight=1.0,
                 mixup_active=True,
                 classes=1000):
        """
        Constructor Token labeling loss.
        """
        super(TokenLabelCrossEntropy, self).__init__()

        self.CE = SoftTargetCrossEntropy()

        self.dense_weight = dense_weight
        self.mixup_active = mixup_active
        self.classes = classes
        self.cls_weight = cls_weight
        assert dense_weight + cls_weight > 0

    def forward(self, x, target):

        output, aux_output, bb = x
        bbx1, bby1, bbx2, bby2 = bb

        B, N, C = aux_output.shape
        if len(target.shape) == 2:
            target_cls = target
            target_aux = target.repeat(1, N).reshape(B * N, C)
        else:
            target_cls = target[:, :, 1]
            target_aux = target[:, :, 2:]
            target_aux = target_aux.transpose(1, 2).reshape(-1, C)
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / N)
        if lam < 1:
            target_cls = lam * target_cls + (1 - lam) * target_cls.flip(0)

        aux_output = aux_output.reshape(-1, C)
        loss_cls = self.CE(output, target_cls)
        loss_aux = self.CE(aux_output, target_aux)
        return self.cls_weight * loss_cls + self.dense_weight * loss_aux
