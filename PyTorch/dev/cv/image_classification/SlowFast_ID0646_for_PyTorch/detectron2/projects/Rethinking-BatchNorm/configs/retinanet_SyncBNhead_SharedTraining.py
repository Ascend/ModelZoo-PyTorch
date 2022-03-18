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
from typing import List
import torch
from torch import Tensor, nn

from detectron2.modeling.meta_arch.retinanet import RetinaNetHead


def apply_sequential(inputs, modules):
    for mod in modules:
        if isinstance(mod, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            # for BN layer, normalize all inputs together
            shapes = [i.shape for i in inputs]
            spatial_sizes = [s[2] * s[3] for s in shapes]
            x = [i.flatten(2) for i in inputs]
            x = torch.cat(x, dim=2).unsqueeze(3)
            x = mod(x).split(spatial_sizes, dim=2)
            inputs = [i.view(s) for s, i in zip(shapes, x)]
        else:
            inputs = [mod(i) for i in inputs]
    return inputs


class RetinaNetHead_SharedTrainingBN(RetinaNetHead):
    def forward(self, features: List[Tensor]):
        logits = apply_sequential(features, list(self.cls_subnet) + [self.cls_score])
        bbox_reg = apply_sequential(features, list(self.bbox_subnet) + [self.bbox_pred])
        return logits, bbox_reg


from .retinanet_SyncBNhead import model, dataloader, lr_multiplier, optimizer, train

model.head._target_ = RetinaNetHead_SharedTrainingBN
