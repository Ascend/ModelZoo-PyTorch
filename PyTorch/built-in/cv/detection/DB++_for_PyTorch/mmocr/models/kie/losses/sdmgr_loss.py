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
from mmdet.models.losses import accuracy
from torch import nn

from mmocr.models.builder import LOSSES


@LOSSES.register_module()
class SDMGRLoss(nn.Module):
    """The implementation the loss of key information extraction proposed in
    the paper: Spatial Dual-Modality Graph Reasoning for Key Information
    Extraction.

    https://arxiv.org/abs/2103.14470.
    """

    def __init__(self, node_weight=1.0, edge_weight=1.0, ignore=-100):
        super().__init__()
        self.loss_node = nn.CrossEntropyLoss(ignore_index=ignore)
        self.loss_edge = nn.CrossEntropyLoss(ignore_index=-1)
        self.node_weight = node_weight
        self.edge_weight = edge_weight
        self.ignore = ignore

    def forward(self, node_preds, edge_preds, gts):
        node_gts, edge_gts = [], []
        for gt in gts:
            node_gts.append(gt[:, 0])
            edge_gts.append(gt[:, 1:].contiguous().view(-1))
        node_gts = torch.cat(node_gts).long()
        edge_gts = torch.cat(edge_gts).long()

        node_valids = torch.nonzero(
            node_gts != self.ignore, as_tuple=False).view(-1)
        edge_valids = torch.nonzero(edge_gts != -1, as_tuple=False).view(-1)
        return dict(
            loss_node=self.node_weight * self.loss_node(node_preds, node_gts),
            loss_edge=self.edge_weight * self.loss_edge(edge_preds, edge_gts),
            acc_node=accuracy(node_preds[node_valids], node_gts[node_valids]),
            acc_edge=accuracy(edge_preds[edge_valids], edge_gts[edge_valids]))
