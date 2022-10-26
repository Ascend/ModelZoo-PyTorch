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
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import dgl.function as fn
from dgl.nn.pytorch import GATConv


class GraphConvLayer(nn.Module):
    def __init__(self, in_feats, out_feats, bias=True):
        super(GraphConvLayer, self).__init__()
        self.mlp = nn.Linear(in_feats * 2, out_feats, bias=bias)

    def forward(self, bipartite, feat):
        if isinstance(feat, tuple):
            srcfeat, dstfeat = feat
        else:
            srcfeat = feat #npu
            dstfeat = feat[:bipartite.num_dst_nodes()]
        graph = bipartite.local_var()#cpu
        srcfeat = srcfeat.to('cpu')
        graph.srcdata['h'] = srcfeat


        graph.update_all(fn.u_mul_e('h', 'affine', 'm'),
                         fn.sum(msg='m', out='h'))
        a = graph.dstdata['h']#cpu
        a = a.to('npu')
        gcn_feat = torch.cat([dstfeat, a], dim=-1)#npu
        out = self.mlp(gcn_feat)
        return out


class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0, use_GAT=False, K=1):
        super(GraphConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        if use_GAT:
            self.gcn_layer = GATConv(in_dim, out_dim, K, allow_zero_in_degree=True)
            self.bias = nn.Parameter(torch.Tensor(K, out_dim))
            init.constant_(self.bias, 0)
        else:
            self.gcn_layer = GraphConvLayer(in_dim, out_dim, bias=True)

        self.dropout = dropout
        self.use_GAT = use_GAT

    def forward(self, bipartite, features):
        out = self.gcn_layer(bipartite, features)

        if self.use_GAT:
            out = torch.mean(out + self.bias, dim=1)

        out = out.reshape(out.shape[0], -1)
        out = F.relu(out)
        if self.dropout > 0:
            out = F.dropout(out, self.dropout, training=self.training)

        return out
