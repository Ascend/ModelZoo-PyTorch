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
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from .graphconv import GraphConv
from .focal_loss import FocalLoss


class LANDER(nn.Module):
    def __init__(self, feature_dim, nhid, num_conv=4, dropout=0,
                 use_GAT=False, K=1, balance=False,
                 use_cluster_feat=True, use_focal_loss=True, **kwargs):
        super(LANDER, self).__init__()
        nhid_half = int(nhid / 2)
        self.use_cluster_feat = use_cluster_feat
        self.use_focal_loss = use_focal_loss

        if self.use_cluster_feat:
            self.feature_dim = feature_dim * 2
        else:
            self.feature_dim = feature_dim

        input_dim = (feature_dim, nhid, nhid, nhid_half)
        output_dim = (nhid, nhid, nhid_half, nhid_half)
        self.conv = nn.ModuleList()
        self.conv.append(GraphConv(self.feature_dim, nhid, dropout, use_GAT, K))
        for i in range(1, num_conv):
            self.conv.append(GraphConv(input_dim[i], output_dim[i], dropout, use_GAT, K))

        self.src_mlp = nn.Linear(output_dim[num_conv - 1], nhid_half)
        self.dst_mlp = nn.Linear(output_dim[num_conv - 1], nhid_half)

        self.classifier_conn = nn.Sequential(nn.PReLU(nhid_half),
                                             nn.Linear(nhid_half, nhid_half),
                                             nn.PReLU(nhid_half),
                                             nn.Linear(nhid_half, 2))

        if self.use_focal_loss:
            self.loss_conn = FocalLoss(2)
        else:
            self.loss_conn = nn.CrossEntropyLoss()
        self.loss_den = nn.MSELoss()

        self.balance = balance

    def pred_conn(self, edges):
        a = edges.src['conv_features']
        b = edges.dst['conv_features']
        a = a.to('npu')
        b = b.to('npu')
        src_feat = self.src_mlp(a)
        dst_feat = self.dst_mlp(b)
        pred_conn = self.classifier_conn(src_feat + dst_feat) #npu
        c = pred_conn
        c = c.to('cpu')
        return {'pred_conn': c}

    def pred_den_msg(self, edges):
        prob = edges.data['prob_conn']
        res = edges.data['raw_affine'] * (prob[:, 1] - prob[:, 0])
        return {'pred_den_msg': res}

    def forward(self, bipartites, device):
        if isinstance(bipartites, dgl.DGLGraph):
            bipartites = [bipartites] * len(self.conv)
            if self.use_cluster_feat:
                neighbor_x = torch.cat([bipartites[0].ndata['features'], bipartites[0].ndata['cluster_features']],
                                       axis=1)
            else:
                neighbor_x = bipartites[0].ndata['features']

            for i in range(len(self.conv)):
                neighbor_x = self.conv[i](bipartites[i], neighbor_x)

            output_bipartite = bipartites[-1]
            output_bipartite.ndata['conv_features'] = neighbor_x
        else:
            if self.use_cluster_feat:
                neighbor_x_src = torch.cat(
                    [bipartites[0].srcdata['features'], bipartites[0].srcdata['cluster_features']], axis=1)
                center_x_src = torch.cat([bipartites[1].srcdata['features'], bipartites[1].srcdata['cluster_features']],
                                         axis=1)
            else:
                neighbor_x_src = bipartites[0].srcdata['features']
                center_x_src = bipartites[1].srcdata['features']
            neighbor_x_src = neighbor_x_src.to(device) #npu
            center_x_src = center_x_src.to(device)
            for i in range(len(self.conv)):
                neighbor_x_dst = neighbor_x_src[:bipartites[i].num_dst_nodes()]
                neighbor_x_src = self.conv[i](bipartites[i], (neighbor_x_src, neighbor_x_dst))
                center_x_dst = center_x_src[:bipartites[i + 1].num_dst_nodes()]
                center_x_src = self.conv[i](bipartites[i + 1], (center_x_src, center_x_dst))
            output_bipartite = bipartites[-1] #cpu
            neighbor_x_src = neighbor_x_src.to('cpu')
            output_bipartite.srcdata['conv_features'] = neighbor_x_src
            center_x_src = center_x_src.to('cpu')
            output_bipartite.dstdata['conv_features'] = center_x_src

        output_bipartite.apply_edges(self.pred_conn)
        output_bipartite.edata['prob_conn'] = F.softmax(output_bipartite.edata['pred_conn'], dim=1)
        output_bipartite.update_all(self.pred_den_msg, fn.mean('pred_den_msg', 'pred_den'))
        return output_bipartite

    def compute_loss(self, bipartite):
        pred_den = bipartite.dstdata['pred_den']
        loss_den = self.loss_den(pred_den, bipartite.dstdata['density'])

        labels_conn = bipartite.edata['labels_conn']
        mask_conn = bipartite.edata['mask_conn']

        if self.balance:
            labels_conn = bipartite.edata['labels_conn']
            neg_check = torch.logical_and(bipartite.edata['labels_conn'] == 0, mask_conn)
            num_neg = torch.sum(neg_check).item()
            neg_indices = torch.where(neg_check)[0]
            pos_check = torch.logical_and(bipartite.edata['labels_conn'] == 1, mask_conn)
            num_pos = torch.sum(pos_check).item()
            pos_indices = torch.where(pos_check)[0]
            if num_pos > num_neg:
                mask_conn[pos_indices[np.random.choice(num_pos, num_pos - num_neg, replace=False)]] = 0
            elif num_pos < num_neg:
                mask_conn[neg_indices[np.random.choice(num_neg, num_neg - num_pos, replace=False)]] = 0

        # In subgraph training, it may happen that all edges are masked in a batch
        if mask_conn.sum() > 0:
            loss_conn = self.loss_conn(bipartite.edata['pred_conn'][mask_conn], labels_conn[mask_conn])
            loss = loss_den + loss_conn
            loss_den_val = loss_den.item()
            loss_conn_val = loss_conn.item()
        else:
            loss = loss_den
            loss_den_val = loss_den.item()
            loss_conn_val = 0

        return loss, loss_den_val, loss_conn_val
