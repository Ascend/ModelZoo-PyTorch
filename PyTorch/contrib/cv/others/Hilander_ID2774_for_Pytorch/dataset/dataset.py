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
import numpy as np
import pickle

import dgl
import torch

from utils import (build_knns, fast_knns2spmat, row_normalize, knns2ordered_nbrs,
                   density_estimation, sparse_mx_to_indices_values, l2norm,
                   decode, build_next_level)

class LanderDataset(object):
    def __init__(self, features, labels, cluster_features=None, k=10, levels=1, faiss_gpu=False):
        self.k = k
        self.gs = []
        self.nbrs = []
        self.dists = []
        self.levels = levels

        # Initialize features and labels
        features = l2norm(features.astype('float32'))
        global_features = features.copy()
        if cluster_features is None:
            cluster_features = features
        global_num_nodes = features.shape[0]
        global_edges = ([], [])
        global_peaks = np.array([], dtype=np.long)
        ids = np.arange(global_num_nodes)

        # Recursive graph construction
        for lvl in range(self.levels):
            if features.shape[0] <= self.k:
                self.levels = lvl
                break
            if faiss_gpu:
                knns = build_knns(features, self.k, 'faiss_gpu')
            else:
                knns = build_knns(features, self.k, 'faiss')
            dists, nbrs = knns2ordered_nbrs(knns)
            self.nbrs.append(nbrs)
            self.dists.append(dists)
            density = density_estimation(dists, nbrs, labels)

            g = self._build_graph(features, cluster_features, labels, density, knns)
            self.gs.append(g)

            if lvl >= self.levels - 1:
                break

            # Decode peak nodes
            new_pred_labels, peaks,\
                global_edges, global_pred_labels, global_peaks = decode(g, 0, 'sim', True,
                                                                        ids, global_edges, global_num_nodes,
                                                                        global_peaks)
            ids = ids[peaks]
            features, labels, cluster_features = build_next_level(features, labels, peaks,
                                                                  global_features, global_pred_labels, global_peaks)

    def _build_graph(self, features, cluster_features, labels, density, knns):
        adj = fast_knns2spmat(knns, self.k)
        adj, adj_row_sum = row_normalize(adj)
        indices, values, shape = sparse_mx_to_indices_values(adj)

        g = dgl.graph((indices[1], indices[0]))
        g.ndata['features'] = torch.FloatTensor(features)
        g.ndata['cluster_features'] = torch.FloatTensor(cluster_features)
        g.ndata['labels'] = torch.LongTensor(labels)
        g.ndata['density'] = torch.FloatTensor(density)
        g.edata['affine'] = torch.FloatTensor(values)
        # A Bipartite from DGL sampler will not store global eid, so we explicitly save it here
        g.edata['global_eid'] = g.edges(form='eid')
        g.ndata['norm'] = torch.FloatTensor(adj_row_sum)
        g.apply_edges(lambda edges: {'raw_affine': edges.data['affine'] / edges.dst['norm']})
        g.apply_edges(lambda edges: {'labels_conn': (edges.src['labels'] == edges.dst['labels']).long()})
        g.apply_edges(lambda edges: {'mask_conn': (edges.src['density'] > edges.dst['density']).bool()})
        return g

    def __getitem__(self, index):
        assert index < len(self.gs)
        return self.gs[index]

    def __len__(self):
        return len(self.gs)
