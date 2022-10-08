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
import argparse, time, os, pickle
import numpy as np

import dgl
import torch
import torch.optim as optim

from models import LANDER
from dataset import LanderDataset
from utils import evaluation, decode, build_next_level, stop_iterating

###########
# ArgParser
parser = argparse.ArgumentParser()

# Dataset
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--model_filename', type=str, default='lander.pth')
parser.add_argument('--faiss_gpu', action='store_true')
parser.add_argument('--early_stop', action='store_true')

# HyperParam
parser.add_argument('--knn_k', type=int, default=10)
parser.add_argument('--levels', type=int, default=1)
parser.add_argument('--tau', type=float, default=0.5)
parser.add_argument('--threshold', type=str, default='prob')
parser.add_argument('--metrics', type=str, default='pairwise,bcubed,nmi')

# Model
parser.add_argument('--hidden', type=int, default=512)
parser.add_argument('--num_conv', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.)
parser.add_argument('--gat', action='store_true')
parser.add_argument('--gat_k', type=int, default=1)
parser.add_argument('--balance', action='store_true')
parser.add_argument('--use_cluster_feat', action='store_true')
parser.add_argument('--use_focal_loss', action='store_true')
parser.add_argument('--use_gt', action='store_true')

args = parser.parse_args()

###########################
# Environment Configuration
if torch.cuda.is_available():
    device = torch.device('cuda')                                           ########
else:
    device = torch.device('cpu')                                           ########

##################
# Data Preparation
with open(args.data_path, 'rb') as f:
    features, labels = pickle.load(f)
global_features = features.copy()
dataset = LanderDataset(features=features, labels=labels, k=args.knn_k,
                        levels=1, faiss_gpu=args.faiss_gpu)
g = dataset.gs[0].to(device)
global_labels = labels.copy()
ids = np.arange(g.number_of_nodes())
global_edges = ([], [])
global_edges_len = len(global_edges[0])
global_num_nodes = g.number_of_nodes()

##################
# Model Definition
if not args.use_gt:
    feature_dim = g.ndata['features'].shape[1]
    model = LANDER(feature_dim=feature_dim, nhid=args.hidden,
                   num_conv=args.num_conv, dropout=args.dropout,
                   use_GAT=args.gat, K=args.gat_k,
                   balance=args.balance,
                   use_cluster_feat=args.use_cluster_feat,
                   use_focal_loss=args.use_focal_loss)
    model.load_state_dict(torch.load(args.model_filename))
    model = model.to(device)
    model.eval()

# number of edges added is the indicator for early stopping
num_edges_add_last_level = np.Inf
##################################
# Predict connectivity and density
for level in range(args.levels):
    if not args.use_gt:
        with torch.no_grad():
            g = model(g)
    new_pred_labels, peaks,\
        global_edges, global_pred_labels, global_peaks = decode(g, args.tau, args.threshold, args.use_gt,
                                                                ids, global_edges, global_num_nodes)
    ids = ids[peaks]
    new_global_edges_len = len(global_edges[0])
    num_edges_add_this_level = new_global_edges_len - global_edges_len
    if stop_iterating(level, args.levels, args.early_stop, num_edges_add_this_level, num_edges_add_last_level, args.knn_k):
        break
    global_edges_len = new_global_edges_len
    num_edges_add_last_level = num_edges_add_this_level

    # build new dataset
    features, labels, cluster_features = build_next_level(features, labels, peaks,
                                                          global_features, global_pred_labels, global_peaks)
    # After the first level, the number of nodes reduce a lot. Using cpu faiss is faster.
    dataset = LanderDataset(features=features, labels=labels, k=args.knn_k,
                            levels=1, faiss_gpu=False, cluster_features = cluster_features)
    if len(dataset.gs) == 0:
        break
    g = dataset.gs[0].to(device)
evaluation(global_pred_labels, global_labels, args.metrics)
