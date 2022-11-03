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
import os
import argparse
import sys




import dgl
import torch
if torch.__version__ >= "1.8":
    import torch_npu
import torch.optim as optim
import datetime
from models import LANDER
from dataset import LanderDataset

###########
# ArgParser
parser = argparse.ArgumentParser()

# Dataset
parser.add_argument('--data_path', type=str, default='data/inat2018_train_dedup_inter_intra_1_in_6_per_class.pkl')
parser.add_argument('--levels', type=str, default='1')
parser.add_argument('--faiss_gpu', action='store_true', default=False)
parser.add_argument('--model_filename', type=str,
                    default='checkpoint/inat2018_train_dedup_inter_intra_1_in_6_per_class.ckpt')

# KNN
parser.add_argument('--knn_k', type=str, default='10')
parser.add_argument('--num_workers', type=int, default=24)

# Model
parser.add_argument('--hidden', type=int, default=512)
parser.add_argument('--num_conv', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.)
parser.add_argument('--gat', action='store_false', default=False)
parser.add_argument('--gat_k', type=int, default=1)
parser.add_argument('--balance', action='store_true', default=True)
parser.add_argument('--use_cluster_feat', action='store_true')
parser.add_argument('--use_focal_loss', action='store_true')

# Training
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=1e-5)

args = parser.parse_args()
print(args)

###########################
# Environment Configuration
if torch.cuda.is_available():
    device = torch.device('cuda')  ########
else:
    device = "npu:0"
    torch.npu.set_device(device)  ########

##################
# Data Preparation
if __name__ == '__main__':
    with open(args.data_path, 'rb') as f:
        features, labels = pickle.load(f)

k_list = [int(k) for k in args.knn_k.split(',')]
lvl_list = [int(l) for l in args.levels.split(',')]
gs = []
nbrs = []
ks = []
for k, l in zip(k_list, lvl_list):
    dataset = LanderDataset(features=features, labels=labels, k=k,
                            levels=l, faiss_gpu=args.faiss_gpu)
    gs += [g for g in dataset.gs]
    ks += [k for g in dataset.gs]
    nbrs += [nbr for nbr in dataset.nbrs]

print('Dataset Prepared.')


def set_train_sampler_loader(g, k):
    fanouts = [k - 1 for i in range(args.num_conv + 1)]
    sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)
    # fix the number of edges
    train_dataloader = dgl.dataloading.NodeDataLoader(
        g, torch.arange(g.number_of_nodes()), sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers
    )
    return train_dataloader


train_loaders = []
for gidx, g in enumerate(gs):
    train_dataloader = set_train_sampler_loader(gs[gidx], ks[gidx])
    train_loaders.append(train_dataloader)

##################
# Model Definition
feature_dim = gs[0].ndata['features'].shape[1]
model = LANDER(feature_dim=feature_dim, nhid=args.hidden,
               num_conv=args.num_conv, dropout=args.dropout,
               use_GAT=args.gat, K=args.gat_k,
               balance=args.balance,
               use_cluster_feat=args.use_cluster_feat,
               use_focal_loss=args.use_focal_loss)
model = model.to(device)
model.train()

#################
# Hyperparameters
opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                weight_decay=args.weight_decay)

# keep num_batch_per_loader the same for every sub_dataloader
num_batch_per_loader = len(train_loaders[0])
train_loaders = [iter(train_loader) for train_loader in train_loaders]
num_loaders = len(train_loaders)
scheduler = optim.lr_scheduler.CosineAnnealingLR(opt,
                                                 T_max=args.epochs * num_batch_per_loader * num_loaders,
                                                 eta_min=1e-5)

# checkpoint = torch.load('checkpoint/1_inat2018_train_dedup_inter_intra_1_in_6_per_class.ckpt')
# model.load_state_dict(checkpoint['model'])
# opt.load_state_dict(checkpoint['optimizer'])
# start_epoch = checkpoint['epoch'] + 1
#

print('Start Training.')

###############
# Training Loop
for epoch in range(args.epochs):
    loss_den_val_total = []
    loss_conn_val_total = []
    loss_val_total = []
    startTime = datetime.datetime.now()
    for batch in range(num_batch_per_loader):
        for loader_id in range(num_loaders):
            try:
                minibatch = next(train_loaders[loader_id])
            except:
                train_loaders[loader_id] = iter(set_train_sampler_loader(gs[loader_id], ks[loader_id]))
                minibatch = next(train_loaders[loader_id])
            input_nodes, sub_g, bipartites = minibatch
            sub_g = sub_g.to(device)
            bipartites = [b.int() for b in bipartites]
            # get the feature for the input_nodes
            opt.zero_grad()
            output_bipartite = model(bipartites, device)
            loss, loss_den_val, loss_conn_val = model.compute_loss(output_bipartite)
            loss_den_val_total.append(loss_den_val)
            loss_conn_val_total.append(loss_conn_val)
            loss_val_total.append(loss.item())
            loss.backward()
            opt.step()
            if (batch + 1) % 10 == 0:
                print('epoch: %d, batch: %d / %d, loader_id : %d / %d, loss: %.6f, loss_den: %.6f, loss_conn: %.6f' %
                      (epoch, batch, num_batch_per_loader, loader_id, num_loaders,
                       loss.item(), loss_den_val, loss_conn_val))
            scheduler.step()
    endTime = datetime.datetime.now()
    durTime = 'epoch time use:%.3fs' % (
            (endTime - startTime).seconds + (endTime - startTime).microseconds / 1000)
    print(durTime)
    print('epoch: %d, loss: %.6f, loss_den: %.6f, loss_conn: %.6f' %
          (epoch, np.array(loss_val_total).mean(),
           np.array(loss_den_val_total).mean(), np.array(loss_conn_val_total).mean()))
    if (epoch + 1) % 50 == 0:
        state = {'model': model.state_dict(), 'optimizer': opt.state_dict(), 'epoch': epoch}
        path = f'checkpoint/{epoch}_inat2018_train_dedup_inter_intra_1_in_6_per_class.ckpt'
        print(path)
        torch.save(state, path)
    torch.save(model.state_dict(), args.model_filename)

torch.save(model.state_dict(), args.model_filename)
