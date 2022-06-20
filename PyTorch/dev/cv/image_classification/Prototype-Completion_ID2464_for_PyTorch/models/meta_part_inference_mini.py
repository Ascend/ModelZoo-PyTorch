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
import torch.nn as nn
import math
import pickle
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import os
import random
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

class ProtoComNet(nn.Module):
    def __init__(self, opt, in_dim=1600):
        super(ProtoComNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=in_dim//2),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=in_dim//2, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=in_dim)
        )
        self.aggregator = nn.Sequential(
            nn.Linear(in_features=600+512, out_features=300),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=300, out_features=1)
        )
        with open('./data/mini_imagenet_part_prior.pickle', 'rb') as handle:
            part_prior = pickle.load(handle)
        self.part_prior = part_prior
        edges = np.array(part_prior['edges'])
        n = len(part_prior['wnids'])
        self.adj = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                                 shape=(n, n), dtype='float32')
        self.adj = self.adj.todense()
        self.adj = torch.from_numpy(self.adj).npu()

        train_class_name_file = './data/mini_imagenet_catname2label_train.pickle'
        val_class_name_file = './data/mini_imagenet_catname2label_val.pickle'
        test_class_name_file = './data/mini_imagenet_catname2label_test.pickle'
        with open(train_class_name_file, 'rb') as handle:
            catname2label_train = pickle.load(handle)
        with open(val_class_name_file, 'rb') as handle:
            catname2label_val = pickle.load(handle)
        with open(test_class_name_file, 'rb') as handle:
            catname2label_test = pickle.load(handle)
        self.catname2label = dict(catname2label_train, **catname2label_val)
        self.catname2label = dict(self.catname2label, **catname2label_test)
        self.label2catname = {v: k for k, v in self.catname2label.items()}
        word_vectors = torch.tensor(part_prior['vectors']).npu()
        word_vectors = F.normalize(word_vectors)
        semantic_feature_0 = word_vectors.unsqueeze(dim=1).expand(-1, n, -1)
        semantic_feature_1 = word_vectors.unsqueeze(dim=0).expand(n, -1, -1)
        self.semantic_feature = torch.cat([semantic_feature_0, semantic_feature_1], dim=-1)
        try:
            with open(os.path.join(opt.save_path, "mini_imagenet_metapart_feature.pickle"), 'rb') as handle:
                self.metapart_feature = pickle.load(handle)

            with open(os.path.join(opt.save_path, "mini_imagenet_class_feature.pickle"), 'rb') as handle:
                self.class_feature = pickle.load(handle)
        except:
            print('no found ' + os.path.join(opt.save_path, "mini_imagenet_metapart_feature.pickle")
                  + ' ' + os.path.join(opt.save_path, "mini_imagenet_class_feature.pickle"))
        self.n = n
        self.in_dim = in_dim

    def forward(self, x, y, use_scale=False, is_infer=False):
        if is_infer == False:
            nb = x.shape[0]
            outputs = []
            targets = []
            for i in range(nb):
                input_feature = torch.zeros(self.n, self.in_dim).npu()
                for k, v in self.metapart_feature.items():
                    input_feature[k:k+1, :] = self.reparameterize(v['mean'], v['std'])
                input_feature[self.part_prior['wnids2id'][self.label2catname[y[i].item()]]:
                                   self.part_prior['wnids2id'][self.label2catname[y[i].item()]] + 1, :] = x[i:i+1, :]

                semantic_feature = self.semantic_feature[self.part_prior['wnids2id'][self.label2catname[y[i].item()]]:
                               self.part_prior['wnids2id'][self.label2catname[y[i].item()]]+1, :, :]

                semantic_feature = torch.cat([semantic_feature, x[i:i+1, :].unsqueeze(0).expand(-1, self.n, -1)], dim=-1)
                fuse_adj = self.aggregator(semantic_feature).squeeze(dim=-1)

                fuse_adj = self.adj[self.part_prior['wnids2id'][self.label2catname[y[i].item()]]:
                                     self.part_prior['wnids2id'][self.label2catname[y[i].item()]] + 1, :] * fuse_adj

                eye = 1 - torch.eye(self.adj.shape[0]).type_as(fuse_adj)
                adj = fuse_adj * eye[self.part_prior['wnids2id'][self.label2catname[y[i].item()]]:
                                     self.part_prior['wnids2id'][self.label2catname[y[i].item()]] + 1, :] + torch.eye(
                    self.adj.shape[0]).type_as(fuse_adj)[self.part_prior['wnids2id'][self.label2catname[y[i].item()]]:
                                                         self.part_prior['wnids2id'][
                                                             self.label2catname[y[i].item()]] + 1, :]

                z = self.encoder(input_feature)
                g = torch.mm(adj, z)
                out = self.decoder(g)
                outputs.append(out)
                targets.append(self.class_feature[y[i].item()]['mean'])
            outputs = torch.cat(outputs, dim=0)
            targets = torch.cat(targets, dim=0)
            return outputs, targets
        else:
            nb = x.shape[0]
            outputs = []
            for i in range(nb):
                input_feature = torch.zeros(self.n, self.in_dim).npu()
                for k, v in self.metapart_feature.items():
                    input_feature[k:k + 1, :] = v['mean']
                input_feature[self.part_prior['wnids2id'][self.label2catname[y[i].item()]]:
                              self.part_prior['wnids2id'][self.label2catname[y[i].item()]] + 1, :] = x[i:i + 1, :]

                semantic_feature = self.semantic_feature[self.part_prior['wnids2id'][self.label2catname[y[i].item()]]:
                                                         self.part_prior['wnids2id'][
                                                             self.label2catname[y[i].item()]] + 1, :, :]
                semantic_feature = torch.cat([semantic_feature, x[i:i + 1, :].unsqueeze(0).expand(-1, self.n, -1)],
                                             dim=-1)
                fuse_adj = self.aggregator(semantic_feature).squeeze(dim=-1)
                fuse_adj = self.adj[self.part_prior['wnids2id'][self.label2catname[y[i].item()]]:
                                    self.part_prior['wnids2id'][self.label2catname[y[i].item()]] + 1, :] * fuse_adj
                eye = 1 - torch.eye(self.adj.shape[0]).type_as(fuse_adj)
                adj = fuse_adj * eye[self.part_prior['wnids2id'][self.label2catname[y[i].item()]]:
                                     self.part_prior['wnids2id'][self.label2catname[y[i].item()]] + 1, :] + torch.eye(
                    self.adj.shape[0]).type_as(fuse_adj)[self.part_prior['wnids2id'][self.label2catname[y[i].item()]]:
                                                         self.part_prior['wnids2id'][
                                                             self.label2catname[y[i].item()]] + 1, :]
                z = self.encoder(input_feature)
                out = torch.mm(adj, z)
                out = self.decoder(out)
                outputs.append(out)
            outputs = torch.cat(outputs, dim=0)

            return outputs, None

    def reparameterize(self, mu, var):
        std = var
        eps = torch.randn_like(std)
        return mu + eps*std
