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
# Copyright (c) Runpei Dong, ArChip Lab.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torch import nn
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


class Cross_Layer(nn.Module):
    def __init__(self):
        super(Cross_Layer, self).__init__()
        pass

    def forward(self, x0, x1):
        '''
        :param x0: 特征进行embedding
        :param x1: 第1层的输出结果
        :return:
        '''
        embed_dim = x1.size(-1)
        w = torch.randn(size=(embed_dim,), dtype=torch.float)
        b = torch.zeros(size=(embed_dim,))
        res = torch.matmul(x1.view(-1, 1, embed_dim), w)   # batch_size, 1
        cross = x0 * res
        return cross + b + x1


class DCN(nn.Module):
    def __init__(self, cate_fea_uniques,
                 num_fea_size=0,
                 emb_size=8,
                 hidden_dims=[256, 128],
                 dropout=[0.2, 0.2],
                 num_layer=2):
        '''
        :param cate_fea_uniques:
        :param num_fea_size: 数字特征  也就是连续特征
        :param emb_size:
        :param hidden_dims:
        :param num_classes:
        :param dropout:
        '''
        super(DCN, self).__init__()
        self.cate_fea_size = len(cate_fea_uniques)
        self.num_fea_size = num_fea_size
        self.num_layers = num_layers

        # sparse特征嵌入
        self.sparse_embedding = nn.ModuleList([
            nn.Embedding(voc_size, emb_size) for voc_size in cate_fea_uniques
        ])

        self.cross_layer = Cross_Layer()

        self.all_dims = [self.cate_fea_size * emb_size+self.num_fea_size] + hidden_dims

        for i in range(1, len(self.all_dims)):
            setattr(self, 'linear_' + str(i), nn.Linear(self.all_dims[i-1], self.all_dims[i]))
            setattr(self, 'batchNorm_' + str(i), nn.BatchNorm1d(self.all_dims[i]))
            setattr(self, 'activation_' + str(i), nn.ReLU())
            setattr(self, 'dropout_' + str(i), nn.Dropout(dropout[i-1]))

        self.output = nn.Linear(hidden_dims[-1] + self.cate_fea_size * emb_size+self.num_fea_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X_sparse, X_dense=None):
        """
        X_sparse: sparse_feature [batch_size, sparse_feature_num]
        X_dense: dense_feature  [batch_size, dense_feature_num]
        """
        batch_size = X_sparse.size(0)

        sparse_feature_embed = [emb(X_sparse[:, i].unsqueeze(1)) for i, emb in enumerate(self.sparse_embedding)]
        sparse_feature_embed = torch.cat(sparse_feature_embed, dim=1)  # batch_size, sparse_feature_nums, emb_size
        # print(sparse_feature_embed.size())   # torch.Size([2, 26, 8])
        concat_sparse_inputs = sparse_feature_embed.view(batch_size, -1)   # torch.Size([2, 208])

        # 直接将dense_feature与sparse_feature的embedding进行concat
        embed_input = torch.cat((concat_sparse_inputs, X_dense), dim=-1)  # torch.Size([2, 221])

        # 特征交叉
        x1 = x0 = embed_input
        for i in range(self.num_layers):
            x1 = self.cross_layer(x0, x1)
        cross_layer_output = x1

        # DNN部分
        dnn_input = embed_input
        for i in range(1, len(self.all_dims)):
            dnn_input = getattr(self, 'linear_' + str(i))(dnn_input)
            dnn_input = getattr(self, 'batchNorm_' + str(i))(dnn_input)
            dnn_input = getattr(self, 'activation_' + str(i))(dnn_input)
            dnn_input = getattr(self, 'dropout_' + str(i))(dnn_input)
        # print(dnn_input.size())   # torch.Size([2, 128])

        # 将cross_layer_output与 dnn的输出拼接
        stack_output = torch.cat((cross_layer_output, dnn_input), dim=-1)
        # print(stack_output.size())   # torch.Size([2, 349])

        out = self.output(stack_output)   # batch_size, 1
        out = self.sigmoid(out)
        return out