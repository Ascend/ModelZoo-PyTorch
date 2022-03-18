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
import torch.nn.functional as F
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


class CIN(nn.Module):
    # Compressed_Interaction_net
    def __init__(self, in_channels, out_channels):
        super(CIN, self).__init__()
        self.conv_1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x0, xl, k):
        '''
        :param x0: 原始输入
        :param xl: 第l层的输入
        :param k: embedding_dim
        :param n_filters: 压缩网络filter的数量
        :return:
        '''
        # print(x0.size())  # torch.Size([2, 26, 8])
        # print(xl.size())  # torch.Size([2, 26, 8])
        # 1. 将x0与xl按照k所在的维度(-1)进行拆分，每个都可以拆成k列
        x0_cols = torch.chunk(x0, k, dim=-1)
        xl_cols = torch.chunk(xl, k, dim=-1)
        assert len(x0_cols) == len(xl_cols), print('error of shape')

        # 2. 遍历k列, 对于x0与xl所在的第i列进行外积计算，存在feature_maps中
        feature_maps = []
        for i in range(k):
            feature_map = torch.matmul(xl_cols[i], x0_cols[i].permute(0, 2, 1))
            # print(feature_map.size())    # torch.Size([2, 26, 26])
            feature_map = feature_map.unsqueeze(dim=-1)
            # print(feature_map.size())    # torch.Size([2, 26, 26, 1])
            feature_maps.append(feature_map)

        feature_maps = torch.cat(feature_maps, -1)
        # print(feature_maps.size())   # torch.Size([2, 26, 26, 8])

        # 3. 压缩网络
        x0_n_feats = x0.size(1)
        xl_n_feats = xl.size(1)

        reshape_feature_maps = feature_maps.view(-1, x0_n_feats * xl_n_feats, k)
        # print(reshape_feature_maps.size())   # torch.Size([2, 676, 8])

        new_feature_maps = self.conv_1(reshape_feature_maps)   # batch_size, n_filter, embed_dim
        # print(new_feature_maps.size())  # # torch.Size([2, 12, 8])
        return new_feature_maps


class xDeepFM(nn.Module):
    def __init__(self, cate_fea_uniques,
                 num_fea_size=0,
                 emb_size=8,
                 hidden_dims=[256, 128],
                 num_classes=1,
                 dropout=[0.2, 0.2]):
        '''
        :param cate_fea_uniques:
        :param num_fea_size: 数字特征  也就是连续特征
        :param emb_size:
        :param hidden_dims:
        :param num_classes:
        :param dropout:
        '''
        super(xDeepFM, self).__init__()
        self.cate_fea_size = len(cate_fea_uniques)
        self.num_fea_size = num_fea_size
        self.n_layers = 3
        self.n_filters = 12
        self.k = emb_size

        # dense特征一阶表示
        if self.num_fea_size != 0:
            self.fm_1st_order_dense = nn.Linear(self.num_fea_size, 1)

        # sparse特征一阶表示
        self.fm_1st_order_sparse_emb = nn.ModuleList([
            nn.Embedding(voc_size, 1) for voc_size in cate_fea_uniques
        ])

        # sparse特征二阶表示
        self.fm_2nd_order_sparse_emb = nn.ModuleList([
            nn.Embedding(voc_size, emb_size) for voc_size in cate_fea_uniques
        ])

        self.compressed_interaction_net = [
            CIN(in_channels=26*26, out_channels=12),
            CIN(in_channels=26*12, out_channels=12),
            CIN(in_channels=26*12, out_channels=12),
        ]
        # DNN部分
        self.dense_linear = nn.Linear(self.num_fea_size, self.cate_fea_size * emb_size)  # # 数值特征的维度变换到FM输出维度一致
        self.relu = nn.ReLU()

        self.all_dims = [self.cate_fea_size * emb_size] + hidden_dims

        for i in range(1, len(self.all_dims)):
            setattr(self, 'linear_' + str(i), nn.Linear(self.all_dims[i-1], self.all_dims[i]))
            setattr(self, 'batchNorm_' + str(i), nn.BatchNorm1d(self.all_dims[i]))
            setattr(self, 'activation_' + str(i), nn.ReLU())
            setattr(self, 'dropout_' + str(i), nn.Dropout(dropout[i-1]))

        self.output = nn.Linear(165, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X_sparse, X_dense=None):
        """
        X_sparse: sparse_feature [batch_size, sparse_feature_num]
        X_dense: dense_feature  [batch_size, dense_feature_num]
        """
        """FM部分"""
        # 一阶  包含sparse_feature和dense_feature的一阶
        fm_1st_sparse_res = [emb(X_sparse[:, i].unsqueeze(1)).view(-1, 1)
                             for i, emb in enumerate(self.fm_1st_order_sparse_emb)]  # sparse特征嵌入成一维度
        fm_1st_sparse_res = torch.cat(fm_1st_sparse_res, dim=1)  # torch.Size([2, 26])
        fm_1st_sparse_res = torch.sum(fm_1st_sparse_res, 1,  keepdim=True)  # [bs, 1] 将sparse_feature通过全连接并相加整成一维度

        if X_dense is not None:
            fm_1st_dense_res = self.fm_1st_order_dense(X_dense)   # 将dense_feature压到一维度
            fm_1st_part = fm_1st_sparse_res + fm_1st_dense_res
        else:
            fm_1st_part = fm_1st_sparse_res   # batch_size, 1
        linear_part = fm_1st_part    # 一阶线性部分

        # sparse的多维嵌入向量
        input_feature_map = [emb(X_sparse[:, i].unsqueeze(1)) for i, emb in enumerate(self.fm_2nd_order_sparse_emb)]
        input_feature_map = torch.cat(input_feature_map, dim=1)  # batch_size, sparse_feature_nums, emb_size
        # print(input_feature_map.size())   # torch.Size([2, 26, 8])

        '''交叉 压缩'''
        x0 = xl = input_feature_map
        cin_layers = []
        pooling_layers = []
        # print(xl.size())   # torch.Size([2, 26, 8])

        for layer in range(self.n_layers):
            xl = self.compressed_interaction_net[layer](x0, xl, self.k)
            # print(xl.size())  # torch.Size([2, 12, 8])
            cin_layers.append(xl)
            # sum pooling
            pooling = torch.sum(xl, dim=-1)
            pooling_layers.append(pooling)

        cin_layers = torch.cat(pooling_layers, dim=-1)
        # print(cin_layers.size())    # torch.Size([2, 36])

        """DNN部分"""
        dnn_out = torch.flatten(input_feature_map, 1)   # [bs, n * emb_size]
        # print(dnn_out.size())   # torch.Size([2, 208])
        # 从sparse_feature_num * emb_size 维度 转为 sparse_feature_num * emb_size 再转为 256
        for i in range(1, len(self.all_dims)):
            dnn_out = getattr(self, 'linear_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'batchNorm_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'activation_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'dropout_' + str(i))(dnn_out)
        # print(dnn_out.size())   # torch.Size([2, 128])

        concat_layers = torch.cat([linear_part, cin_layers, dnn_out], dim=-1)
        # print(concat_layers.size())   # torch.Size([2, 165])
        out = self.output(concat_layers)
        out = self.sigmoid(out)
        return out