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
import torch
import torch.nn as nn
import torch.nn.functional as F

from layer import *
from rnn import *
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

class DIN( nn.Module):
    def __init__(self, n_uid, n_mid, n_cid, embedding_dim, ):
        super().__init__()

        self.embedding_layer = InputEmbedding( n_uid, n_mid, n_cid, embedding_dim )
        self.attention_layer = AttentionLayer( embedding_dim, hidden_size = [ 80, 40], activation_layer='sigmoid')
        # self.output_layer = MLP( embedding_dim * 9, [ 200, 80], 1, 'ReLU')
        self.output_layer = MLP( embedding_dim * 7, [ 200, 80], 1, 'ReLU')

    def forward( self, data, neg_sample = False):
                            
        user, material_historical, category_historical, mask, sequential_length , material, category, \
            material_historical_neg, category_historical_neg = data
        
        user_embedding, material_historical_embedding, category_historical_embedding, \
            material_embedding, category_embedding, material_historical_neg_embedding, category_historical_neg_embedding = \
        self.embedding_layer(  user, material, category, material_historical, category_historical, material_historical_neg, category_historical_neg, neg_sample)

        item_embedding = torch.cat( [ material_embedding, category_embedding], dim = 1)
        item_historical_embedding = torch.cat( [ material_historical_embedding, category_historical_embedding], dim = 2 )

        item_historical_embedding_sum = torch.matmul( mask.unsqueeze( dim = 1), item_historical_embedding).squeeze() / sequential_length.unsqueeze( dim = 1)


        attention_feature = self.attention_layer( item_embedding, item_historical_embedding, mask)

        # combination = torch.cat( [ user_embedding, item_embedding, item_historical_embedding_sum, attention_feature ], dim = 1)
        combination = torch.cat( [ user_embedding, item_embedding, item_historical_embedding_sum, 
                                    # item_embedding * item_historical_embedding_sum, 
                                    attention_feature ], dim = 1)

        scores = self.output_layer( combination)

        return scores.squeeze()

class DIEN( nn.Module):
    def __init__(self, n_uid, n_mid, n_cid, embedding_dim):
        super().__init__()

        self.embedding_layer = InputEmbedding( n_uid, n_mid, n_cid, embedding_dim )
        self.gru_based_layer = nn.GRU( embedding_dim * 2 , embedding_dim * 2, batch_first = True)
        self.attention_layer = AttentionLayer( embedding_dim, hidden_size = [ 80, 40], activation_layer='sigmoid')
        self.gru_customized_layer = DynamicGRU( embedding_dim * 2, embedding_dim * 2)

        self.output_layer = MLP( embedding_dim * 9, [ 200, 80], 1, 'ReLU')
        # self.output_layer = MLP( embedding_dim * 9, [ 200, 80], 1, 'ReLU')

    def forward( self, data, neg_sample = False):
                            
        user, material_historical, category_historical, mask, sequential_length , material, category, \
            material_historical_neg, category_historical_neg = data
        
        user_embedding, material_historical_embedding, category_historical_embedding, \
            material_embedding, category_embedding, material_historical_neg_embedding, category_historical_neg_embedding = \
        self.embedding_layer(  user, material, category, material_historical, category_historical, material_historical_neg, category_historical_neg, neg_sample)

        item_embedding = torch.cat( [ material_embedding, category_embedding], dim = 1)
        item_historical_embedding = torch.cat( [ material_historical_embedding, category_historical_embedding], dim = 2 )

        item_historical_embedding_sum = torch.matmul( mask.unsqueeze( dim = 1), item_historical_embedding).squeeze() / sequential_length.unsqueeze( dim = 1)

        output_based_gru, _ = self.gru_based_layer( item_historical_embedding)
        attention_scores = self.attention_layer( item_embedding, output_based_gru, mask, return_scores = True)
        output_customized_gru = self.gru_customized_layer( output_based_gru, attention_scores)

        attention_feature = output_customized_gru[  range( len( sequential_length)), sequential_length - 1]

        combination = torch.cat( [ user_embedding, item_embedding, item_historical_embedding_sum, item_embedding * item_historical_embedding_sum, attention_feature ], dim = 1)

        scores = self.output_layer( combination)

        return scores.squeeze()