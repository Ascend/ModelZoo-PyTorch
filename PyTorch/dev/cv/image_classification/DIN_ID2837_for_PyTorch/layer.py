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
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

class MLP( nn.Module):
    
    def __init__(self, input_dimension, hidden_size , target_dimension = 1, activation_layer = 'LeakyReLU'):
        super().__init__()

        Activation = nn.LeakyReLU

        # if activation_layer == 'DICE': pass
        # elif activation_layer == 'LeakyReLU': pass

        def _dense( in_dim, out_dim, bias = False):
            return nn.Sequential(
                nn.Linear( in_dim, out_dim, bias = bias),
                nn.BatchNorm1d( out_dim),
                Activation( 0.1 ))

        dimension_pair = [input_dimension] + hidden_size
        layers = [ _dense( dimension_pair[i], dimension_pair[i+1]) for i in range( len( hidden_size))]

        layers.append( nn.Linear( hidden_size[-1], target_dimension))
        layers.insert( 0, nn.BatchNorm1d( input_dimension) )

        self.model = nn.Sequential( *layers )
    
    def forward( self, X): return self.model( X)


class InputEmbedding( nn.Module):

    def __init__(self, n_uid, n_mid, n_cid, embedding_dim ):
        super().__init__()
        self.user_embedding_unit = nn.Embedding( n_uid, embedding_dim)
        self.material_embedding_unit = nn.Embedding( n_mid, embedding_dim)
        self.category_embedding_unit = nn.Embedding( n_cid, embedding_dim)

    def forward( self, user, material, category, material_historical, category_historical, 
                                       material_historical_neg, category_historical_neg, neg_smaple = False ):

        user_embedding = self.user_embedding_unit( user)

        material_embedding = self.material_embedding_unit( material)
        material_historical_embedding = self.material_embedding_unit( material_historical)

        category_embedding = self.category_embedding_unit( category)
        category_historical_embedding = self.category_embedding_unit( category_historical)

        material_historical_neg_embedding = self.material_embedding_unit( material_historical_neg) if neg_smaple else None  
        category_historical_neg_embedding = self.category_embedding_unit( category_historical_neg) if neg_smaple else None

        ans = [ user_embedding, material_historical_embedding, category_historical_embedding, 
            material_embedding, category_embedding, material_historical_neg_embedding, category_historical_neg_embedding ]
        return tuple( map( lambda x: x.squeeze() if x != None else None , ans) )



class AttentionLayer( nn.Module):

    def __init__(self, embedding_dim, hidden_size, activation_layer = 'sigmoid'):
        super().__init__()

        Activation = nn.Sigmoid
        if activation_layer == 'Dice': pass
        
        def _dense( in_dim, out_dim):
            return nn.Sequential( nn.Linear( in_dim, out_dim), Activation() )
        
        dimension_pair = [embedding_dim * 8] + hidden_size
        layers = [ _dense( dimension_pair[i], dimension_pair[i+1]) for i in range( len( hidden_size))]
        layers.append( nn.Linear( hidden_size[-1], 1) )
        self.model = nn.Sequential( *layers)
    
    def forward( self, query, fact, mask, return_scores = False):
        B, T, D = fact.shape
        
        query = torch.ones((B, T, 1), device=f'npu:{NPU_CALCULATE_DEVICE}', dtype=torch.float16) * query.view( (B, 1, D)) 
        # query = query.view(-1).expand( T, -1).view( T, B, D).permute( 1, 0, 2)

        combination = torch.cat( [ fact, query, fact * query, query - fact ], dim = 2)

        scores = self.model( combination).squeeze()
        scores = torch.where( mask == 1, scores, torch.ones_like( scores) * ( -2 ** 31 ) )

        scores = ( scores.softmax( dim = -1) * mask ).view( (B , 1, T))

        if return_scores: return scores.squeeze()
        return torch.matmul( scores, fact).squeeze()