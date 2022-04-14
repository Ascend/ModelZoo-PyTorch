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
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

class AUGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias = True):
        super(AUGRUCell, self).__init__()

        in_dim = input_dim + hidden_dim
        self.reset_gate = nn.Sequential( nn.Linear( in_dim, hidden_dim, bias = bias), nn.Sigmoid())
        self.update_gate = nn.Sequential( nn.Linear( in_dim, hidden_dim, bias = bias), nn.Sigmoid())
        self.h_hat_gate = nn.Sequential( nn.Linear( in_dim, hidden_dim, bias = bias), nn.Tanh())


    def forward(self, X, h_prev, attention_score):
        temp_input = torch.cat( [ h_prev, X ] , dim = -1)
        r = self.reset_gate( temp_input)
        u = self.update_gate( temp_input)

        h_hat = self.h_hat_gate( torch.cat( [ h_prev * r, X], dim = -1) )

        u = attention_score.unsqueeze(1) * u
        h_cur = (1. - u) * h_prev + u * h_hat

        return h_cur


class DynamicGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rnn_cell = AUGRUCell( input_dim, hidden_dim, bias = True)

    def forward(self, X, attenion_scores , h0 = None ):
        B, T, D = X.shape
        H = self.hidden_dim
        
        output = torch.zeros( B, T, H ).type( X.type() )
        h_prev = torch.zeros( B, H ).type( X.type() ) if h0 == None else h0
        for t in range( T): 
            h_prev = output[ : , t, :] = self.rnn_cell( X[ : , t, :], h_prev, attenion_scores[ :, t] )
        return output
