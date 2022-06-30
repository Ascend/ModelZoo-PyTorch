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
"""
Convert weights from https://github.com/google-research/nested-transformer
NOTE: You'll need https://github.com/google/CommonLoopUtils, not included in requirements.txt
"""

import sys

import numpy as np
import torch

from clu import checkpoint
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


arch_depths = {
    'nest_base': [2, 2, 20],
    'nest_small': [2, 2, 20],
    'nest_tiny': [2, 2, 8],
}


def convert_nest(checkpoint_path, arch):
    """
    Expects path to checkpoint which is a dir containing 4 files like in each of these folders
        - https://console.cloud.google.com/storage/browser/gresearch/nest-checkpoints
    `arch` is needed to 
    Returns a state dict that can be used with `torch.nn.Module.load_state_dict`
    Hint: Follow timm.models.nest.Nest.__init__ and 
    https://github.com/google-research/nested-transformer/blob/main/models/nest_net.py
    """
    assert arch in ['nest_base', 'nest_small', 'nest_tiny'], "Your `arch` is not supported"

    flax_dict = checkpoint.load_state_dict(checkpoint_path)['optimizer']['target']
    state_dict = {}

    # Patch embedding
    state_dict['patch_embed.proj.weight'] = torch.tensor(
        flax_dict['PatchEmbedding_0']['Conv_0']['kernel']).permute(3, 2, 0, 1)
    state_dict['patch_embed.proj.bias'] = torch.tensor(flax_dict['PatchEmbedding_0']['Conv_0']['bias'])
    
    # Positional embeddings
    posemb_keys = [k for k in flax_dict.keys() if k.startswith('PositionEmbedding')]
    for i, k in enumerate(posemb_keys):
        state_dict[f'levels.{i}.pos_embed'] = torch.tensor(flax_dict[k]['pos_embedding'])
    
    # Transformer encoders
    depths = arch_depths[arch]
    for level in range(len(depths)):
        for layer in range(depths[level]):
            global_layer_ix = sum(depths[:level]) + layer
            # Norms
            for i in range(2):
                state_dict[f'levels.{level}.transformer_encoder.{layer}.norm{i+1}.weight'] = torch.tensor(
                    flax_dict[f'EncoderNDBlock_{global_layer_ix}'][f'LayerNorm_{i}']['scale'])
                state_dict[f'levels.{level}.transformer_encoder.{layer}.norm{i+1}.bias'] = torch.tensor(
                    flax_dict[f'EncoderNDBlock_{global_layer_ix}'][f'LayerNorm_{i}']['bias'])
            # Attention qkv
            w_q = flax_dict[f'EncoderNDBlock_{global_layer_ix}']['MultiHeadAttention_0']['DenseGeneral_0']['kernel']
            w_kv = flax_dict[f'EncoderNDBlock_{global_layer_ix}']['MultiHeadAttention_0']['DenseGeneral_1']['kernel']
            # Pay attention to dims here (maybe get pen and paper)
            w_kv = np.concatenate(np.split(w_kv, 2, -1), 1)
            w_qkv = np.concatenate([w_q, w_kv], 1)
            state_dict[f'levels.{level}.transformer_encoder.{layer}.attn.qkv.weight'] = torch.tensor(w_qkv).flatten(1).permute(1,0)
            b_q = flax_dict[f'EncoderNDBlock_{global_layer_ix}']['MultiHeadAttention_0']['DenseGeneral_0']['bias']
            b_kv = flax_dict[f'EncoderNDBlock_{global_layer_ix}']['MultiHeadAttention_0']['DenseGeneral_1']['bias']
            # Pay attention to dims here (maybe get pen and paper)
            b_kv = np.concatenate(np.split(b_kv, 2, -1), 0)
            b_qkv = np.concatenate([b_q, b_kv], 0)
            state_dict[f'levels.{level}.transformer_encoder.{layer}.attn.qkv.bias'] = torch.tensor(b_qkv).reshape(-1)
            # Attention proj
            w_proj = flax_dict[f'EncoderNDBlock_{global_layer_ix}']['MultiHeadAttention_0']['proj_kernel']
            w_proj = torch.tensor(w_proj).permute(2, 1, 0).flatten(1)
            state_dict[f'levels.{level}.transformer_encoder.{layer}.attn.proj.weight'] = w_proj
            state_dict[f'levels.{level}.transformer_encoder.{layer}.attn.proj.bias'] = torch.tensor(
                flax_dict[f'EncoderNDBlock_{global_layer_ix}']['MultiHeadAttention_0']['bias'])
            # MLP
            for i in range(2):
                state_dict[f'levels.{level}.transformer_encoder.{layer}.mlp.fc{i+1}.weight'] = torch.tensor(
                    flax_dict[f'EncoderNDBlock_{global_layer_ix}']['MlpBlock_0'][f'Dense_{i}']['kernel']).permute(1, 0)
                state_dict[f'levels.{level}.transformer_encoder.{layer}.mlp.fc{i+1}.bias'] = torch.tensor(
                    flax_dict[f'EncoderNDBlock_{global_layer_ix}']['MlpBlock_0'][f'Dense_{i}']['bias'])

    # Block aggregations (ConvPool)
    for level in range(1, len(depths)):
        # Convs
        state_dict[f'levels.{level}.pool.conv.weight'] = torch.tensor(
            flax_dict[f'ConvPool_{level-1}']['Conv_0']['kernel']).permute(3, 2, 0, 1)
        state_dict[f'levels.{level}.pool.conv.bias'] = torch.tensor(
            flax_dict[f'ConvPool_{level-1}']['Conv_0']['bias'])
        # Norms
        state_dict[f'levels.{level}.pool.norm.weight'] = torch.tensor(
                    flax_dict[f'ConvPool_{level-1}']['LayerNorm_0']['scale'])
        state_dict[f'levels.{level}.pool.norm.bias'] = torch.tensor(
                    flax_dict[f'ConvPool_{level-1}']['LayerNorm_0']['bias'])

    # Final norm
    state_dict[f'norm.weight'] = torch.tensor(flax_dict['LayerNorm_0']['scale'])
    state_dict[f'norm.bias'] = torch.tensor(flax_dict['LayerNorm_0']['bias'])

    # Classifier
    state_dict['head.weight'] = torch.tensor(flax_dict['Dense_0']['kernel']).permute(1, 0)
    state_dict['head.bias'] = torch.tensor(flax_dict['Dense_0']['bias'])

    return state_dict


if __name__ == '__main__':
    variant = sys.argv[1] # base, small, or tiny
    state_dict = convert_nest(f'./nest-{variant[0]}_imagenet', f'nest_{variant}')
    torch.save(state_dict, f'./jx_nest_{variant}.pth')