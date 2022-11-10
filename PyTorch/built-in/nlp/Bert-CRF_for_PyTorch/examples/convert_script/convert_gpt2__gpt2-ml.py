# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017
# All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd
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
# ==========================================================================

#! -*- coding: utf-8 -*-
# gpt2-ml
# 项目链接(tf版本)：https://github.com/imcaspar/gpt2-ml
# pytorch权重转换和下载：https://github.com/ghosthamlet/gpt2-ml-torch
# 最后经过本脚本转成bert4torch适用的权重

import torch

ckpt_dir = 'F:/Projects/pretrain_ckpt/gpt2/[gpt2-ml_torch_15g]'
ckpt_file = f'{ckpt_dir}/pytorch_model.bin'
output_ckpt_file = f'{ckpt_dir}/bert4torch_pytorch_model.bin'
num_hidden_layers = 48


def convert():
    torch_weights = torch.load(ckpt_file)
    new_weights = {}
    prefix = 'gpt2_ml'

    w = torch_weights['wte.weight']
    new_weights[f'{prefix}.embeddings.word_embeddings.weight'] = w

    w = torch_weights['wpe.weight']
    new_weights[f'{prefix}.embeddings.position_embeddings.weight'] = w

    # embedding layernorm
    w = torch_weights['emb_norm.weight']
    new_weights[f'{prefix}.embeddings.LayerNorm.weight'] = w
    b = torch_weights['emb_norm.bias']
    new_weights[f'{prefix}.embeddings.LayerNorm.bias'] = b

    qkv = ['query', 'key', 'value']
    for i in range(num_hidden_layers):
        prefix_i = f'{prefix}.encoder.layer.%d.' % i

        # q, k, v
        w = torch_weights['h.%s.attn.c_attn.weight' % i]
        ws = torch.chunk(w, 3, dim=1)
        for k, w in zip(qkv, ws):
            name = prefix_i + f'attention.self.{k}.weight'
            new_weights[name] = w.T
        b = torch_weights['h.%s.attn.c_attn.bias' % i]
        bs = torch.chunk(b, 3, dim=0)
        for k, b in zip(qkv, bs):
            name = prefix_i + f'attention.self.{k}.bias'
            new_weights[name] = b

        # hdsz-hdsz的全连接
        w = torch_weights['h.%s.attn.c_proj.weight' % i]
        name = prefix_i + 'attention.output.dense.weight'
        new_weights[name] = w.T
        b = torch_weights['h.%s.attn.c_proj.bias' % i]
        name = prefix_i + 'attention.output.dense.bias'
        new_weights[name] = b

        # layernorm1
        w = torch_weights['h.%s.ln_1.weight' % i]
        name = prefix_i + 'attention.output.LayerNorm.weight'
        new_weights[name] = w
        b = torch_weights['h.%s.ln_1.bias' % i]
        name = prefix_i + 'attention.output.LayerNorm.bias'
        new_weights[name] = b

        # feed forward 第一层
        w = torch_weights['h.%s.mlp.c_fc.weight' % i]
        name = prefix_i + 'intermediate.dense.weight'
        new_weights[name] = w.T
        b = torch_weights['h.%s.mlp.c_fc.bias' % i]
        name = prefix_i + 'intermediate.dense.bias'
        new_weights[name] = b

        # feed forward 第二层
        w = torch_weights['h.%s.mlp.c_proj.weight' % i]
        name = prefix_i + 'output.dense.weight'
        new_weights[name] = w.T
        b = torch_weights['h.%s.mlp.c_proj.bias' % i]
        name = prefix_i + 'output.dense.bias'
        new_weights[name] = b

        # layernorm2
        w = torch_weights['h.%s.ln_2.weight' % i]
        name = prefix_i + 'output.LayerNorm.weight'
        new_weights[name] = w
        b = torch_weights['h.%s.ln_2.bias' % i]
        name = prefix_i + 'output.LayerNorm.bias'
        new_weights[name] = b

        
    torch.save(new_weights, output_ckpt_file)

if __name__ == '__main__':
    convert()

# config文件
'''
{
  "vocab_size": 21130,
  "hidden_size": 1536,
  "attention_probs_dropout_prob": 0.1,
  "hidden_dropout_prob": 0.1,
  "hidden_act": "gelu",
  "initializer_range": 0.014142135623731,
  "intermediate_size": 6144,
  "max_position_embeddings": 1024,
  "num_attention_heads": 24,
  "num_hidden_layers": 48
}
'''