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

# t5_pegasus从tf转为bert4torch适配的pytorch版本
# 权重链接：https://github.com/ZhuiyiTechnology/t5-pegasus
import torch
import tensorflow as tf
import json

# small
tf_dir = 'F:/Projects/pretrain_ckpt/t5/[sushen_t5_pegasus_tf_small]--chinese_t5_pegasus_small/'
torch_path = 'F:/Projects/pretrain_ckpt/t5/[sushen_t5_pegasus_torch_small]--chinese_t5_pegasus_small/pytorch_model.bin'

# base:
# tf_dir = 'F:/Projects/pretrain_ckpt/t5/[sushen_t5_pegasus_tf_base]--chinese_t5_pegasus_base/'
# torch_path = 'F:/Projects/pretrain_ckpt/t5/[sushen_t5_pegasus_torch_base]--chinese_t5_pegasus_base/pytorch_model.bin'


tf_path = tf_dir + 'model.ckpt'
with open(tf_dir + 'config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
num_layers = config['num_hidden_layers']
torch_state_dict = {}

mapping = {
'shared/embedding': 'shared.weight',
'encoder/block_000/layer_000/SelfAttention/relative_attention_bias': 'encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight##T', # 自定义标记，##T结尾表示要转置
'encoder/rms_norm/scale': 'encoder.final_layer_norm.weight',
'decoder/block_000/layer_000/SelfAttention/relative_attention_bias': 'decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight##T',
'decoder/rms_norm/scale': 'decoder.final_layer_norm.weight',
'decoder/logits/kernel': 'lm_head.weight##T'
}


for i in range(num_layers):
    i1 = str(i).rjust(3, '0')
    mapping.update({
        f'encoder/block_{i1}/layer_000/SelfAttention/q': f'encoder.block.{i}.layer.0.SelfAttention.q.weight##T',
        f'encoder/block_{i1}/layer_000/SelfAttention/k': f'encoder.block.{i}.layer.0.SelfAttention.k.weight##T',
        f'encoder/block_{i1}/layer_000/SelfAttention/v': f'encoder.block.{i}.layer.0.SelfAttention.v.weight##T',
        f'encoder/block_{i1}/layer_000/SelfAttention/o': f'encoder.block.{i}.layer.0.SelfAttention.o.weight##T',
        f'encoder/block_{i1}/layer_000/rms_norm/scale': f'encoder.block.{i}.layer.0.layer_norm.weight',
        f'encoder/block_{i1}/layer_001/DenseReluDense/wi_0/kernel': f'encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight##T',
        f'encoder/block_{i1}/layer_001/DenseReluDense/wi_1/kernel': f'encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight##T',
        f'encoder/block_{i1}/layer_001/DenseReluDense/wo/kernel': f'encoder.block.{i}.layer.1.DenseReluDense.wo.weight##T',
        f'encoder/block_{i1}/layer_001/rms_norm/scale': f'encoder.block.{i}.layer.1.layer_norm.weight',
        f'decoder/block_{i1}/layer_000/SelfAttention/q': f'decoder.block.{i}.layer.0.SelfAttention.q.weight##T',
        f'decoder/block_{i1}/layer_000/SelfAttention/k': f'decoder.block.{i}.layer.0.SelfAttention.k.weight##T',
        f'decoder/block_{i1}/layer_000/SelfAttention/v': f'decoder.block.{i}.layer.0.SelfAttention.v.weight##T',
        f'decoder/block_{i1}/layer_000/SelfAttention/o': f'decoder.block.{i}.layer.0.SelfAttention.o.weight##T',
        f'decoder/block_{i1}/layer_000/rms_norm/scale': f'decoder.block.{i}.layer.0.layer_norm.weight',
        f'decoder/block_{i1}/layer_001/EncDecAttention/q': f'decoder.block.{i}.layer.1.EncDecAttention.q.weight##T',
        f'decoder/block_{i1}/layer_001/EncDecAttention/k': f'decoder.block.{i}.layer.1.EncDecAttention.k.weight##T',
        f'decoder/block_{i1}/layer_001/EncDecAttention/v': f'decoder.block.{i}.layer.1.EncDecAttention.v.weight##T',
        f'decoder/block_{i1}/layer_001/EncDecAttention/o': f'decoder.block.{i}.layer.1.EncDecAttention.o.weight##T',
        f'decoder/block_{i1}/layer_001/rms_norm/scale': f'decoder.block.{i}.layer.1.layer_norm.weight',
        f'decoder/block_{i1}/layer_002/DenseReluDense/wi_0/kernel': f'decoder.block.{i}.layer.2.DenseReluDense.wi_0.weight##T',
        f'decoder/block_{i1}/layer_002/DenseReluDense/wi_1/kernel': f'decoder.block.{i}.layer.2.DenseReluDense.wi_1.weight##T',
        f'decoder/block_{i1}/layer_002/DenseReluDense/wo/kernel': f'decoder.block.{i}.layer.2.DenseReluDense.wo.weight##T',
        f'decoder/block_{i1}/layer_002/rms_norm/scale': f'decoder.block.{i}.layer.2.layer_norm.weight',
    })

transpose_layers = ['']
for k, v in mapping.items():
    ts = torch.from_numpy(tf.train.load_variable(tf_path, k))
    # if len(ts.shape)==2 and ts.shape[0] == ts.shape[1]:
    #     print(k, v)

    if v.endswith('##T'):
        torch_state_dict[v.rstrip('##T')] = ts.T
    else:
        torch_state_dict[v] = ts

torch.save(torch_state_dict, torch_path)

# config文件
'''
# base版本
{
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 2048,
  "num_attention_heads": 12,
  "attention_head_size": 64,
  "num_hidden_layers": 12,
  "vocab_size": 50000,
  "relative_attention_num_buckets": 32,
  "attention_scale":  false,
  "is_dropout": true
}

# small版本
{
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 512,
  "initializer_range": 0.02,
  "intermediate_size": 1024,
  "num_attention_heads": 6,
  "attention_head_size": 64,
  "num_hidden_layers": 8,
  "vocab_size": 50000,
  "relative_attention_num_buckets": 32,
  "attention_scale":  false,
  "is_dropout": true
}
'''