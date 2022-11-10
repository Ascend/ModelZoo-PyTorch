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

# tensorflow权重链接：https://github.com/ZhuiyiTechnology/GAU-alpha
# 这里直接映射到GAU_alpha的结构上了，因此不需要mapping
import torch
import tensorflow as tf

tf_path = 'F:/Projects/pretrain_ckpt/gau/[sushen-tf]--chinese_GAU-alpha-char_L-24_H-768/bert_model.ckpt'
torch_state_dict = {}

ts = tf.train.load_variable(tf_path, 'bert/embeddings/word_embeddings')
torch_state_dict['embeddings.word_embeddings.weight'] = torch.from_numpy(ts)
torch_state_dict['mlmDecoder.weight'] = torch.from_numpy(ts)

ts = tf.train.load_variable(tf_path, 'bert/embeddings/token_type_embeddings')
torch_state_dict['embeddings.segment_embeddings.weight'] = torch.from_numpy(ts)


for i in range(24):
    ts = tf.train.load_variable(tf_path, f'GAU_alpha/encoder/layer_{i}/gau/i_dense/kernel')
    torch_state_dict[f'encoderLayer.{i}.gau.i_dense.weight'] = torch.from_numpy(ts.T)

    ts = tf.train.load_variable(tf_path, f'GAU_alpha/encoder/layer_{i}/gau/o_dense/kernel')
    torch_state_dict[f'encoderLayer.{i}.gau.o_dense.weight'] = torch.from_numpy(ts.T)
    
    ts1 = tf.train.load_variable(tf_path, f'GAU_alpha/encoder/layer_{i}/gau/q_scaleoffset/gamma')
    ts2 = tf.train.load_variable(tf_path, f'GAU_alpha/encoder/layer_{i}/gau/k_scaleoffset/gamma')
    ts = torch.stack([torch.from_numpy(ts1), torch.from_numpy(ts2)], dim=0)
    torch_state_dict[f'encoderLayer.{i}.gau.offsetscale.gamma'] = ts

torch.save(torch_state_dict, 'F:/Projects/pretrain_ckpt/gau/[sushen-torch]--chinese_GAU-alpha-char_L-24_H-768/pytorch_model.bin')


# config文件
'''
{
  "hidden_act": "swish",
  "hidden_size": 768,
  "hidden_dropout_prob": 0.1,
  "attention_probs_dropout_prob": 0.1,
  "num_attention_heads": 1,
  "attention_key_size": 128,
  "intermediate_size": 1536,
  "num_hidden_layers": 24,
  "type_vocab_size": 2,
  "vocab_size": 12000
}
'''