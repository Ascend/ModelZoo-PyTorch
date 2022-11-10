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

import torch
import tensorflow as tf

tf_path = 'E:/Github/天池新闻分类/top1/pre_models/bert_model.ckpt'
torch_state_dict = {}

mapping = {
    'bert/embeddings/word_embeddings': 'bert.embeddings.word_embeddings.weight',
    'bert/embeddings/token_type_embeddings': 'bert.embeddings.token_type_embeddings.weight',
    'bert/embeddings/position_embeddings': 'bert.embeddings.position_embeddings.weight',
    'bert/embeddings/LayerNorm/beta': 'bert.embeddings.LayerNorm.bias',
    'bert/embeddings/LayerNorm/gamma': 'bert.embeddings.LayerNorm.weight',
    # 'bert/pooler/dense/kernel': 'bert.pooler.dense.weight',
    # 'bert/pooler/dense/bias': 'bert.pooler.dense.bias',
    # 'cls/seq_relationship/output_weights': 'cls.seq_relationship.weight',
    # 'cls/seq_relationship/output_bias': 'cls.seq_relationship.bias',
    'cls/predictions/transform/dense/kernel': 'cls.predictions.transform.dense.weight##T',
    'cls/predictions/transform/dense/bias': 'cls.predictions.transform.dense.bias',
    'cls/predictions/transform/LayerNorm/beta': 'cls.predictions.transform.LayerNorm.bias',
    'cls/predictions/transform/LayerNorm/gamma': 'cls.predictions.transform.LayerNorm.weight',
    'cls/predictions/output_bias': 'cls.predictions.bias',

}


for i in range(12):
    prefix = 'bert/encoder/layer_%d/' % i
    prefix_i = f'bert.encoder.layer.%d.' % i
    mapping.update({
        prefix + 'attention/self/query/kernel': prefix_i + 'attention.self.query.weight##T',
        prefix + 'attention/self/query/bias': prefix_i + 'attention.self.query.bias',
        prefix + 'attention/self/key/kernel': prefix_i + 'attention.self.key.weight##T',
        prefix + 'attention/self/key/bias': prefix_i + 'attention.self.key.bias',
        prefix + 'attention/self/value/kernel': prefix_i + 'attention.self.value.weight##T',
        prefix + 'attention/self/value/bias': prefix_i + 'attention.self.value.bias',
        prefix + 'attention/output/dense/kernel': prefix_i + 'attention.output.dense.weight##T',
        prefix + 'attention/output/dense/bias': prefix_i + 'attention.output.dense.bias',
        prefix + 'attention/output/LayerNorm/beta': prefix_i + 'attention.output.LayerNorm.bias',
        prefix + 'attention/output/LayerNorm/gamma': prefix_i + 'attention.output.LayerNorm.weight',
        prefix + 'intermediate/dense/kernel': prefix_i + 'intermediate.dense.weight##T',
        prefix + 'intermediate/dense/bias': prefix_i + 'intermediate.dense.bias',
        prefix + 'output/dense/kernel': prefix_i + 'output.dense.weight##T',
        prefix + 'output/dense/bias': prefix_i + 'output.dense.bias',
        prefix + 'output/LayerNorm/beta': prefix_i + 'output.LayerNorm.bias',
        prefix + 'output/LayerNorm/gamma': prefix_i + 'output.LayerNorm.weight',
    })

for old_key, new_key in mapping.items():
    try:
        ts = tf.train.load_variable(tf_path, old_key)
        if new_key.endswith('##T'):
            torch_state_dict[new_key.rstrip('##T')] = torch.from_numpy(ts).T
        else:
            torch_state_dict[new_key] = torch.from_numpy(ts)
    except:
        print('Missing ', old_key)
torch.save(torch_state_dict, 'E:/Github/天池新闻分类/top1/pre_models/pytorch_model.bin')
