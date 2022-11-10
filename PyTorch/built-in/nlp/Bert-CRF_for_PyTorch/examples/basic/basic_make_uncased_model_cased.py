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
# 通过简单修改词表，使得不区分大小写的模型有区分大小写的能力
# 基本思路：将英文单词大写化后添加到词表中，并修改模型Embedding层

from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer, load_vocab
import torch

root_model_path = "F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12"
vocab_path = root_model_path + "/vocab.txt"
config_path = root_model_path + "/bert_config.json"
checkpoint_path = root_model_path + '/pytorch_model.bin'


token_dict = load_vocab(vocab_path)
new_token_dict = token_dict.copy()
compound_tokens = []

for t, i in sorted(token_dict.items(), key=lambda s: s[1]):
    # 这里主要考虑两种情况：1、首字母大写；2、整个单词大写。
    # Python2下，新增了5594个token；Python3下，新增了5596个token。
    tokens = []
    if t.isalpha():
        tokens.extend([t[:1].upper() + t[1:], t.upper()])
    elif t[:2] == '##' and t[2:].isalpha():
        tokens.append(t.upper())
    for token in tokens:
        if token not in new_token_dict:
            compound_tokens.append([i])
            new_token_dict[token] = len(new_token_dict)

tokenizer = Tokenizer(new_token_dict, do_lower_case=False)

model = build_transformer_model(
    config_path,
    checkpoint_path,
    compound_tokens=compound_tokens,  # 增加新token，用旧token平均来初始化
)

text = u'Welcome to BEIJING.'
tokens = tokenizer.tokenize(text)
print(tokens)
"""
输出：['[CLS]', u'Welcome', u'to', u'BE', u'##I', u'##JING', u'.', '[SEP]']
"""

token_ids, segment_ids = tokenizer.encode(text)
token_ids, segment_ids = torch.tensor([token_ids]), torch.tensor([segment_ids])
model.eval()
with torch.no_grad():
  print(model([token_ids, segment_ids])[0])
"""
输出：
[[[-1.4999904e-01  1.9651388e-01 -1.7924258e-01 ...  7.8269649e-01
    2.2241375e-01  1.1325148e-01]
  [-4.5268752e-02  5.5090344e-01  7.4699545e-01 ... -4.7773960e-01
   -1.7562288e-01  4.1265407e-01]
  [ 7.0158571e-02  1.7816302e-01  3.6949167e-01 ...  9.6258509e-01
   -8.4678203e-01  6.3776302e-01]
  ...
  [ 9.3637377e-01  3.0232478e-02  8.1411439e-01 ...  7.9186147e-01
    7.5704646e-01 -8.3475001e-04]
  [ 2.3699696e-01  2.9953337e-01  8.1962071e-02 ... -1.3776925e-01
    3.8681498e-01  3.2553676e-01]
  [ 1.9728680e-01  7.7782705e-02  5.2951699e-01 ...  8.9622810e-02
   -2.3932748e-02  6.9600858e-02]]]
"""
