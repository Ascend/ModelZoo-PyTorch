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
# 基础测试：mlm测试roformer、roformer_v2模型

from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer
import torch

choice = 'roformer_v2'  # roformer roformer_v2
if choice == 'roformer':
    args_model_path = "F:/Projects/pretrain_ckpt/roformer/[sushen_torch_base]--roformer_v1_base/"
    args_model = 'roformer'
else:
    args_model_path = "F:/Projects/pretrain_ckpt/roformer/[sushen_torch_base]--roformer_v2_char_base/"
    args_model = 'roformer_v2'
    
# 加载模型，请更换成自己的路径
root_model_path = args_model_path
vocab_path = root_model_path + "/vocab.txt"
config_path = root_model_path + "/config.json"
checkpoint_path = root_model_path + '/pytorch_model.bin'


# 建立分词器
tokenizer = Tokenizer(vocab_path, do_lower_case=True)
model = build_transformer_model(config_path, checkpoint_path, model=args_model, with_mlm='softmax')  # 建立模型，加载权重

token_ids, segments_ids = tokenizer.encode("今天M很好，我M去公园玩。")
token_ids[3] = token_ids[8] = tokenizer._token_mask_id
print(''.join(tokenizer.ids_to_tokens(token_ids)))

tokens_ids_tensor = torch.tensor([token_ids])
segment_ids_tensor = torch.tensor([segments_ids])

# 需要传入参数with_mlm
model.eval()
with torch.no_grad():
    _, logits = model([tokens_ids_tensor, segment_ids_tensor])

pred_str = 'Predict: '
for i, logit in enumerate(logits[0]):
    if token_ids[i] == tokenizer._token_mask_id:
        pred_str += tokenizer.id_to_token(torch.argmax(logit, dim=-1).item())
    else:
        pred_str += tokenizer.id_to_token(token_ids[i])
print(pred_str)
