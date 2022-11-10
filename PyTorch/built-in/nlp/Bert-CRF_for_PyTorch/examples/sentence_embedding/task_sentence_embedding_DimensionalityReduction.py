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

#! -*- coding:utf-8 -*-
# 利用pca压缩句向量
# 从768维压缩到128维，指标从81.82下降到80.10

from task_sentence_embedding_sup_CosineMSELoss import model, train_dataloader, Model, device, valid_dataloader, evaluate
from bert4torch.snippets import get_pool_emb
from sklearn.decomposition import PCA
import numpy as np
import torch
import torch.nn as nn

new_dimension = 128  # 压缩到的维度

train_embeddings = []
for token_ids_list, labels in train_dataloader:
    for token_ids in token_ids_list:
        train_embeddings.append(model.encode(token_ids))
    # if len(train_embeddings) >= 20:
    #     break
train_embeddings = torch.cat(train_embeddings, dim=0).cpu().numpy()
print('train_embeddings done, start pca training...')

pca = PCA(n_components=new_dimension)
pca.fit(train_embeddings)
pca_comp = np.asarray(pca.components_)
print('PCA training done...')

# 定义bert上的模型结构
class NewModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = nn.Linear(768, new_dimension, bias=False)
        self.dense.weight = torch.nn.Parameter(torch.tensor(pca_comp, device=device))

    def encode(self, token_ids):
        self.eval()
        with torch.no_grad():
            hidden_state, pool_cls = self.bert([token_ids])
            attention_mask = token_ids.gt(0).long()
            output = get_pool_emb(hidden_state, pool_cls, attention_mask, self.pool_method)
            output = self.dense(output)
        return output

new_model = NewModel().to(device)
new_model.load_weights('best_model.pt', strict=False)

print('Start evaludating...')
val_consine = evaluate(new_model, valid_dataloader)
print(f'val_consine: {val_consine:.5f}\n')