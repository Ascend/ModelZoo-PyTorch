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

# 模型推理脚本
# cv逐一预测，按照dev的指标加权
from copyreg import pickle
from torch import device
from training import Model, collate_fn
import torch
from torch.utils.data import DataLoader
from bert4torch.snippets import ListDataset
import pandas as pd
from tqdm import tqdm
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 16
def load_data(df):
    """加载数据。"""
    D = list()
    for _, row in df.iterrows():
        text = row['text']
        D.append((text, 0))
    return D

df_test = pd.read_csv('E:/Github/天池新闻分类/data/test_a.csv', sep='\t')
df_test['text'] = df_test['text'].apply(lambda x: x.strip().split())
test_data = load_data(df_test)
dev_dataloader = DataLoader(ListDataset(data=test_data), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 

f1_score = [0.97083, 0.97074, 0.96914, 0.96892, 0.96613]
y_pred_final = 0
for i in range(5):
    model = Model().to(device)
    model.load_weights(f'best_model_fold{i+1}.pt')
    y_pred = []
    for x, _ in tqdm(dev_dataloader, desc=f'evaluate_cv{i}'):
        y_pred.append(model.predict(x).cpu().numpy())
        # if len(y_pred) > 10:
        #     break
    y_pred = np.concatenate(y_pred)
    y_pred_final += y_pred * f1_score[i]
    np.save(f'test_cv{i}_logit.npy', y_pred)

df_test = pd.DataFrame(y_pred_final.argmax(axis=1))
df_test.columns = ['label']
df_test.to_csv('submission.csv', index=False)