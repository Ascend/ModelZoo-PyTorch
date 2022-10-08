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
# Copyright (c) Runpei Dong, ArChip Lab.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
import os
import torch
if torch.__version__ >= "1.8":
    import torch_npu
import time
from torch import nn
import numpy as np
from config import set_args
import pandas as pd
from torch import optim
from model import AutoInt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


def evaluate_model(model):
    model.eval()
    with torch.no_grad():
        valid_labels, valid_preds = [], []
        for step, x in tqdm(enumerate(valid_loader)):
            cat_fea, num_fea, label = x[0], x[1], x[2]
            if torch.npu.is_available():
                cat_fea, num_fea, label = cat_fea.npu(), num_fea.npu(), label.npu()
            logits = model(cat_fea, num_fea)
            logits = logits.view(-1).data.cpu().numpy().tolist()
            valid_preds.extend(logits)
            valid_labels.extend(label.cpu().numpy().tolist())
        cur_auc = roc_auc_score(valid_labels, valid_preds)
        return cur_auc


def train_model(model):
    # 指定多gpu运行
    if torch.npu.is_available():
        model.npu()

    if torch.npu.device_count() > 1:
        args.n_gpu = torch.npu.device_count()
        print("Let's use", torch.npu.device_count(), "GPUs!")
        # 就这一行
        model = model

    loss_fct = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.train_batch_size, gamma=0.8)

    best_auc = 0.0
    for epoch in range(args.Epochs):
        model.train()
        train_loss_sum = 0.0
        start_time = time.time()
        for step, x in enumerate(train_loader):
            cat_fea, num_fea, label = x[0], x[1], x[2]
            if torch.npu.is_available():
                cat_fea, num_fea, label = cat_fea.npu(), num_fea.npu(), label.npu()
            pred = model(cat_fea, num_fea)
            # print(pred.size())  # torch.Size([2, 1])
            pred = pred.view(-1)
            loss = loss_fct(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.cpu().item()
            if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                print("Epoch {:04d} | Step {:04d} / {} | Loss {:.4f} | Time {:.4f}".format(
                    epoch+1, step+1, len(train_loader), train_loss_sum/(step+1), time.time() - start_time))
        scheduler.step()
        cur_auc = evaluate_model(model)
        if cur_auc > best_auc:
            best_auc = cur_auc
            os.makedirs('./save_model', exist_ok=True)
            torch.save(model.state_dict(), './save_model/autoint.bin')


if __name__ == '__main__':
    args = set_args()
    data = pd.read_csv(args.data_path)
    # print(data.shape)   # (600000, 40)

    # 数据预处理
    dense_features = [f for f in data.columns.tolist() if f[0] == "I"]
    sparse_features = [f for f in data.columns.tolist() if f[0] == "C"]

    # 处理缺失值
    data[sparse_features] = data[sparse_features].fillna('-10086',)
    data[dense_features] = data[dense_features].fillna(0,)
    target = ['label']

    # 将类别数据转为数字
    for feat in tqdm(sparse_features):
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    # 将连续值归一化
    for feat in tqdm(dense_features):
        mean = data[feat].mean()
        std = data[feat].std()
        data[feat] = (data[feat] - mean) / (std + 1e-12)
    # print(data.shape)
    # print(data.head())

    train, valid = train_test_split(data, test_size=0.1, random_state=42)
    # print(train.shape)   # (540000, 40)
    # print(valid.shape)   # (60000, 40)
    train_dataset = TensorDataset(torch.LongTensor(train[sparse_features].values),
                                  torch.FloatTensor(train[dense_features].values),
                                  torch.FloatTensor(train['label'].values))
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True)

    valid_dataset = TensorDataset(torch.LongTensor(valid[sparse_features].values),
                                  torch.FloatTensor(valid[dense_features].values),
                                  torch.FloatTensor(valid['label'].values))
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.eval_batch_size, shuffle=False)

    cat_fea_unique = [data[f].nunique() for f in sparse_features]

    model = AutoInt(cat_fea_unique, num_fea_size=len(dense_features))

    train_model(model)
