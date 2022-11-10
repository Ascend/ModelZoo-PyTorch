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
# 以文本分类为例，展示部分tips的使用方法
# torchinfo打印参数，自定义metrics, 断点续训，默认Logger和Tensorboard

from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.snippets import sequence_padding, Callback, Logger, Tensorboard, text_segmentate, ListDataset, seed_everything, get_pool_emb
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary
import os

maxlen = 256
batch_size = 16
config_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
choice = 'train'  # train表示训练，infer表示推理

# 固定seed
seed_everything(42)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 加载数据集
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filenames):
        """加载数据，并尽量划分为不超过maxlen的句子
        """
        D = []
        seps, strips = u'\n。！？!?；;，, ', u'；;，, '
        for filename in filenames:
            with open(filename, encoding='utf-8') as f:
                for l in f:
                    text, label = l.strip().split('\t')
                    for t in text_segmentate(text, maxlen - 2, seps, strips):
                        D.append((t, int(label)))
        return D

def collate_fn(batch):
    batch_token_ids, batch_segment_ids, batch_labels = [], [], []
    for text, label in batch:
        token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)
        batch_labels.append([label])

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long, device=device)
    batch_labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
    return [batch_token_ids, batch_segment_ids], batch_labels.flatten()

# 加载数据集
train_dataloader = DataLoader(MyDataset(['F:/Projects/data/corpus/sentence_classification/sentiment/sentiment.train.data']), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
valid_dataloader = DataLoader(MyDataset(['F:/Projects/data/corpus/sentence_classification/sentiment/sentiment.valid.data']), batch_size=batch_size, collate_fn=collate_fn) 
test_dataloader = DataLoader(MyDataset(['F:/Projects/data/corpus/sentence_classification/sentiment/sentiment.test.data']),  batch_size=batch_size, collate_fn=collate_fn) 

# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self, pool_method='cls') -> None:
        super().__init__()
        self.pool_method = pool_method
        self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, with_pool=True)
        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(self.bert.configs['hidden_size'], 2)

    def forward(self, token_ids, segment_ids):
        hidden_states, pooling = self.bert([token_ids, segment_ids])
        pooled_output = get_pool_emb(hidden_states, pooling, token_ids.gt(0).long(), self.pool_method)
        output = self.dropout(pooled_output)
        output = self.dense(output)
        return output
model = Model().to(device)
summary(model, input_data=next(iter(train_dataloader))[0])

def acc(y_pred, y_true):
    y_pred = torch.argmax(y_pred, dim=-1)
    return torch.sum(y_pred.eq(y_true)).item() / y_true.numel()

# 定义使用的loss和optimizer，这里支持自定义
optimizer = optim.Adam(model.parameters(), lr=2e-5)

if os.path.exists('last_model.pt'):
    model.load_weights('last_model.pt')  # 加载模型权重
if os.path.exists('last_steps.pt'):
    model.load_steps_params('last_steps.pt')  # 加载训练进度参数，断点续训使用
if os.path.exists('last_optimizer.pt'):
    state_dict = torch.load('last_optimizer.pt', map_location='cpu')  # 加载优化器，断点续训使用
    optimizer.load_state_dict(state_dict)

model.compile(
    loss=nn.CrossEntropyLoss(),
    optimizer=optimizer,
    metrics={'acc': acc}
)

class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, global_step, epoch, logs=None):
        val_acc = self.evaluate(valid_dataloader)
        test_acc = self.evaluate(test_dataloader)
        logs['val/acc'] = val_acc
        logs['test/acc'] = test_acc
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            # model.save_weights('best_model.pt')
        print(f'val_acc: {val_acc:.5f}, test_acc: {test_acc:.5f}, best_val_acc: {self.best_val_acc:.5f}\n')

        model.save_weights('last_model.pt', prefix=None)  # 保存模型权重
        model.save_steps_params('last_steps.pt')  # 保存训练进度参数，当前的epoch和step，断点续训使用
        torch.save(optimizer.state_dict(), 'last_optimizer.pt')  # 保存优化器，断点续训使用

    # 定义评价函数
    def evaluate(self, data):
        total, right = 0., 0.
        for x_true, y_true in data:
            y_pred = model.predict(x_true).argmax(axis=1)
            total += len(y_true)
            right += (y_true == y_pred).sum().item()
        return right / total

def inference(texts):
    '''单条样本推理
    '''
    for text in texts:
        token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
        token_ids = torch.tensor(token_ids, dtype=torch.long, device=device)[None, :]
        segment_ids = torch.tensor(segment_ids, dtype=torch.long, device=device)[None, :]

        logit = model.predict([token_ids, segment_ids])
        y_pred = torch.argmax(torch.softmax(logit, dim=-1)).cpu().numpy()
        print(text, ' ----> ', y_pred)

if __name__ == '__main__':
    if choice == 'train':
        evaluator = Evaluator()
        model.fit(train_dataloader, epochs=10, steps_per_epoch=100, callbacks=[evaluator, Logger('test.log'), Tensorboard('./')])
    else:
        model.load_weights('best_model.pt')
        inference(['我今天特别开心', '我今天特别生气'])
