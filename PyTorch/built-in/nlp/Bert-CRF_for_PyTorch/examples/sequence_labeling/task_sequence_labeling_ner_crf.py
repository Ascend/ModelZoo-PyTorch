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
# bert+crf用来做实体识别
# 数据集：http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz
# [valid_f1]  token_level: 97.06； entity_level: 95.90

import os
import time
import numpy as np
import torch
import torch_npu
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import random

from bert4torch.snippets import sequence_padding, Callback, ListDataset, seed_everything
from bert4torch.optimizers import get_linear_schedule_with_warmup
from bert4torch.layers import CRF
from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    # noinspection PyUnresolvedReferences
    import apex
    from apex import amp
except ImportError:
    amp = None
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--train_epochs", type=int, default=20)
parser.add_argument("--data_path", type=str, default='')
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--prof_type", default='None',
                       	 choices=['TORCH', 'CANN', 'GE', 'None'],
                       	 help="The type of profile.")

args = parser.parse_args()
os.environ['PROFILE_TYPE'] = args.prof_type
if args.prof_type == 'GE':
    os.environ['GE_PROFILING_TO_STD_OUT'] = '1'

distributed = 'WORLD_SIZE' in os.environ

if distributed:
    torch.distributed.init_process_group(backend='hccl', world_size=int(os.environ['WORLD_SIZE']), rank=args.local_rank)
    torch.npu.set_device(args.local_rank)
    device = f'npu:{args.local_rank}'
else:
    torch.npu.set_device(0)
    device = f'npu:0'

print(device)

maxlen = 256
batch_size = int(os.environ["BATCH_SIZE"])
warm_factor = 0.1
categories = ['O', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG']
categories_id2label = {i: k for i, k in enumerate(categories)}
categories_label2id = {k: i for i, k in enumerate(categories)}

# BERT base
config_path = f'{args.data_path}/pretrained_model/config.json'
checkpoint_path = f'{args.data_path}/pretrained_model/pytorch_model.bin'
dict_path = f'{args.data_path}/pretrained_model/vocab.txt'
option = {}
option["MM_BMM_ND_ENABLE"] = "enable"
torch_npu.npu.set_option(option)

# 固定seed
seed_everything(42)

# 加载数据集
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        D = []
        with open(filename, encoding='utf-8') as f:
            f = f.read()
            for l in f.split('\n\n'):
                if not l:
                    continue
                d = ['']
                for i, c in enumerate(l.split('\n')):
                    char, flag = c.split(' ')
                    d[0] += char
                    if flag[0] == 'B':
                        d.append([i, i, flag[2:]])
                    elif flag[0] == 'I':
                        d[-1][1] = i
                D.append(d)
        return D

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

def collate_fn(batch):
    batch_token_ids, batch_labels = [], []
    for d in batch:
        tokens = tokenizer.tokenize(d[0], maxlen=maxlen)
        mapping = tokenizer.rematch(d[0], tokens)
        start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
        end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
        token_ids = tokenizer.tokens_to_ids(tokens)
        labels = np.zeros(len(token_ids))
        for start, end, label in d[1:]:
            if start in start_mapping and end in end_mapping:
                start = start_mapping[start]
                end = end_mapping[end]
                labels[start] = categories_label2id['B-'+label]
                labels[start + 1:end + 1] = categories_label2id['I-'+label]
        batch_token_ids.append(token_ids)
        batch_labels.append(labels)
    # , length=maxlen
    batch_token_ids, length = sequence_padding(batch_token_ids, return_max=True)
    batch_token_ids = torch.tensor(batch_token_ids, dtype=torch.long)
    batch_labels, length2 = sequence_padding(batch_labels, return_max=True)
    batch_labels = torch.tensor(batch_labels, dtype=torch.long)
    return batch_token_ids, batch_labels, length

# 转换数据集
if distributed:
    train_dataset = MyDataset(f'{args.data_path}/china-people-daily-ner-corpus/example.train')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=int(os.environ['WORLD_SIZE']), rank=args.local_rank)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, \
        num_workers=args.workers, shuffle=(train_sampler is None), sampler=train_sampler, collate_fn=collate_fn, drop_last=True, pin_memory=True)
else:
    train_dataloader = DataLoader(MyDataset(f'{args.data_path}/china-people-daily-ner-corpus/example.train'), batch_size=batch_size, \
        num_workers=args.workers, shuffle=True, collate_fn=collate_fn, drop_last=True, pin_memory=True)
valid_dataloader = DataLoader(MyDataset(f'{args.data_path}/china-people-daily-ner-corpus/example.dev'), batch_size=batch_size, collate_fn=collate_fn)

# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, segment_vocab_size=0)
        self.fc = nn.Linear(768, len(categories))  # 包含首尾
        self.crf = CRF(len(categories))

    def forward(self, token_ids):
        sequence_output = self.bert([token_ids])  # [btz, seq_len, hdsz]
        emission_score = self.fc(sequence_output)  # [btz, seq_len, tag_size]
        attention_mask = token_ids.gt(0).long()
        return emission_score, attention_mask

    def predict(self, token_ids):
        self.eval()
        with torch.no_grad():
            emission_score, attention_mask = self.forward(token_ids)
            best_path = self.crf.decode(emission_score, attention_mask)  # [btz, seq_len]
        return best_path

model = Model().to(device)

print(model)
if 'npu' in device:
    optimizer = apex.optimizers.NpuFusedAdam(model.parameters(), lr=2e-5)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", loss_scale='dynamic', combine_grad=True)
else:
    optimizer = optim.Adam(model.parameters(), lr=2e-5)

updates_total = len(train_dataloader) * args.train_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_factor*updates_total, num_training_steps=updates_total)

class Loss(nn.Module):
    def forward(self, outputs, labels, seq_length=None):
        if distributed:
            return model.module.crf(*outputs, labels, seq_length=None)
        else:
            return model.crf(*outputs, labels, seq_length=None)

def acc(y_pred, y_true):
    y_pred = y_pred[0]
    y_pred = torch.argmax(y_pred, dim=-1)
    acc = torch.sum(y_pred.eq(y_true)).item() / y_true.numel()
    return {'acc': acc}

if distributed:
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False)

# 支持多种自定义metrics = ['accuracy', acc, {acc: acc}]均可
if distributed:
    model.module.compile(loss=Loss(), optimizer=optimizer, metrics=acc, use_apex=True, scheduler=scheduler, clip_grad_norm=1.0)
else:
    model.compile(loss=Loss(), optimizer=optimizer, metrics=acc, use_apex=True, scheduler=scheduler, clip_grad_norm=1.0)

def evaluate(data):
    X, Y, Z = 1e-10, 1e-10, 1e-10
    X2, Y2, Z2 = 1e-10, 1e-10, 1e-10
    start = time.time()
    for token_ids, label, seq_length in tqdm(data):
        token_ids = token_ids.to('npu', non_blocking=True)
        label = label.to('npu', non_blocking=True)
        if distributed:
            scores = model.module.predict(token_ids)  # [btz, seq_len]
        else:
            scores = model.predict(token_ids)  # [btz, seq_len]
        attention_mask = label.gt(0)

        # token粒度
        X += (scores.eq(label) * attention_mask).sum().item()
        Y += scores.gt(0).sum().item()
        Z += label.gt(0).sum().item()

        # entity粒度
        entity_pred = trans_entity2tuple(scores)
        entity_true = trans_entity2tuple(label)
        X2 += len(entity_pred.intersection(entity_true))
        Y2 += len(entity_pred)
        Z2 += len(entity_true)
    print(' - eval_time_cost: %.3fs' % (time.time() - start), flush=True)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    f2, precision2, recall2 = 2 * X2 / (Y2 + Z2), X2/ Y2, X2 / Z2
    return f1, precision, recall, f2, precision2, recall2


def trans_entity2tuple(scores):
    '''把tensor转为(样本id, start, end, 实体类型)的tuple用于计算指标
    '''
    batch_entity_ids = set()
    for i, one_samp in enumerate(scores):
        entity_ids = []
        for j, item in enumerate(one_samp):
            flag_tag = categories_id2label[item.item()]
            if flag_tag.startswith('B-'):  # B
                entity_ids.append([i, j, j, flag_tag[2:]])
            elif len(entity_ids) == 0:
                continue
            elif (len(entity_ids[-1]) > 0) and flag_tag.startswith('I-') and (flag_tag[2:]==entity_ids[-1][-1]):  # I
                entity_ids[-1][-2] = j
            elif len(entity_ids[-1]) > 0:
                entity_ids.append([])

        for i in entity_ids:
            if i:
                batch_entity_ids.add(tuple(i))
    return batch_entity_ids

class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, steps, epoch, logs=None):
        f1, precision, recall, f2, precision2, recall2 = evaluate(valid_dataloader)
        if f2 > self.best_val_f1:
            self.best_val_f1 = f2
            if distributed and args.local_rank == 0:
                model.module.save_weights('best_model.pt')
            if not distributed:
                model.save_weights('best_model.pt')
        print(f'[val-token  level] f1: {f1:.5f}, p: {precision:.5f} r: {recall:.5f}')
        print(f'[val-entity level] f1: {f2:.5f}, p: {precision2:.5f} r: {recall2:.5f} best_f1: {self.best_val_f1:.5f}\n')


if __name__ == '__main__':
    torch_npu.npu.set_compile_mode(jit_compile=False)
    evaluator = Evaluator()
    if distributed:
        model.module.fit(train_dataloader, train_sampler, epochs=args.train_epochs, steps_per_epoch=None, callbacks=[evaluator]) #evaluator
    else:
        model.fit(train_dataloader, None, epochs=args.train_epochs, steps_per_epoch=None, callbacks=[evaluator]) #evaluator

else:

    model.load_weights('best_model.pt')