#! -*- coding:utf-8 -*-
# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import json
import stat
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from bert4torch.tokenizers import Tokenizer
from bert4torch.snippets import sequence_padding, text_segmentate, ListDataset, get_pool_emb
from bert4torch.models import build_transformer_model, BaseModel
from sklearn.metrics import classification_report

MAXLEN = 256

class MyDataset(ListDataset):
    @staticmethod
    def load_data(filenames):
        # load dataset
        # split sentence < MAXLEN
        D = []
        for filename in filenames:
            with open(filename, encoding='utf-8') as f:
                for l in f:
                    text, label = l.strip().split('\t')
                    for t in text_segmentate(text, MAXLEN - 2, u'\n。！？!?；;，, ', u'；;，, '):
                        D.append((t, int(label)))
        return D

def collate_fn(batch):
    batch_token_ids, batch_segment_ids, batch_labels = [], [], []
    for text, label in batch:
        token_ids, segment_ids = tokenizer.encode(text, maxlen=MAXLEN)
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)
        batch_labels.append([label])

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long, device=device)
    batch_labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
    return [batch_token_ids, batch_segment_ids], batch_labels.flatten()

class Model(BaseModel):
    def __init__(self, config_path):
        super(Model, self).__init__()
        self.bert = build_transformer_model(config_path, checkpoint_path=None, with_pool=True)
        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(self.bert.configs['hidden_size'], 2)

    def forward(self, token_ids, segment_ids):
        hidden_states, pooling = self.bert([token_ids, segment_ids])
        pooled_output = get_pool_emb(hidden_states, pooling, token_ids.gt(0).long(), 'cls')
        output = self.dropout(pooled_output)
        output = self.dense(output)
        return output


def evaluate(args):
    valid_dataloader = DataLoader(MyDataset([args.input_path]), batch_size=args.batch_size, collate_fn=collate_fn)
    results, true_labels = [], []
    model.eval()
    for x, y in valid_dataloader:
        res = model.predict(x).argmax(axis=1).cpu().numpy()
        for pred, label in zip(res, y):
            results.append(int(pred))
            true_labels.append(int(label))
    
    evaluate_results = classification_report(true_labels,
                                             results,
                                             digits=4)
    print(f'Results: {evaluate_results}')
    return evaluate_results


def parse_arguments():
    parser = argparse.ArgumentParser(description='Sentiment Classification Postprocess.')
    parser.add_argument('-i', '--input_path', type=str, required=True,
                        help='input dir for prediction results')
    parser.add_argument('-o', '--out_path', type=str, required=True,
                        help='save path for evaluation result')
    parser.add_argument('-d', '--dict_path', type=str, required=True,
                        help='label dir for label results')
    parser.add_argument('-c', '--config_path', type=str, required=True,
                        help='config path for export model')
    parser.add_argument('-b', '--batch_size', type=int, default=1,
                        help='batch size of model input')
    parser.add_argument('-k', '--ckpt_path', type=str, default="./best_model.pt",
                        help='result dir for prediction results')
    args = parser.parse_args()
    args.out_path = os.path.abspath(args.out_path)
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    return args


if __name__ == '__main__':
    main_args = parse_arguments()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model(main_args.config_path).to(device)
    model.load_weights(main_args.ckpt_path, strict=False)
    model.eval()

    tokenizer = Tokenizer(main_args.dict_path, do_lower_case=True)
    final_results = evaluate(main_args)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL 
    modes = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(main_args.out_path, flags, modes), 'w') as main_f:
        main_f.write(final_results)
        