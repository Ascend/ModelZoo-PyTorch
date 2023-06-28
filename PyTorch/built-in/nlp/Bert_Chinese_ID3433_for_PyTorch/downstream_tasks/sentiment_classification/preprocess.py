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
import argparse
import stat
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from bert4torch.snippets import sequence_padding, text_segmentate, ListDataset
from bert4torch.tokenizers import Tokenizer

MAXLEN = 256

class MyDataset(ListDataset):
    def pad(self):
        # to pass pylint
        # add a pad function
        return self
    
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
        token_ids = [np.dtype('int64').type(x) for x in token_ids]
        segment_ids = [np.dtype('int64').type(x) for x in segment_ids]
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)
        batch_labels.append([label])

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids, MAXLEN), dtype=torch.long, device=device)
    batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids, MAXLEN), dtype=torch.long, device=device)
    batch_labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
    return [batch_token_ids, batch_segment_ids], batch_labels.flatten()


def dump_data(data_loader, save_dir):
    input_data_dir = os.path.join(save_dir, "input_data")
    input_data_token = os.path.join(input_data_dir, "token")
    input_data_segment = os.path.join(input_data_dir, "segment")
    os.makedirs(input_data_dir)
    os.makedirs(input_data_token)
    os.makedirs(input_data_segment)
    label_dir = os.path.join(save_dir, "label")
    os.makedirs(label_dir)
    for idx, data in tqdm(enumerate(data_loader)):
        inputs, labels = data
        token_ids, segment_ids = inputs
        token_ids_path = os.path.join(input_data_token, "{}.bin".format(idx))
        segment_ids_path = os.path.join(input_data_segment, "{}.bin".format(idx))
        label_path = os.path.join(label_dir, "{}.txt".format(idx))
        token_ids.detach().cpu().numpy().tofile(token_ids_path)
        segment_ids.detach().cpu().numpy().tofile(segment_ids_path)
        labels = labels.tolist()

        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL 
        modes = stat.S_IWUSR | stat.S_IRUSR
        with os.fdopen(os.open(label_path, flags, modes), 'w') as f:
            for label in labels:
                f.write(str(label) + '\n')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Sentiment Classification.')
    parser.add_argument('-i', '--input_path', type=str, required=True,
                        help='input dataset path')
    parser.add_argument('-o', '--out_dir', type=str, required=True,
                        help='save dir for preprocessed data')
    parser.add_argument('-d', '--dict_path', type=str, required=True,
                        help='vocab dict path for dataset')
    parser.add_argument('-s', '--seq_len', type=int, default=256,
                        help='max sequence length for output model')
    parser.add_argument('-b', '--batch_size', type=int, default=1,
                        help='batch size of dataloader')
    args = parser.parse_args()
    os.makedirs(args.out_dir)
    return args


if __name__ == '__main__':
    main_args = parse_arguments()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = Tokenizer(main_args.dict_path, do_lower_case=True)
    valid_dataloader = DataLoader(MyDataset([main_args.input_path]),
                                  batch_size=main_args.batch_size,
                                  collate_fn=collate_fn,
                                  shuffle=False)
    dump_data(valid_dataloader, main_args.out_dir)
