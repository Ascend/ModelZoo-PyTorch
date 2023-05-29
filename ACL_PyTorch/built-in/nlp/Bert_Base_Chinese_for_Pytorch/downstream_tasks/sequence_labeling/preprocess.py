# Copyright 2022 Huawei Technologies Co., Ltd
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
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from bert4torch.snippets import sequence_padding, ListDataset
from bert4torch.tokenizers import Tokenizer


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


def collate_fn(batch):
    batch_token_ids, batch_labels = [], []
    for d in batch:
        tokens = tokenizer.tokenize(d[0], maxlen=args.seq_len)
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
    batch_token_ids = torch.tensor(
        sequence_padding(batch_token_ids, length=args.seq_len),
        dtype=torch.long,
        device='cpu')
    batch_labels = torch.tensor(
        sequence_padding(batch_labels, length=args.seq_len),
        dtype=torch.long,
        device='cpu')
    return batch_token_ids, batch_labels


def dump_data(data_loader, save_dir):
    input_data_dir = os.path.join(save_dir, "input_data")
    os.makedirs(input_data_dir, exist_ok=True)
    label_dir = os.path.join(save_dir, "label")
    os.makedirs(label_dir, exist_ok=True)
    for idx, data in tqdm(enumerate(data_loader)):
        token_ids, labels = data
        data_path = os.path.join(input_data_dir, "{}.npy".format(idx))
        label_path = os.path.join(label_dir, "{}.npy".format(idx))
        np.save(data_path, token_ids.detach().numpy())
        np.save(label_path, labels.detach().numpy())


def dump_data_ranks(data_loader, save_dir, rank_list, batch_size):
    if isinstance(rank_list, str):
        rank_list = sorted([int(_) for _ in rank_list.split(",")])
    input_data_dir = os.path.join(save_dir, "input_data")
    label_dir = os.path.join(save_dir, "label")
    os.makedirs(input_data_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    # basic function
    def get_max_valid_length(data):
        valid_num = np.where(data > 0)[1][-1] + 1
        return np.max(valid_num)

    def get_rank_num(valid_num):
        _ranks = np.array(rank_list)
        return _ranks[_ranks >= valid_num][0]

    def pad_data(data, constant_values=(0)):
        return np.pad(data, ((0, batch_size-data.shape[0]), (0, 0)),
                      "constant", constant_values=constant_values)

    def dump(token_ids, labels, idx):
        valid_length = get_max_valid_length(token_ids)
        chosen_rank = get_rank_num(valid_length)
        token_ids = token_ids[:, :chosen_rank]
        labels = labels[:, :chosen_rank]
        data_path = os.path.join(input_data_dir, "{}.npy".format(idx))
        label_path = os.path.join(label_dir, "{}.npy".format(idx))
        if token_ids.shape[0] != batch_size:
            token_ids = pad_data(token_ids)
            labels = pad_data(labels, constant_values=(-1))
        np.save(data_path, token_ids)
        np.save(label_path, labels)
        return chosen_rank

    input_datas = []
    for idx, datas in enumerate(data_loader):
        token_ids, labels = datas
        token_ids = token_ids.detach().numpy()
        labels = labels.detach().numpy()
        if not args.sorted:
            dump(token_ids, labels, idx)
        else:
            token_ids = token_ids.tolist()
            labels = labels.tolist()
            token_ids = [np.array(_) for _ in token_ids]
            labels = [np.array(_) for _ in labels]
            for _d in zip(token_ids, labels):
                input_datas.append(_d)

    if args.sorted:
        # sorted by valid num
        input_datas = sorted(input_datas,
                             key=lambda x:get_max_valid_length(np.expand_dims(x[0], axis=0)))
        data_num = len(input_datas)
        if data_num % batch_size == 0:
            num_list = np.array(list(range(data_num // batch_size)))
        else:
            num_list = np.array(list(range(data_num // batch_size + 1)))
        if args.save_shuffle:
            np.random.shuffle(num_list)
        for batch_idx in range(0, data_num, batch_size):
            input_data = input_datas[batch_idx : batch_idx+batch_size]
            token_ids = np.array([_[0] for _ in input_data])
            labels = np.array([_[1] for _ in input_data])
            dump(token_ids, labels, num_list[batch_idx//batch_size])


def parse_arguments():
    parser = argparse.ArgumentParser(description='Bert_Base_Chinese preprocess for sequence labeling task.')
    parser.add_argument('-i', '--input_path', type=str, required=True,
                        help='input dataset path')
    parser.add_argument('-o', '--out_dir', type=str, required=True,
                        help='save dir for preprocessed data')
    parser.add_argument('-d', '--dict_path', type=str, required=True,
                        help='vocab dict path for dataset')
    parser.add_argument('-s', '--seq_len', type=int, default=256,
                        help='max sequence length for output model')
    parser.add_argument('-r', '--rank', type=bool, default=False,
                        help='enable rank mode')
    parser.add_argument('--rank_list', type=str, default='32,48,64,96,128,192,224,256',
                        help='seq ranks for input data')
    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help='batch size for preprocess process')
    parser.add_argument('--sorted', type=bool, default=True,
                        help='whether to sort data in preprocess for rank mode')
    parser.add_argument('--save_shuffle', type=bool, default=True,
                        help='whether to shuffle preprocessed data')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    return args


if __name__ == '__main__':
    args = parse_arguments()
    categories = ['O', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG']
    categories_id2label = {i: k for i, k in enumerate(categories)}
    categories_label2id = {k: i for i, k in enumerate(categories)}
    tokenizer = Tokenizer(args.dict_path, do_lower_case=True)
    if not args.rank:
        valid_dataloader = DataLoader(MyDataset(args.input_path),
                                      batch_size=1,
                                      collate_fn=collate_fn)
        dump_data(valid_dataloader, args.out_dir)
    else:
        if args.batch_size is None:
            raise ValueError("Batch size should be provided, when ranks mode is on.")
        valid_dataloader = DataLoader(MyDataset(args.input_path),
                                      batch_size=args.batch_size,
                                      collate_fn=collate_fn)
        dump_data_ranks(valid_dataloader, args.out_dir, args.rank_list, args.batch_size)
