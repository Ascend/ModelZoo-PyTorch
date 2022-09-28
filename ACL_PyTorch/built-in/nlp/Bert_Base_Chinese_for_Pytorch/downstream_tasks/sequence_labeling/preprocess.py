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
        data_path = os.path.join(input_data_dir, "{}.bin".format(idx))
        label_path = os.path.join(label_dir, "{}.bin".format(idx))
        token_ids.detach().numpy().tofile(data_path)
        import pdb;pdb.set_trace()
        labels.detach().numpy().tofile(label_path)


def parse_arguments():
    parser = argparse.ArgumentParser(description='SwinTransformer onnx export.')
    parser.add_argument('-i', '--input_path', type=str, required=True,
                        help='input dataset path')
    parser.add_argument('-o', '--out_dir', type=str, required=True,
                        help='save dir for preprocessed data')
    parser.add_argument('-d', '--dict_path', type=str, required=True,
                        help='vocab dict path for dataset')
    parser.add_argument('-s', '--seq_len', type=int, default=256,
                        help='max sequence length for output model')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    return args


if __name__ == '__main__':
    args = parse_arguments()
    categories = ['O', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG']
    categories_id2label = {i: k for i, k in enumerate(categories)}
    categories_label2id = {k: i for i, k in enumerate(categories)}
    tokenizer = Tokenizer(args.dict_path, do_lower_case=True)
    valid_dataloader = DataLoader(MyDataset(args.input_path),
                                  batch_size=1,
                                  collate_fn=collate_fn)
    dump_data(valid_dataloader, args.out_dir)
