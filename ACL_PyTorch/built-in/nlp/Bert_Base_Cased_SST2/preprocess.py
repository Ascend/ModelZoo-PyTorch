# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
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
import argparse

import numpy as np
from transformers import AutoTokenizer


def build_tokenizer(tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        cache_dir=None,
        use_fast=True,
        use_auth_token=None,
    )
    return tokenizer


def padding(result, seq_lens):
    input_ids = result['input_ids']
    attention_mask = result['attention_mask']
    real_len = input_ids.shape[1]
    for seq_len in seq_lens:
        if real_len <= seq_len:
            pad_num = seq_len - real_len
            pad_input = np.pad(input_ids, ((0, 0 ), (0, pad_num)))
            pad_mask = np.pad(attention_mask, ((0, 0 ), (0, pad_num)))
            return pad_input, pad_mask, seq_len
    restore_ids = input_ids[:, :, :seq_len]
    restore_mask = attention_mask[:, :, seq_len]
    return restore_ids, restore_mask, seq_len


def save(save_path, input_ids, attention_mask, label, number):
    ids_path = os.path.join(
        save_path,
        'input_ids',
        '{}.npy'.format(number)
    )
    np.save(ids_path, input_ids)
    mask_path = os.path.join(
        save_path,
        'attention_mask',
        '{}.npy'.format(number)
    )
    np.save(mask_path, attention_mask)
    label_path = os.path.join(
        save_path,
        'labels',
        '{}.npy'.format(number)
    )
    np.save(label_path, label)


def preprocess(cfg):
    tokenizer = build_tokenizer(cfg.tokenizer_path)
    os.makedirs(
        os.path.join(cfg.save_path, 'input_ids'),
        exist_ok=True
    )
    os.makedirs(
        os.path.join(cfg.save_path, 'attention_mask'),
        exist_ok=True
    )
    os.makedirs(
        os.path.join(cfg.save_path, 'labels'),
        exist_ok=True
    )

    seq_lens = [int(seq_len) for seq_len in cfg.seq_len.split(',')]
    seq_lens.sort()
    number = 0

    with open(cfg.text_file, 'r') as data_file:
        data = {seq_len: [[], []] for seq_len in seq_lens}
        labels = {seq_len: [] for seq_len in seq_lens}
        step = 0
        for line in data_file:
            if step > 0:
                context = line.strip().split('\t')
                sentence = context[0]
                result = tokenizer(
                    sentence,
                    truncation=True,
                    return_tensors='np'
                )
                input_ids, attention_mask, real_len = padding(result, seq_lens)
                labels[real_len].append(context[1])
                data[real_len][0].append(input_ids)
                data[real_len][1].append(attention_mask)
                if len(labels[real_len]) == cfg.batch_size:
                    input_ids = np.concatenate(data[real_len][0], axis=0)
                    attention_mask = np.concatenate(data[real_len][1], axis=0)
                    label = np.array(labels[real_len])
                    
                    save(
                        cfg.save_path,
                        input_ids,
                        attention_mask,
                        label,
                        number
                    )
                    data[real_len] = [[],[]]
                    labels[real_len] = []
                    number +=1
            step += 1

    for seq_len, inputs in data.items():
        if len(inputs[0]) > 0:
            pad_num = cfg.batch_size - len(inputs[0])
            pad = np.zeros((pad_num, seq_len)).astype(np.int64)

            input_ids = np.concatenate(inputs[0], axis=0)
            attention_mask = np.concatenate(inputs[1], axis=0)
            input_ids = np.concatenate((input_ids, pad), axis=0)
            attention_mask = np.concatenate((attention_mask, pad), axis=0)

            label = np.array(labels[seq_len])
            save(
                cfg.save_path,
                input_ids,
                attention_mask,
                label,
                number
            )
            number += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_path', required=True,
                        help='path of the folder of tokenizer config')
    parser.add_argument('--text_file', required=True,
                        help='path of the text file to process')
    parser.add_argument('--save_path', required=True,
                        help='path of the onnx model')
    parser.add_argument('--seq_len', default='128',
                        help='length of the input sequence')
    parser.add_argument('--batch_size', required=True, type=int,
                        help='the batch size of input data')
    args = parser.parse_args()

    preprocess(args)
