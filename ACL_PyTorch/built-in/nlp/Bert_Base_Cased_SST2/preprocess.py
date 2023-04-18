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

from transformers import AutoTokenizer


def build_tokenizer(tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        cache_dir=None,
        use_fast=True,
        use_auth_token=None,
    )
    return tokenizer


def preprocess(tokenizer_path, text_file, save_path, seq_len):
    tokenizer = build_tokenizer(tokenizer_path)
    os.makedirs(os.path.join(save_path, 'input_ids'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'attention_mask'), exist_ok=True)

    with open(text_file, 'r') as data_file:
        labels = {}
        step = 0
        for line in data_file:
            if step > 0:
                context = line.strip().split('\t')
                sentence = context[0]
                labels[step] = context[1]
                result = tokenizer(
                    sentence,
                    padding='max_length',
                    max_length=seq_len,
                    truncation=True,
                    return_tensors='np'
                )
                ids_path = os.path.join(
                    save_path,
                    'input_ids',
                    '{}.bin'.format(step)
                )
                result['input_ids'].tofile(ids_path)
                mask_path = os.path.join(
                    save_path,
                    'attention_mask',
                    '{}.bin'.format(step)
                )
                result['attention_mask'].tofile(mask_path)
            step += 1
    file_name = os.path.join(save_path, "labels.json")
    flags = os.O_CREAT | os.O_WRONLY
    with os.fdopen(os.open(file_name, flags, 0o755), 'w') as json_file:
        json.dump(labels, json_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_path', required=True,
                        help='path of the folder of tokenizer config')
    parser.add_argument('--text_file', required=True,
                        help='path of the text file to process')
    parser.add_argument('--save_path', required=True,
                        help='path of the onnx model')
    parser.add_argument('--seq_len', default=128, type=int,
                        help='length of the input sequence')
    args = parser.parse_args()

    preprocess(
        args.tokenizer_path,
        args.text_file,
        args.save_path,
        args.seq_len
    )
