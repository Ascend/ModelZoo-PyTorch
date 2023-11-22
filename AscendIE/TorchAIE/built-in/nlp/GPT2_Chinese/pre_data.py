# Copyright 2021 Huawei Technologies Co., Ltd
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

from tqdm import tqdm
import argparse
import transformers

from tokenizations import tokenization_bert


def build_files(data_path, tokenized_data_path, num_pieces, full_tokenizer, min_length):
    if not os.path.exists(tokenized_data_path):
        os.mkdir(tokenized_data_path)
    with open(data_path, 'r', encoding='utf8') as f:
        print('reading lines')
        lines = json.load(f)
        lines = [line.replace('\n', ' [SEP] ')
                 for line in lines]  # 用[SEP]表示换行, 段落之间使用SEP表示段落结束
        all_len = len(lines)
    for i in tqdm(range(num_pieces)):
        sublines = lines[all_len // num_pieces *
                         i: all_len // num_pieces * (i + 1)]
        if i == num_pieces - 1:
            # 把尾部例子添加到最后一个piece
            sublines.extend(lines[all_len // num_pieces * (i + 1):])
        sublines = [full_tokenizer.tokenize(line) for line in sublines if
                    len(line) > min_length]  # 只考虑长度超过min_length的句子
        sublines = [full_tokenizer.convert_tokens_to_ids(
            line) for line in sublines]
        full_line = []
        for subline in sublines:
            full_line.append(full_tokenizer.convert_tokens_to_ids(
                '[MASK]'))  # 文章开头添加MASK表示文章开始
            full_line.extend(subline)
            full_line.append(full_tokenizer.convert_tokens_to_ids(
                '[CLS]'))  # 文章之间添加CLS表示文章结束
        with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'w') as f:
            for id in full_line:
                f.write(str(id) + ' ')
    print('finish')


def prepare_data(data_path, save_dir):

    data = []
    pre_path = os.listdir(data_path)
    for mid_path in pre_path:
        path_ = os.path.join(data_path, mid_path)
        re_path = os.listdir(path_)
        for pp in re_path:
            p_ = os.path.join(path_, pp)
            with open(p_, 'r', encoding='utf8') as f:
                lines = f.readlines()
                for line in lines:
                    data.append(json.loads(line)['text'])
            break
    with open(save_dir, 'w') as f:
        json.dump(data, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', default='./config.json', type=str, required=False,
                        help='选择模型参数')
    parser.add_argument('--raw_data_path', default='./data/wiki_zh',
                        type=str, required=False, help='原始语料')
    parser.add_argument('--data_json_path', default='./eval.json',
                        type=str, required=False, help='原始语料')
    parser.add_argument('--tokenized_data_path', default='data/tokenized_eval/', type=str, required=False,
                        help='tokenized语料存放位置')
    parser.add_argument('--num_pieces', default=100,
                        type=int, required=False, help='将训练语料分成多少份')
    parser.add_argument('--tokenizer_path', default='./vocab.txt',
                        type=str, required=False, help='选择词库')
    parser.add_argument('--min_length', default=128,
                        type=int, required=False, help='最短收录文章长度')

    args = parser.parse_args()

    model_config = transformers.modeling_gpt2.GPT2Config.from_json_file(
        args.model_config)
    n_ctx = model_config.n_ctx
    full_tokenizer = tokenization_bert.BertTokenizer(
        vocab_file=args.tokenizer_path)
    full_tokenizer.max_len = n_ctx
    raw_data_path = args.raw_data_path
    data_json_path = args.data_json_path
    tokenized_data_path = args.tokenized_data_path
    num_pieces = args.num_pieces
    min_length = args.min_length

    prepare_data(raw_data_path, data_json_path)
    build_files(data_path=data_json_path, tokenized_data_path=tokenized_data_path, num_pieces=num_pieces,
                full_tokenizer=full_tokenizer, min_length=min_length)


if __name__ == '__main__':
    main()
