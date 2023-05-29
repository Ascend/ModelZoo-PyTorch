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
import sys
import json
import stat
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import tqdm

import config
from vocabulary import Vocabulary
from common import seed_everything


class Preprocessor:

    def __init__(self, max_seq_len=50, data_dir=None, vocab_path=None):
        self.max_seq_len = max_seq_len
        self.vocab_path = vocab_path
        if not Path(self.vocab_path).is_file():
            self.build_vocab(data_dir)
        self.vocab = self.load_vocab()

    def load_vocab(self):
        vocab = Vocabulary()
        vocab.load_from_file(self.vocab_path)
        return vocab

    def build_vocab(self, data_dir):
        vocab = Vocabulary()
        files = ["train.json", "dev.json", "test.json"]
        for file in files:
            with open(str(Path(data_dir) / file), 'r') as fr:
                for line in fr:
                    line = json.loads(line.strip())
                    text = line['text']
                    vocab.update(list(text))
        vocab.build_vocab()
        vocab.save(self.vocab_path)

    def process(self, batch, batchsize=1):
        if not isinstance(batch, list):
            batch = [batch]

        ids_batch, mask_batch, lens_batch, tags_batch = [], [], [], []
        for item in batch:
            text = item['text']
            input_ids, input_mask, input_len = self.process_text(text)
            ids_batch.append(input_ids)
            mask_batch.append(input_mask)
            lens_batch.append(input_len)

            anno_label = item.get('label')
            if anno_label:
                input_tags = self.process_label(text, anno_label)
                tags_batch.append(input_tags)

        # Pad
        if len(batch) < batchsize:
            ids_batch += [ids_batch[-1]] * (batchsize - len(batch))
            mask_batch += [mask_batch[-1]] * (batchsize - len(batch))

        ids_batch = torch.vstack(ids_batch).numpy()
        mask_batch = torch.vstack(mask_batch).numpy()

        if tags_batch:
            assert len(tags_batch) == len(batch)
        else:
            tags_batch = None

        return ids_batch, mask_batch, tags_batch, lens_batch

    def process_text(self, text):
        words = list(text)
        context = " ".join(words)

        token_a = context.split(" ")
        input_ids = torch.tensor(
            [[self.vocab.to_index(w) for w in token_a]],
            dtype=torch.long)
        input_mask = torch.tensor(
            [[1] * len(token_a)], dtype=torch.long)
        input_len = len(token_a)

        right_pading = self.max_seq_len - input_ids.size(1)
        zero_pad = nn.ZeroPad2d(padding=(0, right_pading, 0, 0))
        input_ids = zero_pad(input_ids)
        input_mask = zero_pad(input_mask)

        assert input_ids.shape == input_mask.shape
        assert input_ids.size(1) == self.max_seq_len

        return input_ids, input_mask, input_len

    @staticmethod
    def process_label(text, label):        
        tags = ['O'] * len(text)
        if label is None:
            return tags
        for key, value in label.items():
            for sub_name, sub_index in value.items():
                for start, end in sub_index:
                    assert ''.join(text[start: end+1]) == sub_name
                    if start == end:
                        tags[start] = 'S-' + key
                    else:
                        tags[start] = 'B-' + key
                        tags[start+1: end+1] = ['I-' + key] * (len(sub_name) - 1)
        return tags

    def __call__(self, text_batch, label_batch):
        return self.process(text_batch, label_batch)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, help='path to data file')
    parser.add_argument(
        "--vocab", type=str, 
        default='CLUENER2020/bilstm_crf_pytorch/dataset/cluener/vocab.pkl', 
        help='path to om vocab file (.pkl)')
    parser.add_argument(
        "--max_seq_len", type=int, default=50,  
        help="path to checkpoint.")
    parser.add_argument(
        "--output", type=str, default=None, 
        help='a floder to save preprocessed data.')
    args = parser.parse_args()

    output_dir = Path(args.output)
    ids_dir = output_dir / 'inputs' / 'input_ids'
    mask_dir = output_dir / 'inputs' / 'input_mask'
    ids_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    label_path = output_dir / 'label.txt'
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    modes = stat.S_IWUSR | stat.S_IRUSR
    fout = os.fdopen(os.open(str(label_path), flags, modes), 'w')

    seed_everything(1234)
    data_dir = 'CLUENER2020/bilstm_crf_pytorch/dataset/cluener/'
    preprocessor = Preprocessor(max_seq_len=args.max_seq_len, 
                                data_dir=data_dir,
                                vocab_path=args.vocab)

    assert Path(args.data).is_file(), f"path is not a file: {args.data}"
    data_list = []
    with open(args.data, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            assert 'text' in data
            data_list.append(data)

    for i, data in tqdm.tqdm(enumerate(data_list)):
        ids_batch, mask_batch, tags_batch, lens_batch = \
            preprocessor.process(data)
        assert ids_batch.shape[0] == 1
        np.save(ids_dir / f'data_{i:0>8}.npy', ids_batch[0])
        np.save(mask_dir / f'data_{i:0>8}.npy', mask_batch[0])
        line = json.dumps(dict(
            data_index=i, tags=tags_batch, lens=lens_batch))
        fout.write(line + '\n')

    fout.close()


if __name__ == "__main__":
    main()
