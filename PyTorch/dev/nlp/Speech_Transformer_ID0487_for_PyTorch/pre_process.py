# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ============================================================================
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
import pickle

from tqdm import tqdm

from config import wav_folder, tran_file, pickle_file
from utils import ensure_folder


def get_data(split):
    print('getting {} data...'.format(split))

    global VOCAB

    with open(tran_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    tran_dict = dict()
    for line in lines:
        tokens = line.split()
        key = tokens[0]
        trn = ''.join(tokens[1:])
        tran_dict[key] = trn

    samples = []

    folder = os.path.join(wav_folder, split)
    ensure_folder(folder)
    dirs = [os.path.join(folder, d) for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    for dir in tqdm(dirs):
        files = [f for f in os.listdir(dir) if f.endswith('.wav')]

        for f in files:
            wave = os.path.join(dir, f)

            key = f.split('.')[0]
            if key in tran_dict:
                trn = tran_dict[key]
                trn = list(trn.strip()) + ['<eos>']

                for token in trn:
                    build_vocab(token)

                trn = [VOCAB[token] for token in trn]

                samples.append({'trn': trn, 'wave': wave})

    print('split: {}, num_files: {}'.format(split, len(samples)))
    return samples


def build_vocab(token):
    global VOCAB, IVOCAB
    if not token in VOCAB:
        next_index = len(VOCAB)
        VOCAB[token] = next_index
        IVOCAB[next_index] = token


if __name__ == "__main__":
    VOCAB = {'<sos>': 0, '<eos>': 1}
    IVOCAB = {0: '<sos>', 1: '<eos>'}

    data = dict()
    data['VOCAB'] = VOCAB
    data['IVOCAB'] = IVOCAB
    data['train'] = get_data('train')
    data['dev'] = get_data('dev')
    data['test'] = get_data('test')

    with open(pickle_file, 'wb') as file:
        pickle.dump(data, file)

    print('num_train: ' + str(len(data['train'])))
    print('num_dev: ' + str(len(data['dev'])))
    print('num_test: ' + str(len(data['test'])))
    print('vocab_size: ' + str(len(data['VOCAB'])))
