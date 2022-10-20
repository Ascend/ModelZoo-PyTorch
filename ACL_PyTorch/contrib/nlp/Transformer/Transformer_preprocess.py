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
import sys
import spacy
import torchtext
import torch
import argparse
import numpy as np
from tqdm import tqdm

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'


def text_data_preprocess():
    src_lang_model = spacy.load(opt.src_lang_mode_path)
    trg_lang_model = spacy.load(opt.trg_lang_mode_path)

    def tokenize_src(text):
        return [tok.text for tok in src_lang_model.tokenizer(text)]

    def tokenize_trg(text):
        return [tok.text for tok in trg_lang_model.tokenizer(text)]

    SRC = torchtext.data.Field(
        tokenize=tokenize_src, lower=opt.lowercase,
        pad_token=PAD_WORD, init_token=BOS_WORD, eos_token=EOS_WORD)

    TRG = torchtext.data.Field(
        tokenize=tokenize_trg, lower=opt.lowercase,
        pad_token=PAD_WORD, init_token=BOS_WORD, eos_token=EOS_WORD)

    def filter_examples_with_length(x):
        return len(vars(x)['src']) <= opt.filter_max_len and len(vars(x)['trg']) <= opt.filter_max_len

    train, _, test = torchtext.datasets.Multi30k.splits(
        exts=('.' + opt.src_lang, '.' + opt.trg_lang),
        fields=(SRC, TRG),
        root=opt.dataset_parent_path,
        filter_pred=filter_examples_with_length)

    SRC.build_vocab(train.src, min_freq=opt.filter_min_freq)
    print('source language vocabulary size:', len(SRC.vocab))
    TRG.build_vocab(train.trg, min_freq=opt.filter_min_freq)
    print('target language vocabulary size:', len(TRG.vocab))

    if opt.share_vocab:
        for w, _ in SRC.vocab.stoi.items():
            if w not in TRG.vocab.stoi:
                TRG.vocab.stoi[w] = len(TRG.vocab.stoi)
        TRG.vocab.itos = [None] * len(TRG.vocab.stoi)
        for w, i in TRG.vocab.stoi.items():
            TRG.vocab.itos[i] = w
        SRC.vocab.stoi = TRG.vocab.stoi
        SRC.vocab.itos = TRG.vocab.itos
        print('get merged vocabulary size:', len(TRG.vocab))

    data = test.examples

    if os.path.exists(opt.pre_data_save_path) == False:
        os.mkdir(opt.pre_data_save_path)

    f_test_en = open(os.path.join(opt.dataset_parent_path, "multi30k/test2016.en"), "rt")
    f_test_de_len15 = open(os.path.join(opt.pre_data_save_path, "test_de_len15.txt"), "wt")
    f_test_en_len15 = open(os.path.join(opt.pre_data_save_path, "test_en_len15.txt"), "wt")
    f_test_de_array_len15 = open(os.path.join(opt.pre_data_save_path, "test_de_array_len15.txt"), "wt")
    f_info_file = open(os.path.join(opt.pre_data_save_path, "bin_file.info"), "wt")

    unk_idx = SRC.vocab.stoi[SRC.unk_token]
    test_loader = torchtext.data.Dataset(examples=data, fields={'src': SRC, 'trg': TRG})
    count = 0
    for index, example in enumerate(tqdm(test_loader)):

        src_seq = [SRC.vocab.stoi.get(word, unk_idx) for word in example.src]
        if len(src_seq) > opt.align_length:
            f_test_en.readline()
            continue
        else:
            src_seq = list(src_seq + [1] * (opt.align_length - len(src_seq)))
            f_test_de_len15.write(" ".join(example.src) + "\n")
            f_test_en_len15.write(f_test_en.readline())
            f_test_de_array_len15.write(str(src_seq) + "\n")

        src_seq = np.asarray(src_seq, dtype=np.int64)
        bin_file_path = os.path.join(opt.pre_data_save_path, str(count) + ".bin")
        src_seq.tofile(bin_file_path)
        f_info_file.write(str(count) + " ./" + str(count) + ".bin" + '\n')
        count += 1

    f_test_en.close()
    f_test_de_len15.close()
    f_test_en_len15.close()
    f_test_de_array_len15.close()
    f_info_file.close()


if __name__ == "__main__":
    """
    Usage Example:
    preprocess_to_bin.py \
    --src_lang=de \
    --trg_lang=en \
    --src_lang_mode_path=de \
    --trg_lang_mode_path=en \
    --dataset_parent_path=.data \
    --pre_data_save_path=./pre_data/len15 \
    --align_length 15
    """

    src_langs = ['de']
    trg_langs = ['en']

    parser = argparse.ArgumentParser(description='preprocess the text data')

    parser.add_argument('--src_lang', required=True, choices=src_langs, help='source language')
    parser.add_argument('--trg_lang', required=True, choices=trg_langs, help='target language')
    parser.add_argument('--src_lang_mode_path', required=True, help='source language model path')
    parser.add_argument('--trg_lang_mode_path', required=True, help='target language model path')
    parser.add_argument('--dataset_parent_path', required=True, help='Multi30k dataset parent path')
    parser.add_argument('--pre_data_save_path', required=True, help='preprocess data save folder path')
    parser.add_argument('--filter_max_len', type=int, default=100,
                        help="sentences that are longer than this size are filtered when the dataset is created")
    parser.add_argument('--filter_min_freq', type=int, default=3,
                        help="a word must appear more than this number of times to be added to the vocab")
    parser.add_argument('--lowercase', type=bool, default=True, help="convert all words to lowercase")
    parser.add_argument('--share_vocab', type=bool, default=True, help="if share vocab between src and trg language")
    parser.add_argument('--align_length', type=int, default=15,
                        help="the alignment length when constructing an array from the sentence")

    opt = parser.parse_args()

    text_data_preprocess()
