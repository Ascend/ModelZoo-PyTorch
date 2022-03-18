#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================
#
"""
pytorch-dl
Created by raj at 7:48 AM,  7/31/20
"""
import os
import time

import torch

from dataset.iwslt_data import rebatch_source_only
from models.decoding import batched_beam_search
from models.utils.model_utils import load_model_state
from onmt import opts, inputters
from onmt.utils import set_random_seed
from onmt.utils.parse import ArgumentParser
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


def translate(opt):
    set_random_seed(opt.seed, False)

    start_steps, model, fields = load_model_state(os.path.join(opt.models[0], 'checkpoints_best.pt'), opt,
                                                  data_parallel=False)
    model.eval()

    src_vocab = fields['src'].base_field.vocab
    trg_vocab = fields['tgt'].base_field.vocab

    pad_idx = src_vocab.stoi["<blank>"]
    unk_idx = src_vocab.stoi["<unk>"]
    start_symbol = trg_vocab.stoi["<s>"]
    if start_symbol == unk_idx:
        if opt.tgt_lang_id:
            start_symbol = trg_vocab.stoi["<" + opt.tgt_lang_id + ">"]
        else:
            raise AssertionError("For mBart fine-tuned model, --tgt_lang_id is necessary to set. eg DE EN etc.")

    with open(opt.src) as input:
        src = input.readlines()

    src_reader = inputters.str2reader['text'].from_opt(opt)
    src_data = {"reader": src_reader, "data": src, "dir": ''}

    _readers, _data, _dir = inputters.Dataset.config(
        [('src', src_data)])

    # corpus_id field is useless here
    if fields.get("corpus_id", None) is not None:
        fields.pop('corpus_id')

    data = inputters.Dataset(fields, readers=_readers, dirs=_dir, data=_data, sort_key=inputters.str2sortkey['text'])

    data_iter = inputters.OrderedIterator(
        dataset=data,
        batch_size=1,
        train=False,
        sort=False,
        sort_within_batch=True,
        shuffle=False
    )

    cuda_condition = torch.npu.is_available() and not opt.cpu
    device = torch.device(f'npu:{NPU_CALCULATE_DEVICE}')

    if cuda_condition:
        model.npu()

    with torch.no_grad():
        translated = list()
        reference = list()
        start = time.time()
        for k, batch in enumerate(rebatch_source_only(pad_idx, b, device=device) for b in data_iter):
            print('Processing: {0}'.format(k))
            # out = greedy_decode(model, batch.src, batch.src_mask, start_symbol=start_symbol)
            # out = beam_search(model, batch.src, batch.src_mask,
            #                           start_symbol=start_symbol, pad_symbol=pad_idx,
            #                           max=batch.ntokens + 10)
            out = batched_beam_search(model, batch.src, batch.src_mask,
                                      start_symbol=start_symbol, pad_symbol=pad_idx,
                                      max=batch.ntokens + 10)

            # print("Source:", end="\t")
            # for i in range(1, batch.src.size(1)):
            #     sym = SRC.vocab.itos[batch.src.data[0, i]]
            #     if sym == "<eos>": break
            #     print(sym, end=" ")
            # print()
            # print("Translation:", end="\t")

            transl = list()
            start_idx = 0  # for greedy decoding the start index should be 1 that will exclude the <sos> symbol
            for i in range(start_idx, out.size(1)):
                sym = trg_vocab.itos[out[0, i]]
                if sym == "</s>": break
                transl.append(sym)
            text_transl = " ".join(transl).replace("@@ ", '')
            translated.append(text_transl)

            print(text_transl)

            # print()
            # print("Target:", end="\t")
            # ref = list()
            # for i in range(1, batch.trg.size(1)):
            #     sym = trg_vocab.itos[batch.trg.data[0, i]]
            #     if sym == "</s>": break
            #     ref.append(sym)
            # reference.append(" ".join(ref))

            # if k == 1:
            #     break

    with open('test-beam-decode.de-en.en', 'w', encoding='utf8') as outfile:
        outfile.write('\n'.join(translated))
    # with open('valid-ref.de-en.en', 'w', encoding='utf-8') as outfile:
    #     outfile.write('\n'.join(reference))
    print('Time elapsed:{}'.format(time.time() - start))


def _get_parser():
    parser = ArgumentParser(description='translate.py')

    opts.config_opts(parser)
    opts.translate_opts(parser)
    return parser


def main():
    parser = _get_parser()

    opt = parser.parse_args()
    translate(opt)


if __name__ == "__main__":
    main()
