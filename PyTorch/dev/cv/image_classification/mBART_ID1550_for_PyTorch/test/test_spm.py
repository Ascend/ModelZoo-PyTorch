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
Created by raj at 4:40 PM,  2/20/20
"""
import sentencepiece as spm
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))


def concat_src_trg(prefix, src, trg, input_sentnce_size=100_000):
    # Reading data from file1
    nlines = int(input_sentnce_size / 2)
    with open(prefix + '.' + src) as fp:
        data = fp.read()
    data = data[:nlines]
    print(data)

    # Reading data from file2
    with open(prefix + '.' + trg) as fp:
        data2 = fp.read()
    # data2 = data2[:nlines]

    # Merging 2 files
    # To add the data of file2
    # from next line
    data += "\n"
    data += data2

    print(len(data))

    output = prefix + '.' + src + '-' + trg
    with open(output, 'w') as fp:
        fp.write(data)
    return output


input_sentnce_size = 200_000

input_file = '.data/iwslt//de-en/train.de-en'
concatenated = concat_src_trg(input_file, 'de', 'en', input_sentnce_size)

spm.SentencePieceTrainer.Train('--input={} --input_sentence_size={} --shuffle_input_sentence=true '
                               '--model_prefix=m --vocab_size={}'.format(concatenated, input_sentnce_size, 32_000))

