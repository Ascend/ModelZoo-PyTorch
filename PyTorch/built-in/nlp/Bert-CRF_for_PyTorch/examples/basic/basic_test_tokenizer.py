# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017
# All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd
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
# ==========================================================================

# 测试tokenizer和transformers自带的tokenizer是否一致，测试后是一致的
from transformers import BertTokenizer, XLNetTokenizer, XLNetTokenizerFast
from bert4torch.tokenizers import Tokenizer, SpTokenizer
from tqdm import tqdm

choice = 1
if choice:
    print('Test BertTokenizer')
    tokenizer_transformers = BertTokenizer.from_pretrained("F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12")
    tokenizer_bert4torch = Tokenizer('F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/vocab.txt', do_lower_case=True, do_tokenize_unk=True)
else:
    print('Test SpTokenizer')
    tokenizer_transformers = XLNetTokenizer.from_pretrained("F:/Projects/pretrain_ckpt/xlnet/[hit_torch_base]--chinese-xlnet-base")
    # tokenizer_transformers = XLNetTokenizerFast.from_pretrained("F:/Projects/pretrain_ckpt/xlnet/[hit_torch_base]--chinese-xlnet-base")  # fast版本有些许不一样
    tokenizer_bert4torch = tokenizer = SpTokenizer('F:/Projects/pretrain_ckpt/xlnet/[hit_torch_base]--chinese-xlnet-base/spiece.model', token_start=None, token_end=None)

with open('F:/Projects/data/corpus/sentence_classification/sentiment/sentiment.train.data', 'r', encoding='utf-8') as f:
    for l in tqdm(f):
        l = l.split('\t')[0].strip()
        tokens1 = tokenizer_transformers.tokenize(l)
        tokens2 = tokenizer_bert4torch.tokenize(l)
        tokens2 = tokens2[1:-1] if choice == 1 else tokens2
        if tokens1 != tokens2:
            print(''.join(tokens1))
            print(''.join(tokens2))
            print('------------------------------')