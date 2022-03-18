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
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.data.encoders import register_bpe


@register_bpe('bert')
class BertBPE(object):

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--bpe-cased', action='store_true',
                            help='set for cased BPE',
                            default=False)
        parser.add_argument('--bpe-vocab-file', type=str,
                            help='bpe vocab file.')
        # fmt: on

    def __init__(self, args):
        try:
            from pytorch_transformers import BertTokenizer
            from pytorch_transformers.tokenization_utils import clean_up_tokenization
        except ImportError:
            raise ImportError(
                'Please install 1.0.0 version of pytorch_transformers'
                'with: pip install pytorch-transformers'
            )

        if 'bpe_vocab_file' in args:
            self.bert_tokenizer = BertTokenizer(
                args.bpe_vocab_file,
                do_lower_case=not args.bpe_cased
            )
        else:
            vocab_file_name = 'bert-base-cased' if args.bpe_cased else 'bert-base-uncased'
            self.bert_tokenizer = BertTokenizer.from_pretrained(vocab_file_name)
            self.clean_up_tokenization = clean_up_tokenization

    def encode(self, x: str) -> str:
        return ' '.join(self.bert_tokenizer.tokenize(x))

    def decode(self, x: str) -> str:
        return self.clean_up_tokenization(
            self.bert_tokenizer.convert_tokens_to_string(x.split(' '))
        )

    def is_beginning_of_word(self, x: str) -> bool:
        return not x.startswith('##')
