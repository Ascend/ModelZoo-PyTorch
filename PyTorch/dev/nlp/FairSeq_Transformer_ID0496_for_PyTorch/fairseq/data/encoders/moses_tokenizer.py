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

from fairseq.data.encoders import register_tokenizer


@register_tokenizer('moses')
class MosesTokenizer(object):

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--moses-source-lang', metavar='SRC',
                            help='source language')
        parser.add_argument('--moses-target-lang', metavar='TARGET',
                            help='target language')
        parser.add_argument('--moses-no-dash-splits', action='store_true', default=False,
                            help='don\'t apply dash split rules')
        parser.add_argument('--moses-no-escape', action='store_true', default=False,
                            help='don\'t perform HTML escaping on apostrophy, quotes, etc.')
        # fmt: on

    def __init__(self, args):
        self.args = args

        if getattr(args, 'moses_source_lang', None) is None:
            args.moses_source_lang = getattr(args, 'source_lang', 'en')
        if getattr(args, 'moses_target_lang', None) is None:
            args.moses_target_lang = getattr(args, 'target_lang', 'en')

        try:
            from sacremoses import MosesTokenizer, MosesDetokenizer
            self.tok = MosesTokenizer(args.moses_source_lang)
            self.detok = MosesDetokenizer(args.moses_target_lang)
        except ImportError:
            raise ImportError('Please install Moses tokenizer with: pip install sacremoses')

    def encode(self, x: str) -> str:
        return self.tok.tokenize(
            x,
            aggressive_dash_splits=(not self.args.moses_no_dash_splits),
            return_str=True,
            escape=(not self.args.moses_no_escape),
        )

    def decode(self, x: str) -> str:
        return self.detok.detokenize(x.split())
