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

from fairseq import file_utils
from fairseq.data.encoders import register_bpe

from .gpt2_bpe_utils import get_encoder


DEFAULT_ENCODER_JSON = 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
DEFAULT_VOCAB_BPE = 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'


@register_bpe('gpt2')
class GPT2BPE(object):

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--gpt2-encoder-json', type=str,
                            default=DEFAULT_ENCODER_JSON,
                            help='path to encoder.json')
        parser.add_argument('--gpt2-vocab-bpe', type=str,
                            default=DEFAULT_VOCAB_BPE,
                            help='path to vocab.bpe')
        # fmt: on

    def __init__(self, args):
        encoder_json = file_utils.cached_path(
            getattr(args, 'gpt2_encoder_json', DEFAULT_ENCODER_JSON)
        )
        vocab_bpe = file_utils.cached_path(
            getattr(args, 'gpt2_vocab_bpe', DEFAULT_VOCAB_BPE)
        )
        self.bpe = get_encoder(encoder_json, vocab_bpe)

    def encode(self, x: str) -> str:
        return ' '.join(map(str, self.bpe.encode(x)))

    def decode(self, x: str) -> str:
        return self.bpe.decode(map(int, x.split()))

    def is_beginning_of_word(self, x: str) -> bool:
        return self.decode(x).startswith(' ')
