#!/usr/bin/env python
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
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse

import sentencepiece as spm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="sentencepiece model to use for decoding")
    parser.add_argument("--input", required=True, help="input file to decode")
    parser.add_argument("--input_format", choices=["piece", "id"], default="piece")
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor()
    sp.Load(args.model)

    if args.input_format == "piece":
        def decode(l):
            return "".join(sp.DecodePieces(l))
    elif args.input_format == "id":
        def decode(l):
            return "".join(sp.DecodeIds(l))
    else:
        raise NotImplementedError

    def tok2int(tok):
        # remap reference-side <unk> (represented as <<unk>>) to 0
        return int(tok) if tok != "<<unk>>" else 0

    with open(args.input, "r", encoding="utf-8") as h:
        for line in h:
            print(decode(list(map(tok2int, line.rstrip().split()))))


if __name__ == "__main__":
    main()
