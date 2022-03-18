#!/usr/bin/env python3
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
"""
Count the number of documents and average number of lines and tokens per
document in a large file. Documents should be separated by a single empty line.
"""

import argparse
import gzip
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('--gzip', action='store_true')
    args = parser.parse_args()

    def gopen():
        if args.gzip:
            return gzip.open(args.input, 'r')
        else:
            return open(args.input, 'r', encoding='utf-8')

    num_lines = []
    num_toks = []
    with gopen() as h:
        num_docs = 1
        num_lines_in_doc = 0
        num_toks_in_doc = 0
        for i, line in enumerate(h):
            if len(line.strip()) == 0:  # empty line indicates new document
                num_docs += 1
                num_lines.append(num_lines_in_doc)
                num_toks.append(num_toks_in_doc)
                num_lines_in_doc = 0
                num_toks_in_doc = 0
            else:
                num_lines_in_doc += 1
                num_toks_in_doc += len(line.rstrip().split())
            if i % 1000000 == 0:
                print(i, file=sys.stderr, end="", flush=True)
            elif i % 100000 == 0:
                print(".", file=sys.stderr, end="", flush=True)
        print(file=sys.stderr, flush=True)

    print("found {} docs".format(num_docs))
    print("average num lines per doc: {}".format(np.mean(num_lines)))
    print("average num toks per doc: {}".format(np.mean(num_toks)))


if __name__ == '__main__':
    main()
