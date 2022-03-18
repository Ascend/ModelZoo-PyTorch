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
Split a large file into a train and valid set while respecting document
boundaries. Documents should be separated by a single empty line.
"""

import argparse
import random
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('sample_output', help='train output file')
    parser.add_argument('remainder_output', help='valid output file')
    parser.add_argument('-k', type=int, help="remainder size")
    parser.add_argument('--lines', action='store_true',
                        help='split lines instead of docs')
    args = parser.parse_args()

    assert args.k is not None

    sample = []
    remainder = []
    num_docs = [0]

    def update_sample(doc):
        if len(sample) < args.k:
            sample.append(doc.copy())
        else:
            i = num_docs[0]
            j = random.randrange(i + 1)
            if j < args.k:
                remainder.append(sample[j])
                sample[j] = doc.copy()
            else:
                remainder.append(doc.copy())
        num_docs[0] += 1
        doc.clear()

    with open(args.input, 'r', encoding='utf-8') as h:
        doc = []
        for i, line in enumerate(h):
            if line.strip() == "":  # empty line indicates new document
                update_sample(doc)
            else:
                doc.append(line)
            if args.lines:
                update_sample(doc)
            if i % 1000000 == 0:
                print(i, file=sys.stderr, end="", flush=True)
            elif i % 100000 == 0:
                print(".", file=sys.stderr, end="", flush=True)
        if len(doc) > 0:
            update_sample(doc)
    print(file=sys.stderr, flush=True)

    assert len(sample) == args.k

    with open(args.sample_output, 'w', encoding='utf-8') as out:
        first = True
        for doc in sample:
            if not first and not args.lines:
                out.write("\n")
            first = False
            for line in doc:
                out.write(line)

    with open(args.remainder_output, 'w', encoding='utf-8') as out:
        first = True
        for doc in remainder:
            if not first and not args.lines:
                out.write("\n")
            first = False
            for line in doc:
                out.write(line)


if __name__ == '__main__':
    main()
