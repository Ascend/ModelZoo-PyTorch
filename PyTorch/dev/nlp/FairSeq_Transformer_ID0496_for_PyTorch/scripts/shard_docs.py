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
Split a large file into shards while respecting document boundaries. Documents
should be separated by a single empty line.
"""

import argparse
import contextlib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('--num-shards', type=int)
    args = parser.parse_args()

    assert args.num_shards is not None and args.num_shards > 1

    with open(args.input, 'r', encoding='utf-8') as h:
        with contextlib.ExitStack() as stack:
            outputs = [
                stack.enter_context(open(args.input + ".shard" + str(i), "w", encoding="utf-8"))
                for i in range(args.num_shards)
            ]

            doc = []
            first_doc = [True]*args.num_shards
            def output_doc(i):
                if not first_doc[i]:
                    outputs[i].write("\n")
                first_doc[i] = False
                for line in doc:
                    outputs[i].write(line)
                doc.clear()

            num_docs = 0
            for line in h:
                if line.strip() == "":  # empty line indicates new document
                    output_doc(num_docs % args.num_shards)
                    num_docs += 1
                else:
                    doc.append(line)
            output_doc(num_docs % args.num_shards)


if __name__ == '__main__':
    main()
