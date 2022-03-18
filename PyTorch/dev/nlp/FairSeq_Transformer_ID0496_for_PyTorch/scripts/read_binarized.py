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

import argparse

from fairseq.data import data_utils, Dictionary, indexed_dataset


def get_parser():
    parser = argparse.ArgumentParser(
        description='writes text from binarized file to stdout')
    # fmt: off
    parser.add_argument('--dataset-impl', help='dataset implementation',
                        choices=indexed_dataset.get_available_dataset_impl())
    parser.add_argument('--dict', metavar='FP', help='dictionary containing known words', default=None)
    parser.add_argument('--input', metavar='FP', required=True, help='binarized file to read')
    # fmt: on

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    dictionary = Dictionary.load(args.dict) if args.dict is not None else None
    dataset = data_utils.load_indexed_dataset(
        args.input,
        dictionary,
        dataset_impl=args.dataset_impl,
        default='lazy',
    )

    for tensor_line in dataset:
        if dictionary is None:
            line = ' '.join([str(int(x)) for x in tensor_line])
        else:
            line = dictionary.string(tensor_line)

        print(line)


if __name__ == '__main__':
    main()
