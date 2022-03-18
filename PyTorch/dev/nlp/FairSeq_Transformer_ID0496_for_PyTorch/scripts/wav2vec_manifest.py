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
Data pre-processing: build vocabularies and binarize training data.
"""

import argparse
import glob
import os
import soundfile
import random


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', metavar='DIR', help='root directory containing flac files to index')
    parser.add_argument('--valid-percent', default=0.01, type=float, metavar='D',
                        help='percentage of data to use as validation set (between 0 and 1)')
    parser.add_argument('--dest', default='.', type=str, metavar='DIR', help='output directory')
    parser.add_argument('--ext', default='flac', type=str, metavar='EXT', help='extension to look for')
    parser.add_argument('--seed', default=42, type=int, metavar='N', help='random seed')
    parser.add_argument('--path-must-contain', default=None, type=str, metavar='FRAG',
                        help='if set, path must contain this substring for a file to be included in the manifest')
    return parser


def main(args):
    assert args.valid_percent >= 0 and args.valid_percent <= 1.

    dir_path = os.path.realpath(args.root)
    search_path = os.path.join(dir_path, '**/*.' + args.ext)
    rand = random.Random(args.seed)

    with open(os.path.join(args.dest, 'train.tsv'), 'w') as train_f, open(
            os.path.join(args.dest, 'valid.tsv'), 'w') as valid_f:
        print(dir_path, file=train_f)
        print(dir_path, file=valid_f)

        for fname in glob.iglob(search_path, recursive=True):
            file_path = os.path.realpath(fname)

            if args.path_must_contain and args.path_must_contain not in file_path:
                continue

            frames = soundfile.info(fname).frames
            dest = train_f if rand.random() > args.valid_percent else valid_f
            print('{}\t{}'.format(os.path.relpath(file_path, dir_path), frames), file=dest)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
