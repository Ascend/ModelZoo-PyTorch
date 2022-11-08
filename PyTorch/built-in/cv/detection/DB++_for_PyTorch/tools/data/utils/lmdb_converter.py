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

# Copyright (c) OpenMMLab. All rights reserved.
import argparse

from mmocr.utils import recog2lmdb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('label_path', type=str, help='Path to label file')
    parser.add_argument('output', type=str, help='Output lmdb path')
    parser.add_argument(
        '--img-root', '-i', type=str, help='Input imglist path')
    parser.add_argument(
        '--label-only',
        action='store_true',
        help='Only converter label to lmdb')
    parser.add_argument(
        '--label-format',
        '-f',
        default='txt',
        choices=['txt', 'jsonl'],
        help='The format of the label file, either txt or jsonl')
    parser.add_argument(
        '--batch-size',
        '-b',
        type=int,
        default=1000,
        help='Processing batch size, defaults to 1000')
    parser.add_argument(
        '--encoding',
        '-e',
        type=str,
        default='utf8',
        help='Bytes coding scheme, defaults to utf8')
    parser.add_argument(
        '--lmdb-map-size',
        '-m',
        type=int,
        default=1099511627776,
        help='Maximum size database may grow to, '
        'defaults to 1099511627776 bytes (1TB)')
    opt = parser.parse_args()

    assert opt.img_root or opt.label_only
    recog2lmdb(opt.img_root, opt.label_path, opt.output, opt.label_format,
               opt.label_only, opt.batch_size, opt.encoding, opt.lmdb_map_size)


if __name__ == '__main__':
    main()
