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
import json
import os.path as osp

from mmocr.utils.fileio import list_to_file


def convert_annotations(root_path, split, format):
    """Convert original annotations to mmocr format
    The annotation format is as the following:
        word_1.png, "flying"
        word_2.png, "today"
        word_3.png, "means"
    After this module, the annotation has been changed to the format below:
        txt:
        word_1.png flying
        word_2.png today
        word_3.png means

        jsonl:
        {'filename': 'word_1.png', 'text': 'flying'}
        {'filename': 'word_2.png', 'text': 'today'}
        {'filename': 'word_3.png', 'text': 'means'}

    Args:
        root_path (str): The root path of the dataset
        split (str): The split of dataset. Namely: training or test
        format (str): Annotation format, should be either 'txt' or 'jsonl'

    """
    assert isinstance(root_path, str)
    assert isinstance(split, str)

    lines = []
    with open(
            osp.join(root_path, 'annotations',
                     f'Challenge2_{split}_Task3_GT.txt'),
            'r',
            encoding='"utf-8-sig') as f:
        annos = f.readlines()
    dst_image_root = osp.join(root_path, split.lower())
    for anno in annos:
        # text may contain comma ','
        dst_img_name, word = anno.split(', "')
        word = word.replace('"\n', '')

        if format == 'txt':
            lines.append(f'{osp.basename(dst_image_root)}/{dst_img_name} '
                         f'{word}')
        elif format == 'jsonl':
            lines.append(
                json.dumps({
                    'filename':
                    f'{osp.basename(dst_image_root)}/{dst_img_name}',
                    'text': word
                }))
        else:
            raise NotImplementedError

    list_to_file(osp.join(root_path, f'{split.lower()}_label.{format}'), lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate training and test set of IC13')
    parser.add_argument('root_path', help='Root dir path of IC13')
    parser.add_argument(
        '--format',
        default='jsonl',
        help='Use jsonl or string to format annotations',
        choices=['jsonl', 'txt'])
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    root_path = args.root_path

    for split in ['Train', 'Test']:
        convert_annotations(root_path, split, args.format)
        print(f'{split} split converted.')


if __name__ == '__main__':
    main()
