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
    """Convert original annotations to mmocr format.

    The annotation format is as the following:
        Crops/val/11/1/1.png weighted
        Crops/val/11/1/2.png 26
        Crops/val/11/1/3.png casting
        Crops/val/11/1/4.png 28
    After this module, the annotation has been changed to the format below:
        jsonl:
        {'filename': 'Crops/val/11/1/1.png', 'text': 'weighted'}
        {'filename': 'Crops/val/11/1/1.png', 'text': '26'}
        {'filename': 'Crops/val/11/1/1.png', 'text': 'casting'}
        {'filename': 'Crops/val/11/1/1.png', 'text': '28'}

    Args:
        root_path (str): The root path of the dataset
        split (str): The split of dataset. Namely: training or test
        format (str): Annotation format, should be either 'txt' or 'jsonl'
    """
    assert isinstance(root_path, str)
    assert isinstance(split, str)

    if format == 'txt':  # LV has already provided txt format annos
        return

    if format == 'jsonl':
        lines = []
        with open(
                osp.join(root_path, f'{split}_label.txt'),
                'r',
                encoding='"utf-8-sig') as f:
            annos = f.readlines()
        for anno in annos:
            if anno:
                # Text may contain spaces
                dst_img_name, word = anno.split('png ')
                word = word.strip('\n')
                lines.append(
                    json.dumps({
                        'filename': dst_img_name + 'png',
                        'text': word
                    }))
    else:
        raise NotImplementedError

    list_to_file(osp.join(root_path, f'{split}_label.{format}'), lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate training and test set of Lecture Video DB')
    parser.add_argument('root_path', help='Root dir path of Lecture Video DB')
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

    for split in ['train', 'val', 'test']:
        convert_annotations(root_path, split, args.format)
        print(f'{split} split converted.')


if __name__ == '__main__':
    main()
