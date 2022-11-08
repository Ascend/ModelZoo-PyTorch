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
import os
import os.path as osp
import xml.etree.ElementTree as ET

import cv2

from mmocr.utils.fileio import list_to_file


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate testset of svt by cropping box image.')
    parser.add_argument(
        'root_path',
        help='Root dir path of svt, where test.xml in,'
        'for example, "data/mixture/svt/svt1/"')
    parser.add_argument(
        '--resize',
        action='store_true',
        help='Whether resize cropped image to certain size.')
    parser.add_argument('--height', default=32, help='Resize height.')
    parser.add_argument('--width', default=100, help='Resize width.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    root_path = args.root_path

    # inputs
    src_label_file = osp.join(root_path, 'test.xml')
    if not osp.exists(src_label_file):
        raise Exception(
            f'{src_label_file} not exists, please check and try again.')
    src_image_root = root_path

    # outputs
    dst_label_file = osp.join(root_path, 'test_label.txt')
    dst_image_root = osp.join(root_path, 'image')
    os.makedirs(dst_image_root, exist_ok=True)

    tree = ET.parse(src_label_file)
    root = tree.getroot()

    index = 1
    lines = []
    total_img_num = len(root)
    i = 1
    for image_node in root.findall('image'):
        image_name = image_node.find('imageName').text
        print(f'[{i}/{total_img_num}] Process image: {image_name}')
        i += 1
        lexicon = image_node.find('lex').text.lower()
        lexicon_list = lexicon.split(',')
        lex_size = len(lexicon_list)
        src_img = cv2.imread(osp.join(src_image_root, image_name))
        for rectangle in image_node.find('taggedRectangles'):
            x = int(rectangle.get('x'))
            y = int(rectangle.get('y'))
            w = int(rectangle.get('width'))
            h = int(rectangle.get('height'))
            rb, re = max(0, y), max(0, y + h)
            cb, ce = max(0, x), max(0, x + w)
            dst_img = src_img[rb:re, cb:ce]
            text_label = rectangle.find('tag').text.lower()
            if args.resize:
                dst_img = cv2.resize(dst_img, (args.width, args.height))
            dst_img_name = f'img_{index:04}' + '.jpg'
            index += 1
            dst_img_path = osp.join(dst_image_root, dst_img_name)
            cv2.imwrite(dst_img_path, dst_img)
            lines.append(f'{osp.basename(dst_image_root)}/{dst_img_name} '
                         f'{text_label} {lex_size} {lexicon}')
    list_to_file(dst_label_file, lines)
    print(f'Finish to generate svt testset, '
          f'with label file {dst_label_file}')


if __name__ == '__main__':
    main()
