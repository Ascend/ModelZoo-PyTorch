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

import cv2

from mmocr.utils import list_from_file, list_to_file


def parse_old_label(data_root, in_path, img_size=False):
    imgid2imgname = {}
    imgid2anno = {}
    idx = 0
    for line in list_from_file(in_path):
        line = line.strip().split()
        img_full_path = osp.join(data_root, line[0])
        if not osp.exists(img_full_path):
            continue
        ann_file = osp.join(data_root, line[1])
        if not osp.exists(ann_file):
            continue

        img_info = {}
        img_info['file_name'] = line[0]
        if img_size:
            img = cv2.imread(img_full_path)
            h, w = img.shape[:2]
            img_info['height'] = h
            img_info['width'] = w
        imgid2imgname[idx] = img_info

        imgid2anno[idx] = []
        char_annos = []
        for t, ann_line in enumerate(list_from_file(ann_file)):
            ann_line = ann_line.strip()
            if t == 0:
                img_info['text'] = ann_line
            else:
                char_box = [float(x) for x in ann_line.split()]
                char_text = img_info['text'][t - 1]
                char_ann = dict(char_box=char_box, char_text=char_text)
                char_annos.append(char_ann)
        imgid2anno[idx] = char_annos
        idx += 1

    return imgid2imgname, imgid2anno


def gen_line_dict_file(out_path, imgid2imgname, imgid2anno, img_size=False):
    lines = []
    for key, value in imgid2imgname.items():
        if key in imgid2anno:
            anno = imgid2anno[key]
            line_dict = {}
            line_dict['file_name'] = value['file_name']
            line_dict['text'] = value['text']
            if img_size:
                line_dict['height'] = value['height']
                line_dict['width'] = value['width']
            line_dict['annotations'] = anno
            lines.append(json.dumps(line_dict))
    list_to_file(out_path, lines)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-root', help='data root for both image file and anno file')
    parser.add_argument(
        '--in-path',
        help='mapping file of image_name and ann_file,'
        ' "image_name ann_file" in each line')
    parser.add_argument(
        '--out-path', help='output txt path with line-json format')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    imgid2imgname, imgid2anno = parse_old_label(args.data_root, args.in_path)
    gen_line_dict_file(args.out_path, imgid2imgname, imgid2anno)
    print('finish')


if __name__ == '__main__':
    main()
