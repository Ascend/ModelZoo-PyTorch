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
import math
import os.path as osp

import mmcv

from mmocr.utils import convert_annotations


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate training and validation set of TextOCR ')
    parser.add_argument('root_path', help='Root dir path of TextOCR')
    args = parser.parse_args()
    return args


def collect_textocr_info(root_path, annotation_filename, print_every=1000):

    annotation_path = osp.join(root_path, annotation_filename)
    if not osp.exists(annotation_path):
        raise Exception(
            f'{annotation_path} not exists, please check and try again.')

    annotation = mmcv.load(annotation_path)

    # img_idx = img_start_idx
    img_infos = []
    for i, img_info in enumerate(annotation['imgs'].values()):
        if i > 0 and i % print_every == 0:
            print(f'{i}/{len(annotation["imgs"].values())}')

        img_info['segm_file'] = annotation_path
        ann_ids = annotation['imgToAnns'][img_info['id']]
        anno_info = []
        for ann_id in ann_ids:
            ann = annotation['anns'][ann_id]

            # Ignore illegible or non-English words
            text_label = ann['utf8_string']
            iscrowd = 1 if text_label == '.' else 0

            x, y, w, h = ann['bbox']
            x, y = max(0, math.floor(x)), max(0, math.floor(y))
            w, h = math.ceil(w), math.ceil(h)
            bbox = [x, y, w, h]
            segmentation = [max(0, int(x)) for x in ann['points']]
            anno = dict(
                iscrowd=iscrowd,
                category_id=1,
                bbox=bbox,
                area=ann['area'],
                segmentation=[segmentation])
            anno_info.append(anno)
        img_info.update(anno_info=anno_info)
        img_infos.append(img_info)
    return img_infos


def main():
    args = parse_args()
    root_path = args.root_path
    print('Processing training set...')
    training_infos = collect_textocr_info(root_path, 'TextOCR_0.1_train.json')
    convert_annotations(training_infos,
                        osp.join(root_path, 'instances_training.json'))
    print('Processing validation set...')
    val_infos = collect_textocr_info(root_path, 'TextOCR_0.1_val.json')
    convert_annotations(val_infos, osp.join(root_path, 'instances_val.json'))
    print('Finish')


if __name__ == '__main__':
    main()
