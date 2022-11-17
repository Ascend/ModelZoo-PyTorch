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

import numpy as np
from shapely.geometry import Polygon

from mmocr.utils import convert_annotations


def collect_level_info(annotation):
    """Collect information from any level in HierText.

    Args:
        annotation (dict): dict at each level

    Return:
        anno (dict): dict containing annotations
    """
    iscrowd = 0 if annotation['legible'] else 1
    vertices = np.array(annotation['vertices'])
    polygon = Polygon(vertices)
    area = polygon.area
    min_x, min_y, max_x, max_y = polygon.bounds
    bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
    segmentation = [i for j in vertices for i in j]
    anno = dict(
        iscrowd=iscrowd,
        category_id=1,
        bbox=bbox,
        area=area,
        segmentation=[segmentation])
    return anno


def collect_hiertext_info(root_path, level, split, print_every=1000):
    """Collect the annotation information.

    The annotation format is as the following:
    {
        "info": {
            "date": "release date",
            "version": "current version"
        },
        "annotations": [  // List of dictionaries, one for each image.
            {
            "image_id": "the filename of corresponding image.",
            "image_width": image_width,  // (int) The image width.
            "image_height": image_height, // (int) The image height.
            "paragraphs": [  // List of paragraphs.
                {
                "vertices": [[x1, y1], [x2, y2],...,[xn, yn]]
                "legible": true
                "lines": [
                    {
                    "vertices": [[x1, y1], [x2, y2],...,[x4, y4]]
                    "text": L
                    "legible": true,
                    "handwritten": false
                    "vertical": false,
                    "words": [
                        {
                        "vertices": [[x1, y1], [x2, y2],...,[xm, ym]]
                        "text": "the text content of this word",
                        "legible": true
                        "handwritten": false,
                        "vertical": false,
                        }, ...
                    ]
                    }, ...
                ]
                }, ...
            ]
            }, ...
        ]
        }

    Args:
        root_path (str): Root path to the dataset
        level (str): Level of annotations,  which should be 'word', 'line',
            or 'paragraphs'
        split (str): Dataset split, which should be 'train' or 'validation'
        print_every (int): Print log information per iter

    Returns:
        img_info (dict): The dict of the img and annotation information
    """

    annotation_path = osp.join(root_path, 'annotations/' + split + '.jsonl')
    if not osp.exists(annotation_path):
        raise Exception(
            f'{annotation_path} not exists, please check and try again.')

    annotation = json.load(open(annotation_path, 'r'))['annotations']
    img_infos = []
    for i, img_annos in enumerate(annotation):
        if i > 0 and i % print_every == 0:
            print(f'{i}/{len(annotation)}')
        img_info = {}
        img_info['file_name'] = img_annos['image_id'] + '.jpg'
        img_info['height'] = img_annos['image_height']
        img_info['width'] = img_annos['image_width']
        img_info['segm_file'] = annotation_path
        anno_info = []
        for paragraph in img_annos['paragraphs']:
            if level == 'paragraph':
                anno = collect_level_info(paragraph)
                anno_info.append(anno)
            elif level == 'line':
                for line in paragraph['lines']:
                    anno = collect_level_info(line)
                    anno_info.append(anno)
            elif level == 'word':
                for line in paragraph['lines']:
                    for word in line['words']:
                        anno = collect_level_info(line)
                        anno_info.append(anno)
        img_info.update(anno_info=anno_info)
        img_infos.append(img_info)
    return img_infos


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate training and validation set of HierText ')
    parser.add_argument('root_path', help='Root dir path of HierText')
    parser.add_argument(
        '--level',
        default='word',
        help='HierText provides three levels of annotation',
        choices=['word', 'line', 'paragraph'])
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    root_path = args.root_path
    print('Processing training set...')
    training_infos = collect_hiertext_info(root_path, args.level, 'train')
    convert_annotations(training_infos,
                        osp.join(root_path, 'instances_training.json'))
    print('Processing validation set...')
    val_infos = collect_hiertext_info(root_path, args.level, 'val')
    convert_annotations(val_infos, osp.join(root_path, 'instances_val.json'))
    print('Finish')


if __name__ == '__main__':
    main()
