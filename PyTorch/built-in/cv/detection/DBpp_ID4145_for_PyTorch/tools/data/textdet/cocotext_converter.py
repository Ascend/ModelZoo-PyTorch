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
        description='Generate training and validation set of COCO Text v2 ')
    parser.add_argument('root_path', help='Root dir path of COCO Text v2')
    args = parser.parse_args()
    return args


def collect_cocotext_info(root_path, split, print_every=1000):
    """Collect the annotation information.

    The annotation format is as the following:
    {
        'anns':{
            '45346':{
                'mask': [468.9,286.7,468.9,295.2,493.0,295.8,493.0,287.2],
                'class': 'machine printed',
                'bbox': [468.9, 286.7, 24.1, 9.1], # x, y, w, h
                'image_id': 217925,
                'id': 45346,
                'language': 'english', # 'english' or 'not english'
                'area': 206.06,
                'utf8_string': 'New',
                'legibility': 'legible', # 'legible' or 'illegible'
            },
            ...
        }
        'imgs':{
            '540965':{
                'id': 540965,
                'set': 'train', # 'train' or 'val'
                'width': 640,
                'height': 360,
                'file_name': 'COCO_train2014_000000540965.jpg'
            },
            ...
        }
        'imgToAnns':{
            '540965': [],
            '260932': [63993, 63994, 63995, 63996, 63997, 63998, 63999],
            ...
        }
    }

    Args:
        root_path (str): Root path to the dataset
        split (str): Dataset split, which should be 'train' or 'val'
        print_every (int): Print log information per iter

    Returns:
        img_info (dict): The dict of the img and annotation information
    """

    annotation_path = osp.join(root_path, 'annotations/cocotext.v2.json')
    if not osp.exists(annotation_path):
        raise Exception(
            f'{annotation_path} not exists, please check and try again.')

    annotation = mmcv.load(annotation_path)

    img_infos = []
    for i, img_info in enumerate(annotation['imgs'].values()):
        if i > 0 and i % print_every == 0:
            print(f'{i}/{len(annotation["imgs"].values())}')

        if img_info['set'] == split:
            img_info['segm_file'] = annotation_path
            ann_ids = annotation['imgToAnns'][str(img_info['id'])]
            # Filter out images without text
            if len(ann_ids) == 0:
                continue
            anno_info = []
            for ann_id in ann_ids:
                ann = annotation['anns'][str(ann_id)]

                # Ignore illegible or non-English words
                iscrowd = 1 if ann['language'] == 'not english' or ann[
                    'legibility'] == 'illegible' else 0

                x, y, w, h = ann['bbox']
                x, y = max(0, math.floor(x)), max(0, math.floor(y))
                w, h = math.ceil(w), math.ceil(h)
                bbox = [x, y, w, h]
                segmentation = [max(0, int(x)) for x in ann['mask']]
                if len(segmentation) < 8 or len(segmentation) % 2 != 0:
                    segmentation = [x, y, x + w, y, x + w, y + h, x, y + h]
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
    training_infos = collect_cocotext_info(root_path, 'train')
    convert_annotations(training_infos,
                        osp.join(root_path, 'instances_training.json'))
    print('Processing validation set...')
    val_infos = collect_cocotext_info(root_path, 'val')
    convert_annotations(val_infos, osp.join(root_path, 'instances_val.json'))
    print('Finish')


if __name__ == '__main__':
    main()
