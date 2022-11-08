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
from mmdet.datasets.builder import DATASETS

import mmocr.utils as utils
from mmocr.datasets.ocr_dataset import OCRDataset


@DATASETS.register_module()
class OCRSegDataset(OCRDataset):

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix

    def _parse_anno_info(self, annotations):
        """Parse char boxes annotations.
        Args:
            annotations (list[dict]): Annotations of one image, where
                each dict is for one character.

        Returns:
            dict: A dict containing the following keys:

                - chars (list[str]): List of character strings.
                - char_rects (list[list[float]]): List of char box, with each
                    in style of rectangle: [x_min, y_min, x_max, y_max].
                - char_quads (list[list[float]]): List of char box, with each
                    in style of quadrangle: [x1, y1, x2, y2, x3, y3, x4, y4].
        """

        assert utils.is_type_list(annotations, dict)
        assert 'char_box' in annotations[0]
        assert 'char_text' in annotations[0]
        assert len(annotations[0]['char_box']) in [4, 8]

        chars, char_rects, char_quads = [], [], []
        for ann in annotations:
            char_box = ann['char_box']
            if len(char_box) == 4:
                char_box_type = ann.get('char_box_type', 'xyxy')
                if char_box_type == 'xyxy':
                    char_rects.append(char_box)
                    char_quads.append([
                        char_box[0], char_box[1], char_box[2], char_box[1],
                        char_box[2], char_box[3], char_box[0], char_box[3]
                    ])
                elif char_box_type == 'xywh':
                    x1, y1, w, h = char_box
                    x2 = x1 + w
                    y2 = y1 + h
                    char_rects.append([x1, y1, x2, y2])
                    char_quads.append([x1, y1, x2, y1, x2, y2, x1, y2])
                else:
                    raise ValueError(f'invalid char_box_type {char_box_type}')
            elif len(char_box) == 8:
                x_list, y_list = [], []
                for i in range(4):
                    x_list.append(char_box[2 * i])
                    y_list.append(char_box[2 * i + 1])
                x_max, x_min = max(x_list), min(x_list)
                y_max, y_min = max(y_list), min(y_list)
                char_rects.append([x_min, y_min, x_max, y_max])
                char_quads.append(char_box)
            else:
                raise Exception(
                    f'invalid num in char box: {len(char_box)} not in (4, 8)')
            chars.append(ann['char_text'])

        ann = dict(chars=chars, char_rects=char_rects, char_quads=char_quads)

        return ann

    def prepare_train_img(self, index):
        """Get training data and annotations from pipeline.

        Args:
            index (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """
        img_ann_info = self.data_infos[index]
        img_info = {
            'filename': img_ann_info['file_name'],
        }
        ann_info = self._parse_anno_info(img_ann_info['annotations'])
        results = dict(img_info=img_info, ann_info=ann_info)

        self.pre_pipeline(results)

        return self.pipeline(results)
