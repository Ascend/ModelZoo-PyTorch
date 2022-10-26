# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import os

from pprint import pprint
from pathlib import Path
import numpy as np
import torch

from mmocr.models.builder import build_convertor
from mmocr.core.evaluation.ocr_metric import eval_ocr_metric


def parse_args():
    parser = argparse.ArgumentParser(
                        description='postprocess.')
    parser.add_argument('--result-dir', type=str, required=True,
                        help='output directory of inferencing.')
    parser.add_argument('--gt-path', type=str, required=True,
                        help='path to groundtruth file.')
    args = parser.parse_args()
    return args


def get_pred_texts(result_dir):

    label_convertor = build_convertor({
        'type': 'AttnConvertor', 'dict_type': 'DICT90',
        'with_unknown': True, 'max_seq_len': 25
    })

    result_files = [
        res_path.__str__() for res_path in Path(result_dir).iterdir()
    ]
    result_files.sort()

    pred_texts = []
    file_stems = []
    for res_path in result_files:
        stem = Path(res_path).name.replace('_0.bin', '')
        if stem.startswith('padding') or stem.startswith('sumary'):
            continue
        result = np.fromfile(res_path, np.float32).reshape(1, 25, 92)
        result = torch.from_numpy(result)

        label_indexes, label_scores = label_convertor.tensor2idx(result)
        label_strings = label_convertor.idx2str(label_indexes)

        pred_texts.extend(label_strings)
        file_stems.append(stem)

    return pred_texts, file_stems


def get_gt_texts(gt_path, file_stems):

    gt_dict = {}
    for line in open(gt_path, 'r', encoding='utf-8'):
        img_path, text = line.strip().split(' ', 1)
        gt_dict[Path(img_path).stem] = text
    
    gt_texts = [gt_dict[stem] for stem in file_stems if stem in gt_dict]
    assert len(gt_texts) == len(file_stems)

    return gt_texts


def evaluate(result_dir, gt_path, out_path=None):

    pred_texts, file_stems = get_pred_texts(result_dir)
    gt_texts = get_gt_texts(gt_path, file_stems)
    eval_results = eval_ocr_metric(pred_texts, gt_texts, metric='acc')

    pprint(eval_results)


if __name__ == '__main__':
    args = parse_args()
    evaluate(args.result_dir, args.gt_path)
