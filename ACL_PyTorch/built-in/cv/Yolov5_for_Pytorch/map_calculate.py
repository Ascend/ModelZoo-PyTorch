# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import argparse
from collections import OrderedDict


def eval(flags):
    annType = ['segm', 'bbox', 'keypoints']
    annType = annType[1]  # specify type here
    print('Start evaluate *%s* results...' % (annType))
    cocoGt_file = flags.ground_truth_json
    cocoDt_file = flags.detection_results_json
    cocoGt = COCO(cocoGt_file)
    cocoDt = cocoGt.loadRes(cocoDt_file)
    imgIds = cocoGt.getImgIds()
    print('get %d images' % len(imgIds))
    imgIds = sorted(imgIds)
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    # copy-paste style
    eval_results = OrderedDict()
    metric = annType
    metric_items = [
        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
    ]
    coco_metric_names = {
        'mAP': 0,
        'mAP_50': 1,
        'mAP_75': 2,
        'mAP_s': 3,
        'mAP_m': 4,
        'mAP_l': 5,
        'AR@100': 6,
        'AR@300': 7,
        'AR@1000': 8,
        'AR_s@1000': 9,
        'AR_m@1000': 10,
        'AR_l@1000': 11
    }

    for metric_item in metric_items:
        key = f'{metric}_{metric_item}'
        val = float(
            f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
        )
        eval_results[key] = val
    ap = cocoEval.stats[:6]
    eval_results[f'{metric}_mAP_copypaste'] = (
        f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} {ap[4]:.3f} {ap[5]:.3f}'
    )
    print(dict(eval_results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_truth_json", type=str, default="./instances_val2017.json")
    parser.add_argument("--detection_results_json", type=str, default="./predictions.json")
    flags = parser.parse_args()
    eval(flags)
