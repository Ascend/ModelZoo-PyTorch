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
# ============================================================================
import os
import numpy as np
import cv2
import os
import argparse
import json_tricks as json
import sys
sys.path.append("../sdk")
from preprocess import data_cfg,load_coco_person_detection_results,get_mapping_id_name
from main import mkdir_or_exist
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval
from collections import defaultdict
from collections import OrderedDict
from cal_accuracy import evaluate
from postprocess import process_result

def parse_args():
    parser = argparse.ArgumentParser(description="ENET process")
     # dataset
    parser.add_argument('--data_root',default = '/home/dataset/coco2017' ,help='path of root')
    parser.add_argument('--result_bin', default='../mxbase/result_bin', help='bin_result dir')
    parser.add_argument('--out', default = './output', help='output result file')
    parser.add_argument(
        '--eval',
        default=None,
        nargs='+',
        help='evaluation metric, which depends on the dataset,'
        ' e.g., "mAP" for MSCOCO')

    args_opt = parser.parse_args()
    return args_opt

if __name__ == '__main__':
    args = parse_args()
    ann_file = os.path.join(args.data_root, 'annotations/person_keypoints_val2017.json')
    coco = COCO(ann_file)
    mkdir_or_exist(args.out)
    id2name, name2id = get_mapping_id_name(coco.imgs)
    db = load_coco_person_detection_results(args.data_root, id2name)
    outputs = [] # predictions
    length = len(db)
    for idx in range(length):
        print("------{}",idx/length)
        kpt = db[idx]
        img_full_name = kpt['image_file']
        c = kpt['center']
        s = kpt['scale']
        r = kpt['rotation'] # rotation
        score = kpt['bbox_score']
        bbox_id = kpt['bbox_id']
        img_name = os.path.basename(img_full_name)
        pre_bin = os.path.join(args.result_bin, str(bbox_id)+img_name+"._bin")
        output_heatmap = np.fromfile(pre_bin, dtype=np.float32).reshape(1, 17, 2)      
        output = process_result(output_heatmap, c, s, data_cfg['image_size'], score, bbox_id, img_full_name)
        outputs.append(output)
    results = evaluate(outputs, args.out, data_cfg, name2id, coco, args)
    for k, v in sorted(results.items()):
        print(f'{k}: {v}')
