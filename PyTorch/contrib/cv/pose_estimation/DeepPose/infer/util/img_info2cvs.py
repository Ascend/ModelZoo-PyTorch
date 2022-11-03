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

import csv  
import os
import argparse
import sys
sys.path.append("../sdk")
from preprocess import load_coco_person_detection_results
from preprocess import get_mapping_id_name
from main import mkdir_or_exist
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval

def parse_args():
    parser = argparse.ArgumentParser(description='mmpose t')
    parser.add_argument('--data_root',default = '../../../coco2017' ,help='path of root')
    parser.add_argument('--out_dir',default = './' ,help='csv output dir')
    args = parser.parse_args()
    return args   

if __name__ == '__main__':
    args = parse_args()
    ann_file = os.path.join(args.data_root, 'annotations/person_keypoints_val2017.json')
    coco = COCO(ann_file)
    id2name, name2id = get_mapping_id_name(coco.imgs)
    db = load_coco_person_detection_results(args.data_root, id2name)
    mkdir_or_exist(args.out_dir)
    output_file = os.path.join(args.out_dir, 'info.csv')
    with open(output_file, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        for idx in range(len(db)):
            img_name = os.path.basename(db[idx]['image_file'])
            c = db[idx]['center']
            s = db[idx]['scale']
            bbox_id = db[idx]['bbox_id']
            writer.writerow([img_name, c[0], c[1], s[0], s[1], bbox_id])
        f.close()
