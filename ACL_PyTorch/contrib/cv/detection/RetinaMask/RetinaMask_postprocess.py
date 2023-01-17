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
# limitations under the License

import os
import sys
import argparse
import numpy as np
import pycocotools.coco as coco
import pycocotools.mask as mask_util

cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path + "/..")

from tqdm import tqdm
from PIL import Image
from tools.utils import np_batched_nms, post_process, convert_to_coco_format, evaluate_prediction, Masker


def display_results(info_file, results_path, coco_path, fix_shape, output_path):
    with open(info_file, 'r') as f:
        infos = f.readlines()
    masker = Masker(threshold=0.5, padding=1)
    coco_dataset = coco.COCO(os.path.join(coco_path, 'annotations', 'instances_val2017.json'))
    data_list = []
    for info in tqdm(infos):
        name = info.split(' ')[1].split('/')[-1].split('.')[0]
        image_id = int(name)
        image_file = os.path.join(coco_path, 'val2017', name + '.jpg')
        img = Image.open(image_file).convert('RGB')
        ori_w, ori_h = img.size
        long_side = ori_w if ori_w > ori_h else ori_h
        ratio = fix_shape / long_side

        bboxes_file, labels_file, scores_file, masks_file = \
            name + '_0.bin', name + '_1.bin', name + '_2.bin', name + '_3.bin'
        bboxes = np.fromfile(os.path.join(results_path, bboxes_file), dtype=np.float32).reshape(-1, 4)
        labels = np.fromfile(os.path.join(results_path, labels_file), dtype=np.int32)
        scores = np.fromfile(os.path.join(results_path, scores_file), dtype=np.float32)
        masks = np.fromfile(os.path.join(results_path, masks_file), dtype=np.float32).reshape(-1, 1, 28, 28)

        bboxes, labels, scores, masks = np_batched_nms(bboxes, scores, labels, masks, iou_threshold=0.4)
        bboxes, labels, scores, masks = post_process(ori_h, ori_w, ratio, bboxes, labels, scores, masks)

        masks = masker(masks, bboxes, ori_h, ori_w)
        rles = [mask_util.encode(np.array(mask[0, :, :, np.newaxis], order="F").astype('uint8'))[0] for mask in masks]
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")

        data_list.extend(convert_to_coco_format(coco_dataset, image_id, bboxes, labels, scores, rles))

    bbox_info, segm_info = evaluate_prediction(coco_dataset, data_list)
    ttxxt = open(output_path, 'w') 
    print('bbox_info: \n', bbox_info, '\n', file=ttxxt)
    print('segm_info: \n', segm_info, '\n', file=ttxxt)
    ttxxt.close()


if __name__ == '__main__':
    # cur_path = os.path.abspath(os.path.dirname(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_text_path", type=str, default="../retinamask_coco2017.info")
    parser.add_argument("--infer_results_path", type=str, default="../result/dumpOutput_device0")
    parser.add_argument("--fix_shape", type=int, default=1344)
    parser.add_argument("--coco_path", type=str, default="/opt/npu/coco")
    parser.add_argument("--output_path", type=str, default="../evaluation_results.txt")

    args = parser.parse_args()

    display_results(args.input_text_path, args.infer_results_path, args.coco_path, args.fix_shape, args.output_path)
