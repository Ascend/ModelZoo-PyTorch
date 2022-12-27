# Copyright 2022 Huawei Technologies Co., Ltd
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
import os
import argparse
import glob
import numpy as np
import cv2
import torch
from mmdet.core import bbox2result
from mmdet.datasets import CocoDataset


def postprocess_bboxes(bboxes, image_size, net_input_width, net_input_height):
    org_w = image_size[0]
    org_h = image_size[1]
    scale = min(net_input_width / org_w, net_input_height / org_h)
    bboxes[:, 0] = (bboxes[:, 0]) / scale
    bboxes[:, 1] = (bboxes[:, 1]) / scale
    bboxes[:, 2] = (bboxes[:, 2]) / scale
    bboxes[:, 3] = (bboxes[:, 3]) / scale
    return bboxes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_annotation", default="./origin_pictures.info")
    parser.add_argument("--bin_data_path", default="./result/dumpOutput_device0")
    parser.add_argument("--net_out_num", type=int, default=2)
    parser.add_argument("--net_input_width", type=int, default=1216)
    parser.add_argument("--net_input_height", type=int, default=800)
    parser.add_argument("--annotations_path", default="/root/datasets")
    flags = parser.parse_args()

    img_size_dict = dict()
    with open(flags.test_annotation)as f:
        for line in f.readlines():
            temp = line.split(" ")
            img_file_path = temp[1]
            img_name = temp[1].split("/")[-1].split(".")[0]
            img_width = int(temp[2])
            img_height = int(temp[3])
            img_size_dict[img_name] = (img_width, img_height, img_file_path)

    bin_path = flags.bin_data_path

    coco_dataset = CocoDataset(ann_file='{}/coco/annotations/instances_val2017.json'.format(flags.annotations_path), pipeline=[])
    coco_class_map = {id:name for id, name in enumerate(coco_dataset.CLASSES)}
    results = []
    cnt = 0
    for ids in coco_dataset.img_ids:
        cnt = cnt + 1
        bin_file = glob.glob(f'{bin_path}/*0{str(ids)}_0.txt')[0]
        bin_file = bin_file[bin_file.rfind('/') + 1:]
        bin_file = bin_file[:bin_file.find('_')]
        print(cnt - 1, bin_file)
        path_base = os.path.join(bin_path, bin_file)

        res_buff = []
        bbox_results = []
        cls_segms = []
        if os.path.exists(f'{path_base}_0.txt') and os.path.exists(f'{path_base}_1.txt'):
            bboxes = np.loadtxt(f'{path_base}_{str(flags.net_out_num - 2)}.txt', dtype="float32")
            bboxes = np.reshape(bboxes, [100, 5])
            bboxes = torch.from_numpy(bboxes)
            labels = np.loadtxt(f'{path_base}_{str(flags.net_out_num - 1)}.txt', dtype="int64")
            labels = np.reshape(labels, [100, 1])
            labels = torch.from_numpy(labels)

            img_shape = (flags.net_input_height, flags.net_input_width)
            bboxes = postprocess_bboxes(bboxes, img_size_dict[bin_file], flags.net_input_width, flags.net_input_height)
            bbox_results = [bbox2result(bboxes, labels[:, 0], 80)]
        else:
            print("[ERROR] file not exist", f'{path_base}_{str(0)}.txt', f'{path_base}_ {str(1)}.txt')

        result = bbox_results
        results.extend(result)

    eval_results = coco_dataset.evaluate(results, metric=['bbox', ], classwise=True)
