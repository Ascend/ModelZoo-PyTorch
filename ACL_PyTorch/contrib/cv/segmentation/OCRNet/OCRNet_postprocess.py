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
import numpy as np
import torch
from tqdm import tqdm
import argparse

import paddle
import paddle.nn.functional as F

import sys
sys.path.append('PaddleSeg')
from paddleseg.utils import metrics

def parse_args():
    parser = argparse.ArgumentParser(description='preprocess for OCRNet mocel')
    parser.add_argument('--pred_path', default="preds")
    parser.add_argument('--bin_file_path', default="cityscapes_bin", help='cityscapes processed by preprocessor')
    return parser.parse_args()

def read_files(preds_path, labels_path):
    pred_list = [os.path.join(preds_path, file) for file in os.listdir(preds_path)]
    label_list = [os.path.join(labels_path, label) for label in os.listdir(labels_path)]
    files = [{
        "id": int(file.split('_')[1][3:]), 
        "pred_path": os.path.join(preds_path, file),
        "label_path": os.path.join(labels_path, "label_bin"+file.split('_')[1][3:]+".bin" )}
        for file in os.listdir(preds_path) 
    ]
    return files
  

def get_pred(pred_path):
    pred = []
    with open(pred_path) as file:
        for line in file.readlines():
            pred.append(line.strip().split(' '))
        pred = np.asarray(pred).astype(np.int64)
        pred = paddle.to_tensor(pred)
        pred = pred.reshape((-1, 1024, 2048))

    return pred

  
def get_label(label_path):
    label = np.fromfile(label_path, dtype=np.int32).astype(np.int64)
    label = label.reshape((-1, 1024, 2048))
    label = paddle.to_tensor(label)
    return label

def main(args):
    preds_path = args.pred_path
    labels_path = os.path.join(args.bin_file_path, 'labels')

    files = read_files(preds_path, labels_path)


    num_classes = 19
    ignore_index = 255
    intersect_area_all = 0
    pred_area_all = 0
    label_area_all = 0
    for i, item in tqdm(enumerate(files)):
        pred_path = item["pred_path"]
        label_path = item["label_path"]
        pred = get_pred(pred_path)
        label = get_label(label_path)
        intersect_area, pred_area, label_area = metrics.calculate_area(
            pred,
            label,
            num_classes,
            ignore_index)
        intersect_area_all = intersect_area_all + intersect_area
        pred_area_all = pred_area_all + pred_area
        label_area_all = label_area_all + label_area
    class_iou, miou = metrics.mean_iou(intersect_area_all, pred_area_all,
                                   label_area_all)
    class_acc, acc = metrics.accuracy(intersect_area_all, pred_area_all)
    kappa = metrics.kappa(intersect_area_all, pred_area_all, label_area_all)
    print("[EVAL] #mIoU: {:.4f} Acc: {:.4f} Kappa: {:.4f} ".format(
        miou, acc, kappa))
    print("[EVAL] Class IoU: \n" + str(np.round(class_iou, 4)))
    print("[EVAL] Class Acc: \n" + str(np.round(class_acc, 4)))


if __name__ == '__main__':
    args = parse_args()
    main(args)
