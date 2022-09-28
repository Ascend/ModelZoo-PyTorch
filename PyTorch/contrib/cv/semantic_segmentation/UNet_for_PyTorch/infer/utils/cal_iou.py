# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
# ============================================================================
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
import numpy as np
import cv2
from glob import glob
import os
import argparse

import sys
sys.path.append('..')
from sdk.metrics import iou_score


def parse_args():
    parser = argparse.ArgumentParser(description="ENET process")
     # dataset
    parser.add_argument('--dataset', default='inputs/dsb2018_96', help='dataset dir')
    parser.add_argument('--bin_result', default='../mxbase/bin_result', help='bin_result dir')

    args_opt = parser.parse_args()
    return args_opt

class AverageMeter(object):#括号表示继承
    """Computes and stores the average and current value""" 
    def __init__(self, name, fmt=':f', start_count_index=0):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.start_count_index = start_count_index

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0


    # iou.update(iou_now, input.size(0))
    def update(self, val, n=1):
        if self.count == 0:
            self.N = n

        self.val = val
        self.count += n
        
        if self.count > (self.start_count_index * self.N):
            self.sum += val * n
            self.avg = self.sum / (self.count - self.start_count_index * self.N)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)         

def getMask(dir, class_num, img_id):
    mask = []
    for i in range(class_num):
        mask.append(cv2.imread(os.path.join(dir, str(i), img_id + ".png"), cv2.IMREAD_GRAYSCALE)[..., None])
    mask = np.dstack(mask)

    return mask


if __name__ == '__main__':
    class_num = 1

    args = parse_args()
     # Data loading code
    img_ids = glob(os.path.join(args.dataset, 'images', '*' + '.png'))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    mask_dir = os.path.join(args.dataset, 'masks')

    if not img_ids:
        raise RuntimeError(
            "Found 0 images in subfolders of:" + args.dataset + "\n")

    iou = AverageMeter('Iou', ':6.4f')

    for img_id in img_ids:
        print("Processing ---> ", img_id)

        mask_label = getMask(mask_dir, class_num, img_id)
        predict_path = os.path.join(args.bin_result, img_id + "_0.bin")
        predict_result = np.fromfile(predict_path, dtype=np.float32)      
        predict_result = predict_result.reshape(1, 1, 96, 96)
        mask_label = mask_label.astype(np.float32) / 255.0
        mask_label = mask_label.transpose(2, 0, 1)
        mask_label = np.expand_dims(mask_label, 0)  # NHW


        iou_now = iou_score(predict_result, mask_label)
        iou.update(iou_now, mask_label.shape[0])
     
    print('[AVG-IOU] * Iou {iou.avg:.4f}'
                  .format(iou=iou))
    print("val finished....")
