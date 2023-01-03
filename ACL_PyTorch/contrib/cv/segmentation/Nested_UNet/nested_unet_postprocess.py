# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
import sys
from glob import glob
import json
import cv2
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def iou_score(output, target):
    smooth = 1e-5
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)

def main():

    file_names = os.listdir(result_dir_path)
    avg_meter = AverageMeter()

    for file_name in file_names:
        res_path = os.path.join(result_dir_path, file_name)
        result = np.fromfile(res_path, dtype='float32')
        result = np.reshape(result, (1, 96, 96))

        mask = cv2.imread(os.path.join(mask_path, file_name.split('_')[0] + '.png'))
        mask = mask.astype('float32') / 255

        mask = mask.transpose(2, 0, 1)[0]
        iou = iou_score(result, mask)
        avg_meter.update(iou)

    print('IoU: %.4f' % avg_meter.avg)


if __name__ == "__main__":
    result_dir_path = sys.argv[1]
    mask_path = sys.argv[2]
    main()
