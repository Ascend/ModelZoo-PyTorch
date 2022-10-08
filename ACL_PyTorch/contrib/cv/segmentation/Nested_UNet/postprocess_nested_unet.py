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

def read_info_from_json(json_path):
    '''
    此函数用于读取inference_tools生成的json文件
    input: json文件地址
    output: dict结构: 为原始的json转换出来的结构
    '''
    if os.path.exists(json_path) is False:
        print(json_path, 'is not exist')
    with open(json_path, 'r') as f:
        load_data = json.load(f)
        file_info = load_data['filesinfo']
        return file_info

def main():

    file_info = read_info_from_json(result_json_path)

    avg_meter = AverageMeter()

    for i in file_info.items():
        # 获取推理结果文件地址
        result_file_name = os.path.basename(i[1]['outfiles'][0])
        # 使用result_dir的路径作为结果文件的路径，可以使得运行该脚本的路径更通用
        res_path = os.path.join(result_dir_path, result_file_name)
        # 获取对应的标签
        label_id = os.path.splitext(os.path.basename(i[1]['infiles'][0]))[0]

        result = np.fromfile(res_path, dtype='float32')
        result = np.reshape(result, (1, 96, 96))

        mask = cv2.imread(os.path.join(mask_path, label_id + '.png'))
        mask = mask.astype('float32') / 255

        mask = mask.transpose(2, 0, 1)[0]
        iou = iou_score(result, mask)
        avg_meter.update(iou)

    print('IoU: %.4f' % avg_meter.avg)


if __name__ == "__main__":
    result_json_path = sys.argv[1]
    mask_path = sys.argv[2]
    result_dir_path = os.path.dirname(result_json_path)
    main()
