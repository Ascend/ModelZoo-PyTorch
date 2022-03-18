# Copyright 2020 Huawei Technologies Co., Ltd
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

import cv2
import torch
from torch.nn import functional as F
import numpy as np
import time
import json
from tqdm import tqdm


def is_label(filename):
    return filename.endswith("_labelIds.png")

def convert_label(label, inverse=False):
    temp = label.copy()
    label_mapping = {-1: 255, 0: 255, 
                    1: 255, 2: 255, 
                    3: 255, 4: 255, 
                    5: 255, 6: 255, 
                    7: 0, 8: 1, 9: 255, 
                    10: 255, 11: 2, 12: 3, 
                    13: 4, 14: 255, 15: 255, 
                    16: 255, 17: 5, 18: 255, 
                    19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                    25: 12, 26: 13, 27: 14, 28: 15, 
                    29: 255, 30: 255, 
                    31: 16, 32: 17, 33: 18}
    if inverse:
        for v, k in label_mapping.items():
            label[temp == k] = v
    else:
        for k, v in label_mapping.items():
            label[temp == k] = v
    return label

def main():

    writer = open(os.path.join(result_json_path, json_file_name), 'w')
    table_dict = {}
    table_dict["title"] = "overall statistical evaluation"

    result1_ids = glob(os.path.join(result_dir, '*' + '1.bin'))
    result1_ids = [os.path.splitext(os.path.basename(p))[0] for p in result1_ids]
    result1_ids.sort()
    # print('result1_ids\'length:', len(result1_ids))

    result2_ids = glob(os.path.join(result_dir, '*' + '2.bin'))
    result2_ids = [os.path.splitext(os.path.basename(p))[0] for p in result2_ids]
    result2_ids.sort()
    # print('result2_ids\'length:', len(result2_ids))

    gt_ids = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(gt_dir)) for f in fn if is_label(f)]
    gt_ids.sort()
    # print('gt_ids\'length:' , len(gt_ids))

    start = time.time()

    confusion_matrix = np.zeros(
        (19, 19, 2))
    
    for index in tqdm(range(len(result1_ids))):
        # print("now is image ", index)

        # output1:out_aux
        file1path = os.path.join(result_dir, result1_ids[index] + '.bin')
        size = os.path.getsize(file1path)
        result1 = np.fromfile(file1path, dtype='float32')
        out_aux = np.array(result1).reshape(19, 256, 512)
        out_aux = torch.tensor(out_aux, dtype=torch.float32) 
        out_aux = out_aux.unsqueeze(0)

        # output2: out
        file2path = os.path.join(result_dir, result2_ids[index] + '.bin')
        size = os.path.getsize(file2path)
        result2 = np.fromfile(file2path, dtype='float32')
        out = np.array(result2).reshape(19, 256, 512)
        out = torch.tensor(out, dtype=torch.float32) 
        out = out.unsqueeze(0)

        pred=[]
        pred.append(out_aux)
        pred.append(out)

        # label
        # print(gt_ids[index])
        label = cv2.imread(gt_ids[index], cv2.IMREAD_GRAYSCALE)
        label = convert_label(label)
        label = np.array(label).astype('int32') # 1024*2048
        label = torch.tensor(label).unsqueeze(0)
        size = label.size()

        for i, x in enumerate(pred):
            x = F.interpolate(
                input=x, size=size[-2:],
                mode='bilinear', align_corners=True
            )

            confusion_matrix[..., i] += get_confusion_matrix(
                label, x, size, 19, 255)

    # 计算mIou
    nums=2
    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        if (i == 1):
            print('IoU_array: {}\n mean_IoU: {}'.format(IoU_array, mean_IoU))
            table_dict["value"] = {"Iou_array": IoU_array.tolist(), "mean_Iou": mean_IoU.tolist()}
            json.dump(table_dict, writer)

    writer.close()

    print('total time: {}s'.format(time.time()-start))


def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(
        label.cpu().numpy()[:, :size[-2], :size[-1]], dtype='int')

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix


if __name__ == "__main__":
    result_dir = sys.argv[1]  # "result/dumpOutput_device0/"
    gt_dir = sys.argv[2]  # "/root/jyf/cityscapes/gtFine/val/"
    result_json_path = sys.argv[3] # "./"
    json_file_name = sys.argv[4] # "result.json"
    main()
