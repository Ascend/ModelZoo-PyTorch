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
import json
import torch
import sys


def process_label(label_file):
    label = dict()
    with open(label_file, 'r') as f:
        x = f.readlines()
    for i in range(len(x)):
        class_name = x[i].split(' ')[0].split('/')[1]
        class_idx = x[i].split(' ')[2].replace('\n', '').replace('\r', '')
        label[class_name] = class_idx
    return label


def postprocess(result_file, label_file, json_file):
    # evaluate
    file_names = os.listdir(result_file)
    num_correct_top1 = 0
    num_total = len(file_names)
    for file_idx in range(num_total):
        x = file_names[file_idx]  
        with open(os.path.join(result_file, x), 'r') as f:
            scores = f.readlines()
            s = [[] for _ in range(10)]
            for i, score in enumerate(scores):
                score_ = score.replace('\n', '').replace('\r', '').split(' ')
                s[i] = [float(i) for i in score_ if i!='']        
        cls_score = torch.tensor(s).mean(dim=0)
        max_value = cls_score[0]
        idx = 0
        for i in range(len(cls_score)):
            if cls_score[i] >= max_value:
                max_value = cls_score[i]
                idx = i
        label = process_label(label_file)
        if label[x.split('.')[0].replace('_0', '')] == str(idx):
            num_correct_top1 += 1

    # generate result json file
    top1_acc = num_correct_top1 / num_total
    result_dict = {"top1_acc": top1_acc}
    print(result_dict)
    json_str = json.dumps(result_dict)
    with open(json_file, 'w') as f:
        f.write(json_str)


if __name__=="__main__":
    result_dir = sys.argv[1]  
    label_dir = sys.argv[2] 
    json_dir = sys.argv[3]
    postprocess(result_dir, label_dir, json_dir)
