# Copyright (C) Huawei Technologies Co., Ltd 2022-2022. All rights reserved.
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

import torch
from timm.utils import accuracy
import os
import numpy as np
import argparse

def postprocess(txt_path, label_path):

    with open(label_path, 'r') as target_file:
        target_lines = target_file.readlines()
        target = []
        for line_n in target_lines:
            target_n = int(line_n.split(" ")[1])
            target.append(target_n)

    pre_file_list = sorted(os.listdir(txt_path))
    output = []
    for file in pre_file_list:
        out = np.loadtxt(os.path.join(txt_path, file))
        output.append(out)
    
    output = torch.tensor(np.array(output))
    target = torch.tensor(np.array(target))
    acc1, acc5 = accuracy(output, target, topk=(1, 5))
    print("{"+"key: Top1 accuracy, value: {top1:.2f}%".format(top1=acc1.item())+"}"+
    "{"+"key: Top5 accuracy, value: {top5:.2f}%".format(top5=acc5.item())+"}")
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--txt-path', type=str, default = None)
    parser.add_argument('--label-path', type=str, default = None)
    args = parser.parse_args()
    postprocess(args.txt_path, args.label_path)