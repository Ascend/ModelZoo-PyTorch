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
# limitations under the License.
import math
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import utils
import os
import sys
import json
import time
import argparse
import numpy as np
from tqdm import tqdm

def target_fromtxt(gtfile_path):
    img_gt_dict = []
    with open(gtfile_path, 'r') as f:
        for line in f.readlines():
            temp = line.strip().split(" ")
            imgLab = int(temp[1])
            img_gt_dict.append(imgLab)
    return torch.tensor(img_gt_dict)

def get_file_list(file_dir):
    for root, dirs, files in os.walk(file_dir):
        print('root_dir:', root) 
        return root 
    return root

@torch.no_grad()
def evaluate(data_path, result_json_path, json_file_name, val_path):
    target2 = target_fromtxt(val_path)
    target2 = target2.to('cpu', non_blocking=True)
    metric_logger = utils.MetricLogger(delimiter="  ")

    root = get_file_list(data_path)
    O1 = None
    index1 = 0
    writer = open(os.path.join(result_json_path, json_file_name), 'w')

    for number in range(1, 50001):
        pbar.update(1)
        output2 = torch.from_numpy(np.fromfile(f''+root+"/"+"ILSVRC2012_val_000"+str(number).zfill(5)+'_0.bin', dtype = np.float32))       
        if(O1 is not None):
            O2 = torch.cat([O1,output2.unsqueeze(0)],dim=0)
            O1 = O2
        else:
            O1 = output2.unsqueeze(0)
        if(O1.shape[0]<96):
            continue
        
        O1 = O1.to('cpu', non_blocking = True)
        acc1, acc5 = accuracy(O1, target2[index1:index1+96], topk = (1, 5))
        batch_size = O1.shape[0]
        metric_logger.meters['acc1'].update(acc1.item(), n = batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n = batch_size)
        O1 = None
        index1 = index1+96

    if(O1 is not None) :
        O1 = O1.to('cpu', non_blocking = True)    
        acc1, acc5 = accuracy(O1, target2[index1:], topk = (1, 5))
        batch_size = O1.shape[0]
        metric_logger.meters['acc1'].update(acc1.item(), n = batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n = batch_size)
    metric_logger.synchronize_between_processes()   
    json.dump('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1 = metric_logger.acc1, top5 = metric_logger.acc5), writer)
    writer.close()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1 = metric_logger.acc1, top5 = metric_logger.acc5))

if __name__ == '__main__':
    data_path = sys.argv[1]
    val_path = sys.argv[4]
    result_json_path = sys.argv[2]
    json_file_name = sys.argv[3]

    pbar = tqdm(total = 50000)
    evaluate(data_path, result_json_path, json_file_name, val_path)
    pbar.close()

