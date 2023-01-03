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
import torch
from timm.utils import accuracy, AverageMeter
import os
from tqdm import tqdm
import argparse
import json

def postprocess(result_path, val_label, output_path):
      if not os.path.exists(os.path.split(output_path)[0]):
            os.makedirs(os.path.split(output_path)[0], exist_ok=True)
      acc1_meter = AverageMeter()
      acc5_meter = AverageMeter()
      result = sorted(os.listdir(result_path))
      target = np.loadtxt(val_label, dtype=np.int32, usecols=1)
      print(f'val size is {len(target)}')

      filename = output_path
      with open(filename, 'w') as file_obj:
            for i in tqdm(range(len(target))):
                  output = torch.tensor(np.loadtxt(os.path.join(result_path,result[i]),dtype=np.float32)).unsqueeze(dim=0)
                  acc1, acc5 = accuracy(output, torch.tensor([target[i]]), topk=(1, 5))
                  acc1_meter.update(acc1.item())
                  acc5_meter.update(acc5.item())
                  val_result = (f'val index {i}    '
                                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})    '
                                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})')
                  json.dump(val_result, file_obj)
                  file_obj.write('\n')

if __name__ == '__main__':
      parser = argparse.ArgumentParser()
      parser.add_argument('--input_path', type=str, default='./infer/result/2022_08_27-14_15_13')
      parser.add_argument('--label_path', type=str, default='./imageNet/val_label.txt')
      parser.add_argument('--output_path', type=str, default='./infer/result.json')
      args = parser.parse_args()
      postprocess(args.input_path, args.label_path, args.output_path)
