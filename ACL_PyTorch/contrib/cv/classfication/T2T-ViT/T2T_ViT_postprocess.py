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

import argparse
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from timm.utils import *


def _parse_args():

    parser = argparse.ArgumentParser(description='T2T-ViT postprocess.')
    parser.add_argument('--result-dir', type=str, metavar='DIR', help='path to model output')
    parser.add_argument('--gt-path', type=str, metavar='PATH', help='path to groundtruth')
    parser.add_argument('-b', '--batch-size', type=int, default=1, metavar='N',help='input batch size for training (default: 64)')
    args = parser.parse_args()

    args.num_classes = 1000

    return args


def post_precess(result_dir, gt_path, args):

    labels = torch.from_numpy(np.load(gt_path))
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    for res in tqdm(Path(result_dir).iterdir()):
        i = int(res.stem.replace('_0', ''))    
        target = torch.tensor([labels[i][0]])
        output = np.fromfile(res.__str__(), dtype=np.float32)
        output = torch.from_numpy(output.reshape(1, args.num_classes))

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1_m.update(acc1.item(), output.size(0))
        top5_m.update(acc5.item(), output.size(0))

    metrics = OrderedDict([('top1', top1_m.avg), ('top5', top5_m.avg)])
    print(f"Top-1 accuracy of the model is: {metrics['top1']:.1f}%")
    print(f"val_metrics: {metrics}")

    return metrics


def main():
    args = _parse_args()
    post_precess(args.result_dir, args.gt_path, args)


if __name__ == '__main__':
    main()