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
import argparse
import numpy as np
from collections import OrderedDict
from mmaction.core import top_k_accuracy
import torch
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(
        description='Dataset K400 Postprocessing')
    parser.add_argument('--result_path', type=str)
    parser.add_argument('--info_path', default='k400.info', type=str)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # load info file
    gt_labels = []
    with open(args.info_path, 'r') as f:
        for line in f.readlines():
            t = line.split( )[-1]
            gt_labels.append(int(t))


    # load inference result
    results = []
    num_file = len(os.listdir(args.result_path)) - 1
    for idx in range(num_file):
        file = os.path.join(args.result_path, str(idx) + '_0.txt')
        result = np.loadtxt(file)
        result = torch.from_numpy(result)
        batch_size = result.shape[0]
        result = result.view(batch_size // 3, 3, -1)
        result = F.softmax(result, dim=2).mean(dim=1).numpy()
        results.extend(result)


    metrics = ['top_k_accuracy']
    metric_options = dict(top_k_accuracy=dict(topk=(1, 5)))
    eval_results = OrderedDict()
    for metric in metrics:
        print(f'Evaluating {metric} ...')
        if metric == 'top_k_accuracy':
            topk = metric_options.setdefault('top_k_accuracy',
                                             {}).setdefault('topk', (1, 5))
            if not isinstance(topk, (int, tuple)):
                raise TypeError(
                    f'topk must be int or tuple of int, but got {type(topk)}')
            if isinstance(topk, int):
                topk = (topk, )

            top_k_acc = top_k_accuracy(results, gt_labels, topk)
            log_msg = []
            for k, acc in zip(topk, top_k_acc):
                eval_results[f'top{k}_acc'] = acc
                log_msg.append(f'\ntop{k}_acc\t{acc:.4f}')
            log_msg = ''.join(log_msg)
            print(log_msg)
            continue


if __name__ == '__main__':
    main()