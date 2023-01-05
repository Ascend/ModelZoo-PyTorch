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
from collections import OrderedDict

from tqdm import tqdm
import numpy as np
from mmaction.core import top_k_accuracy


def parse_args():
    parser = argparse.ArgumentParser(
        description='Dataset K400 Postprocessing')
    parser.add_argument('--result_dir', type=str)
    parser.add_argument('--label_file', type=str)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # load info file
    gt_labels = []
    with open(args.label_file, 'r') as f:
        for line in f.readlines():
            gt_labels.append(int(line.split(' ')[1]))

    # load inference result
    results = []
    all_files = [file for file in os.listdir(args.result_dir)
                 if not file.endswith('.json')]
    num_file = len(all_files)
    for idx in tqdm(range(num_file)):
        file = os.path.join(args.result_dir, str(idx) + '_0.txt')
        with open(file, 'r') as f:
            for batch in f.readlines():
                line = batch.split(' ')[:-1]
                line = np.array([float(x) for x in line])
                results.append(line)
    results = results[:len(gt_labels)]

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
