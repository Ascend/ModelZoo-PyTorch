"""
Copyright 2020 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
============================================================================
"""
# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
from mmaction.core import top_k_accuracy


def parse_args():
    """
    input argument receiving function
    :return: input argument
    """
    parser = argparse.ArgumentParser(description='Dataset UCF101 Postprocessing')
    parser.add_argument('--result_path', type=str)
    parser.add_argument('--info_path', type=str)

    args = parser.parse_args()

    return args


def main():
    """
    main function
    :return: None
    """
    args = parse_args()

    # load info file
    gt_labels = []
    with open(args.info_path, 'r') as f:
        for line in f.readlines():
            gt_labels.append(int(line.split(' ')[1]))

    name = args.result_path.split('_')[1]
    subdir = os.listdir(args.result_path)[0]
    # args.result_path = os.path.join(args.result_path, subdir)

    # load inference result
    results = []
    num_file = len(os.listdir(args.result_path))
    for idx in range(num_file):
        file_ = os.path.join(args.result_path, str(idx) + '_output_0.txt')
        with open(file_, 'r') as f:
            for batch in f.readlines():
                line = batch.split(' ')[:-1]
                line = np.array([float(x) for x in line])
                results.append(line)
                break
    results = results[:len(gt_labels)]

    metrics = ['top_k_accuracy']
    metric_options = dict(top_k_accuracy=dict(topk=(1, 5)))
    for metric in metrics:
        if metric == 'top_k_accuracy':
            topk = metric_options.setdefault('top_k_accuracy', {}).setdefault('topk', (1, 5))
            if not isinstance(topk, (int, tuple)):
                raise TypeError('topk must be int or tuple of int')
            if isinstance(topk, int):
                topk = (topk,)

            top_k_acc = top_k_accuracy(results, gt_labels, topk)
            print('om {} top1:{:.4f} top5:{:.4f}'.format(name, top_k_acc[0], top_k_acc[1]))


if __name__ == '__main__':
    main()
