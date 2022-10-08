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
from collections import OrderedDict
from mmaction.core import top_k_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description='Dataset UCF101 Postprocessing')
    parser.add_argument('--result_path', type=str)
    parser.add_argument('--info_path', type=str)
    parser.add_argument('--batch_size', type=int, default=1)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    with open(args.info_path, "r") as f:
        l = list(map(lambda x: int(x.strip()), f.readlines()))

    num_samples = len(l) // args.batch_size
    i = 0
    acc = 0
    while i < num_samples:
        with open(args.result_path+str(i)+'_0.txt', 'r') as f:
            lines = f.readlines()
            lines = list(map(lambda x:x.strip().split(), lines))
            lines = np.array([[float(lines[m][n]) for n in range(101)]for m in range(args.batch_size)]).argmax(1)
        for k in range(args.batch_size):
            acc += int(lines[k] == l[i*args.batch_size + k])
        i += 1

    print(acc / len(l))


if __name__ == '__main__':
    main()
