# Copyright (c) Soumith Chintala 2016,
# All rights reserved
#
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://spdx.org/licenses/BSD-3-Clause.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-
"""用于导出动态shape算子
"""

import os
import sys
import argparse


def func(log_path, split_flag):
    """
    :param log_path: where log_path addr is.
    :return:
    """
    recompile_flag = 'To compile op: '
    output_list = [[]]

    with open(log_path, 'r')as f:
        log_list = f.read().split('\n')
        for log in log_list:
            log = log.strip()
            if split_flag in log:
                output_list.append([])
            elif recompile_flag in log:
                op_name = log.split(recompile_flag)[1]
                if op_name not in output_list[-1]:
                    output_list[-1].append(op_name)

    with open('recompile_op_list.txt', 'w')as f:
        for idx, output in enumerate(output_list):
            f.write('iter: %d' % idx + '\n')
            f.write(','.join(output) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='trans the log')
    parser.add_argument('--log_path', default="./recompile_op.log",
                        help="input the dir name, trans the current dir with default")
    parser.add_argument('--split_flag', default='=====iter',
                        help="flag for split epochs")
    args = parser.parse_args()
    func(args.log_path, args.split_flag)