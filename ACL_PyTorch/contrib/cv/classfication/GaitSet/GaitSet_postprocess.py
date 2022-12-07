# -*- coding: utf-8 -*-
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
# limitations under the License


from datetime import datetime
import numpy as np
import os
import argparse
import sys
import torch.distributed as dist
sys.path.append("./GaitSet")
from model.initialization import initialization
from model.utils import evaluation
from config import conf
from tqdm import tqdm


def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--batch_size', default='1', type=int,
                    help='batch_size: batch size for parallel test. Default: 1')
parser.add_argument('--post_process', default=True, type=boolean_string,
                    help='post_process: if set as TRUE then calculate from .bin files'
                         ' after the benchmark model evaluation. Default: FALSE')
parser.add_argument('--output_path', default='./result', type=str)
args = parser.parse_args()

conf['WORK_PATH'] = os.path.abspath('./')

m = initialization(conf, test=True)[0]



test = m.transform('test', args.batch_size, post_process=args.post_process, output_path=args.output_path)

# add post_process

print('Evaluating Model...')
acc = evaluation(test, conf['data'])


# Print rank-1 accuracy of the best model
for i in range(1):
    print('===Rank-%d (Include identical-view cases)===' % (i + 1))
    print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
        np.mean(acc[0, :, :, i]),
        np.mean(acc[1, :, :, i]),
        np.mean(acc[2, :, :, i])))