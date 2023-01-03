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
parser.add_argument('--data_path', default='./predata', type=str)
parser.add_argument('--bin_file_path', default='./GASIS-B-bin', type=str)  
parser.add_argument('--pre_process', default=True, type=boolean_string,
                    help='pre_process: if set as TRUE then convert images to .bin'
                         ' before the benchmark model evaluation. Default: FALSE')                
args = parser.parse_args()


# Exclude identical-view cases
def de_diag(acc, each_angle=False):
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / 10.0
    if not each_angle:
        result = np.mean(result)
    return result



conf['WORK_PATH'] = os.path.abspath('./')
conf['data']['dataset_path'] = args.data_path

m = initialization(conf, test=True)[0]
# add post_process
test = m.transform('test', bin_file_path=args.bin_file_path, batch_size=args.batch_size, pre_process=args.pre_process)

