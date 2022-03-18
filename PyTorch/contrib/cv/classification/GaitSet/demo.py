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

from model.initialization import initialization
from model.utils import evaluation
from config import conf_1p as conf


def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--iter', default='25600', type=int,
                    help='iter: iteration of the checkpoint to load. Default: 80000')
parser.add_argument('--batch_size', default='64', type=int,
                    help='batch_size: batch size for parallel test. Default: 1')
parser.add_argument('--cache', default=True, type=boolean_string,
                    help='cache: if set as TRUE all the test data will be loaded at once'
                         ' before the transforming start. Default: FALSE')
args = parser.parse_args()


# Exclude identical-view cases
def de_diag(acc, each_angle=False):
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / 10.0
    if not each_angle:
        result = np.mean(result)
    return result


conf['data']['dataset_path'] = os.path.abspath(conf['data']['dataset_path'])
conf['WORK_PATH'] = os.path.abspath(conf['WORK_PATH'])

m = initialization(conf, test=args.cache)[0]

# load model checkpoint of iteration args.iter
print('Loading the model of iteration %d...' % args.iter)
m.load(args.iter)
print('Transforming...')
time = datetime.now()
test = m.transform('test', args.batch_size)

if test == None:
    print('Pre-process Finished!')
else:
    print('Evaluating...')
    acc = evaluation(test, conf['data'])
    print('Evaluation complete. Cost:', datetime.now() - time)
    
    # Print rank-1 accuracy of the best model
    for i in range(1):
        print('===Rank-%d (Include identical-view cases)===' % (i + 1))
        print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
            np.mean(acc[0, :, :, i]),
            np.mean(acc[1, :, :, i]),
            np.mean(acc[2, :, :, i])))
    
    # Print rank-1 accuracy of the best model£¬excluding identical-view cases
    for i in range(1):
        print('===Rank-%d (Exclude identical-view cases)===' % (i + 1))
        print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
            de_diag(acc[0, :, :, i]),
            de_diag(acc[1, :, :, i]),
            de_diag(acc[2, :, :, i])))
    
    # Print rank-1 accuracy of the best model (Each Angle)
    np.set_printoptions(precision=2, floatmode='fixed')
    for i in range(1):
        print('===Rank-%d of each angle (Exclude identical-view cases)===' % (i + 1))
        print('NM:', de_diag(acc[0, :, :, i], True))
        print('BG:', de_diag(acc[1, :, :, i], True))
        print('CL:', de_diag(acc[2, :, :, i], True))
