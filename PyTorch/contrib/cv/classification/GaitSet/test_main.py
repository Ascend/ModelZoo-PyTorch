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

from datetime import datetime
import numpy as np
import os
import argparse
import torch
if torch.__version__>="1.8":
    import torch_npu

from model.initialization import initialization
from model.utils import evaluation
from config import conf_1p as conf


def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--iter', default='80000', type=int,
                    help='iter: iteration of the checkpoint to load. Default: 80000')
parser.add_argument('--batch_size', default='1', type=int,
                    help='batch_size: batch size for parallel test. Default: 1')
parser.add_argument('--cache', default=False, type=boolean_string,
                    help='cache: if set as TRUE all the test data will be loaded at once'
                         ' before the transforming start. Default: FALSE')
parser.add_argument('--data_path',default='../../CASIA-B-Pre/', type=str,help='data_path')
args = parser.parse_args()


# Exclude identical-view cases
def de_diag(acc, each_angle=False):
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / 10.0
    if not each_angle:
        result = np.mean(result)
    return result


if os.getenv('ALLOW_FP32') or os.getenv('ALLOW_HF32'):
    torch.npu.config.allow_internal_format = False
    if os.getenv('ALLOW_FP32'):
        torch.npu.conv.allow_hf32 = False
        torch.npu.matmul.allow_hf32 = False
conf['data']['dataset_path'] = args.data_path
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
    
    # Print rank-1 accuracy of the best model，excluding identical-view cases
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
