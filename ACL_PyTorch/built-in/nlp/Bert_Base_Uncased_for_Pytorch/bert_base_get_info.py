# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import os
import argparse


base_path = './bert_bin/'
# three inputs in each eval
test_num = len(os.listdir(base_path)) // 3
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--batchsize', type=int, default=8)
args = parser.parse_args()
batchsize = args.batchsize
real_test = test_num // batchsize * batchsize

with open('./bert_base_uncased.info', 'w') as f:
    for i in range(real_test):
        ids_name = base_path + 'input_ids_{}.bin'.format(i)
        segment_name = base_path + 'segment_ids_{}.bin'.format(i)
        mask_name = base_path + 'input_mask_{}.bin'.format(i)
        f.write(str(i) + ' ' + ids_name)
        f.write('\n')
        f.write(str(i) + ' ' + segment_name)
        f.write('\n')
        f.write(str(i) + ' ' + mask_name)
        f.write('\n')
