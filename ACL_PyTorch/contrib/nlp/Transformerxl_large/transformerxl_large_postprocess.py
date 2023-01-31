# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding: utf-8
import math
import os, sys
import numpy as np
import torch
import argparse

parser = argparse.ArgumentParser('Set data_bin and target_bin path', add_help=False)
parser.add_argument('--om_out_path', default="/home/huangwei/eval_only_hw/tools/msame/out/2022126_19_0_54_936547/", type=str)
parser.add_argument('--target_path', default="/home/huangwei/eval_only_hw/bin_target/", type=str)
args = parser.parse_args()
device = torch.device("cpu")
output_dir = args.om_out_path
target_dir = args.target_path
filenames = os.listdir(output_dir)
i = 0
total_len, total_loss = 0, 0.
for file in filenames:
    idx = file.split('_')[1]
    target_filename = target_dir + 'data_' + str(idx) + '.bin'
    target = np.fromfile(target_filename, dtype=np.int64)
    with open(output_dir + file) as f:
        line = f.readlines()[0].split()
        line_f = list(map(float, line))
        ret = torch.from_numpy(np.array(line_f))
        seq_len = len(ret)
        loss = ret.mean()
        total_loss += seq_len * loss.item()
        total_len += seq_len
        i += 1
        print('\rHave done {} batches'.format(i), end='')

print("\nloss = {:.2f}  | bpc {:.4f} ".format(total_loss / total_len, loss / math.log(2)))
print('completed!')




