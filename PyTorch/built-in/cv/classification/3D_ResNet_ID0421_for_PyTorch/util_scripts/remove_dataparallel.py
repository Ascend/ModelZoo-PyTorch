# Copyright 2020 Huawei Technologies Co., Ltd
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

import argparse
from collections import OrderedDict

import torch

parser = argparse.ArgumentParser()
parser.add_argument('file_path', type=str)
parser.add_argument('--dst_file_path', default=None, type=str)
args = parser.parse_args()

if args.dst_file_path is None:
    args.dst_file_path = args.file_path

x = torch.load(args.file_path)
state_dict = x['state_dict']
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    new_k = '.'.join(k.split('.')[1:])
    new_state_dict[new_k] = v

x['state_dict'] = new_state_dict

torch.save(x, args.dst_file_path)