# Copyright 2021 Huawei Technologies Co., Ltd
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
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output', default='result/bs1')
args = parser.parse_args()

file_dir = args.output

with open(file_dir.rstrip('/') + "_summary.json",'r') as load_f:
    load_dict = json.load(load_f)

print(load_dict.keys())

load_dict = load_dict['filesinfo']
load_dict_0 = load_dict['0']
print(load_dict_0.keys())

for i in load_dict.keys():
    data = load_dict[i]
    infile = os.path.basename(data['infiles'][0])
    outfile = os.path.basename(data['outfiles'][0])
    os.rename(os.path.join(file_dir + outfile), os.path.join(file_dir + infile))
