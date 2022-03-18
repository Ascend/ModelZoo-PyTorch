# Copyright 2020 Huawei Technologies Co., Ltd
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

import sys

if len(sys.argv) != 2:
    print("We need training text to generate the modelling units.")
    sys.exit(1)

train_text = sys.argv[1]
units_file = 'data/units'

units = {}
with open(train_text, 'r') as fin:   
    line = fin.readline()
    while line:
        line = line.strip().split(' ')
        for char in line[1:]:
            try:
                if units[char] == True:
                    continue
            except:
                units[char] = True
        line = fin.readline()

fwriter = open(units_file, 'w')
for char in units:
    print(char, file=fwriter)


